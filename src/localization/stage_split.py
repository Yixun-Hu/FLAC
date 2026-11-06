"""exp_37 instrumented stage-split pass (plan_loc_efficiency.md §10 BLOCKING 1) -- NEVER the headline.

``execute_query_stage_split`` mirrors ``runner.execute_query`` statement for statement (same caches, seeds, batching,
syncs and perf_counter windows, so its ``latency_seconds`` are comparable with the headline pass) and adds, inside
the timed windows, CUDA-event marks (no host sync; read after the existing end-of-batch sync) and per-segment
allocator resets / reads:

  context encoding        = the existing context window (FA: the per-angle ViT passes included)
  candidate conditioning  = batch start -> after ``merge_cached_conditioning`` + ``get_conditioning_inputs``
                            (candidate metadata, the source / FA dynamic branches incl. the rotated ViT passes, seeds)
  DiT                     = ``deterministic_noise`` + the one Euler step (``sample_discrete_euler``)
  VAE decode              = after ``sample_discrete_euler`` -> the decoded, clamped batch

Memory: absolute peaks per segment (allocated and reserved, not additive) and a running whole-query maximum over
every GPU segment (observed encode and AGREE scoring included) -- the bundle-compatible ``peak_memory_bytes``.
Its totals are compared with the uninstrumented headline before any split is published (``exp37_latency``).
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""

from __future__ import annotations

import time

import torch

from src.inference.sampling import sample_discrete_euler
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS, FA_CONTEXT_CONDITIONING_IDS, FA_DYNAMIC_CONDITIONING_IDS, SOURCE_CONDITIONING_IDS,
    candidate_metadata, candidate_seed, deterministic_noise, encode_audio_features, filter_frozen_query_candidates,
    load_frozen_query, merge_cached_conditioning, prepare_generated_audio,
)
from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.localization.runner import _build_query_result, _cache_branch, _repeat_branch, validate_frame_angles

STAGES = ("context_conditioning", "candidate_conditioning", "dit", "vae_decode")


class HostClock:
    """perf_counter marks (CPU devices and tests)."""

    def mark(self):
        return time.perf_counter()

    @staticmethod
    def seconds(start, stop):
        return stop - start


class EventClock:
    """CUDA events on the current stream: recording adds no host sync."""

    def mark(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def seconds(start, stop):
        return start.elapsed_time(stop) / 1000.0


class PeakMemory:
    """Absolute per-segment allocator peaks and their running maximum (host-side counters, no device sync)."""

    def __init__(self, device, enabled=False, reset=None, allocated=None, reserved=None):
        self.device, self.enabled, self.segments, self.query_peak = device, enabled, {}, None
        self._reset = reset or torch.cuda.reset_peak_memory_stats
        self._allocated = allocated or torch.cuda.max_memory_allocated
        self._reserved = reserved or torch.cuda.max_memory_reserved

    def begin(self):
        if self.enabled:
            self._reset(self.device)

    def end(self, name):
        if not self.enabled:
            self.segments[name] = {"allocated_bytes": None, "reserved_bytes": None}
            return
        peak = {"allocated_bytes": int(self._allocated(self.device)), "reserved_bytes": int(self._reserved(self.device))}
        old = self.segments.get(name, {"allocated_bytes": 0, "reserved_bytes": 0})
        self.segments[name] = {k: max(old[k], v) for k, v in peak.items()}
        self.query_peak = {k: max((self.query_peak or peak)[k], v) for k, v in peak.items()}


@torch.inference_mode()
def generate_rir_batch_staged(module, source_branch, context_branch, seeds, dynamic_branch=None, *, clock, memory,
                              marks, steps=1, cfg_scale=1.0):
    """``engine.generate_rir_batch`` with the three stage marks; the computation is the same."""
    diffusion = module.diffusion
    if diffusion.diffusion_objective != "rectified_flow":
        raise ValueError("exp_09 probe is pinned to the rectified-flow checkpoint")
    batch_size = len(seeds)
    conditioning = merge_cached_conditioning(source_branch, context_branch, batch_size, dynamic_branch=dynamic_branch)
    inputs = diffusion.get_conditioning_inputs(conditioning)
    marks["conditioned"] = clock.mark()
    memory.end("candidate_conditioning")
    memory.begin()
    latent_samples = 10240 // diffusion.pretransform.downsampling_ratio
    dtype = next(diffusion.model.parameters()).dtype
    noise = deterministic_noise(seeds, (diffusion.io_channels, latent_samples), device=module.device, dtype=dtype)
    fakes = sample_discrete_euler(diffusion.model, noise, steps, **inputs, cfg_scale=cfg_scale,
                                  dist_shift=diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
    marks["dit"] = clock.mark()
    memory.end("dit")
    memory.begin()
    fakes = diffusion.pretransform.decode(fakes).float()
    generated = prepare_generated_audio(fakes, sample_size=10240)
    marks["decoded"] = clock.mark()
    memory.end("vae_decode")
    return generated


@torch.inference_mode()
def execute_query_stage_split(*, module, retrieval, selected, record, audit, room_base, dataset_root, device,
                              cond_method, candidate_batch_size, sample_seed, tau, run_sha256, score_sample_counts=(1,),
                              synchronize_timing=False, measure_core_forward=False, context_count=8,
                              frame_angles=DEFAULT_FRAME_ANGLES, candidate_subset=None):
    """The headline ``execute_query`` (``--measure-core-forward``) with the stage marks and memory segments."""
    if not measure_core_forward or tuple(score_sample_counts) != (1,):
        raise ValueError("the stage split needs measure_core_forward and K_gen = 1")
    if candidate_subset is not None:
        raise ValueError("the stage split runs on full candidate sets only")
    validate_frame_angles(cond_method, frame_angles)
    cuda = torch.device(device).type == "cuda"
    device_type = torch.device(device).type
    clock, memory = (EventClock() if cuda else HostClock()), PeakMemory(device, enabled=cuda)
    if synchronize_timing and cuda:
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    candidates = filter_frozen_query_candidates(record, audit, room_base)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    memory.begin()
    observed_gpu = observed.unsqueeze(0).to(device=device, dtype=torch.float32)
    observation_features = encode_audio_features(retrieval, observed_gpu)
    memory.end("observed_encode")

    if cuda:
        torch.cuda.synchronize(device)
    context_started = time.perf_counter()
    memory.begin()
    context_mark = clock.mark()
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [metadata], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    context_done = clock.mark()
    if cuda:
        torch.cuda.synchronize(device)
    context_seconds = time.perf_counter() - context_started
    memory.end("context_conditioning")
    stage = dict.fromkeys(STAGES, 0.0)
    stage["context_conditioning"] = clock.seconds(context_mark, context_done)

    score_chunks, candidate_generation_seconds = [], 0.0
    for batch_start in range(0, len(candidates), candidate_batch_size):
        batch_stop = min(batch_start + candidate_batch_size, len(candidates))
        if cuda:
            torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        memory.begin()
        batch_mark, marks = clock.mark(), {}
        candidate_items = candidate_metadata(metadata, candidates[batch_start:batch_stop], record["receiver_global"])
        with torch.amp.autocast(device_type):
            source = _cache_branch(module, candidate_items, SOURCE_CONDITIONING_IDS, device, cond_method,
                                   frame_angles=frame_angles)
            dynamic = (_cache_branch(module, candidate_items, FA_DYNAMIC_CONDITIONING_IDS, device, cond_method,
                                     frame_angles=frame_angles)
                       if cond_method == "fa_invariant" else None)
        repeated_source = _repeat_branch(source, 1)
        repeated_dynamic = _repeat_branch(dynamic, 1) if dynamic else None
        seeds = [candidate_seed(sample_seed, int(record["index"]), i, 0) for i in range(batch_start, batch_stop)]
        generated = generate_rir_batch_staged(module, repeated_source, context, seeds, dynamic_branch=repeated_dynamic,
                                              clock=clock, memory=memory, marks=marks)
        if cuda:
            torch.cuda.synchronize(device)
        candidate_generation_seconds += time.perf_counter() - generation_started
        stage["candidate_conditioning"] += clock.seconds(batch_mark, marks["conditioned"])
        stage["dit"] += clock.seconds(marks["conditioned"], marks["dit"])
        stage["vae_decode"] += clock.seconds(marks["dit"], marks["decoded"])
        memory.begin()
        scores = encode_audio_features(retrieval, generated) @ observation_features.T
        score_chunks.append(scores.reshape(batch_stop - batch_start, 1).cpu())
        memory.end("agree_scoring")

    similarities = torch.cat(score_chunks, dim=0).float()
    if similarities.shape != (len(candidates), 1):
        raise RuntimeError("generated score matrix has an unexpected shape")
    peak_memory = memory.query_peak["allocated_bytes"] if memory.query_peak else 0
    if synchronize_timing and cuda:
        torch.cuda.synchronize(device)
    result = _build_query_result(selected=selected, record=record, candidates=candidates, similarities=similarities,
                                 run_sha256=run_sha256, elapsed_seconds=time.perf_counter() - start_time,
                                 peak_memory_bytes=peak_memory, tau=tau, random_seed=sample_seed,
                                 score_sample_counts=(1,))
    result["latency_protocol"] = {
        "name": f"fem_core_aligned_kctx{context_count}_kgen1", "context_count": context_count,
        "generated_rirs_per_candidate": 1, "input_loading_included": False, "candidate_filtering_included": False,
        "candidate_scoring_included": False, "candidate_selection_and_metrics_included": False,
        "result_serialization_included": False, "stage_split": True}
    result["latency_seconds"] = {"context_conditioning": context_seconds,
                                 "candidate_conditioning_and_generation": candidate_generation_seconds,
                                 "core_forward_total": context_seconds + candidate_generation_seconds}
    result["stage_seconds"] = stage
    result["stage_sum_seconds"] = float(sum(stage.values()))
    result["stage_peak_memory"] = {**memory.segments, "whole_query": memory.query_peak}
    return result, candidates, similarities.numpy()
