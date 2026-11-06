"""exp_37 untimed artifact executor and artifact replay (announcement 08; Codex code review 1, BLOCKING 1).

``execute_query_artifacts`` computes exactly what the timed headline computes -- the same context and candidate
caches, full-grid seeds, batches, ``generate_rir_batch`` and AGREE scoring -- but never times, and dumps each query's
scored waveforms (the exact ``prepare_generated_audio`` output) through a ``WaveformDump``. Accuracy-only runs use it
inline. Timing passes never dump; their ARTIFACT REPLAY runs this executor with ``reference`` = the timing run's stored
similarities and refuses unless the regenerated similarities are bitwise identical. A coarse-to-fine pass is replayed
with ``c2f_replay_plan`` (its coarse batches, then its new refined batches; output in union order).
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS, FA_CONTEXT_CONDITIONING_IDS, FA_DYNAMIC_CONDITIONING_IDS, SOURCE_CONDITIONING_IDS,
    candidate_metadata, candidate_seed, encode_audio_features, filter_frozen_query_candidates, generate_rir_batch,
    load_frozen_query,
)
from src.localization.exp37_grid import nested_subset
from src.localization.runner import _build_query_result, _cache_branch, _repeat_branch, validate_frame_angles


def c2f_replay_plan(c2f: dict):
    """The batch plan of a recorded c2f pass: its coarse (nested) batches, then its new refined batches."""
    union = np.asarray(c2f["union_indices"])

    def plan(full, batch_size):
        coarse = nested_subset(full)
        new = np.setdiff1d(union, coarse)
        batches = [coarse[i:i + batch_size] for i in range(0, len(coarse), batch_size)]
        return batches + [new[i:i + batch_size] for i in range(0, len(new), batch_size)], union
    return plan


@torch.inference_mode()
def execute_query_artifacts(*, module, retrieval, selected, record, audit, room_base, dataset_root, device, cond_method,
                            candidate_batch_size, sample_seed, tau, run_sha256, dump, score_sample_counts=(1,),
                            synchronize_timing=False, measure_core_forward=False, context_count=8,
                            frame_angles=DEFAULT_FRAME_ANGLES, candidate_subset=None, reference=None, plan=None):
    if measure_core_forward or synchronize_timing:
        raise ValueError("the artifact executor is untimed (timing passes never dump; replay them instead)")
    if tuple(score_sample_counts) != (1,):
        raise ValueError("exp_37 artifacts are K_gen = 1")
    validate_frame_angles(cond_method, frame_angles)
    device_type, started = torch.device(device).type, time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    full = filter_frozen_query_candidates(record, audit, room_base)
    if len(full) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    if plan is not None:
        batches, output = plan(full, candidate_batch_size)
    else:
        output = np.asarray(candidate_subset(full)) if candidate_subset is not None else np.arange(len(full))
        batches = [output[i:i + candidate_batch_size] for i in range(0, len(output), candidate_batch_size)]
    observation_features = encode_audio_features(retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32))
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [metadata], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    scores, waves = {}, {}
    for batch in batches:
        items = candidate_metadata(metadata, full[batch], record["receiver_global"])
        with torch.amp.autocast(device_type):
            source = _cache_branch(module, items, SOURCE_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
            dynamic = (_cache_branch(module, items, FA_DYNAMIC_CONDITIONING_IDS, device, cond_method,
                                     frame_angles=frame_angles) if cond_method == "fa_invariant" else None)
        seeds = [candidate_seed(sample_seed, int(record["index"]), int(i), 0) for i in batch]
        generated = generate_rir_batch(module, _repeat_branch(source, 1), context, seeds,
                                       dynamic_branch=_repeat_branch(dynamic, 1) if dynamic else None)
        batch_scores = (encode_audio_features(retrieval, generated) @ observation_features.T).reshape(-1).cpu()
        generated = generated.cpu()
        for row, index in enumerate(batch):
            scores[int(index)], waves[int(index)] = batch_scores[row], generated[row]
    similarities = torch.stack([scores[int(i)] for i in output]).float().reshape(-1, 1)
    if reference is not None:
        expected = np.asarray(reference(int(record["index"])))
        if expected.dtype != np.float32 or not np.array_equal(similarities.numpy(), expected):
            delta = float(np.max(np.abs(similarities.numpy().astype(np.float64) - expected.astype(np.float64)))) \
                if expected.shape == tuple(similarities.shape) else float("nan")
            raise RuntimeError(f"replay: query {record['index']} similarities are not bitwise the timing run's "
                               f"(max |delta| {delta:.3e})")
    entry = dump.write(int(record["index"]), torch.stack([waves[int(i)] for i in output]).numpy(), output)
    peak = int(torch.cuda.max_memory_allocated(device)) if device_type == "cuda" else 0
    result = _build_query_result(selected=selected, record=record, candidates=full[output], similarities=similarities,
                                 run_sha256=run_sha256, elapsed_seconds=time.perf_counter() - started,
                                 peak_memory_bytes=peak, tau=tau, random_seed=sample_seed, score_sample_counts=(1,))
    if candidate_subset is not None or plan is not None:
        active = hashlib.sha256(np.asarray(output).astype("<u4").tobytes()).hexdigest()
        result["parent_candidate_indices_sha256"] = result["candidate_indices_sha256"]
        result["candidate_indices_sha256"] = active
        if candidate_subset is not None:
            result["candidate_subset"] = {"name": candidate_subset.__name__, "full_candidate_count": len(full),
                                          "indices_sha256": active}
    result["waveforms"] = {"dump_dir": str(dump.dir), **entry}
    return result, full[output], similarities.numpy()
