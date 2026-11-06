"""exp_37 B3 rotation contract (plan_loc_efficiency.md §5, §10 "B3 rotation contract"; Codex HIGH 3).

A rigid yaw rotation of the whole scene about the receiver's vertical axis, applied to the CONDITIONING only:
  * ``base`` = the frozen query metadata (``load_frozen_query``), never mutated;
  * the query / context cache is built from ``rotate_scene_metadata(base, θ, 512)`` -- ONE 128-column roll at 90°
    plus ONE rotation of every pose vector;
  * every candidate item is built from ``base`` (``candidate_metadata``) and THEN rotated once -- never derived from
    the rotated query metadata (which would rotate depth and context poses twice);
  * candidate arrays, the frozen filtering, the truth and the metrics stay in the global frame.
θ = 0 is the existing headline path (``runner.execute_query``), so this accuracy-only executor refuses it; the angle
must be a whole number of panorama columns (exact roll). Seeds, batching and scoring are the headline's.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES, rotate_scene_metadata
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS, FA_CONTEXT_CONDITIONING_IDS, FA_DYNAMIC_CONDITIONING_IDS, SOURCE_CONDITIONING_IDS,
    candidate_metadata, candidate_seed, encode_audio_features, filter_frozen_query_candidates,
    generate_and_score_batch, load_frozen_query,
)
from src.localization.runner import _build_query_result, _cache_branch, _repeat_branch, validate_frame_angles

PANORAMA_WIDTH = 512


def column_shift(theta_deg: float) -> int:
    columns = float(theta_deg) * PANORAMA_WIDTH / 360.0
    if abs(columns - round(columns)) > 1e-9:
        raise ValueError(f"rotate_deg {theta_deg} is not a whole number of the {PANORAMA_WIDTH} panorama columns")
    return int(round(columns)) % PANORAMA_WIDTH


def rotated_query_metadata(base: dict, theta_deg: float) -> dict:
    column_shift(theta_deg)
    return rotate_scene_metadata(base, math.radians(theta_deg), PANORAMA_WIDTH)


def rotated_candidate_items(base: dict, candidates, receiver_global, theta_deg: float) -> list[dict]:
    column_shift(theta_deg)
    return [rotate_scene_metadata(item, math.radians(theta_deg), PANORAMA_WIDTH)
            for item in candidate_metadata(base, candidates, receiver_global)]


@torch.inference_mode()
def execute_query_rotated(*, module, retrieval, selected, record, audit, room_base, dataset_root, device, cond_method,
                          candidate_batch_size, sample_seed, tau, run_sha256, rotate_deg, score_sample_counts=(1,),
                          synchronize_timing=False, measure_core_forward=False, context_count=8,
                          frame_angles=DEFAULT_FRAME_ANGLES, candidate_subset=None):
    if measure_core_forward:
        raise ValueError("rotated runs are accuracy-only (no --measure-core-forward timing)")
    if candidate_subset is not None:
        raise ValueError("rotated runs use the full frozen candidate set")
    shift = column_shift(rotate_deg)
    if shift == 0:
        raise ValueError("rotate_deg 0 is the existing headline path, not this executor")
    validate_frame_angles(cond_method, frame_angles)
    device_type, start_time = torch.device(device).type, time.perf_counter()   # bookkeeping, not a timing
    observed, base = load_frozen_query(record, dataset_root)
    candidates = filter_frozen_query_candidates(record, audit, room_base)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    observation_features = encode_audio_features(retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32))
    query = rotated_query_metadata(base, rotate_deg)
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [query], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    samples = max(score_sample_counts)
    chunks = []
    for start in range(0, len(candidates), candidate_batch_size):
        stop = min(start + candidate_batch_size, len(candidates))
        items = rotated_candidate_items(base, candidates[start:stop], record["receiver_global"], rotate_deg)
        with torch.amp.autocast(device_type):
            source = _cache_branch(module, items, SOURCE_CONDITIONING_IDS, device, cond_method,
                                   frame_angles=frame_angles)
            dynamic = (_cache_branch(module, items, FA_DYNAMIC_CONDITIONING_IDS, device, cond_method,
                                     frame_angles=frame_angles) if cond_method == "fa_invariant" else None)
        seeds = [candidate_seed(sample_seed, int(record["index"]), i, s) for i in range(start, stop)
                 for s in range(samples)]
        scores = generate_and_score_batch(module, retrieval, _repeat_branch(source, samples), context,
                                          observation_features, seeds,
                                          dynamic_branch=_repeat_branch(dynamic, samples) if dynamic else None)
        chunks.append(scores.reshape(stop - start, samples).cpu())
    similarities = torch.cat(chunks, dim=0).float()
    peak = int(torch.cuda.max_memory_allocated(device)) if device_type == "cuda" else 0
    result = _build_query_result(selected=selected, record=record, candidates=candidates, similarities=similarities,
                                 run_sha256=run_sha256, elapsed_seconds=time.perf_counter() - start_time, peak_memory_bytes=peak, tau=tau,
                                 random_seed=sample_seed, score_sample_counts=tuple(score_sample_counts))
    result["rotation"] = {"rotate_deg": float(rotate_deg), "panorama_width": PANORAMA_WIDTH, "column_shift": shift,
                          "effective_deg": 360.0 * shift / PANORAMA_WIDTH,
                          "contract": "query rotated once; candidates built from the base then rotated once; "
                                      "candidates / truth / metrics global"}
    return result, np.asarray(candidates), similarities.numpy()
