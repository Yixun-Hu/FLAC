"""exp_37 coarse-to-fine SEQUENTIAL timing pass (plan_loc_efficiency.md §4, §10 "B2 time axis").

Times the path a deployed c2f search actually runs, end to end, with the bundle's sync placement (a CUDA sync before
every ``perf_counter`` read): the observed RIR encoded ONCE -> the context cache -> coarse = the nested 1.0 m subset
(generation, then AGREE encoding + similarity) -> top-k (score desc, full-grid index asc) + the Chebyshev-0.5 m
neighbourhood -> refined = only the candidates not yet scored (generation, encoding, similarity) -> argmax over
coarse U refined in full-grid order. Seeds stay keyed by the full 0.5 m index. ``c2f_total`` is the c2f L_q; the
generator-only part (context + both generations) is ``core_forward_total``. The c2f ACCURACY of record is the
offline readout of the 0.5 m runs (``exp37_grid``); this pass's own winner is recorded as a cross-check.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
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
from src.localization.exp37_grid import CHEBYSHEV_RADIUS_M, TOLERANCE_M, nested_subset, query_scores, top_k
from src.localization.runner import _build_query_result, _cache_branch, _repeat_branch, validate_frame_angles


@torch.inference_mode()
def execute_query_c2f(*, module, retrieval, selected, record, audit, room_base, dataset_root, device, cond_method,
                      candidate_batch_size, sample_seed, tau, run_sha256, score_sample_counts=(1,),
                      synchronize_timing=False, measure_core_forward=False, context_count=8,
                      frame_angles=DEFAULT_FRAME_ANGLES, candidate_subset=None, top_k_count=5):
    if not measure_core_forward or tuple(score_sample_counts) != (1,):
        raise ValueError("the c2f timing pass needs measure_core_forward and K_gen = 1")
    if candidate_subset is not None:
        raise ValueError("the c2f pass defines its own candidate sets (no candidate_subset)")
    if int(top_k_count) <= 0:
        raise ValueError("top_k_count must be positive")
    validate_frame_angles(cond_method, frame_angles)
    cuda, device_type = torch.device(device).type == "cuda", torch.device(device).type

    def clock():
        if cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    start_time = clock() if synchronize_timing else time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    full = filter_frozen_query_candidates(record, audit, room_base)
    if len(full) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    marks = {"start": clock()}
    observation_features = encode_audio_features(retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32))
    marks["observed_encoding"] = clock()
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [metadata], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    marks["context_conditioning"] = clock()
    similarities = np.full(len(full), np.nan, dtype=np.float32)
    generated_count, seconds = 0, {}

    def score(indices, stage):
        nonlocal generated_count
        generation = scoring = 0.0
        for start in range(0, len(indices), candidate_batch_size):
            batch = indices[start:start + candidate_batch_size]
            t0 = clock()
            items = candidate_metadata(metadata, full[batch], record["receiver_global"])
            with torch.amp.autocast(device_type):
                source = _cache_branch(module, items, SOURCE_CONDITIONING_IDS, device, cond_method,
                                       frame_angles=frame_angles)
                dynamic = (_cache_branch(module, items, FA_DYNAMIC_CONDITIONING_IDS, device, cond_method,
                                         frame_angles=frame_angles) if cond_method == "fa_invariant" else None)
            seeds = [candidate_seed(sample_seed, int(record["index"]), int(i), 0) for i in batch]
            generated = generate_rir_batch(module, _repeat_branch(source, 1), context, seeds,
                                           dynamic_branch=_repeat_branch(dynamic, 1) if dynamic else None)
            t1 = clock()
            scores = (encode_audio_features(retrieval, generated) @ observation_features.T).reshape(-1).cpu()
            similarities[batch] = scores.float().numpy()
            t2 = clock()
            generation, scoring, generated_count = generation + t1 - t0, scoring + t2 - t1, generated_count + len(batch)
        seconds[f"{stage}_generation"], seconds[f"{stage}_scoring"] = generation, scoring

    coarse = nested_subset(full)
    score(coarse, "coarse")
    t_selection = clock()
    coarse_scores = np.full(len(full), np.nan)                      # only the coarse rows are scored yet
    coarse_scores[coarse] = query_scores(similarities[coarse][:, None], tau)
    top = top_k(coarse_scores, coarse, int(top_k_count))
    near = np.zeros(len(full), dtype=bool)
    for index in top:
        near |= np.abs(full.astype(np.float64) - full[index].astype(np.float64)).max(axis=1) <= CHEBYSHEV_RADIUS_M + TOLERANCE_M
    refined = np.flatnonzero(near)
    new = np.setdiff1d(refined, coarse)
    seconds["selection"] = clock() - t_selection
    score(new, "refined")
    t_union = clock()
    union = np.union1d(coarse, refined)
    union_scores = query_scores(similarities[union][:, None], tau)
    winner = int(union[int(np.argmax(union_scores))])
    seconds["union_argmax"] = clock() - t_union
    peak = int(torch.cuda.max_memory_allocated(device)) if cuda else 0
    seconds.update(observed_encoding=marks["observed_encoding"] - marks["start"],
                   context_conditioning=marks["context_conditioning"] - marks["observed_encoding"])
    result = _build_query_result(selected=selected, record=record, candidates=full[union],
                                 similarities=torch.from_numpy(similarities[union][:, None]), run_sha256=run_sha256,
                                 elapsed_seconds=time.perf_counter() - start_time, peak_memory_bytes=peak, tau=tau,
                                 random_seed=sample_seed, score_sample_counts=(1,))
    parts = ("observed_encoding", "context_conditioning", "coarse_generation", "coarse_scoring", "selection",
             "refined_generation", "refined_scoring", "union_argmax")
    result["latency_seconds"] = {**{p: float(seconds[p]) for p in parts},
                                 "c2f_total": float(sum(seconds[p] for p in parts)),
                                 "core_forward_total": float(seconds["context_conditioning"] + seconds["coarse_generation"]
                                                             + seconds["refined_generation"])}
    result["latency_protocol"] = {"name": f"exp37_c2f_sequential_kctx{context_count}_kgen1", "context_count": context_count,
                                  "top_k": int(top_k_count), "chebyshev_radius_m": CHEBYSHEV_RADIUS_M,
                                  "observed_encoding_included": True, "candidate_scoring_included": True,
                                  "input_loading_included": False, "candidate_filtering_included": False}
    result["c2f"] = {"full_candidate_count": len(full), "coarse_count": len(coarse), "top_indices": top.tolist(),
                     "refined_count": len(refined), "new_count": len(new), "union_indices": union.tolist(),
                     "generated_count": generated_count, "winner_full_index": winner,
                     "union_indices_sha256": hashlib.sha256(union.astype("<u4").tobytes()).hexdigest()}
    return result, full[union], similarities[union][:, None]
