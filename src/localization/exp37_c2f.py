"""exp_37 coarse-to-fine SEQUENTIAL timing pass (plan_loc_efficiency.md §4, §10 "B2 time axis").

Times the path a deployed c2f search actually runs, end to end, with the bundle's sync placement (a CUDA sync before
every ``perf_counter`` read): the observed RIR encoded ONCE -> the context cache -> coarse = the nested 1.0 m subset
(generation, then AGREE encoding + similarity) -> top-k (score desc, full-grid index asc) + the Chebyshev-0.5 m
neighbourhood -> refined = only the candidates not yet scored (generation, encoding, similarity) -> argmax over
coarse U refined in full-grid order. Seeds stay keyed by the full 0.5 m index. ``c2f_total`` is the c2f L_q; the
generator-only part (context + both generations) is ``core_forward_total``. The c2f accuracy comes from
sequential c2f runs for all three seeds (plan §12; the fix-round-2 c2f-parity choice): this seed-42 timing pass,
whose untimed replay supplies the waveforms, plus the seed-43 / 44 accuracy-only runs ``b2a_c2f_*`` executed
untimed with the same adaptive path (``exp37_artifacts``). The offline readout of the parent 0.5 m runs
(``exp37_grid``) is only a parity cross-check and a sensitivity row, never the accuracy of record.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c; the policy text updated
in the final fix round (fixed after review 3, not re-reviewed (cap)).
"""

from __future__ import annotations

import hashlib
import statistics
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

REPEATS = 3                       # the bundle selector's repeat count (median-of-three per component)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, tuple):
        return all(np.array_equal(x, y) for x, y in zip(a, b))
    return a == b


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

    def repeated(label, work):
        """``work`` run REPEATS times, each between synced clocks; its outputs must agree across the repeats."""
        times, first = [], None
        for _ in range(REPEATS):
            t0 = clock()
            out = work()
            times.append(clock() - t0)
            if first is None:
                first = out
            elif not _same(out, first):
                raise RuntimeError(f"the c2f {label} differs across the selector repeats")
        return first, times

    features, repeats = None, {}
    features, repeats["observed_encoding"] = repeated("observed encoding", lambda: encode_audio_features(
        retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32)))
    t0 = clock()
    with torch.amp.autocast(device_type):
        context = _cache_branch(module, [metadata], FA_CONTEXT_CONDITIONING_IDS if cond_method == "fa_invariant"
                                else CONTEXT_CONDITIONING_IDS, device, cond_method, frame_angles=frame_angles)
    seconds = {"context_conditioning": clock() - t0}
    similarities = np.full(len(full), np.nan, dtype=np.float32)
    generated_count = 0

    def score(indices, stage):
        nonlocal generated_count
        generation, scoring = 0.0, [0.0] * REPEATS
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
            generation += clock() - t0
            scores, times = repeated(f"{stage} scoring", lambda: (
                encode_audio_features(retrieval, generated) @ features.T).reshape(-1).cpu())
            scoring = [a + b for a, b in zip(scoring, times)]
            similarities[batch] = scores.float().numpy()
            generated_count += len(batch)
        seconds[f"{stage}_generation"], repeats[f"{stage}_scoring"] = generation, scoring

    coarse = nested_subset(full)
    score(coarse, "coarse")

    def select():
        coarse_scores = np.full(len(full), np.nan)                      # only the coarse rows are scored yet
        coarse_scores[coarse] = query_scores(similarities[coarse][:, None], tau)
        top = top_k(coarse_scores, coarse, int(top_k_count))
        near = np.zeros(len(full), dtype=bool)
        for index in top:
            near |= np.abs(full.astype(np.float64) - full[index].astype(np.float64)).max(axis=1) \
                <= CHEBYSHEV_RADIUS_M + TOLERANCE_M
        return top, np.flatnonzero(near)

    (top, refined), repeats["selection"] = repeated("top-k / neighbour selection", select)
    new = np.setdiff1d(refined, coarse)
    score(new, "refined")
    union = np.union1d(coarse, refined)
    winner, repeats["union_argmax"] = repeated("union argmax", lambda: int(
        union[int(np.argmax(query_scores(similarities[union][:, None], tau)))]))
    peak = int(torch.cuda.max_memory_allocated(device)) if cuda else 0
    seconds.update({part: float(statistics.median(times)) for part, times in repeats.items()})
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
    result["selector_repeat_seconds"] = {part: [float(t) for t in times] for part, times in repeats.items()}
    result["latency_protocol"] = {"name": f"exp37_c2f_sequential_kctx{context_count}_kgen1", "context_count": context_count,
                                  "top_k": int(top_k_count), "chebyshev_radius_m": CHEBYSHEV_RADIUS_M,
                                  "observed_encoding_included": True, "candidate_scoring_included": True,
                                  "input_loading_included": False, "candidate_filtering_included": False,
                                  "selector_repeat_count": REPEATS,
                                  "selector_statistic": "per-component median over the repeats on the exact generated "
                                                        "tensors (the bundle selector's median-of-three)"}
    union_sha = hashlib.sha256(union.astype("<u4").tobytes()).hexdigest()
    result["parent_candidate_indices_sha256"] = result["candidate_indices_sha256"]
    result["candidate_indices_sha256"] = union_sha
    result["c2f"] = {"full_candidate_count": len(full), "coarse_count": len(coarse), "top_indices": top.tolist(),
                     "refined_count": len(refined), "new_count": len(new), "union_indices": union.tolist(),
                     "generated_count": generated_count, "winner_full_index": winner, "union_indices_sha256": union_sha,
                     "top_k": int(top_k_count), "chebyshev_radius_m": CHEBYSHEV_RADIUS_M}
    return result, full[union], similarities[union][:, None]
