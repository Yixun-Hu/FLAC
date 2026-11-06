"""exp_37 candidate-density readouts of a 0.5 m run (plan_loc_efficiency.md §4, §10 "B2 mathematics").

- ``nested_subset``: the rows of an origin-anchored 0.5 m candidate array that lie on the 1.0 m lattice (every
  coordinate an integer). Reading a 0.5 m run's scores on these rows is the "nested 1.0 m subset with full-grid
  common random numbers" (the noise seeds stay keyed by the 0.5 m index) -- not a native reindexed 1.0 m rerun.
- ``c2f_indices``: coarse = the nested subset; top-k by descending score, ties by the original full-grid index;
  refined = every 0.5 m candidate of the query within Chebyshev distance 0.5 m of a top-k point (validity =
  membership in the 0.5 m set; <= 26 neighbours each); union in full-grid order (argmax ties inherited from it).
- ``subset_readout``: the runner's K=1 score, argmax over the subset, the runner's metrics with a FRESH oracle over
  that candidate set (the c2f resolution-aware metric never reuses the full 0.5 m oracle).
k = 5 is the registered c2f; k in {1, 3, 10} is a labelled post-primary sensitivity only.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.localization.reporting import _summarize_metric_rows
from src.localization.runner import completed_query_result, verify_hashed_payload
from src.localization.scoring import localization_metrics, log_mean_exp_scores, stable_argmax

REGISTERED_K = 5
SENSITIVITY_K = (1, 3, 10)
CHEBYSHEV_RADIUS_M = 0.5
TOLERANCE_M = 1e-4


def nested_subset(candidates) -> np.ndarray:
    points = np.asarray(candidates, dtype=np.float64)
    if not np.all(np.abs(2.0 * points - np.round(2.0 * points)) <= 2.0 * TOLERANCE_M):
        raise ValueError("the candidates are not on the origin-anchored 0.5 m lattice")
    return np.flatnonzero(np.all(np.abs(points - np.round(points)) <= TOLERANCE_M, axis=1))


def top_k(scores, subset, k: int) -> np.ndarray:
    subset = np.asarray(subset)
    order = np.lexsort((subset, -np.asarray(scores, dtype=np.float64)[subset]))   # score desc, then index asc
    return subset[order[:k]]


def c2f_indices(candidates, scores, k: int = REGISTERED_K, radius: float = CHEBYSHEV_RADIUS_M) -> dict:
    points = np.asarray(candidates, dtype=np.float64)
    coarse = nested_subset(points)
    top = top_k(scores, coarse, k)
    near = np.zeros(len(points), dtype=bool)
    for index in top:
        near |= np.abs(points - points[index]).max(axis=1) <= radius + TOLERANCE_M
    refined = np.flatnonzero(near)
    return {"coarse": coarse, "top": top, "refined": refined, "union": np.union1d(coarse, refined)}


def query_scores(similarities, tau: float) -> np.ndarray:
    """The runner's K=1 candidate score (``log_mean_exp_scores`` over the first sample)."""
    return log_mean_exp_scores(torch.as_tensor(np.asarray(similarities)), (1,), tau=tau)[1].numpy()


def subset_readout(candidates, scores, truth, indices) -> dict:
    indices = np.asarray(indices)
    local = stable_argmax(np.asarray(scores)[indices])
    metrics = localization_metrics(np.asarray(candidates)[indices], truth, local)
    return {**metrics, "prediction_index_full": int(indices[local]), "candidate_count": int(len(indices))}


def readout_run(run_dir: Path, pilot: dict) -> dict:
    """Per grid, per pilot query: the readout rows (with ``room``) of one complete 0.5 m run; the full-grid readout
    must reproduce each query's recorded K=1 metrics, else the run is refused."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run_manifest.json").read_text())
    verify_hashed_payload(run, "run manifest")
    identity = run["identity"]
    if identity["pilot_manifest_sha256"] != pilot["sha256"] or identity["query_indices"] != [
            int(r["index"]) for r in pilot["records"]]:
        raise RuntimeError(f"{run_dir} is not a complete run of this pilot manifest")
    grids = ["0.50", "1.00_nested", f"c2f_k{REGISTERED_K}"] + [f"c2f_k{k}" for k in SENSITIVITY_K]
    rows = {name: [] for name in grids}
    for record in pilot["records"]:
        result = completed_query_result(run_dir, query_index=int(record["index"]), query_id=record["query_id"],
                                        candidate_count=int(record["candidate_count"]), run_sha256=run["sha256"],
                                        score_sample_counts=tuple(identity["score_sample_counts"]))
        if result is None:
            raise RuntimeError(f"{run_dir}: query {record['index']} is incomplete")
        with np.load(run_dir / result["arrays_file"]) as arrays:
            candidates, similarities = arrays["candidates"], arrays["similarities"]
        scores, truth = query_scores(similarities, float(identity["tau"])), np.asarray(result["source_global"])
        full = subset_readout(candidates, scores, truth, np.arange(len(candidates)))
        recorded = result["metrics"]["1"]
        if full["prediction_index"] != recorded["prediction_index"] or full["localization_error_m"] != recorded[
                "localization_error_m"]:
            raise RuntimeError(f"{run_dir}: query {record['index']} arrays disagree with its recorded K=1 metrics")
        sets = {"0.50": np.arange(len(candidates)), "1.00_nested": nested_subset(candidates)}
        sets.update({f"c2f_k{k}": c2f_indices(candidates, scores, k)["union"] for k in (REGISTERED_K, *SENSITIVITY_K)})
        for name in grids:
            rows[name].append({**subset_readout(candidates, scores, truth, sets[name]), "room": result["room"],
                               "query_index": int(record["index"])})
    return rows


def summarize_readout(rows: dict) -> dict:
    """The exp_09 pooled metric set per grid (median / mean error, oracle, excess, SR@0.5 / 1.0, resolution-aware
    SR) plus the mean candidate count."""
    return {name: {**_summarize_metric_rows(items),
                   "mean_candidate_count": float(np.mean([r["candidate_count"] for r in items])),
                   "summed_candidate_count": int(sum(r["candidate_count"] for r in items))}
            for name, items in rows.items()}
