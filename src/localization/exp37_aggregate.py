"""exp_37 B2 / B3 aggregation helpers (plan_loc_efficiency.md §4, §5, §10 "B2 time axis", "Pareto references").

- accuracy: each run's pooled 64-query metrics (``exp37_grid.summarize_readout``), then the mean and the SAMPLE SD
  (ddof 1) over the seeds 42-44 per grid -- localization seed uncertainty; one repeat of timing is not a variance;
- time: s/query = the query-micro mean over the matched queries of L_q = core_forward_total_q + the shape-matched
  AGREE selector term (never ``elapsed_seconds``); the c2f point uses its measured sequential ``c2f_total``;
- the Pareto frontier is computed over the internal FLAC / OrbitRIR points ONLY;
- FewShotRIR and FEM are external reference markers: exp_23 accuracy on the same 64 queries, latency = the bundle
  host's per-query medians restricted to those 64 (FEM: policy-aware, mixed hosts, 7 strict-coverage failures).
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
"""

from __future__ import annotations

import statistics

EXTERNAL = {
    "few_shot_rir": {"metrics_key": "few_shot", "latency_column": "few_shot_rir_median_seconds",
                     "label": "Few-ShotRIR (exp_23 accuracy, bundle host latency)"},
    "fem_omp": {"metrics_key": "fem_omp", "latency_column": "fem_omp_median_seconds",
                "label": "FEM-OMP (exp_23 random-fallback accuracy, policy-aware mixed host latency, {failed} failures)"},
}
EXTERNAL_METRICS = {"median_localization_error_m": "median_localization_error_m", "success_0_5m": "success_rate_at_0_5m",
                    "success_1_0m": "success_rate_at_1_0m",
                    "oracle_normalized_success_0_5m": "resolution_aware_success_rate_at_0_5m"}


def seed_stats(values) -> dict:
    values = [float(v) for v in values]
    return {"n": len(values), "mean": float(statistics.fmean(values)),
            "sd": float(statistics.stdev(values)) if len(values) >= 2 else None, "values": values}


def aggregate_accuracy(summaries_by_seed: dict, metrics) -> dict:
    """``{seed: {grid: summary}}`` -> ``{grid: {metric: seed_stats, "seeds": [...]}}``."""
    seeds = sorted(summaries_by_seed)
    grids = [sorted(summaries_by_seed[seed]) for seed in seeds]
    if any(g != grids[0] for g in grids):
        raise RuntimeError("the seeds do not share the same grids")
    return {grid: {**{m: seed_stats(summaries_by_seed[s][grid][m] for s in seeds) for m in metrics}, "seeds": seeds}
            for grid in grids[0]}


def mean_latency(core: dict, selector: dict, indices) -> float:
    indices = [int(i) for i in indices]
    if not set(indices) <= set(core) or not set(indices) <= set(selector):
        raise RuntimeError("latency coverage mismatch")
    return float(sum(core[i] + selector[i] for i in indices) / len(indices))


def pareto_frontier(points, minimize_y: bool) -> list[int]:
    """Indices of the non-dominated (x = s/query lower is better, y = error lower / success higher) points by x."""
    def dominates(p, q):
        better_y = p[1] <= q[1] if minimize_y else p[1] >= q[1]
        strictly = p[0] < q[0] or (p[1] < q[1] if minimize_y else p[1] > q[1])
        return p[0] <= q[0] and better_y and strictly
    keep = [i for i, q in enumerate(points) if not any(dominates(p, q) for j, p in enumerate(points) if j != i)]
    return sorted(keep, key=lambda i: (points[i][0], points[i][1]))


def external_markers(exp23_summary: dict, latency_rows: list[dict], pilot: dict) -> dict:
    indices = [int(r["index"]) for r in pilot["records"]]
    scope = sorted(int(q["query_index"]) for q in exp23_summary["per_query"])
    if scope != sorted(indices) or int(exp23_summary["query_count"]) != len(indices):
        raise RuntimeError(f"the exp_23 summary is not the {len(indices)} pilot queries")
    by_index = {}
    for row in latency_rows:
        if int(row["query_index"]) in set(indices):
            if int(row["query_index"]) in by_index:
                raise RuntimeError(f"duplicate latency row for query {row['query_index']}")
            by_index[int(row["query_index"])] = row
    if sorted(by_index) != sorted(indices):
        raise RuntimeError(f"the bundle latency rows do not cover the {len(indices)} pilot queries")
    failed = int(exp23_summary["fem_coverage"]["failed_query_count"])
    out = {}
    for name, spec in EXTERNAL.items():
        raw = exp23_summary["metrics"][spec["metrics_key"]]
        out[name] = {"external": True, "label": spec["label"].format(failed=failed),
                     "accuracy": {**raw, **{ours: raw[theirs] for ours, theirs in EXTERNAL_METRICS.items()}},
                     "seconds_per_query": float(sum(float(by_index[i][spec["latency_column"]]) for i in indices)
                                                / len(indices)),
                     "latency_source": f"bundle expected/latency_per_query.csv::{spec['latency_column']}",
                     **({"failed_query_count": failed} if name == "fem_omp" else {})}
    return out
