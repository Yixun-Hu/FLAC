"""exp_37 round B-c: pure aggregation helpers for B2 / B3 (plan §4, §5, §10): seed mean ± SD (sample SD, ddof 1),
the query-micro s/query of L_q = core + selector, the Pareto frontier over the internal (FLAC / OrbitRIR) points only,
and the FewShotRIR / FEM reference markers (exp_23 accuracy on the same 64 queries; the bundle host's per-query
latencies restricted to those 64). Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import csv
import json
from pathlib import Path

import pytest

from src.localization import exp37_aggregate as agg
from src.localization.pilot import load_pilot_manifest

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
EXP23 = REPO / "worklog/worklog_yixun/exp_23_five_method_64_random_fallback/summary.json"
CSV = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility/expected/latency_per_query.csv"


def test_seed_stats_use_the_sample_sd():
    stats = agg.seed_stats([1.0, 2.0, 3.0])
    assert stats == {"n": 3, "mean": 2.0, "sd": 1.0, "values": [1.0, 2.0, 3.0]}
    assert agg.seed_stats([4.0])["sd"] is None


def test_accuracy_is_aggregated_per_grid_over_seeds():
    summaries = {seed: {"0.50": {"median_localization_error_m": m, "success_1_0m": s}}
                 for seed, m, s in ((42, 1.0, 0.5), (43, 1.2, 0.4), (44, 1.1, 0.6))}
    out = agg.aggregate_accuracy(summaries, metrics=("median_localization_error_m", "success_1_0m"))
    assert out["0.50"]["median_localization_error_m"]["mean"] == pytest.approx(1.1)
    assert out["0.50"]["success_1_0m"]["sd"] == pytest.approx(0.1) and out["0.50"]["seeds"] == [42, 43, 44]
    with pytest.raises(RuntimeError, match="grids"):
        agg.aggregate_accuracy({42: {"0.50": {}}, 43: {"0.75": {}}}, metrics=())


def test_the_query_micro_mean_of_core_plus_selector():
    assert agg.mean_latency({1: 1.0, 2: 3.0}, {1: 0.5, 2: 0.5, 3: 9.0}, [1, 2]) == pytest.approx(2.5)
    with pytest.raises(RuntimeError, match="coverage"):
        agg.mean_latency({1: 1.0}, {1: 0.5}, [1, 2])


def test_the_frontier_keeps_the_non_dominated_points():
    points = [(1.0, 3.0), (2.0, 2.0), (3.0, 2.5), (4.0, 1.0), (2.0, 4.0)]
    assert agg.pareto_frontier(points, minimize_y=True) == [0, 1, 3]
    assert agg.pareto_frontier([(1.0, 0.2), (2.0, 0.5), (3.0, 0.4)], minimize_y=False) == [0, 1]


def test_the_reference_markers_come_from_exp23_and_the_bundle_on_the_64_queries():
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    markers = agg.external_markers(json.loads(EXP23.read_text()), list(csv.DictReader(CSV.open())), pilot)
    few, fem = markers["few_shot_rir"], markers["fem_omp"]
    assert few["accuracy"]["median_localization_error_m"] == pytest.approx(1.5173, abs=1e-4)
    assert fem["accuracy"]["success_rate_at_1_0m"] == 0.5 and fem["failed_query_count"] == 7
    rows = [r for r in csv.DictReader(CSV.open()) if int(r["query_index"]) in {r2["index"] for r2 in pilot["records"]}]
    assert len(rows) == 64
    assert few["seconds_per_query"] == pytest.approx(sum(float(r["few_shot_rir_median_seconds"]) for r in rows) / 64)
    assert "bundle host" in few["label"] and "mixed host" in fem["label"] and "7 failures" in fem["label"]
    assert few["external"] is True and fem["external"] is True


def test_markers_refuse_another_query_scope():
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    summary = json.loads(EXP23.read_text())
    summary["per_query"] = summary["per_query"][:-1]
    with pytest.raises(RuntimeError, match="64"):
        agg.external_markers(summary, list(csv.DictReader(CSV.open())), pilot)
