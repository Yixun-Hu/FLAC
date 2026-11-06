"""exp_37 round B-c: the nested 1.0 m subset and the coarse-to-fine (c2f) offline readout of a 0.5 m run
(plan_loc_efficiency.md §4, §10 "B2 mathematics").

"1.0 m" = the integer-coordinate subset of the origin-anchored 0.5 m candidates with FULL-GRID common random numbers;
c2f = that subset -> top-k (k = 5 registered; 1 / 3 / 10 sensitivity) by descending score then the original full-grid
index -> every 0.5 m candidate within Chebyshev 0.5 m of a top-k point -> argmax over coarse U refined in full-grid
order. Each candidate set gets a FRESH oracle (never the full 0.5 m one). Written by the exp_37 Coder (Claude Opus
5.5, max effort), round B-c.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_grid as grid
from src.localization.pilot import load_pilot_manifest

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
LINE = np.array([[x, 0.0, 1.0] for x in np.arange(0.0, 3.01, 0.5)], dtype=np.float32)   # 0, .5, ..., 3.0


def test_the_nested_subset_is_the_integer_coordinate_rows_in_order():
    assert grid.nested_subset(LINE).tolist() == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="0.5 m lattice"):
        grid.nested_subset(LINE + np.float32(0.25))


def test_top_k_orders_by_descending_score_then_full_grid_index():
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.5, 0.2, 0.5], dtype=np.float32)
    assert grid.top_k(scores, np.array([0, 2, 4, 6]), 2).tolist() == [2, 4]          # tie 0.5: index 2 before 4
    assert grid.top_k(scores, np.array([0, 2, 4, 6]), 10).tolist() == [2, 4, 6, 0]


def test_c2f_refines_the_chebyshev_neighbourhood_and_keeps_full_grid_order():
    cube = np.array([[x, y, z] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0) for z in (1.0, 1.5)], dtype=np.float32)
    scores = np.zeros(len(cube), dtype=np.float32)
    scores[grid.nested_subset(cube)] = np.arange(len(grid.nested_subset(cube)), dtype=np.float32)
    out = grid.c2f_indices(cube, scores, k=1)
    top = out["top"][0]
    assert np.allclose(cube[top], [1.0, 1.0, 1.0])
    near = np.flatnonzero(np.abs(cube - cube[top]).max(axis=1) <= 0.5 + 1e-6)
    assert out["refined"].tolist() == near.tolist()
    assert out["union"].tolist() == sorted(set(out["coarse"]) | set(near))


def test_the_readout_uses_a_fresh_oracle_over_its_own_set():
    truth = np.array([0.5, 0.0, 1.0])
    scores = np.array([0.0, 0.1, 0.9, 0.2, 0.3, 0.1, 0.0], dtype=np.float32)       # argmax at x = 1.0
    full = grid.subset_readout(LINE, scores, truth, np.arange(7))
    nested = grid.subset_readout(LINE, scores, truth, grid.nested_subset(LINE))
    assert full["oracle_error_m"] == 0.0 and nested["oracle_error_m"] == pytest.approx(0.5)
    assert nested["prediction_index_full"] == 2 and nested["localization_error_m"] == pytest.approx(0.5)
    assert nested["excess_error_m"] == 0.0 and nested["oracle_normalized_success_0_5m"] == 1
    assert nested["candidate_count"] == 4


def _committed(arm):
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    return pilot, grid.readout_run(E9 / "pilot_results" / arm, pilot)


@pytest.mark.parametrize("arm", ["vanilla", "fa_bf"])
def test_on_the_committed_exp09_scores_the_full_grid_readout_reproduces_the_runs_k1_metrics(arm):
    pilot, rows = _committed(arm)
    assert set(rows) == {"0.50", "1.00_nested", "c2f_k5", "c2f_k1", "c2f_k3", "c2f_k10"}
    assert len(rows["0.50"]) == 64                       # readout_run itself refuses any mismatch with metrics["1"]
    counts = [r["candidate_count"] for r in rows["1.00_nested"]]
    assert sum(counts) == 6610 and min(counts) >= 6                                     # research report, verified
    for k in (1, 3, 5, 10):
        assert all(c2f["candidate_count"] <= nested["candidate_count"] + 27 * k for c2f, nested in
                   zip(rows[f"c2f_k{k}"], rows["1.00_nested"]))
        assert all(c2f["oracle_error_m"] <= nested["oracle_error_m"] + 1e-12 for c2f, nested in
                   zip(rows[f"c2f_k{k}"], rows["1.00_nested"]))                       # a superset's oracle is no worse


def test_the_summary_pools_the_queries_per_grid():
    _pilot, rows = _committed("vanilla")
    summary = grid.summarize_readout(rows)
    committed = json.loads((E9 / "pilot_results" / "pilot_results.json").read_text())
    assert summary["0.50"]["median_localization_error_m"] == pytest.approx(1.0093, abs=1e-4)   # the paper's K=1 Vanilla
    assert summary["0.50"]["query_count"] == 64 and summary["1.00_nested"]["query_count"] == 64
    assert summary["1.00_nested"]["mean_oracle_error_m"] > summary["0.50"]["mean_oracle_error_m"]
    assert isinstance(committed, dict)


def test_a_run_whose_arrays_disagree_with_its_recorded_metrics_is_refused(tmp_path):
    import shutil
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    copy = tmp_path / "vanilla"
    shutil.copytree(E9 / "pilot_results" / "vanilla", copy)
    first = copy / "queries" / f"query_{int(pilot['records'][0]['index']):05d}.json"
    payload = json.loads(first.read_text())
    payload["metrics"]["1"]["prediction_index"] += 1
    from src.localization.pilot import canonical_sha256
    payload["sha256"] = canonical_sha256({k: v for k, v in payload.items() if k != "sha256"})
    first.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="recorded"):
        grid.readout_run(copy, pilot)


@pytest.mark.parametrize("field, value", [("oracle_error_m", 9.0), ("excess_error_m", 9.0), ("success_0_5m", 7),
                                          ("success_1_0m", 7), ("oracle_normalized_success_0_5m", 7),
                                          ("oracle_normalized_success_1_0m", 7), ("prediction_global", [0.0, 0.0, 0.0]),
                                          ("winning_score", 9.0), ("mean_candidate_score", 9.0)])
def test_every_recorded_k1_field_must_be_reproduced(tmp_path, field, value):
    """Codex NIT 1: the full recorded K=1 block, not only the prediction index and the error."""
    import shutil
    from src.localization.pilot import canonical_sha256
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    copy = tmp_path / "vanilla"
    shutil.copytree(E9 / "pilot_results" / "vanilla", copy)
    first = copy / "queries" / f"query_{int(pilot['records'][0]['index']):05d}.json"
    payload = json.loads(first.read_text())
    payload["metrics"]["1"][field] = value
    payload["sha256"] = canonical_sha256({k: v for k, v in payload.items() if k != "sha256"})
    first.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="recorded"):
        grid.readout_run(copy, pilot)


def test_the_recomputed_block_equals_the_recorded_one_on_real_data():
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    record = pilot["records"][0]
    stem = E9 / "pilot_results" / "fa_bf" / "queries" / f"query_{int(record['index']):05d}"
    result = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz")) as arrays:
        block = grid.recomputed_k1_metrics(arrays["candidates"], arrays["similarities"], result["source_global"], 0.1)
    assert block == result["metrics"]["1"]
