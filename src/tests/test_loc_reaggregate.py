"""Tests for ``src.localization.reaggregate`` (exp_18, r4 item 2d).

R1 selects tau offline from the logged per-sample similarities -- no regeneration
-- so that path must be reviewed code with tests, not a one-off ``python -c``.
The registered selection is LME at K'=8, objective = pooled MEAN e_loc,
smallest-tau tie-break (plan §2.5).
"""
import json
import math

import numpy as np
import pytest
import torch

from src.localization import reaggregate as ra
from src.localization.scoring import aggregate, localization_error, predict_index


# --------------------------------------------------------------------------- #
# the exact float32 codec (moved here so the driver and the offline path share one)
# --------------------------------------------------------------------------- #
def test_sims_codec_round_trips_bitwise():
    g = torch.Generator().manual_seed(3)
    sims = (torch.rand(4, 8, generator=g) * 2 - 1).float()
    assert torch.equal(ra.decode_sims(ra.encode_sims(sims)), sims)
    assert isinstance(ra.encode_sims(sims)[0][0], str)


def test_scores_codec_round_trips():
    scores = torch.tensor([0.1, -0.7, 1.0], dtype=torch.float32)
    assert torch.equal(ra.decode_scores(ra.encode_sims(scores.unsqueeze(0))[0]), scores)


# --------------------------------------------------------------------------- #
# fixtures: rows whose optimum is known by construction
# --------------------------------------------------------------------------- #
_CAND = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]


def _row(query_id, room_id, sims, gt_index=0, available=None):
    sims = torch.tensor(sims, dtype=torch.float32)
    return {"query_id": query_id, "room_id": room_id,
            "candidate_xyz_world": _CAND, "gt_index": gt_index,
            "gt_xyz_world": _CAND[gt_index], "candidate_nodes": [0, 1, 2],
            "candidate_available": available or [True] * len(_CAND),
            "sims_hex": ra.encode_sims(sims), "n_samples": sims.shape[1]}


def _rows():
    # candidate 0 (GT) wins on the K=4 mean; candidate 1 wins on the first sample
    return [
        _row("q0", "A/A_idx_0", [[0.1, 0.9, 0.9, 0.9], [0.8, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]),
        _row("q1", "A/A_idx_0", [[0.2, 0.8, 0.8, 0.8], [0.7, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]]),
        _row("q2", "B/B_idx_1", [[0.3, 0.7, 0.7, 0.7], [0.6, 0.3, 0.3, 0.3], [0.0, 0.0, 0.0, 0.0]]),
    ]


# --------------------------------------------------------------------------- #
# recompute_row
# --------------------------------------------------------------------------- #
def test_recompute_row_matches_direct_scoring_calls():
    row = _rows()[0]
    sims = ra.decode_sims(row["sims_hex"])
    for method, tau in (("lme", 0.05), ("mean", None), ("max", None)):
        scores = aggregate(sims, method, tau)
        expected_index = predict_index(scores)
        got = ra.recompute_row(row, method=method, tau=tau, k_prime=4)
        assert got["pred_index"] == expected_index
        assert got["e_loc"] == pytest.approx(
            localization_error(_CAND[expected_index], row["gt_xyz_world"]))
        assert got["top1"] == (1.0 if expected_index == row["gt_index"] else 0.0)


def test_recompute_row_uses_the_first_k_prime_samples_in_generation_order():
    row = _rows()[0]
    first_only = ra.recompute_row(row, method="mean", tau=None, k_prime=1)
    all_four = ra.recompute_row(row, method="mean", tau=None, k_prime=4)
    assert first_only["pred_index"] == 1                 # sample 0 favours candidate 1
    assert all_four["pred_index"] == 0                   # the K=4 mean favours the GT
    sims = ra.decode_sims(row["sims_hex"])
    assert torch.equal(ra.decode_sims(row["sims_hex"])[:, :2], sims[:, :2])


def test_recompute_row_refuses_a_k_prime_beyond_the_logged_samples():
    with pytest.raises(ValueError):
        ra.recompute_row(_rows()[0], method="mean", tau=None, k_prime=8)


def test_recompute_row_respects_availability():
    row = _row("q", "A/A_idx_0", [[0.1], [0.9], [0.5]], available=[True, False, True])
    assert ra.recompute_row(row, method="max", tau=None, k_prime=1)["pred_index"] == 2


# --------------------------------------------------------------------------- #
# sweep + registered selection
# --------------------------------------------------------------------------- #
def test_sweep_covers_the_registered_grid_without_meaningless_tau_duplicates():
    results = ra.sweep(_rows(), k_primes=(1, 2, 4))
    lme = [r for r in results if r["method"] == "lme"]
    assert len(lme) == len(ra.DEFAULT_TAUS) * 3
    for method in ("mean", "max"):
        entries = [r for r in results if r["method"] == method]
        assert len(entries) == 3                          # tau is not a parameter of these
        assert all(entry["tau"] is None for entry in entries)
    assert {r["k_prime"] for r in results} == {1, 2, 4}
    assert all(r["n_queries"] == 3 for r in results)


def test_sweep_reports_pooled_mean_and_median_from_summarize():
    from src.localization.scoring import summarize
    results = ra.sweep(_rows(), taus=(0.05,), methods=("mean",), k_primes=(4,))
    entry = results[0]
    records = [{"query_id": row["query_id"], "room_id": row["room_id"],
                "e_loc": ra.recompute_row(row, "mean", None, 4)["e_loc"],
                "top1": ra.recompute_row(row, "mean", None, 4)["top1"]} for row in _rows()]
    expected = summarize(records)
    assert entry["pooled_mean_e_loc"] == pytest.approx(expected["pooled"]["mean_e_loc"])
    assert entry["pooled_median_e_loc"] == pytest.approx(expected["pooled"]["median_e_loc"])
    assert entry["top1"] == pytest.approx(expected["pooled"]["top1"])


def test_select_registered_minimizes_pooled_mean_error_over_lme_at_k8():
    results = [
        {"method": "lme", "tau": 0.02, "k_prime": 8, "pooled_mean_e_loc": 2.0},
        {"method": "lme", "tau": 0.05, "k_prime": 8, "pooled_mean_e_loc": 1.0},
        {"method": "lme", "tau": 0.10, "k_prime": 8, "pooled_mean_e_loc": 3.0},
        {"method": "mean", "tau": None, "k_prime": 8, "pooled_mean_e_loc": 0.1},
        {"method": "lme", "tau": 0.02, "k_prime": 4, "pooled_mean_e_loc": 0.2},
    ]
    chosen = ra.select_registered(results)
    assert chosen["method"] == "lme" and chosen["k_prime"] == 8 and chosen["tau"] == 0.05


def test_select_registered_breaks_ties_towards_the_smallest_tau():
    results = [{"method": "lme", "tau": tau, "k_prime": 8, "pooled_mean_e_loc": 1.0}
               for tau in (0.5, 0.2, 0.05, 0.02)]
    assert ra.select_registered(results)["tau"] == 0.02


def test_select_registered_refuses_when_the_registered_cell_is_absent():
    with pytest.raises(ValueError):
        ra.select_registered([{"method": "mean", "tau": None, "k_prime": 8,
                               "pooled_mean_e_loc": 1.0}])


# --------------------------------------------------------------------------- #
# end to end over JSONL row files
# --------------------------------------------------------------------------- #
def test_reaggregate_reads_multiple_row_files_and_reports_the_selection(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    with open(first, "w") as handle:
        for row in _rows()[:2]:
            handle.write(json.dumps(row) + "\n")
    with open(second, "w") as handle:
        handle.write(json.dumps(_rows()[2]) + "\n")

    report = ra.reaggregate([str(first), str(second)], k_primes=(1, 2, 4))
    assert report["n_rows"] == 3 and report["row_files"] == [str(first), str(second)]
    assert report["k_primes_evaluated"] == [1, 2, 4]
    assert len(report["sweep"]) == len(ra.DEFAULT_TAUS) * 3 + 2 * 3
    selected = report["selected"]
    assert selected["method"] == "lme" and selected["k_prime"] == 4
    assert selected["objective"] == "pooled_mean_e_loc"
    assert report["registered_rule"].startswith("LME")


def test_reaggregate_refuses_rows_with_differing_sample_counts(tmp_path):
    path = tmp_path / "mixed.jsonl"
    with open(path, "w") as handle:
        handle.write(json.dumps(_rows()[0]) + "\n")
        handle.write(json.dumps(_row("q9", "A/A_idx_0", [[0.1], [0.2], [0.3]])) + "\n")
    with pytest.raises(ValueError):
        ra.reaggregate([str(path)], k_primes=(1, 4))


def test_sweep_skips_k_primes_beyond_the_logged_samples(tmp_path):
    """Rows from a K=4 run cannot answer K'=8; the sweep evaluates what the rows
    support and the report says which K' were actually run."""
    results = ra.sweep(_rows(), taus=(0.05,), methods=("mean",), k_primes=(1, 4, 8))
    assert {r["k_prime"] for r in results} == {1, 4}
    with pytest.raises(ValueError):
        ra.sweep(_rows(), taus=(0.05,), methods=("mean",), k_primes=(8,))

    path = tmp_path / "rows.jsonl"
    with open(path, "w") as handle:
        for row in _rows():
            handle.write(json.dumps(row) + "\n")
    report = ra.reaggregate([str(path)])
    assert report["k_primes_requested"] == [1, 2, 4, 8]
    assert report["k_primes_evaluated"] == [1, 2, 4]
    assert report["selected"]["k_prime"] == 4          # largest supported


def test_reaggregate_is_deterministic(tmp_path):
    path = tmp_path / "rows.jsonl"
    with open(path, "w") as handle:
        for row in _rows():
            handle.write(json.dumps(row) + "\n")
    assert ra.reaggregate([str(path)], k_primes=(2,)) == ra.reaggregate([str(path)], k_primes=(2,))
