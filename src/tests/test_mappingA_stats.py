"""Placement-level statistics (exp_21, contract E, cycle 10).

The clustering unit is the placement: 36 items of one placement share a room
position, an array, a target source and overlapping context, so item-i.i.d.
intervals would understate the uncertainty. Every oracle here is hand-computed.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_stats as stats  # noqa: E402


def _records(values_by_placement, room="EmptyRoom", metric="C50"):
    records = []
    for placement, values in values_by_placement.items():
        for i, value in enumerate(values):
            records.append({"room": room, "placement_id": placement,
                            "item_id": f"{room}/{placement}/slot{i:02d}",
                            "metrics": {metric: value}})
    return records


def test_within_placement_aggregation_is_the_item_mean():
    records = _records({"p000": [1.0, 2.0, 3.0], "p001": [10.0, 20.0]})
    means, counts = stats.aggregate_within_placement(records, "C50")
    assert means == {("EmptyRoom", "p000"): 2.0, ("EmptyRoom", "p001"): 15.0}
    assert counts[("EmptyRoom", "p000")] == 3


def test_a_missing_metric_is_refused_rather_than_dropped():
    records = _records({"p000": [1.0, float("nan")]})
    with pytest.raises(ValueError) as exc:
        stats.aggregate_within_placement(records, "C50")
    assert "different estimand" in str(exc.value)


def test_the_macro_weights_rooms_equally_not_placements():
    """EmptyRoom contributes 3 placements averaging 2, FurnishedRoom 1 placement at
    10: the macro is (2 + 10)/2 = 6, not the placement mean 4."""
    means = {("EmptyRoom", "p000"): 1.0, ("EmptyRoom", "p001"): 2.0,
             ("EmptyRoom", "p002"): 3.0, ("FurnishedRoom", "p003"): 10.0}
    out = stats.macro_two_room(means)
    assert out["rooms"] == {"EmptyRoom": 2.0, "FurnishedRoom": 10.0}
    assert out["macro"] == 6.0
    assert out["n_placements"] == {"EmptyRoom": 3, "FurnishedRoom": 1}
    assert out["macro"] != np.mean([1.0, 2.0, 3.0, 10.0])


def test_the_bootstrap_collapses_when_every_placement_agrees():
    means = {("EmptyRoom", f"p{i:03d}"): 5.0 for i in range(4)}
    means.update({("FurnishedRoom", f"p1{i:02d}"): 7.0 for i in range(4)})
    out = stats.cluster_bootstrap(means, n_resamples=200)
    assert out["point"] == 6.0
    assert out["ci_low"] == out["ci_high"] == 6.0
    assert out["unit"] == "placement" and out["stratified_by"] == "room"


def test_the_bootstrap_interval_brackets_the_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    means = {("EmptyRoom", f"p{i:03d}"): float(v)
             for i, v in enumerate(rng.normal(5.0, 1.0, 8))}
    means.update({("FurnishedRoom", f"q{i:03d}"): float(v)
                  for i, v in enumerate(rng.normal(7.0, 1.0, 8))})
    first = stats.cluster_bootstrap(means, n_resamples=500)
    second = stats.cluster_bootstrap(means, n_resamples=500)
    assert first == second                                  # stable seed
    assert first["ci_low"] < first["point"] < first["ci_high"]


def test_the_paired_test_is_exact_and_hand_checkable():
    """Four placements, every difference +1. Of the 2^4 = 16 sign assignments only
    all-plus and all-minus give |mean| >= 1, so the two-sided p is 2/16 = 0.125."""
    arm_a = {("EmptyRoom", f"p{i:03d}"): 1.0 for i in range(4)}
    arm_b = {("EmptyRoom", f"p{i:03d}"): 0.0 for i in range(4)}
    out = stats.paired_randomization(arm_a, arm_b)
    assert out["exact"] is True
    assert out["n_assignments"] == 16
    assert out["observed_difference"] == 1.0
    assert out["p_value"] == pytest.approx(2 / 16)


def test_the_paired_test_finds_no_effect_when_differences_cancel():
    arm_a = {("EmptyRoom", "p000"): 1.0, ("EmptyRoom", "p001"): 0.0}
    arm_b = {("EmptyRoom", "p000"): 0.0, ("EmptyRoom", "p001"): 1.0}
    out = stats.paired_randomization(arm_a, arm_b)
    assert out["observed_difference"] == 0.0
    assert out["p_value"] == 1.0


def test_the_paired_test_refuses_unpaired_placements():
    with pytest.raises(ValueError) as exc:
        stats.paired_randomization({("EmptyRoom", "p000"): 1.0},
                                   {("EmptyRoom", "p001"): 1.0})
    assert "not a paired test" in str(exc.value)


def test_a_large_design_falls_back_to_sampling_and_says_so():
    arm_a = {("EmptyRoom", f"p{i:03d}"): float(i) for i in range(32)}
    arm_b = {("EmptyRoom", f"p{i:03d}"): 0.0 for i in range(32)}
    out = stats.paired_randomization(arm_a, arm_b, n_resamples=500)
    assert out["exact"] is False
    assert out["n_assignments"] == 500
    assert out["n_placements"] == 32


def test_seed_variability_is_reported_separately_from_the_interval():
    out = stats.seed_variability({42: 1.0, 43: 1.2, 44: 0.8, 45: 1.1, 46: 0.9})
    assert out["n_seeds"] == 5
    assert out["mean"] == pytest.approx(1.0)
    assert out["sd"] == pytest.approx(np.std([1.0, 1.2, 0.8, 1.1, 0.9], ddof=1))
    assert "not an interval" in out["note"]


def test_the_full_pipeline_runs_from_item_records():
    records = (_records({f"p{i:03d}": [1.0 + 0.1 * i] * 36 for i in range(4)}) +
               _records({f"q{i:03d}": [2.0 + 0.1 * i] * 36 for i in range(4)},
                        room="FurnishedRoom"))
    means, counts = stats.aggregate_within_placement(records, "C50")
    assert sum(counts.values()) == 8 * 36
    macro = stats.macro_two_room(means)
    assert macro["macro"] == pytest.approx((1.15 + 2.15) / 2)
    interval = stats.cluster_bootstrap(means, n_resamples=200)
    assert interval["n_placements"] == {"EmptyRoom": 4, "FurnishedRoom": 4}
