"""exp_22 R2 -- the paired cross-arm comparison on the mesh grid (plan r2 §1, §3 R2-A).

Two halves, tested separately:

* the STATISTICS are pure functions of authenticated per-query records, and every
  expected number here is computed BY HAND or by numpy directly in the test --
  never by calling the code under test: room-first endpoints (the mean across the
  registered rooms of the within-room median ``e_loc`` and of ``success_raw@1.0``),
  paired room deltas (treatment - P1), ONE shared room-resample matrix, bootstrap
  tail p-values with draws exactly 0 counted as crossing and no finite-sample
  correction, BF one-sided / YAW two-sided, Holm over exactly four tests, and the
  exhaustive verdict classes (Codex B1/B2);
* the ADMISSION and PAIRING take those records only from the report's admission
  path, and refuse to pair runs that differ on anything outside an explicit
  whitelist (Codex B3/H1).
"""
import json
import os

import numpy as np
import pytest

from src.localization import meshgrid_crossarm as mx
from src.localization import meshgrid_report as mr


# --------------------------------------------------------------------------- #
# the registered procedure, as constants
# --------------------------------------------------------------------------- #
def test_the_registered_procedure_is_pinned():
    assert mx.CONTROL_ARM == "P1" and mx.TREATMENT_ARMS == ("BF", "YAW")
    assert (mx.PRIMARY_AGGREGATOR, mx.PRIMARY_K) == ("lme", 8)
    assert mx.PRIMARY_ENDPOINTS == ("median_e_loc", "success_raw@1.0")
    assert mx.SIDEDNESS == {"BF": "one_sided", "YAW": "two_sided"}
    assert mx.IMPROVEMENT == {"median_e_loc": -1, "success_raw@1.0": +1}
    assert mx.FAMILY_ALPHA == 0.05
    assert [mx.family_label(arm, stat) for arm, stat in mx.REGISTERED_FAMILY] == [
        "BF_vs_P1_median_e_loc_K8", "BF_vs_P1_success_raw@1.0_K8",
        "YAW_vs_P1_median_e_loc_K8", "YAW_vs_P1_success_raw@1.0_K8"]
    # the report's registered bootstrap, reused -- not a second convention
    assert (mx.BOOTSTRAP_SEED, mx.BOOTSTRAP_N) == (mr.BOOTSTRAP_SEED, mr.BOOTSTRAP_N)
    assert (mx.BOOTSTRAP_SEED, mx.BOOTSTRAP_N) == (20260825, 10000)


# --------------------------------------------------------------------------- #
# room-first endpoints and paired room deltas (B1)
# --------------------------------------------------------------------------- #
def _results(arm_rows):
    """``[(room, e_loc), ...]`` -> report-shaped authenticated results (K = 8, lme)."""
    out = []
    for index, (room, e_loc) in enumerate(arm_rows):
        block = {"e_loc": float(e_loc), "e_excess": max(0.0, float(e_loc) - 0.1)}
        out.append({"query_id": f"{index}|q", "room_id": room, "position": index,
                    "by": {"lme": {8: block, 1: block, 4: block}}})
    return out


P1_ROWS = [("R1", 1.0), ("R1", 2.0), ("R1", 4.0), ("R2", 0.5), ("R2", 3.0)]
YAW_ROWS = [("R1", 0.5), ("R1", 1.0), ("R1", 5.0), ("R2", 0.2), ("R2", 0.4)]


def test_the_endpoints_are_room_first_by_hand():
    p1 = mx.room_statistics(_results(P1_ROWS))
    yaw = mx.room_statistics(_results(YAW_ROWS))
    # within-room medians: P1 R1 median(1,2,4)=2, R2 median(0.5,3)=1.75;
    # YAW R1 median(0.5,1,5)=1, R2 median(0.2,0.4)=0.3
    assert (p1["R1"]["median_e_loc"], p1["R2"]["median_e_loc"]) == (2.0, 1.75)
    assert (yaw["R1"]["median_e_loc"], yaw["R2"]["median_e_loc"]) == (1.0, pytest.approx(0.3))
    # success_raw@1.0 = mean(e_loc <= 1.0): P1 1/3, 1/2; YAW 2/3, 1
    assert p1["R1"]["success_raw@1.0"] == pytest.approx(1 / 3)
    assert yaw["R2"]["success_raw@1.0"] == 1.0
    rooms, delta_e = mx.paired_room_deltas(yaw, p1, "median_e_loc")
    assert rooms == ["R1", "R2"]
    assert delta_e.tolist() == pytest.approx([-1.0, -1.45])
    assert float(delta_e.mean()) == pytest.approx(-1.225)         # (-1 - 1.45) / 2
    _rooms, delta_s = mx.paired_room_deltas(yaw, p1, "success_raw@1.0")
    assert float(delta_s.mean()) == pytest.approx(5 / 12)        # (1/3 + 1/2) / 2


def test_the_endpoint_is_not_a_pooled_or_per_query_median():
    p1 = mx.room_statistics(_results(P1_ROWS))
    yaw = mx.room_statistics(_results(YAW_ROWS))
    _rooms, delta_e = mx.paired_room_deltas(yaw, p1, "median_e_loc")
    pooled = np.median([e for _r, e in YAW_ROWS]) - np.median([e for _r, e in P1_ROWS])
    per_query = np.median([a[1] - b[1] for a, b in zip(YAW_ROWS, P1_ROWS)])
    assert float(delta_e.mean()) != pytest.approx(pooled)       # -1.0 vs -1.225
    assert float(delta_e.mean()) != pytest.approx(per_query)    # -0.5 vs -1.225


def test_pairing_rooms_must_be_the_same_set():
    p1 = mx.room_statistics(_results(P1_ROWS))
    yaw = mx.room_statistics(_results(YAW_ROWS[:3]))            # R1 only
    with pytest.raises(ValueError, match="rooms"):
        mx.paired_room_deltas(yaw, p1, "median_e_loc")


# --------------------------------------------------------------------------- #
# the shared matrix, the tails and the zero-crossing rule (B2)
# --------------------------------------------------------------------------- #
DRAWS = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2], [0, 0, 1]])


def test_the_tails_are_the_registered_ones_by_hand():
    deltas = np.array([-1.0, 1.0, 0.0])
    # resampled means: -1, 1, 0, 0, -1/3  ->  P(<=0) = 4/5, P(>=0) = 3/5
    e_bf = mx.endpoint_test(deltas, DRAWS, "BF", "median_e_loc")
    s_bf = mx.endpoint_test(deltas, DRAWS, "BF", "success_raw@1.0")
    e_yaw = mx.endpoint_test(deltas, DRAWS, "YAW", "median_e_loc")
    assert e_bf["p_raw"] == pytest.approx(3 / 5)        # one-sided: P(draw >= 0)
    assert s_bf["p_raw"] == pytest.approx(4 / 5)        # one-sided: P(draw <= 0)
    assert e_yaw["p_raw"] == 1.0                        # 2 * min(4/5, 3/5) = 1.2 -> capped
    assert e_bf["delta"] == pytest.approx(0.0)
    assert e_bf["n_boot"] == 5 and e_bf["n_rooms"] == 3


def test_draws_exactly_zero_count_as_crossing_with_no_finite_sample_correction():
    zero = mx.endpoint_test(np.zeros(3), DRAWS, "BF", "median_e_loc")
    assert zero["p_raw"] == 1.0                         # every draw is exactly 0
    clean = mx.endpoint_test(np.array([-1.0, -2.0, -3.0]), DRAWS, "BF", "median_e_loc")
    assert clean["p_raw"] == 0.0                        # NOT (0 + 1) / (5 + 1)
    two = mx.endpoint_test(np.array([-1.0, -2.0, -3.0]), DRAWS, "YAW", "median_e_loc")
    assert two["p_raw"] == 0.0
    up = mx.endpoint_test(np.array([1.0, 2.0, 0.0]), DRAWS, "BF", "success_raw@1.0")
    # resampled means 1, 2, 0, 1, 4/3 -> P(<=0) = 1/5 (the exact zero counts)
    assert up["p_raw"] == pytest.approx(1 / 5)


def test_the_confidence_interval_is_the_descriptive_percentile_of_the_same_draws():
    deltas = np.array([-1.0, 1.0, 0.0])
    test = mx.endpoint_test(deltas, DRAWS, "YAW", "median_e_loc")
    boot = np.array([-1.0, 1.0, 0.0, 0.0, -1 / 3])
    low, high = np.percentile(boot, [2.5, 97.5], method="linear")
    assert test["ci95"] == pytest.approx([low, high])
    assert test["ci_role"].startswith("DESCRIPTIVE")


def test_the_shared_matrix_is_the_registered_rng_and_is_deterministic():
    draws = mx.shared_draws(16)
    expected = np.random.default_rng(20260825).integers(0, 16, size=(10000, 16))
    assert np.array_equal(draws, expected)
    assert np.array_equal(draws, mx.shared_draws(16))
    assert mx.draws_digest(draws) == mx.draws_digest(expected)


# --------------------------------------------------------------------------- #
# Holm over exactly four, and the exhaustive verdicts (B2)
# --------------------------------------------------------------------------- #
def _tests(p_values, deltas):
    out = []
    for (arm, stat), p_raw, delta in zip(mx.REGISTERED_FAMILY, p_values, deltas):
        out.append({"label": mx.family_label(arm, stat), "arm": arm, "stat": stat,
                    "p_raw": p_raw, "delta": delta})
    return out


def test_holm_by_hand_over_exactly_the_four_registered_tests():
    tests = _tests([0.01, 0.04, 0.03, 0.20], [-0.1, 0.02, -0.05, 0.01])
    family = mx.holm_family(tests)
    by_label = {t["label"]: t for t in family["tests"]}
    # sorted: 0.01 x4 = 0.04 (reject) | 0.03 x3 = 0.09 (stop) | 0.04 x2 = 0.08 -> 0.09 |
    # 0.20 x1 = 0.20
    assert by_label["BF_vs_P1_median_e_loc_K8"]["p_adjusted"] == pytest.approx(0.04)
    assert by_label["YAW_vs_P1_median_e_loc_K8"]["p_adjusted"] == pytest.approx(0.09)
    assert by_label["BF_vs_P1_success_raw@1.0_K8"]["p_adjusted"] == pytest.approx(0.09)
    assert by_label["YAW_vs_P1_success_raw@1.0_K8"]["p_adjusted"] == pytest.approx(0.20)
    assert [label for label, t in sorted(by_label.items()) if t["rejected"]] == [
        "BF_vs_P1_median_e_loc_K8"]
    with pytest.raises(ValueError, match="exactly"):
        mx.holm_family(tests[:3])
    extra = tests + [dict(tests[0], label="BF_vs_P1_success_raw@0.5_K8")]
    with pytest.raises(ValueError, match="exactly"):
        mx.holm_family(extra)


def _verdict_tests(arm, reject_e, reject_s, delta_e, delta_s):
    return ({"stat": "median_e_loc", "rejected": reject_e, "delta": delta_e, "arm": arm},
            {"stat": "success_raw@1.0", "rejected": reject_s, "delta": delta_s, "arm": arm})


@pytest.mark.parametrize("reject_e, reject_s, verdict", [
    (True, True, "SUPERIOR"), (True, False, "PARTIAL"), (False, True, "PARTIAL"),
    (False, False, "NOT SUPERIOR")])
def test_every_bf_verdict_class(reject_e, reject_s, verdict):
    assert mx.bf_verdict(*_verdict_tests("BF", reject_e, reject_s, -0.2, 0.05)) == verdict


def test_a_bf_rejection_with_the_wrong_sign_breaches_the_procedure():
    with pytest.raises(ValueError, match="wrong sign"):
        mx.bf_verdict(*_verdict_tests("BF", True, False, +0.2, 0.05))


@pytest.mark.parametrize("reject_e, reject_s, delta_e, delta_s, verdict", [
    (True, True, -0.2, 0.05, "DIFFERENT-CONSISTENT"),     # both improve
    (True, True, 0.2, -0.05, "DIFFERENT-CONSISTENT"),     # both worsen
    (True, True, -0.2, -0.05, "DIFFERENT-DISCORDANT"),    # e improves, s worsens
    (True, True, 0.2, 0.05, "DIFFERENT-DISCORDANT"),      # e worsens, s improves
    (True, False, 0.2, 0.05, "PARTIAL"), (False, True, -0.2, 0.05, "PARTIAL"),
    (False, False, -0.2, 0.05, "NOT DIFFERENT")])
def test_every_yaw_verdict_class(reject_e, reject_s, delta_e, delta_s, verdict):
    assert mx.yaw_verdict(*_verdict_tests("YAW", reject_e, reject_s, delta_e,
                                          delta_s)) == verdict


# --------------------------------------------------------------------------- #
# the confirmatory family end to end, and the incomplete family
# --------------------------------------------------------------------------- #
ROOMS = [f"Room/Room_idx_{i:02d}" for i in range(16)]


def _room_stats(median, success):
    return {room: {"median_e_loc": float(m), "success_raw@1.0": float(s)}
            for room, m, s in zip(ROOMS, median, success)}


def _arms():
    rng = np.random.default_rng(7)
    p1 = _room_stats(rng.uniform(0.7, 5.0, 16), rng.uniform(0.0, 0.8, 16))
    bf = _room_stats([p1[r]["median_e_loc"] - 0.15 + 0.005 * i for i, r in enumerate(ROOMS)],
                     [p1[r]["success_raw@1.0"] + 0.04 for r in ROOMS])
    yaw = _room_stats([p1[r]["median_e_loc"] + rng.normal(0.0, 0.2) for r in ROOMS],
                      [p1[r]["success_raw@1.0"] + rng.normal(0.0, 0.05) for r in ROOMS])
    return {"P1": p1, "BF": bf, "YAW": yaw}


def _expected_p(treatment, control, stat, arm):
    """Independent re-derivation with numpy, straight from plan r2 §1."""
    deltas = np.array([treatment[r][stat] - control[r][stat] for r in ROOMS])
    draws = np.random.default_rng(20260825).integers(0, 16, size=(10000, 16))
    boot = deltas[draws].mean(axis=1)
    le, ge = np.mean(boot <= 0.0), np.mean(boot >= 0.0)
    if arm == "BF":
        return float(ge if stat == "median_e_loc" else le), float(deltas.mean())
    return float(min(1.0, 2.0 * min(le, ge))), float(deltas.mean())


def test_the_confirmatory_family_matches_an_independent_derivation():
    arms = _arms()
    result = mx.confirmatory(arms)
    assert result["family_complete"] is True
    assert result["draws"]["seed"] == 20260825 and result["draws"]["n_boot"] == 10000
    assert result["rooms"] == ROOMS
    raw = {}
    for test in result["tests"]:
        p, delta = _expected_p(arms[test["arm"]], arms["P1"], test["stat"], test["arm"])
        assert test["p_raw"] == pytest.approx(p, abs=0.0)
        assert test["delta"] == pytest.approx(delta)
        raw[test["label"]] = p
    # Holm, independently: sort, scale by (4 - i), running max, stop at first failure
    ordered = sorted(raw.items(), key=lambda kv: (kv[1], kv[0]))
    running, rejecting, adjusted = 0.0, True, {}
    for index, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (4 - index)))
        rejecting = rejecting and running <= 0.05
        adjusted[label] = (running, rejecting)
    for test in result["tests"]:
        assert (test["p_adjusted"], test["rejected"]) == pytest.approx(adjusted[test["label"]])
    # the BF effect above is a clean improvement on every room: both reject
    assert result["verdicts"]["BF"] == "SUPERIOR"
    assert result["verdicts"]["YAW"] in {"NOT DIFFERENT", "PARTIAL", "DIFFERENT-CONSISTENT",
                                         "DIFFERENT-DISCORDANT"}
    # ONE matrix, shared by all four tests
    assert len({test["draws_sha256"] for test in result["tests"]}) == 1
    assert result["draws"]["sha256"] == result["tests"][0]["draws_sha256"]


def test_an_incomplete_family_reports_raw_p_and_never_a_holm_verdict():
    arms = _arms()
    result = mx.confirmatory({"P1": arms["P1"], "YAW": arms["YAW"]})
    assert result["family_complete"] is False
    assert result["holm"] is None
    assert {test["arm"] for test in result["tests"]} == {"YAW"}
    for test in result["tests"]:
        assert test["p_raw"] is not None
        assert test["p_adjusted"] is None and test["rejected"] is None
    assert result["verdicts"] == {"YAW": mx.INCOMPLETE_FAMILY_VERDICT}
    assert "INCOMPLETE" in mx.INCOMPLETE_FAMILY_VERDICT
    with pytest.raises(ValueError, match="P1"):
        mx.confirmatory({"YAW": arms["YAW"], "BF": arms["BF"]})


def test_the_secondaries_are_descriptive_deltas_only():
    results = {"P1": _results(P1_ROWS), "YAW": _results(YAW_ROWS)}
    secondary = mx.secondary_descriptives(results, draws=DRAWS[:, :2] % 2)
    yaw = secondary["YAW"]
    cell = yaw["K8"]["success_raw@0.5"]
    # success@0.5: P1 R1 0/3, R2 1/2 ; YAW R1 1/3, R2 2/2 -> deltas 1/3, 1/2
    assert cell["delta"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert "p_raw" not in cell and cell["role"].startswith("DESCRIPTIVE")
    assert set(yaw) == {"K1", "K4", "K8"}
    assert "median_e_loc" not in yaw["K8"]                       # primaries live elsewhere
    assert "median_e_loc" in yaw["K1"]
