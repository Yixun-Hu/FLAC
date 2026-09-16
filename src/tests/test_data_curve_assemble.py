"""Tests for exp_14's data-curve assembler (plan §1: endpoints, verdict, data equivalence).

The assembler is the only thing that turns 60 metric JSONs into a claim, so every rule it
applies is pre-registered in the plan and pinned here rather than argued about afterwards:

* **Orientation.** ``B_f = van - cyl`` for T60/C50/EDT and ``cyl - van`` for R@k, so a
  positive benefit always means "CylDINO is better". A sign slip would invert the whole
  experiment's conclusion while every number still looks plausible.
* **The ordered verdict (plan §1).** NOT SUPPORTED -> MIXED -> SUPPORTED -> PARTIAL, first
  match wins. The branches overlap on purpose (an all-non-positive curve can also be
  monotone), so the ORDER is the rule and is tested directly, not just the four outcomes.
* **Data equivalence (plan §1, finding r2-6/r3-4).** An upward scan with first match over
  g(f) = cyl(f) - van@100 %, oriented; exactly one of four outcomes fires. A non-monotone
  curve must report only its *first* crossing -- reporting the last one would quietly turn
  "recovers briefly at 50 %" into "needs 75 % of the data".
* **Five seeds or nothing (plan §1, D7).** A cell aggregated from four seeds is not the
  quantity the anchors were measured as; such a row is marked, never silently averaged.

CPU-only and filesystem-free except the loader cases, which write small JSONs in
``tmp_path``. No GPU, no NAS, no network.
"""
import pytest

from src.tools.data_curve import assemble


# --------------------------------------------------------------------------- orientation
def test_lower_is_better_metrics_orient_van_minus_cyl():
    # T60/C50/EDT are error metrics: cyl better (smaller) => positive benefit.
    assert assemble.oriented_benefit("T60", 10.0, 9.0) == pytest.approx(1.0)
    assert assemble.oriented_benefit("C50", 1.0093, 1.0786) == pytest.approx(-0.0693)
    assert assemble.oriented_benefit("EDT", 40.65, 39.0764) == pytest.approx(1.5736)


def test_higher_is_better_metrics_orient_cyl_minus_van():
    # Recall is a score: cyl better (larger) => positive benefit.
    assert assemble.oriented_benefit("R@1", 5.173, 5.4789) == pytest.approx(0.3059)
    assert assemble.oriented_benefit("R@5", 15.43, 16.3421) == pytest.approx(0.9121)
    assert assemble.oriented_benefit("R@10", 23.409, 24.226) == pytest.approx(0.817)


def test_oriented_benefit_refuses_an_unscored_metric():
    with pytest.raises(ValueError):
        assemble.oriented_benefit("FD", 1.0, 2.0)


# ------------------------------------------------------------------------------ mean / sd
def test_mean_sd_is_the_sample_sd_over_seeds():
    assert assemble.mean_sd([1.0, 2.0, 3.0]) == pytest.approx((2.0, 1.0))


def test_mean_sd_of_one_seed_reports_zero_spread_not_an_error():
    assert assemble.mean_sd([8.31]) == pytest.approx((8.31, 0.0))


def test_mean_sd_of_nothing_raises():
    with pytest.raises(ValueError):
        assemble.mean_sd([])


# -------------------------------------------------------------------------------- verdict
def _benefits(t60, edt):
    return {"T60": dict(zip((25, 50, 75, 100), t60)),
            "EDT": dict(zip((25, 50, 75, 100), edt))}


def test_verdict_supported_when_both_primaries_are_positive_and_non_increasing():
    out = assemble.verdict(_benefits([0.9, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "SUPPORTED"
    assert out["annotations"] == []


def test_verdict_partial_when_positive_everywhere_but_monotonicity_fails():
    out = assemble.verdict(_benefits([0.9, 0.5, 0.7, 0.4], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "PARTIAL"
    # 0.9 at 25 % is still above 0.4 at 100 %, so the benefit is not *shrinking*.
    assert out["annotations"] == []


def test_verdict_partial_annotates_the_metric_whose_benefit_shrinks():
    out = assemble.verdict(_benefits([0.4, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "PARTIAL"
    assert out["annotations"] == ["T60: benefit shrinking (B_25 < B_100)"]


def test_verdict_mixed_when_exactly_one_primary_goes_non_positive():
    out = assemble.verdict(_benefits([-0.1, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "MIXED"
    assert out["non_positive"] == {"T60": [25], "EDT": []}


def test_verdict_not_supported_when_both_fail_at_different_fractions():
    out = assemble.verdict(_benefits([-0.1, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, -0.2]))
    assert out["verdict"] == "NOT SUPPORTED"
    assert out["non_positive"] == {"T60": [25], "EDT": [100]}


def test_a_benefit_of_exactly_zero_counts_as_non_positive():
    out = assemble.verdict(_benefits([0.0, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "MIXED"


def test_branch_order_non_positive_beats_monotone():
    # Both curves are perfectly non-increasing (rule 3's condition) AND both dip to <= 0
    # (rule 1's). The ORDER decides: NOT SUPPORTED, never SUPPORTED.
    out = assemble.verdict(_benefits([0.5, 0.2, 0.0, -0.3], [0.9, 0.4, 0.1, -0.1]))
    assert out["verdict"] == "NOT SUPPORTED"


def test_branch_order_one_non_positive_beats_monotone():
    out = assemble.verdict(_benefits([0.5, 0.2, 0.1, -0.3], [1.8, 1.4, 1.2, 1.0]))
    assert out["verdict"] == "MIXED"


def test_verdict_refuses_a_missing_primary_cell():
    incomplete = _benefits([0.9, 0.7, 0.6, 0.5], [1.8, 1.4, 1.2, 1.0])
    incomplete["EDT"][75] = None
    with pytest.raises(ValueError):
        assemble.verdict(incomplete)


# ---------------------------------------------------------------------- data equivalence
def _g(*values):
    return dict(zip((25, 50, 75, 100), values))


def test_data_equivalence_is_censored_when_the_smallest_fraction_already_wins():
    out = assemble.data_equivalence(_g(0.5, 0.4, 0.3, 0.2))
    assert out["kind"] == "censored_at_min"
    assert out["outcome"] == "<= 25 %"
    assert out["f_star"] is None


def test_data_equivalence_treats_zero_at_the_smallest_fraction_as_censored():
    assert assemble.data_equivalence(_g(0.0, 0.4, 0.3, 0.2))["kind"] == "censored_at_min"


@pytest.mark.parametrize("values, where", [
    ((-0.3, 0.0, 0.2, 0.3), 50),
    ((-0.3, -0.1, 0.0, 0.2), 75),
    ((-0.3, -0.2, -0.1, 0.0), 100),
])
def test_data_equivalence_reports_an_exact_fraction(values, where):
    out = assemble.data_equivalence(_g(*values))
    assert out["kind"] == "exact"
    assert out["f_star"] == pytest.approx(float(where))
    assert out["outcome"] == f"exact {where} %"


@pytest.mark.parametrize("values, bracket, f_star", [
    ((-1.0, 1.0, 1.2, 1.3), (25, 50), 37.5),
    ((-1.0, -0.5, 0.5, 0.8), (50, 75), 62.5),
    ((-1.0, -1.0, -0.25, 0.75), (75, 100), 81.25),
])
def test_data_equivalence_interpolates_inside_the_bracket_that_crosses(values, bracket,
                                                                      f_star):
    out = assemble.data_equivalence(_g(*values))
    assert out["kind"] == "crossing"
    assert tuple(out["bracket"]) == bracket
    assert out["f_star"] == pytest.approx(f_star)
    assert out["outcome"] == f"crossing in ({bracket[0]} %, {bracket[1]} %)"


def test_an_oscillating_curve_reports_only_its_first_crossing():
    out = assemble.data_equivalence(_g(-1.0, 1.0, -1.0, 1.0))
    assert out["kind"] == "crossing"
    assert tuple(out["bracket"]) == (25, 50)
    assert out["f_star"] == pytest.approx(37.5)


def test_data_equivalence_reports_no_crossing_when_the_scan_ends_negative():
    out = assemble.data_equivalence(_g(-1.0, -0.8, -0.5, -0.1))
    assert out["kind"] == "none"
    assert out["outcome"] == "no crossing through 100 %"
    assert out["f_star"] is None


def test_equality_is_decided_at_the_reporting_precision():
    # 4 decimals is what the tables print; a difference invisible there is "exact".
    assert assemble.data_equivalence(_g(-0.3, 0.00004, 0.2, 0.3))["kind"] == "exact"
    assert assemble.data_equivalence(_g(-0.3, 0.0001, 0.2, 0.3))["kind"] == "crossing"


def test_data_equivalence_prints_the_whole_g_vector_beside_the_label():
    out = assemble.data_equivalence(_g(-1.0, -0.5, 0.5, 0.8))
    assert out["g"] == {25: -1.0, 50: -0.5, 75: 0.5, 100: 0.8}
