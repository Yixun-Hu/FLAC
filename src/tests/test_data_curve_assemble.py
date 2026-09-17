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
import hashlib
import json
import os

import pytest
import torch

from src.tools.data_curve import assemble, names


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


# ===================================================================================
# Loading the cells off the NAS: whose numbers are these, and are there five of them?
# ===================================================================================
BASE_METRICS = {
    "T60": 9.0, "Invalid T60": 0.0, "C50": 1.0, "EDT": 40.0, "FD": 0.32,
    "RIR_to_GT_RIR_R@1": 5.0, "RIR_to_GT_RIR_R@5": 15.0, "RIR_to_GT_RIR_R@10": 23.0,
    "RIR_to_geom_R@1": 3.8, "RIR_to_geom_R@5": 13.0, "RIR_to_geom_R@10": 20.0,
}
#: The fake checkpoint every fixture run is scored from. Its digest is the REAL sha256 of
#: those bytes, because the assembler now hashes the discovered final checkpoint itself and
#: compares -- an invented constant would be rejected, as it should be.
CKPT_BYTES = b"not a real checkpoint"
CKPT_SHA = hashlib.sha256(CKPT_BYTES).hexdigest()
#: Fixture bundles are (2, 1, 10240) instead of the real (6337, 1, 10240) = 260 MB.
FIXTURE_N = 2
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "data_curve")


def cell_record(arm, ckpt, values=None, ckpt_sha256=CKPT_SHA, **overrides):
    """One metrics JSON exactly as ``eval_FLAC.build_metrics_record`` writes it."""
    metrics = dict(BASE_METRICS)
    for metric, value in (values or {}).items():
        metrics[assemble.METRIC_KEYS[metric]] = value
    record = {
        "metrics": metrics,
        "ckpt_path": ckpt,
        "rotate_deg": names.ROTATE_DEG,
        "cond_method": names.ARM_COND_METHOD[arm],
        "frame_avg_angles": [0.0] if arm == "cyl" else None,
        "cond_autocast": names.COND_AUTOCAST,
        "ckpt_sha256": ckpt_sha256,
    }
    record.update(overrides)
    return record


def write_bundle(ckpt, arm, tag, K, seed, meta_patch=None):
    """The sibling prediction bundle -- the only artifact that proves the split and seed."""
    meta = {"dataset_config": names.EVAL_DATASET_CONFIGS[K], "seed": seed,
            "n_samples": FIXTURE_N, "n_items": FIXTURE_N, "batch_size": 8,
            "cond_method": names.ARM_COND_METHOD[arm],
            "frame_avg_angles": [0.0] if arm == "cyl" else None,
            "rotate_deg": names.ROTATE_DEG, "cond_autocast": names.COND_AUTOCAST,
            "ckpt_path": ckpt, "ckpt_sha256": CKPT_SHA,
            "eval_name": names.eval_name(arm, tag, K, seed), "steps": names.EVAL_STEPS,
            "cfg_scale": names.EVAL_CFG_SCALE, "stored_after_clamp_pad": True,
            "artifact_contract": names.PREDICTIONS_ARTIFACT_CONTRACT}
    meta.update(meta_patch or {})
    torch.save({"predictions": torch.zeros(FIXTURE_N, 1, names.SAMPLE_LEN), "meta": meta},
               names.predictions_pt_path(ckpt, arm, tag, K, seed))


def write_run(nas_root, arm, tag, cells=None, epoch=8, step=names.MAX_STEPS,
              record_patch=None, bundle_patch=None, **kwargs):
    """A finished run directory on a fake NAS: the final checkpoint, its cells and bundles.

    ``cells`` maps ``(K, seed)`` to either a ``{metric: value}`` dict or a whole record
    override; omitted cells are simply absent, which is how an unfinished arm looks.
    ``record_patch`` / ``bundle_patch`` override fields of one cell's JSON / bundle meta.
    """
    run_dir = os.path.join(str(nas_root), names.run_id(arm, tag))
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, f"epoch={epoch}-step={step}.ckpt")
    with open(ckpt, "wb") as fout:
        fout.write(CKPT_BYTES)
    if cells is None:
        cells = {(K, seed): {} for K in names.K_VALUES for seed in names.SEEDS}
    for (K, seed), spec in cells.items():
        patch = (record_patch or {}).get((K, seed), {})
        record = spec if "metrics" in spec else cell_record(arm, ckpt, spec,
                                                            **dict(kwargs, **patch))
        with open(names.metrics_json_path(ckpt, arm, tag, K, seed), "w") as fout:
            json.dump(record, fout)
        write_bundle(ckpt, arm, tag, K, seed, (bundle_patch or {}).get((K, seed)))
    return ckpt


def test_final_checkpoint_is_the_single_step_40000_file(tmp_path):
    ckpt = write_run(tmp_path, "cyl", "025", cells={})
    run_dir = os.path.dirname(ckpt)
    open(os.path.join(run_dir, "epoch=1-step=2500.ckpt"), "wb").close()
    assert assemble.final_checkpoint(run_dir) == ckpt


def test_final_checkpoint_refuses_a_run_that_has_not_reached_40000(tmp_path):
    run_dir = tmp_path / "dc_cyl_f025"
    run_dir.mkdir()
    (run_dir / "epoch=1-step=2500.ckpt").write_bytes(b"")
    with pytest.raises(assemble.DataCurveError):
        assemble.final_checkpoint(str(run_dir))


def test_final_checkpoint_refuses_two_candidates(tmp_path):
    ckpt = write_run(tmp_path, "cyl", "025", cells={})
    run_dir = os.path.dirname(ckpt)
    open(os.path.join(run_dir, "epoch=9-step=40000.ckpt"), "wb").close()
    with pytest.raises(assemble.DataCurveError):
        assemble.final_checkpoint(run_dir)


def test_load_run_reads_all_ten_cells(tmp_path):
    write_run(tmp_path, "cyl", "025")
    run = assemble.load_run(str(tmp_path), "cyl", "025", expect_n=FIXTURE_N)
    assert run["violations"] == []
    assert run["run_id"] == "dc_cyl_f025"
    assert run["ckpt_sha256"] == CKPT_SHA
    assert sorted(run["cells"][8]) == list(names.SEEDS)
    assert run["cells"][1][42]["metrics"]["T60"] == pytest.approx(9.0)


def test_load_run_marks_a_missing_seed_instead_of_averaging_four(tmp_path):
    cells = {(K, seed): {} for K in names.K_VALUES for seed in names.SEEDS}
    del cells[(8, 45)]
    write_run(tmp_path, "van", "050", cells=cells)
    run = assemble.load_run(str(tmp_path), "van", "050", expect_n=FIXTURE_N)
    assert 45 not in run["cells"][8]
    assert any("s45" in v and "K8" in v for v in run["violations"])


@pytest.mark.parametrize("field, value", [
    ("cond_method", "vanilla"),          # a cyl cell scored down the stock path
    ("frame_avg_angles", None),
    ("rotate_deg", 45.0),
    ("cond_autocast", "off"),
])
def test_load_run_refuses_a_cell_scored_under_another_protocol(tmp_path, field, value):
    cells = {(K, seed): {} for K in names.K_VALUES for seed in names.SEEDS}
    run_dir = tmp_path / "dc_cyl_f075"
    run_dir.mkdir()
    ckpt = str(run_dir / f"epoch=8-step={names.MAX_STEPS}.ckpt")
    cells[(8, 44)] = cell_record("cyl", ckpt, **{field: value})
    write_run(tmp_path, "cyl", "075", cells=cells)
    run = assemble.load_run(str(tmp_path), "cyl", "075", expect_n=FIXTURE_N)
    assert 44 not in run["cells"][8]
    assert any(field in v for v in run["violations"])


@pytest.mark.parametrize("patch, needle", [
    ({"ckpt_sha256": None}, "ckpt_sha256"),
    ({"ckpt_sha256": "b" * 64}, "ckpt_sha256"),
    ({"ckpt_path": "/elsewhere/epoch=9-step=40000.ckpt"}, "ckpt_path"),
])
def test_a_cell_not_bound_to_the_discovered_checkpoint_is_dropped(tmp_path, patch, needle):
    write_run(tmp_path, "van", "025", record_patch={(1, 43): patch})
    run = assemble.load_run(str(tmp_path), "van", "025", expect_n=FIXTURE_N)
    assert 43 not in run["cells"][1]                 # dropped, not inserted with a note
    assert any(needle in v for v in run["violations"])


def test_a_cell_whose_bundle_is_missing_is_not_a_complete_cell(tmp_path):
    ckpt = write_run(tmp_path, "van", "025")
    os.remove(names.predictions_pt_path(ckpt, "van", "025", 8, 46))
    run = assemble.load_run(str(tmp_path), "van", "025", expect_n=FIXTURE_N)
    assert 46 not in run["cells"][8]
    assert any("bundle" in v for v in run["violations"])


def test_a_cell_whose_bundle_claims_another_seed_is_dropped(tmp_path):
    write_run(tmp_path, "cyl", "075", bundle_patch={(8, 44): {"seed": 45}})
    run = assemble.load_run(str(tmp_path), "cyl", "075", expect_n=FIXTURE_N)
    assert 44 not in run["cells"][8]
    assert any("seed" in v for v in run["violations"])


# --------------------------------------------------------- aggregation and paired benefit
def test_aggregate_marks_a_row_that_is_short_of_five_seeds():
    out = assemble.aggregate({42: 1.0, 43: 2.0, 44: 3.0, 45: 4.0})
    assert out["complete"] is False
    assert out["n"] == 4
    assert assemble.aggregate({s: 1.0 for s in names.SEEDS})["complete"] is True


def test_paired_benefit_differences_per_seed_not_differences_of_means():
    # Same means, but the per-seed differences are constant: the paired sd must be 0
    # while the marginal arms each have a visible spread.
    cyl = {42: 8.0, 43: 9.0, 44: 10.0, 45: 11.0, 46: 12.0}
    van = {s: v + 1.0 for s, v in cyl.items()}
    out = assemble.paired_benefit("T60", cyl, van)
    assert out["form"] == "paired"
    assert out["mean"] == pytest.approx(1.0)
    assert out["sd"] == pytest.approx(0.0)
    assert out["n"] == 5


def test_paired_benefit_orients_recall_the_other_way():
    cyl = {s: 6.0 for s in names.SEEDS}
    van = {s: 5.0 for s in names.SEEDS}
    assert assemble.paired_benefit("R@1", cyl, van)["mean"] == pytest.approx(1.0)
    assert assemble.paired_benefit("T60", cyl, van)["mean"] == pytest.approx(-1.0)


def test_paired_benefit_refuses_when_the_two_arms_do_not_share_seeds():
    cyl = {42: 1.0, 43: 1.0, 44: 1.0, 45: 1.0, 46: 1.0}
    van = {42: 2.0, 43: 2.0, 44: 2.0, 45: 2.0}
    assert assemble.paired_benefit("T60", cyl, van) is None


def test_marginal_benefit_is_a_difference_of_means_with_no_paired_sd():
    out = assemble.marginal_benefit("T60", {"mean": 9.5, "sd": 0.2, "n": 5},
                                    {"mean": 10.0, "sd": 0.1, "n": 5})
    assert out["form"] == "marginal"
    assert out["mean"] == pytest.approx(0.5)
    assert out["sd"] is None


# ---------------------------------------------------------------------------- anchors
def test_anchor_reference_maps_each_arm_to_its_tier_s_block():
    anchors = assemble.load_anchor_reference(os.path.join(FIXTURES, "anchors_tier_S.json"))
    assert anchors["van"][8]["T60"]["mean"] == pytest.approx(10.0)
    assert anchors["cyl"][8]["T60"]["mean"] == pytest.approx(9.5)
    assert anchors["cyl"][1]["R@10"]["sd"] == pytest.approx(0.3)
    assert anchors["van"][8]["T60"]["form"] == "marginal"


def write_anchor_cells(directory, arm, names_by_cell, manifest=None):
    os.makedirs(directory, exist_ok=True)
    for (K, seed), basename in names_by_cell.items():
        record = cell_record(arm, "anchor.ckpt", {"T60": 10.0 + seed - 42 + K / 100.0})
        record.pop("ckpt_sha256")          # the historical anchors predate the digest
        with open(os.path.join(directory, basename), "w") as fout:
            json.dump(record, fout)
    if manifest is not None:
        with open(os.path.join(directory, "anchor_cells.json"), "w") as fout:
            json.dump(manifest, fout)
    return directory


def _anchor_basenames():
    return {(K, s): f"epoch=8-step=40000_metrics_1_1.0_ref_K{K}_s{s}.json"
            for K in names.K_VALUES for s in names.SEEDS}


def test_anchor_cells_are_paired_by_the_K_and_seed_in_their_names(tmp_path):
    directory = write_anchor_cells(str(tmp_path / "p1"), "van", _anchor_basenames())
    cells, violations = assemble.load_anchor_cells(directory, "van")
    assert violations == []
    assert sorted(cells[8]) == list(names.SEEDS)
    assert cells[8][43]["metrics"]["T60"] == pytest.approx(11.08)


def test_a_cell_whose_name_carries_no_K_or_seed_is_placed_by_the_pinned_manifest(tmp_path):
    # D10: P1's seed-42 K=8 anchor is the screen cell `..._exp07_P1_screen_S40000_ema.json`,
    # whose name says neither. Guessing is not an option; the mapping is pinned.
    basenames = _anchor_basenames()
    screen = "epoch=8-step=40000_metrics_1_1.0_exp07_P1_screen_S40000_ema.json"
    basenames[(8, 42)] = screen
    directory = write_anchor_cells(str(tmp_path / "p1"), "van", basenames,
                                   manifest={screen: {"K": 8, "seed": 42}})
    cells, violations = assemble.load_anchor_cells(directory, "van")
    assert violations == []
    assert os.path.basename(cells[8][42]["path"]) == screen


def test_an_unplaceable_anchor_cell_is_refused_not_guessed(tmp_path):
    basenames = _anchor_basenames()
    basenames[(8, 42)] = "epoch=8-step=40000_metrics_1_1.0_exp07_P1_screen_S40000_ema.json"
    directory = write_anchor_cells(str(tmp_path / "p1"), "van", basenames)
    cells, violations = assemble.load_anchor_cells(directory, "van")
    assert cells[8].get(42) is None
    assert any("screen" in v for v in violations)


def test_a_missing_anchor_seed_refuses_the_paired_form(tmp_path):
    basenames = _anchor_basenames()
    del basenames[(1, 46)]
    directory = write_anchor_cells(str(tmp_path / "cyl"), "cyl", basenames)
    _, violations = assemble.load_anchor_cells(directory, "cyl")
    assert any("K1" in v and "s46" in v for v in violations)


def test_two_cells_claiming_one_anchor_slot_are_refused(tmp_path):
    basenames = _anchor_basenames()
    directory = write_anchor_cells(str(tmp_path / "cyl"), "cyl", basenames)
    duplicate = os.path.join(directory, "epoch=8-step=40000_metrics_1_1.0_dup_K8_s42.json")
    with open(duplicate, "w") as fout:
        json.dump(cell_record("cyl", "anchor.ckpt"), fout)
    _, violations = assemble.load_anchor_cells(directory, "cyl")
    assert any("more than one" in v for v in violations)


# ------------------------------------------------------------------- effective epochs
def test_effective_epochs_are_recomputed_and_cross_checked_against_the_manifest():
    with open(os.path.join(FIXTURES, "split_manifest.json")) as fin:
        manifest = json.load(fin)
    epochs, violations = assemble.effective_epochs(manifest)
    assert violations == []
    assert epochs[25] == pytest.approx(10240.0)
    assert epochs[75] == pytest.approx(3413.3333333333335)
    # 100 % is not in the manifest's `fractions` block: it is the full split, whose size
    # the histogram carries (100 + 900 = 1000 targets).
    assert epochs[100] == pytest.approx(2560.0)


def test_a_manifest_that_disagrees_with_its_own_arithmetic_is_flagged():
    with open(os.path.join(FIXTURES, "split_manifest.json")) as fin:
        manifest = json.load(fin)
    manifest["fractions"]["0.5"]["effective_epochs_at_40k_x64"] = 1.0
    _, violations = assemble.effective_epochs(manifest)
    assert any("0.5" in v for v in violations)


# =====================================================================================
# The whole document: six runs + two anchors -> one table, one verdict, one provenance
# =====================================================================================
JITTER = {42: -0.02, 43: -0.01, 44: 0.0, 45: 0.01, 46: 0.02}   # mean exactly 0
BENEFIT = {"025": 0.9, "050": 0.7, "075": 0.6}                 # shrinking with more data
ANCHORS_JSON = os.path.join(FIXTURES, "anchors_tier_S.json")


def _anchor_means(arm, K):
    with open(ANCHORS_JSON) as fin:
        reference = json.load(fin)
    block = reference[assemble.ANCHOR_REFERENCE_KEYS[arm]][f"K{K}"]
    return {metric: block[metric][0] for metric in assemble.METRICS}


def run_cell_values(arm, tag, K, seed):
    """Synthetic cell values: van sits on its 100 % anchor, cyl beats it by BENEFIT[tag].

    The per-seed jitter is shared by both arms, so the PAIRED benefit has sd 0 while each
    arm on its own has a visible spread -- which is exactly the property the paired form
    exists for, and it would vanish if the assembler differenced the means instead.
    """
    values = {m: v + JITTER[seed] for m, v in _anchor_means("van", K).items()}
    if arm == "van":
        return values
    benefit = BENEFIT[tag]
    values["T60"] -= benefit
    values["EDT"] -= 2 * benefit
    values["C50"] += 0.1                       # CylDINO is WORSE on C50, as at 100 %
    values["R@1"] += benefit / 2
    values["R@5"] += benefit
    values["R@10"] += benefit
    return values


def fixture_nas(tmp_path, drop=(), patch=None):
    """A NAS root holding all six finished runs.

    ``drop`` removes ``(arm, tag, K, seed)`` cells; ``patch`` maps the same key to record
    overrides, which is how a cell that is not bound to its run's checkpoint is built.
    """
    nas = tmp_path / "nas"
    nas.mkdir(exist_ok=True)
    for tag in sorted(names.FRACTIONS):
        for arm in names.ARMS:
            cells = {(K, seed): run_cell_values(arm, tag, K, seed)
                     for K in names.K_VALUES for seed in names.SEEDS
                     if (arm, tag, K, seed) not in drop}
            patches = {(K, seed): fields for (a, t, K, seed), fields in (patch or {}).items()
                       if (a, t) == (arm, tag)}
            write_run(nas, arm, tag, cells=cells, record_patch=patches)
    return str(nas)


def fixture_anchor_dirs(tmp_path):
    """Raw 100 % anchor cells whose per-seed means reproduce the reference exactly."""
    out = {}
    for arm in names.ARMS:
        directory = str(tmp_path / f"anchor_{arm}")
        os.makedirs(directory, exist_ok=True)
        for K in names.K_VALUES:
            base = _anchor_means(arm, K)
            for seed in names.SEEDS:
                record = cell_record(arm, "anchor.ckpt",
                                     {m: v + JITTER[seed] for m, v in base.items()})
                record.pop("ckpt_sha256")
                path = os.path.join(directory, f"anchor_ref_K{K}_s{seed}.json")
                with open(path, "w") as fout:
                    json.dump(record, fout)
        out[arm] = directory
    return out


def build(tmp_path, **kwargs):
    return assemble.build_curve(
        kwargs.pop("nas_root", None) or fixture_nas(tmp_path),
        ANCHORS_JSON,
        split_manifest_path=os.path.join(FIXTURES, "split_manifest.json"),
        expect_n=FIXTURE_N, **kwargs)


def test_a_complete_curve_reaches_the_pre_registered_verdict(tmp_path):
    doc = build(tmp_path)
    assert doc["violations"] == []
    assert doc["complete"] is True
    assert doc["verdict"]["verdict"] == "SUPPORTED"
    assert doc["curve"]["K8"]["T60"]["25"]["benefit"]["mean"] == pytest.approx(0.9)
    assert doc["curve"]["K8"]["T60"]["100"]["benefit"]["mean"] == pytest.approx(0.5)


def test_the_paired_benefit_survives_the_whole_pipeline(tmp_path):
    doc = build(tmp_path)
    row = doc["curve"]["K8"]["EDT"]["50"]
    assert row["benefit"]["form"] == "paired"
    assert row["benefit"]["sd"] == pytest.approx(0.0)     # shared jitter cancels per seed
    assert row["cyl"]["sd"] > 0                            # but each arm alone does vary


def test_without_raw_anchor_cells_the_hundred_percent_point_is_marginal(tmp_path):
    doc = build(tmp_path)
    assert doc["anchor_form"] == "marginal"
    assert doc["curve"]["K8"]["T60"]["100"]["benefit"]["sd"] is None
    assert any("marginal" in d for d in doc["disclosures"])


def test_raw_anchor_cells_promote_the_hundred_percent_point_to_the_paired_form(tmp_path):
    doc = build(tmp_path, anchor_cell_dirs=fixture_anchor_dirs(tmp_path))
    assert doc["anchor_form"] == "paired"
    row = doc["curve"]["K8"]["T60"]["100"]
    assert row["benefit"]["form"] == "paired"
    assert row["benefit"]["mean"] == pytest.approx(0.5)
    assert row["benefit"]["sd"] == pytest.approx(0.0)


def test_an_incomplete_anchor_directory_falls_back_to_the_marginal_form(tmp_path):
    dirs = fixture_anchor_dirs(tmp_path)
    os.remove(os.path.join(dirs["cyl"], "anchor_ref_K8_s46.json"))
    doc = build(tmp_path, anchor_cell_dirs=dirs)
    assert doc["anchor_form"] == "marginal"
    assert any("s46" in v for v in doc["violations"])


@pytest.mark.parametrize("patch", [
    {"ckpt_sha256": None},                                   # evaluator stamped nothing in
    {"ckpt_sha256": "b" * 64},                               # a digest nobody validated
    {"ckpt_path": "/elsewhere/epoch=9-step=40000.ckpt"},     # scored from another file
])
def test_an_unbound_primary_cell_pends_the_verdict_it_does_not_just_warn(tmp_path, patch):
    nas = fixture_nas(tmp_path, patch={("cyl", "050", 8, 44): patch})
    doc = build(tmp_path, nas_root=nas)
    assert doc["curve"]["K8"]["T60"]["50"]["cyl"]["n"] == 4      # the cell is GONE
    assert doc["curve"]["K8"]["T60"]["50"]["complete"] is False
    assert doc["verdict"]["verdict"] == "PENDING"
    assert doc["complete"] is False


def test_a_run_whose_every_cell_carries_the_wrong_digest_yields_no_cells(tmp_path):
    # Uniform is not the same as correct: all ten agree with each other and none of them
    # agrees with the bytes on disk, which is exactly the case the old check missed.
    patch = {("van", "025", K, seed): {"ckpt_sha256": "b" * 64}
             for K in names.K_VALUES for seed in names.SEEDS}
    doc = build(tmp_path, nas_root=fixture_nas(tmp_path, patch=patch))
    assert doc["curve"]["K8"]["T60"]["25"]["van"]["n"] == 0
    assert doc["verdict"]["verdict"] == "PENDING"


def test_a_row_short_of_five_seeds_makes_the_verdict_pending(tmp_path):
    nas = fixture_nas(tmp_path, drop=[("cyl", "050", 8, 44)])
    doc = build(tmp_path, nas_root=nas)
    assert doc["curve"]["K8"]["T60"]["50"]["complete"] is False
    assert doc["verdict"]["verdict"] == "PENDING"
    assert doc["complete"] is False


def test_data_equivalence_is_reported_for_every_metric_and_K(tmp_path):
    doc = build(tmp_path)
    assert sorted(doc["data_equivalence"]) == ["K1", "K8"]
    assert sorted(doc["data_equivalence"]["K8"]) == sorted(assemble.METRICS)
    # cyl at 25 % already beats van@100 on T60 in this fixture: boundary-censored.
    assert doc["data_equivalence"]["K8"]["T60"]["kind"] == "censored_at_min"
    # ... and never catches up on C50, where CylDINO is worse by construction.
    assert doc["data_equivalence"]["K8"]["C50"]["kind"] == "none"


def test_the_document_round_trips_through_json_with_stable_keys(tmp_path):
    doc = build(tmp_path)
    path = tmp_path / "data_curve.json"
    with open(path, "w") as fout:
        json.dump(doc, fout)
    with open(path) as fin:
        reloaded = json.load(fin)
    assert reloaded == doc


def test_the_markdown_carries_the_numbers_the_verdict_and_the_provenance(tmp_path):
    doc = build(tmp_path)
    md = assemble.render_markdown(doc)
    for K in names.K_VALUES:
        assert f"K = {K}" in md
    for metric in assemble.METRICS:
        assert metric in md
    assert "SUPPORTED" in md
    assert "Data equivalence" in md
    assert CKPT_SHA in md                       # which checkpoint bytes produced the cells
    assert "fa_invariant" in md and "vanilla" in md      # both protocols, stated
    assert "dc_cyl_f025_K8_s42" in md                    # a cell path, not just a number
    assert "34.84" not in md and "10240.0" in md         # the FIXTURE's effective epochs


def test_the_markdown_states_the_fixed_compute_estimand(tmp_path):
    md = assemble.render_markdown(build(tmp_path))
    assert "fixed-compute" in md
    assert "item-weighted" in md and "per-scene" in md


# ------------------------------------------------------------------------------- CLI
def test_cli_writes_both_artifacts_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", FIXTURE_N)
    out_json, out_md = tmp_path / "curve.json", tmp_path / "curve.md"
    rc = assemble.main(["--nas-root", fixture_nas(tmp_path), "--anchors", ANCHORS_JSON,
                        "--split-manifest", os.path.join(FIXTURES, "split_manifest.json"),
                        "--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    with open(out_json) as fin:
        assert json.load(fin)["verdict"]["verdict"] == "SUPPORTED"
    assert "SUPPORTED" in out_md.read_text()


def test_cli_accepts_the_two_anchor_cell_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", FIXTURE_N)
    dirs = fixture_anchor_dirs(tmp_path)
    out_json = tmp_path / "curve.json"
    rc = assemble.main(["--nas-root", fixture_nas(tmp_path), "--anchors", ANCHORS_JSON,
                        "--split-manifest", os.path.join(FIXTURES, "split_manifest.json"),
                        "--anchor-cells-p1", dirs["van"], "--anchor-cells-cyl", dirs["cyl"],
                        "--out-json", str(out_json)])
    assert rc == 0
    with open(out_json) as fin:
        assert json.load(fin)["anchor_form"] == "paired"


def test_cli_strict_exits_non_zero_on_a_missing_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(names, "N_ITEMS_UNSEEN", FIXTURE_N)
    nas = fixture_nas(tmp_path, drop=[("van", "075", 1, 42)])
    argv = ["--nas-root", nas, "--anchors", ANCHORS_JSON, "--split-manifest",
            os.path.join(FIXTURES, "split_manifest.json"), "--out-json",
            str(tmp_path / "curve.json")]
    assert assemble.main(argv) == 0                       # marked, but rendered
    assert assemble.main(argv + ["--strict"]) != 0        # and fatal when it must be
