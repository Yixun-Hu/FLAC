"""exp_15 — TDD for ``yaw_aug_collect.py`` (plan §6.8's per-function inventory).

Written RED first, per the SOP. The collector is the last thing between the 42
landed cells and a number in a table, so every one of its refusals is tested by
constructing the artifact that should be refused — not by asserting that a
correct artifact is accepted and hoping the converse holds.

The statistics are pinned to PRECOMPUTED reference values (plan §6.8): a CI or a
Holm ordering that is merely self-consistent proves nothing.

No torch, no GPU, no filesystem outside tmp_path.
"""
import json
import math
import os
import sys

import pytest

_EXPDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "worklog", "worklog_yixun", "exp_15_yaw_aug_claude")
if _EXPDIR not in sys.path:
    sys.path.insert(0, _EXPDIR)

collect = pytest.importorskip("yaw_aug_collect")
V = pytest.importorskip("exp15_validate_cell")


# --------------------------------------------------------------------------- #
# fixtures: a complete, well-formed cell on disk
# --------------------------------------------------------------------------- #
PIN = "deadbeef00000000000000000000000000000042"
CKPT_SHA = "a" * 64


def _cell(arm, kind, seed=42, k=8, deg=None):
    return V.Cell(arm, kind, V.STEP, seed, k, deg)


def write_cell(root, cell, *, n=8, pin=PIN, ckpt_sha=CKPT_SHA,
               split=None, scene=None, scene_drop=(), offsets=None, targets=None,
               meta_override=None, rec_override=None):
    """Write one cell's three artifacts under ``root``; return the metrics path."""
    mode, deg, rseed = V.rotation_expectation(cell)
    ckpt = os.path.join(root, "epoch=8-step=40000.ckpt")
    open(ckpt, "a").close()
    metrics = V.metrics_path(ckpt, cell)

    split_block = dict({m: 1.0 for m in V.REQUIRED_SPLIT_METRICS}, **(split or {}))
    scene_block = dict({m: 1.0 for m in V.REQUIRED_SCENE_METRICS}, **(scene or {}))
    for _drop in scene_drop:                 # a MISSING key, not an overridden one
        scene_block.pop(_drop, None)

    rel = list(targets or [f"Cafe/Cafe_idx_0/{i}.wav" for i in range(n)])
    inp = [[i, rel[i], ["ctx0"], V.IMG_W] for i in range(n)]
    if offsets is None:
        offsets = ([(i * 37 + 5) % V.IMG_W for i in range(n)] if mode == "random"
                   else [V.expected_column_shift(deg)] * n)
    asg = [[i, rel[i], offsets[i]] for i in range(n)]
    ih, ah = V.canonical_stream_hash(inp), V.canonical_stream_hash(asg)

    record = {
        "metrics": split_block, "ckpt_path": ckpt, "rotate_deg": deg,
        "cond_method": "vanilla", "frame_avg_angles": None, "cond_autocast": "bf16",
        "source_sha": pin, "batch_size": V.BATCH_SIZE, "n_samples": n,
        "dataset_config": "src/configs/dataset_configs/AR/eval/"
                          + (V.SPLIT_K8 if int(cell.k) == 8 else V.SPLIT_K1),
        "seed": int(cell.seed), "cfg_scale": 1.0, "steps": 1,
        "eval_name": V.eval_name(cell), "weights_source": "ema", "device": "cuda",
        "by_scene": {s: dict(scene_block) for s in V.EXPECTED_SCENE_KEYS},
        "per_scene_schema": 1, "scene_count": 10,
    }
    if mode == "random":
        record.update({"rotate_mode": "random", "rotate_deg": None,
                       "rotate_seed": rseed, "input_hash": ih,
                       "assignment_hash": ah, "stream_count": n, "img_w": V.IMG_W})
    record.update(rec_override or {})

    meta = {
        "arm": cell.arm, "cell": cell.cell, "step": V.STEP, "seed": int(cell.seed),
        "K": int(cell.k), "eval_name": V.eval_name(cell), "cond_method": "vanilla",
        "cond_autocast": "bf16", "frame_avg_angles": None,
        "frame_avg_angles_flag": V.FRAME_AVG_ANGLES_FLAG, "rotate_mode": mode,
        "rotate_seed": rseed, "rotate_deg": deg, "batch_size": V.BATCH_SIZE,
        "num_workers": V.NUM_WORKERS, "expected_stream_count": n,
        "record_stream": True, "record_per_scene": True, "use_ema": True,
        "train_yaw_aug": V.TRAIN_YAW_AUG[cell.arm], "commit": pin,
        "ckpt_sha256": ckpt_sha,
    }
    meta.update(meta_override or {})

    stream = {"schema_version": 1, "fingerprint_schema": 1, "rotate_mode": mode,
              "rotate_seed": rseed, "rotate_deg": deg, "stream_count": n,
              "img_w": V.IMG_W, "input_tuples": inp, "assignment_tuples": asg,
              "offsets": offsets, "input_hash": ih, "assignment_hash": ah}

    json.dump(record, open(metrics, "w"))
    json.dump(meta, open(metrics + ".screenmeta.json", "w"))
    json.dump(stream, open(metrics[:-len(".json")] + ".stream.json", "w"))
    return metrics


# --------------------------------------------------------------------------- #
# parse_cell
# --------------------------------------------------------------------------- #
class TestParseCell:
    def test_reads_a_well_formed_cell(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"))
        art = collect.parse_cell(path)
        assert art.cell == _cell("YAWAUG", "tbl")
        assert art.record["eval_name"] == "exp15_YAWAUG_tbl_S40000_s42_K8"
        assert art.meta["train_yaw_aug"] is True
        assert art.stream["stream_count"] == 8

    def test_a_registered_cell_that_has_not_landed_is_named(self, tmp_path):
        ckpt = str(tmp_path / "epoch=8-step=40000.ckpt")
        path = V.metrics_path(ckpt, _cell("YAWAUG", "tbl"))
        with pytest.raises(collect.ArtifactError, match="metrics artifact missing"):
            collect.parse_cell(path)

    def test_a_path_naming_no_registered_cell_is_refused_as_such(self, tmp_path):
        with pytest.raises(collect.ArtifactError, match="not a metrics artifact"):
            collect.parse_cell(str(tmp_path / "nope.json"))

    def test_malformed_json_is_named_not_crashed(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"))
        open(path, "w").write("{not json")
        with pytest.raises(collect.ArtifactError, match="could not be parsed"):
            collect.parse_cell(path)

    def test_missing_sidecar_is_named(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"))
        os.remove(path + ".screenmeta.json")
        with pytest.raises(collect.ArtifactError, match="screenmeta"):
            collect.parse_cell(path)

    def test_missing_stream_is_named(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "rrob"))
        os.remove(path[:-len(".json")] + ".stream.json")
        with pytest.raises(collect.ArtifactError, match="stream"):
            collect.parse_cell(path)

    def test_an_unregistered_eval_name_is_refused(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"))
        bad = str(tmp_path / "epoch=8-step=40000_metrics_1_1.0_exp15_C4L_tbl_S40000_s42_K8.json")
        os.rename(path, bad)
        with pytest.raises(collect.ArtifactError):
            collect.parse_cell(bad)


# --------------------------------------------------------------------------- #
# validate_protocol — delegates to exp15_validate_cell (ONE source of truth)
# --------------------------------------------------------------------------- #
class TestValidateProtocol:
    def test_a_conformant_cell_has_no_reasons(self, tmp_path):
        for kind, deg in (("tbl", None), ("rrob", None), ("vctl", 90.0)):
            d = tmp_path / kind
            d.mkdir()
            path = write_cell(str(d), _cell("VANL", kind, deg=deg), n=8)
            assert collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                             expected_count=8) == []

    @pytest.mark.parametrize("field,value,needle", [
        ("cond_method", "fa_invariant", "cond_method"),
        ("cond_autocast", "default", "cond_autocast"),
        ("batch_size", 32, "batch_size"),
        ("weights_source", "online", "weights_source"),
    ])
    def test_wrong_protocol_field_is_rejected(self, tmp_path, field, value, needle):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"),
                          rec_override={field: value})
        reasons = collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                            expected_count=8)
        assert any(needle in r for r in reasons), reasons

    def test_a_T_cell_carrying_random_provenance_is_rejected(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"),
                          rec_override={"rotate_mode": "random", "rotate_seed": 42})
        reasons = collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                            expected_count=8)
        assert any("random-mode keys" in r for r in reasons), reasons

    def test_the_pinned_frame_angle_flag_must_be_witnessed(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"),
                          meta_override={"frame_avg_angles_flag": "0,180"})
        reasons = collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                            expected_count=8)
        assert any("frame_avg_angles_flag" in r for r in reasons), reasons

    def test_nan_metric_is_rejected_here_too(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"),
                          split={"FD": float("nan")})
        reasons = collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                            expected_count=8)
        assert any("non-finite" in r for r in reasons), reasons


# --------------------------------------------------------------------------- #
# verify_hashes — the §4.3 integrity contract
# --------------------------------------------------------------------------- #
class TestVerifyHashes:
    def _pair(self, tmp_path, overrides=None):
        overrides = overrides or {}
        cells = []
        for arm in ("YAWAUG", "VANL"):
            for kind in ("tbl", "rrob"):
                d = tmp_path / f"{arm}_{kind}"
                d.mkdir()
                p = write_cell(str(d), _cell(arm, kind),
                               **overrides.get((arm, kind), {}))
                cells.append(collect.parse_cell(p))
        return cells

    def test_matched_assignments_pass(self, tmp_path):
        problems = collect.verify_hashes(self._pair(tmp_path))
        assert problems == [], problems

    def test_cross_arm_input_hash_mismatch_is_named(self, tmp_path):
        cells = self._pair(tmp_path, {("VANL", "tbl"): {
            "targets": [f"Office/Office_idx_9/{i}.wav" for i in range(8)]}})
        problems = collect.verify_hashes(cells)
        assert any("input_hash" in p for p in problems), problems

    def test_cross_arm_assignment_hash_mismatch_is_named(self, tmp_path):
        cells = self._pair(tmp_path, {("VANL", "rrob"): {
            "offsets": [1] * 7 + [2]}})
        problems = collect.verify_hashes(cells)
        assert any("assignment_hash" in p for p in problems), problems

    def test_within_arm_Z_R_input_hash_mismatch_is_named(self, tmp_path):
        cells = self._pair(tmp_path, {("YAWAUG", "rrob"): {
            "targets": [f"Bedrooms/Bedrooms_idx_1/{i}.wav" for i in range(8)]}})
        problems = collect.verify_hashes(cells)
        assert any("T<->R" in p or "pairing" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# check_completeness
# --------------------------------------------------------------------------- #
class TestCheckCompleteness:
    def test_five_of_five_is_complete(self):
        got = [(a, "tbl", 8, s) for a in ("YAWAUG", "VANL") for s in V.SEEDS]
        status = collect.check_completeness(got, block=("YAWAUG", "tbl", 8))
        assert status.status == "OK" and status.missing == ()

    def test_four_of_five_is_PENDING_and_carries_no_numbers(self):
        got = [("YAWAUG", "tbl", 8, s) for s in (42, 43, 44, 45)]
        status = collect.check_completeness(got, block=("YAWAUG", "tbl", 8))
        assert status.status == "PENDING"
        assert status.missing == (46,)

    def test_an_unregistered_cell_is_rejected(self):
        with pytest.raises(ValueError, match="not registered"):
            collect.check_completeness([("C4L", "tbl", 8, 42)],
                                       block=("C4L", "tbl", 8))

    def test_the_full_grid_is_exactly_42(self):
        assert len(collect.registered_cells()) == 42


# --------------------------------------------------------------------------- #
# pair_seeds
# --------------------------------------------------------------------------- #
class TestPairSeeds:
    def test_aligns_matching_seeds(self, tmp_path):
        z, r = [], []
        for s in V.SEEDS:
            dz = tmp_path / f"z{s}"; dz.mkdir()
            dr = tmp_path / f"r{s}"; dr.mkdir()
            z.append(collect.parse_cell(write_cell(str(dz), _cell("YAWAUG", "tbl", seed=s))))
            r.append(collect.parse_cell(write_cell(str(dr), _cell("YAWAUG", "rrob", seed=s))))
        pairs, problems = collect.pair_seeds(z, r)
        assert problems == [] and sorted(pairs) == list(V.SEEDS)

    def test_a_missing_partner_is_refused_not_dropped(self, tmp_path):
        dz = tmp_path / "z"; dz.mkdir()
        z = [collect.parse_cell(write_cell(str(dz), _cell("YAWAUG", "tbl", seed=42)))]
        pairs, problems = collect.pair_seeds(z, [])
        assert pairs == {} and any("42" in p for p in problems)


# --------------------------------------------------------------------------- #
# orient_metric
# --------------------------------------------------------------------------- #
class TestOrientMetric:
    def test_lower_better_metric_is_unchanged(self):
        assert collect.orient_metric("T60", [1.0, -2.0]) == [1.0, -2.0]

    def test_higher_better_metric_flips_sign(self):
        # positive must mean WORSE for every metric, so an improvement in a
        # higher-is-better metric becomes a negative (better) oriented value.
        assert collect.orient_metric("RIR_to_GT_RIR_R@1", [1.0, -2.0]) == [-1.0, 2.0]

    def test_aliases_resolve(self):
        assert collect.orient_metric("R@1", [1.0]) == [-1.0]
        assert collect.orient_metric("T60%", [1.0]) == [1.0]

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            collect.orient_metric("made_up", [1.0])


# --------------------------------------------------------------------------- #
# paired_t_ci / holm — against PRECOMPUTED reference values
# --------------------------------------------------------------------------- #
class TestStatistics:
    def test_paired_t_ci_reference(self):
        res = collect.paired_t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert res.n == 5 and res.df == 4
        assert res.mean == pytest.approx(3.0)
        assert res.sd == pytest.approx(1.5811388300841898)
        assert res.se == pytest.approx(0.7071067811865476)
        assert res.t == pytest.approx(4.242640687119285)
        assert res.p == pytest.approx(0.013235599563682695, rel=1e-9)
        assert res.lo == pytest.approx(1.036756838522439)
        assert res.hi == pytest.approx(4.9632431614775605)

    def test_the_critical_value_is_the_df4_two_sided_95pct_one(self):
        assert collect.t_critical(0.05, 4) == pytest.approx(2.7764451051977987)

    def test_ci_is_symmetric_about_the_mean(self):
        res = collect.paired_t_ci([0.5, -0.25, 2.0, 1.25, -1.0])
        assert (res.lo + res.hi) / 2 == pytest.approx(res.mean)

    def test_zero_spread_nonzero_mean_is_a_real_effect(self):
        res = collect.paired_t_ci([2.0] * 5)
        assert res.p == 0.0 and math.isinf(res.t)

    def test_zero_spread_zero_mean_is_no_effect(self):
        res = collect.paired_t_ci([0.0] * 5)
        assert res.p == 1.0 and res.t == 0.0

    def test_fewer_than_two_observations_raises(self):
        with pytest.raises(ValueError):
            collect.paired_t_ci([1.0])

    @pytest.mark.parametrize("pvals,expected", [
        ([0.01, 0.04], [0.02, 0.04]),
        ([0.04, 0.01], [0.04, 0.02]),
        ([0.03, 0.03], [0.06, 0.06]),
        ([0.5, 0.5], [1.0, 1.0]),
        ([0.001, 0.002], [0.002, 0.002]),
    ])
    def test_holm_reference_orderings(self, pvals, expected):
        assert collect.holm(pvals) == pytest.approx(expected)

    def test_holm_is_monotone_in_input_order(self):
        # step-down: an adjusted p never falls below an earlier-ranked one
        adj = collect.holm([0.001, 0.5, 0.02])
        order = sorted(range(3), key=lambda i: [0.001, 0.5, 0.02][i])
        vals = [adj[i] for i in order]
        assert vals == sorted(vals)

    def test_holm_of_empty_is_empty(self):
        assert collect.holm([]) == []


# --------------------------------------------------------------------------- #
# verdicts — plan §5's vocabulary, and NO equivalence claims
# --------------------------------------------------------------------------- #
class TestVerdict:
    def test_significant_and_better_is_superior(self):
        assert collect.verdict_for("T60", mean=-1.0, p_holm=0.01) == "YAWAUG-SUPERIOR"

    def test_significant_and_worse_is_inferior(self):
        assert collect.verdict_for("T60", mean=1.0, p_holm=0.01) == "YAWAUG-INFERIOR"

    def test_direction_is_read_from_the_metric(self):
        # higher-is-better: a POSITIVE difference favours YAWAUG
        assert collect.verdict_for("RIR_to_GT_RIR_R@1", mean=1.0,
                                   p_holm=0.01) == "YAWAUG-SUPERIOR"

    def test_non_significant_is_not_resolved_never_equivalent(self):
        v = collect.verdict_for("T60", mean=-1.0, p_holm=0.2)
        assert v == "NOT STATISTICALLY RESOLVED"
        assert "EQUIV" not in v

    def test_the_vocabulary_is_closed(self):
        assert set(collect.VERDICTS) == {"YAWAUG-SUPERIOR", "YAWAUG-INFERIOR",
                                         "NOT STATISTICALLY RESOLVED"}


# --------------------------------------------------------------------------- #
# gate_report
# --------------------------------------------------------------------------- #
class TestGateReport:
    def test_g1_passes_when_the_control_degrades_by_5_sigma(self):
        g = collect.gate_g1(vctl_t60=10.0, tbl_t60_by_seed={42: 5.0, 43: 5.1, 44: 4.9,
                                                           45: 5.05, 46: 4.95},
                            factor=5.0)
        assert g["status"] == "PASS"
        assert g["observed"] == pytest.approx(10.0 - 5.0)

    def test_g1_fails_when_the_harness_detects_no_degradation(self):
        g = collect.gate_g1(vctl_t60=5.02, tbl_t60_by_seed={42: 5.0, 43: 5.1, 44: 4.9,
                                                           45: 5.05, 46: 4.95},
                            factor=5.0)
        assert g["status"] == "FAIL"

    def test_g1_needs_the_full_seed_block(self):
        g = collect.gate_g1(vctl_t60=10.0, tbl_t60_by_seed={42: 5.0}, factor=5.0)
        assert g["status"] == "PENDING"

    def test_g5_completeness_is_pending_below_five_seeds(self):
        g = collect.gate_g5({("YAWAUG", "tbl", 8): (42, 43, 44, 45)})
        assert g["status"] == "PENDING"

    def test_external_check_formula(self):
        # 3 * sqrt(sa^2 + sb^2) / sqrt(5)
        tol = collect.external_tolerance(0.3, 0.4)
        assert tol == pytest.approx(3.0 * math.sqrt(0.09 + 0.16) / math.sqrt(5))

    def test_external_check_does_not_halt(self):
        chk = collect.external_check("T60", ours=1.0, theirs=5.0, sd_ours=0.1,
                                     sd_theirs=0.1)
        assert chk["exceeds"] is True and chk["halting"] is False

    def test_gate_names_are_the_planned_five(self):
        assert collect.GATE_NAMES == ("G1", "G2", "G3", "G4", "G5")

    # --- a gate that verified NOTHING must never read as green ---------------
    # Found by smoking the collector against the real (still empty) output tree:
    # G3/G4/G5 all returned PASS by vacuous truth over zero cells, which is the
    # exact fail-open reading the whole design exists to refuse.
    def test_g3_over_zero_cells_is_pending_not_pass(self):
        assert collect.gate_g3([])["status"] == "PENDING"

    def test_g3_needs_two_comparable_cells_to_pass(self, tmp_path):
        d = tmp_path / "one"; d.mkdir()
        one = collect.parse_cell(write_cell(str(d), _cell("YAWAUG", "tbl")))
        assert collect.gate_g3([one])["status"] == "PENDING"

    def test_g4_over_zero_cells_is_pending_not_pass(self):
        g = collect.gate_g4([])
        assert g["status"] in ("PENDING", "FAIL")
        assert g["status"] != "PASS"

    def test_g4_surfaces_a_missing_admission_record_for_a_REGISTERED_arm(self, tmp_path):
        # driven by the registered arms, not by whichever arms happen to be on
        # disk, so an unavailable expectation cannot hide behind an absent cell
        g = collect.gate_g4([], registry=str(tmp_path / "no_such_registry.json"))
        assert g["status"] == "FAIL"
        assert any("YAWAUG" in p for p in g["problems"])

    def test_g5_over_zero_blocks_is_pending_not_pass(self):
        g = collect.gate_g5({})
        assert g["status"] == "PENDING"
        assert len(g["pending"]) == len(collect.registered_blocks()) == 8

    def test_g5_passes_only_when_every_registered_block_is_full(self):
        full = {b: tuple(V.SEEDS) for b in collect.registered_blocks()}
        assert collect.gate_g5(full)["status"] == "PASS"
        one_short = dict(full)
        one_short[("YAWAUG", "tbl", 8)] = (42, 43, 44, 45)
        assert collect.gate_g5(one_short)["status"] == "PENDING"


# --------------------------------------------------------------------------- #
# render_tables — golden fixture
# --------------------------------------------------------------------------- #
GOLDEN_H1 = """\
| metric | Δ (YAWAUG − VANL) | 95% CI | p | p (Holm-2) | verdict |
| --- | --- | --- | --- | --- | --- |
| T60 ↓ | -0.500 | [-0.900, -0.100] | 0.0300 | 0.0600 | NOT STATISTICALLY RESOLVED |
| R@1 ↑ | 1.250 | [0.750, 1.750] | 0.0010 | 0.0020 | YAWAUG-SUPERIOR |
"""


class TestRenderTables:
    def test_h1_table_matches_the_golden_fixture(self):
        rows = [
            {"metric": "T60", "mean": -0.5, "lo": -0.9, "hi": -0.1, "p": 0.03,
             "p_holm": 0.06, "verdict": "NOT STATISTICALLY RESOLVED"},
            {"metric": "RIR_to_GT_RIR_R@1", "mean": 1.25, "lo": 0.75, "hi": 1.75,
             "p": 0.001, "p_holm": 0.002, "verdict": "YAWAUG-SUPERIOR"},
        ]
        assert collect.render_h1_table(rows) == GOLDEN_H1

    def test_a_blocked_contrast_renders_the_word_not_a_number(self):
        md = collect.render_h1_table([], blocked="input_hash mismatch at (K=8, s=43)")
        assert "BLOCKED" in md and "input_hash mismatch" in md
        assert not any(ch.isdigit() and ch not in "128" for ch in md.split("BLOCKED")[0])

    def test_a_pending_block_renders_PENDING_not_a_mean(self):
        md = collect.render_block_table([{"arm": "YAWAUG", "K": 8,
                                          "status": "PENDING", "seeds": (42, 43)}],
                                        metrics=("T60",))
        assert "PENDING" in md and "±" not in md

    def test_the_json_bundle_round_trips(self):
        bundle = collect.results_bundle({"gates": {}, "h1": {}, "cells": []})
        assert json.loads(json.dumps(bundle))["schema_version"] == collect.SCHEMA_VERSION


# =========================================================================== #
# eval-r2 REVISE: the findings the 64-test suite could not have caught.
# Everything below builds a SYNTHETIC FULL GRID on disk and drives the real
# build_results()/render_report(), because findings 1 and 2 were both invisible
# to leaf-function tests: the gates were correct in isolation and simply never
# consulted, and the report was correct in isolation and simply never emitted.
# =========================================================================== #
import hashlib

ARM_TREE = {"YAWAUG": ("exp15_YAWAUG", "FLAC_exp15_YAWAUG", "exp15_YAWAUG"),
            "VANL": ("exp11_VANL", "FLAC_exp11_VANL", "exp11_VANL")}
SHA = {"YAWAUG": "1" * 64, "VANL": "2" * 64}


def _admission(tmp_path):
    """Synthetic control + registry admitting the fixtures' digests."""
    control = tmp_path / "control.json"
    registry = tmp_path / "registry.json"
    json.dump({"_meta": {"expect_step": V.STEP},
               "checkpoint": {"path": "x.ckpt", "sha256": SHA["VANL"], "bytes": 1,
                              "global_step": V.STEP,
                              "embedded_config_canonical_sha256": "c" * 64,
                              "ema_inventory_sha256": "e" * 64},
               "config": {"sha256": "d" * 64, "canonical_sha256": "c" * 64},
               "exp_11_cross_references": {"manifest_sha256": "f" * 64}},
              open(control, "w"))
    json.dump({"arms": {"YAWAUG": {"final_ckpt_sha256": SHA["YAWAUG"],
                                   "final_step": V.STEP, "config_sha256": "d" * 64,
                                   "manifest_sha256": "f" * 64}},
               "legs": {"YAWAUG": [{"step": V.STEP, "ckpt_sha256": SHA["YAWAUG"],
                                    "ckpt_bytes": 1, "ckpt_path": "y.ckpt",
                                    "audit": {"embedded_config_canonical_sha256": "c" * 64,
                                              "ema_inventory_sha256": "e" * 64}}]}},
              open(registry, "w"))
    return str(control), str(registry)


def build_full_grid(tmp_path, values=None, n=8, pin=PIN, skip=(), mutate=None):
    """Write all 42 registered cells under a synthetic output root.

    ``values(cell, metric) -> float`` decides every number, so a test can make
    YAWAUG flatter, make seed 42 differ from the block mean, or blow a hash.
    """
    root = tmp_path / "outputs"
    for arm, tree in ARM_TREE.items():
        d = root.joinpath(*tree, "checkpoints")
        d.mkdir(parents=True, exist_ok=True)
        (d / "epoch=8-step=40000.ckpt").touch()
    for cell in V.expected_grid():
        if (cell.arm, cell.cell, int(cell.k), int(cell.seed)) in skip:
            continue
        ckdir = str(root.joinpath(*ARM_TREE[cell.arm], "checkpoints"))
        split = {m: (values(cell, m) if values else 1.0)
                 for m in V.REQUIRED_SPLIT_METRICS}
        scene = {m: (values(cell, m) if values else 1.0)
                 for m in V.REQUIRED_SCENE_METRICS}
        kw = dict(n=n, pin=pin, ckpt_sha=SHA[cell.arm], split=split, scene=scene)
        if cell.cell == "rrob":
            # G2 recomputes the draw with the evaluator's own draw_yaw_offsets, so
            # a happy-path fixture has to BE the registered draw. (The first
            # version used a made-up formula and G2 caught it — the gate working
            # exactly as intended.)
            kw["offsets"] = list(collect.golden_offsets(int(cell.seed), n))
        if mutate:
            kw = mutate(cell, kw) or kw
        write_cell(ckdir, cell, **kw)
    return str(root)


def run(tmp_path, **kw):
    control, registry = _admission(tmp_path)
    root = build_full_grid(tmp_path, **kw)
    return collect.build_results(root, pin=PIN, expected_count=kw.get("n", 8),
                                 control=control, registry=registry)


def flat_values(cell, metric):
    """YAWAUG is FLATTER under rotation and slightly better at theta=0."""
    base = {"T60": 10.0, "C50": 5.0, "EDT": 20.0, "FD": 0.5,
            "Invalid T60": 0.0}.get(metric, 30.0)
    seed_jitter = (int(cell.seed) - 44) * 0.01
    arm_gain = -0.5 if cell.arm == "YAWAUG" else 0.0
    if metric in ("RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10",
                  "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10"):
        arm_gain = 0.5 if cell.arm == "YAWAUG" else 0.0     # higher = better
        rot = -1.0 if cell.cell == V.CELLS[1] else 0.0
        if cell.cell == "rrob":
            rot = -3.0 if cell.arm == "VANL" else -0.5      # VANL degrades more
        return base + arm_gain + rot + seed_jitter
    rot = 0.0
    if cell.cell == "rrob":
        rot = 3.0 if cell.arm == "VANL" else 0.5            # VANL degrades more
    if cell.cell == "vctl":
        rot = 40.0 if cell.arm == "VANL" else 1.0           # G1 positive control
    return base + arm_gain + rot + seed_jitter


class TestFullGridHappyPath:
    def test_every_gate_passes_and_every_hypothesis_renders(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        for name in collect.GATE_NAMES:
            assert res["gates"][name]["status"] == "PASS", (name, res["gates"][name])
        assert res["disposition"]["halt"] == [] and res["disposition"]["pending"] == []
        for k in ("8", "1"):
            for h in ("H1", "H2", "H3"):
                block = res["hypotheses"][k][h]
                assert block["blocked"] is None and block["pending"] is None, (k, h)
                assert len(block["rows"]) == 2, (k, h)

    def test_g3_and_g4_report_the_COMPLETE_obligation_set(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        g3, g4 = res["gates"]["G3"], res["gates"]["G4"]
        assert g3["checked"] == g3["expected"] == 50
        # 40, not 42: review F3's V-gating leak closed in G4 too — a V cell that
        # has not landed must not make the gate PENDING and suppress the readout.
        assert g4["checked"] == g4["expected"] == 40

    def test_h1_yields_a_directional_verdict(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        verdicts = {r["metric"]: r["verdict"] for r in res["hypotheses"]["8"]["H1"]["rows"]}
        assert verdicts["T60"] == "YAWAUG-SUPERIOR"
        assert verdicts["RIR_to_GT_RIR_R@1"] == "YAWAUG-SUPERIOR"

    def test_h2_orientation_a_flatter_YAWAUG_gives_positive_d(self, tmp_path):
        # delta = orient(m_R - m_T) is POSITIVE-is-worse; VANL degrades more, so
        # d = delta(VANL) - delta(YAWAUG) must come out POSITIVE on both
        # co-primaries regardless of each metric's own direction.
        res = run(tmp_path, values=flat_values)
        for row in res["hypotheses"]["8"]["H2"]["rows"]:
            assert row["mean"] > 0, row

    def test_h3_is_secondary_and_carries_no_holm_or_verdict(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        for row in res["hypotheses"]["8"]["H3"]["rows"]:
            assert "SECONDARY" in row["family"]
            assert "p_holm" not in row and "verdict" not in row

    def test_the_report_has_every_planned_section(self, tmp_path):
        md = collect.render_report(run(tmp_path, values=flat_values))
        for heading in ("## Validity gates", "## K = 8 (confirmatory)",
                        "## K = 1 (descriptive repeat)",
                        "### H1 — clean cost/benefit",
                        "### H2 — does augmentation buy flatness?",
                        "### H3 — absolute deployment under rotation",
                        "### T block (θ=0) — mean ± std over seeds",
                        "### R block (random yaw) — mean ± std over seeds",
                        "## Validity-control cells (V block)",
                        "## External reproduction checks (non-halting)",
                        "## Quarantined, descriptive only — RIR_to_geom_R@k",
                        "## Scope of inference"):
            assert heading in md, heading

    def test_the_scope_statement_is_mandatory_and_complete(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        md = collect.render_report(res)
        for claim in ("exactly ONE", "EVALUATION-time", "checkpoint-band",
                      "16-leg chain", "plan §13"):
            assert claim in res["scope_of_inference"], claim
            assert claim in md, claim
            assert claim in json.dumps(collect.results_bundle(res),
                                       ensure_ascii=False)

    def test_the_v_readouts_separate_the_gate_from_the_mechanism(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        roles = {r["arm"]: r["role"] for r in res["v_readouts"]}
        assert roles["VANL"] == "G1 positive control"
        assert "DESCRIPTIVE ONLY" in next(r["note"] for r in res["v_readouts"]
                                          if r["arm"] == "YAWAUG")

    def test_the_quarantined_family_is_rendered_and_labelled(self, tmp_path):
        md = collect.render_report(run(tmp_path, values=flat_values))
        tail = md.split("## Quarantined")[1]
        assert "Confounded by construction" in tail
        assert "R@1" in tail or "RIR_to_geom" in tail

    def test_the_json_bundle_carries_the_whole_report(self, tmp_path):
        bundle = collect.results_bundle(run(tmp_path, values=flat_values))
        for key in ("gates", "disposition", "hypotheses", "blocks", "v_readouts",
                    "externals", "scope_of_inference", "aggregation"):
            assert key in bundle, key
        json.loads(json.dumps(bundle))          # must round-trip


class TestGatesActuallyGate:
    """Finding 1: the gates were right and simply never consulted."""

    def test_g1_uses_seed_42_not_the_five_seed_mean(self):
        # ASYMMETRIC on purpose: seed 42 is far from the block mean, so a
        # implementation subtracting the mean gets a different answer. The old
        # test used a symmetric block where the two coincide.
        by_seed = {42: 4.0, 43: 5.0, 44: 5.0, 45: 6.0, 46: 10.0}
        g = collect.gate_g1(vctl_t60=9.0, tbl_t60_by_seed=by_seed, factor=1.0)
        assert g["reference_seed"] == 42
        assert g["reference"] == pytest.approx(4.0)
        assert g["observed"] == pytest.approx(5.0)          # 9 - 4, NOT 9 - 6
        assert g["sigma"] == pytest.approx(st_stdev(by_seed.values()))

    def test_g1_failure_HALTS_every_hypothesis(self, tmp_path):
        def no_degradation(cell, metric):
            v = flat_values(cell, metric)
            if cell.cell == "vctl" and cell.arm == "VANL" and metric == "T60":
                return 10.02                    # essentially no degradation
            return v
        res = run(tmp_path, values=no_degradation)
        assert res["gates"]["G1"]["status"] == "FAIL"
        assert res["disposition"]["halt"]
        for k in ("8", "1"):
            for h in ("H1", "H2", "H3"):
                assert res["hypotheses"][k][h]["blocked"], (k, h)
                assert res["hypotheses"][k][h]["rows"] == []
        md = collect.render_report(res)
        assert "HALTED" in md and "BLOCKED" in md

    def test_g2_pending_suppresses_every_number(self, tmp_path):
        # drop the K=8 seed-42 R cell: G2 has nothing to recompute against
        res = run(tmp_path, values=flat_values,
                  skip={("YAWAUG", "rrob", 8, 42), ("VANL", "rrob", 8, 42)})
        assert res["gates"]["G2"]["status"] == "PENDING"
        assert res["disposition"]["pending"]
        for h in ("H1", "H2", "H3"):
            assert res["hypotheses"]["8"][h]["rows"] == []

    def test_g4_failure_HALTS_every_hypothesis(self, tmp_path):
        def wrong_digest(cell, kw):
            if cell.arm == "YAWAUG":
                kw["ckpt_sha"] = "9" * 64
            return kw
        res = run(tmp_path, values=flat_values, mutate=wrong_digest)
        # a wrong digest is caught by the per-cell validator first, so the cells
        # are REJECTED and G4's obligations go unmet — either way, no numbers
        assert res["gates"]["G4"]["status"] in ("FAIL", "PENDING")
        for h in ("H1", "H2", "H3"):
            assert res["hypotheses"]["8"][h]["rows"] == []

    def test_g5_partial_suppresses_every_number(self, tmp_path):
        res = run(tmp_path, values=flat_values, skip={("YAWAUG", "tbl", 1, 46)})
        assert res["gates"]["G5"]["status"] == "PENDING"
        for h in ("H1", "H2", "H3"):
            assert res["hypotheses"]["8"][h]["rows"] == []
        assert "PENDING" in collect.render_report(res)

    def test_only_the_ten_K8_T_cells_is_NOT_enough_to_publish(self, tmp_path):
        # the exact scenario finding 1 described: H1's inputs are complete while
        # G1/G2/G5 are still pending. It must not print a number.
        keep = {("YAWAUG", "tbl", 8), ("VANL", "tbl", 8)}
        skip = {(c.arm, c.cell, int(c.k), int(c.seed)) for c in V.expected_grid()
                if (c.arm, c.cell, int(c.k)) not in keep}
        res = run(tmp_path, values=flat_values, skip=skip)
        assert res["hypotheses"]["8"]["H1"]["rows"] == []
        assert "BLOCKED" in collect.render_report(res) or "PENDING" in collect.render_report(res)


class TestG3ScopedBlocking:
    def test_a_T_block_hash_violation_blocks_H1_at_that_K(self, tmp_path):
        def break_t_hash(cell, kw):
            if cell.arm == "VANL" and cell.cell == "tbl" and int(cell.k) == 8 \
                    and int(cell.seed) == 43:
                kw["targets"] = [f"Other/Other_idx_0/{i}.wav" for i in range(8)]
            return kw
        res = run(tmp_path, values=flat_values, mutate=break_t_hash)
        assert res["gates"]["G3"]["status"] == "FAIL"
        assert res["hypotheses"]["8"]["H1"]["blocked"]
        assert res["hypotheses"]["8"]["H1"]["rows"] == []

    def test_g3_partial_evidence_is_pending_not_pass(self, tmp_path):
        res = run(tmp_path, values=flat_values, skip={("VANL", "rrob", 1, 46)})
        g3 = res["gates"]["G3"]
        assert g3["status"] == "PENDING"
        assert g3["checked"] < g3["expected"] == 50


class TestExternalChecksIntegration:
    def test_externals_are_present_and_non_halting(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        assert res["externals"]
        for chk in res["externals"]:
            assert chk["halting"] is False
        assert "non-halting" in collect.render_report(res).lower()

    def test_within_and_beyond_tolerance(self):
        near = collect.external_check("T60", ours=1.0, theirs=1.01,
                                      sd_ours=0.5, sd_theirs=0.5)
        far = collect.external_check("T60", ours=1.0, theirs=9.0,
                                     sd_ours=0.01, sd_theirs=0.01)
        assert near["exceeds"] is False and far["exceeds"] is True
        assert near["halting"] is False and far["halting"] is False


class TestSnapshotConsistency:
    """Finding 5: validate exactly the payloads that get aggregated."""

    def test_validator_core_accepts_parsed_payloads(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"))
        art = collect.parse_cell(path)
        assert V.validate_payloads(art.record, art.meta, art.stream, art.cell,
                                   pin=PIN, ckpt_sha=CKPT_SHA,
                                   expected_count=8) == []

    def test_a_concurrent_replacement_cannot_validate_A_and_aggregate_B(self, tmp_path):
        """Validate the snapshot we keep, not whatever the file says later."""
        control, registry = _admission(tmp_path)
        root = build_full_grid(tmp_path, values=flat_values)
        cells, _, rejected = collect.collect_cells(root, pin=PIN, expected_count=8)
        assert rejected == [] and len(cells) == 42
        kept = {V.eval_name(a.cell): a.record["metrics"]["T60"] for a in cells}
        # now REPLACE one artifact on disk with a corrupted version
        victim = next(a for a in cells if a.cell.arm == "YAWAUG"
                      and a.cell.cell == "tbl" and int(a.cell.k) == 8)
        bad = dict(victim.record, metrics=dict(victim.record["metrics"], T60=999.0))
        json.dump(bad, open(victim.path, "w"))
        # the retained snapshot is untouched: the aggregate uses what was validated
        assert kept[V.eval_name(victim.cell)] != 999.0
        routed = collect.route_observations(cells)
        assert 999.0 not in routed[("YAWAUG", "tbl", 8)]["T60"].values()


class TestGoldenReport:
    def test_markdown_golden_is_stable(self, tmp_path):
        md = collect.render_report(run(tmp_path, values=flat_values))
        # the H1 table is the one a reader acts on; pin it exactly
        h1 = md.split("### H1 — clean cost/benefit (m_T YAWAUG vs VANL)")[1] \
               .split("### H2")[0].strip()
        lines = [l for l in h1.splitlines() if l.strip()]
        # the family label now precedes the table (K=8 confirmatory, K=1 not)
        assert lines[0] == "_CONFIRMATORY (Holm over 2 co-primaries)_"
        assert lines[1] == (
            "| metric | Δ (YAWAUG − VANL) | 95% CI | p | p (Holm-2) | verdict |")
        assert lines[3].startswith("| T60 ↓ | -0.500 |")
        assert lines[4].startswith("| R@1 ↑ | 0.500 |")

    def test_the_bundle_is_STRICTLY_valid_json(self, tmp_path):
        # allow_nan=False is what JSON.parse enforces; a bare Infinity token
        # would make the HTML page fail to load the results it is built on.
        bundle = collect.results_bundle(run(tmp_path, values=flat_values))
        text = json.dumps(bundle, allow_nan=False, ensure_ascii=False)
        assert json.loads(text)["schema_version"] == collect.SCHEMA_VERSION
        # the zero-spread t is preserved, as a name rather than a bare token
        assert '"Infinity"' in text or "Infinity" not in text

    def test_json_golden_keys_are_stable(self, tmp_path):
        bundle = collect.results_bundle(run(tmp_path, values=flat_values))
        assert bundle["schema_version"] == collect.SCHEMA_VERSION
        assert bundle["co_primary"] == ["T60", "RIR_to_GT_RIR_R@1"]
        assert set(bundle["hypotheses"]) == {"8", "1"}
        assert set(bundle["hypotheses"]["8"]) == {"H1", "H2", "H3"}
        assert bundle["aggregation"]["split_level"][0] == "FD"


def st_stdev(values):
    import statistics
    return statistics.stdev(list(values))


# =========================================================================== #
# INTEGRATIVE review (NO-GO) — F2, F3, F4, F5, F7, F8.
# =========================================================================== #
class TestK1IsDescriptiveOnly:
    """F2: exactly ONE confirmatory family — H1's two K=8 co-primaries."""

    def test_k8_is_confirmatory_with_holm_and_verdicts(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        h1 = res["hypotheses"]["8"]["H1"]
        assert h1["confirmatory"] is True and "CONFIRMATORY" in h1["family"]
        for row in h1["rows"]:
            assert "p_holm" in row and "verdict" in row

    def test_k1_is_descriptive_with_neither(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        h1 = res["hypotheses"]["1"]["H1"]
        assert h1["confirmatory"] is False and "DESCRIPTIVE" in h1["family"]
        for row in h1["rows"]:
            assert "p_holm" not in row and "verdict" not in row

    def test_the_k1_table_has_no_holm_or_verdict_columns(self, tmp_path):
        md = collect.render_report(run(tmp_path, values=flat_values))
        k1 = md.split("## K = 1 (descriptive repeat)")[1].split("### H2")[0]
        assert "Holm-2" not in k1 and "YAWAUG-SUPERIOR" not in k1
        k8 = md.split("## K = 8 (confirmatory)")[1].split("### H2")[0]
        assert "Holm-2" in k8

    def test_the_k1_heading_says_descriptive(self, tmp_path):
        md = collect.render_report(run(tmp_path, values=flat_values))
        assert "## K = 1 (descriptive repeat)" in md
        assert "**H1 (K=1, DESCRIPTIVE repeat)" in md or "K=1, DESCRIPTIVE" in md \
            or "DESCRIPTIVE (K=1 repeat" in md


class TestVCellsDoNotGate:
    """F3: YAWAUG V is descriptive/mechanistic and carries no gate role."""

    def test_v_owes_no_assignment_obligation(self):
        kinds = {o[1] for o in collect.g3_obligations() if o[0] == "assignment_hash"}
        assert kinds == {"rrob"}, kinds

    def test_the_obligation_count_drops_to_50(self):
        # 20 cross-arm input_hash (T and R only) + 10 R assignment + 20 pairings.
        # V contributes NOTHING: it is descriptive and carries no gate role.
        assert len(collect.g3_obligations()) == 50
        assert not any(o[1] == "vctl" for o in collect.g3_obligations())

    def test_a_V_hash_mismatch_does_not_block_any_hypothesis(self, tmp_path):
        def break_v(cell, kw):
            if cell.cell == "vctl" and cell.arm == "YAWAUG":
                kw["offsets"] = [7] * 8          # not the fixed 90-degree shift
            return kw
        res = run(tmp_path, values=flat_values, mutate=break_v)
        for k in ("8", "1"):
            for h in ("H1", "H2", "H3"):
                assert res["hypotheses"][k][h]["blocked"] is None, (k, h)

    def test_a_missing_YAWAUG_V_cell_does_not_block_inference(self, tmp_path):
        res = run(tmp_path, values=flat_values, skip={("YAWAUG", "vctl", 8, 42)})
        assert res["gates"]["G3"]["status"] == "PASS"
        assert res["hypotheses"]["8"]["H1"]["rows"], "H1 was blocked by a V cell"


class TestInvalidT60Routing:
    """F4: §13 puts Invalid T60 in the acoustic family."""

    def test_the_validator_requires_it_per_scene(self):
        assert "Invalid T60" in V.REQUIRED_SCENE_METRICS

    def test_a_scene_missing_it_is_refused(self, tmp_path):
        path = write_cell(str(tmp_path), _cell("YAWAUG", "tbl"),
                          scene_drop=("Invalid T60",))
        reasons = collect.validate_protocol(path, pin=PIN, ckpt_sha=CKPT_SHA,
                                            expected_count=8)
        assert any("Invalid T60" in r for r in reasons), reasons

    def test_it_routes_as_a_scene_mean_not_split_level(self):
        assert collect.aggregation_source("Invalid T60") == "scene-mean"

    def test_it_has_a_direction_without_mutating_exp14s_table(self):
        assert collect.metric_direction("Invalid T60") == "lower"
        import yaw_gen_collect as G14
        assert "Invalid T60" not in G14.METRIC_DIRECTION

    def test_it_appears_in_the_descriptive_tables_not_the_confirmatory_family(self, tmp_path):
        assert "Invalid T60" in collect.DESCRIPTIVE_METRICS
        assert "Invalid T60" not in collect.CO_PRIMARY
        md = collect.render_report(run(tmp_path, values=flat_values))
        assert "Invalid T60" in md

    def test_the_real_exp14_artifact_still_satisfies_the_widened_schema(self):
        real = ("outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/"
                "epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8"
                "_rotrand42_rotrand42.json")
        if not os.path.isfile(real):
            pytest.skip("exp_14 campaign artifacts not present in this tree")
        rec = json.load(open(real))
        assert V._per_scene_reasons(rec) == []


class TestExp11ExternalCheck:
    """F5: both pre-declared external references, descriptive and non-halting."""

    def test_both_sources_are_reported(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        assert {c["source"] for c in res["externals"]} == {"exp_14 Z", "exp_11 Q9"}

    def test_neither_source_halts(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        assert all(c["halting"] is False for c in res["externals"])

    def test_exp11_acoustic_is_declared_incomparable_not_compared(self):
        # exp_11's Q9 cells predate --record-per-scene, so their T60 is a
        # split-level quantity — a different estimand from §13's scene mean.
        ours = {"T60": {s: 1.0 for s in V.SEEDS},
                "RIR_to_GT_RIR_R@1": {s: 1.0 for s in V.SEEDS}}
        checks = collect._one_external("exp_11 Q9", ours, {"T60": {}, "RIR_to_GT_RIR_R@1": {}},
                                       collect.CO_PRIMARY)
        t60 = next(c for c in checks if c["metric"] == "T60")
        assert t60["status"] == "UNAVAILABLE"
        assert "different estimand" in t60["detail"]


class TestG2ProbeIdentity:
    """F8: the probe is the registered YAWAUG/rrob/K8/s42 cell, and is named."""

    def test_g2_names_the_registered_probe(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        assert res["gates"]["G2"]["probe"] == "exp15_YAWAUG_rrob_rotrand42_S40000_s42_K8"
        assert "exp15_YAWAUG_rrob" in res["gates"]["G2"]["detail"]

    def test_a_VANL_R_cell_cannot_stand_in_for_the_probe(self, tmp_path):
        res = run(tmp_path, values=flat_values, skip={("YAWAUG", "rrob", 8, 42)})
        assert res["gates"]["G2"]["status"] == "PENDING"
        assert "has not landed" in res["gates"]["G2"]["detail"]


class TestPublishRowTransaction:
    """F7: the §6.9 trigger, executable and tested."""

    def _publish(self):
        import importlib.util
        path = ("worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_publish_row.py")
        spec = importlib.util.spec_from_file_location("yaw_aug_publish_row", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_the_two_row_specs_are_registered_additively(self):
        pub = self._publish()
        specs, module = pub.row_specs()
        assert len(specs) == 2
        assert {s[2] for s in specs} == {1, 8}
        assert all(s[4] == "exp15" for s in specs)
        # ...and exp_11 / exp_14 specs are still present and untouched
        labels = [r[0] for r in module.ROWS]
        assert any("exp_11 baseline" in l for l in labels)
        assert any("exp_14 Z" in l for l in labels)

    def test_readiness_is_T_only_and_fires_when_T_is_complete(self, tmp_path):
        pub = self._publish()
        control, registry = _admission(tmp_path)
        # ONLY the YAWAUG+VANL T cells: no R, no V at all
        keep = {("YAWAUG", "tbl", 8), ("YAWAUG", "tbl", 1),
                ("VANL", "tbl", 8), ("VANL", "tbl", 1)}
        skip = {(c.arm, c.cell, int(c.k), int(c.seed)) for c in V.expected_grid()
                if (c.arm, c.cell, int(c.k)) not in keep}
        root = build_full_grid(tmp_path, values=flat_values, skip=skip)
        got = pub.readiness(root, pin=PIN, expected_count=8,
                            control=control, registry=registry)
        assert got["ready"] is True, got["reasons"]
        # keys are per ARM and K now: readiness requires BOTH arms (re-review F3)
        assert got["seeds"]["YAWAUG/K8"] == list(V.SEEDS)
        assert got["seeds"]["VANL/K8"] == list(V.SEEDS)

    def test_readiness_refuses_a_four_seed_block(self, tmp_path):
        pub = self._publish()
        control, registry = _admission(tmp_path)
        root = build_full_grid(tmp_path, values=flat_values,
                               skip={("YAWAUG", "tbl", 8, 46)})
        got = pub.readiness(root, pin=PIN, expected_count=8,
                            control=control, registry=registry)
        assert got["ready"] is False
        assert any("4/5" in r for r in got["reasons"]), got["reasons"]

    def test_the_generator_routes_exp15_rows_through_scene_means(self, tmp_path):
        """§13: T60/C50/EDT are ten-family means, retrieval stays split-level."""
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "raws"
        d.mkdir()
        files = []
        for seed in V.SEEDS:
            cell = _cell("YAWAUG", "tbl", seed=seed)
            files.append(write_cell(str(d), cell,
                                    split={"T60": 99.0, "RIR_to_GT_RIR_R@1": 7.0},
                                    scene={"T60": 3.0}))
        values, n = module.agg_files_exp15(files)
        assert n == 5
        # the scene mean (3.0) is published, NOT the split-level 99.0
        assert values["T60"][0] == pytest.approx(3.0)
        # retrieval stays split-level
        assert values["R@1"][0] == pytest.approx(7.0)

    def test_the_generator_refuses_a_row_without_by_scene(self, tmp_path):
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "raws2"
        d.mkdir()
        files = [write_cell(str(d), _cell("YAWAUG", "tbl", seed=s)) for s in V.SEEDS]
        for f in files:                       # strip the block §13 routing needs
            rec = json.load(open(f))
            rec.pop("by_scene")
            json.dump(rec, open(f, "w"))
        with pytest.raises(ValueError, match="by_scene"):
            module.agg_files_exp15(files)

    def test_the_generator_refuses_a_non_T_cell_as_a_model_row(self, tmp_path):
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "raws3"
        d.mkdir()
        files = [write_cell(str(d), _cell("YAWAUG", "rrob", seed=s)) for s in V.SEEDS]
        ok, problems = module.validate_exp15_cell(files, expected_k=8)
        assert ok is False
        assert any("theta=0" in p or "T (" in p for p in problems), problems


# =========================================================================== #
# RE-REVIEW (NO-GO): F3 two-K transaction + both arms, F4 V readout withheld.
# =========================================================================== #
class TestVReadoutWithheld:
    """Re-review F4: a defective V cell publishes the WORD, never the number."""

    def _defective(self, tmp_path):
        def break_v(cell, kw):
            if cell.cell == "vctl" and cell.arm == "YAWAUG":
                # individually VALID (offsets still the fixed 90-degree shift) but
                # a different item set from VANL's V cell -> input_hash mismatch
                kw["targets"] = [f"Office/Office_idx_9/{i}.wav" for i in range(8)]
            return kw
        return run(tmp_path, values=flat_values, mutate=break_v)

    def test_the_defect_is_recorded_against_the_cell(self, tmp_path):
        res = self._defective(tmp_path)
        row = next(r for r in res["v_readouts"] if r["arm"] == "YAWAUG")
        assert row["withheld"], row
        assert row["T60"] is None

    def test_the_number_is_absent_from_the_markdown(self, tmp_path):
        res = self._defective(tmp_path)
        md = collect.render_report(res)
        v_section = md.split("## Validity-control cells (V block)")[1].split("##")[0]
        assert "WITHHELD (hash defect" in v_section
        # the YAWAUG V value under flat_values is 10.480 — it must not appear
        assert "10.480" not in v_section

    def test_the_number_is_absent_from_the_json(self, tmp_path):
        res = self._defective(tmp_path)
        blob = json.dumps(collect.results_bundle(res), ensure_ascii=False)
        row = next(r for r in res["v_readouts"] if r["arm"] == "YAWAUG")
        assert row["T60"] is None
        assert '"T60": 10.48' not in blob

    def test_a_healthy_V_cell_still_publishes_its_number(self, tmp_path):
        res = run(tmp_path, values=flat_values)
        for row in res["v_readouts"]:
            assert row["withheld"] is None and row["T60"] is not None

    def test_hypotheses_remain_unblocked_by_the_defect(self, tmp_path):
        res = self._defective(tmp_path)
        assert res["hypotheses"]["8"]["H1"]["blocked"] is None
        assert res["hypotheses"]["8"]["H1"]["rows"]


class TestTwoKTransaction:
    """Re-review F3: both arms, both K, or nothing publishes."""

    def _publish(self):
        import importlib.util
        path = "worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_publish_row.py"
        spec = importlib.util.spec_from_file_location("yaw_aug_publish_row", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_yawaug_alone_is_NOT_ready(self, tmp_path):
        """The cross-arm equalities cannot be tested from one arm's cells."""
        pub = self._publish()
        control, registry = _admission(tmp_path)
        keep = {("YAWAUG", "tbl", 8), ("YAWAUG", "tbl", 1)}
        skip = {(c.arm, c.cell, int(c.k), int(c.seed)) for c in V.expected_grid()
                if (c.arm, c.cell, int(c.k)) not in keep}
        root = build_full_grid(tmp_path, values=flat_values, skip=skip)
        got = pub.readiness(root, pin=PIN, expected_count=8,
                            control=control, registry=registry)
        assert got["ready"] is False
        assert any("VANL" in r for r in got["reasons"]), got["reasons"]

    def test_both_arms_both_K_is_ready(self, tmp_path):
        pub = self._publish()
        control, registry = _admission(tmp_path)
        keep = {(a, "tbl", k) for a in ("YAWAUG", "VANL") for k in (1, 8)}
        skip = {(c.arm, c.cell, int(c.k), int(c.seed)) for c in V.expected_grid()
                if (c.arm, c.cell, int(c.k)) not in keep}
        root = build_full_grid(tmp_path, values=flat_values, skip=skip)
        got = pub.readiness(root, pin=PIN, expected_count=8,
                            control=control, registry=registry)
        assert got["ready"] is True, got["reasons"]
        assert set(got["seeds"]) == {"YAWAUG/K1", "YAWAUG/K8", "VANL/K1", "VANL/K8"}

    def test_the_untested_equalities_are_named_not_assumed(self, tmp_path):
        pub = self._publish()
        control, registry = _admission(tmp_path)
        keep = {("YAWAUG", "tbl", 8), ("YAWAUG", "tbl", 1), ("VANL", "tbl", 8)}
        skip = {(c.arm, c.cell, int(c.k), int(c.seed)) for c in V.expected_grid()
                if (c.arm, c.cell, int(c.k)) not in keep}
        root = build_full_grid(tmp_path, values=flat_values, skip=skip)
        got = pub.readiness(root, pin=PIN, expected_count=8,
                            control=control, registry=registry)
        assert got["ready"] is False
        assert any("cross-arm input_hash" in r for r in got["reasons"]), got["reasons"]

    def test_the_checklist_is_concrete_and_emitted(self):
        pub = self._publish()
        for fragment in ("gen_model_comparison.py", "git add", "git pull --rebase",
                         "exp_15 results:", "DO NOT repin", "yaw_aug_analysis.md"):
            assert fragment in pub.PUBLISH_CHECKLIST, fragment

    def test_the_generator_requires_exactly_five_files(self, tmp_path):
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "six"
        d.mkdir()
        files = [write_cell(str(d), _cell("YAWAUG", "tbl", seed=s)) for s in V.SEEDS]
        files.append(files[0])                    # a duplicate the glob might catch
        ok, problems = module.validate_exp15_cell(files, expected_k=8)
        assert ok is False
        assert any("need exactly 5" in p for p in problems), problems

    def test_the_generator_requires_the_campaign_pin_in_the_sidecar(self, tmp_path):
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "nopin"
        d.mkdir()
        files = [write_cell(str(d), _cell("YAWAUG", "tbl", seed=s)) for s in V.SEEDS]
        for f in files:                           # strip the pin the row must cite
            meta = json.load(open(f + ".screenmeta.json"))
            meta["commit"] = None
            json.dump(meta, open(f + ".screenmeta.json", "w"))
        ok, problems = module.validate_exp15_cell(files, expected_k=8)
        assert ok is False
        assert any("40-hex campaign pin" in p for p in problems), problems

    def test_the_generator_refuses_cells_spanning_two_pins(self, tmp_path):
        pub = self._publish()
        _specs, module = pub.row_specs()
        d = tmp_path / "twopins"
        d.mkdir()
        files = [write_cell(str(d), _cell("YAWAUG", "tbl", seed=s)) for s in V.SEEDS]
        meta = json.load(open(files[0] + ".screenmeta.json"))
        meta["commit"] = "b" * 40
        json.dump(meta, open(files[0] + ".screenmeta.json", "w"))
        ok, problems = module.validate_exp15_cell(files, expected_k=8)
        assert ok is False
        assert any("campaign pins" in p for p in problems), problems
