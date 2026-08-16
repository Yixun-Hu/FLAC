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
               split=None, scene=None, offsets=None, targets=None,
               meta_override=None, rec_override=None):
    """Write one cell's three artifacts under ``root``; return the metrics path."""
    mode, deg, rseed = V.rotation_expectation(cell)
    ckpt = os.path.join(root, "epoch=8-step=40000.ckpt")
    open(ckpt, "a").close()
    metrics = V.metrics_path(ckpt, cell)

    split_block = dict({m: 1.0 for m in V.REQUIRED_SPLIT_METRICS}, **(split or {}))
    scene_block = dict({m: 1.0 for m in V.REQUIRED_SCENE_METRICS}, **(scene or {}))

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
