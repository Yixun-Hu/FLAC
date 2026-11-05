"""exp_35 round 3: the collector's per-arm protocol (``ARM_PROTOCOL``: record validation and
artifact paths for the B-F arm, plan_orbitrir_yaw_evidence.md r2 §3 P2, §9 round 3).

Synthetic cells are written with the REAL evaluator functions through test_depth_canon_collect's
fixtures (N = 8 items), now arm-aware: a B-F cell is written exactly as eval_FLAC writes a
``--cond-method fa_invariant`` evaluation.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import json
import os

import pytest

import eval_FLAC
import test_depth_canon_collect as T

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
VANILLA = {"cond_method": "vanilla", "frame_avg_angles": None, "orbit_execution": "n/a",
           "frame_avg_fwd_cap": None, "cond_autocast": "bf16"}
BF = {"cond_method": "fa_invariant", "frame_avg_angles": [0.0, 90.0, 180.0, 270.0], "orbit_execution": "batched",
      "frame_avg_fwd_cap": 64, "cond_autocast": "bf16"}


def xrow(block, sub, K, seed, deg):
    hits = [c for c in MANIFEST["cells"] if c["block"] == block and c["subblock"] == sub and c["K"] == K
            and c["seed"] == seed and c["rotate_deg"] == deg]
    assert len(hits) == 1, (block, sub, K, seed, deg, len(hits))
    return hits[0]


BF_CASES = [("BF-pin", "T", 1, 42, 0), ("BF-pin", "R", 8, 44, None), ("BF-pin", "C4", 1, 42, 90),
            ("BF-pin", "C4", 8, 42, 45), ("BF-dup", "T", 8, 42, 0), ("OFF-BF", "OFF", 1, 43, 135)]
VANILLA_CASES = [("CTL", "T", 1, 42, 0), ("OFF-P1", "OFF", 8, 46, 225), ("OFF-CANON", "OFF", 1, 42, 135),
                 ("CANON-45", "OFF", 8, 43, 45)]
EXP35_CASES = BF_CASES + VANILLA_CASES


def test_the_arm_protocol_table():
    assert C.arm_protocol("BF") == BF and C.ARM_PROTOCOL == {"BF": BF}
    assert all(C.arm_protocol(a) == VANILLA for a in ("CANON", "P1", "YAW"))
    flags = MANIFEST["arm_cond_flags"]["BF"]                 # the generator's argv states the same protocol
    got = dict(zip(flags[::2], flags[1::2]))
    assert got["--cond-method"] == BF["cond_method"] and int(got["--frame-avg-max-fwd-samples"]) == BF["frame_avg_fwd_cap"]
    assert [float(a) for a in got["--frame-avg-angles"].split(",")] == BF["frame_avg_angles"]
    assert [c["block"] for c in MANIFEST["cells"] if C.is_exp35(c)] == [c["block"] for c in MANIFEST["cells"][230:]]


@pytest.mark.parametrize("spec", EXP35_CASES, ids=[f"{c[0]}_{c[1]}_K{c[2]}_s{c[3]}_{c[4]}" for c in EXP35_CASES])
def test_a_true_exp35_record_passes(tmp_path, spec):
    r = xrow(*spec)
    rec = json.load(open(tmp_path / T.write_cell(tmp_path, r, receipt=False)))
    want = BF if r["arm"] == "BF" else VANILLA
    assert {k: rec[k] for k in want} == want
    assert C.record_reasons(rec, r, pin=PIN, link=rec["ckpt_path"]) == []


BF_EDITS = [(lambda d: d.update(cond_method="vanilla"), "cond_method"),
            (lambda d: d.update(frame_avg_angles=[0, 90, 180, 270]), "frame_avg_angles"),       # ints: type-strict
            (lambda d: d.update(frame_avg_angles=[0.0, 90.0, 180.0]), "frame_avg_angles"),
            (lambda d: d.update(frame_avg_angles=None), "frame_avg_angles"),
            (lambda d: d.update(orbit_execution="loop"), "orbit_execution"),
            (lambda d: d.update(frame_avg_fwd_cap=32), "frame_avg_fwd_cap"),
            (lambda d: d.update(frame_avg_fwd_cap=None), "frame_avg_fwd_cap"),
            (lambda d: d.update(cond_autocast="default"), "cond_autocast")]


@pytest.mark.parametrize("edit, needle", BF_EDITS, ids=[f"{n}{i}" for i, (_, n) in enumerate(BF_EDITS)])
def test_a_bf_record_off_its_protocol_is_refused(tmp_path, edit, needle):
    r = xrow("BF-pin", "R", 1, 45, None)
    rec = json.load(open(tmp_path / T.write_cell(tmp_path, r, receipt=False)))
    edit(rec)
    reasons = C.record_reasons(rec, r, pin=PIN, link=rec["ckpt_path"])
    assert reasons and any(needle in x for x in reasons), reasons


def test_a_vanilla_record_is_refused_for_a_bf_cell_and_vice_versa(tmp_path):
    bf, p1 = xrow("BF-pin", "T", 8, 43, 0), xrow("OFF-P1", "OFF", 1, 44, 135)
    rec = json.load(open(tmp_path / T.write_cell(tmp_path, bf, receipt=False)))
    rec.update(VANILLA)
    assert {x.split(":")[0] for x in C.record_reasons(rec, bf, pin=PIN, link=rec["ckpt_path"])} >= {
        "cond_method", "frame_avg_angles", "orbit_execution", "frame_avg_fwd_cap"}
    rec = json.load(open(tmp_path / T.write_cell(tmp_path, p1, receipt=False)))
    rec.update(BF)
    assert {x.split(":")[0] for x in C.record_reasons(rec, p1, pin=PIN, link=rec["ckpt_path"])} >= {
        "cond_method", "frame_avg_angles", "orbit_execution", "frame_avg_fwd_cap"}


def test_cell_paths_follow_the_arm_protocol():
    for r in MANIFEST["cells"]:
        base = "epoch=8-step=40000" if r["arm"] != "CANON" else f"epoch={r['step'] // 4551}-step={r['step']}"
        p = C.cell_paths(r, f"farm/{r['arm']}/{base}.ckpt", "/dumps")
        want = eval_FLAC.build_output_paths(
            f"farm/{r['arm']}/{base}.ckpt", 1, 1.0, r["cell"], cond_method=C.arm_protocol(r["arm"])["cond_method"],
            rotate_deg=float(r["rotate_deg"] or 0), n_angles=4, rotate_mode=r["rotate_mode"],
            rotate_seed=r["rotate_seed"], are_lambda=None, canonicalize=r["canonicalize"],
            canon_mismatch=r["canon_mismatch"])["metrics"]
        assert p.metrics == want and p.stem == f"{base}_metrics_{r['metrics_suffix']}", r["cell"]
        assert ("_fa_invariant_a4" in p.stem) is (r["arm"] == "BF")


def test_a_row_whose_name_and_block_name_different_campaigns_is_refused():
    import copy
    m = copy.deepcopy(MANIFEST)
    r = next(c for c in m["cells"] if c["cell"] == "exp35_P1_T_S40000_K1_s42")
    r["block"] = "P1-pin"                                     # an exp35_ name in an exp_25 block
    reasons = C.verify_cell(f"epoch=8-step=40000_metrics_{r['metrics_suffix']}", manifest=m, farm_root=T.FARM,
                            dump_root="/d", pin=PIN)
    assert len(reasons) == 1 and "different campaigns" in reasons[0], reasons


def test_plan_entries_of_exp35_cells():
    names = ["exp35_P1_T_S40000_K1_s42", "exp35_BF_T_S40000_K8_s42", "exp35_BF_R_S40000_K1_s46",
             "exp35_CANON_OFF_S40000_K8_s44_th225", "exp35_CANON_OFF_S40000_K1_s43_th45"]
    plan = C.plan_cells(MANIFEST, C.select_cells(MANIFEST, only=names), farm_root="farm", dump_root="/d",
                        canon_index=T.canon_index(), no_dump_blocks=[])
    by = {e["cell"]: e for e in plan}
    ctl, bf, canon = by[names[0]], by[names[1]], by[names[3]]
    assert (ctl["target"], ctl["sha256"]) == (MANIFEST["arms"]["P1"]["ckpt"]["path"], MANIFEST["arms"]["P1"]["ckpt"]["sha256"])
    assert (bf["target"], bf["sha256"], bf["nas_manifest_key"]) == (
        MANIFEST["arms"]["BF"]["ckpt"]["path"], MANIFEST["arms"]["BF"]["ckpt"]["sha256"], "BF/epoch=8-step=40000.ckpt")
    assert bf["link"] == "farm/BF/epoch=8-step=40000.ckpt" and bf["canon"] is None
    assert bf["argv"] == C.select_cells(MANIFEST, only=[names[1]])[0]["eval_args"] + [
        "--ckpt-path", bf["link"], "--dump-scored-waveforms", "/d"]
    assert (canon["target"], canon["sha256"]) == T.canon_index()[40000] and canon["canon"].endswith(".canon.json")
    assert all("_fa_invariant_a4" in by[n]["stem"] for n in names[1:3])
