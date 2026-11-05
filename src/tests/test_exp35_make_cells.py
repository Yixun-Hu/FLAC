"""exp_35 round 2: the exp_35 rows of the evaluation cell manifest (plan_orbitrir_yaw_evidence.md
r2 §1 frozen scope, §9 round 2; the Planner's §14 A1 of 2026-11-05: the 16 E2-FAIL contingency rows).
``depth_canon_make_cells.py`` now renders 346 rows: the 230 exp_25 rows byte-for-byte as committed at the exp_25
campaign pin, then 116 exp_35 rows (orders 230-345) in the exact exp_25 26-field row schema, with the B-F arm's
frame-averaged protocol in its argv.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import collections
import hashlib
import importlib.util
import json
import os
import subprocess

import pytest

import eval_FLAC
from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")
EXP25_PIN = "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"
EXP25_CELLS_SHA256 = "1073ef647434318d219dba3c20e3a18b5a909c90aa3ee6e6bd86021c965f0575"   # plan §3 P2 guard
EXP25_BLOCKS = ("T", "T-dup", "C4+45", "R", "D-i", "D-ii", "S", "P1-pin", "YAW-pin", "TRAJ")
EXP35_BLOCKS = ("CTL", "BF-pin", "BF-dup", "OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45", "E2-FAIL")
EXP35_COUNTS = {"CTL": 2, "BF-pin": 28, "BF-dup": 2, "OFF-P1": 20, "OFF-BF": 20, "OFF-CANON": 20, "CANON-45": 8,
                "E2-FAIL": 16}
SEEDS = (42, 43, 44, 45, 46)
BF_FLAGS = ["--cond-method", "fa_invariant", "--frame-avg-angles", "0,90,180,270", "--frame-avg-max-fwd-samples", "64"]
NAS = "/media/diskstation/yixunhu/FLAC/checkpoints"
BF_ARM = {"model_config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
          "model_config_sha256": "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328",
          "trained_canon": False,
          "ckpt": {"kind": "endpoint", "path": f"{NAS}/ar_40k_endpoints/BF/epoch=8-step=40000.ckpt",
                   "sha256": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
                   "nas_manifest": f"{NAS}/ar_40k_endpoints/MANIFEST.sha256",
                   "nas_manifest_key": "BF/epoch=8-step=40000.ckpt"}}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = _load(os.path.join(REPO, EXPREL, "depth_canon_make_cells.py"), "depth_canon_make_cells_exp35")


@pytest.fixture(scope="module")
def full():
    return MC.build_manifest()


@pytest.fixture(scope="module")
def exp25():
    return MC.build_manifest(exp35=False)


def _rows(manifest, block):
    return [c for c in manifest["cells"] if c["block"] == block]


def _flags(row):
    args, out, i = row["eval_args"], {}, 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[args[i]], i = args[i + 1], i + 2
        else:
            out[args[i]], i = None, i + 1
    return out


# --------------------------------------------------------------------------- #
# the regression guard: the exp_25 rows and header are untouched
# --------------------------------------------------------------------------- #
def test_the_exp25_only_render_is_the_committed_exp25_manifest(exp25):
    text = MC.render(exp25)
    assert hashlib.sha256(text.encode()).hexdigest() == EXP25_CELLS_SHA256
    pinned = subprocess.run(["git", "-C", REPO, "show", f"{EXP25_PIN}:{EXPREL}/depth_canon_cells.json"],
                            capture_output=True, text=True, check=True).stdout
    assert text == pinned and len(exp25["cells"]) == 230


def test_the_full_manifest_appends_116_exp35_rows_to_the_exp25_rows(full, exp25):
    assert full["cells"][:230] == exp25["cells"]
    rows = full["cells"][230:]
    assert [c["order"] for c in rows] == list(range(230, 346))
    assert collections.Counter(c["block"] for c in rows) == EXP35_COUNTS and sum(EXP35_COUNTS.values()) == 116
    assert {c["block"]: c["priority"] for c in rows} == {b: 10 + i for i, b in enumerate(EXP35_BLOCKS)}
    assert full["blocks"] == list(EXP25_BLOCKS + EXP35_BLOCKS)
    assert full["block_counts"] == dict(exp25["block_counts"], **EXP35_COUNTS)
    assert full["arms"] == dict(exp25["arms"], BF=BF_ARM) and "BF" not in exp25["arms"]
    assert full["arm_cond_flags"] == {"BF": BF_FLAGS} and "arm_cond_flags" not in exp25
    assert "plan_orbitrir_yaw_evidence.md" in full["plan"] and exp25["plan"] == "plan_depth_canon.md v9 §5"
    same = [k for k in exp25 if k not in ("cells", "blocks", "block_counts", "arms", "plan")]
    assert {k: full[k] for k in same} == {k: exp25[k] for k in same}
    assert all(list(c) == list(exp25["cells"][0]) for c in rows)           # the exact 26-field row schema
    assert len(exp25["cells"][0]) == 26


def test_the_committed_file_is_the_full_render(full):
    with open(os.path.join(REPO, EXPREL, "depth_canon_cells.json")) as f:
        assert f.read() == MC.render(full)
    assert MC.main(["--check"]) == 0


# --------------------------------------------------------------------------- #
# the exp_35 rows: names, blocks, protocol
# --------------------------------------------------------------------------- #
def test_exp35_names_are_registered_and_injective(full, exp25):
    names = {b: {c["cell"] for c in _rows(full, b)} for b in EXP35_BLOCKS}
    assert names["CTL"] == {f"exp35_P1_T_S40000_K{k}_s42" for k in (1, 8)}               # plan §3c
    assert names["BF-pin"] == (
        {f"exp35_BF_T_S40000_K{k}_s{s}" for k in (1, 8) for s in SEEDS}                  # §4 E2 names s42
        | {f"exp35_BF_R_S40000_K{k}_s{s}" for k in (1, 8) for s in SEEDS}
        | {f"exp35_BF_C4_S40000_K{k}_s42_th{d}" for k in (1, 8) for d in (90, 180, 270, 45)})
    assert names["BF-dup"] == {f"exp35_BF_T_S40000_K{k}_s42_dup" for k in (1, 8)}
    for block, arm in (("OFF-P1", "P1"), ("OFF-BF", "BF"), ("OFF-CANON", "CANON")):
        assert names[block] == {f"exp35_{arm}_OFF_S40000_K{k}_s{s}_th{d}" for k in (1, 8) for s in SEEDS
                                for d in (135, 225)}
    assert names["CANON-45"] == {f"exp35_CANON_OFF_S40000_K{k}_s{s}_th45" for k in (1, 8) for s in SEEDS[1:]}
    assert names["E2-FAIL"] == {f"exp35_{arm}_OFF_S40000_K{k}_s{s}_th45" for arm in ("P1", "BF") for k in (1, 8)
                                for s in SEEDS[1:]}                                     # §14 A1
    every = [c["cell"] for c in full["cells"]]
    assert len(set(every)) == 346 and len({c["metrics_suffix"] for c in full["cells"]}) == 346
    assert all(not any(t in n for t in ("exp14_", "exp11_", "exp25_")) for b in EXP35_BLOCKS for n in names[b])
    old = {c["metrics_suffix"] for c in exp25["cells"]}
    assert not any(c["metrics_suffix"] in old for c in full["cells"][230:])


def _shape(c):
    return (c["arm"], c["subblock"], c["split"], c["rotate_mode"], c["rotate_deg"], c["rotate_seed"],
            c["canonicalize"], c["allow_canon_mismatch"], c["step"], c["dump"])


def test_exp35_rows_per_block(full):
    want = {"CTL": {("P1", "T", 0)}, "BF-dup": {("BF", "T", 0)}, "OFF-P1": {("P1", "OFF", 135), ("P1", "OFF", 225)},
            "OFF-BF": {("BF", "OFF", 135), ("BF", "OFF", 225)},
            "OFF-CANON": {("CANON", "OFF", 135), ("CANON", "OFF", 225)}, "CANON-45": {("CANON", "OFF", 45)},
            "E2-FAIL": {("P1", "OFF", 45), ("BF", "OFF", 45)},
            "BF-pin": {("BF", "T", 0), ("BF", "R", None), ("BF", "C4", 90), ("BF", "C4", 180), ("BF", "C4", 270),
                       ("BF", "C4", 45)}}
    for block in EXP35_BLOCKS:
        rows = _rows(full, block)
        assert {(c["arm"], c["subblock"], c["rotate_deg"]) for c in rows} == want[block], block
        for c in rows:
            canon = c["arm"] == "CANON"
            random = c["subblock"] == "R"
            assert _shape(c) == (c["arm"], c["subblock"], "unseen", "random" if random else "fixed", c["rotate_deg"],
                                 c["seed"] if random else None, canon, False, 40000, True), c["cell"]
            assert (c["canon_in_play"], c["canon_trained"], c["canon_mismatch"]) == (
                (True, True, False) if canon else (False, None, False)), c["cell"]
            assert c["expected_stream_count"] == 6337 and c["seed"] in SEEDS
    seeds = lambda b, **kw: collections.Counter((c["K"], c["seed"]) for c in _rows(full, b)  # noqa: E731
                                                if all(c[k] == v for k, v in kw.items()))
    assert seeds("CTL") == {(1, 42): 1, (8, 42): 1} and seeds("BF-dup") == {(1, 42): 1, (8, 42): 1}
    assert seeds("BF-pin", subblock="T") == {(k, s): 1 for k in (1, 8) for s in SEEDS}
    assert seeds("BF-pin", subblock="R") == {(k, s): 1 for k in (1, 8) for s in SEEDS}
    assert seeds("BF-pin", subblock="C4") == {(1, 42): 4, (8, 42): 4}
    for block in ("OFF-P1", "OFF-BF", "OFF-CANON"):
        assert seeds(block) == {(k, s): 2 for k in (1, 8) for s in SEEDS}
    assert seeds("CANON-45") == {(k, s): 1 for k in (1, 8) for s in SEEDS[1:]}
    assert seeds("E2-FAIL") == {(k, s): 2 for k in (1, 8) for s in SEEDS[1:]}


def test_bf_rows_carry_the_registered_frame_average_protocol(full):
    for c in [c for c in full["cells"] if c["arm"] == "BF"]:
        args = c["eval_args"]
        assert args[:4] == ["--model-config", BF_ARM["model_config"], "--dataset-config", c["dataset_config"]]
        assert args[4:10] == BF_FLAGS and args[10:10 + len(MC.COMMON_FLAGS) - 2] == MC.COMMON_FLAGS[2:]
        assert "--canonicalize" not in args and "--allow-canon-mismatch" not in args
        f = _flags(c)
        assert tuple(float(a) for a in f["--frame-avg-angles"].split(",")) == DEFAULT_FRAME_ANGLES
        assert f["--eval-name"] == c["cell"] and f["--seed"] == str(c["seed"])
        stem = eval_FLAC.build_output_paths(
            "farm/x.ckpt", 1, 1.0, c["cell"], cond_method="fa_invariant", rotate_deg=float(c["rotate_deg"] or 0),
            n_angles=4, rotate_mode=c["rotate_mode"], rotate_seed=c["rotate_seed"], are_lambda=None,
            canonicalize=False, canon_mismatch=False)["metrics"]
        assert c["metrics_suffix"] == os.path.basename(stem)[len("x_metrics_"):-len(".json")]
        assert f"{c['cell']}_fa_invariant_a4" in c["metrics_suffix"]


def test_vanilla_exp35_rows_carry_the_exp25_flags(full, exp25):
    for c in [c for c in full["cells"][230:] if c["arm"] != "BF"]:
        assert c["eval_args"][4:4 + len(MC.COMMON_FLAGS)] == MC.COMMON_FLAGS, c["cell"]
    for k in (1, 8):   # E1: the control cell IS the exp_25 P1-pin T s42 cell under another name
        ctl = next(c for c in _rows(full, "CTL") if c["K"] == k)
        ref = next(c for c in exp25["cells"] if c["cell"] == f"exp25_P1_T_S40000_K{k}_s42")
        swap = lambda a: [ctl["cell"] if x == ref["cell"] else x for x in a]  # noqa: E731
        assert ctl["eval_args"] == swap(ref["eval_args"])
        differ = ("cell", "order", "priority", "block", "metrics_suffix", "eval_args")
        assert {x: v for x, v in ctl.items() if x not in differ} == {x: v for x, v in ref.items() if x not in differ}
        assert ctl["metrics_suffix"] == ref["metrics_suffix"].replace("exp25_", "exp35_")


def test_the_bf_arm_is_the_endpoint_checkpoint(full):
    assert full["arms"]["BF"] == BF_ARM
    with open(os.path.join(REPO, BF_ARM["model_config"]), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == BF_ARM["model_config_sha256"]
    nas = BF_ARM["ckpt"]["nas_manifest"]
    if os.path.exists(nas):                                   # read-only: the NAS endpoint manifest lists it
        with open(nas) as f:
            assert f"{BF_ARM['ckpt']['sha256']}  {BF_ARM['ckpt']['nas_manifest_key']}" in f.read().splitlines()


def test_phase_i_is_the_32_cells_of_ctl_bf_pin_and_bf_dup(full):
    phase = [c for c in full["cells"] if c["block"] in ("CTL", "BF-pin", "BF-dup")]
    assert len(phase) == 32 and [c["order"] for c in phase] == list(range(230, 262))
    assert {c["arm"] for c in phase} == {"P1", "BF"} and sum(c["arm"] == "BF" for c in phase) == 30


def test_the_e2_fail_rows_are_the_matching_45_degree_rows_at_seeds_43_to_46(full):
    """§14 A1: {P1, BF} x 45 deg x K1/K8 x s43-46, orders 330-345 (arm, then K, then seed), priority 17; each row
    is its arm's s42 45-degree row (P1: exp_25's P1-pin C4 th45; B-F: BF-pin C4 th45) at another seed."""
    rows = _rows(full, "E2-FAIL")
    assert [(c["order"], c["arm"], c["K"], c["seed"]) for c in rows] == [
        (330 + i, arm, k, s) for i, (arm, k, s) in enumerate((a, k, s) for a in ("P1", "BF") for k in (1, 8)
                                                              for s in SEEDS[1:])]
    assert {c["priority"] for c in rows} == {17}
    for c in rows:
        ref = next(r for r in full["cells"] if r["cell"] == (
            f"exp25_P1_C4_S40000_K{c['K']}_s42_th45" if c["arm"] == "P1" else f"exp35_BF_C4_S40000_K{c['K']}_s42_th45"))
        swap = lambda a: [c["cell"] if x == ref["cell"] else str(c["seed"]) if (x, a[i - 1]) == ("42", "--seed")  # noqa
                          else x for i, x in enumerate(a)]
        assert c["eval_args"] == swap(ref["eval_args"]), c["cell"]
        same = ("block", "subblock", "cell", "order", "priority", "seed", "metrics_suffix", "eval_args")
        assert {x: v for x, v in c.items() if x not in same} == {x: v for x, v in ref.items() if x not in same}
        assert c["metrics_suffix"] == ref["metrics_suffix"].replace(ref["cell"], c["cell"]), c["cell"]
        assert (f"{c['cell']}_fa_invariant_a4" in c["metrics_suffix"]) == (c["arm"] == "BF")
