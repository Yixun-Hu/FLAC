#!/usr/bin/env python3
"""exp_25 depth_canon -- the evaluation cell manifest ("cell classes as data",
plan_depth_canon.md v9 §5): every generative cell of the campaign as one row, in the
registered priority order T, T-dup, C4+45, R, D-i, D-ii, S, P1-pin, YAW-pin, TRAJ.

exp_35 (plan_orbitrir_yaw_evidence.md r2 §1, §9 round 2, §14 A1) appends 116 rows (orders 230-345) in
the same 26-field row schema: CTL, BF-pin, BF-dup, OFF-P1, OFF-BF, OFF-CANON, CANON-45 and the E2-FAIL
contingency block (run only when named), with the B-F arm's frame-averaged evaluation flags (``ARM_COND_FLAGS``). ``build_manifest(exp35=False)``
renders the exp_25 manifest byte for byte as committed at the exp_25 pin (the regression guard).

Deterministic, no I/O beyond reading nothing at all: ``build_manifest()`` is a pure
function of the constants below and eval_FLAC's own ``build_output_paths`` (which names
each cell's metrics stem, hence its dump directory). ``--write`` (re)writes
``depth_canon_cells.json``; ``--check`` exits 1 if the committed file has drifted.

A row carries the evaluator argv EXCEPT ``--ckpt-path`` (a local farm link, named after
the resolved checkpoint) and ``--dump-scored-waveforms <ROOT>`` (iff the cell dumps);
the driver ``depth_canon_eval_run.sh`` adds those two at run time. ``dump`` is true for
every row (A3 pending: the driver's ``--no-dump-blocks`` deselects a block's dumps).

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9 §5.
"""
import json
import os
import sys


def _repo_root(p):
    p = os.path.abspath(p)
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # this checkout's eval_FLAC, never a stale installed copy

EXPREL = "worklog/worklog_yixun/exp_25_depth_canon_claude"
OUT = os.path.join(REPO, EXPREL, "depth_canon_cells.json")
SCHEMA_VERSION = 1
EXP25_BLOCKS = ("T", "T-dup", "C4+45", "R", "D-i", "D-ii", "S", "P1-pin", "YAW-pin", "TRAJ")
EXP35_BLOCKS = ("CTL", "BF-pin", "BF-dup", "OFF-P1", "OFF-BF", "OFF-CANON", "CANON-45", "E2-FAIL")
BLOCKS = EXP25_BLOCKS + EXP35_BLOCKS
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
C4_ANGLES = (90, 180, 270)
OFF_ORBIT = 45
OFF_ANGLES = (135, 225)   # exp_35 phase (ii): the off-orbit figure angles
ENDPOINT, CADENCE = 40000, 2500
IMG_W = 512
NAS = "/media/diskstation/yixunhu/FLAC"
DUMP_ROOT = f"{NAS}/exp25_pred_waveforms"
FARM_ROOT = "outputs_FLAC/exp25_depth_canon"
COMMON_FLAGS = ["--cond-method", "vanilla", "--cond-autocast", "bf16", "--cfg-scale", "1.0",
                "--steps", "1", "--batch-size", "64", "--device", "cuda", "--record-per-scene",
                "--record-stream"]
# exp_35: B-F is evaluated frame-averaged (announcements 05/06: every flag written out); its rows
# replace the two leading ``--cond-method vanilla`` tokens of COMMON_FLAGS with these
ARM_COND_FLAGS = {"BF": ["--cond-method", "fa_invariant", "--frame-avg-angles", "0,90,180,270",
                         "--frame-avg-max-fwd-samples", "64"]}
_AR_EVAL = "src/configs/dataset_configs/AR/eval"
SPLITS = {
    "unseen": {"split_json": "data/AR/unseen_eval.json", "expected_stream_count": 6337,
               "configs": {1: (f"{_AR_EVAL}/acousticroom_unseeneval_1.json",
                               "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"),
                           8: (f"{_AR_EVAL}/acousticroom_unseeneval.json",
                               "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55")}},
    "seen": {"split_json": "data/AR/seen_eval.json", "expected_stream_count": 6217,
             "configs": {1: (f"{_AR_EVAL}/acousticroom_seeneval_1.json",
                             "15ab6a948c79861b8d8a8f598537949d88cbbf53d834cf8e18dd3747d6066a52"),
                         8: (f"{_AR_EVAL}/acousticroom_seeneval.json",
                             "fa4546060426079116e5e2b8a4ad4afe743e014decad3ad7af5b44afd0e99838")}},
}
_ENDPOINTS = f"{NAS}/checkpoints/ar_40k_endpoints"
ARMS = {  # model configs == each checkpoint's EMBEDDED model_config (verified 2026-10-23)
    "CANON": {"model_config": f"{EXPREL}/FLAC_AR_CANON.json",
              "model_config_sha256": "aba94032b1e33d05d8cf5d2d188f320f70e2915bde18b8f0be7813b6b9336626",
              "trained_canon": True,
              "ckpt": {"kind": "canon_run",
                       "run_ns": f"{NAS}/checkpoints/exp25_depth_canon/FLAC_exp25_CANON/exp25_CANON"}},
    "P1": {"model_config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
           "model_config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
           "trained_canon": False,
           "ckpt": {"kind": "endpoint", "path": f"{_ENDPOINTS}/P1/epoch=8-step=40000.ckpt",
                    "sha256": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
                    "nas_manifest": f"{_ENDPOINTS}/MANIFEST.sha256",
                    "nas_manifest_key": "P1/epoch=8-step=40000.ckpt"}},
    "YAW": {"model_config": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
            "model_config_sha256": "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004",
            "trained_canon": False,
            "ckpt": {"kind": "endpoint", "path": f"{_ENDPOINTS}/YAW/epoch=8-step=40000.ckpt",
                     "sha256": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
                     "nas_manifest": f"{_ENDPOINTS}/MANIFEST.sha256",
                     "nas_manifest_key": "YAW/epoch=8-step=40000.ckpt"}},
    "BF": {"model_config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",   # exp_35
           "model_config_sha256": "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328",
           "trained_canon": False,
           "ckpt": {"kind": "endpoint", "path": f"{_ENDPOINTS}/BF/epoch=8-step=40000.ckpt",
                    "sha256": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328",
                    "nas_manifest": f"{_ENDPOINTS}/MANIFEST.sha256",
                    "nas_manifest_key": "BF/epoch=8-step=40000.ckpt"}},
}
EXP25_ARMS = ("CANON", "P1", "YAW")


def _row(block, sub, arm, step, split, k, seed, rot, canonicalize=False, allow=False, dup=False):
    """One cell. ``rot`` = ``("fixed", deg)`` or ``("random", None)`` (rotate seed = eval seed)."""
    import eval_FLAC
    mode, deg = rot
    tag = {"T-dup": "T", "C4+45": "C4", "D-i": "Di", "D-ii": "Dii", "CTL": "T", "BF-dup": "T"}.get(block, block)
    if block in ("P1-pin", "YAW-pin", "BF-pin", "E2-FAIL") or block.startswith(("OFF-", "CANON-")):
        tag = sub
    name = f"{'exp35' if block in EXP35_BLOCKS else 'exp25'}_{arm}_{tag}_S{step}_K{k}_s{seed}"
    if tag in ("C4", "TRAJ", "OFF"):
        name += f"_th{deg}"
    name += "_dup" if dup else ""
    trained = ARMS[arm]["trained_canon"]
    in_play = trained or canonicalize or allow
    mismatch = bool(in_play and trained != canonicalize)
    config, config_sha = SPLITS[split]["configs"][k]
    n = SPLITS[split]["expected_stream_count"]
    rotate_seed = seed if mode == "random" else None
    cond = ARM_COND_FLAGS[arm] + COMMON_FLAGS[2:] if arm in ARM_COND_FLAGS else COMMON_FLAGS
    args = ["--model-config", ARMS[arm]["model_config"], "--dataset-config", config, *cond,
            "--expected-stream-count", str(n), "--seed", str(seed), "--rotate-mode", mode]
    args += ["--rotate-seed", str(seed)] if mode == "random" else ["--rotate-deg", str(deg)]
    args += (["--canonicalize"] if canonicalize else []) + (["--allow-canon-mismatch"] if allow else [])
    args += ["--eval-name", name]
    metrics = eval_FLAC.build_output_paths(
        "farm/x.ckpt", 1, 1.0, name, cond_method=cond[1], rotate_deg=float(deg or 0), n_angles=4,
        rotate_mode=mode, rotate_seed=rotate_seed, are_lambda=None, canonicalize=canonicalize,
        canon_mismatch=mismatch)["metrics"]
    suffix = os.path.basename(metrics)[len("x_metrics_"):-len(".json")]
    return {"cell": name, "block": block, "subblock": sub, "arm": arm, "step": step, "split": split,
            "K": k, "seed": seed, "rotate_mode": mode, "rotate_deg": deg, "rotate_seed": rotate_seed,
            "canonicalize": canonicalize, "allow_canon_mismatch": allow, "canon_in_play": in_play,
            "canon_trained": trained if in_play else None, "canon_mismatch": mismatch,
            "model_config": ARMS[arm]["model_config"],
            "model_config_sha256": ARMS[arm]["model_config_sha256"], "dataset_config": config,
            "dataset_config_sha256": config_sha, "expected_stream_count": n, "dump": True,
            "metrics_suffix": suffix, "eval_args": args}


def _pin_rows(arm):
    """P1-pin / YAW-pin: T (theta 0) + R (random) + C4/45 (seed 42) [+ S for YAW]."""
    block = f"{arm}-pin"
    rows = [_row(block, "T", arm, ENDPOINT, "unseen", k, s, ("fixed", 0)) for k in KS for s in SEEDS]
    rows += [_row(block, "R", arm, ENDPOINT, "unseen", k, s, ("random", None)) for k in KS for s in SEEDS]
    rows += [_row(block, "C4", arm, ENDPOINT, "unseen", k, 42, ("fixed", d))
             for k in KS for d in C4_ANGLES + (OFF_ORBIT,)]
    if arm == "YAW":
        rows += [_row(block, "S", arm, ENDPOINT, "seen", k, s, ("fixed", 0)) for k in KS for s in SEEDS]
    return rows


def _exp35_blocks():
    """exp_35 (plan_orbitrir_yaw_evidence.md r2 §1): phase (i) CTL / BF-pin / BF-dup, phase (ii) the
    off-orbit figure cells; every row seed 42-46 / K1, K8 unless the plan narrows it."""
    E, U = ENDPOINT, "unseen"
    off = lambda block, arm: [_row(block, "OFF", arm, E, U, k, s, ("fixed", d), arm == "CANON")  # noqa: E731
                              for k in KS for s in SEEDS for d in OFF_ANGLES]
    return {
        "CTL": [_row("CTL", "T", "P1", E, U, k, 42, ("fixed", 0)) for k in KS],
        "BF-pin": [_row("BF-pin", "T", "BF", E, U, k, s, ("fixed", 0)) for k in KS for s in SEEDS]
        + [_row("BF-pin", "R", "BF", E, U, k, s, ("random", None)) for k in KS for s in SEEDS]
        + [_row("BF-pin", "C4", "BF", E, U, k, 42, ("fixed", d)) for k in KS for d in C4_ANGLES + (OFF_ORBIT,)],
        "BF-dup": [_row("BF-dup", "T", "BF", E, U, k, 42, ("fixed", 0), dup=True) for k in KS],
        "OFF-P1": off("OFF-P1", "P1"), "OFF-BF": off("OFF-BF", "BF"), "OFF-CANON": off("OFF-CANON", "CANON"),
        "CANON-45": [_row("CANON-45", "OFF", "CANON", E, U, k, s, ("fixed", OFF_ORBIT), True)
                     for k in KS for s in SEEDS[1:]],
        # §14 A1: the contingency of an E2 FAIL (§4) -- P1 and B-F at 45 deg, s43-46; run only when named
        "E2-FAIL": [_row("E2-FAIL", "OFF", arm, E, U, k, s, ("fixed", OFF_ORBIT))
                    for arm in ("P1", "BF") for k in KS for s in SEEDS[1:]],
    }


def build_manifest(exp35=True):
    """The 346-row manifest; ``exp35=False``: the exp_25 manifest exactly as committed at its pin."""
    E, U = ENDPOINT, "unseen"
    blocks = {
        "T": [_row("T", "T", "CANON", E, U, k, s, ("fixed", 0), True) for k in KS for s in SEEDS],
        "T-dup": [_row("T-dup", "T", "CANON", E, U, k, 42, ("fixed", 0), True, dup=True) for k in KS],
        "C4+45": [_row("C4+45", "C4", "CANON", E, U, k, 42, ("fixed", d), True)
                  for k in KS for d in C4_ANGLES + (OFF_ORBIT,)],
        "R": [_row("R", "R", "CANON", E, U, k, s, ("random", None), True) for k in KS for s in SEEDS],
        "D-i": [_row("D-i", "D", "CANON", E, U, k, 42, ("fixed", 0), allow=True) for k in KS],
        "D-ii": [_row("D-ii", "D", "P1", E, U, k, 42, ("fixed", 0), True, True) for k in KS],
        "S": [_row("S", "S", "CANON", E, "seen", k, s, ("fixed", 0), True) for k in KS for s in SEEDS],
        "P1-pin": _pin_rows("P1"),
        "YAW-pin": _pin_rows("YAW"),
        "TRAJ": [_row("TRAJ", "TRAJ", "CANON", step, U, k, 42, ("fixed", d), True)
                 for step in range(CADENCE, ENDPOINT, CADENCE) for k in KS for d in (0,) + C4_ANGLES],
    }
    names = EXP25_BLOCKS + (EXP35_BLOCKS if exp35 else ())
    blocks.update(_exp35_blocks() if exp35 else {})
    cells = []
    for priority, block in enumerate(names):
        for row in blocks[block]:
            cells.append({"cell": row.pop("cell"), "order": len(cells), "priority": priority, **row})
    head = {"schema_version": SCHEMA_VERSION, "generator": f"{EXPREL}/depth_canon_make_cells.py",
            "plan": "plan_depth_canon.md v9 §5" + (
                "; exp_35 rows: plan_orbitrir_yaw_evidence.md r2 §1, §11, §14 A1" if exp35 else ""),
            "blocks": list(names), "block_counts": {b: len(blocks[b]) for b in names}, "common_flags": COMMON_FLAGS}
    head.update({"arm_cond_flags": ARM_COND_FLAGS} if exp35 else {})
    return {**head, "img_w": IMG_W, "seeds": list(SEEDS), "dump_root": DUMP_ROOT, "farm_root": FARM_ROOT,
            "splits": {s: {"split_json": v["split_json"], "expected_stream_count": v["expected_stream_count"]}
                       for s, v in SPLITS.items()},
            "arms": {a: ARMS[a] for a in EXP25_ARMS + (("BF",) if exp35 else ())}, "cells": cells}


def render(manifest):
    """Deterministic bytes: a pretty header, then one compact cell per line."""
    head = {k: v for k, v in manifest.items() if k != "cells"}
    return (json.dumps(head, indent=1)[:-2] + ',\n "cells": [\n'
            + ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in manifest["cells"])
            + "\n ]\n}\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help=f"write {os.path.relpath(OUT, REPO)}")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file drifted")
    args = ap.parse_args(argv)
    text = render(build_manifest())
    if args.check:
        with open(OUT) as f:
            same = f.read() == text
        print("cells manifest OK" if same else f"cells manifest DRIFTED from the generator: {OUT}")
        return 0 if same else 1
    if args.write:
        with open(OUT, "w") as f:
            f.write(text)
        print(f"wrote {OUT}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
