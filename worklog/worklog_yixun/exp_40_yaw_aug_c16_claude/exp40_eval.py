#!/usr/bin/env python3
"""exp_40 yaw_aug_c16 -- the evaluation driver (plan_yaw_aug_c16.md §5 as amended by the BINDING
§14; Codex code review 1 B7-B10, H4 (eval half), H7-H12, M2, M5, M7 and code review 2 B4, M2,
M3 (eval half) fixed here).

Thin by design: it reuses the exp_25 YAW-pin protocol (each argv derived from the committed
exp_25 manifest row of the same kind), ``eval_FLAC.py``, the committed no-flash launcher,
a content-addressed checkpoint cache and the exp_36 replay semantics, without touching any
exp_25/35/36 file. Subcommands:

  plan [--write|--check]       the deterministic 85-cell list -> exp40_cells.json
  build-e0-reference           ONE-TIME: pin the three exp_25 E0 reference cells and the 20
                               expected-hash stream files by sha256 -> exp40_e0_reference.json
  register-comparators         ONE-TIME (§14 H8 step 1): validate the 92 reused exp_25 / exp_36
                               cells with their source experiment's OWN validator at its pin and
                               pin every byte -> exp40_comparators.json
  run --lanes 0[,1]            from the detached eval worktree at EVAL_SHA (§14 H7). Gates: the
      [--blocks|--only]        worktree, the runtime-input manifest (AR re-hash receipt, the
                               C16 admission), the AR quick gates (IR counts + 16-item eval
                               sample), the environment probe (interpreter, versions, module
                               origins, no-flash preflight, sealed-HF resolution, driver pin),
                               the exp_40 card lock, per-card GPU_LOCK_<uuid> + occupancy re-check
                               right before each spawn, the runtime identity (AR receipt 0..48 h
                               old at THAT cell's start) right before each spawn. E0 first (writes
                               the E0 receipt, derived from its three cell receipts); any other
                               cell needs an EQUAL E0 receipt for EVAL_SHA + closure + environment
  run --dry-run                every gate of run (exit 2 on any refusal), then the argv; only the
                               locks and the spawns are skipped
  verify                       re-verify every selected cell's receipt and artifacts with the
                               SAME strict validators as run / collect (non-zero if any is
                               missing or invalid)
  collect                      validated CSV + all tables + off-grid readout + paper-import CSV
                               + collect manifest
  figure                       the exp_40 figure, built from the frozen, re-validated CSV
  freeze-table-evidence        the committed manifest the offline row validator checks
  validate-row                 offline validator used by gen_model_comparison.py (contract exp40)

Exit: 0 PASS / all done; 1 a cell or verdict FAILED (or the run was interrupted); 2 a gate REFUSED.
Written by the exp_40 Coder seat (Claude Opus 5.5) per plan_yaw_aug_c16.md §5 / §14.
"""
import sys

sys.dont_write_bytecode = True     # B7: before ANY import from an evaluation tree (and the kit)

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_kit():
    spec = importlib.util.spec_from_file_location("exp40_kit", os.path.join(HERE, "exp40_kit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kit = _load_kit()
Refused = kit.Refused
ROOT = kit.ROOT
EXPREL = kit.EXPREL
EXP25_CELLS = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cells.json"
EXP25_CELLS_SHA256 = "1073ef647434318d219dba3c20e3a18b5a909c90aa3ee6e6bd86021c965f0575"
P1_CONFIG = "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARMS = {
    "C16": {"config": kit.ARM_CONFIG, "config_sha256": kit.PINS[kit.ARM_CONFIG]},
    "YAW": {"config": kit.BASE_CONFIG, "config_sha256": kit.PINS[kit.BASE_CONFIG],
            "ckpt": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/YAW/epoch=8-step=40000.ckpt",
            "ckpt_sha256": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7",
            "ckpt_bytes": 723922667},
    "P1": {"config": P1_CONFIG,
           "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
           "ckpt": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/P1/epoch=8-step=40000.ckpt",
           "ckpt_sha256": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c",
           "ckpt_bytes": 723927531},
}
ADMISSION = f"{EXPREL}/exp40_C16_admission.json"       # where the admission is committed at results time
CELLS_FILE = f"{EXPREL}/exp40_cells.json"
COMPARATORS_FILE = f"{EXPREL}/exp40_comparators.json"
FIGURE_CONTRACT = f"{EXPREL}/exp40_figure_contract.json"
E0_REFERENCE = f"{EXPREL}/exp40_e0_reference.json"
E0_REFERENCE_SHA256 = "180a4534df2d8ea2f7cf84ee9a73b6e9a715fffd667e79c277dcf19072280c49"   # built 2026-11-07 (build-e0-reference)
TABLE_EVIDENCE = f"{EXPREL}/exp40_table_evidence.json"
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
C4 = (90, 180, 270)
G45 = (45, 135, 225, 315)
OFF = (11.25, 101.25, 191.25, 281.25)
FARM = "outputs_FLAC/exp40_yaw_aug_c16/farm"
DUMP_ROOT = "/media/diskstation/yixunhu/FLAC/exp40_pred_waveforms"
EVAL_PATHS = ("eval_FLAC.py", "src", "data/AR", "overlays", EXPREL, EXP25_CELLS, kit.BASE_CONFIG, P1_CONFIG)
EVAL_LINK_NAMES = ("AcousticRooms", "weights/FLAC/VAE.ckpt", "weights/AGREE/AGREE_fullAR.pt")
NO_FLASH_OVERLAY = "overlays/no_flash/flash_attn.py"
NO_FLASH_LAUNCHER = "overlays/no_flash/exp35_no_flash_launch.py"
EVAL_PINS = {
    "weights/FLAC/VAE.ckpt": "6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf",
    "weights/AGREE/AGREE_fullAR.pt": "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787",
    "data/AR/unseen_eval.json": "9a9d817abc3e19f41351e07325ffa929c1f2846d0c77e80d538d9e4a21342ba8",
    "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json":
        "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3",
    "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json":
        "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55",
    NO_FLASH_OVERLAY: "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023",
    NO_FLASH_LAUNCHER: "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de",
}
DATASET_OF_K = {1: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
                8: "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"}
METRIC_KEYS = ("T60", "Invalid T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
               "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
SCENES = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
          "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants")
# the exact exp_25 metrics-record schema (read from the archived YAW cells 2026-11-07)
METRICS_BASE_KEYS = frozenset((
    "batch_size", "by_scene", "cfg_scale", "ckpt_bytes_loaded", "ckpt_path", "ckpt_sha256_loaded",
    "cond_autocast", "cond_method", "dataset_config", "device", "eval_name", "frame_avg_angles",
    "frame_avg_fwd_cap", "metrics", "n_samples", "orbit_execution", "per_scene_schema", "rotate_deg",
    "scene_count", "seed", "source_sha", "steps", "weights_source"))
METRICS_RANDOM_KEYS = frozenset(("assignment_hash", "img_w", "input_hash", "rotate_mode", "rotate_seed",
                                 "stream_count"))
STREAM_KEYS = frozenset(("schema_version", "fingerprint_schema", "rotate_mode", "rotate_seed", "rotate_deg",
                         "img_w", "stream_count", "input_tuples", "offsets", "assignment_tuples",
                         "input_hash", "assignment_hash"))
DUMP_KEYS = frozenset(("schema_version", "n", "dtype", "shape", "sha256", "input_hash", "metrics_stem",
                       "source_sha", "ckpt_path", "metrics_path", "metrics_sha256", "stream_sidecar_path",
                       "stream_sidecar_sha256", "canon_sidecar_path", "canon_sidecar_sha256"))
IMG_W, N_ITEMS, WAVE_T = 512, 6337, 10240
# E0 references: the archived exp_25 YAW cells (paths + sha256 pinned in E0_REFERENCE)
EXP25_FARM = "/home/yixunhu/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/YAW"
EXP25_DUMPS = "/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms"
E0 = {
    "exp40_E0_YAW_T_S40000_K1_s42": "exp25_YAW_T_S40000_K1_s42",
    "exp40_E0_YAW_C4_S40000_K8_s42_th90": "exp25_YAW_C4_S40000_K8_s42_th90",
    "exp40_E0_YAW_R_S40000_K8_s42": "exp25_YAW_R_S40000_K8_s42",
}
ORIENT = {"T60": -1, "C50": -1, "EDT": -1, "FD": -1, "RIR_to_GT_RIR_R@1": 1,
          "RIR_to_GT_RIR_R@5": 1, "RIR_to_GT_RIR_R@10": 1}
MIN_GPU_FREE_MIB = 10240
GPU_NAME = kit.PLATFORMS["a6000"]["gpu_name"]
SENTINEL = "EXP35_NO_FLASH_LAUNCHER"
# comparator sources (§5.2): each cell is validated by its source experiment's own validator at
# its own pin (git archive of that commit, run -B in a temporary tree), data read in place
SOURCES = {
    "exp25": {"pin": "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e",
              "base_dir": "/home/yixunhu/codespace/exp-25-depth-canon",
              "validator_blob": "631fe678c33c99fdb63a9c8343a02a18b78fcd73"},
    "exp36": {"pin": "6be49f46ec31e899f1bee01f72c6a5827e701c6c",
              "base_dir": "/home/yixunhu/codespace/exp-36-run",
              "validator_blob": "33ef6ecddaf14f3408fcd6341196010cfa89c86b"},
}
COMP_FARM_ROOT = "outputs_FLAC/exp25_depth_canon"
COMP_DUMP_ROOT = EXP25_DUMPS
VALIDATOR = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py"
VALIDATOR_ARCHIVE = ("eval_FLAC.py", "src", "AGREE", "overlays", "data/AR",
                     "worklog/worklog_yixun/exp_25_depth_canon_claude",
                     "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
                     "worklog/worklog_yixun/exp_07_fa_scratch_claude")
_EXP36_DIR = "/home/yixunhu/codespace/exp-36-run/worklog/worklog_yixun/exp_36_yaw_c4_spread_claude"
EXP36_VERDICTS = {     # the replay hard stops that admitted the exp_36 pin, per chain and arm
    ("offorbit", "YAW"): (f"{_EXP36_DIR}/yaw_c4_spread_2026-11-06_19-03-38_offorbit_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json",
                          "fce4adeb38daa1238c26f3b636342dcc70e9ea0f3f638c4a484f9000de193574"),
    ("offorbit", "P1"): (f"{_EXP36_DIR}/yaw_c4_spread_2026-11-06_19-03-38_offorbit_verdict_exp36_P1_C4_S40000_K8_s42_th90.json",
                         "06cbe045cd199344e0acdae1638779270da7bee5bff5a6fb0d5af62aa71e4245"),
    ("A", "YAW"): (f"{_EXP36_DIR}/yaw_c4_spread_2026-11-07_01-44-20_A_verdict_exp36_YAW_C4_S40000_K8_s42_th90.json",
                   "306c645dbc19b14c0aa8747d3f7d2cb0d765f8c647ab40e9904210612c645e9e"),
}
RUNTIME_INPUT_NAMES = ("ar_verify", "admission")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def rot_token(deg):
    d = float(deg)
    return str(int(d)) if d.is_integer() else repr(d).replace(".", "p")


def _hex64(x):
    return isinstance(x, str) and _HEX64.fullmatch(x) is not None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _int(v):
    return type(v) is int


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _read(path):
    """``(bytes, sha256, parsed JSON)`` from ONE read of ``path``."""
    with open(path, "rb") as f:
        b = f.read()
    return b, _sha(b), json.loads(b)


def _strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


# --------------------------------------------------------------------------- #
# the cell list (plan §5.2)
# --------------------------------------------------------------------------- #
def _templates(cells_json):
    rows = {c["cell"]: c for c in cells_json["cells"] if c["block"] == "YAW-pin"}
    out = {}
    for k in KS:
        out[("T", k)] = rows[f"exp25_YAW_T_S40000_K{k}_s42"]
        out[("C4", k)] = rows[f"exp25_YAW_C4_S40000_K{k}_s42_th90"]
        out[("R", k)] = rows[f"exp25_YAW_R_S40000_K{k}_s42"]
    return out


def _argv_from(template, arm, seed, mode, deg, name):
    """Copy the exp_25 row's argv, substituting ONLY model config, seed, rotation, name."""
    a = list(template["eval_args"])
    out, i = [], 0
    while i < len(a):
        flag = a[i]
        if flag in ("--model-config", "--seed", "--rotate-mode", "--rotate-deg", "--rotate-seed", "--eval-name"):
            i += 2
            continue
        out.append(flag)
        i += 1
    rot = ["--rotate-mode", mode] + (["--rotate-seed", str(seed)] if mode == "random"
                                     else ["--rotate-deg", str(deg)])
    return ["--model-config", ARMS[arm]["config"]] + out[:out.index("--expected-stream-count") + 2] \
        + ["--seed", str(seed)] + rot + out[out.index("--expected-stream-count") + 2:] + ["--eval-name", name]


def _row(templates, block, arm, k, seed, mode, deg, name, replay_of=None):
    kind = "R" if mode == "random" else ("T" if float(deg) == 0 else "C4")
    t = templates[(kind, k)]
    return {"cell": name, "block": block, "arm": arm, "K": k, "seed": seed, "rotate_mode": mode,
            "rotate_deg": None if mode == "random" else float(deg), "rotate_seed": seed if mode == "random" else None,
            "dataset_config": t["dataset_config"], "dataset_config_sha256": t["dataset_config_sha256"],
            "model_config": ARMS[arm]["config"], "model_config_sha256": ARMS[arm]["config_sha256"],
            "expected_stream_count": 6337, "template": t["cell"], "replay_of": replay_of,
            "eval_args": _argv_from(t, arm, seed, mode, deg, name)}


def build_cells(cells_json):
    T = _templates(cells_json)
    rows = []
    for name, ref in E0.items():
        k = int(re.search(r"_K(\d)_", name).group(1))
        mode = "random" if "_R_" in name else "fixed"
        deg = 90 if "_th90" in name else 0
        rows.append(_row(T, "E0", "YAW", k, 42, mode, deg, name, replay_of=ref))
    for k in KS:
        for s in SEEDS:
            rows.append(_row(T, "T", "C16", k, s, "fixed", 0, f"exp40_C16_T_S40000_K{k}_s{s}"))
    for k in KS:
        for s in SEEDS:
            for d in C4:
                rows.append(_row(T, "C4", "C16", k, s, "fixed", d, f"exp40_C16_C4_S40000_K{k}_s{s}_th{d}"))
    for k in KS:
        for d in G45:
            rows.append(_row(T, "G45", "C16", k, 42, "fixed", d, f"exp40_C16_G45_S40000_K{k}_s42_th{d}"))
    for k in KS:
        for d in OFF:
            rows.append(_row(T, "OFF", "C16", k, 42, "fixed", d,
                             f"exp40_C16_OFF_S40000_K{k}_s42_th{rot_token(d)}"))
    for k in KS:
        for s in SEEDS:
            rows.append(_row(T, "R", "C16", k, s, "random", 0, f"exp40_C16_R_S40000_K{k}_s{s}"))
    for arm in ("YAW", "P1"):
        for k in KS:
            for d in OFF:
                rows.append(_row(T, "FILL", arm, k, 42, "fixed", d,
                                 f"exp40_{arm}_OFF_S40000_K{k}_s42_th{rot_token(d)}"))
    names = [r["cell"] for r in rows]
    if len(set(names)) != len(names):
        raise Refused("duplicate cell names")
    return {"schema_version": 1, "plan": "plan_yaw_aug_c16.md r2 §5.2", "count": len(rows),
            "block_counts": {b: sum(r["block"] == b for r in rows)
                             for b in ("E0", "T", "C4", "G45", "OFF", "R", "FILL")},
            "cells": rows}


def render(manifest):
    return json.dumps(manifest, indent=1, sort_keys=True) + "\n"


def load_cells(tree):
    return kit.read_json(os.path.join(tree, CELLS_FILE))["cells"]


# --------------------------------------------------------------------------- #
# the ONE evaluation tree whose evaluator helpers this process uses (B7)
# --------------------------------------------------------------------------- #
_EVALUATOR = {}
_EVAL_LOCK = threading.Lock()
_DRAWS = {}


def evaluator(tree):
    """``(eval_FLAC, src.data.yaw_rotation)`` of ONE evaluation tree per process. Bytecode is
    disabled before the import (an import may never dirty the detached eval worktree), and a
    second, different tree is refused rather than silently served the first one's modules."""
    real = os.path.realpath(tree)
    with _EVAL_LOCK:
        if _EVALUATOR:
            if _EVALUATOR["tree"] != real:
                raise Refused(f"this process already bound the evaluator of {_EVALUATOR['tree']}; "
                              f"refusing a second evaluation tree {real}")
            return _EVALUATOR["eval"], _EVALUATOR["yaw"]
        sys.dont_write_bytecode = True
        if real not in [os.path.realpath(p) for p in sys.path if p]:
            sys.path.insert(0, real)
        import eval_FLAC
        from src.data import yaw_rotation
        for mod, rel in ((eval_FLAC, "eval_FLAC.py"), (yaw_rotation, "src/data/yaw_rotation.py")):
            if os.path.realpath(mod.__file__) != os.path.join(real, rel):
                raise Refused(f"{mod.__name__} resolves to {mod.__file__}, not {real}/{rel}")
        _EVALUATOR.update(tree=real, eval=eval_FLAC, yaw=yaw_rotation)
        return eval_FLAC, yaw_rotation


def stem_of(cell, tree):
    ev, _ = evaluator(tree)
    m = ev.build_output_paths(
        "farm/x.ckpt", 1, 1.0, cell["cell"], cond_method="vanilla",
        rotate_deg=float(cell["rotate_deg"] or 0), n_angles=4, rotate_mode=cell["rotate_mode"],
        rotate_seed=cell["rotate_seed"], are_lambda=None, canonicalize=False, canon_mismatch=False)["metrics"]
    return "epoch=8-step=40000" + os.path.basename(m)[len("x"):-len(".json")]


def expected_offsets(cell, tree):
    """The per-position offsets the cell MUST record: fixed = the angle's column shift (the
    evaluator's own rule); random = the evaluator's own draw from the eval seed."""
    ev, yaw = evaluator(tree)
    if cell["rotate_mode"] == "random":
        seed = int(cell["rotate_seed"])
        with _EVAL_LOCK:
            if seed not in _DRAWS:
                import torch
                g = torch.Generator(device="cpu")
                g.manual_seed(seed)
                _DRAWS[seed] = yaw.draw_yaw_offsets(N_ITEMS, IMG_W, g).tolist()
            return _DRAWS[seed]
    return [yaw.yaw_column_shift(math.radians(float(cell["rotate_deg"])), IMG_W)] * N_ITEMS


# --------------------------------------------------------------------------- #
# evaluation tree identity, closure digest, environment, AR gates (plan §5.1, §14 H2/H3/H7)
# --------------------------------------------------------------------------- #
EVAL_UNTRACKED = {"AcousticRooms": "/media/diskstation/yixunhu/FLAC/AcousticRooms",
                  "weights/AGREE": None, "weights/FLAC/VAE.ckpt": None}     # allowlist (plan §14 N2)


def closure_digest(repo, eval_sha):
    """EVAL_SHA's evaluator closure (``git ls-tree`` of EVAL_PATHS) + the pinned assets: E0 is
    bound to it, so a later record-only commit elsewhere cannot stale E0 (plan §14 H8)."""
    blobs = kit.tracked_blobs(repo, eval_sha, EVAL_PATHS)
    return kit.digest_obj({"eval_sha": eval_sha, "blobs": blobs, "pins": EVAL_PINS})


def tree_identity_reasons(wt, eval_sha):
    """HEAD == EVAL_SHA, detached, the tracked tree clean, only the allowlisted untracked entries."""
    reasons = []
    head = kit.git(wt, "rev-parse", "HEAD").strip()
    if head != eval_sha:
        reasons.append(f"eval worktree HEAD {head} != EVAL_SHA {eval_sha}")
    if kit.git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() != "HEAD":
        reasons.append("the eval worktree must be detached")
    for line in kit.git(wt, "status", "--porcelain", "--untracked-files=all").splitlines():
        code, path = line[:2], line[3:].rstrip("/")
        if code != "??" or not any(path == u or path.startswith(u + "/") for u in EVAL_UNTRACKED):
            reasons.append(f"eval worktree not clean: {line!r}")
    return reasons


def eval_worktree_reasons(wt, eval_sha, hf_sealed):
    """The detached eval worktree at EVAL_SHA, checked before EVERY spawn (plan §14 H7)."""
    reasons = tree_identity_reasons(wt, eval_sha)
    for rel, target in EVAL_UNTRACKED.items():
        p = os.path.join(wt, rel)
        if not os.path.lexists(p):
            reasons.append(f"allowlisted asset {rel} missing")
        elif target is not None and os.path.realpath(p) != os.path.realpath(target):
            reasons.append(f"{rel} resolves to {os.path.realpath(p)}, not {target}")
    for rel, want in EVAL_PINS.items():
        p = os.path.join(wt, rel)
        if not os.path.isfile(p) or kit.sha256_file(p) != want:
            reasons.append(f"{rel} is not the pinned {want}")
    for arm, a in ARMS.items():
        p = os.path.join(wt, a["config"])
        if not os.path.isfile(p) or kit.sha256_file(p) != a["config_sha256"]:
            reasons.append(f"{a['config']} is not the pinned {a['config_sha256']}")
    reasons += kit.hf_sealed_reasons(hf_sealed)
    man_path = os.path.join(wt, kit.AR_MANIFEST)
    if not os.path.isfile(man_path):
        reasons.append(f"{kit.AR_MANIFEST} missing from EVAL_SHA")
    else:
        man = kit.read_json(man_path)
        reasons += kit.ar_counts_reasons(os.path.join(wt, "AcousticRooms"),
                                         kit.split_rooms(os.path.join(wt, "data/AR/unseen_eval.json")),
                                         man["rooms"])
    return reasons


def ar_gate_reasons(tree, inputs, now=None):
    """§14 H2/H7 before every cell batch: a 0..48 h full AcousticRooms re-hash receipt of the
    committed manifest over THIS root (a runtime input) + the 16-item eval-split content sample."""
    if not inputs or "ar_verify" not in inputs:
        return ["the runtime inputs must name the full AcousticRooms re-hash receipt (ar_verify)"]
    try:
        man = kit.read_json(os.path.join(tree, kit.AR_MANIFEST))
    except (OSError, ValueError) as err:
        return [f"{kit.AR_MANIFEST} unreadable in the eval tree ({err})"]
    ar_root = os.path.join(tree, "AcousticRooms")
    reasons = kit.ar_verify_reasons(inputs["ar_verify"]["path"], man, ar_root, now=now)
    try:
        got = kit.ar_sample_digest(tree, "data/AR/unseen_eval.json", ar_root)
    except (OSError, ValueError) as err:
        reasons.append(f"the AcousticRooms eval sample is unreadable ({err})")
    else:
        if got != kit.AR_EVAL_SAMPLE_DIGEST:
            reasons.append(f"the AcousticRooms eval-split sample digest {got[:12]} != the pinned "
                           f"{kit.AR_EVAL_SAMPLE_DIGEST[:12]}")
    return reasons


def sentinel_line(target):
    return (f"{SENTINEL} attention_backend=sdpa overlay_sha256={EVAL_PINS[NO_FLASH_OVERLAY]} "
            f"launcher_sha256={EVAL_PINS[NO_FLASH_LAUNCHER]} target={target}")


def spawn_env(tree, hf_sealed, base=None):
    base = os.environ if base is None else base
    env = {k: v for k, v in base.items()
           if not k.startswith(("PYTHON", "HF_", "TRANSFORMERS_", "HUGGINGFACE_"))}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.path.join(tree, "overlays/no_flash"), **kit.hf_env(hf_sealed))
    return env


def eval_environment_reasons(measured, tree, py):
    """The measured interpreter, versions and module origins against the A6000 pins."""
    reasons = []
    if measured.get("executable") != os.path.realpath(py):
        reasons.append(f"interpreter {measured.get('executable')} is not the pinned {py}")
    prefix = str(measured.get("prefix")) + os.sep
    for mod, origin in sorted((measured.get("origins") or {}).items()):
        if not isinstance(origin, str) or not origin.startswith(prefix):
            reasons.append(f"module {mod} resolves to {origin}, outside the env prefix {prefix}")
    if measured.get("transformer_module") != os.path.realpath(os.path.join(tree, "src/models/transformer.py")):
        reasons.append(f"src.models.transformer resolves to {measured.get('transformer_module')}, "
                       "not the eval tree's")
    dists = measured.get("dists") or {}
    got = {"python": measured.get("python"), "cuda_runtime": measured.get("cuda_runtime"),
           "cudnn": measured.get("cudnn"), "nccl": measured.get("nccl"),
           **{k: (v.split("+")[0] if isinstance(v, str) else v) for k, v in dists.items()}}
    for key, want in kit.ENV_PINS_A6000.items():
        if got.get(key) != want:
            reasons.append(f"{key} {got.get(key)!r} != pinned {want!r}")
    return reasons


def driver_version(runner=None):
    runner = runner or kit.run
    rc, out, _ = runner(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    vs = {ln.strip() for ln in out.splitlines() if ln.strip()} if rc == 0 else set()
    return vs.pop() if len(vs) == 1 else None


def eval_environment(tree, py, hf_sealed, runner=None, measure=None, driver=None):
    """§14 H3 + code-r1 H4 for evaluation: the environment probe in the eval tree's interpreter,
    the no-flash launcher's own --preflight (SDPA under the overlay, its exact sentinel), the
    sealed hub + offline HF resolution inside it, the A6000 driver pin. Returns
    (reasons, {measured, driver, env_digest}); the digest is bound into E0 and every receipt."""
    runner = runner or kit.run
    measure = measure or kit.measure_environment
    env = spawn_env(tree, hf_sealed)
    plain = {k: v for k, v in env.items() if k != "PYTHONPATH"}
    reasons = []
    try:
        measured = measure(tree, py, plain)
    except Refused as err:
        return [str(err)], None
    reasons += eval_environment_reasons(measured, tree, py)
    rc, out, err = runner([py, os.path.join(tree, NO_FLASH_LAUNCHER), "--preflight"], cwd=tree, env=env)
    lines = [ln for ln in out.splitlines() if ln.startswith(SENTINEL)]
    if rc != 0 or lines != [sentinel_line("--preflight")]:
        reasons.append(f"no-flash preflight rc {rc}, sentinel {lines!r} ({err.strip()[-300:]})")
    reasons += kit.hf_sealed_reasons(hf_sealed)
    reasons += kit.hf_resolution_reasons(hf_sealed, py, env)
    drv = driver if driver is not None else driver_version(runner)
    reasons += kit.driver_reasons("a6000", drv)
    return reasons, {"measured": measured, "driver": drv, "env_digest": kit.environment_digest(measured, drv)}


def runtime_entries_reasons(ri, tree=None):
    """The exact runtime-input entry schema (plan §14 H7; code-r2 B4): ``ar_verify`` required,
    ``admission`` optional, each exactly ``{path, sha256}`` with an absolute path outside the
    eval worktree. Shared by the manifest loader and every receipt validator."""
    if not isinstance(ri, dict) or "ar_verify" not in ri or set(ri) - set(RUNTIME_INPUT_NAMES):
        got = sorted(ri) if isinstance(ri, dict) else type(ri).__name__
        return [f"runtime inputs name {got}; need exactly ar_verify (+ admission for C16)"]
    r = []
    for name in sorted(ri):
        ent = ri[name]
        if not isinstance(ent, dict) or set(ent) != {"path", "sha256"} or not isinstance(ent["path"], str) \
                or not os.path.isabs(ent["path"]) or not _hex64(ent["sha256"]):
            r.append(f"runtime input {name}: need exactly an absolute path and a sha256")
        elif tree is not None and os.path.realpath(ent["path"]).startswith(os.path.realpath(tree) + os.sep):
            r.append(f"runtime input {name} lives inside the eval worktree (§14 H7: never)")
    return r


def inputs_digest(ri):
    """The canonical identity of one runtime-input set (code-r2 B4): what every receipt's
    ``runtime_inputs_sha256`` must equal, recomputable from the receipt's own entries."""
    return kit.digest_obj(ri)


def runtime_inputs(path, tree=None):
    """The hash-pinned post-freeze inputs (plan §14 H7): exactly ``{name: {path, sha256}}`` with
    ``ar_verify`` required and ``admission`` optional, absolute paths outside the eval worktree,
    every file hashing to its pin. Returns (inputs, their canonical digest)."""
    try:
        _, _, spec = _read(path)
    except (OSError, ValueError) as err:
        raise Refused(f"runtime-input manifest {path} unreadable ({err})") from None
    r = runtime_entries_reasons(spec, tree)
    if r:
        raise Refused("; ".join(r))
    for name, ent in spec.items():
        if not os.path.isfile(ent["path"]) or kit.sha256_file(ent["path"]) != ent["sha256"]:
            raise Refused(f"runtime input {name} ({ent['path']}) does not hash to {ent['sha256']}")
    return spec, inputs_digest(spec)


_AR_MANIFESTS = {}


def ar_manifest_of(tree):
    path = os.path.join(tree, kit.AR_MANIFEST)
    if path not in _AR_MANIFESTS:
        _AR_MANIFESTS[path] = kit.read_json(path)
    return _AR_MANIFESTS[path]


def runtime_identity_reasons(ri, ri_sha, cell, tree, t_start, admission=None, resolve=None):
    """The ONE runtime-identity validator (code-r2 B4), used at spawn time and by every receipt
    check (run, verify, collect, figure, table): the exact entry schema; ``ri_sha`` equal to the
    canonical digest of the entries; every entry's bytes still present (``resolve`` maps an entry
    to a local copy, e.g. the committed table evidence); for C16 the admission entry equal to the
    validated admission, which must predate the cell; the AR re-hash receipt a PASS of the
    committed manifest that was 0..48 h old AT THE CELL'S START (``t_start``)."""
    r = runtime_entries_reasons(ri, tree)
    if r:
        return r
    if ri_sha != inputs_digest(ri):
        r.append("runtime_inputs_sha256 != the canonical digest of the recorded runtime inputs")
    if not _num(t_start):
        return r + ["the cell start time is malformed"]
    local = {}
    for name in sorted(ri):
        ent = ri[name]
        p = resolve(name, ent) if resolve is not None else ent["path"]
        if not p or not os.path.isfile(p) or kit.sha256_file(p) != ent["sha256"]:
            r.append(f"runtime input {name} ({ent['path']}) no longer holds its recorded bytes")
        else:
            local[name] = p
    if cell["arm"] == "C16":
        if "admission" not in ri:
            r.append("a C16 cell's runtime inputs name no admission")
        elif admission is None or ri["admission"]["sha256"] != admission[1]:
            r.append("the recorded admission input != the validated C16 admission")
        elif not _num(admission[0].get("t")) or admission[0]["t"] > t_start:
            r.append("the C16 admission postdates the cell's start")
    if "ar_verify" in local:
        try:
            man = ar_manifest_of(tree)
        except (OSError, ValueError) as err:
            r.append(f"{kit.AR_MANIFEST} unreadable ({err})")
        else:
            r += [f"AR receipt at the cell's start: {x}" for x in
                  kit.ar_verify_reasons(local["ar_verify"], man, man["ar_root"], now=t_start)]
    return r


# --------------------------------------------------------------------------- #
# admission and checkpoints: one read, embedded config type-strict equal to the arm's
# --------------------------------------------------------------------------- #
def admission_of(inputs, nas_root=None, rehash="final"):
    """The C16 admission through the kit's ONE validator (B10); its file sha must equal the
    runtime-input pin. Returns (record, sha256)."""
    if not inputs or "admission" not in inputs:
        raise Refused("the C16 arm needs the admission record in --runtime-inputs")
    reasons, rec, sha = kit.admission_reasons(inputs["admission"]["path"],
                                              nas_root=nas_root or kit.NAS_ROOT, rehash=rehash)
    if reasons:
        raise Refused("admission refused: " + "; ".join(reasons[:4]))
    if sha != inputs["admission"]["sha256"]:
        raise Refused("the admission file changed after the runtime-input manifest pinned it")
    return rec, sha


def arm_checkpoint(arm, admission=None):
    """``(source path, sha256, bytes)``: C16 from the validated admission, YAW / P1 pinned."""
    if arm == "C16":
        if admission is None:
            raise Refused("the C16 arm needs the admission record in --runtime-inputs")
        rec = admission[0]
        return rec["ckpt_path"], rec["sha256"], rec["bytes"]
    a = ARMS[arm]
    return a["ckpt"], a["ckpt_sha256"], a["ckpt_bytes"]


def farm_link(farm, arm):
    return os.path.join(farm, arm, "epoch=8-step=40000.ckpt")


def cache_checkpoint(arm, tree, farm, src, sha, nbytes):
    """Content-addressed cache entry ``<farm>/.ckpt_cache/<sha>.ckpt`` (0444) + the farm hard
    link, serialised per SHA by an flock (threads and processes alike), written through a
    unique temporary name; the bytes are hashed and ``torch.load``-parsed from ONE read, and
    the embedded model_config must equal the arm's config, canonical and type-strict (B8)."""
    cache_dir = os.path.join(farm, ".ckpt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{sha}.ckpt")
    link = farm_link(farm, arm)
    fd = os.open(os.path.join(cache_dir, f"{sha}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not os.path.lexists(cached):
            with open(src, "rb") as f:
                blob = f.read()
            if len(blob) != nbytes or _sha(blob) != sha:
                raise Refused(f"{src} is not the pinned {sha} ({nbytes} bytes)")
            import torch
            ck = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            with open(os.path.join(tree, ARMS[arm]["config"])) as f:
                cfg = json.load(f)
            if not isinstance(ck, dict) or kit.canonical(ck.get("model_config")) != kit.canonical(cfg) \
                    or not _strict_equal(ck.get("model_config"), cfg):
                raise Refused(f"{arm}: the checkpoint's embedded model_config != {ARMS[arm]['config']}")
            tmp = f"{cached}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
            try:
                with open(tmp, "xb") as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o444)
                kit.rename_noreplace(tmp, cached)
            finally:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            kit.fsync_path(cache_dir, directory=True)
        st = os.lstat(cached)
        if not os.path.isfile(cached) or os.path.islink(cached) or st.st_mode & 0o222 \
                or st.st_size != nbytes or kit.sha256_file(cached) != sha:
            raise Refused(f"cache entry {cached} is not a read-only regular file re-hashing to {sha}")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.link(cached, link)
        if os.path.islink(link) or not os.path.samestat(os.stat(link), st):
            raise Refused(f"stale farm entry {link} (not the hard link of {cached})")
    finally:
        os.close(fd)
    return link, sha, st.st_size


# --------------------------------------------------------------------------- #
# per-cell verification (plan §5.3, code-r1 H9): exact, type-strict schemas
# --------------------------------------------------------------------------- #
def metrics_reasons(rec, cell, head, ckpt_sha, ckpt_bytes, link, stream=None):
    if not isinstance(rec, dict):
        return ["metrics record is not an object"]
    rnd = cell["rotate_mode"] == "random"
    keys = METRICS_BASE_KEYS | (METRICS_RANDOM_KEYS if rnd else frozenset())
    r = []
    if set(rec) != keys:
        r.append(f"metrics keys differ from the exp_25 schema: {sorted(set(rec) ^ keys)}")
    want = {"cond_method": "vanilla", "cond_autocast": "bf16", "frame_avg_angles": None,
            "frame_avg_fwd_cap": None, "orbit_execution": "n/a", "batch_size": 64, "cfg_scale": 1.0,
            "steps": 1, "device": "cuda", "weights_source": "ema", "n_samples": N_ITEMS,
            "scene_count": 10, "per_scene_schema": 1, "seed": cell["seed"],
            "rotate_deg": cell["rotate_deg"], "dataset_config": cell["dataset_config"],
            "source_sha": head, "ckpt_sha256_loaded": ckpt_sha, "ckpt_bytes_loaded": ckpt_bytes,
            "eval_name": cell["cell"], "ckpt_path": link}
    if rnd:
        st = stream if isinstance(stream, dict) else {}
        want.update(rotate_mode="random", rotate_seed=cell["seed"], stream_count=N_ITEMS, img_w=IMG_W,
                    input_hash=st.get("input_hash"), assignment_hash=st.get("assignment_hash"))
    for key, val in want.items():
        if key in rec and not _strict_equal(rec[key], val):
            r.append(f"metrics {key} = {rec[key]!r}, expected {val!r}")
    m = rec.get("metrics")
    if not (isinstance(m, dict) and set(m) == set(METRIC_KEYS) and all(_num(v) for v in m.values())):
        r.append(f"metric keys / finiteness wrong: {sorted(m) if isinstance(m, dict) else m!r}")
    bs = rec.get("by_scene")
    if not isinstance(bs, dict) or set(bs) != set(SCENES):
        r.append(f"by_scene keys {sorted(bs) if isinstance(bs, dict) else bs!r}")
    else:
        for s in SCENES:
            blk = bs[s]
            if not (isinstance(blk, dict) and set(blk) == set(METRIC_KEYS) and all(_num(v) for v in blk.values())):
                r.append(f"by_scene {s}: keys / finiteness wrong")
    return r


def stream_reasons(stream, cell, offsets, hashes, ev):
    if not isinstance(stream, dict) or set(stream) != STREAM_KEYS:
        got = sorted(stream) if isinstance(stream, dict) else stream
        return [f"stream keys {got!r} != the exp_25 sidecar schema"]
    r = []
    for key, val in (("schema_version", 1), ("fingerprint_schema", 1), ("stream_count", N_ITEMS),
                     ("img_w", IMG_W), ("rotate_mode", cell["rotate_mode"]),
                     ("rotate_deg", cell["rotate_deg"]), ("rotate_seed", cell["rotate_seed"])):
        if not _strict_equal(stream[key], val):
            r.append(f"stream {key} = {stream[key]!r}, expected {val!r}")
    it, at, offs = stream["input_tuples"], stream["assignment_tuples"], stream["offsets"]
    if not (isinstance(it, list) and isinstance(at, list) and isinstance(offs, list)) \
            or not len(it) == len(at) == len(offs) == N_ITEMS:
        return r + [f"stream tuple / offset counts are not all {N_ITEMS}"]
    k = cell["K"]
    for i, (a, b, d) in enumerate(zip(it, at, offs)):
        if not (isinstance(a, list) and len(a) == 4 and _int(a[0]) and a[0] == i and isinstance(a[1], str)
                and isinstance(a[2], list) and len(a[2]) == k and all(isinstance(x, str) for x in a[2])
                and _int(a[3]) and a[3] == IMG_W
                and isinstance(b, list) and len(b) == 3 and _int(b[0]) and b[0] == i and b[1] == a[1]
                and _int(b[2]) and b[2] == d and _int(d) and 0 <= d < IMG_W):
            r.append(f"stream tuple {i} is malformed or inconsistent (input / assignment / offset)")
            break
    if offs != offsets:
        r.append("offsets are not the expected "
                 + ("eval-seed draw" if cell["rotate_mode"] == "random" else "column shift of the angle"))
    if ev.canonical_stream_hash(it) != stream["input_hash"] or \
            ev.canonical_stream_hash(at) != stream["assignment_hash"]:
        r.append("stream hashes do not recompute from their tuples")
    exp_in, exp_assign = hashes[(cell["K"], cell["seed"])]
    if stream["input_hash"] != exp_in:
        r.append("input_hash != the exp_25 YAW cell's at the same (K, seed)")
    if cell["rotate_mode"] == "random" and stream["assignment_hash"] != exp_assign:
        r.append("assignment_hash != the exp_25 YAW R cell's at the same (K, seed)")
    return r


def dump_reasons(ddir, stem, head, link, mpath, spath, msha, ssha, input_hash):
    """``(reasons, waveform sha, manifest sha)`` for one dump directory (exact schema)."""
    try:
        _, man_sha, dman = _read(os.path.join(ddir, "manifest.json"))
        with open(os.path.join(ddir, "COMPLETE"), "rb") as f:
            complete = f.read().decode()
        with open(os.path.join(ddir, "waveforms.sha256"), "rb") as f:
            sidecar = f.read().decode()
    except (OSError, ValueError, UnicodeDecodeError) as err:
        return [f"dump {ddir} incomplete ({err})"], None, None
    if not isinstance(dman, dict) or set(dman) != DUMP_KEYS:
        return [f"dump manifest keys {sorted(dman) if isinstance(dman, dict) else dman!r} != the schema"], None, man_sha
    r = []
    wsha = dman["sha256"]
    if not _hex64(wsha):
        return [f"dump manifest sha256 {wsha!r} is not a sha256"], None, man_sha
    real = lambda p: os.path.realpath(p) if isinstance(p, str) else p  # noqa: E731
    for key, val in (("schema_version", 1), ("n", N_ITEMS), ("dtype", "float32"), ("shape", [N_ITEMS, WAVE_T]),
                     ("input_hash", input_hash), ("metrics_stem", stem), ("source_sha", head),
                     ("ckpt_path", link), ("metrics_sha256", msha), ("stream_sidecar_sha256", ssha),
                     ("canon_sidecar_path", None), ("canon_sidecar_sha256", None)):
        if not _strict_equal(dman[key], val):
            r.append(f"dump manifest {key} = {dman[key]!r}, expected {val!r}")
    for key, val in (("metrics_path", mpath), ("stream_sidecar_path", spath)):
        if not isinstance(dman[key], str) or real(dman[key]) != real(val):
            r.append(f"dump manifest {key} = {dman[key]!r}, expected {val!r}")
    if complete != wsha + "\n":
        r.append("dump COMPLETE marker is not exactly the waveform sha")
    if sidecar != f"{wsha}  waveforms.npy\n":
        r.append("waveforms.sha256 is not exactly the waveform sha line")
    npy = os.path.join(ddir, "waveforms.npy")
    if not os.path.isfile(npy) or kit.sha256_file(npy) != wsha:
        r.append("waveforms.npy does not re-hash to waveforms.sha256")
    if glob.glob(glob.escape(ddir) + ".staging-*"):
        r.append(f"staging leftovers next to {ddir}")
    return r, wsha, man_sha


def sentinel_reasons(log_path, target):
    try:
        with open(log_path, "rb") as f:
            b = f.read()
    except (OSError, TypeError):
        return [f"cell log {log_path!r} unreadable"], None
    text = b.decode(errors="replace")
    lines = [ln for ln in text.splitlines() if ln.startswith(SENTINEL)]
    r = []
    if lines != [sentinel_line(target)]:
        r.append(f"no-flash sentinel lines {lines!r} != exactly the one expected line")
    if "NO-FLASH LAUNCH REFUSED" in text:
        r.append("the no-flash launcher refused")
    return r, _sha(b)


def cell_paths(cell, farm, tree, dump_root):
    stem = stem_of(cell, tree)
    base = os.path.join(farm, cell["arm"], stem)
    return {"stem": stem, "metrics": base + ".json", "stream": base + ".stream.json",
            "receipt": base + ".exp40_receipt.json", "dump": os.path.join(dump_root, stem)}


def verify_cell(cell, farm, tree, head, ckpt_sha, ckpt_bytes, link, log_path, target, hashes,
                dump_root=DUMP_ROOT):
    """``(reasons, digests)``: every artifact of one cell against §5.3, exactly."""
    ev, _ = evaluator(tree)
    p = cell_paths(cell, farm, tree, dump_root)
    try:
        _, msha, rec = _read(p["metrics"])
        _, ssha, stream = _read(p["stream"])
    except (OSError, ValueError) as err:
        return [f"artifacts unreadable ({err})"], None
    if os.path.realpath(target) != os.path.realpath(os.path.join(tree, "eval_FLAC.py")):
        return [f"launcher target {target} is not the eval tree's eval_FLAC.py"], None
    reasons = metrics_reasons(rec, cell, head, ckpt_sha, ckpt_bytes, link, stream)
    reasons += stream_reasons(stream, cell, expected_offsets(cell, tree), hashes, ev)
    in_hash = stream.get("input_hash") if isinstance(stream, dict) else None
    dr, wsha, man_sha = dump_reasons(p["dump"], p["stem"], head, link, p["metrics"], p["stream"],
                                     msha, ssha, in_hash)
    reasons += dr
    sr, log_sha = sentinel_reasons(log_path, target)
    reasons += sr
    digests = {"stem": p["stem"], "metrics_sha256": msha, "stream_sha256": ssha, "input_hash": in_hash,
               "assignment_hash": stream.get("assignment_hash") if isinstance(stream, dict) else None,
               "waveforms_sha256": wsha, "dump_manifest_sha256": man_sha, "log_sha256": log_sha,
               "metrics": rec.get("metrics") if isinstance(rec, dict) else None, "record": rec}
    return reasons, digests


# --------------------------------------------------------------------------- #
# E0 (plan §5.2): references pinned by sha256 in a committed manifest (code-r1 B9)
# --------------------------------------------------------------------------- #
E0_REF_KEYS = frozenset(("schema_version", "kind", "exp25_pin", "exp25_cells_sha256", "farm", "dump_root",
                         "e0", "expected"))


def _ref_file(path):
    _, sha, doc = _read(path)
    return {"path": path, "sha256": sha}, doc


def build_e0_reference(tree=ROOT, farm=EXP25_FARM, dumps=EXP25_DUMPS):
    """ONE-TIME: hash every exp_25 byte E0 and the per-cell hash checks rely on."""
    ev, _ = evaluator(tree)
    pin = SOURCES["exp25"]["pin"]
    if kit.sha256_file(os.path.join(tree, EXP25_CELLS)) != EXP25_CELLS_SHA256:
        raise Refused(f"{EXP25_CELLS} is not the pinned {EXP25_CELLS_SHA256}")
    cells = {c["cell"]: c for c in load_cells(tree)}
    out = {"schema_version": 1, "kind": "exp40_e0_reference", "exp25_pin": pin,
           "exp25_cells_sha256": EXP25_CELLS_SHA256, "farm": farm, "dump_root": dumps, "e0": {}, "expected": {}}
    for name, ref in sorted(E0.items()):
        stem = stem_of({**cells[name], "cell": ref}, tree)
        mref, m = _ref_file(os.path.join(farm, stem + ".json"))
        sref, s = _ref_file(os.path.join(farm, stem + ".stream.json"))
        rref, rc = _ref_file(os.path.join(farm, stem + ".cell.json"))
        if m.get("source_sha") != pin or m.get("eval_name") != ref or rc.get("cell") != ref \
                or rc.get("stem") != stem or rc.get("dump") is not True:
            raise Refused(f"{ref}: the archived record is not the exp_25 pin's cell")
        if ev.canonical_stream_hash(s["input_tuples"]) != s["input_hash"] or \
                ev.canonical_stream_hash(s["assignment_tuples"]) != s["assignment_hash"]:
            raise Refused(f"{ref}: the archived stream does not recompute")
        ddir = os.path.join(dumps, stem)
        _, man_sha, dman = _read(os.path.join(ddir, "manifest.json"))
        with open(os.path.join(ddir, "COMPLETE"), "rb") as f:
            cb = f.read()
        with open(os.path.join(ddir, "waveforms.sha256"), "rb") as f:
            sb = f.read()
        wsha = cb.decode().strip()
        if cb != (wsha + "\n").encode() or sb != f"{wsha}  waveforms.npy\n".encode() or dman.get("sha256") != wsha \
                or dman.get("metrics_sha256") != mref["sha256"] or dman.get("stream_sidecar_sha256") != sref["sha256"]:
            raise Refused(f"{ref}: the archived dump does not bind its metrics / stream / waveforms")
        out["e0"][name] = {"reference": ref, "stem": stem, "metrics": mref, "stream": sref, "receipt": rref,
                           "dump": {"dir": ddir, "manifest_sha256": man_sha, "complete_sha256": _sha(cb),
                                    "sidecar_sha256": _sha(sb), "waveforms_sha256": wsha}}
    for k in KS:
        for s in SEEDS:
            tref, t = _ref_file(os.path.join(farm, f"epoch=8-step=40000_metrics_1_1.0_exp25_YAW_T_S40000_K{k}_s{s}.stream.json"))
            rref, r = _ref_file(os.path.join(farm, f"epoch=8-step=40000_metrics_1_1.0_exp25_YAW_R_S40000_K{k}_s{s}_rotrand{s}.stream.json"))
            for doc in (t, r):
                if ev.canonical_stream_hash(doc["input_tuples"]) != doc["input_hash"] or \
                        ev.canonical_stream_hash(doc["assignment_tuples"]) != doc["assignment_hash"]:
                    raise Refused(f"K{k} s{s}: an archived stream does not recompute")
            if t["input_hash"] != r["input_hash"]:
                raise Refused(f"K{k} s{s}: the T and R input hashes differ")
            out["expected"][f"K{k}_s{s}"] = {"input_hash": t["input_hash"], "assignment_hash_random": r["assignment_hash"],
                                             "T_stream": tref, "R_stream": rref}
    return out


def e0_reference(tree, pinned=None):
    """The committed E0 reference manifest, its sha256 pinned in this file. Returns (ref, sha)."""
    pinned = E0_REFERENCE_SHA256 if pinned is None else pinned
    try:
        _, sha, ref = _read(os.path.join(tree, E0_REFERENCE))
    except (OSError, ValueError) as err:
        raise Refused(f"{E0_REFERENCE} unreadable ({err})") from None
    if sha != pinned:
        raise Refused(f"{E0_REFERENCE} sha256 {sha[:12]} != the pinned {pinned[:12]}")
    if not isinstance(ref, dict) or set(ref) != E0_REF_KEYS or ref["kind"] != "exp40_e0_reference" \
            or sorted(ref["e0"]) != sorted(E0) or sorted(ref["expected"]) != sorted(f"K{k}_s{s}" for k in KS for s in SEEDS) \
            or ref["exp25_pin"] != SOURCES["exp25"]["pin"] or ref["exp25_cells_sha256"] != EXP25_CELLS_SHA256:
        raise Refused(f"{E0_REFERENCE} does not have the exact E0 reference schema")
    return ref, sha


def expected_hashes(ref):
    """``{(K, seed): (input_hash, assignment_hash of the random cell)}`` from the COMMITTED
    reference (no read of any mutable external file)."""
    out = {}
    for k in KS:
        for s in SEEDS:
            e = ref["expected"][f"K{k}_s{s}"]
            out[(k, s)] = (e["input_hash"], e["assignment_hash_random"])
    return out


def replay_reasons(cell, digests, ref):
    """E0: the new YAW cell vs the archived exp_25 cell (exp_36 replay semantics), every
    reference byte validated against its committed sha256 before the comparison."""
    e = ref["e0"].get(cell["cell"])
    if e is None or e["reference"] != cell["replay_of"]:
        return [f"E0 {cell['cell']}: not a pinned reference cell"]
    try:
        with open(e["metrics"]["path"], "rb") as f:
            b = f.read()
    except OSError as err:
        return [f"E0 {cell['cell']}: reference metrics unreadable ({err})"]
    if _sha(b) != e["metrics"]["sha256"]:
        return [f"E0 {cell['cell']}: the reference metrics changed since they were pinned"]
    refm = json.loads(b)
    new = digests.get("record")
    reasons = []
    if not isinstance(new, dict) or set(new) != set(refm):
        return [f"E0 {cell['cell']}: metrics key sets differ from {cell['replay_of']}"]
    for key in sorted(refm):
        if key in ("source_sha", "eval_name", "ckpt_path"):
            continue
        if not _strict_equal(refm[key], new[key]):
            reasons.append(f"E0 {cell['cell']}: {key} differs from {cell['replay_of']}")
    if digests.get("stream_sha256") != e["stream"]["sha256"]:
        reasons.append(f"E0 {cell['cell']}: stream sidecar differs")
    if digests.get("waveforms_sha256") != e["dump"]["waveforms_sha256"]:
        reasons.append(f"E0 {cell['cell']}: scored waveforms differ")
    return reasons


# --------------------------------------------------------------------------- #
# receipts: ONE strict validator for run, verify, collect, figure, table (code-r1 B10)
# --------------------------------------------------------------------------- #
RECEIPT_KEYS = frozenset((
    "schema_version", "kind", "cell", "stem", "eval_sha", "closure", "env_digest", "runtime_inputs",
    "runtime_inputs_sha256", "ar_receipt_sha256", "admission_sha256", "e0_receipt_sha256",
    "e0_reference_sha256", "ckpt_link", "ckpt_sha256", "ckpt_bytes", "target", "lane", "gpu_uuid", "log",
    "log_sha256", "seconds", "metrics_sha256", "stream_sha256", "input_hash", "assignment_hash",
    "waveforms_sha256", "dump_manifest_sha256", "metrics", "t_start", "t"))
RECEIPT_SCHEMA = 3
E0_KEYS = frozenset((
    "schema_version", "kind", "verdict", "eval_sha", "closure", "env_digest", "eval_pins",
    "e0_reference_sha256", "runtime_inputs_sha256", "ar_receipt_sha256", "ckpt_sha256", "ckpt_bytes",
    "model_config_sha256", "dataset_config_sha256", "input_hash_map", "cells", "cell_receipts",
    "failures", "t"))


def identity(eval_sha, closure, ref_sha, hashes, farm, tree, dump_root=None, env_digest=None,
             e0_sha=None, admission=None, inputs_sha=None, run_t0=None):
    """The identities every receipt must carry (admission = (record, sha) or None). In a run,
    ``inputs_sha`` / ``run_t0``: every receipt started by THIS run must carry this run's
    runtime-input identity."""
    return {"eval_sha": eval_sha, "closure": closure, "e0_reference_sha": ref_sha, "hashes": hashes,
            "farm": farm, "tree": tree, "dump_root": dump_root or DUMP_ROOT, "env_digest": env_digest,
            "e0_sha": e0_sha, "admission": admission, "inputs_sha": inputs_sha, "run_t0": run_t0}


def expected_ckpt(cell, ident):
    if cell["arm"] == "C16":
        if ident["admission"] is None:
            return None
        rec = ident["admission"][0]
        return rec["sha256"], rec["bytes"]
    return ARMS[cell["arm"]]["ckpt_sha256"], ARMS[cell["arm"]]["ckpt_bytes"]


def receipt_reasons(rec, cell, ident, offline=False, resolve=None):
    """The ONE cell-receipt validator (code-r1 B10, code-r2 B4) for run, verify, collect, figure
    and the offline table validator (``offline``: no eval-tree / farm path checks, runtime
    inputs resolved through ``resolve``)."""
    if not isinstance(rec, dict) or set(rec) != RECEIPT_KEYS:
        got = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
        return [f"receipt keys {got} != the exact receipt schema"]
    r = []

    def need(ok, msg):
        if not ok:
            r.append(msg)
    need(rec["schema_version"] == RECEIPT_SCHEMA and _int(rec["schema_version"])
         and rec["kind"] == "exp40_cell_receipt", "not an exp_40 cell receipt")
    need(_strict_equal(rec["cell"], cell), "the receipt's cell row != the committed manifest's")
    need(rec["eval_sha"] == ident["eval_sha"], f"receipt eval_sha {rec['eval_sha']} != EVAL_SHA {ident['eval_sha']}")
    need(rec["closure"] == ident["closure"], "receipt closure != the EVAL_SHA closure")
    need(rec["e0_reference_sha256"] == ident["e0_reference_sha"], "receipt E0 reference != the committed one")
    need(ident["env_digest"] is not None and rec["env_digest"] == ident["env_digest"],
         "receipt environment digest != the E0 / current one")
    ri = rec["runtime_inputs"]
    need(isinstance(ri, dict) and isinstance(ri.get("ar_verify"), dict)
         and rec["ar_receipt_sha256"] == ri["ar_verify"].get("sha256"), "receipt AR receipt sha != its runtime input")
    need(_num(rec["t_start"]) and _num(rec["t"]) and _num(rec["seconds"]) and rec["t_start"] <= rec["t"]
         and abs(rec["t"] - rec["t_start"] - rec["seconds"]) <= 5.0, "receipt start / end / duration inconsistent")
    adm = ident["admission"] if cell["arm"] == "C16" else None
    r += runtime_identity_reasons(ri, rec["runtime_inputs_sha256"], cell, ident["tree"], rec["t_start"],
                                  admission=adm, resolve=resolve)
    if ident.get("inputs_sha") is not None and _num(rec["t_start"]) and rec["t_start"] >= ident["run_t0"]:
        need(rec["runtime_inputs_sha256"] == ident["inputs_sha"],
             "receipt runtime inputs != this run's runtime-input manifest")
    if cell["arm"] == "C16":
        need(adm is not None and rec["admission_sha256"] == adm[1], "receipt admission != the validated C16 admission")
    else:
        need(rec["admission_sha256"] is None, "a non-C16 receipt names an admission")
    if cell["block"] == "E0":
        need(rec["e0_receipt_sha256"] is None, "an E0 cell receipt names an E0 receipt")
    else:
        need(ident["e0_sha"] is not None and rec["e0_receipt_sha256"] == ident["e0_sha"],
             "receipt E0 receipt != the EQUAL E0 receipt of EVAL_SHA + closure")
    ck = expected_ckpt(cell, ident)
    need(ck is not None and rec["ckpt_sha256"] == ck[0] and rec["ckpt_bytes"] == ck[1] and _int(rec["ckpt_bytes"]),
         "receipt checkpoint != the arm's pinned / admitted checkpoint")
    if not offline:
        need(rec["stem"] == stem_of(cell, ident["tree"]), "receipt stem != the cell's")
        need(rec["ckpt_link"] == farm_link(ident["farm"], cell["arm"]), "receipt checkpoint link != the farm entry")
        need(isinstance(rec["target"], str) and os.path.realpath(rec["target"]) ==
             os.path.realpath(os.path.join(ident["tree"], "eval_FLAC.py")), "receipt target != the eval tree's eval_FLAC.py")
    for k in ("metrics_sha256", "stream_sha256", "waveforms_sha256", "dump_manifest_sha256", "log_sha256",
              "input_hash", "assignment_hash"):
        need(_hex64(rec[k]), f"receipt {k} is not a sha256")
    need(_int(rec["lane"]) and isinstance(rec["gpu_uuid"], str) and isinstance(rec["log"], str),
         "receipt lane / gpu / log malformed")
    return r


def full_cell_reasons(cell, ident, ref=None):
    """Receipt + every artifact, re-verified now (and E0's replay against the pinned reference).
    Returns (reasons, receipt sha256 or None)."""
    p = cell_paths(cell, ident["farm"], ident["tree"], ident["dump_root"])
    try:
        _, rsha, rec = _read(p["receipt"])
    except (OSError, ValueError) as err:
        return [f"no readable receipt {p['receipt']} ({type(err).__name__})"], None
    r = receipt_reasons(rec, cell, ident)
    if r:
        return r, rsha
    vr, d = verify_cell(cell, ident["farm"], ident["tree"], ident["eval_sha"], rec["ckpt_sha256"],
                        rec["ckpt_bytes"], rec["ckpt_link"], rec["log"], rec["target"], ident["hashes"],
                        dump_root=ident["dump_root"])
    r += vr
    if d is not None:
        for k in ("metrics_sha256", "stream_sha256", "input_hash", "assignment_hash", "waveforms_sha256",
                  "dump_manifest_sha256", "log_sha256"):
            if d[k] != rec[k]:
                r.append(f"{k} now differs from the receipt")
        if not _strict_equal(d["metrics"], rec["metrics"]):
            r.append("metrics now differ from the receipt")
        if not vr and cell["block"] == "E0" and ref is not None:
            r += replay_reasons(cell, d, ref)
    return r, rsha


def e0_receipt_path(receipts, eval_sha, closure):
    return os.path.join(receipts, f"e0_{eval_sha}_{closure}.json")


def e0_dataset_configs(tree):
    """The E0 cells' dataset-config pins, derived from the committed cell rows."""
    return {str(c["K"]): c["dataset_config_sha256"] for c in load_cells(tree) if c["block"] == "E0"}


def e0_receipt_doc(verdict, ident, ref, inputs_sha, ar_sha, cell_receipts, failures):
    return {"schema_version": 3, "kind": "exp40_e0_receipt", "verdict": verdict,
            "eval_sha": ident["eval_sha"], "closure": ident["closure"], "env_digest": ident["env_digest"],
            "eval_pins": dict(EVAL_PINS), "e0_reference_sha256": ident["e0_reference_sha"],
            "runtime_inputs_sha256": inputs_sha, "ar_receipt_sha256": ar_sha,
            "ckpt_sha256": ARMS["YAW"]["ckpt_sha256"], "ckpt_bytes": ARMS["YAW"]["ckpt_bytes"],
            "model_config_sha256": ARMS["YAW"]["config_sha256"],
            "dataset_config_sha256": e0_dataset_configs(ident["tree"]),
            "input_hash_map": {f"K{k}_s{s}": list(v) for (k, s), v in sorted(ident["hashes"].items())},
            "cells": sorted(E0), "cell_receipts": dict(sorted(cell_receipts.items())),
            "failures": [[n, list(rs)] for n, rs in failures], "t": time.time()}


def e0_gate_reasons(e0_path, ident, check_cells=True, cell_recs=None, resolve=None):
    """``(reasons, sha256, doc)``: an EQUAL E0 receipt for EVAL_SHA + closure (+ the current
    environment digest when the identity carries one), bound to the committed reference and to
    the three cell receipts, which must still hash to what it recorded, pass the ONE receipt
    validator (runtime inputs + AR freshness at their own start) and agree with it on the
    runtime-input identity, the AR receipt, the environment, the dataset configs and time
    (code-r2 B4). ``cell_recs`` ({cell: (record, sha256)}) + ``resolve``: the offline table path."""
    try:
        _, sha, e0 = _read(e0_path)
    except (OSError, ValueError):
        return [f"no E0 receipt {e0_path}: run --blocks E0 first"], None, None
    if not isinstance(e0, dict) or set(e0) != E0_KEYS:
        return [f"{e0_path} does not have the exact E0 receipt schema"], sha, None
    r = []

    def need(ok, msg):
        if not ok:
            r.append(msg)
    need(e0["kind"] == "exp40_e0_receipt" and e0["schema_version"] == 3, "not an exp_40 E0 receipt")
    need(e0["verdict"] == "EQUAL" and e0["failures"] == [], f"E0 verdict {e0['verdict']!r} is not EQUAL")
    need(e0["eval_sha"] == ident["eval_sha"] and e0["closure"] == ident["closure"],
         f"E0 receipt is not for EVAL_SHA {ident['eval_sha'][:12]} / closure {ident['closure'][:12]}")
    need(ident["env_digest"] is None or e0["env_digest"] == ident["env_digest"],
         "the environment digest differs from E0's: the environment changed since E0")
    need(_hex64(e0["env_digest"]), "E0 environment digest is not a sha256")
    need(e0["e0_reference_sha256"] == ident["e0_reference_sha"], "E0 receipt reference != the committed one")
    need(e0["eval_pins"] == EVAL_PINS and e0["cells"] == sorted(E0), "E0 receipt pins / cells differ")
    need(e0["ckpt_sha256"] == ARMS["YAW"]["ckpt_sha256"] and e0["ckpt_bytes"] == ARMS["YAW"]["ckpt_bytes"]
         and e0["model_config_sha256"] == ARMS["YAW"]["config_sha256"], "E0 receipt checkpoint / config differ")
    need(e0["input_hash_map"] == {f"K{k}_s{s}": list(v) for (k, s), v in sorted(ident["hashes"].items())},
         "E0 receipt input-hash map != the committed reference")
    want_ds = e0_dataset_configs(ident["tree"])
    need(e0["dataset_config_sha256"] == want_ds
         and all(v == EVAL_PINS[DATASET_OF_K[int(k)]] for k, v in want_ds.items()),
         "E0 receipt dataset configs != its cells' pinned dataset configs")
    need(_hex64(e0["runtime_inputs_sha256"]) and _hex64(e0["ar_receipt_sha256"]),
         "E0 receipt runtime-input / AR receipt identities are not sha256s")
    need(_num(e0["t"]) and e0["t"] <= time.time() + 60.0, "E0 receipt time malformed or in the future")
    cr = e0["cell_receipts"]
    need(isinstance(cr, dict) and sorted(cr) == sorted(E0) and all(_hex64(v) for v in cr.values()),
         "E0 receipt does not bind the three cell receipts")
    if not r and check_cells:
        cell_ident = {**ident, "env_digest": e0["env_digest"], "e0_sha": None, "admission": None,
                      "inputs_sha": None, "run_t0": None}
        for c in [c for c in load_cells(ident["tree"]) if c["block"] == "E0"]:
            if cell_recs is not None:
                rec, rsha = cell_recs.get(c["cell"], (None, None))
            else:
                rp = cell_paths(c, ident["farm"], ident["tree"], ident["dump_root"])["receipt"]
                try:
                    _, rsha, rec = _read(rp)
                except (OSError, ValueError):
                    rec, rsha = None, None
            if rec is None or rsha != cr[c["cell"]]:
                r.append(f"E0 cell receipt of {c['cell']} changed since the E0 receipt (or is missing)")
                continue
            cr_r = receipt_reasons(rec, c, cell_ident, offline=cell_recs is not None, resolve=resolve)
            if cr_r:
                r.append(f"E0 cell receipt of {c['cell']} is invalid: {cr_r[:3]}")
                continue
            need(rec["runtime_inputs_sha256"] == e0["runtime_inputs_sha256"],
                 f"E0 receipt runtime_inputs_sha256 != its cell receipt's ({c['cell']})")
            need(rec["ar_receipt_sha256"] == e0["ar_receipt_sha256"],
                 f"E0 receipt ar_receipt_sha256 != its cell receipt's ({c['cell']})")
            need(rec["t"] <= e0["t"], f"E0 receipt predates its cell receipt ({c['cell']})")
    return r, sha, e0


# --------------------------------------------------------------------------- #
# running (plan §5.3; code-r1 B8, H8): card lock, per-card GPU lock, owned children only
# --------------------------------------------------------------------------- #
def hold_cards_lock(path=None):
    """The exp_40 card flock, held for the whole evaluation (the training launcher takes it too)."""
    path = path or kit.CARDS_LOCK
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Refused(f"another exp_40 training / evaluation holds {path}") from None
    return fd


def lane_state(lane, query):
    """``(uuid, reasons)``: the lane's card is the pinned model, has no compute process and
    >= MIN_GPU_FREE_MIB free."""
    gpus, apps = query()
    g = {x["index"]: x for x in gpus}.get(lane)
    if g is None:
        return None, [f"GPU {lane} not visible"]
    r = []
    if g["name"] != GPU_NAME:
        r.append(f"GPU {lane} is {g['name']!r}, not {GPU_NAME!r}")
    users = [a["pid"] for a in apps if a["uuid"] == g["uuid"]]
    if users:
        r.append(f"GPU {lane} holds compute process(es) {users}")
    if g["free_mib"] < MIN_GPU_FREE_MIB:
        r.append(f"GPU {lane} has {g['free_mib']} MiB free < {MIN_GPU_FREE_MIB}")
    return g["uuid"], r


def acquire_lane(lane, payload, query, handshake_dir):
    """The cooperative ``GPU_LOCK_<uuid>`` of the lane's card, taken only on a free card and
    re-checked under the lock. Returns ((uuid, token, paths) or None, reasons)."""
    uid, r = lane_state(lane, query)
    if r:
        return None, r
    try:
        token, paths = kit.acquire_gpu_locks(handshake_dir, [uid], payload)
    except Refused as err:
        return None, [str(err)]
    uid2, r = lane_state(lane, query)
    if r or uid2 != uid:
        kit.release_gpu_locks(paths, token)
        return None, r or [f"GPU {lane} changed uuid {uid} -> {uid2}"]
    return (uid, token, paths), []


class Children:
    """Owned evaluator processes, each in its own session: a signal terminates ONLY those
    process groups (TERM, bounded wait, then KILL) and stops new spawns."""

    def __init__(self):
        self.stop = threading.Event()
        self.interrupted = False
        self._lock = threading.Lock()
        self._procs = {}

    def run(self, argv, cwd, env, log_path):
        with open(log_path, "wb") as log:
            with self._lock:
                if self.stop.is_set():
                    raise Refused("stopping: no new evaluator is spawned")
                p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                self._procs[p.pid] = p
            try:
                return p.wait()
            finally:
                with self._lock:
                    self._procs.pop(p.pid, None)

    def terminate(self, grace_s=60.0):
        self.stop.set()
        self.interrupted = True
        with self._lock:
            procs = list(self._procs.values())
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        def escalate():
            deadline = time.time() + grace_s
            for p in procs:
                try:
                    p.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        threading.Thread(target=escalate, daemon=True).start()


def eval_argv(cell, tree, link, py, dump_root):
    return [py, os.path.join(tree, NO_FLASH_LAUNCHER), os.path.join(tree, "eval_FLAC.py"),
            *cell["eval_args"], "--ckpt-path", link, "--dump-scored-waveforms", dump_root]


def default_spawn(ctx, cell, link, lane, log_path):
    env = spawn_env(ctx["tree"], ctx["hf_sealed"])
    env["CUDA_VISIBLE_DEVICES"] = str(lane)
    return ctx["children"].run(eval_argv(cell, ctx["tree"], link, ctx["py"], ctx["dump_root"]),
                               ctx["tree"], env, log_path)


def _process_cell(cell, lane, ctx, fail):
    ident, tree, farm = ctx["ident"], ctx["tree"], ctx["farm"]
    p = cell_paths(cell, farm, tree, ctx["dump_root"])
    if os.path.lexists(p["receipt"]):
        r, _ = full_cell_reasons(cell, ident, ctx["ref"])
        if not r:
            print(f"SKIP {cell['cell']} (verified)", flush=True)
            return
        fail(cell, ["existing receipt does not re-verify"] + r)
        return
    if any(os.path.lexists(p[k]) for k in ("metrics", "stream", "dump")) or \
            glob.glob(glob.escape(p["dump"]) + ".staging-*"):
        fail(cell, ["artifacts exist without a receipt: refusing to overwrite"])
        return
    src, sha, nbytes = arm_checkpoint(cell["arm"], ident["admission"])
    link, sha, nbytes = cache_checkpoint(cell["arm"], tree, farm, src, sha, nbytes)
    payload = {"experiment": "exp_40", "role": "eval", "cell": cell["cell"], "host": socket.gethostname(),
               "pid": os.getpid()}
    waited = 0.0
    while True:
        if ctx["children"].stop.is_set():
            fail(cell, ["interrupted before spawn"])
            return
        got, why = acquire_lane(lane, payload, ctx["query"], ctx["handshake_dir"])
        if got:
            break
        if waited >= ctx["max_wait_s"]:
            fail(cell, [f"lane {lane} not free after {waited:.0f} s"] + why)
            return
        ctx["sleep"](ctx["poll_s"])
        waited += ctx["poll_s"]
    uid, token, paths = got
    lock_rec = os.path.join(ctx["lockrec_dir"], f"{token}.json")
    rc, t0, log_path = None, time.time(), None
    try:
        os.makedirs(ctx["lockrec_dir"], exist_ok=True)
        kit.atomic_write_json(lock_rec, {"token": token, "paths": paths, "cell": cell["cell"], "pid": os.getpid()})
        r = eval_worktree_reasons(tree, ident["eval_sha"], ctx["hf_sealed"])     # before EVERY spawn
        if r:
            ctx["children"].stop.set()
            fail(cell, ["eval worktree drifted; stopping"] + r)
            return
        uid_now, r = lane_state(lane, ctx["query"])                            # immediately before spawn
        if r or uid_now != uid:
            fail(cell, ["the lane changed under the lock"] + r)
            return
        t0 = time.time()                                    # the cell's start (code-r2 B4)
        r = runtime_identity_reasons(ctx["inputs"], ctx["inputs_sha"], cell, tree, t0,
                                     admission=ident["admission"] if cell["arm"] == "C16" else None)
        if r:
            fail(cell, ["runtime inputs refused at the cell's start"] + r)
            return
        log_path = os.path.join(ctx["logdir"], f"yaw_aug_c16_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{cell['cell']}_eval.log")
        rc = ctx["spawn"](ctx, cell, link, lane, log_path)
    finally:
        kit.release_gpu_locks(paths, token)
        if os.path.lexists(lock_rec):
            os.unlink(lock_rec)
    if rc != 0:
        fail(cell, [f"eval rc {rc}"])
        return
    target = os.path.join(tree, "eval_FLAC.py")
    vr, d = verify_cell(cell, farm, tree, ident["eval_sha"], sha, nbytes, link, log_path, target,
                        ident["hashes"], dump_root=ctx["dump_root"])
    if not vr and cell["block"] == "E0":
        vr = replay_reasons(cell, d, ctx["ref"])
    if vr:
        fail(cell, vr)
        return
    inputs = ctx["inputs"]
    t_end = time.time()
    kit.atomic_write_json(p["receipt"], {
        "schema_version": RECEIPT_SCHEMA, "kind": "exp40_cell_receipt", "cell": cell, "stem": p["stem"],
        "eval_sha": ident["eval_sha"], "closure": ident["closure"], "env_digest": ident["env_digest"],
        "runtime_inputs": inputs, "runtime_inputs_sha256": ctx["inputs_sha"],
        "ar_receipt_sha256": inputs["ar_verify"]["sha256"],
        "admission_sha256": ident["admission"][1] if cell["arm"] == "C16" else None,
        "e0_receipt_sha256": None if cell["block"] == "E0" else ident["e0_sha"],
        "e0_reference_sha256": ident["e0_reference_sha"], "ckpt_link": link, "ckpt_sha256": sha,
        "ckpt_bytes": nbytes, "target": target, "lane": lane, "gpu_uuid": uid, "log": log_path,
        "log_sha256": d["log_sha256"], "seconds": t_end - t0, "metrics_sha256": d["metrics_sha256"],
        "stream_sha256": d["stream_sha256"], "input_hash": d["input_hash"],
        "assignment_hash": d["assignment_hash"], "waveforms_sha256": d["waveforms_sha256"],
        "dump_manifest_sha256": d["dump_manifest_sha256"], "metrics": d["metrics"], "t_start": t0, "t": t_end})
    print(f"DONE {cell['cell']} ({time.time() - t0:.0f} s)", flush=True)


def run_cells(cells, ctx):
    """Every selected cell on the lanes; a worker exception is a recorded failure, and success
    means every selected cell ends with exactly one receipt that re-verifies now (B8)."""
    work = queue.Queue()
    for c in cells:
        work.put(c)
    failures, flock_ = [], threading.Lock()

    def fail(cell, reasons):
        with flock_:
            failures.append((cell["cell"], list(reasons)))
        print(f"FAIL {cell['cell']}: {list(reasons)[:3]}", flush=True)

    def worker(lane):
        while not ctx["children"].stop.is_set():
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                _process_cell(cell, lane, ctx, fail)
            except Exception as err:          # never a silent thread death
                fail(cell, [f"worker exception {type(err).__name__}: {err}"])

    threads = [threading.Thread(target=worker, args=(lane,)) for lane in ctx["lanes"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = {n for n, _ in failures}
    for c in cells:
        if c["cell"] in failed:
            continue
        r, _ = full_cell_reasons(c, ctx["ident"], ctx["ref"])
        if r:
            failures.append((c["cell"], ["no verified receipt after the run"] + r))
    return failures


# --------------------------------------------------------------------------- #
# comparator registry (plan §5.2; code-r1 H10)
# --------------------------------------------------------------------------- #
REG_KEYS = frozenset(("schema_version", "kind", "count", "exp25_pin", "exp36_pin", "validators",
                      "e0_reference_sha256", "cells"))
REG_CELL_KEYS = frozenset((
    "source", "arm", "cell", "block", "K", "seed", "rotate_deg", "rotate_mode", "stem", "metrics_path",
    "metrics_sha256", "stream_path", "stream_sha256", "receipt_path", "receipt_sha256", "source_sha",
    "ckpt_sha256", "ckpt_bytes", "model_config", "model_config_sha256", "dataset_config",
    "dataset_config_sha256", "input_hash", "assignment_hash", "dump_dir", "waveforms_sha256",
    "dump_manifest_sha256", "complete_sha256", "sidecar_sha256", "replay_verdict"))


def registry_want():
    """EXACTLY the reused cells (§5.2): exp_25 YAW / P1 T + R (s42-46), C4 + 45 (s42); exp_36
    YAW / P1 135 / 225 / 315 (s42) and YAW C4 s43-46."""
    want = []

    def add(src, arm, name, k, s, deg, mode, block, chain=None):
        want.append({"source": src, "arm": arm, "cell": name, "K": k, "seed": s, "rotate_deg": deg,
                     "rotate_mode": mode, "block": block, "chain": chain})
    for arm in ("YAW", "P1"):
        for k in KS:
            for s in SEEDS:
                add("exp25", arm, f"exp25_{arm}_T_S40000_K{k}_s{s}", k, s, 0.0, "fixed", "T")
                add("exp25", arm, f"exp25_{arm}_R_S40000_K{k}_s{s}", k, s, None, "random", "R")
            for d in C4 + (45,):
                add("exp25", arm, f"exp25_{arm}_C4_S40000_K{k}_s42_th{d}", k, 42, float(d), "fixed",
                    "C4" if d in C4 else "G45")
            for d in (135, 225, 315):
                name = f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" if arm == "YAW" else f"exp35_P1_OFF_S40000_K{k}_s42_th{d}"
                add("exp36", arm, name, k, 42, float(d), "fixed", "G45", "offorbit")
    for k in KS:
        for s in (43, 44, 45, 46):
            for d in C4:
                add("exp36", "YAW", f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}", k, s, float(d), "fixed", "C4", "A")
    return want


AUTH_SNIPPET = r'''
import json, sys
sys.dont_write_bytecode = True
d, pin, base, farm_root, dump_root, stems = sys.argv[1:6] + [json.loads(sys.argv[6])]
sys.path.insert(0, d + "/worklog/worklog_yixun/exp_25_depth_canon_claude")
import depth_canon_collect as C
assert C.REPO == d, (C.REPO, d)
man = C.load_cells()
out = {s: C.verify_cell(s, manifest=man, farm_root=farm_root, dump_root=dump_root, pin=pin,
                        base_dir=base, dump_check="full") for s in stems}
print("EXP40_AUTH_RESULT " + json.dumps(out, sort_keys=True))
'''


def authoritative_validate(source, stems, py=None, repo=ROOT):
    """``{stem: reasons}`` from the source experiment's own ``depth_canon_collect.verify_cell``
    at its pin: ``git archive <pin>`` into a temporary tree (the validator blob checked against
    its pin), run ``-B`` there with the data read in place under the source worktree."""
    src = SOURCES[source]
    py = py or kit.PLATFORMS["a6000"]["python"]
    blob = kit.git(repo, "rev-parse", f"{src['pin']}:{VALIDATOR}").strip()
    if blob != src["validator_blob"]:
        raise Refused(f"{source}: validator blob {blob} != pinned {src['validator_blob']}")
    names = kit.git(repo, "ls-tree", "--name-only", src["pin"], "--", *VALIDATOR_ARCHIVE).split()
    with tempfile.TemporaryDirectory(prefix="exp40_auth_") as d:
        a = subprocess.run(["git", "-C", repo, "archive", src["pin"], "--", *names], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", d], input=a.stdout, check=True)
        os.mkdir(os.path.join(d, ".git"))                # the validator locates its repo root by .git
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        p = subprocess.run([py, "-B", "-c", AUTH_SNIPPET, d, src["pin"], src["base_dir"], COMP_FARM_ROOT,
                            COMP_DUMP_ROOT, json.dumps(sorted(stems))], cwd=d, env=env,
                           capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("EXP40_AUTH_RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise Refused(f"{source} validator failed (rc {p.returncode}): {p.stderr.strip()[-600:]}")
    return json.loads(lines[0][len("EXP40_AUTH_RESULT "):])


def _comparator_entry(w, receipt_path, hashes, verdicts):
    """Our own strict checks on top of the authoritative validator, and every pinned field."""
    src = SOURCES[w["source"]]
    farm = os.path.dirname(receipt_path)
    _, rsha, rc = _read(receipt_path)
    stem = rc.get("stem")
    if rc.get("cell") != w["cell"] or not isinstance(stem, str) or rc.get("arm") != w["arm"] \
            or rc.get("dump") is not True or rc.get("dump_root") != COMP_DUMP_ROOT \
            or rc.get("ckpt_sha256") != ARMS[w["arm"]]["ckpt_sha256"] \
            or rc.get("model_config") != ARMS[w["arm"]]["config"] \
            or rc.get("model_config_sha256") != ARMS[w["arm"]]["config_sha256"] \
            or rc.get("dataset_config") != DATASET_OF_K[w["K"]] \
            or rc.get("dataset_config_sha256") != EVAL_PINS[DATASET_OF_K[w["K"]]] \
            or (w["source"] == "exp36" and rc.get("expect_head") != src["pin"]):
        raise Refused(f"{w['cell']}: its receipt does not match the registry entry")
    mpath, spath = os.path.join(farm, stem + ".json"), os.path.join(farm, stem + ".stream.json")
    _, msha, m = _read(mpath)
    _, ssha, st = _read(spath)
    cell = {"cell": w["cell"], "seed": w["seed"], "rotate_deg": w["rotate_deg"], "rotate_mode": w["rotate_mode"],
            "dataset_config": DATASET_OF_K[w["K"]]}
    want = {"source_sha": src["pin"], "cond_method": "vanilla", "seed": w["seed"], "rotate_deg": w["rotate_deg"],
            "ckpt_sha256_loaded": ARMS[w["arm"]]["ckpt_sha256"], "ckpt_bytes_loaded": ARMS[w["arm"]]["ckpt_bytes"],
            "n_samples": N_ITEMS, "dataset_config": cell["dataset_config"], "eval_name": w["cell"]}
    bad = [k for k, v in want.items() if not isinstance(m, dict) or not _strict_equal(m.get(k), v)]
    keys = METRICS_BASE_KEYS | (METRICS_RANDOM_KEYS if w["rotate_mode"] == "random" else frozenset())
    mm = m.get("metrics") if isinstance(m, dict) else None
    if bad or set(m) != keys or not (isinstance(mm, dict) and set(mm) == set(METRIC_KEYS)
                                     and all(_num(v) for v in mm.values())):
        raise Refused(f"{w['cell']}: protocol / schema fields do not match the registry entry {bad}")
    exp_in, exp_assign = hashes[(w["K"], w["seed"])]
    if not isinstance(st, dict) or st.get("input_hash") != exp_in or \
            (w["rotate_mode"] == "random" and st.get("assignment_hash") != exp_assign):
        raise Refused(f"{w['cell']}: stream hashes != the exp_25 YAW stream at (K{w['K']}, s{w['seed']})")
    ddir = os.path.join(COMP_DUMP_ROOT, stem)
    _, man_sha, dman = _read(os.path.join(ddir, "manifest.json"))
    with open(os.path.join(ddir, "COMPLETE"), "rb") as f:
        cb = f.read()
    with open(os.path.join(ddir, "waveforms.sha256"), "rb") as f:
        sb = f.read()
    wsha = dman.get("sha256")
    if not _hex64(wsha) or cb != (wsha + "\n").encode() or sb != f"{wsha}  waveforms.npy\n".encode() \
            or dman.get("metrics_sha256") != msha or dman.get("stream_sidecar_sha256") != ssha \
            or dman.get("metrics_stem") != stem or dman.get("n") != N_ITEMS:
        raise Refused(f"{w['cell']}: its dump does not bind its metrics / stream / waveforms")
    verdict = None
    if w["source"] == "exp36":
        vpath, vsha = verdicts[(w["chain"], w["arm"])]
        _, got, v = _read(vpath)
        if got != vsha or v.get("check") != "exp36_replay" or v.get("result") != "equal" or v.get("pin") != src["pin"]:
            raise Refused(f"{w['cell']}: the exp_36 replay verdict {vpath} is not the pinned EQUAL one")
        verdict = {"path": vpath, "sha256": vsha}
    return {"source": w["source"], "arm": w["arm"], "cell": w["cell"], "block": w["block"], "K": w["K"],
            "seed": w["seed"], "rotate_deg": w["rotate_deg"], "rotate_mode": w["rotate_mode"], "stem": stem,
            "metrics_path": mpath, "metrics_sha256": msha, "stream_path": spath, "stream_sha256": ssha,
            "receipt_path": receipt_path, "receipt_sha256": rsha, "source_sha": src["pin"],
            "ckpt_sha256": ARMS[w["arm"]]["ckpt_sha256"], "ckpt_bytes": ARMS[w["arm"]]["ckpt_bytes"],
            "model_config": ARMS[w["arm"]]["config"], "model_config_sha256": ARMS[w["arm"]]["config_sha256"],
            "dataset_config": DATASET_OF_K[w["K"]], "dataset_config_sha256": EVAL_PINS[DATASET_OF_K[w["K"]]],
            "input_hash": st["input_hash"], "assignment_hash": st["assignment_hash"], "dump_dir": ddir,
            "waveforms_sha256": wsha, "dump_manifest_sha256": man_sha, "complete_sha256": _sha(cb),
            "sidecar_sha256": _sha(sb), "replay_verdict": verdict}


def register_comparators(tree=ROOT, sources=None, validate=None, verdicts=None):
    """Pin the reused cells (§5.2): exactly ``registry_want()``, each receipt found by its own
    ``*.cell.json`` (never inferred from names), each cell validated by its source experiment's
    authoritative validator and then by our strict checks."""
    sources = sources or SOURCES
    validate = validate or authoritative_validate
    verdicts = verdicts or EXP36_VERDICTS
    ref, ref_sha = e0_reference(tree)
    hashes = expected_hashes(ref)
    want = registry_want()
    index = {}
    for src, s in sources.items():
        for arm in ("YAW", "P1"):
            d = os.path.join(s["base_dir"], COMP_FARM_ROOT, arm)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if f.endswith(".cell.json"):
                    index.setdefault((src, arm, kit.read_json(os.path.join(d, f)).get("cell")), []).append(
                        os.path.join(d, f))
    located = []
    for w in want:
        cands = index.get((w["source"], w["arm"], w["cell"]), [])
        if len(cands) != 1:
            raise Refused(f"{w['source']} {w['arm']} {w['cell']}: {len(cands)} receipts")
        located.append((w, cands[0]))
    for src in sources:
        stems = {kit.read_json(p)["stem"]: w["cell"] for w, p in located if w["source"] == src}
        res = validate(src, sorted(stems))
        bad = {s: r for s, r in res.items() if r}
        if sorted(res) != sorted(stems) or bad:
            raise Refused(f"{src}: the authoritative validator refused {sorted(bad)[:4]} "
                          f"({next(iter(bad.values()))[:2] if bad else 'missing results'})")
    cells = [_comparator_entry(w, p, hashes, verdicts) for w, p in located]
    return {"schema_version": 2, "kind": "exp40_comparators", "count": len(cells),
            "exp25_pin": sources["exp25"]["pin"], "exp36_pin": sources["exp36"]["pin"],
            "validators": {s: {"path": VALIDATOR, "pin": v["pin"], "blob": v["validator_blob"]}
                           for s, v in sorted(sources.items())},
            "e0_reference_sha256": ref_sha, "cells": cells}


def registry_reasons(reg, ref_sha, sources=None, rehash_npy=True):
    """The committed registry against the declared contract, every registered byte re-hashed."""
    sources = sources or SOURCES
    if not isinstance(reg, dict) or set(reg) != REG_KEYS or reg["kind"] != "exp40_comparators" \
            or reg["schema_version"] != 2:
        return ["the comparator registry does not have the exact schema"]
    r = []
    if reg["exp25_pin"] != sources["exp25"]["pin"] or reg["exp36_pin"] != sources["exp36"]["pin"]:
        r.append("registry pins are not the declared exp_25 / exp_36 pins")
    if reg["validators"] != {s: {"path": VALIDATOR, "pin": v["pin"], "blob": v["validator_blob"]}
                             for s, v in sorted(sources.items())}:
        r.append("registry validators are not the declared ones")
    if reg["e0_reference_sha256"] != ref_sha:
        r.append("registry was built against another E0 reference")
    want = [(w["source"], w["arm"], w["cell"]) for w in registry_want()]
    got = [(e.get("source"), e.get("arm"), e.get("cell")) for e in reg["cells"]]
    if got != want or reg["count"] != len(want):
        return r + [f"registry cells are not exactly the {len(want)} registered ones"]
    for e in reg["cells"]:
        if set(e) != REG_CELL_KEYS:
            r.append(f"{e['cell']}: registry entry schema")
            continue
        files = [("metrics", e["metrics_path"], e["metrics_sha256"]), ("stream", e["stream_path"], e["stream_sha256"]),
                 ("receipt", e["receipt_path"], e["receipt_sha256"]),
                 ("dump manifest", os.path.join(e["dump_dir"], "manifest.json"), e["dump_manifest_sha256"]),
                 ("COMPLETE", os.path.join(e["dump_dir"], "COMPLETE"), e["complete_sha256"]),
                 ("waveforms.sha256", os.path.join(e["dump_dir"], "waveforms.sha256"), e["sidecar_sha256"])]
        if rehash_npy:
            files.append(("waveforms.npy", os.path.join(e["dump_dir"], "waveforms.npy"), e["waveforms_sha256"]))
        if e["replay_verdict"] is not None:
            files.append(("replay verdict", e["replay_verdict"]["path"], e["replay_verdict"]["sha256"]))
        elif e["source"] == "exp36":
            r.append(f"{e['cell']}: an exp_36 cell without its replay verdict")
        for label, path, sha in files:
            if not os.path.isfile(path) or kit.sha256_file(path) != sha:
                r.append(f"{e['cell']}: {label} changed since registration")
    return r


# --------------------------------------------------------------------------- #
# collect, off-grid readout, tables, paper import, figure (plan §1, §5.4; code-r1 H12)
# --------------------------------------------------------------------------- #
def offgrid_readout(x):
    """``x[(arm, K, metric, theta)] -> value`` (seed 42). Per (K, metric): the descriptive
    ``d`` of C16, YAW and (when present) P1, the margin P = d_C16 - d_YAW, YAW's C4 spread S and
    the plan §1 label; PENDING if any C16 / YAW input is missing."""
    out = {}
    grid = (0.0, 90.0, 180.0, 270.0)

    def d_of(a, k, m, q):
        return sum(q * (x[(a, k, m, t)] - x[(a, k, m, t - 11.25)]) for t in OFF) / 4.0
    for k in KS:
        for m, q in ORIENT.items():
            need = [(a, k, m, t) for a in ("C16", "YAW") for t in OFF + grid]
            if any(key not in x for key in need):
                out[(k, m)] = {"label": "PENDING"}
                continue
            d = {a: d_of(a, k, m, q) for a in ("C16", "YAW")}
            p1 = d_of("P1", k, m, q) if all(("P1", k, m, t) in x for t in OFF + grid) else None
            P = d["C16"] - d["YAW"]
            S = max(x[("YAW", k, m, f)] for f in grid) - min(x[("YAW", k, m, f)] for f in grid)
            label = ("within YAW's C4 spread" if abs(P) <= S else
                     ("C16 worse off-grid" if P < 0 else "C16 better off-grid"))
            out[(k, m)] = {"d_C16": d["C16"], "d_YAW": d["YAW"], "d_P1": p1, "P": P, "S": S, "label": label}
    return out


def offgrid_from_rows(rows):
    x = {}
    for r in rows:
        if r["seed"] == 42 and r["rotate_mode"] == "fixed":
            for m in ORIENT:
                x[(r["arm"], r["K"], m, float(r["rotate_deg"]))] = r[m]
    return offgrid_readout(x)


CSV_FIELDS = ("cell", "arm", "block", "K", "seed", "rotate_mode", "rotate_deg", *METRIC_KEYS,
              "metrics_path", "metrics_sha256", "stream_sha256", "receipt_sha256", "source_sha", "origin")
PAPER_METRICS = ("T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5", "RIR_to_GT_RIR_R@10",
                 "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
PAPER_FIELDS = ("method", "K", "angle_deg", "seed", "source_json", *PAPER_METRICS)
PAPER_METHOD = {"C16": "YawAug C16@40k (exp_40)", "YAW": "YawAug YAW@40k", "P1": "Vanilla P1@40k"}


def _sort_key(r):
    return (r["arm"], r["K"], r["block"], -1.0 if r["rotate_deg"] is None else r["rotate_deg"], r["seed"], r["cell"])


def render_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in sorted(rows, key=_sort_key):
        w.writerow({k: ("" if r[k] is None else r[k]) for k in CSV_FIELDS})
    return buf.getvalue()


def parse_csv(text):
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        row = dict(r)
        row["K"], row["seed"] = int(row["K"]), int(row["seed"])
        row["rotate_deg"] = None if row["rotate_deg"] == "" else float(row["rotate_deg"])
        for m in METRIC_KEYS:
            row[m] = float(row[m])
        rows.append(row)
    return rows


def render_paper_import(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PAPER_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in sorted((r for r in rows if r["rotate_mode"] == "fixed"), key=_sort_key):
        deg = r["rotate_deg"]
        w.writerow({"method": PAPER_METHOD[r["arm"]], "K": r["K"],
                    "angle_deg": int(deg) if float(deg).is_integer() else deg, "seed": r["seed"],
                    "source_json": r["metrics_path"], **{m: r[m] for m in PAPER_METRICS}})
    return buf.getvalue()


def collect_rows(repo, tree, eval_sha, admission_path, nas_root=None, rehash_npy=True, sources=None,
                 registry_path=None):
    """Every exp_40 cell whose receipt re-verifies against EVAL_SHA, the closure, the EQUAL E0
    receipt and the validated admission + every registered comparator (re-hashed). Returns
    (rows, pending, provenance)."""
    r = tree_identity_reasons(tree, eval_sha)
    if r:
        raise Refused("; ".join(r[:5]))
    closure = closure_digest(tree, eval_sha)
    ref, ref_sha = e0_reference(tree)
    farm = os.path.join(repo, FARM)
    ident = identity(eval_sha, closure, ref_sha, expected_hashes(ref), farm, tree)
    er, e0_sha, e0 = e0_gate_reasons(e0_receipt_path(os.path.join(repo, EXPREL, "receipts"), eval_sha, closure), ident)
    if er:
        raise Refused("; ".join(er[:5]))
    ident["env_digest"], ident["e0_sha"] = e0["env_digest"], e0_sha
    ar, adm, adm_sha = kit.admission_reasons(admission_path, nas_root=nas_root or kit.NAS_ROOT, rehash=None)
    if ar:
        raise Refused("admission: " + "; ".join(ar[:4]))
    ident["admission"] = (adm, adm_sha)
    rows, pending = [], []
    for c in load_cells(tree):
        if c["block"] == "E0":
            continue
        if not os.path.lexists(cell_paths(c, farm, tree, ident["dump_root"])["receipt"]):
            pending.append(c["cell"])
            continue
        cr, rsha = full_cell_reasons(c, ident, ref)
        if cr:
            raise Refused(f"{c['cell']}: receipt does not re-verify: {cr[:3]}")
        rec = kit.read_json(cell_paths(c, farm, tree, ident["dump_root"])["receipt"])
        rows.append({"cell": c["cell"], "arm": c["arm"], "block": c["block"], "K": c["K"], "seed": c["seed"],
                     "rotate_mode": c["rotate_mode"], "rotate_deg": c["rotate_deg"], **rec["metrics"],
                     "metrics_path": cell_paths(c, farm, tree, ident["dump_root"])["metrics"],
                     "metrics_sha256": rec["metrics_sha256"], "stream_sha256": rec["stream_sha256"],
                     "receipt_sha256": rsha, "source_sha": eval_sha, "origin": "exp40"})
    reg_path = registry_path or os.path.join(tree, COMPARATORS_FILE)
    try:
        _, reg_sha, reg = _read(reg_path)
    except (OSError, ValueError) as err:
        raise Refused(f"{COMPARATORS_FILE} unreadable ({err})") from None
    rr = registry_reasons(reg, ref_sha, sources=sources, rehash_npy=rehash_npy)
    if rr:
        raise Refused("; ".join(rr[:5]))
    for e in reg["cells"]:
        _, msha, m = _read(e["metrics_path"])
        if msha != e["metrics_sha256"]:
            raise Refused(f"{e['cell']}: metrics changed since registration")
        rows.append({"cell": e["cell"], "arm": e["arm"], "block": e["block"], "K": e["K"], "seed": e["seed"],
                     "rotate_mode": e["rotate_mode"], "rotate_deg": e["rotate_deg"], **m["metrics"],
                     "metrics_path": e["metrics_path"], "metrics_sha256": msha, "stream_sha256": e["stream_sha256"],
                     "receipt_sha256": e["receipt_sha256"], "source_sha": e["source_sha"], "origin": e["source"]})
    prov = {"eval_sha": eval_sha, "closure": closure, "e0_receipt_sha256": e0_sha, "admission_sha256": adm_sha,
            "registry_sha256": reg_sha, "e0_reference_sha256": ref_sha,
            "cells_manifest_sha256": kit.sha256_file(os.path.join(tree, CELLS_FILE))}
    return rows, pending, prov


def _mean_sd(v):
    mu = sum(v) / len(v)
    return mu, ((sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan"))


# per (block-of-table, arm): the blocks a row may come from and the seeds the cell must have
TABLES = (
    ("theta = 0 (mean ± sample SD over eval seeds 42-46)", (0.0,),
     {"C16": (("T",), SEEDS), "YAW": (("T",), SEEDS), "P1": (("T",), SEEDS)}),
    ("C4 per angle (C16 / YAW: seeds 42-46; P1: seed 42 only)", tuple(float(d) for d in C4),
     {"C16": (("C4",), SEEDS), "YAW": (("C4",), SEEDS), "P1": (("C4",), (42,))}),
    ("45° grid (on the C16 grid; seed 42)", tuple(float(d) for d in G45),
     {"C16": (("G45",), (42,)), "YAW": (("G45",), (42,)), "P1": (("G45",), (42,))}),
    ("off-grid half-step coset (seed 42)", OFF,
     {"C16": (("OFF",), (42,)), "YAW": (("FILL",), (42,)), "P1": (("FILL",), (42,))}),
    ("random yaw (mean ± sample SD over eval seeds 42-46)", (None,),
     {"C16": (("R",), SEEDS), "YAW": (("R",), SEEDS), "P1": (("R",), SEEDS)}),
)


def table_lines(rows, title, angles, spec):
    lines = [f"## {title}", "", "| arm | K | angle | n | " + " | ".join(ORIENT) + " |",
             "|---|---|---|---|" + "---|" * len(ORIENT)]
    for arm, (blocks, seeds) in spec.items():
        for k in KS:
            for a in angles:
                sel = [r for r in rows if r["arm"] == arm and r["K"] == k and r["block"] in blocks
                       and r["rotate_deg"] == a]
                got = sorted(r["seed"] for r in sel)
                ang = "random" if a is None else (str(int(a)) if float(a).is_integer() else str(a))
                if got != list(seeds):
                    lines.append(f"| {arm} | {k} | {ang} | {len(sel)} | " + " | ".join("PENDING" for _ in ORIENT) + " |")
                    continue
                cells = []
                for m in ORIENT:
                    mu, sd = _mean_sd([r[m] for r in sel])
                    cells.append(f"{mu:.3f} ± {sd:.3f}" if len(sel) > 1 else f"{mu:.3f}")
                lines.append(f"| {arm} | {k} | {ang} | {len(sel)} | " + " | ".join(cells) + " |")
    return lines + [""]


def render_tables(rows, pending):
    lines = ["# exp_40 yaw_aug_c16 -- collected tables (generated by exp40_eval.py collect; never hand-edited)", ""]
    if pending:
        lines += [f"**PENDING** ({len(pending)} exp_40 cells without a verified receipt): " + ", ".join(pending[:12])
                  + (" ..." if len(pending) > 12 else ""), ""]
    for title, angles, spec in TABLES:
        lines += table_lines(rows, title, angles, spec)
    lines += ["## off-grid readout (plan §1; seed 42; descriptive compatibility margin; P1's d is descriptive only)",
              "", "| K | metric | d C16 | d YAW | d P1 | P | S (YAW C4 spread) | readout |",
              "|---|---|---|---|---|---|---|---|"]
    for (k, m), v in sorted(offgrid_from_rows(rows).items()):
        if v["label"] == "PENDING":
            lines.append(f"| {k} | {m} | | | | | | PENDING |")
        else:
            p1 = "PENDING" if v["d_P1"] is None else f"{v['d_P1']:.4f}"
            lines.append(f"| {k} | {m} | {v['d_C16']:.4f} | {v['d_YAW']:.4f} | {p1} | {v['P']:.4f} | "
                         f"{v['S']:.4f} | {v['label']} |")
    return "\n".join(lines) + "\n"


OUT_CSV = "yaw_aug_c16_cells_raw.csv"
OUT_TABLES = "yaw_aug_c16_results_tables.md"
OUT_PAPER = "yaw_aug_c16_paper_import.csv"
OUT_COLLECT = "yaw_aug_c16_collect_manifest.json"


def write_collect(rows, pending, prov, outdir):
    outs = {}
    for name, text in ((OUT_CSV, render_csv(rows)), (OUT_TABLES, render_tables(rows, pending)),
                       (OUT_PAPER, render_paper_import(rows))):
        path = os.path.join(outdir, name)
        with open(path, "w") as f:
            f.write(text)
        outs[name] = _sha(text.encode())
    kit.atomic_write_json(os.path.join(outdir, OUT_COLLECT),
                          {"schema_version": 1, "kind": "exp40_collect", "provenance": prov,
                           "outputs": outs, "rows": len(rows), "pending": pending})
    return [os.path.join(outdir, n) for n in (OUT_CSV, OUT_TABLES, OUT_PAPER, OUT_COLLECT)]


def figure_selection(rows, contract):
    """``(selected rows in contract order, reasons)``: every contract cell exactly once."""
    reasons, sel = [], []
    for e in contract["cells"]:
        key = (e["arm"], e["K"], e["seed"], e["rotate_mode"], e["rotate_deg"])
        hit = [r for r in rows if (r["arm"], r["K"], r["seed"], r["rotate_mode"], r["rotate_deg"]) == key]
        if not hit:
            reasons.append(f"figure cell missing: {key}")
        elif len(hit) > 1:
            reasons.append(f"duplicate cells for {key}")
        else:
            sel.append(hit[0])
    return sel, reasons


def figure_reasons(rows, contract):
    return figure_selection(rows, contract)[1]


def frozen_rows(rows, prov, outdir):
    """The frozen CSV, re-validated: its bytes == a fresh render of the re-collected rows and
    == the collect manifest's sha, the provenance unchanged. Returns (parsed rows, csv sha, manifest sha)."""
    path = os.path.join(outdir, OUT_CSV)
    try:
        with open(path, "rb") as f:
            frozen = f.read()
        _, cm_sha, cm = _read(os.path.join(outdir, OUT_COLLECT))
    except (OSError, ValueError) as err:
        raise Refused(f"no frozen collect output in {outdir} ({err}); run collect first") from None
    if frozen != render_csv(rows).encode():
        raise Refused(f"{OUT_CSV} is stale: it differs from the re-validated cells")
    if cm.get("outputs", {}).get(OUT_CSV) != _sha(frozen) or cm.get("provenance") != prov:
        raise Refused(f"{OUT_COLLECT} does not bind the frozen CSV / the current provenance")
    return parse_csv(frozen.decode()), _sha(frozen), cm_sha


def make_figure(rows, contract, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    angles = contract["angles"]
    metrics = contract["metrics"]
    fig, axes = plt.subplots(len(contract["Ks"]), len(metrics), figsize=(3.2 * len(metrics), 2.8 * len(contract["Ks"])),
                             squeeze=False)
    for i, k in enumerate(contract["Ks"]):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            for arm in contract["arms"]:
                ys = []
                for a in angles:
                    v = [r[m] for r in rows if r["arm"] == arm and r["K"] == k and r["seed"] == 42
                         and r["rotate_mode"] == "fixed" and float(r["rotate_deg"]) == float(a)]
                    ys.append(v[0] if v else float("nan"))
                ax.plot(angles, ys, marker="o", ms=3, label=arm)
            for g in range(0, 360, 90):
                ax.axvline(g, color="0.85", lw=0.6, zorder=0)
            ax.set_title(f"{m} (K={k})", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(contract["Ks"]) - 1:
                ax.set_xlabel("yaw (deg), seed 42", fontsize=7)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    paths = [os.path.join(outdir, f"yaw_aug_c16_figure.{ext}") for ext in ("png", "pdf")]
    for pth in paths:
        fig.savefig(pth, dpi=160)
    return paths


# --------------------------------------------------------------------------- #
# the offline table validator (plan §5.4; code-r1 B10, M7)
# --------------------------------------------------------------------------- #
EVIDENCE_KEYS = frozenset(("schema_version", "kind", "eval_sha", "closure", "admission", "e0_receipt",
                           "e0_cells", "ar_receipts", "cells"))


def freeze_table_evidence(repo, tree, eval_sha, admission_path, nas_root=None, rehash_npy=True, **kw):
    """After a full collect: the committed manifest binding the ten C16 theta = 0 cells'
    metrics / stream / receipt bytes, the admission, the E0 receipt, its three cell receipts and
    every AR re-hash receipt those receipts were run under (repo-relative paths; all must be
    committed with it) -- so the offline validator re-runs the SAME receipt / E0 / runtime-input
    validators (code-r2 B4)."""
    rows, pending, prov = collect_rows(repo, tree, eval_sha, admission_path, nas_root=nas_root,
                                       rehash_npy=rehash_npy, **kw)
    cells_all = load_cells(tree)
    t_cells = [c for c in cells_all if c["block"] == "T"]
    if any(c["cell"] in pending for c in t_cells):
        raise Refused("the C16 theta = 0 block is not complete")
    root = os.path.realpath(repo)

    def rel(p_):
        rp = os.path.relpath(os.path.realpath(p_), root)
        if rp.startswith(".."):
            raise Refused(f"{p_} lives outside the repo; the table evidence must be committable")
        return rp
    farm = os.path.join(repo, FARM)
    cells, e0_cells, ar = {}, {}, {}
    for c in cells_all:
        if c["block"] not in ("T", "E0"):
            continue
        p = cell_paths(c, farm, tree, DUMP_ROOT)
        rec = kit.read_json(p["receipt"])
        ent = rec["runtime_inputs"]["ar_verify"]
        ar[ent["sha256"]] = rel(ent["path"])
        if c["block"] == "T":
            cells[c["cell"]] = {k: {"path": rel(p[k]), "sha256": kit.sha256_file(p[k])}
                                for k in ("metrics", "stream", "receipt")}
        else:
            e0_cells[c["cell"]] = {"path": rel(p["receipt"]), "sha256": kit.sha256_file(p["receipt"])}
    e0p = e0_receipt_path(os.path.join(repo, EXPREL, "receipts"), eval_sha, prov["closure"])
    return {"schema_version": 2, "kind": "exp40_table_evidence", "eval_sha": eval_sha, "closure": prov["closure"],
            "admission": {"path": rel(admission_path), "sha256": prov["admission_sha256"]},
            "e0_receipt": {"path": rel(e0p), "sha256": prov["e0_receipt_sha256"]},
            "e0_cells": e0_cells, "ar_receipts": dict(sorted(ar.items())), "cells": cells}


def validate_row(files, repo, tree, evidence_path=None):
    """Offline validator for gen_model_comparison's contract ``exp40`` (plan §5.4): the files
    must be EXACTLY the five canonical C16 theta = 0 metrics files of one K (seeds 42-46); every
    byte they and their stream / receipt / admission / E0 receipt / E0 cell receipts / AR
    receipts carry must equal the committed table-evidence manifest; then the SAME validators as
    run / verify / collect: ``e0_gate_reasons`` (offline, from the evidence copies),
    ``receipt_reasons(offline=True)`` with the runtime inputs resolved through the evidence, and
    the metrics / stream contract. No checkpoint, NAS, farm-path or log access."""
    problems = []
    try:
        _, _, ev = _read(evidence_path or os.path.join(tree, TABLE_EVIDENCE))
    except (OSError, ValueError) as err:
        return False, [f"no committed table evidence {TABLE_EVIDENCE} ({type(err).__name__})"]
    if not isinstance(ev, dict) or set(ev) != EVIDENCE_KEYS or ev["kind"] != "exp40_table_evidence" \
            or ev["schema_version"] != 2:
        return False, [f"{TABLE_EVIDENCE} does not have the exact schema"]
    farm = os.path.join(repo, FARM)
    cells_all = load_cells(tree)
    t_cells = [c for c in cells_all if c["block"] == "T"]
    by_path = {os.path.realpath(cell_paths(c, farm, tree, DUMP_ROOT)["metrics"]): c for c in t_cells}
    got = [os.path.realpath(f) for f in files]
    cells = [by_path.get(g) for g in got]
    if any(c is None for c in cells):
        return False, [f"{files[i]}: not a canonical exp_40 C16 theta = 0 metrics file" for i, c in enumerate(cells) if c is None]
    ks = {c["K"] for c in cells}
    if len(set(got)) != len(got) or len(ks) != 1 or sorted(c["seed"] for c in cells) != list(SEEDS):
        return False, [f"need exactly the five seeds {SEEDS} of one K, got {sorted((c['K'], c['seed']) for c in cells)}"]
    if sorted(ev["cells"]) != sorted(c["cell"] for c in t_cells) or sorted(ev["e0_cells"]) != sorted(E0) \
            or not isinstance(ev["ar_receipts"], dict) or not ev["ar_receipts"]:
        return False, ["the table evidence does not name exactly the ten C16 theta = 0 cells, the three "
                       "E0 cells and their AR receipts"]
    at = lambda rel: os.path.join(repo, rel)  # noqa: E731
    pinned = [(label, ev[label]["path"], ev[label]["sha256"]) for label in ("admission", "e0_receipt")]
    pinned += [(f"E0 cell receipt {n}", e["path"], e["sha256"]) for n, e in sorted(ev["e0_cells"].items())]
    pinned += [("AR receipt", rel, sha) for sha, rel in sorted(ev["ar_receipts"].items())]
    for label, rel, sha in pinned:
        if not os.path.isfile(at(rel)) or kit.sha256_file(at(rel)) != sha:
            problems.append(f"{label} {rel} does not hash to the table evidence")
    if problems:
        return False, problems
    ar, adm, adm_sha = kit.admission_reasons(at(ev["admission"]["path"]), rehash=None)
    if ar or adm_sha != ev["admission"]["sha256"]:
        return False, [f"admission invalid: {ar[:2]}"]

    def resolve(name, ent):
        if name == "admission":
            return at(ev["admission"]["path"]) if ent["sha256"] == ev["admission"]["sha256"] else None
        rel = ev["ar_receipts"].get(ent["sha256"])
        return at(rel) if rel else None
    try:
        ref, ref_sha = e0_reference(tree)
    except Refused as err:
        return False, [str(err)]
    ident = identity(ev["eval_sha"], ev["closure"], ref_sha, expected_hashes(ref), farm, tree,
                     admission=(adm, adm_sha))
    e0_recs = {n: (kit.read_json(at(e["path"])), e["sha256"]) for n, e in ev["e0_cells"].items()}
    gate, e0_sha, e0 = e0_gate_reasons(at(ev["e0_receipt"]["path"]), ident, cell_recs=e0_recs, resolve=resolve)
    if gate:
        return False, [f"E0 receipt invalid: {g}" for g in gate[:4]]
    ident.update(env_digest=e0["env_digest"], e0_sha=e0_sha)
    for c in cells:
        p = cell_paths(c, farm, tree, DUMP_ROOT)
        e = ev["cells"][c["cell"]]
        for k in ("metrics", "stream", "receipt"):
            if os.path.realpath(at(e[k]["path"])) != os.path.realpath(p[k]) or not os.path.isfile(p[k]) \
                    or kit.sha256_file(p[k]) != e[k]["sha256"]:
                problems.append(f"{c['cell']}: {k} bytes / path differ from the table evidence")
        if problems:
            continue
        rec, m, st = kit.read_json(p["receipt"]), kit.read_json(p["metrics"]), kit.read_json(p["stream"])
        rr = receipt_reasons(rec, c, ident, offline=True, resolve=resolve)
        if rr:
            problems.append(f"{c['cell']}: receipt invalid: {rr[:3]}")
            continue
        if rec["metrics_sha256"] != e["metrics"]["sha256"] or rec["stream_sha256"] != e["stream"]["sha256"]:
            problems.append(f"{c['cell']}: the receipt is not bound to the table evidence")
            continue
        mr = metrics_reasons(m, c, ev["eval_sha"], rec["ckpt_sha256"], rec["ckpt_bytes"], rec["ckpt_link"], st)
        if mr or not isinstance(st, dict) or st.get("input_hash") != rec.get("input_hash") \
                or not _strict_equal(m.get("metrics"), rec.get("metrics")):
            problems.append(f"{c['cell']}: metrics / stream do not match the receipt ({mr[:2]})")
    return (not problems), problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _select(cells, blocks, only):
    if blocks:
        cells = [c for c in cells if c["block"] in blocks.split(",")]
    if only:
        names = only.split(",")
        unknown = sorted(set(names) - {c["cell"] for c in cells})
        if unknown:
            raise Refused(f"unknown / unselected cells {unknown}")
        cells = [c for c in cells if c["cell"] in names]
    if not cells:
        raise Refused("no cells selected")
    return cells


def _run(args):
    """Every gate, then (unless --dry-run) the cells. --dry-run skips ONLY the locks and the
    evaluator spawns: any failing gate is the same named refusal, exit 2 (code-r2 M2)."""
    wt, eval_sha = args.eval_worktree, args.eval_sha
    run_t0 = time.time()
    inputs, inputs_sha = runtime_inputs(args.runtime_inputs, tree=wt)
    reasons = eval_worktree_reasons(wt, eval_sha, args.hf_sealed) + ar_gate_reasons(wt, inputs)
    py = kit.PLATFORMS["a6000"]["python"]
    env_r, env = eval_environment(wt, py, args.hf_sealed)
    reasons += env_r
    if reasons:
        raise Refused("; ".join(reasons[:8]))
    closure = closure_digest(wt, eval_sha)
    ref, ref_sha = e0_reference(wt)
    cells = _select(load_cells(wt), args.blocks, args.only)
    farm = os.path.join(args.repo, FARM)
    receipts = os.path.join(args.repo, EXPREL, "receipts")
    ident = identity(eval_sha, closure, ref_sha, expected_hashes(ref), farm, wt,
                     env_digest=env["env_digest"], inputs_sha=inputs_sha, run_t0=run_t0)
    if any(c["arm"] == "C16" for c in cells):
        ident["admission"] = admission_of(inputs, rehash="final")
    e0_path = e0_receipt_path(receipts, eval_sha, closure)
    e0_cells = [c for c in cells if c["block"] == "E0"]
    rest = [c for c in cells if c["block"] != "E0"]
    if rest and not (sorted(c["cell"] for c in e0_cells) == sorted(E0)):
        gate, e0_sha, _ = e0_gate_reasons(e0_path, ident)
        if gate:
            raise Refused("; ".join(gate))
    if args.dry_run:
        for c in cells:
            print(c["cell"], " ".join(eval_argv(c, wt, farm_link(farm, c["arm"]), py, DUMP_ROOT)))
        print(f"DRY_RUN: EVAL_SHA {eval_sha} closure {closure} runtime inputs {inputs_sha[:12]}: every gate "
              f"passed for {len(cells)} cell(s); nothing locked, nothing spawned")
        return 0
    os.makedirs(receipts, exist_ok=True)
    children = Children()
    ctx = {"tree": wt, "farm": farm, "py": py, "hf_sealed": args.hf_sealed, "inputs": inputs,
           "inputs_sha": inputs_sha, "ident": ident, "ref": ref, "logdir": os.path.join(args.repo, EXPREL),
           "lockrec_dir": os.path.join(receipts, "eval_gpu_locks"), "lanes": [int(x) for x in args.lanes.split(",")],
           "dump_root": DUMP_ROOT, "query": kit.query_gpus, "sleep": time.sleep, "poll_s": 60.0,
           "max_wait_s": float(args.max_wait_s), "handshake_dir": kit.HANDSHAKE_DIR, "children": children,
           "spawn": default_spawn}
    cards = hold_cards_lock()
    old = {s: signal.signal(s, lambda *_: children.terminate()) for s in (signal.SIGINT, signal.SIGTERM)}
    fails = []
    try:
        if e0_cells:
            f0 = run_cells(e0_cells, ctx)
            if not children.interrupted:
                existing = e0_gate_reasons(e0_path, ident)
                if existing[0]:
                    shas, ids = {}, set()
                    for c in e0_cells:
                        rp = cell_paths(c, farm, wt, DUMP_ROOT)["receipt"]
                        if os.path.isfile(rp):
                            _, shas[c["cell"]], rec = _read(rp)
                            ids.add((rec.get("runtime_inputs_sha256"), rec.get("ar_receipt_sha256")))
                    if len(ids) > 1:                        # receipts from different runtime inputs
                        f0 = f0 + [("E0", [f"the E0 cell receipts carry {len(ids)} runtime-input identities"])]
                    verdict = "EQUAL" if not f0 and sorted(shas) == sorted(E0) else ("NOT_EQUAL" if f0 else "PARTIAL")
                    ri_sha, ar_sha = ids.pop() if len(ids) == 1 else (inputs_sha, inputs["ar_verify"]["sha256"])
                    kit.atomic_write_json(e0_path, e0_receipt_doc(verdict, ident, ref, ri_sha, ar_sha, shas, f0))
            fails += f0
        if rest and not children.interrupted:
            gate, e0_sha, _ = e0_gate_reasons(e0_path, ident)
            if gate:
                raise Refused("; ".join(gate))
            ident["e0_sha"] = e0_sha
            r = ar_gate_reasons(wt, inputs)                      # before every cell batch
            if r:
                raise Refused("; ".join(r))
            fails += run_cells(rest, ctx)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        os.close(cards)
    for name, rs in fails:
        print(f"FAILED {name}: {rs}")
    if children.interrupted:
        print("INTERRUPTED: owned evaluators terminated, locks released")
        return 1
    return 0 if not fails else 1


def _verify(args):
    wt, eval_sha = args.eval_worktree, args.eval_sha
    r = tree_identity_reasons(wt, eval_sha)
    if r:
        raise Refused("; ".join(r[:5]))
    closure = closure_digest(wt, eval_sha)
    ref, ref_sha = e0_reference(wt)
    cells = _select(load_cells(wt), args.blocks, args.only)
    farm = os.path.join(args.repo, FARM)
    ident = identity(eval_sha, closure, ref_sha, expected_hashes(ref), farm, wt)
    gate, e0_sha, e0 = e0_gate_reasons(e0_receipt_path(os.path.join(args.repo, EXPREL, "receipts"), eval_sha, closure), ident)
    bad = 0
    if gate:
        print(f"E0 RECEIPT INVALID: {gate}")
        return 1
    ident["env_digest"], ident["e0_sha"] = e0["env_digest"], e0_sha
    if any(c["arm"] == "C16" for c in cells):
        if not args.runtime_inputs:
            raise Refused("verifying C16 cells needs --runtime-inputs (the admission)")
        inputs, _ = runtime_inputs(args.runtime_inputs, tree=wt)
        ident["admission"] = admission_of(inputs, rehash=None)
    for c in cells:
        cr, _ = full_cell_reasons(c, ident, ref)
        print(("VERIFIED " if not cr else "INVALID ") + c["cell"] + ("" if not cr else f": {cr[:3]}"))
        bad += bool(cr)
    print(f"{len(cells) - bad}/{len(cells)} verified")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--write", action="store_true"); p.add_argument("--check", action="store_true")
    b = sub.add_parser("build-e0-reference"); b.add_argument("--out", required=True)
    for name in ("run", "verify"):
        r = sub.add_parser(name)
        r.add_argument("--repo", required=True, help="the main exp_40 worktree (farm, receipts, logs)")
        r.add_argument("--eval-worktree", required=True); r.add_argument("--eval-sha", required=True)
        r.add_argument("--runtime-inputs", required=(name == "run")); r.add_argument("--only"); r.add_argument("--blocks")
        if name == "run":
            r.add_argument("--hf-sealed", required=True); r.add_argument("--lanes", default="0")
            r.add_argument("--dry-run", action="store_true")
            r.add_argument("--max-wait-s", type=float, default=24 * 3600.0)
    g = sub.add_parser("register-comparators"); g.add_argument("--out", required=True)
    for name in ("collect", "figure", "freeze-table-evidence"):
        c = sub.add_parser(name); c.add_argument("--repo", required=True); c.add_argument("--tree", required=True)
        c.add_argument("--eval-sha", required=True); c.add_argument("--admission", required=True)
        if name == "freeze-table-evidence":
            c.add_argument("--out", required=True)
    w = sub.add_parser("validate-row"); w.add_argument("--repo", required=True); w.add_argument("--tree", required=True)
    w.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            with open(os.path.join(ROOT, EXP25_CELLS)) as f:
                man = build_cells(json.load(f))
            text = render(man)
            path = os.path.join(ROOT, CELLS_FILE)
            if args.check:
                with open(path) as f:
                    same = f.read() == text
                print("cells OK" if same else "cells DRIFTED")
                return 0 if same else 1
            if args.write:
                with open(path, "w") as f:
                    f.write(text)
            print(f"{man['count']} cells {man['block_counts']}")
            return 0
        if args.cmd == "build-e0-reference":
            if os.path.exists(args.out):
                raise Refused(f"{args.out} exists; the E0 reference is built once")
            ref = build_e0_reference()
            kit.atomic_write_json(args.out, ref)
            print(f"E0 reference -> {args.out} sha256 {kit.sha256_file(args.out)}")
            return 0
        if args.cmd == "register-comparators":
            if os.path.exists(args.out):
                raise Refused(f"{args.out} exists; the registry is written once")
            reg = register_comparators()
            kit.atomic_write_json(args.out, reg)
            print(f"registered {reg['count']} comparator cells -> {args.out}")
            return 0
        if args.cmd in ("collect", "figure", "freeze-table-evidence"):
            outdir = os.path.join(args.repo, EXPREL)
            if args.cmd == "freeze-table-evidence":
                doc = freeze_table_evidence(args.repo, args.tree, args.eval_sha, args.admission)
                kit.atomic_write_json(args.out, doc)
                print(f"table evidence -> {args.out}")
                return 0
            rows, pending, prov = collect_rows(args.repo, args.tree, args.eval_sha, args.admission)
            if args.cmd == "collect":
                print("\n".join(write_collect(rows, pending, prov, outdir)))
                return 0
            parsed, csv_sha, cm_sha = frozen_rows(rows, prov, outdir)
            contract_path = os.path.join(args.tree, FIGURE_CONTRACT)
            contract = kit.read_json(contract_path)
            sel, reasons = figure_selection(parsed, contract)
            if reasons:
                raise Refused("; ".join(reasons[:8]))
            paths = make_figure(sel, contract, outdir)
            kit.atomic_write_json(os.path.join(outdir, "yaw_aug_c16_figure_manifest.json"), {
                "schema_version": 2, "csv_sha256": csv_sha, "collect_manifest_sha256": cm_sha,
                "contract_sha256": kit.sha256_file(contract_path), "provenance": prov,
                "figures": {os.path.basename(pth): kit.sha256_file(pth) for pth in paths},
                "cells": [[r["arm"], r["K"], r["seed"], r["rotate_mode"], r["rotate_deg"], r["metrics_sha256"]]
                          for r in sel]})
            print("\n".join(paths))
            return 0
        if args.cmd == "validate-row":
            ok, problems = validate_row(args.files, args.repo, args.tree)
            for pr in problems:
                print(pr)
            return 0 if ok else 1
        return _run(args) if args.cmd == "run" else _verify(args)
    except Refused as err:
        print(f"REFUSED: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
