#!/usr/bin/env python3
"""exp_40 yaw_aug_c16 -- the evaluation driver (plan r2 §5).

Thin by design: it reuses the exp_25 YAW-pin protocol (each argv derived from the committed
exp_25 manifest row of the same kind), ``eval_FLAC.py``, the committed no-flash launcher,
a content-addressed checkpoint cache and the exp_36 replay semantics, without touching any
exp_25/35/36 file. Subcommands:

  plan [--write|--check]     the deterministic 85-cell list -> exp40_cells.json
  run --lanes 0[,1] [--only] from the detached eval worktree at EVAL_SHA (plan §14 H7): E0
                             first (writes the E0 receipt), then every other cell; refuses any
                             non-E0 cell without an EQUAL E0 receipt for EVAL_SHA + closure
  verify                     re-verify every receipt
  register-comparators       pin the reused exp_25 / exp_36 cells (exp40_comparators.json)
  collect | figure           validated CSV + tables + off-grid readout | the exp_40 figure
  validate-row               offline validator used by gen_model_comparison.py (contract exp40)

Exit: 0 PASS / all done; 1 a cell or verdict FAILED; 2 a gate REFUSED.
Written by the exp_40 Coder seat (Claude Opus 5.5) per plan_yaw_aug_c16.md r2 §5.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time

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
EXP25_CELLS_SHA256 = None            # pinned below by PINS (computed at plan time, see plan --check)
P1_CONFIG = "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARMS = {
    "C16": {"config": kit.ARM_CONFIG, "config_sha256": kit.PINS[kit.ARM_CONFIG]},
    "YAW": {"config": kit.BASE_CONFIG, "config_sha256": kit.PINS[kit.BASE_CONFIG],
            "ckpt": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/YAW/epoch=8-step=40000.ckpt",
            "ckpt_sha256": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7"},
    "P1": {"config": P1_CONFIG,
           "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
           "ckpt": "/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/P1/epoch=8-step=40000.ckpt",
           "ckpt_sha256": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c"},
}
ADMISSION = f"{EXPREL}/exp40_C16_admission.json"
CELLS_FILE = f"{EXPREL}/exp40_cells.json"
COMPARATORS_FILE = f"{EXPREL}/exp40_comparators.json"
FIGURE_CONTRACT = f"{EXPREL}/exp40_figure_contract.json"
SEEDS = (42, 43, 44, 45, 46)
KS = (1, 8)
C4 = (90, 180, 270)
G45 = (45, 135, 225, 315)
OFF = (11.25, 101.25, 191.25, 281.25)
FARM = "outputs_FLAC/exp40_yaw_aug_c16/farm"
DUMP_ROOT = "/media/diskstation/yixunhu/FLAC/exp40_pred_waveforms"
EVAL_PATHS = ("eval_FLAC.py", "src", "data/AR", "overlays", EXPREL, EXP25_CELLS, kit.BASE_CONFIG, P1_CONFIG)
EVAL_LINK_NAMES = ("AcousticRooms", "weights/FLAC/VAE.ckpt", "weights/AGREE/AGREE_fullAR.pt")
EVAL_PINS = {
    "weights/FLAC/VAE.ckpt": "6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf",
    "weights/AGREE/AGREE_fullAR.pt": "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787",
    "data/AR/unseen_eval.json": "9a9d817abc3e19f41351e07325ffa929c1f2846d0c77e80d538d9e4a21342ba8",
    "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json":
        "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3",
    "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json":
        "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55",
    "overlays/no_flash/flash_attn.py": "0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023",
    "overlays/no_flash/exp35_no_flash_launch.py": "bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de",
}
METRIC_KEYS = ("T60", "Invalid T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
               "RIR_to_GT_RIR_R@10", "RIR_to_geom_R@1", "RIR_to_geom_R@5", "RIR_to_geom_R@10")
SCENES = ("Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
          "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants")
# E0 references: the archived exp_25 YAW cells (paths + sha256 pinned; plan r2 §5.2)
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


def rot_token(deg):
    d = float(deg)
    return str(int(d)) if d.is_integer() else repr(d).replace(".", "p")


# --------------------------------------------------------------------------- #
# the cell list (plan r2 §5.2)
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


# --------------------------------------------------------------------------- #
# evaluation tree + closure digest (plan r2 §5.1)
# --------------------------------------------------------------------------- #
EVAL_UNTRACKED = {"AcousticRooms": "/media/diskstation/yixunhu/FLAC/AcousticRooms",
                  "weights/AGREE": None, "weights/FLAC/VAE.ckpt": None}     # allowlist (plan §14 N2)


def closure_digest(repo, eval_sha):
    """EVAL_SHA's evaluator closure (``git ls-tree`` of EVAL_PATHS) + the pinned assets: E0 is
    bound to it, so a later record-only commit elsewhere cannot stale E0 (plan §14 H8)."""
    blobs = kit.tracked_blobs(repo, eval_sha, EVAL_PATHS)
    return kit.digest_obj({"eval_sha": eval_sha, "blobs": blobs, "pins": EVAL_PINS})


def eval_worktree_reasons(wt, eval_sha, hf_sealed):
    """The detached eval worktree at EVAL_SHA, checked before EVERY spawn (plan §14 H7)."""
    reasons = []
    head = kit.git(wt, "rev-parse", "HEAD").strip()
    if head != eval_sha:
        reasons.append(f"eval worktree HEAD {head} != EVAL_SHA {eval_sha}")
    if kit.git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() != "HEAD":
        reasons.append("the eval worktree must be detached")
    status = kit.git(wt, "status", "--porcelain", "--untracked-files=all").splitlines()
    for line in status:
        code, path = line[:2], line[3:].rstrip("/")
        if code != "??" or not any(path == u or path.startswith(u + "/") for u in EVAL_UNTRACKED):
            reasons.append(f"eval worktree not clean: {line!r}")
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


def runtime_inputs(path):
    """The hash-pinned post-freeze inputs (plan §14 H7): ``{name: {path, sha256}}``."""
    spec = kit.read_json(path)
    out = {}
    for name, ent in spec.items():
        if kit.sha256_file(ent["path"]) != ent["sha256"]:
            raise Refused(f"runtime input {name} ({ent['path']}) does not hash to {ent['sha256']}")
        out[name] = ent
    return out, kit.sha256_file(path)


# --------------------------------------------------------------------------- #
# checkpoints: one read, embedded config type-strict equal to the arm's (plan r2 §5.1)
# --------------------------------------------------------------------------- #
def arm_checkpoint(arm, tree, inputs=None):
    if arm == "C16":
        if not inputs or "admission" not in inputs:
            raise Refused("the C16 arm needs the admission record in --runtime-inputs")
        adm = kit.read_json(inputs["admission"]["path"])
        if adm.get("admitted") is not True or adm.get("arm") != "C16":
            raise Refused(f"{ADMISSION} is not an admission of the C16 arm")
        return adm["ckpt_path"], adm["sha256"]
    return ARMS[arm]["ckpt"], ARMS[arm]["ckpt_sha256"]


def cache_checkpoint(arm, tree, farm, inputs=None):
    src, sha = arm_checkpoint(arm, tree, inputs)
    cache_dir = os.path.join(farm, ".ckpt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"{sha}.ckpt")
    link = os.path.join(farm, arm, "epoch=8-step=40000.ckpt")
    if not os.path.exists(cached):
        with open(src, "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != sha:
            raise Refused(f"{src} does not hash to the pinned {sha}")
        import torch
        ck = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        with open(os.path.join(tree, ARMS[arm]["config"])) as f:
            cfg = json.load(f)
        if kit.canonical(ck.get("model_config")) != kit.canonical(cfg) \
                or not _strict_equal(ck.get("model_config"), cfg):
            raise Refused(f"{arm}: the checkpoint's embedded model_config != {ARMS[arm]['config']}")
        tmp = f"{cached}.tmp-{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o444)
        kit.rename_noreplace(tmp, cached)
    st = os.stat(cached)
    if st.st_mode & 0o222 or kit.sha256_file(cached) != sha:
        raise Refused(f"cache entry {cached} is writable or does not re-hash to {sha}")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        if os.path.islink(link) or os.stat(link).st_ino != st.st_ino:
            raise Refused(f"stale farm entry {link}")
    else:
        os.link(cached, link)
    return link, sha, st.st_size


def _strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


# --------------------------------------------------------------------------- #
# per-cell verification (plan r2 §5.3)
# --------------------------------------------------------------------------- #
def stem_of(cell, tree):
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import eval_FLAC
    m = eval_FLAC.build_output_paths(
        "farm/x.ckpt", 1, 1.0, cell["cell"], cond_method="vanilla",
        rotate_deg=float(cell["rotate_deg"] or 0), n_angles=4, rotate_mode=cell["rotate_mode"],
        rotate_seed=cell["rotate_seed"], are_lambda=None, canonicalize=False, canon_mismatch=False)["metrics"]
    return "epoch=8-step=40000" + os.path.basename(m)[len("x"):-len(".json")]


def expected_hashes():
    """``{(K, seed): (input_hash, assignment_hash_random)}`` from the archived exp_25 YAW cells."""
    out = {}
    for k in KS:
        for s in SEEDS:
            t = kit.read_json(os.path.join(EXP25_FARM, f"epoch=8-step=40000_metrics_1_1.0_exp25_YAW_T_S40000_K{k}_s{s}.stream.json"))
            r = kit.read_json(os.path.join(EXP25_FARM, f"epoch=8-step=40000_metrics_1_1.0_exp25_YAW_R_S40000_K{k}_s{s}_rotrand{s}.stream.json"))
            out[(k, s)] = (t["input_hash"], r["assignment_hash"])
    return out


def verify_cell(cell, farm, tree, head, ckpt_sha, ckpt_bytes, log_path, hashes, dump_root=DUMP_ROOT):
    reasons = []
    stem = stem_of(cell, tree)
    mpath = os.path.join(farm, cell["arm"], stem + ".json")
    spath = os.path.join(farm, cell["arm"], stem + ".stream.json")
    try:
        with open(mpath, "rb") as f:
            mbytes = f.read()
        rec = json.loads(mbytes)
        with open(spath, "rb") as f:
            sbytes = f.read()
        stream = json.loads(sbytes)
    except (OSError, ValueError) as err:
        return [f"artifacts unreadable ({err})"], None
    want = {"cond_method": "vanilla", "cond_autocast": "bf16", "frame_avg_angles": None,
            "frame_avg_fwd_cap": None, "orbit_execution": "n/a", "batch_size": 64, "cfg_scale": 1.0,
            "steps": 1, "device": "cuda", "weights_source": "ema", "n_samples": 6337,
            "scene_count": 10, "per_scene_schema": 1, "seed": cell["seed"],
            "rotate_deg": cell["rotate_deg"], "dataset_config": cell["dataset_config"],
            "source_sha": head, "ckpt_sha256_loaded": ckpt_sha, "ckpt_bytes_loaded": ckpt_bytes,
            "eval_name": cell["cell"]}
    for key, val in want.items():
        if rec.get(key) != val or type(rec.get(key)) is not type(val):
            reasons.append(f"metrics {key} = {rec.get(key)!r}, expected {val!r}")
    m = rec.get("metrics", {})
    if tuple(sorted(m)) != tuple(sorted(METRIC_KEYS)) or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in m.values()):
        reasons.append(f"metric keys / finiteness wrong: {sorted(m)}")
    if sorted((rec.get("by_scene") or {}).keys()) != sorted(SCENES):
        reasons.append(f"by_scene keys {sorted((rec.get('by_scene') or {}).keys())}")
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import eval_FLAC
    for key, val in (("schema_version", 1), ("stream_count", 6337), ("img_w", 512),
                     ("rotate_mode", cell["rotate_mode"]), ("rotate_deg", cell["rotate_deg"]),
                     ("rotate_seed", cell["rotate_seed"])):
        if stream.get(key) != val:
            reasons.append(f"stream {key} = {stream.get(key)!r}, expected {val!r}")
    if eval_FLAC.canonical_stream_hash(stream.get("input_tuples", [])) != stream.get("input_hash") or \
            eval_FLAC.canonical_stream_hash(stream.get("assignment_tuples", [])) != stream.get("assignment_hash"):
        reasons.append("stream hashes do not recompute from their tuples")
    exp_in, exp_assign = hashes[(cell["K"], cell["seed"])]
    if stream.get("input_hash") != exp_in:
        reasons.append("input_hash != the exp_25 YAW cell's at the same (K, seed)")
    if cell["rotate_mode"] == "random":
        if stream.get("assignment_hash") != exp_assign:
            reasons.append("assignment_hash != the exp_25 YAW R cell's at the same (K, seed)")
    else:
        shift = round(float(cell["rotate_deg"]) * 512 / 360.0) % 512
        if set(stream.get("offsets", [])) != {shift}:
            reasons.append(f"fixed-mode offsets are not all {shift}")
    ddir = os.path.join(dump_root, stem)
    try:
        dman = kit.read_json(os.path.join(ddir, "manifest.json"))
        with open(os.path.join(ddir, "COMPLETE")) as f:
            complete = f.read().strip()
        with open(os.path.join(ddir, "waveforms.sha256")) as f:
            wsha = f.read().split()[0]
    except (OSError, ValueError, IndexError) as err:
        reasons.append(f"dump {ddir} incomplete ({err})")
    else:
        for key, val in (("n", 6337), ("shape", [6337, 10240]), ("dtype", "float32"),
                         ("input_hash", stream.get("input_hash")), ("metrics_stem", stem),
                         ("metrics_sha256", hashlib.sha256(mbytes).hexdigest()),
                         ("stream_sidecar_sha256", hashlib.sha256(sbytes).hexdigest()), ("sha256", wsha)):
            if dman.get(key) != val:
                reasons.append(f"dump manifest {key} = {dman.get(key)!r}, expected {val!r}")
        if wsha not in complete:
            reasons.append("dump COMPLETE marker does not name the waveform sha")
        if kit.sha256_file(os.path.join(ddir, "waveforms.npy")) != wsha:
            reasons.append("waveforms.npy does not re-hash to waveforms.sha256")
    try:
        with open(log_path, errors="replace") as f:
            sentinel = [ln for ln in f if ln.startswith("EXP35_NO_FLASH_LAUNCHER")]
    except OSError:
        sentinel = []
    if len(sentinel) != 1 or "sdpa" not in sentinel[0] or EVAL_PINS["overlays/no_flash/flash_attn.py"] not in sentinel[0]:
        reasons.append(f"no-flash sentinel lines {sentinel!r}")
    digests = {"metrics_sha256": hashlib.sha256(mbytes).hexdigest(),
               "stream_sha256": hashlib.sha256(sbytes).hexdigest(),
               "input_hash": stream.get("input_hash"), "assignment_hash": stream.get("assignment_hash"),
               "waveforms_sha256": locals().get("wsha"), "metrics": m, "stem": stem}
    return reasons, digests


def replay_reasons(cell, digests, farm):
    """E0: the new YAW cell vs the archived exp_25 cell (exp_36 replay semantics)."""
    ref_stem = stem_of({**cell, "cell": cell["replay_of"]}, ROOT)
    ref = kit.read_json(os.path.join(EXP25_FARM, ref_stem + ".json"))
    new = kit.read_json(os.path.join(farm, cell["arm"], digests["stem"] + ".json"))
    reasons = []
    for key in sorted(set(ref) | set(new)):
        if key in ("source_sha", "eval_name", "ckpt_path"):
            continue
        if not _strict_equal(ref.get(key), new.get(key)):
            reasons.append(f"E0 {cell['cell']}: {key} differs from {cell['replay_of']}")
    if kit.sha256_file(os.path.join(EXP25_FARM, ref_stem + ".stream.json")) != digests["stream_sha256"]:
        reasons.append(f"E0 {cell['cell']}: stream sidecar differs")
    with open(os.path.join(EXP25_DUMPS, ref_stem, "waveforms.sha256")) as f:
        if f.read().split()[0] != digests["waveforms_sha256"]:
            reasons.append(f"E0 {cell['cell']}: scored waveforms differ")
    return reasons


# --------------------------------------------------------------------------- #
# running (plan r2 §5.3)
# --------------------------------------------------------------------------- #
def lane_free(lane):
    gpus, apps = kit.query_gpus()
    g = {x["index"]: x for x in gpus}.get(lane)
    if g is None:
        raise Refused(f"GPU {lane} not visible")
    return not [a for a in apps if a["uuid"] == g["uuid"]] and g["free_mib"] >= MIN_GPU_FREE_MIB


def spawn_cell(cell, tree, link, lane, log_path, py, hf_sealed):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "HF_", "TRANSFORMERS_", "HUGGINGFACE_"))}
    env.update(CUDA_VISIBLE_DEVICES=str(lane), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTHONPATH=os.path.join(tree, "overlays/no_flash"), **kit.hf_env(hf_sealed))
    argv = [py, os.path.join(tree, "overlays/no_flash/exp35_no_flash_launch.py"),
            os.path.join(tree, "eval_FLAC.py"), *cell["eval_args"], "--ckpt-path", link,
            "--dump-scored-waveforms", DUMP_ROOT]
    with open(log_path, "w") as log:
        return subprocess.run(argv, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def receipt_path(farm, cell, stem):
    return os.path.join(farm, cell["arm"], stem + ".exp40_receipt.json")


def run_cells(cells, tree, head, lanes, farm, py, e0_receipt_path, closure, logdir, hf_sealed,
              inputs=None, inputs_sha=None, poll_s=60):
    hashes = expected_hashes()
    work = queue.Queue()
    for c in cells:
        work.put(c)
    failures, lock = [], threading.Lock()

    def worker(lane):
        while True:
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            stem = stem_of(cell, tree)
            rpath = receipt_path(farm, cell, stem)
            link, sha, nbytes = cache_checkpoint(cell["arm"], tree, farm, inputs)
            if os.path.exists(rpath):
                rec = kit.read_json(rpath)
                r, _ = verify_cell(cell, farm, tree, rec.get("head"), sha, nbytes, rec.get("log"), hashes)
                if not r and rec.get("closure") == closure:
                    print(f"SKIP {cell['cell']} (verified)", flush=True)
                    continue
                with lock:
                    failures.append((cell["cell"], ["existing receipt does not re-verify"] + r))
                continue
            if os.path.exists(os.path.join(farm, cell["arm"], stem + ".json")):
                with lock:
                    failures.append((cell["cell"], ["artifacts exist without a receipt: refusing to overwrite"]))
                continue
            while not lane_free(lane):
                time.sleep(poll_s)
            r = eval_worktree_reasons(tree, head, hf_sealed)        # re-checked before EVERY spawn
            if r:
                with lock:
                    failures.append((cell["cell"], ["eval worktree drifted"] + r))
                return
            log_path = os.path.join(logdir, f"yaw_aug_c16_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{cell['cell']}_eval.log")
            t0 = time.time()
            rc = spawn_cell(cell, tree, link, lane, log_path, py, hf_sealed)
            r, digests = verify_cell(cell, farm, tree, head, sha, nbytes, log_path, hashes) if rc == 0 else (
                [f"eval rc {rc}"], None)
            if not r and cell["block"] == "E0":
                r = replay_reasons(cell, digests, farm)
            if r:
                with lock:
                    failures.append((cell["cell"], r))
                print(f"FAIL {cell['cell']}: {r[:3]}", flush=True)
                continue
            kit.atomic_write_json(rpath, {"cell": cell, "head": head, "closure": closure,
                                          "runtime_inputs_sha256": inputs_sha,
                                          "lane": lane, "log": log_path, "seconds": time.time() - t0,
                                          "ckpt_sha256": sha, "ckpt_bytes": nbytes, **digests})
            print(f"DONE {cell['cell']} ({time.time() - t0:.0f} s)", flush=True)

    threads = [threading.Thread(target=worker, args=(lane,)) for lane in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failures


# --------------------------------------------------------------------------- #
# comparator registry, collect, off-grid readout, figure (plan r2 §5.2, §1, §5.4)
# --------------------------------------------------------------------------- #
def offgrid_readout(x):
    """``x[(arm, K, metric, theta)] -> value`` (seed 42). Returns per (K, metric) the d, P, S
    and the label of plan r2 §1, PENDING if any input cell is missing."""
    out = {}
    for k in KS:
        for m, q in ORIENT.items():
            need = [(a, k, m, t) for a in ("C16", "YAW") for t in OFF + (0.0, 90.0, 180.0, 270.0)]
            if any(key not in x for key in need):
                out[(k, m)] = {"label": "PENDING"}
                continue
            d = {a: sum(q * (x[(a, k, m, t)] - x[(a, k, m, t - 11.25)]) for t in OFF) / 4.0
                 for a in ("C16", "YAW")}
            P = d["C16"] - d["YAW"]
            S = max(x[("YAW", k, m, f)] for f in (0.0, 90.0, 180.0, 270.0)) - \
                min(x[("YAW", k, m, f)] for f in (0.0, 90.0, 180.0, 270.0))
            label = ("within YAW's C4 spread" if abs(P) <= S else
                     ("C16 worse off-grid" if P < 0 else "C16 better off-grid"))
            out[(k, m)] = {"d_C16": d["C16"], "d_YAW": d["YAW"], "P": P, "S": S, "label": label}
    return out


def register_comparators(exp25_farm, exp36_farm):
    """Pin the reused cells (plan r2 §5.2): exp_25 YAW/P1 T, R, C4+45 (pin d6ca307a) and
    exp_36 YAW/P1 135/225/315 + YAW C4 s43-46 -- EXACTLY these, paths from their receipts."""
    want = []
    for arm in ("YAW", "P1"):
        for k in KS:
            for s in SEEDS:
                want.append(("exp25", arm, f"exp25_{arm}_T_S40000_K{k}_s{s}", k, s, 0.0, "fixed"))
                want.append(("exp25", arm, f"exp25_{arm}_R_S40000_K{k}_s{s}", k, s, None, "random"))
            for d in C4 + (45,):
                want.append(("exp25", arm, f"exp25_{arm}_C4_S40000_K{k}_s42_th{d}", k, 42, float(d), "fixed"))
            for d in (135, 225, 315):
                name = f"exp25_YAW_C4_S40000_K{k}_s42_th{d}" if arm == "YAW" else f"exp35_P1_OFF_S40000_K{k}_s42_th{d}"
                want.append(("exp36", arm, name, k, 42, float(d), "fixed"))
    for k in KS:
        for s in (43, 44, 45, 46):
            for d in C4:
                want.append(("exp36", "YAW", f"exp25_YAW_C4_S40000_K{k}_s{s}_th{d}", k, s, float(d), "fixed"))
    pins = {"exp25": "d6ca307a85b9b0a75ec9d47f4d237e63a3db527e"}
    index = {}
    for src, root in (("exp25", exp25_farm), ("exp36", exp36_farm)):
        for arm in ("YAW", "P1"):
            d = os.path.join(root, arm)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if f.endswith(".cell.json"):
                    index.setdefault((src, arm, kit.read_json(os.path.join(d, f)).get("cell")), []).append(f)
    out = []
    for src, arm, name, k, s, deg, mode in want:
        farm = os.path.join(exp25_farm if src == "exp25" else exp36_farm, arm)
        cands = index.get((src, arm, name), [])
        if len(cands) != 1:
            raise Refused(f"{src} {arm} {name}: {len(cands)} receipts in {farm}")
        receipt = kit.read_json(os.path.join(farm, cands[0]))
        stem = receipt["stem"]
        mpath, spath = os.path.join(farm, stem + ".json"), os.path.join(farm, stem + ".stream.json")
        m, st = kit.read_json(mpath), kit.read_json(spath)
        if src == "exp25" and m.get("source_sha") != pins["exp25"]:
            raise Refused(f"{name}: source_sha {m.get('source_sha')} is not the exp_25 pin")
        if m.get("cond_method") != "vanilla" or m.get("seed") != s or m.get("rotate_deg") != deg \
                or m.get("ckpt_sha256_loaded") != ARMS[arm]["ckpt_sha256"] or m.get("n_samples") != 6337:
            raise Refused(f"{name}: protocol fields do not match the registry entry")
        if receipt.get("dump") is not True:
            raise Refused(f"{name}: receipt has no dump")
        ddir = os.path.join(receipt["dump_root"], stem)
        with open(os.path.join(ddir, "waveforms.sha256")) as f:
            wsha = f.read().split()[0]
        out.append({"source": src, "arm": arm, "cell": name, "K": k, "seed": s, "rotate_deg": deg,
                    "rotate_mode": mode, "metrics_path": mpath, "metrics_sha256": kit.sha256_file(mpath),
                    "stream_path": spath, "stream_sha256": kit.sha256_file(spath),
                    "receipt_path": os.path.join(farm, cands[0]),
                    "receipt_sha256": kit.sha256_file(os.path.join(farm, cands[0])),
                    "source_sha": m.get("source_sha"), "ckpt_sha256": m.get("ckpt_sha256_loaded"),
                    "input_hash": st.get("input_hash"), "assignment_hash": st.get("assignment_hash"),
                    "dump_dir": ddir, "waveforms_sha256": wsha,
                    "dump_manifest_sha256": kit.sha256_file(os.path.join(ddir, "manifest.json"))})
    srcs = {e["source_sha"] for e in out if e["source"] == "exp36"}
    if len(srcs) != 1:
        raise Refused(f"exp_36 cells come from {len(srcs)} evaluator pins {sorted(srcs)}: not the registered mix")
    return {"schema_version": 1, "count": len(out), "exp25_pin": pins["exp25"],
            "exp36_pin": srcs.pop(), "cells": out}


def registry_reasons(reg):
    reasons = []
    for e in reg["cells"]:
        for key in ("metrics", "stream", "receipt"):
            if kit.sha256_file(e[f"{key}_path"]) != e[f"{key}_sha256"]:
                reasons.append(f"{e['cell']}: {key} file changed since registration")
    return reasons


CSV_FIELDS = ("arm", "block", "K", "seed", "rotate_mode", "rotate_deg", *METRIC_KEYS,
              "metrics_sha256", "stream_sha256", "source_sha", "origin")


def collect_rows(repo, tree):
    """Every validated exp_40 cell (receipt re-verified) + every registered comparator."""
    man = kit.read_json(os.path.join(tree, CELLS_FILE))
    farm = os.path.join(repo, FARM)
    hashes = expected_hashes()
    rows, pending = [], []
    for c in man["cells"]:
        if c["block"] == "E0":
            continue
        stem = stem_of(c, tree)
        rp = receipt_path(farm, c, stem)
        if not os.path.exists(rp):
            pending.append(c["cell"])
            continue
        rec = kit.read_json(rp)
        r, d = verify_cell(c, farm, tree, rec["head"], rec["ckpt_sha256"], rec["ckpt_bytes"], rec["log"], hashes)
        if r:
            raise Refused(f"{c['cell']}: receipt does not re-verify: {r[:3]}")
        rows.append({"arm": c["arm"], "block": c["block"], "K": c["K"], "seed": c["seed"],
                     "rotate_mode": c["rotate_mode"], "rotate_deg": c["rotate_deg"], **d["metrics"],
                     "metrics_sha256": d["metrics_sha256"], "stream_sha256": d["stream_sha256"],
                     "source_sha": rec["head"], "origin": "exp40"})
    reg = kit.read_json(os.path.join(tree, COMPARATORS_FILE))
    rr = registry_reasons(reg)
    if rr:
        raise Refused("; ".join(rr[:5]))
    for e in reg["cells"]:
        m = kit.read_json(e["metrics_path"])["metrics"]
        block = {0.0: "T", None: "R"}.get(e["rotate_deg"], "C4" if e["rotate_deg"] in (90.0, 180.0, 270.0) else "G45")
        rows.append({"arm": e["arm"], "block": block, "K": e["K"], "seed": e["seed"],
                     "rotate_mode": e["rotate_mode"], "rotate_deg": e["rotate_deg"], **m,
                     "metrics_sha256": e["metrics_sha256"], "stream_sha256": e["stream_sha256"],
                     "source_sha": e["source_sha"], "origin": e["source"]})
    return rows, pending


def offgrid_from_rows(rows):
    x = {}
    for r in rows:
        if r["seed"] == 42 and r["rotate_mode"] == "fixed":
            for m in ORIENT:
                x[(r["arm"], r["K"], m, float(r["rotate_deg"]))] = r[m]
    return offgrid_readout(x)


def write_collect(rows, pending, outdir):
    csv_path = os.path.join(outdir, "yaw_aug_c16_cells_raw.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["arm"], r["K"], r["block"], r["rotate_deg"] or -1, r["seed"])):
            w.writerow(r)
    lines = ["# exp_40 yaw_aug_c16 -- collected tables (generated by exp40_eval.py collect; never hand-edited)", ""]
    if pending:
        lines += [f"**PENDING** ({len(pending)} exp_40 cells without a verified receipt): " + ", ".join(pending[:12])
                  + (" ..." if len(pending) > 12 else ""), ""]
    lines += ["## theta = 0 (mean ± sample SD over eval seeds 42-46)", "",
              "| arm | K | n | " + " | ".join(ORIENT) + " |", "|---|---|---|" + "---|" * len(ORIENT)]
    for arm in ("C16", "YAW", "P1"):
        for k in KS:
            sel = [r for r in rows if r["arm"] == arm and r["K"] == k and r["block"] == "T"]
            if len(sel) != 5:
                lines.append(f"| {arm} | {k} | {len(sel)} | " + " | ".join("PENDING" for _ in ORIENT) + " |")
                continue
            cells = []
            for m in ORIENT:
                v = [r[m] for r in sel]
                mu = sum(v) / 5
                sd = (sum((x - mu) ** 2 for x in v) / 4) ** 0.5
                cells.append(f"{mu:.3f} ± {sd:.3f}")
            lines.append(f"| {arm} | {k} | 5 | " + " | ".join(cells) + " |")
    lines += ["", "## off-grid readout (plan r2 §1; seed 42; descriptive compatibility margin)", "",
              "| K | metric | d C16 | d YAW | P | S (YAW C4 spread) | readout |", "|---|---|---|---|---|---|---|"]
    for (k, m), v in sorted(offgrid_from_rows(rows).items()):
        if v["label"] == "PENDING":
            lines.append(f"| {k} | {m} | | | | | PENDING |")
        else:
            lines.append(f"| {k} | {m} | {v['d_C16']:.4f} | {v['d_YAW']:.4f} | {v['P']:.4f} | {v['S']:.4f} | {v['label']} |")
    md_path = os.path.join(outdir, "yaw_aug_c16_results_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def figure_reasons(rows, contract):
    reasons = []
    have = {(r["arm"], r["K"], r["seed"], r["rotate_mode"], r["rotate_deg"]) for r in rows}
    for e in contract["cells"]:
        key = (e["arm"], e["K"], e["seed"], e["rotate_mode"], e["rotate_deg"])
        if key not in have:
            reasons.append(f"figure cell missing: {key}")
    allowed = {(e["arm"], e["K"], e["seed"], e["rotate_mode"], e["rotate_deg"]) for e in contract["cells"]}
    dup = [k for k in have if sum(1 for r in rows if (r["arm"], r["K"], r["seed"], r["rotate_mode"], r["rotate_deg"]) == k) > 1]
    if dup:
        reasons.append(f"duplicate cells {dup[:5]}")
    return reasons


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


def validate_row(files, repo, tree):
    """Offline validator for gen_model_comparison's contract ``exp40`` (plan r2 §5.4): each
    file must be a registered C16 theta=0 table cell whose committed receipt re-verifies."""
    man = kit.read_json(os.path.join(tree, CELLS_FILE))
    by_stem = {stem_of(c, tree): c for c in man["cells"] if c["block"] == "T"}
    problems = []
    for f in files:
        stem = os.path.basename(f)[:-len(".json")]
        c = by_stem.get(stem)
        if c is None:
            problems.append(f"{f}: not a registered exp_40 T cell")
            continue
        rp = receipt_path(os.path.join(repo, FARM), c, stem)
        if not os.path.exists(rp):
            problems.append(f"{f}: no exp_40 receipt")
            continue
        rec = kit.read_json(rp)
        if kit.sha256_file(f) != rec.get("metrics_sha256"):
            problems.append(f"{f}: metrics bytes differ from the receipt")
    return (not problems), problems


def e0_gate_reasons(e0_path, eval_sha, closure):
    try:
        e0 = kit.read_json(e0_path)
    except (OSError, ValueError):
        return [f"no E0 receipt {e0_path}: run --blocks E0 first"]
    reasons = []
    if e0.get("verdict") != "EQUAL" or e0.get("eval_sha") != eval_sha or e0.get("closure") != closure \
            or sorted(e0.get("cells", [])) != sorted(E0):
        reasons.append(f"{e0_path} is not an EQUAL E0 over all three replays for EVAL_SHA {eval_sha[:12]}"
                       f" / closure {closure[:12]}")
    return reasons


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--write", action="store_true"); p.add_argument("--check", action="store_true")
    for name in ("run", "verify"):
        r = sub.add_parser(name)
        r.add_argument("--repo", required=True, help="the main exp_40 worktree (farm, receipts)")
        r.add_argument("--eval-worktree", required=True); r.add_argument("--eval-sha", required=True)
        r.add_argument("--hf-sealed", required=True); r.add_argument("--runtime-inputs")
        r.add_argument("--lanes", default="0"); r.add_argument("--only"); r.add_argument("--blocks")
        r.add_argument("--dry-run", action="store_true")
    g = sub.add_parser("register-comparators"); g.add_argument("--exp25-farm", required=True)
    g.add_argument("--exp36-farm", required=True)
    for name in ("collect", "figure"):
        c = sub.add_parser(name); c.add_argument("--repo", required=True); c.add_argument("--tree", required=True)
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
        if args.cmd == "register-comparators":
            reg = register_comparators(args.exp25_farm, args.exp36_farm)
            out = os.path.join(ROOT, COMPARATORS_FILE)
            if os.path.exists(out):
                raise Refused(f"{out} exists; the registry is written once")
            kit.atomic_write_json(out, reg)
            print(f"registered {reg['count']} comparator cells -> {out}")
            return 0
        if args.cmd in ("collect", "figure"):
            rows, pending = collect_rows(args.repo, args.tree)
            outdir = os.path.join(args.repo, EXPREL)
            if args.cmd == "collect":
                print("\n".join(write_collect(rows, pending, outdir)))
                return 0
            contract = kit.read_json(os.path.join(args.tree, FIGURE_CONTRACT))
            reasons = figure_reasons(rows, contract)
            if reasons:
                raise Refused("; ".join(reasons[:8]))
            paths = make_figure(rows, contract, outdir)
            kit.atomic_write_json(os.path.join(outdir, "yaw_aug_c16_figure_manifest.json"),
                                  {"contract_sha256": kit.sha256_file(os.path.join(args.tree, FIGURE_CONTRACT)),
                                   "figures": {pth: kit.sha256_file(pth) for pth in paths},
                                   "cells": sorted((r["arm"], r["K"], r["seed"], r["rotate_mode"],
                                                    str(r["rotate_deg"]), r["metrics_sha256"]) for r in rows)})
            print("\n".join(paths))
            return 0
        if args.cmd == "validate-row":
            ok, problems = validate_row(args.files, args.repo, args.tree)
            for pr in problems:
                print(pr)
            return 0 if ok else 1
        # run / verify ------------------------------------------------------------------ #
        wt, eval_sha = args.eval_worktree, args.eval_sha
        reasons = eval_worktree_reasons(wt, eval_sha, args.hf_sealed)
        if reasons:
            raise Refused("; ".join(reasons[:8]))
        closure = closure_digest(wt, eval_sha)
        inputs, inputs_sha = runtime_inputs(args.runtime_inputs) if args.runtime_inputs else (None, None)
        man = kit.read_json(os.path.join(wt, CELLS_FILE))
        cells = man["cells"]
        if args.blocks:
            cells = [c for c in cells if c["block"] in args.blocks.split(",")]
        if args.only:
            cells = [c for c in cells if c["cell"] in args.only.split(",")]
        if not cells:
            raise Refused("no cells selected")
        farm = os.path.join(args.repo, FARM)
        receipts = os.path.join(args.repo, EXPREL, "receipts")
        os.makedirs(receipts, exist_ok=True)
        e0_path = os.path.join(receipts, f"e0_{eval_sha}_{closure}.json")
        e0_cells = [c for c in cells if c["block"] == "E0"]
        rest = [c for c in cells if c["block"] != "E0"]
        if rest:
            gate = [] if e0_cells and sorted(c["cell"] for c in e0_cells) == sorted(E0) else \
                e0_gate_reasons(e0_path, eval_sha, closure)
            if gate and not args.dry_run:
                raise Refused("; ".join(gate))
        if args.cmd == "verify" or args.dry_run:
            for c in cells:
                print(c["cell"], "eval_FLAC.py", " ".join(c["eval_args"]))
            print(f"EVAL_SHA {eval_sha} closure {closure} E0 gate: "
                  f"{'OK' if not rest or not e0_gate_reasons(e0_path, eval_sha, closure) else 'NOT YET'}")
            return 0
        py = kit.PLATFORMS["a6000"]["python"]
        lanes = [int(x) for x in args.lanes.split(",")]
        logdir = os.path.join(args.repo, EXPREL)
        fails = []
        if e0_cells:
            f0 = run_cells(e0_cells, wt, eval_sha, lanes, farm, py, e0_path, closure, logdir,
                           args.hf_sealed, inputs, inputs_sha)
            kit.atomic_write_json(e0_path, {"verdict": "EQUAL" if not f0 and sorted(c["cell"] for c in e0_cells) == sorted(E0)
                                            else ("MISMATCH" if f0 else "PARTIAL"),
                                            "eval_sha": eval_sha, "closure": closure,
                                            "cells": [c["cell"] for c in e0_cells], "failures": f0,
                                            "eval_pins": EVAL_PINS})
            fails += f0
        if rest:
            gate = e0_gate_reasons(e0_path, eval_sha, closure)
            if gate:
                raise Refused("; ".join(gate))
            fails += run_cells(rest, wt, eval_sha, lanes, farm, py, e0_path, closure, logdir,
                               args.hf_sealed, inputs, inputs_sha)
        for name, rs in fails:
            print(f"FAILED {name}: {rs}")
        return 0 if not fails else 1
    except Refused as err:
        print(f"REFUSED: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
