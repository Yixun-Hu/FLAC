#!/usr/bin/env python3
"""exp_40 yaw_aug_c16 -- the training kit's single source of truth (plan r1 §4).

The bash wrappers (``exp40_train_local.sh`` for the 2xA6000 box, ``exp40_submit_neuronic.sh``
+ ``exp40_train.sbatch`` for neuronic) only sequence these subcommands; every decision is
here and unit-tested (``src/tests/test_exp40_kit.py``).

  argv | parsed-argv             the literal train.py argv; its prefigure parse (in a tree)
  build-runtree | verify-runtree  the immutable run tree (git archive + allowlisted links)
  check                          every no-GPU gate against a run tree
  gpus                           GPU identity / occupancy (nvidia-smi, uuid-mapped, fail-closed)
  watch                          the start watcher (banner + ranks before the first step;
                                 terminates ONLY its own process group otherwise)
  evidence                       aggregate the rank-private evidence; write a smoke / pilot
                                 receipt, or the FULL run's evidence verdict
  publish-cadence                finished cadence checkpoints -> NAS (watch / pulled modes)
  audit | admit                  the full-state end audit; the admission record
  neuronic-submit                the Slurm submit-exactly-once state machine

Exit codes: 0 PASS; 1 a verdict FAILED; 2 a gate REFUSED; 8 / 9 / 10 watcher (see ``watch``).

Written by the exp_40 Coder seat (Claude Opus 5.5) per plan_yaw_aug_c16.md r1 §4.
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

# --------------------------------------------------------------------------- #
# constants (plan r1 §4)
# --------------------------------------------------------------------------- #
EXPREL = "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
ARM_CONFIG = f"{EXPREL}/FLAC_AR_YAWAUG_C16.json"
BASE_CONFIG = "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"
DATASET_CONFIG = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE = "weights/FLAC/VAE.safetensors"
FIXTURE = "src/tests/fixtures/exp40_yaw_aug_enabled_golden.json"
PINS = {
    ARM_CONFIG: "b497dfe937debde029057d7442863a9fb6ef87dd333d868f90f5bc311326f5c9",
    BASE_CONFIG: "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004",
    DATASET_CONFIG: "71f11e80b9b09db754e2ce42f517480a9fb85977f27990f7f01cdbdebfc9b242",
    "data/AR/train.json": "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03",
    VAE: "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
}
AR_SAMPLE_DIGEST = "a6ec819505de2bf1b196731ad1160aa59ff82735eb18d2ae95e944d48797d703"
DINO_REPO_DIR = "models--facebook--dinov3-vits16-pretrain-lvd1689m"
DINO_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
DINO_CONFIG_SHA256 = "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b"
AR_MANIFEST = f"{EXPREL}/exp40_ar_manifest.json"      # committed once the one-time manifest exists
HANDSHAKE_DIR = "/home/yixunhu/codespace/claude_gpu_yield"
RUNTREE_MANIFEST = ".exp40_runtree_manifest.json"
# What the run tree holds: the training closure + this folder (kit, driver, config) + the
# reference config + the E1 fixture (read by the evidence checks).
ARCHIVE_PATHS = ("train.py", "defaults.ini", "src", "data/AR", EXPREL, BASE_CONFIG)
CLOSURE = ("train.py", "defaults.ini", "src", "data/AR", BASE_CONFIG, ARM_CONFIG,
           f"{EXPREL}/exp40_kit.py", f"{EXPREL}/exp40_evidence_driver.py",
           f"{EXPREL}/exp40_train_local.sh", f"{EXPREL}/exp40_train.sbatch",
           f"{EXPREL}/exp40_submit_neuronic.sh")
LINKS = ("AcousticRooms", VAE)                     # the ONLY non-archive entries of a run tree

YAW_BLOCK = {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 32}
YAW_KWARGS = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
              "yaw_aug_offset_step": 32}
BANNER = "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"
RUNGS = {"2x32": (2, 32), "8x8": (8, 8)}
PLATFORMS = {
    "a6000": {"gpu_name": "NVIDIA RTX A6000", "rungs": ("2x32",),
              "python": "/home/yixunhu/miniconda3/envs/flac/bin/python",
              "hf_hub": "/home/yixunhu/.cache/huggingface/hub",
              "modes": ("SMOKE", "DDP_SMOKE", "FULL")},
    "neuronic": {"gpu_name": "NVIDIA L40", "rungs": ("2x32", "8x8"),
                 "python": "/n/fs/gatrdp/envs/flac/bin/python",
                 "hf_hub": "/n/fs/gatrdp/hf_cache/hub",
                 "modes": ("PILOT", "FULL")},
}
# Pinned on the A6000 box (read 2026-11-07); on neuronic these are RECORDED by the PILOT and
# FULL refuses any drift from the PILOT receipt's environment digest.
ENV_PINS_A6000 = {
    "python": "3.10.20", "torch": "2.7.0", "torchaudio": "2.7.0", "pytorch-lightning": "2.1.0",
    "transformers": "4.57.0", "flash-attn": "2.7.4.post1", "numpy": "1.23.5",
    "ema-pytorch": "0.2.3", "prefigure": "0.0.9", "safetensors": "0.7.0",
    "huggingface-hub": "0.35.3", "einops": "0.8.2", "wandb": "0.15.4", "torchvision": "0.22.0",
    "cuda_runtime": "12.6",
    "cudnn": 90501, "nccl": [2, 26, 2],
}
DRIVER_A6000 = "590.48.01"
SMOKE_STEPS = {"SMOKE": 20, "DDP_SMOKE": 30, "PILOT": 300}
PILOT_WINDOW = (100, 300)
ENDPOINT, CADENCE = 40000, 2500
CKPT_STEPS = tuple(range(CADENCE, ENDPOINT + 1, CADENCE))
PROJECT = "FLAC_exp40_C16"
NAS_ROOT = "/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16"
MIN_FREE_GIB = 30
SHARED_MIN_FREE_MIB = 24576
RUN_ID_RE = re.compile(r"^exp40_C16_(a6000|neuronic)_(2x32|8x8)(_r[1-9][0-9]?)?$")
FORBIDDEN_TRAINING_KEYS = ("cond_method", "canonicalize", "are_lambda", "are_anchor",
                           "frame_avg_max_fwd_samples", "frame_avg_angles")
ADAMW = {"lr": 5e-05, "betas": [0.9, 0.999], "weight_decay": 0.001}
SCHED = {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
         "base_lrs": [5e-05], "last_epoch": ENDPOINT, "_step_count": ENDPOINT + 1}
STEPS_PER_EPOCH = 4550          # 291,210 items / 64 (DistributedSampler, drop_last) -- exp_15/17 logs


class Refused(Exception):
    """A gate refused; the message names the reason."""


def _repo_root(p):
    """A git checkout (``.git``) or a run tree (``train.py`` + ``src`` + this folder)."""
    p = os.path.abspath(p)
    while True:
        if os.path.exists(os.path.join(p, ".git")) or (
                os.path.isfile(os.path.join(p, "train.py")) and os.path.isdir(os.path.join(p, "src"))):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo / run tree root not found")
        p = parent


ROOT = _repo_root(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest_obj(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def run(cmd, cwd=None, env=None, input_bytes=None):
    out = subprocess.run(cmd, cwd=cwd, env=env, input=input_bytes, capture_output=True)
    return out.returncode, out.stdout.decode(errors="replace"), out.stderr.decode(errors="replace")


def git(repo, *args):
    rc, out, err = run(["git", *args], cwd=repo)
    if rc != 0:
        raise Refused(f"git {' '.join(args)} failed in {repo}: {err.strip()}")
    return out


def atomic_write_json(path, payload):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def validate_identity(platform, rung, mode, run_id):
    reasons = []
    if platform not in PLATFORMS:
        return [f"platform {platform!r} is not one of {sorted(PLATFORMS)}"]
    if rung not in PLATFORMS[platform]["rungs"]:
        reasons.append(f"rung {rung!r} is not allowed on {platform} "
                       f"(allowed: {list(PLATFORMS[platform]['rungs'])})")
    if mode not in PLATFORMS[platform]["modes"]:
        reasons.append(f"mode {mode!r} is not one of {list(PLATFORMS[platform]['modes'])} on {platform}")
    m = RUN_ID_RE.match(run_id or "")
    if not m:
        reasons.append(f"run id {run_id!r} does not match {RUN_ID_RE.pattern}")
    elif (m.group(1), m.group(2)) != (platform, rung):
        reasons.append(f"run id {run_id!r} names {m.group(1)}/{m.group(2)}, the launch is "
                       f"{platform}/{rung}")
    return reasons


def is_smoke(mode):
    return mode in ("SMOKE", "DDP_SMOKE", "PILOT")


def world_of(mode, rung):
    return 1 if mode == "SMOKE" else RUNGS[rung][0]


def check_stamp(stamp):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}", stamp or ""):
        raise Refused(f"stamp must be YYYY-MM-DD_HH-MM-SS, got {stamp!r}")
    return stamp


def namespace(run_id, mode, stamp):
    """The experiment name (= W&B name = checkpoint subdirectory) of a launch."""
    return run_id if mode == "FULL" else f"{run_id}_{mode.lower()}_{check_stamp(stamp)}"


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #
def train_argv(platform, rung, mode, run_id, save_root, stamp=""):
    """The literal train.py argv (plan §4.1). FULL / DDP_SMOKE / PILOT share every recipe
    flag (``recipe_projection``); SMOKE is the one-rank preliminary (no SyncBN)."""
    reasons = validate_identity(platform, rung, mode, run_id)
    if reasons:
        raise Refused("; ".join(reasons))
    if not os.path.isabs(save_root):
        raise Refused(f"save root must be absolute, got {save_root!r}")
    ranks, micro = RUNGS[rung]
    full = mode == "FULL"
    argv = ["--model-config", ARM_CONFIG, "--dataset-config", DATASET_CONFIG,
            "--pretransform-ckpt-path", VAE,
            "--max-steps", str(ENDPOINT if full else SMOKE_STEPS[mode]),
            "--batch-size", str(micro), "--accum-batches", "1", "--num-workers", "6",
            "--seed", "42"]
    if mode == "SMOKE":
        argv += ["--num-gpus", "1", "--num-nodes", "1", "--strategy", "auto"]
    else:
        argv += ["--num-gpus", str(ranks), "--num-nodes", "1",
                 "--strategy", "ddp_find_unused_parameters_true", "--sync-batchnorm", "true"]
    argv += ["--precision", "bf16-mixed", "--val-every", "-1", "--val-dataset-config", "",
             "--gradient-clip-val", "0.0", "--logger", "wandb",
             "--checkpoint-every", str(CADENCE),
             "--name", PROJECT, "--experiment-name", namespace(run_id, mode, stamp),
             "--save-dir", save_root]
    return argv


VARIABLE_FLAGS = ("--max-steps", "--save-dir", "--experiment-name")     # plan §14 H5


def recipe_projection(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in VARIABLE_FLAGS:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def expected_parse(platform, rung, mode, run_id, save_root, stamp=""):
    """What train.py's prefigure parse of ``train_argv`` must yield, key by key, type-strict."""
    ranks, micro = RUNGS[rung]
    full = mode == "FULL"
    one = mode == "SMOKE"
    return {
        "config_file": "defaults.ini", "wandb_config": None, "name": PROJECT,
        "experiment_name": namespace(run_id, mode, stamp), "project": "FLAC",
        "batch_size": micro, "recover": "false", "save_top_k": -1, "num_nodes": 1,
        "num_gpus": 1 if one else ranks,
        "strategy": "auto" if one else "ddp_find_unused_parameters_true",
        "precision": "bf16-mixed", "sync_batchnorm": "false" if one else "true",
        "num_workers": 6, "seed": 42, "accum_batches": 1,
        "checkpoint_every": CADENCE, "val_every": -1,
        "max_steps": ENDPOINT if full else SMOKE_STEPS[mode], "ckpt_path": "",
        "pretrained_ckpt_path": "", "pretransform_ckpt_path": VAE, "model_config": ARM_CONFIG,
        "dataset_config": DATASET_CONFIG, "val_dataset_config": "", "save_dir": save_root,
        "gradient_clip_val": 0.0, "remove_pretransform_weight_norm": "", "logger": "wandb",
        "steps": 1.0, "cfg_scale": 1.0, "store_predictions": "false",
    }


_PARSE_SNIPPET = ("import json,sys;from prefigure.prefigure import get_all_args;"
                  "a=vars(get_all_args());print(json.dumps(a,sort_keys=True))")


def parsed_argv_reasons(tree, argv, expected, python):
    rc, out, err = run([python, "-I", "-c", f"import sys;sys.argv=['train.py']+{argv!r};" + _PARSE_SNIPPET],
                       cwd=tree)
    if rc != 0:
        return [f"prefigure could not parse the argv in {tree}: {err.strip()[-400:]}"]
    got = json.loads(out.strip().splitlines()[-1])
    reasons = []
    for key in sorted(set(got) | set(expected)):
        g, e = got.get(key, "<absent>"), expected.get(key, "<absent>")
        if g != e or type(g) is not type(e):
            reasons.append(f"parsed {key} = {g!r} ({type(g).__name__}), expected {e!r}")
    return reasons


# --------------------------------------------------------------------------- #
# run tree (plan §4.2)
# --------------------------------------------------------------------------- #
def tracked_blobs(repo, sha, paths=ARCHIVE_PATHS):
    """``{path: blob sha1}`` of the archive paths at ``sha`` (regular files only)."""
    out = git(repo, "ls-tree", "-r", "-z", sha, "--", *paths)
    blobs = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise Refused(f"unsupported tracked entry {mode} {kind} {path}")
        blobs[path] = blob
    return blobs


def build_runtree(repo, sha, dest, links, paths=ARCHIVE_PATHS, allowed_links=LINKS):
    """``git archive <sha> <ARCHIVE_PATHS>`` into a FRESH ``dest``, the allowlisted links, then
    read-only. ``links`` = {link path: absolute target}; exactly ``LINKS``."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Refused(f"run tree sha must be a full 40-hex commit, got {sha!r}")
    if set(links) != set(allowed_links):
        raise Refused(f"tree links must be exactly {list(allowed_links)}, got {sorted(links)}")
    if os.path.lexists(dest):
        raise Refused(f"run tree {dest} exists; a run tree is built once")
    for target in links.values():
        if not os.path.isabs(target) or not os.path.exists(target):
            raise Refused(f"link target {target!r} must be an existing absolute path")
    os.makedirs(dest)
    proc = subprocess.run(["git", "archive", "--format=tar", sha, "--", *paths],
                          cwd=repo, capture_output=True)
    if proc.returncode != 0:
        raise Refused(f"git archive failed: {proc.stderr.decode(errors='replace').strip()}")
    untar = subprocess.run(["tar", "-x", "-C", dest], input=proc.stdout, capture_output=True)
    if untar.returncode != 0:
        raise Refused(f"tar -x failed: {untar.stderr.decode(errors='replace').strip()}")
    for rel, target in sorted(links.items()):
        path = os.path.join(dest, rel)
        if os.path.lexists(path):
            raise Refused(f"link path {rel} already exists in the archive")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.symlink(target, path)
    atomic_write_json(os.path.join(dest, RUNTREE_MANIFEST),
                      {"sha": sha, "paths": list(paths), "blobs": tracked_blobs(repo, sha, paths),
                       "links": links})
    for dirpath, dirnames, filenames in os.walk(dest, topdown=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o444)
        os.chmod(dirpath, 0o555)
    return dest


def verify_runtree(repo, sha, tree, links, paths=ARCHIVE_PATHS, allowed_links=LINKS):
    """Every regular file == the commit's blob; the file set == the commit's archive paths
    plus exactly the allowlisted links with their recorded targets; nothing writable."""
    reasons = []
    try:
        carried = read_json(os.path.join(tree, RUNTREE_MANIFEST))
    except (OSError, ValueError) as err:
        return [f"the run tree carries no readable {RUNTREE_MANIFEST} ({err})"]
    want = carried.get("blobs", {})
    if carried.get("sha") != sha or carried.get("links") != links or carried.get("paths") != list(paths):
        reasons.append(f"carried manifest names sha {carried.get('sha')} / links {carried.get('links')}")
    if set(links) != set(allowed_links):
        reasons.append(f"links {sorted(links)} are not exactly {list(allowed_links)}")
    if repo is not None and want != tracked_blobs(repo, sha, paths):
        reasons.append("the carried manifest differs from git ls-tree of the commit")
    seen = {}
    for dirpath, dirnames, filenames in os.walk(tree):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, tree)
            if os.path.islink(p):
                if rel not in links:
                    reasons.append(f"unexpected symlink {rel}")
                elif os.readlink(p) != links[rel] or os.path.realpath(p) != os.path.realpath(links[rel]):
                    reasons.append(f"link {rel} -> {os.readlink(p)}, recorded {links[rel]}")
                if name in dirnames:
                    dirnames.remove(name)
                continue
            if os.path.isdir(p):
                if os.stat(p).st_mode & 0o222:
                    reasons.append(f"directory {rel} is writable")
                continue
            if rel == RUNTREE_MANIFEST:
                if os.stat(p).st_mode & 0o222:
                    reasons.append(f"{rel} is writable")
                continue
            if os.stat(p).st_mode & 0o222:
                reasons.append(f"{rel} is writable")
            seen[rel] = git_blob_sha1(p)
    for rel in sorted(set(want) - set(seen)):
        reasons.append(f"missing tracked file {rel}")
    for rel in sorted(set(seen) - set(want)):
        reasons.append(f"untracked file {rel} in the run tree")
    for rel in sorted(set(want) & set(seen)):
        if want[rel] != seen[rel]:
            reasons.append(f"{rel} differs from {sha[:12]}")
    for rel in links:
        if not os.path.islink(os.path.join(tree, rel)):
            reasons.append(f"allowlisted link {rel} missing")
    return reasons


def tree_digest(repo, sha, links, paths=ARCHIVE_PATHS):
    return digest_obj({"sha": sha, "blobs": tracked_blobs(repo, sha, paths), "links": links})


def ar_sample_digest(tree):
    """16 deterministic train items x {IR wav, depth npy, metadata json} (plan §4.2)."""
    with open(os.path.join(tree, "data/AR/train.json")) as f:
        split = json.load(f)
    items = sorted((t, r, x) for t in split for r in split[t] for x in split[t][r])
    h = hashlib.sha256()
    for t, r, x in (items[i * len(items) // 16] for i in range(16)):
        stem = x.split(".")[0]
        src, rec = int(stem.split("_")[0][1:]), int(stem.split("_")[1][1:])
        for rel in (f"single_channel_ir_1/{t}/{r}/{x}", f"depth_map/{t}/{r}/{rec}.npy",
                    f"metadata/{t}/{r}/S00{src}_R00{rec}.json"):
            h.update(f"{rel} {sha256_file(os.path.join(tree, 'AcousticRooms', rel))}\n".encode())
    return h.hexdigest()


AR_SUBDIRS = ("single_channel_ir_1", "depth_map", "metadata")


def split_rooms(split_json_path):
    with open(split_json_path) as f:
        split = json.load(f)
    return sorted((t, r) for t in split for r in split[t])


def room_manifest(ar_root, room_type, room, content):
    """Per room folder and sub-tree: file count, the (name, size) listing digest and -- with
    ``content`` -- the ordered content digest. Only regular files are allowed (AR_md lists
    the IR folder to build the context pool, so the listing itself is training input)."""
    out = {}
    for sub in AR_SUBDIRS:
        d = os.path.join(ar_root, sub, room_type, room)
        names = sorted(os.listdir(d))
        entries, h = [], hashlib.sha256()
        for n in names:
            p = os.path.join(d, n)
            st = os.lstat(p)
            if not os.path.isfile(p) or os.path.islink(p):
                raise Refused(f"{p} is not a regular file")
            entries.append([n, st.st_size])
            if content:
                h.update(f"{n} {sha256_file(p)}\n".encode())
        out[sub] = {"n": len(names), "listing": digest_obj(entries),
                    "content": h.hexdigest() if content else None}
    return out


def ar_manifest(ar_root, rooms, content, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {f"{t}/{r}": pool.submit(room_manifest, ar_root, t, r, content) for t, r in rooms}
        return {k: f.result() for k, f in sorted(futs.items())}


def ar_counts_reasons(ar_root, rooms, pinned):
    """Quick gate (plan §14 H2): every used room's IR-folder file count == the pinned manifest's."""
    reasons = []
    for t, r in rooms:
        key = f"{t}/{r}"
        ref = pinned.get(key)
        if ref is None:
            reasons.append(f"room {key} is not in the pinned AcousticRooms manifest")
            continue
        n = len(os.listdir(os.path.join(ar_root, "single_channel_ir_1", t, r)))
        if n != ref["single_channel_ir_1"]["n"]:
            reasons.append(f"AcousticRooms single_channel_ir_1/{key}: {n} files, pinned {ref['single_channel_ir_1']['n']}")
    return reasons


def ar_verify_reasons(receipt_path_, manifest, max_age_s=48 * 3600):
    """FULL / eval-freeze gate: a PASS full content re-hash of THIS manifest, <= 48 h old."""
    try:
        rec = read_json(receipt_path_)
    except (OSError, ValueError) as err:
        return [f"no readable full AcousticRooms re-hash receipt {receipt_path_} ({err}); run "
                "exp40_kit.py ar-verify first"]
    reasons = []
    if rec.get("verdict") != "PASS" or rec.get("manifest_digest") != manifest["digest"] \
            or rec.get("got_digest") != manifest["digest"]:
        reasons.append(f"AR re-hash receipt {receipt_path_} is not a PASS of manifest {manifest['digest'][:12]}")
    if time.time() - float(rec.get("t", 0)) > max_age_s:
        reasons.append(f"AR re-hash receipt {receipt_path_} is older than {max_age_s // 3600} h")
    return reasons


def ar_listing_reasons(ar_root, rooms, pinned, workers=8):
    """Per-launch re-check: every room's per-sub-tree file count and (name, size) listing
    equals the pinned full-content manifest's (plan §4.2)."""
    got = ar_manifest(ar_root, rooms, content=False, workers=workers)
    reasons = []
    for room, subs in got.items():
        ref = pinned.get(room)
        if ref is None:
            reasons.append(f"room {room} is not in the pinned AcousticRooms manifest")
            continue
        for sub, v in subs.items():
            if v["n"] != ref[sub]["n"] or v["listing"] != ref[sub]["listing"]:
                reasons.append(f"AcousticRooms {sub}/{room}: listing differs from the pinned manifest")
    return reasons


# --------------------------------------------------------------------------- #
# environment (plan §4.3-2), measured INSIDE the run tree's interpreter
# --------------------------------------------------------------------------- #
_ENV_SNIPPET = r"""
import importlib, importlib.metadata as md, json, os, sys
sys.path.insert(0, os.getcwd())
out = {"python": sys.version.split()[0], "executable": os.path.realpath(sys.executable),
       "prefix": os.path.realpath(sys.prefix), "dists": {}, "origins": {}}
for d in ("torch", "torchaudio", "pytorch-lightning", "transformers", "flash-attn", "numpy",
          "ema-pytorch", "prefigure", "safetensors", "huggingface-hub", "einops", "wandb",
          "torchvision"):
    try:
        out["dists"][d] = md.version(d)
    except Exception as e:
        out["dists"][d] = None
import torch
out["cuda_runtime"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()
try:
    out["nccl"] = list(torch.cuda.nccl.version())
except Exception:
    out["nccl"] = None
for m in ("torch", "pytorch_lightning", "transformers", "flash_attn", "numpy", "prefigure",
          "ema_pytorch", "safetensors"):
    try:
        out["origins"][m] = os.path.realpath(importlib.import_module(m).__file__)
    except Exception as e:
        out["origins"][m] = None
import src.models.transformer as T
out["transformer_module"] = os.path.realpath(T.__file__)
out["flash_attn_func_is_none"] = T.flash_attn_func is None
print(json.dumps(out, sort_keys=True))
"""


def measure_environment(tree, python, env):
    rc, out, err = run([python, "-c", _ENV_SNIPPET], cwd=tree, env=env)
    if rc != 0:
        raise Refused(f"environment probe failed in {tree}: {err.strip()[-600:]}")
    return json.loads(out.strip().splitlines()[-1])


def environment_reasons(platform, measured, tree, env):
    reasons = []
    for var in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONUSERBASE"):
        if env.get(var):
            reasons.append(f"{var} is set ({env[var]!r})")
    for var in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"):
        if env.get(var) != "1":
            reasons.append(f"{var}=1 is required")
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if env.get(var) != "1":
            reasons.append(f"{var}=1 is required")
    if measured["executable"] != os.path.realpath(PLATFORMS[platform]["python"]):
        reasons.append(f"interpreter {measured['executable']} is not the pinned "
                       f"{PLATFORMS[platform]['python']}")
    prefix = measured["prefix"] + os.sep
    for mod, origin in measured["origins"].items():
        if not origin or not origin.startswith(prefix):
            reasons.append(f"module {mod} resolves to {origin}, outside the env prefix {prefix}")
    if measured["transformer_module"] != os.path.realpath(os.path.join(tree, "src/models/transformer.py")):
        reasons.append(f"src.models.transformer resolves to {measured['transformer_module']}, "
                       "not the run tree's")
    if measured["flash_attn_func_is_none"]:
        reasons.append("src.models.transformer.flash_attn_func is None: the DiT would run SDPA, "
                       "not the reference arms' (inferred) flash-attn backend")
    if platform == "a6000":
        got = {"python": measured["python"], "cuda_runtime": measured["cuda_runtime"],
               "cudnn": measured["cudnn"], "nccl": measured["nccl"],
               **{k: (v.split("+")[0] if isinstance(v, str) else v)
                  for k, v in measured["dists"].items()}}
        for key, want in ENV_PINS_A6000.items():
            if got.get(key) != want:
                reasons.append(f"{key} {got.get(key)!r} != pinned {want!r}")
    return reasons


def environment_digest(measured, driver_version=None):
    keep = {k: measured[k] for k in ("python", "executable", "dists", "cuda_runtime", "cudnn",
                                     "nccl", "origins")}
    keep["driver"] = driver_version
    return digest_obj(keep)


# --------------------------------------------------------------------------- #
# pins, weights, config contract
# --------------------------------------------------------------------------- #
def pin_reasons(tree):
    reasons = []
    for rel, want in PINS.items():
        path = os.path.join(tree, rel)
        if not os.path.isfile(path):
            reasons.append(f"pinned file missing: {rel}")
        elif sha256_file(path) != want:
            reasons.append(f"{rel} sha256 != pinned {want}")
    return reasons


def dino_reasons(hub):
    snaps = os.path.join(hub, DINO_REPO_DIR, "snapshots")
    try:
        cands = sorted(os.listdir(snaps))
    except OSError:
        return [f"DINOv3 snapshots missing under {snaps}"]
    for c in cands:
        d = os.path.join(snaps, c)
        if sorted(os.listdir(d)) != ["config.json", "model.safetensors"]:
            continue
        if (sha256_file(os.path.join(d, "model.safetensors")) == DINO_SHA256
                and sha256_file(os.path.join(d, "config.json")) == DINO_CONFIG_SHA256):
            if len(cands) != 1:
                return [f"{len(cands)} DINOv3 snapshots under {snaps}; exactly one (the pinned) is "
                        "allowed so HF offline resolution cannot pick another"]
            return []
    return [f"no DINOv3 snapshot under {snaps} holds exactly config.json {DINO_CONFIG_SHA256} + "
            f"model.safetensors {DINO_SHA256}"]


DINO_REPO_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"


def build_hf_sealed(src_hub, dest):
    """A read-only HF hub holding exactly the pinned DINOv3 snapshot (plan §14 H3): refs/main
    + regular-file config.json / model.safetensors, verified against the pins."""
    if os.path.lexists(dest):
        raise Refused(f"{dest} exists; the sealed hub is built once")
    src = os.path.join(src_hub, DINO_REPO_DIR, "snapshots", DINO_REVISION)
    for name, want in (("config.json", DINO_CONFIG_SHA256), ("model.safetensors", DINO_SHA256)):
        if sha256_file(os.path.join(src, name)) != want:
            raise Refused(f"{src}/{name} is not the pinned {want}")
    root = os.path.join(dest, "hub", DINO_REPO_DIR)
    snap = os.path.join(root, "snapshots", DINO_REVISION)
    os.makedirs(snap)
    os.makedirs(os.path.join(root, "refs"))
    with open(os.path.join(root, "refs", "main"), "w") as f:
        f.write(DINO_REVISION)
    for name in ("config.json", "model.safetensors"):
        shutil.copyfile(os.path.join(src, name), os.path.join(snap, name))
    for dirpath, dirnames, filenames in os.walk(dest, topdown=False):
        for n in filenames:
            os.chmod(os.path.join(dirpath, n), 0o444)
        os.chmod(dirpath, 0o555)
    reasons = hf_sealed_reasons(dest)
    if reasons:
        raise Refused("; ".join(reasons))
    return dest


def hf_sealed_reasons(dest):
    reasons = []
    root = os.path.join(dest, "hub", DINO_REPO_DIR)
    want_files = {os.path.join("refs", "main"), os.path.join("snapshots", DINO_REVISION, "config.json"),
                  os.path.join("snapshots", DINO_REVISION, "model.safetensors")}
    got = set()
    for dirpath, dirnames, filenames in os.walk(os.path.join(dest, "hub")):
        for n in dirnames + filenames:
            p = os.path.join(dirpath, n)
            if os.path.islink(p):
                reasons.append(f"sealed hub holds a symlink {p}")
            if os.stat(p).st_mode & 0o222:
                reasons.append(f"sealed hub entry {p} is writable")
        for n in filenames:
            got.add(os.path.relpath(os.path.join(dirpath, n), root))
    if got != want_files:
        reasons.append(f"sealed hub files {sorted(got)} != {sorted(want_files)}")
        return reasons
    with open(os.path.join(root, "refs", "main")) as f:
        if f.read().strip() != DINO_REVISION:
            reasons.append("sealed hub refs/main is not the pinned revision")
    snap = os.path.join(root, "snapshots", DINO_REVISION)
    if sha256_file(os.path.join(snap, "config.json")) != DINO_CONFIG_SHA256:
        reasons.append("sealed config.json is not the pinned bytes")
    if sha256_file(os.path.join(snap, "model.safetensors")) != DINO_SHA256:
        reasons.append("sealed model.safetensors is not the pinned bytes")
    return reasons


def hf_env(sealed):
    return {"HF_HOME": sealed, "HF_HUB_CACHE": os.path.join(sealed, "hub"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"}


_HF_RESOLVE = ("import os,json;from huggingface_hub import try_to_load_from_cache as t;"
               f"print(json.dumps([t({DINO_REPO_ID!r},'model.safetensors'),t({DINO_REPO_ID!r},'config.json')]))")


def hf_resolution_reasons(sealed, python, env):
    e = {k: v for k, v in env.items() if not k.startswith(("HF_", "TRANSFORMERS_"))}
    e.update(hf_env(sealed))
    rc, out, err = run([python, "-c", _HF_RESOLVE], env=e)
    if rc != 0:
        return [f"HF resolution probe failed: {err.strip()[-300:]}"]
    paths = json.loads(out.strip().splitlines()[-1])
    hub = os.path.realpath(os.path.join(sealed, "hub")) + os.sep
    return [f"DINOv3 {p} resolves outside the sealed hub" for p in paths
            if not isinstance(p, str) or not os.path.realpath(p).startswith(hub)]


def config_contract_reasons(tree, python):
    """The arm config through the run tree's REAL factory parsers (plan §4.3-3)."""
    snippet = (
        "import json,os,sys;sys.path.insert(0,os.getcwd());"
        "from src.training import factory as F;"
        f"cfg=json.load(open({ARM_CONFIG!r}));t=cfg['training'];"
        "print(json.dumps({'yaw':F._parse_yaw_aug_config(t),'canon':F._parse_canonicalize_config(t),"
        "'cap':F._parse_frame_avg_cap_config(t),'are':F._parse_are_config(cfg,t)},sort_keys=True))")
    rc, out, err = run([python, "-c", snippet], cwd=tree)
    if rc != 0:
        return [f"the factory refuses the arm config: {err.strip()[-400:]}"]
    got = json.loads(out.strip().splitlines()[-1])
    reasons = []
    if got["yaw"] != YAW_KWARGS:
        reasons.append(f"factory yaw_aug kwargs {got['yaw']} != {YAW_KWARGS}")
    for k in ("canon", "cap", "are"):
        if got[k] != {}:
            reasons.append(f"factory {k} kwargs {got[k]} not empty")
    with open(os.path.join(tree, ARM_CONFIG)) as f:
        cfg = json.load(f)
    t = cfg["training"]
    if canonical(t.get("yaw_aug")) != canonical(YAW_BLOCK):
        reasons.append(f"training.yaw_aug is {t.get('yaw_aug')!r}")
    for key in FORBIDDEN_TRAINING_KEYS:
        if key in t:
            reasons.append(f"training.{key} must be absent")
    if t.get("use_ema") is not True:
        reasons.append("training.use_ema must be true")
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c.get("type") == "ViTCoordinates"]
    if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
        reasons.append("expected exactly the source_vit and context_poses_vit conditioners")
    for c in vits:
        if c["config"].get("gradient_checkpointing") is not True:
            reasons.append(f"{c['id']}: gradient_checkpointing must be true")
        if c["config"]["ViT"].get("img_w") != 512 or c["config"]["ViT"].get("img_h") != 256:
            reasons.append(f"{c['id']}: ViT must be 256x512")
    return reasons


def disk_reasons(path, min_gib):
    os.makedirs(path, exist_ok=True)
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize / 2 ** 30
    return [] if free >= min_gib else [f"only {free:.1f} GiB free under {path} (< {min_gib})"]


# --------------------------------------------------------------------------- #
# receipts (plan §4.3-7)
# --------------------------------------------------------------------------- #
def receipt_name(kind, platform, rung, sha):
    return f"{kind}_{platform}_{rung}_{sha}.json"


def receipt_reasons(rec, kind, platform, rung, sha, tree_dig, env_dig, full_projection):
    reasons = []
    want = {"verdict": "PASS", "kind": kind, "platform": platform, "rung": rung, "sha": sha,
            "tree_digest": tree_dig, "env_digest": env_dig, "recipe_projection": full_projection}
    for key, val in want.items():
        if rec.get(key) != val:
            reasons.append(f"{kind} receipt field {key} = {rec.get(key)!r}, need {val!r}")
    ev = rec.get("evidence", {})
    if ev.get("verdict") != "PASS":
        reasons.append(f"{kind} receipt evidence verdict {ev.get('verdict')!r}")
    for key in ("train_rc", "watcher_rc"):
        if rec.get(key) != 0:
            reasons.append(f"{kind} receipt {key} = {rec.get(key)!r}, need 0")
    return reasons


# --------------------------------------------------------------------------- #
# check: every no-GPU gate (plan §4.3)
# --------------------------------------------------------------------------- #
def check(platform, rung, mode, run_id, sha, repo, tree, links, save_root, stamp,
          receipts_dir, hf_sealed, env=None, nas_root=NAS_ROOT, in_job=False):
    """``in_job`` (the neuronic sbatch): the source checkout may have moved since submission,
    so HEAD equality and the clean-closure check were made by the submitter; the job re-checks
    the immutable run tree against the carried manifest and ``git ls-tree <sha>``."""
    env = dict(os.environ) if env is None else env
    reasons = validate_identity(platform, rung, mode, run_id)
    if reasons:
        return reasons, {}
    info = {"sha": sha}
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        return [f"EXPECT_SHA must be a full 40-hex commit, got {sha!r}"], info
    if not in_job:
        head = git(repo, "rev-parse", "HEAD").strip()
        if head != sha:
            reasons.append(f"source checkout HEAD {head} != EXPECT_SHA {sha}")
        dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", *CLOSURE).strip()
        if dirty:
            reasons.append("the training closure is not committed (or holds untracked files): " + dirty)
    if reasons:
        return reasons, info
    reasons += verify_runtree(repo, sha, tree, links)
    if reasons:
        return reasons, info
    info["tree_digest"] = tree_digest(repo, sha, links)
    reasons += pin_reasons(tree)
    if ar_sample_digest(tree) != AR_SAMPLE_DIGEST:
        reasons.append("the AcousticRooms sample digest differs from the pinned "
                       f"{AR_SAMPLE_DIGEST}: wrong or modified dataset behind the link")
    if mode != "SMOKE":
        man_path = os.path.join(tree, AR_MANIFEST)
        if not os.path.isfile(man_path):
            reasons.append(f"{AR_MANIFEST} is not committed: the full AcousticRooms manifest is "
                           "required for DDP_SMOKE / PILOT / FULL")
        else:
            man = read_json(man_path)
            if digest_obj(man["rooms"]) != man["digest"]:
                reasons.append(f"{AR_MANIFEST} is internally inconsistent")
            else:
                reasons += ar_counts_reasons(os.path.join(tree, "AcousticRooms"),
                                             split_rooms(os.path.join(tree, "data/AR/train.json")),
                                             man["rooms"])
                if mode == "FULL":
                    reasons += ar_verify_reasons(os.path.join(receipts_dir, f"ar_verify_{platform}.json"), man)
    reasons += hf_sealed_reasons(hf_sealed)
    reasons += hf_resolution_reasons(hf_sealed, PLATFORMS[platform]["python"], env)
    for k, v in hf_env(hf_sealed).items():
        if env.get(k) != v:
            reasons.append(f"{k} must be {v!r} (the sealed hub), got {env.get(k)!r}")
    measured = measure_environment(tree, PLATFORMS[platform]["python"], env)
    info["environment"] = measured
    reasons += environment_reasons(platform, measured, tree, env)
    info["env_digest"] = environment_digest(measured, env.get("EXP40_DRIVER_VERSION"))
    reasons += config_contract_reasons(tree, PLATFORMS[platform]["python"])
    argv = train_argv(platform, rung, mode, run_id, save_root, stamp)
    info["argv"] = argv
    reasons += parsed_argv_reasons(tree, argv, expected_parse(platform, rung, mode, run_id,
                                                              save_root, stamp),
                                   PLATFORMS[platform]["python"])
    ns_dir = os.path.join(save_root, PROJECT, namespace(run_id, mode, stamp))
    if os.path.lexists(ns_dir):
        reasons.append(f"run namespace {ns_dir} exists: every launch starts fresh (plan D3)")
    reasons += disk_reasons(save_root, MIN_FREE_GIB if mode == "FULL" else 5)
    if mode == "FULL":
        if platform == "a6000" and os.path.lexists(os.path.join(nas_root, run_id)):
            reasons.append(f"NAS cadence directory {os.path.join(nas_root, run_id)} exists")
        kind = "ddp_smoke" if platform == "a6000" else "pilot"
        path = os.path.join(receipts_dir, receipt_name(kind, platform, rung, sha))
        try:
            rec = read_json(path)
        except (OSError, ValueError) as err:
            reasons.append(f"no readable {kind} receipt {path} ({err})")
        else:
            reasons += receipt_reasons(rec, kind, platform, rung, sha, info["tree_digest"],
                                       info["env_digest"], recipe_projection(argv))
            info["receipt_sha256"] = sha256_file(path)
    return reasons, info


# --------------------------------------------------------------------------- #
# gpus (plan §4.3-5)
# --------------------------------------------------------------------------- #
def parse_gpu_query(text):
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit() or not parts[3].isdigit():
            raise Refused(f"unparseable nvidia-smi GPU row {line!r}")
        rows.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                     "free_mib": int(parts[3])})
    if not rows:
        raise Refused("nvidia-smi listed no GPU")
    return rows


def parse_apps_query(text):
    apps = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[1].isdigit():
            raise Refused(f"unparseable nvidia-smi compute-app row {line!r}")
        apps.append({"uuid": parts[0], "pid": int(parts[1]), "used": parts[2]})
    return apps


def gpu_reasons(platform, mode, rung, lanes, policy, gpus, apps, expect_total):
    reasons = []
    if policy not in ("idle", "shared"):
        return [f"GPU_POLICY must be idle or shared, got {policy!r}"]
    if mode == "FULL" and policy != "idle":
        return ["a primary FULL run is idle-only (GPU_POLICY=idle); shared cards are a "
                "different resource condition (plan D3 / Codex r1 HIGH 6)"]
    name = PLATFORMS[platform]["gpu_name"]
    if len(gpus) != expect_total:
        reasons.append(f"{len(gpus)} GPUs visible, expected exactly {expect_total}")
    by_index = {g["index"]: g for g in gpus}
    if len(set(lanes)) != len(lanes) or len(lanes) != world_of(mode, rung):
        reasons.append(f"lanes {lanes} must be {world_of(mode, rung)} distinct GPU(s) for "
                       f"{mode} on {rung}")
    for lane in lanes:
        g = by_index.get(lane)
        if g is None:
            reasons.append(f"GPU {lane} is not visible")
            continue
        if g["name"] != name:
            reasons.append(f"GPU {lane} is {g['name']!r}, not {name!r}")
        users = [a["pid"] for a in apps if a["uuid"] == g["uuid"]]
        if policy == "idle" and users:
            reasons.append(f"GPU {lane} holds compute process(es) {users} (GPU_POLICY=idle)")
        if policy == "shared" and g["free_mib"] < SHARED_MIN_FREE_MIB:
            reasons.append(f"GPU {lane} has {g['free_mib']} MiB free < {SHARED_MIN_FREE_MIB}")
    known = {g["uuid"] for g in gpus}
    stray = [a for a in apps if a["uuid"] not in known]
    if stray:
        reasons.append(f"compute processes on unknown GPU uuids {stray}: query inconsistent")
    return reasons


def query_gpus():
    rc1, o1, e1 = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
                       "--format=csv,noheader,nounits"])
    rc2, o2, e2 = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                       "--format=csv,noheader,nounits"])
    if rc1 or rc2:
        raise Refused(f"nvidia-smi failed (rc {rc1}/{rc2}): {e1.strip()} {e2.strip()}")
    return parse_gpu_query(o1), parse_apps_query(o2)


# --------------------------------------------------------------------------- #
# watch (plan §4.4)
# --------------------------------------------------------------------------- #
_STEP_RE = re.compile(r"Epoch \d+:\s+\d+%\|[^|]*\|\s*([0-9]+)/")
_MEMBER_RE = re.compile(r"Initializing distributed: GLOBAL_RANK: (\d+), MEMBER: (\d+)/(\d+)")


def start_gate(text, members, banner=BANNER):
    """``("pending"|"pass"|"fail", why)``: PASS once the exact banner and (members > 1) every
    MEMBER k/members line precede the first progress token with a completed step."""
    banner_seen, ranks = False, set()
    for raw in re.split(r"[\r\n]", text):
        line = raw.strip()
        if line.startswith("yaw_aug ENABLED"):
            if line != banner:
                return "fail", f"wrong treatment banner {line!r} (expected {banner!r})"
            banner_seen = True
        mm = _MEMBER_RE.search(line)
        if mm:
            if int(mm.group(3)) != members:
                return "fail", f"world size {mm.group(3)} != {members}"
            ranks.add(int(mm.group(1)))
        for sm in _STEP_RE.finditer(raw):
            if int(sm.group(1)) >= 1:
                if not banner_seen:
                    return "fail", "a training step completed before the C16 banner"
                if members > 1 and ranks != set(range(members)):
                    return "fail", f"a step completed with ranks {sorted(ranks)} registered"
                return "pass", "banner and ranks before the first step"
    return "pending", ""


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch(log, members, pid, pgid, timeout_s, poll_s=5.0):
    """0 PASS; 8 FAIL / 9 timeout (after SIGTERM to OUR process group ``pgid``); 10 the
    training process ended first. ``pgid`` must be the session leader the launcher created
    with ``setsid`` -- never this watcher's own group."""
    if pgid == os.getpgrp():
        raise Refused("refusing to watch a process group that contains the watcher itself")
    t0 = time.time()
    while True:
        try:
            with open(log, errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        state, why = start_gate(text, members)
        if state == "pass":
            print(f"START GATE PASS: {why}", flush=True)
            return 0
        timed_out = time.time() - t0 > timeout_s
        if state == "fail" or timed_out:
            why = why or f"no decision within {timeout_s} s"
            print(f"START GATE FAIL: {why} -- terminating process group {pgid}", flush=True)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return 8 if state == "fail" else 9
        if not _alive(pid):
            print("START GATE: the training process exited before the gate decided", flush=True)
            return 10
        time.sleep(poll_s)


# --------------------------------------------------------------------------- #
# evidence (plan §4.7, §4.3-7)
# --------------------------------------------------------------------------- #
_STREAM_SNIPPET = r"""
import hashlib, json, os, sys
sys.path.insert(0, os.getcwd())
import torch
import src.training.diffusion as D
from src.data import yaw_rotation as Y
steps, ranks, micro, sampled, verbatim = json.loads(sys.argv[1])
out = {}
for r in ranks:
    h, hist, samp, verb = hashlib.sha256(), [0] * 16, {}, []
    for s in range(steps):
        g = torch.Generator(); g.manual_seed(D._yaw_aug_step_seed(42, s, r))
        o = Y.quantize_yaw_offsets(Y.draw_yaw_offsets(micro, 512, g), 512, 32).tolist()
        h.update(f"{s}:{','.join(map(str, o))}\n".encode())
        for d in o:
            hist[d // 32] += 1
        if s in sampled:
            samp[str(s)] = o
        if verbatim:
            verb.append(o)
    out[str(r)] = {"digest": h.hexdigest(), "hist": hist, "n": steps * micro,
                   "sampled": samp, "verbatim": verb}
print(json.dumps(out))
"""
SAMPLED_STEPS = (0, 1, 2, 3, 1000, 2499, 2500, 39997, 39998, 39999)


def expected_stream(tree, steps, ranks, micro, verbatim):
    """Recompute, in the run tree, every rank's C16 offset stream from the counter-based seed:
    the streaming digest, histogram, count, the sampled steps and (smokes) every step."""
    arg = json.dumps([steps, list(ranks), micro, list(SAMPLED_STEPS), verbatim])
    rc, out, err = run([sys.executable, "-c", _STREAM_SNIPPET, arg], cwd=tree)
    if rc != 0:
        raise Refused(f"offset recomputation failed: {err.strip()[-400:]}")
    return {int(k): v for k, v in json.loads(out.strip().splitlines()[-1]).items()}


def load_rank_files(evdir):
    recs, summaries = {}, {}
    for name in sorted(os.listdir(evdir)):
        m = re.fullmatch(r"rank(\d+)\.jsonl", name)
        if m:
            with open(os.path.join(evdir, name)) as f:
                recs[int(m.group(1))] = [json.loads(line) for line in f if line.strip()]
        m = re.fullmatch(r"rank(\d+)\.summary\.json", name)
        if m:
            summaries[int(m.group(1))] = read_json(os.path.join(evdir, name))
    return recs, summaries


SIGNATURE_KEYS = ("state_signature", "param_signature", "rng_signature", "numerics", "yaw_aug")


def evidence_reasons(recs, summaries, log_text, mode, rung, steps, expected, fixture, reference=None):
    """The complete rank-private evidence contract (plan r2 §4.3-7 / §4.6). ``reference`` is
    the DDP_SMOKE / PILOT receipt a FULL run's initial state must equal."""
    world = world_of(mode, rung)
    micro = RUNGS[rung][1]
    verbatim = mode != "FULL"
    reasons = []
    if sorted(recs) != list(range(world)) or sorted(summaries) != list(range(world)):
        return [f"evidence from ranks {sorted(recs)} / summaries {sorted(summaries)}, expected "
                f"0..{world - 1}"]
    fits = {}
    for rank in range(world):
        rows, s = recs[rank], summaries[rank]
        if any(r.get("rank") != rank for r in rows) or s.get("rank") != rank:
            reasons.append(f"rank {rank}: records of another rank in its private files")
        f = [r for r in rows if r.get("kind") == "fit_start"]
        if len(f) != 1:
            reasons.append(f"rank {rank}: {len(f)} fit_start records (need 1)")
            continue
        f = fits[rank] = f[0]
        if f.get("world_size") != world:
            reasons.append(f"rank {rank}: world size {f.get('world_size')} != {world}")
        if f.get("yaw_aug") != YAW_BLOCK:
            reasons.append(f"rank {rank}: wrapper yaw_aug state {f.get('yaw_aug')}")
        if f.get("flash_attn_func_is_none") is not False:
            reasons.append(f"rank {rank}: flash_attn_func was None in the training process")
        if f.get("global_step") != 0:
            reasons.append(f"rank {rank}: fit started at global_step {f.get('global_step')}")
        if (world > 1) != bool(f.get("syncbn_modules")):
            reasons.append(f"rank {rank}: {f.get('syncbn_modules')} SyncBatchNorm modules with "
                           f"world size {world}")
        if not str(f.get("gpu_name", "")).startswith("NVIDIA"):
            reasons.append(f"rank {rank}: GPU identity {f.get('gpu_name')!r}")
        c = s.get("counts", {})
        for key, want in (("apply_yaw_aug", steps), ("training_step", steps), ("ckpt_reentrant", 0),
                          ("ckpt_default", 0), ("ckpt_unclassified", 0)):
            if c.get(key) != want:
                reasons.append(f"rank {rank}: {key} = {c.get(key)}, need {want}")
        for key in ("flash_calls", "ckpt_dinov3", "ckpt_dit"):
            if not c.get(key):
                reasons.append(f"rank {rank}: {key} = {c.get(key)} (need > 0)")
        if s.get("all_losses_finite") is not True:
            reasons.append(f"rank {rank}: all_losses_finite = {s.get('all_losses_finite')}")
        want_pub = len(CKPT_STEPS) if (mode == "FULL" and rank == 0) else 0
        if c.get("checkpoints_published") != want_pub:
            reasons.append(f"rank {rank}: {c.get('checkpoints_published')} checkpoints acknowledged, need {want_pub}")
        if s.get("global_step") != steps or s.get("n_steps") != steps or s.get("steps_contiguous_from_0") is not True:
            reasons.append(f"rank {rank}: steps {s.get('n_steps')} / global_step {s.get('global_step')} "
                           f"/ contiguous {s.get('steps_contiguous_from_0')} (need {steps})")
        e = expected[rank]
        if s.get("offset_digest") != e["digest"] or s.get("offset_histogram") != e["hist"] \
                or s.get("n_offsets") != e["n"]:
            reasons.append(f"rank {rank}: offset stream != the counter-based C16 recomputation")
        if s.get("sampled_offsets") != e["sampled"]:
            reasons.append(f"rank {rank}: sampled offsets != the recomputation")
        for step, ref in (fixture or {}).items():
            if step in s.get("sampled_offsets", {}) and rank < len(ref) and len(ref[rank]) == micro:
                if s["sampled_offsets"][step] != [32 * (d // 32) for d in ref[rank]]:
                    reasons.append(f"rank {rank} step {step}: != the E1 fixture snapped down")
        if verbatim:
            offs = [r for r in rows if r.get("kind") == "offsets"]
            if [r.get("offsets") for r in offs] != e["verbatim"] or [r.get("step") for r in offs] != list(range(steps)):
                reasons.append(f"rank {rank}: verbatim offsets differ from the recomputation")
    if len(fits) == world:
        for key in SIGNATURE_KEYS:
            if len({canonical(fits[r].get(key)) for r in range(world)}) != 1:
                reasons.append(f"initial {key} differs across ranks")
        if reference is not None:
            ref_fits = reference.get("fit_start") or {}
            for r in range(world):
                ref = ref_fits.get(str(r)) or ref_fits.get(r) or {}
                for key in SIGNATURE_KEYS:
                    if canonical(fits[r].get(key)) != canonical(ref.get(key)):
                        reasons.append(f"rank {r}: initial {key} differs from the "
                                       f"{reference.get('kind')} receipt's")
    lines = [ln.strip() for ln in re.split(r"[\r\n]", log_text)]
    if lines.count(BANNER) != 1:
        reasons.append(f"the C16 banner appears {lines.count(BANNER)} times (need 1)")
    state, why = start_gate(log_text, world)
    if state != "pass":
        reasons.append(f"start gate {state}: {why}")
    return reasons


def pilot_rate(recs, window=PILOT_WINDOW):
    """Rank 0's post-warm-up seconds per optimizer step over ``window`` (from the driver's
    per-step wall times)."""
    t = {r["step"]: r["t"] for r in recs.get(0, []) if r.get("kind") in ("offsets", "tick") and "t" in r}
    a, b = window
    if a not in t or b - 1 not in t:
        raise Refused(f"pilot evidence lacks steps {a} and {b - 1}")
    return (t[b - 1] - t[a]) / (b - 1 - a)


def full_time_request(sec_per_step, maxtime_min):
    """``--time`` minutes for a monolithic FULL: 40k x rate x 1.15 + 45 min, refused if it
    exceeds the partition's MaxTime (plan §4.4)."""
    need = math.ceil(ENDPOINT * sec_per_step * 1.15 / 60.0 + 45)
    if maxtime_min is not None and need > maxtime_min:
        raise Refused(f"a monolithic FULL needs {need} min > the partition MaxTime {maxtime_min} "
                      "min: the run cannot be monolithic here (plan D1/D3, Yixun's decision)")
    return need


def parse_slurm_time(text):
    """sinfo %l -> minutes (``infinite`` -> None)."""
    text = text.strip()
    if text in ("infinite", "UNLIMITED"):
        return None
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", text)
    if not m:
        m2 = re.fullmatch(r"(\d+):(\d+)", text)
        if not m2:
            raise Refused(f"unparseable partition time limit {text!r}")
        return int(m2.group(1))
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return d * 1440 + h * 60 + mi + (1 if s else 0)


# --------------------------------------------------------------------------- #
# cadence publication (plan §4.5)
# --------------------------------------------------------------------------- #
_CKPT_RE = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")


def ckpt_steps(names):
    out = {}
    for n in names:
        m = _CKPT_RE.match(n)
        if m:
            out.setdefault(int(m.group(2)), []).append(n)
    return out


def mark_saved(path):
    """The atomic 'this checkpoint's synchronous save returned' marker (plan §14 B2)."""
    if not _CKPT_RE.match(os.path.basename(path)) or not os.path.isfile(path):
        raise Refused(f"{path} is not a cadence checkpoint file")
    atomic_write_json(f"{path}.saved", {"sha256": sha256_file(path), "bytes": os.path.getsize(path),
                                        "t": time.time()})


def _saved_sha(path):
    rec = read_json(f"{path}.saved")
    if rec.get("bytes") != os.path.getsize(path):
        raise Refused(f"{path} changed size after its save marker")
    return rec["sha256"]


def publish_saved(path, run_id, timeout_s=1800, dest_root=NAS_ROOT):
    """Publish ONE saved checkpoint to ``dest_root/run_id`` and acknowledge it in the NAS
    manifest, within ``timeout_s`` (else raise: the caller fails the run)."""
    import threading
    result = {}

    def work():
        try:
            result["names"] = publish_pass(os.path.dirname(path), run_id, dest_root=dest_root,
                                           only=[os.path.basename(path)])
        except Exception as err:      # noqa: BLE001
            result["error"] = err
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise Refused(f"publishing {path} exceeded {timeout_s} s")
    if "error" in result:
        raise result["error"]
    man = read_manifest(os.path.join(dest_root, run_id, "MANIFEST.sha256"))
    if man.get(os.path.basename(path)) != _saved_sha(path):
        raise Refused(f"{path} is not acknowledged in the NAS manifest")


def record_saved_local(path):
    """Neuronic (D4): record the saved checkpoint's sha in CHECKPOINTS.partial.sha256."""
    manifest = os.path.join(os.path.dirname(os.path.dirname(path)), "CHECKPOINTS.partial.sha256")
    entries = read_manifest(manifest)
    entries[os.path.basename(path)] = _saved_sha(path)
    write_manifest(manifest, entries)


def read_manifest(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            sha, _, name = line.rstrip("\n").partition("  ")
            if not re.fullmatch(r"[0-9a-f]{64}", sha) or not _CKPT_RE.match(name):
                raise Refused(f"malformed manifest line {line!r} in {path}")
            if name in out:
                raise Refused(f"duplicate manifest entry {name} in {path}")
            out[name] = sha
    return out


def write_manifest(path, entries):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        for name in sorted(entries, key=lambda n: int(_CKPT_RE.match(n).group(2))):
            f.write(f"{entries[name]}  {name}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_path(os.path.dirname(os.path.abspath(path)), directory=True)


_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


def rename_noreplace(src, dst):
    """``renameat2(RENAME_NOREPLACE)``: atomically move ``src`` to ``dst`` iff ``dst`` does not
    exist (a plain ``rename`` would silently replace it). Refuses if unsupported."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "renameat2", None)
    if fn is None:
        raise Refused("renameat2 is unavailable; refusing a replace-capable rename")
    if fn(_AT_FDCWD, os.fsencode(src), _AT_FDCWD, os.fsencode(dst), _RENAME_NOREPLACE) != 0:
        err = ctypes.get_errno()
        raise Refused(f"renameat2({src} -> {dst}, NOREPLACE) failed: {os.strerror(err)}")


def fsync_path(path, directory=False):
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if directory else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_one(src_dir, name, dest, known_sha=None):
    """Hash -> (adopt an identical orphan | staged copy -> fsync -> re-hash -> NOREPLACE move ->
    dir fsync) -> source re-hash. Returns the sha256. Never deletes a published file."""
    src = os.path.join(src_dir, name)
    sha = sha256_file(src)
    if known_sha is not None and sha != known_sha:
        raise Refused(f"{src} sha256 {sha} != the manifest's {known_sha}")
    final = os.path.join(dest, name)
    if os.path.lexists(final):
        if os.path.islink(final) or sha256_file(final) != sha:
            raise Refused(f"{final} exists and is not an identical copy; refusing to clobber")
        return sha                                   # an orphan of a crash before the manifest
    tmp = os.path.join(dest, f".{name}.partial-{os.getpid()}")
    shutil.copyfile(src, tmp)
    fsync_path(tmp)
    if sha256_file(tmp) != sha:
        raise Refused(f"the NAS copy {tmp} of {name} does not re-hash to {sha}")
    rename_noreplace(tmp, final)
    fsync_path(dest, directory=True)
    if sha256_file(src) != sha:
        raise Refused(f"{src} changed while it was published")
    return sha


def publish_pass(src_dir, run_id, dest_root=NAS_ROOT, only=None, pulled_manifest=None):
    """Publish saved checkpoints to ``dest_root/run_id`` under the local flock: ``only`` (the
    in-run hook: one just-saved checkpoint, sha from its ``.saved`` marker) or
    ``pulled_manifest`` (the neuronic pull: exactly the listed, hash-verified files)."""
    import fcntl
    if not RUN_ID_RE.match(run_id):
        raise Refused(f"run id {run_id!r} does not match {RUN_ID_RE.pattern}")
    if (only is None) == (pulled_manifest is None):
        raise Refused("publish_pass needs exactly one of only= / pulled_manifest=")
    if only is not None:
        todo = [(n, _saved_sha(os.path.join(src_dir, n))) for n in only]
    else:
        todo = sorted(read_manifest(pulled_manifest).items(),
                      key=lambda kv: int(_CKPT_RE.match(kv[0]).group(2)))
    dest = os.path.join(dest_root, run_id)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.expanduser("~/.exp40_publish.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        nas_manifest = os.path.join(dest, "MANIFEST.sha256")
        published = read_manifest(nas_manifest)
        done = []
        for name, sha in todo:
            if name in published:
                if published[name] != sha:
                    raise Refused(f"{name} is on the NAS with {published[name]}, expected {sha}")
                continue
            published[name] = publish_one(src_dir, name, dest, known_sha=sha)
            write_manifest(nas_manifest, published)
            done.append(name)
        return done


# --------------------------------------------------------------------------- #
# end audit / admission (plan §4.6)
# --------------------------------------------------------------------------- #
def checkpoint_state_reasons(ck, arm_cfg, expected_dirpath, wrapper=None):
    reasons = []
    if ck.get("global_step") != ENDPOINT:
        reasons.append(f"global_step {ck.get('global_step')} != {ENDPOINT}")
    if ck.get("epoch") != (ENDPOINT - 1) // STEPS_PER_EPOCH:
        reasons.append(f"epoch {ck.get('epoch')} != {(ENDPOINT - 1) // STEPS_PER_EPOCH}")
    if canonical(ck.get("model_config")) != canonical(arm_cfg):
        reasons.append("the embedded model_config differs from the arm config")
    sd = ck.get("state_dict") or {}
    import torch
    nonfinite = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()
                 and not torch.isfinite(v).all()]
    if nonfinite:
        reasons.append(f"non-finite tensors {nonfinite[:5]}")
    ema = {k[len("diffusion_ema.ema_model."):]: v for k, v in sd.items()
           if k.startswith("diffusion_ema.ema_model.")}
    online = {k[len("diffusion.model."):]: v for k, v in sd.items() if k.startswith("diffusion.model.")}
    if not ema or set(ema) != set(online) or any(
            tuple(ema[k].shape) != tuple(online[k].shape) or ema[k].dtype != online[k].dtype for k in ema):
        reasons.append(f"EMA ({len(ema)}) does not mirror the online DiT ({len(online)})")
    initted, step = sd.get("diffusion_ema.initted"), sd.get("diffusion_ema.step")
    if initted is None or not bool(initted) or step is None or int(step) <= 0:
        reasons.append(f"EMA state initted={initted} step={step}")
    opt = ck.get("optimizer_states") or []
    if len(opt) != 1 or len(opt[0].get("param_groups", [])) != 1:
        reasons.append("expected one optimizer with one param group")
    else:
        pg = opt[0]["param_groups"][0]
        for k, v in ADAMW.items():
            got = list(pg.get(k)) if isinstance(pg.get(k), (tuple, list)) else pg.get(k)
            if got != v:
                reasons.append(f"AdamW {k} {got!r} != {v!r}")
    sch = ck.get("lr_schedulers") or []
    if len(sch) != 1 or any(sch[0].get(k) != v for k, v in SCHED.items()):
        reasons.append(f"InverseLR state {({k: sch[0].get(k) for k in SCHED} if sch else None)} != {SCHED}")
    loops = ck.get("loops", {}).get("fit_loop", {})
    try:
        opt_steps = loops["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"]
    except (KeyError, TypeError):
        opt_steps = None
    if opt_steps != ENDPOINT:
        reasons.append(f"fit_loop optimizer steps completed {opt_steps} != {ENDPOINT}")
    cbs = ck.get("callbacks", {})
    key = ("ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2500, "
           "'every_n_epochs': 0, 'train_time_interval': None}")
    if list(cbs) != [key]:
        reasons.append(f"callbacks {list(cbs)} != [{key}]")
    elif os.path.realpath(str(cbs[key].get("dirpath"))) != os.path.realpath(expected_dirpath):
        reasons.append(f"ModelCheckpoint dirpath {cbs[key].get('dirpath')} != {expected_dirpath}")
    if wrapper is not None and not reasons:
        fresh = wrapper.state_dict()
        inv = lambda d: {k: (tuple(v.shape), str(v.dtype)) for k, v in d.items()}
        if inv(fresh) != inv(sd):
            missing = sorted(set(fresh) - set(sd))[:5]
            extra = sorted(set(sd) - set(fresh))[:5]
            reasons.append(f"state inventory differs from a fresh wrapper (missing {missing}, "
                           f"extra {extra})")
        else:
            wrapper.load_state_dict(sd, strict=True)
            params = list(wrapper.diffusion.parameters())
            trainable = {i for i, p in enumerate(params) if p.requires_grad}
            state = opt[0]["state"]
            if set(map(int, state)) != trainable or sorted(opt[0]["param_groups"][0]["params"]) != sorted(range(len(params))):
                reasons.append(f"optimizer state covers {len(state)} params, {len(trainable)} trainable")
            for idx, st in state.items():
                p = params[int(idx)]
                bad = [k for k in ("exp_avg", "exp_avg_sq")
                       if st.get(k) is None or tuple(st[k].shape) != tuple(p.shape)
                       or not torch.isfinite(st[k]).all()]
                if bad or float(st.get("step", -1)) != float(ENDPOINT):
                    reasons.append(f"optimizer state {idx} malformed ({bad}, step {st.get('step')})")
                    break
    return reasons


def build_wrapper_cpu(tree, arm_cfg):
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from src.models.factory import create_model_from_config
    from src.training.factory import create_training_wrapper_from_config
    cfg = json.loads(canonical(arm_cfg))
    return create_training_wrapper_from_config(cfg, create_model_from_config(cfg))


def load_checkpoint_once(path):
    import torch
    with open(path, "rb") as f:
        blob = f.read()
    return hashlib.sha256(blob).hexdigest(), len(blob), torch.load(
        io.BytesIO(blob), map_location="cpu", weights_only=False)


def full_evidence_reasons(path, run_id):
    try:
        rec = read_json(path)
    except (OSError, ValueError) as err:
        return [f"FULL evidence receipt {path} unreadable ({err})"], None
    reasons = []
    for key, want in (("kind", "full_evidence"), ("verdict", "PASS"), ("run_id", run_id), ("mode", "FULL")):
        if rec.get(key) != want:
            reasons.append(f"FULL evidence {key} = {rec.get(key)!r}, need {want!r}")
    return reasons, sha256_file(path)


def audit(ckpt_dir, tree, nas_dir=None, strict_load=True, evidence=None, run_id=None):
    names = sorted(os.listdir(ckpt_dir))
    reasons = []
    ev_sha = None
    if evidence is not None:
        r, ev_sha = full_evidence_reasons(evidence, run_id)
        reasons += r
    others = [n for n in names if not _CKPT_RE.match(n) and n != "CHECKPOINTS.sha256"
              and not (n.endswith(".ckpt.saved") and _CKPT_RE.match(n[:-len(".saved")]))]
    if others:
        reasons.append(f"unexpected files in the checkpoint dir: {others}")
    steps = ckpt_steps(n for n in names if _CKPT_RE.match(n))
    if sorted(steps) != list(CKPT_STEPS) or any(len(v) != 1 for v in steps.values()):
        reasons.append(f"checkpoint steps {sorted(steps)} != {list(CKPT_STEPS)} (one each)")
    for s_, v in steps.items():
        if f"{v[0]}.saved" not in names:
            reasons.append(f"{v[0]} has no save marker")
    shas = {}
    if not reasons:
        for s in CKPT_STEPS:
            shas[steps[s][0]] = sha256_file(os.path.join(ckpt_dir, steps[s][0]))
        final = steps[ENDPOINT][0]
        sha, _, ck = load_checkpoint_once(os.path.join(ckpt_dir, final))
        if sha != shas[final]:
            reasons.append("the 40k checkpoint changed while it was audited")
        with open(os.path.join(tree, ARM_CONFIG)) as f:
            arm_cfg = json.load(f)
        wrapper = build_wrapper_cpu(tree, arm_cfg) if strict_load else None
        reasons += checkpoint_state_reasons(ck, arm_cfg, ckpt_dir, wrapper)
        if nas_dir is not None:
            on_nas = read_manifest(os.path.join(nas_dir, "MANIFEST.sha256"))
            if on_nas != shas:
                reasons.append(f"NAS inventory {sorted(on_nas)} does not equal the 16 local "
                               "checkpoints with identical sha256")
            for name, s in on_nas.items():
                if sha256_file(os.path.join(nas_dir, name)) != s:
                    reasons.append(f"NAS {name} does not re-hash to its manifest entry")
    if not reasons:
        write_manifest(os.path.join(ckpt_dir, "CHECKPOINTS.sha256"), shas)
    return {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons,
            "ckpt_dir": os.path.abspath(ckpt_dir), "sha256": shas, "strict_load": strict_load,
            "nas_dir": nas_dir, "full_evidence_sha256": ev_sha}


def admit(nas_run_dir, tree, expected_dirpath, meta, evidence, audit_json, strict_load=True):
    r, ev_sha = full_evidence_reasons(evidence, meta.get("run_id"))
    au = read_json(audit_json)
    if au.get("verdict") != "PASS" or au.get("full_evidence_sha256") != ev_sha:
        r.append(f"audit {audit_json} is not a PASS bound to the FULL evidence {ev_sha}")
    if r:
        raise Refused("; ".join(r))
    manifest = read_manifest(os.path.join(nas_run_dir, "MANIFEST.sha256"))
    if manifest != au.get("sha256"):
        raise Refused("the NAS manifest differs from the audited CHECKPOINTS.sha256")
    steps = ckpt_steps(manifest)
    if sorted(steps) != list(CKPT_STEPS):
        raise Refused(f"NAS manifest steps {sorted(steps)} != {list(CKPT_STEPS)}")
    final = steps[ENDPOINT][0]
    sha, nbytes, ck = load_checkpoint_once(os.path.join(nas_run_dir, final))
    if sha != manifest[final]:
        raise Refused(f"{final} sha256 {sha} != its MANIFEST entry")
    with open(os.path.join(tree, ARM_CONFIG)) as f:
        arm_cfg = json.load(f)
    reasons = checkpoint_state_reasons(ck, arm_cfg, expected_dirpath,
                                       build_wrapper_cpu(tree, arm_cfg) if strict_load else None)
    if reasons:
        raise Refused("; ".join(reasons))
    ema_inv = digest_obj(sorted((k, list(v.shape), str(v.dtype)) for k, v in ck["state_dict"].items()
                                if k.startswith("diffusion_ema.ema_model.")))
    historical = {"exp17_acousticrooms_snapshot": "unknown", "exp17_dinov3_revision_and_bytes": "unknown",
                  "exp17_vae_bytes": "unknown", "exp17_dependency_versions": "unknown",
                  "exp17_wandb_behaviour": "unknown", "exp17_launch_commit": "unknown (audited base 41aa31dc)",
                  "exp17_attention_backend": "flash-attn (inferred, not proven)"}
    current = {"acousticrooms_manifest_digest": read_json(os.path.join(tree, AR_MANIFEST))["digest"]
               if os.path.exists(os.path.join(tree, AR_MANIFEST)) else None,
               "dinov3_revision": DINO_REVISION, "dinov3_config_sha256": DINO_CONFIG_SHA256,
               "dinov3_weights_sha256": DINO_SHA256, "vae_sha256": PINS[VAE]}
    return {"admitted": True, "arm": "C16", **meta, "historical": historical, "current_assets": current,
            "trajectory_equivalence_claim_allowed": False,
            "ckpt_path": os.path.join(nas_run_dir, final),
            "sha256": sha, "bytes": nbytes, "global_step": ck["global_step"],
            "model_config": ARM_CONFIG, "model_config_sha256": PINS[ARM_CONFIG],
            "embedded_config_sha256": digest_obj(ck["model_config"]),
            "ema_inventory_sha256": ema_inv, "strict_load": strict_load,
            "full_evidence_sha256": ev_sha, "audit_sha256": sha256_file(audit_json),
            "manifest_sha256": sha256_file(os.path.join(nas_run_dir, "MANIFEST.sha256"))}


# --------------------------------------------------------------------------- #
# reservation token (handshake dir) and execution-side claim (plan r2 §4.3-5, §4.4)
# --------------------------------------------------------------------------- #
def create_exclusive(path, payload):
    """``O_CREAT|O_EXCL`` + fsync: exactly one creator ever succeeds."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_path(os.path.dirname(os.path.abspath(path)), directory=True)


def acquire_gpu_locks(handshake_dir, uuids, payload):
    """Cooperative per-card locks (plan §14 H1): ``GPU_LOCK_<uuid>`` created ``O_EXCL`` in
    sorted uuid order with an unguessable owner token; any failure rolls back exactly the
    locks this attempt created. Returns (token, paths)."""
    token = uuid.uuid4().hex
    taken = []
    try:
        for u in sorted(set(uuids)):
            if not re.fullmatch(r"GPU-[0-9a-f-]{36}", u):
                raise Refused(f"not a GPU uuid: {u!r}")
            path = os.path.join(handshake_dir, f"GPU_LOCK_{u}")
            try:
                create_exclusive(path, {**payload, "owner_token": token, "uuid": u, "t": time.time()})
            except FileExistsError:
                raise Refused(f"{path} is held: {read_json(path) if os.path.exists(path) else '?'}") from None
            taken.append(path)
    except BaseException:
        for path in taken:
            _release_one(path, token)
        raise
    return token, taken


def _release_one(path, token):
    try:
        rec = read_json(path)
    except (OSError, ValueError) as err:
        raise Refused(f"cannot read lock {path} ({err})") from None
    if rec.get("owner_token") != token:
        raise Refused(f"{path} is owned by another token; not releasing")
    released = os.path.join(os.path.dirname(path), "released")
    os.makedirs(released, exist_ok=True)
    rename_noreplace(path, os.path.join(released, f"{os.path.basename(path)}.released-"
                                                  f"{time.strftime('%Y-%m-%d_%H-%M-%S')}-{token[:8]}"))


def release_gpu_locks(paths, token):
    errors = []
    for path in paths:
        try:
            _release_one(path, token)
        except Refused as err:
            errors.append(str(err))
    if errors:
        raise Refused("; ".join(errors))


def execution_id(run_id, mode, stamp=""):
    """FULL: the RUN_ID (a rerun needs a new _r<n>); PILOT: one per submission stamp."""
    return run_id if mode == "FULL" else f"{run_id}_pilot_{check_stamp(stamp)}"


def claim(claims_dir, exec_id, mode, job_id, intent):
    """The execution-side guarantee (plan §14 B1): at most one training execution per
    execution identity. The path never contains the submission intent."""
    os.makedirs(claims_dir, exist_ok=True)
    path = os.path.join(claims_dir, f"{exec_id}__{mode}.claim")
    try:
        create_exclusive(path, {"execution_id": exec_id, "mode": mode, "job_id": job_id,
                                "intent": intent, "host": os.uname().nodename, "t": time.time()})
    except FileExistsError:
        raise Refused(f"claim {path} exists: another job owns this execution; this one must not train") from None
    return path


# --------------------------------------------------------------------------- #
# neuronic submit-exactly-once (plan §4.4, Codex r1 BLOCKING 1)
# --------------------------------------------------------------------------- #
TERMINAL_OK = ("COMPLETED",)
TERMINAL_FAIL = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
                 "BOOT_FAIL", "DEADLINE", "REVOKED")
LIVE = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED",
        "STOPPED", "SIGNALING", "STAGE_OUT", "REQUEUE_HOLD", "REQUEUE_FED", "SPECIAL_EXIT")


def normalise_state(raw):
    """sacct/squeue State (possibly truncated, e.g. ``OUT_OF_ME+``) -> one canonical state;
    an ambiguous or unknown prefix is FATAL."""
    s = raw.strip().split()[0] if raw.strip() else ""
    s = s.rstrip("+")
    cands = [x for x in TERMINAL_OK + TERMINAL_FAIL + LIVE if x.startswith(s)] if s else []
    if len(cands) != 1:
        raise Refused(f"cannot normalise scheduler state {raw!r} (candidates {cands})")
    return cands[0]


def job_name(exec_id, mode):
    return f"exp40-{exec_id}-{mode}"


def reconcile(name, runner):
    """``[(jobid, state)]`` for ``name`` from BOTH squeue and sacct; any query failure fatal."""
    rc, out, err = runner(["squeue", "-h", "-n", name, "-o", "%i|%T"])
    if rc != 0:
        raise Refused(f"squeue failed: {err.strip()}")
    jobs = {}
    for line in out.strip().splitlines():
        jid, _, st = line.partition("|")
        jobs[jid.strip()] = normalise_state(st)
    rc, out, err = runner(["sacct", "-n", "-X", "-P", "--name", name, "-o", "JobID,State%-40"])
    if rc != 0:
        raise Refused(f"sacct failed: {err.strip()}")
    for line in out.strip().splitlines():
        jid, _, st = line.partition("|")
        jid = jid.strip()
        if jid and jid not in jobs:
            jobs[jid] = normalise_state(st)
    return sorted(jobs.items())


def execution_status(state, runner, grace_s, sleep):
    """``live`` | ``completed`` | ``failed`` | ``ambiguous`` | ``never`` for one execution."""
    if state["status"] == "prepared":
        return "never", []                           # sbatch was never attempted
    name = job_name(state["execution_id"], state["mode"])
    jobs = reconcile(name, runner)
    if not jobs:
        sleep(grace_s)                               # accounting can lag a short-lived job
        jobs = reconcile(name, runner)
    sts = [st for _, st in jobs]
    if any(st in LIVE for st in sts):
        return "live", jobs
    if not jobs:
        return "ambiguous", jobs
    if all(st in TERMINAL_OK for st in sts):
        return "completed", jobs
    return "failed", jobs


def neuronic_submit(states_dir, run_id, mode, exec_id, sbatch_argv, runner, dry_run=False,
                    grace_s=120.0, sleep=time.sleep, relaxation=None, probe_ok=False):
    """Submission bookkeeping (plan §14 B1). One state file per execution; the claim in the
    job is the guarantee. Refuses a new execution of RUN_ID while any earlier execution of it
    is live or ambiguous; a proven failure closes that execution (a fresh identity is needed)."""
    if mode not in ("PILOT", "FULL"):
        raise Refused(f"neuronic mode must be PILOT or FULL, got {mode!r}")
    if mode == "FULL" and relaxation != "post-run-publish":
        raise Refused("a neuronic FULL needs Yixun's advance NAS relaxation "
                      "EXP40_NAS_RELAXATION=post-run-publish (plan D4)")
    if mode == "FULL" and not probe_ok:
        raise Refused("a neuronic FULL needs a PASS receipt of exp40_claim_probe.sbatch (plan §14 B1)")
    os.makedirs(states_dir, exist_ok=True)
    mine = os.path.join(states_dir, f"{exec_id}__{mode}.state.json")
    for name in sorted(os.listdir(states_dir)):
        if not name.endswith(".state.json"):
            continue
        st = read_json(os.path.join(states_dir, name))
        if st.get("run_id") != run_id:
            continue
        status, jobs = execution_status(st, runner, grace_s, sleep)
        st["reconciled"] = {"status": status, "jobs": jobs, "t": time.time()}
        if not dry_run:
            atomic_write_json(os.path.join(states_dir, name), st)
        if status in ("live", "ambiguous"):
            raise Refused(f"execution {st['execution_id']} ({st['mode']}) is {status} {jobs}: no new "
                          "submission for this RUN_ID until it is proven terminal")
        if os.path.join(states_dir, name) == mine:
            raise Refused(f"execution {exec_id} already exists ({status}); a new attempt needs a "
                          "fresh execution identity (PILOT: a new stamp; FULL: a new _r<n> RUN_ID)")
        if mode == "FULL" and st.get("mode") == "FULL" and status == "completed":
            raise Refused(f"FULL {run_id} already COMPLETED ({jobs})")
    state = {"run_id": run_id, "mode": mode, "execution_id": exec_id, "status": "prepared",
             "intent": uuid.uuid4().hex[:12], "relaxation": relaxation, "job_ids": []}
    cmd = sbatch_argv(job_name(exec_id, mode), state["intent"])
    if dry_run:
        return "dry-run", {**state, "would_run": cmd}
    atomic_write_json(mine, state)
    state["status"] = "sbatch_attempted"             # the durable boundary BEFORE sbatch runs
    state["sbatch"] = cmd
    atomic_write_json(mine, state)
    rc, out, err = runner(cmd)
    if rc != 0:
        raise Refused(f"sbatch failed (rc {rc}): {err.strip()} -- state 'sbatch_attempted'; "
                      "reconcile before any new submission")
    jid = out.strip().split(";")[0]
    if not jid.isdigit():
        raise Refused(f"sbatch --parsable printed {out!r}; state stays 'sbatch_attempted'")
    state.update(status="submitted", job_ids=[jid])
    atomic_write_json(mine, state)
    return "submitted", state


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _links_from(args):
    links = {}
    for item in args.link or []:
        k, _, v = item.partition("=")
        links[k] = v
    return links


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def ident(p, stamp=True):
        p.add_argument("--platform", required=True); p.add_argument("--rung", required=True)
        p.add_argument("--mode", required=True); p.add_argument("--run-id", required=True)
        p.add_argument("--save-root", required=True)
        if stamp:
            p.add_argument("--stamp", default="")

    a = sub.add_parser("argv"); ident(a)
    b = sub.add_parser("build-runtree"); b.add_argument("--repo", required=True)
    b.add_argument("--sha", required=True); b.add_argument("--dest", required=True)
    b.add_argument("--link", action="append")
    v = sub.add_parser("verify-runtree"); v.add_argument("--repo", required=True)
    v.add_argument("--sha", required=True); v.add_argument("--tree", required=True)
    v.add_argument("--link", action="append")
    c = sub.add_parser("check"); ident(c); c.add_argument("--sha", required=True)
    c.add_argument("--repo", required=True); c.add_argument("--tree", required=True)
    c.add_argument("--link", action="append"); c.add_argument("--receipts", required=True)
    c.add_argument("--out", required=True); c.add_argument("--in-job", action="store_true")
    c.add_argument("--hf-sealed", required=True)
    g = sub.add_parser("gpus"); g.add_argument("--platform", required=True)
    g.add_argument("--rung", required=True)
    g.add_argument("--mode", required=True); g.add_argument("--lanes", required=True)
    g.add_argument("--policy", required=True); g.add_argument("--expect-total", type=int, required=True)
    g.add_argument("--out")
    w = sub.add_parser("watch"); w.add_argument("--log", required=True)
    w.add_argument("--members", type=int, required=True); w.add_argument("--pid", type=int, required=True)
    w.add_argument("--pgid", type=int, required=True); w.add_argument("--timeout", type=int, default=3600)
    e = sub.add_parser("evidence"); ident(e)
    e.add_argument("--evidence-dir", required=True); e.add_argument("--log", required=True)
    e.add_argument("--check-json", required=True); e.add_argument("--receipts", required=True)
    e.add_argument("--train-rc", type=int, required=True)
    e.add_argument("--watcher-rc", type=int, required=True); e.add_argument("--out", required=True)
    e.add_argument("--reference"); e.add_argument("--publisher-rc", type=int, default=0)
    t = sub.add_parser("neuronic-time"); t.add_argument("--receipt", required=True)
    t.add_argument("--maxtime", required=True)
    p = sub.add_parser("publish-cadence"); p.add_argument("--src", required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--pulled-manifest", required=True)
    p.add_argument("--dest-root", default=NAS_ROOT)
    u = sub.add_parser("audit"); u.add_argument("--ckpt-dir", required=True)
    u.add_argument("--nas-dir"); u.add_argument("--out", required=True)
    u.add_argument("--evidence", required=True); u.add_argument("--run-id", required=True)
    d = sub.add_parser("admit"); d.add_argument("--nas-run-dir", required=True)
    d.add_argument("--expected-dirpath", required=True); d.add_argument("--meta-json", required=True)
    d.add_argument("--evidence", required=True); d.add_argument("--audit", required=True)
    d.add_argument("--out", required=True)
    m = sub.add_parser("ar-manifest"); m.add_argument("--ar-root", required=True)
    m.add_argument("--split", action="append", required=True); m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=4)
    r_ = sub.add_parser("gpu-lock"); r_.add_argument("--uuids", required=True)
    r_.add_argument("--payload-json", required=True); r_.add_argument("--handshake-dir", default=HANDSHAKE_DIR)
    r_.add_argument("--out", required=True)
    rl = sub.add_parser("gpu-unlock"); rl.add_argument("--lock-json", required=True)
    cl = sub.add_parser("claim"); cl.add_argument("--claims-dir", required=True)
    cl.add_argument("--execution-id", required=True); cl.add_argument("--mode", required=True)
    cl.add_argument("--intent", required=True); cl.add_argument("--job-id", required=True)
    n = sub.add_parser("neuronic-submit"); n.add_argument("--states-dir", required=True)
    n.add_argument("--run-id", required=True); n.add_argument("--mode", required=True)
    n.add_argument("--execution-id", required=True)
    n.add_argument("--sbatch-json", required=True); n.add_argument("--dry-run", action="store_true")
    n.add_argument("--probe-receipt")
    hs = sub.add_parser("build-hf-sealed"); hs.add_argument("--src-hub", required=True)
    hs.add_argument("--dest", required=True)
    av = sub.add_parser("ar-verify"); av.add_argument("--ar-root", required=True)
    av.add_argument("--manifest", required=True); av.add_argument("--out", required=True)
    av.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "argv":
            print("\n".join(train_argv(args.platform, args.rung, args.mode, args.run_id,
                                       args.save_root, args.stamp)))
            return 0
        if args.cmd == "build-runtree":
            print(build_runtree(args.repo, args.sha, args.dest, _links_from(args)))
            return 0
        if args.cmd == "verify-runtree":
            reasons = verify_runtree(args.repo, args.sha, args.tree, _links_from(args))
            for r in reasons:
                print(f"REFUSED: {r}")
            return 0 if not reasons else 2
        if args.cmd == "check":
            reasons, info = check(args.platform, args.rung, args.mode, args.run_id, args.sha,
                                  args.repo, args.tree, _links_from(args), args.save_root,
                                  args.stamp, args.receipts, args.hf_sealed, in_job=args.in_job)
            info.update(verdict="PASS" if not reasons else "REFUSED", reasons=reasons)
            atomic_write_json(args.out, info)
            for r in reasons:
                print(f"REFUSED: {r}")
            print(f"CHECK {info['verdict']} -> {args.out}")
            return 0 if not reasons else 2
        if args.cmd == "gpus":
            lanes = [int(x) for x in args.lanes.split(",")]
            gpus, apps = query_gpus()
            reasons = gpu_reasons(args.platform, args.mode, args.rung, lanes, args.policy, gpus,
                                  apps, args.expect_total)
            rec = {"gpus": gpus, "compute_apps": apps, "lanes": lanes, "policy": args.policy,
                   "verdict": "PASS" if not reasons else "REFUSED", "reasons": reasons,
                   "t": time.time()}
            if args.out:
                atomic_write_json(args.out, rec)
            for r in reasons:
                print(f"REFUSED: {r}")
            return 0 if not reasons else 2
        if args.cmd == "watch":
            return watch(args.log, args.members, args.pid, args.pgid, args.timeout)
        if args.cmd == "evidence":
            chk = read_json(args.check_json)
            world, micro = world_of(args.mode, args.rung), RUNGS[args.rung][1]
            steps = ENDPOINT if args.mode == "FULL" else SMOKE_STEPS[args.mode]
            recs, summaries = load_rank_files(args.evidence_dir)
            expected = expected_stream(ROOT, steps, range(world), micro, verbatim=args.mode != "FULL")
            with open(os.path.join(ROOT, FIXTURE)) as f:
                fx = json.load(f)["record"]["real"]["offsets"].get(args.rung)
            if args.mode == "SMOKE" and fx:
                fx = {k: v[:1] for k, v in fx.items()}
            with open(args.log, errors="replace") as f:
                text = f.read()
            reference = read_json(args.reference) if args.reference else None
            if args.mode == "FULL" and reference is None:
                raise Refused("FULL evidence needs --reference (the DDP_SMOKE / PILOT receipt)")
            reasons = evidence_reasons(recs, summaries, text, args.mode, args.rung, steps,
                                       expected, fx, reference)
            argv_ = chk.get("argv") or []
            rec = {"verdict": "PASS" if not reasons and args.train_rc == 0
                   and args.watcher_rc == 0 and args.publisher_rc == 0 else "FAIL",
                   "kind": {"DDP_SMOKE": "ddp_smoke", "PILOT": "pilot", "SMOKE": "smoke",
                            "FULL": "full_evidence"}[args.mode],
                   "platform": args.platform, "rung": args.rung, "mode": args.mode,
                   "run_id": args.run_id, "stamp": args.stamp, "sha": chk.get("sha"),
                   "tree_digest": chk.get("tree_digest"), "env_digest": chk.get("env_digest"),
                   "recipe_projection": recipe_projection(argv_), "argv": argv_,
                   "train_rc": args.train_rc, "watcher_rc": args.watcher_rc,
                   "publisher_rc": args.publisher_rc,
                   "reference_sha256": sha256_file(args.reference) if args.reference else None,
                   "evidence": {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons},
                   "fit_start": {str(r): next((x for x in recs.get(r, []) if x.get("kind") == "fit_start"), None)
                                 for r in range(world)},
                   "summaries": summaries}
            if args.mode == "PILOT" and not reasons:
                rec["sec_per_step"] = pilot_rate(recs)
            atomic_write_json(args.out, rec)
            if rec["verdict"] == "PASS" and args.mode in ("DDP_SMOKE", "PILOT"):
                os.makedirs(args.receipts, exist_ok=True)
                atomic_write_json(os.path.join(args.receipts, receipt_name(
                    rec["kind"], args.platform, args.rung, rec["sha"])), rec)
            print(f"EVIDENCE {rec['verdict']} -> {args.out}")
            for r in reasons:
                print(f"  {r}")
            return 0 if rec["verdict"] == "PASS" else 1
        if args.cmd == "ar-manifest":
            rooms = sorted({room for sp in args.split for room in split_rooms(sp)})
            t0 = time.time()
            man = ar_manifest(args.ar_root, rooms, content=True, workers=args.workers)
            rec = {"ar_root": os.path.realpath(args.ar_root), "splits": args.split,
                   "rooms": man, "digest": digest_obj(man), "seconds": time.time() - t0}
            atomic_write_json(args.out, rec)
            print(f"AR MANIFEST {rec['digest']} ({len(man)} rooms, {rec['seconds']:.0f} s) -> {args.out}")
            return 0
        if args.cmd == "neuronic-time":
            rec = read_json(args.receipt)
            if rec.get("kind") != "pilot" or rec.get("verdict") != "PASS" or "sec_per_step" not in rec:
                raise Refused(f"{args.receipt} is not a PASS pilot receipt with a measured rate")
            print(full_time_request(float(rec["sec_per_step"]), parse_slurm_time(args.maxtime)))
            return 0
        if args.cmd == "publish-cadence":
            done = publish_pass(args.src, args.run_id, dest_root=args.dest_root,
                                pulled_manifest=args.pulled_manifest)
            print(f"PUBLISHED {done}")
            return 0
        if args.cmd == "audit":
            v = audit(args.ckpt_dir, ROOT, nas_dir=args.nas_dir, evidence=args.evidence,
                      run_id=args.run_id)
            atomic_write_json(args.out, v)
            print(f"AUDIT {v['verdict']} -> {args.out}")
            for r in v["reasons"]:
                print(f"  {r}")
            return 0 if v["verdict"] == "PASS" else 1
        if args.cmd == "admit":
            if os.path.exists(args.out):
                raise Refused(f"{args.out} exists; an admission record is written once")
            rec = admit(args.nas_run_dir, ROOT, args.expected_dirpath, json.loads(args.meta_json),
                        args.evidence, args.audit)
            atomic_write_json(args.out, rec)
            print(f"ADMITTED {rec['sha256']} -> {args.out}")
            return 0
        if args.cmd == "gpu-lock":
            token, paths = acquire_gpu_locks(args.handshake_dir, args.uuids.split(","),
                                             json.loads(args.payload_json))
            atomic_write_json(args.out, {"token": token, "paths": paths})
            print(f"LOCKED {paths}")
            return 0
        if args.cmd == "gpu-unlock":
            rec = read_json(args.lock_json)
            release_gpu_locks(rec["paths"], rec["token"])
            print(f"UNLOCKED {rec['paths']}")
            return 0
        if args.cmd == "claim":
            print(claim(args.claims_dir, args.execution_id, args.mode, args.job_id, args.intent))
            return 0
        if args.cmd == "build-hf-sealed":
            print(build_hf_sealed(args.src_hub, args.dest))
            return 0
        if args.cmd == "ar-verify":
            man = read_json(args.manifest)
            t0 = time.time()
            rooms = sorted(tuple(r.split("/")) for r in man["rooms"])
            got = ar_manifest(args.ar_root, rooms, content=True, workers=args.workers)
            rec = {"verdict": "PASS" if got == man["rooms"] else "FAIL",
                   "manifest_digest": man["digest"], "got_digest": digest_obj(got),
                   "ar_root": os.path.realpath(args.ar_root), "t": time.time(),
                   "seconds": time.time() - t0,
                   "differing_rooms": sorted(r for r in man["rooms"] if got.get(r) != man["rooms"][r])}
            atomic_write_json(args.out, rec)
            print(f"AR VERIFY {rec['verdict']} ({rec['seconds']:.0f} s) -> {args.out}")
            return 0 if rec["verdict"] == "PASS" else 1
        if args.cmd == "neuronic-submit":
            spec = json.loads(args.sbatch_json)

            exports = [o for o in spec["options"] if o.startswith("--export=")]
            if len(exports) != 1 or not exports[0].startswith("--export=ALL,"):
                raise Refused("the sbatch spec needs exactly one --export=ALL,... option")

            def sbatch_argv(name, intent):
                opts = [o + f",EXP40_INTENT={intent}" if o.startswith("--export=") else o
                        for o in spec["options"]]
                return ["sbatch", "--parsable", "--job-name", name, "--comment", intent] + opts + [spec["script"]]
            probe_ok = False
            if args.probe_receipt and os.path.exists(args.probe_receipt):
                probe_ok = read_json(args.probe_receipt).get("verdict") == "PASS"
            action, state = neuronic_submit(args.states_dir, args.run_id, args.mode, args.execution_id,
                                            sbatch_argv, run, dry_run=args.dry_run,
                                            relaxation=os.environ.get("EXP40_NAS_RELAXATION"),
                                            probe_ok=probe_ok)
            print(json.dumps({"action": action, "state": state}, sort_keys=True))
            return 0
    except Refused as err:
        print(f"REFUSED: {err}")
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
