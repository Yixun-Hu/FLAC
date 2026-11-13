#!/usr/bin/env python3
"""exp_43 std_aug -- the training kit's single source of truth (plan_std_aug.md §5.1).

Forked from the exp_40 kit (whose folder stays byte-frozen). The bash wrapper
``exp43_train_local.sh`` (the 2xA6000 box, the only platform of exp_43) only sequences these
subcommands; every decision is here and unit-tested (``src/tests/test_exp43_kit.py``).

  argv                           the literal train.py argv
  build-runtree | verify-runtree  the immutable run tree (git archive + allowlisted links)
  check                          every no-GPU gate against a run tree
  gpus                           GPU identity / occupancy (nvidia-smi, uuid-mapped)
  gpu-lock | gpu-unlock          the cooperative per-card GPU_LOCK_<uuid> files
  watch | stop-group             the start watcher; bounded termination of an OWNED group
  evidence                       aggregate the rank-private evidence; a smoke receipt, or the
                                 FULL run's evidence
  publish-saved                  (in-run, killable subprocess) one saved checkpoint -> NAS
  nas-probe                      the real-NAS primitive capability receipt (pre-FULL)
  audit | run-result | admit     the end audit; the stable run-result manifest; the admission
                                 (every identity derived from validated records)
  ar-manifest | ar-verify | ar-receipt-check   the AcousticRooms snapshot identity
  build-hf-sealed                the read-only DINOv3 hub

Exit codes: 0 PASS; 1 a verdict FAILED; 2 a gate REFUSED; 8 / 9 / 10 / 11 watcher.

Written by the exp_43 Coder seat (Claude Opus 5.5) per plan_std_aug.md §5 on the exp_40 kit
(code-review provenance tags such as "code-r1 H2" name the exp_40 reviews that introduced a check).
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
# constants (plan §4; §14 binding)
# --------------------------------------------------------------------------- #
EXPREL = "worklog/worklog_yixun/exp_43_std_aug_claude"
# the RF arm: the exp_15 YAW config with training.yaw_aug.flip = true (plan §3.5)
ARM_CONFIG = f"{EXPREL}/FLAC_AR_YAWAUG_FLIP.json"
BASE_CONFIG = "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"
DATASET_CONFIG = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE = "weights/FLAC/VAE.safetensors"
FIXTURE = "src/tests/fixtures/exp40_yaw_aug_enabled_golden.json"
# The full AcousticRooms content manifest: a BYTE COPY of exp_40's (the same snapshot, built
# 2026-11-07 over train + unseen_eval), pinned by sha256 (plan §5.1); its 48-hour re-hash
# receipts are issued fresh under exp_43's own receipts directory.
AR_MANIFEST = f"{EXPREL}/exp43_ar_manifest.json"
AR_MANIFEST_SHA256 = "0c109cd2fe0c03381a1d87cee234da5782b0754b4af3385da0917c54e189f12f"
PINS = {
    AR_MANIFEST: AR_MANIFEST_SHA256,
    ARM_CONFIG: "d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d",
    BASE_CONFIG: "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004",
    DATASET_CONFIG: "71f11e80b9b09db754e2ce42f517480a9fb85977f27990f7f01cdbdebfc9b242",
    "data/AR/train.json": "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03",
    VAE: "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
}
AR_SAMPLE_DIGEST = "a6ec819505de2bf1b196731ad1160aa59ff82735eb18d2ae95e944d48797d703"
# the same 16-item construction over data/AR/unseen_eval.json (read 2026-11-07 23:20 from
# /media/diskstation/yixunhu/FLAC/AcousticRooms; the evaluation per-batch quick gate, code-r1 H7)
AR_EVAL_SAMPLE_DIGEST = "ce890a70df2f81b6b7385b47155812f12ccc55e982681c7ab2df9007477a01dd"
DINO_REPO_DIR = "models--facebook--dinov3-vits16-pretrain-lvd1689m"
DINO_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
DINO_CONFIG_SHA256 = "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b"
# Shared vs private at run time (plan §5.1, Codex r1 finding 11). SHARED ON PURPOSE: the GPU
# handshake directory, the per-GPU-UUID GPU_LOCK_<uuid> files in it, and exp_40's card lock --
# kept at its exp_40 path as documented cross-experiment serialisation, so an exp_40 (eval),
# exp_42 or exp_43 job can never co-occupy an A6000 card (a fixed path, not under any worktree,
# code-r1 H8). Everything else is private to exp_43: receipts, admissions, run trees, W&B
# directories, the checkpoint project, the NAS root, the publication lock.
HANDSHAKE_DIR = "/home/yixunhu/codespace/claude_gpu_yield"
CARDS_LOCK = os.path.expanduser("~/.exp40_cards.lock")
PUBLISH_LOCK = os.path.expanduser("~/.exp43_publish.lock")
RUNTREE_MANIFEST = ".exp43_runtree_manifest.json"
# What the run tree holds: the training closure + this folder (kit, driver, config, AR
# manifest) + the reference config + the E1 fixture (read by the evidence checks).
ARCHIVE_PATHS = ("train.py", "defaults.ini", "src", "data/AR", EXPREL, BASE_CONFIG)
CLOSURE = ("train.py", "defaults.ini", "src", "data/AR", BASE_CONFIG, ARM_CONFIG,
           f"{EXPREL}/exp43_kit.py", f"{EXPREL}/exp43_evidence_driver.py",
           f"{EXPREL}/exp43_train_local.sh", AR_MANIFEST)
LINKS = ("AcousticRooms", VAE)                     # the ONLY non-archive entries of a run tree

# The RF treatment (plan §2): the exp_15 continuous roll d ~ Uniform{0..511} (UNQUANTISED),
# then a horizontal mirror flip with p = 1/2, both from the per-(seed, step, rank) generator.
YAW_BLOCK = {"enabled": True, "img_w": 512, "seed": 42, "flip": True}
YAW_KWARGS = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42, "yaw_aug_flip": True}
BANNER = "yaw_aug ENABLED img_w=512 seed=42 flip=0.5"
HIST_BINS, HIST_WIDTH = 16, 32           # 16 coarse offset bins: bin = d // 32 (512 = 16 x 32)
# exp_43 trains on ONE platform and ONE rung: the 2xRTX A6000 box at 2 ranks x micro 32 (plan
# §1). Every other platform, rung, grammar and CLI path of the exp_40 kit is deleted.
PLATFORM, RUNG = "a6000", "2x32"
RUNGS = {RUNG: (2, 32)}
PLATFORMS = {
    PLATFORM: {"gpu_name": "NVIDIA RTX A6000", "rungs": (RUNG,),
               "python": "/home/yixunhu/miniconda3/envs/flac/bin/python",
               "hf_hub": "/home/yixunhu/.cache/huggingface/hub",
               "modes": ("SMOKE", "DDP_SMOKE", "FULL")},
}
# Pinned on the A6000 box (read 2026-11-07).
ENV_PINS_A6000 = {
    "python": "3.10.20", "torch": "2.7.0", "torchaudio": "2.7.0", "pytorch-lightning": "2.1.0",
    "transformers": "4.57.0", "flash-attn": "2.7.4.post1", "numpy": "1.23.5",
    "ema-pytorch": "0.2.3", "prefigure": "0.0.9", "safetensors": "0.7.0",
    "huggingface-hub": "0.35.3", "einops": "0.8.2", "wandb": "0.15.4", "torchvision": "0.22.0",
    "cuda_runtime": "12.6",
    "cudnn": 90501, "nccl": [2, 26, 2],
}
DRIVER_A6000 = "590.48.01"
SMOKE_STEPS = {"SMOKE": 20, "DDP_SMOKE": 30}
ENDPOINT, CADENCE = 40000, 2500
CKPT_STEPS = tuple(range(CADENCE, ENDPOINT + 1, CADENCE))
PROJECT = "exp43_std_aug"              # train.py --name: the W&B project and the checkpoint project
NAS_ROOT = "/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug"
MIN_FREE_GIB = 30
SHARED_MIN_FREE_MIB = 24576
RUN_ID_RE = re.compile(r"^exp43_RF_a6000_2x32(_r[1-9][0-9]?)?$")


def run_id_ok(run_id):
    """The exp_43 RUN_ID grammar, matched over the WHOLE string (``$`` alone would accept a
    trailing newline)."""
    return isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id) is not None


FORBIDDEN_TRAINING_KEYS = ("cond_method", "canonicalize", "are_lambda", "are_anchor",
                           "frame_avg_max_fwd_samples", "frame_avg_angles")
ADAMW = {"lr": 5e-05, "betas": [0.9, 0.999], "weight_decay": 0.001}
SCHED = {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
         "base_lrs": [5e-05], "last_epoch": ENDPOINT, "_step_count": ENDPOINT + 1}
# Audit fix 2026-11-11 (Yixun's go; no Codex round left under the cap): the trainable parameters
# that never receive a gradient in the FLAC forward pass, so AdamW never creates state for
# them. The _r1 40k checkpoint and the exp_20 references YAW_40k / P1_40k lack state for exactly
# these seven (indices 174, 175, 179, 392, 393, 456, 457), bit-identical from step 2,500 to 40,000.
NO_GRAD_PARAMS = frozenset((
    "conditioner.conditioners.source.proj_out.weight",
    "conditioner.conditioners.source.proj_out.bias",
    "conditioner.conditioners.source_vit.vit.embeddings.mask_token",
    "conditioner.conditioners.context_poses.proj_out.weight",
    "conditioner.conditioners.context_poses.proj_out.bias",
    "conditioner.conditioners.context_audio.net.cnn.fc_backup.weight",
    "conditioner.conditioners.context_audio.net.cnn.fc_backup.bias",
))


def scheduled_lr(step):
    """InverseLR's closed form (src/training/utils.py) at ``last_epoch = step`` for ADAMW / SCHED:
    the lr a checkpoint's param group holds after ``step`` scheduler steps (audit fix 2026-11-11:
    the group's ``lr`` is the SCHEDULED value, its ``initial_lr`` the base lr)."""
    warm = 1 - SCHED["warmup"] ** (step + 1)
    return warm * max(SCHED["final_lr"], ADAMW["lr"] * (1 + step / SCHED["inv_gamma"]) ** -SCHED["power"])
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
    """Unique temp -> fsync -> rename -> parent-dir fsync (code-r1 B2: the rename itself is
    durable before the caller acts on it)."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise
    fsync_path(d, directory=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def validate_identity(platform, rung, mode, run_id):
    """exp_43: exactly platform ``a6000``, rung ``2x32``, a mode of that platform and a RUN_ID
    of the exp_43 grammar; anything else (every neuronic identity included) is refused."""
    if platform != PLATFORM:
        return [f"platform {platform!r} is not {PLATFORM!r}: exp_43 trains on the 2xA6000 box only"]
    reasons = []
    if rung != RUNG:
        reasons.append(f"rung {rung!r} is not allowed on {platform} (allowed: [{RUNG!r}])")
    if mode not in PLATFORMS[PLATFORM]["modes"]:
        reasons.append(f"mode {mode!r} is not one of {list(PLATFORMS[PLATFORM]['modes'])} on {platform}")
    if not run_id_ok(run_id):
        reasons.append(f"run id {run_id!r} does not match {RUN_ID_RE.pattern}")
    return reasons


def is_smoke(mode):
    return mode in ("SMOKE", "DDP_SMOKE")


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
    """The literal train.py argv (plan §4.1). FULL / DDP_SMOKE share every recipe
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


def ar_sample_digest(tree, split_rel="data/AR/train.json", ar_root=None):
    """16 deterministic items of a split x {IR wav, depth npy, metadata json} (plan §4.2).
    Training pins the train split (AR_SAMPLE_DIGEST); evaluation pins the eval split
    (AR_EVAL_SAMPLE_DIGEST, code-r1 H7)."""
    with open(os.path.join(tree, split_rel)) as f:
        split = json.load(f)
    root = ar_root if ar_root is not None else os.path.join(tree, "AcousticRooms")
    items = sorted((t, r, x) for t in split for r in split[t] for x in split[t][r])
    h = hashlib.sha256()
    for t, r, x in (items[i * len(items) // 16] for i in range(16)):
        stem = x.split(".")[0]
        src, rec = int(stem.split("_")[0][1:]), int(stem.split("_")[1][1:])
        for rel in (f"single_channel_ir_1/{t}/{r}/{x}", f"depth_map/{t}/{r}/{rec}.npy",
                    f"metadata/{t}/{r}/S00{src}_R00{rec}.json"):
            h.update(f"{rel} {sha256_file(os.path.join(root, rel))}\n".encode())
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


AR_RECEIPT_KEYS = {"verdict", "manifest_digest", "got_digest", "ar_root", "t", "seconds",
                   "differing_rooms"}


def ar_verify_reasons(receipt_path_, manifest, ar_root, now=None, max_age_s=48 * 3600):
    """FULL / eval-freeze gate (code-r1 M3/H7): a PASS full content re-hash of THIS manifest
    over THIS AcousticRooms root, 0 <= age <= 48 h, exact receipt schema. The one validator
    used by the FULL gate (``check``) and the ``ar-receipt-check`` CLI."""
    try:
        rec = read_json(receipt_path_)
    except (OSError, ValueError) as err:
        return [f"no readable full AcousticRooms re-hash receipt {receipt_path_} ({err}); run "
                "exp43_kit.py ar-verify first"]
    if not isinstance(rec, dict) or set(rec) != AR_RECEIPT_KEYS:
        return [f"AR re-hash receipt {receipt_path_} keys {sorted(rec) if isinstance(rec, dict) else rec!r} "
                f"!= {sorted(AR_RECEIPT_KEYS)}"]
    reasons = []
    if digest_obj(manifest.get("rooms")) != manifest.get("digest"):
        reasons.append("the pinned AcousticRooms manifest is internally inconsistent")
    if rec["verdict"] != "PASS" or rec["differing_rooms"] != [] \
            or rec["manifest_digest"] != manifest.get("digest") or rec["got_digest"] != manifest.get("digest"):
        reasons.append(f"AR re-hash receipt {receipt_path_} is not a PASS of manifest "
                       f"{str(manifest.get('digest'))[:12]}")
    if rec["ar_root"] != os.path.realpath(ar_root):
        reasons.append(f"AR re-hash receipt root {rec['ar_root']} != {os.path.realpath(ar_root)}")
    t = rec["t"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        reasons.append(f"AR re-hash receipt time {t!r} is not a number")
    else:
        age = (time.time() if now is None else now) - t
        if not 0 <= age <= max_age_s:
            reasons.append(f"AR re-hash receipt {receipt_path_} age {age / 3600:.1f} h is outside "
                           f"[0, {max_age_s // 3600}] h")
    return reasons


def driver_reasons(platform, driver_version):
    """code-r1 H4: the A6000 NVIDIA driver is pinned (exp_43 has no other platform)."""
    if platform != PLATFORM:
        return [f"platform {platform!r} is not {PLATFORM!r}"]
    if not driver_version:
        return ["the NVIDIA driver version was not measured (EXP43_DRIVER_VERSION)"]
    if driver_version != DRIVER_A6000:
        return [f"NVIDIA driver {driver_version!r} != pinned {DRIVER_A6000!r}"]
    return []


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


def sealed_env(sealed, base):
    """``base`` with every competing HF / transformers cache variable removed and the sealed
    hub + offline flags set (code-r3 B1)."""
    e = {k: v for k, v in base.items() if not k.startswith(("HF_", "TRANSFORMERS_", "HUGGINGFACE_"))}
    e.update(hf_env(sealed))
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.pop("PYTHONPATH", None)
    return e


def dino_resolved_paths():
    """Where THIS process's huggingface_hub resolves the pinned DINOv3 files."""
    from huggingface_hub import try_to_load_from_cache as t
    return [t(DINO_REPO_ID, "model.safetensors"), t(DINO_REPO_ID, "config.json")]


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
    with open(os.path.join(tree, BASE_CONFIG)) as f:          # plan §3.5: the ONLY difference
        want = json.load(f)
    try:
        want["training"]["yaw_aug"] = dict(want["training"]["yaw_aug"], flip=True)
    except (KeyError, TypeError):
        reasons.append("the base config has no training.yaw_aug block")
    else:
        if canonical(want) != canonical(cfg):
            reasons.append("the arm config is not the exp_15 YAW config with exactly training.yaw_aug.flip "
                           "= true added")
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


FIT_START_KEYS = frozenset((
    "kind", "rank", "world_size", "local_rank", "gpu_name", "gpu_uuid", "syncbn_modules",
    "batchnorm_modules", "yaw_aug", "flash_attn_func_is_none", "state_signature", "param_signature",
    "rng_signature", "global_step", "numerics"))
NUMERICS_KEYS = frozenset(("matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
                           "cudnn_deterministic", "float32_matmul_precision"))
SUMMARY_KEYS = frozenset((
    "rank", "counts", "stream_digest", "all_losses_finite", "offset_histogram", "n_offsets", "n_steps",
    "steps_contiguous_from_0", "sampled_offsets", "sampled_flips", "flips_applied", "sampled_losses",
    "global_step", "pid"))
RNG_KEYS = frozenset(("python", "numpy", "torch_cpu", "torch_cuda", "sobol"))     # the GPU producer's set
COUNT_KEYS = frozenset(("flash_calls", "ckpt_dinov3", "ckpt_dit", "ckpt_unclassified", "ckpt_reentrant",
                        "ckpt_default", "apply_yaw_aug", "training_step", "checkpoints_published",
                        # the mirror spy (exp43_evidence_driver.py): every mirror_scene_metadata call,
                        # the calls not attributable to exactly one sample of the step being augmented
                        # (or a batch returned without exactly those mirrored samples), and the steps
                        # whose applied (offsets, flip bits) differ from the private recomputation
                        "mirror_calls", "flip_anomalies", "stream_mismatch_steps"))
STREAM_FIELDS = (("stream_digest", "digest"), ("offset_histogram", "hist"), ("n_offsets", "n"),
                 ("flips_applied", "flips"), ("sampled_offsets", "sampled"), ("sampled_flips", "sampled_flips"))


def stream_mismatch(summary, expected):
    """The summary's stream fields that differ from the recomputation ``expected`` (type-strict:
    compared as canonical JSON, so ``True`` is not ``1``)."""
    return [k for k, ek in STREAM_FIELDS if canonical(summary.get(k)) != canonical(expected[ek])]


def _int(x):
    return type(x) is int


def nested_evidence_reasons(fits, sums, platform, rung, mode):
    """code-r2 B3: the EXACT, type-strict producer schema of every rank's ``fit_start`` record
    and summary (exp43_evidence_driver.py), for DDP_SMOKE / FULL."""
    world, micro = world_of(mode, rung), RUNGS[rung][1]
    steps = ENDPOINT if mode == "FULL" else SMOKE_STEPS[mode]
    ranks = [str(i) for i in range(world)]
    if not isinstance(fits, dict) or sorted(fits) != ranks or not isinstance(sums, dict) or sorted(sums) != ranks:
        return [f"fit_start / summaries are not exactly ranks {ranks}"]
    r, uuids = [], []
    sampled_keys = sorted(str(t) for t in SAMPLED_STEPS if t < steps)
    for k in ranks:
        i, f, s_ = int(k), fits[k], sums[k]
        if not isinstance(f, dict) or set(f) != FIT_START_KEYS:
            r.append(f"rank {k}: fit_start keys {sorted(f) if isinstance(f, dict) else f!r} != the exact schema")
        else:
            for key, ok in (("kind", f["kind"] == "fit_start"), ("rank", _int(f["rank"]) and f["rank"] == i),
                            ("world_size", _int(f["world_size"]) and f["world_size"] == world),
                            ("local_rank", _int(f["local_rank"]) and f["local_rank"] == i),
                            ("gpu_name", f["gpu_name"] == PLATFORMS[platform]["gpu_name"]),
                            ("gpu_uuid", isinstance(f["gpu_uuid"], str) and _UUID_RE.fullmatch(f["gpu_uuid"]) is not None),
                            ("syncbn_modules", _int(f["syncbn_modules"]) and _int(f["batchnorm_modules"])
                             and f["syncbn_modules"] > 0 and f["syncbn_modules"] == f["batchnorm_modules"]),
                            ("yaw_aug", canonical(f["yaw_aug"]) == canonical(YAW_BLOCK)),
                            ("flash_attn_func_is_none", f["flash_attn_func_is_none"] is False),
                            ("global_step", _int(f["global_step"]) and f["global_step"] == 0),
                            ("state_signature", _is_sha(f["state_signature"])),
                            ("param_signature", _is_sha(f["param_signature"])),
                            ("rng_signature", isinstance(f["rng_signature"], dict)
                             and set(f["rng_signature"]) == RNG_KEYS
                             and all(_is_sha(v) for v in f["rng_signature"].values())),
                            ("numerics", isinstance(f["numerics"], dict) and set(f["numerics"]) == NUMERICS_KEYS
                             and all(type(f["numerics"][x]) is bool for x in NUMERICS_KEYS - {"float32_matmul_precision"})
                             and isinstance(f["numerics"]["float32_matmul_precision"], str))):
                if not ok:
                    r.append(f"rank {k}: fit_start {key} = {f[key]!r} is not the exp_43 {mode} start")
            if isinstance(f["gpu_uuid"], str):
                uuids.append(f["gpu_uuid"])
        if not isinstance(s_, dict) or set(s_) != SUMMARY_KEYS:
            r.append(f"rank {k}: summary keys {sorted(s_) if isinstance(s_, dict) else s_!r} != the exact schema")
            continue
        c = s_["counts"]
        if not isinstance(c, dict) or set(c) != COUNT_KEYS or not all(_int(v) for v in c.values()):
            r.append(f"rank {k}: counters {c!r} are not exactly {sorted(COUNT_KEYS)} as ints")
        else:
            for key, want in (("apply_yaw_aug", steps), ("training_step", steps), ("ckpt_reentrant", 0),
                              ("ckpt_default", 0), ("ckpt_unclassified", 0), ("flip_anomalies", 0),
                              ("stream_mismatch_steps", 0),
                              ("checkpoints_published", len(CKPT_STEPS) if (mode == "FULL" and i == 0) else 0)):
                if c[key] != want:
                    r.append(f"rank {k}: {key} = {c[key]}, need {want}")
            for key in ("flash_calls", "ckpt_dinov3", "ckpt_dit"):
                if c[key] <= 0:
                    r.append(f"rank {k}: {key} = {c[key]} (need > 0)")
        if not (_int(s_["rank"]) and s_["rank"] == i):
            r.append(f"rank {k}: summary of rank {s_['rank']!r}")
        if s_["all_losses_finite"] is not True:
            r.append(f"rank {k}: all_losses_finite = {s_['all_losses_finite']!r}")
        if not (_int(s_["global_step"]) and _int(s_["n_steps"]) and s_["global_step"] == steps
                and s_["n_steps"] == steps and s_["steps_contiguous_from_0"] is True):
            r.append(f"rank {k}: not {steps} contiguous steps")
        h = s_["offset_histogram"]
        if not (_int(s_["n_offsets"]) and s_["n_offsets"] == steps * micro and isinstance(h, list)
                and len(h) == HIST_BINS and all(_int(x) and x >= 0 for x in h) and sum(h) == s_["n_offsets"]):
            r.append(f"rank {k}: offset count / {HIST_BINS}-bin histogram inconsistent ({s_['n_offsets']!r})")
        if not _is_sha(s_["stream_digest"]):
            r.append(f"rank {k}: stream_digest is not a sha256")
        so, sl, sf = s_["sampled_offsets"], s_["sampled_losses"], s_["sampled_flips"]
        if not (isinstance(so, dict) and sorted(so) == sampled_keys and all(
                isinstance(v, list) and len(v) == micro and all(_int(d) and 0 <= d < YAW_BLOCK["img_w"] for d in v)
                for v in so.values())):
            r.append(f"rank {k}: sampled offsets are not {micro} column offsets in "
                     f"[0, {YAW_BLOCK['img_w']}) at steps {sampled_keys}")
        if not (isinstance(sf, dict) and sorted(sf) == sampled_keys and all(
                isinstance(v, list) and len(v) == micro and all(_int(b) and b in (0, 1) for b in v)
                for v in sf.values())):
            r.append(f"rank {k}: sampled flips are not {micro} 0/1 bits at steps {sampled_keys}")
        fa = s_["flips_applied"]                       # plan §5.1 / §8 L4: applied-flip evidence
        if not (_int(fa) and 0 < fa <= steps * micro and isinstance(c, dict) and c.get("mirror_calls") == fa):
            r.append(f"rank {k}: flips_applied = {fa!r} is not a positive count of mirrored samples equal "
                     f"to the mirror calls {c.get('mirror_calls') if isinstance(c, dict) else None!r}")
        try:
            losses_ok = isinstance(sl, dict) and sorted(sl) == sampled_keys and all(
                isinstance(v, str) and math.isfinite(float.fromhex(v)) for v in sl.values())
        except (TypeError, ValueError):
            losses_ok = False
        if not losses_ok:
            r.append(f"rank {k}: sampled losses are not finite float.hex strings at steps {sampled_keys}")
        if not _int(s_["pid"]):
            r.append(f"rank {k}: pid {s_['pid']!r}")
    if len(set(uuids)) != len(uuids):
        r.append(f"two ranks report the same GPU {uuids}")
    if all(isinstance(fits[k], dict) for k in ranks):
        for key in SIGNATURE_KEYS:
            if len({canonical(fits[k].get(key)) for k in ranks}) != 1:
                r.append(f"initial {key} differs across ranks")
    return r


def receipt_reasons(rec, kind, platform, rung, sha, tree_dig, env_dig, full_projection):
    """code-r2 B3: the EXACT DDP_SMOKE receipt (the evidence CLI's output) -- every produced
    acceptance field and the nested per-rank records. Used by the FULL gate and the
    FULL-evidence validator. exp_43 has no other receipt kind."""
    mode = {"ddp_smoke": "DDP_SMOKE"}.get(kind)
    if mode is None:
        return [f"receipt kind {kind!r} is not ddp_smoke"]
    want_keys = EVIDENCE_KEYS
    if not isinstance(rec, dict) or set(rec) != want_keys:
        got = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
        return [f"{kind} receipt keys {got} != the exact schema"]
    reasons = []
    want = {"verdict": "PASS", "kind": kind, "mode": mode, "platform": platform, "rung": rung, "sha": sha,
            "tree_digest": tree_dig, "env_digest": env_dig, "recipe_projection": full_projection,
            "evidence": {"verdict": "PASS", "reasons": []}, "reference_sha256": None}
    for key, val in want.items():
        if rec[key] != val:
            reasons.append(f"{kind} receipt field {key} = {rec[key]!r}, need {val!r}")
    if not run_id_ok(rec["run_id"]) or (platform, rung) != (PLATFORM, RUNG):
        reasons.append(f"{kind} receipt run_id {rec['run_id']!r} is not a {platform} {rung} RUN_ID")
    if not (isinstance(rec["stamp"], str) and re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", rec["stamp"])):
        reasons.append(f"{kind} receipt stamp {rec['stamp']!r}")
    for key in ("train_rc", "watcher_rc", "publisher_rc"):
        if not _int(rec[key]) or rec[key] != 0:
            reasons.append(f"{kind} receipt {key} = {rec[key]!r}, need 0")
    if not _is_sha(rec["check_sha256"]):
        reasons.append(f"{kind} receipt check_sha256 is not a sha256")
    argv = rec["argv"]
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)) or recipe_projection(argv) != full_projection:
        reasons.append(f"{kind} receipt argv is not the FULL recipe")
    else:
        i = argv.index("--max-steps") if "--max-steps" in argv else -1
        if i < 0 or argv[i + 1] != str(SMOKE_STEPS[mode]):
            reasons.append(f"{kind} receipt argv --max-steps is not {SMOKE_STEPS[mode]}")
    reasons += [f"{kind} receipt: {x}" for x in nested_evidence_reasons(rec["fit_start"], rec["summaries"],
                                                                       platform, rung, mode)]
    return reasons


def receipt_stream_reasons(rec, tree, mode, rung):
    """code-r3 H1: a DDP_SMOKE receipt's per-rank stream digest (offsets AND applied flip bits),
    histogram, offset and flip counts and sampled offsets / flips must equal the counter-based
    RF stream recomputed in ``tree``."""
    world, micro = world_of(mode, rung), RUNGS[rung][1]
    expected = expected_stream(tree, SMOKE_STEPS[mode], range(world), micro, verbatim=False)
    r = []
    for k in range(world):
        s_, e = (rec.get("summaries") or {}).get(str(k)) or {}, expected[k]
        bad = stream_mismatch(s_, e)
        if bad:
            r.append(f"{mode} receipt rank {k}: {', '.join(bad)} != the counter-based RF recomputation")
    return r


# --------------------------------------------------------------------------- #
# check: every no-GPU gate (plan §4.3)
# --------------------------------------------------------------------------- #
def check(platform, rung, mode, run_id, sha, repo, tree, links, save_root, stamp,
          receipts_dir, hf_sealed, env=None, nas_root=NAS_ROOT):
    """Every no-GPU gate of one launch, from the run tree's own kit: identity, HEAD == EXPECT_SHA
    with a clean closure, the run tree, pins, data, sealed hub, environment, driver, config
    contract, prefigure parse, fresh namespace, disk; FULL also the AR re-hash receipt, a fresh
    NAS destination, the NAS capability receipt and the DDP_SMOKE receipt of this commit."""
    env = dict(os.environ) if env is None else env
    reasons = validate_identity(platform, rung, mode, run_id)
    if reasons:
        return reasons, {}
    info = {"sha": sha}
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        return [f"EXPECT_SHA must be a full 40-hex commit, got {sha!r}"], info
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
    info["links"] = dict(links)                     # admission re-derives tree_digest (code-r2 H2)
    reasons += pin_reasons(tree)
    if ar_sample_digest(tree) != AR_SAMPLE_DIGEST:
        reasons.append("the AcousticRooms sample digest differs from the pinned "
                       f"{AR_SAMPLE_DIGEST}: wrong or modified dataset behind the link")
    if mode != "SMOKE":
        man_path = os.path.join(tree, AR_MANIFEST)
        if not os.path.isfile(man_path):
            reasons.append(f"{AR_MANIFEST} is not committed: the full AcousticRooms manifest is "
                           "required for DDP_SMOKE / FULL")
        else:
            man = read_json(man_path)
            if digest_obj(man["rooms"]) != man["digest"]:
                reasons.append(f"{AR_MANIFEST} is internally inconsistent")
            else:
                reasons += ar_counts_reasons(os.path.join(tree, "AcousticRooms"),
                                             split_rooms(os.path.join(tree, "data/AR/train.json")),
                                             man["rooms"])
                if mode == "FULL":
                    reasons += ar_verify_reasons(os.path.join(receipts_dir, f"ar_verify_{platform}.json"),
                                                 man, os.path.join(tree, "AcousticRooms"))
    reasons += hf_sealed_reasons(hf_sealed)
    reasons += hf_resolution_reasons(hf_sealed, PLATFORMS[platform]["python"], env)
    for k, v in hf_env(hf_sealed).items():
        if env.get(k) != v:
            reasons.append(f"{k} must be {v!r} (the sealed hub), got {env.get(k)!r}")
    measured = measure_environment(tree, PLATFORMS[platform]["python"], env)
    info["environment"] = measured
    reasons += environment_reasons(platform, measured, tree, env)
    reasons += driver_reasons(platform, env.get("EXP43_DRIVER_VERSION"))
    info["env_digest"] = environment_digest(measured, env.get("EXP43_DRIVER_VERSION"))
    info["driver_version"] = env.get("EXP43_DRIVER_VERSION")        # code-r3 H3: re-derivable digest
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
        if os.path.lexists(os.path.join(nas_root, run_id)):
            reasons.append(f"NAS cadence directory {os.path.join(nas_root, run_id)} exists")
        # code-r1 H2: the real-NAS primitive probe, <= 48 h old
        reasons += nas_receipt_reasons(os.path.join(receipts_dir, "nas_capability_a6000.json"),
                                       dest_root=nas_root)
        kind = "ddp_smoke"
        path = os.path.join(receipts_dir, receipt_name(kind, platform, rung, sha))
        try:
            rec = read_json(path)
        except (OSError, ValueError) as err:
            reasons.append(f"no readable {kind} receipt {path} ({err})")
        else:
            rr_ = receipt_reasons(rec, kind, platform, rung, sha, info["tree_digest"],
                                  info["env_digest"], recipe_projection(argv))
            reasons += rr_
            if not rr_:
                reasons += receipt_stream_reasons(rec, tree, "DDP_SMOKE", rung)
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
    if platform != PLATFORM or rung != RUNG or mode not in PLATFORMS[PLATFORM]["modes"]:
        return [f"GPU gate for {platform!r} / {rung!r} / {mode!r}: not an exp_43 launch"]
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


_UUID_RE = re.compile(r"(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
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
                    return "fail", "a training step completed before the RF banner"
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


def group_members(pgid):
    """Live (non-zombie) processes of process group ``pgid`` from /proc: a zombie has exited
    and holds nothing; it only awaits its parent's wait()."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            out.append(int(name))
    return out


def group_alive(pgid):
    return bool(group_members(pgid))


def terminate_group(pgid, grace_s=120.0, kill_wait_s=30.0, poll_s=0.5):
    """SIGTERM the OWNED process group ``pgid``, wait up to ``grace_s`` for it to vanish,
    escalate to SIGKILL of that same group only, wait again. True iff the group is gone
    (code-r1 B4). Refuses the caller's own group."""
    if pgid <= 1 or pgid == os.getpgrp():
        raise Refused(f"refusing to signal process group {pgid} (own group or invalid)")
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, kill_wait_s)):
        if not group_alive(pgid):
            return True
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if not group_alive(pgid):
                return True
            time.sleep(poll_s)
    return not group_alive(pgid)


def watch(log, members, pid, pgid, timeout_s, poll_s=5.0, grace_s=120.0):
    """0 PASS; 8 FAIL / 9 timeout (after terminating OUR process group ``pgid``: SIGTERM,
    bounded wait, SIGKILL escalation; 11 if the group still survives); 10 the training
    process ended first. ``pgid`` must be the session leader the launcher created with
    ``setsid`` -- never this watcher's own group."""
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
            gone = terminate_group(pgid, grace_s=grace_s)
            print(f"process group {pgid} {'terminated' if gone else 'SURVIVED SIGKILL'}", flush=True)
            if not gone:
                return 11
            return 8 if state == "fail" else 9
        if not _alive(pid):
            print("START GATE: the training process exited before the gate decided", flush=True)
            return 10
        time.sleep(poll_s)


# --------------------------------------------------------------------------- #
# evidence (plan §4.7, §4.3-7)
# --------------------------------------------------------------------------- #
# plan §2.3: per (seed, step, rank) ONE generator; the n offsets d ~ Uniform{0..511} are drawn
# FIRST (the unchanged exp_15 draw, unquantised), then the n flip bits by
# torch.randint(0, 2, (n,), generator=g, dtype=torch.long) from the SAME generator. The digest is
# sha256 over "<step>:<d0>,<d1>,...|<b0>,<b1>,...\n"; the histogram has 16 coarse bins d // 32.
_STREAM_SNIPPET = r"""
import hashlib, json, os, sys
sys.path.insert(0, os.getcwd())
import torch
import src.training.diffusion as D
from src.data import yaw_rotation as Y
steps, ranks, micro, sampled, verbatim, seed, img_w, bins, width = json.loads(sys.argv[1])
out = {}
for r in ranks:
    h, hist, so, sf, verb, nflip = hashlib.sha256(), [0] * bins, {}, {}, [], 0
    for s in range(steps):
        g = torch.Generator(); g.manual_seed(D._yaw_aug_step_seed(seed, s, r))
        o = Y.draw_yaw_offsets(micro, img_w, g).tolist()
        f = torch.randint(0, 2, (micro,), generator=g, dtype=torch.long).tolist()
        h.update(f"{s}:{','.join(map(str, o))}|{','.join(map(str, f))}\n".encode())
        for d in o:
            hist[d // width] += 1
        nflip += sum(f)
        if s in sampled:
            so[str(s)], sf[str(s)] = o, f
        if verbatim:
            verb.append([o, f])
    out[str(r)] = {"digest": h.hexdigest(), "hist": hist, "n": steps * micro, "flips": nflip,
                   "sampled": so, "sampled_flips": sf, "verbatim": verb}
print(json.dumps(out))
"""
SAMPLED_STEPS = (0, 1, 2, 3, 1000, 2499, 2500, 39997, 39998, 39999)


def expected_stream(tree, steps, ranks, micro, verbatim):
    """Recompute, in the run tree, every rank's RF stream from the counter-based seed: the
    streaming digest over offsets + flip bits, the histogram, the offset and flip counts, the
    sampled steps' offsets and bits and (smokes) every step's ``[offsets, bits]``."""
    arg = json.dumps([steps, list(ranks), micro, list(SAMPLED_STEPS), verbatim, YAW_BLOCK["seed"],
                      YAW_BLOCK["img_w"], HIST_BINS, HIST_WIDTH])
    rc, out, err = run([sys.executable, "-c", _STREAM_SNIPPET, arg], cwd=tree)
    if rc != 0:
        raise Refused(f"stream recomputation failed: {err.strip()[-400:]}")
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
    """The complete rank-private evidence contract (plan §4.3-7 / §4.6 / §14). ``reference`` is
    the DDP_SMOKE receipt a FULL run's initial state must equal."""
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
                          ("ckpt_default", 0), ("ckpt_unclassified", 0), ("flip_anomalies", 0),
                          ("stream_mismatch_steps", 0)):
            if c.get(key) != want:
                reasons.append(f"rank {rank}: {key} = {c.get(key)}, need {want}")
        fa = s.get("flips_applied")                    # plan §5.1 / §8 L4: applied-flip evidence
        if not (_int(fa) and fa > 0 and c.get("mirror_calls") == fa):
            reasons.append(f"rank {rank}: flips_applied = {fa!r}, mirror_calls = {c.get('mirror_calls')!r} "
                           "(need the same positive count of mirrored samples)")
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
        bad = stream_mismatch(s, e)
        if bad:
            reasons.append(f"rank {rank}: {', '.join(bad)} != the counter-based RF recomputation")
        for step, ref in (fixture or {}).items():   # RF's offsets ARE the reference YAW draws (§2.3)
            if step in (s.get("sampled_offsets") or {}) and rank < len(ref) and len(ref[rank]) == micro:
                if s["sampled_offsets"][step] != list(ref[rank]):
                    reasons.append(f"rank {rank} step {step}: offsets != the E1 fixture's reference YAW draws")
        if verbatim:
            offs = [r for r in rows if r.get("kind") == "offsets"]
            if [[r.get("offsets"), r.get("flips")] for r in offs] != e["verbatim"] \
                    or [r.get("step") for r in offs] != list(range(steps)):
                reasons.append(f"rank {rank}: verbatim offsets / flip bits differ from the recomputation")
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
        reasons.append(f"the RF banner appears {lines.count(BANNER)} times (need 1)")
    state, why = start_gate(log_text, world)
    if state != "pass":
        reasons.append(f"start gate {state}: {why}")
    return reasons


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


def publish_saved(path, run_id, timeout_s=1800, dest_root=NAS_ROOT, python=None):
    """Publish ONE saved checkpoint to ``dest_root/run_id`` and acknowledge it in the NAS
    manifest within ``timeout_s`` (else raise: the caller fails the run). The copy runs in its
    OWN process group so a timeout kills it -- no background write outlives the failure
    (code-r1 H2); a killed copy can only leave a staged ``.partial`` file, which admission
    rejects."""
    cmd = [python or sys.executable, os.path.abspath(__file__), "publish-saved", "--path", path,
           "--run-id", run_id, "--dest-root", dest_root]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        gone = terminate_group(proc.pid, grace_s=10.0, kill_wait_s=30.0)
        proc.wait()
        raise Refused(f"publishing {path} exceeded {timeout_s} s; the copy was "
                      f"{'killed' if gone else 'NOT confirmed dead'}") from None
    if proc.returncode != 0:
        raise Refused(f"publishing {path} failed (rc {proc.returncode}): {out.strip()[-600:]}")
    man = read_manifest(os.path.join(dest_root, run_id, "MANIFEST.sha256"))
    want = _saved_sha(path)
    if man.get(os.path.basename(path)) != want:
        raise Refused(f"{path} is not acknowledged in the NAS manifest")
    if sha256_file(os.path.join(dest_root, run_id, os.path.basename(path))) != want:
        raise Refused(f"the NAS copy of {path} does not re-hash to its save marker")


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
        if sha256_file(src) != sha:                  # code-r1 H2: re-hash the source after adoption
            raise Refused(f"{src} changed while an identical NAS orphan was adopted")
        return sha                                   # an orphan of a crash before the manifest
    tmp = os.path.join(dest, f".{name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    shutil.copyfile(src, tmp)
    fsync_path(tmp)
    if sha256_file(tmp) != sha:
        raise Refused(f"the NAS copy {tmp} of {name} does not re-hash to {sha}")
    rename_noreplace(tmp, final)
    fsync_path(dest, directory=True)
    if sha256_file(src) != sha:
        raise Refused(f"{src} changed while it was published")
    return sha


def publish_pass(src_dir, run_id, dest_root=NAS_ROOT, only=None):
    """Publish saved checkpoints to ``dest_root/run_id`` under the local flock: ``only`` (the
    in-run hook: the just-saved checkpoint(s), sha from the ``.saved`` marker)."""
    import fcntl
    if not run_id_ok(run_id):
        raise Refused(f"run id {run_id!r} does not match {RUN_ID_RE.pattern}")
    if not only or isinstance(only, str):
        raise Refused("publish_pass needs only= (a non-empty list of saved checkpoint names)")
    todo = [(n, _saved_sha(os.path.join(src_dir, n))) for n in only]
    dest = os.path.join(dest_root, run_id)
    os.makedirs(dest, exist_ok=True)
    with open(PUBLISH_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        nas_manifest = os.path.join(dest, "MANIFEST.sha256")
        published = read_manifest(nas_manifest)
        done = []
        for name, sha in todo:
            if name in published:
                if published[name] != sha:
                    raise Refused(f"{name} is on the NAS with {published[name]}, expected {sha}")
                # code-r1 H2: an already-manifested name is acknowledged only after its NAS
                # bytes re-hash to the manifest entry
                on_nas = os.path.join(dest, name)
                if os.path.islink(on_nas) or not os.path.isfile(on_nas) or sha256_file(on_nas) != sha:
                    raise Refused(f"NAS {on_nas} is manifested but does not re-hash to {sha}")
                # code-r2 H3: the SOURCE is re-hashed on this acknowledgement path too
                if sha256_file(os.path.join(src_dir, name)) != sha:
                    raise Refused(f"{os.path.join(src_dir, name)} no longer hashes to {sha}")
                continue
            published[name] = publish_one(src_dir, name, dest, known_sha=sha)
            write_manifest(nas_manifest, published)
            done.append(name)
        return done


NAS_PROBE_KEYS = frozenset(("kind", "verdict", "reasons", "dest_root", "dir", "checks", "mount", "t"))
MOUNT_KEYS = frozenset(("source", "fstype", "dev", "findmnt_rc"))


def mount_identity(path):
    rc, out, err = run(["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", path])
    lines = out.strip().splitlines() or [""]      # an autofs direct mount lists autofs, then the fs
    src, _, fstype = lines[-1].strip().partition(" ")
    return {"source": src.strip(), "fstype": fstype.strip(), "dev": os.stat(path).st_dev,
            "findmnt_rc": rc}


def mount_reasons(m, what):
    """An exact, successful mount identity (code-r2 B2/H3)."""
    if not isinstance(m, dict) or set(m) != MOUNT_KEYS:
        return [f"{what}: mount record {m!r} does not have the exact schema {sorted(MOUNT_KEYS)}"]
    r = []
    if m["findmnt_rc"] != 0 or type(m["findmnt_rc"]) is not int:
        r.append(f"{what}: findmnt failed (rc {m['findmnt_rc']!r})")
    if not (isinstance(m["source"], str) and m["source"] and isinstance(m["fstype"], str) and m["fstype"]
            and "\n" not in m["source"] + m["fstype"]):
        r.append(f"{what}: empty or malformed mount source/fstype {m['source']!r} {m['fstype']!r}")
    if type(m["dev"]) is not int:
        r.append(f"{what}: device id {m['dev']!r} is not an int")
    return r


def nas_capability_probe(dest_root, size=4 << 20):
    """Exercise, ON the real NAS destination, every primitive the publication relies on
    (code-r1 H2): O_EXCL create (+ its refusal), staged copy + fsync + re-hash,
    renameat2(RENAME_NOREPLACE) (+ its refusal onto an existing name), directory fsync,
    atomic manifest replace. Leaves its small probe directory in place (never deletes)."""
    d = os.path.join(dest_root, f".capability_probe_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:6]}")
    os.makedirs(d)
    checks, reasons = {}, []

    def step(name, fn):
        try:
            checks[name] = fn()
        except Exception as err:          # noqa: BLE001 -- every failure is a recorded reason
            checks[name] = f"FAILED: {type(err).__name__}: {err}"
            reasons.append(f"{name}: {err}")
    blob = os.urandom(size)
    want = hashlib.sha256(blob).hexdigest()
    step("o_excl_create", lambda: create_exclusive(os.path.join(d, "excl"), {"t": time.time()}) or True)

    def excl_refused():
        try:
            create_exclusive(os.path.join(d, "excl"), {})
        except FileExistsError:
            return True
        raise Refused("a second O_EXCL create of the same name succeeded")
    step("o_excl_refuses_existing", excl_refused)

    def staged():
        tmp = os.path.join(d, ".blob.partial")
        with open(tmp, "wb") as f:
            f.write(blob)
        fsync_path(tmp)
        if sha256_file(tmp) != want:
            raise Refused("staged copy does not re-hash")
        rename_noreplace(tmp, os.path.join(d, "blob"))
        fsync_path(d, directory=True)
        if sha256_file(os.path.join(d, "blob")) != want:
            raise Refused("renamed copy does not re-hash")
        return True
    step("staged_copy_fsync_rehash_noreplace_rename_dirfsync", staged)

    def noreplace_refused():
        with open(os.path.join(d, ".other"), "wb") as f:
            f.write(b"x")
        try:
            rename_noreplace(os.path.join(d, ".other"), os.path.join(d, "blob"))
        except Refused:
            if sha256_file(os.path.join(d, "blob")) != want:
                raise Refused("a refused NOREPLACE rename still changed the target") from None
            return True
        raise Refused("RENAME_NOREPLACE replaced an existing file")
    step("noreplace_refuses_existing", noreplace_refused)

    def manifest():
        write_manifest(os.path.join(d, "MANIFEST.sha256"), {ckpt_name(CADENCE): want})
        write_manifest(os.path.join(d, "MANIFEST.sha256"), {ckpt_name(CADENCE): want, ckpt_name(2 * CADENCE): want})
        if len(read_manifest(os.path.join(d, "MANIFEST.sha256"))) != 2:
            raise Refused("the atomic manifest replace did not read back")
        return True
    step("atomic_manifest_replace", manifest)
    return {"kind": "nas_capability", "verdict": "PASS" if not reasons else "FAIL", "reasons": reasons,
            "dest_root": dest_root, "dir": d, "checks": checks, "mount": mount_identity(dest_root),
            "t": time.time()}


NAS_PROBE_CHECKS = ("o_excl_create", "o_excl_refuses_existing",
                    "staged_copy_fsync_rehash_noreplace_rename_dirfsync", "noreplace_refuses_existing",
                    "atomic_manifest_replace")


def nas_receipt_reasons(path, dest_root=NAS_ROOT, now=None, max_age_s=48 * 3600, current_mount=None):
    """code-r2 H3: exact schema; exactly the five named primitive checks, each True; the
    probe dir directly under THIS destination; an exact, successful mount identity equal to
    the destination's CURRENT mount; 0 <= age <= 48 h."""
    try:
        rec = read_json(path)
    except (OSError, ValueError) as err:
        return [f"no readable NAS capability receipt {path} ({err}); run exp43_kit.py nas-probe"]
    if not isinstance(rec, dict) or set(rec) != NAS_PROBE_KEYS:
        return [f"NAS capability receipt {path} does not have the exact schema"]
    r = []
    if rec["kind"] != "nas_capability" or rec["verdict"] != "PASS" or rec["reasons"] != []:
        r.append("the NAS capability probe did not PASS")
    if not isinstance(rec["checks"], dict) or sorted(rec["checks"]) != sorted(NAS_PROBE_CHECKS) \
            or any(rec["checks"][k] is not True for k in NAS_PROBE_CHECKS):
        r.append(f"the NAS capability checks are not exactly {list(NAS_PROBE_CHECKS)}, each True")
    if rec["dest_root"] != dest_root or os.path.dirname(str(rec["dir"])) != dest_root:
        r.append(f"the NAS capability probe tested {rec['dest_root']}, not {dest_root}")
    mr = mount_reasons(rec["mount"], "the NAS capability probe")
    r += mr
    if not mr:
        cur = current_mount if current_mount is not None else mount_identity(dest_root)
        cr = mount_reasons(cur, "the NAS destination now")
        r += cr
        if not cr and (cur["source"], cur["fstype"]) != (rec["mount"]["source"], rec["mount"]["fstype"]):
            r.append(f"the NAS destination is now on {cur['source']} {cur['fstype']}, the probe tested "
                     f"{rec['mount']['source']} {rec['mount']['fstype']}")
    t = rec["t"]
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        r.append("NAS capability receipt time missing")
    else:
        age = (time.time() if now is None else now) - t
        if not 0 <= age <= max_age_s:
            r.append(f"NAS capability receipt age {age / 3600:.1f} h is outside [0, {max_age_s // 3600}] h")
    return r


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
            if k == "lr":
                continue
            got = list(pg.get(k)) if isinstance(pg.get(k), (tuple, list)) else pg.get(k)
            if got != v:
                reasons.append(f"AdamW {k} {got!r} != {v!r}")
        if pg.get("initial_lr") != ADAMW["lr"]:
            reasons.append(f"AdamW initial_lr {pg.get('initial_lr')!r} != {ADAMW['lr']!r}")
        want, got = scheduled_lr(ENDPOINT), pg.get("lr")
        if isinstance(got, bool) or not isinstance(got, (int, float)) \
                or not math.isclose(got, want, rel_tol=1e-12, abs_tol=0.0):
            reasons.append(f"AdamW lr {got!r} != the scheduled InverseLR lr {want!r} at step {ENDPOINT}")
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
            named = list(wrapper.diffusion.named_parameters())
            params = [p for _, p in named]
            trainable = {i for i, p in enumerate(params) if p.requires_grad}
            exempt = {i for i in trainable if named[i][0] in NO_GRAD_PARAMS}   # audit fix 2026-11-11
            state = opt[0]["state"]
            have = set(map(int, state))
            if have != trainable - exempt or sorted(opt[0]["param_groups"][0]["params"]) != sorted(range(len(params))):
                odd = sorted(named[i][0] if 0 <= i < len(named) else str(i) for i in have ^ (trainable - exempt))
                reasons.append(f"optimizer state covers {len(state)} params, {len(trainable)} trainable, "
                               f"{len(exempt)} never-gradient exempt; differing: {odd[:5]}")
            for idx, st in state.items():
                p = params[int(idx)]
                bad = [k for k in ("exp_avg", "exp_avg_sq")
                       if st.get(k) is None or tuple(st[k].shape) != tuple(p.shape)
                       or not torch.isfinite(st[k]).all()]
                if bad or float(st.get("step", -1)) != float(ENDPOINT):
                    reasons.append(f"optimizer state {idx} malformed ({bad}, step {st.get('step')})")
                    break
    return reasons


def frozen_param_reasons(first, last):
    """Audit fix 2026-11-11: every NO_GRAD_PARAMS tensor present in the online model is
    bit-identical at the first cadence checkpoint and at the endpoint (AdamW never stepped it)."""
    import torch
    a, b = first.get("state_dict") or {}, last.get("state_dict") or {}
    r = []
    for name in sorted(NO_GRAD_PARAMS):
        k = "diffusion." + name
        if (k in a) != (k in b):
            r.append(f"never-gradient parameter {name} is in only one of the step-{CADENCE} / "
                     f"step-{ENDPOINT} checkpoints")
        elif k in a and not (torch.is_tensor(a[k]) and torch.is_tensor(b[k]) and torch.equal(a[k], b[k])):
            r.append(f"never-gradient parameter {name} changed between step {CADENCE} and step {ENDPOINT}")
    return r


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


EVIDENCE_KEYS = frozenset((
    "verdict", "kind", "platform", "rung", "mode", "run_id", "stamp", "sha", "tree_digest",
    "env_digest", "recipe_projection", "argv", "train_rc", "watcher_rc", "publisher_rc",
    "reference_sha256", "evidence", "fit_start", "summaries", "check_sha256"))


def _is_sha(x, n=64):
    return isinstance(x, str) and re.fullmatch(f"[0-9a-f]{{{n}}}", x) is not None


def full_evidence_reasons(path, run_id, check_path=None, reference_path=None, recompute=False, tree=None):
    """The ONE strict FULL-evidence validator (code-r1 B5), used by audit and admission:
    exact schema; identity derived from the RUN_ID; every acceptance field (exit codes,
    evidence verdict, per-rank counts, finite-loss flag, 16 save-time acknowledgements, step
    contiguity); the recipe projection; optionally the binding to the check record and the
    DDP_SMOKE reference receipt (sha256 + identity + initial-state signatures) and a
    re-computation of every rank's 40k-step offset + flip stream IN ``tree`` (admission: the immutable
    tree at the training commit, code-r2 H2). Returns (reasons, sha256, record)."""
    try:
        rec = read_json(path)
        file_sha = sha256_file(path)
    except (OSError, ValueError) as err:
        return [f"FULL evidence receipt {path} unreadable ({err})"], None, None
    if not isinstance(rec, dict) or set(rec) != EVIDENCE_KEYS:
        got = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
        return [f"FULL evidence {path}: keys {got} != the exact schema"], file_sha, None
    r = []
    if not run_id_ok(run_id):
        return [f"run id {run_id!r} is not an exp_43 RUN_ID"], file_sha, None
    platform, rung = PLATFORM, RUNG
    for key, want in (("kind", "full_evidence"), ("verdict", "PASS"), ("mode", "FULL"),
                      ("run_id", run_id), ("platform", platform), ("rung", rung),
                      ("evidence", {"verdict": "PASS", "reasons": []})):
        if rec[key] != want:
            r.append(f"FULL evidence {key} = {rec[key]!r}, need {want!r}")
    for key in ("train_rc", "watcher_rc", "publisher_rc"):
        if type(rec[key]) is not int or rec[key] != 0:
            r.append(f"FULL evidence {key} = {rec[key]!r}, need 0")
    if not _is_sha(rec["sha"], 40):
        r.append("FULL evidence sha is not a 40-hex commit")
    if not (isinstance(rec["stamp"], str) and re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", rec["stamp"])):
        r.append(f"FULL evidence stamp {rec['stamp']!r}")
    for key in ("tree_digest", "env_digest", "reference_sha256", "check_sha256"):
        if not _is_sha(rec[key]):
            r.append(f"FULL evidence {key} is not a sha256")
    argv = rec["argv"]
    if not (isinstance(argv, list) and all(isinstance(x, str) for x in argv)):
        r.append("FULL evidence argv is not a list of strings")
    else:
        want_proj = recipe_projection(train_argv(platform, rung, "FULL", run_id, "/save-root"))
        if rec["recipe_projection"] != recipe_projection(argv) or rec["recipe_projection"] != want_proj:
            r.append("FULL evidence recipe projection is not the exp_43 FULL recipe")
    world, micro = world_of("FULL", rung), RUNGS[rung][1]
    ranks = [str(i) for i in range(world)]
    fits, sums = rec["fit_start"], rec["summaries"]
    nested = nested_evidence_reasons(fits, sums, platform, rung, "FULL")
    r += nested
    if nested and (not isinstance(fits, dict) or sorted(fits) != ranks or not isinstance(sums, dict)
                   or sorted(sums) != ranks):
        return r, file_sha, None
    if recompute and not r:
        expected = expected_stream(tree or ROOT, ENDPOINT, range(world), micro, verbatim=False)
        for k in ranks:
            bad = stream_mismatch(sums[k], expected[int(k)])
            if bad:
                r.append(f"rank {k}: {', '.join(bad)} != the counter-based RF recomputation")
    if check_path is not None:
        try:
            chk = read_json(check_path)
            chk_sha = sha256_file(check_path)
        except (OSError, ValueError) as err:
            r.append(f"check record {check_path} unreadable ({err})")
        else:
            if not isinstance(chk, dict):
                chk = {}
            if chk_sha != rec["check_sha256"]:
                r.append("the check record is not the one this evidence is bound to")
            for key in ("sha", "tree_digest", "env_digest", "argv"):
                if chk.get(key) != rec[key]:
                    r.append(f"check record {key} != the evidence's")
            if chk.get("verdict") != "PASS" or chk.get("reasons") != []:
                r.append("the check record is not a PASS")
    if reference_path is not None:
        kind = "ddp_smoke"
        try:
            ref = read_json(reference_path)
            ref_sha = sha256_file(reference_path)
        except (OSError, ValueError) as err:
            r.append(f"reference receipt {reference_path} unreadable ({err})")
        else:
            if ref_sha != rec["reference_sha256"]:
                r.append("the reference receipt is not the one this evidence is bound to")
            rref = receipt_reasons(ref, kind, platform, rung, rec["sha"], rec["tree_digest"],
                                   rec["env_digest"], rec["recipe_projection"])
            r += rref
            if recompute and not rref:
                r += receipt_stream_reasons(ref, tree or ROOT, "DDP_SMOKE", rung)
            ref_fits = ref.get("fit_start") if isinstance(ref, dict) and isinstance(ref.get("fit_start"), dict) else {}
            for k in ranks:
                mine = fits[k] if isinstance(fits[k], dict) else {}
                for key in SIGNATURE_KEYS:
                    if canonical((ref_fits.get(k) or {}).get(key)) != canonical(mine.get(key)):
                        r.append(f"rank {k}: initial {key} != the {kind} receipt's")
    return r, file_sha, (rec if not r else None)


def audit(ckpt_dir, tree, nas_dir=None, strict_load=True, evidence=None, run_id=None,
          expected_dirpath=None, check_path=None, reference_path=None):
    names = sorted(os.listdir(ckpt_dir))
    reasons = []
    ev_sha = None
    if evidence is not None:
        r, ev_sha, _ = full_evidence_reasons(evidence, run_id, check_path, reference_path)
        reasons += r
    allowed = set(EXPECTED_CKPTS) | {f"{n}.saved" for n in EXPECTED_CKPTS} | {"CHECKPOINTS.sha256"}
    others = [n for n in names if n not in allowed]
    if others:
        reasons.append(f"unexpected files in the checkpoint dir: {others}")
    for n in EXPECTED_CKPTS:
        if n not in names:
            reasons.append(f"{n} is missing")
        elif f"{n}.saved" not in names:
            reasons.append(f"{n} has no save marker")
    shas = {}
    if not reasons:
        for n in EXPECTED_CKPTS:
            shas[n] = sha256_file(os.path.join(ckpt_dir, n))
            if _saved_sha(os.path.join(ckpt_dir, n)) != shas[n]:
                reasons.append(f"{n} does not hash to its save marker")
        final = ckpt_name(ENDPOINT)
        sha, _, ck = load_checkpoint_once(os.path.join(ckpt_dir, final))
        if sha != shas[final]:
            reasons.append("the 40k checkpoint changed while it was audited")
        with open(os.path.join(tree, ARM_CONFIG)) as f:
            arm_cfg = json.load(f)
        wrapper = build_wrapper_cpu(tree, arm_cfg) if strict_load else None
        reasons += checkpoint_state_reasons(ck, arm_cfg, expected_dirpath or ckpt_dir, wrapper)
        first_name = ckpt_name(CADENCE)                                   # audit fix 2026-11-11
        fsha, _, first = load_checkpoint_once(os.path.join(ckpt_dir, first_name))
        if fsha != shas[first_name]:
            reasons.append(f"{first_name} changed while it was audited")
        reasons += frozen_param_reasons(first, ck)
        del first
        if nas_dir is not None:
            reasons += nas_run_dir_reasons(nas_dir, shas)
    if not reasons:
        write_manifest(os.path.join(ckpt_dir, "CHECKPOINTS.sha256"), shas)
    return {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons,
            "ckpt_dir": os.path.abspath(ckpt_dir), "sha256": shas, "strict_load": strict_load,
            "nas_dir": nas_dir, "full_evidence_sha256": ev_sha,
            "expected_dirpath": expected_dirpath or os.path.abspath(ckpt_dir)}


def nas_run_dir_reasons(nas_dir, shas):
    """The NAS run dir holds EXACTLY the 16 checkpoints + MANIFEST.sha256 (no extra, no
    partial), the manifest equals ``shas`` and every file re-hashes to it (code-r1 B6)."""
    try:
        names = sorted(os.listdir(nas_dir))
    except OSError as err:
        return [f"NAS run dir {nas_dir} unreadable ({err})"]
    want = sorted(list(EXPECTED_CKPTS) + ["MANIFEST.sha256"])
    if names != want:
        return [f"NAS run dir holds {sorted(set(names) ^ set(want))} beyond / short of the 16 "
                "checkpoints + MANIFEST.sha256"]
    r = []
    if read_manifest(os.path.join(nas_dir, "MANIFEST.sha256")) != shas:
        r.append("the NAS MANIFEST.sha256 differs from the audited checkpoint hashes")
    for name, want_sha in sorted(shas.items()):
        if sha256_file(os.path.join(nas_dir, name)) != want_sha:
            r.append(f"NAS {name} does not re-hash to the audited sha256")
    return r


RUN_RESULT_KEYS = frozenset(("kind", "run_id", "platform", "rung", "mode", "sha", "verdict",
                             "files", "expected_dirpath", "ckpt_dir", "nas_dir", "runtree", "t"))
RUN_RESULT_FILES = ("check", "reference", "evidence", "audit")


def run_result(run_id, sha, verdict, files, expected_dirpath, ckpt_dir, runtree, nas_root=NAS_ROOT):
    """The launcher's stable FULL run-result manifest (code-r1 M6): names + sha256 of the
    check, reference, evidence and audit records; the training-time checkpoint dirpath."""
    if not run_id_ok(run_id):
        raise Refused(f"run id {run_id!r}")
    if sorted(files) != sorted(RUN_RESULT_FILES):
        raise Refused(f"run result files {sorted(files)} != {sorted(RUN_RESULT_FILES)}")
    return {"kind": "exp40_run_result", "run_id": run_id, "platform": PLATFORM, "rung": RUNG,
            "mode": "FULL", "sha": sha, "verdict": verdict,
            "files": {k: {"name": os.path.basename(p), "sha256": sha256_file(p)} for k, p in files.items()},
            "expected_dirpath": expected_dirpath, "ckpt_dir": ckpt_dir,
            "nas_dir": os.path.join(nas_root, run_id), "runtree": runtree, "t": time.time()}


def resolve_run_result(path, receipts_dir):
    """Read a run-result manifest and resolve every named record in ``receipts_dir`` by name,
    verifying its sha256."""
    rr = read_json(path)
    if not isinstance(rr, dict) or set(rr) != RUN_RESULT_KEYS or rr["kind"] != "exp40_run_result" \
            or rr["mode"] != "FULL" or sorted(rr["files"]) != sorted(RUN_RESULT_FILES):
        raise Refused(f"{path} is not an exp_40 FULL run-result manifest")
    if not run_id_ok(rr["run_id"]) or (rr["platform"], rr["rung"]) != (PLATFORM, RUNG):
        raise Refused(f"run result names {rr['run_id']!r} on {rr['platform']!r} / {rr['rung']!r}, not an "
                      f"exp_43 {PLATFORM} {RUNG} run")
    want = "TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION"
    if rr["verdict"] != want:
        raise Refused(f"run result verdict {rr['verdict']!r}, need {want!r}")
    out = {}
    for k, ent in rr["files"].items():
        p = os.path.join(receipts_dir, ent["name"])
        if os.path.basename(ent["name"]) != ent["name"] or not os.path.isfile(p) \
                or sha256_file(p) != ent["sha256"]:
            raise Refused(f"run-result record {k} ({p}) is missing or not the recorded bytes")
        out[k] = p
    return rr, out


CURRENT_ASSET_KEYS = frozenset(("acousticrooms_manifest_digest", "dinov3_revision", "dinov3_config_sha256",
                                "dinov3_weights_sha256", "vae_sha256", "hf_sealed_hub", "dinov3_resolved"))


def current_assets(hf_sealed, resolved, tree=None):
    """The admission's asset identities (code-r3 B1/M2): the committed AR manifest, the pinned
    DINOv3 / VAE bytes, and the sealed hub the strict load actually resolved DINOv3 from."""
    man = read_json(os.path.join(tree or ROOT, AR_MANIFEST))
    return {"acousticrooms_manifest_digest": man["digest"], "dinov3_revision": DINO_REVISION,
            "dinov3_config_sha256": DINO_CONFIG_SHA256, "dinov3_weights_sha256": DINO_SHA256,
            "vae_sha256": PINS[VAE], "hf_sealed_hub": os.path.realpath(hf_sealed),
            "dinov3_resolved": [os.path.realpath(x) for x in resolved]}


def current_assets_reasons(ca):
    if not isinstance(ca, dict) or set(ca) != CURRENT_ASSET_KEYS:
        return [f"current_assets keys {sorted(ca) if isinstance(ca, dict) else ca!r} != {sorted(CURRENT_ASSET_KEYS)}"]
    r = []
    try:
        man_digest = read_json(os.path.join(ROOT, AR_MANIFEST))["digest"]
    except (OSError, ValueError, KeyError):
        man_digest = None
    for key, want in (("acousticrooms_manifest_digest", man_digest), ("dinov3_revision", DINO_REVISION),
                      ("dinov3_config_sha256", DINO_CONFIG_SHA256), ("dinov3_weights_sha256", DINO_SHA256),
                      ("vae_sha256", PINS[VAE])):
        if want is None or ca[key] != want:
            r.append(f"current_assets {key} = {ca[key]!r}, not the pinned {want!r}")
    hub = ca["hf_sealed_hub"]
    if not (isinstance(hub, str) and os.path.isabs(hub)):
        r.append(f"current_assets hf_sealed_hub {hub!r} is not an absolute path")
    elif not (isinstance(ca["dinov3_resolved"], list) and len(ca["dinov3_resolved"]) == 2 and all(
            isinstance(x, str) and x.startswith(os.path.join(hub, "hub", DINO_REPO_DIR, "snapshots", DINO_REVISION) + os.sep)
            for x in ca["dinov3_resolved"])):
        r.append(f"current_assets dinov3_resolved {ca['dinov3_resolved']!r} is not the pinned snapshot in {hub}")
    return r


def build_admission_tree(repo, sha, check_rec, ev, dest_parent):
    """code-r2 H2: an IMMUTABLE tree at the training commit for admission. The commit's archive
    plus the TRAINING-time links (from the bound check record) must reproduce the evidence's
    run-tree digest; the admission tree itself gets this box's links (only the AcousticRooms
    and VAE paths differ), is built fresh, read-only, and verified."""
    train_links = check_rec.get("links") if isinstance(check_rec, dict) else None
    if not isinstance(train_links, dict) or set(train_links) != set(LINKS):
        raise Refused("the check record carries no training-time links")
    if tree_digest(repo, sha, train_links) != ev["tree_digest"]:
        raise Refused(f"git archive of {sha[:12]} + the training links do not reproduce the evidence's "
                      "run-tree digest: the evidence was not produced from this commit")
    local = {rel: os.path.realpath(os.path.join(repo, rel)) for rel in LINKS}
    dest = os.path.join(dest_parent, f"admission_{ev['run_id']}__{sha[:12]}__{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    build_runtree(repo, sha, dest, local)
    reasons = verify_runtree(repo, sha, dest, local)
    if reasons:
        raise Refused("admission tree: " + "; ".join(reasons))
    return dest


def admission_state(ckpt_path, expected_dirpath, tree=None, strict_load=True, hf_sealed=None):
    """The 40k checkpoint's full state interpreted by THIS kit's tree (run inside the
    admission tree by admission_state_subprocess), with DINOv3 resolved ONLY from the sealed
    hub (code-r3 B1): the process environment must name exactly that hub, offline."""
    tree = tree or ROOT
    pre = []
    if hf_sealed is None:
        pre.append("admission-state needs the sealed HF hub")
    else:
        for k, v in hf_env(hf_sealed).items():
            if os.environ.get(k) != v:
                pre.append(f"{k} is {os.environ.get(k)!r}, not {v!r} (the sealed hub)")
        stray = sorted(k for k in os.environ if k.startswith(("HF_", "TRANSFORMERS_", "HUGGINGFACE_"))
                       and k not in hf_env(hf_sealed))
        if stray:
            pre.append(f"competing HF variables are set: {stray}")
    hub = os.path.realpath(os.path.join(hf_sealed, "hub")) + os.sep if hf_sealed else None
    resolved = dino_resolved_paths() if hub else []
    if hub and not all(isinstance(x, str) and os.path.realpath(x).startswith(hub) for x in resolved):
        pre.append(f"DINOv3 resolves to {resolved}, outside the sealed hub {hub}")
    if pre:
        return {"sha256": None, "bytes": None, "reasons": pre, "global_step": None,
                "embedded_config_sha256": None, "ema_inventory_sha256": None, "strict_load": strict_load,
                "tree": os.path.realpath(tree), "hf_sealed": hf_sealed, "dino_resolved": resolved}
    st = checkpoint_admission_state(ckpt_path, expected_dirpath, tree, strict_load)
    st.update(hf_sealed=os.path.realpath(hf_sealed), dino_resolved=[os.path.realpath(x) for x in resolved])
    return st


def checkpoint_admission_state(ckpt_path, expected_dirpath, tree, strict_load=True):
    """One read of the checkpoint; its full state against ``tree``'s arm config and (strict)
    a wrapper built from ``tree``'s own source."""
    sha, nbytes, ck = load_checkpoint_once(ckpt_path)
    with open(os.path.join(tree, ARM_CONFIG)) as f:
        arm_cfg = json.load(f)
    reasons = checkpoint_state_reasons(ck, arm_cfg, expected_dirpath,
                                       build_wrapper_cpu(tree, arm_cfg) if strict_load else None)
    ema_inv = digest_obj(sorted((k, list(v.shape), str(v.dtype)) for k, v in ck["state_dict"].items()
                                if k.startswith("diffusion_ema.ema_model.")))
    return {"sha256": sha, "bytes": nbytes, "reasons": reasons, "global_step": ck.get("global_step"),
            "embedded_config_sha256": digest_obj(ck.get("model_config")), "ema_inventory_sha256": ema_inv,
            "strict_load": strict_load, "tree": os.path.realpath(tree)}


def admission_state_subprocess(tree, ckpt_path, expected_dirpath, strict_load=True, python=None,
                               hf_sealed=None):
    """Run ``admission-state`` in the admission tree (cwd = that tree, ``--tree`` = that tree, no
    bytecode): the arm config and the strict-load wrapper's source come from the training commit,
    never from the live checkout (code-r2 H2). Audit fix 2026-11-11: the CHECKS are this (live)
    kit's -- the training commit's kit carries the two false-positive checks fixed here."""
    if hf_sealed is None:
        raise Refused("admission-state needs the sealed HF hub")
    cmd = [python or sys.executable, "-B", os.path.abspath(__file__), "admission-state", "--tree", tree,
           "--ckpt", ckpt_path, "--expected-dirpath", expected_dirpath, "--hf-sealed", hf_sealed] \
        + ([] if strict_load else ["--no-strict"])
    env = sealed_env(hf_sealed, os.environ)                    # code-r3 B1: ONLY the sealed hub
    rc, out, err = run(cmd, cwd=tree, env=env)
    if rc != 0:
        raise Refused(f"admission-state in {tree} failed (rc {rc}): {(err or out).strip()[-600:]}")
    st = json.loads(out.strip().splitlines()[-1])
    if st.get("tree") != os.path.realpath(tree):
        raise Refused(f"admission-state ran from {st.get('tree')}, not the admission tree {tree}")
    return st


HISTORICAL = {"exp17_acousticrooms_snapshot": "unknown", "exp17_dinov3_revision_and_bytes": "unknown",
              "exp17_vae_bytes": "unknown", "exp17_dependency_versions": "unknown",
              "exp17_wandb_behaviour": "unknown", "exp17_launch_commit": "unknown (audited base 41aa31dc)",
              "exp17_attention_backend": "flash-attn (inferred, not proven)"}


def admit(run_result_path, receipts_dir, repo, note="", strict_load=True, nas_root=NAS_ROOT,
          tree_parent=None, tree_builder=build_admission_tree, state_fn=admission_state_subprocess,
          hf_sealed=None):
    """Admission (plan §4.6; code-r1 B6, code-r2 H2): every identity field is DERIVED from
    validated records -- the run result, the strict FULL evidence (bound to its check record
    and its DDP_SMOKE reference), the audit bound to that evidence; all 16 NAS
    checkpoints are re-hashed; the offset recomputation, the arm config and the strict-load
    wrapper come from an immutable tree at the TRAINING commit whose archive reproduces the
    evidence's run-tree digest. The only caller input is a free-text note."""
    if not isinstance(note, str):
        raise Refused("the admission note must be a string")
    if not hf_sealed:                                        # code-r3 B1
        raise Refused("admission needs --hf-sealed (the verified read-only DINOv3 hub)")
    hf_r = hf_sealed_reasons(hf_sealed) + hf_resolution_reasons(hf_sealed, sys.executable, dict(os.environ))
    if hf_r:
        raise Refused("sealed HF hub: " + "; ".join(hf_r))
    rr, files = resolve_run_result(run_result_path, receipts_dir)
    run_id = rr["run_id"]
    r, ev_sha, ev = full_evidence_reasons(files["evidence"], run_id, files["check"], files["reference"])
    if r:
        raise Refused("FULL evidence: " + "; ".join(r))
    if ev["sha"] != rr["sha"]:
        raise Refused("the run result and the FULL evidence name different commits")
    tree = tree_builder(repo, rr["sha"], read_json(files["check"]), ev,
                        tree_parent or os.path.join(repo, "outputs_FLAC", "exp43_std_aug", "runtrees"))
    r, _, _ = full_evidence_reasons(files["evidence"], run_id, files["check"], files["reference"],
                                    recompute=True, tree=tree)
    if r:
        raise Refused("FULL evidence (recomputed in the admission tree): " + "; ".join(r))
    au = read_json(files["audit"])
    if au.get("verdict") != "PASS" or au.get("reasons") != [] or au.get("full_evidence_sha256") != ev_sha \
            or au.get("strict_load") is not True or sorted(au.get("sha256") or {}) != sorted(EXPECTED_CKPTS) \
            or au.get("expected_dirpath") != rr["expected_dirpath"]:
        raise Refused(f"audit {files['audit']} is not a strict PASS bound to the FULL evidence {ev_sha}")
    shas = au["sha256"]
    nas_dir = os.path.join(nas_root, run_id)
    reasons = nas_run_dir_reasons(nas_dir, shas)
    if reasons:
        raise Refused("; ".join(reasons))
    final = ckpt_name(ENDPOINT)
    st = state_fn(tree, os.path.join(nas_dir, final), rr["expected_dirpath"], strict_load=strict_load,
                  hf_sealed=hf_sealed)
    if st["reasons"]:
        raise Refused("; ".join(st["reasons"]))
    if st["sha256"] != shas[final]:
        raise Refused(f"{final} sha256 {st['sha256']} != the audited one")
    if st["reasons"]:
        raise Refused("; ".join(st["reasons"]))
    current = current_assets(hf_sealed, st["dino_resolved"], tree)
    ca_r = current_assets_reasons(current)
    if ca_r:
        raise Refused("admission assets: " + "; ".join(ca_r))
    return {"kind": "exp40_admission", "admitted": True, "arm": "C16", "run_id": run_id,
            "platform": rr["platform"], "rung": rr["rung"], "train_sha": ev["sha"],
            "runtree_digest": ev["tree_digest"], "env_digest": ev["env_digest"], "nas_dir": nas_dir,
            "checkpoints": dict(shas), "manifest_sha256": sha256_file(os.path.join(nas_dir, "MANIFEST.sha256")),
            "ckpt_path": os.path.join(nas_dir, final), "sha256": st["sha256"], "bytes": st["bytes"],
            "global_step": st["global_step"], "model_config": ARM_CONFIG,
            "model_config_sha256": PINS[ARM_CONFIG], "embedded_config_sha256": st["embedded_config_sha256"],
            "ema_inventory_sha256": st["ema_inventory_sha256"], "strict_load": st["strict_load"],
            "full_evidence_sha256": ev_sha, "check_sha256": ev["check_sha256"],
            "audit_sha256": sha256_file(files["audit"]), "historical": dict(HISTORICAL),
            "current_assets": current, "trajectory_equivalence_claim_allowed": False, "note": note,
            "t": time.time()}


def ckpt_name(step):
    """The exact Lightning name of the step's cadence checkpoint (epoch = (step-1) // 4550)."""
    return f"epoch={(step - 1) // STEPS_PER_EPOCH}-step={step}.ckpt"


EXPECTED_CKPTS = tuple(ckpt_name(s) for s in CKPT_STEPS)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
CLOCK_SKEW_S = 300
ADMISSION_KEYS = frozenset((
    "kind", "admitted", "arm", "run_id", "platform", "rung", "train_sha", "runtree_digest",
    "env_digest", "nas_dir", "checkpoints", "manifest_sha256", "ckpt_path", "sha256", "bytes",
    "global_step", "model_config", "model_config_sha256", "embedded_config_sha256",
    "ema_inventory_sha256", "strict_load", "full_evidence_sha256", "check_sha256", "audit_sha256",
    "historical", "current_assets", "trajectory_equivalence_claim_allowed", "note", "t"))


def admission_reasons(path, nas_root=NAS_ROOT, rehash="final"):
    """The ONE admission validator (code-r1 B6/B10), shared by the evaluation driver, collect,
    figure and table publication. ``rehash``: "final" re-hashes the 40k checkpoint, "all" also
    re-hashes every cadence checkpoint and MANIFEST.sha256 and rejects extra/partial files,
    None checks the record only. Returns (reasons, record or None, sha256 of the file)."""
    try:
        rec = read_json(path)
        file_sha = sha256_file(path)
    except (OSError, ValueError) as err:
        return [f"admission {path} unreadable ({err})"], None, None
    if not isinstance(rec, dict) or set(rec) != ADMISSION_KEYS:
        got = sorted(rec) if isinstance(rec, dict) else type(rec).__name__
        return [f"admission {path} keys {got} != the exact admission schema"], None, file_sha
    r = []

    def need(cond, msg):
        if not cond:
            r.append(msg)
    need(rec["kind"] == "exp40_admission" and rec["admitted"] is True and rec["arm"] == "C16",
         "not an exp_40 C16 admission")
    need(run_id_ok(rec["run_id"]) and rec["platform"] == PLATFORM and rec["rung"] == RUNG,
         f"run_id {rec['run_id']!r} / platform / rung inconsistent")
    need(isinstance(rec["train_sha"], str) and _HEX40.fullmatch(rec["train_sha"]) is not None,
         "train_sha is not a 40-hex commit")
    for k in ("runtree_digest", "env_digest", "manifest_sha256", "sha256", "embedded_config_sha256",
              "ema_inventory_sha256", "full_evidence_sha256", "check_sha256", "audit_sha256"):
        need(isinstance(rec[k], str) and _HEX64.fullmatch(rec[k]) is not None, f"{k} is not a sha256")
    need(rec["nas_dir"] == os.path.join(nas_root, str(rec["run_id"])),
         f"nas_dir {rec['nas_dir']!r} != {os.path.join(nas_root, str(rec['run_id']))}")
    ck = rec["checkpoints"]
    need(isinstance(ck, dict) and sorted(ck) == sorted(EXPECTED_CKPTS)
         and all(isinstance(v, str) and _HEX64.fullmatch(v) for v in ck.values()),
         "checkpoints is not exactly the 16 cadence names -> sha256")
    final = ckpt_name(ENDPOINT)
    need(rec["ckpt_path"] == os.path.join(str(rec["nas_dir"]), final), f"ckpt_path is not <nas_dir>/{final}")
    need(isinstance(ck, dict) and ck.get(final) == rec["sha256"], "sha256 != checkpoints[40k]")
    need(type(rec["bytes"]) is int and rec["bytes"] > 0, "bytes is not a positive int")
    need(type(rec["global_step"]) is int and rec["global_step"] == ENDPOINT, "global_step != 40000")
    need(rec["model_config"] == ARM_CONFIG and rec["model_config_sha256"] == PINS[ARM_CONFIG],
         "model_config is not the pinned RF arm config")
    need(rec["strict_load"] is True, "admission without a strict load")
    need(rec["trajectory_equivalence_claim_allowed"] is False, "trajectory claim must be False")
    need(rec["historical"] == HISTORICAL, "historical is not the fixed exp_17 provenance record")
    r += current_assets_reasons(rec["current_assets"])
    need(isinstance(rec["note"], str), "note must be a string")
    need(isinstance(rec["t"], (int, float)) and not isinstance(rec["t"], bool) and math.isfinite(rec["t"])
         and 0 < rec["t"] <= time.time() + CLOCK_SKEW_S, "admission time t is not a finite, non-future time")
    if r or rehash is None:
        return r, (rec if not r else None), file_sha
    if rehash == "all":
        try:
            names = sorted(os.listdir(rec["nas_dir"]))
        except OSError as err:
            return [f"NAS run dir unreadable ({err})"], None, file_sha
        if names != sorted(list(EXPECTED_CKPTS) + ["MANIFEST.sha256"]):
            r.append(f"NAS run dir holds {names}, not exactly the 16 checkpoints + MANIFEST.sha256")
        elif sha256_file(os.path.join(rec["nas_dir"], "MANIFEST.sha256")) != rec["manifest_sha256"]:
            r.append("NAS MANIFEST.sha256 changed since admission")
        else:
            for name, want in sorted(ck.items()):
                if sha256_file(os.path.join(rec["nas_dir"], name)) != want:
                    r.append(f"NAS {name} no longer hashes to its admitted sha256")
    elif rehash == "final":
        if not os.path.isfile(rec["ckpt_path"]) or sha256_file(rec["ckpt_path"]) != rec["sha256"]:
            r.append(f"{rec['ckpt_path']} does not re-hash to the admitted {rec['sha256']}")
    else:
        raise ValueError(f"rehash={rehash!r}")
    return r, (rec if not r else None), file_sha


# --------------------------------------------------------------------------- #
# the cooperative per-card locks in the shared handshake dir (plan §4.3-5; exp_40 §14 H1)
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
    c.add_argument("--out", required=True)
    c.add_argument("--hf-sealed", required=True)
    g = sub.add_parser("gpus"); g.add_argument("--platform", required=True)
    g.add_argument("--rung", required=True)
    g.add_argument("--mode", required=True); g.add_argument("--lanes", required=True)
    g.add_argument("--policy", required=True); g.add_argument("--expect-total", type=int, required=True)
    g.add_argument("--out")
    w = sub.add_parser("watch"); w.add_argument("--log", required=True)
    w.add_argument("--members", type=int, required=True); w.add_argument("--pid", type=int, required=True)
    w.add_argument("--pgid", type=int, required=True); w.add_argument("--timeout", type=int, default=3600)
    sg = sub.add_parser("stop-group")         # code-r1 B4: the launchers' signal-safe cleanup
    sg.add_argument("--pgid", type=int, required=True); sg.add_argument("--grace", type=float, default=120.0)
    e = sub.add_parser("evidence"); ident(e)
    e.add_argument("--evidence-dir", required=True); e.add_argument("--log", required=True)
    e.add_argument("--check-json", required=True); e.add_argument("--receipts", required=True)
    e.add_argument("--train-rc", type=int, required=True)
    e.add_argument("--watcher-rc", type=int, required=True); e.add_argument("--out", required=True)
    e.add_argument("--reference"); e.add_argument("--publisher-rc", type=int, default=0)
    ps = sub.add_parser("publish-saved"); ps.add_argument("--path", required=True)
    ps.add_argument("--run-id", required=True); ps.add_argument("--dest-root", required=True)
    np_ = sub.add_parser("nas-probe"); np_.add_argument("--dest-root", default=NAS_ROOT)
    np_.add_argument("--out", required=True)
    u = sub.add_parser("audit"); u.add_argument("--ckpt-dir", required=True)
    u.add_argument("--nas-dir"); u.add_argument("--out", required=True)
    u.add_argument("--evidence", required=True); u.add_argument("--run-id", required=True)
    u.add_argument("--expected-dirpath", help="the ModelCheckpoint dirpath recorded at training time "
                                              "(default: --ckpt-dir)")
    u.add_argument("--check", required=True); u.add_argument("--reference", required=True)
    u.add_argument("--tree", help="the run tree whose arm config + source build the strict wrapper "
                                  "(default: this kit's own tree; audit fix 2026-11-11 re-audit)")
    rr_ = sub.add_parser("run-result"); rr_.add_argument("--run-id", required=True)
    rr_.add_argument("--sha", required=True); rr_.add_argument("--verdict", required=True)
    for k in RUN_RESULT_FILES:
        rr_.add_argument(f"--{k}", required=True)
    rr_.add_argument("--expected-dirpath", required=True); rr_.add_argument("--ckpt-dir", required=True)
    rr_.add_argument("--runtree", required=True); rr_.add_argument("--out", required=True)
    st_ = sub.add_parser("admission-state"); st_.add_argument("--ckpt", required=True)
    st_.add_argument("--expected-dirpath", required=True); st_.add_argument("--no-strict", action="store_true")
    st_.add_argument("--hf-sealed", required=True)
    st_.add_argument("--tree", help="the admission tree whose arm config + source are interpreted "
                                    "(default: this kit's own tree)")
    d = sub.add_parser("admit"); d.add_argument("--run-result", required=True)
    d.add_argument("--receipts-dir", required=True); d.add_argument("--note", default="")
    d.add_argument("--hf-sealed", required=True)
    d.add_argument("--out", required=True)
    m = sub.add_parser("ar-manifest"); m.add_argument("--ar-root", required=True)
    m.add_argument("--split", action="append", required=True); m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=4)
    r_ = sub.add_parser("gpu-lock"); r_.add_argument("--uuids", required=True)
    r_.add_argument("--payload-json", required=True); r_.add_argument("--handshake-dir", default=HANDSHAKE_DIR)
    r_.add_argument("--out", required=True)
    rl = sub.add_parser("gpu-unlock"); rl.add_argument("--lock-json", required=True)
    hs = sub.add_parser("build-hf-sealed"); hs.add_argument("--src-hub", required=True)
    hs.add_argument("--dest", required=True)
    av = sub.add_parser("ar-verify"); av.add_argument("--ar-root", required=True)
    av.add_argument("--manifest", required=True); av.add_argument("--out", required=True)
    av.add_argument("--workers", type=int, default=4)
    ac = sub.add_parser("ar-receipt-check")      # code-r1 M3: the SAME validator as the FULL gate
    ac.add_argument("--receipt", required=True); ac.add_argument("--manifest", required=True)
    ac.add_argument("--ar-root", required=True)
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
                                  args.stamp, args.receipts, args.hf_sealed)
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
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "verdict": "PASS" if not reasons else "REFUSED", "reasons": reasons,
                   "t": time.time()}
            if args.out:
                atomic_write_json(args.out, rec)
            for r in reasons:
                print(f"REFUSED: {r}")
            return 0 if not reasons else 2
        if args.cmd == "watch":
            return watch(args.log, args.members, args.pid, args.pgid, args.timeout)
        if args.cmd == "stop-group":
            gone = terminate_group(args.pgid, grace_s=args.grace)
            print(f"process group {args.pgid} {'gone' if gone else 'SURVIVED SIGKILL'}")
            return 0 if gone else 1
        if args.cmd == "evidence":
            idr = validate_identity(args.platform, args.rung, args.mode, args.run_id)
            if idr:
                raise Refused("; ".join(idr))
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
                raise Refused("FULL evidence needs --reference (the DDP_SMOKE receipt)")
            reasons = evidence_reasons(recs, summaries, text, args.mode, args.rung, steps,
                                       expected, fx, reference)
            argv_ = chk.get("argv") or []
            rec = {"verdict": "PASS" if not reasons and args.train_rc == 0
                   and args.watcher_rc == 0 and args.publisher_rc == 0 else "FAIL",
                   "kind": {"DDP_SMOKE": "ddp_smoke", "SMOKE": "smoke", "FULL": "full_evidence"}[args.mode],
                   "platform": args.platform, "rung": args.rung, "mode": args.mode,
                   "run_id": args.run_id, "stamp": args.stamp, "sha": chk.get("sha"),
                   "tree_digest": chk.get("tree_digest"), "env_digest": chk.get("env_digest"),
                   "recipe_projection": recipe_projection(argv_), "argv": argv_,
                   "train_rc": args.train_rc, "watcher_rc": args.watcher_rc,
                   "publisher_rc": args.publisher_rc,
                   "reference_sha256": sha256_file(args.reference) if args.reference else None,
                   "check_sha256": sha256_file(args.check_json),
                   "evidence": {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons},
                   "fit_start": {str(r): next((x for x in recs.get(r, []) if x.get("kind") == "fit_start"), None)
                                 for r in range(world)},
                   "summaries": summaries}
            atomic_write_json(args.out, rec)
            if rec["verdict"] == "PASS" and args.mode == "DDP_SMOKE":
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
        if args.cmd == "publish-saved":
            print(publish_pass(os.path.dirname(args.path), args.run_id, dest_root=args.dest_root,
                               only=[os.path.basename(args.path)]))
            return 0
        if args.cmd == "nas-probe":
            rec = nas_capability_probe(args.dest_root)
            atomic_write_json(args.out, rec)
            print(f"NAS PROBE {rec['verdict']} -> {args.out}")
            for r in rec["reasons"]:
                print(f"  {r}")
            return 0 if rec["verdict"] == "PASS" else 1
        if args.cmd == "audit":
            v = audit(args.ckpt_dir, args.tree or ROOT, nas_dir=args.nas_dir, evidence=args.evidence,
                      run_id=args.run_id, expected_dirpath=args.expected_dirpath,
                      check_path=args.check, reference_path=args.reference)
            atomic_write_json(args.out, v)
            print(f"AUDIT {v['verdict']} -> {args.out}")
            for r in v["reasons"]:
                print(f"  {r}")
            return 0 if v["verdict"] == "PASS" else 1
        if args.cmd == "run-result":
            if args.verdict not in ("TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION", "FAIL"):
                raise Refused(f"run-result verdict {args.verdict!r}")
            rec = run_result(args.run_id, args.sha, args.verdict,
                             {k: getattr(args, k) for k in RUN_RESULT_FILES}, args.expected_dirpath,
                             args.ckpt_dir, args.runtree)
            atomic_write_json(args.out, rec)
            print(f"RUN RESULT {rec['verdict']} -> {args.out}")
            return 0
        if args.cmd == "admission-state":
            print(json.dumps(admission_state(args.ckpt, args.expected_dirpath, tree=args.tree,
                                             strict_load=not args.no_strict, hf_sealed=args.hf_sealed),
                             sort_keys=True))
            return 0
        if args.cmd == "admit":
            if os.path.lexists(args.out):
                raise Refused(f"{args.out} exists; an admission record is written once")
            rec = admit(args.run_result, args.receipts_dir, ROOT, note=args.note, hf_sealed=args.hf_sealed)
            tmp = f"{args.out}.candidate-{os.getpid()}"
            atomic_write_json(tmp, rec)
            reasons, _, _ = admission_reasons(tmp, rehash="all")
            if reasons:
                os.unlink(tmp)
                raise Refused("the derived admission fails its own validator: " + "; ".join(reasons))
            os.chmod(tmp, 0o444)
            rename_noreplace(tmp, args.out)
            fsync_path(os.path.dirname(os.path.abspath(args.out)), directory=True)
            print(f"ADMITTED {rec['sha256']} -> {args.out}")
            return 0
        if args.cmd == "gpu-lock":
            token, paths = acquire_gpu_locks(args.handshake_dir, args.uuids.split(","),
                                             json.loads(args.payload_json))
            try:                                      # code-r1 H3: no token record -> no locks
                atomic_write_json(args.out, {"token": token, "paths": paths})
            except BaseException as err:
                release_gpu_locks(paths, token)
                raise Refused(f"could not record the lock token in {args.out} ({err}); "
                              f"released {paths}") from None
            print(f"LOCKED {paths}")
            return 0
        if args.cmd == "gpu-unlock":
            rec = read_json(args.lock_json)
            release_gpu_locks(rec["paths"], rec["token"])
            print(f"UNLOCKED {rec['paths']}")
            return 0
        if args.cmd == "build-hf-sealed":
            print(build_hf_sealed(args.src_hub, args.dest))
            return 0
        if args.cmd == "ar-receipt-check":
            reasons = ar_verify_reasons(args.receipt, read_json(args.manifest), args.ar_root)
            for r in reasons:
                print(f"  {r}")
            print(f"AR RECEIPT {'PASS' if not reasons else 'REFUSED'} {args.receipt}")
            return 0 if not reasons else 2
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
    except Refused as err:
        print(f"REFUSED: {err}")
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
