#!/usr/bin/env python3
"""exp_38 data_efficiency — the single source of truth for every training run.

Every launch path (the A6000 launcher, the neuronic sbatch + submitter, the eval driver)
asks THIS module for the run's config, dataset config, argv, chunk plan, expected
checkpoint names and gates, so the six runs cannot drift between platforms
(plan_data_efficiency.md §6-§7, §10 item 4).

Pure functions on top; a small CLI at the bottom (``python exp38_registry.py -h``). torch is
imported lazily (only ``audit-ckpt`` and the chunk-plan probe need it), so ``dry-run`` and
``check`` run in a plain interpreter.

Written by the exp_38 Coder (Claude Opus 5.5) per plan_data_efficiency.md.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))


def _repo_root(start):
    d = start
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"no .git above {start}")


REPO = _repo_root(HERE)
EXP_DIR = "worklog/worklog_yixun/exp_38_data_efficiency_claude"

# ---------------------------------------------------------------- recipe constants
MAX_STEPS = 40000
CADENCE = 2500
SEED = 42
EFF_BATCH = 64
N_ROOMS = 243
VAE_PATH = "weights/FLAC/VAE.safetensors"
VAE_SHA256 = "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
# exp_07 / exp_25 DINOv3 initializer pin (assert_arm_configs.py:56-62): one snapshot, these files.
VIT_REPO_DIR = "models--facebook--dinov3-vits16-pretrain-lvd1689m"
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_FILES = {"config.json": "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b",
             "model.safetensors": "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"}
YAW_BANNER = "yaw_aug ENABLED img_w=512 seed=42"
FLASH_MISSING = "flash_attn not installed"
OVERLAY_MESSAGE = "exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA"
FIT_END = "`Trainer.fit` stopped: `max_steps={n}` reached."

FRACS = {
    "025": {"split": "data/AR/train_frac025_s2026.json",
            "split_sha256": "ce33f34f74b055d98a576c5aa5398d058c698d4cce9049e9648bcf3edf27122d",
            "dataset_config": "src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json",
            "dataset_config_sha256": "7ac8243f94cf9abc652e1a667c51263436041ea7a1fb0e8ed868c7e07fae7a21",
            "n_items": 73473, "steps_per_epoch": 1148},
    "050": {"split": "data/AR/train_frac050_s2026.json",
            "split_sha256": "bf33084671130a87735a47f97ff9c5851676f65fe1828bc8f8916e55af113ce4",
            "dataset_config": "src/configs/dataset_configs/AR/train/acousticroom_train_frac050.json",
            "dataset_config_sha256": "b74687f7a8f314436a4220eb461a0bf222f65276fb33eff4486a51e6d32a4025",
            "n_items": 145665, "steps_per_epoch": 2276},
    "075": {"split": "data/AR/train_frac075_s2026.json",
            "split_sha256": "57388ce870f63d9d64fda0c6722f85625580f5572b44b85add4130a8788d26c1",
            "dataset_config": "src/configs/dataset_configs/AR/train/acousticroom_train_frac075.json",
            "dataset_config_sha256": "cc0e5fb8b5f221f393f19122d60c862763688f088329bcf0e42fb4ac338a17b5",
            "n_items": 218463, "steps_per_epoch": 3413},
}

BF_BASE_CONFIG = "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
BF_BASE_SHA256 = "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328"
ARMS = {
    # the YAW 100 % checkpoint ac1f2603... embeds exactly this file (plan §3)
    "yaw": {"config": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
            "config_sha256": "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004",
            "cond_method": "vanilla", "banner": YAW_BANNER, "min_free_mib": 12288, "required": True},
    # B-F: FLAC_AR_BF.json + training.frame_avg_max_fwd_samples = per-rank micro (plan §5)
    "bf": {"config": None, "config_sha256": None,
           "cond_method": "fa_invariant", "banner": None, "min_free_mib": 21900, "required": True},
    # optional platform bridge (plan §6, D2): dc_van's config byte-for-byte
    "p1": {"config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
           "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8",
           "cond_method": "vanilla", "banner": None, "min_free_mib": 12288, "required": False},
}
RUNGS = {"2x32": {"ngpu": 2, "micro": 32, "default": True},
         "8x8": {"ngpu": 8, "micro": 8, "default": False}}
PLATFORMS = {
    "a6000": {"rungs": ("2x32",), "launcher": "python",
              "save_root": "/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency"},
    "neuronic": {"rungs": ("2x32", "8x8"), "launcher": "torchrun",
                 "save_root": "/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency"},
}
OPTIONAL_RUNS = (("p1", "025"),)


# seed-42 initial wrapper state (exp38_init_digest.py on the CPU == the in-process evidence of the
# GPU-0 smokes, 2026-11-07; identical for YAW, B-F and P1 -- their .model sections are equal)
INIT_STATE_SHA256 = "378440d09589c8901f43bdb9e4da97b7401e7e51534d02d00488676cec9ccf95"
INIT_SOBOL_SHA256 = "2b099850ab9d888b168c8ca18dd693819a41814b682c5df0fa5ca9be7bc58798"
FLASH_VERSION = "2.7.4.post1"
CAMPAIGN_ROOT = {"a6000": "/home/yixunhu/codespace/exp-38-campaign"}
# exact untracked symlinks a campaign tree may carry on this box (content is pinned separately)
SYMLINK_ALLOWLIST = {"a6000": {"AcousticRooms": "/media/diskstation/yixunhu/FLAC/AcousticRooms",
                               "weights/AGREE": "/home/yixunhu/codespace/FLAC/weights/AGREE",
                               "weights/FLAC/VAE.safetensors": "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors",
                               "weights/FLAC/VAE.ckpt": "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt"}}
EXTERNAL_CONFIGS = ("worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
                    "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                    "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json")
SHADOW_NAMES = ("sitecustomize.py", "usercustomize.py", "flash_attn", "flash_attn.py")


class GateError(RuntimeError):
    """A fail-closed gate refused."""


# ---------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _abs(rel, repo=None):
    return rel if os.path.isabs(rel) else os.path.join(repo or REPO, rel)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------- runs (registered matrix only)
def run_id(arm, tag, rung="2x32"):
    if arm not in ARMS or tag not in FRACS or rung not in RUNGS:
        raise GateError(f"unknown run ({arm!r}, {tag!r}, {rung!r})")
    return f"de_{arm}_f{tag}" + ("" if rung == "2x32" else f"_r{rung}")


def required_runs(rung="2x32"):
    return [run_id(a, t, rung) for a in ("yaw", "bf") for t in ("025", "050", "075")]


def all_runs(rung="2x32"):
    return required_runs(rung) + ([run_id(a, t, rung) for a, t in OPTIONAL_RUNS] if rung == "2x32" else [])


# the launchable matrix (Codex code r1 HIGH 3): the six primary runs and the P1 bridge at 2x32, and the
# six primary runs on the opt-in 8x8 sensitivity rung -- nothing else (no P1 f050 / f075, no bridge 8x8)
REGISTERED_RUNS = frozenset(all_runs("2x32") + required_runs("8x8"))
_RUN_RE = re.compile(r"^de_(yaw|bf|p1)_f(025|050|075)(?:_r(8x8))?$")


def parse_run_id(run):
    m = _RUN_RE.match(run or "")
    if not m or run not in REGISTERED_RUNS:
        raise GateError(f"not a registered exp_38 run id: {run!r}")
    return m.group(1), m.group(2), m.group(3) or "2x32"


def check_rung(platform, rung, allow_nonmatched=False):
    if platform not in PLATFORMS:
        raise GateError(f"unknown platform {platform!r}")
    if rung not in PLATFORMS[platform]["rungs"]:
        raise GateError(f"rung {rung} is not available on {platform}")
    if not RUNGS[rung]["default"] and not allow_nonmatched:
        raise GateError(f"rung {rung} changes the per-rank micro-batch of every reference run "
                        "(plan §7); it needs ALLOW_NONMATCHED_RUNG=1 and is labelled cross-recipe")
    return RUNGS[rung]


# ---------------------------------------------------------------- configs
def derive_bf_config_text(base_text, cap):
    """FLAC_AR_BF.json + exactly ``training.frame_avg_max_fwd_samples = cap`` (appended last)."""
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise GateError(f"cap must be a positive int, got {cap!r}")
    cfg = json.loads(base_text, object_pairs_hook=collections.OrderedDict)
    if "frame_avg_max_fwd_samples" in cfg["training"]:
        raise GateError("the base B-F config already declares a chunk cap")
    cfg["training"]["frame_avg_max_fwd_samples"] = cap
    return json.dumps(cfg, indent=4) + "\n"


def bf_config_rel(micro):
    return f"{EXP_DIR}/FLAC_AR_BF_cap{micro}.json"


def model_config_rel(arm, rung):
    return bf_config_rel(RUNGS[rung]["micro"]) if arm == "bf" else ARMS[arm]["config"]


def chunk_plan(arm, rung):
    """Announcement 06 record. B-F: cap = per-rank micro -> one angle per orbit forward."""
    if ARMS[arm]["cond_method"] != "fa_invariant":
        return None
    micro = RUNGS[rung]["micro"]
    cap = micro
    per_chunk = max(1, cap // micro)
    return {"frame_avg_max_fwd_samples": cap, "per_rank_micro_batch": micro, "orbit_size": 4,
            "nonzero_angles": 3, "angles_per_chunk": per_chunk,
            "shared_rope_draws": 0 if per_chunk == 1 else per_chunk,
            "equals": "legacy per-angle loop (B-F 100 %)"}


def check_config(arm, rung, repo=None):
    """Verify the arm's config file on disk; return its parsed dict."""
    rel = model_config_rel(arm, rung)
    path = _abs(rel, repo)
    if arm == "bf":
        base = _abs(BF_BASE_CONFIG, repo)
        if sha256_file(base) != BF_BASE_SHA256:
            raise GateError(f"{BF_BASE_CONFIG} is not the B-F 100 % config")
        want = derive_bf_config_text(open(base).read(), RUNGS[rung]["micro"])
        if open(path).read() != want:
            raise GateError(f"{rel} is not FLAC_AR_BF.json + the cap-{RUNGS[rung]['micro']} key")
    elif sha256_file(path) != ARMS[arm]["config_sha256"]:
        raise GateError(f"{rel} sha256 is not the pinned {ARMS[arm]['config_sha256'][:12]}")
    cfg = json.load(open(path))
    tr = cfg["training"]
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    if len(vits) != 2 or not all(c["config"].get("gradient_checkpointing") is True for c in vits):
        raise GateError(f"{rel}: both ViT conditioners must have gradient_checkpointing true")
    if not all(c["config"]["ViT"]["img_w"] == 512 for c in vits) or tr.get("use_ema") is not True:
        raise GateError(f"{rel}: img_w 512 and use_ema true required")
    if tr.get("cond_method", "vanilla") != ARMS[arm]["cond_method"]:
        raise GateError(f"{rel}: cond_method {tr.get('cond_method')!r} != {ARMS[arm]['cond_method']}")
    if arm == "yaw" and tr.get("yaw_aug") != {"enabled": True, "img_w": 512, "seed": 42}:
        raise GateError(f"{rel}: training.yaw_aug is not the registered block")
    if arm != "yaw" and "yaw_aug" in tr:
        raise GateError(f"{rel}: unexpected yaw_aug")
    if arm == "bf":
        if tr.get("frame_avg_angles") != [0.0, 90.0, 180.0, 270.0]:
            raise GateError(f"{rel}: frame_avg_angles must be the C4 orbit")
        if tr.get("frame_avg_max_fwd_samples") != chunk_plan(arm, rung)["frame_avg_max_fwd_samples"]:
            raise GateError(f"{rel}: chunk cap != per-rank micro-batch")
    opt = tr.get("optimizer_configs", {}).get("diffusion", {})
    if opt.get("optimizer") != {"type": "AdamW", "config": {"lr": 5e-05, "betas": [0.9, 0.999], "weight_decay": 0.001}} \
            or opt.get("scheduler") != {"type": "InverseLR", "config": {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99}}:
        raise GateError(f"{rel}: optimizer / scheduler is not AdamW(5e-5, (0.9,0.999), 1e-3) + InverseLR(1e6, 0.5, 0.99)")
    if (tr.get("timestep_sampler"), tr.get("cfg_dropout_prob")) != ("log_snr", 0.1):
        raise GateError(f"{rel}: timestep_sampler / cfg_dropout_prob off-recipe")
    for key in ("canonicalize", "are_lambda", "are_anchor"):
        if key in tr:
            raise GateError(f"{rel}: unexpected training.{key}")
    return cfg


def check_subset(tag, repo=None):
    f = FRACS[tag]
    if sha256_file(_abs(f["split"], repo)) != f["split_sha256"]:
        raise GateError(f"{f['split']} is not exp_14's split")
    if sha256_file(_abs(f["dataset_config"], repo)) != f["dataset_config_sha256"]:
        raise GateError(f"{f['dataset_config']} is not exp_14's dataset config")
    dc = json.load(open(_abs(f["dataset_config"], repo)))
    ctx = dc["modalities"]["acoustic_context"]
    if dc["datasets"][0]["json_file_path"] != f["split"] or ctx.get("restrict_to_split") is not True \
            or ctx.get("max_context") != 8:
        raise GateError(f"{f['dataset_config']}: split path / restrict_to_split / K=8 wrong")
    split = json.load(open(_abs(f["split"], repo)))
    n = sum(len(v) for rooms in split.values() for v in rooms.values())
    rooms = sum(len(r) for r in split.values())
    if n != f["n_items"] or rooms != N_ROOMS:
        raise GateError(f"{f['split']}: {n} items / {rooms} rooms != {f['n_items']} / {N_ROOMS}")
    return {"split": f["split"], "split_sha256": f["split_sha256"], "n_items": n, "rooms": rooms,
            "dataset_config": f["dataset_config"], "restrict_to_split": True}


SUBSET_AUDIT = f"{EXP_DIR}/data_efficiency_subset_audit.json"
SUBSET_AUDIT_BOUND = frozenset({"src/configs/dataset_configs/custom_metadata/AR_md.py", "src/data/dataset.py",
                                "data/AR/train_frac_manifest_s2026.json", "data/AR/train.json",
                                f"{EXP_DIR}/exp38_subset_audit.py"}
                               | {FRACS[t][k] for t in FRACS for k in ("split", "dataset_config")})


def check_subset_audit(repo=None):
    """The committed exhaustive subset audit (plan §10 A4) must exist, match its .sha256, say PASS
    and be bound -- exactly this file set -- to the CURRENT bytes. The dataset ROOT is checked by
    content (``exp38_subset_audit.py --verify`` recomputes the room-listing fingerprint), never by
    path, so the same artifact serves both platforms (Codex code r1 HIGH 8)."""
    path = _abs(SUBSET_AUDIT, repo)
    try:
        text = open(path, "rb").read()
        side = open(path + ".sha256").read().split()[0]
    except OSError as err:
        raise GateError(f"subset audit artifact missing: {err}")
    if hashlib.sha256(text).hexdigest() != side:
        raise GateError("subset audit artifact does not match its .sha256")
    res = json.loads(text)
    if res.get("verdict") != "PASS" or res.get("problems"):
        raise GateError(f"subset audit verdict {res.get('verdict')}")
    bound = res.get("bound", {})
    if set(bound) != SUBSET_AUDIT_BOUND or sorted(res.get("fractions", {})) != sorted(FRACS) \
            or not isinstance(res.get("root_fingerprint"), str):
        raise GateError("subset audit artifact is incomplete / not the registered schema")
    for rel, digest in bound.items():
        if sha256_file(_abs(rel, repo)) != digest:
            raise GateError(f"{rel} changed since the subset audit")
    for tag, f in FRACS.items():
        if res["fractions"][tag]["split_sha256"] != f["split_sha256"]:
            raise GateError(f"subset audit is not bound to the {tag} split")
    return {"subset_audit": SUBSET_AUDIT, "sha256": side}


# ---------------------------------------------------------------- checkpoints and run dirs
def epoch_of_step(step, tag):
    return (step - 1) // FRACS[tag]["steps_per_epoch"]


def ckpt_name(step, tag):
    return f"epoch={epoch_of_step(step, tag)}-step={step}.ckpt"


def cadence_steps(upto=MAX_STEPS):
    return list(range(CADENCE, upto + 1, CADENCE))


RUN_DIR_OWN = (".exp38_lock", ".exp38_tools", "CHECKPOINTS.sha256", "evidence", "TRAINING_COMPLETE.json",
               "TRAINING_COMPLETE.json.sha256")


def _own_entry(name):
    return (name.endswith(".ckpt") or name.startswith("at_launch_") or name.startswith(".exp38_lock.released_")
            or name.startswith(".exp38_probe_") or name in RUN_DIR_OWN)


def check_run_dir(path, run, upto, allow_lock=False):
    """``upto`` 0: a fresh run dir (no checkpoint, nothing foreign, no lock). ``upto`` 40000: a
    finished run's dir -- exactly the 16 cadence checkpoints. There is no resume state (plan §6)."""
    if upto not in (0, MAX_STEPS):
        raise GateError(f"run-dir state {upto}: only 0 (fresh) or {MAX_STEPS} (complete) exist")
    _, tag, _ = parse_run_id(run)
    present = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "*.ckpt")))
    want = sorted(ckpt_name(s, tag) for s in cadence_steps(upto)) if upto else []
    if present != want:
        raise GateError(f"{path}: checkpoints {present} != expected {want} for state {upto}")
    if os.path.isdir(path):
        names = os.listdir(path)
        stray = sorted(n for n in names if not _own_entry(n))
        if stray:
            raise GateError(f"{path}: foreign entries {stray}")
        if ".exp38_lock" in names and not allow_lock:
            raise GateError(f"{path}: run lock .exp38_lock present (a run is live or crashed; archive first)")
        if upto == 0 and any(n in names for n in ("TRAINING_COMPLETE.json", "CHECKPOINTS.sha256")):
            raise GateError(f"{path}: a completed run's records are present")
    return {"run_dir": path, "state": "fresh" if not upto else "complete", "checkpoints": present}


def inverse_lr(step, base=5e-5, inv_gamma=1e6, power=0.5, warmup=0.99):
    """src/training/utils.py::InverseLR closed form (final_lr 0) at a completed step."""
    return base * (1 - warmup ** (step + 1)) * (1 + step / inv_gamma) ** (-power)


# exact checkpoint schema of a 40k checkpoint of this architecture under PL 2.1.0 (read from the
# YAW / B-F 100 % checkpoints and dc_van_f025, 2026-11-07; dc_van carries one extra exp_14 key)
N_OPT_STATE = 449            # of the 753 group parameters, those that receive gradients carry Adam state
N_STATE_TENSORS = 1279
CKPT_TOP_KEYS = frozenset({"callbacks", "epoch", "global_step", "loops", "lr_schedulers", "model_config",
                           "optimizer_states", "pytorch-lightning_version", "state_dict"})
LOOP_KEYS = frozenset({"fit_loop", "predict_loop", "test_loop", "validate_loop"})
FIT_LOOP_KEYS = frozenset({
    "epoch_loop.automatic_optimization.optim_progress", "epoch_loop.automatic_optimization.state_dict",
    "epoch_loop.batch_progress", "epoch_loop.manual_optimization.optim_step_progress",
    "epoch_loop.manual_optimization.state_dict", "epoch_loop.scheduler_progress", "epoch_loop.state_dict",
    "epoch_loop.val_loop.batch_progress", "epoch_loop.val_loop.state_dict", "epoch_progress", "state_dict"})
OPT_GROUP_KEYS = frozenset({"amsgrad", "betas", "capturable", "decoupled_weight_decay", "differentiable", "eps",
                            "foreach", "fused", "initial_lr", "lr", "maximize", "params", "weight_decay"})
OPT_STATE_KEYS = frozenset({"exp_avg", "exp_avg_sq", "step"})
SCHED_KEYS = frozenset({"_get_lr_called_within_step", "_last_lr", "_step_count", "base_lrs", "final_lr",
                        "inv_gamma", "last_epoch", "power", "warmup"})
MODEL_CHECKPOINT_KEY = ("ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2500, "
                        "'every_n_epochs': 0, 'train_time_interval': None}")
MODEL_CHECKPOINT_STATE_KEYS = frozenset({"best_k_models", "best_model_path", "best_model_score", "current_score",
                                         "dirpath", "kth_best_model_path", "kth_value", "last_model_path", "monitor"})
_FRESH = {}


def fresh_inventory(arm, rung, repo=None):
    """(state_dict {key: (shape, dtype)}, optimizer-group param shapes, wrapper) of a freshly built
    CPU wrapper of the arm config -- the reference the checkpoint inventory must equal."""
    key = (arm, rung, repo or REPO)
    if key not in _FRESH:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if (repo or REPO) not in sys.path:
            sys.path.insert(0, repo or REPO)
        from src.models import create_model_from_config
        from src.training import create_training_wrapper_from_config
        cfg = json.load(open(_abs(model_config_rel(arm, rung), repo)))
        wrapper = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
        inv = {k: (tuple(v.shape), str(v.dtype)) for k, v in wrapper.state_dict().items()}
        opt = wrapper.configure_optimizers()
        opt = opt[0][0] if isinstance(opt, tuple) else (opt["optimizer"] if isinstance(opt, dict) else opt)
        opt = opt[0] if isinstance(opt, (list, tuple)) else opt
        shapes = [tuple(p.shape) for p in opt.param_groups[0]["params"]]
        _FRESH[key] = (inv, shapes, wrapper)
    return _FRESH[key]


def compare_inventory(got, want):
    """Problems of a state_dict inventory {key: (shape, dtype)} against the reference."""
    if got == want:
        return []
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    diff = sorted(k for k in set(got) & set(want) if got[k] != want[k])
    return [f"state_dict inventory != a fresh wrapper's (missing {missing[:3]}, extra {extra[:3]}, "
            f"shape/dtype {diff[:3]})"]


def inventory_problems(sd, arm, rung, repo=None):
    """(problems, optimizer param shapes): the state_dict's {key: (shape, dtype)} must equal a fresh
    CPU wrapper's of the arm config, and the fresh wrapper must strict-load it."""
    import torch
    inv, opt_shapes, wrapper = fresh_inventory(arm, rung, repo)
    got = {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items() if torch.is_tensor(v)}
    p = compare_inventory(got, inv)
    if not p:
        try:
            wrapper.load_state_dict(sd, strict=True)
        except Exception as err:                          # noqa: BLE001 -- any load failure is a refusal
            p.append(f"strict load failed: {err}")
    return p, opt_shapes


def ckpt_schema_problems(ck, step, tag, run_dir=None, opt_shapes=None, extra_top_keys=frozenset()):
    """Exact-schema problems of a loaded PL 2.1 checkpoint beyond its state_dict (Codex code r1 HIGH 1)."""
    import torch
    p = []
    if set(ck) != CKPT_TOP_KEYS | set(extra_top_keys):
        p.append(f"top-level keys {sorted(set(ck) ^ (CKPT_TOP_KEYS | set(extra_top_keys)))} differ")
    if ck.get("pytorch-lightning_version") != "2.1.0":
        p.append(f"pytorch-lightning_version {ck.get('pytorch-lightning_version')!r} != '2.1.0'")
    if type(ck.get("global_step")) is not int or ck.get("global_step") != step:
        p.append(f"global_step {ck.get('global_step')!r} != {step}")
    if type(ck.get("epoch")) is not int or ck.get("epoch") != epoch_of_step(step, tag):
        p.append(f"epoch {ck.get('epoch')!r} != {epoch_of_step(step, tag)}")
    opts = ck.get("optimizer_states")
    sch_list = ck.get("lr_schedulers")
    sch = sch_list[0] if isinstance(sch_list, list) and len(sch_list) == 1 and isinstance(sch_list[0], dict) else {}
    if not (isinstance(opts, list) and len(opts) == 1 and isinstance(opts[0], dict)
            and set(opts[0]) == {"state", "param_groups"} and len(opts[0]["param_groups"]) == 1):
        p.append("expected exactly one optimizer state {state, param_groups} with one group")
    else:
        g, st = opts[0]["param_groups"][0], opts[0]["state"]
        if set(g) != OPT_GROUP_KEYS:
            p.append(f"param-group keys {sorted(set(g) ^ OPT_GROUP_KEYS)} differ")
        want_g = {"betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0.001, "initial_lr": 5e-05,
                  "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False,
                  "decoupled_weight_decay": True, "foreach": None, "fused": None}
        for k, v in want_g.items():
            gv = tuple(g.get(k, ())) if k == "betas" else g.get(k)
            if gv != v:
                p.append(f"param group {k} {g.get(k)!r} != {v!r}")
        if g.get("lr") != (sch.get("_last_lr") or [None])[0]:
            p.append("optimizer lr != scheduler _last_lr")
        if len(st) != N_OPT_STATE or not all(type(i) is int for i in st):
            p.append(f"optimizer state for {len(st)} parameters != {N_OPT_STATE}")
        if opt_shapes is not None and g.get("params") != list(range(len(opt_shapes))):
            p.append(f"param group lists {len(g.get('params', []))} params != the fresh wrapper's {len(opt_shapes)}")
        for i, s_ in st.items():
            if not isinstance(s_, dict) or set(s_) != OPT_STATE_KEYS:
                p.append(f"optimizer state {i} keys {sorted(s_) if isinstance(s_, dict) else s_!r}")
                break
            shp = opt_shapes[i] if opt_shapes is not None and isinstance(i, int) and i < len(opt_shapes) else None
            if (shp is not None and not (tuple(s_["exp_avg"].shape) == shp == tuple(s_["exp_avg_sq"].shape))) \
                    or not bool(torch.isfinite(s_["exp_avg"]).all()) or not bool(torch.isfinite(s_["exp_avg_sq"]).all()) \
                    or float(s_["step"]) != float(step):
                p.append(f"optimizer state {i} shape / finiteness / step off")
                break
    if set(sch) != SCHED_KEYS:
        p.append(f"scheduler keys {sorted(set(sch) ^ SCHED_KEYS)} differ")
    elif (sch["inv_gamma"], sch["power"], sch["warmup"], sch["final_lr"], sch["base_lrs"], sch["last_epoch"],
          sch["_step_count"], sch["_get_lr_called_within_step"]) != (1000000, 0.5, 0.99, 0.0, [5e-05], step, step + 1, False):
        p.append(f"scheduler {[sch[k] for k in sorted(SCHED_KEYS - {'_last_lr'})]} off-recipe")
    elif not (isinstance(sch["_last_lr"], list) and len(sch["_last_lr"]) == 1
              and abs(sch["_last_lr"][0] - inverse_lr(step)) <= 1e-12):
        p.append(f"scheduler _last_lr {sch['_last_lr']} != InverseLR({step}) {inverse_lr(step)}")
    loops = ck.get("loops")
    if not isinstance(loops, dict) or set(loops) != LOOP_KEYS or set(loops.get("fit_loop", {})) != FIT_LOOP_KEYS:
        p.append("loop state keys differ from PL 2.1's")
    else:
        fl = loops["fit_loop"]
        try:
            done = fl["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"]
            bp, ep = fl["epoch_loop.batch_progress"]["total"], fl["epoch_progress"]["total"]
        except (KeyError, TypeError):
            done, bp, ep = None, {}, {}
        if done != step or bp.get("ready") != step or bp.get("processed") != step:
            p.append(f"fit-loop progress {done} / {bp} != step {step}")
        if ep.get("processed") not in (epoch_of_step(step, tag), epoch_of_step(step, tag) + 1):
            p.append(f"epoch progress {ep} inconsistent with epoch {epoch_of_step(step, tag)}")
    cbs = ck.get("callbacks")
    if not isinstance(cbs, dict) or list(cbs) != [MODEL_CHECKPOINT_KEY]:
        p.append(f"callback state keys {list(cbs) if isinstance(cbs, dict) else cbs!r} != [{MODEL_CHECKPOINT_KEY}]")
    else:
        mc = cbs[MODEL_CHECKPOINT_KEY]
        if not isinstance(mc, dict) or set(mc) != MODEL_CHECKPOINT_STATE_KEYS:
            p.append("ModelCheckpoint state keys differ")
        elif run_dir is not None and os.path.realpath(str(mc.get("dirpath", ""))) != os.path.realpath(run_dir):
            p.append(f"ModelCheckpoint dirpath {mc.get('dirpath')} != {run_dir}")
        elif mc.get("monitor") is not None:
            p.append("ModelCheckpoint monitors a metric")
    return p


def audit_ckpt(path, run, step, repo=None, run_dir=None, inventory=True, extra_top_keys=frozenset()):
    """One read of the bytes: sha256 + full training-state admission (plan r2 §10 B2)."""
    import io
    import torch
    arm, tag, rung = parse_run_id(run)
    if os.path.basename(path) != ckpt_name(step, tag):
        raise GateError(f"{path} is not named {ckpt_name(step, tag)}")
    with open(path, "rb") as fh:
        data = fh.read()
    digest = hashlib.sha256(data).hexdigest()
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    want = json.load(open(_abs(model_config_rel(arm, rung), repo)))
    p = []
    if canonical(ck.get("model_config")) != canonical(want):
        p.append("embedded model_config != the arm config")
    sd = ck.get("state_dict", {})
    if len(sd) != N_STATE_TENSORS or not all(torch.is_tensor(v) for v in sd.values()):
        p.append(f"state_dict holds {len(sd)} entries (want {N_STATE_TENSORS} tensors)")
    bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        p.append(f"non-finite tensors {bad[:3]}")
    ema = {k[len("diffusion_ema.ema_model."):]: v for k, v in sd.items() if k.startswith("diffusion_ema.ema_model.")}
    onl = {k[len("diffusion.model."):]: v for k, v in sd.items() if k.startswith("diffusion.model.")}
    if not ema or set(ema) != set(onl):
        p.append(f"EMA suffix set ({len(ema)}) != online diffusion.model suffix set ({len(onl)})")
    elif any(ema[k].shape != onl[k].shape or ema[k].dtype != onl[k].dtype for k in ema):
        p.append("EMA / online tensors differ in shape or dtype")
    est, ein = sd.get("diffusion_ema.step"), sd.get("diffusion_ema.initted")
    if est is None or ein is None or int(est.reshape(-1)[0]) != step or float(ein.reshape(-1)[0]) != 1.0:
        p.append(f"EMA step / initted {est} / {ein} (want {step} / 1)")
    opt_shapes = None
    if inventory:
        inv_p, opt_shapes = inventory_problems(sd, arm, rung, repo)
        p += inv_p
    p += ckpt_schema_problems(ck, step, tag, run_dir, opt_shapes, extra_top_keys)
    if p:
        raise GateError(f"{path}: " + "; ".join(p))
    return {"ckpt": path, "sha256": digest, "bytes": len(data), "global_step": step}


def write_ckpt_manifest(path, run, allow_lock=False):
    """CHECKPOINTS.sha256 of a complete run dir (the supervisor still holds the run lock then)."""
    _, tag, _ = parse_run_id(run)
    check_run_dir(path, run, MAX_STEPS, allow_lock=allow_lock)
    lines = [f"{sha256_file(os.path.join(path, ckpt_name(s, tag)))}  {ckpt_name(s, tag)}"
             for s in cadence_steps()]
    out = os.path.join(path, "CHECKPOINTS.sha256")
    if os.path.exists(out):
        raise GateError(f"{out} already exists")
    atomic_write_bytes(out, ("\n".join(lines) + "\n").encode())
    return out


# ---------------------------------------------------------------- argv (monolithic only, Codex code r1 HIGH 2)
def save_dir(platform, run, save_root=None):
    return os.path.join(save_root or PLATFORMS[platform]["save_root"], run)


def build_train_argv(run, platform, save_root=None, allow_nonmatched=False):
    """argv for train.py (the caller prefixes ``python`` or ``torchrun …``): one monolithic 0 -> 40,000
    run. There is no resume argv: a crashed primary run is archived and re-run from step 0 (plan §6)."""
    arm, tag, rung = parse_run_id(run)
    r = check_rung(platform, rung, allow_nonmatched)
    sd = save_dir(platform, run, save_root)
    argv = ["train.py",
            "--model-config", model_config_rel(arm, rung),
            "--dataset-config", FRACS[tag]["dataset_config"],
            "--pretransform-ckpt-path", VAE_PATH,
            "--max-steps", str(MAX_STEPS),
            "--batch-size", str(r["micro"]), "--accum-batches", "1", "--num-workers", "6",
            "--seed", str(SEED), "--num-gpus", str(r["ngpu"]), "--num-nodes", "1",
            "--strategy", "ddp_find_unused_parameters_true", "--sync-batchnorm", "true",
            "--precision", "bf16-mixed", "--val-every", "-1", "--gradient-clip-val", "0.0",
            "--logger", "none", "--checkpoint-every", str(CADENCE),
            "--name", run, "--experiment-name", run, "--save-dir", sd,
            "--evidence-json", os.path.join(sd, "evidence", "run_evidence.json")]
    if r["ngpu"] * r["micro"] != EFF_BATCH or any("exp14_" in a for a in argv) or "--ckpt-path" in argv:
        raise GateError("argv invariant broken")
    return argv


def argv_projection(argv):
    """The production-relevant part of a train.py argv (drops budget / cadence / names / paths)."""
    drop_value = ("--max-steps", "--checkpoint-every", "--name", "--experiment-name", "--save-dir", "--evidence-json")
    return [a for i, a in enumerate(argv) if a not in drop_value and not (i and argv[i - 1] in drop_value)]


def build_smoke_argv(run, steps, smoke_dir, ddp=False, platform="a6000"):
    """Storage-light smoke at the real per-rank micro-batch. ``ddp=False``: one rank (no DDP /
    SyncBN; a preliminary rung only). ``ddp=True``: the production topology of the rung, exactly
    the FULL argv but for the step budget, cadence, names and save dir."""
    arm, tag, rung = parse_run_id(run)
    if not 1 <= steps <= 200:
        raise GateError("smoke: 1..200 steps")
    if ddp:
        argv = build_train_argv(run, platform, save_root=os.path.dirname(smoke_dir.rstrip("/")),
                                allow_nonmatched=(rung != "2x32"))
        argv[argv.index("--max-steps") + 1] = str(steps)
        argv[argv.index("--checkpoint-every") + 1] = "1000000"
        argv[argv.index("--save-dir") + 1] = smoke_dir
        argv[argv.index("--evidence-json") + 1] = os.path.join(smoke_dir, "evidence", "run_evidence.json")
        for flag in ("--name", "--experiment-name"):
            argv[argv.index(flag) + 1] = f"smoke_{run}"
        return argv
    if rung != "2x32":
        raise GateError("single-rank smoke: 2x32 runs only")
    return ["train.py", "--model-config", model_config_rel(arm, rung),
            "--dataset-config", FRACS[tag]["dataset_config"], "--pretransform-ckpt-path", VAE_PATH,
            "--max-steps", str(steps), "--batch-size", "32", "--accum-batches", "1",
            "--num-workers", "6", "--seed", str(SEED), "--num-gpus", "1", "--num-nodes", "1",
            "--precision", "bf16-mixed", "--val-every", "-1", "--gradient-clip-val", "0.0",
            "--logger", "none", "--checkpoint-every", "1000000",
            "--name", f"smoke_{run}", "--experiment-name", f"smoke_{run}", "--save-dir", smoke_dir,
            "--evidence-json", os.path.join(smoke_dir, "evidence", "run_evidence.json")]


# ---------------------------------------------------------------- pins and backend
def vit_pin_check(hub_cache=None):
    if hub_cache is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub_cache = HF_HUB_CACHE
    snap_dir = os.path.join(hub_cache, VIT_REPO_DIR, "snapshots")
    snaps = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []
    if snaps != [VIT_REV]:
        raise GateError(f"DINOv3 snapshots {snaps} != [{VIT_REV}] in {snap_dir}")
    snap = os.path.join(snap_dir, VIT_REV)
    files = sorted(os.path.relpath(os.path.join(d, f), snap) for d, _, fs in os.walk(snap) for f in fs)
    if files != sorted(VIT_FILES):
        raise GateError(f"DINOv3 snapshot files {files} != {sorted(VIT_FILES)}")
    for name, want in VIT_FILES.items():
        if sha256_file(os.path.join(snap, name)) != want:
            raise GateError(f"DINOv3 {name} is not the pinned file")
    return {"vit_rev": VIT_REV, "hub_cache": hub_cache}


def vae_check(repo=None):
    if sha256_file(_abs(VAE_PATH, repo)) != VAE_SHA256:
        raise GateError(f"{VAE_PATH} is not the pinned VAE")
    return {"vae_sha256": VAE_SHA256}


def backend_probe(backend, repo=None):
    """flash: a real flash_attn must be importable; sdpa: the name must resolve to the exp_35
    overlay (PYTHONPATH=overlays/no_flash) or be absent from the environment."""
    import importlib.util
    spec = importlib.util.find_spec("flash_attn")
    origin = os.path.realpath(spec.origin) if spec and spec.origin else None
    overlay = os.path.realpath(_abs("overlays/no_flash/flash_attn.py", repo))
    if backend == "flash":
        if origin is None or origin == overlay:
            raise GateError("ATTN_BACKEND=flash but flash_attn is not importable (or shadowed)")
        return {"attention_backend": "flash", "flash_attn": origin}
    if backend == "sdpa":
        if origin not in (None, overlay):
            raise GateError(f"ATTN_BACKEND=sdpa but a real flash_attn ({origin}) is reachable; "
                            "put overlays/no_flash on PYTHONPATH")
        return {"attention_backend": "sdpa", "flash_attn": origin or "absent"}
    raise GateError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------- log gates
_FOUND = re.compile(r"Found (\d+) files in (\d+) subfolders")
_MEMBER = re.compile(r"Initializing distributed: GLOBAL_RANK: (\d+), MEMBER: (\d+)/(\d+)")
_STEP = re.compile(r"Epoch \d+:\s+\d+%\|[^|]*\|\s*([1-9]\d*)/\d+")
_LOSS = re.compile(r"train/loss=([^,\]\s]+)")


def log_lines(text):
    return text.replace("\r", "\n").split("\n")


def first_step_index(lines):
    for i, line in enumerate(lines):
        if _STEP.search(line):
            return i
    return None


def parse_losses(text):
    """Every ``train/loss=<value>`` the progress bar printed: (finite floats, non-finite tokens)."""
    import math
    good, bad = [], []
    for tok in _LOSS.findall(text):
        try:
            v = float(tok)
        except ValueError:
            bad.append(tok)
            continue
        (good if math.isfinite(v) else bad).append(v if math.isfinite(v) else tok)
    return good, bad


def log_gate(text, run, phase, backend, ngpu, n_items=None, max_steps=MAX_STEPS, min_losses=1):
    """Return a list of problems (empty = PASS) for the start or end phase of a training log."""
    arm, tag, _ = parse_run_id(run)
    n_items = FRACS[tag]["n_items"] if n_items is None else n_items
    lines = log_lines(text)
    problems = []
    step_i = first_step_index(lines)
    if step_i is None:
        problems.append("no optimizer-step progress line yet")
    found = [m.groups() for m in map(_FOUND.search, lines) if m]
    if len(found) < ngpu or any(g != (str(n_items), str(N_ROOMS)) for g in found):
        problems.append(f"dataset lines {found} != {ngpu} x ({n_items}, {N_ROOMS})")
    if ngpu > 1:
        members = {m.groups() for m in map(_MEMBER.search, lines) if m}
        want = {(str(r), str(r + 1), str(ngpu)) for r in range(ngpu)}
        if members != want:
            problems.append(f"DDP members {sorted(members)} != {sorted(want)}")
    elif any(_MEMBER.search(line) for line in lines):
        problems.append("a single-rank run initialised distributed")
    banner = ARMS[arm]["banner"]
    if banner:
        idx = [i for i, line in enumerate(lines) if line.strip() == banner]
        if not idx or (step_i is not None and idx[0] > step_i):
            problems.append(f"banner {banner!r} missing or after the first step")
    elif any(YAW_BANNER.split(" img_w")[0] in line for line in lines):
        problems.append("a non-YAW run printed the yaw_aug banner")
    missing = [line for line in lines if FLASH_MISSING in line]
    if backend == "flash" and missing:
        problems.append("ATTN_BACKEND=flash but the log says flash_attn is not installed")
    if backend == "sdpa" and len(missing) < ngpu:
        problems.append(f"ATTN_BACKEND=sdpa but only {len(missing)} '{FLASH_MISSING}' lines for {ngpu} ranks")
    good, bad = parse_losses(text)
    if bad:
        problems.append(f"non-finite / unparseable train/loss values {bad[:3]}")
    if len(good) < (min_losses if phase == "end" else 1):
        problems.append(f"only {len(good)} finite train/loss values (need {min_losses if phase == 'end' else 1})")
    if phase == "end" and FIT_END.format(n=max_steps) not in text:
        problems.append(f"Lightning's max_steps={max_steps} termination marker missing")
    return problems


# ---------------------------------------------------------------- evaluator weight pins (Codex code r1 BLOCKING 6)
AGREE_REL = "weights/AGREE/AGREE_fullAR.pt"
AGREE_SHA256 = "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
VAE_CKPT_REL = "weights/FLAC/VAE.ckpt"           # the AGREE audio tower's cwd-relative "pretrained"
VAE_CKPT_SHA256 = "6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf"


def eval_pins(repo=None):
    """AGREE + the AGREE audio tower's VAE.ckpt + the DINOv3 snapshot: checked before and after
    every evaluation cell and on admission."""
    out = {}
    for rel, want, key in ((AGREE_REL, AGREE_SHA256, "agree_sha256"), (VAE_CKPT_REL, VAE_CKPT_SHA256, "vae_ckpt_sha256")):
        got = sha256_file(_abs(rel, repo))
        if got != want:
            raise GateError(f"{rel} sha {got} != pinned {want}")
        out[key] = got
    out.update(vit_pin_check())
    return out


# ---------------------------------------------------------------- atomic, sealed records (NIT 2)
def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as err:                       # CIFS may refuse a directory fsync
        if err.errno not in (22, 95):
            raise
    finally:
        os.close(fd)


def atomic_write_bytes(path, data):
    """temp file -> fsync -> os.replace -> directory fsync."""
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(d)


def write_sealed(path, rec):
    """A JSON receipt + its ``.sha256`` sidecar; refuses an existing receipt. The sidecar is written
    first, the JSON last (the JSON is the commit point: a sidecar alone is never admitted)."""
    if os.path.exists(path):
        raise GateError(f"{path} exists (receipts are written once)")
    text = (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode()
    atomic_write_bytes(path + ".sha256", f"{hashlib.sha256(text).hexdigest()}  {os.path.basename(path)}\n".encode())
    atomic_write_bytes(path, text)
    return hashlib.sha256(text).hexdigest()


def read_sealed(path, keys):
    """(record, sha256) of a sealed receipt with EXACTLY ``keys``; anything else refuses."""
    try:
        text = open(path, "rb").read()
        side = open(path + ".sha256").read().split()
    except OSError as err:
        raise GateError(f"receipt {path} missing / unreadable: {err}")
    digest = hashlib.sha256(text).hexdigest()
    if side != [digest, os.path.basename(path)]:
        raise GateError(f"{path} does not match its .sha256 sidecar")
    try:
        rec = json.loads(text)
    except ValueError as err:
        raise GateError(f"{path}: not JSON ({err})")
    if not isinstance(rec, dict) or set(rec) != set(keys):
        got = set(rec) if isinstance(rec, dict) else set()
        raise GateError(f"{path}: schema differs (missing {sorted(set(keys) - got)}, extra {sorted(got - set(keys))})")
    return rec, digest


def _is_sha(x):
    return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None


# ---------------------------------------------------------------- storage (Codex code r1 BLOCKING 6, HIGH 7)
# the registered output filesystems: the A6000 NAS is CIFS behind autofs; neuronic's /n/fs is a
# network filesystem (its exact type is recorded by the first SMOKE; local disks / tmpfs refuse)
STORAGE_FS = {"a6000": ("cifs",), "neuronic": ("nfs", "nfs4", "lustre", "gpfs", "beegfs", "cifs", "smb3")}
TRAIN_FREE_GIB = 30           # 16 checkpoints = 11.6 GB per run (plan §11) + margin
DUMP_CELL_BYTES = 259_600_000  # one scored-waveform cell


def mount_fstype(path):
    """fstype of the filesystem holding ``path`` (the most recent mount on the longest prefix)."""
    real = os.path.realpath(path)
    best, fstype = "", None
    for line in open("/proc/self/mounts"):
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt = parts[1].replace("\\040", " ")
        if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
            best, fstype = mnt, parts[2]
    return fstype, best


def storage_gate(platform, path, need_bytes, tag="exp38"):
    """``path`` must exist on a registered filesystem, have ``need_bytes`` free and pass a
    write / read-back / delete probe. Returns the observed facts."""
    if not os.path.isdir(path):
        raise GateError(f"storage root {path} does not exist (it is never created here: a missing "
                        "mount must not become a local directory)")
    fstype, mnt = mount_fstype(path)
    if fstype not in STORAGE_FS[platform]:
        raise GateError(f"{path} is on {fstype} ({mnt}), not a registered {platform} filesystem {STORAGE_FS[platform]}")
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    if free < need_bytes:
        raise GateError(f"{path}: {free / 2**30:.1f} GiB free < {need_bytes / 2**30:.1f} GiB needed")
    probe = os.path.join(path, f".{tag}_probe_{os.getpid()}_{int(time.time())}")
    payload = os.urandom(16).hex()
    try:
        with open(probe, "x") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        ok = open(probe).read() == payload
    finally:
        if os.path.exists(probe):
            os.unlink(probe)
    if not ok:
        raise GateError(f"{path}: write probe read back different bytes")
    return {"path": path, "fstype": fstype, "mountpoint": mnt, "free_bytes": free}


# ---------------------------------------------------------------- campaign closure + immutable snapshot
CLOSURE_ROOTS = ("train.py", "defaults.ini", "eval_FLAC.py", "pyproject.toml", "src", "overlays", "data/AR")
CLOSURE_EXP_PREFIXES = ("exp38_", "FLAC_AR_BF_cap", "data_efficiency_subset_audit.json")


def _git_out(repo, *args):
    import subprocess
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout


def closure_files(repo=None):
    """Tracked code + configs + splits the campaign runs (tests and worklog prose excluded). A
    snapshot (no git) lists its own SNAPSHOT.sha256 instead."""
    repo = repo or REPO
    if os.path.isfile(os.path.join(repo, "SNAPSHOT.sha256")):
        rels = [ln.split("  ", 1)[1] for ln in open(os.path.join(repo, "SNAPSHOT.sha256")).read().strip().split("\n")]
        return sorted(r for r in rels if _in_closure(r))
    out = _git_out(repo, "ls-files", "-z", "--", *CLOSURE_ROOTS, EXP_DIR, *EXTERNAL_CONFIGS).decode().split("\0")
    return sorted({r for r in out if r and _in_closure(r)})


def _in_closure(rel):
    if rel.startswith("src/tests/"):
        return False
    if rel.startswith(EXP_DIR + "/"):
        return os.path.basename(rel).startswith(CLOSURE_EXP_PREFIXES) and "/" not in rel[len(EXP_DIR) + 1:]
    return rel in EXTERNAL_CONFIGS or any(rel == r or rel.startswith(r + "/") for r in CLOSURE_ROOTS)


def closure_digest(repo=None):
    repo = repo or REPO
    h = hashlib.sha256()
    files = closure_files(repo)
    for rel in files:
        h.update(f"{rel}\0{sha256_file(os.path.join(repo, rel))}\n".encode())
    return {"closure_sha256": h.hexdigest(), "n_files": len(files)}


# the snapshot's own links (train.py runs with the snapshot as its cwd; the dataset config names
# "AcousticRooms" and the argv names weights/FLAC/VAE.safetensors, both cwd-relative)
SNAPSHOT_LINKS = ("AcousticRooms", VAE_PATH)
RECEIPT_FILES = (SUBSET_AUDIT + ".sha256",)


def snapshot_files(run, repo=None):
    return sorted(set(closure_files(repo)) | {r for r in RECEIPT_FILES if os.path.exists(_abs(r, repo))})


def snapshot_tools(run, dest, repo=None):
    """Copy the closure into ``dest`` (refused if it exists), link the two data inputs to the
    campaign tree's verified targets, and seal SNAPSHOT.sha256 + SNAPSHOT.links. Training is
    launched FROM the snapshot and every later audit reads it (Codex code r1 HIGH 5)."""
    import shutil
    repo = repo or REPO
    if os.path.lexists(dest):
        raise GateError(f"snapshot dir {dest} exists")
    lines = []
    for rel in snapshot_files(run, repo):
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(_abs(rel, repo), out)
        lines.append(f"{sha256_file(out)}  {rel}")
    links = []
    for rel in SNAPSHOT_LINKS:
        target = os.path.realpath(_abs(rel, repo))
        if not os.path.exists(target):
            raise GateError(f"{rel} does not resolve in {repo}")
        os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
        os.symlink(target, os.path.join(dest, rel))
        links.append(f"{target}  {rel}")
    atomic_write_bytes(os.path.join(dest, "SNAPSHOT.sha256"), ("\n".join(lines) + "\n").encode())
    atomic_write_bytes(os.path.join(dest, "SNAPSHOT.links"), ("\n".join(links) + "\n").encode())
    return {"snapshot": dest, "files": len(lines), "snapshot_sha256": snapshot_digest(dest)}


def snapshot_digest(dest):
    return hashlib.sha256(open(os.path.join(dest, "SNAPSHOT.sha256"), "rb").read()
                          + open(os.path.join(dest, "SNAPSHOT.links"), "rb").read()).hexdigest()


def verify_snapshot(dest, want_digest=None):
    p = []
    for line in open(os.path.join(dest, "SNAPSHOT.sha256")).read().strip().split("\n"):
        digest, rel = line.split("  ", 1)
        path = os.path.join(dest, rel)
        if os.path.islink(path) or not os.path.isfile(path) or sha256_file(path) != digest:
            p.append(f"snapshot file {rel} changed")
    for line in open(os.path.join(dest, "SNAPSHOT.links")).read().strip().split("\n"):
        target, rel = line.split("  ", 1)
        if not os.path.islink(os.path.join(dest, rel)) or os.readlink(os.path.join(dest, rel)) != target:
            p.append(f"snapshot link {rel} changed")
    if VAE_PATH in SNAPSHOT_LINKS and sha256_file(os.path.join(dest, VAE_PATH)) != VAE_SHA256:
        p.append("snapshot VAE is not the pinned file")
    if want_digest is not None and snapshot_digest(dest) != want_digest:
        p.append("snapshot manifests differ from the launch record")
    if p:
        raise GateError("; ".join(p))
    return {"snapshot": dest, "verified": True, "snapshot_sha256": snapshot_digest(dest)}


# ---------------------------------------------------------------- in-process evidence gate (Codex code r1 BLOCKING 2)
EVIDENCE_KEYS = frozenset({
    "accumulate_grad_batches", "batchnorm_nonsync_modules", "checkpoint_calls", "conditioner_calls",
    "flash_attn_func_set", "flash_calls", "global_step_at_fit_start", "global_step_recorded",
    "gradient_clip_val", "local_rank", "max_steps", "micro_batch", "n_state_tensors", "num_nodes", "precision",
    "rank", "rng_at_fit_start", "sdpa_calls", "sobol", "state_dict_sha256", "strategy", "syncbn_modules",
    "vit_ckpt_wrapped_layers", "world_size"})
INIT_SOBOL = {"dimension": 1, "num_generated": 0, "scramble": True, "state_sha256": INIT_SOBOL_SHA256}
# RNG digests at fit start. One rank (SingleDeviceStrategy): train.py's seed_everything(42) state
# after the model build drew from torch's CPU generator -- measured by the GPU-0 smokes (YAW, B-F).
# DDP (local subprocess launcher and torchrun alike): PL 2.1 DDPStrategy.setup_distributed() calls
# lightning_fabric reset_seed(), re-seeding every rank with PL_GLOBAL_SEED=42 after the build, so the
# torch CPU state is the FRESH seed-42 state (computed on the CPU; python / numpy are untouched by
# the build and equal in both). CUDA: seed 42, nothing drawn before fit -> identical in both.
# The CPU 2-rank ddp_spawn test (test_exp38_ddp_evidence.py) exercises the DDP rule; SMOKE_DDP
# confirms the full set on the cards before FULL. Every rank must carry exactly these values.
RNG_PY = "a52169fac48aa2d80acadd41f776ce38656ab70adc3b32c80c1fb1f0ee56b71f"
RNG_NP = "9701712d219c85a63378972112819f19e48d92e38babd180e9bdcde9a5e36cc6"
RNG_CUDA = "aefdce03def94ffed42159e55677ec20e91200e1981f25b1f271b6fd99c3c263"
RNG_AT_FIT_START = {
    "single": {"python": RNG_PY, "numpy": RNG_NP, "torch_cuda": RNG_CUDA,
               "torch_cpu": "7461f975015a0a9df867cc0521d9a6bc71fc82a8f2e47da7289e94a0645b15b6"},
    "ddp": {"python": RNG_PY, "numpy": RNG_NP, "torch_cuda": RNG_CUDA,
            "torch_cpu": "5feb54a46230d321d888e9b43cf5a57492665f199e4377408d79ee460e36679c"},
}
N_BATCHNORM = 20
DIT_ATTN_CALLS = 48            # DiT attention calls in step 1 (measured: YAW and B-F, flash)
VIT_CKPT_CALLS = {"vanilla": 120, "fa_invariant": 444}   # measured: YAW 120, B-F 444 (12 layers)


def load_evidence(evidence_dir, ngpu):
    """{rank: record} or raises: exactly ``ngpu`` rank files and nothing else."""
    names = sorted(os.listdir(evidence_dir)) if os.path.isdir(evidence_dir) else []
    want = [f"run_evidence.json.rank{r}.json" for r in range(ngpu)]
    if names != sorted(want):
        raise GateError(f"{evidence_dir}: evidence files {names} != {sorted(want)}")
    out = {}
    for r in range(ngpu):
        try:
            out[r] = json.load(open(os.path.join(evidence_dir, want[r])))
        except (OSError, ValueError) as err:
            raise GateError(f"rank {r} evidence unreadable: {err}")
    return out


def evidence_problems(recs, run, ngpu, backend, max_steps=MAX_STEPS):
    """Problems (empty = PASS) of the per-rank first-step records ``{rank: record}``: the exact
    schema and every recorded value, rank-indexed where they differ and equal where they agree."""
    arm, _, rung = parse_run_id(run)
    micro = RUNGS[rung]["micro"] if ngpu > 1 else 32
    topo = "ddp" if ngpu > 1 else "single"
    cond = ARMS[arm]["cond_method"]
    p = []
    if sorted(recs) != list(range(ngpu)):
        return [f"evidence ranks {sorted(recs)} != {list(range(ngpu))}"]
    for r in range(ngpu):
        e = recs[r]
        if not isinstance(e, dict) or set(e) != EVIDENCE_KEYS:
            got = set(e) if isinstance(e, dict) else set()
            p.append(f"rank {r}: evidence schema (missing {sorted(EVIDENCE_KEYS - got)}, extra {sorted(got - EVIDENCE_KEYS)})")
            continue
        want = {"rank": r, "local_rank": r, "world_size": ngpu, "num_nodes": 1, "precision": "bf16-mixed",
                "accumulate_grad_batches": 1, "max_steps": max_steps, "gradient_clip_val": 0.0,
                "strategy": "DDPStrategy" if ngpu > 1 else "SingleDeviceStrategy",
                "global_step_at_fit_start": 0, "global_step_recorded": 1, "micro_batch": micro,
                "state_dict_sha256": INIT_STATE_SHA256, "n_state_tensors": N_STATE_TENSORS,
                "vit_ckpt_wrapped_layers": 12, "flash_attn_func_set": backend == "flash",
                "sobol": INIT_SOBOL, "rng_at_fit_start": RNG_AT_FIT_START[topo],
                "syncbn_modules": N_BATCHNORM if ngpu > 1 else 0,
                "batchnorm_nonsync_modules": 0 if ngpu > 1 else N_BATCHNORM,
                "flash_calls": DIT_ATTN_CALLS if backend == "flash" else 0,
                "sdpa_calls": 0 if backend == "flash" else DIT_ATTN_CALLS,
                "checkpoint_calls": [{"use_reentrant": False}] * VIT_CKPT_CALLS[cond],
                "conditioner_calls": [[micro, None]] + ([[micro, ["source_vit", "context_poses_vit"]]] * 3
                                                        if cond == "fa_invariant" else [])}
        for k, v in want.items():
            if canonical(e[k]) != canonical(v) or type(e[k]) is not type(v):
                shown = e[k] if k != "checkpoint_calls" else f"{len(e[k])} calls"
                p.append(f"rank {r}: {k} {shown!r} != {v if k != 'checkpoint_calls' else len(v)!r}")
    return p


def evidence_gate(evidence_dir, run, ngpu, backend, max_steps=MAX_STEPS):
    try:
        recs = load_evidence(evidence_dir, ngpu)
    except GateError as err:
        return [str(err)]
    return evidence_problems(recs, run, ngpu, backend, max_steps)


# ---------------------------------------------------------------- tree hygiene + environment (Codex code r1 HIGH 9)
# untracked entries a campaign tree may carry, each validated by CONTENT (not by name)
TREE_LINKS = {
    "AcousticRooms": "dir",                      # content: exp38_subset_audit.py --verify (root fingerprint)
    VAE_PATH: VAE_SHA256,
    VAE_CKPT_REL: VAE_CKPT_SHA256,
    "weights/AGREE": "agree_dir",                 # content: AGREE_fullAR.pt sha
}
TREE_LINKS_REQUIRED = {"a6000": ("AcousticRooms", VAE_PATH, VAE_CKPT_REL, "weights/AGREE"),
                       "neuronic": ("AcousticRooms", VAE_PATH)}
# on this box the links must also name these exact targets
A6000_LINK_TARGETS = {"AcousticRooms": "/media/diskstation/yixunhu/FLAC/AcousticRooms",
                      "weights/AGREE": "/home/yixunhu/codespace/FLAC/weights/AGREE",
                      VAE_PATH: "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors",
                      VAE_CKPT_REL: "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt"}
# variables that change what or how the interpreter / torch runs: must be unset (the numeric /
# communication ones a site may legitimately set are pinned by the environment receipt instead)
FORBIDDEN_ENV = ("PYTHONOPTIMIZE", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME", "ENABLE_TORCH_COMPILE",
                 "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE", "LD_PRELOAD")
ENV_PREFIXES = ("NCCL_", "TORCH", "PYTORCH_", "CUBLAS", "CUDNN", "PL_", "LIGHTNING_", "PYTHONHASHSEED")


def check_untracked(platform, repo=None):
    """Problems (empty = PASS): import shadows, unexpected untracked entries, and every allowed link
    validated by content (and, on this box, by its registered target)."""
    repo = repo or REPO
    p = []
    untracked = _git_out(repo, "ls-files", "-z", "--others", "--exclude-standard").decode().split("\0")
    for rel in filter(None, untracked):
        if rel.startswith(("worklog/", "outputs_FLAC/")) and not rel.endswith(".py"):
            continue
        if rel in TREE_LINKS or any(rel.startswith(k + "/") for k in TREE_LINKS):
            continue
        p.append(f"unexpected untracked entry {rel}")
    for rel, kind in TREE_LINKS.items():
        path = os.path.join(repo, rel)
        if not os.path.lexists(path):
            if rel in TREE_LINKS_REQUIRED[platform]:
                p.append(f"required link {rel} is missing")
            continue
        if not os.path.islink(path):
            p.append(f"{rel} must be a symlink")
            continue
        if platform == "a6000" and os.readlink(path) != A6000_LINK_TARGETS[rel]:
            p.append(f"{rel} -> {os.readlink(path)}, not {A6000_LINK_TARGETS[rel]}")
        try:
            if kind == "dir" and not os.path.isdir(path):
                p.append(f"{rel} does not resolve to a directory")
            elif kind == "agree_dir" and sha256_file(os.path.join(path, "AGREE_fullAR.pt")) != AGREE_SHA256:
                p.append(f"{rel}/AGREE_fullAR.pt is not the pinned AGREE")
            elif _is_sha(kind) and sha256_file(path) != kind:
                p.append(f"{rel} is not the pinned file")
        except OSError as err:
            p.append(f"{rel}: {err}")
    for top in ("", "src", "src/models", "src/training", "src/data", "overlays"):
        d = os.path.join(repo, top)
        for name in (os.listdir(d) if os.path.isdir(d) else []):
            if name in SHADOW_NAMES:
                p.append(f"import shadow {os.path.join(top, name)}")
    p += env_problems()
    return p


def env_problems(environ=None):
    """The interpreter flags / variables every launch and evaluation must run under (NIT: PYTHONOPTIMIZE)."""
    environ = os.environ if environ is None else environ
    p = [f"environment variable {k} is set" for k in FORBIDDEN_ENV if environ.get(k)]
    if environ.get("PYTHONNOUSERSITE") != "1":
        p.append("PYTHONNOUSERSITE=1 is required")
    if environ is os.environ and (sys.flags.optimize or not sys.flags.no_user_site):
        p.append(f"interpreter flags optimize={sys.flags.optimize} no_user_site={sys.flags.no_user_site}")
    return p


ENV_RECEIPT_KEYS = frozenset({"schema", "platform", "python", "executable", "torch", "cuda", "cudnn", "nccl",
                              "driver", "gpu_names", "packages", "packages_sha256", "flags", "env_vars"})


def _gpu_rows():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode or not out.stdout.strip():
        raise GateError(f"nvidia-smi failed (rc {out.returncode}): {out.stderr.strip()}")
    return [[x.strip() for x in line.split(",")] for line in out.stdout.strip().split("\n")]


def env_snapshot(platform):
    """The environment identity (fails closed on every probe): interpreter, the CUDA stack, the
    driver, the GPU model, the complete installed-package inventory, flags and numeric env vars."""
    import importlib.metadata as md
    import platform as pf
    import torch
    try:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as err:                      # noqa: BLE001 -- a missing NCCL refuses
        raise GateError(f"NCCL version probe failed: {err}")
    rows = _gpu_rows()
    drivers = sorted({r[3] for r in rows})
    if len(drivers) != 1:
        raise GateError(f"driver versions {drivers}")
    pk = {}
    for dist in md.distributions():
        name = re.sub(r"[-_.]+", "-", (dist.metadata["Name"] or "").lower())
        if name:
            pk[name] = dist.version
    rec = {"schema": 1, "platform": platform, "python": pf.python_version(),
           "executable": os.path.realpath(sys.executable), "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "nccl": nccl, "driver": drivers[0],
           "gpu_names": sorted({r[2] for r in rows}), "packages": dict(sorted(pk.items())),
           "packages_sha256": hashlib.sha256(canonical(dict(sorted(pk.items()))).encode()).hexdigest(),
           "flags": {"optimize": sys.flags.optimize, "no_user_site": sys.flags.no_user_site,
                     "ignore_environment": sys.flags.ignore_environment, "isolated": sys.flags.isolated},
           "env_vars": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(ENV_PREFIXES) and k != "CUDA_VISIBLE_DEVICES"}}
    if None in (rec["cuda"], rec["cudnn"]) or not rec["gpu_names"]:
        raise GateError("CUDA / cuDNN / GPU probe returned nothing")
    if platform == "a6000" and pk.get("flash-attn") != FLASH_VERSION:
        raise GateError(f"flash-attn {pk.get('flash-attn')} != pinned {FLASH_VERSION}")
    return rec


def env_receipt_rel(platform):
    return f"{EXP_DIR}/data_efficiency_env_receipt_{platform}.json"


def env_receipt(platform, write=False, repo=None):
    """Write once (``write=True``, sealed) or check (exact equality with the live environment)."""
    path = _abs(env_receipt_rel(platform), repo)
    probs = env_problems()
    if probs:
        raise GateError("; ".join(probs))
    now = env_snapshot(platform)
    if write:
        digest = write_sealed(path, now)
        return {"env_receipt": path, "sha256": digest, "written": True}
    pinned, digest = read_sealed(path, ENV_RECEIPT_KEYS)
    if canonical(pinned) != canonical(now):
        diff = sorted(k for k in ENV_RECEIPT_KEYS if canonical(pinned.get(k)) != canonical(now.get(k)))
        if "packages" in diff:
            pk0, pk1 = pinned["packages"], now["packages"]
            diff.append({k: (pk0.get(k), pk1.get(k)) for k in set(pk0) | set(pk1) if pk0.get(k) != pk1.get(k)})
        raise GateError(f"environment drift vs the campaign receipt: {diff}")
    return {"env_receipt": path, "sha256": digest}


def env_receipt_sha(platform, repo=None):
    return read_sealed(_abs(env_receipt_rel(platform), repo), ENV_RECEIPT_KEYS)[1]


# ---------------------------------------------------------------- E0 (the validator lives with its writer)
E0_RECEIPT = f"{EXP_DIR}/data_efficiency_e0_receipt.json"


def check_e0_receipt(platform, repo=None):
    """The E0 receipt must be a complete PASS at THIS closure (exp38_eval.validate_e0_receipt)."""
    sys.path.insert(0, HERE)
    import exp38_eval
    return exp38_eval.validate_e0_receipt(repo=repo)


# ---------------------------------------------------------------- SMOKE_DDP receipt (Codex code r1 BLOCKING 1)
SMOKE_STEPS_DDP = 30
SMOKE_KEYS = frozenset({"schema", "mode", "run", "platform", "backend", "head", "closure_sha256",
                        "env_receipt_sha256", "subset_audit_sha256", "snapshot_sha256", "argv_projection",
                        "model_config_sha256", "dataset_config_sha256", "split_sha256", "rc", "checkpoints_written",
                        "log", "log_sha256", "log_gate", "evidence", "evidence_gate", "gpus", "verdict",
                        "launched_at", "finished_at"})


def smoke_receipt_rel(run, platform, ts):
    return f"{EXP_DIR}/data_efficiency_smoke_{run}_{platform}_{ts}.json"


def gpu_identity(platform, ngpu, rows=None):
    """[[uuid, name], ...] the run used: this box's two registered A6000s, or neuronic's allocation."""
    if platform == "a6000":
        rows = rows if rows is not None else _gpu_rows()
        got = sorted([r[1], r[2]] for r in rows)
        want = sorted([u, "NVIDIA RTX A6000"] for u in A6000_UUIDS)
        if got != want:
            raise GateError(f"GPUs {got} != this box's registered A6000s {want}")
        return got
    if rows is None:
        raise GateError("neuronic GPU identity comes from the allocation probe")
    got = sorted([r[0], r[1]] for r in rows)
    if len(got) != ngpu or any("L40" not in n for _, n in got):
        raise GateError(f"allocation GPUs {got} are not {ngpu} x L40")
    return got


A6000_UUIDS = ("GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da")


def write_smoke_receipt(run, platform, backend, head, smoke_dir, log_path, rc, gpus, snapshot_sha, launched_at,
                        ts, repo=None, out_repo=None):
    """Recomputes every gate from the artifacts (never trusts a caller's verdict) and seals the
    receipt into ``out_repo`` (the campaign tree); the bindings are computed from ``repo`` (the
    snapshot the smoke ran from)."""
    arm, tag, rung = parse_run_id(run)
    ngpu = RUNGS[rung]["ngpu"]
    text = open(log_path, errors="replace").read()
    n_ckpt = len(glob.glob(os.path.join(smoke_dir, "**", "*.ckpt"), recursive=True))
    lg = log_gate(text, run, "end", backend, ngpu, max_steps=SMOKE_STEPS_DDP, min_losses=1)
    try:
        recs = load_evidence(os.path.join(smoke_dir, "evidence"), ngpu)
        eg = evidence_problems(recs, run, ngpu, backend, SMOKE_STEPS_DDP)
    except GateError as err:
        recs, eg = {}, [str(err)]
    verdict = "PASS" if (rc == 0 and n_ckpt == 0 and not lg and not eg) else "FAIL"
    rec = {"schema": 1, "mode": "SMOKE_DDP", "run": run, "platform": platform, "backend": backend, "head": head,
           "closure_sha256": closure_digest(repo)["closure_sha256"],
           "env_receipt_sha256": env_receipt_sha(platform, repo),
           "subset_audit_sha256": check_subset_audit(repo)["sha256"], "snapshot_sha256": snapshot_sha,
           "argv_projection": argv_projection(build_train_argv(run, platform, allow_nonmatched=rung != "2x32")),
           "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
           "dataset_config_sha256": sha256_file(_abs(FRACS[tag]["dataset_config"], repo)),
           "split_sha256": sha256_file(_abs(FRACS[tag]["split"], repo)), "rc": int(rc),
           "checkpoints_written": n_ckpt, "log": log_path, "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "log_gate": "PASS" if not lg else lg, "evidence": {str(r): v for r, v in sorted(recs.items())},
           "evidence_gate": "PASS" if not eg else eg, "gpus": gpus, "verdict": verdict,
           "launched_at": launched_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = _abs(smoke_receipt_rel(run, platform, ts), out_repo or repo)
    write_sealed(out, rec)
    return {"smoke_receipt": out, "verdict": verdict, "log_gate": rec["log_gate"], "evidence_gate": rec["evidence_gate"]}


def smoke_receipt_problems(rec, run, platform, backend, head, gpus=None, repo=None):
    """Every field of a sealed SMOKE_DDP receipt against the CURRENT tree / environment / request."""
    arm, tag, rung = parse_run_id(run)
    ngpu = RUNGS[rung]["ngpu"]
    want = {"schema": 1, "mode": "SMOKE_DDP", "run": run, "platform": platform, "backend": backend, "head": head,
            "closure_sha256": closure_digest(repo)["closure_sha256"],
            "env_receipt_sha256": env_receipt_sha(platform, repo),
            "subset_audit_sha256": check_subset_audit(repo)["sha256"],
            "argv_projection": argv_projection(build_train_argv(run, platform, allow_nonmatched=rung != "2x32")),
            "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
            "dataset_config_sha256": FRACS[tag]["dataset_config_sha256"], "split_sha256": FRACS[tag]["split_sha256"],
            "rc": 0, "checkpoints_written": 0, "log_gate": "PASS", "evidence_gate": "PASS", "verdict": "PASS"}
    p = [f"{k} {rec.get(k)!r} != {v!r}" for k, v in want.items() if canonical(rec.get(k)) != canonical(v)]
    if not (_is_sha(rec.get("snapshot_sha256")) and _is_sha(rec.get("log_sha256"))
            and isinstance(rec.get("log"), str) and isinstance(rec.get("launched_at"), str)
            and isinstance(rec.get("finished_at"), str)):
        p.append("snapshot / log / time fields malformed")
    if gpus is not None and canonical(rec.get("gpus")) != canonical(gpus):
        p.append(f"GPUs {rec.get('gpus')} != this launch's {gpus}")
    g = rec.get("gpus")
    if platform == "a6000" and canonical(g) != canonical(sorted([u, "NVIDIA RTX A6000"] for u in A6000_UUIDS)):
        p.append("smoke GPUs are not this box's registered A6000s")
    if platform == "neuronic" and not (isinstance(g, list) and len(g) == ngpu
                                       and all(isinstance(x, list) and len(x) == 2 and "L40" in str(x[1]) for x in g)):
        p.append(f"smoke GPUs {g} are not {ngpu} x L40")
    ev = rec.get("evidence")
    if not isinstance(ev, dict) or sorted(ev) != [str(r) for r in range(ngpu)]:
        p.append("smoke evidence ranks")
    else:
        p += evidence_problems({int(r): v for r, v in ev.items()}, run, ngpu, backend, SMOKE_STEPS_DDP)
    return p


def check_smoke_receipt(run, platform, backend, head, gpus=None, repo=None):
    """The newest sealed SMOKE_DDP receipt for (run, platform) must pass every field check; an
    older PASS cannot stand in for a newer FAIL (the newest receipt decides)."""
    paths = sorted(glob.glob(_abs(f"{EXP_DIR}/data_efficiency_smoke_{run}_{platform}_*.json", repo)))
    if not paths:
        raise GateError(f"no SMOKE_DDP receipt for {run} on {platform}")
    rec, digest = read_sealed(paths[-1], SMOKE_KEYS)
    p = smoke_receipt_problems(rec, run, platform, backend, head, gpus, repo)
    if p:
        raise GateError(f"{paths[-1]}: " + "; ".join(p))
    return {"smoke_receipt": paths[-1], "sha256": digest}


# ---------------------------------------------------------------- terminal training receipt (Codex code r1 BLOCKING 4)
TRAINING_RECEIPT = "TRAINING_COMPLETE.json"
FULL_MIN_LOSSES = 50          # the end gate of a 40k run: at least 50 finite progress-bar losses
TRAINING_KEYS = frozenset({"schema", "run", "platform", "backend", "head", "closure_sha256", "snapshot_sha256",
                           "env_receipt_sha256", "e0_receipt_sha256", "smoke_receipt", "smoke_receipt_sha256",
                           "model_config_sha256", "dataset_config_sha256", "split_sha256", "argv", "rc", "log",
                           "log_sha256", "log_gate", "evidence_sha256", "evidence_gate", "checkpoints_sha256",
                           "final_ckpt", "final_ckpt_sha256", "final_ckpt_bytes", "gpus", "launched_at",
                           "finished_at"})


def evidence_digest(evidence_dir):
    h = hashlib.sha256()
    for name in sorted(os.listdir(evidence_dir)):
        h.update(f"{name}\0{sha256_file(os.path.join(evidence_dir, name))}\n".encode())
    return h.hexdigest()


def write_training_receipt(run, platform, backend, head, rd, log_path, rc, snapshot_sha, e0_sha, smoke, gpus,
                           launched_at, repo=None):
    """Published only after every end audit passes (re-run here, from the snapshot ``repo``)."""
    arm, tag, rung = parse_run_id(run)
    ngpu = RUNGS[rung]["ngpu"]
    if rc != 0:
        raise GateError(f"train exit status {rc}")
    text = open(log_path, errors="replace").read()
    lg = log_gate(text, run, "end", backend, ngpu, min_losses=FULL_MIN_LOSSES)
    eg = evidence_gate(os.path.join(rd, "evidence"), run, ngpu, backend)
    if lg or eg:
        raise GateError("end gates: " + "; ".join(lg + eg))
    check_run_dir(rd, run, MAX_STEPS, allow_lock=True)
    final = ckpt_name(MAX_STEPS, tag)
    audit = audit_ckpt(os.path.join(rd, final), run, MAX_STEPS, repo=repo, run_dir=rd)
    manifest = write_ckpt_manifest(rd, run, allow_lock=True)
    lines = dict(reversed(ln.split("  ", 1)) for ln in open(manifest).read().strip().split("\n"))
    if lines.get(final) != audit["sha256"]:
        raise GateError("final checkpoint changed between the audit and the manifest")
    rec = {"schema": 1, "run": run, "platform": platform, "backend": backend, "head": head,
           "closure_sha256": closure_digest(repo)["closure_sha256"], "snapshot_sha256": snapshot_sha,
           "env_receipt_sha256": env_receipt_sha(platform, repo), "e0_receipt_sha256": e0_sha,
           "smoke_receipt": smoke["smoke_receipt"], "smoke_receipt_sha256": smoke["sha256"],
           "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
           "dataset_config_sha256": sha256_file(_abs(FRACS[tag]["dataset_config"], repo)),
           "split_sha256": sha256_file(_abs(FRACS[tag]["split"], repo)),
           "argv": build_train_argv(run, platform, allow_nonmatched=rung != "2x32"), "rc": 0, "log": log_path,
           "log_sha256": hashlib.sha256(text.encode()).hexdigest(), "log_gate": "PASS",
           "evidence_sha256": evidence_digest(os.path.join(rd, "evidence")), "evidence_gate": "PASS",
           "checkpoints_sha256": sha256_file(manifest), "final_ckpt": final, "final_ckpt_sha256": audit["sha256"],
           "final_ckpt_bytes": audit["bytes"], "gpus": gpus, "launched_at": launched_at,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_sealed(os.path.join(rd, TRAINING_RECEIPT), rec)
    return rec


def check_training_receipt(run, rd, closure=None, receipt_dir=None, verify_bytes=True):
    """The sealed terminal receipt of a run (in ``receipt_dir`` or the run dir), bound to the run, the
    campaign ``closure``, the checkpoint manifest and -- with ``verify_bytes`` -- the final
    checkpoint's current sha256 and byte count. Returns (receipt, receipt sha256)."""
    arm, tag, rung = parse_run_id(run)
    src = receipt_dir or rd
    rec, digest = read_sealed(os.path.join(src, TRAINING_RECEIPT), TRAINING_KEYS)
    final = ckpt_name(MAX_STEPS, tag)
    p = []
    want = {"schema": 1, "run": run, "rc": 0, "log_gate": "PASS", "evidence_gate": "PASS", "final_ckpt": final,
            "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung))),
            "dataset_config_sha256": FRACS[tag]["dataset_config_sha256"], "split_sha256": FRACS[tag]["split_sha256"]}
    p += [f"{k} {rec.get(k)!r} != {v!r}" for k, v in want.items() if rec.get(k) != v]
    if closure is not None and rec.get("closure_sha256") != closure:
        p.append(f"trained at closure {str(rec.get('closure_sha256'))[:12]}, not the campaign's {closure[:12]}")
    if not (type(rec.get("final_ckpt_bytes")) is int and _is_sha(rec.get("final_ckpt_sha256"))):
        p.append("final checkpoint sha / bytes malformed")
    try:
        manifest = os.path.join(src, "CHECKPOINTS.sha256")
        if sha256_file(manifest) != rec.get("checkpoints_sha256"):
            p.append("CHECKPOINTS.sha256 != the receipt's")
        lines = dict(reversed(ln.split("  ", 1)) for ln in open(manifest).read().strip().split("\n"))
        if sorted(lines) != sorted(ckpt_name(s, tag) for s in cadence_steps()) or lines[final] != rec.get("final_ckpt_sha256"):
            p.append("checkpoint manifest set / final line")
        if verify_bytes:
            path = os.path.join(rd, final)
            if os.path.getsize(path) != rec.get("final_ckpt_bytes") or sha256_file(path) != rec.get("final_ckpt_sha256"):
                p.append(f"{path} sha / bytes != the training receipt")
    except (OSError, ValueError) as err:
        p.append(f"checkpoint manifest: {err}")
    if p:
        raise GateError(f"{run} training receipt: " + "; ".join(p))
    return rec, digest


# ---------------------------------------------------------------- dry run
def describe_run(run, platform, save_root=None, allow_nonmatched=False, repo=None):
    arm, tag, rung = parse_run_id(run)
    check_config(arm, rung, repo)
    sub = check_subset(tag, repo)
    audit = check_subset_audit(repo)
    argv = build_train_argv(run, platform, save_root, allow_nonmatched=allow_nonmatched)
    sd = save_dir(platform, run, save_root)
    local = platform == "a6000" or save_root is not None or os.path.isdir(os.path.dirname(sd))
    state = check_run_dir(sd, run, 0)["state"] if local else "not verifiable on this host"
    return {"run": run, "arm": arm, "fraction": tag, "rung": rung, "platform": platform,
            "model_config": model_config_rel(arm, rung),
            "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
            "subset": sub, "subset_audit": audit, "chunk_plan": chunk_plan(arm, rung),
            "run_dir_state": state, "final_ckpt": ckpt_name(MAX_STEPS, tag), "argv": argv}


# ---------------------------------------------------------------- CLI (no resume, no receipt writers: the supervisor writes them)
def _cli(argv=None):
    global REPO
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", help="file-lookup root (an immutable tool snapshot)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--rung", default="2x32", choices=sorted(RUNGS))
    for name in ("check", "argv"):
        s = sub.add_parser(name)
        s.add_argument("run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
        s.add_argument("--save-root"); s.add_argument("--allow-nonmatched-rung", action="store_true")
    s = sub.add_parser("smoke-argv"); s.add_argument("run"); s.add_argument("--steps", type=int, required=True)
    s.add_argument("--save-dir", required=True); s.add_argument("--ddp", action="store_true")
    s.add_argument("--platform", default="a6000", choices=sorted(PLATFORMS))
    s = sub.add_parser("run-dir-check"); s.add_argument("dir"); s.add_argument("--run", required=True)
    s.add_argument("--state", type=int, default=0, choices=(0, MAX_STEPS))
    s = sub.add_parser("audit-ckpt"); s.add_argument("path"); s.add_argument("--run", required=True)
    s.add_argument("--step", type=int, required=True); s.add_argument("--run-dir")
    s = sub.add_parser("pins"); s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    sub.add_parser("eval-pins")
    s = sub.add_parser("log-gate"); s.add_argument("log"); s.add_argument("--run", required=True)
    s.add_argument("--phase", required=True, choices=("start", "end"))
    s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    s.add_argument("--ngpu", type=int, required=True); s.add_argument("--max-steps", type=int, default=MAX_STEPS)
    s.add_argument("--min-losses", type=int, default=50)
    s = sub.add_parser("verify-snapshot"); s.add_argument("dest")
    s = sub.add_parser("evidence-gate"); s.add_argument("dir"); s.add_argument("--run", required=True)
    s.add_argument("--ngpu", type=int, required=True); s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    s.add_argument("--max-steps", type=int, default=MAX_STEPS)
    sub.add_parser("closure")
    s = sub.add_parser("untracked"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s = sub.add_parser("env-receipt"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s.add_argument("--write", action="store_true")
    s = sub.add_parser("e0-check"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s = sub.add_parser("smoke-check"); s.add_argument("run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s.add_argument("--backend", required=True, choices=("flash", "sdpa")); s.add_argument("--head", required=True)
    s = sub.add_parser("training-check"); s.add_argument("run"); s.add_argument("--run-dir", required=True)
    s = sub.add_parser("dry-run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s.add_argument("--rung", default="2x32", choices=sorted(RUNGS)); s.add_argument("--save-root")
    s.add_argument("--allow-nonmatched-rung", action="store_true"); s.add_argument("--include-optional", action="store_true")
    a = p.parse_args(argv)
    if a.repo:
        REPO = os.path.realpath(a.repo)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        if a.cmd == "list":
            print("\n".join(all_runs(a.rung)))
        elif a.cmd == "check":
            out = describe_run(a.run, a.platform, a.save_root, a.allow_nonmatched_rung)
            print(json.dumps({k: v for k, v in out.items() if k != "argv"}, indent=1))
        elif a.cmd == "argv":
            arm, tag, rung = parse_run_id(a.run)
            check_config(arm, rung); check_subset(tag)
            print("\n".join(build_train_argv(a.run, a.platform, a.save_root, a.allow_nonmatched_rung)))
        elif a.cmd == "smoke-argv":
            arm, tag, rung = parse_run_id(a.run)
            check_config(arm, rung); check_subset(tag)
            print("\n".join(build_smoke_argv(a.run, a.steps, a.save_dir, a.ddp, a.platform)))
        elif a.cmd == "run-dir-check":
            print(json.dumps(check_run_dir(a.dir, a.run, a.state)))
        elif a.cmd == "audit-ckpt":
            print(json.dumps(audit_ckpt(a.path, a.run, a.step, run_dir=a.run_dir)))
        elif a.cmd == "pins":
            print(json.dumps({**vae_check(), **vit_pin_check(), **backend_probe(a.backend)}))
        elif a.cmd == "eval-pins":
            print(json.dumps(eval_pins()))
        elif a.cmd == "log-gate":
            problems = log_gate(open(a.log, errors="replace").read(), a.run, a.phase, a.backend,
                                a.ngpu, max_steps=a.max_steps, min_losses=a.min_losses)
            print("LOG GATE " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "verify-snapshot":
            print(json.dumps(verify_snapshot(a.dest)))
        elif a.cmd == "evidence-gate":
            problems = evidence_gate(a.dir, a.run, a.ngpu, a.backend, a.max_steps)
            print("EVIDENCE GATE " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "closure":
            print(json.dumps(closure_digest()))
        elif a.cmd == "untracked":
            problems = check_untracked(a.platform)
            print("UNTRACKED " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "env-receipt":
            print(json.dumps(env_receipt(a.platform, write=a.write)))
        elif a.cmd == "e0-check":
            print(json.dumps(check_e0_receipt(a.platform)))
        elif a.cmd == "smoke-check":
            print(json.dumps(check_smoke_receipt(a.run, a.platform, a.backend, a.head)))
        elif a.cmd == "training-check":
            print(json.dumps(check_training_receipt(a.run, a.run_dir)[0], indent=1))
        elif a.cmd == "dry-run":
            runs = all_runs(a.rung) if a.include_optional else required_runs(a.rung)
            for run in runs:
                d = describe_run(run, a.platform, a.save_root, a.allow_nonmatched_rung)
                print(f"=== {run} [{a.platform} {d['rung']}] config {d['model_config']} "
                      f"({d['model_config_sha256'][:12]}) split {d['subset']['split']} "
                      f"({d['subset']['n_items']} items / {d['subset']['rooms']} rooms, restricted; audit "
                      f"{d['subset_audit']['sha256'][:12]}) chunk {d['chunk_plan']} run dir {d['run_dir_state']} "
                      f"final {d['final_ckpt']}")
                print("    argv: " + " ".join(d["argv"]))
            print(f"DRY-RUN OK: {len(runs)} run(s) on {a.platform}")
    except (GateError, OSError, KeyError, ValueError) as err:
        print(f"EXP38 GATE REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
