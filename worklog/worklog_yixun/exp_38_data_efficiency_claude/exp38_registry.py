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


# ---------------------------------------------------------------- runs
def run_id(arm, tag, rung="2x32"):
    if arm not in ARMS or tag not in FRACS or rung not in RUNGS:
        raise GateError(f"unknown run ({arm!r}, {tag!r}, {rung!r})")
    return f"de_{arm}_f{tag}" + ("" if rung == "2x32" else f"_r{rung}")


_RUN_RE = re.compile(r"^de_(yaw|bf|p1)_f(025|050|075)(?:_r(8x8))?$")


def parse_run_id(run):
    m = _RUN_RE.match(run or "")
    if not m:
        raise GateError(f"not an exp_38 run id: {run!r}")
    return m.group(1), m.group(2), m.group(3) or "2x32"


def required_runs(rung="2x32"):
    return [run_id(a, t, rung) for a in ("yaw", "bf") for t in ("025", "050", "075")]


def all_runs(rung="2x32"):
    return required_runs(rung) + [run_id(a, t, rung) for a, t in OPTIONAL_RUNS]


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


def check_subset_audit(repo=None):
    """The committed exhaustive subset audit (plan §10 A4) must exist, match its .sha256, say
    PASS and be bound to the CURRENT AR_md.py / dataset.py / manifest / dataset-config bytes."""
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
    need = {"src/configs/dataset_configs/custom_metadata/AR_md.py", "src/data/dataset.py",
            "data/AR/train_frac_manifest_s2026.json", "data/AR/train.json", f"{EXP_DIR}/exp38_subset_audit.py"}
    need |= {FRACS[t][k] for t in FRACS for k in ("split", "dataset_config")}
    if not need <= set(bound) or sorted(res.get("fractions", {})) != sorted(FRACS):
        raise GateError("subset audit artifact is incomplete")
    for rel, digest in bound.items():
        if rel.startswith("realpath("):
            continue
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


def check_leg(start, end):
    for v in (start, end):
        if isinstance(v, bool) or not isinstance(v, int) or v % CADENCE:
            raise GateError(f"leg boundary {v!r} is not on the {CADENCE}-step cadence")
    if not 0 <= start < end <= MAX_STEPS:
        raise GateError(f"leg [{start}, {end}] outside [0, {MAX_STEPS}]")


RUN_DIR_OWN = (".exp38_lock", ".exp38_submitted", ".exp38_tools", "CHECKPOINTS.sha256", "evidence")


def _own_entry(name):
    return (name.endswith(".ckpt") or name.startswith("at_launch_") or name.startswith(".exp38_lock.released_")
            or name in RUN_DIR_OWN)


def check_run_dir(path, run, leg_start, allow_submission_marker=False):
    """Fresh (leg_start 0): no checkpoint and nothing foreign. Resume: exactly the cadence set
    <= leg_start. ``.exp38_lock`` (a live or crashed run) always refuses a new start;
    ``.exp38_submitted`` is allowed only to the job its submission created."""
    _, tag, _ = parse_run_id(run)
    present = sorted(os.path.basename(p) for p in glob.glob(os.path.join(path, "*.ckpt")))
    want = sorted(ckpt_name(s, tag) for s in cadence_steps(leg_start)) if leg_start else []
    if present != want:
        raise GateError(f"{path}: checkpoints {present} != expected {want} for a start at step {leg_start}")
    if os.path.isdir(path):
        names = os.listdir(path)
        stray = sorted(n for n in names if not _own_entry(n))
        if stray:
            raise GateError(f"{path}: foreign entries {stray}")
        if ".exp38_lock" in names:
            raise GateError(f"{path}: run lock .exp38_lock present (a run is live or crashed; archive first)")
        if ".exp38_submitted" in names and not allow_submission_marker:
            raise GateError(f"{path}: .exp38_submitted present (already submitted; archive the run dir first)")
    return {"run_dir": path, "state": "fresh" if not leg_start else f"resume@{leg_start}",
            "checkpoints": present}


def inverse_lr(step, base=5e-5, inv_gamma=1e6, power=0.5, warmup=0.99):
    """src/training/utils.py::InverseLR closed form (final_lr 0) at a completed step."""
    return base * (1 - warmup ** (step + 1)) * (1 + step / inv_gamma) ** (-power)


# optimizer state entries of a 40k checkpoint of this architecture (dc_van_f025, YAW / B-F 100 %:
# 449 of the 753 group parameters receive gradients; the rest never get Adam state)
N_OPT_STATE = 449
MODEL_CHECKPOINT_KEY = ("ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2500, "
                        "'every_n_epochs': 0, 'train_time_interval': None}")
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


def audit_ckpt(path, run, step, repo=None, run_dir=None, inventory=True):
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
    if ck.get("global_step") != step:
        p.append(f"global_step {ck.get('global_step')} != {step}")
    if ck.get("epoch") != epoch_of_step(step, tag):
        p.append(f"epoch {ck.get('epoch')} != {epoch_of_step(step, tag)}")
    if canonical(ck.get("model_config")) != canonical(want):
        p.append("embedded model_config != the arm config")
    sd = ck.get("state_dict", {})
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
    if inventory:
        inv, shapes, wrapper = fresh_inventory(arm, rung, repo)
        got = {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items() if torch.is_tensor(v)}
        if got != inv:
            p.append(f"state_dict inventory != a fresh wrapper's (missing {sorted(set(inv) - set(got))[:3]}, "
                     f"extra {sorted(set(got) - set(inv))[:3]})")
        else:
            try:
                wrapper.load_state_dict(sd, strict=True)
            except Exception as err:                      # noqa: BLE001 -- any load failure is a refusal
                p.append(f"strict load failed: {err}")
    opts = ck.get("optimizer_states") or []
    sch = (ck.get("lr_schedulers") or [{}])[0]
    if len(opts) != 1 or len(opts[0].get("param_groups", [])) != 1:
        p.append("expected one optimizer with one param group")
    else:
        g, st = opts[0]["param_groups"][0], opts[0].get("state", {})
        if (tuple(g.get("betas", ())), g.get("eps"), g.get("weight_decay"), g.get("initial_lr")) != ((0.9, 0.999), 1e-08, 0.001, 5e-05):
            p.append(f"AdamW group {g.get('betas')}/{g.get('eps')}/{g.get('weight_decay')}/{g.get('initial_lr')} off-recipe")
        if g.get("lr") != (sch.get("_last_lr") or [None])[0]:
            p.append("optimizer lr != scheduler _last_lr")
        if len(st) != N_OPT_STATE:
            p.append(f"optimizer state for {len(st)} parameters != {N_OPT_STATE}")
        if inventory:
            if g.get("params") != list(range(len(shapes))):
                p.append(f"param group lists {len(g.get('params', []))} params != the fresh wrapper's {len(shapes)}")
            else:
                for i, s_ in st.items():
                    ok = (tuple(s_.get("exp_avg", torch.empty(0)).shape) == shapes[i] ==
                          tuple(s_.get("exp_avg_sq", torch.empty(0)).shape))
                    fin = all(bool(torch.isfinite(s_[k]).all()) for k in ("exp_avg", "exp_avg_sq") if k in s_)
                    if not ok or not fin or float(s_.get("step", -1)) != float(step):
                        p.append(f"optimizer state {i} shape / finiteness / step off")
                        break
    if (sch.get("inv_gamma"), sch.get("power"), sch.get("warmup"), sch.get("final_lr"), sch.get("base_lrs"),
            sch.get("last_epoch"), sch.get("_step_count")) != (1000000, 0.5, 0.99, 0.0, [5e-05], step, step + 1):
        p.append(f"scheduler {[sch.get(k) for k in ('inv_gamma', 'power', 'warmup', 'final_lr', 'base_lrs', 'last_epoch', '_step_count')]} off-recipe")
    elif abs((sch.get("_last_lr") or [0])[0] - inverse_lr(step)) > 1e-12:
        p.append(f"scheduler _last_lr {(sch.get('_last_lr') or [0])[0]} != InverseLR({step}) {inverse_lr(step)}")
    try:
        fl = ck["loops"]["fit_loop"]
        done = fl["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"]
        bp = fl["epoch_loop.batch_progress"]["total"]
        ep = fl["epoch_progress"]["total"]
    except (KeyError, TypeError):
        done, bp, ep = None, {}, {}
    if done != step or bp.get("ready") != step or bp.get("processed") != step:
        p.append(f"fit-loop progress {done} / {bp} != step {step}")
    if ep.get("processed") != epoch_of_step(step, tag) and ep.get("processed") != epoch_of_step(step, tag) + 1:
        p.append(f"epoch progress {ep} inconsistent with epoch {epoch_of_step(step, tag)}")
    cbs = ck.get("callbacks") or {}
    if list(cbs) != [MODEL_CHECKPOINT_KEY]:
        p.append(f"callback state keys {list(cbs)} != [{MODEL_CHECKPOINT_KEY}]")
    elif run_dir is not None and os.path.realpath(cbs[MODEL_CHECKPOINT_KEY].get("dirpath", "")) != os.path.realpath(run_dir):
        p.append(f"ModelCheckpoint dirpath {cbs[MODEL_CHECKPOINT_KEY].get('dirpath')} != {run_dir}")
    if p:
        raise GateError(f"{path}: " + "; ".join(p))
    return {"ckpt": path, "sha256": digest, "bytes": len(data), "global_step": step}


def write_ckpt_manifest(path, run):
    _, tag, _ = parse_run_id(run)
    check_run_dir(path, run, MAX_STEPS, allow_submission_marker=True)
    lines = [f"{sha256_file(os.path.join(path, ckpt_name(s, tag)))}  {ckpt_name(s, tag)}"
             for s in cadence_steps()]
    out = os.path.join(path, "CHECKPOINTS.sha256")
    if os.path.exists(out):
        raise GateError(f"{out} already exists")
    with open(out + ".tmp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(out + ".tmp", out)
    return out


# ---------------------------------------------------------------- argv
def save_dir(platform, run, save_root=None):
    return os.path.join(save_root or PLATFORMS[platform]["save_root"], run)


def build_train_argv(run, platform, save_root=None, resume_step=0, allow_nonmatched=False):
    """argv for train.py (the caller prefixes ``python`` or ``torchrun …``). Monolithic 0 -> 40,000;
    ``resume_step`` > 0 only for a crash restart (the run is then non-primary, plan §6)."""
    arm, tag, rung = parse_run_id(run)
    r = check_rung(platform, rung, allow_nonmatched)
    leg_start, leg_end = resume_step, MAX_STEPS
    check_leg(leg_start, leg_end)
    sd = save_dir(platform, run, save_root)
    argv = ["train.py",
            "--model-config", model_config_rel(arm, rung),
            "--dataset-config", FRACS[tag]["dataset_config"],
            "--pretransform-ckpt-path", VAE_PATH,
            "--max-steps", str(leg_end),
            "--batch-size", str(r["micro"]), "--accum-batches", "1", "--num-workers", "6",
            "--seed", str(SEED), "--num-gpus", str(r["ngpu"]), "--num-nodes", "1",
            "--strategy", "ddp_find_unused_parameters_true", "--sync-batchnorm", "true",
            "--precision", "bf16-mixed", "--val-every", "-1", "--gradient-clip-val", "0.0",
            "--logger", "none", "--checkpoint-every", str(CADENCE),
            "--name", run, "--experiment-name", run, "--save-dir", sd,
            "--evidence-json", os.path.join(sd, "evidence", "run_evidence.json")]
    if leg_start:
        argv += ["--ckpt-path", os.path.join(sd, ckpt_name(leg_start, tag))]
    if r["ngpu"] * r["micro"] != EFF_BATCH or any("exp14_" in a for a in argv):
        raise GateError("argv invariant broken")
    return argv


def build_smoke_argv(run, steps, smoke_dir, ddp=False, platform="a6000"):
    """Storage-light smoke at the real per-rank micro-batch. ``ddp=False``: one rank (no DDP /
    SyncBN; a preliminary rung only). ``ddp=True``: the production topology of the rung (2 x 32
    + SyncBN), exactly the FULL argv but for the step budget, cadence and save dir."""
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


def wait_start_gate(log_path, run, backend, ngpu, pid, timeout_s, poll_s=10.0):
    """Poll the log until the first optimizer step (or the pid exits / timeout), then gate it."""
    deadline = time.time() + timeout_s
    while True:
        text = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""
        if first_step_index(log_lines(text)) is not None:
            return log_gate(text, run, "start", backend, ngpu)
        alive = pid is None or os.path.exists(f"/proc/{pid}")
        if not alive:
            return ["the training process exited before its first step"]
        if time.time() > deadline:
            return [f"no first step within {timeout_s} s"]
        time.sleep(poll_s)


# ---------------------------------------------------------------- immutable tool snapshot
def snapshot_files(run, repo=None):
    """Every campaign-closure file + the receipts the audits read (Codex plan r2 HIGH 7)."""
    files = set(closure_files(repo))
    for rel in (SUBSET_AUDIT + ".sha256", f"{EXP_DIR}/data_efficiency_e0_receipt.json",
                f"{EXP_DIR}/data_efficiency_e0_receipt.json.sha256", env_receipt_rel("a6000")):
        if os.path.exists(_abs(rel, repo)):
            files.add(rel)
    for path in glob.glob(_abs(smoke_receipt_glob(run), repo)):
        rel = os.path.relpath(path, repo or REPO)
        files.update({rel, rel + ".sha256"} if os.path.exists(path + ".sha256") else {rel})
    return sorted(files)


def snapshot_tools(run, dest, repo=None):
    """Copy the files the start / end audits read into ``dest`` (refuses an existing dest) and
    write ``SNAPSHOT.sha256``; the audits then run ``python <dest>/<registry> --repo <dest> …``."""
    import shutil
    if os.path.exists(dest):
        raise GateError(f"snapshot dir {dest} exists")
    lines = []
    for rel in snapshot_files(run, repo):
        src = _abs(rel, repo)
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(src, out)
        lines.append(f"{sha256_file(out)}  {rel}")
    os.makedirs(os.path.join(dest, ".git"), exist_ok=True)      # repo marker for _repo_root
    with open(os.path.join(dest, "SNAPSHOT.sha256"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"snapshot": dest, "files": len(lines)}


def verify_snapshot(dest):
    for line in open(os.path.join(dest, "SNAPSHOT.sha256")).read().strip().split("\n"):
        digest, rel = line.split("  ", 1)
        if sha256_file(os.path.join(dest, rel)) != digest:
            raise GateError(f"snapshot file {rel} changed")
    return {"snapshot": dest, "verified": True}


# ---------------------------------------------------------------- in-process evidence gate
def evidence_gate(evidence_dir, run, ngpu, backend, max_steps=MAX_STEPS):
    """Problems (empty = PASS) of the per-rank first-step evidence (plan r2 §10 B3)."""
    arm, _, rung = parse_run_id(run)
    micro = RUNGS[rung]["micro"] if ngpu > 1 else 32
    p, recs = [], []
    for r in range(ngpu):
        path = os.path.join(evidence_dir, f"run_evidence.json.rank{r}.json")
        try:
            recs.append(json.load(open(path)))
        except (OSError, ValueError) as err:
            p.append(f"rank {r} evidence unreadable: {err}")
    if p:
        return p
    for r, e in enumerate(recs):
        want = {"rank": r, "world_size": ngpu, "num_nodes": 1, "precision": "bf16-mixed",
                "accumulate_grad_batches": 1, "max_steps": max_steps, "gradient_clip_val": 0.0,
                "global_step_at_fit_start": 0, "global_step_recorded": 1, "micro_batch": micro,
                "state_dict_sha256": INIT_STATE_SHA256, "vit_ckpt_wrapped_layers": 12,
                "flash_attn_func_set": backend == "flash"}
        for k, v in want.items():
            if e.get(k) != v:
                p.append(f"rank {r}: {k} {e.get(k)!r} != {v!r}")
        sob = e.get("sobol") or {}
        if (sob.get("state_sha256"), sob.get("num_generated")) != (INIT_SOBOL_SHA256, 0):
            p.append(f"rank {r}: Sobol state off the registered initial state")
        if ngpu > 1 and (e.get("syncbn_modules", 0) < 1 or e.get("batchnorm_nonsync_modules") != 0):
            p.append(f"rank {r}: SyncBN not active ({e.get('syncbn_modules')} sync / {e.get('batchnorm_nonsync_modules')} plain)")
        ck = e.get("checkpoint_calls") or []
        if not ck or any(c.get("use_reentrant") is not False for c in ck):
            p.append(f"rank {r}: checkpoint calls {len(ck)}, not all use_reentrant=False")
        if backend == "flash" and not (e.get("flash_calls", 0) > 0 and e.get("sdpa_calls") == 0):
            p.append(f"rank {r}: flash {e.get('flash_calls')} / sdpa {e.get('sdpa_calls')} calls in the DiT")
        if backend == "sdpa" and not (e.get("flash_calls") == 0 and e.get("sdpa_calls", 0) > 0):
            p.append(f"rank {r}: flash {e.get('flash_calls')} / sdpa {e.get('sdpa_calls')} calls in the DiT")
        want_calls = [[micro, None]]
        if ARMS[arm]["cond_method"] == "fa_invariant":
            want_calls += [[micro, ["source_vit", "context_poses_vit"]]] * 3
        if e.get("conditioner_calls") != want_calls:
            p.append(f"rank {r}: conditioner calls {e.get('conditioner_calls')} != {want_calls}")
    if len({e.get("rng_at_fit_start", {}).get("torch_cpu") for e in recs}) != 1:
        p.append("ranks disagree on the torch CPU RNG state at fit start")
    return p


# ---------------------------------------------------------------- campaign closure, environment, receipts
CLOSURE_ROOTS = ("train.py", "defaults.ini", "eval_FLAC.py", "pyproject.toml", "src", "overlays", "data/AR")
CLOSURE_EXP_PREFIXES = ("exp38_", "FLAC_AR_BF_cap", "data_efficiency_subset_audit.json")


def closure_files(repo=None):
    import subprocess
    repo = repo or REPO
    out = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--", *CLOSURE_ROOTS, EXP_DIR, *EXTERNAL_CONFIGS],
                         capture_output=True, check=True).stdout.decode().split("\0")
    files = []
    for rel in out:
        if not rel or rel.startswith("src/tests/"):
            continue
        if rel.startswith(EXP_DIR + "/") and not os.path.basename(rel).startswith(CLOSURE_EXP_PREFIXES):
            continue
        files.append(rel)
    return sorted(set(files))


def closure_digest(repo=None):
    repo = repo or REPO
    h = hashlib.sha256()
    files = closure_files(repo)
    for rel in files:
        h.update(f"{rel}\0{sha256_file(os.path.join(repo, rel))}\n".encode())
    return {"closure_sha256": h.hexdigest(), "n_files": len(files)}


def check_untracked(platform, repo=None):
    """Refuse import shadows and unexpected links in the campaign tree (Codex plan r2 HIGH 7)."""
    import subprocess
    repo = repo or REPO
    p = []
    untracked = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--others", "--exclude-standard"],
                               capture_output=True, check=True).stdout.decode().split("\0")
    allow = SYMLINK_ALLOWLIST.get(platform)
    names = set(allow if allow is not None else SYMLINK_ALLOWLIST["a6000"])   # neuronic: names only
    for rel in filter(None, untracked):
        if rel.startswith(("worklog/", "outputs_FLAC/")) and not rel.endswith(".py"):
            continue
        if rel in names:
            continue
        p.append(f"unexpected untracked entry {rel}")
    if allow is not None:
        for rel, target in allow.items():
            path = os.path.join(repo, rel)
            if not os.path.islink(path) or os.readlink(path) != target:
                p.append(f"{rel} is not the symlink -> {target}")
    for top in ("", "src", "src/models", "src/training", "src/data", "overlays"):
        d = os.path.join(repo, top)
        for name in (os.listdir(d) if os.path.isdir(d) else []):
            if name in SHADOW_NAMES:
                p.append(f"import shadow {os.path.join(top, name)}")
    for k in ("ENABLE_TORCH_COMPILE", "PYTHONSTARTUP", "PYTHONINSPECT"):
        if os.environ.get(k):
            p.append(f"environment variable {k} is set")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        p.append("PYTHONNOUSERSITE=1 is required")
    return p


def env_snapshot():
    import importlib.metadata as md
    import platform as pf
    import subprocess
    import torch
    vers = {"python": pf.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}
    try:
        vers["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception:
        vers["nccl"] = None
    for dist in ("pytorch_lightning", "lightning_fabric", "transformers", "flash_attn", "torchaudio", "numpy",
                 "prefigure", "ema_pytorch", "huggingface_hub", "safetensors", "einops"):
        try:
            vers[dist] = md.version(dist.replace("_", "-"))
        except md.PackageNotFoundError:
            try:
                vers[dist] = md.version(dist)
            except md.PackageNotFoundError:
                vers[dist] = None
    try:
        vers["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=True).stdout.split()[0]
    except Exception:
        vers["driver"] = None
    return vers


def env_receipt_rel(platform):
    return f"{EXP_DIR}/data_efficiency_env_receipt_{platform}.json"


def env_receipt(platform, write=False, repo=None):
    """Write once (``write=True``) or check: the campaign environment must not drift between runs."""
    path = _abs(env_receipt_rel(platform), repo)
    now = env_snapshot()
    if platform == "a6000" and now.get("flash_attn") != FLASH_VERSION:
        raise GateError(f"flash-attn {now.get('flash_attn')} != pinned {FLASH_VERSION}")
    if write:
        if os.path.exists(path):
            raise GateError(f"{path} exists (the environment receipt is written once)")
        with open(path, "w") as fh:
            json.dump(now, fh, indent=1, sort_keys=True)
            fh.write("\n")
        return {"env_receipt": path, "sha256": sha256_file(path), "written": True}
    try:
        pinned = json.load(open(path))
    except OSError:
        raise GateError(f"no environment receipt {path}: write it at campaign start (E0)")
    if pinned != now:
        diff = {k: (pinned.get(k), now.get(k)) for k in set(pinned) | set(now) if pinned.get(k) != now.get(k)}
        raise GateError(f"environment drift vs the campaign receipt: {diff}")
    return {"env_receipt": path, "sha256": sha256_file(path)}


def _receipt(path):
    text = open(path, "rb").read()
    side = open(path + ".sha256").read().split()[0]
    if hashlib.sha256(text).hexdigest() != side:
        raise GateError(f"{path} does not match its .sha256")
    return json.loads(text)


def check_e0_receipt(platform, repo=None):
    path = _abs(f"{EXP_DIR}/data_efficiency_e0_receipt.json", repo)
    try:
        rec = _receipt(path)
    except (OSError, ValueError) as err:
        raise GateError(f"E0 receipt missing / unreadable: {err}")
    if rec.get("verdict") != "PASS":
        raise GateError(f"E0 verdict {rec.get('verdict')}")
    cur = closure_digest(repo)["closure_sha256"]
    if rec.get("closure_sha256") != cur:
        raise GateError("E0 receipt was produced at a different campaign closure")
    for rel, digest in (rec.get("evaluator_files") or {}).items():
        if sha256_file(_abs(rel, repo)) != digest:
            raise GateError(f"{rel} changed since E0")
    if not rec.get("evaluator_files"):
        raise GateError("E0 receipt binds no evaluator files")
    return {"e0_receipt": path, "closure_sha256": cur}


def smoke_receipt_glob(run):
    return f"{EXP_DIR}/data_efficiency_smoke_{run}_*.json"


def check_smoke_receipt(run, platform, repo=None):
    cur = closure_digest(repo)["closure_sha256"]
    for path in sorted(glob.glob(_abs(smoke_receipt_glob(run), repo)), reverse=True):
        try:
            rec = _receipt(path)
        except (OSError, ValueError):
            continue
        if (rec.get("mode"), rec.get("run"), rec.get("platform"), rec.get("verdict"), rec.get("closure_sha256")) \
                == ("SMOKE_DDP", run, platform, "PASS", cur):
            return {"smoke_receipt": path}
    raise GateError(f"no PASS SMOKE_DDP receipt for {run} on {platform} at closure {cur[:12]}")


def write_smoke_receipt(out, run, platform, evidence_dir, rc, n_ckpt, log_gate_line, evidence_gate_line,
                        verdict, head, repo=None):
    """The SMOKE_DDP receipt FULL validates (plan r2 §10 B6 schema)."""
    import subprocess
    if os.path.exists(out):
        raise GateError(f"{out} exists")
    full = build_train_argv(run, platform)
    proj = [a for i, a in enumerate(full)
            if not (a.startswith("/") or (i and full[i - 1] in ("--max-steps", "--checkpoint-every", "--name",
                                                                  "--experiment-name", "--save-dir", "--evidence-json")))]
    ev = {}
    for path in sorted(glob.glob(os.path.join(evidence_dir, "run_evidence.json.rank*.json"))):
        ev[os.path.basename(path)] = json.load(open(path))
    arm, tag, rung = parse_run_id(run)
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().split("\n")
    rec = {"mode": "SMOKE_DDP", "run": run, "platform": platform, "head": head,
           "closure_sha256": closure_digest(repo)["closure_sha256"],
           "env_receipt_sha256": sha256_file(_abs(env_receipt_rel(platform), repo)),
           "argv_projection": proj, "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
           "dataset_config_sha256": FRACS[tag]["dataset_config_sha256"], "split_sha256": FRACS[tag]["split_sha256"],
           "rc": int(rc), "checkpoints_written": int(n_ckpt), "log_gate": log_gate_line,
           "evidence_gate": evidence_gate_line, "evidence": ev, "gpus": gpus, "verdict": verdict}
    if verdict == "PASS" and (rec["rc"] or rec["checkpoints_written"] or log_gate_line != "LOG GATE PASS"
                              or evidence_gate_line != "EVIDENCE GATE PASS" or len(ev) != RUNGS[rung]["ngpu"]):
        raise GateError("a PASS smoke receipt needs rc 0, no checkpoint, both gates PASS and every rank's evidence")
    text = json.dumps(rec, indent=1, sort_keys=True).encode()
    with open(out, "wb") as fh:
        fh.write(text)
    with open(out + ".sha256", "w") as fh:
        fh.write(f"{hashlib.sha256(text).hexdigest()}  {os.path.basename(out)}\n")
    return {"smoke_receipt": out, "verdict": verdict}


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


# ---------------------------------------------------------------- CLI
def _cli(argv=None):
    global REPO
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--repo", help="file-lookup root (an immutable tool snapshot)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--rung", default="2x32")
    for name in ("check", "argv"):
        s = sub.add_parser(name)
        s.add_argument("run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
        s.add_argument("--save-root"); s.add_argument("--resume-step", type=int, default=0)
        s.add_argument("--allow-nonmatched-rung", action="store_true")
    s = sub.add_parser("smoke-argv"); s.add_argument("run"); s.add_argument("--steps", type=int, required=True)
    s.add_argument("--save-dir", required=True); s.add_argument("--ddp", action="store_true")
    s.add_argument("--platform", default="a6000", choices=sorted(PLATFORMS))
    s = sub.add_parser("run-dir-check"); s.add_argument("dir"); s.add_argument("--run", required=True)
    s.add_argument("--leg-start", type=int, default=0); s.add_argument("--allow-submission-marker", action="store_true")
    s = sub.add_parser("audit-ckpt"); s.add_argument("path"); s.add_argument("--run", required=True)
    s.add_argument("--step", type=int, required=True); s.add_argument("--run-dir")
    s = sub.add_parser("ckpt-manifest"); s.add_argument("dir"); s.add_argument("--run", required=True)
    s = sub.add_parser("pins"); s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    s = sub.add_parser("log-gate"); s.add_argument("log"); s.add_argument("--run", required=True)
    s.add_argument("--phase", required=True, choices=("start", "end"))
    s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    s.add_argument("--ngpu", type=int, required=True); s.add_argument("--max-steps", type=int, default=MAX_STEPS)
    s.add_argument("--min-losses", type=int, default=50)
    s.add_argument("--wait", type=float, default=0.0); s.add_argument("--pid", type=int)
    s = sub.add_parser("snapshot"); s.add_argument("run"); s.add_argument("--dest", required=True)
    s = sub.add_parser("verify-snapshot"); s.add_argument("dest")
    s = sub.add_parser("evidence-gate"); s.add_argument("dir"); s.add_argument("--run", required=True)
    s.add_argument("--ngpu", type=int, required=True); s.add_argument("--backend", required=True, choices=("flash", "sdpa"))
    s.add_argument("--max-steps", type=int, default=MAX_STEPS)
    s = sub.add_parser("closure")
    s = sub.add_parser("smoke-receipt"); s.add_argument("out"); s.add_argument("--run", required=True)
    s.add_argument("--platform", required=True, choices=sorted(PLATFORMS)); s.add_argument("--evidence", required=True)
    s.add_argument("--rc", type=int, required=True); s.add_argument("--checkpoints", type=int, required=True)
    s.add_argument("--log-gate", required=True); s.add_argument("--evidence-gate", required=True)
    s.add_argument("--verdict", required=True, choices=("PASS", "FAIL")); s.add_argument("--head", required=True)
    s = sub.add_parser("untracked"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s = sub.add_parser("env-receipt"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s.add_argument("--write", action="store_true")
    s = sub.add_parser("e0-check"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s = sub.add_parser("smoke-check"); s.add_argument("run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s = sub.add_parser("dry-run"); s.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    s.add_argument("--rung", default="2x32"); s.add_argument("--save-root")
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
            print("\n".join(build_train_argv(a.run, a.platform, a.save_root, a.resume_step,
                                             a.allow_nonmatched_rung)))
        elif a.cmd == "smoke-argv":
            arm, tag, rung = parse_run_id(a.run)
            check_config(arm, rung); check_subset(tag)
            print("\n".join(build_smoke_argv(a.run, a.steps, a.save_dir, a.ddp, a.platform)))
        elif a.cmd == "run-dir-check":
            print(json.dumps(check_run_dir(a.dir, a.run, a.leg_start, a.allow_submission_marker)))
        elif a.cmd == "audit-ckpt":
            print(json.dumps(audit_ckpt(a.path, a.run, a.step, run_dir=a.run_dir)))
        elif a.cmd == "ckpt-manifest":
            print(write_ckpt_manifest(a.dir, a.run))
        elif a.cmd == "pins":
            print(json.dumps({**vae_check(), **vit_pin_check(), **backend_probe(a.backend)}))
        elif a.cmd == "log-gate":
            if a.wait:
                problems = wait_start_gate(a.log, a.run, a.backend, a.ngpu, a.pid, a.wait)
            else:
                problems = log_gate(open(a.log, errors="replace").read(), a.run, a.phase, a.backend,
                                    a.ngpu, max_steps=a.max_steps, min_losses=a.min_losses)
            print("LOG GATE " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "snapshot":
            print(json.dumps(snapshot_tools(a.run, a.dest)))
        elif a.cmd == "verify-snapshot":
            print(json.dumps(verify_snapshot(a.dest)))
        elif a.cmd == "evidence-gate":
            problems = evidence_gate(a.dir, a.run, a.ngpu, a.backend, a.max_steps)
            print("EVIDENCE GATE " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "closure":
            print(json.dumps(closure_digest()))
        elif a.cmd == "smoke-receipt":
            print(json.dumps(write_smoke_receipt(a.out, a.run, a.platform, a.evidence, a.rc, a.checkpoints,
                                                 a.log_gate, a.evidence_gate, a.verdict, a.head)))
        elif a.cmd == "untracked":
            problems = check_untracked(a.platform)
            print("UNTRACKED " + ("PASS" if not problems else "FAIL: " + " | ".join(problems)))
            return 0 if not problems else 1
        elif a.cmd == "env-receipt":
            print(json.dumps(env_receipt(a.platform, write=a.write)))
        elif a.cmd == "e0-check":
            print(json.dumps(check_e0_receipt(a.platform)))
        elif a.cmd == "smoke-check":
            print(json.dumps(check_smoke_receipt(a.run, a.platform)))
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
