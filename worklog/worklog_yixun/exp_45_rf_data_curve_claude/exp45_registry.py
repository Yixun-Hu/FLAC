#!/usr/bin/env python3
"""exp_45 rf_data_curve — the single source of truth for every training run.

Forked from exp_38's exp38_registry.py (exp_45 commit A, a byte copy); every later change is a delta of
plan_rf_data_curve.md r3 §3.2. exp_38's tools and artifacts are reused READ-ONLY at their own paths
(``EXP38_DIR``): its registry (imported by its two tools), the subset audit + content manifest, the 68
reference cells and the sealed E0a decision.

Every launch path (the neuronic sbatch + submitter, the eval driver) asks THIS module for the run's
config, dataset config, argv, expected checkpoint names and gates. Exactly three runs exist: exp_43's RF
arm (full-circle roll + mirror flip p = 1/2) at exp_38's 25 / 50 / 75 % subsets on neuronic 2 x L40 x 32
(plan_rf_data_curve.md r3 §2, §3.2; exp_38: plan_data_efficiency.md §6-§7, §10 item 4).

Pure functions on top; a small CLI at the bottom (``python exp45_registry.py -h``). torch is
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
    """The checkout (``.git``) or the closure snapshot (``SNAPSHOT.sha256``) this file lives in."""
    d = start
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")) or os.path.isfile(os.path.join(d, "SNAPSHOT.sha256")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"no .git / SNAPSHOT.sha256 above {start}")


REPO = _repo_root(HERE)
EXP_DIR = "worklog/worklog_yixun/exp_45_rf_data_curve_claude"
# exp_38's folder (plan r3 §3.2): bound paths of the reused tools / artifacts keep naming it
EXP38_DIR = "worklog/worklog_yixun/exp_38_data_efficiency_claude"
REF_DIR = f"{EXP38_DIR}/ref_cells"                       # the ONE reference directory (Codex plan r2 finding 1)
REF_MANIFEST = f"{REF_DIR}/REF_MANIFEST.sha256"
E0A_DECISION = f"{EXP38_DIR}/data_efficiency_e0a_decision.json"   # unchanged bytes: E0a's sealed exception

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
YAW_BANNER = "yaw_aug ENABLED img_w=512 seed=42"            # exp_15's YAW line: never in an exp_45 log
# exp_43's RF line: printed ONCE, by rank 0 (diffusion.py _print_yaw_aug_banner is rank_zero_only; plan F1)
RF_BANNER = "yaw_aug ENABLED img_w=512 seed=42 flip=0.5"
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

ARMS = {
    # exp_43's RF arm, its config used in place and unchanged (sha256 at 4a7e3b3a; plan F1): exp_15's YAW config
    # + exactly training.yaw_aug.flip = true
    "rf": {"config": "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json",
           "config_sha256": "d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d",
           "cond_method": "vanilla", "banner": RF_BANNER, "min_free_mib": 12288, "required": True},
}
RF_YAW_AUG = {"enabled": True, "img_w": 512, "seed": 42, "flip": True}
# exp_38's E0 reference configs (E0a / E0d: P1, E0b: B-F 100 %, E0c: YAW 100 %): read by the E0 cells, never
# launchable training arms (plan r3 §3.2)
E0_CONFIGS = {
    "bvp1": {"config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
             "config_sha256": "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"},
    "bf_base": {"config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                "config_sha256": "3e677204902c543801679610b58d818de3f6353e7b95baded2667078135ea328"},
    "yaw": {"config": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
            "config_sha256": "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004"},
}
RUNGS = {"2x32": {"ngpu": 2, "micro": 32, "default": True}}
PLATFORMS = {
    "a6000": {"rungs": ("2x32",), "launcher": "python",
              "save_root": "/media/diskstation/yixunhu/FLAC/checkpoints/exp45_rf_data_curve"},
    "neuronic": {"rungs": ("2x32",), "launcher": "torchrun",
                 "save_root": "/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp45_rf_data_curve"},
}
OPTIONAL_RUNS = ()


# seed-42 initial wrapper state (exp38_init_digest.py on the CPU == the in-process evidence of the
# GPU-0 smokes, 2026-11-07; identical for YAW, B-F and P1 -- their .model sections are equal). exp_45's
# REGISTERED PREDICTION for RF is the same value (plan r3 §3.2: the flip adds no state tensor and no RNG draw at
# construction); verified fail-closed by exp45_init_digest.py (L3) and the SMOKE evidence
INIT_STATE_SHA256 = "378440d09589c8901f43bdb9e4da97b7401e7e51534d02d00488676cec9ccf95"
INIT_SOBOL_SHA256 = "2b099850ab9d888b168c8ca18dd693819a41814b682c5df0fa5ca9be7bc58798"
FLASH_VERSION = "2.7.4.post1"
CAMPAIGN_ROOT = {"a6000": "/home/yixunhu/codespace/exp-45-campaign"}   # tree 45 (plan r3 §4)
# exact untracked symlinks a campaign tree may carry on this box (content is pinned separately)
SYMLINK_ALLOWLIST = {"a6000": {"AcousticRooms": "/media/diskstation/yixunhu/FLAC/AcousticRooms",
                               "weights/AGREE": "/home/yixunhu/codespace/FLAC/weights/AGREE",
                               "weights/FLAC/VAE.safetensors": "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.safetensors",
                               "weights/FLAC/VAE.ckpt": "/home/yixunhu/codespace/FLAC/weights/FLAC/VAE.ckpt"}}
EXTERNAL_CONFIGS = (ARMS["rf"]["config"],) + tuple(v["config"] for v in E0_CONFIGS.values())
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
    return f"exp45_{arm.upper()}f{tag}_l40_{rung}"


def required_runs(rung="2x32"):
    return [run_id("rf", t, rung) for t in ("025", "050", "075")]


def all_runs(rung="2x32"):
    return required_runs(rung)


# the launchable matrix: exactly the three RF runs (plan r3 §2, §3.2) -- no P1 bridge, no 8x8, no _r suffix
REGISTERED_RUNS = frozenset(required_runs("2x32"))
# D1 (Yixun, 2026-11-14): exp_45 trains on neuronic 2 x L40 x 32 (exp_38's D2). FULL refuses elsewhere.
TRAINING_PLATFORM = "neuronic"


def campaign_runs():
    """The runs the campaign must train and evaluate: the three RF runs."""
    return required_runs("2x32")
_RUN_RE = re.compile(r"^exp45_RFf(025|050|075)_l40_2x32$")


def parse_run_id(run):
    m = _RUN_RE.match(run if isinstance(run, str) else "")
    if not m or run not in REGISTERED_RUNS:
        raise GateError(f"not a registered exp_45 run id: {run!r}")
    return "rf", m.group(1), "2x32"


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
def model_config_rel(arm, rung):
    return ARMS[arm]["config"]


def chunk_plan(arm, rung):
    """Announcement 06 record: the RF arm runs no frame-averaged conditioning, so there is no chunk plan."""
    if ARMS[arm]["cond_method"] != "vanilla":
        raise GateError(f"arm {arm}: only vanilla-conditioned arms are registered")
    return None


def check_config(arm, rung, repo=None):
    """Verify the arm's config file on disk; return its parsed dict."""
    rel = model_config_rel(arm, rung)
    path = _abs(rel, repo)
    if sha256_file(path) != ARMS[arm]["config_sha256"]:
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
    if canonical(tr.get("yaw_aug")) != canonical(RF_YAW_AUG):                    # type-strict (1 is not True)
        raise GateError(f"{rel}: training.yaw_aug is not the registered RF block {RF_YAW_AUG}")
    if any(k.startswith("frame_avg") for k in tr):
        raise GateError(f"{rel}: unexpected frame-averaging keys")
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


SUBSET_AUDIT = f"{EXP38_DIR}/data_efficiency_subset_audit.json"
# the BYTES of every AR file the registered runs read (exp38_content_manifest.py; Codex code r2 BLOCKING 3)
AR_CONTENT_MANIFEST = f"{EXP38_DIR}/data_efficiency_ar_content_manifest.txt.gz"
AR_CONTENT_SHA256 = "369bd4fcf015f232f5dbd0561ea77ccc6b32e1dcc6a71788e264bd9825f0042c"   # 445,414 files, 17.87 GB, 729 groups (2026-11-08)
SUBSET_AUDIT_BOUND = frozenset({"src/configs/dataset_configs/custom_metadata/AR_md.py", "src/data/dataset.py",
                                "data/AR/train_frac_manifest_s2026.json", "data/AR/train.json",
                                f"{EXP38_DIR}/exp38_subset_audit.py"}
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


RUN_DIR_OWN = (".exp38_lock", ".exp38_tools", "CHECKPOINTS.sha256", "evidence", "receipts", "TRAINING_COMPLETE.json",
               "TRAINING_COMPLETE.json.sha256", "TRAINING_COMMITTED.json", "TRAINING_COMMITTED.json.sha256")


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
        if upto == 0 and any(n in names for n in ("TRAINING_COMPLETE.json", "CHECKPOINTS.sha256", "TRAINING_COMMITTED.json")):
            raise GateError(f"{path}: a completed run's records are present")
    return {"run_dir": path, "state": "fresh" if not upto else "complete", "checkpoints": present}


def inverse_lr(step, base=5e-5, inv_gamma=1e6, power=0.5, warmup=0.99):
    """src/training/utils.py::InverseLR closed form (final_lr 0) at a completed step."""
    return base * (1 - warmup ** (step + 1)) * (1 + step / inv_gamma) ** (-power)


# exact checkpoint schema of a 40k checkpoint of this architecture under PL 2.1.0 (read from the
# YAW / B-F 100 % checkpoints and dc_van_f025, 2026-11-07; dc_van carries one extra exp_14 key)
N_OPT_STATE = 449            # of the 753 group parameters, those that receive gradients carry Adam state
N_OPT_PARAMS = 753
# the parameters WITHOUT Adam state (no gradient ever reaches them) -- identical in the dc_van_f025, YAW and B-F
# 40k endpoints (read 2026-11-07); every other index carries state
OPT_NO_STATE_RANGES = ((173, 175), (179, 179), (391, 393), (456, 752))
OPT_STATE_IDS = tuple(i for i in range(N_OPT_PARAMS) if not any(a <= i <= b for a, b in OPT_NO_STATE_RANGES))
# sha256 of the recursive key / type template of ck["loops"] (identical in the three real endpoints)
LOOPS_TEMPLATE_SHA256 = "b4c3a3568ee36f831ce4badb398f7e23a8cb634fd4e1d36b1ec8b820f07437ef"


def loops_template(x):
    if isinstance(x, dict):
        return {str(k): loops_template(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [type(x).__name__, [loops_template(v) for v in x]]
    return type(x).__name__


def _zero_progress(d):
    """All counters of a never-run loop's progress tree are 0 (bools False)."""
    if isinstance(d, dict):
        return all(_zero_progress(v) for v in d.values())
    return d in (0, False) and type(d) in (int, bool)
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
        import src
        from src.models import create_model_from_config
        from src.training import create_training_wrapper_from_config
        # the flac env also carries an upstream `src` package in site-packages: refuse to audit against it
        if not os.path.realpath(src.__file__).startswith(os.path.realpath(repo or REPO) + os.sep):
            raise GateError(f"`src` resolves to {src.__file__}, not inside {repo or REPO}")
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
        if not all(type(i) is int for i in st) or tuple(sorted(st)) != OPT_STATE_IDS:
            p.append(f"optimizer state ids ({len(st)}) != the registered {N_OPT_STATE} parameter ids")
        if g.get("params") != list(range(N_OPT_PARAMS)) or (opt_shapes is not None and len(opt_shapes) != N_OPT_PARAMS):
            p.append(f"param group lists {len(g.get('params', []))} params != {N_OPT_PARAMS}")
        for i, s_ in st.items():
            if not isinstance(s_, dict) or set(s_) != OPT_STATE_KEYS:
                p.append(f"optimizer state {i} keys {sorted(s_) if isinstance(s_, dict) else s_!r}")
                break
            if opt_shapes is not None and not (isinstance(i, int) and 0 <= i < len(opt_shapes)):
                p.append(f"optimizer state id {i!r} names no parameter")
                break
            shp = opt_shapes[i] if opt_shapes is not None else None
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
    elif hashlib.sha256(json.dumps(loops_template(loops), sort_keys=True).encode()).hexdigest() != LOOPS_TEMPLATE_SHA256:
        p.append("loop state structure / types differ from the registered PL 2.1 template")
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
        if fl["epoch_loop.state_dict"].get("_batches_that_stepped") != step - 1 \
                or fl["epoch_loop.scheduler_progress"]["total"] != {"ready": step, "completed": step} \
                or not _zero_progress(fl["epoch_loop.manual_optimization.optim_step_progress"]) \
                or not _zero_progress(fl["epoch_loop.val_loop.batch_progress"]) \
                or not all(_zero_progress(loops[k]["batch_progress"]) for k in ("predict_loop", "test_loop", "validate_loop")):
            p.append("loop counters off (batches stepped / scheduler / manual / validation / test / predict loops)")
    cbs = ck.get("callbacks")
    if not isinstance(cbs, dict) or list(cbs) != [MODEL_CHECKPOINT_KEY]:
        p.append(f"callback state keys {list(cbs) if isinstance(cbs, dict) else cbs!r} != [{MODEL_CHECKPOINT_KEY}]")
    else:
        mc = cbs[MODEL_CHECKPOINT_KEY]
        if not isinstance(mc, dict) or set(mc) != MODEL_CHECKPOINT_STATE_KEYS:
            p.append("ModelCheckpoint state keys differ")
        elif run_dir is not None and os.path.realpath(str(mc.get("dirpath", ""))) != os.path.realpath(run_dir):
            p.append(f"ModelCheckpoint dirpath {mc.get('dirpath')} != {run_dir}")
        else:
            kv = mc.get("kth_value")
            want_mc = {"monitor": None, "best_model_score": None, "current_score": None, "best_k_models": {},
                       "kth_best_model_path": "", "last_model_path": "",
                       "best_model_path": os.path.join(str(mc.get("dirpath")), ckpt_name(step, tag))}
            bad = sorted(k for k, v in want_mc.items() if mc.get(k) != v or type(mc.get(k)) is not type(v))
            if not (torch.is_tensor(kv) and kv.dim() == 0 and kv.dtype == torch.float32 and float(kv) == float("inf")):
                bad.append("kth_value")
            if bad:
                p.append(f"ModelCheckpoint state values {bad} off the registered save_top_k=-1 / monitor=None state")
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
_EPOCH_TOTAL = re.compile(r"Epoch \d+:\s+\d+%\|[^|]*\|\s*\d+/(\d+)")      # the loader length per epoch


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
    # exactly ONE RF banner line (rank 0 only, plan F1), before the first step; no other arm's yaw_aug line
    banner = ARMS[arm]["banner"]
    idx = [i for i, line in enumerate(lines) if line.strip() == banner]
    if len(idx) != 1 or (step_i is not None and idx[0] > step_i):
        problems.append(f"banner {banner!r}: {len(idx)} line(s); exactly one, before the first step")
    other = sorted({line.strip() for line in lines if "yaw_aug ENABLED" in line and line.strip() != banner})
    if other:
        problems.append(f"another arm's yaw_aug banner {other[:2]} (exp_15 YAW / exp_40 C16)")
    # the loader length, observed from the progress bar's per-epoch step count (plan r3 §5 L5, evidence wording)
    totals = sorted({int(m.group(1)) for m in map(_EPOCH_TOTAL.search, lines) if m})
    if step_i is not None and totals != [FRACS[tag]["steps_per_epoch"]]:
        problems.append(f"progress-bar steps per epoch {totals} != the {tag} loader length "
                        f"{FRACS[tag]['steps_per_epoch']}")
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


# ---------------------------------------------------------------- the shared storage reservation (plan r3 §3.2)
# Codex plan r2 finding 7 / r3 finding 2: ONE ledger per save root, under an exclusive flock. The submitter reserves
# a candidate's full need before sbatch and any owner intent, refusing unless
#     free - sum(outstanding of the ledger's existing reservations) - need >= floor
# (the candidate subtracted ONCE, before insertion); the supervisor re-checks at FULL start
#     free - sum(outstanding of all reservations, its own included) >= floor
# (no extra candidate term; idempotent). outstanding = reserved - bytes already written in the run dir. A
# reservation is released only when its run is sealed (supervisor) or abandoned (submitter); a crash keeps it held.
STORAGE_LEDGER = ".exp45_storage.json"
STORAGE_LOCK = ".exp45_storage.lock"
STORAGE_FLOOR_BYTES = TRAIN_FREE_GIB * 2**30
CKPT_BYTES = 724_000_000             # >= one 40k checkpoint of this architecture (723,922,667-723,927,531 B measured)
RUN_RESERVE_BYTES = len(cadence_steps()) * CKPT_BYTES + 2 * 2**30      # 16 checkpoints + 2 GiB logs / snapshot
LEDGER_KEYS = frozenset({"run", "token", "run_dir", "reserved_bytes", "reserved_at"})


def free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def written_bytes(path):
    """Bytes already written under a run dir (lstat sizes, links not followed; 0 if absent)."""
    total = 0
    for d, _, files in os.walk(path, followlinks=False):
        for name in files:
            if not os.path.islink(os.path.join(d, name)):
                total += os.lstat(os.path.join(d, name)).st_size
    return total


class _Ledger:
    """The flock'd reservation ledger of one save root (the lock file is separate: the ledger is replaced atomically)."""

    def __init__(self, root):
        if not os.path.isdir(root):
            raise GateError(f"storage root {root} does not exist")
        self.root, self.path = root, os.path.join(root, STORAGE_LEDGER)

    def __enter__(self):
        import fcntl
        self.lock = open(os.path.join(self.root, STORAGE_LOCK), "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            data = json.load(open(self.path)) if os.path.exists(self.path) else {"schema": 1, "reservations": {}}
        except (OSError, ValueError) as err:
            self.__exit__(None, None, None)
            raise GateError(f"storage ledger {self.path} unreadable: {err}")
        if set(data) != {"schema", "reservations"} or data["schema"] != 1 or not isinstance(data["reservations"], dict) \
                or any(not isinstance(r, dict) or set(r) != LEDGER_KEYS or type(r["reserved_bytes"]) is not int
                       for r in data["reservations"].values()):
            self.__exit__(None, None, None)
            raise GateError(f"storage ledger {self.path} is not the registered schema")
        self.data = data
        return self

    def outstanding(self, exclude=None):
        return sum(max(0, r["reserved_bytes"] - written_bytes(r["run_dir"]))
                   for key, r in self.data["reservations"].items() if key != exclude)

    def save(self):
        atomic_write_bytes(self.path, (json.dumps(self.data, indent=1, sort_keys=True) + "\n").encode())

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()
        return False


def reserve_storage(run, token, run_dir, root, need=RUN_RESERVE_BYTES, floor=STORAGE_FLOOR_BYTES):
    """Submitter, before sbatch and any owner intent: insert the run's reservation or refuse (GateError)."""
    parse_run_id(run)
    if not (isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token)):
        raise GateError(f"reservation token {token!r} is not a 32-hex intent token")
    with _Ledger(root) as led:
        if run in led.data["reservations"]:
            raise GateError(f"{run} already holds a storage reservation (abandon / seal it first)")
        free, out = free_bytes(root), led.outstanding()
        if free - out - need < floor:
            raise GateError(f"storage: {free / 2**30:.1f} GiB free - {out / 2**30:.1f} GiB outstanding - "
                            f"{need / 2**30:.1f} GiB needed < the {floor / 2**30:.0f} GiB floor")
        led.data["reservations"][run] = {"run": run, "token": token, "run_dir": run_dir, "reserved_bytes": int(need),
                                         "reserved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        led.save()
        return {"reserved_bytes": int(need), "free_bytes": free, "outstanding_bytes": out}


def recheck_storage(run, token, root, floor=STORAGE_FLOOR_BYTES):
    """Supervisor at FULL start, under the same lock: the run's own reservation (its token) must exist, and
    free - sum(outstanding of all reservations, its own included) >= floor. Changes nothing."""
    with _Ledger(root) as led:
        mine = led.data["reservations"].get(run)
        if mine is None or mine["token"] != token:
            raise GateError(f"{run} holds no storage reservation of token {str(token)[:8]}")
        free, out = free_bytes(root), led.outstanding()
        if free - out < floor:
            raise GateError(f"storage: {free / 2**30:.1f} GiB free - {out / 2**30:.1f} GiB outstanding "
                            f"< the {floor / 2**30:.0f} GiB floor")
        return {"free_bytes": free, "outstanding_bytes": out, "reserved_bytes": mine["reserved_bytes"]}


def release_storage(run, token, root):
    """Sealed (supervisor) or abandoned (submitter): drop the run's reservation if it is this token's."""
    with _Ledger(root) as led:
        mine = led.data["reservations"].get(run)
        if mine is None:
            return None
        if mine["token"] != token:
            raise GateError(f"{run}'s storage reservation belongs to token {mine['token'][:8]}, not {str(token)[:8]}")
        del led.data["reservations"][run]
        led.save()
        return mine


# ---------------------------------------------------------------- campaign closure + immutable snapshot
# AGREE/: the metric code imports `AGREE.AGREE` -- every reference cell ran with the repo root (so the
# repo copy, not the site-packages one) first on sys.path; the snapshots keep that
CLOSURE_ROOTS = ("train.py", "defaults.ini", "eval_FLAC.py", "pyproject.toml", "src", "overlays", "data/AR", "AGREE")
CLOSURE_EXP_PREFIXES = ("exp45_",)


def ref_set():
    """The exact 68 exp_38 reference cells, relative to REF_DIR (exp38_eval.ref_set(): 8 E0 replays + the
    dc_van 25 / 50 / 75 % and P1 / YAW / B-F 100 % anchors)."""
    seeds = (42, 43, 44, 45, 46)
    out = [f"E0/{g}_K{k}_s42.json" for g in ("E0a", "E0b", "E0c", "E0d") for k in (8, 1)]
    out += [f"anchors/{m}/{m}_K{k}_s{s}.json" for m in ("P1", "YAW", "BF") for k in (8, 1) for s in seeds]
    out += [f"anchors/dc_van/dc_van_f{t}_K{k}_s{s}.json" for t in ("025", "050", "075") for k in (8, 1) for s in seeds]
    return sorted(out)


# the exact read-only exp_38 dependencies (plan r3 §3.2), at exp_38's own paths: they enter closure_files()
# and every snapshot through EXTERNAL_CONFIGS, without a logic change
EXP38_READONLY = (f"{EXP38_DIR}/exp38_registry.py", f"{EXP38_DIR}/exp38_subset_audit.py",
                  f"{EXP38_DIR}/exp38_content_manifest.py", SUBSET_AUDIT, SUBSET_AUDIT + ".sha256",
                  AR_CONTENT_MANIFEST, REF_MANIFEST, *(f"{REF_DIR}/{r}" for r in ref_set()),
                  E0A_DECISION, E0A_DECISION + ".sha256")
EXTERNAL_CONFIGS = EXTERNAL_CONFIGS + EXP38_READONLY


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


def snapshot_tools(run, dest, repo=None, links=SNAPSHOT_LINKS):
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
    made = []
    for rel in links:
        target = os.path.realpath(_abs(rel, repo))
        if not os.path.exists(target):
            raise GateError(f"{rel} does not resolve in {repo}")
        os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
        os.symlink(target, os.path.join(dest, rel))
        made.append(f"{target}  {rel}")
    atomic_write_bytes(os.path.join(dest, "SNAPSHOT.sha256"), ("\n".join(lines) + "\n").encode())
    atomic_write_bytes(os.path.join(dest, "SNAPSHOT.links"), ("\n".join(made) + "\n").encode())
    return {"snapshot": dest, "files": len(lines), "snapshot_sha256": snapshot_digest(dest)}


def snapshot_digest(dest):
    return hashlib.sha256(open(os.path.join(dest, "SNAPSHOT.sha256"), "rb").read()
                          + open(os.path.join(dest, "SNAPSHOT.links"), "rb").read()).hexdigest()


def snapshot_extras(dest, files, links):
    """Entries of the snapshot tree (walked lexically, links not followed) that are neither a listed
    file, a listed link, a parent directory of one, the two manifests, nor the bytecode Python itself
    writes for a listed module (``<dir>/__pycache__/<stem>.cpython-*.pyc``)."""
    allowed_dirs = {""}
    for rel in list(files) + list(links):
        d = os.path.dirname(rel)
        while d:
            allowed_dirs.add(d)
            d = os.path.dirname(d)
    stems = {(os.path.dirname(r), os.path.basename(r)[:-3]) for r in files if r.endswith(".py")}
    extras = []
    for root, dirs, fnames in os.walk(dest, followlinks=False):
        rel_root = os.path.relpath(root, dest)
        rel_root = "" if rel_root == "." else rel_root
        for d in list(dirs):
            rel = os.path.join(rel_root, d)
            if os.path.islink(os.path.join(root, d)):
                if rel not in links:
                    extras.append(rel)
                dirs.remove(d)
            elif rel not in allowed_dirs and not (d == "__pycache__" and rel_root in allowed_dirs):
                extras.append(rel + "/")
                dirs.remove(d)
        for f in fnames:
            rel = os.path.join(rel_root, f)
            if rel in files or rel in links or rel in ("SNAPSHOT.sha256", "SNAPSHOT.links"):
                continue
            if os.path.basename(rel_root) == "__pycache__" and not os.path.islink(os.path.join(root, f)):
                parent = os.path.dirname(rel_root)
                m = re.fullmatch(r"(.+)\.cpython-\d+(?:\.opt-\d)?\.pyc", f)
                if m and (parent, m.group(1)) in stems:
                    continue
            extras.append(rel)
    return sorted(extras)


def verify_snapshot(dest, want_digest=None):
    p = []
    files, links = [], []
    for line in filter(None, open(os.path.join(dest, "SNAPSHOT.sha256")).read().split("\n")):
        files.append(line.split("  ", 1)[1])
    for line in filter(None, open(os.path.join(dest, "SNAPSHOT.links")).read().split("\n")):
        links.append(line.split("  ", 1)[1])
    extras = snapshot_extras(dest, set(files), set(links))
    if extras:
        p.append(f"unmanifested snapshot entries {extras[:5]}")
    for line in filter(None, open(os.path.join(dest, "SNAPSHOT.sha256")).read().split("\n")):
        digest, rel = line.split("  ", 1)
        path = os.path.join(dest, rel)
        if os.path.islink(path) or not os.path.isfile(path) or sha256_file(path) != digest:
            p.append(f"snapshot file {rel} changed")
    linked = []
    for line in filter(None, open(os.path.join(dest, "SNAPSHOT.links")).read().split("\n")):
        target, rel = line.split("  ", 1)
        linked.append(rel)
        if not os.path.islink(os.path.join(dest, rel)) or os.readlink(os.path.join(dest, rel)) != target:
            p.append(f"snapshot link {rel} changed")
    for rel, want in ((VAE_PATH, VAE_SHA256), (VAE_CKPT_REL, VAE_CKPT_SHA256), (AGREE_REL, AGREE_SHA256)):
        if (rel in linked or os.path.dirname(rel) in linked) and sha256_file(os.path.join(dest, rel)) != want:
            p.append(f"snapshot {rel} is not the pinned file")
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


# the ONLY untracked (not ignored) entries a campaign tree may hold besides TREE_LINKS: the records the
# kit itself writes into its own folder (Codex code r2 HIGH 10)
UNTRACKED_RECORDS = re.compile(r"^" + re.escape(EXP_DIR) + r"/(rf_data_curve_[^/]+\.(json|sha256|log|out|md|txt\.gz)"
                               r"|at_launch_[^/]+\.json|eval_logs/rf_data_curve_[^/]+\.log|slurm_train_[^/]+\.out"
                               r"|exp45_neuronic_submissions\.log|\.collect\.lock)$")
IMPORTABLE = (".py", ".pyc", ".pyo", ".so", ".pyd", ".pth", ".egg-link", ".egg-info", ".dist-info")


def _importable(rel):
    base = os.path.basename(rel.rstrip("/"))
    return base.endswith(IMPORTABLE) or base.startswith(("sitecustomize", "usercustomize"))


def check_untracked(platform, repo=None):
    """Problems (empty = PASS): import shadows, unexpected untracked entries (an exact allowlist of
    the kit's record names), importable files anywhere outside the tracked tree (ignored ones too,
    ``__pycache__`` excepted), and every allowed link validated by content (and, on this box, by its
    registered target)."""
    repo = repo or REPO
    p = []
    untracked = _git_out(repo, "ls-files", "-z", "--others", "--exclude-standard").decode().split("\0")
    for rel in filter(None, untracked):
        if rel in TREE_LINKS or any(rel.startswith(k + "/") for k in TREE_LINKS):
            continue
        if UNTRACKED_RECORDS.match(rel) and not _importable(rel):
            continue
        p.append(f"unexpected untracked entry {rel}")
    ignored = _git_out(repo, "ls-files", "-z", "--others", "--ignored", "--exclude-standard").decode().split("\0")
    for rel in filter(None, ignored):
        if "__pycache__" in rel.split("/") or rel.rstrip("/") in TREE_LINKS:
            continue
        if _importable(rel):
            p.append(f"ignored importable entry {rel}")
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


# variables only the chosen launcher may create (PL's subprocess launcher / torchrun); an inherited one
# could change the topology, the seed (SLURM_PROCID) or a Trainer argument (PL_TRAINER_*)
LAUNCHER_VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE", "LOCAL_WORLD_SIZE",
                 "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "NODE_RANK")
LAUNCHER_PREFIXES = ("TORCHELASTIC_", "PL_", "LIGHTNING_")


def launcher_env_problems(platform, environ=None):
    environ = os.environ if environ is None else environ
    bad = sorted(k for k in environ if k in LAUNCHER_VARS or k.startswith(LAUNCHER_PREFIXES)
                 or (platform == "a6000" and k.startswith("SLURM_")))
    return [f"inherited launcher variables {bad} (only the launcher may set them)"] if bad else []


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
                              "driver", "gpu_names", "packages", "packages_sha256", "flags", "env_vars",
                              "module_origins", "dist_records"})
# the modules a run imports from site-packages: their resolved origin must lie inside the interpreter's
# prefix, and their distributions' RECORD files (per-file hashes of the installed bytes) are pinned
CRITICAL_MODULES = ("torch", "pytorch_lightning", "lightning_fabric", "transformers", "flash_attn", "numpy",
                    "torchaudio", "einops", "safetensors", "prefigure", "ema_pytorch", "huggingface_hub")


def _tree_digest(d):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            path = os.path.join(root, f)
            h.update(f"{os.path.relpath(path, d)}\0{sha256_file(path)}\n".encode())
    return h.hexdigest()


OPTIONAL_MODULES = ("flash_attn",)    # absent in an SDPA-only environment (recorded as "absent")


def record_identity(base, record_text):
    """sha256 over (path, CURRENT sha256) of every file a distribution RECORD lists (bytecode
    excluded: Python may regenerate it); a listed hash that no longer matches the file refuses
    (an installed file edited without its RECORD)."""
    import base64
    import csv
    h = hashlib.sha256()
    rows = sorted(r for r in csv.reader(record_text.splitlines()) if r)
    for row in rows:
        rel, digest = row[0], row[1] if len(row) > 1 else ""
        if "__pycache__" in rel.split("/") or rel.endswith(".pyc"):
            continue
        path = os.path.normpath(os.path.join(base, rel))
        if not os.path.isfile(path):
            raise GateError(f"installed file {rel} listed in RECORD is missing")
        cur = sha256_file(path)
        if digest.startswith("sha256="):
            want = base64.urlsafe_b64decode(digest[7:] + "=" * (-len(digest[7:]) % 4)).hex()
            if want != cur:
                raise GateError(f"installed file {rel} differs from its RECORD hash")
        h.update(f"{rel}\0{cur}\n".encode())
    return h.hexdigest()


def module_identity():
    """({module: resolved origin}, {module: identity of its CURRENT installed bytes}) -- fail-closed
    (Codex code r3 HIGH 4). With a RECORD: every listed file re-hashed (and checked against the
    recorded hash); without one: every file of the package directory. An optional backend module that
    is not installed is recorded as "absent" (Codex code r3 HIGH 8)."""
    import importlib.metadata as md
    import importlib.util
    prefix = os.path.realpath(sys.prefix)
    dists = md.packages_distributions()
    origins, records = {}, {}
    for mod in CRITICAL_MODULES:
        spec = importlib.util.find_spec(mod)
        origin = os.path.realpath(spec.origin) if spec is not None and spec.origin else None
        if origin is None and mod in OPTIONAL_MODULES:
            origins[mod] = records[mod] = "absent"
            continue
        if origin is None or not origin.startswith(prefix + os.sep):
            raise GateError(f"module {mod} resolves to {origin}, not inside {prefix}")
        origins[mod] = origin
        parts = []
        for name in sorted(dists.get(mod) or []):
            try:
                dist = md.distribution(name)
                text = dist.read_text("RECORD")
            except md.PackageNotFoundError:
                text = None
            if not text:
                parts = None
                break
            parts.append(record_identity(str(dist.locate_file("")), text))
        records[mod] = ("record:" + hashlib.sha256("\0".join(parts).encode()).hexdigest() if parts
                        else "tree:" + _tree_digest(os.path.dirname(origin)))
    return origins, records


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
           "cudnn": torch.backends.cudnn.version(), "nccl": nccl,
           # one box here; neuronic nodes may differ in driver -> probed (fail-closed) and recorded per job
           # in the launch record, not pinned across nodes
           "driver": drivers[0] if platform == "a6000" else "per-job (launch record)",
           "gpu_names": sorted({r[2] for r in rows}), "packages": dict(sorted(pk.items())),
           "packages_sha256": hashlib.sha256(canonical(dict(sorted(pk.items()))).encode()).hexdigest(),
           "flags": {"optimize": sys.flags.optimize, "no_user_site": sys.flags.no_user_site,
                     "ignore_environment": sys.flags.ignore_environment, "isolated": sys.flags.isolated},
           "env_vars": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(ENV_PREFIXES) and k != "CUDA_VISIBLE_DEVICES"}}
    rec["module_origins"], rec["dist_records"] = module_identity()
    if None in (rec["cuda"], rec["cudnn"]) or not rec["gpu_names"]:
        raise GateError("CUDA / cuDNN / GPU probe returned nothing")
    if platform == "a6000" and pk.get("flash-attn") != FLASH_VERSION:
        raise GateError(f"flash-attn {pk.get('flash-attn')} != pinned {FLASH_VERSION}")
    return rec


def env_receipt_rel(platform):
    return f"{EXP_DIR}/rf_data_curve_env_receipt_{platform}.json"


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
E0_RECEIPT = f"{EXP_DIR}/rf_data_curve_e0_receipt.json"


def check_e0_receipt(platform, repo=None):
    """The E0 receipt must be a complete PASS at THIS closure (exp45_eval.validate_e0_receipt; the
    environment-receipt binding applies on the A6000 box, where E0 ran)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import exp45_eval
    # on neuronic the eight E0 cells are copied beside the receipt and re-admitted OFFLINE (Codex code r3 HIGH 5)
    return exp45_eval.validate_e0_receipt(repo=repo, platform=platform, readmit=True)


# ---------------------------------------------------------------- SMOKE_DDP receipt (Codex code r1 BLOCKING 1)
SMOKE_STEPS_DDP = 30
# --logger none under the recorded flac env (no TensorBoard package) selects PL's CSV logger at the run dir (plan F4):
# a 30-step SMOKE leaves exactly lightning_logs/version_0/hparams.yaml (no metrics row exists before step 99, F5)
LOGGER_DIR = "lightning_logs"
SMOKE_LOGGER_ENTRIES = ["version_0", "version_0/hparams.yaml"]


def logger_entries(run_dir):
    """Every entry under <run_dir>/lightning_logs (relative, directories included), links not followed."""
    ll = os.path.join(run_dir, LOGGER_DIR)
    out = []
    for d, dirs, files in os.walk(ll, followlinks=False):
        rel = os.path.relpath(d, ll)
        out += [os.path.normpath(os.path.join(rel, x)) for x in dirs + files]
    return sorted(out)


def smoke_logger_problems(smoke_dir):
    """The SMOKE gate's logger check (plan r3 §3.2): the CSV logger was selected -- exactly
    lightning_logs/version_0/hparams.yaml, no TensorBoard event file -- before any FULL. The entries are checked by
    TYPE too, as the FULL mover does (Codex code r1 finding 16): version_0 a real directory, hparams.yaml a regular
    file, neither a symlink."""
    ll = os.path.join(smoke_dir, LOGGER_DIR)
    if not os.path.isdir(ll) or os.path.islink(ll):
        return [f"{ll}: no logger directory (PL's CSV logger was not selected)"]
    got = logger_entries(smoke_dir)
    p = [f"{ll}: a TensorBoard event file {e}" for e in got if "tfevents" in os.path.basename(e)]
    if got != SMOKE_LOGGER_ENTRIES:
        p.append(f"{ll}: entries {got} != exactly {SMOKE_LOGGER_ENTRIES}")
        return p
    v0 = os.path.join(ll, "version_0")
    hp = os.path.join(v0, "hparams.yaml")
    if os.path.islink(v0) or not os.path.isdir(v0) or os.path.islink(hp) or not os.path.isfile(hp):
        p.append(f"{v0}: version_0 must be a directory and hparams.yaml a regular file, neither a symlink")
    return p


SMOKE_KEYS = frozenset({"schema", "mode", "run", "platform", "backend", "head", "closure_sha256",
                        "logger_gate", "smoke_dir",
                        "env_receipt_sha256", "subset_audit_sha256", "snapshot_sha256", "snapshot_dir", "argv_projection",
                        "model_config_sha256", "dataset_config_sha256", "split_sha256", "rc", "checkpoints_written",
                        "log", "log_sha256", "log_gate", "evidence", "evidence_gate", "gpus", "verdict",
                        "launched_at", "finished_at"})


def smoke_receipt_rel(run, platform, ts):
    return f"{EXP_DIR}/rf_data_curve_smoke_{run}_{platform}_{ts}.json"


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


def _same_env(platform, env_sha, tree):
    """The campaign tree's sealed environment receipt is still the one checked at launch."""
    if env_receipt_sha(platform, tree) != env_sha:
        raise GateError("the environment receipt changed during the run")
    return env_sha


def write_smoke_receipt(run, platform, backend, head, smoke_dir, log_path, rc, gpus, snapshot_sha, launched_at,
                        ts, env_sha, repo=None, out_repo=None, snapshot_dir=None):
    """Recomputes every gate from the artifacts (never trusts a caller's verdict) and seals the
    receipt into ``out_repo`` (the campaign tree); the code bindings are computed from ``repo``
    (the snapshot the smoke ran from), the environment binding from the tree's receipt (``env_sha``
    = the sha checked at launch, re-checked here)."""
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
    xg = smoke_logger_problems(smoke_dir)
    verdict = "PASS" if (rc == 0 and n_ckpt == 0 and not lg and not eg and not xg) else "FAIL"
    rec = {"schema": 1, "mode": "SMOKE_DDP", "run": run, "platform": platform, "backend": backend, "head": head,
           "logger_gate": "PASS" if not xg else xg, "smoke_dir": smoke_dir,
           "closure_sha256": closure_digest(repo)["closure_sha256"],
           "env_receipt_sha256": _same_env(platform, env_sha, out_repo or repo),
           "subset_audit_sha256": check_subset_audit(repo)["sha256"], "snapshot_sha256": snapshot_sha,
           "snapshot_dir": snapshot_dir or repo,
           "argv_projection": argv_projection(build_train_argv(run, platform, allow_nonmatched=rung != "2x32")),
           "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
           "dataset_config_sha256": sha256_file(_abs(FRACS[tag]["dataset_config"], repo)),
           "split_sha256": sha256_file(_abs(FRACS[tag]["split"], repo)), "rc": int(rc),
           "checkpoints_written": n_ckpt, "log": log_path, "log_sha256": sha256_file(log_path),
           "log_gate": "PASS" if not lg else lg, "evidence": {str(r): v for r, v in sorted(recs.items())},
           "evidence_gate": "PASS" if not eg else eg, "gpus": gpus, "verdict": verdict,
           "launched_at": launched_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = _abs(smoke_receipt_rel(run, platform, ts), out_repo or repo)
    write_sealed(out, rec)
    return {"smoke_receipt": out, "verdict": verdict, "log_gate": rec["log_gate"], "evidence_gate": rec["evidence_gate"],
            "logger_gate": rec["logger_gate"]}


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
            "rc": 0, "checkpoints_written": 0, "log_gate": "PASS", "evidence_gate": "PASS", "verdict": "PASS",
            "logger_gate": "PASS"}
    p = [f"{k} {rec.get(k)!r} != {v!r}" for k, v in want.items() if canonical(rec.get(k)) != canonical(v)]
    if not (_is_sha(rec.get("snapshot_sha256")) and _is_sha(rec.get("log_sha256"))
            and isinstance(rec.get("log"), str) and isinstance(rec.get("snapshot_dir"), str)
            and isinstance(rec.get("smoke_dir"), str)
            and isinstance(rec.get("launched_at"), str) and isinstance(rec.get("finished_at"), str)):
        p.append("snapshot / log / time fields malformed")
        return p
    # the durable bindings are re-opened, not trusted (Codex code r2 HIGH 3)
    try:
        verify_snapshot(rec["snapshot_dir"], rec["snapshot_sha256"])
        if closure_digest(rec["snapshot_dir"])["closure_sha256"] != want["closure_sha256"]:
            p.append("the smoke's snapshot is not this closure")
    except (GateError, OSError, ValueError) as err:
        p.append(f"smoke snapshot: {err}")
    try:
        if sha256_file(rec["log"]) != rec["log_sha256"]:
            p.append("the smoke log changed since its receipt")
        else:
            p += log_gate(open(rec["log"], errors="replace").read(), run, "end", backend, ngpu,
                          max_steps=SMOKE_STEPS_DDP, min_losses=1)
    except OSError as err:
        p.append(f"smoke log: {err}")
    p += [f"smoke logger: {x}" for x in smoke_logger_problems(rec["smoke_dir"])]   # re-opened, not trusted
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
    paths = sorted(glob.glob(_abs(f"{EXP_DIR}/rf_data_curve_smoke_{run}_{platform}_*.json", repo)))
    if not paths:
        raise GateError(f"no SMOKE_DDP receipt for {run} on {platform}")
    rec, digest = read_sealed(paths[-1], SMOKE_KEYS)
    p = smoke_receipt_problems(rec, run, platform, backend, head, gpus, repo)
    if p:
        raise GateError(f"{paths[-1]}: " + "; ".join(p))
    return {"smoke_receipt": paths[-1], "sha256": digest}


# ---------------------------------------------------------------- the training-loss record (plan r3 §3.2, N3, F4 / F5)
LOSS_RECORD = "loss_record.csv"                     # receipts/: PL's CSV-logger metrics.csv (moved there by the FULL
LOSS_RECORD_HPARAMS = "loss_record_hparams.yaml"    # supervisor before the end audit), with its hparams.yaml
LOSS_COLUMNS = frozenset({"step", "train/std_data", "train/loss", "train/mse_loss", "epoch", "train/lr"})
LOSS_EVERY = 100                                    # PL 2.1 log_every_n_steps: the logging rank's batch loss


def check_loss_record(path, max_steps=MAX_STEPS):
    """{"loss_record_sha256", "loss_record_rows"} of a run's every-100-step batch-loss record, else GateError. The
    header is parsed BY NAME -- PL 2.1 builds the column order from a set, so any order passes (Codex plan r1
    finding 10) -- with no duplicate column (r2 finding 9) and exactly the registered column set; one row per logged
    step, step = 99 + 100 k for k = 0 .. 399 in order; every train/loss finite. The sha256 is of the file bytes."""
    import csv
    import io
    import math
    try:
        data = open(path, "rb").read()
        rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        raise GateError(f"loss record {path} missing / unreadable: {err}")
    head, body = (rows[0], rows[1:]) if rows else ([], [])
    if len(head) != len(set(head)):
        raise GateError(f"{path}: duplicate column in the header {head}")
    if set(head) != LOSS_COLUMNS:
        raise GateError(f"{path}: columns {sorted(head)} != {sorted(LOSS_COLUMNS)}")
    want = [LOSS_EVERY - 1 + LOSS_EVERY * k for k in range(max_steps // LOSS_EVERY)]
    if len(body) != len(want):
        raise GateError(f"{path}: {len(body)} rows != {len(want)} (one per {LOSS_EVERY} steps)")
    col = {name: i for i, name in enumerate(head)}
    for i, (row, step) in enumerate(zip(body, want), start=1):
        if len(row) != len(head) or row[col["step"]] != str(step):
            raise GateError(f"{path}: row {i} ({len(row)} fields, step {row[col['step']] if len(row) == len(head) else '?'}) "
                            f"!= step {step}")
        try:
            loss = float(row[col["train/loss"]])
        except ValueError:
            loss = float("nan")
        if not math.isfinite(loss):
            raise GateError(f"{path}: row {i} (step {step}) train/loss {row[col['train/loss']]!r} is not finite")
    return {"loss_record_sha256": hashlib.sha256(data).hexdigest(), "loss_record_rows": len(body)}


# ---------------------------------------------------------------- terminal training receipt (Codex code r1 BLOCKING 4)
TRAINING_RECEIPT = "TRAINING_COMPLETE.json"
FULL_MIN_LOSSES = 50          # the end gate of a 40k run: at least 50 finite progress-bar losses
TRAINING_KEYS = frozenset({"schema", "run", "platform", "backend", "head", "closure_sha256", "snapshot_sha256",
                           "data_content_checks",
                           "env_receipt_sha256", "e0_receipt_sha256", "smoke_receipt", "smoke_receipt_sha256",
                           "model_config_sha256", "dataset_config_sha256", "split_sha256", "data_content_sha256",
                           "receipts", "argv", "rc", "log",
                           "log_sha256", "log_gate", "evidence_sha256", "evidence_gate", "checkpoints_sha256",
                           "final_ckpt", "final_ckpt_sha256", "final_ckpt_bytes", "gpus", "launched_at",
                           "finished_at", "loss_record_sha256", "loss_record_rows"})


def evidence_digest(evidence_dir):
    h = hashlib.sha256()
    for name in sorted(os.listdir(evidence_dir)):
        h.update(f"{name}\0{sha256_file(os.path.join(evidence_dir, name))}\n".encode())
    return h.hexdigest()


RECEIPTS_DIR = "receipts"          # in the run dir: byte copies of the receipts the run was admitted under


def retain_receipts(rd, paths, plain=None):
    """Copy each sealed receipt (+ sidecar) the launch validated into <run dir>/receipts/ under its own
    basename (the sidecar names it), and each ``plain`` backing file (the SMOKE log) under
    ``<role>.<ext>``; returns {role: basename}. Called before the child starts."""
    import shutil
    d = os.path.join(rd, RECEIPTS_DIR)
    try:
        os.makedirs(d, exist_ok=False)
    except FileExistsError:
        raise GateError(f"{d} exists (receipts are retained once, at launch)")
    out = {}
    for role, path in paths.items():
        for src in (path, path + ".sha256"):
            shutil.copyfile(src, os.path.join(d, os.path.basename(src)))
        out[role] = os.path.basename(path)
    for role, path in (plain or {}).items():
        name = f"{role}{os.path.splitext(path)[1]}"
        shutil.copyfile(path, os.path.join(d, name))
        out[role] = name
    return out


COMMIT_MARKER = "TRAINING_COMMITTED.json"
COMMIT_KEYS = frozenset({"schema", "run", "training_receipt_sha256", "launch_record", "launch_record_sha256"})


def write_commit_marker(rd, run, training_sha, record_path):
    """The run's single final commit (Codex code r3 HIGH 7): written only after the sealed training
    receipt AND the terminal launch record (its run-dir copy) exist; binds both by sha."""
    rec = json.load(open(record_path))
    if rec.get("status") != "COMPLETE" or rec.get("run") != run or rec.get("training_receipt_sha256") != training_sha:
        raise GateError("the terminal launch record does not name this completed training receipt")
    return write_sealed(os.path.join(rd, COMMIT_MARKER), {
        "schema": 1, "run": run, "training_receipt_sha256": training_sha,
        "launch_record": os.path.basename(record_path), "launch_record_sha256": sha256_file(record_path)})


def closure_of_manifest(text):
    """closure_digest of a snapshot from its SNAPSHOT.sha256 text alone (reproducible offline)."""
    h = hashlib.sha256()
    rows = sorted((rel, d) for d, rel in (ln.split("  ", 1) for ln in text.split("\n") if ln) if _in_closure(rel))
    for rel, d in rows:
        h.update(f"{rel}\0{d}\n".encode())
    return h.hexdigest()


def _iso(t):
    import datetime
    return datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")


def write_training_receipt(run, platform, backend, head, rd, log_path, rc, snapshot_sha, e0_sha, smoke, gpus,
                           launched_at, env_sha, retained, content_sha, repo=None, tree=None):
    """Published only after every end audit passes (re-run here, from the snapshot ``repo``; the
    environment receipt is the campaign ``tree``'s, re-checked against the launch-time ``env_sha``).
    The train log is retained beside the launch-time receipt copies (``retained``)."""
    import shutil
    arm, tag, rung = parse_run_id(run)
    ngpu = RUNGS[rung]["ngpu"]
    if rc != 0:
        raise GateError(f"train exit status {rc}")
    if content_sha != AR_CONTENT_SHA256:
        raise GateError("the run was not admitted under the pinned AR content manifest")
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
    log_copy = os.path.join(rd, RECEIPTS_DIR, "train.log")
    shutil.copyfile(log_path, log_copy)
    if sha256_file(log_copy) != sha256_file(log_path):
        raise GateError("the retained train log differs from the log")
    for f in ("SNAPSHOT.sha256", "SNAPSHOT.links"):             # the snapshot is reproducible from the bundle
        shutil.copyfile(os.path.join(repo, f), os.path.join(rd, RECEIPTS_DIR, f))
    if snapshot_digest(os.path.join(rd, RECEIPTS_DIR)) != snapshot_sha:
        raise GateError("the retained snapshot manifests are not the launched snapshot's")
    loss = check_loss_record(os.path.join(rd, RECEIPTS_DIR, LOSS_RECORD))        # moved in by the supervisor
    if not os.path.isfile(os.path.join(rd, RECEIPTS_DIR, LOSS_RECORD_HPARAMS)):
        raise GateError(f"the loss record's {LOSS_RECORD_HPARAMS} is not in {RECEIPTS_DIR}/")
    rec = {"schema": 1, "run": run, "platform": platform, "backend": backend, "head": head,
           "closure_sha256": closure_digest(repo)["closure_sha256"], "snapshot_sha256": snapshot_sha,
           "env_receipt_sha256": _same_env(platform, env_sha, tree), "e0_receipt_sha256": e0_sha,
           "smoke_receipt": smoke["smoke_receipt"], "smoke_receipt_sha256": smoke["sha256"],
           "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung), repo)),
           "dataset_config_sha256": sha256_file(_abs(FRACS[tag]["dataset_config"], repo)),
           "split_sha256": sha256_file(_abs(FRACS[tag]["split"], repo)), "data_content_sha256": content_sha,
           "argv": build_train_argv(run, platform, allow_nonmatched=rung != "2x32"), "rc": 0, "log": log_path,
           "log_sha256": sha256_file(log_copy), "log_gate": "PASS",
           "evidence_sha256": evidence_digest(os.path.join(rd, "evidence")), "evidence_gate": "PASS",
           "checkpoints_sha256": sha256_file(manifest), "final_ckpt": final, "final_ckpt_sha256": audit["sha256"],
           "final_ckpt_bytes": audit["bytes"], "gpus": gpus,
           "receipts": dict(retained, log="train.log", snapshot_manifest="SNAPSHOT.sha256", snapshot_links="SNAPSHOT.links",
                            loss_record=LOSS_RECORD, loss_record_hparams=LOSS_RECORD_HPARAMS), **loss,
           "data_content_checks": ["pre-launch", "post-run"],
           "launched_at": launched_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_sealed(os.path.join(rd, TRAINING_RECEIPT), rec)
    return rec


def _gpus_ok(platform, ngpu, g):
    if platform == "a6000":
        return canonical(g) == canonical(sorted([u, "NVIDIA RTX A6000"] for u in A6000_UUIDS))
    return (isinstance(g, list) and len(g) == ngpu and len({str(x[0]) for x in g if isinstance(x, list)}) == ngpu
            and all(isinstance(x, list) and len(x) == 2 and str(x[0]).startswith("GPU-") and "L40" in str(x[1])
                    for x in g))


def check_training_receipt(run, rd, closure=None, receipt_dir=None, verify_bytes=True, e0_sha=None, platform=None):
    """The sealed terminal receipt of a run (in ``receipt_dir`` or the run dir), every field checked
    (Codex code r2 BLOCKING 1): the run, platform, backend, HEAD, the campaign ``closure`` / ``e0_sha``,
    the registered argv, configs, split and AR content, GPU topology; the RETAINED copies of the E0,
    SMOKE_DDP and environment receipts (sealed, their shas == the receipt's, the SMOKE one re-validated
    incl. its evidence); the retained train log (sha + the end gate re-run); the evidence (digest + the
    gate re-run); the checkpoint manifest and -- with ``verify_bytes`` -- the final checkpoint's current
    sha256 and byte count. Returns (receipt, receipt sha256)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import exp45_eval
    arm, tag, rung = parse_run_id(run)
    ngpu = RUNGS[rung]["ngpu"]
    src = receipt_dir or rd
    rec, digest = read_sealed(os.path.join(src, TRAINING_RECEIPT), TRAINING_KEYS)
    final = ckpt_name(MAX_STEPS, tag)
    p = []
    plat, backend = rec["platform"], rec["backend"]
    if plat not in PLATFORMS or (platform is not None and plat != platform) or backend not in ("flash", "sdpa"):
        raise GateError(f"{run} training receipt: platform {plat!r} / backend {backend!r}")
    want = {"schema": 1, "run": run, "rc": 0, "log_gate": "PASS", "evidence_gate": "PASS", "final_ckpt": final,
            "model_config_sha256": sha256_file(_abs(model_config_rel(arm, rung))),
            "dataset_config_sha256": FRACS[tag]["dataset_config_sha256"], "split_sha256": FRACS[tag]["split_sha256"],
            "data_content_sha256": AR_CONTENT_SHA256,
            "argv": build_train_argv(run, plat, allow_nonmatched=rung != "2x32"),
            "loss_record_rows": MAX_STEPS // LOSS_EVERY}
    p += [f"{k} {str(rec.get(k))[:60]!r} != {str(v)[:60]!r}" for k, v in want.items() if canonical(rec.get(k)) != canonical(v)]
    if closure is not None and rec["closure_sha256"] != closure:
        p.append(f"trained at closure {str(rec['closure_sha256'])[:12]}, not the campaign's {closure[:12]}")
    if e0_sha is not None and rec["e0_receipt_sha256"] != e0_sha:
        p.append("trained under another E0 receipt than the campaign's")
    if not (isinstance(rec["head"], str) and re.fullmatch(r"[0-9a-f]{40}", rec["head"])
            and all(_is_sha(rec[k]) for k in ("closure_sha256", "snapshot_sha256", "env_receipt_sha256",
                                              "e0_receipt_sha256", "smoke_receipt_sha256", "log_sha256",
                                              "evidence_sha256", "checkpoints_sha256", "final_ckpt_sha256",
                                              "loss_record_sha256"))
            and type(rec["final_ckpt_bytes"]) is int and isinstance(rec["launched_at"], str)
            and isinstance(rec["finished_at"], str) and isinstance(rec["log"], str)
            and isinstance(rec["smoke_receipt"], str)):
        p.append("hash / byte / time / path fields malformed")
    if not _gpus_ok(plat, ngpu, rec["gpus"]):
        p.append(f"GPUs {rec['gpus']} are not the {plat} topology")
    # ---- the retained receipts and log
    rdir = os.path.join(src, RECEIPTS_DIR)
    names = rec["receipts"] if isinstance(rec["receipts"], dict) else {}
    try:
        if sorted(names) != ["e0", "env", "log", "loss_record", "loss_record_hparams", "smoke", "smoke_log",
                             "snapshot_links", "snapshot_manifest"] \
                or (names["loss_record"], names["loss_record_hparams"]) != (LOSS_RECORD, LOSS_RECORD_HPARAMS):
            raise GateError(f"retained receipts {sorted(names)}")
        # the loss record is re-opened and re-validated, online and offline (plan r3 §3.2, Codex plan r1 finding 9)
        got = check_loss_record(os.path.join(rdir, LOSS_RECORD))
        if (got["loss_record_sha256"], got["loss_record_rows"]) != (rec["loss_record_sha256"], rec["loss_record_rows"]):
            p.append("the retained loss record != the training receipt's sha256 / rows")
        if not os.path.isfile(os.path.join(rdir, LOSS_RECORD_HPARAMS)):
            p.append(f"the retained {LOSS_RECORD_HPARAMS} is missing")
        e0c, d_e0 = read_sealed(os.path.join(rdir, names["e0"]), exp45_eval.E0_KEYS)
        envc, d_env = read_sealed(os.path.join(rdir, names["env"]), ENV_RECEIPT_KEYS)
        smk, d_smk = read_sealed(os.path.join(rdir, names["smoke"]), SMOKE_KEYS)
        if (d_e0, d_env, d_smk) != (rec["e0_receipt_sha256"], rec["env_receipt_sha256"], rec["smoke_receipt_sha256"]):
            p.append("a retained receipt copy != the sha the training receipt bound")
        if e0c["verdict"] != "PASS" or e0c["closure_sha256"] != rec["closure_sha256"]:
            p.append("the retained E0 receipt is not a PASS at the run's closure")
        p += [f"retained E0: {x}" for x in exp45_eval.e0_static_problems(e0c)]
        if envc["platform"] != plat:
            p.append("the retained environment receipt is another platform's")
        if not (isinstance(envc["packages"], dict) and envc["packages_sha256"] ==
                hashlib.sha256(canonical(dict(sorted(envc["packages"].items()))).encode()).hexdigest()
                and isinstance(envc["flags"], dict) and envc["flags"].get("optimize") == 0
                and envc["flags"].get("no_user_site") == 1
                and (plat != "a6000" or envc["packages"].get("flash-attn") == FLASH_VERSION)):
            p.append("the retained environment receipt is internally inconsistent / off the registered flags")
        snap_dir = rdir
        if snapshot_digest(snap_dir) != rec["snapshot_sha256"] or \
                closure_of_manifest(open(os.path.join(snap_dir, names["snapshot_manifest"])).read()) != rec["closure_sha256"]:
            p.append("the retained snapshot manifests are not the run's snapshot / closure")
        if not (_iso(rec["launched_at"]) <= _iso(rec["finished_at"])):
            p.append("finished before it launched")
        want_s = {"mode": "SMOKE_DDP", "run": run, "platform": plat, "backend": backend, "head": rec["head"],
                  "closure_sha256": rec["closure_sha256"], "env_receipt_sha256": rec["env_receipt_sha256"],
                  "rc": 0, "checkpoints_written": 0, "log_gate": "PASS", "evidence_gate": "PASS", "verdict": "PASS",
                  "logger_gate": "PASS", "argv_projection": argv_projection(rec["argv"])}
        p += [f"retained SMOKE {k} {str(smk.get(k))[:40]!r} != {str(v)[:40]!r}" for k, v in want_s.items()
              if canonical(smk.get(k)) != canonical(v)]
        ev = smk.get("evidence") or {}
        p += evidence_problems({int(r): v for r, v in ev.items()} if sorted(ev) == [str(r) for r in range(ngpu)] else {},
                               run, ngpu, backend, SMOKE_STEPS_DDP)
        if smk.get("schema") != 1 or not _gpus_ok(plat, ngpu, smk.get("gpus")) or not _is_sha(smk.get("snapshot_sha256")):
            p.append("retained SMOKE schema / GPUs / snapshot")
        slog = os.path.join(rdir, names["smoke_log"])
        if sha256_file(slog) != smk.get("log_sha256"):
            p.append("the retained SMOKE log != the SMOKE receipt's sha")
        else:
            p += [f"retained SMOKE log: {x}" for x in log_gate(open(slog, errors="replace").read(), run, "end", backend,
                                                               ngpu, max_steps=SMOKE_STEPS_DDP, min_losses=1)]
        if not (_iso(smk["finished_at"]) <= _iso(rec["launched_at"])):
            p.append("the SMOKE finished after this run launched")
        logc = os.path.join(rdir, names["log"])
        if sha256_file(logc) != rec["log_sha256"]:
            p.append("the retained train log != the receipt's sha")
        else:
            p += log_gate(open(logc, errors="replace").read(), run, "end", backend, ngpu, min_losses=FULL_MIN_LOSSES)
        evd = os.path.join(src, "evidence")
        if evidence_digest(evd) != rec["evidence_sha256"]:
            p.append("the evidence files != the receipt's digest")
        else:
            p += evidence_gate(evd, run, ngpu, backend)
    except (GateError, OSError, ValueError, KeyError, TypeError) as err:
        p.append(f"retained records: {err}")
    try:
        manifest = os.path.join(src, "CHECKPOINTS.sha256")
        if sha256_file(manifest) != rec["checkpoints_sha256"]:
            p.append("CHECKPOINTS.sha256 != the receipt's")
        lines = dict(reversed(ln.split("  ", 1)) for ln in open(manifest).read().strip().split("\n"))
        if sorted(lines) != sorted(ckpt_name(s, tag) for s in cadence_steps()) or lines[final] != rec["final_ckpt_sha256"]:
            p.append("checkpoint manifest set / final line")
        if verify_bytes:
            path = os.path.join(rd, final)
            if os.path.getsize(path) != rec["final_ckpt_bytes"] or sha256_file(path) != rec["final_ckpt_sha256"]:
                p.append(f"{path} sha / bytes != the training receipt")
    except (OSError, ValueError) as err:
        p.append(f"checkpoint manifest: {err}")
    # the single final commit (Codex code r3 HIGH 7)
    try:
        cm, _ = read_sealed(os.path.join(src, COMMIT_MARKER), COMMIT_KEYS)
        lr = os.path.join(src, str(cm["launch_record"]))
        lrec = json.load(open(lr))
        if (cm["schema"], cm["run"], cm["training_receipt_sha256"]) != (1, run, digest) \
                or sha256_file(lr) != cm["launch_record_sha256"] or lrec.get("status") != "COMPLETE" \
                or lrec.get("training_receipt_sha256") != digest:
            p.append("the commit marker / terminal launch record do not bind this receipt")
    except (GateError, OSError, ValueError, KeyError) as err:
        p.append(f"no committed completion: {err}")
    if rec["data_content_checks"] != ["pre-launch", "post-run"]:
        p.append("the AR content was not verified before and after the run")
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
    s.add_argument("--platform", required=True, choices=sorted(PLATFORMS)); s.add_argument("--closure", required=True)
    s.add_argument("--e0-sha", required=True)
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
            print(json.dumps(check_training_receipt(a.run, a.run_dir, closure=a.closure, e0_sha=a.e0_sha,
                                                    platform=a.platform)[0], indent=1))
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
