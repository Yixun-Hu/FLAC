#!/usr/bin/env python3
"""exp_07 pre-launch audit: instantiate both arm configs and assert their wiring.

Builds the full model + training wrapper from FLAC_AR_BV.json and FLAC_AR_BF.json
(CPU, no pretransform weights needed for wiring checks) and asserts:
  - B-V wrapper: cond_method == 'vanilla' (default), EMA on, InverseLR schedule,
    lr 5e-5 / betas (0.9, 0.999) / wd 1e-3, log_snr sampler at (-1.2, 2.0).
  - B-F wrapper: cond_method == 'fa_invariant', frame_avg_angles == C4.
  - v2 (review fix): FACTORY wiring, not JSON re-reads — wrapper.cfg_dropout_prob,
    wrapper.optimizer_configs, and a real configure_optimizers() call whose
    returned AdamW/InverseLR objects are field-checked.
  - v2 (review fix): seeded init-identity — both arms are built under the same
    RNG seed and their full state_dicts must hash identically (proves matched
    initialization, not just matched architecture).
  - Both arms build the same architecture: identical parameter count and
    identical parameter-name set (the FA path changes forward-time conditioning
    only, never the module tree).

Run from repo root:  python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py
"""
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

def _repo_root(p):  # marker-walk: survives worklog relocations (worklog_yixun move, 2026-07-12)
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # guard against the stale pip-installed src copy

from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config

HERE = os.path.dirname(os.path.abspath(__file__))  # self-locating (layout-proof)
SEED = 42

# Fail-closed DINOv3 initializer pin (audit §3/§5.iii): the ViT is trainable, so
# its init weights are lineage-relevant. This gate refuses to proceed unless the
# HF cache holds EXACTLY the pinned snapshot — combined with HF_HUB_OFFLINE=1
# at launch, the run can never silently pick up a different revision.
# Fail-closed by construction: explicit raises (survive `python -O`, unlike
# assert) and the cache dir resolved via huggingface_hub itself (honors
# HF_HOME/HF_HUB_CACHE exactly like the transformers loader does).
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
# exp_25 rB-F1 A2 (Codex rA H5): EVERY file AutoModel.from_pretrained consumes from the snapshot is
# pinned -- config.json too (a compatible edit, e.g. dropout or layer_norm_eps, would keep the weights
# loadable and change the run); any other file in the snapshot refuses.
VIT_FILES = {"config.json": "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b",
             "model.safetensors": VIT_SHA256}


def assert_vit_pin(hub_cache=None):
    from huggingface_hub.constants import HF_HUB_CACHE  # env-resolved cache root
    cache = hub_cache or HF_HUB_CACHE
    hub = os.path.join(cache, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
    snap_dir = os.path.join(hub, "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir} — refuse to launch")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(
            f"DINOv3 cache snapshots {snaps} != pinned [{VIT_REV!r}] — refuse to launch")
    snap = os.path.join(snap_dir, VIT_REV)
    files = sorted(os.path.relpath(os.path.join(d, f), snap) for d, _, fs in os.walk(snap) for f in fs)
    if files != sorted(VIT_FILES):
        raise RuntimeError(f"DINOv3 snapshot files {files} != the pinned {sorted(VIT_FILES)} — refuse to launch")
    for name, want in sorted(VIT_FILES.items()):
        path = os.path.join(snap, name)
        if not os.path.isfile(path):
            raise RuntimeError(f"DINOv3 {name} is not a readable regular file (dangling link?) — refuse to launch")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise RuntimeError(f"DINOv3 {name} sha256 {h.hexdigest()} != pinned {want}")
    print(f"ViT pin OK: cache {cache}, single snapshot {VIT_REV[:12]}…, every file pinned "
          f"({', '.join(f'{n} {s[:12]}…' for n, s in sorted(VIT_FILES.items()))}; launch with HF_HUB_OFFLINE=1)")


def build(name):
    # identical RNG state before each build -> init weights must match across arms
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg = json.load(open(os.path.join(HERE, name)))
    model = create_model_from_config(cfg)
    wrapper = create_training_wrapper_from_config(cfg, model)
    return cfg, model, wrapper


def state_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        v = model.state_dict()[k]
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


assert_vit_pin()

print("building B-V (vanilla) ...")
cfg_v, model_v, wrap_v = build("FLAC_AR_BV.json")
print("building B-F (fa_invariant) ...")
cfg_f, model_f, wrap_f = build("FLAC_AR_BF.json")

# --- conditioning-method wiring ---
assert wrap_v.cond_method == "vanilla", wrap_v.cond_method
assert wrap_f.cond_method == "fa_invariant", wrap_f.cond_method
assert wrap_f.frame_avg_angles == (0.0, 90.0, 180.0, 270.0), wrap_f.frame_avg_angles
print(f"cond_method: BV={wrap_v.cond_method!r}  BF={wrap_f.cond_method!r}  "
      f"BF angles={wrap_f.frame_avg_angles}")

# --- recipe wiring shared by both arms: assert the WRAPPER (factory output), then
# --- exercise configure_optimizers() and field-check the real objects ---
for tag, w in (("BV", wrap_v), ("BF", wrap_f)):
    assert w.diffusion_ema is not None, f"{tag}: EMA missing"
    assert w.cfg_dropout_prob == 0.1, w.cfg_dropout_prob
    assert w.timestep_sampler == "log_snr"
    assert (w.mean_logsnr, w.std_logsnr) == (-1.2, 2.0)
    assert w.optimizer_configs["diffusion"]["optimizer"]["config"] == {
        "lr": 5e-5, "betas": [0.9, 0.999], "weight_decay": 1e-3}

    opts, scheds = w.configure_optimizers()
    opt = opts[0]
    assert type(opt).__name__ == "AdamW", type(opt)
    pg = opt.param_groups[0]
    # Constructing InverseLR immediately applies its step-0 multiplier to the
    # live lr: (1 - 0.99^(0+1)) * 5e-5 * (1 + 0/1e6)^-0.5 = 5e-7. The base lr
    # survives in initial_lr (scheduler-recorded) — same layout as the released
    # ckpt (initial_lr 5e-5, stepped lr 4.8393e-5 at 67.5k).
    step0_lr = (1 - 0.99 ** 1) * 5e-5
    assert abs(pg["lr"] - step0_lr) < 1e-18, (pg["lr"], step0_lr)
    assert pg["initial_lr"] == 5e-5, pg.get("initial_lr")
    assert tuple(pg["betas"]) == (0.9, 0.999) and pg["weight_decay"] == 1e-3, \
        {k: pg[k] for k in ("betas", "weight_decay")}
    sc = scheds[0]
    assert sc["interval"] == "step", sc
    sched = sc["scheduler"]
    assert type(sched).__name__ == "InverseLR", type(sched)
    assert (sched.inv_gamma, sched.power, sched.warmup) == (1000000, 0.5, 0.99), \
        (sched.inv_gamma, sched.power, sched.warmup)
    print(f"{tag}: wrapper wiring OK — EMA on, cfg_dropout 0.1, log_snr(-1.2,2.0); "
          f"configure_optimizers() -> AdamW(initial_lr 5e-5, step-0 lr {pg['lr']:.3e} "
          f"= warmup-live, (0.9,0.999), wd1e-3) + InverseLR(1e6,0.5,0.99)@step  OK")

# --- identical architecture across arms ---
names_v = [n for n, _ in model_v.named_parameters()]
names_f = [n for n, _ in model_f.named_parameters()]
count_v = sum(p.numel() for p in model_v.parameters())
count_f = sum(p.numel() for p in model_f.parameters())
assert names_v == names_f, "parameter-name sets differ between arms"
assert count_v == count_f, (count_v, count_f)
print(f"architecture: identical param names ({len(names_v)} tensors) "
      f"and count ({count_v/1e6:.2f}M) in both arms")

# --- seeded init-identity: same seed -> byte-identical initial weights ---
hv, hf = state_hash(model_v), state_hash(model_f)
assert hv == hf, f"init state hashes differ:\n  BV {hv}\n  BF {hf}"
print(f"init identity: state_dict sha256 match under seed {SEED}: {hv[:16]}…")

print("\nALL ASSERTS PASSED — arms are init-identical and differ only in "
      "forward-time conditioning method.")
