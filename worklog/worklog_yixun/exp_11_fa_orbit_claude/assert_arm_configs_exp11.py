#!/usr/bin/env python3
"""exp_11 pre-launch audit — DINOv3 pin + per-arm wiring + init-identity.

Replicated from exp_07's ``assert_arm_configs.py`` (which does NOT generalize:
it hardcodes ``FLAC_AR_BV.json``/``FLAC_AR_BF.json``, the C4 orbit and the
vanilla-vs-fa pairing). Same checks, re-pointed at the exp_11 arm configs and
extended to the orbit sweep:

  - DINOv3 initializer pin: the ViT is trainable, so its init weights are
    lineage-relevant; the HF cache must hold EXACTLY the pinned snapshot
    (constants copied verbatim from exp_07's gate — the single upstream source).
  - FACTORY wiring for the requested arm: cond_method 'fa_invariant', the arm's
    uniform orbit, EMA on, cfg_dropout 0.1, log_snr(-1.2, 2.0), and a real
    configure_optimizers() whose AdamW/InverseLR objects are field-checked.
  - Architecture identity against the C4L bridge arm: identical parameter names
    and count (the orbit changes forward-time conditioning only).
  - Seeded init-identity: built under the same seed, this arm's state_dict must
    hash identically to C4L's — so every arm starts from the same weights and
    the sweep's only delta is the averaging orbit.

Explicit raises (not ``assert``) everywhere the outcome gates a launch, so the
gate survives an inherited ``PYTHONOPTIMIZE``.

Run from the repo root:
    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_11_fa_orbit_claude/assert_arm_configs_exp11.py C8
"""
import hashlib
import json
import os
import random
import sys


def _repo_root(p):  # marker-walk (same helper as exp_07's gate)
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # guard against a stale pip-installed src copy

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.factory import create_model_from_config  # noqa: E402
from src.training.factory import create_training_wrapper_from_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
REFERENCE_ARM = "C4L"                       # the bridge arm every arm must match
ARM_ORBIT = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}

# Verbatim from worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py
VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"


def assert_vit_pin():
    from huggingface_hub.constants import HF_HUB_CACHE
    hub = os.path.join(HF_HUB_CACHE, "models--facebook--dinov3-vits16-pretrain-lvd1689m")
    snap_dir = os.path.join(hub, "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir} — refuse to launch")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(
            f"DINOv3 cache snapshots {snaps} != pinned [{VIT_REV!r}] — refuse to launch")
    st = os.path.join(snap_dir, VIT_REV, "model.safetensors")
    h = hashlib.sha256()
    with open(st, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != VIT_SHA256:
        raise RuntimeError(
            f"DINOv3 model.safetensors sha256 {h.hexdigest()} != pinned {VIT_SHA256}")
    print(f"ViT pin OK: cache {HF_HUB_CACHE}, single snapshot {VIT_REV[:12]}…, "
          f"sha256 {VIT_SHA256[:12]}… (launch with HF_HUB_OFFLINE=1)")


def build(arm):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg = json.load(open(os.path.join(HERE, f"FLAC_AR_BF_{arm}.json")))
    model = create_model_from_config(cfg)
    wrapper = create_training_wrapper_from_config(cfg, model)
    return cfg, model, wrapper


def state_hash(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_recipe(tag, w, n_angles):
    if w.cond_method != "fa_invariant":
        raise RuntimeError(f"{tag}: cond_method {w.cond_method!r} != 'fa_invariant'")
    want = tuple(k * 360.0 / n_angles for k in range(n_angles))
    got = tuple(w.frame_avg_angles)
    if got != want:
        raise RuntimeError(f"{tag}: frame_avg_angles {got} != the uniform C{n_angles} orbit {want}")
    if w.diffusion_ema is None:
        raise RuntimeError(f"{tag}: EMA missing")
    if w.cfg_dropout_prob != 0.1:
        raise RuntimeError(f"{tag}: cfg_dropout_prob {w.cfg_dropout_prob} != 0.1")
    if w.timestep_sampler != "log_snr":
        raise RuntimeError(f"{tag}: timestep_sampler {w.timestep_sampler!r} != 'log_snr'")
    if (w.mean_logsnr, w.std_logsnr) != (-1.2, 2.0):
        raise RuntimeError(f"{tag}: log_snr ({w.mean_logsnr}, {w.std_logsnr}) != (-1.2, 2.0)")
    if w.optimizer_configs["diffusion"]["optimizer"]["config"] != {
            "lr": 5e-5, "betas": [0.9, 0.999], "weight_decay": 1e-3}:
        raise RuntimeError(f"{tag}: optimizer config drifted")

    opts, scheds = w.configure_optimizers()
    opt, pg = opts[0], opts[0].param_groups[0]
    if type(opt).__name__ != "AdamW":
        raise RuntimeError(f"{tag}: optimizer {type(opt).__name__} != AdamW")
    step0_lr = (1 - 0.99 ** 1) * 5e-5          # InverseLR applies its step-0 multiplier at build
    if abs(pg["lr"] - step0_lr) > 1e-18 or pg["initial_lr"] != 5e-5:
        raise RuntimeError(f"{tag}: lr {pg['lr']} / initial_lr {pg.get('initial_lr')} unexpected")
    if tuple(pg["betas"]) != (0.9, 0.999) or pg["weight_decay"] != 1e-3:
        raise RuntimeError(f"{tag}: betas/wd drifted: {pg['betas']}, {pg['weight_decay']}")
    sc = scheds[0]
    sched = sc["scheduler"]
    if sc["interval"] != "step" or type(sched).__name__ != "InverseLR":
        raise RuntimeError(f"{tag}: scheduler {type(sched).__name__} @ {sc['interval']}")
    if (sched.inv_gamma, sched.power, sched.warmup) != (1000000, 0.5, 0.99):
        raise RuntimeError(f"{tag}: InverseLR fields drifted")
    print(f"{tag}: wiring OK — fa_invariant C{n_angles} orbit, EMA on, cfg_dropout 0.1, "
          f"log_snr(-1.2,2.0), AdamW(initial_lr 5e-5, step-0 {pg['lr']:.3e}) + InverseLR@step")


def main(argv):
    if len(argv) != 2 or argv[1] not in ARM_ORBIT:
        raise SystemExit(f"usage: {os.path.basename(argv[0])} <{'|'.join(ARM_ORBIT)}>")
    arm = argv[1]

    assert_vit_pin()

    print(f"building {arm} (fa_invariant C{ARM_ORBIT[arm]}) ...")
    _, model_a, wrap_a = build(arm)
    check_recipe(arm, wrap_a, ARM_ORBIT[arm])

    if arm == REFERENCE_ARM:
        print(f"init identity: {arm} IS the reference arm; state_dict sha256 "
              f"{state_hash(model_a)[:16]}… under seed {SEED}")
        print(f"\nALL ASSERTS PASSED — {arm} is launch-ready.")
        return 0

    print(f"building the {REFERENCE_ARM} reference for init-identity ...")
    _, model_r, wrap_r = build(REFERENCE_ARM)
    check_recipe(REFERENCE_ARM, wrap_r, ARM_ORBIT[REFERENCE_ARM])

    names_a = [n for n, _ in model_a.named_parameters()]
    names_r = [n for n, _ in model_r.named_parameters()]
    count_a = sum(p.numel() for p in model_a.parameters())
    count_r = sum(p.numel() for p in model_r.parameters())
    if names_a != names_r:
        raise RuntimeError("parameter-name sets differ between the arms")
    if count_a != count_r:
        raise RuntimeError(f"parameter counts differ: {arm} {count_a} vs {REFERENCE_ARM} {count_r}")
    print(f"architecture: identical param names ({len(names_a)} tensors) and count "
          f"({count_a / 1e6:.2f}M) in {arm} and {REFERENCE_ARM}")

    ha, hr = state_hash(model_a), state_hash(model_r)
    if ha != hr:
        raise RuntimeError(f"init state hashes differ:\n  {arm} {ha}\n  {REFERENCE_ARM} {hr}")
    print(f"init identity: state_dict sha256 match under seed {SEED}: {ha[:16]}…")
    print(f"\nALL ASSERTS PASSED — {arm} is init-identical to {REFERENCE_ARM} and differs "
          "only in the averaging orbit.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
