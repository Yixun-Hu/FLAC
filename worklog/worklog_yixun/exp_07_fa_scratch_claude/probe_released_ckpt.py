#!/usr/bin/env python3
"""exp_07 pre-launch audit: read the released FLAC.ckpt's own training records.

Extracts everything the wrapper checkpoint recorded about how the released model
was actually trained — epoch/step counters (ALL Lightning phases: ready/started/
processed/completed, since the checkpoint callback fires before `completed`
increments), the accumulation factor from like-for-like `processed` counters,
optimizer param groups (lr/betas/weight decay), scheduler class+state (last_lr
at 67.5k), EMA buffers, embedded hyper_parameters, and — v2 (review fix) — the
full diff of the ckpt-embedded model_config against the repo FLAC_AR.json plus
the pinned DINOv3 revision/hash used by OUR runs. CPU-only, read-only.

Run from repo root:  python worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py
"""
import glob
import hashlib
import json
import os

import torch

CKPT = "weights/FLAC/FLAC.ckpt"
REPO_CFG = "src/configs/model_configs/FLAC/AR/FLAC_AR.json"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)

print("=" * 80)
print(f"released wrapper checkpoint: {CKPT}")
print("=" * 80)
print("top-level keys:", sorted(ck.keys()))
print()
print(f"epoch:                {ck.get('epoch')}")
print(f"global_step (optim):  {ck.get('global_step')}")
print(f"pytorch-lightning:    {ck.get('pytorch-lightning_version')}")

# ---- loops: ALL counter phases (PL2 nests under 'loops'/'fit_loop').
# The ModelCheckpoint callback saves after `processed` increments but before
# `completed` does, so completed lags processed by exactly 1 at save time —
# like-for-like comparisons must use the same phase on both counters.
loops = ck.get("loops", {})
fit = loops.get("fit_loop", {})


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


print("\nloops.fit_loop (all phases):")
bp = dig(fit, "epoch_loop.batch_progress") or {}
for scope in ("total", "current"):
    print(f"  batch_progress.{scope}: {bp.get(scope)}")
op = dig(fit, "epoch_loop.automatic_optimization.optim_progress", "optimizer", "step") or {}
for scope in ("total", "current"):
    print(f"  optim.step.{scope}:     {op.get(scope)}")
print(f"  epoch_progress.total:  {dig(fit, 'epoch_progress', 'total')}")

micro_processed = (bp.get("total") or {}).get("processed")
optim_completed = (op.get("total") or {}).get("completed")
micro_completed = (bp.get("total") or {}).get("completed")
if micro_processed and optim_completed:
    print(f"\n  accumulation = micro(processed)/optim(completed) = "
          f"{micro_processed}/{optim_completed} = {micro_processed / optim_completed:.6f}")
    print(f"  (micro completed={micro_completed} lags processed by "
          f"{micro_processed - micro_completed}: checkpoint fires pre-`completed`.)")
    cur_micro = (bp.get("current") or {}).get("processed")
    epochs_done = (dig(fit, "epoch_progress", "total") or {}).get("completed")
    if cur_micro is not None and epochs_done:
        per_epoch = (micro_processed - cur_micro) / epochs_done
        print(f"  steps/epoch = ({micro_processed} - {cur_micro}) / {epochs_done} = {per_epoch:.1f}")
        print("  NOTE: per-epoch step count constrains only the GLOBAL effective batch")
        print("  (given the shipped split + drop_last, src/data/dataset.py:405); the")
        print("  micro-batch x GPU decomposition (64x1 vs 32x2) is NOT counter-distinguishable —")
        print("  '64 on a single H100' comes from the paper text (FLAC_pdf.md B.1).")

# ---- optimizer: actual lr/betas/wd used ----
for i, opt in enumerate(ck.get("optimizer_states", [])):
    for g, pg in enumerate(opt.get("param_groups", [])):
        keep = {k: v for k, v in pg.items() if k != "params"}
        print(f"\noptimizer[{i}].param_groups[{g}]: {keep}")

# ---- scheduler state + analytic check with the FULL InverseLR formula ----
for i, sch in enumerate(ck.get("lr_schedulers", [])):
    print(f"\nlr_schedulers[{i}]: {sch}")
    if sch.get("inv_gamma"):
        step = sch["last_epoch"]
        base = sch["base_lrs"][0]
        # src/training/utils.py InverseLR._get_closed_form_lr:
        #   (1 - warmup**(step+1)) * base * (1 + step/inv_gamma)**-power
        warm = 1 - sch["warmup"] ** (step + 1)
        analytic = warm * base * (1 + step / sch["inv_gamma"]) ** -sch["power"]
        print(f"  analytic (full formula incl. warmup term {warm!r}): {analytic!r}")
        print(f"  recorded _last_lr:                                  {sch['_last_lr'][0]!r}")

# ---- embedded hyper_parameters ----
hp = ck.get("hyper_parameters")
print(f"\nhyper_parameters: {'ABSENT' if hp is None else hp}")

# ---- EMA evidence ----
sd = ck.get("state_dict", {})
ema_n = len([k for k in sd if k.startswith("diffusion_ema.")])
ema_meta = {k: sd[k].item() for k in ("diffusion_ema.initted", "diffusion_ema.step") if k in sd}
print(f"\nEMA keys present: {ema_n}; buffers: {ema_meta}")

# ---- callback state ----
print(f"\ncallback keys: {list(ck.get('callbacks', {}).keys())}")

# ================= v2: embedded model_config vs repo config =================
print("\n" + "=" * 80)
print(f"embedded model_config vs {REPO_CFG}")
print("=" * 80)
mc_ckpt = ck["model_config"]
mc_repo = json.load(open(REPO_CFG))


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list):
        out[p] = json.dumps(d)
    else:
        out[p] = d
    return out


a, b = flat(mc_ckpt), flat(mc_repo)
only_ckpt = sorted(set(a) - set(b))
only_repo = sorted(set(b) - set(a))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print("keys only in ckpt:", only_ckpt if only_ckpt else "none")
print("keys only in repo:", only_repo if only_repo else "none")
for k in diff:
    print(f"  DIFF {k}:\n    ckpt={a[k]!r}\n    repo={b[k]!r}")
if not diff:
    print("value diffs: none")
print("VERDICT:", "IDENTICAL" if not (only_ckpt or only_repo or diff) else
      "DIFFERS (see audit doc for the training-relevance classification)")

# ================= v2: our DINOv3 initializer pin =================
print("\n" + "=" * 80)
print("our DINOv3 initializer pin (trainable ViT -> init weights are lineage-relevant)")
print("=" * 80)
hub = os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m")
snaps = sorted(glob.glob(os.path.join(hub, "snapshots", "*")))
for s in snaps:
    print(f"revision: {os.path.basename(s)}")
    st = os.path.join(s, "model.safetensors")
    if os.path.exists(st):
        h = hashlib.sha256()
        with open(st, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print(f"model.safetensors sha256: {h.hexdigest()}")
print("(authors loaded a LOCAL snapshot './Models/dinov3-vits16-...' — revision "
      "unknowable from the ckpt; both exp_07 arms share the pin above.)")

print("\ndone.")
