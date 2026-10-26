#!/usr/bin/env python3
"""exp_25 depth_canon - read-only CPU inspector for a PL boundary checkpoint on the NAS.
Usage: CUDA_VISIBLE_DEVICES="" <flac python> depth_canon_ckpt_inspect.py <ckpt>
Prints step/epoch, tensor census by prefix, non-finite tensor count, optimizer/scheduler state. Never writes.
Planner-authored 2026-10-26 (first use: epoch=0-step=2500.ckpt); Codex review batched into the endpoint / MODE=RESUME round."""
import sys, torch, collections
p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
print("top keys:", sorted(ck.keys()))
print("epoch:", ck.get("epoch"), "global_step:", ck.get("global_step"))
sd = ck["state_dict"]
pre = collections.Counter(k.split(".")[0] for k in sd)
print("state_dict tensors:", len(sd), "by prefix:", dict(pre))
ema = [k for k in sd if k.startswith("diffusion_ema.")]
print("ema tensors:", len(ema))
bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
print("non-finite tensors in state_dict:", len(bad), bad[:5])
opt = ck.get("optimizer_states")
print("optimizer_states:", type(opt).__name__, len(opt) if opt else None, "param_groups lr:", [g.get("lr") for g in opt[0]["param_groups"]] if opt else None, "state entries:", len(opt[0]["state"]) if opt else None)
ls = ck.get("lr_schedulers")
print("lr_schedulers:", [{k: s.get(k) for k in ("last_epoch","inv_gamma","power","warmup","final_lr","base_lrs")} for s in ls] if ls else None)
loops = ck.get("loops", {})
fl = loops.get("fit_loop", {})
print("fit_loop keys:", sorted(fl.keys())[:8])
try:
    print("optimizer_step completed:", fl["epoch_loop.batch_loop.optimizer_loop.optim_progress"]["optimizer"]["step"]["total"]["completed"])
except Exception as e:
    try:
        print("optimizer progress:", fl["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"])
    except Exception as e2:
        print("optimizer progress: n/a", e, e2)
