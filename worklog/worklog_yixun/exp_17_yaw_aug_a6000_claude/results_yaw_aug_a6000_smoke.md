# exp_17 — SMOKE results (rate measurement + treatment liveness)

**Run:** `MODE=SMOKE bash worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh`
**Log:** `yaw_aug_a6000_2026-08-15_13-51-57_exp17_YAWAUG_smoke_train.log`
**Repo state:** branch `exp17-yawaug-scratch`, HEAD `b2af05b`, tree clean under `src/ train.py baselines/`
**W&B:** `FLAC_exp17_YAWAUG_smoke/runs/33vj0pzu` (identity gate passed: yh4742@princeton.edu)
**Date:** 2026-08-15 13:51:57 → 13:53:51

## Verdict: PASS — FULL is cleared to launch

| Gate | Result |
|---|---|
| Treatment banner (exact whole-line match) | **FOUND** — `yaw_aug ENABLED img_w=512 seed=42` |
| Checkpoints written by SMOKE | **0** (cadence 1,000,000 ≫ 25-step endpoint) |
| nan/inf in log | none |
| `Trainer.fit` termination | `max_steps=25 reached`, rc=0 |
| R3 projection (startup-inclusive upper bound) | 38.2 h ≤ 55 h → **PASS** |

The banner match is the load-bearing one: it is asserted as a whole line against
the literal print at `src/training/diffusion.py:407`, and the launcher's own
preflight output is deliberately worded so it cannot satisfy that match
(guardtest H1 asserts exactly this). A silently-disabled treatment is the one
failure mode this experiment could not survive, and it would otherwise look
like a perfectly successful run.

## Measured rate

The R3 number (38.2 h) includes process startup, ViT download/load, and the
243-subfolder scandir, so it over-estimates by design — an upper bound is the
safe direction for an abort threshold. The steady-state rate, taken from the
tqdm elapsed stamps over steps 10 → 25:

| Quantity | Value |
|---|---|
| Steady state | **2.200 s / optimizer step** |
| 40,000 steps | **24.4 h** |
| First checkpoint (2,500 steps) | 1.53 h |
| Loss at step 25 | 2.47 (train/mse_loss), std_data 1.20 — normal cold-start range |

For comparison, P1 vanilla measured 0.259 opt-steps/s = 3.86 s/step → 42.9 h at
40k. The 1.75× speed-up is the ViT gradient-checkpointing change (registered
deltas 2/3), not the augmentation: turning checkpointing off trades VRAM for
skipped recompute. Its numerical status, stated at the strength the evidence
actually supports (corrected after Codex r2): the regression test pins ON-vs-OFF
parameter gradients as `torch.allclose(atol=1e-6, rtol=1e-5)` over >=100 tensors
on an **fp32 CPU** probe, not `torch.equal`; the "210 tensors, max abs diff 0.0"
figure is an exp_07 worklog observation, not a pinned invariant, and this arm
trains bf16-mixed on CUDA. Exactness is expected by construction, not asserted
in CI. Yixun
freed both A6000s for this arm, which is what made the trade available.

## Topology confirmed in the log

- 2 processes, `distributed_backend=nccl`, `LOCAL_RANK 0/1`, `CUDA_VISIBLE_DEVICES=[0,1]`
- 64.5 M diffusion + 16.7 M EMA; 50.3 M trainable / 30.9 M frozen
- ViT: `facebook/dinov3-vits16-pretrain-lvd1689m`, 21.60 M / 21.60 M trainable
- Dataset: 291,210 files in 243 subfolders (the full AR training set)
- 4,550 steps/epoch at the pinned rung → 40,000 steps ≈ 8.8 epochs

*Recorded by the main session seat (Claude Opus 5, max effort).*
