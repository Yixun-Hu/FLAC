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

## Measured rate — and why it is a RANGE, not a number

The R3 number includes process startup, ViT load, and the 243-subfolder scandir,
so it over-estimates by design — an upper bound is the safe direction for an
abort threshold. Steady state is taken from the tqdm elapsed stamps over
steps 10 → 25.

| Smoke run | Wall (25 steps) | R3 upper bound | Steady state | 40k projection |
|---|---|---|---|---|
| 13:51:57 (pre-r2 launcher) | 86 s | 38.2 h | 2.200 s/step | 24.4 h |
| 14:14 (final launcher) | 104 s | 46.2 h | 2.933 s/step | 32.6 h |

**The 33 % spread is GPU contention, not noise in our arm.** At the time of the
second run two `eval_FLAC.py` processes owned by a *different* checkout
(`/workspace`, evaluating `FLAC_AR_BVp1.json`) were holding ~1.9 GB each and
drawing **22–23 % utilisation on both cards**. They are not ours and were not
touched, per the standing rule against interfering with unowned runs.

So the honest FULL estimate is **24 h if the cards are ours alone, ~33 h under
the contention observed today**, and the R3 bound of 46.2 h holds either way.
Loss at step 25 was identical across all three smokes (2.470 / std_data 1.200 /
lr 1.11e-5), so the variance is throughput, not trajectory.

For comparison, P1 vanilla measured 3.86 s/step → 42.9 h at 40k. The speed-up is
the ViT gradient-checkpointing change (registered deltas 2/3), not the
augmentation: checkpointing trades VRAM for recompute in the backward pass.
Yixun freed both A6000s for this arm, which is what made the trade available.

**Gate behaviour worth recording.** The new endpoint gate fired on the *second*
smoke even though it had completed 25/25 steps: Lightning prints
`` `Trainer.fit` stopped: `max_steps=25` reached. `` with backticks around the
assignment, and the pattern omitted them. It failed closed — a re-run, not a
false pass — but a gate that can never match is worthless, so guardtests
I4a/I4b now pin both directions against the literal text.

## Topology confirmed in the log

- 2 processes, `distributed_backend=nccl`, `LOCAL_RANK 0/1`, `CUDA_VISIBLE_DEVICES=[0,1]`
- 64.5 M diffusion + 16.7 M EMA; 50.3 M trainable / 30.9 M frozen
- ViT: `facebook/dinov3-vits16-pretrain-lvd1689m`, 21.60 M / 21.60 M trainable
- Dataset: 291,210 files in 243 subfolders (the full AR training set)
- 4,550 steps/epoch at the pinned rung → 40,000 steps ≈ 8.8 epochs

*Recorded by the main session seat (Claude Opus 5, max effort).*
