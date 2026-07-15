# Phase 3 Recipe: 5k Weight Warm Start to Total 30k

Date: 2026-07-11

## Goal

Continue matched SimpleViT/CylViT models from their exported 5k weights for a
new 25k-step optimizer phase. Optimizer, scheduler, and EMA state start fresh;
the effective total-step labels are used only to compare training maturity.

## Matched Recipe

| Setting | Value |
|---|---:|
| New optimizer steps | 25,000 |
| Effective total label | 30,000 |
| Micro batch / accumulation | 4 / 16 |
| Effective batch | 64 |
| Geometry encoder LR | 2e-5 |
| DiT LR | 2e-6 |
| Other conditioner LR | 1e-6 |
| Warmup | 500 new optimizer steps |
| Scheduler | cosine, 0.1 minimum LR ratio |
| EMA | enabled for DiT, beta 0.9999 |
| Gradient clipping | 1.0 |
| Periodic checkpoint | every 5,000 new steps |
| Validation | 32 seen-room K=1 batches every 2,000 steps |

## Convergence Record

- Full unseen-room K=1 evaluation at yaw 0 for total 5k, 10k, 15k, 20k,
  25k, and 30k.
- Final total-30k evaluation at yaw 0, 90, 180, and 270 degrees.
- Per-model Markdown reports contain absolute metrics and changes from the
  preceding milestone.
- Existing metrics JSON files are reused, so interrupted evaluation can be
  restarted without repeating completed points.

The 5k starting point is an exported weight checkpoint. The 10k through 30k
points use Phase 3 Lightning checkpoints and therefore include the Phase 3 EMA
state used by `eval_FLAC.py`.
