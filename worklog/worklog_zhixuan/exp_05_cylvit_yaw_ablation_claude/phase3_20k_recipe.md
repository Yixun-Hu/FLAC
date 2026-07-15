# Phase 3 Recipe: 5k Weight Warm Start to Total-20k Comparison

Date: 2026-07-11

## Goal

Continue the matched SimpleViT/CylViT experiment from their exported 5k model
weights while resetting optimizer, scheduler, and EMA state. This is a new
15k-step optimization phase, not a Lightning optimizer-state resume.

## Matched Recipe

| Setting | Value |
|---|---:|
| New optimizer steps | 15,000 |
| Effective total label | 20,000 |
| Micro batch / accumulation | 4 / 16 |
| Effective batch | 64 |
| Geometry encoder LR | 2e-5 |
| DiT LR | 2e-6 |
| Other conditioner LR | 1e-6 |
| Warmup | 500 new optimizer steps |
| Scheduler | cosine, 0.1 minimum LR ratio |
| EMA | enabled for DiT, beta 0.9999 |
| Gradient clipping | 1.0 |
| Periodic checkpoint | every 2,000 steps |
| Validation | 32 batches of seen-room K=1 every 2,000 steps |

SimpleViT and CylViT keep `token_pool=mean` and `max_value=1`; changing either
would break continuity with the 5k symmetry ablation. Table-4 parity with linear
pooling and xRIR scaling remains a separate experiment.

## Acceptance

- Both models load their exported 5k checkpoints without architecture mismatch.
- Optimizer parameter groups are disjoint and cover every trainable non-VAE parameter.
- Validation produces `val/avg_loss`, a best checkpoint, and periodic checkpoints.
- Full K=1 evaluation at 0/90/180/270 degrees is run at equivalent total 20k.
