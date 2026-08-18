# exp_19 — Yixun query (2026-08-17)

**Verbatim:** "Please use the same HAA fintuning recipe as @FLAC_pdf.md (410 steps from 40k checkpoint) of P1-vanilla and B-F FA method. Before doing this, please give me plan to do this."

**Summary:** HAA-finetune the P1-vanilla@40k and B-F FA@40k checkpoints with the paper's recipe, then (implied) evaluate on HAA and compare. Plan first, approval before implementation.

**Assumption flagged:** "410 steps" does NOT appear in FLAC_pdf.md (grep: zero hits); the released recipe (README) is **1,000 steps** (batch 16 × accum 4 = eff 64, lr 5e-6, InverseLR, `--pretrained-ckpt-path` weights-only init, ckpt/val every 10). The plan reconciles both: train 1,000 steps with checkpoints every 10, which yields the step-410 checkpoint anyway. Yixun to confirm the endpoint convention.

**Why this experiment:** transfer test — does the equivariance/augmentation story survive domain shift to HAA (real rooms, source-position panoramas, reversed convention)?
