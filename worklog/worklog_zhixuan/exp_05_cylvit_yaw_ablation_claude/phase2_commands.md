# Phase 2 Commands — exp_05_cylvit_yaw_ablation

**Date:** 2026-07-08  
**Status:** scripts prepared; training not launched by Codex.

## What Phase 2 Runs

Phase 2 trains two matched FLAC variants:

- `SimpleViT-FLAC`
- `CylViT-FLAC`

Both variants:

- use the same AcousticRooms train config,
- start from `weights/FLAC/FLAC_EMA.ckpt` where weights are shape-compatible,
- skip incompatible released DINOv3 geometry-conditioner weights,
- randomly initialize the new SimpleViT/CylViT geometry branches,
- freeze the VAE/pretransform,
- train conditioner + DiT,
- use constant LR `5e-6` with `200` optimizer-step warmup,
- use matched seed, batch, accumulation, and max-step settings.

## Dry Run Already Completed

Commands run by Codex:

```bash
../venv/bin/python worklog/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json \
  --save-dir outputs_FLAC/exp05_dryrun_simple \
  --name FLAC_exp05_dryrun_simple \
  --dry-run \
  --accelerator cpu

../venv/bin/python worklog/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json \
  --save-dir outputs_FLAC/exp05_dryrun_cyl \
  --name FLAC_exp05_dryrun_cyl \
  --dry-run \
  --accelerator cpu
```

Observed:

- SimpleViT dry run loaded 640 compatible tensors.
- CylViT dry run loaded 640 compatible tensors.
- In both cases the incompatible DINOv3 geometry `lin_proj` tensors were skipped.
- The selected `source_vit` classes were correct:
  - `SimpleViT`
  - `CylindricalViT`

## Step 1 — Smoke Run

Run this first. It trains each model for only 10 steps and does not export checkpoints.

```bash
cd /home/zhixuanzhao/projects/rir2rir/FLAC
GPU=0 bash worklog/exp_05_cylvit_yaw_ablation_claude/run_phase2_smoke.sh
```

If GPU 0 is busy, change `GPU=1` or another visible device:

```bash
GPU=1 bash worklog/exp_05_cylvit_yaw_ablation_claude/run_phase2_smoke.sh
```

Pass condition:

- both SimpleViT and CylViT reach 10 steps,
- losses are finite,
- no dataloader or checkpoint loading error occurs.

## Step 2 — Short-Run Trend Training

Default short run:

```bash
cd /home/zhixuanzhao/projects/rir2rir/FLAC
GPU=0 bash worklog/exp_05_cylvit_yaw_ablation_claude/run_phase2_short_train.sh
```

Default settings:

```text
MAX_STEPS=625
SEED=42
BATCH_SIZE=4
ACCUM=32
effective batch = 128
```

To adjust:

```bash
GPU=0 MAX_STEPS=200 SEED=42 BATCH_SIZE=4 ACCUM=16 \
  bash worklog/exp_05_cylvit_yaw_ablation_claude/run_phase2_short_train.sh
```

Expected outputs:

```text
outputs_FLAC/exp05_simplevit_short_s42/FLAC_exp05_simplevit_short_s42.ckpt
outputs_FLAC/exp05_simplevit_short_s42/FLAC_exp05_simplevit_short_s42_load_report.json

outputs_FLAC/exp05_cylvit_short_s42/FLAC_exp05_cylvit_short_s42.ckpt
outputs_FLAC/exp05_cylvit_short_s42/FLAC_exp05_cylvit_short_s42_load_report.json
```

## Step 3 — Minimal Post-Training Eval Smoke

After Step 2 finishes, run a small exp01-style eval on K=1 and a few yaw angles:

```bash
cd /home/zhixuanzhao/projects/rir2rir/FLAC
GPU=0 K=1 SEED=42 ANGLES=0,22.5,90 \
  bash worklog/exp_05_cylvit_yaw_ablation_claude/run_phase2_posttrain_eval_smoke.sh
```

This is not the full Phase 3 sweep. It only checks that:

- `eval_FLAC.py` can load both exported checkpoints,
- exp01 metrics run,
- yaw-rotated conditioning path is functional.

## Full Phase 3 Later

If the short-run trend looks promising, Phase 3 should run the full exp01 metric suite:

```text
K = 1, 8
seeds = 42, 43, 44, 45, 46
angles = 0, 22.5, 45, ..., 337.5 plus 5, 10, 15
```

That full sweep is intentionally not launched in Phase 2.

