#!/usr/bin/env bash
# Phase 2 short-run trend test: matched SimpleViT vs CylViT training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-625}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-32}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-train}"

COMMON=(
  worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt
  --lr 5e-6
  --warmup-steps 200
  --max-steps "$MAX_STEPS"
  --checkpoint-every 250
  --batch-size "$BATCH_SIZE"
  --accumulate-grad-batches "$ACCUM"
  --num-workers 4
  --precision bf16-mixed
  --seed "$SEED"
)

echo "=== exp05 short train SimpleViT seed=${SEED} steps=${MAX_STEPS} ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${COMMON[@]}" \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json \
  --save-dir outputs_FLAC/exp05_simplevit_short_s${SEED} \
  --name FLAC_exp05_simplevit_short_s${SEED}

echo "=== exp05 short train CylViT seed=${SEED} steps=${MAX_STEPS} ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${COMMON[@]}" \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json \
  --save-dir outputs_FLAC/exp05_cylvit_short_s${SEED} \
  --name FLAC_exp05_cylvit_short_s${SEED}

echo "=== exp05 short train complete ==="
