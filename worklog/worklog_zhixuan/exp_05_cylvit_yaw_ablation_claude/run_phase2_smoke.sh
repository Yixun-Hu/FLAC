#!/usr/bin/env bash
# Phase 2 smoke: 10 training steps for each geometry encoder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
GPU="${GPU:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-smoke}"

COMMON=(
  worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt
  --lr 5e-6
  --warmup-steps 200
  --batch-size 2
  --accumulate-grad-batches 1
  --num-workers 4
  --precision bf16-mixed
  --seed 42
  --smoke
)

echo "=== exp05 smoke SimpleViT ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${COMMON[@]}" \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json \
  --save-dir outputs_FLAC/exp05_simplevit_smoke \
  --name FLAC_exp05_simplevit_smoke

echo "=== exp05 smoke CylViT ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${COMMON[@]}" \
  --model-config src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json \
  --save-dir outputs_FLAC/exp05_cylvit_smoke \
  --name FLAC_exp05_cylvit_smoke

echo "=== exp05 smoke complete ==="
