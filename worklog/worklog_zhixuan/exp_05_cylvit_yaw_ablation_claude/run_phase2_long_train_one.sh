#!/usr/bin/env bash
# Long training for one exp05 model.
#
# Environment variables:
#   MODEL=simplevit|cylvit
#   GPU=0
#   MAX_STEPS=5000
#   SEED=42
#   RESUME_FROM=outputs_FLAC/.../last.ckpt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-cylvit}"
MAX_STEPS="${MAX_STEPS:-5000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-32}"
LR="${LR:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
RESUME_FROM="${RESUME_FROM:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-long-train}"

case "$MODEL" in
  simplevit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json"
    SAVE_DIR="outputs_FLAC/exp05_simplevit_${MAX_STEPS}s_s${SEED}"
    NAME="FLAC_exp05_simplevit_${MAX_STEPS}s_s${SEED}"
    ;;
  cylvit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json"
    SAVE_DIR="outputs_FLAC/exp05_cylvit_${MAX_STEPS}s_s${SEED}"
    NAME="FLAC_exp05_cylvit_${MAX_STEPS}s_s${SEED}"
    ;;
  *)
    echo "MODEL must be 'simplevit' or 'cylvit', got '$MODEL'" >&2
    exit 2
    ;;
esac

CMD=(
  "$PYTHON"
  worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py
  --model-config "$MODEL_CONFIG"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt
  --save-dir "$SAVE_DIR"
  --name "$NAME"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --max-steps "$MAX_STEPS"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --batch-size "$BATCH_SIZE"
  --accumulate-grad-batches "$ACCUM"
  --num-workers "$NUM_WORKERS"
  --precision "$PRECISION"
  --seed "$SEED"
)

if [ -n "$RESUME_FROM" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "RESUME_FROM does not exist: $RESUME_FROM" >&2
    echo "Use a Lightning training checkpoint such as ${SAVE_DIR}/last.ckpt or ${SAVE_DIR}/epoch=...ckpt." >&2
    echo "The exported ${SAVE_DIR}/${NAME}.ckpt is for eval/export loading, not optimizer-state resume." >&2
    exit 2
  fi
  CMD+=(--resume-from "$RESUME_FROM")
fi

echo "=== exp05 long train ==="
echo "MODEL=${MODEL}"
echo "GPU=${GPU}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SEED=${SEED}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "NAME=${NAME}"
echo "RESUME_FROM=${RESUME_FROM:-none}"
echo "CMD=${CMD[*]}"

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
