#!/usr/bin/env bash
# Train or resume exactly one exp05 model: MODEL=simplevit or MODEL=cylvit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
GPU="${GPU:-0}"
MODEL="${MODEL:-cylvit}"
MAX_STEPS="${MAX_STEPS:-625}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-32}"
RESUME_FROM="${RESUME_FROM:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-train-one}"

case "$MODEL" in
  simplevit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json"
    SAVE_DIR="outputs_FLAC/exp05_simplevit_short_s${SEED}"
    NAME="FLAC_exp05_simplevit_short_s${SEED}"
    ;;
  cylvit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json"
    SAVE_DIR="outputs_FLAC/exp05_cylvit_short_s${SEED}"
    NAME="FLAC_exp05_cylvit_short_s${SEED}"
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

if [ -n "$RESUME_FROM" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "RESUME_FROM does not exist: $RESUME_FROM" >&2
    echo "Use a Lightning training checkpoint such as ${SAVE_DIR}/last.ckpt or ${SAVE_DIR}/epoch=...ckpt." >&2
    echo "The exported ${SAVE_DIR}/${NAME}.ckpt is for eval/export loading, not optimizer-state resume." >&2
    exit 2
  fi
  CMD+=(--resume-from "$RESUME_FROM")
fi

echo "=== exp05 train MODEL=${MODEL} seed=${SEED} steps=${MAX_STEPS} resume=${RESUME_FROM:-none} ==="
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
