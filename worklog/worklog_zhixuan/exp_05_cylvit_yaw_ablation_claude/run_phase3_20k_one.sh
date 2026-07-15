#!/usr/bin/env bash
# Start a fresh optimizer phase from an exported 5k SimpleViT/CylViT model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
MODEL="${MODEL:-cylvit}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
PHASE_STEPS="${PHASE_STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-16}"
GEOMETRY_LR="${GEOMETRY_LR:-2e-5}"
DIT_LR="${DIT_LR:-2e-6}"
OTHER_COND_LR="${OTHER_COND_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
VAL_EVERY="${VAL_EVERY:-2000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16-mixed}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-phase3-${MODEL}}"

case "$MODEL" in
  simplevit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json"
    INIT_CKPT="outputs_FLAC/exp05_simplevit_resume2500to5000_s42/FLAC_exp05_simplevit_resume2500to5000_s42.ckpt"
    ;;
  cylvit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json"
    INIT_CKPT="outputs_FLAC/exp05_cylvit_5000s_s42/FLAC_exp05_cylvit_5000s_s42.ckpt"
    ;;
  *)
    echo "MODEL must be 'simplevit' or 'cylvit', got '$MODEL'" >&2
    exit 2
    ;;
esac

if [ ! -f "$INIT_CKPT" ]; then
  echo "Missing 5k initialization checkpoint: $INIT_CKPT" >&2
  exit 2
fi

SAVE_DIR="outputs_FLAC/exp05_${MODEL}_phase3_total20k_s${SEED}"
NAME="FLAC_exp05_${MODEL}_phase3_total20k_s${SEED}"

echo "=== exp05 Phase 3 from 5k weights ==="
echo "MODEL=${MODEL} GPU=${GPU} SEED=${SEED}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "PHASE_STEPS=${PHASE_STEPS} EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM))"
echo "GEOMETRY_LR=${GEOMETRY_LR} DIT_LR=${DIT_LR} OTHER_COND_LR=${OTHER_COND_LR}"
echo "SAVE_DIR=${SAVE_DIR}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/train_vit_ablation.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json \
  --ckpt-path "$INIT_CKPT" \
  --save-dir "$SAVE_DIR" \
  --name "$NAME" \
  --geometry-lr "$GEOMETRY_LR" \
  --dit-lr "$DIT_LR" \
  --other-cond-lr "$OTHER_COND_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --scheduler cosine \
  --min-lr-ratio 0.1 \
  --max-steps "$PHASE_STEPS" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --val-every "$VAL_EVERY" \
  --limit-val-batches "$LIMIT_VAL_BATCHES" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$BATCH_SIZE" \
  --accumulate-grad-batches "$ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --precision "$PRECISION" \
  --gradient-clip-val 1.0 \
  --seed "$SEED" \
  --use-ema \
  --require-full-load
