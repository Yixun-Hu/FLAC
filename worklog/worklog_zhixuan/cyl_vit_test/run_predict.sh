#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

EXPDIR="worklog/worklog_zhixuan/cyl_vit_test"
PYTHON="${PYTHON:-python3}"
MODEL_CONFIG="$EXPDIR/FLAC_AR_CylViT.json"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a CylindricalViT Lightning checkpoint}"
K="${K:-1}"
SPLIT="${SPLIT:-unseen}"
SEED="${SEED:-42}"
YAW="${YAW:-0}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
STORE_PREDICTIONS="${STORE_PREDICTIONS:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

case "$SPLIT:$K" in
  unseen:1) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json" ;;
  unseen:8) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json" ;;
  seen:1) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json" ;;
  seen:8) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json" ;;
  seen:*|unseen:*) echo "K must be 1 or 8, got '$K'" >&2; exit 2 ;;
  *) echo "SPLIT must be seen or unseen, got '$SPLIT'" >&2; exit 2 ;;
esac

VERIFY_ARGS=("$EXPDIR/verify_config.py")
if [ "$DRY_RUN" != "1" ]; then
  if [ ! -f "$CKPT_PATH" ]; then
    echo "Checkpoint does not exist: $CKPT_PATH" >&2
    exit 2
  fi
  VERIFY_ARGS+=(--assets eval)
fi
"$PYTHON" "${VERIFY_ARGS[@]}"

EVAL_NAME="cylvit_${SPLIT}_K${K}_s${SEED}_yaw${YAW}"
ROT_SUFFIX=""
if [ "$YAW" != "0" ] && [ "$YAW" != "0.0" ]; then
  ROT_SUFFIX="_rot${YAW%%.*}"
fi
CKPT_STEM="$(dirname "$CKPT_PATH")/$(basename "${CKPT_PATH%.ckpt}")"
METRICS_PATH="${CKPT_STEM}_metrics_1_1.0_${EVAL_NAME}${ROT_SUFFIX}.json"
PREDICTIONS_PATH="${CKPT_STEM}_predictions_1_1.0_${EVAL_NAME}${ROT_SUFFIX}.pt"

if [ "$DRY_RUN" != "1" ] && [ "$FORCE" != "1" ] && [ -f "$METRICS_PATH" ]; then
  if [ "$STORE_PREDICTIONS" = "0" ] || [ -f "$PREDICTIONS_PATH" ]; then
    echo "Prediction output already exists; skip: $METRICS_PATH"
    exit 0
  fi
fi

CMD=("$PYTHON" eval_FLAC.py
  --model-config "$MODEL_CONFIG"
  --dataset-config "$DATASET_CONFIG"
  --ckpt-path "$CKPT_PATH"
  --cfg-scale 1.0
  --steps 1
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --rotate-deg "$YAW"
  --cond-method vanilla
  --eval-name "$EVAL_NAME")

if [ "$STORE_PREDICTIONS" = "1" ]; then
  CMD+=(--store_predictions)
elif [ "$STORE_PREDICTIONS" != "0" ]; then
  echo "STORE_PREDICTIONS must be 0 or 1" >&2
  exit 2
fi

if [ "$FORCE" != "0" ] && [ "$FORCE" != "1" ]; then
  echo "FORCE must be 0 or 1" >&2
  exit 2
fi

if [ "$DRY_RUN" = "1" ]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
