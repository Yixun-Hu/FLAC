#!/usr/bin/env bash
# C16 yaw-sweep evaluation: all 16 patch-grid angles (n * 22.5 deg) for one model.
# Defaults to the matched total-25k milestone checkpoint (epoch=4-step=20000).
# Safe to re-run: skips angles whose metrics JSON already exists. Locates the
# repo root via git, so the script works from any location inside the checkout.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
MODEL="${MODEL:?Set MODEL to simplevit or cylvit}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
TOTAL_LABEL="${TOTAL_LABEL:-25k}"
SAVE_DIR="outputs_FLAC/exp05_${MODEL}_phase3_total30k_s${SEED}"
CKPT="${CKPT:-$SAVE_DIR/epoch=4-step=20000.ckpt}"

case "$MODEL" in
  simplevit) MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json" ;;
  cylvit)    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json" ;;
  *) echo "MODEL must be 'simplevit' or 'cylvit', got '$MODEL'" >&2; exit 2 ;;
esac

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

ANGLES=(0 22.5 45 67.5 90 112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5)

# STORE_PRED_C4=1: additionally store prediction tensors at the C4 angles
# (0/90/180/270), consumed by gen_rotation_waveform_visuals.py.
STORE_PRED_C4="${STORE_PRED_C4:-0}"

for YAW in "${ANGLES[@]}"; do
  EVAL_NAME="exp05_${MODEL}_c16_total${TOTAL_LABEL}_yaw${YAW}"
  SUFFIX=""
  if [ "$YAW" != "0" ]; then SUFFIX="_rot${YAW%%.*}"; fi
  STEM="$(dirname "$CKPT")/$(basename "${CKPT%.ckpt}")"
  METRICS="${STEM}_metrics_1_1.0_${EVAL_NAME}${SUFFIX}.json"
  EXTRA_ARGS=()
  case "$YAW" in
    0|90|180|270)
      if [ "$STORE_PRED_C4" = "1" ]; then
        EXTRA_ARGS+=(--store_predictions)
        # redo the angle if metrics exist but the predictions file does not
        if [ -f "$METRICS" ] && [ ! -f "${STEM}_predictions_1_1.0_${EVAL_NAME}${SUFFIX}.pt" ]; then
          METRICS="/nonexistent"
        fi
      fi
      ;;
  esac
  if [ -f "$METRICS" ]; then
    echo "[c16] exists, skip: model=${MODEL} yaw=${YAW}"
    continue
  fi
  echo "[c16] model=${MODEL} yaw=${YAW} ckpt=${CKPT} store_pred=${STORE_PRED_C4}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval_FLAC.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
    --ckpt-path "$CKPT" \
    --cfg-scale 1.0 \
    --steps 1 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --rotate-deg "$YAW" \
    --eval-name "$EVAL_NAME" \
    "${EXTRA_ARGS[@]}"
done

echo "[c16] done: model=${MODEL}"
