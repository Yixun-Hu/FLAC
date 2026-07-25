#!/usr/bin/env bash
# Train one exp06 patch-embedding variant on one physical GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-${FLAC_ROOT}/../venv/bin/python}"

VARIANT="${VARIANT:?Set VARIANT=linear or VARIANT=cnn}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp06-${VARIANT}}"
export MPLCONFIGDIR
GPU="${GPU:-0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-30000}"
MILESTONES="${MILESTONES:-5000,10000,20000,30000}"
LAST_EVERY="${LAST_EVERY:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUM="${ACCUM:-16}"
NUM_WORKERS="${NUM_WORKERS:-6}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VAL_EVERY="${VAL_EVERY:-2500}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
PRECISION="${PRECISION:-bf16-mixed}"
RESUME_FROM="${RESUME_FROM:-auto}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
DRY_RUN="${DRY_RUN:-0}"

DRY_RUN_ARGS=()
case "$DRY_RUN" in
  0) ;;
  1) DRY_RUN_ARGS+=(--dry-run) ;;
  *)
    echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  linear)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json"
    ;;
  cnn)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json"
    ;;
  *)
    echo "VARIANT must be linear or cnn, got: $VARIANT" >&2
    exit 2
    ;;
esac

INIT_DIR="${FLAC_ROOT}/outputs_FLAC/exp06_cylvit_pe_matched_initializations"
INIT_CKPT="${INIT_DIR}/cylvit_pe_${VARIANT}_trainS${SEED}_init.ckpt"
if [ "$VARIANT" = "cnn" ]; then
  SAVE_DIR="${FLAC_ROOT}/outputs_FLAC/exp06_cylvit_pe_cnn_patchlocal_trainS${SEED}"
else
  SAVE_DIR="${FLAC_ROOT}/outputs_FLAC/exp06_cylvit_pe_linear_trainS${SEED}"
fi

cd "$FLAC_ROOT"
if [ "$SKIP_PREPARE" != "1" ]; then
  "$PYTHON" "${SCRIPT_DIR}/prepare_matched_initializations.py" \
    --seed "$SEED" \
    --output-dir "$INIT_DIR"
fi

if [ ! -f "$INIT_CKPT" ]; then
  echo "Missing audited initialization: $INIT_CKPT" >&2
  exit 2
fi

echo "[exp06:launch] variant=${VARIANT} physical_gpu=${GPU} process_devices=1 (not DDP)"
echo "[exp06:launch] seed=${SEED} max_steps=${MAX_STEPS} effective_batch=$((BATCH_SIZE * ACCUM))"
echo "[exp06:launch] validation=${VAL_DATASET_CONFIG} every_optimizer_steps=${VAL_EVERY} limit=${LIMIT_VAL_BATCHES}"
echo "[exp06:launch] init=${INIT_CKPT}"
echo "[exp06:launch] save_dir=${SAVE_DIR} resume=${RESUME_FROM}"
echo "[exp06:launch] dry_run=${DRY_RUN}"

exec env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${SCRIPT_DIR}/train_patch_ablation.py" \
  --variant "$VARIANT" \
  --model-config "$MODEL_CONFIG" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --val-dataset-config "$VAL_DATASET_CONFIG" \
  --init-checkpoint "$INIT_CKPT" \
  --save-dir "$SAVE_DIR" \
  --max-steps "$MAX_STEPS" \
  --milestones "$MILESTONES" \
  --last-every "$LAST_EVERY" \
  --batch-size "$BATCH_SIZE" \
  --accumulate-grad-batches "$ACCUM" \
  --num-workers "$NUM_WORKERS" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --val-every "$VAL_EVERY" \
  --limit-val-batches "$LIMIT_VAL_BATCHES" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  --accelerator gpu \
  --devices 1 \
  --resume-from "$RESUME_FROM" \
  "${DRY_RUN_ARGS[@]}"
