#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

EXPDIR="worklog/worklog_zhixuan/cyl_vit_test"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-67500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-42}"
LOGGER="${LOGGER:-none}"
SAVE_DIR="${SAVE_DIR:-outputs_FLAC/cyl_vit_test_s${SEED}}"
MODEL_CONFIG="$EXPDIR/FLAC_AR_CylViT.json"
DATASET_CONFIG="src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE_CKPT="${VAE_CKPT:-weights/FLAC/VAE.safetensors}"
CKPT_PATH="${CKPT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  case "$gpu" in
    ''|*[!0-9]*) echo "GPU_IDS must be comma-separated non-negative integers, got '$GPU_IDS'" >&2; exit 2 ;;
  esac
done
for value_name in BATCH_SIZE GLOBAL_BATCH_SIZE MAX_STEPS CHECKPOINT_EVERY NUM_WORKERS; do
  value="${!value_name}"
  case "$value" in
    ''|*[!0-9]*) echo "$value_name must be a positive integer, got '$value'" >&2; exit 2 ;;
  esac
  if [ "$value" -lt 1 ] && [ "$value_name" != "NUM_WORKERS" ]; then
    echo "$value_name must be positive, got '$value'" >&2
    exit 2
  fi
done

MICRO_GLOBAL=$((BATCH_SIZE * NUM_GPUS))
if [ $((GLOBAL_BATCH_SIZE % MICRO_GLOBAL)) -ne 0 ]; then
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is not divisible by BATCH_SIZE*NUM_GPUS=$MICRO_GLOBAL" >&2
  exit 2
fi
ACCUM_BATCHES=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL))
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * ACCUM_BATCHES))
if [ "$EFFECTIVE_BATCH" -ne "$GLOBAL_BATCH_SIZE" ]; then
  echo "effective-batch calculation failed: $EFFECTIVE_BATCH != $GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  STRATEGY="${STRATEGY:-ddp_find_unused_parameters_true}"
else
  STRATEGY="${STRATEGY:-auto}"
fi

VERIFY_ARGS=("$EXPDIR/verify_config.py" --instantiate)
if [ "$DRY_RUN" != "1" ]; then
  VERIFY_ARGS+=(--assets train --vae-ckpt "$VAE_CKPT")
fi
"$PYTHON" "${VERIFY_ARGS[@]}"

RESUME_ARGS=()
if [ -n "$CKPT_PATH" ]; then
  if [ "$DRY_RUN" != "1" ] && [ ! -f "$CKPT_PATH" ]; then
    echo "CKPT_PATH does not exist: $CKPT_PATH" >&2
    exit 2
  fi
  RESUME_ARGS=(--ckpt-path "$CKPT_PATH")
fi

CMD=("$PYTHON" train.py
  --model-config "$MODEL_CONFIG"
  --dataset-config "$DATASET_CONFIG"
  --pretransform-ckpt-path "$VAE_CKPT"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --accum-batches "$ACCUM_BATCHES"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --num-gpus "$NUM_GPUS"
  --strategy "$STRATEGY"
  --precision bf16-mixed
  --logger "$LOGGER"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --name FLAC_cyl_vit_test
  --experiment-name "cyl_vit_test_s${SEED}"
  --save-dir "$SAVE_DIR"
  "${RESUME_ARGS[@]}")

echo "CylindricalViT training: GPUs=$GPU_IDS micro=$BATCH_SIZE accum=$ACCUM_BATCHES effective=$EFFECTIVE_BATCH steps=$MAX_STEPS"
if [ "$DRY_RUN" = "1" ]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU_IDS"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"
