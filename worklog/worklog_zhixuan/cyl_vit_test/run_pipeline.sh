#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

EXPDIR="worklog/worklog_zhixuan/cyl_vit_test"
PYTHON="${PYTHON:-python3}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_YAW="${RUN_YAW:-0}"
MAX_STEPS="${MAX_STEPS:-67500}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-outputs_FLAC/cyl_vit_test_s${SEED}}"
CKPT_PATH="${CKPT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$RUN_TRAIN" = "1" ]; then
  PYTHON="$PYTHON" MAX_STEPS="$MAX_STEPS" SEED="$SEED" SAVE_DIR="$SAVE_DIR" DRY_RUN="$DRY_RUN" \
    bash "$EXPDIR/run_train.sh"
elif [ "$RUN_TRAIN" != "0" ]; then
  echo "RUN_TRAIN must be 0 or 1" >&2
  exit 2
fi

if [ -z "$CKPT_PATH" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    CKPT_PATH="$SAVE_DIR/epoch=N-step=${MAX_STEPS}.ckpt"
  else
    CKPT_PATH="$("$PYTHON" "$EXPDIR/find_checkpoint.py" --root "$SAVE_DIR" --step "$MAX_STEPS")"
  fi
fi

CKPT_PATH="$CKPT_PATH" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" bash "$EXPDIR/run_eval_suite.sh"
if [ "$RUN_YAW" = "1" ]; then
  CKPT_PATH="$CKPT_PATH" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" bash "$EXPDIR/run_yaw_suite.sh"
elif [ "$RUN_YAW" != "0" ]; then
  echo "RUN_YAW must be 0 or 1" >&2
  exit 2
fi
