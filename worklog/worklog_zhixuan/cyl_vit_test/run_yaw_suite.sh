#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

EXPDIR="worklog/worklog_zhixuan/cyl_vit_test"
PYTHON="${PYTHON:-python3}"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to the final CylindricalViT checkpoint}"
SEED="${SEED:-42}"
STORE_PREDICTIONS="${STORE_PREDICTIONS:-0}"
SPLIT="${SPLIT:-unseen}"
DRY_RUN="${DRY_RUN:-0}"
C16_ANGLES="${C16_ANGLES:-0 22.5 45 67.5 90 112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5}"
K8_ANGLES="${K8_ANGLES:-0 90}"

for YAW_VALUE in $C16_ANGLES; do
  SPLIT="$SPLIT" K=1 YAW="$YAW_VALUE" SEED="$SEED" STORE_PREDICTIONS="$STORE_PREDICTIONS" \
    CKPT_PATH="$CKPT_PATH" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" \
    bash "$EXPDIR/run_predict.sh"
done

for YAW_VALUE in $K8_ANGLES; do
  SPLIT="$SPLIT" K=8 YAW="$YAW_VALUE" SEED="$SEED" STORE_PREDICTIONS="$STORE_PREDICTIONS" \
    CKPT_PATH="$CKPT_PATH" PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" \
    bash "$EXPDIR/run_predict.sh"
done

if [ "$DRY_RUN" != "1" ]; then
  SUMMARY_PATH="${SUMMARY_PATH:-$(dirname "$CKPT_PATH")/cyl_vit_test_yaw_summary.md}"
  "$PYTHON" "$EXPDIR/summarize_metrics.py" --checkpoint "$CKPT_PATH" --output "$SUMMARY_PATH"
  echo "Yaw summary: $SUMMARY_PATH"
fi
