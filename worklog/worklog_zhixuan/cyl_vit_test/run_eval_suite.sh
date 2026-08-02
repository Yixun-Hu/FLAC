#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

EXPDIR="worklog/worklog_zhixuan/cyl_vit_test"
PYTHON="${PYTHON:-python3}"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to the final CylindricalViT checkpoint}"
SEEDS="${SEEDS:-42 43 44 45 46}"
KS="${KS:-1 8}"
STORE_PREDICTIONS="${STORE_PREDICTIONS:-0}"
SPLITS="${SPLITS:-unseen}"
DRY_RUN="${DRY_RUN:-0}"

for SPLIT_VALUE in $SPLITS; do
  for K_VALUE in $KS; do
    for SEED_VALUE in $SEEDS; do
      SPLIT="$SPLIT_VALUE" K="$K_VALUE" SEED="$SEED_VALUE" YAW=0 \
        STORE_PREDICTIONS="$STORE_PREDICTIONS" CKPT_PATH="$CKPT_PATH" \
        PYTHON="$PYTHON" DRY_RUN="$DRY_RUN" bash "$EXPDIR/run_predict.sh"
    done
  done
done

if [ "$DRY_RUN" != "1" ]; then
  SUMMARY_PATH="${SUMMARY_PATH:-$(dirname "$CKPT_PATH")/cyl_vit_test_metrics_summary.md}"
  "$PYTHON" "$EXPDIR/summarize_metrics.py" --checkpoint "$CKPT_PATH" --output "$SUMMARY_PATH"
  echo "Metric summary: $SUMMARY_PATH"
fi
