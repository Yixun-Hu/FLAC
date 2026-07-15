#!/usr/bin/env bash
# Run the exp06 evaluation suite as two independent single-GPU workers:
#   GPU_LINEAR -> CylViT linear patch embedding
#   GPU_CNN    -> CylViT CNN patch embedding
#
# SUITE=all (default) runs 50 unique full-split conditions:
#   * Table 1 clean: 2 variants x K{1,8} x eval seeds 42..46 x yaw0 = 20
#   * C16 gate:       2 variants x K1 x seed42 x 16 yaws = 32
#     The two K1/seed42/yaw0 conditions are reused from Table 1, not rerun.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAC_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$FLAC_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash run_eval_suite_dual_gpu.sh

Suite selection:
  SUITE=all      Table 1 clean plus C16 K=1 gate (default; 50 unique jobs)
  SUITE=table1   yaw0, K=1/8, eval seeds 42..46 (20 jobs)
  SUITE=c16      K=1, eval seed42, all 16 C16 yaws (32 jobs)

Common overrides:
  GPU_LINEAR=0 GPU_CNN=1
  TRAIN_SEED=42 TRAIN_STEP=30000
  PYTHON=../venv/bin/python
  EVAL_BATCH_SIZE=64 EVAL_NUM_WORKERS=4
  FORCE=1 DRY_RUN=1 RUN_SUMMARY=0

Per-variant checkpoint/config overrides:
  LINEAR_CKPT=... CNN_CKPT=...
  LINEAR_BARE_CKPT=... CNN_BARE_CKPT=...
  LINEAR_SAVE_DIR=... CNN_SAVE_DIR=...
  LINEAR_MODEL_CONFIG=... CNN_MODEL_CONFIG=...

This launcher is two concurrent one-GPU jobs, not DDP. Run it inside the
project's persistent job mechanism when the full suite must survive disconnects.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SUITE="${SUITE:-all}"
case "$SUITE" in
  all|table1|c16) ;;
  *) echo "SUITE must be all, table1, or c16; got '$SUITE'." >&2; exit 2 ;;
esac

GPU_LINEAR="${GPU_LINEAR:-0}"
GPU_CNN="${GPU_CNN:-1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_STEP="${TRAIN_STEP:-30000}"
PYTHON="${PYTHON:-../venv/bin/python}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"

LINEAR_SAVE_DIR="${LINEAR_SAVE_DIR:-outputs_FLAC/exp06_cylvit_pe_linear_trainS${TRAIN_SEED}}"
CNN_SAVE_DIR="${CNN_SAVE_DIR:-outputs_FLAC/exp06_cylvit_pe_cnn_trainS${TRAIN_SEED}}"
LINEAR_MODEL_CONFIG="${LINEAR_MODEL_CONFIG:-src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json}"
CNN_MODEL_CONFIG="${CNN_MODEL_CONFIG:-src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json}"
LINEAR_CKPT="${LINEAR_CKPT:-}"
CNN_CKPT="${CNN_CKPT:-}"
LINEAR_BARE_CKPT="${LINEAR_BARE_CKPT:-}"
CNN_BARE_CKPT="${CNN_BARE_CKPT:-}"

if [[ "$GPU_LINEAR" == "$GPU_CNN" && "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  echo "GPU_LINEAR and GPU_CNN are both $GPU_LINEAR; set distinct devices or ALLOW_SHARED_GPU=1." >&2
  exit 2
fi

RUN_ONE="$SCRIPT_DIR/run_eval_one.sh"
[[ -f "$RUN_ONE" ]] || { echo "Missing one-condition runner: $RUN_ONE" >&2; exit 2; }

TABLE1_SEEDS=(42 43 44 45 46)
C16_ALL=(0 22.5 45 67.5 90 112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5)
C16_AFTER_CLEAN=(22.5 45 67.5 90 112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5)

run_condition() {
  local variant="$1"
  local gpu="$2"
  local save_dir="$3"
  local model_config="$4"
  local ckpt="$5"
  local bare_ckpt="$6"
  local k="$7"
  local eval_seed="$8"
  local yaw="$9"

  GPU="$gpu" \
  TRAIN_SEED="$TRAIN_SEED" \
  TRAIN_STEP="$TRAIN_STEP" \
  PYTHON="$PYTHON" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  EVAL_NUM_WORKERS="$EVAL_NUM_WORKERS" \
  FORCE="$FORCE" \
  DRY_RUN="$DRY_RUN" \
  SAVE_DIR="$save_dir" \
  MODEL_CONFIG="$model_config" \
  CKPT="$ckpt" \
  BARE_CKPT="$bare_ckpt" \
    bash "$RUN_ONE" "$variant" "$k" "$eval_seed" "$yaw"
}

run_variant_suite() {
  local variant="$1"
  local gpu="$2"
  local save_dir="$3"
  local model_config="$4"
  local ckpt="$5"
  local bare_ckpt="$6"
  local k seed yaw

  echo "[exp06-suite] START variant=$variant gpu=$gpu suite=$SUITE"

  if [[ "$SUITE" == "all" || "$SUITE" == "table1" ]]; then
    for k in 1 8; do
      for seed in "${TABLE1_SEEDS[@]}"; do
        run_condition "$variant" "$gpu" "$save_dir" "$model_config" "$ckpt" "$bare_ckpt" "$k" "$seed" 0
      done
    done
  fi

  if [[ "$SUITE" == "c16" ]]; then
    # Standalone C16 includes yaw0 because no Table-1 pass preceded it.
    for yaw in "${C16_ALL[@]}"; do
      run_condition "$variant" "$gpu" "$save_dir" "$model_config" "$ckpt" "$bare_ckpt" 1 42 "$yaw"
    done
  elif [[ "$SUITE" == "all" ]]; then
    # K1/seed42/yaw0 already exists from Table 1; do not regenerate it under a
    # second name.  The remaining 15 rotations complete the C16 group.
    for yaw in "${C16_AFTER_CLEAN[@]}"; do
      run_condition "$variant" "$gpu" "$save_dir" "$model_config" "$ckpt" "$bare_ckpt" 1 42 "$yaw"
    done
  fi

  echo "[exp06-suite] DONE variant=$variant gpu=$gpu suite=$SUITE"
}

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LINEAR_LOG="$LOG_DIR/eval_${SUITE}_linear_step${TRAIN_STEP}_${STAMP}.log"
CNN_LOG="$LOG_DIR/eval_${SUITE}_cnn_step${TRAIN_STEP}_${STAMP}.log"

echo "[exp06-suite] launching two independent workers"
echo "[exp06-suite] linear: GPU=$GPU_LINEAR log=$LINEAR_LOG"
echo "[exp06-suite] cnn:    GPU=$GPU_CNN log=$CNN_LOG"

run_variant_suite linear "$GPU_LINEAR" "$LINEAR_SAVE_DIR" "$LINEAR_MODEL_CONFIG" \
  "$LINEAR_CKPT" "$LINEAR_BARE_CKPT" >"$LINEAR_LOG" 2>&1 &
PID_LINEAR=$!
run_variant_suite cnn "$GPU_CNN" "$CNN_SAVE_DIR" "$CNN_MODEL_CONFIG" \
  "$CNN_CKPT" "$CNN_BARE_CKPT" >"$CNN_LOG" 2>&1 &
PID_CNN=$!

cleanup_children() {
  kill "$PID_LINEAR" "$PID_CNN" 2>/dev/null || true
}
trap cleanup_children INT TERM

STATUS_LINEAR=0
STATUS_CNN=0
set +e
wait "$PID_LINEAR"
STATUS_LINEAR=$?
wait "$PID_CNN"
STATUS_CNN=$?
set -e
trap - INT TERM

echo "[exp06-suite] worker exits: linear=$STATUS_LINEAR cnn=$STATUS_CNN"

SUMMARY_STATUS=0
if [[ "$RUN_SUMMARY" == "1" && "$DRY_RUN" != "1" && "$STATUS_LINEAR" == "0" && "$STATUS_CNN" == "0" ]]; then
  if [[ -z "${LINEAR_RESULTS_DIR:-}" ]]; then
    if [[ -n "$LINEAR_CKPT" ]]; then
      LINEAR_RESULTS_DIR="$(dirname "$LINEAR_CKPT")"
    elif [[ -n "$LINEAR_BARE_CKPT" ]]; then
      LINEAR_RESULTS_DIR="$(dirname "$LINEAR_BARE_CKPT")"
    fi
  fi
  if [[ -z "${CNN_RESULTS_DIR:-}" ]]; then
    if [[ -n "$CNN_CKPT" ]]; then
      CNN_RESULTS_DIR="$(dirname "$CNN_CKPT")"
    elif [[ -n "$CNN_BARE_CKPT" ]]; then
      CNN_RESULTS_DIR="$(dirname "$CNN_BARE_CKPT")"
    fi
  fi
  LINEAR_RESULTS_DIR="${LINEAR_RESULTS_DIR:-$LINEAR_SAVE_DIR}"
  CNN_RESULTS_DIR="${CNN_RESULTS_DIR:-$CNN_SAVE_DIR}"
  SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$SCRIPT_DIR/eval_summary_trainS${TRAIN_SEED}_step${TRAIN_STEP}.md}"

  SUMMARY_CMD=(
    "$PYTHON" "$SCRIPT_DIR/summarize_eval.py"
    --suite "$SUITE"
    --train-seed "$TRAIN_SEED"
    --train-step "$TRAIN_STEP"
    --linear-dir "$LINEAR_RESULTS_DIR"
    --cnn-dir "$CNN_RESULTS_DIR"
    --output "$SUMMARY_OUTPUT"
    --require-complete
  )
  set +e
  "${SUMMARY_CMD[@]}"
  SUMMARY_STATUS=$?
  set -e
elif [[ "$RUN_SUMMARY" == "1" && "$DRY_RUN" != "1" ]]; then
  echo "[exp06-suite] summary skipped because at least one worker failed."
elif [[ "$DRY_RUN" == "1" ]]; then
  echo "[exp06-suite] DRY_RUN=1; summary intentionally skipped."
fi

if (( STATUS_LINEAR != 0 || STATUS_CNN != 0 || SUMMARY_STATUS != 0 )); then
  echo "[exp06-suite] FAILED; inspect $LINEAR_LOG and $CNN_LOG" >&2
  exit 1
fi
echo "[exp06-suite] complete"
