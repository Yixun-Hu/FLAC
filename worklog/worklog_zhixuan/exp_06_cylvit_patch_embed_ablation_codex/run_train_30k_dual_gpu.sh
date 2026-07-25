#!/usr/bin/env bash
# Prepare one matched initialization pair, then run both variants concurrently.
# The launcher remains in the foreground and waits for both single-GPU jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-${FLAC_ROOT}/../venv/bin/python}"

LINEAR_GPU="${LINEAR_GPU:-0}"
CNN_GPU="${CNN_GPU:-1}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-30000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
INIT_DIR="${FLAC_ROOT}/outputs_FLAC/exp06_cylvit_pe_matched_initializations"
ONE_SCRIPT="${SCRIPT_DIR}/run_train_30k_one.sh"

if [ "$LINEAR_GPU" = "$CNN_GPU" ] && [ "$ALLOW_SHARED_GPU" != "1" ]; then
  echo "LINEAR_GPU and CNN_GPU both resolve to ${LINEAR_GPU}; set ALLOW_SHARED_GPU=1 to override intentionally" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$FLAC_ROOT"

echo "[exp06:dual] validating or creating the matched seed-${SEED} initialization pair"
"$PYTHON" "${SCRIPT_DIR}/prepare_matched_initializations.py" \
  --seed "$SEED" \
  --output-dir "$INIT_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LINEAR_LOG="${LOG_DIR}/linear_trainS${SEED}_${STAMP}.log"
CNN_LOG="${LOG_DIR}/cnn_trainS${SEED}_${STAMP}.log"
LINEAR_PID_FILE="${LOG_DIR}/linear_trainS${SEED}.pid"
CNN_PID_FILE="${LOG_DIR}/cnn_trainS${SEED}.pid"

env VARIANT=linear GPU="$LINEAR_GPU" SEED="$SEED" MAX_STEPS="$MAX_STEPS" \
  PYTHON="$PYTHON" SKIP_PREPARE=1 \
  bash "$ONE_SCRIPT" >>"$LINEAR_LOG" 2>&1 &
LINEAR_PID=$!

env VARIANT=cnn GPU="$CNN_GPU" SEED="$SEED" MAX_STEPS="$MAX_STEPS" \
  PYTHON="$PYTHON" SKIP_PREPARE=1 \
  bash "$ONE_SCRIPT" >>"$CNN_LOG" 2>&1 &
CNN_PID=$!

printf '%s\n' "$LINEAR_PID" >"$LINEAR_PID_FILE"
printf '%s\n' "$CNN_PID" >"$CNN_PID_FILE"

cleanup() {
  local signal="${1:-TERM}"
  echo "[exp06:dual] received ${signal}; forwarding TERM to ${LINEAR_PID},${CNN_PID}" >&2
  kill "$LINEAR_PID" "$CNN_PID" 2>/dev/null || true
  rm -f "$LINEAR_PID_FILE" "$CNN_PID_FILE"
}
trap 'cleanup INT; exit 130' INT
trap 'cleanup TERM; exit 143' TERM

echo "[exp06:dual] linear physical GPU=${LINEAR_GPU} PID=${LINEAR_PID} log=${LINEAR_LOG}"
echo "[exp06:dual] cnn    physical GPU=${CNN_GPU} PID=${CNN_PID} log=${CNN_LOG}"
echo "[exp06:dual] two independent devices=1 jobs; launcher PID=$$ waits in foreground"

LINEAR_STATUS=0
CNN_STATUS=0
FINISHED_PID=""

set +e
wait -n -p FINISHED_PID "$LINEAR_PID" "$CNN_PID"
FIRST_STATUS=$?

if [ "$FINISHED_PID" = "$LINEAR_PID" ]; then
  LINEAR_STATUS=$FIRST_STATUS
  if [ "$LINEAR_STATUS" -ne 0 ]; then
    echo "[exp06:dual] linear failed (exit=${LINEAR_STATUS}); terminating cnn to preserve the paired run" >&2
    kill "$CNN_PID" 2>/dev/null || true
  fi
  wait "$CNN_PID"
  CNN_STATUS=$?
elif [ "$FINISHED_PID" = "$CNN_PID" ]; then
  CNN_STATUS=$FIRST_STATUS
  if [ "$CNN_STATUS" -ne 0 ]; then
    echo "[exp06:dual] cnn failed (exit=${CNN_STATUS}); terminating linear to preserve the paired run" >&2
    kill "$LINEAR_PID" 2>/dev/null || true
  fi
  wait "$LINEAR_PID"
  LINEAR_STATUS=$?
else
  echo "[exp06:dual] could not identify the first completed child process" >&2
  kill "$LINEAR_PID" "$CNN_PID" 2>/dev/null || true
  wait "$LINEAR_PID"
  LINEAR_STATUS=$?
  wait "$CNN_PID"
  CNN_STATUS=$?
fi
set -e

rm -f "$LINEAR_PID_FILE" "$CNN_PID_FILE"
trap - INT TERM

echo "[exp06:dual] complete linear_exit=${LINEAR_STATUS} cnn_exit=${CNN_STATUS}"
if [ "$LINEAR_STATUS" -ne 0 ] || [ "$CNN_STATUS" -ne 0 ]; then
  exit 1
fi
