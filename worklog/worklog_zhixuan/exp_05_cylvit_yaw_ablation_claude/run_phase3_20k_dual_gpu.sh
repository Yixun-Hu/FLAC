#!/usr/bin/env bash
# Launch matched SimpleViT/CylViT Phase 3 runs on two GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SIMPLE_GPU="${SIMPLE_GPU:-0}"
CYL_GPU="${CYL_GPU:-1}"
SEED="${SEED:-42}"
LOG_DIR="worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude"
ONE_SCRIPT="${LOG_DIR}/run_phase3_20k_one.sh"
SIMPLE_LOG="${LOG_DIR}/simplevit_phase3_total20k_s${SEED}_gpu${SIMPLE_GPU}.log"
CYL_LOG="${LOG_DIR}/cylvit_phase3_total20k_s${SEED}_gpu${CYL_GPU}.log"

env MODEL=simplevit GPU="$SIMPLE_GPU" SEED="$SEED" \
  bash "$ONE_SCRIPT" > "$SIMPLE_LOG" 2>&1 &
SIMPLE_PID=$!

env MODEL=cylvit GPU="$CYL_GPU" SEED="$SEED" \
  bash "$ONE_SCRIPT" > "$CYL_LOG" 2>&1 &
CYL_PID=$!

echo "SimpleViT PID=${SIMPLE_PID} log=${SIMPLE_LOG}"
echo "CylViT PID=${CYL_PID} log=${CYL_LOG}"
echo "Monitor: ps -p ${SIMPLE_PID},${CYL_PID} -o pid,ppid,stat,etime,cmd"
