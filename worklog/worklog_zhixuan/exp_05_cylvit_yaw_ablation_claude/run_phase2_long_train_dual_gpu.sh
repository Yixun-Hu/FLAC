#!/usr/bin/env bash
# Launch SimpleViT and CylViT long training in parallel on two GPUs.
#
# Defaults:
#   SimpleViT -> GPU 0
#   CylViT    -> GPU 1
#
# Environment variables:
#   MAX_STEPS=5000
#   SEED=42
#   SIMPLE_GPU=0
#   CYL_GPU=1
#   SIMPLE_RESUME_FROM=outputs_FLAC/exp05_simplevit_5000s_s42/last.ckpt
#   CYL_RESUME_FROM=outputs_FLAC/exp05_cylvit_5000s_s42/last.ckpt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MAX_STEPS="${MAX_STEPS:-5000}"
SEED="${SEED:-42}"
SIMPLE_GPU="${SIMPLE_GPU:-0}"
CYL_GPU="${CYL_GPU:-1}"
SIMPLE_RESUME_FROM="${SIMPLE_RESUME_FROM:-}"
CYL_RESUME_FROM="${CYL_RESUME_FROM:-}"

LOG_DIR="worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude"
mkdir -p "$LOG_DIR"

SIMPLE_LOG="${LOG_DIR}/simplevit_${MAX_STEPS}s_s${SEED}_gpu${SIMPLE_GPU}.log"
CYL_LOG="${LOG_DIR}/cylvit_${MAX_STEPS}s_s${SEED}_gpu${CYL_GPU}.log"

COMMON_SCRIPT="${LOG_DIR}/run_phase2_long_train_one.sh"

echo "=== launching exp05 dual-GPU long train ==="
echo "MAX_STEPS=${MAX_STEPS}"
echo "SEED=${SEED}"
echo "SimpleViT GPU=${SIMPLE_GPU}, log=${SIMPLE_LOG}, resume=${SIMPLE_RESUME_FROM:-none}"
echo "CylViT    GPU=${CYL_GPU}, log=${CYL_LOG}, resume=${CYL_RESUME_FROM:-none}"

env MODEL=simplevit GPU="$SIMPLE_GPU" MAX_STEPS="$MAX_STEPS" SEED="$SEED" \
  RESUME_FROM="$SIMPLE_RESUME_FROM" \
  bash "$COMMON_SCRIPT" > "$SIMPLE_LOG" 2>&1 &
SIMPLE_PID=$!

env MODEL=cylvit GPU="$CYL_GPU" MAX_STEPS="$MAX_STEPS" SEED="$SEED" \
  RESUME_FROM="$CYL_RESUME_FROM" \
  bash "$COMMON_SCRIPT" > "$CYL_LOG" 2>&1 &
CYL_PID=$!

echo "SimpleViT PID=${SIMPLE_PID}"
echo "CylViT PID=${CYL_PID}"
echo
echo "Follow logs with:"
echo "  tail -f ${SIMPLE_LOG}"
echo "  tail -f ${CYL_LOG}"
echo
echo "Check processes with:"
echo "  ps -p ${SIMPLE_PID},${CYL_PID} -o pid,ppid,stat,etime,cmd"
