#!/usr/bin/env bash
# Evaluate the exp05 SimpleViT/CylViT step-2000 Lightning checkpoints.
#
# Defaults:
#   SimpleViT eval -> GPU 0
#   CylViT eval    -> GPU 1
#   K=1
#   ANGLES=0,90,180,270
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
SEED="${SEED:-42}"
K="${K:-1}"
ANGLES="${ANGLES:-0,90,180,270}"
SIMPLE_GPU="${SIMPLE_GPU:-0}"
CYL_GPU="${CYL_GPU:-1}"
STEPS="${STEPS:-1}"
CFG_SCALE="${CFG_SCALE:-1.0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-eval-step2000}"

if [ "$K" = "1" ]; then
  DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"
else
  DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
fi

LOG_DIR="worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude"
mkdir -p "$LOG_DIR"

run_eval_set () {
  local model_name="$1"
  local gpu="$2"
  local model_config="$3"
  local ckpt_path="$4"

  if [ ! -f "$ckpt_path" ]; then
    echo "Missing checkpoint: $ckpt_path" >&2
    exit 2
  fi

  IFS=',' read -ra angle_list <<< "$ANGLES"
  for angle in "${angle_list[@]}"; do
    echo "=== eval ${model_name} K=${K} seed=${SEED} angle=${angle} ckpt=${ckpt_path} ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" eval_FLAC.py \
      --model-config "$model_config" \
      --dataset-config "$DATASET_CONFIG" \
      --ckpt-path "$ckpt_path" \
      --cfg-scale "$CFG_SCALE" \
      --steps "$STEPS" \
      --seed "$SEED" \
      --rotate-deg "$angle" \
      --eval-name "exp05_${model_name}_step2000_K${K}_seed${SEED}_rot${angle}"
  done
}

run_eval_set \
  "simplevit" \
  "$SIMPLE_GPU" \
  "src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json" \
  "outputs_FLAC/exp05_simplevit_2500s_s42/epoch=0-step=2000.ckpt" &
SIMPLE_PID=$!

run_eval_set \
  "cylvit" \
  "$CYL_GPU" \
  "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json" \
  "outputs_FLAC/exp05_cylvit_2500s_s42/epoch=0-step=2000.ckpt" &
CYL_PID=$!

echo "SimpleViT eval PID=${SIMPLE_PID}"
echo "CylViT eval PID=${CYL_PID}"

wait "$SIMPLE_PID"
wait "$CYL_PID"

echo "=== exp05 step2000 eval complete ==="
