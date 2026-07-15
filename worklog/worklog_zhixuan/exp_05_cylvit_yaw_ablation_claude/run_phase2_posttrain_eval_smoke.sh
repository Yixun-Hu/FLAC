#!/usr/bin/env bash
# Minimal post-training eval smoke for exported SimpleViT/CylViT checkpoints.
# This is not the full Phase 3 sweep; it just checks eval_FLAC.py loads both
# trained checkpoints and produces exp01-style metrics at a few yaw angles.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
K="${K:-1}"
ANGLES="${ANGLES:-0,22.5,90}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp05-eval}"

if [ "$K" = "1" ]; then
  DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"
else
  DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
fi

run_eval () {
  local model_name="$1"
  local model_config="$2"
  local ckpt_path="$3"
  IFS=',' read -ra angle_list <<< "$ANGLES"
  for angle in "${angle_list[@]}"; do
    echo "=== eval ${model_name} K=${K} seed=${SEED} angle=${angle} ==="
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval_FLAC.py \
      --model-config "$model_config" \
      --dataset-config "$DATASET_CONFIG" \
      --ckpt-path "$ckpt_path" \
      --cfg-scale 1.0 \
      --steps 1 \
      --seed "$SEED" \
      --rotate-deg "$angle" \
      --eval-name "exp05_${model_name}_K${K}_seed${SEED}_rot${angle}"
  done
}

run_eval \
  "simplevit_short" \
  "src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json" \
  "outputs_FLAC/exp05_simplevit_short_s${SEED}/FLAC_exp05_simplevit_short_s${SEED}.ckpt"

run_eval \
  "cylvit_short" \
  "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json" \
  "outputs_FLAC/exp05_cylvit_short_s${SEED}/FLAC_exp05_cylvit_short_s${SEED}.ckpt"

echo "=== exp05 posttrain eval smoke complete ==="
