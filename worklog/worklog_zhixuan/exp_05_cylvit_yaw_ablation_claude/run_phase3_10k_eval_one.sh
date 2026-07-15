#!/usr/bin/env bash
# Evaluate one model's Phase 3 step-5000 checkpoint (effective total 10k).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-../venv/bin/python}"
MODEL="${MODEL:?Set MODEL to simplevit or cylvit}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
SAVE_DIR="outputs_FLAC/exp05_${MODEL}_phase3_total30k_s${SEED}"
CKPT="${SAVE_DIR}/epoch=1-step=5000.ckpt"

case "$MODEL" in
  simplevit) MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json" ;;
  cylvit) MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json" ;;
  *) echo "Unknown MODEL=$MODEL" >&2; exit 2 ;;
esac

if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

for yaw in 0 90 180 270; do
  eval_name="exp05_${MODEL}_convergence_total10k_yaw${yaw}"
  suffix=""
  if [ "$yaw" != "0" ]; then suffix="_rot${yaw}"; fi
  metrics="${CKPT%.ckpt}_metrics_1_1.0_${eval_name}${suffix}.json"
  if [ -f "$metrics" ]; then
    echo "[eval] already exists: $metrics"
    continue
  fi
  echo "[eval] model=${MODEL} total=10k yaw=${yaw}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval_FLAC.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
    --ckpt-path "$CKPT" \
    --cfg-scale 1.0 \
    --steps 1 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --rotate-deg "$yaw" \
    --eval-name "$eval_name"
done

"$PYTHON" worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/summarize_phase3_convergence.py \
  --model "$MODEL" \
  --search-root outputs_FLAC \
  --output "worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/phase3_${MODEL}_convergence.md"
