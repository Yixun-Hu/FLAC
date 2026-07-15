#!/usr/bin/env bash
# Evaluate yaw=0 at each total-step milestone and C4 yaw at total 30k.
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

case "$MODEL" in
  simplevit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_SimpleViT.json"
    INIT_CKPT="outputs_FLAC/exp05_simplevit_resume2500to5000_s42/FLAC_exp05_simplevit_resume2500to5000_s42.ckpt"
    ;;
  cylvit)
    MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT.json"
    INIT_CKPT="outputs_FLAC/exp05_cylvit_5000s_s42/FLAC_exp05_cylvit_5000s_s42.ckpt"
    ;;
  *) echo "Unknown MODEL=$MODEL" >&2; exit 2 ;;
esac

find_phase_ckpt() {
  local phase_step="$1"
  find "$SAVE_DIR" -maxdepth 1 -type f -name "epoch=*-step=${phase_step}.ckpt" -print -quit
}

run_eval() {
  local ckpt="$1"
  local total_k="$2"
  local yaw="$3"
  local eval_name="exp05_${MODEL}_convergence_total${total_k}k_yaw${yaw}"
  local suffix=""
  if [ "$yaw" != "0" ]; then suffix="_rot${yaw}"; fi
  if find "$(dirname "$ckpt")" -maxdepth 1 -type f \
      -name "$(basename "${ckpt%.ckpt}")_metrics_1_1.0_${eval_name}${suffix}.json" \
      -print -quit | grep -q .; then
    echo "[eval] already exists: total=${total_k}k yaw=${yaw}"
    return
  fi
  echo "[eval] model=${MODEL} total=${total_k}k yaw=${yaw} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" eval_FLAC.py \
    --model-config "$MODEL_CONFIG" \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
    --ckpt-path "$ckpt" \
    --cfg-scale 1.0 \
    --steps 1 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --rotate-deg "$yaw" \
    --eval-name "$eval_name"
}

run_eval "$INIT_CKPT" 5 0

for phase_step in 5000 10000 15000 20000 25000; do
  ckpt="$(find_phase_ckpt "$phase_step")"
  if [ -z "$ckpt" ]; then
    echo "Missing periodic checkpoint for phase step ${phase_step} in ${SAVE_DIR}" >&2
    exit 2
  fi
  total_k=$((phase_step / 1000 + 5))
  run_eval "$ckpt" "$total_k" 0
  if [ "$phase_step" = "25000" ]; then
    run_eval "$ckpt" 30 90
    run_eval "$ckpt" 30 180
    run_eval "$ckpt" 30 270
  fi
done

"$PYTHON" worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/summarize_phase3_convergence.py \
  --model "$MODEL" \
  --search-root outputs_FLAC \
  --output "worklog/worklog_zhixuan/exp_05_cylvit_yaw_ablation_claude/phase3_${MODEL}_convergence.md"
