#!/bin/bash
# exp_12 D1 endpoint eval for ONE arm: K in {1,8} x seeds 42..46 on the 6,337-entry
# unseen_eval split. LEAN (Yixun's no-gates directive): the exp-09 pin gate is NOT invoked;
# the two mandatory conditioning flags ARE, because eval_FLAC.py defaults --cond-method to
# 'vanilla' and would otherwise silently evaluate the raw-pose path (THE trap, HANDOFF #1).
#
#   bash eval_arm.sh <run_name> <model_config> <gpu> [ckpt_path]
set -uo pipefail

RUN="${1:?run name, e.g. exp12A_c3c4}"
CFG="${2:?model config json}"
GPU="${3:?gpu index}"
CKPT="${4:-}"

cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

if [ -z "$CKPT" ]; then
  CKPT=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1)
fi
if [ ! -f "$CKPT" ]; then
  echo "REFUSE: no step=67500 checkpoint for $RUN (looked for outputs_FLAC/$RUN/*/*/checkpoints/*step=67500.ckpt)"
  exit 2
fi

LOG=worklog/worklog_yixun/exp_12_arms/eval_$RUN${K_VALUES:+_K${K_VALUES// /}}.log
echo "=== eval $RUN | ckpt $CKPT | gpu $GPU | $(date -Is) ===" | tee -a "$LOG"

for K in ${K_VALUES:-1 8}; do
  case $K in
    1) DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json ;;
    8) DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json ;;
  esac
  for SEED in 42 43 44 45 46; do
    NAME="${RUN}_D1_K${K}_s${SEED}"
    OUT="$(dirname "$CKPT")/$(basename "$CKPT" .ckpt)_metrics_1_1.0_${NAME}_fa_invariant_a1.json"
    if [ -f "$OUT" ]; then
      echo "[skip] $NAME already has metrics" | tee -a "$LOG"
      continue
    fi
    echo "[run ] $NAME $(date -Is)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python eval_FLAC.py \
      --model-config "$CFG" --dataset-config "$DS" --ckpt-path "$CKPT" \
      --cond-method fa_invariant --frame-avg-angles 0 \
      --cond-autocast bf16 --seed "$SEED" --steps 1 --cfg-scale 1.0 \
      --eval-name "$NAME" >> "$LOG" 2>&1
    rc=$?
    echo "[done] $NAME rc=$rc $(date -Is)" | tee -a "$LOG"
  done
done
echo "=== eval $RUN COMPLETE $(date -Is) ===" | tee -a "$LOG"
