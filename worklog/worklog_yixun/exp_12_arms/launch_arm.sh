#!/bin/bash
# exp_12: launch ONE conditioning arm on ONE GPU, exp_06 C2 recipe verbatim except
# 1 GPU + --accum-batches 2 (effective batch 64 = the registered 2x32) and no sync-bn.
# Prints "pid: <n>" as its last line so a chain can wait on it.
#
#   bash launch_arm.sh <gpu> <model_config> <run_name>
set -euo pipefail

GPU="${1:?gpu}"; CFG="${2:?model config}"; RUN="${3:?run name}"
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

REC=worklog/worklog_yixun/exp_12_arms
mkdir -p outputs_FLAC

cmd=(python train.py
  --model-config "$CFG"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps 67500 --batch-size 32 --accum-batches 2 --num-workers 6 --seed 42
  --num-gpus 1 --logger wandb --checkpoint-every 2500
  --name "$RUN" --experiment-name "$RUN" --save-dir "outputs_FLAC/$RUN")

{
  echo "launched_at: $(date -Is)"
  echo "gpu: $GPU  run: $RUN"
  echo "exp12_sha: $(git rev-parse HEAD)"
  echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
  echo "model_config_sha256: $(sha256sum "$CFG" | cut -d' ' -f1)"
  echo "train_manifest_sha256: $(sha256sum data/AR/train.json | cut -d' ' -f1)"
  ssl=$(python -c "
import json,sys
c=json.load(open('$CFG'))
for b in c['model']['conditioning']['configs']:
    if b['type']=='ViTCoordinates':
        print(b['config']['ViT'].get('ssl_ckpt','<none>')); break")
  echo "ssl_ckpt: $ssl"
  # An `if`, not an `&&` chain: a false `&&` list as the last command of a block would
  # make this script exit non-zero and the chain would read that as a failed launch.
  if [ "$ssl" != "<none>" ] && [ -f "$ssl" ]; then
    echo "ssl_ckpt_sha256: $(sha256sum "$ssl" | cut -d' ' -f1)"
  fi
  echo "command: CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
} > "$REC/at_launch_$RUN.txt"

if [ -n "${DRY_RUN:-}" ]; then
  echo "DRY_RUN: would launch on GPU$GPU: ${cmd[*]}"
  echo "pid: 0"
  exit 0
fi

CUDA_VISIBLE_DEVICES=$GPU nohup "${cmd[@]}" > "$REC/train_$RUN.log" 2>&1 &
PID=$!
echo "pid: $PID" >> "$REC/at_launch_$RUN.txt"
echo "pid: $PID"
