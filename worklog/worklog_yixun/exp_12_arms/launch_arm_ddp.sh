#!/bin/bash
# exp_12: launch a conditioning arm on TWO GPUs with DDP + SyncBatchNorm -- the original
# exp-09 / exp_06 C2 recipe: 2 GPUs x batch 32, accumulation 1, effective batch 64,
# sync_batchnorm=true, ddp_find_unused_parameters_true.
#
#   bash launch_arm_ddp.sh <model_config> <run_name>
#
# Why this exists (Yixun's request, 2026-08-17, after Codex correctly flagged my imprecision):
# 1 GPU x 32 x accum 2 and 2 GPUs x 32 x accum 1 both give effective batch 64 but are NOT
# numerically equivalent. This model contains 20 BatchNorm modules (60 buffers) in the
# context_audio RIR CNN -- verified in the checkpoint -- so BN statistics come from 32
# samples on the single-GPU scheme and from 64 under SyncBN. That is a semantic difference,
# not float noise, which is exactly why this must be a run from step 0 rather than a
# mid-flight switch: a resumed run would carry 7,500 steps of BN-over-32 running statistics
# into a BN-over-64 regime.
set -euo pipefail

CFG="${1:?model config}"; RUN="${2:?run name}"
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
  --max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --logger wandb --checkpoint-every 2500
  --name "$RUN" --experiment-name "$RUN" --save-dir "outputs_FLAC/$RUN")

{
  echo "launched_at: $(date -Is)"
  echo "run: $RUN   recipe: 2-GPU DDP, batch 32/card, accum 1, effective 64, SyncBN ON"
  echo "exp12_sha: $(git rev-parse HEAD)"
  echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
  echo "model_config_sha256: $(sha256sum "$CFG" | cut -d' ' -f1)"
  echo "train_manifest_sha256: $(sha256sum data/AR/train.json | cut -d' ' -f1)"
  ssl=$(python -c "
import json
c=json.load(open('$CFG'))
for b in c['model']['conditioning']['configs']:
    if b['type']=='ViTCoordinates':
        print(b['config']['ViT'].get('ssl_ckpt','<none>')); break")
  echo "ssl_ckpt: $ssl"
  if [ "$ssl" != "<none>" ] && [ -f "$ssl" ]; then
    echo "ssl_ckpt_sha256: $(sha256sum "$ssl" | cut -d' ' -f1)"
  fi
  echo "command: ${cmd[*]}"
} > "$REC/at_launch_$RUN.txt"

nohup "${cmd[@]}" > "$REC/train_$RUN.log" 2>&1 &
PID=$!
echo "pid: $PID" >> "$REC/at_launch_$RUN.txt"
echo "pid: $PID"
