#!/bin/bash
# exp_12 arms A & C — LEAN launch (Yixun's explicit no-gates directive, 2026-08-07:
# "don't set gate to save fund/gpu hours, I need the results!").
# One GPU per arm, contemporaneous (satisfies the exp_02 contemporaneity rule for A-vs-C).
# Recipe = exp_06 C2 verbatim except: 1 GPU with --accum-batches 2 (keeps effective
# batch 64 = 2x32 of the registered 2-GPU recipe), no sync-batchnorm (1-GPU fail-closed).
set -euo pipefail
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1

REC=worklog/worklog_yixun/exp_12_arms
mkdir -p outputs_FLAC "$REC"

launch () {  # $1=gpu $2=arm-letter $3=run-name
  local cfg="$REC/FLAC_AR_exp12$2.json"
  local cmd=(python train.py
    --model-config "$cfg"
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors
    --max-steps 67500 --batch-size 32 --accum-batches 2 --num-workers 6 --seed 42
    --num-gpus 1 --logger wandb --checkpoint-every 2500
    --name "$3" --experiment-name "$3" --save-dir "outputs_FLAC/$3")
  {
    echo "launched_at: $(date -Is)"
    echo "gpu: $1  arm: $2  run: $3"
    echo "exp12_sha: $(git rev-parse HEAD)"
    echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
    echo "model_config_sha256: $(sha256sum "$cfg" | cut -d' ' -f1)"
    echo "train_manifest_sha256: $(sha256sum data/AR/train.json | cut -d' ' -f1)"
    echo "command: CUDA_VISIBLE_DEVICES=$1 ${cmd[*]}"
  } > "$REC/at_launch_$3.txt"
  CUDA_VISIBLE_DEVICES=$1 nohup "${cmd[@]}" > "$REC/train_$3.log" 2>&1 &
  echo "pid: $!" >> "$REC/at_launch_$3.txt"
  echo "arm $2 -> GPU$1 pid $!"
}

launch 0 A exp12A_c3c4
launch 1 C exp12C_ray12
