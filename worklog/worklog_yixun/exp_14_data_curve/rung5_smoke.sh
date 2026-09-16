#!/bin/bash
# exp_14 ladder rung 5: smallest-batch, NO-checkpoint smoke per arm, 2-GPU DDP+SyncBN (co-tenant with exp13_vanB by Yixun's decision).
# Planner one-off; reviewed in the integrative review. Never touches other processes.
set -u
WT=/home/yixunhu/codespace/exp-14-data-curve; KIT=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude
REC=$WT/worklog/worklog_yixun/exp_14_data_curve; NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
cd "$WT"; export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
TS=$(date +%Y-%m-%d_%H-%M-%S)
for ARM in cyl van; do
  CFG=$KIT/configs/FLAC_AR_exp14_${ARM}S.json; LOG=$REC/rung5_smoke_${ARM}_${TS}.log
  echo "=== rung5 $ARM start $(date -Is) FLAC=$(git rev-parse --short HEAD) cfg=$(sha256sum $CFG | cut -c1-16)" | tee "$LOG"
  CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps 3 --batch-size 2 --accum-batches 1 --num-workers 2 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 100000 \
    --name smoke5_dc_$ARM --experiment-name smoke5_dc_$ARM --save-dir "$NAS/smoke5_$ARM" >> "$LOG" 2>&1
  echo "=== rung5 $ARM rc=$? end $(date -Is)" | tee -a "$LOG"
done
echo "RUNG5_DONE $(date -Is)" > $REC/rung5_smoke_${TS}.done
