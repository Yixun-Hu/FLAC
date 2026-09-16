#!/bin/bash
# exp_14 ladder rung 5b: contract-violation termination probe (van arm, 2-GPU DDP) on a split where every receiver keeps ONE source
# => every restricted context pool is empty => DatasetContractError on the first batch; expect non-zero rc, both ranks gone, no checkpoint.
set -u
WT=/home/yixunhu/codespace/exp-14-data-curve; KIT=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude
REC=$WT/worklog/worklog_yixun/exp_14_data_curve; NAS=/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
cd "$WT"; export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
TS=$(date +%Y-%m-%d_%H-%M-%S); LOG=$REC/rung5b_probe_van_${TS}.log
echo "=== rung5b start $(date -Is) FLAC=$(git rev-parse --short HEAD)" | tee "$LOG"
CUDA_VISIBLE_DEVICES=0,1 timeout 900 python train.py --model-config "$KIT/configs/FLAC_AR_exp14_vanS.json" \
  --dataset-config worklog/worklog_yixun/exp_14_data_curve/rung5b_probe_dataset_config.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 3 --batch-size 2 --accum-batches 1 --num-workers 2 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every 100000 \
  --name smoke5b_probe --experiment-name smoke5b_probe --save-dir "$NAS/smoke5b_probe" >> "$LOG" 2>&1
RC=$?; echo "=== rung5b rc=$RC end $(date -Is)" | tee -a "$LOG"
sleep 5; echo "=== surviving train.py procs from this worktree: $(pgrep -f "$WT/train.py|smoke5b_probe" | wc -l)" | tee -a "$LOG"
echo "=== checkpoints written: $(find $NAS/smoke5b_probe -name '*.ckpt' 2>/dev/null | wc -l)" | tee -a "$LOG"
echo "RUNG5B_DONE rc=$RC $(date -Is)" > $REC/rung5b_probe_${TS}.done
