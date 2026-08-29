#!/bin/bash
# exp_21 trajectory screen (Yixun-approved 2026-08-28): K=8, seed 42, all 15 pre-40k
# boundary checkpoints, GPU 1, CONCURRENT with the registered 34-cell block on GPU 0
# (deliberate, logged exception to the driver's one-eval-at-a-time guard; distinct
# filenames, separate GPU). Protocol flags mirror the registered template
# (announcement 05, every flag explicit). Single-seed SCREEN cells: structurally
# excluded from model_comparison by design; they feed the trajectory analysis only.
# Resume-safe: a cell whose metrics JSON already exists is skipped.
set -u
cd /home/yixunhu/codespace/FLAC
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
CKDIR=outputs_FLAC/exp21_BFC/FLAC_exp21_BFC/exp21_BFC/checkpoints
for STEP in 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000 32500 35000 37500; do
  CKPT=$(ls $CKDIR/epoch=*-step=${STEP}.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "MISSING ckpt for step $STEP - abort"; exit 2; fi
  DONE=$(ls "$CKDIR"/*"exp21_BFC_TRAJ_S${STEP}_K8_s42"*.json 2>/dev/null | grep -v stream | head -1)
  if [ -n "$DONE" ]; then echo "SKIP step $STEP (exists)"; continue; fi
  echo "=== TRAJ cell step $STEP start $(date +%H:%M:%S) ==="
  CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
    --model-config worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json \
    --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
    --ckpt-path "$CKPT" \
    --cond-method fa_cartesian --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64 \
    --rotate-mode fixed --rotate-deg 0 --cond-autocast bf16 \
    --batch-size 64 --cfg-scale 1.0 --steps 1 --record-per-scene \
    --seed 42 --eval-name exp21_BFC_TRAJ_S${STEP}_K8_s42
  rc=$?
  echo "=== TRAJ cell step $STEP rc=$rc $(date +%H:%M:%S) ==="
  [ $rc -ne 0 ] && { echo "TRAJ SCREEN ABORT at step $STEP"; exit $rc; }
done
echo "=== TRAJ SCREEN COMPLETE (15 cells) ==="
