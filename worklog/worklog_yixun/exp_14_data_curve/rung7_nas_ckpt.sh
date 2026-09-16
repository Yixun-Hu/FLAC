#!/bin/bash
# exp_14 ladder rung 7: NAS checkpoint write/reload probe with the run contract (cyl arm): 5 steps, checkpoint at step 5, then validate.
set -u
WT=/home/yixunhu/codespace/exp-14-data-curve; KIT=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude
REC=$WT/worklog/worklog_yixun/exp_14_data_curve; RUN=/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke/smoke7_cyl
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac
cd "$WT"; export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
TS=$(date +%Y-%m-%d_%H-%M-%S); LOG=$REC/rung7_nas_cyl_${TS}.log; mkdir -p "$RUN"
CFG=$KIT/configs/FLAC_AR_exp14_cylS.json; DS=src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json
echo "=== rung7 start $(date -Is) FLAC=$(git rev-parse --short HEAD)" | tee "$LOG"
CONTRACT=$(python -m src.training.run_contract make-contract --run-dir "$RUN" --run-id smoke7_cyl --fraction 0.25 --dataset-config "$DS" --split-json data/AR/train_frac025_s2026.json --model-config "$CFG" --seed 42 --micro-batch 32 --num-gpus 2 --accum-batches 1 --sync-batchnorm true --flac-sha $(git rev-parse HEAD) --package-sha $(git -C /home/yixunhu/codespace/cylindrical-dinov3-exp13pin rev-parse HEAD) 2>>"$LOG"); echo "contract: $CONTRACT rc=$?" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" --dataset-config "$DS" \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 5 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every 5 \
  --name smoke7_cyl --experiment-name smoke7_cyl --save-dir "$RUN" --run-contract-json "$CONTRACT" >> "$LOG" 2>&1
echo "=== rung7 train rc=$? $(date -Is)" | tee -a "$LOG"
CK=$(find "$RUN" -name '*step=5.ckpt' | head -1); echo "ckpt: $CK size=$(stat -c%s "$CK" 2>/dev/null)" | tee -a "$LOG"
sleep 60; echo "size after 60s: $(stat -c%s "$CK" 2>/dev/null)" | tee -a "$LOG"
python -m src.training.run_contract validate --ckpt "$CK" --contract "$CONTRACT" --model-config "$CFG" --expect-step 5 --for-resume 2>&1 | tee -a "$LOG"; echo "=== validate rc=${PIPESTATUS[0]}" | tee -a "$LOG"
echo "RUNG7_DONE $(date -Is)" > $REC/rung7_nas_${TS}.done
