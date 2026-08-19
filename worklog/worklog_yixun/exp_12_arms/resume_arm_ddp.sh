#!/bin/bash
# exp_12: resume a 2-GPU DDP + SyncBN run from its newest checkpoint, ON THE SAME RECIPE.
#
#   bash resume_arm_ddp.sh <run_name>
#
# resume_arm.sh is the SINGLE-GPU path (1 GPU, accum 2). Using it on a DDP run would resume
# with BN statistics over 32 instead of 64 -- the exact regime mix the from-scratch DDP
# restart existed to avoid -- so DDP runs must resume through this script instead. It
# refuses any run name that is not a DDP run for that reason.
set -euo pipefail

RUN="${1:?run name}"
case "$RUN" in
  *_ddp) : ;;
  *) echo "REFUSE: '$RUN' is not a DDP run; use resume_arm.sh for single-GPU runs."; exit 2 ;;
esac

cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
REC=worklog/worklog_yixun/exp_12_arms

case "$RUN" in
  exp12B_ssl_cond_ddp) CFG=$REC/FLAC_AR_exp12B.json ;;
  exp12A_c3c4_ddp)     CFG=$REC/FLAC_AR_exp12A.json ;;
  *) echo "REFUSE: unknown DDP run '$RUN'"; exit 2 ;;
esac

CKPT=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null \
       | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "REFUSE: no checkpoint found for $RUN"; exit 2; }

python - "$CKPT" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert d.get("optimizer_states"), "no optimizer state"
assert d.get("lr_schedulers"), "no lr scheduler state"
assert len(d["state_dict"]) == 1279, f"unexpected key count {len(d['state_dict'])}"
print(f"  checkpoint OK: step {d['global_step']} epoch {d['epoch']}")
PY

STEP=$(echo "$CKPT" | grep -oP 'step=\K[0-9]+')
cmd=(python train.py
  --model-config "$CFG"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --ckpt-path "$CKPT"
  --max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --logger wandb --checkpoint-every 2500
  --name "$RUN" --experiment-name "$RUN" --save-dir "outputs_FLAC/$RUN")

{
  echo "resumed_at: $(date -Is)"
  echo "run: $RUN  from_step: $STEP  recipe: 2-GPU DDP, batch 32/card, accum 1, eff 64, SyncBN ON"
  echo "ckpt: $CKPT"
  echo "ckpt_sha256: $(sha256sum "$CKPT" | cut -d' ' -f1)"
  echo "exp12_sha: $(git rev-parse HEAD)"
  echo "NOTE: fresh stochastic continuation -- RNG and dataloader position are NOT restored."
  echo "command: ${cmd[*]}"
} >> "$REC/at_resume_$RUN.txt"

nohup "${cmd[@]}" >> "$REC/train_$RUN.log" 2>&1 &
echo "pid: $!" >> "$REC/at_resume_$RUN.txt"
echo "$RUN resumed from step $STEP on 2 GPUs -> pid $!"
