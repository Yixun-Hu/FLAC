#!/bin/bash
# exp_12: resume a STOPPED conditioning arm from its newest checkpoint.
#
#   bash resume_arm.sh <gpu> <run_name>            # e.g. bash resume_arm.sh 1 exp12C_ray12
#
# Picks the highest-step checkpoint in the run's own directory, verifies it is complete and
# loadable (optimizer + LR-scheduler state present), and restarts training with
# `--ckpt-path`, which restores global_step / optimizer / scheduler and continues to 67,500.
#
# DISCLOSED CAVEAT (HANDOFF trap): a Lightning resume is a FRESH STOCHASTIC CONTINUATION --
# RNG state and dataloader position are NOT restored. The resumed run therefore sees a
# different sample order and different noise draws than an uninterrupted run would have.
# Weights, optimizer moments and the LR schedule ARE restored. This is the same resume
# semantics used by earlier exp_09 runs and must be disclosed in any result that uses it.
set -euo pipefail

GPU="${1:?gpu}"; RUN="${2:?run name}"
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

REC=worklog/worklog_yixun/exp_12_arms
case "$RUN" in
  exp12A_c3c4)     CFG=$REC/FLAC_AR_exp12A.json ;;
  exp12C_ray12)    CFG=$REC/FLAC_AR_exp12C.json ;;
  exp12B_ssl_cond) CFG=$REC/FLAC_AR_exp12B.json ;;
  *) echo "REFUSE: unknown run '$RUN'"; exit 2 ;;
esac

CKPT=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null \
       | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "REFUSE: no checkpoint found for $RUN"; exit 2; }

# Fail closed on a truncated checkpoint: resuming from a half-written file would either
# crash six hours in or, worse, silently restore garbage.
python - "$CKPT" <<'PY'
import sys, torch
p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
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
  --max-steps 67500 --batch-size 32 --accum-batches 2 --num-workers 6 --seed 42
  --num-gpus 1 --logger wandb --checkpoint-every 2500
  --name "$RUN" --experiment-name "$RUN" --save-dir "outputs_FLAC/$RUN")

{
  echo "resumed_at: $(date -Is)"
  echo "gpu: $GPU  run: $RUN  from_step: $STEP"
  echo "ckpt: $CKPT"
  echo "ckpt_sha256: $(sha256sum "$CKPT" | cut -d' ' -f1)"
  echo "exp12_sha: $(git rev-parse HEAD)"
  echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
  echo "NOTE: fresh stochastic continuation -- RNG and dataloader position are NOT restored."
  echo "command: CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
} >> "$REC/at_resume_$RUN.txt"

CUDA_VISIBLE_DEVICES=$GPU nohup "${cmd[@]}" >> "$REC/train_$RUN.log" 2>&1 &
echo "pid: $!" >> "$REC/at_resume_$RUN.txt"
echo "$RUN resumed from step $STEP -> GPU$GPU pid $!"
