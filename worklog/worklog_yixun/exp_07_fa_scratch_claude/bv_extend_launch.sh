#!/usr/bin/env bash
# ============================================================================
# bv_extend_launch.sh - exp_07 phase 2, "extend past 67.5k" run.
#
# Resumes the phase-1 B-V run (exp07_BV) from its step-67500 checkpoint and
# continues training to --max-steps (arg 1, default 100000) to search for a
# better checkpoint beyond 67.5k. Yixun: "continue our previous train on
# B-V@67.5k to check what is the best ckpt we have." PL restores
# optimizer/scheduler/EMA/loop/global-step via ckpt_path (train.py:191).
#
# NOTE (not bit-exact): Lightning 2.1 does NOT restore RNG here and this
# checkpoint carries no RNG state, so the continuation is a FRESH stochastic
# trajectory from step 67500 onward -- fine for a "find the best later ckpt"
# search. Same 8x8 (eff 64) recipe + seed 42 as phase 1; new ckpts
# step=70000..MAXSTEPS every 2500. InverseLR barely decays (lr ~4.84e-5 at
# 67.5k) so late gains are plausible -- tests the "under-training" hypothesis.
#
# wandb: default OFF (LOGGER=none) because the current WANDB_API_KEY belongs to
# yixunhu21@gmail.com, NOT the requested yh4742@princeton.edu. To log to wandb,
# set yh4742's key (export WANDB_API_KEY=...; it OVERRIDES `wandb login`) then
# run with LOGGER=wandb -- the fail-closed identity gate below verifies the email.
# With wandb, train.py:129 nests ckpts under <save-dir>/<name>/<exp-name>/checkpoints/.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
MAXSTEPS="${1:-100000}"
LOGGER="${LOGGER:-none}"
# env-overridable so restarts resume from the NEWEST extend ckpt instead of 67.5k
# (see bv_extend_stop_restart.md; default preserved for the original launch semantics)
RESUME_CKPT="${RESUME_CKPT:-outputs_FLAC/exp07_BV/epoch=14-step=67500.ckpt}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_BVextend_train.log"

# --- validate target: integer strictly > 67500 (else nothing to extend) ---
case "$MAXSTEPS" in
  ''|*[!0-9]*) echo "MAXSTEPS must be a positive integer > 67500 (got: '${MAXSTEPS}') - abort"; exit 2 ;;
esac
[ "$MAXSTEPS" -gt 67500 ] || { echo "MAXSTEPS must be strictly > 67500 (got: ${MAXSTEPS}) - nothing to extend - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== B-V EXTEND - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - resume ${RESUME_CKPT} -> max ${MAXSTEPS} - logger=${LOGGER} ==="
[ -f "$RESUME_CKPT" ] || { echo "MISSING resume ckpt ${RESUME_CKPT} - abort"; exit 1; }

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  # ~/.bashrc's interactive guard blocks non-interactive sourcing; extract the
  # newest exported key directly (mirrors the reviewed bf_scratch_launch.sh line)
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
  python - <<'PY'
import sys
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY
  gate=$?
  if [ "$gate" -ne 0 ]; then echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - set the right WANDB_API_KEY or use LOGGER=none - abort"; exit 2; fi
fi

# --- pre-launch gate: DINOv3 pin + arm init-identity (same fail-closed gate as phase 1) ---
python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader -i 1

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_BV.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --ckpt-path "$RESUME_CKPT" \
  --max-steps "$MAXSTEPS" --batch-size 8 --accum-batches 8 --num-workers 6 --seed 42 \
  --precision bf16-mixed --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp07_BVextend --experiment-name exp07_BVextend \
  --save-dir outputs_FLAC/exp07_BVextend
rc=$?
echo "=== B-V EXTEND exited rc=${rc} at $(date '+%H:%M:%S') ==="
exit $rc
