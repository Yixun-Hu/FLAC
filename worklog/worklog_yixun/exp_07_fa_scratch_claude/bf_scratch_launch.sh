#!/usr/bin/env bash
# ============================================================================
# bf_scratch_launch.sh - exp_07: B-F FROM-SCRATCH training launch (GPU 1).
#
# Yixun go (2026-07-16): B-F takes the GPU-1 slot right after the B-V extend
# completes (extend -> B-F -> P1). Mirrors the B-V launch manifest exactly,
# differing ONLY via FLAC_AR_BF.json (+cond_method: fa_invariant,
# +frame_avg_angles: [0,90,180,270]) - arms proven init-identical under seed
# 42 by assert_arm_configs.py (state_dict sha256 match).
#
# Recipe (= B-V phase 1): 8x8 (eff 64, the only rung that fits B-F on 48 GiB
# per M0), seed 42, --max-steps 67500, EMA on (config), ckpt every 2500 into
# outputs_FLAC/exp07_BF/, bf16-mixed, workers 6, HF_HUB_OFFLINE=1. ~9.6 d.
# Screens: bf_screen.sh at each 10k ckpt (EMA+online, K=8 s42 full split).
#
# wandb: default OFF (LOGGER=none) - current WANDB_API_KEY belongs to
# yixunhu21@gmail.com, NOT the requested yh4742@princeton.edu. To enable:
# export yh4742's key (env var OVERRIDES `wandb login`), run LOGGER=wandb;
# the fail-closed identity gate below verifies the email. With wandb,
# train.py:129 nests ckpts under <save-dir>/<name>/<exp-name>/checkpoints/.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
LOGGER="${LOGGER:-none}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_BF_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== B-F FROM-SCRATCH - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - 8x8 eff64 seed42 -> 67500 - logger=${LOGGER} ==="

# --- GPU 1 must be free (do not co-launch onto the extend or anything else) ---
# fail-CLOSED: a failed query must not read as "free" (pipefail is set above)
BUSY="$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')"
rc_q=$?
[ "$rc_q" -eq 0 ] || { echo "nvidia-smi query failed (rc=${rc_q}) - refusing to launch blind"; exit 2; }
[ -z "$BUSY" ] || { echo "GPU 1 busy (pids: ${BUSY}) - refusing to launch"; exit 2; }

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
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

# --- pre-launch gate: DINOv3 pin + arm init-identity (fail-closed, as B-V) ---
# offline so the gate's own model construction cannot contact the Hub and
# mutate the cache it just validated
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader -i 1
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_BF.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size 8 --accum-batches 8 --num-workers 6 --seed 42 \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp07_BF --experiment-name exp07_BF \
  --save-dir outputs_FLAC/exp07_BF
rc=$?
echo "=== B-F FROM-SCRATCH exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
