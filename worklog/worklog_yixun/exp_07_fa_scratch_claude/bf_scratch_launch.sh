#!/usr/bin/env bash
# ============================================================================
# bf_scratch_launch.sh - exp_07: B-F FROM-SCRATCH training, 2-GPU DDP + SyncBN.
#
# Yixun mandate (2026-07-16): 2x A6000 DDP with SyncBatchNorm so the effective
# BN batch = 64 = paper (deliberate deviation from the release CODE, which got
# BN-64 via micro-64 on one H100 without SyncBN - we match the paper's BN
# statistics, not the release's code path). Global eff batch 64 fixed; the
# micro/accum rung comes from the M1 probe: MB x 2 GPUs x ACC = 64.
# LAUNCH-GATED: only run after Yixun's explicit go on the probed rung.
#
# Otherwise mirrors the B-V phase-1 manifest exactly: FLAC_AR_BF.json (byte-copy
# of release config + cond_method fa_invariant + frame_avg_angles [0,90,180,270];
# init-identity vs B-V sha256-asserted), seed 42, --max-steps 67500, EMA on
# (config), ckpt every 2500, bf16-mixed (ini default, no flag), workers 6/rank,
# HF_HUB_OFFLINE=1. Strategy passed EXPLICITLY (defaults.ini strategy="auto" is
# forwarded verbatim by train.py:159-170; the num_gpus>1 fallback never fires).
#
# wandb: default ON per Yixun (LOGGER=wandb) - project FLAC_exp07_BF, run
# exp07_BF, account yh4742@princeton.edu; key self-extracted past ~/.bashrc's
# interactive guard; fail-closed identity gate. Ckpts nest under
# outputs_FLAC/exp07_BF/<project>/<run>/checkpoints/ (train.py:129).
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"    # micro-batch per GPU (M1 winning rung)
ACC="${ACC:-1}"   # accumulation      (M1 winning rung)
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_BF_train.log"

# invariant (review findings): accumulation never feeds BN statistics, so the
# BN=64 mandate leaves exactly ONE legal rung - pin it literally (string
# equality; arithmetic like MB*2*ACC==64 is bypassable via bash int overflow):
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== B-F FROM-SCRATCH DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> 67500 - logger=${LOGGER} ==="

# --- BOTH GPUs must be free (fail-CLOSED: query failure refuses launch) ---
for G in 0 1; do
  BUSY="$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] || { echo "nvidia-smi query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ -z "$BUSY" ] || { echo "GPU ${G} busy (pids: ${BUSY}) - refusing to launch"; exit 2; }
done

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  # ~/.bashrc's interactive guard blocks non-interactive sourcing, so
  # extract the newest exported key directly (verified 2026-07-16 ->
  # yh4742@princeton.edu; the gate below re-verifies at every launch)
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

# --- pre-launch gate: DINOv3 pin + arm init-identity (fail-closed, as B-V) ---
# offline so the gate's own model construction cannot contact the Hub and
# mutate the cache it just validated
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true (fail-closed in train.py below num_gpus 2) | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC}"

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_BF.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp07_BF --experiment-name exp07_BF \
  --save-dir outputs_FLAC/exp07_BF
rc=$?
echo "=== B-F DDP+SyncBN exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
