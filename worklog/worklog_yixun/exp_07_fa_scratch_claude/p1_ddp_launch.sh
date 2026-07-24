#!/usr/bin/env bash
# ============================================================================
# p1_ddp_launch.sh - exp_07 P1: B-V at the IDENTICAL B-F recipe (attribution arm).
#
# Yixun 2026-07-23 ("Stop now -> P1 immediately"): B-F stopped at 40k for
# futility (EDT/C50 plateaued ~2x the 8x8 anchor, R@1 0.15x); P1 answers the
# open question - recipe or fa_invariant? Recipe = EXACTLY B-F's:
# 32/GPU x 2 GPUs x accum 1 (eff 64), SyncBN (BN=64), ViT grad-ckpt ON (for
# recipe identity; numerics-identical), env flac, seed 42, 67.5k steps,
# ckpt/2500, wandb. Config FLAC_AR_BVp1.json = BV byte-copy + the 2
# gradient_checkpointing keys; BVp1 vs BF diff = cond_method+frame_avg_angles
# ONLY (single-delta matched pair; asserted below).
#
# Estimands: (a) ATTRIBUTION - P1's 10k/20k/30k/40k screens vs B-F's at matched
# steps (single-delta) and vs the 8x8 B-V anchor (recipe effect); (b) parity
# secondary at endpoint (plan_bv_parity.md tiers). Mirrors bf_scratch_launch.sh
# (reviewed SHIP) except config/names/dirs + the diff pre-flight.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_P1_train.log"

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== P1 (B-V @ B-F recipe) DDP+SyncBN+ckpt - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> 67500 - logger=${LOGGER} ==="

# --- config-contract pre-flight (fail-closed): the single-delta guarantees ---
d1=$(diff <(python3 -m json.tool "${EXPDIR}/FLAC_AR_BVp1.json") <(python3 -m json.tool "${EXPDIR}/FLAC_AR_BF.json") | grep -cE "^[<>]") || true
d2=$(diff <(python3 -m json.tool "${EXPDIR}/FLAC_AR_BV.json") <(python3 -m json.tool "${EXPDIR}/FLAC_AR_BVp1.json") | grep -cE "^[<>]") || true
# BVp1 vs BF: exactly 9 changed lines (cond_method + 6-line frame_avg_angles array + 2 context braces)
# BVp1 vs BV: exactly 4 changed lines (2x gradient_checkpointing + 2 context)
[ "$d1" -ge 8 ] && [ "$d1" -le 10 ] && [ "$d2" -ge 3 ] && [ "$d2" -le 5 ] || { echo "config-contract drift (BVp1-vs-BF ${d1} lines, BVp1-vs-BV ${d2} lines) - abort"; exit 2; }
echo "config contract OK (BVp1-vs-BF ${d1} diff lines, BVp1-vs-BV ${d2})"

# --- per-GPU FREE-VRAM gate (co-tenancy policy; M1-measured rank ~15.9 GB) ---
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate ---
if [ "$LOGGER" = "wandb" ]; then
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
  [ "$gate" -eq 0 ] || { echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - abort"; exit 2; }
fi

# --- DINOv3 pin + arm init-identity (offline, fail-closed) ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | grad-ckpt: config"

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_BVp1.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp07_P1 --experiment-name exp07_P1 \
  --save-dir outputs_FLAC/exp07_P1
rc=$?
echo "=== P1 exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
