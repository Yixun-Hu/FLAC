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
# full-state resume (PL ckpt_path; mirrors the reviewed bv_extend pattern). Added
# 2026-07-25 after the session-restart teardown killed the run at ~34.3k (last
# ckpt 32,500): RESUME_CKPT="<path>" resumes; empty = fresh launch.
RESUME_CKPT="${RESUME_CKPT:-}"
# R@1-extension support (Yixun 2026-07-27 "Extend P1 for R@1, then close"):
# override the training budget; default = the original 67,500.
MAXSTEPS="${MAXSTEPS:-67500}"
case "$MAXSTEPS" in ''|*[!0-9]*) echo "MAXSTEPS must be a positive integer (got '${MAXSTEPS}') - abort"; exit 2;; esac
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_P1_train.log"

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== P1 (B-V @ B-F recipe) DDP+SyncBN+ckpt - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> 67500 - logger=${LOGGER} - resume='${RESUME_CKPT:-fresh}' ==="
if [ -n "$RESUME_CKPT" ]; then [ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }; fi

# --- config-contract pre-flight (fail-closed; review-prescribed EXACT parsed-object
# assertions, replacing the diff-count windows which could pass a wrong config) ---
python3 - <<'PY' || { echo "config-contract FAILED - abort"; exit 2; }
import json, copy
base = "worklog/worklog_yixun/exp_07_fa_scratch_claude/"
bv  = json.load(open(base + "FLAC_AR_BV.json"))
bvp = json.load(open(base + "FLAC_AR_BVp1.json"))
bf  = json.load(open(base + "FLAC_AR_BF.json"))
exp = copy.deepcopy(bv)
n = 0
for c in exp["model"]["conditioning"]["configs"]:
    if c.get("id") in ("source_vit", "context_poses_vit"):
        c["config"]["gradient_checkpointing"] = True; n += 1
assert n == 2, f"expected exactly 2 ViT conditioner blocks, found {n}"
assert bvp == exp, "BVp1 != BV + the 2 gradient_checkpointing keys (parsed-object mismatch)"
exp2 = copy.deepcopy(bvp)
exp2["training"]["cond_method"] = "fa_invariant"
exp2["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
assert bf == exp2, "BF != BVp1 + cond_method/frame_avg_angles (parsed-object mismatch)"
print("config contract OK (exact parsed-object assertions)")
PY

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
  ${RESUME_CKPT:+--ckpt-path "$RESUME_CKPT"} \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp07_P1 --experiment-name exp07_P1 \
  --save-dir outputs_FLAC/exp07_P1
rc=$?
echo "=== P1 exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
