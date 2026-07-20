#!/usr/bin/env bash
# ============================================================================
# exp09_launch.sh - exp-09 (FLAC no-SSL cylindrical DINOv3): C2 full training,
# 2-GPU DDP + SyncBN. B-F/P1 protocol EXACTLY (micro 32/GPU x2 x accum1 = eff 64,
# SyncBN, seed 42, AdamW 5e-5/wd 1e-3/InverseLR(1e6,0.5,0.99), EMA, bf16-mixed,
# 67,500 steps, ckpt every 2,500) - the ONLY differences from B-F are the ViT
# backbone (cylindrical_dinov3 + gauge=cylindrical_xyz, official weights, NO SSL)
# and cond_method fa_invariant with frame_avg_angles [0.0] (one base/frame pass,
# no extra frame-average passes). See FLAC_AR_exp09.json and its config-delta test.
#
# Cloned from exp_07's bf_scratch_launch.sh: same reviewed skeleton (fail-closed
# free-VRAM gate, wandb identity gate, pre-launch pin gate, `set -uo pipefail`).
#
# GPU-GATED (plan §6): B-F completion does NOT authorize this launch. C1/C2/D each
# need an explicit Yixun go AND a free-VRAM check. THIS launcher (C2/full) BINDS the
# FROZEN threshold MIN_FREE_MB (= measured exp-09 peak + 4,096 MiB) that the C1 fit probe
# derived, read from the records file c1_frozen_min_free.txt: absent file, non-numeric
# value, or a MIN_FREE_MB override != the frozen value all make this script REFUSE
# (fail-closed). It can no longer launch against B-F's inherited number or an arbitrary
# override. Optional resume via CKPT_PATH (--ckpt-path passthrough; see below).
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"    # micro-batch per GPU (B-F/P1 BN-64 rung)
ACC="${ACC:-1}"   # accumulation      (B-F/P1 BN-64 rung)
CKPT_PATH="${CKPT_PATH:-}"   # optional resume checkpoint (train.py:230; documented below)
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
# The train log is a C2 records artifact kept in EXPDIR by default; EXP09_LOG_DIR relocates
# it (e.g. for CPU tests) so a gate rehearsal never dirties the worktree.
LOG="${EXP09_LOG_DIR:-$EXPDIR}/exp09_${TS}_cylNoSSL_train.log"

# invariant (review findings): accumulation never feeds BN statistics, so the
# BN=64 mandate leaves exactly ONE legal rung - pin it literally (string equality;
# arithmetic like MB*2*ACC==64 is bypassable via bash int overflow):
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

# --- EXACT-MATCH frozen-threshold binding (integrative-review findings 3 & 5) --------------
# The launcher must NOT accept an arbitrary numeric MIN_FREE_MB (the old gate did, so a value
# BELOW the derived threshold could launch). Instead it BINDS to the exact value the C1 fit
# probe FROZE into a records file, and cross-checks any provided MIN_FREE_MB against it:
#   * frozen-records file absent            -> REFUSE (C1 has not frozen the threshold yet);
#   * frozen value non-numeric / empty      -> REFUSE;
#   * MIN_FREE_MB provided and != frozen    -> REFUSE (arbitrary override rejected);
#   * MIN_FREE_MB absent, or == frozen      -> bind to the frozen value and proceed.
# Done BEFORE the tee so a refusal never opens a log. C1_FROZEN_MIN_FREE_FILE relocates the
# records file (tests/records only); the exact-VALUE cross-check is the real protection.
FROZEN_FILE="${C1_FROZEN_MIN_FREE_FILE:-${EXPDIR}/c1_frozen_min_free.txt}"
[ -f "$FROZEN_FILE" ] || { echo "REFUSING TO LAUNCH: frozen-records file '${FROZEN_FILE}' is absent - the C1 fit probe (plan §3/§6) has not FROZEN the exp-09 free-VRAM threshold yet."; exit 3; }
FROZEN_RAW="$(cat "$FROZEN_FILE")"
FROZEN="$(printf '%s' "$FROZEN_RAW" | tr -d '[:space:]')"
case "$FROZEN" in
  ''|*[!0-9]*) echo "REFUSING TO LAUNCH: frozen value '${FROZEN_RAW}' in ${FROZEN_FILE} is not a plain integer."; exit 3;;
esac
MIN_FREE_MB="${MIN_FREE_MB:-$FROZEN}"
[ "$MIN_FREE_MB" = "$FROZEN" ] || { echo "REFUSING TO LAUNCH: MIN_FREE_MB=${MIN_FREE_MB} does not match the FROZEN C1 threshold ${FROZEN} (arbitrary override rejected - the launcher binds the exact derived value)."; exit 3; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp-09 cyl-no-SSL DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> 67500 - logger=${LOGGER} ==="

# MIN_FREE_MB is bound to the FROZEN C1 threshold above (= measured exp-09 peak + 4,096 MiB);
# it is NEVER B-F's inherited 21,900 MiB and can no longer be an arbitrary numeric override.

# --- per-GPU FREE-VRAM gate (fail-CLOSED on query errors; mirrors bf_scratch_launch.sh)
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate (only when wandb is requested) ---
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
  if [ "$gate" -ne 0 ]; then echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - set the right WANDB_API_KEY or use LOGGER=none - abort"; exit 2; fi
fi

# --- pre-launch pin gate: custom-class + gauge + official-weight + config-delta
# (fail-closed, as B-F). Offline so the gate's own model construction cannot
# contact the Hub and mutate the cache it just validated.
EXPECT_PACKAGE_SHA="${EXPECT_PACKAGE_SHA:-}"   # records freeze may pin one package SHA (finding 3)
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs_exp09.py" ${EXPECT_PACKAGE_SHA:+--expect-package-sha "$EXPECT_PACKAGE_SHA"} || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import cylindrical_dinov3 as c; print('cylindrical_dinov3', c.__version__, c.__file__)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | MIN_FREE_MB=${MIN_FREE_MB}"

# --- optional RESUME passthrough (integrative-review finding 4; train.py:230 forwards
# --ckpt-path to trainer.fit(ckpt_path=...)). Set CKPT_PATH=<file> to resume the reviewed
# exact command. DISCLOSURE (plan §3): a Lightning resume is a FRESH stochastic continuation
# - RNG and dataloader position are NOT restored - so it is disclosed, never a bit-exact restart.
RESUME_ARGS=()
if [ -n "$CKPT_PATH" ]; then
  [ -f "$CKPT_PATH" ] || { echo "CKPT_PATH='${CKPT_PATH}' not found - abort"; exit 3; }
  RESUME_ARGS=(--ckpt-path "$CKPT_PATH")
  echo "--- RESUME from ${CKPT_PATH} (fresh stochastic continuation, plan §3) ---"
fi

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_exp09.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp09_cylNoSSL --experiment-name exp09_cylNoSSL \
  --save-dir outputs_FLAC/exp09_cylNoSSL \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
rc=$?
echo "=== exp-09 cyl-no-SSL DDP+SyncBN exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
