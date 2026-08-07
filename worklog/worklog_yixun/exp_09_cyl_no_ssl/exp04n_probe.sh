#!/usr/bin/env bash
# ============================================================================
# exp04n_probe.sh - exp_04 (mean-pool + MLP head) LIFECYCLE PROBE (plan §4.4).
#
# exp04n_launch.sh pins --max-steps 67500 with no override interface, so the probe is a
# SEPARATE reviewed wrapper (exp_02 house style, modelled on probe_p1_rerun_saveresume):
#   phase A: fresh run, 30 steps, checkpoint every 10  -> assert a step-30 ckpt exists;
#   phase B: RESUME from that ckpt to ABSOLUTE step 40 -> assert a step-40 ckpt exists.
# It samples per-rank VRAM throughout and PRINTS the measured peak, the RECOMMENDED frozen
# free-VRAM value (peak x 1.15) and the checkpoint size. It deliberately does NOT write
# exp04n_frozen_min_free.txt: that number is committed into the records by hand, pre-launch,
# after review - which is why this probe is the ONE script EXEMPT from the frozen-file gate
# the launcher enforces (it is the script that PRODUCES the number).
#
# Every OTHER gate is the launcher's, unchanged and fail-closed: the same BN-64 rung, the
# same pin gate (assert_arm_configs_exp04n.py with BOTH pins REQUIRED), the same wandb
# identity gate (~/.netrc on this cluster), and a PROVISIONAL per-GPU free-VRAM floor of
# 21,900 MiB (B-F's inherited number - used ONLY here, to keep the probe itself co-tenant
# safe, and never for the full run).
#
# Scientific shape is the arm's: FLAC_AR_exp04n.json, micro-batch 32/GPU x 2 GPUs x accum 1
# (eff 64), SyncBN, ddp_find_unused_parameters_true, seed 42, num-workers 6.
#
# Usage (from the Slurm wrapper probe_exp04n.sbatch):
#   EXPECT_PACKAGE_SHA=<cyl HEAD> EXPECT_EXP04_SHA=<worktree HEAD> bash exp04n_probe.sh
# ============================================================================
set -uo pipefail
cd /n/fs/gatrdp/codespace/exp04-meanpool-mlp-cond || exit 3   # absolute worktree root for THIS cluster
export PYTHONPATH=/n/fs/gatrdp/codespace/cylindrical-dinov3/src   # process-local package src

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
RECORDS="/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/exp_04_meanpool_mlp_cond_claude"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"
SAVE="${SAVE:-/n/fs/gatrdp/outputs/exp04n_meanpoolmlp_PROBE}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXP04N_LOG_DIR:-$RECORDS}/meanpool_mlp_cond_${TS}_j${SLURM_JOB_ID:-nojob}_probe.log"
# The frozen free-VRAM file this probe FEEDS (it must NOT exist yet as far as the launcher is
# concerned; this script never writes it - see the header).
FROZEN_FILE="${EXPDIR}/exp04n_frozen_min_free.txt"
# PROVISIONAL floor for the probe itself (B-F's inherited 21,900 MiB). The full launch binds
# the value DERIVED below instead, and refuses without the records file.
MIN_FREE_MB="${MIN_FREE_MB:-21900}"

# --- the BN-64 rung pin (identical to the launcher; string equality, not arithmetic) ---
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

# --- fresh probe save-dir (a leftover tree would make the step-30/40 assertions vacuous) ---
[ ! -e "$SAVE" ] || { echo "REFUSING: probe save dir '${SAVE}' exists - a stale tree would satisfy the step-30/40 checkpoint assertions without running anything. Remove it or set SAVE=."; exit 3; }

# --- BOTH pins REQUIRED (no defaults, no fallback set) ---
EXPECT_PACKAGE_SHA="${EXPECT_PACKAGE_SHA:-}"; EXPECT_EXP04_SHA="${EXPECT_EXP04_SHA:-}"
[ -n "$EXPECT_PACKAGE_SHA" ] && [ -n "$EXPECT_EXP04_SHA" ] || { echo "REFUSING: both EXPECT_PACKAGE_SHA (cylindrical-dinov3 HEAD) and EXPECT_EXP04_SHA (worktree HEAD) must be set"; exit 3; }

mkdir -p "$(dirname "$LOG")" || exit 3
exec > >(tee -a "$LOG") 2>&1
echo "=== exp04n lifecycle probe (30 -> resume -> 40) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 - job ${SLURM_JOB_ID:-nojob} ==="
echo "probe save-dir: ${SAVE} | provisional MIN_FREE_MB=${MIN_FREE_MB} (frozen file to be fed: ${FROZEN_FILE})"

# --- per-GPU FREE-VRAM gate (fail-CLOSED on query errors; provisional floor) ---
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=name,index,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  [ -r "$HOME/.netrc" ] || { echo "wandb identity: ~/.netrc is absent/unreadable - run 'wandb login' or use LOGGER=none - abort"; exit 2; }
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

# --- the launcher's pin + arm-wiring gate, bound to the SAME config the probe trains ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs_exp04n.py" --expect-package-sha "$EXPECT_PACKAGE_SHA" --expect-exp04-sha "$EXPECT_EXP04_SHA" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import cylindrical_dinov3 as c; print('cylindrical_dinov3', c.__version__, c.__file__)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'

# --- per-rank VRAM sampler (background; killed on every exit path) ---
SAMPLES="${RECORDS}/exp04n_probe_vram_samples_j${SLURM_JOB_ID:-nojob}.txt"
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 15; done ) > "$SAMPLES" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

COMMON=( --model-config "${EXPDIR}/FLAC_AR_exp04n.json"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --checkpoint-every 10 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --logger "$LOGGER" --name FLAC_exp04n_meanpoolmlp_probe --experiment-name exp04n_meanpoolmlp_probe
  --save-dir "$SAVE" )

echo "--- phase A: fresh, max-steps 30 (ckpt every 10) ---"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py "${COMMON[@]}" --max-steps 30
rc_a=$?
[ "$rc_a" -eq 0 ] || { echo "PROBE FAIL: phase A exited rc=${rc_a}"; exit 1; }
CKPT30=$(find "$SAVE" -name '*step=30.ckpt' | head -1)
[ -n "$CKPT30" ] || { echo "PROBE FAIL: no step-30 ckpt"; exit 1; }
echo "step-30 ckpt: $CKPT30 ($(stat -c%s "$CKPT30") bytes)"
sha256sum "$CKPT30"

echo "--- phase B: resume from step-30, ABSOLUTE max-steps 40 ---"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py "${COMMON[@]}" --max-steps 40 --ckpt-path "$CKPT30"
rc_b=$?
[ "$rc_b" -eq 0 ] || { echo "PROBE FAIL: phase B exited rc=${rc_b}"; exit 1; }
CKPT40=$(find "$SAVE" -name '*step=40.ckpt' | head -1)
[ -n "$CKPT40" ] || { echo "PROBE FAIL: no step-40 ckpt after resume"; exit 1; }
echo "step-40 ckpt: $CKPT40 ($(stat -c%s "$CKPT40") bytes)"

kill $SAMPLER 2>/dev/null || true
PEAK=$(awk -F', ' '{print $2}' "$SAMPLES" | sort -n | tail -1)
case "${PEAK:-}" in ''|*[!0-9]*) echo "PROBE FAIL: no usable VRAM samples in ${SAMPLES}"; exit 1;; esac
RECOMMENDED=$(( (PEAK * 115 + 99) / 100 ))   # peak x 1.15, rounded UP
echo "PEAK per-rank VRAM used (MiB): ${PEAK}"
echo "RECOMMENDED frozen MIN_FREE_MB (peak x 1.15, round up): ${RECOMMENDED}"
echo "CKPT size (bytes): $(stat -c%s "$CKPT30")"
echo "NEXT STEP (records, by hand + review): write ${RECOMMENDED} into ${FROZEN_FILE} and commit it BEFORE the full launch; exp04n_launch.sh REFUSES while that file is absent."
echo "PROBE PASS: save + resume validated (30 -> 40) with the exp04n mean-pool + MLP head"
