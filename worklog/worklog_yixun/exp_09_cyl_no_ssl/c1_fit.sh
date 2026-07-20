#!/usr/bin/env bash
# ============================================================================
# c1_fit.sh <EXTERNAL_LOG_DIR> - exp-09 C1 FIT PROBE (integrative-review finding 2).
#
# Records-referenced fit-probe launcher. Measures the exp-09 per-GPU peak VRAM under a
# SHORT training invocation (exact B-F CLI shape) wrapped by the EXTERNAL peak sampler,
# so the C1 records step can FREEZE  MIN_FREE_MB = max(peak_gpu0, peak_gpu1) + 4,096 MiB.
# This script only MEASURES + serialises the derived gate; the freeze happens in records.
#
# GPU-GATED (plan §6): needs an explicit Yixun C1 go. The fit probe ALONE uses the
# conservative BOOTSTRAP gate (free >= 21,900 MiB/GPU = B-F's registered threshold; exp-09
# is B-F minus 4x frame-averaging, so B-F's envelope is an upper bound). It CANNOT use the
# derived threshold it exists to produce (plan §6, r3 #2). Every LATER launch (C1 smoke,
# C2, D) uses the FROZEN MIN_FREE_MB instead.
#
# $1 is a REQUIRED external log directory: the tee log and the peak JSON land THERE, never
# in the worktree (audit attempt-1 lesson: an in-worktree log dirties the clean-tree gate).
# ============================================================================
set -o pipefail    # verbatim (records-referenced, review finding 2)
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
WORKTREE="$(readlink -f .)"

# --- $1 = EXTERNAL log dir (required; must be OUTSIDE the worktree) ---
LOGDIR="${1:-}"
[ -n "$LOGDIR" ] || { echo "usage: c1_fit.sh <EXTERNAL_LOG_DIR>  (required; the log+peak JSON must land OUTSIDE the worktree)"; exit 3; }
mkdir -p "$LOGDIR" 2>/dev/null || { echo "cannot create log dir '${LOGDIR}' - abort"; exit 3; }
LOGDIR_ABS="$(readlink -f "$LOGDIR")"
case "${LOGDIR_ABS}/" in
  "${WORKTREE}"/*)
    echo "REFUSING: log dir '${LOGDIR_ABS}' is INSIDE the worktree '${WORKTREE}' - it would dirty"
    echo "the clean-tree pin gate (audit attempt-1 lesson). Pass an EXTERNAL directory."
    exit 3;;
esac

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${LOGDIR_ABS}/exp09_${TS}_c1_fit.log"
PEAK_JSON="${LOGDIR_ABS}/exp09_${TS}_c1_fit_peak.json"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp-09 C1 FIT PROBE - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - log+peak -> ${LOGDIR_ABS} ==="

# --- BOOTSTRAP free-VRAM gate: 21,900 MiB/GPU on BOTH GPUs (fail-CLOSED on query errors) ---
BOOTSTRAP_MIN_FREE_MB=21900
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to probe blind"; exit 2; }
  [ "$FREE" -ge "$BOOTSTRAP_MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < bootstrap ${BOOTSTRAP_MIN_FREE_MB} MiB/GPU - refusing to probe"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at probe start ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- pin gate FIRST (fail-closed; offline so it cannot mutate the cache it validates).
# EXPECT_PACKAGE_SHA (optional) pins the accepted package HEAD to one SHA at records freeze,
# without editing the gate (integrative-review finding 3). ---
EXPECT_PACKAGE_SHA="${EXPECT_PACKAGE_SHA:-}"
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs_exp09.py" ${EXPECT_PACKAGE_SHA:+--expect-package-sha "$EXPECT_PACKAGE_SHA"} || { echo "PIN GATE FAILED - abort"; exit 1; }

# --- SHORT training under the external peak sampler (exact B-F CLI shape; small max-steps;
# checkpoint-every <= max-steps so a ckpt is written and the run exercises the ckpt path) ---
MB=32; ACC=1
FIT_STEPS="${FIT_STEPS:-15}"
CKPT_EVERY="${CKPT_EVERY:-$FIT_STEPS}"
[ "$CKPT_EVERY" -le "$FIT_STEPS" ] || { echo "checkpoint-every ${CKPT_EVERY} must be <= max-steps ${FIT_STEPS} - abort"; exit 3; }
SCRATCH="${LOGDIR_ABS}/c1_fit_run_${TS}"
echo "--- fit: ${FIT_STEPS} steps, ckpt every ${CKPT_EVERY}, rung ${MB}x2x${ACC} eff64 SyncBN, under gpu_peak_sampler.py ---"

python "${EXPDIR}/gpu_peak_sampler.py" --out "$PEAK_JSON" --interval 1.0 --gpus 0,1 --gate-margin-mib 4096 -- \
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model-config "${EXPDIR}/FLAC_AR_exp09.json" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps "$FIT_STEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every "$CKPT_EVERY" \
    --name exp09_c1_fit --experiment-name exp09_c1_fit --save-dir "$SCRATCH"
rc=$?
echo "=== C1 FIT exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') - peak JSON: ${PEAK_JSON} ==="
echo "--- derived-gate readout (COMPUTATION ONLY; the FREEZE into c1_frozen_min_free.txt happens in the C1 records step) ---"
python - "$PEAK_JSON" <<'PY' || echo "(peak JSON unreadable)"
import json, sys
d = json.load(open(sys.argv[1]))
print("per_gpu_peaks_mib =", d["per_gpu_peaks_mib"])
print("samples_per_gpu   =", d["samples_per_gpu"])
print("sampler_ok        =", d["sampler_ok"])
print("derived_gate_mib  =", d["derived_gate_mib"], "(= max(peaks) + 4096; freeze this in records)")
sys.exit(0 if d["sampler_ok"] else 3)
PY
exit $rc
