#!/usr/bin/env bash
# ============================================================================
# c1_smoke.sh <FROZEN_MIN_FREE_MB> <EXTERNAL_LOG_DIR> - exp-09 C1 100-STEP SMOKE
# (integrative-review finding 2, item 3).
#
# Continues C1 after the fit probe has FROZEN the threshold. $1 is the FROZEN
# MIN_FREE_MB (= measured peak + 4,096 MiB): it is REQUIRED and must be a plain integer -
# a TBD / non-numeric / absent value is REFUSED (this script must never gate on a
# placeholder). Both GPUs are gated on it. $2 is a REQUIRED external log dir (the tee log
# + verification JSONs land there, never in the worktree).
#
# Runs 100 steps of the exact config (checkpoint at/before step 100), then VERIFIES:
#   (a) a finite loss was logged;           (d) SUSTAINED throughput (steps/wall) >= 0.0395/s;
#   (b) a checkpoint file exists;            (e) the fa_invariant backbone-forward count is
#   (c) it strict-reloads (eval load chain);     NINE per batch (K=8 + 1) - no extra frame pass.
# Exits nonzero on ANY miss.
#
# GPU-GATED (plan §6): needs an explicit Yixun C1 go and the frozen threshold.
# ============================================================================
set -o pipefail    # verbatim (records-referenced, review finding 2)
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
WORKTREE="$(readlink -f .)"

# --- $1 = FROZEN MIN_FREE_MB (required; refuse TBD / non-numeric / absent). NB: the refuse
# messages must NOT interpolate $1/$2 literally - under `set -u` that expansion is itself an
# 'unbound variable' trip (exit 1) precisely when the arg is absent (r2 LOW). Reference the
# already-guarded ${1:-}/${2:-} captures / plain words instead. ---
MIN_FREE_MB="${1:-}"
[ -n "$MIN_FREE_MB" ] || { echo "usage: c1_smoke.sh <FROZEN_MIN_FREE_MB> <EXTERNAL_LOG_DIR>  (arg 1 must be the FROZEN numeric threshold)"; exit 3; }
case "$MIN_FREE_MB" in
  ''|*[!0-9]*)
    echo "REFUSING: MIN_FREE_MB='${MIN_FREE_MB}' is not a frozen numeric value (TBD / non-numeric /"
    echo "absent are all refused). The C1 fit probe must FREEZE  peak+4096  first, then pass it here."
    exit 3;;
esac

# --- $2 = EXTERNAL log dir (required; must be OUTSIDE the worktree) ---
LOGDIR="${2:-}"
[ -n "$LOGDIR" ] || { echo "usage: c1_smoke.sh <FROZEN_MIN_FREE_MB> <EXTERNAL_LOG_DIR>  (arg 2: external log dir required)"; exit 3; }
mkdir -p "$LOGDIR" 2>/dev/null || { echo "cannot create log dir '${LOGDIR}' - abort"; exit 3; }
LOGDIR_ABS="$(readlink -f "$LOGDIR")"
case "${LOGDIR_ABS}/" in
  "${WORKTREE}"/*)
    echo "REFUSING: log dir '${LOGDIR_ABS}' is INSIDE the worktree - it would dirty the clean-tree gate. Pass an EXTERNAL directory."
    exit 3;;
esac

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${LOGDIR_ABS}/exp09_${TS}_c1_smoke.log"
VERIFY_JSON="${LOGDIR_ABS}/exp09_${TS}_c1_smoke_logverify.json"
COUNT_JSON="${LOGDIR_ABS}/exp09_${TS}_c1_smoke_fwdcount.json"
SCRATCH="${LOGDIR_ABS}/c1_smoke_run_${TS}"
DATASET="src/configs/dataset_configs/AR/train/acousticroom_train.json"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp-09 C1 SMOKE (100 steps) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - MIN_FREE_MB=${MIN_FREE_MB} - log -> ${LOGDIR_ABS} ==="

# --- per-GPU FREE-VRAM gate on the FROZEN value (fail-CLOSED on query errors) ---
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < frozen ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- pin gate (fail-closed). EXPECT_PACKAGE_SHA / EXPECT_EXP09_SHA pin the accepted package and
# exp-09 worktree HEADs to exact SHAs at records freeze without editing the gate (r2 blocker 1);
# the exp-09 HEAD pin is REQUIRED on this blessed path (absent it, the gate refuses). ---
EXPECT_PACKAGE_SHA="${EXPECT_PACKAGE_SHA:-}"
EXPECT_EXP09_SHA="${EXPECT_EXP09_SHA:-}"
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs_exp09.py" \
  ${EXPECT_PACKAGE_SHA:+--expect-package-sha "$EXPECT_PACKAGE_SHA"} \
  ${EXPECT_EXP09_SHA:+--expect-exp09-sha "$EXPECT_EXP09_SHA"} \
  || { echo "PIN GATE FAILED - abort"; exit 1; }

# --- 100 steps of the exact config; checkpoint AT/BEFORE step 100 ---
MB=32; ACC=1; SMOKE_STEPS=100; CKPT_EVERY=100
echo "--- smoke: ${SMOKE_STEPS} steps, ckpt every ${CKPT_EVERY}, rung ${MB}x2x${ACC} eff64 SyncBN ---"
# MONOTONIC high-resolution timer (r3 blocker): bash $SECONDS truncates BOTH readings to whole
# seconds, so its integer delta can UNDER-count elapsed and false-pass the rate at the boundary.
# time.monotonic() is immune to NTP/wall-clock steps during the run and is sub-second; the delta
# is passed FRACTIONAL and the verifier CEILs it (rounds elapsed up / rate down = fail-closed).
SMOKE_START="$(python -c 'import time; print(repr(time.monotonic()))')"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_exp09.json" \
  --dataset-config "$DATASET" \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps "$SMOKE_STEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every "$CKPT_EVERY" \
  --name exp09_c1_smoke --experiment-name exp09_c1_smoke --save-dir "$SCRATCH"
rc=$?
SMOKE_WALL="$(python -c "import time; print(time.monotonic() - ${SMOKE_START})")"
[ "$rc" -eq 0 ] || { echo "SMOKE FAIL: training exited rc=${rc}"; exit 1; }
[ -n "$SMOKE_WALL" ] || { echo "SMOKE FAIL: could not measure monotonic wall time"; exit 1; }
echo "--- smoke wall time: ${SMOKE_WALL}s (monotonic, fractional) for ${SMOKE_STEPS} DECLARED steps -> sustained gate (verifier reconciles observed vs declared + ceils elapsed) ---"

FAIL=0
# --- (a)+(d) finite loss + OBSERVED-step reconciliation + SUSTAINED throughput (declared steps /
# CEIL'd monotonic wall time), not the max tick (r2 blocker 2 / r3). The verifier parses the log's
# OBSERVED furthest step and FAILS if it != the DECLARED ${SMOKE_STEPS} (an interrupted / swallowed
# -KeyboardInterrupt run short of max-steps is caught; r3 LOW). ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/forward_counter_probe.py" verify-log \
  --log "$LOG" --min-steps-per-s 0.0395 \
  --sustained-steps "$SMOKE_STEPS" --sustained-wall-s "$SMOKE_WALL" --out "$VERIFY_JSON" \
  || { echo "SMOKE FAIL: finite-loss / SUSTAINED-throughput verification failed"; FAIL=1; }

# --- (b) checkpoint exists (step <= 100) ---
CKPT="$(find "$SCRATCH" -name '*.ckpt' 2>/dev/null | head -1)"
if [ -n "$CKPT" ]; then
  echo "checkpoint: ${CKPT}"
else
  echo "SMOKE FAIL: no checkpoint written under ${SCRATCH}"; FAIL=1
fi

# --- (c) strict reload via the EXISTING eval load chain (eval_FLAC.check_load_integrity) ---
if [ -n "$CKPT" ]; then
  HF_HUB_OFFLINE=1 python - "$CKPT" "${EXPDIR}/FLAC_AR_exp09.json" <<'PY' || { echo "SMOKE FAIL: strict reload failed"; FAIL=1; }
import json, sys
import torch
from eval_FLAC import check_load_integrity          # the existing load-chain helper
from src.models import create_model_from_config
ckpt_path, cfg_path = sys.argv[1], sys.argv[2]
model_config = json.load(open(cfg_path))
training_config = model_config.get("training", {}) or {}
ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"]
for k in list(state_dict):                            # eval_FLAC remap (diffusion.* -> *)
    if k.startswith("diffusion."):
        state_dict[k.replace("diffusion.", "", 1)] = state_dict.pop(k)
if training_config.get("use_ema", False) and any(k.startswith("diffusion_ema.ema_model.") for k in state_dict):
    for k in list(state_dict):
        if k.startswith("diffusion_ema.ema_model."):
            state_dict[k.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(k)
model = create_model_from_config(model_config)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
check_load_integrity(missing, unexpected)             # raises on any real mismatch
print("strict reload OK: no unexpected/mismatched keys")
PY
fi

# --- (e) forward-counter evidence: NINE backbone calls per batch through the REAL backbone
# (K=8 = AcousticRooms max_context; the count is structural, so a fabricated K=8 batch through
# the real FLAC_AR_exp09.json backbone is valid evidence - review: parametrize by K, pin 9) ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/forward_counter_probe.py" count \
  --config "${EXPDIR}/FLAC_AR_exp09.json" --k 8 --expect 9 --out "$COUNT_JSON" \
  || { echo "SMOKE FAIL: forward-counter evidence != 9 (K=8 context + 1 source)"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then echo "=== C1 SMOKE FAILED (see above) - log ${LOG} ==="; exit 1; fi
echo "=== C1 SMOKE PASSED: finite loss, ckpt+strict-reload, SUSTAINED throughput >= 0.0395 steps/s, 9 backbone calls/batch - log ${LOG} ==="
exit 0
