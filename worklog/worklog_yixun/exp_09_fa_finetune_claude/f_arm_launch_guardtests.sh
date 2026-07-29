#!/usr/bin/env bash
# ============================================================================
# f_arm_launch_guardtests.sh - guard-branch exercise for f_arm_launch.sh
# (codex r1: "add launcher branch tests"; shell-pragmatic form).
#
# Drives the real launcher through every fail-closed branch plus the three
# VALID identities in *dry-fail* mode: MIN_FREE_MB is set impossibly high so the
# per-GPU VRAM gate aborts the run AFTER all config/lineage/strip gates have
# executed and BEFORE train.py is ever invoked. No training is launched, no GPU
# is touched, and the 700 MB anchor is never read - the resume checkpoints are
# tiny synthetic PL-shaped dicts written into a temp dir.
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR09="worklog/worklog_yixun/exp_09_fa_finetune_claude"
LAUNCHER="${EXPDIR09}/f_arm_launch.sh"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR09}/fa_finetune_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort every valid case
RESET_DIR="outputs_FLAC/exp09_Fr"  # the F-reset save-dir the strip case writes into

exec > >(tee -a "$LOG") 2>&1
echo "=== f_arm_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

TMP="$(mktemp -d)"
# launcher logs created by this exercise are removed at the end (they are test noise)
LOGS_BEFORE="$(ls "${EXPDIR09}" | grep '_train\.log$' | sort || true)"

cleanup() {
  rm -rf "$TMP"
  local now removed
  now="$(ls "${EXPDIR09}" | grep '_train\.log$' | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR09}/${f}" && echo "  rm ${f}"; done
  fi
}
trap cleanup EXIT

# --- synthetic resume checkpoints (PL-2.1 shaped, a few KB each) ---
TMP="$TMP" python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import os, torch
tmp = os.environ["TMP"]
def ck(step=87500, cleared=False, weights_only=False):
    d = {
        "epoch": 19, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [{"state": {} if cleared else {0: {"step": torch.tensor(float(step)),
                                                               "exp_avg": torch.ones(2),
                                                               "exp_avg_sq": torch.ones(2)}},
                              "param_groups": [{"lr": 4.794633014853842e-05, "initial_lr": 5e-05,
                                                "betas": (0.9, 0.999), "weight_decay": 1e-3,
                                                "params": [0]}]}],
        "lr_schedulers": [{"last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [4.794633014853842e-05], "base_lrs": [5e-05]}],
        "model_config": {"model_type": "diffusion_cond"},
    }
    if weights_only:
        del d["optimizer_states"]
    return d
torch.save(ck(), os.path.join(tmp, "epoch=19-step=87500.ckpt"))          # warm anchor stand-in
torch.save(ck(cleared=True), os.path.join(tmp, "stripped-step=87500.ckpt"))
torch.save(ck(step=12345), os.path.join(tmp, "wrongstep-step=12345.ckpt"))
torch.save(ck(weights_only=True), os.path.join(tmp, "weightsonly-step=87500.ckpt"))
print("synthetic checkpoints written to", tmp)
PY

ANCHOR="${TMP}/epoch=19-step=87500.ckpt"
STRIPPED="${TMP}/stripped-step=87500.ckpt"
WRONGSTEP="${TMP}/wrongstep-step=12345.ckpt"
WEIGHTSONLY="${TMP}/weightsonly-step=87500.ckpt"

PASS=0; FAIL=0
# case <label> <want_rc> <want_substring> -- <ENV=VAL>...
case_run() {
  local label="$1" want_rc="$2" want_msg="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  local ok=1
  [ "$rc" = "$want_rc" ] || ok=0
  case "$out" in *"$want_msg"*) ;; *) ok=0;; esac
  if [ "$ok" = "1" ]; then
    PASS=$((PASS+1)); printf 'PASS  rc=%-3s %s\n' "$rc" "$label"
  else
    FAIL=$((FAIL+1))
    printf 'FAIL  rc=%s (want %s) %s\n      wanted substring: %s\n' "$rc" "$want_rc" "$label" "$want_msg"
    echo "$out" | sed 's/^/      | /' | tail -12
  fi
}

echo "--- A. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- CONDA_DEFAULT_ENV=rir2rir
case_run "MODEL_CONFIG unset" 2 "MODEL_CONFIG must be exactly"
case_run "MODEL_CONFIG=FLAC_AR_BV.json (not an arm)" 2 "MODEL_CONFIG must be exactly" -- MODEL_CONFIG=FLAC_AR_BV.json
case_run "MODEL_CONFIG path traversal" 2 "MODEL_CONFIG must be exactly" -- MODEL_CONFIG=../../../src/configs/model_configs/FLAC/AR/FLAC_AR.json
case_run "OPT_RESET=2" 2 "OPT_RESET must be 0 or 1" -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=2
case_run "OPT_RESET_FORCE=2" 2 "OPT_RESET_FORCE must be 0 or 1" -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET_FORCE=2
case_run "RESET_LINEAGE=yes" 2 "RESET_LINEAGE must be 0 or 1" -- MODEL_CONFIG=FLAC_AR_BF.json RESET_LINEAGE=yes
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- MODEL_CONFIG=FLAC_AR_BF.json CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- MODEL_CONFIG=FLAC_AR_BF.json CHECKPOINT_EVERY=abc

echo "--- B. arm/variant scope ---"
case_run "V arm + OPT_RESET=1 rejected" 2 "not an approved treatment for the V control arm" \
  -- MODEL_CONFIG=FLAC_AR_BVp1.json OPT_RESET=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750
case_run "V arm + RESET_LINEAGE=1 rejected" 2 "meaningless on the V arm" \
  -- MODEL_CONFIG=FLAC_AR_BVp1.json RESET_LINEAGE=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750
case_run "OPT_RESET=1 + RESET_LINEAGE=1 contradictory" 2 "contradictory" \
  -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 RESET_LINEAGE=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750

echo "--- C. resume / budget requirements ---"
case_run "RESUME_CKPT unset" 2 "RESUME_CKPT is REQUIRED" -- MODEL_CONFIG=FLAC_AR_BF.json
case_run "RESUME_CKPT missing file" 2 "RESUME_CKPT not found" -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT=/nope.ckpt
case_run "MAXSTEPS unset" 2 "MAXSTEPS is REQUIRED" -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR"
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=abc
case_run "MAXSTEPS == anchor steps" 2 "must exceed the anchor's 87500 steps" -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=87500
case_run "MB=8 ACC=8 rung rejected" 2 "only the BN-compliant rung MB=32 ACC=1" -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 MB=8 ACC=8

echo "--- D. resume-lineage validation ---"
case_run "wrong-step ckpt rejected" 2 "global_step 12345 != expected 87500" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$WRONGSTEP" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "weights-only ckpt rejected" 2 "no 'optimizer_states' key" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$WEIGHTSONLY" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "stripped ckpt mislabelled as F-warm" 2 "relaunch with RESET_LINEAGE=1" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$STRIPPED" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "re-stripping an already-stripped ckpt rejected" 2 "ALREADY has cleared optimizer state" \
  -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 RESUME_CKPT="$STRIPPED" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "stripped ckpt on the V arm rejected" 2 "V control declared but the resume ckpt has CLEARED" \
  -- MODEL_CONFIG=FLAC_AR_BVp1.json RESUME_CKPT="$STRIPPED" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- E. the three VALID identities (dry-fail at the VRAM gate) ---"
case_run "F-warm identity" 2 "identity: --name FLAC_exp09_Fw --experiment-name exp09_Fw --save-dir outputs_FLAC/exp09_Fw" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 CHECKPOINT_EVERY=625 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "F-warm reaches the VRAM gate (all earlier gates passed)" 2 "refusing to launch" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 CHECKPOINT_EVERY=625 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "F-warm lineage accepted" 2 "lineage OK for variant Fw" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "F-warm cadence override echoed" 2 "ckpt-every 625" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 CHECKPOINT_EVERY=625 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "V identity" 2 "identity: --name FLAC_exp09_V --experiment-name exp09_V --save-dir outputs_FLAC/exp09_V" \
  -- MODEL_CONFIG=FLAC_AR_BVp1.json RESUME_CKPT="$ANCHOR" MAXSTEPS=97500 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "F-reset (continuation) identity via RESET_LINEAGE" 2 "identity: --name FLAC_exp09_Fr --experiment-name exp09_Fr --save-dir outputs_FLAC/exp09_Fr" \
  -- MODEL_CONFIG=FLAC_AR_BF.json RESET_LINEAGE=1 RESUME_CKPT="$STRIPPED" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- F. OPT_RESET strip path (synthetic anchor -> ${RESET_DIR}) ---"
if [ -e "$RESET_DIR" ]; then
  echo "SKIP  ${RESET_DIR} already exists - refusing to touch a real run's directory."
  echo "      (delete it manually if it is stale, then re-run this exercise)"
else
  case_run "OPT_RESET strips, verifies and switches to the copy" 2 "adam_state=CLEARED param_groups=KEPT" \
    -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 CHECKPOINT_EVERY=625 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "existing stripped copy is not silently overwritten" 2 "refusing to overwrite" \
    -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "OPT_RESET_FORCE=1 regenerates it" 2 "adam_state=CLEARED param_groups=KEPT" \
    -- MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 OPT_RESET_FORCE=1 RESUME_CKPT="$ANCHOR" MAXSTEPS=88750 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

  echo "--- post-conditions on the produced copy ---"
  RESET_CKPT="${RESET_DIR}/optreset_from_87500.ckpt" ANCHOR="$ANCHOR" python3 - <<'PY'
import os, torch
c = torch.load(os.environ["RESET_CKPT"], map_location="cpu", weights_only=False)
a = torch.load(os.environ["ANCHOR"], map_location="cpu", weights_only=False)
assert len(c["optimizer_states"]) == 1 and c["optimizer_states"][0]["state"] == {}
assert c["optimizer_states"][0]["param_groups"] == a["optimizer_states"][0]["param_groups"]
assert c["global_step"] == 87500 and c["lr_schedulers"][0]["last_epoch"] == 87500
assert a["optimizer_states"][0]["state"], "synthetic anchor was mutated"
print("  OK: entry kept, state cleared, param_groups identical to the anchor, anchor intact")
PY
  [ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
  rm -rf "$RESET_DIR" && echo "cleaned up ${RESET_DIR}"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
