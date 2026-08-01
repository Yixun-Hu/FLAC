#!/usr/bin/env bash
# ============================================================================
# bf_resume_launch_guardtests.sh - guard-branch exercise for bf_resume_launch.sh
# (same shell-pragmatic form as exp_09's f_arm_launch_guardtests.sh).
#
# Drives the REAL launcher through every fail-closed branch plus the two VALID
# lineage modes in *dry-fail* mode: MIN_FREE_MB is set impossibly high so the
# per-GPU VRAM gate aborts AFTER all config/sha/lineage gates have executed and
# BEFORE train.py is ever invoked. No training is launched and no GPU is used.
#
# Safety rules honoured here:
#   * the real 40k anchor is only ever READ (sha256sum / torch.load) - never
#     copied, moved, written or deleted;
#   * synthetic checkpoints are tiny PL-shaped dicts in a mktemp dir;
#   * the RESTART cases need files inside outputs_FLAC/exp10_BF - that directory
#     is created ONLY if it does not already exist, only the exact files created
#     here are removed, and the whole block is SKIPPED if the namespace already
#     exists (a live exp_10 run may own it);
#   * the sha-pin cases temporarily swap ${EXPDIR10}/bf40k_anchor.sha256 (a
#     worklog file, not the anchor). The original is backed up, restored
#     immediately after the block AND from the EXIT trap, the restoration is
#     verified, and the original digest is echoed into this log so it can be
#     rewritten by hand if the script is hard-killed mid-block.
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR10="worklog/worklog_yixun/exp_10_fa_scratch_resume_claude"
LAUNCHER="${EXPDIR10}/bf_resume_launch.sh"
BF_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
ANCHOR="outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt"
PIN_FILE="${EXPDIR10}/bf40k_anchor.sha256"
SAVEDIR="outputs_FLAC/exp10_BF"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR10}/fa_scratch_resume_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort every valid case

exec > >(tee -a "$LOG") 2>&1
echo "=== bf_resume_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ] || { echo "launcher not found - abort"; exit 3; }
[ -f "$ANCHOR" ]   || { echo "real anchor not found: ${ANCHOR} - abort"; exit 3; }
[ -f "$PIN_FILE" ] || { echo "pin file not found: ${PIN_FILE} - abort"; exit 3; }

TMP="$(mktemp -d)"
PIN_BACKUP="${TMP}/bf40k_anchor.sha256.orig"
cp "$PIN_FILE" "$PIN_BACKUP"
PIN_ORIG="$(awk 'NR==1{print $1}' "$PIN_FILE")"
echo "pin file digest (for manual recovery if this script is hard-killed): ${PIN_ORIG}"

LOGS_BEFORE="$(ls "${EXPDIR10}" | grep '_train\.log$' | sort || true)"

restore_pin() {
  [ -f "$PIN_BACKUP" ] || return 0
  cp -f "$PIN_BACKUP" "$PIN_FILE"
  if cmp -s "$PIN_BACKUP" "$PIN_FILE"; then
    echo "  pin file restored OK (${PIN_ORIG})"
  else
    echo "  !!! PIN FILE NOT RESTORED - rewrite ${PIN_FILE} with: ${PIN_ORIG}"
  fi
}

cleanup() {
  restore_pin
  rm -rf "$TMP"
  local now removed
  now="$(ls "${EXPDIR10}" | grep '_train\.log$' | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR10}/${f}" && echo "  rm ${f}"; done
  fi
}
trap cleanup EXIT

# --- synthetic checkpoints (PL-2.1 shaped, a few KB each; the fa ones embed the
# --- REAL FLAC_AR_BF.json so the launcher's parsed-object equality holds) ---
TMP="$TMP" BF_CONFIG="$BF_CONFIG" python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import json, os, torch
tmp = os.environ["TMP"]
cfg = json.load(open(os.environ["BF_CONFIG"]))

def ck(step, model_config, cleared=False):
    return {
        "epoch": step // 4444, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [{"state": {} if cleared else {0: {"step": torch.tensor(float(step)),
                                                               "exp_avg": torch.ones(2),
                                                               "exp_avg_sq": torch.ones(2)}},
                              "param_groups": [{"lr": 4.902903378454601e-05, "initial_lr": 5e-05,
                                                "betas": (0.9, 0.999), "weight_decay": 1e-3,
                                                "params": [0]}]}],
        "lr_schedulers": [{"last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [4.902903378454601e-05], "base_lrs": [5e-05]}],
        "model_config": model_config,
    }

# outside-the-anchor-path stand-ins (temp dir)
torch.save(ck(40000, cfg), os.path.join(tmp, "notanchor-step=40000.ckpt"))
torch.save(ck(42500, cfg), os.path.join(tmp, "outside-step=42500.ckpt"))
print("synthetic checkpoints written to", tmp)
PY

NOTANCHOR="${TMP}/notanchor-step=40000.ckpt"
OUTSIDE="${TMP}/outside-step=42500.ckpt"

PASS=0; FAIL=0
# case <label> <want_rc> <want_substrings, '|||'-separated> -- <ENV=VAL>...
case_run() {
  local label="$1" want_rc="$2" want_msgs="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local out rc ok=1 m rest missing=""
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  [ "$rc" = "$want_rc" ] || ok=0
  rest="$want_msgs"
  while [ -n "$rest" ]; do
    m="${rest%%|||*}"
    if [ "$rest" = "$m" ]; then rest=""; else rest="${rest#*|||}"; fi
    case "$out" in *"$m"*) ;; *) ok=0; missing="${missing}
      missing substring: ${m}";; esac
  done
  if [ "$ok" = "1" ]; then
    PASS=$((PASS+1)); printf 'PASS  rc=%-3s %s\n' "$rc" "$label"
  else
    FAIL=$((FAIL+1))
    printf 'FAIL  rc=%s (want %s) %s%s\n' "$rc" "$want_rc" "$label" "$missing"
    echo "$out" | sed 's/^/      | /' | tail -12
  fi
}

echo "--- A. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- CONDA_DEFAULT_ENV=rir2rir
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- CHECKPOINT_EVERY=abc
case_run "CHECKPOINT_EVERY=-5" 2 "CHECKPOINT_EVERY must be a positive integer" -- CHECKPOINT_EVERY=-5
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- MAXSTEPS=abc
case_run "MAXSTEPS=0" 2 "MAXSTEPS must be > 0" -- MAXSTEPS=0
case_run "EXPECTED_STEP=abc" 2 "EXPECTED_STEP must be a positive integer" -- EXPECTED_STEP=abc RESUME_CKPT="$ANCHOR"

echo "--- B. resume requirement ---"
case_run "RESUME_CKPT unset" 2 "RESUME_CKPT is REQUIRED"
case_run "RESUME_CKPT missing file" 2 "RESUME_CKPT not found" -- RESUME_CKPT=/nope.ckpt

echo "--- C. lineage-mode path gates ---"
case_run "INITIAL with a non-anchor step-40000 ckpt rejected (wrong path, right step)" 2 \
  "INITIAL launch requires RESUME_CKPT to BE the B-F 40k anchor" -- RESUME_CKPT="$NOTANCHOR"
case_run "EXPECTED_STEP=40000 explicit + non-anchor path rejected" 2 \
  "INITIAL launch requires RESUME_CKPT to BE the B-F 40k anchor" -- EXPECTED_STEP=40000 RESUME_CKPT="$NOTANCHOR"
case_run "EXPECTED_STEP=39000 (pre-anchor restart) rejected" 2 \
  "is below the B-F anchor step 40000" -- EXPECTED_STEP=39000 RESUME_CKPT="$NOTANCHOR"
case_run "RESTART from outside the exp10 namespace rejected" 2 \
  "checkpoint written by this run, i.e. one inside" -- EXPECTED_STEP=42500 RESUME_CKPT="$OUTSIDE"
case_run "INITIAL MAXSTEPS == 40000 rejected" 2 \
  "must exceed the anchor's 40000 steps" -- MAXSTEPS=40000 RESUME_CKPT="$ANCHOR"
case_run "INITIAL MAXSTEPS < 40000 rejected" 2 \
  "must exceed the anchor's 40000 steps" -- MAXSTEPS=30000 RESUME_CKPT="$ANCHOR"
case_run "MB=8 ACC=8 rung rejected" 2 "only the BN-compliant rung MB=32 ACC=1" -- MB=8 ACC=8 RESUME_CKPT="$ANCHOR"

echo "--- D. anchor sha-256 pin (pin file temporarily swapped; anchor untouched) ---"
printf '%s\n' "0000000000000000000000000000000000000000000000000000000000000000" > "$PIN_FILE"
case_run "sha mismatch (wrong pin) rejected" 2 "ANCHOR SHA-256 MISMATCH" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
printf '%s\n' "not-a-digest" > "$PIN_FILE"
case_run "malformed pin file rejected" 2 "does not contain a 64-hex digest" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
rm -f "$PIN_FILE"
case_run "missing pin file rejected" 2 "sha256 pin file missing" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
restore_pin

echo "--- E. exp10 namespace / RESTART cases (synthetic ckpts in ${SAVEDIR}) ---"
if [ -e "$SAVEDIR" ]; then
  echo "SKIP  ${SAVEDIR} already exists - refusing to touch a possibly-live run's directory."
  echo "      (verify it is stale and remove it yourself, then re-run this exercise)"
  echo "SKIP  cases: RESTART accept / vanilla-config rejection / stripped-optimizer rejection / MAXSTEPS<=EXPECTED_STEP"
else
  MAKE_NS=1 TMP="$TMP" SAVEDIR="$SAVEDIR" BF_CONFIG="$BF_CONFIG" python3 - <<'PY' || { echo "could not build namespace checkpoints - abort"; exit 3; }
import copy, json, os, torch
savedir = os.environ["SAVEDIR"]
cfg = json.load(open(os.environ["BF_CONFIG"]))
vanilla = copy.deepcopy(cfg)
vanilla["training"]["cond_method"] = "vanilla"
vanilla["training"].pop("frame_avg_angles", None)
def ck(step, model_config, cleared=False):
    return {
        "epoch": 10, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [{"state": {} if cleared else {0: {"step": torch.tensor(float(step)),
                                                               "exp_avg": torch.ones(2),
                                                               "exp_avg_sq": torch.ones(2)}},
                              "param_groups": [{"lr": 4.9e-05, "initial_lr": 5e-05,
                                                "betas": (0.9, 0.999), "weight_decay": 1e-3,
                                                "params": [0]}]}],
        "lr_schedulers": [{"last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [4.9e-05], "base_lrs": [5e-05]}],
        "model_config": model_config,
    }
os.makedirs(savedir, exist_ok=False)
torch.save(ck(42500, cfg), os.path.join(savedir, "guardtest-restart-step=42500.ckpt"))
torch.save(ck(42500, vanilla), os.path.join(savedir, "guardtest-vanillacfg-step=42500.ckpt"))
torch.save(ck(42500, cfg, cleared=True), os.path.join(savedir, "guardtest-stripped-step=42500.ckpt"))
print("synthetic RESTART checkpoints written to", savedir)
PY
  NS_OK="${SAVEDIR}/guardtest-restart-step=42500.ckpt"
  NS_VANILLA="${SAVEDIR}/guardtest-vanillacfg-step=42500.ckpt"
  NS_STRIPPED="${SAVEDIR}/guardtest-stripped-step=42500.ckpt"

  case_run "RESTART MAXSTEPS == EXPECTED_STEP rejected" 2 "must exceed the resume step 42500" \
    -- EXPECTED_STEP=42500 MAXSTEPS=42500 RESUME_CKPT="$NS_OK"
  case_run "RESTART with a wrong EXPECTED_STEP rejected" 2 "global_step 42500 != expected 45000" \
    -- EXPECTED_STEP=45000 MAXSTEPS=67500 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "vanilla-config ckpt rejected (not the fa lineage)" 2 "expected 'fa_invariant'" \
    -- EXPECTED_STEP=42500 MAXSTEPS=67500 RESUME_CKPT="$NS_VANILLA" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "optimizer-stripped ckpt rejected (exp_10 is warm-only)" 2 "optimizer state is CLEARED" \
    -- EXPECTED_STEP=42500 MAXSTEPS=67500 RESUME_CKPT="$NS_STRIPPED" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "VALID RESTART (dry-fail at the VRAM gate)" 2 \
    "mode=RESTART|||lineage OK for mode RESTART|||identity: --name FLAC_exp10_BF --experiment-name exp10_BF --save-dir outputs_FLAC/exp10_BF|||recipe: 32x2x1 eff64 seed42 -> 67500 | ckpt-every 1250|||refusing to launch" \
    -- EXPECTED_STEP=42500 MAXSTEPS=67500 CHECKPOINT_EVERY=1250 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"

  echo "--- targeted cleanup of ${SAVEDIR} ---"
  for f in "$NS_OK" "$NS_VANILLA" "$NS_STRIPPED"; do
    [ -f "$f" ] && rm -f "$f" && echo "  rm ${f}"
  done
  if rmdir "$SAVEDIR" 2>/dev/null; then
    echo "  rmdir ${SAVEDIR} (was empty)"
  else
    echo "  KEPT ${SAVEDIR} - not empty (something else wrote into it); left untouched:"
    ls -la "$SAVEDIR" | sed 's/^/    /'
  fi
fi

echo "--- F. the VALID INITIAL launch (real anchor, real pin, dry-fail at the VRAM gate) ---"
case_run "INITIAL accept: sha OK + fa lineage OK + identity + defaults" 2 \
  "mode=INITIAL|||anchor sha256 OK|||cond_method=fa_invariant|||lineage OK for mode INITIAL|||identity: --name FLAC_exp10_BF --experiment-name exp10_BF --save-dir outputs_FLAC/exp10_BF|||recipe: 32x2x1 eff64 seed42 -> 67500 | ckpt-every 2500|||config contract OK|||refusing to launch" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- anchor integrity re-check (must equal the pin; read-only throughout) ---"
NOW_SHA="$(sha256sum "$ANCHOR" | awk '{print $1}')"
if [ "$NOW_SHA" = "$PIN_ORIG" ]; then
  PASS=$((PASS+1)); echo "PASS  anchor unchanged (${NOW_SHA})"
else
  FAIL=$((FAIL+1)); echo "FAIL  ANCHOR SHA CHANGED: ${NOW_SHA} != ${PIN_ORIG}"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
