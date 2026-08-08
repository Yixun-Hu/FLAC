#!/usr/bin/env bash
# ============================================================================
# dtail_launch_guardtests.sh - guard-branch exercise for dtail_launch.sh
# (same shell-pragmatic form as exp_10's bf_resume_launch_guardtests.sh).
#
# Drives the REAL launcher through every fail-closed branch plus the VALID
# lineage modes in *dry-fail* mode: a gate is forced to abort AFTER the gates
# under test have executed and BEFORE train.py (or the 690 MB retune) is ever
# reached. Two dry-fail stops are used:
#   * MIN_FREE_MB=99000000  - the per-GPU VRAM gate; the late stop, after the
#     full torch.load lineage check. Used for the "valid launch" cases.
#   * MIN_FREE_DISK_MB=99999999 - the df floor; the early stop, before the
#     690 MB anchor torch.load. Used where only the pre-lineage gates matter.
# No training is launched, no GPU is used, and the retune tool never runs.
#
# The df floor is ALSO a gate under test (case E). Its documented bypass for
# guard-testing is MIN_FREE_DISK_MB=1; the real launch must never set it.
#
# Safety rules honoured here:
#   * the real 87.5k anchor is only ever READ (sha256sum / torch.load) - never
#     copied, moved, written or deleted;
#   * synthetic checkpoints are tiny PL-shaped dicts in a mktemp dir;
#   * the RESTART/retune cases need files inside outputs_FLAC/exp13_DT - that
#     directory is created ONLY if it does not already exist, only the exact
#     files created here are removed, and the whole block is SKIPPED if the
#     namespace already exists (a live exp_13 run may own it);
#   * the sha-pin cases temporarily swap ${EXPDIR13}/anchor_87500.sha256 and the
#     config cases temporarily swap ${EXPDIR13}/FLAC_AR_BVp1_dtail.json - both
#     are worklog files, not the anchor. Originals are backed up, restored
#     immediately after each block AND from the EXIT trap, restoration is
#     verified, and the original digests are echoed into this log so they can be
#     recovered by hand if the script is hard-killed mid-block.
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_13_decay_tail_claude/dtail_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR13="worklog/worklog_yixun/exp_13_decay_tail_claude"
LAUNCHER="${EXPDIR13}/dtail_launch.sh"
DTAIL_CONFIG="${EXPDIR13}/FLAC_AR_BVp1_dtail.json"
BVP1_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ANCHOR="outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt"
PIN_FILE="${EXPDIR13}/anchor_87500.sha256"
SAVEDIR="outputs_FLAC/exp13_DT"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR13}/decay_tail_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort (late stop)
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (early stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== dtail_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]     || { echo "launcher not found - abort"; exit 3; }
[ -f "$ANCHOR" ]       || { echo "real anchor not found: ${ANCHOR} - abort"; exit 3; }
[ -f "$PIN_FILE" ]     || { echo "pin file not found: ${PIN_FILE} - abort"; exit 3; }
[ -f "$DTAIL_CONFIG" ] || { echo "dtail config not found: ${DTAIL_CONFIG} - abort"; exit 3; }

TMP="$(mktemp -d)"
PIN_BACKUP="${TMP}/anchor_87500.sha256.orig"
CFG_BACKUP="${TMP}/FLAC_AR_BVp1_dtail.json.orig"
cp "$PIN_FILE" "$PIN_BACKUP"
cp "$DTAIL_CONFIG" "$CFG_BACKUP"
PIN_ORIG="$(awk 'NR==1{print $1}' "$PIN_FILE")"
CFG_ORIG_SHA="$(sha256sum "$DTAIL_CONFIG" | awk '{print $1}')"
echo "pin file digest    (manual recovery if hard-killed): ${PIN_ORIG}"
echo "dtail config sha256(manual recovery if hard-killed): ${CFG_ORIG_SHA}"

LOGS_BEFORE="$(ls "${EXPDIR13}" | grep '_train\.log$' | sort || true)"

restore_pin() {
  [ -f "$PIN_BACKUP" ] || return 0
  cp -f "$PIN_BACKUP" "$PIN_FILE"
  if cmp -s "$PIN_BACKUP" "$PIN_FILE"; then
    echo "  pin file restored OK (${PIN_ORIG})"
  else
    echo "  !!! PIN FILE NOT RESTORED - rewrite ${PIN_FILE} with: ${PIN_ORIG}"
  fi
}

restore_cfg() {
  [ -f "$CFG_BACKUP" ] || return 0
  cp -f "$CFG_BACKUP" "$DTAIL_CONFIG"
  if cmp -s "$CFG_BACKUP" "$DTAIL_CONFIG"; then
    echo "  dtail config restored OK (sha256 ${CFG_ORIG_SHA})"
  else
    echo "  !!! DTAIL CONFIG NOT RESTORED - restore ${DTAIL_CONFIG} from git (sha256 ${CFG_ORIG_SHA})"
  fi
}

cleanup() {
  restore_pin
  restore_cfg
  rm -rf "$TMP"
  local now removed
  now="$(ls "${EXPDIR13}" | grep '_train\.log$' | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR13}/${f}" && echo "  rm ${f}"; done
  fi
}
trap cleanup EXIT

# --- synthetic checkpoints outside the anchor path (PL-2.1 shaped, a few KB) ---
TMP="$TMP" BVP1_CONFIG="$BVP1_CONFIG" python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import json, os, torch
tmp = os.environ["TMP"]
cfg = json.load(open(os.environ["BVP1_CONFIG"]))

def ck(step, model_config, cleared=False):
    return {
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
        "lr_schedulers": [{"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
                           "base_lrs": [5e-05], "last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [4.794633014853842e-05]}],
        "model_config": model_config,
    }

torch.save(ck(87500, cfg), os.path.join(tmp, "notanchor-step=87500.ckpt"))
torch.save(ck(90000, cfg), os.path.join(tmp, "outside-step=90000.ckpt"))
print("synthetic checkpoints written to", tmp)
PY

NOTANCHOR="${TMP}/notanchor-step=87500.ckpt"
OUTSIDE="${TMP}/outside-step=90000.ckpt"

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
    echo "$out" | sed 's/^/      | /' | tail -14
  fi
}

echo "--- A. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- CONDA_DEFAULT_ENV=rir2rir
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- CHECKPOINT_EVERY=abc
case_run "CHECKPOINT_EVERY=-5" 2 "CHECKPOINT_EVERY must be a positive integer" -- CHECKPOINT_EVERY=-5
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- MAXSTEPS=abc
case_run "MAXSTEPS=0" 2 "MAXSTEPS must be > 0" -- MAXSTEPS=0
case_run "MAXSTEPS=97500.5 (non-integer) rejected" 2 "MAXSTEPS must be a positive integer" -- MAXSTEPS=97500.5
case_run "EXPECTED_STEP=abc" 2 "EXPECTED_STEP must be a positive integer" -- EXPECTED_STEP=abc RESUME_CKPT="$ANCHOR"
case_run "MIN_FREE_DISK_MB=abc" 2 "MIN_FREE_DISK_MB must be a positive integer" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB=abc

echo "--- B. resume requirement ---"
case_run "RESUME_CKPT unset" 2 "RESUME_CKPT is REQUIRED" --
case_run "RESUME_CKPT missing file" 2 "RESUME_CKPT not found" -- RESUME_CKPT=/nope.ckpt

echo "--- C. lineage-mode path gates ---"
case_run "INITIAL with a non-anchor step-87500 ckpt rejected (wrong path, right step)" 2 \
  "INITIAL launch requires RESUME_CKPT to BE the P1 87.5k anchor" -- RESUME_CKPT="$NOTANCHOR"
case_run "EXPECTED_STEP=87500 explicit + non-anchor path rejected" 2 \
  "INITIAL launch requires RESUME_CKPT to BE the P1 87.5k anchor" -- EXPECTED_STEP=87500 RESUME_CKPT="$NOTANCHOR"
case_run "EXPECTED_STEP=80000 (pre-anchor restart) rejected" 2 \
  "is below the P1 anchor step 87500" -- EXPECTED_STEP=80000 RESUME_CKPT="$NOTANCHOR"
case_run "RESTART from outside the exp13 namespace rejected" 2 \
  "checkpoint written by this run, i.e. one inside" -- EXPECTED_STEP=90000 RESUME_CKPT="$OUTSIDE"
case_run "INITIAL MAXSTEPS == 87500 rejected" 2 \
  "must exceed the anchor's 87500 steps" -- MAXSTEPS=87500 RESUME_CKPT="$ANCHOR"
case_run "INITIAL MAXSTEPS < 87500 rejected" 2 \
  "must exceed the anchor's 87500 steps" -- MAXSTEPS=60000 RESUME_CKPT="$ANCHOR"
case_run "MB=8 ACC=8 rung rejected" 2 "only the BN-compliant rung MB=32 ACC=1" -- MB=8 ACC=8 RESUME_CKPT="$ANCHOR"
case_run "PROBE with a cadence that never saves rejected" 2 \
  "would not save" -- MAXSTEPS=87510 CHECKPOINT_EVERY=1250 RESUME_CKPT="$ANCHOR"

echo "--- D. exp_13 scheduler-block asserts (dtail config temporarily swapped) ---"
# D1 POSITIVE (byte-exact): revert the scheduler block to BVp1's non-decaying values
DTAIL_CONFIG="$DTAIL_CONFIG" python3 - <<'PY'
import json, os
p = os.environ["DTAIL_CONFIG"]; c = json.load(open(p))
c["training"]["optimizer_configs"]["diffusion"]["scheduler"]["config"].update(
    {"inv_gamma": 1000000, "power": 0.5})
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D1 scheduler block reverted to BVp1 (1e6/0.5) rejected" 2 \
  "!= registered|||exp_13 config gate FAILED" -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# D2 POSITIVE (byte-exact): a plausible-but-wrong warmup
DTAIL_CONFIG="$DTAIL_CONFIG" python3 - <<'PY'
import json, os
p = os.environ["DTAIL_CONFIG"]; c = json.load(open(p))
c["training"]["optimizer_configs"]["diffusion"]["scheduler"]["config"]["warmup"] = 0.98
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D2 warmup 0.98 (not the preserved 0.99) rejected" 2 \
  "!= registered|||exp_13 config gate FAILED" -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# D3 POSITIVE: an extra key inside the scheduler config
DTAIL_CONFIG="$DTAIL_CONFIG" python3 - <<'PY'
import json, os
p = os.environ["DTAIL_CONFIG"]; c = json.load(open(p))
c["training"]["optimizer_configs"]["diffusion"]["scheduler"]["config"]["last_epoch"] = 87500
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D3 extra key in the scheduler config rejected" 2 \
  "!= registered|||exp_13 config gate FAILED" -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# D4 the final_lr_ratio trap (review B5: not an InverseLR kwarg -> TypeError at runtime)
DTAIL_CONFIG="$DTAIL_CONFIG" python3 - <<'PY'
import json, os
p = os.environ["DTAIL_CONFIG"]; c = json.load(open(p))
c["training"]["final_lr_ratio"] = 0.25          # outside the scheduler block on purpose
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D4 stray final_lr_ratio key rejected" 2 \
  "final_lr_ratio' is not an InverseLR kwarg|||exp_13 config gate FAILED" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# D5 NEGATIVE: dtail must equal BVp1 everywhere OUTSIDE the scheduler block
DTAIL_CONFIG="$DTAIL_CONFIG" python3 - <<'PY'
import json, os
p = os.environ["DTAIL_CONFIG"]; c = json.load(open(p))
c["training"]["cfg_dropout_prob"] = 0.2         # BVp1 says 0.1
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D5 non-scheduler drift from BVp1 rejected" 2 \
  "somewhere OTHER than the scheduler block|||exp_13 config gate FAILED" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

echo "--- E. df floor on the outputs volume (bypass for guard-testing: MIN_FREE_DISK_MB=1) ---"
case_run "df floor rejects an impossible free-space requirement" 2 \
  "free disk|||refusing to launch" -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
case_run "df floor bypass (MIN_FREE_DISK_MB=1) lets the run reach the sha pin" 2 \
  "anchor sha256 OK|||exp_13 config gate OK" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- F. anchor sha-256 pin (pin file temporarily swapped; anchor untouched) ---"
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

echo "--- G. exp13 namespace: RESTART + retune-destination cases (synthetic ckpts in ${SAVEDIR}) ---"
if [ -e "$SAVEDIR" ]; then
  echo "SKIP  ${SAVEDIR} already exists - refusing to touch a possibly-live run's directory."
  echo "      (verify it is stale and remove it yourself, then re-run this exercise)"
  echo "SKIP  cases: RESTART accept / BVp1-embedded rejection / stripped-optimizer rejection /"
  echo "SKIP         MAXSTEPS<=EXPECTED_STEP / wrong EXPECTED_STEP / retune-refusal / RETUNE_FORCE"
else
  TMP="$TMP" SAVEDIR="$SAVEDIR" DTAIL_CONFIG="$DTAIL_CONFIG" BVP1_CONFIG="$BVP1_CONFIG" \
  python3 - <<'PY' || { echo "could not build namespace checkpoints - abort"; exit 3; }
import json, os, torch
savedir = os.environ["SAVEDIR"]
dtail = json.load(open(os.environ["DTAIL_CONFIG"]))
bvp1 = json.load(open(os.environ["BVP1_CONFIG"]))

def ck(step, model_config, cleared=False):
    return {
        "epoch": 20, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [{"state": {} if cleared else {0: {"step": torch.tensor(float(step)),
                                                               "exp_avg": torch.ones(2),
                                                               "exp_avg_sq": torch.ones(2)}},
                              "param_groups": [{"lr": 1.2611275964391691e-05, "initial_lr": 5e-05,
                                                "betas": (0.9, 0.999), "weight_decay": 1e-3,
                                                "params": [0]}]}],
        "lr_schedulers": [{"inv_gamma": 30000, "power": 1.0, "warmup": 0.99, "final_lr": 0.0,
                           "base_lrs": [5e-05], "last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [1.2611275964391691e-05]}],
        "model_config": model_config,
    }

os.makedirs(savedir, exist_ok=False)
torch.save(ck(90000, dtail), os.path.join(savedir, "guardtest-restart-step=90000.ckpt"))
torch.save(ck(90000, bvp1), os.path.join(savedir, "guardtest-bvp1cfg-step=90000.ckpt"))
torch.save(ck(90000, dtail, cleared=True), os.path.join(savedir, "guardtest-stripped-step=90000.ckpt"))
# a stand-in for an already-built retuned copy (content irrelevant: the launcher's
# pre-check is an existence test that must fire before anything expensive runs)
open(os.path.join(savedir, "retuned_from_87500.ckpt"), "w").write("guardtest placeholder\n")
print("synthetic RESTART checkpoints + retuned placeholder written to", savedir)
PY
  NS_OK="${SAVEDIR}/guardtest-restart-step=90000.ckpt"
  NS_BVP1="${SAVEDIR}/guardtest-bvp1cfg-step=90000.ckpt"
  NS_STRIPPED="${SAVEDIR}/guardtest-stripped-step=90000.ckpt"
  NS_RETUNED="${SAVEDIR}/retuned_from_87500.ckpt"

  case_run "RESTART MAXSTEPS == EXPECTED_STEP rejected" 2 "must exceed the resume step 90000" \
    -- EXPECTED_STEP=90000 MAXSTEPS=90000 RESUME_CKPT="$NS_OK"
  case_run "RESTART with a wrong EXPECTED_STEP rejected" 2 "global_step 90000 != expected 92500" \
    -- EXPECTED_STEP=92500 MAXSTEPS=97500 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "RESTART on a BVp1-embedded ckpt rejected (exp_13 ckpts embed the dtail config)" 2 \
    "embedded model_config != worklog/worklog_yixun/exp_13_decay_tail_claude/FLAC_AR_BVp1_dtail.json" \
    -- EXPECTED_STEP=90000 MAXSTEPS=97500 RESUME_CKPT="$NS_BVP1" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "optimizer-stripped ckpt rejected (exp_13 is warm-only)" 2 "optimizer state is CLEARED" \
    -- EXPECTED_STEP=90000 MAXSTEPS=97500 RESUME_CKPT="$NS_STRIPPED" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "VALID RESTART (dry-fail at the VRAM gate)" 2 \
    "mode=RESTART|||exp_13 config gate OK|||lineage OK for mode RESTART|||identity: --name FLAC_exp13_DT --experiment-name exp13_DT --save-dir outputs_FLAC/exp13_DT|||recipe: 32x2x1 eff64 seed42 -> 97500 | ckpt-every 1250|||inv_gamma=30000 power=1.0|||refusing to launch" \
    -- EXPECTED_STEP=90000 MAXSTEPS=97500 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"

  echo "--- G2. retune-destination guard (a retuned copy already exists) ---"
  case_run "INITIAL refuses to clobber an existing retuned copy" 2 \
    "the retuned checkpoint already exists|||RETUNE_FORCE=1" \
    -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "RETUNE_FORCE=1 gets past the retune pre-check (then dry-fails at VRAM)" 2 \
    "anchor sha256 OK|||lineage OK for mode INITIAL|||refusing to launch" \
    -- RESUME_CKPT="$ANCHOR" RETUNE_FORCE=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  if grep -q 'guardtest placeholder' "$NS_RETUNED" 2>/dev/null; then
    PASS=$((PASS+1)); echo "PASS  the retuned placeholder was never overwritten (the retune step is post-VRAM-gate)"
  else
    FAIL=$((FAIL+1)); echo "FAIL  the retuned placeholder was modified by a dry-fail run"
  fi

  echo "--- targeted cleanup of ${SAVEDIR} ---"
  for f in "$NS_OK" "$NS_BVP1" "$NS_STRIPPED" "$NS_RETUNED"; do
    [ -f "$f" ] && rm -f "$f" && echo "  rm ${f}"
  done
  if rmdir "$SAVEDIR" 2>/dev/null; then
    echo "  rmdir ${SAVEDIR} (was empty)"
  else
    echo "  KEPT ${SAVEDIR} - not empty (something else wrote into it); left untouched:"
    ls -la "$SAVEDIR" | sed 's/^/    /'
  fi
fi

echo "--- H. the VALID launches (real anchor, real pin, dry-fail at the VRAM gate) ---"
case_run "H1 INITIAL accept: sha OK + BVp1 lineage OK + identity + defaults" 2 \
  "mode=INITIAL|||exp_13 config gate OK|||anchor sha256 OK|||lineage OK for mode INITIAL|||identity: --name FLAC_exp13_DT --experiment-name exp13_DT --save-dir outputs_FLAC/exp13_DT|||recipe: 32x2x1 eff64 seed42 -> 97500 | ckpt-every 1250|||config contract OK|||probe=0|||InverseLR(inv_gamma=30000, power=1.0, warmup=0.99)|||refusing to launch" \
  -- RESUME_CKPT="$ANCHOR" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "H2 PROBE accept: MAXSTEPS 87515 / cadence 5 reaches the df gate with probe=1" 2 \
  "probe=1|||recipe: 32x2x1 eff64 seed42 -> 87515 | ckpt-every 5|||free disk" \
  -- MAXSTEPS=87515 CHECKPOINT_EVERY=5 RESUME_CKPT="$ANCHOR" MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"

echo "--- anchor + config integrity re-check (read-only throughout) ---"
NOW_SHA="$(sha256sum "$ANCHOR" | awk '{print $1}')"
if [ "$NOW_SHA" = "$PIN_ORIG" ]; then
  PASS=$((PASS+1)); echo "PASS  anchor unchanged (${NOW_SHA})"
else
  FAIL=$((FAIL+1)); echo "FAIL  ANCHOR SHA CHANGED: ${NOW_SHA} != ${PIN_ORIG}"
fi
NOW_CFG_SHA="$(sha256sum "$DTAIL_CONFIG" | awk '{print $1}')"
if [ "$NOW_CFG_SHA" = "$CFG_ORIG_SHA" ]; then
  PASS=$((PASS+1)); echo "PASS  dtail config restored (${NOW_CFG_SHA})"
else
  FAIL=$((FAIL+1)); echo "FAIL  DTAIL CONFIG NOT RESTORED: ${NOW_CFG_SHA} != ${CFG_ORIG_SHA}"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
