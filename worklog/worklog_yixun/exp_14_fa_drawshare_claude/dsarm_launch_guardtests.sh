#!/usr/bin/env bash
# ============================================================================
# dsarm_launch_guardtests.sh - guard-branch exercise for dsarm_launch.sh
# (same shell-pragmatic form as exp_10's bf_resume_launch_guardtests.sh and
# exp_13's dtail_launch_guardtests.sh).
#
# Seats this round: main session claude-fable-5 (xhigh) per the model-change
# flag; code by the Opus 5 Coder seat (SOP §Roles).
#
# Drives the REAL launcher through every fail-closed branch plus the VALID
# launches in *dry-fail* mode: a gate is forced to abort AFTER the gates under
# test have executed and BEFORE train.py (or the expensive DINOv3 pin audit) is
# ever reached. Two dry-fail stops are used:
#   * MIN_FREE_MB=99000000     - the per-GPU VRAM gate; the LATE stop, after the
#     config contract, the df floor and the torch.load resume-lineage check.
#     Used for the "valid launch" and resume cases.
#   * MIN_FREE_DISK_MB=99999999 - the df floor; the EARLY stop, right after the
#     config contract. Used where only the contract matters.
# No training is launched and no GPU work is done (the VRAM gate only queries
# nvidia-smi).
#
# The df floor is ALSO a gate under test (case E). Its documented bypass for
# guard-testing is MIN_FREE_DISK_MB=1; the real launch must never set it.
#
# Safety rules honoured here:
#   * synthetic checkpoints are tiny PL-shaped dicts in a mktemp dir;
#   * the namespace cases need files inside outputs_FLAC/exp14_DSPA - that
#     directory is created ONLY if it does not already exist, only the exact
#     files created here are removed, and the whole block is SKIPPED if the
#     namespace already exists (a live exp_14 run may own it);
#   * the config-contract cases temporarily swap the ARM config files
#     ${EXPDIR14}/FLAC_AR_BF_DS{PA,CS3}.json - worklog files, never a checkpoint.
#     Originals are backed up, restored immediately after each block AND from the
#     EXIT trap, restoration is verified, and the original digests are echoed
#     into this log so they can be recovered by hand if the script is hard-killed
#     mid-block. exp_07's FLAC_AR_BF.json is READ ONLY and never written.
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_14_fa_drawshare_claude/dsarm_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR14="worklog/worklog_yixun/exp_14_fa_drawshare_claude"
LAUNCHER="${EXPDIR14}/dsarm_launch.sh"
DSPA_CONFIG="${EXPDIR14}/FLAC_AR_BF_DSPA.json"
DSCS3_CONFIG="${EXPDIR14}/FLAC_AR_BF_DSCS3.json"
BF_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
SAVEDIR_DSPA="outputs_FLAC/exp14_DSPA"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR14}/fa_drawshare_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort (late stop)
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (early stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== dsarm_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]     || { echo "launcher not found - abort"; exit 3; }
[ -f "$DSPA_CONFIG" ]  || { echo "DSPA config not found: ${DSPA_CONFIG} - abort"; exit 3; }
[ -f "$DSCS3_CONFIG" ] || { echo "DSCS3 config not found: ${DSCS3_CONFIG} - abort"; exit 3; }
[ -f "$BF_CONFIG" ]    || { echo "BF reference not found: ${BF_CONFIG} - abort"; exit 3; }

TMP="$(mktemp -d)"
cp "$DSPA_CONFIG"  "${TMP}/FLAC_AR_BF_DSPA.json.orig"
cp "$DSCS3_CONFIG" "${TMP}/FLAC_AR_BF_DSCS3.json.orig"
DSPA_ORIG_SHA="$(sha256sum "$DSPA_CONFIG" | awk '{print $1}')"
DSCS3_ORIG_SHA="$(sha256sum "$DSCS3_CONFIG" | awk '{print $1}')"
BF_ORIG_SHA="$(sha256sum "$BF_CONFIG" | awk '{print $1}')"
echo "DSPA  config sha256 (manual recovery if hard-killed): ${DSPA_ORIG_SHA}"
echo "DSCS3 config sha256 (manual recovery if hard-killed): ${DSCS3_ORIG_SHA}"
echo "BF    config sha256 (read-only throughout)          : ${BF_ORIG_SHA}"

LOGS_BEFORE="$(ls "${EXPDIR14}" | grep '_train\.log$' | sort || true)"

restore_cfgs() {
  local ok=1
  for pair in "${TMP}/FLAC_AR_BF_DSPA.json.orig:${DSPA_CONFIG}" \
              "${TMP}/FLAC_AR_BF_DSCS3.json.orig:${DSCS3_CONFIG}"; do
    local src="${pair%%:*}" dst="${pair##*:}"
    [ -f "$src" ] || continue
    cp -f "$src" "$dst"
    cmp -s "$src" "$dst" || { ok=0; echo "  !!! ${dst} NOT RESTORED - restore it from git"; }
  done
  [ "$ok" = "1" ] && echo "  arm configs restored OK"
}

cleanup() {
  restore_cfgs
  rm -rf "$TMP"
  local now removed
  now="$(ls "${EXPDIR14}" | grep '_train\.log$' | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR14}/${f}" && echo "  rm ${f}"; done
  fi
}
trap cleanup EXIT

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
    echo "$out" | sed 's/^/      | /' | tail -16
  fi
}

echo "--- A. arm selection ---"
case_run "ARM unset rejected" 2 "ARM must be exactly one of DSPA" --
case_run "ARM=DSPA2 (near-miss) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=DSPA2
case_run "ARM=dspa (wrong case) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=dspa
case_run "ARM=BF (another experiment's arm) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=BF

echo "--- B. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- ARM=DSPA CONDA_DEFAULT_ENV=rir2rir
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MAXSTEPS=abc
case_run "MAXSTEPS=0" 2 "MAXSTEPS must be > 0" -- ARM=DSPA MAXSTEPS=0
case_run "MAXSTEPS=40000.5 (non-integer) rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MAXSTEPS=40000.5
case_run "MAXSTEPS=-40000 rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MAXSTEPS=-40000
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- ARM=DSPA CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- ARM=DSPA CHECKPOINT_EVERY=abc
# unset / empty are NOT errors: `${VAR:-default}` treats both as "use the default".
# Asserted rather than assumed, because the whole recipe hangs off those defaults.
case_run "MAXSTEPS + CHECKPOINT_EVERY empty -> the registered defaults (40000 / 2500)" 2 \
  "recipe: 32x2x1 eff64 seed42 -> 40000 | ckpt-every 2500|||refusing to launch" \
  -- ARM=DSPA MAXSTEPS= CHECKPOINT_EVERY= MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "MB=8 ACC=8 rung rejected (accumulation never feeds BN)" 2 \
  "only the BN-compliant rung MB=32 ACC=1" -- ARM=DSPA MB=8 ACC=8
case_run "MIN_FREE_DISK_MB=abc rejected" 2 "MIN_FREE_DISK_MB must be a positive integer" \
  -- ARM=DSPA MIN_FREE_DISK_MB=abc
case_run "PROBE with a cadence that never saves rejected" 2 "would not save" \
  -- ARM=DSPA MAXSTEPS=15 CHECKPOINT_EVERY=2500

echo "--- C. resume-declaration coupling ---"
case_run "RESUME_CKPT without EXPECTED_STEP rejected" 2 \
  "a resume must state the step it claims to be at" -- ARM=DSPA RESUME_CKPT=/nope.ckpt
case_run "EXPECTED_STEP without RESUME_CKPT rejected" 2 \
  "declares a RESTART but RESUME_CKPT is unset" -- ARM=DSPA EXPECTED_STEP=20000
case_run "EXPECTED_STEP=abc rejected" 2 "EXPECTED_STEP must be a non-negative integer" \
  -- ARM=DSPA EXPECTED_STEP=abc
case_run "RESUME_CKPT missing file rejected" 2 "RESUME_CKPT not found" \
  -- ARM=DSPA EXPECTED_STEP=20000 RESUME_CKPT=/nope.ckpt

echo "--- D. config contract (arm configs temporarily swapped) ---"
# D1: the cap is wrong for the arm
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = 64      # neither arm's cap
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D1 wrong cap (64) for DSPA rejected" 2 \
  "is 64, expected 32|||config contract FAILED" -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D1b: the two arms' caps swapped - each must reject the other's number
python3 - "$DSCS3_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = 32       # DS-PA's cap in the DS-CS3 file
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D1b DSCS3 carrying DSPA's cap rejected" 2 \
  "is 32, expected 96|||config contract FAILED" -- ARM=DSCS3 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D2: the key is missing entirely (the arm would inherit the module default 64)
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
del c["training"]["frame_avg_max_fwd_samples"]
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D2 missing cap key rejected (would silently inherit the module default)" 2 \
  "is None, expected 32|||config contract FAILED" -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D3: a bool that isinstance(_, int) would happily accept
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = True
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D3 boolean cap rejected" 2 \
  "config contract FAILED" -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D4: NON-CAP drift - the arm must equal BF everywhere else
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["cfg_dropout_prob"] = 0.2               # BF says 0.1
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D4 non-cap drift from BF rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D5: the conditioning method itself drifts (announcement 05)
python3 - "$DSCS3_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["cond_method"] = "vanilla"
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D5 cond_method drift to vanilla rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSCS3 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

# D6: the orbit shrinks - C4 is the registered orbit for both arms
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_angles"] = [0.0, 180.0]
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D6 non-C4 orbit rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

echo "--- E. df floor on the outputs volume (bypass for guard-testing: MIN_FREE_DISK_MB=1) ---"
case_run "df floor rejects an impossible free-space requirement" 2 \
  "free disk|||refusing to launch" -- ARM=DSPA MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
case_run "df floor bypass lets the run reach the VRAM gate" 2 \
  "config contract OK (DSPA)|||refusing to launch" \
  -- ARM=DSPA MIN_FREE_DISK_MB=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- F. RESTART lineage (synthetic ckpts in ${SAVEDIR_DSPA}) ---"
if [ -e "$SAVEDIR_DSPA" ]; then
  echo "SKIP  ${SAVEDIR_DSPA} already exists - refusing to touch a possibly-live run's directory."
  echo "      (verify it is stale and remove it yourself, then re-run this exercise)"
  echo "SKIP  cases: RESTART accept / embedded-config mismatch / wrong step / outside-namespace"
else
  TMP="$TMP" SAVEDIR="$SAVEDIR_DSPA" DSPA_CONFIG="$DSPA_CONFIG" DSCS3_CONFIG="$DSCS3_CONFIG" \
  python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import json, os, torch
savedir = os.environ["SAVEDIR"]; tmp = os.environ["TMP"]
dspa = json.load(open(os.environ["DSPA_CONFIG"]))
dscs3 = json.load(open(os.environ["DSCS3_CONFIG"]))
bf_like = json.loads(json.dumps(dspa)); del bf_like["training"]["frame_avg_max_fwd_samples"]

def ck(step, model_config):
    return {
        "epoch": 4, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [{"state": {0: {"step": torch.tensor(float(step)),
                                            "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}},
                              "param_groups": [{"lr": 4.79e-05, "initial_lr": 5e-05,
                                                "betas": (0.9, 0.999), "weight_decay": 1e-3,
                                                "params": [0]}]}],
        "lr_schedulers": [{"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99, "final_lr": 0.0,
                           "base_lrs": [5e-05], "last_epoch": step, "_step_count": step + 1,
                           "_last_lr": [4.79e-05]}],
        "model_config": model_config,
    }

os.makedirs(savedir, exist_ok=False)
torch.save(ck(20000, dspa),    os.path.join(savedir, "guardtest-ok-step=20000.ckpt"))
torch.save(ck(20000, dscs3),   os.path.join(savedir, "guardtest-othercap-step=20000.ckpt"))
torch.save(ck(20000, bf_like), os.path.join(savedir, "guardtest-nocapkey-step=20000.ckpt"))
torch.save(ck(20000, dspa),    os.path.join(tmp, "guardtest-outside-step=20000.ckpt"))
print("synthetic RESTART checkpoints written to", savedir, "and", tmp)
PY
  NS_OK="${SAVEDIR_DSPA}/guardtest-ok-step=20000.ckpt"
  NS_OTHER="${SAVEDIR_DSPA}/guardtest-othercap-step=20000.ckpt"
  NS_NOKEY="${SAVEDIR_DSPA}/guardtest-nocapkey-step=20000.ckpt"
  OUTSIDE="${TMP}/guardtest-outside-step=20000.ckpt"

  case_run "F1 RESTART from outside this arm's namespace rejected" 2 \
    "may only resume a checkpoint written by THIS arm" \
    -- ARM=DSPA EXPECTED_STEP=20000 RESUME_CKPT="$OUTSIDE"
  case_run "F2 RESTART MAXSTEPS <= EXPECTED_STEP rejected" 2 "must exceed the resume step 20000" \
    -- ARM=DSPA EXPECTED_STEP=20000 MAXSTEPS=20000 RESUME_CKPT="$NS_OK"
  case_run "F3 RESTART with a wrong EXPECTED_STEP rejected" 2 "global_step 20000 != expected 22500" \
    -- ARM=DSPA EXPECTED_STEP=22500 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "F4 RESTART on a ckpt embedding the OTHER arm's cap rejected" 2 \
    "embedded model_config != worklog/worklog_yixun/exp_14_fa_drawshare_claude/FLAC_AR_BF_DSPA.json|||training.frame_avg_max_fwd_samples=96|||resume-lineage check FAILED" \
    -- ARM=DSPA EXPECTED_STEP=20000 RESUME_CKPT="$NS_OTHER" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "F5 RESTART on a ckpt with NO cap key rejected" 2 \
    "training.frame_avg_max_fwd_samples='<absent>'|||resume-lineage check FAILED" \
    -- ARM=DSPA EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOKEY" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "F6 VALID RESTART (dry-fail at the VRAM gate)" 2 \
    "mode=RESTART|||config contract OK (DSPA)|||resume lineage OK|||identity: --name FLAC_exp14_DSPA --experiment-name exp14_DSPA --save-dir outputs_FLAC/exp14_DSPA|||refusing to launch" \
    -- ARM=DSPA EXPECTED_STEP=20000 MAXSTEPS=40000 RESUME_CKPT="$NS_OK" MIN_FREE_MB="$DRYFAIL_MIN_FREE"

  echo "--- targeted cleanup of ${SAVEDIR_DSPA} ---"
  for f in "$NS_OK" "$NS_OTHER" "$NS_NOKEY"; do
    [ -f "$f" ] && rm -f "$f" && echo "  rm ${f}"
  done
  if rmdir "$SAVEDIR_DSPA" 2>/dev/null; then
    echo "  rmdir ${SAVEDIR_DSPA} (was empty)"
  else
    echo "  KEPT ${SAVEDIR_DSPA} - not empty (something else wrote into it); left untouched:"
    ls -la "$SAVEDIR_DSPA" | sed 's/^/    /'
  fi
fi

echo "--- G. the VALID launches: both arms + their declared chunk plans ---"
case_run "G1 DSPA accept: identity, recipe, 1/3 per-angle chunk plan" 2 \
  "exp_14 fa_drawshare DSPA (SCRATCH) DDP+SyncBN|||mode=SCRATCH|||identity: --name FLAC_exp14_DSPA --experiment-name exp14_DSPA --save-dir outputs_FLAC/exp14_DSPA|||recipe: 32x2x1 eff64 seed42 -> 40000 | ckpt-every 2500|||arm config: worklog/worklog_yixun/exp_14_fa_drawshare_claude/FLAC_AR_BF_DSPA.json (single delta vs worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json: training.frame_avg_max_fwd_samples=32)|||config contract OK (DSPA)|||CHUNK PLAN (announcement 06): cap=32 per-rank micro-batch=32 orbit=4 angles (3 rotated)|||angles_per_chunk = max(1, 32 // 32) = 1|||rotated-angle chunks = [1, 1, 1] (3 conditioner forward(s) of [32, 32, 32] samples)|||shared-angle count = 1/3|||per-angle draws (the July path)|||probe=0|||refusing to launch" \
  -- ARM=DSPA MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "G2 DSCS3 accept: identity, recipe, 3/3 shared chunk plan + the cap-96 VRAM note" 2 \
  "exp_14 fa_drawshare DSCS3 (SCRATCH) DDP+SyncBN|||mode=SCRATCH|||identity: --name FLAC_exp14_DSCS3 --experiment-name exp14_DSCS3 --save-dir outputs_FLAC/exp14_DSCS3|||recipe: 32x2x1 eff64 seed42 -> 40000 | ckpt-every 2500|||config contract OK (DSCS3)|||CHUNK PLAN (announcement 06): cap=96 per-rank micro-batch=32 orbit=4 angles (3 rotated)|||angles_per_chunk = max(1, 96 // 32) = 3|||rotated-angle chunks = [3] (1 conditioner forward(s) of [96] samples)|||shared-angle count = 3/3|||3 angles share one RoPE draw|||floor is NOT requalified for this arm|||refusing to launch" \
  -- ARM=DSCS3 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "G3 PROBE mode accepted with a cadence that does save" 2 \
  "probe=1|||recipe: 32x2x1 eff64 seed42 -> 15 | ckpt-every 5|||config contract OK (DSPA)|||refusing to launch" \
  -- ARM=DSPA MAXSTEPS=15 CHECKPOINT_EVERY=5 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- config integrity re-check ---"
NOW_DSPA="$(sha256sum "$DSPA_CONFIG" | awk '{print $1}')"
NOW_DSCS3="$(sha256sum "$DSCS3_CONFIG" | awk '{print $1}')"
NOW_BF="$(sha256sum "$BF_CONFIG" | awk '{print $1}')"
for pair in "DSPA:${NOW_DSPA}:${DSPA_ORIG_SHA}" "DSCS3:${NOW_DSCS3}:${DSCS3_ORIG_SHA}" \
            "BF:${NOW_BF}:${BF_ORIG_SHA}"; do
  label="${pair%%:*}"; rest="${pair#*:}"; now="${rest%%:*}"; want="${rest##*:}"
  if [ "$now" = "$want" ]; then
    PASS=$((PASS+1)); echo "PASS  ${label} config unchanged (${now})"
  else
    FAIL=$((FAIL+1)); echo "FAIL  ${label} CONFIG CHANGED: ${now} != ${want}"
  fi
done

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
