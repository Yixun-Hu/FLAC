#!/usr/bin/env bash
# ============================================================================
# are_launch_guardtests.sh - guard-branch exercise for are_launch.sh
# (same shell-pragmatic form as exp_14's dsarm_launch_guardtests.sh, exp_13's
# dtail_launch_guardtests.sh and exp_10's bf_resume_launch_guardtests.sh).
#
# Seat: Opus 5 Coder (SOP §Roles).
#
# Drives the REAL launcher through every fail-closed branch plus the VALID
# launches in *dry-fail* mode: a gate is forced to abort AFTER the gates under
# test have executed and BEFORE train.py (or the expensive DINOv3 pin audit) is
# ever reached. Two dry-fail stops are used:
#   * MIN_FREE_MB=99000000     - the per-GPU VRAM gate; the LATE stop, after the
#     config contract, the admission-evidence gate, the df floor and the
#     torch.load resume-lineage check. Used for the "valid launch" cases.
#   * MIN_FREE_DISK_MB=99999999 - the df floor; the EARLY stop, right after the
#     config contract and the evidence gate. Used where only those matter.
# No training is launched and no GPU work is done (the VRAM gate only queries
# nvidia-smi).
#
# The df floor is ALSO a gate under test (case E). Its documented bypass for
# guard-testing is MIN_FREE_DISK_MB=1; the real launch must never set it.
# ALLOW_DIRTY_TREATMENT=1 is likewise guard-test-only: this branch is developed
# uncommitted, so a production-shaped FULL launch could not otherwise be
# exercised at all. Its own gate is tested in case H.
#
# Safety rules honoured here:
#   * synthetic checkpoints are tiny PL-shaped dicts in a mktemp dir;
#   * the namespace cases need files inside outputs_FLAC/exp16_AREV - that
#     directory is created ONLY if it does not already exist, only the exact
#     files created here are removed, and the whole block is SKIPPED if the
#     namespace already exists (a live exp_16 run may own it);
#   * the evidence cases write into worklog/.../evidence/ under the same rule:
#     SKIPPED wholesale if that directory already exists, so a real campaign's
#     stamped admission record can never be overwritten or deleted here;
#   * the config-contract cases temporarily swap ${EXPDIR16}/FLAC_AR_ARE.json -
#     a worklog file, never a checkpoint. The original is backed up, restored
#     immediately after each block AND from the EXIT trap, restoration is
#     verified, and the original digest is echoed into this log so it can be
#     recovered by hand if the script is hard-killed mid-block. exp_07's
#     FLAC_AR_BVp1.json is READ ONLY and never written.
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_16_are_port_claude/are_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR16="worklog/worklog_yixun/exp_16_are_port_claude"
LAUNCHER="${EXPDIR16}/are_launch.sh"
STAMPER="${EXPDIR16}/stamp_evidence.py"
ARE_CONFIG="${EXPDIR16}/FLAC_AR_ARE.json"
BASE_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
EVIDENCE_DIR="${EXPDIR16}/evidence"
SAVEDIR_AREV="outputs_FLAC/exp16_AREV"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR16}/are_port_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort (late stop)
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (early stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== are_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]    || { echo "launcher not found - abort"; exit 3; }
[ -f "$STAMPER" ]     || { echo "evidence stamper not found: ${STAMPER} - abort"; exit 3; }
[ -f "$ARE_CONFIG" ]  || { echo "ARE arm config not found: ${ARE_CONFIG} - abort"; exit 3; }
[ -f "$BASE_CONFIG" ] || { echo "BVp1 reference not found: ${BASE_CONFIG} - abort"; exit 3; }

TMP="$(mktemp -d)"
cp "$ARE_CONFIG" "${TMP}/FLAC_AR_ARE.json.orig"
ARE_ORIG_SHA="$(sha256sum "$ARE_CONFIG" | awk '{print $1}')"
BASE_ORIG_SHA="$(sha256sum "$BASE_CONFIG" | awk '{print $1}')"
echo "ARE  config sha256 (manual recovery if hard-killed): ${ARE_ORIG_SHA}"
echo "BVp1 config sha256 (read-only throughout)          : ${BASE_ORIG_SHA}"

LOGS_BEFORE="$(ls "${EXPDIR16}" | grep '_train\.log$' | sort || true)"
EVIDENCE_OWNED=0            # 1 once THIS script created the evidence dir

restore_cfgs() {
  local src="${TMP}/FLAC_AR_ARE.json.orig"
  [ -f "$src" ] || return 0
  cp -f "$src" "$ARE_CONFIG"
  if cmp -s "$src" "$ARE_CONFIG"; then
    echo "  arm config restored OK"
  else
    echo "  !!! ${ARE_CONFIG} NOT RESTORED - restore it from git"
  fi
}

cleanup_evidence() {
  # ONLY ever removes an evidence directory this script created itself.
  [ "$EVIDENCE_OWNED" = "1" ] || return 0
  rm -f "${EVIDENCE_DIR}/are_fit_AREV.json"
  if rmdir "$EVIDENCE_DIR" 2>/dev/null; then
    echo "  rmdir ${EVIDENCE_DIR} (created by this exercise, now empty)"
  else
    echo "  KEPT ${EVIDENCE_DIR} - not empty; left untouched:"
    ls -la "$EVIDENCE_DIR" | sed 's/^/    /'
  fi
}

cleanup() {
  restore_cfgs
  cleanup_evidence
  rm -rf "$TMP"
  rm -f "${EXPDIR16}/are_port_${TS}_guardtest_evidence_source.log"
  rm -rf "${EXPDIR16}/__pycache__"
  local now removed
  now="$(ls "${EXPDIR16}" | grep '_train\.log$' | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR16}/${f}" && echo "  rm ${f}"; done
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
    echo "$out" | sed 's/^/      | /' | tail -18
  fi
}

mutate() { # $1 = python snippet operating on `c` (the parsed ARE config)
  ARE_CONFIG="$ARE_CONFIG" python3 - "$1" <<'PY'
import json, os, sys
p = os.environ["ARE_CONFIG"]
c = json.load(open(p))
exec(sys.argv[1])
json.dump(c, open(p, "w"), indent=4)
PY
}

echo "--- A. arm + mode selection (both explicit, both fail-closed) ---"
case_run "ARM unset rejected" 2 "ARM must be exactly AREV" --
case_run "ARM=AREV2 (near-miss) rejected" 2 "ARM must be exactly AREV" -- ARM=AREV2
case_run "ARM=arev (wrong case) rejected" 2 "ARM must be exactly AREV" -- ARM=arev
case_run "ARM=DSPA (another experiment's arm) rejected" 2 "ARM must be exactly AREV" -- ARM=DSPA
case_run "ARM=AREFA (a Phase-2 arm with no config) rejected" 2 \
  "ARM must be exactly AREV|||Phases 2/3" -- ARM=AREFA
case_run "MODE unset rejected (never inferred from MAXSTEPS)" 2 \
  "MODE must be exactly one of PROBE|||never inferred from MAXSTEPS" -- ARM=AREV
case_run "MODE=probe (wrong case) rejected" 2 "MODE must be exactly one of PROBE" -- ARM=AREV MODE=probe
case_run "MODE=FIT (invented) rejected" 2 "MODE must be exactly one of PROBE" -- ARM=AREV MODE=FIT

echo "--- B. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- ARM=AREV MODE=FULL CONDA_DEFAULT_ENV=rir2rir
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- ARM=AREV MODE=FULL MAXSTEPS=abc
case_run "MAXSTEPS=0" 2 "MAXSTEPS must be > 0" -- ARM=AREV MODE=FULL MAXSTEPS=0
case_run "MAXSTEPS=40000.5 (non-integer) rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=AREV MODE=FULL MAXSTEPS=40000.5
case_run "MAXSTEPS=-40000 rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=AREV MODE=FULL MAXSTEPS=-40000
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- ARM=AREV MODE=FULL CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- ARM=AREV MODE=FULL CHECKPOINT_EVERY=abc
case_run "MB=8 ACC=8 rung rejected (accumulation never feeds BN)" 2 \
  "only the BN-compliant rung MB=32 ACC=1" -- ARM=AREV MODE=FULL MB=8 ACC=8
case_run "MIN_FREE_DISK_MB=abc rejected" 2 "MIN_FREE_DISK_MB must be a positive integer" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB=abc
case_run "a cadence that never saves is rejected in EVERY mode (readback is mandatory)" 2 \
  "would not save before MAXSTEPS" -- ARM=AREV MODE=PROBE MAXSTEPS=15 CHECKPOINT_EVERY=2500
case_run "FULL with a cadence that never saves rejected too" 2 \
  "would not save before MAXSTEPS" -- ARM=AREV MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=2500

echo "--- C. mode/resume coupling ---"
case_run "MODE=RESTART without RESUME_CKPT rejected" 2 \
  "MODE=RESTART requires RESUME_CKPT" -- ARM=AREV MODE=RESTART
case_run "MODE=RESTART without EXPECTED_STEP rejected" 2 \
  "MODE=RESTART requires EXPECTED_STEP > 0" -- ARM=AREV MODE=RESTART RESUME_CKPT=/nope.ckpt
case_run "MODE=RESTART with a missing file rejected" 2 "RESUME_CKPT not found" \
  -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT=/nope.ckpt
case_run "MODE=FULL with RESUME_CKPT rejected (use MODE=RESTART)" 2 \
  "is a from-scratch launch; RESUME_CKPT must be unset" -- ARM=AREV MODE=FULL RESUME_CKPT=/nope.ckpt
case_run "MODE=PROBE with EXPECTED_STEP rejected" 2 \
  "is a from-scratch launch; EXPECTED_STEP must be unset" -- ARM=AREV MODE=PROBE EXPECTED_STEP=20000
case_run "EXPECTED_STEP=abc rejected" 2 "EXPECTED_STEP must be a non-negative integer" \
  -- ARM=AREV MODE=RESTART EXPECTED_STEP=abc RESUME_CKPT=/nope.ckpt

echo "--- D. config contract (arm config temporarily swapped) ---"
mutate 'c["training"]["are_lambda"] = 0.5'
case_run "D1 wrong lambda (0.5) rejected" 2 \
  "is 0.5, expected 1.0|||config contract FAILED" -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'del c["training"]["are_lambda"]'
case_run "D2 missing lambda key rejected (an ARE arm that trains today's objective)" 2 \
  "is None, expected 1.0|||config contract FAILED" -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["are_lambda"] = True'
case_run "D3 boolean lambda rejected" 2 \
  "a typed float|||config contract FAILED" -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["are_lambda"] = 1'
case_run "D3b integer lambda 1 rejected (1 == 1.0 == True in python)" 2 \
  "is 1, expected 1.0|||a typed float|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'del c["training"]["are_anchor"]'
case_run "D4 missing anchor block rejected (no calibrated constants)" 2 \
  "training.are_anchor is None|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["are_anchor"]["typo"] = 1.0'
case_run "D5 extra anchor key rejected (a typo would be silently ignored)" 2 \
  "training.are_anchor keys|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'del c["training"]["are_anchor"]["delta_hat"]'
case_run "D6 missing delta_hat rejected" 2 \
  "training.are_anchor keys|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["are_anchor"]["a_g"] = 1'
case_run "D7 integer A_g rejected (a typed float is required)" 2 \
  "training.are_anchor.a_g is 1|||a typed float|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["cfg_dropout_prob"] = 0.2'
case_run "D8 non-ARE drift from BVp1 rejected" 2 \
  "somewhere OTHER than training.are_lambda / training.are_anchor|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["cond_method"] = "fa_invariant"'
case_run "D9 cond_method drift to fa_invariant rejected (ARE-FA is Phase 2)" 2 \
  "somewhere OTHER than training.are_lambda / training.are_anchor|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

mutate 'c["training"]["are_anchor"]["delta_hat"] = 3000.0'
case_run "D10 a delta_hat that pushes AR's longest path out of the kept frames rejected" 2 \
  "ARE anchor window too narrow|||config contract FAILED" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

echo "--- E. df floor on the outputs volume (bypass for guard-testing: MIN_FREE_DISK_MB=1) ---"
case_run "df floor rejects an impossible free-space requirement" 2 \
  "free disk|||refusing to launch" -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
case_run "df floor bypass lets a PROBE reach the VRAM gate" 2 \
  "config contract OK (AREV)|||refusing to launch" \
  -- ARM=AREV MODE=PROBE MIN_FREE_DISK_MB=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- F. ADMISSION EVIDENCE: FULL/RESTART are GATED ---"
if [ -e "$EVIDENCE_DIR" ]; then
  echo "SKIP  ${EVIDENCE_DIR} already exists - refusing to touch a real campaign's stamped evidence."
  echo "SKIP  cases: no-evidence rejection / stale SHA / mutated config / FAIL verdict /"
  echo "SKIP         unsourced log / valid FULL accept"
else
  # A PROBE needs no evidence: that is how evidence is earned.
  case_run "F0 PROBE needs no admission evidence" 2 \
    "admission evidence: NOT required in MODE=PROBE|||refusing to launch" \
    -- ARM=AREV MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "F1 FULL with NO evidence rejected" 2 \
    "missing evidence file worklog/worklog_yixun/exp_16_are_port_claude/evidence/are_fit_AREV.json|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"

  # from here on this script owns the evidence directory
  mkdir -p "$EVIDENCE_DIR" && EVIDENCE_OWNED=1
  EV_LOG="${EXPDIR16}/are_port_${TS}_guardtest_evidence_source.log"
  echo "synthetic evidence source log for the guard exercise" > "$EV_LOG"
  stamp() { # $1=kind $2=arm  [$3=verdict]
    python3 "$STAMPER" --kind "$1" --arm "$2" --verdict "${3:-PASS}" --log "$EV_LOG" \
      --notes "guard-test synthetic evidence (${TS})" --force >/dev/null || {
        echo "could not stamp ${1}/${2} - abort"; exit 3; }
  }

  stamp are_fit AREV
  case_run "F2 FULL with valid are_fit evidence accepted (dry-fail at VRAM)" 2 \
    "admission evidence for AREV (FULL)|||OK   are_fit (AREV)|||admission evidence OK|||refusing to launch" \
    -- ARM=AREV MODE=FULL MIN_FREE_MB="$DRYFAIL_MIN_FREE" ALLOW_DIRTY_TREATMENT=1

  # --- adversarial evidence: stale source SHA -------------------------------
  python3 - "${EVIDENCE_DIR}/are_fit_AREV.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["source_sha"] = "0" * 40                 # a commit that is not HEAD
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F3 stale source_sha rejected (a source change is a hard abort)" 2 \
    "source_sha 000000000000... != HEAD|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp are_fit AREV

  # --- adversarial evidence: the ARE code changed since stamping -------------
  python3 - "${EVIDENCE_DIR}/are_fit_AREV.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["treatment_sha256"] = "f" * 64
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F4 mismatched treatment_sha256 rejected (the ARE code changed)" 2 \
    "treatment_sha256 ffffffffffffffff... != current|||describes a different method|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp are_fit AREV

  # --- adversarial evidence: the arm config moved after stamping -------------
  # (in production this is exactly what a RE-CALIBRATION looks like: a different
  #  delta_hat / A_g is a different anchor, so the probe no longer describes it)
  python3 - "$ARE_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
json.dump(c, open(p, "w"), indent=2)          # same object, different bytes
PY
  case_run "F5 arm config re-serialised after stamping rejected (model_config_sha256)" 2 \
    "model_config_sha256|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  restore_cfgs
  stamp are_fit AREV

  # --- adversarial evidence: a recorded FAIL verdict -------------------------
  stamp are_fit AREV FAIL
  case_run "F6 FAIL-verdict evidence rejected (a failed gate is still evidence)" 2 \
    "verdict 'FAIL' != 'PASS'|||the gate did NOT pass|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp are_fit AREV

  # --- adversarial evidence: the verdict's log is gone ----------------------
  python3 - "${EVIDENCE_DIR}/are_fit_AREV.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["log"] = "worklog/worklog_yixun/exp_16_are_port_claude/does_not_exist.log"
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F7 unsourced verdict rejected (recorded log missing)" 2 \
    "does not exist -> the verdict is unsourced|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp are_fit AREV

  # --- adversarial evidence: the record is about a different lambda ----------
  python3 - "${EVIDENCE_DIR}/are_fit_AREV.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["are_lambda"] = 0.5
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F8 evidence about a different lambda rejected" 2 \
    "are_lambda 0.5 != 1.0|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp are_fit AREV

  # --- adversarial evidence: the calibration record or the frozen VAE moved --
  # (the anchor is a function of BOTH; nothing else in the record would notice)
  for FIELD in calibration_sha256 vae_sha256; do
    FIELD="$FIELD" python3 - "${EVIDENCE_DIR}/are_fit_AREV.json" <<'FUZZ'
import json, os, sys
p = sys.argv[1]; r = json.load(open(p))
r[os.environ["FIELD"]] = "a" * 64
json.dump(r, open(p, "w"), indent=2)
FUZZ
    case_run "F9 ${FIELD} mismatch rejected (a different anchor input)" 2 \
      "${FIELD} aaaaaaaaaaaaaaaa... != current|||ADMISSION EVIDENCE GATE FAILED" \
      -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
    stamp are_fit AREV
  done

  echo "--- H. dirty-treatment gate (ALLOW_DIRTY_TREATMENT is guard-test-only) ---"
  if [ -n "$(git status --porcelain -- src/data/are_anchor.py src/data/dataset.py src/data/utils.py src/training/diffusion.py src/training/factory.py train.py)" ]; then
    case_run "H1 FULL refuses a dirty treatment path without the bypass" 2 \
      "treatment paths are DIRTY|||does not describe what would run|||ADMISSION EVIDENCE GATE FAILED" \
      -- ARM=AREV MODE=FULL MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
    case_run "H2 the bypass is announced in the log when used" 2 \
      "GUARD-TESTING ONLY; never on a real launch|||admission evidence OK" \
      -- ARM=AREV MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  else
    echo "SKIP  H1/H2 - the treatment paths are clean in this checkout, so the dirty branch"
    echo "SKIP        cannot be exercised without dirtying tracked source (refused)."
  fi
fi

echo "--- G. RESTART lineage (synthetic ckpts in ${SAVEDIR_AREV}) ---"
if [ -e "$SAVEDIR_AREV" ]; then
  echo "SKIP  ${SAVEDIR_AREV} already exists - refusing to touch a possibly-live run's directory."
  echo "SKIP  cases: scratch-into-dirty-namespace / outside-namespace / wrong step / other lambda /"
  echo "SKIP         absent lambda / int-typed lambda / cleared optimizer / no param_groups /"
  echo "SKIP         no scheduler / accept"
else
  TMP="$TMP" SAVEDIR="$SAVEDIR_AREV" ARE_CONFIG="$ARE_CONFIG" \
  python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import json, os, torch
savedir = os.environ["SAVEDIR"]; tmp = os.environ["TMP"]
arm = json.load(open(os.environ["ARE_CONFIG"]))
otherlam = json.loads(json.dumps(arm)); otherlam["training"]["are_lambda"] = 0.5
nolam = json.loads(json.dumps(arm)); del nolam["training"]["are_lambda"]
intlam = json.loads(json.dumps(arm)); intlam["training"]["are_lambda"] = 1


def ck(step, model_config, cleared=False, groups=True, scheds=True):
    opt = {"state": {} if cleared else {0: {"step": torch.tensor(float(step)),
                                            "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}}}
    if groups:
        opt["param_groups"] = [{"lr": 4.79e-05, "initial_lr": 5e-05, "betas": (0.9, 0.999),
                                "weight_decay": 1e-3, "params": [0]}]
    d = {
        "epoch": 4, "global_step": step, "pytorch-lightning_version": "2.1.0",
        "state_dict": {"diffusion.w": torch.ones(2), "diffusion_ema.step": torch.tensor(step),
                       "diffusion_ema.ema_model.w": torch.zeros(2)},
        "loops": {}, "callbacks": {},
        "optimizer_states": [opt],
        "model_config": model_config,
    }
    if scheds:
        d["lr_schedulers"] = [{"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99,
                               "final_lr": 0.0, "base_lrs": [5e-05], "last_epoch": step,
                               "_step_count": step + 1, "_last_lr": [4.79e-05]}]
    return d


os.makedirs(savedir, exist_ok=False)
torch.save(ck(20000, arm),                 os.path.join(savedir, "guardtest-ok-step=20000.ckpt"))
torch.save(ck(20000, otherlam),            os.path.join(savedir, "guardtest-otherlam-step=20000.ckpt"))
torch.save(ck(20000, nolam),               os.path.join(savedir, "guardtest-nolamkey-step=20000.ckpt"))
torch.save(ck(20000, intlam),              os.path.join(savedir, "guardtest-intlam-step=20000.ckpt"))
torch.save(ck(20000, arm, cleared=True),   os.path.join(savedir, "guardtest-stripped-step=20000.ckpt"))
torch.save(ck(20000, arm, groups=False),   os.path.join(savedir, "guardtest-nogroups-step=20000.ckpt"))
torch.save(ck(20000, arm, scheds=False),   os.path.join(savedir, "guardtest-nosched-step=20000.ckpt"))
torch.save(ck(20000, arm),                 os.path.join(tmp, "guardtest-outside-step=20000.ckpt"))
print("synthetic RESTART checkpoints written to", savedir, "and", tmp)
PY
  NS_OK="${SAVEDIR_AREV}/guardtest-ok-step=20000.ckpt"
  NS_OTHER="${SAVEDIR_AREV}/guardtest-otherlam-step=20000.ckpt"
  NS_NOKEY="${SAVEDIR_AREV}/guardtest-nolamkey-step=20000.ckpt"
  NS_INT="${SAVEDIR_AREV}/guardtest-intlam-step=20000.ckpt"
  NS_STRIPPED="${SAVEDIR_AREV}/guardtest-stripped-step=20000.ckpt"
  NS_NOGROUPS="${SAVEDIR_AREV}/guardtest-nogroups-step=20000.ckpt"
  NS_NOSCHED="${SAVEDIR_AREV}/guardtest-nosched-step=20000.ckpt"
  OUTSIDE="${TMP}/guardtest-outside-step=20000.ckpt"
  EV_ARGS=(ALLOW_DIRTY_TREATMENT=1)
  [ "$EVIDENCE_OWNED" = "1" ] || EV_ARGS=()   # without evidence the gate stops earlier

  case_run "G0 MODE=FULL into a namespace that already holds checkpoints rejected" 2 \
    "refuses to launch into a namespace that already holds checkpoints|||use MODE=RESTART" \
    -- ARM=AREV MODE=FULL
  case_run "G1 RESTART from outside this arm's namespace rejected" 2 \
    "may only resume a checkpoint written by THIS arm's FULL run" \
    -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$OUTSIDE"
  case_run "G2 RESTART MAXSTEPS <= EXPECTED_STEP rejected" 2 "must exceed the resume step 20000" \
    -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 MAXSTEPS=20000 RESUME_CKPT="$NS_OK"

  if [ "$EVIDENCE_OWNED" = "1" ]; then
    case_run "G3 RESTART with a wrong EXPECTED_STEP rejected" 2 "global_step 20000 != expected 22500" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=22500 RESUME_CKPT="$NS_OK" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G4 RESTART on a ckpt embedding a DIFFERENT lambda rejected" 2 \
      "type-strict mismatch|||model_config.training.are_lambda: 0.5 != 1.0|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_OTHER" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G5 RESTART on a ckpt with NO lambda key rejected (today's objective)" 2 \
      "present in the config, absent from the checkpoint|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOKEY" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G6 RESTART on a ckpt whose lambda is int 1 rejected (TYPE-strict: 1 == 1.0 in python)" 2 \
      "type int != float|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_INT" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G7 optimizer-stripped ckpt rejected (exp_16's arm is warm-only)" 2 \
      "optimizer state is CLEARED|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_STRIPPED" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G8 ckpt with no param_groups rejected" 2 \
      "no 'param_groups'|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOGROUPS" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G9 ckpt with no lr_schedulers rejected" 2 \
      "no 'lr_schedulers'|||resume-lineage check FAILED" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOSCHED" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G10 VALID RESTART (evidence + full-state lineage, dry-fail at VRAM)" 2 \
      "mode=RESTART|||config contract OK (AREV)|||admission evidence OK|||resume lineage OK|||optimizer_state=FULL|||identity: --name FLAC_exp16_AREV --experiment-name exp16_AREV --save-dir outputs_FLAC/exp16_AREV|||refusing to launch" \
      -- ARM=AREV MODE=RESTART EXPECTED_STEP=20000 MAXSTEPS=40000 RESUME_CKPT="$NS_OK" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  else
    echo "SKIP  G3-G10 - a real evidence directory exists, so RESTART cannot reach the lineage gate here."
  fi

  echo "--- targeted cleanup of ${SAVEDIR_AREV} ---"
  for f in "$NS_OK" "$NS_OTHER" "$NS_NOKEY" "$NS_INT" "$NS_STRIPPED" "$NS_NOGROUPS" "$NS_NOSCHED"; do
    [ -f "$f" ] && rm -f "$f" && echo "  rm ${f}"
  done
  if rmdir "$SAVEDIR_AREV" 2>/dev/null; then
    echo "  rmdir ${SAVEDIR_AREV} (was empty)"
  else
    echo "  KEPT ${SAVEDIR_AREV} - not empty (something else wrote into it); left untouched:"
    ls -la "$SAVEDIR_AREV" | sed 's/^/    /'
  fi
fi

echo "--- I. the VALID PROBE launch: identity, defaults, anchor + worst-case disclosure ---"
case_run "I1 AREV PROBE accept: probe identity, 15/5 defaults, calibrated anchor, AR worst case" 2 \
  "exp_16 are_port AREV (PROBE) DDP+SyncBN|||identity: --name FLAC_exp16_AREV_probe --experiment-name exp16_AREV_probe --save-dir outputs_FLAC/exp16_AREV_probe|||recipe: 32x2x1 eff64 seed42 -> 15 | ckpt-every 5|||config contract OK (AREV)|||ANCHOR: delta_hat=0.0|||TIME BASE: fs=22050 sample_size=10240 hop=1024 -> 10 latent frames, keeping 0..2|||AUGMENTATION: the AR train loader|||WORST CASE (AR max 27.1047 m + max shift 10)|||needs 2 frame(s) <= 3 kept|||It is NOT requalified|||refusing to launch" \
  -- ARM=AREV MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "I2 a PROBE never writes into the production namespace" 2 \
  "--save-dir outputs_FLAC/exp16_AREV_probe|||refusing to launch" \
  -- ARM=AREV MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- config integrity re-check ---"
NOW_ARE="$(sha256sum "$ARE_CONFIG" | awk '{print $1}')"
NOW_BASE="$(sha256sum "$BASE_CONFIG" | awk '{print $1}')"
for pair in "ARE:${NOW_ARE}:${ARE_ORIG_SHA}" "BVp1:${NOW_BASE}:${BASE_ORIG_SHA}"; do
  label="${pair%%:*}"; rest="${pair#*:}"; now="${rest%%:*}"; want="${rest##*:}"
  if [ "$now" = "$want" ]; then
    PASS=$((PASS+1)); echo "PASS  ${label} config unchanged (${now})"
  else
    FAIL=$((FAIL+1)); echo "FAIL  ${label} CONFIG CHANGED: ${now} != ${want}"
  fi
done
if [ -d "outputs_FLAC/exp16_AREV_probe" ]; then
  FAIL=$((FAIL+1)); echo "FAIL  a probe namespace was created by a dry-fail run (nothing should have run)"
else
  PASS=$((PASS+1)); echo "PASS  no probe namespace was created (every case dry-failed before train.py)"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
