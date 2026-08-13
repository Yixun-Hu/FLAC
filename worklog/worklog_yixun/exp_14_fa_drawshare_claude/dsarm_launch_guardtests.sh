#!/usr/bin/env bash
# ============================================================================
# dsarm_launch_guardtests.sh - guard-branch exercise for dsarm_launch.sh
# (same shell-pragmatic form as exp_10's bf_resume_launch_guardtests.sh and
# exp_13's dtail_launch_guardtests.sh).  Rev 2 - r1 review findings 2/3/4.
#
# Seats this round: main session claude-fable-5 (xhigh) per the model-change
# flag; code by the Opus 5 Coder seat (SOP §Roles).
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
#   * the namespace cases need files inside outputs_FLAC/exp14_DSPA - that
#     directory is created ONLY if it does not already exist, only the exact
#     files created here are removed, and the whole block is SKIPPED if the
#     namespace already exists (a live exp_14 run may own it);
#   * the evidence cases write into worklog/.../evidence/ under the same rule:
#     SKIPPED wholesale if that directory already exists, so a real campaign's
#     stamped admission record can never be overwritten or deleted here;
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
STAMPER="${EXPDIR14}/stamp_evidence.py"
DSPA_CONFIG="${EXPDIR14}/FLAC_AR_BF_DSPA.json"
DSCS3_CONFIG="${EXPDIR14}/FLAC_AR_BF_DSCS3.json"
BF_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
EVIDENCE_DIR="${EXPDIR14}/evidence"
SAVEDIR_DSPA="outputs_FLAC/exp14_DSPA"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR14}/fa_drawshare_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the VRAM gate to abort (late stop)
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (early stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== dsarm_launch guard exercise (rev 2) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]     || { echo "launcher not found - abort"; exit 3; }
[ -f "$STAMPER" ]      || { echo "evidence stamper not found: ${STAMPER} - abort"; exit 3; }
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
EVIDENCE_OWNED=0            # 1 once THIS script created the evidence dir

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

cleanup_evidence() {
  # ONLY ever removes an evidence directory this script created itself.
  [ "$EVIDENCE_OWNED" = "1" ] || return 0
  rm -f "${EVIDENCE_DIR}/cap_fit_DSPA.json" "${EVIDENCE_DIR}/cap_fit_DSCS3.json" \
        "${EVIDENCE_DIR}/dspa_40k_audit_DSPA.json"
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
  # the synthetic log the stamped evidence pointed at, and any __pycache__ the
  # launcher's stamp_evidence import may have left in the worklog directory
  rm -f "${EXPDIR14}/fa_drawshare_${TS}_guardtest_evidence_source.log"
  rm -rf "${EXPDIR14}/__pycache__"
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
    echo "$out" | sed 's/^/      | /' | tail -18
  fi
}

echo "--- A. arm + mode selection (both explicit, both fail-closed) ---"
case_run "ARM unset rejected" 2 "ARM must be exactly one of DSPA" --
case_run "ARM=DSPA2 (near-miss) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=DSPA2
case_run "ARM=dspa (wrong case) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=dspa
case_run "ARM=BF (another experiment's arm) rejected" 2 "ARM must be exactly one of DSPA" -- ARM=BF
case_run "MODE unset rejected (never inferred from MAXSTEPS)" 2 \
  "MODE must be exactly one of PROBE|||never inferred from MAXSTEPS" -- ARM=DSPA
case_run "MODE=probe (wrong case) rejected" 2 "MODE must be exactly one of PROBE" -- ARM=DSPA MODE=probe
case_run "MODE=FIT (invented) rejected" 2 "MODE must be exactly one of PROBE" -- ARM=DSPA MODE=FIT

echo "--- B. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- ARM=DSPA MODE=FULL CONDA_DEFAULT_ENV=rir2rir
case_run "MAXSTEPS=abc" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MODE=FULL MAXSTEPS=abc
case_run "MAXSTEPS=0" 2 "MAXSTEPS must be > 0" -- ARM=DSPA MODE=FULL MAXSTEPS=0
case_run "MAXSTEPS=40000.5 (non-integer) rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MODE=FULL MAXSTEPS=40000.5
case_run "MAXSTEPS=-40000 rejected" 2 "MAXSTEPS must be a positive integer" -- ARM=DSPA MODE=FULL MAXSTEPS=-40000
case_run "CHECKPOINT_EVERY=0" 2 "CHECKPOINT_EVERY must be > 0" -- ARM=DSPA MODE=FULL CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc" 2 "CHECKPOINT_EVERY must be a positive integer" -- ARM=DSPA MODE=FULL CHECKPOINT_EVERY=abc
case_run "MB=8 ACC=8 rung rejected (accumulation never feeds BN)" 2 \
  "only the BN-compliant rung MB=32 ACC=1" -- ARM=DSPA MODE=FULL MB=8 ACC=8
case_run "MIN_FREE_DISK_MB=abc rejected" 2 "MIN_FREE_DISK_MB must be a positive integer" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB=abc
case_run "a cadence that never saves is rejected in EVERY mode (readback is mandatory)" 2 \
  "would not save before MAXSTEPS" -- ARM=DSPA MODE=PROBE MAXSTEPS=15 CHECKPOINT_EVERY=2500
case_run "FULL with a cadence that never saves rejected too" 2 \
  "would not save before MAXSTEPS" -- ARM=DSPA MODE=FULL MAXSTEPS=1000 CHECKPOINT_EVERY=2500

echo "--- C. mode/resume coupling ---"
case_run "MODE=RESTART without RESUME_CKPT rejected" 2 \
  "MODE=RESTART requires RESUME_CKPT" -- ARM=DSPA MODE=RESTART
case_run "MODE=RESTART without EXPECTED_STEP rejected" 2 \
  "MODE=RESTART requires EXPECTED_STEP > 0" -- ARM=DSPA MODE=RESTART RESUME_CKPT=/nope.ckpt
case_run "MODE=RESTART with a missing file rejected" 2 "RESUME_CKPT not found" \
  -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT=/nope.ckpt
case_run "MODE=FULL with RESUME_CKPT rejected (use MODE=RESTART)" 2 \
  "is a from-scratch launch; RESUME_CKPT must be unset" -- ARM=DSPA MODE=FULL RESUME_CKPT=/nope.ckpt
case_run "MODE=PROBE with EXPECTED_STEP rejected" 2 \
  "is a from-scratch launch; EXPECTED_STEP must be unset" -- ARM=DSPA MODE=PROBE EXPECTED_STEP=20000
case_run "EXPECTED_STEP=abc rejected" 2 "EXPECTED_STEP must be a non-negative integer" \
  -- ARM=DSPA MODE=RESTART EXPECTED_STEP=abc RESUME_CKPT=/nope.ckpt

echo "--- D. config contract (arm configs temporarily swapped) ---"
python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = 64      # neither arm's cap
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D1 wrong cap (64) for DSPA rejected" 2 \
  "is 64, expected 32|||config contract FAILED" -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSCS3_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = 32       # DS-PA's cap in the DS-CS3 file
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D1b DSCS3 carrying DSPA's cap rejected" 2 \
  "is 32, expected 96|||config contract FAILED" -- ARM=DSCS3 MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
del c["training"]["frame_avg_max_fwd_samples"]
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D2 missing cap key rejected (would silently inherit the module default)" 2 \
  "is None, expected 32|||config contract FAILED" -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = True
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D3 boolean cap rejected" 2 "config contract FAILED" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_max_fwd_samples"] = 32.0     # float 32 == int 32 in python
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D3b float cap 32.0 rejected (a typed int is required)" 2 \
  "a typed int|||config contract FAILED" -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["cfg_dropout_prob"] = 0.2               # BF says 0.1
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D4 non-cap drift from BF rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSCS3_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["cond_method"] = "vanilla"
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D5 cond_method drift to vanilla rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSCS3 MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c["training"]["frame_avg_angles"] = [0.0, 180.0]
json.dump(c, open(p, "w"), indent=4)
PY
case_run "D6 non-C4 orbit rejected" 2 \
  "somewhere OTHER than training.frame_avg_max_fwd_samples|||config contract FAILED" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfgs

echo "--- E. df floor on the outputs volume (bypass for guard-testing: MIN_FREE_DISK_MB=1) ---"
case_run "df floor rejects an impossible free-space requirement" 2 \
  "free disk|||refusing to launch" -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
case_run "df floor bypass lets a PROBE reach the VRAM gate" 2 \
  "config contract OK (DSPA)|||refusing to launch" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_DISK_MB=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- F. ADMISSION EVIDENCE (r1 review finding 3): FULL/RESTART are GATED ---"
if [ -e "$EVIDENCE_DIR" ]; then
  echo "SKIP  ${EVIDENCE_DIR} already exists - refusing to touch a real campaign's stamped evidence."
  echo "SKIP  cases: no-evidence rejection / DSCS3-without-audit / stale SHA / mutated config /"
  echo "SKIP         FAIL verdict / unsourced log / valid FULL accept for both arms"
else
  # A PROBE needs no evidence: that is how evidence is earned.
  case_run "F0 PROBE needs no admission evidence" 2 \
    "admission evidence: NOT required in MODE=PROBE|||refusing to launch" \
    -- ARM=DSPA MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  case_run "F1 FULL DSPA with NO evidence rejected" 2 \
    "missing evidence file worklog/worklog_yixun/exp_14_fa_drawshare_claude/evidence/cap_fit_DSPA.json|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  case_run "F2 FULL DSCS3 with NO evidence rejected (both records named)" 2 \
    "evidence/cap_fit_DSCS3.json|||evidence/dspa_40k_audit_DSPA.json|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSCS3 MODE=FULL MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"

  # from here on this script owns the evidence directory
  mkdir -p "$EVIDENCE_DIR" && EVIDENCE_OWNED=1
  EV_LOG="${EXPDIR14}/fa_drawshare_${TS}_guardtest_evidence_source.log"
  echo "synthetic evidence source log for the guard exercise" > "$EV_LOG"
  stamp() { # $1=kind $2=arm  [$3=verdict]
    python3 "$STAMPER" --kind "$1" --arm "$2" --verdict "${3:-PASS}" --log "$EV_LOG" \
      --notes "guard-test synthetic evidence (${TS})" --force >/dev/null || {
        echo "could not stamp ${1}/${2} - abort"; exit 3; }
  }

  stamp cap_fit DSPA
  case_run "F3 FULL DSPA with valid cap_fit evidence accepted (dry-fail at VRAM)" 2 \
    "admission evidence for DSPA (FULL)|||OK   cap_fit (DSPA)|||admission evidence OK|||refusing to launch" \
    -- ARM=DSPA MODE=FULL MIN_FREE_MB="$DRYFAIL_MIN_FREE" ALLOW_DIRTY_TREATMENT=1

  stamp cap_fit DSCS3
  case_run "F4 FULL DSCS3 with cap_fit but WITHOUT the DS-PA 40k audit rejected" 2 \
    "OK   cap_fit (DSCS3)|||FAIL dspa_40k_audit (DSPA)|||missing evidence file|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSCS3 MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"

  stamp dspa_40k_audit DSPA
  case_run "F5 FULL DSCS3 with BOTH records accepted (the sequencing gate discharged)" 2 \
    "OK   cap_fit (DSCS3)|||OK   dspa_40k_audit (DSPA)|||admission evidence OK|||refusing to launch" \
    -- ARM=DSCS3 MODE=FULL MIN_FREE_MB="$DRYFAIL_MIN_FREE" ALLOW_DIRTY_TREATMENT=1

  # --- adversarial evidence: stale source SHA -------------------------------
  python3 - "${EVIDENCE_DIR}/cap_fit_DSPA.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["source_sha"] = "0" * 40                 # a commit that is not HEAD
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F6 stale source_sha rejected (plan §5: a source change is a hard abort)" 2 \
    "source_sha 000000000000... != HEAD|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp cap_fit DSPA

  # --- adversarial evidence: the treatment code changed since stamping ------
  python3 - "${EVIDENCE_DIR}/cap_fit_DSPA.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["treatment_sha256"] = "f" * 64
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F7 mismatched treatment_sha256 rejected (cap-threading code changed)" 2 \
    "treatment_sha256 ffffffffffffffff... != current|||describes a different method|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp cap_fit DSPA

  # --- adversarial evidence: the arm config moved after stamping ------------
  python3 - "$DSPA_CONFIG" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
json.dump(c, open(p, "w"), indent=2)          # same object, different bytes
PY
  case_run "F8 arm config re-serialised after stamping rejected (model_config_sha256)" 2 \
    "model_config_sha256|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  restore_cfgs
  stamp cap_fit DSPA

  # --- adversarial evidence: a recorded FAIL verdict ------------------------
  stamp cap_fit DSPA FAIL
  case_run "F9 FAIL-verdict evidence rejected (a failed gate is still evidence)" 2 \
    "verdict 'FAIL' != 'PASS'|||the gate did NOT pass|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp cap_fit DSPA

  # --- adversarial evidence: the verdict's log is gone ----------------------
  python3 - "${EVIDENCE_DIR}/cap_fit_DSPA.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["log"] = "worklog/worklog_yixun/exp_14_fa_drawshare_claude/does_not_exist.log"
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F10 unsourced verdict rejected (recorded log missing)" 2 \
    "does not exist -> the verdict is unsourced|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp cap_fit DSPA

  # --- adversarial evidence: wrong kind under the right filename -----------
  python3 - "${EVIDENCE_DIR}/dspa_40k_audit_DSPA.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
r["kind"] = "cap_fit"                      # a fit probe passed off as the 40k audit
json.dump(r, open(p, "w"), indent=2)
PY
  case_run "F11 a cap_fit record passed off as the 40k audit rejected" 2 \
    "kind 'cap_fit' != 'dspa_40k_audit'|||ADMISSION EVIDENCE GATE FAILED" \
    -- ARM=DSCS3 MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
  stamp dspa_40k_audit DSPA

  echo "--- H. dirty-treatment gate (ALLOW_DIRTY_TREATMENT is guard-test-only) ---"
  if [ -n "$(git status --porcelain -- src/data/yaw_rotation.py src/training/diffusion.py src/training/factory.py train.py)" ]; then
    case_run "H1 FULL refuses a dirty treatment path without the bypass" 2 \
      "treatment paths are DIRTY|||does not describe what would run|||ADMISSION EVIDENCE GATE FAILED" \
      -- ARM=DSPA MODE=FULL MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
    case_run "H2 the bypass is announced in the log when used" 2 \
      "GUARD-TESTING ONLY; never on a real launch|||admission evidence OK" \
      -- ARM=DSPA MODE=FULL ALLOW_DIRTY_TREATMENT=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  else
    echo "SKIP  H1/H2 - the treatment paths are clean in this checkout, so the dirty branch"
    echo "SKIP        cannot be exercised without dirtying tracked source (refused)."
  fi
fi

echo "--- G. RESTART lineage (synthetic ckpts in ${SAVEDIR_DSPA}) ---"
if [ -e "$SAVEDIR_DSPA" ]; then
  echo "SKIP  ${SAVEDIR_DSPA} already exists - refusing to touch a possibly-live run's directory."
  echo "SKIP  cases: scratch-into-dirty-namespace / outside-namespace / wrong step / other-arm cap /"
  echo "SKIP         float-typed cap / cleared optimizer / no param_groups / no scheduler / accept"
else
  TMP="$TMP" SAVEDIR="$SAVEDIR_DSPA" DSPA_CONFIG="$DSPA_CONFIG" DSCS3_CONFIG="$DSCS3_CONFIG" \
  python3 - <<'PY' || { echo "could not build synthetic checkpoints - abort"; exit 3; }
import json, os, torch
savedir = os.environ["SAVEDIR"]; tmp = os.environ["TMP"]
dspa = json.load(open(os.environ["DSPA_CONFIG"]))
dscs3 = json.load(open(os.environ["DSCS3_CONFIG"]))
nocap = json.loads(json.dumps(dspa)); del nocap["training"]["frame_avg_max_fwd_samples"]
floatcap = json.loads(json.dumps(dspa)); floatcap["training"]["frame_avg_max_fwd_samples"] = 32.0

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
torch.save(ck(20000, dspa),                     os.path.join(savedir, "guardtest-ok-step=20000.ckpt"))
torch.save(ck(20000, dscs3),                    os.path.join(savedir, "guardtest-othercap-step=20000.ckpt"))
torch.save(ck(20000, nocap),                    os.path.join(savedir, "guardtest-nocapkey-step=20000.ckpt"))
torch.save(ck(20000, floatcap),                 os.path.join(savedir, "guardtest-floatcap-step=20000.ckpt"))
torch.save(ck(20000, dspa, cleared=True),       os.path.join(savedir, "guardtest-stripped-step=20000.ckpt"))
torch.save(ck(20000, dspa, groups=False),       os.path.join(savedir, "guardtest-nogroups-step=20000.ckpt"))
torch.save(ck(20000, dspa, scheds=False),       os.path.join(savedir, "guardtest-nosched-step=20000.ckpt"))
torch.save(ck(20000, dspa),                     os.path.join(tmp, "guardtest-outside-step=20000.ckpt"))
print("synthetic RESTART checkpoints written to", savedir, "and", tmp)
PY
  NS_OK="${SAVEDIR_DSPA}/guardtest-ok-step=20000.ckpt"
  NS_OTHER="${SAVEDIR_DSPA}/guardtest-othercap-step=20000.ckpt"
  NS_NOKEY="${SAVEDIR_DSPA}/guardtest-nocapkey-step=20000.ckpt"
  NS_FLOAT="${SAVEDIR_DSPA}/guardtest-floatcap-step=20000.ckpt"
  NS_STRIPPED="${SAVEDIR_DSPA}/guardtest-stripped-step=20000.ckpt"
  NS_NOGROUPS="${SAVEDIR_DSPA}/guardtest-nogroups-step=20000.ckpt"
  NS_NOSCHED="${SAVEDIR_DSPA}/guardtest-nosched-step=20000.ckpt"
  OUTSIDE="${TMP}/guardtest-outside-step=20000.ckpt"
  EV_ARGS=(ALLOW_DIRTY_TREATMENT=1)
  [ "$EVIDENCE_OWNED" = "1" ] || EV_ARGS=()   # without evidence the gate stops earlier

  case_run "G0 MODE=FULL into a namespace that already holds checkpoints rejected" 2 \
    "refuses to launch into a namespace that already holds checkpoints|||use MODE=RESTART" \
    -- ARM=DSPA MODE=FULL
  case_run "G1 RESTART from outside this arm's namespace rejected" 2 \
    "may only resume a checkpoint written by THIS arm's FULL run" \
    -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$OUTSIDE"
  case_run "G2 RESTART MAXSTEPS <= EXPECTED_STEP rejected" 2 "must exceed the resume step 20000" \
    -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 MAXSTEPS=20000 RESUME_CKPT="$NS_OK"

  if [ "$EVIDENCE_OWNED" = "1" ]; then
    case_run "G3 RESTART with a wrong EXPECTED_STEP rejected" 2 "global_step 20000 != expected 22500" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=22500 RESUME_CKPT="$NS_OK" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G4 RESTART on a ckpt embedding the OTHER arm's cap rejected" 2 \
      "type-strict mismatch|||model_config.training.frame_avg_max_fwd_samples: 96 != 32|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_OTHER" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G5 RESTART on a ckpt with NO cap key rejected" 2 \
      "present in the config, absent from the checkpoint|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOKEY" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G6 RESTART on a ckpt whose cap is 32.0 rejected (TYPE-strict: 32.0 == 32 in python)" 2 \
      "type float != int|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_FLOAT" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G7 optimizer-stripped ckpt rejected (exp_14 arms are warm-only)" 2 \
      "optimizer state is CLEARED|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_STRIPPED" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G8 ckpt with no param_groups rejected" 2 \
      "no 'param_groups'|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOGROUPS" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G9 ckpt with no lr_schedulers rejected" 2 \
      "no 'lr_schedulers'|||resume-lineage check FAILED" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 RESUME_CKPT="$NS_NOSCHED" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
    case_run "G10 VALID RESTART (evidence + full-state lineage, dry-fail at VRAM)" 2 \
      "mode=RESTART|||config contract OK (DSPA)|||admission evidence OK|||resume lineage OK|||optimizer_state=FULL|||identity: --name FLAC_exp14_DSPA --experiment-name exp14_DSPA --save-dir outputs_FLAC/exp14_DSPA|||refusing to launch" \
      -- ARM=DSPA MODE=RESTART EXPECTED_STEP=20000 MAXSTEPS=40000 RESUME_CKPT="$NS_OK" "${EV_ARGS[@]}" MIN_FREE_MB="$DRYFAIL_MIN_FREE"
  else
    echo "SKIP  G3-G10 - a real evidence directory exists, so RESTART cannot reach the lineage gate here."
  fi

  echo "--- targeted cleanup of ${SAVEDIR_DSPA} ---"
  for f in "$NS_OK" "$NS_OTHER" "$NS_NOKEY" "$NS_FLOAT" "$NS_STRIPPED" "$NS_NOGROUPS" "$NS_NOSCHED"; do
    [ -f "$f" ] && rm -f "$f" && echo "  rm ${f}"
  done
  if rmdir "$SAVEDIR_DSPA" 2>/dev/null; then
    echo "  rmdir ${SAVEDIR_DSPA} (was empty)"
  else
    echo "  KEPT ${SAVEDIR_DSPA} - not empty (something else wrote into it); left untouched:"
    ls -la "$SAVEDIR_DSPA" | sed 's/^/    /'
  fi
fi

echo "--- I. the VALID PROBE launches: both arms + their declared chunk plans ---"
case_run "I1 DSPA PROBE accept: probe identity, 15/5 defaults, 1/3 per-angle chunk plan" 2 \
  "exp_14 fa_drawshare DSPA (PROBE) DDP+SyncBN|||identity: --name FLAC_exp14_DSPA_probe --experiment-name exp14_DSPA_probe --save-dir outputs_FLAC/exp14_DSPA_probe|||recipe: 32x2x1 eff64 seed42 -> 15 | ckpt-every 5|||config contract OK (DSPA)|||CHUNK PLAN (announcement 06): cap=32 per-rank micro-batch=32 orbit=4 angles (3 rotated)|||angles_per_chunk = max(1, 32 // 32) = 1|||rotated-angle chunks = [1, 1, 1] (3 conditioner forward(s) of [32, 32, 32] samples)|||shared-angle count = 1/3|||per-angle draws (the July path)|||refusing to launch" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "I2 DSCS3 PROBE accept: probe identity, 3/3 shared chunk plan + the cap-96 VRAM note" 2 \
  "exp_14 fa_drawshare DSCS3 (PROBE) DDP+SyncBN|||identity: --name FLAC_exp14_DSCS3_probe --experiment-name exp14_DSCS3_probe --save-dir outputs_FLAC/exp14_DSCS3_probe|||config contract OK (DSCS3)|||CHUNK PLAN (announcement 06): cap=96 per-rank micro-batch=32 orbit=4 angles (3 rotated)|||angles_per_chunk = max(1, 96 // 32) = 3|||rotated-angle chunks = [3] (1 conditioner forward(s) of [96] samples)|||shared-angle count = 3/3|||3 angles share one RoPE draw|||floor is NOT requalified for this arm|||refusing to launch" \
  -- ARM=DSCS3 MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"
case_run "I3 a PROBE never writes into the production namespace" 2 \
  "--save-dir outputs_FLAC/exp14_DSPA_probe|||refusing to launch" \
  -- ARM=DSPA MODE=PROBE MIN_FREE_MB="$DRYFAIL_MIN_FREE"

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
if [ -d "outputs_FLAC/exp14_DSPA_probe" ] || [ -d "outputs_FLAC/exp14_DSCS3_probe" ]; then
  FAIL=$((FAIL+1)); echo "FAIL  a probe namespace was created by a dry-fail run (nothing should have run)"
else
  PASS=$((PASS+1)); echo "PASS  no probe namespace was created (every case dry-failed before train.py)"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
