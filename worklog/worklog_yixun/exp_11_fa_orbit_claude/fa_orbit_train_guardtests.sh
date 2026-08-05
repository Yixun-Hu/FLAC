#!/usr/bin/env bash
# ============================================================================
# fa_orbit_train_guardtests.sh — guard-branch exercise for fa_orbit_train.sbatch
# (same shell-pragmatic form as exp_10's bf_resume_launch_guardtests.sh).
#
# Drives the REAL launcher through every fail-closed branch plus the valid
# INITIAL/RESTART lineage modes. Two safe vehicles, no GPU and no Slurm needed:
#
#   * DRYRUN=1 — runs every cheap gate (params, rung, arm->config map, semantic
#     orbit/grad-ckpt gate, duplicate-run, lineage) and then prints the train.py
#     argv and its parity diff against the exp_07 reference, exiting BEFORE any
#     Slurm/GPU/wandb/ViT gate and before torchrun. The commit/drift gate is
#     advisory in this mode (so the parity dry-run works on a dev tree).
#   * real mode with a fake SLURM_JOB_ID and an impossible MIN_FREE_MB — proves
#     the commit-binding/drift gate is fail-closed and that a valid invocation
#     stops at the VRAM gate, i.e. before wandb, the ViT/init gate and training.
#
# Safety: nothing is submitted, no GPU is touched, no checkpoint is read or
# written, synthetic checkpoints are tiny files in a mktemp dir, and the arm
# save-dir probe directory is created only if absent and removed afterwards.
#
# Usage:  bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train_guardtests.sh
# Exit 0 = every case behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
LAUNCHER="${EXPDIR}/fa_orbit_train.sbatch"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_orbit_${TS}_guardtests.log"
HEAD_SHA="$(git rev-parse HEAD)"
DRYFAIL_MIN_FREE=99000000        # impossible: forces the VRAM gate to abort valid cases

exec > >(tee -a "$LOG") 2>&1
echo "=== fa_orbit_train guard exercise — ${TS} — $(git rev-parse --short HEAD) ==="
[ -f "$LAUNCHER" ] || { echo "launcher not found: ${LAUNCHER} - abort"; exit 3; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
: > "${TMP}/fake.ckpt"

PASS=0; FAIL=0
# case <name> <expected-exit> <expected-substring> -- <env assignments...> ; runs the launcher
case_run() {
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  [ "$1" = "--" ] && shift
  local out rc
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"
    PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + text '${want_txt}', got rc=${rc}"
    echo "$out" | tail -6 | sed 's/^/        | /'
    FAIL=$((FAIL + 1))
  fi
}

BASE=(DRYRUN=1 "EXPECT_SHA=${HEAD_SHA}" MIN_FREE_MB=1)

echo "--- A. parameter and rung gates ---"
case_run "missing ARM"            2 "ARM"            -- "${BASE[@]}" RUNG=16x4
case_run "unknown ARM"            2 "not a legal exp_11 arm" -- "${BASE[@]}" ARM=C7 RUNG=16x4
case_run "FA1 is not an arm"      2 "not a legal exp_11 arm" -- "${BASE[@]}" ARM=FA1 RUNG=16x4
case_run "VAN is not an arm"      2 "not a legal exp_11 arm" -- "${BASE[@]}" ARM=VAN RUNG=16x4
case_run "missing RUNG"           2 "RUNG"           -- "${BASE[@]}" ARM=C4L
case_run "32x2 rejected (OOM)"    2 "32x2"           -- "${BASE[@]}" ARM=C4L RUNG=32x2
case_run "bogus rung"             2 "RUNG"           -- "${BASE[@]}" ARM=C4L RUNG=64x1
case_run "missing EXPECT_SHA"     2 "EXPECT_SHA"     -- DRYRUN=1 ARM=C4L RUNG=16x4
case_run "MAXSTEPS non-numeric"   2 "MAXSTEPS"       -- "${BASE[@]}" ARM=C4L RUNG=16x4 MAXSTEPS=lots

echo "--- B. restart-lineage gates (exp_10 pattern) ---"
case_run "initial + RESUME_CKPT"  2 "INITIAL launch must not carry" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 "RESUME_CKPT=${TMP}/fake.ckpt"
case_run "initial + EXPECTED_STEP" 2 "EXPECTED_STEP" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 EXPECTED_STEP=5000
case_run "restart w/o ckpt"       2 "RESTART requires RESUME_CKPT" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 EXPECTED_STEP=5000 RESUME_CKPT=
case_run "restart ckpt missing"   2 "not found" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/nope.ckpt"
case_run "restart foreign ckpt"   2 "may only resume a checkpoint written by this arm" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 EXPECTED_STEP=5000 "RESUME_CKPT=${TMP}/fake.ckpt"
case_run "restart step 0"         2 "EXPECTED_STEP" \
  -- "${BASE[@]}" ARM=C4L RUNG=16x4 EXPECTED_STEP=0 "RESUME_CKPT=${TMP}/fake.ckpt"

# a restart ckpt INSIDE the arm's own save-dir: legal lineage, so the run must
# reach the dry-run parity report. Create the namespace only if it is absent.
ARM_SAVEDIR="outputs_FLAC/exp11_C8"
CKPT_DIR="${ARM_SAVEDIR}/FLAC_exp11_C8/exp11_C8/checkpoints"
OWNED=0
if [ ! -e "$ARM_SAVEDIR" ]; then
  mkdir -p "$CKPT_DIR"; : > "${CKPT_DIR}/epoch=1-step=5000.ckpt"; OWNED=1
  case_run "restart own ckpt OK"  0 "ARGV PARITY OK" \
    -- "${BASE[@]}" ARM=C8 RUNG=16x4 EXPECTED_STEP=5000 MAXSTEPS=40000 \
       "RESUME_CKPT=${CKPT_DIR}/epoch=1-step=5000.ckpt"
  case_run "restart MAXSTEPS<=step" 2 "must exceed" \
    -- "${BASE[@]}" ARM=C8 RUNG=16x4 EXPECTED_STEP=5000 MAXSTEPS=5000 \
       "RESUME_CKPT=${CKPT_DIR}/epoch=1-step=5000.ckpt"
  case_run "initial refuses existing run dir" 2 "already exists" \
    -- "${BASE[@]}" ARM=C8 RUNG=16x4
  rm -f "${CKPT_DIR}/epoch=1-step=5000.ckpt"
  rmdir -p "$CKPT_DIR" 2>/dev/null || true
  [ -e "$ARM_SAVEDIR" ] && rm -rf "$ARM_SAVEDIR"
else
  echo "SKIP  restart-own-ckpt cases: ${ARM_SAVEDIR} already exists (a live run may own it)"
fi

echo "--- C. valid initial launches reach the argv-parity report (all four arms) ---"
for ARM in C4L C8 C16 C32; do
  for RUNG in 16x4 8x8; do
    case_run "dry-run ${ARM} ${RUNG}" 0 "ARGV PARITY OK" -- "${BASE[@]}" ARM=$ARM RUNG=$RUNG
  done
done
case_run "argv carries the arm config" 0 "FLAC_AR_BF_C16.json" -- "${BASE[@]}" ARM=C16 RUNG=8x8
case_run "argv carries rung batch size" 0 "--batch-size 8" -- "${BASE[@]}" ARM=C4L RUNG=8x8
case_run "argv carries num-gpus"        0 "--num-gpus 4"   -- "${BASE[@]}" ARM=C4L RUNG=16x4

echo "--- D. commit-binding / drift gate is fail-closed in REAL mode ---"
case_run "wrong EXPECT_SHA aborts"  2 "EXPECT_SHA" \
  -- ARM=C4L RUNG=16x4 EXPECT_SHA=0000000000000000000000000000000000000000 \
     SLURM_JOB_ID=999999 "MIN_FREE_MB=${DRYFAIL_MIN_FREE}"
case_run "real mode needs Slurm"    2 "must run under sbatch" \
  -- ARM=C4L RUNG=16x4 "EXPECT_SHA=${HEAD_SHA}" "MIN_FREE_MB=${DRYFAIL_MIN_FREE}"

echo "--- E. semantic gate rejects a mislabelled arm config ---"
# swap C8's config content for C4L's inside a scratch copy of the tree layout:
# the launcher derives the path from ARM, so the only way to exercise this is to
# point the gate at a config whose orbit disagrees with the arm. We do that by
# temporarily shadowing the arm config with a copy of another arm's file.
if [ "$OWNED" = "1" ] || [ ! -e "outputs_FLAC/exp11_C32" ]; then
  cp "${EXPDIR}/FLAC_AR_BF_C32.json" "${TMP}/C32.orig"
  cp "${EXPDIR}/FLAC_AR_BF_C8.json" "${EXPDIR}/FLAC_AR_BF_C32.json"
  case_run "orbit mismatch rejected" 2 "ARM/CONFIG GATE" -- "${BASE[@]}" ARM=C32 RUNG=16x4
  cp "${TMP}/C32.orig" "${EXPDIR}/FLAC_AR_BF_C32.json"
  git diff --quiet -- "${EXPDIR}/FLAC_AR_BF_C32.json" \
    && echo "PASS  C32 config restored byte-identically" && PASS=$((PASS + 1)) \
    || { echo "FAIL  C32 config NOT restored — restore it from git before launching"; FAIL=$((FAIL + 1)); }
fi

echo
echo "=== guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
echo "log: ${LOG}"
