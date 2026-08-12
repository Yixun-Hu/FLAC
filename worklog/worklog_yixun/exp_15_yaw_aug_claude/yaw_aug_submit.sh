#!/usr/bin/env bash
# ============================================================================
# yaw_aug_submit.sh — the ONLY sanctioned way to submit the exp_15 arm.
#
# Verbatim from exp_11's fa_orbit_submit.sh, then re-pinned (plan §6.6). exp_11
# round-3 B1: an operator must never hand-assemble --gres/--cpus/--mem/--time.
# Every resource flag is derived here from the pins inside yaw_aug_train.sbatch
# (read out of the script itself, so the two can never disagree), and the job is
# refused unless the tracked tree is clean.
#
#   ./yaw_aug_submit.sh YAWAUG
#   ./yaw_aug_submit.sh YAWAUG --resume <ckpt> --expected-step 12500   # crash restart, <= 40k
#   SMOKE=1 SMOKE_RUNG=8x8 SMOKE_MIN_FREE_MB=14000 ./yaw_aug_submit.sh YAWAUG
#   DRYRUN=1 ./yaw_aug_submit.sh YAWAUG     # print the sbatch line, submit nothing
#
# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
# in an atomic, no-clobber manifest next to the launcher.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_15_yaw_aug_claude"
EXP11DIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"   # READ-ONLY (helpers + the control config)
SBATCH_FILE="${EXPDIR}/yaw_aug_train.sbatch"
DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
PLACEHOLDER="TO-PIN-AFTER-P0"

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

ARM="${1:-}"
[ -n "$ARM" ] || { echo "usage: $0 YAWAUG [--resume <ckpt> --expected-step <n>] - abort"; exit 2; }
shift
case "$ARM" in YAWAUG) ;; *) echo "ARM '${ARM}' must be YAWAUG (the only exp_15 arm) - abort"; exit 2;; esac

RESUME_CKPT=""; EXPECTED_STEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    *) echo "unknown argument '$1' - abort"; exit 2 ;;
  esac
done
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac

# --- pins are read FROM the launcher, so submitter and job cannot disagree ----
pin() {  # read one PINNED_* value out of the launcher (quoted or bare)
  awk -v k="$1" '$0 ~ "^"k"=" {
        if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
        else { split($0, a, "="); split(a[2], b, " "); print b[1] }
        exit }' "$SBATCH_FILE"
}
if [ "$SMOKE" = "1" ]; then
  # The approved smoke runs the REAL topology (plan §7-6) — see the launcher.
  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (8x8, the production rung)}"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
  JOBNAME="exp15-smoke-${ARM}"
else
  RUNG="$(pin PINNED_RUNG)"
  # The wall pin follows the LEG. For exp_15 both pins are 24:00:00 (a crash
  # restart finishes the same 40k budget), but the selection is kept so the
  # submitter and the job provably allocate and enforce the same pin.
  if [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
  else
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
  fi
  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
  done
  JOBNAME="exp15-${ARM}-train"
fi
case "$RUNG" in 8x8) ;; *) echo "rung '${RUNG}' must be 8x8 — exp_15 has ONE topology, smoke included - abort"; exit 2;; esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }

# --- drift gate: a queued job must run reviewed, committed code --------------
# THE SAME CLOSURE THE WORKER ENFORCES (yaw_aug_train.sbatch section C), so the
# submitter cannot queue a job the worker will refuse. Pathspecs are QUOTED so
# git expands them and a DELETED tracked file still matches; data/AR carries the
# split JSONs the dataloader opens (train.json is the one this run reads), whose
# uncommitted edit would change the samples trained while every code hash stayed
# put; src/tests is excluded because pytest-only code is never imported by
# train.py and the concurrent TDD sessions land test files continuously; and a
# FAILING git invocation is fatal, never an empty "clean" answer.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
         "$EXPDIR/FLAC_AR_YAWAUG.json" "$EXPDIR/yaw_aug_train.sbatch" "$EXPDIR/yaw_aug_submit.sh" \
         "$EXPDIR/yaw_aug_record_control.py" "$EXPDIR/yaw_aug_pin_allowlist.txt" \
         "$EXPDIR/yaw_aug_control_admission.json" \
         "$EXP11DIR/fa_orbit_ckpt_preflight.py" "$EXP11DIR/fa_orbit_classify.py" \
         "$EXP11DIR/fa_orbit_wandb_readback.py" "$EXP11DIR/FLAC_AR_VANCKPT.json" \
         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>&1)" \
  || { echo "git status for the drift gate failed: ${DRIFT} - abort"; exit 2; }
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD 2>&1)" || { echo "git rev-parse HEAD failed: ${SHA} - abort"; exit 2; }
printf '%s\n' "$SHA" | grep -qE '^[0-9a-f]{40}$' \
  || { echo "HEAD did not resolve to a full 40-hex commit id ('${SHA}') - abort"; exit 2; }

ARGS=(
  --job-name="$JOBNAME"
  --gres="gpu:l40:${NGPU}"
  --cpus-per-task="$((8 + 7 * NGPU))"
  --mem="$(((12 * NGPU + 12)))G"
  --time="$TIME_LIMIT"
  --export="ALL,ARM=${ARM},EXPECT_SHA=${SHA},OUTPUT_ROOT=outputs_FLAC"
)
[ "$SMOKE" = "1" ] && ARGS[5]="${ARGS[5]},SMOKE=1,SMOKE_RUNG=${SMOKE_RUNG},SMOKE_MIN_FREE_MB=${SMOKE_MIN_FREE_MB},SMOKE_MAXSTEPS=${SMOKE_MAXSTEPS:-30},SMOKE_TIME=${TIME_LIMIT}"
[ -n "$RESUME_CKPT" ] && ARGS[5]="${ARGS[5]},RESUME_CKPT=${RESUME_CKPT},EXPECTED_STEP=${EXPECTED_STEP}"
ARGS+=("$SBATCH_FILE")

echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE}"
if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN sbatch ${ARGS[*]}"
  exit 0
fi

# --- NEW-3: publish the INTENT before submitting -----------------------------
# The provenance record must exist before the job can exist, otherwise a local
# write failure leaves a queued job nobody recorded. The intent manifest carries
# the exact command and pins; the job id is appended afterwards, and if that
# append fails the exact job we just created is cancelled.
INTENT_ID="$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
MANIFEST="${EXPDIR}/yaw_aug_submission_${ARM}_${INTENT_ID}.txt"
[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
{
  echo "# exp_15 arm submission (intent published BEFORE sbatch)"
  echo "intent_id ${INTENT_ID}"
  echo "submitted_at $(date -Is)"
  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU}"
  echo "jobname ${JOBNAME} time ${TIME_LIMIT} smoke ${SMOKE}"
  echo "commit ${SHA}"
  echo "pins rung=${RUNG} maxsteps=$(pin PINNED_MAXSTEPS) ckpt_every=$(pin PINNED_CHECKPOINT_EVERY) min_free_mb=$(pin PINNED_MIN_FREE_MB) p0_manifest_sha256=$(pin PINNED_P0_MANIFEST_SHA256)"
  echo "resume ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
  echo "sbatch sbatch ${ARGS[*]}"
} >> "$TMP" || { echo "intent manifest write failed - abort"; exit 3; }
mv -n "$TMP" "$MANIFEST" || { echo "intent manifest publication failed - abort"; exit 2; }
[ -e "$MANIFEST" ] || { echo "intent manifest ${MANIFEST} did not appear - abort"; exit 2; }
echo "intent manifest: ${MANIFEST}"

OUT="$(sbatch "${ARGS[@]}" 2>&1)"; JID="$(echo "$OUT" | awk '/Submitted batch job/ {print $NF}')"
if [ -z "$JID" ]; then
  echo "SUBMIT FAILED: ${OUT}"
  echo "submit_failed $(date -Is)" >> "$MANIFEST"
  exit 1
fi
echo "submitted ${ARM} -> job ${JID}"
if ! echo "jobid ${JID}" >> "$MANIFEST"; then
  echo "could not append job id ${JID} to ${MANIFEST} — cancelling the job rather than leave it unrecorded"
  scancel "$JID" || echo "scancel ${JID} FAILED — cancel it by hand NOW"
  exit 2
fi
echo "submission recorded: ${MANIFEST} (job ${JID})"
