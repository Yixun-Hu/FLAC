#!/usr/bin/env bash
# ============================================================================
# fa_orbit_submit.sh — the ONLY sanctioned way to submit an exp_11 arm.
#
# Round-3 review B1: an operator must never hand-assemble --gres/--cpus/--mem/
# --time. Every resource flag is derived here from the pins inside
# fa_orbit_train.sbatch (read out of the script itself, so the two can never
# disagree), and the job is refused unless the tracked tree is clean.
#
#   ./fa_orbit_submit.sh C8
#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 12500
#   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
#   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
#
# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
# in an atomic, no-clobber manifest next to the launcher.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH_FILE="${EXPDIR}/fa_orbit_train.sbatch"
DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
PLACEHOLDER="TO-PIN-AFTER-P0"

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

ARM="${1:-}"
[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32> [--resume <ckpt> --expected-step <n>] - abort"; exit 2; }
shift
case "$ARM" in C4L|C8|C16|C32) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32 - abort"; exit 2;; esac

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
  RUNG="${SMOKE_RUNG:?SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8)}"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
  [ -n "${SMOKE_MIN_FREE_MB:-}" ] || { echo "SMOKE=1 requires SMOKE_MIN_FREE_MB - abort"; exit 2; }
  JOBNAME="exp11-smoke-${ARM}"
else
  RUNG="$(pin PINNED_RUNG)"
  TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
  done
  JOBNAME="exp11-${ARM}-train"
fi
case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }

# --- drift gate: a queued job must run reviewed, committed code --------------
# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
# arms are running and Slurm appends to their tracked *.out logs continuously, so
# a folder-wide check would abort every screen on a live-log write. Configs,
# drivers and validators are still fully covered.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
         worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD)"

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
MANIFEST="${EXPDIR}/fa_orbit_submission_${ARM}_${INTENT_ID}.txt"
[ ! -e "$MANIFEST" ] || { echo "submission manifest ${MANIFEST} already exists - abort"; exit 2; }
TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
{
  echo "# exp_11 arm submission (intent published BEFORE sbatch)"
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
