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
#   ./fa_orbit_submit.sh C8 --resume <ckpt> --expected-step 40000 --chunk-end 42500
#   SMOKE=1 SMOKE_RUNG=16x4 SMOKE_MIN_FREE_MB=14000 ./fa_orbit_submit.sh C4L
#   DRYRUN=1 ./fa_orbit_submit.sh C8        # print the sbatch line, submit nothing
#
# CHUNKED legs (round 5). The partition never backfills a 34-160 h allocation, so
# a leg may declare --chunk-end <n>: it trains to that boundary, saves, and exits,
# and the next leg resumes from there. A chunk leg is walled by
# PINNED_TIME_LIMIT_CHUNK_<ARM> (hours, not days), which is what makes it
# schedulable. The chain is driven by fa_orbit_chunk_watchdog.sh, which submits
# every chunk through THIS script — never through sbatch directly.
#
# Resources per rung (micro x N = 64): --gres=gpu:l40:N, --cpus-per-task=8+7N,
# --mem=(12N+12)G, --time=<the arm's pinned limit>. Each submission is recorded
# in an atomic, no-clobber manifest next to the launcher.
#
# ANTI-DUPLICATE RESERVATION (round-5 r2 review, blocking 1). Every real
# submission takes an exclusive flock on .submit_<ARM>.lock and re-checks the
# queue for a live exp11-<ARM>-train INSIDE that lock before calling sbatch, so a
# manual invocation and the watchdog cannot both queue the same boundary. See the
# block below for the ordering argument.
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
[ -n "$ARM" ] || { echo "usage: $0 <C4L|C8|C16|C32|VANL> [--resume <ckpt> --expected-step <n> [--chunk-end <n>]] - abort"; exit 2; }
shift
case "$ARM" in C4L|C8|C16|C32|VANL) ;; *) echo "ARM '${ARM}' must be C4L|C8|C16|C32|VANL - abort"; exit 2;; esac

RESUME_CKPT=""; EXPECTED_STEP=0; CHUNK_END=""
while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME_CKPT="${2:?--resume needs a path}"; shift 2 ;;
    --expected-step) EXPECTED_STEP="${2:?--expected-step needs a number}"; shift 2 ;;
    --chunk-end) CHUNK_END="${2:?--chunk-end needs a number}"; shift 2 ;;
    *) echo "unknown argument '$1' - abort"; exit 2 ;;
  esac
done
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "--expected-step must be a non-negative integer - abort"; exit 2;; esac
# --- --chunk-end: shape-checked HERE, and again inside the job ---------------
# A chunk is meaningful only for a leg that resumes: it narrows where THIS job
# stops, never what the campaign may reach (the budget pin stays 100000). It must
# land on a saved checkpoint, or the next chunk has nothing to resume from.
CHUNK_BUDGET="$(awk -F= '/^PINNED_MAXSTEPS=/{split($2,a," "); print a[1]; exit}' "$SBATCH_FILE")"
case "$CHUNK_BUDGET" in ''|*[!0-9]*) echo "could not read PINNED_MAXSTEPS from ${SBATCH_FILE} - abort"; exit 3;; esac
if [ -n "$CHUNK_END" ]; then
  case "$CHUNK_END" in ''|*[!0-9]*) echo "--chunk-end must be a positive integer - abort"; exit 2;; esac
  [ "$SMOKE" != "1" ] || { echo "--chunk-end is a production chunk-chain input and has no meaning under SMOKE=1 - abort"; exit 2; }
  { [ -n "$RESUME_CKPT" ] && [ "$EXPECTED_STEP" -gt 0 ]; } \
    || { echo "--chunk-end is valid only together with --resume/--expected-step - abort"; exit 2; }
  [ "$((CHUNK_END % 2500))" -eq 0 ] || { echo "--chunk-end ${CHUNK_END} is not a multiple of 2500 (the pinned checkpoint cadence) - abort"; exit 2; }
  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || { echo "--chunk-end ${CHUNK_END} must exceed --expected-step ${EXPECTED_STEP} - abort"; exit 2; }
  [ "$CHUNK_END" -le "$CHUNK_BUDGET" ] || { echo "--chunk-end ${CHUNK_END} exceeds the pinned budget ${CHUNK_BUDGET} - abort"; exit 2; }
fi

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
  # A RESTART leg is a different budget from the INITIAL one: 60k further steps,
  # not 40k from scratch. Selecting the INITIAL limit for a restart would wall-kill
  # every arm partway through the extension.
  # A CHUNK leg is shorter still: it stops at the next boundary, so it is walled
  # by the arm's CHUNK pin (hours) rather than its whole-extension RESTART pin
  # (days) — the whole point of chunking is an allocation the scheduler backfills.
  if [ -n "$CHUNK_END" ]; then
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_CHUNK_${ARM}")"
  elif [ -n "${EXPECTED_STEP:-}" ] && [ "${EXPECTED_STEP:-0}" -gt 0 ]; then
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_RESTART_${ARM}")"
  else
    TIME_LIMIT="$(pin "PINNED_TIME_LIMIT_${ARM}")"
  fi
  [ -n "$TIME_LIMIT" ] || { echo "the launcher carries no wall pin for this ${ARM} leg - abort"; exit 2; }
  for V in "$RUNG" "$TIME_LIMIT" "$(pin PINNED_MIN_FREE_MB)" "$(pin PINNED_P0_MANIFEST_SHA256)"; do
    [ "$V" != "$PLACEHOLDER" ] || { echo "the launcher still carries ${PLACEHOLDER} pins: the P0 report has not been pinned yet — no arm may be submitted (use SMOKE=1 for the smoke) - abort"; exit 2; }
  done
  JOBNAME="exp11-${ARM}-train"
fi
case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung '${RUNG}' must be 32x2|16x4|8x8 - abort"; exit 2;; esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "rung ${RUNG}: MB*NGPU != 64 - abort"; exit 2; }

# --- SUBMISSION RESERVATION (round-5 r2 review, blocking 1) -------------------
# The chunk watchdog's singleton lock only prevents a second WATCHDOG. A human
# running this script by hand could still slip between the watchdog's queue check
# and its sbatch and queue the same boundary twice, and the job-side run-directory
# flock cannot help: it is taken long after scheduling, so it serialises execution
# while still burning a second allocation. The reservation therefore lives HERE,
# in the ONE sanctioned submitter that every path — watchdog or human — goes
# through:
#
#     flock .submit_<ARM>.lock  ->  squeue -n exp11-<ARM>-train  ->  sbatch
#
# The queue check is INSIDE the lock, so no other submitter can observe an empty
# queue and sbatch between our check and ours. Both directions are fail-CLOSED: a
# held lock refuses, and a squeue that exits NONZERO refuses too — an unreadable
# queue is never read as an empty one. The lock is released by the kernel when
# this process exits, on every path (refusal, sbatch failure, success).
#
# It is placed BEFORE the code-drift gate deliberately: a duplicate submission
# must be refused as a duplicate, not masked by whichever gate happens to fire
# first, and holding the reservation across the (cheap, local) drift check costs
# nothing. SMOKE and DRYRUN are excluded by design — a smoke leg carries its own
# job name and identity, and a dry run submits nothing at all.
if [ "$SMOKE" != "1" ] && [ "$DRYRUN" != "1" ]; then
  SUBMIT_LOCK="${EXPDIR}/.submit_${ARM}.lock"
  WHO="${USER:-$(id -un)}"
  exec 9>"$SUBMIT_LOCK" || { echo "could not open the ${ARM} submission lock ${SUBMIT_LOCK} - abort"; exit 3; }
  if ! flock -n 9; then
    echo "another submission for ${ARM} already holds ${SUBMIT_LOCK} — a leg for this arm is already being submitted - abort"
    exit 2
  fi
  # stderr is captured SEPARATELY on purpose: folded into stdout, a harmless
  # scheduler warning would read as a live job and refuse a legitimate leg.
  SQ_ERR="$(mktemp "${TMPDIR:-/tmp}/exp11_squeue.XXXXXX")" || { echo "could not create a temp file for the queue check - abort"; exit 3; }
  LIVE="$(squeue -h -u "$WHO" -n "$JOBNAME" -o '%i %T' 2>"$SQ_ERR")"; QRC=$?
  SQ_MSG="$(head -3 "$SQ_ERR" 2>/dev/null | tr '\n' ' ')"; rm -f "$SQ_ERR"
  if [ "$QRC" -ne 0 ]; then
    echo "squeue exited ${QRC} (${SQ_MSG:-no message}) — the queue state is UNKNOWN and an unreadable queue is never read as an empty one - abort"
    exit 2
  fi
  if [ -n "$LIVE" ]; then
    echo "a leg for this arm is already queued/running as ${JOBNAME} (${LIVE//$'\n'/; }) - abort"
    exit 2
  fi
fi

# --- drift gate: a queued job must run reviewed, committed code --------------
# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
# arms are running and Slurm appends to their tracked *.out logs continuously, so
# a folder-wide check would abort every screen on a live-log write. Configs,
# drivers and validators are still fully covered.
#
# arm_launch_registry.json is EXCLUDED (round-5 review B1). It is a lineage
# RECORD written by the reviewed recorder as each chunk finishes — a measurement
# *product*, not a measurement *surface* — and it is already outside the
# launcher's commit-binding closure. Gating it deadlocks the chunk chain: the
# recorder dirties it at 42500, so the very next submission (and every later
# one, for every arm) is refused until a human commits. Keep it outside.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src \
         "$EXPDIR"/*.json "$EXPDIR"/*.py "$EXPDIR"/*.sbatch "$EXPDIR"/*.sh \
         ":(exclude)${EXPDIR}/arm_launch_registry.json" \
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
[ -n "$CHUNK_END" ] && ARGS[5]="${ARGS[5]},CHUNK_END=${CHUNK_END}"
ARGS+=("$SBATCH_FILE")

echo "arm ${ARM} | rung ${RUNG} (${MB}x${NGPU}) | time ${TIME_LIMIT} | commit ${SHA} | smoke ${SMOKE} | chunk_end ${CHUNK_END:-<none>}"
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
  echo "chunk_end ${CHUNK_END:-<none>}"
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
