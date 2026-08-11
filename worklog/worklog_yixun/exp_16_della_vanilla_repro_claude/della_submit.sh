#!/usr/bin/env bash
# ============================================================================
# della_submit.sh — the ONLY supported way to submit an exp_16 job (plan §4d).
#
# It exists so that three things cannot drift apart: what was submitted, what
# commit it is bound to, and what the worklog says was submitted. Submitting
# `sbatch` by hand skips all three.
#
# Usage:
#   della_submit.sh eval  {unseen_s42|unseen_s43|seen_s42}
#   della_submit.sh train [--smoke] [--time D-HH:MM:SS]
#
#   --time is REQUIRED for a full training leg (the ETA comes from the smoke
#   measurement x1.5, plan §5 Phase 2) and defaults to 01:00:00 for --smoke.
#   della derives the QOS from --time; there is no --qos/--partition here, and
#   della REJECTS an explicit partition for GPU jobs.
#
# Gates (all fail-closed, distinct exit codes): tracked-file drift; HEAD not
# pushed to origin/della-flac-chequity. A queued job reads the checkout at RUN
# time, so an unpushed or dirty HEAD means the job's code identity is not the one
# under review — and the sbatch drivers re-check the same SHA on the node.
#
# DRYRUN=1 prints the sbatch line it WOULD run and exits: it submits nothing,
# writes nothing to the command record, and reports gate failures as advisories
# so the argv stays inspectable on a dirty/unpushed development tree.
#
# Exit codes: 2 usage, 3 tracked drift, 4 HEAD not pushed, 5 sbatch failed/
# unparsable, 6 command-record write failed.
# ============================================================================
set -euo pipefail

REPO=/home/yh4742/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"
BRANCH=della-flac-chequity
REMOTE=origin
RECORD="$EXPDIR/della_vanilla_repro_command.md"
DRYRUN="${DRYRUN:-0}"

die() { echo "GATEFAIL rc=${2} ${1}" >&2; exit "${2}"; }

usage() {
  echo "usage: della_submit.sh eval {unseen_s42|unseen_s43|seen_s42}" >&2
  echo "       della_submit.sh train [--smoke] [--time D-HH:MM:SS]" >&2
}

# --- A. arguments -------------------------------------------------------------
KIND="${1:-}"; shift || true
case "$KIND" in
  eval|train) ;;
  *) usage; die "first argument must be 'eval' or 'train' (got '${KIND}')" 2 ;;
esac

CELL=""; SMOKE=0; TIME_LIMIT=""
if [ "$KIND" = "eval" ]; then
  CELL="${1:-}"; shift || true
  case "$CELL" in
    unseen_s42|unseen_s43|seen_s42) ;;
    *) usage; die "eval CELL must be unseen_s42|unseen_s43|seen_s42 (got '${CELL}')" 2 ;;
  esac
  [ "$#" -eq 0 ] || die "eval takes no further options (its --time is fixed at 04:00:00 in della_eval.sbatch)" 2
else
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --smoke) SMOKE=1; shift ;;
      --time)  TIME_LIMIT="${2:-}"; shift 2 || die "--time needs a value" 2 ;;
      --time=*) TIME_LIMIT="${1#--time=}"; shift ;;
      *) usage; die "unknown train option '${1}'" 2 ;;
    esac
  done
  if [ "$SMOKE" = "1" ]; then
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
  else
    # No default on purpose: a full leg's wall-clock budget is a measured
    # quantity (smoke rate x1.5), and a defaulted one would silently truncate
    # a multi-day run at whatever the default happened to be.
    [ -n "$TIME_LIMIT" ] || die "a full training leg needs an explicit --time D-HH:MM:SS (smoke defaults to 01:00:00)" 2
  fi
  case "$TIME_LIMIT" in
    [0-9]-[0-9][0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9]-[0-9][0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9]:[0-9][0-9]:[0-9][0-9]) ;;
    *) die "--time '${TIME_LIMIT}' is not D-HH:MM:SS or HH:MM:SS" 2 ;;
  esac
fi

# --- B. tracked drift ---------------------------------------------------------
DRIFT="$(git -C "$REPO" status --porcelain --untracked-files=no)" \
  || die "git status FAILED - refusing to treat an error as a clean tree" 3
if [ -n "$DRIFT" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: tracked files are dirty (a real submit aborts here):"
    echo "$DRIFT"
  else
    echo "$DRIFT" >&2
    die "tracked files are dirty - commit or stash before submitting" 3
  fi
fi

# --- C. HEAD must be PUSHED ---------------------------------------------------
# The node re-checks EXPECT_SHA against the checkout at run time; binding a job
# to a commit that exists only on this filesystem makes that check unverifiable
# by anyone else (and unrecoverable if the checkout moves).
EXPECT_SHA="$(git -C "$REPO" rev-parse HEAD)" || die "git rev-parse HEAD FAILED" 4
if [ "$DRYRUN" = "1" ]; then
  git -C "$REPO" fetch --quiet "$REMOTE" "$BRANCH" 2>/dev/null || true
else
  git -C "$REPO" fetch --quiet "$REMOTE" "$BRANCH" || die "git fetch ${REMOTE} ${BRANCH} FAILED" 4
fi
REMOTE_SHA="$(git -C "$REPO" rev-parse "${REMOTE}/${BRANCH}" 2>/dev/null || echo '')"
if [ "$EXPECT_SHA" != "$REMOTE_SHA" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: HEAD ${EXPECT_SHA} != ${REMOTE}/${BRANCH} ${REMOTE_SHA:-<none>} (a real submit aborts here)"
  else
    die "HEAD ${EXPECT_SHA} is not pushed to ${REMOTE}/${BRANCH} (${REMOTE_SHA:-<none>})" 4
  fi
fi

# --- D. build the sbatch command ---------------------------------------------
if [ "$KIND" = "eval" ]; then
  JOB_NAME="exp16-eval-${CELL}"
  SBATCH_ARGV=(
    --job-name="$JOB_NAME"
    --export="ALL,EXPECT_SHA=${EXPECT_SHA},CELL=${CELL}"
    "$EXPDIR/della_eval.sbatch"
  )
else
  JOB_NAME="exp16-train"
  EXPORTS="ALL,EXPECT_SHA=${EXPECT_SHA}"
  if [ "$SMOKE" = "1" ]; then
    JOB_NAME="exp16-train-smoke"
    EXPORTS="${EXPORTS},SMOKE=1"
  fi
  SBATCH_ARGV=(
    --job-name="$JOB_NAME"
    --time="$TIME_LIMIT"
    --export="$EXPORTS"
    "$EXPDIR/della_train.sbatch"
  )
fi
SBATCH_LINE="sbatch ${SBATCH_ARGV[*]}"
echo "$SBATCH_LINE"

if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN: nothing submitted, command record untouched (EXPECT_SHA would be ${EXPECT_SHA})"
  exit 0
fi

# --- E. submit ----------------------------------------------------------------
SBATCH_OUT="$(sbatch "${SBATCH_ARGV[@]}")" || die "sbatch FAILED - nothing submitted" 5
# Parsed from sbatch's own words rather than --parsable so the line recorded in
# the worklog is byte-identical to the line that ran.
JOBID="$(printf '%s\n' "$SBATCH_OUT" | awk '/Submitted batch job/ {print $NF}')"
case "$JOBID" in
  ''|*[!0-9]*) echo "$SBATCH_OUT" >&2; die "could not parse a job id out of sbatch's output" 5 ;;
esac

# --- F. record it (append-only) ----------------------------------------------
if [ ! -f "$RECORD" ]; then
  {
    echo "# della_vanilla_repro — submission command record"
    echo
    echo "Append-only, written by \`della_submit.sh\` at submit time (plan §4d): the exact"
    echo "sbatch line, the commit the job is bound to, and the job id Slurm returned."
    echo
  } > "$RECORD" || die "could not create the command record ${RECORD}" 6
fi
{
  echo "## $(date -Is) — ${JOB_NAME} — job ${JOBID}"
  echo
  echo "- EXPECT_SHA: \`${EXPECT_SHA}\`"
  echo "- job id: \`${JOBID}\`"
  echo "- submitted by: \`$(whoami)@$(hostname)\`"
  echo
  echo '```bash'
  echo "$SBATCH_LINE"
  echo '```'
  echo
} >> "$RECORD" || die "could not append to the command record ${RECORD}" 6

echo "$JOBID"
