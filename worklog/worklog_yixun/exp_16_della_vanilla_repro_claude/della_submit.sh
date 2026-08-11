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
#   della_submit.sh train [--smoke] [--resume] [--time D-HH:MM:SS]
#
#   --time is REQUIRED for a full training leg (the ETA comes from the smoke
#   measurement x1.5, plan §5 Phase 2) and defaults to 01:00:00 for --smoke.
#   della derives the QOS from --time; there is no --qos/--partition here, and
#   della REJECTS an explicit partition for GPU jobs.
#   --resume opts a production leg into continuing from its newest checkpoint;
#   without it the driver REFUSES to start when checkpoints already exist.
#
# ENVIRONMENT ISOLATION (review B1): jobs are submitted with an explicit
# --export list, i.e. `--export=EXPECT_SHA=...,CELL=...`. Per sbatch(1) on Slurm
# 25.11, the `[ALL,]<environment_variables>` form without the leading ALL exports
# "all SLURM_* and SPANK option environment variables along with explicitly
# defined variables" — and NOTHING else from this shell. `--export=NONE,VAR=...`
# is NOT usable for this: the same man page states "User can not specify explicit
# environment variables with NONE", so the parameters would never arrive (the
# drivers would then fail closed at their EXPECT_SHA gate).
#
# TRANSACTION (review B3) — no released job may lack its record:
#   sbatch --hold --parsable  ->  jobid          (queued, cannot start)
#   append the record block under flock, then read the jobid back
#   scontrol release <jobid>                     (only now can it start)
#   any failure in between  ->  scancel <jobid>, report, exit nonzero
#
# GATES: the measurement CLOSURE must be clean (worklog files may be dirty — the
# command record itself is written while jobs are queued), and HEAD must be
# pushed. The pushed check uses `git ls-remote` so a DRYRUN writes NOTHING, not
# even FETCH_HEAD. A non-smoke `train` submission additionally requires the
# committed Phase-1 verdict (see PHASE1_PASS below) — the plan's "Phase 1 gates
# all training compute" is enforced here, not left to memory.
#
# DRYRUN=1 prints the sbatch line it WOULD run and exits: it submits nothing,
# writes nothing, and reports gate failures as advisories so the argv stays
# inspectable on a dirty/unpushed development tree.
#
# Exit codes: 2 usage, 3 closure dirty, 4 HEAD not pushed, 5 sbatch failed or
# unparsable, 6 record write/verify failed (job cancelled), 7 release failed,
# 8 Phase-1 verdict missing for a production training leg.
# ============================================================================
set -euo pipefail

REPO=/home/yh4742/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"
BRANCH=della-flac-chequity
REMOTE=origin
RECORD="$EXPDIR/della_vanilla_repro_command.md"
# The committed Phase-1 verdict. Plan §5: Phase 1 gates ALL training compute, and
# Rev 3 §2 makes that a file — values, deltas vs the pre-registered thresholds,
# loader/count/load evidence and a verdict line — so "the gate passed" is a
# reviewable artifact rather than something a submitter remembers.
PHASE1_PASS_REL="worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/PHASE1_PASS.md"
DRYRUN="${DRYRUN:-0}"
export PYTHONDONTWRITEBYTECODE=1   # nothing here runs python, but the kit is uniform

# --- THE MEASUREMENT CLOSURE (review B3; same list verbatim in all three kit files) --
CLOSURE=(
  src
  data
  train.py
  eval_FLAC.py
  finetune_cond.py
  eval_pl.py
  eval_VAE.py
  unwrap_model.py
  defaults.ini
  pyproject.toml
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_eval.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_train.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_submit.sh
)

die() { echo "GATEFAIL rc=${2} ${1}" >&2; exit "${2}"; }

usage() {
  echo "usage: della_submit.sh eval {unseen_s42|unseen_s43|seen_s42}" >&2
  echo "       della_submit.sh train [--smoke] [--resume] [--time D-HH:MM:SS]" >&2
}

# Slurm accepts HH:MM:SS with HH > 23, but MM and SS are minutes and seconds:
# "99:99:99" is a typo, not a 4-day budget, and must not become one.
validate_time() {   # <value>; 0 iff D-HH:MM:SS or HH:MM:SS with MM,SS <= 59
  local t="$1" days="" hms hh mm ss
  case "$t" in
    *-*) days="${t%%-*}"; hms="${t#*-}" ;;
    *)   hms="$t" ;;
  esac
  if [ -n "$days" ]; then
    case "$days" in ''|*[!0-9]*) return 1 ;; esac
  fi
  case "$hms" in
    [0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9][0-9]:[0-9][0-9]:[0-9][0-9]) ;;
    *) return 1 ;;
  esac
  hh="${hms%%:*}"; mm="${hms#*:}"; mm="${mm%%:*}"; ss="${hms##*:}"
  [ "$((10#$mm))" -le 59 ] || return 1
  [ "$((10#$ss))" -le 59 ] || return 1
  # In the D-HH:MM:SS form the hour field is an hour-of-day and cannot exceed 23.
  if [ -n "$days" ]; then
    [ "$((10#$hh))" -le 23 ] || return 1
  fi
  return 0
}

# --- A. arguments -------------------------------------------------------------
KIND="${1:-}"; shift || true
case "$KIND" in
  eval|train) ;;
  *) usage; die "first argument must be 'eval' or 'train' (got '${KIND}')" 2 ;;
esac

CELL=""; SMOKE=0; RESUME=0; TIME_LIMIT=""
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
      --smoke)  SMOKE=1; shift ;;
      --resume) RESUME=1; shift ;;
      --time)   TIME_LIMIT="${2:-}"; shift 2 || die "--time needs a value" 2 ;;
      --time=*) TIME_LIMIT="${1#--time=}"; shift ;;
      *) usage; die "unknown train option '${1}'" 2 ;;
    esac
  done
  [ "$SMOKE" = "0" ] || [ "$RESUME" = "0" ] || die "--smoke and --resume are mutually exclusive: a smoke always starts fresh" 2
  if [ "$SMOKE" = "1" ]; then
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
  else
    # No default on purpose: a full leg's wall-clock budget is a measured
    # quantity (smoke rate x1.5), and a defaulted one would silently truncate
    # a multi-day run at whatever the default happened to be.
    [ -n "$TIME_LIMIT" ] || die "a full training leg needs an explicit --time D-HH:MM:SS (smoke defaults to 01:00:00)" 2
  fi
  validate_time "$TIME_LIMIT" \
    || die "--time '${TIME_LIMIT}' is not D-HH:MM:SS or HH:MM:SS with minutes/seconds <= 59" 2
fi

# --- B. the measurement closure must be clean --------------------------------
# Scoped, not whole-tree (review B3): this script APPENDS to the command record
# in the same directory, and the drivers write their logs there while jobs are
# queued, so a whole-tree gate would make the kit unable to record itself.
CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
  || die "git status FAILED - refusing to treat an error as a clean closure" 3
if [ -n "$CLOSURE_DIRT" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: the measurement closure is dirty (a real submit aborts here):"
    echo "$CLOSURE_DIRT"
  else
    echo "$CLOSURE_DIRT" >&2
    die "the measurement closure is dirty - commit or stash before submitting" 3
  fi
fi

# --- B2. the Phase-1 verdict gates all production training (plan §5, Rev 3 §2) --
# Fatal even under DRYRUN: this is not an argv-formatting question but the
# experiment's own precondition, and a dry run that "passed" without it would be
# read as evidence the leg is ready to go. --smoke is exempt because the smoke is
# what PRICES the leg (it must be runnable before Phase 1 concludes); eval is
# exempt because eval IS Phase 1.
if [ "$KIND" = "train" ] && [ "$SMOKE" != "1" ]; then
  [ -f "$REPO/$PHASE1_PASS_REL" ] \
    || die "the Phase-1 gate (plan §5) has no verdict: ${PHASE1_PASS_REL} does not exist — Phase 1 gates all training compute; --smoke is exempt" 8
  git -C "$REPO" cat-file -e "HEAD:${PHASE1_PASS_REL}" 2>/dev/null \
    || die "the Phase-1 verdict ${PHASE1_PASS_REL} is not tracked at HEAD — an uncommitted verdict is not a reviewable gate (plan §5 / Rev 3 §2)" 8
  echo "Phase-1 gate OK: ${PHASE1_PASS_REL} exists and is tracked at HEAD"
fi

# --- C. HEAD must be PUSHED ---------------------------------------------------
# The node re-checks EXPECT_SHA against the checkout at run time; binding a job
# to a commit that exists only on this filesystem makes that check unverifiable
# by anyone else (and unrecoverable if the checkout moves). ls-remote is a
# READ-ONLY query: unlike `git fetch` it writes no FETCH_HEAD, so a DRYRUN leaves
# the filesystem byte-identical. A failed query is FATAL, never assumed-OK.
EXPECT_SHA="$(git -C "$REPO" rev-parse HEAD)" || die "git rev-parse HEAD FAILED" 4
REMOTE_SHA="$(git -C "$REPO" ls-remote "$REMOTE" "refs/heads/${BRANCH}" | awk '{print $1}')" \
  || die "git ls-remote ${REMOTE} refs/heads/${BRANCH} FAILED - cannot prove HEAD is pushed" 4
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
    --hold --parsable
    --job-name="$JOB_NAME"
    --export="EXPECT_SHA=${EXPECT_SHA},CELL=${CELL}"
    "$EXPDIR/della_eval.sbatch"
  )
else
  JOB_NAME="exp16-train"
  EXPORTS="EXPECT_SHA=${EXPECT_SHA}"
  if [ "$SMOKE" = "1" ]; then
    JOB_NAME="exp16-train-smoke"
    EXPORTS="${EXPORTS},SMOKE=1"
  fi
  if [ "$RESUME" = "1" ]; then
    JOB_NAME="exp16-train-resume"
    EXPORTS="${EXPORTS},RESUME=1"
  fi
  SBATCH_ARGV=(
    --hold --parsable
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

# --- E. submit HELD -----------------------------------------------------------
# Held, so the record below is written while the job is queued but CANNOT start.
SBATCH_OUT="$(sbatch "${SBATCH_ARGV[@]}")" || die "sbatch FAILED - nothing submitted" 5
JOBID="${SBATCH_OUT%%;*}"        # --parsable prints "<jobid>[;<cluster>]"
case "$JOBID" in
  ''|*[!0-9]*)
    echo "sbatch returned: '${SBATCH_OUT}'" >&2
    die "sbatch output is not a numeric job id - the job (if any) cannot be cancelled by name; check squeue" 5 ;;
esac
echo "submitted HELD as ${JOBID}"

# --- F. record it, then verify the record, then release ----------------------
# flock serialises appends against a second submitter; the read-back is what makes
# "recorded" a fact rather than an assumption (a full disk truncates silently).
record_submission() {
  {
    flock -w 30 200 || return 1
    if [ ! -s "$RECORD" ]; then
      {
        echo "# della_vanilla_repro — submission command record"
        echo
        echo "Append-only, written by \`della_submit.sh\` at submit time (plan §4d): the exact"
        echo "sbatch line, the commit the job is bound to, and the job id Slurm returned."
        echo "Each job is submitted HELD, recorded here, and only then released."
        echo
      } >&200 || return 1
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
    } >&200 || return 1
  } 200>>"$RECORD"
  grep -q "job id: \`${JOBID}\`" "$RECORD"
}

if ! record_submission; then
  echo "command-record write/verify FAILED for job ${JOBID} - cancelling it" >&2
  scancel "$JOBID" || echo "scancel FAILED - cancel job ${JOBID} by hand" >&2
  die "no released job may lack its record in ${RECORD}" 6
fi
echo "recorded in ${RECORD}"

if ! scontrol release "$JOBID"; then
  echo "scontrol release FAILED for ${JOBID} - cancelling it; its record block stays as evidence" >&2
  scancel "$JOBID" || echo "scancel FAILED too - job ${JOBID} is HELD; release or cancel it by hand" >&2
  die "could not release job ${JOBID}" 7
fi

echo "$JOBID"
