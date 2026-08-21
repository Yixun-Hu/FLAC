#!/usr/bin/env bash
# ============================================================================
# della_repro_eval_submit.sh — submit the exp_16 Phase-3 evaluation of the
# REPRODUCTION checkpoint as one recorded transaction.
#
# Usage:
#   della_repro_eval_submit.sh all           # every registered cell (14)
#   della_repro_eval_submit.sh all67500      # only the headline-checkpoint cells (12)
#   della_repro_eval_submit.sh u8_s42 u1_s43 # any explicit subset
#
# Cells are INDEPENDENT: no --dependency, no mutual exclusion. They only read the
# checkpoint, so running them concurrently is the point.
#
# VENUE: cells run on ailab/H200 (the header's own directives). The single
# arch-spot-check cell u8_s42_a100 is submitted with
#   --partition= --qos= --gres=gpu:a100:1
# Empty CLI values UNSET the in-file --partition/--qos, after which della
# auto-routes from the GRES; verified with `sbatch --test-only`, which placed the
# job "in partition gpu" on an A100 node. An explicit `--partition=gpu` is
# rejected by della ("You specified a partition of gpu. This is not allowed."),
# and keeping ailab with an A100 GRES fails as an impossible configuration — so
# the empty-value route is the only working one, and it needs no second script.
#
# GATES: closure clean, HEAD pushed (ls-remote, fatal), the headline checkpoint
# present and non-empty. There is deliberately NO PHASE1_PASS interlock here —
# that gate exists to stop unjustified TRAINING compute, and Phase 3 is the
# read-only measurement of a finished run.
#
# TRANSACTION: every cell is submitted HELD, one dated block naming every cell and
# id is appended to the command record under flock and read back, and only then is
# every id released. Any failure cancels every id created.
#
# Exit codes: 2 usage, 3 closure dirty, 4 HEAD not pushed, 5 sbatch failed or
# unparsable, 6 record write/verify failed, 7 release failed, 10 checkpoint
# missing/empty, 12 another submission holds the lock.
# ============================================================================
set -euo pipefail

REPO=/home/yh4742/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"
BRANCH=della-flac-chequity
REMOTE=origin
RECORD="$EXPDIR/della_vanilla_repro_command.md"
CELL_SBATCH="$EXPDIR/della_repro_eval.sbatch"
SUBMIT_LOCK="$EXPDIR/.repro_eval_submit.lock"
CKD=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro/FLAC_vanilla_repro/exp16_della_vanilla_repro/checkpoints
CKPT_67500="$CKD/epoch=14-step=67500.ckpt"
JOB_PREFIX=exp16-reval
DRYRUN="${DRYRUN:-0}"
export PYTHONDONTWRITEBYTECODE=1

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
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_repro_eval.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_repro_eval_submit.sh
)

# Registered cells, in submission order. ALL67500 is the subset that measures the
# headline artifact (everything except the two endpoint-draw screens).
ALL_CELLS=(u8_s42 u8_s43 u8_s44 u8_s45 u8_s46
           u1_s42 u1_s43 u1_s44 u1_s45 u1_s46
           u8_s42_step62500 u8_s42_step65000
           seen_s42
           u8_s42_a100)
ALL67500_CELLS=(u8_s42 u8_s43 u8_s44 u8_s45 u8_s46
                u1_s42 u1_s43 u1_s44 u1_s45 u1_s46
                seen_s42
                u8_s42_a100)

die() { echo "GATEFAIL rc=${2} ${1}" >&2; exit "${2}"; }
usage() {
  echo "usage: della_repro_eval_submit.sh {all | all67500 | CELL [CELL...]}" >&2
  echo "       cells: ${ALL_CELLS[*]}" >&2
}

# --- ONE LOCK OVER THE WHOLE TRANSACTION -------------------------------------
# Two concurrent invocations would each submit the same cells, and duplicate cells
# would race to write one metrics JSON. A DRYRUN takes no lock (and so creates no
# lockfile), because it submits nothing.
if [ "${REPRO_EVAL_LOCK_HELD:-0}" != "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
  export REPRO_EVAL_LOCK_HELD=1
  RC=0
  flock -w 10 -E 99 "$SUBMIT_LOCK" bash "$0" "$@" || RC=$?
  [ "$RC" -ne 99 ] || die "another Phase-3 eval submission is in progress (could not take ${SUBMIT_LOCK} within 10s)" 12
  exit "$RC"
fi

# --- A. arguments -------------------------------------------------------------
[ "$#" -gt 0 ] || { usage; die "no cells requested" 2; }
CELLS=()
case "$1" in
  all)      [ "$#" -eq 1 ] || die "'all' takes no further arguments" 2;      CELLS=("${ALL_CELLS[@]}") ;;
  all67500) [ "$#" -eq 1 ] || die "'all67500' takes no further arguments" 2; CELLS=("${ALL67500_CELLS[@]}") ;;
  *)
    for c in "$@"; do
      ok=0
      for k in "${ALL_CELLS[@]}"; do [ "$c" != "$k" ] || ok=1; done
      [ "$ok" = "1" ] || { usage; die "cell '${c}' is not registered" 2; }
      CELLS+=("$c")
    done ;;
esac
echo "cells (${#CELLS[@]}): ${CELLS[*]}"

# --- B. the closure must be clean --------------------------------------------
CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
  || die "git status FAILED - refusing to treat an error as a clean closure" 3
if [ -n "$CLOSURE_DIRT" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: the eval closure is dirty (a real submit aborts here):"
    echo "$CLOSURE_DIRT"
  else
    echo "$CLOSURE_DIRT" >&2
    die "the eval closure is dirty - commit or stash before submitting" 3
  fi
fi

# --- C. the headline checkpoint must exist ------------------------------------
if [ ! -s "$CKPT_67500" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: ${CKPT_67500} is missing or empty (a real submit aborts here)"
  else
    die "the reproduction checkpoint ${CKPT_67500} is missing or empty" 10
  fi
else
  echo "checkpoint OK: ${CKPT_67500} ($(stat -c '%s' "$CKPT_67500") bytes)"
fi

# --- D. HEAD must be PUSHED ---------------------------------------------------
# ls-remote is a read-only query: unlike `git fetch` it writes no FETCH_HEAD, so a
# DRYRUN leaves the filesystem byte-identical. A failed query is FATAL.
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

# --- E. build every leg's sbatch argv -----------------------------------------
# Only the A100 cell carries venue overrides; every other cell runs on the file's
# own ailab/H200 directives.
build_argv() {   # <cell> -> fills SBATCH_ARGV
  SBATCH_ARGV=(--hold --parsable --job-name="${JOB_PREFIX}-${1}")
  if [ "$1" = "u8_s42_a100" ]; then
    SBATCH_ARGV+=(--partition= --qos= --gres=gpu:a100:1)
  fi
  SBATCH_ARGV+=(--export="EXPECT_SHA=${EXPECT_SHA},CELL=${1}" "$CELL_SBATCH")
}

if [ "$DRYRUN" = "1" ]; then
  for c in "${CELLS[@]}"; do
    build_argv "$c"
    echo "sbatch ${SBATCH_ARGV[*]}"
  done
  echo "DRY RUN: nothing submitted, command record untouched (EXPECT_SHA would be ${EXPECT_SHA})"
  exit 0
fi

# --- F. submit every cell HELD ------------------------------------------------
JOBIDS=(); SUBMITTED_CELLS=(); SBATCH_LINES=()
abort_transaction() {   # <message>
  if [ "${#JOBIDS[@]}" -gt 0 ]; then
    echo "cancelling the partial submission: ${JOBIDS[*]}" >&2
    scancel "${JOBIDS[@]}" || echo "scancel FAILED - cancel ${JOBIDS[*]} by hand" >&2
  fi
  die "$1" 5
}
for c in "${CELLS[@]}"; do
  build_argv "$c"
  OUT="$(sbatch "${SBATCH_ARGV[@]}")" || abort_transaction "sbatch FAILED for cell ${c}"
  ID="${OUT%%;*}"
  case "$ID" in
    ''|*[!0-9]*) echo "sbatch returned: '${OUT}'" >&2; abort_transaction "cell ${c}: sbatch output is not a numeric job id" ;;
  esac
  JOBIDS+=("$ID"); SUBMITTED_CELLS+=("${c}=${ID}"); SBATCH_LINES+=("sbatch ${SBATCH_ARGV[*]}")
  echo "cell ${c} submitted HELD as ${ID}"
done

# --- G. record it, verify it, then release ------------------------------------
write_record() {
  {
    flock -w 30 200 || return 1
    if [ ! -s "$RECORD" ]; then
      { echo "# della_vanilla_repro — submission command record"; echo; } >&200 || return 1
    fi
    {
      echo "## $(date -Is) — ${JOB_PREFIX} (Phase 3) — ${#CELLS[@]} cells — jobs ${JOBIDS[*]}"
      echo
      echo "- EXPECT_SHA: \`${EXPECT_SHA}\`"
      echo "- checkpoint: \`${CKPT_67500}\` (+ the 62500/65000 endpoint screens where requested)"
      echo "- cells: \`${SUBMITTED_CELLS[*]}\`"
      echo "- submitted by: \`$(whoami)@$(hostname)\`"
      echo
      echo '```bash'
      printf '%s\n' "${SBATCH_LINES[@]}"
      echo '```'
      echo
    } >&200 || return 1
  } 200>>"$RECORD"
  grep -q "jobs ${JOBIDS[*]}" "$RECORD" || return 1
}
if ! write_record; then
  echo "command-record write or verify FAILED - cancelling every submitted cell" >&2
  scancel "${JOBIDS[@]}" || echo "scancel FAILED - cancel ${JOBIDS[*]} by hand" >&2
  die "no released cell may lack its record in ${RECORD}" 6
fi
echo "recorded in ${RECORD}"

# --- H. release ----------------------------------------------------------------
if ! scontrol release "${JOBIDS[@]}"; then
  echo "scontrol release FAILED - cancelling the cells; the record block stays as evidence" >&2
  scancel "${JOBIDS[@]}" || echo "scancel FAILED too - jobs ${JOBIDS[*]} are HELD; release or cancel by hand" >&2
  die "could not release the submitted cells" 7
fi
echo "released ${#JOBIDS[@]} cells: ${JOBIDS[*]}"
printf '%s\n' "${SUBMITTED_CELLS[@]}"
