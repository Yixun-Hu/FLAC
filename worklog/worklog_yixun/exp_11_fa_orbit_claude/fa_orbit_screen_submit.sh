#!/usr/bin/env bash
# ============================================================================
# fa_orbit_screen_submit.sh — submit ONE exp_11 screen against a pinned,
# LEASED measurement worktree.
#
# Screens used to be submitted by hand, which is how job 3649599 ended up
# reading code that moved underneath it. Submitting through this script makes
# the three things that must be atomic actually atomic:
#
#   0. LOCK SPAN — the whole sequence runs inside the store lock (the script
#                  re-execs itself under `fa_orbit_measure_worktree.sh
#                  --with-lock`). Locking each step separately still left a gap
#                  between "tree prepared" and "lease written" in which a prune
#                  sweep sees a brand-new unleased tree and deletes it.
#   1. pin      — a worktree at the submission SHA, assets provisioned
#   2. LEASE     — via sbatch --hold. Submitting held is what removes the last
#                  two races: the job id exists BEFORE the lease is written, and
#                  the job cannot start until it is released, so there is no
#                  window in which a queued job has no lease (a pruner would
#                  have seen its tree as free) and none in which a lease names a
#                  placeholder instead of a job. If the lease write or the
#                  release fails, the held job is CANCELLED — never orphaned.
#   3. EXPECT_SHA — taken from the pinned tree, not from the (moving) main HEAD
#
set -euo pipefail
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$MAIN_REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
HELPER="$EXPDIR/fa_orbit_measure_worktree.sh"

# --- OUTER ENTRY: prove the lock before anything transactional happens -------
# Re-exec under the store lock so that preparation, submission and leasing are
# ONE atomic span. fd 8 is inherited, so the nested helper calls below re-enter
# the same lock instead of deadlocking on a second one.
#
# The marker alone is not proof. It is an ordinary environment variable, and
# anything that inherits it — a stale export, a wrapper, a rerun of a crashed
# shell — would carry it into a run holding no lock at all. Verifying it inside
# the helper is too late: by then this script has already prepared a tree. So
# the check lives HERE, at the outer entry, before any transaction step.
LOCKFILE="${MAIN_REPO}/.measure_worktrees/.store.lock"

fd8_is_the_store_lock() {
  local have want
  [ -e /proc/self/fd/8 ] || return 1
  have="$(readlink -f /proc/self/fd/8 2>/dev/null)" || return 1
  want="$(readlink -f "$LOCKFILE" 2>/dev/null)" || return 1
  # An unresolvable path yields the EMPTY string, and two empty strings compare
  # equal — which would turn a double failure into a "match". Require both.
  [ -n "$have" ] && [ -n "$want" ] && [ "$have" = "$want" ]
}

if [ "${FA_ORBIT_STORE_LOCK_HELD:-0}" = "1" ]; then
  fd8_is_the_store_lock || {
    echo "FA_ORBIT_STORE_LOCK_HELD is set but fd 8 is not ${LOCKFILE} —" >&2
    echo "refusing to run a submission that only CLAIMS to hold the store lock" >&2
    exit 2
  }
else
  exec bash "$HELPER" --with-lock bash "$0" "$@"
fi

ARM=""; STEP=""; SEED=42; K=8; CELL=screen; EXCLUDE=""; ROTATE_DEG=""; EVAL_ORBIT=""
for kv in "$@"; do
  case "$kv" in
    ARM=*|STEP=*|SEED=*|K=*|CELL=*|EXCLUDE=*|ROTATE_DEG=*|EVAL_ORBIT=*) eval "${kv%%=*}='${kv#*=}'" ;;
    *) echo "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=/EXCLUDE=/ROTATE_DEG=/EVAL_ORBIT=)" >&2; exit 2 ;;
  esac
done
[ -n "$ARM" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C4L STEP=10000 [SEED=42] [K=8] [CELL=screen] [EXCLUDE=node[,node]] [ROTATE_DEG=..] [EVAL_ORBIT=..]" >&2; exit 2; }
case "$EXCLUDE" in *[!A-Za-z0-9,._\[\]-]*) echo "EXCLUDE='${EXCLUDE}' is not a node list" >&2; exit 2 ;; esac

# 0b. PREFLIGHT: the campaign freeze must be engaged.
# The campaign's validity argument rests on "no worktree is deleted while it
# runs". Leaving that to operator discipline makes it a promise; requiring the
# marker here makes it a precondition — a submission cannot happen unless the
# guarantee is mechanically in force.
FREEZE_MARKER="${MAIN_REPO}/.measure_worktrees/.campaign_freeze"
if [ -e "$FREEZE_MARKER" ]; then
  echo "campaign freeze: ACTIVE — $(head -1 "$FREEZE_MARKER" 2>/dev/null)"
else
  echo "campaign freeze: ABSENT" >&2
  echo "refusing to submit: a measurement campaign requires the deletion freeze." >&2
  echo "  engage it with: bash ${HELPER} --freeze" >&2
  exit 2
fi

# 1. pin + assets (the helper refuses a dirty or mismatched tree)
WT="$("$HELPER" | tail -1)"
[ -d "$WT" ] || { echo "could not prepare a measurement worktree" >&2; exit 3; }
EXPECT_SHA="$(git -C "$WT" rev-parse HEAD)"

# 2. submit HELD: the id exists before the lease, the job runs after it
JOB_NAME="exp11-screen-${ARM}-${CELL}-${STEP}-s${SEED}-K${K}"
SBATCH="${FA_ORBIT_SBATCH:-sbatch}"          # guard-suite seam; a real run uses sbatch
SCONTROL="${FA_ORBIT_SCONTROL:-scontrol}"
SCANCEL="${FA_ORBIT_SCANCEL:-scancel}"
# Node exclusion is passed as an EXPLICIT FLAG, never through the environment.
# SBATCH_EXCLUDE does not exist: of the 58 input environment variables sbatch
# documents there is no --exclude equivalent (the lookalike SBATCH_EXCLUSIVE is
# --exclusive, a different option entirely). sbatch therefore ignored it in
# silence, and every batch that believed it was excluding sick nodes was not.
EXCLUDE_ARGV=()
if [ -n "$EXCLUDE" ]; then
  EXCLUDE_ARGV=(--exclude="$EXCLUDE")
  echo "excluding nodes: ${EXCLUDE}"
elif [ -n "${SBATCH_EXCLUDE:-}" ]; then
  echo "NOTE: SBATCH_EXCLUDE='${SBATCH_EXCLUDE}' is set but sbatch does not honour it;" >&2
  echo "      pass EXCLUDE=${SBATCH_EXCLUDE} to this script instead - abort" >&2
  exit 2
fi
# CELL-specific parameters travel with the job, not as ambient state.
CELL_EXPORT=""
[ -n "$ROTATE_DEG" ] && CELL_EXPORT="${CELL_EXPORT},ROTATE_DEG=${ROTATE_DEG}"
[ -n "$EVAL_ORBIT" ] && CELL_EXPORT="${CELL_EXPORT},EVAL_ORBIT=${EVAL_ORBIT}"
JOBID="$("$SBATCH" --hold --parsable \
  --job-name="$JOB_NAME" \
  --output="${EXPDIR}/slurm_screen_%x_%j.out" \
  "${EXCLUDE_ARGV[@]}" \
  --export=ALL,MEASURE_ROOT="$WT",EXPECT_SHA="$EXPECT_SHA",ARM="$ARM",STEP="$STEP",SEED="$SEED",K="$K",CELL="$CELL""$CELL_EXPORT" \
  "$EXPDIR/fa_orbit_screen.sbatch")" || { echo "sbatch FAILED - nothing submitted" >&2; exit 4; }
JOBID="${JOBID%%;*}"
case "$JOBID" in ''|*[!0-9]*) echo "sbatch returned '${JOBID}', not a job id - abort" >&2; exit 4 ;; esac
echo "submitted HELD as ${JOBID}"

# 3. lease it by its real id, VALIDATE the lease, and only THEN release the job.
#    All of this still runs inside the store lock taken at step 0.
#
#    THE LEASE IS NEVER DROPPED ON AN ERROR PATH. Once a lease exists, the only
#    safe thing to do with an ambiguous outcome is to keep it: if scancel failed,
#    or succeeded but we cannot be sure, the job may yet run — and a job whose
#    lease we tidied away is a job whose worktree a sweep will delete underneath
#    it. A lease that outlives its job costs one held worktree until the reaper
#    proves, via a successful squeue, that the id is gone. That asymmetry is the
#    whole design: retaining is cheap, dropping is unrecoverable.
#
#    add_lease re-checks the worktree's identity before writing: a lease on a
#    directory that is not a live registered worktree would send the job to a
#    tree with no code in it.
if ! bash "$HELPER" --lease "$JOBID" "$WT"; then
  echo "could not write the lease for ${JOBID} - cancelling the held job" >&2
  "$SCANCEL" "$JOBID" || echo "scancel FAILED - job ${JOBID} is held and UNLEASED, cancel it by hand" >&2
  exit 5
fi

# Validate BEFORE releasing: while the job is still held, a bad lease costs a
# cancellation; after release it is a running job we cannot vouch for.
if ! grep -q "^jobid ${JOBID}$" "${WT}/.leases/${JOBID}" 2>/dev/null; then
  echo "lease ${WT}/.leases/${JOBID} does not name ${JOBID} - cancelling (lease RETAINED)" >&2
  "$SCANCEL" "$JOBID" || echo "scancel FAILED - cancel job ${JOBID} by hand" >&2
  exit 7
fi
echo "lease validated: ${WT}/.leases/${JOBID}"

if ! "$SCONTROL" release "$JOBID"; then
  echo "could not release ${JOBID} - cancelling it; the lease is RETAINED" >&2
  if "$SCANCEL" "$JOBID"; then
    echo "cancelled ${JOBID}; its lease stays until squeue proves the id is gone" >&2
  else
    echo "scancel FAILED too - job ${JOBID} may still run; cancel it by hand." >&2
    echo "The lease is deliberately kept: dropping it would let a sweep delete" >&2
    echo "the worktree out from under a job that is still alive." >&2
  fi
  exit 6
fi

echo "released ${JOB_NAME} (${JOBID})"
echo "  MEASURE_ROOT ${WT}"
echo "  EXPECT_SHA   ${EXPECT_SHA}"
echo "  lease        ${WT}/.leases/${JOBID}"
