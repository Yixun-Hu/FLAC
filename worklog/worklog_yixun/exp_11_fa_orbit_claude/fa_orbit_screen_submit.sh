#!/usr/bin/env bash
# ============================================================================
# fa_orbit_screen_submit.sh — submit ONE exp_11 screen against a pinned,
# LEASED measurement worktree.
#
# Screens used to be submitted by hand, which is how job 3649599 ended up
# reading code that moved underneath it. Submitting through this script makes
# the three things that must be atomic actually atomic:
#
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

ARM=""; STEP=""; SEED=42; K=8; CELL=screen
for kv in "$@"; do
  case "$kv" in
    ARM=*|STEP=*|SEED=*|K=*|CELL=*) eval "${kv%%=*}='${kv#*=}'" ;;
    *) echo "unknown argument '${kv}' (expected ARM=/STEP=/SEED=/K=/CELL=)" >&2; exit 2 ;;
  esac
done
[ -n "$ARM" ] && [ -n "$STEP" ] || { echo "usage: bash $0 ARM=C4L STEP=10000 [SEED=42] [K=8] [CELL=screen]" >&2; exit 2; }

# 1. pin + assets (the helper refuses a dirty or mismatched tree)
WT="$("$HELPER" | tail -1)"
[ -d "$WT" ] || { echo "could not prepare a measurement worktree" >&2; exit 3; }
EXPECT_SHA="$(git -C "$WT" rev-parse HEAD)"

# 2. submit HELD: the id exists before the lease, the job runs after it
JOB_NAME="exp11-screen-${ARM}-${STEP}-s${SEED}-K${K}"
SBATCH="${FA_ORBIT_SBATCH:-sbatch}"          # guard-suite seam; a real run uses sbatch
SCONTROL="${FA_ORBIT_SCONTROL:-scontrol}"
SCANCEL="${FA_ORBIT_SCANCEL:-scancel}"
JOBID="$("$SBATCH" --hold --parsable \
  --job-name="$JOB_NAME" \
  --output="${EXPDIR}/slurm_screen_%x_%j.out" \
  --export=ALL,MEASURE_ROOT="$WT",EXPECT_SHA="$EXPECT_SHA",ARM="$ARM",STEP="$STEP",SEED="$SEED",K="$K",CELL="$CELL" \
  "$EXPDIR/fa_orbit_screen.sbatch")" || { echo "sbatch FAILED - nothing submitted" >&2; exit 4; }
JOBID="${JOBID%%;*}"
case "$JOBID" in ''|*[!0-9]*) echo "sbatch returned '${JOBID}', not a job id - abort" >&2; exit 4 ;; esac
echo "submitted HELD as ${JOBID}"

# 3. lease it by its real id, then release it. Any failure cancels the held job:
#    an unleased job would be one a sweep could pull the tree out from under.
if ! bash "$HELPER" --lease "$JOBID" "$WT"; then
  echo "could not write the lease for ${JOBID} - cancelling the held job" >&2
  "$SCANCEL" "$JOBID" || echo "scancel FAILED - job ${JOBID} is held and UNLEASED, cancel it by hand" >&2
  exit 5
fi
if ! "$SCONTROL" release "$JOBID"; then
  echo "could not release ${JOBID} - cancelling it and dropping its lease" >&2
  "$SCANCEL" "$JOBID" || echo "scancel FAILED - cancel job ${JOBID} by hand" >&2
  bash "$HELPER" --release "$JOBID" "$WT" || true
  exit 6
fi

echo "released ${JOB_NAME} (${JOBID})"
echo "  MEASURE_ROOT ${WT}"
echo "  EXPECT_SHA   ${EXPECT_SHA}"
echo "  lease        ${WT}/.leases/${JOBID}"
