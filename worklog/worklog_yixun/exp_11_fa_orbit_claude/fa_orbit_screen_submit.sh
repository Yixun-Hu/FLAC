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
#   2. LEASE    — written BEFORE sbatch can return, so no pruner can ever see
#                 the tree as free while a job for it is queued. It is written
#                 under a placeholder token and promoted to the real job id the
#                 moment sbatch reports it; if sbatch fails, it is released.
#   3. EXPECT_SHA — taken from the pinned tree, not from the (moving) main HEAD
#
#   bash fa_orbit_screen_submit.sh ARM=C4L STEP=10000 [SEED=42] [K=8] [CELL=screen]
#
# Prints the job id on success. Nothing here submits by itself: it needs the
# ARM/STEP arguments and an explicit run.
# ============================================================================
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

# 2. lease under a placeholder BEFORE sbatch
PENDING="pending-$$-$(date +%s)"
bash "$HELPER" --lease "$PENDING" "$WT"
cleanup() { bash "$HELPER" --release "$PENDING" "$WT" 2>/dev/null || true; }
trap cleanup EXIT

JOB_NAME="exp11-screen-${ARM}-${STEP}-s${SEED}-K${K}"
JOBID="$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --output="${EXPDIR}/slurm_screen_%x_%j.out" \
  --export=ALL,MEASURE_ROOT="$WT",EXPECT_SHA="$EXPECT_SHA",ARM="$ARM",STEP="$STEP",SEED="$SEED",K="$K",CELL="$CELL" \
  "$EXPDIR/fa_orbit_screen.sbatch")" || { echo "sbatch FAILED - lease released" >&2; exit 4; }
JOBID="${JOBID%%;*}"

# 3. promote the placeholder to the real job id (the job validates this at start)
bash "$HELPER" --promote "$PENDING" "$JOBID" "$WT"
trap - EXIT

echo "submitted ${JOB_NAME} as ${JOBID}"
echo "  MEASURE_ROOT ${WT}"
echo "  EXPECT_SHA   ${EXPECT_SHA}"
echo "  lease        ${WT}/.leases/${JOBID}"
