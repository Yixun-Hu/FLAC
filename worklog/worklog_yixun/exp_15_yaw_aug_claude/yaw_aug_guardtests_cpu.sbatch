#!/bin/bash
# exp_15: run the eval-kit STRICT guard suite on a scheduled CPU node (login node saturated).
# CPU-only; the suite self-checks that it performs no sbatch of its own.
#SBATCH --job-name=exp15-guardtests-strict
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_15_yaw_aug_claude/slurm_guardtests_strict_%j.out
set -uo pipefail
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh && conda activate flac
# RUN_ROOT lets this run from a CLEAN WORKTREE at the fix commit rather than the
# dirty main checkout, so the transcript's tested HEAD is the committed state the
# reviewer will read (arming-check finding 4). Defaults to the main tree.
RUN_ROOT="${RUN_ROOT:-/n/fs/gatrdp/codespace/FLAC}"
cd "$RUN_ROOT" || exit 3
TS=$(date +%Y-%m-%d_%H-%M-%S)
EXPD=worklog/worklog_yixun/exp_15_yaw_aug_claude
# --- COMMIT AND CONTENT BINDING ----------------------------------------------
# HEAD alone is not enough: a dirty tree reports the same HEAD as a clean one.
# Digest the eight files this suite actually drives, so the transcript states
# exactly what was tested.
echo "=== tested-root : ${RUN_ROOT}"
echo "=== tested-HEAD : $(git -C "$RUN_ROOT" rev-parse HEAD)"
echo "=== tree-dirty  : $(if git -C "$RUN_ROOT" diff --quiet && git -C "$RUN_ROOT" diff --cached --quiet; then echo no; else echo YES; fi)"
echo "=== driven-file sha256:"
for f in "$EXPD/yaw_aug_screen.sbatch" "$EXPD/yaw_aug_screen_submit.sh" \
         "$EXPD/yaw_aug_submit_grid.sh" "$EXPD/yaw_aug_screen_guardtests.sh" \
         "$EXPD/exp15_validate_cell.py" "$EXPD/exp15_admit_ckpt.py" \
         "$EXPD/yaw_aug_collect.py" "$EXPD/yaw_aug_publish_row.py"; do
  printf '===   %s  %s\n' "$(sha256sum "$f" 2>/dev/null | cut -c1-16)" "$f"
done
# The suite is written for a login-shell environment: with SLURM_* set, the
# driver's under-sbatch branch (MEASURE_ROOT demand, yaw_aug_screen.sbatch:331)
# fires before the parameter gates the refusal cases assert on. Scrub them for
# the suite only — the suite itself never submits (self-checked).
for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset "$v"; done
# TAG is a parameter so each review round gets its own ledger namespace and the
# union check compares like with like. Default is this round's.
TAG="${GUARD_TAG:-evalfinal2}"
STRICT=1 GUARD_TAG="$TAG" bash "$EXPD/yaw_aug_screen_guardtests.sh" 2>&1 | tee "$EXPD/yaw_aug_${TS}_guardtests_${TAG}.log"
rc=${PIPESTATUS[0]}
python3 "$EXPD/yaw_aug_union_coverage.py" "$EXPD"/yaw_aug_*_guardtests_${TAG}.ledger 2>&1 | tee "$EXPD/yaw_aug_${TS}_union_${TAG}.log" || rc=1
echo "GUARDTESTS_RC=$rc"
exit $rc
