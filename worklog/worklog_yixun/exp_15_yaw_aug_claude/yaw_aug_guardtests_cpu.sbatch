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
cd /n/fs/gatrdp/codespace/FLAC
TS=$(date +%Y-%m-%d_%H-%M-%S)
EXPD=worklog/worklog_yixun/exp_15_yaw_aug_claude
# The suite is written for a login-shell environment: with SLURM_* set, the
# driver's under-sbatch branch (MEASURE_ROOT demand, yaw_aug_screen.sbatch:331)
# fires before the parameter gates the refusal cases assert on. Scrub them for
# the suite only — the suite itself never submits (self-checked).
for v in $(env | grep -o '^SLURM[A-Z_]*'); do unset "$v"; done
STRICT=1 GUARD_TAG=evalfinal bash "$EXPD/yaw_aug_screen_guardtests.sh" 2>&1 | tee "$EXPD/yaw_aug_${TS}_guardtests_evalfinal.log"
rc=${PIPESTATUS[0]}
python3 "$EXPD/yaw_aug_union_coverage.py" "$EXPD"/yaw_aug_*_guardtests_evalfinal.ledger 2>&1 | tee "$EXPD/yaw_aug_${TS}_union_evalfinal.log" || rc=1
echo "GUARDTESTS_RC=$rc"
exit $rc
