#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --job-name=exp40-claim-probe
# ============================================================================
# exp40_claim_probe.sbatch -- plan §14 B1 / code-r1 B3 / code-r2 B2 / code-r3 H2: two tasks on two
# DIFFERENT nodes race an O_CREAT|O_EXCL create of the same claim path inside THE neuronic control
# directory's claims/ (where the training jobs claim), 20 rounds, each fenced by a file barrier
# before and after the race; each task records its own view of the mount. Submitted ONLY by
# exp40_submit_claim_probe.sh, which builds an IMMUTABLE run tree at EXPECT_SHA and records the
# sha256 of this script: the job refuses unless the script Slurm actually runs (its spooled
# copy, "$0") is byte-identical to the tree's committed one and the tree verifies against git, and
# it runs the TREE's kit -- so a checkout that moves while the job is queued cannot change what
# the receipt claims. The verdict (atomic, fsync'd) goes to <control>/receipts/claim_probe.json.
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan §14 B1 and code reviews 1-3.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }
for v in EXP40_PROBE_TREE EXP40_EXPECT_SHA EXP40_PROBE_SCRIPT_SHA256 EXP40_REPO; do
  [ -n "${!v:-}" ] || die "$v must be exported by exp40_submit_claim_probe.sh"
done
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
PY="/n/fs/gatrdp/envs/flac/bin/python"
CONTROL="/n/fs/gatrdp/exp40_control"            # THE control directory (fixed; code-r2 B1)
TREE="$EXP40_PROBE_TREE"; KIT="$TREE/$EXPREL/exp40_kit.py"
"$PY" -B "$KIT" probe-preflight --repo "$EXP40_REPO" --sha "$EXP40_EXPECT_SHA" --tree "$TREE" \
  --script-self "$0" --script-sha256 "$EXP40_PROBE_SCRIPT_SHA256" || die "probe preflight refused"
DIR="$CONTROL/claims/probe_${SLURM_JOB_ID}"
mkdir -p "$DIR" "$CONTROL/receipts" || die "cannot create $DIR"
srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 "$PY" -B "$KIT" probe-task --dir "$DIR"
SRUN_RC=$?
"$PY" -B "$KIT" probe-verdict --dir "$DIR" --job-id "$SLURM_JOB_ID" \
  --script "$0" --source-sha "$EXP40_EXPECT_SHA" --srun-rc "$SRUN_RC"
