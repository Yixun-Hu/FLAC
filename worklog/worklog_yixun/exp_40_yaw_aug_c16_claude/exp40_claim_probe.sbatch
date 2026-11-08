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
# exp40_claim_probe.sbatch -- plan §14 B1 / code-r1 B3: two tasks on two DIFFERENT nodes race
# an O_CREAT|O_EXCL create of the same claim path inside the ONE neuronic control directory's
# claims/ (the directory the training jobs claim in), 20 rounds, each round fenced by a file
# barrier before and after the race. Each task records its own view of the mount (findmnt
# SOURCE/FSTYPE + st_dev). The verdict is written atomically (fsync'd) by exp40_kit.py
# probe-verdict to <control>/receipts/claim_probe.json and holds the hosts, the 20 per-round
# winners, the job id, the sha256 of THIS committed script, the source commit, the tested
# directory and the time; the submitter validates every field (probe_receipt_reasons) and
# its age (<= 14 d) before any FULL; it is bound to the source commit it ran from, so run it
# from the clean checkout at the FULL's EXPECT_SHA. CPU only. Submit from that checkout:
#   sbatch --output=<repo>/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/slurm_claim_probe_%j.out \
#          worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_claim_probe.sbatch
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan §14 B1 and code review 1 B3.
# ============================================================================
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:?}"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
PY="/n/fs/gatrdp/envs/flac/bin/python"
CONTROL="/n/fs/gatrdp/exp40_control"            # THE control directory (fixed; code-r2 B1)
KIT="$REPO/$EXPREL/exp40_kit.py"
SCRIPT="$REPO/$EXPREL/exp40_claim_probe.sbatch"      # the committed copy (Slurm runs a spooled one)
DIR="$CONTROL/claims/probe_${SLURM_JOB_ID}"
mkdir -p "$DIR" "$CONTROL/receipts" || { echo "REFUSED: cannot create $DIR"; exit 2; }
srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 "$PY" "$KIT" probe-task --dir "$DIR"
SRUN_RC=$?
"$PY" "$KIT" probe-verdict --dir "$DIR" --job-id "$SLURM_JOB_ID" \
  --script "$SCRIPT" --source-sha "$(git -C "$REPO" rev-parse HEAD)" --srun-rc "$SRUN_RC"
