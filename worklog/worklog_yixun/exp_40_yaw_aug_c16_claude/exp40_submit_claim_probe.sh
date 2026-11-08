#!/usr/bin/env bash
# ============================================================================
# exp40_submit_claim_probe.sh -- submit the two-node claim probe (plan §14 B1) from an IMMUTABLE
# run tree at EXPECT_SHA (code-r3 H2). Run by Yixun on a neuronic login node from the clean
# checkout of branch exp-40-yaw-aug-c16 at EXPECT_SHA (pushed). CPU only, ~1 min.
#   EXPECT_SHA=<40-hex> bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_submit_claim_probe.sh
# The receipt is bound to EXPECT_SHA: rerun it for every new launch commit (valid 14 days).
# Written by the exp_40 Coder seat (Claude Opus 5.5) per code review 3 HIGH 2.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }
SELF_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(git -C "$SELF_DIR" rev-parse --show-toplevel)" || die "not inside a git checkout"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
PY="/n/fs/gatrdp/envs/flac/bin/python"
EXPECT_SHA="${EXPECT_SHA:-}"
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_SHA must be the full 40-hex commit"
cd "$REPO" || die "cd $REPO"
[ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || die "HEAD $(git rev-parse HEAD) != EXPECT_SHA"
git fetch -q origin exp-40-yaw-aug-c16 || die "git fetch failed"
git merge-base --is-ancestor "$EXPECT_SHA" FETCH_HEAD || die "EXPECT_SHA is not pushed to origin/exp-40-yaw-aug-c16"
[ -z "$(git status --porcelain --untracked-files=all -- "$EXPREL"/*.py "$EXPREL"/*.sh "$EXPREL"/*.sbatch)" ] \
  || die "the kit / probe scripts are not committed"
SAVE_ROOT="$REPO/outputs_FLAC/exp40_yaw_aug_c16"; mkdir -p "$SAVE_ROOT/runtrees" || die "dirs"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
TREE="$SAVE_ROOT/runtrees/claim_probe__${EXPECT_SHA:0:12}__${TS}"
AR_REAL="$(readlink -f AcousticRooms)" || die "AcousticRooms link missing"
VAE_REAL="$(readlink -f weights/FLAC/VAE.safetensors)" || die "VAE missing"
"$PY" "$EXPREL/exp40_kit.py" build-runtree --repo "$REPO" --sha "$EXPECT_SHA" --dest "$TREE" \
  --link "AcousticRooms=$AR_REAL" --link "weights/FLAC/VAE.safetensors=$VAE_REAL" || die "probe tree"
SCRIPT="$TREE/$EXPREL/exp40_claim_probe.sbatch"
SCRIPT_SHA="$(sha256sum "$SCRIPT" | cut -d' ' -f1)" || die "sha256"
sbatch --parsable --output="$REPO/$EXPREL/slurm_claim_probe_%j.out" \
  --export=ALL,EXP40_PROBE_TREE="$TREE",EXP40_EXPECT_SHA="$EXPECT_SHA",EXP40_PROBE_SCRIPT_SHA256="$SCRIPT_SHA",EXP40_REPO="$REPO" \
  "$SCRIPT" | tee -a "$REPO/$EXPREL/yaw_aug_c16_neuronic_submissions.log"
exit "${PIPESTATUS[0]}"
