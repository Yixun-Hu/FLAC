#!/usr/bin/env bash
# ============================================================================
# exp_11 P0 matrix submitter — plan Rev 3 §10 P0.1/P0.2.
#
# Submits the PAIRED (10-step, 30-step) profiling jobs for the throughput/fit
# matrix, so the collector can cancel startup cost per cell:
#   rungs {32x2, 16x4, 8x8} x orbits {C4L, C8} ......... rung + orbit scaling
#   VAN_{32x2,16x4,8x8} (canonical FLAC_AR.json) ....... vanilla baseline for
#                                                        the per-orbit-pass fit
#   CKPT4_32x2 (exp_07 FLAC_AR_BF.json, grad-ckpt ON) .. recompute cost
# C16/C32 spot cells are NOT submitted by default (plan: spot-check at the
# WINNING rung only, after the matrix lands):
#   ./p0_submit_matrix.sh spot 16x4
#
# Every job is bound to the current HEAD (EXPECT_SHA) and refuses to run if the
# tracked tree is dirty — same drift philosophy as exp_12's mem_probe. Nothing
# is submitted by sourcing this file; run it explicitly. DRYRUN=1 prints the
# sbatch commands without submitting.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH_FILE="$EXPDIR/p0_profile.sbatch"
EXP07="worklog/worklog_yixun/exp_07_fa_scratch_claude"
CANON="src/configs/model_configs/FLAC/AR/FLAC_AR.json"
DRYRUN="${DRYRUN:-0}"

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

# --- drift gate: a queued job must run reviewed, committed code ---
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD)"
echo "submitting P0 matrix at ${SHA}"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
MANIFEST="$EXPDIR/p0_manifest_${TS}.txt"
: > "$MANIFEST"
echo "# exp_11 P0 submission manifest - ${TS} - sha ${SHA}" >> "$MANIFEST"
echo "# cell maxsteps ngpu mb jobid" >> "$MANIFEST"

config_for() {  # family -> model config path
  case "$1" in
    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
    VAN)            echo "$CANON" ;;
    CKPT4)          echo "$EXP07/FLAC_AR_BF.json" ;;
    *)              echo "" ;;
  esac
}

submit_pair() {  # $1 = family, $2 = rung "MBxNGPU"
  local family="$1" rung="$2" mb ngpu cfg cell
  mb="${rung%x*}"; ngpu="${rung#*x}"
  cfg="$(config_for "$family")"
  [ -n "$cfg" ] || { echo "unknown family '${family}' - skip"; return 1; }
  [ -f "$cfg" ] || { echo "missing config ${cfg} - skip"; return 1; }
  [ "$((mb * ngpu))" -eq 64 ] || { echo "rung ${rung}: MB*NGPU != 64 - skip"; return 1; }
  cell="${family}_${rung}"
  for steps in 10 30; do
    local args=(
      --job-name="p0-${cell}-s${steps}"
      --gres="gpu:l40:${ngpu}"
      --cpus-per-task="$((8 + 7 * ngpu))"
      --mem="$((12 * ngpu + 12))G"
      --time=00:40:00
      --export="ALL,EXPECT_SHA=${SHA},CELL=${cell},MODEL_CONFIG=${cfg},NGPU=${ngpu},MB=${mb},MAXSTEPS=${steps}"
      "$SBATCH_FILE"
    )
    if [ "$DRYRUN" = "1" ]; then
      echo "DRYRUN sbatch ${args[*]}"
      echo "${cell} ${steps} ${ngpu} ${mb} DRYRUN" >> "$MANIFEST"
      continue
    fi
    local out jid
    out="$(sbatch "${args[@]}" 2>&1)"
    jid="$(echo "$out" | awk '/Submitted batch job/ {print $NF}')"
    if [ -z "$jid" ]; then
      echo "SUBMIT FAILED ${cell} s${steps}: ${out}"
      echo "${cell} ${steps} ${ngpu} ${mb} SUBMIT_FAILED" >> "$MANIFEST"
    else
      echo "submitted ${cell} s${steps} (${ngpu} GPU x MB ${mb}) -> job ${jid}"
      echo "${cell} ${steps} ${ngpu} ${mb} ${jid}" >> "$MANIFEST"
    fi
  done
}

MODE="${1:-matrix}"
case "$MODE" in
  matrix)
    for rung in 32x2 16x4 8x8; do
      for family in VAN C4L C8; do submit_pair "$family" "$rung"; done
    done
    submit_pair CKPT4 32x2
    ;;
  spot)
    RUNG="${2:?usage: $0 spot <RUNG e.g. 16x4>}"
    submit_pair C16 "$RUNG"
    submit_pair C32 "$RUNG"
    ;;
  *)
    echo "usage: $0 [matrix | spot <RUNG>]   (DRYRUN=1 to preview)"; exit 2 ;;
esac

echo "manifest: ${MANIFEST}"
echo "collect with: python ${EXPDIR}/p0_collect.py"
