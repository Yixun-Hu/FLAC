#!/usr/bin/env bash
# ============================================================================
# exp_11 P0 matrix submitter — plan Rev 3 §10 P0.1/P0.2 (Rev 2: round-2 review).
#
# ONE 30-step job per cell; the steady-state rate comes from the runner's in-fit
# step-10/step-30 marks, so no job pairing is needed (review B2).
#
#   matrix (default) ... rungs {32x2,16x4,8x8} x {VAN, C4L, C8} + CKPT4_32x2
#   spot <RUNG> ........ C16 and C32 at one of {32x2,16x4,8x8}
#   workers <CELL> <RUNG> ... the 0-vs-6-worker pair for one cell (review B3)
#
# Every submission gets a collision-proof RUNID and an ATOMIC manifest (written
# to a temp file and mv'd into place) listing runid, commit sha and every
# expected (cell, maxsteps, jobid, config_sha). p0_collect.py consumes that
# manifest and admits only rows that match it (review B4). The config path/sha
# comes from p0_profile.sbatch's own map via P0_PRINT_CONFIG, so the submitter
# cannot disagree with the job about which config a cell runs (review B5).
#
# Refuses to run on a dirty tracked tree, and exits nonzero if ANY sbatch fails.
# DRYRUN=1 prints the sbatch commands without submitting.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH_FILE="$EXPDIR/p0_profile.sbatch"
MAXSTEPS="${MAXSTEPS:-30}"
DRYRUN="${DRYRUN:-0}"
FAILURES=0

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

# --- drift gate: a queued job must run reviewed, committed code ---
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD)"
RUNID="$(git rev-parse --short HEAD)-$(date +%s)"
echo "submitting P0 run ${RUNID} at ${SHA}"

manifest_begin() {  # $1 = run id for this manifest
  RUNID="$1"
  MANIFEST="$EXPDIR/p0_manifest_${RUNID}.txt"
  MANIFEST_TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
  trap 'rm -f "$MANIFEST_TMP"' EXIT
  {
    echo "# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)"
    echo "runid ${RUNID}"
    echo "sha ${SHA}"
    echo "submitted_at $(date +%s)"
  } >> "$MANIFEST_TMP"
}

manifest_publish() {
  mv "$MANIFEST_TMP" "$MANIFEST"   # atomic publish
  trap - EXIT
  echo "manifest: ${MANIFEST}"
  echo "collect with: python ${EXPDIR}/p0_collect.py --manifest ${MANIFEST}"
}

submit_cell() {  # $1 = CELL, $2 = optional NUM_WORKERS override
  local cell="$1" workers="${2:-}" mb ngpu cfg cfg_sha out jid
  local rung="${cell#*_}"
  mb="${rung%x*}"; ngpu="${rung#*x}"
  [ "$((mb * ngpu))" -eq 64 ] || { echo "cell ${cell}: MB*NGPU != 64 - skip"; FAILURES=$((FAILURES+1)); return 1; }

  # config path from the sbatch's own map (single source of truth)
  cfg="$(P0_PRINT_CONFIG="$cell" bash "$SBATCH_FILE" 2>/dev/null)"
  [ -n "$cfg" ] && [ -f "$cfg" ] || { echo "cell ${cell}: no config from the sbatch map - skip"; FAILURES=$((FAILURES+1)); return 1; }
  cfg_sha="$(sha256sum "$cfg" | awk '{print $1}')"

  local tag="$cell"
  local export_list="ALL,EXPECT_SHA=${SHA},RUNID=${RUNID},CELL=${cell},MAXSTEPS=${MAXSTEPS}"
  if [ -n "$workers" ]; then
    tag="${cell}-w${workers}"
    export_list="${export_list},NUM_WORKERS=${workers}"
  fi

  local args=(
    --job-name="p0-${tag}"
    --gres="gpu:l40:${ngpu}"
    --cpus-per-task="$((8 + 7 * ngpu))"
    --mem="$((12 * ngpu + 12))G"
    --time=00:40:00
    --export="$export_list"
    "$SBATCH_FILE"
  )
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN sbatch ${args[*]}"
    echo "cell ${cell} ${MAXSTEPS} DRYRUN ${cfg_sha}" >> "$MANIFEST_TMP"
    return 0
  fi
  out="$(sbatch "${args[@]}" 2>&1)"
  jid="$(echo "$out" | awk '/Submitted batch job/ {print $NF}')"
  if [ -z "$jid" ]; then
    echo "SUBMIT FAILED ${cell}: ${out}"
    echo "cell ${cell} ${MAXSTEPS} SUBMIT_FAILED ${cfg_sha}" >> "$MANIFEST_TMP"
    FAILURES=$((FAILURES + 1))
    return 1
  fi
  echo "submitted ${tag} (${ngpu} GPU x MB ${mb}, workers ${workers:-6}) -> job ${jid}"
  echo "cell ${cell} ${MAXSTEPS} ${jid} ${cfg_sha}" >> "$MANIFEST_TMP"
}

MODE="${1:-matrix}"
BASE_RUNID="$RUNID"
case "$MODE" in
  matrix)
    manifest_begin "$BASE_RUNID"
    for rung in 32x2 16x4 8x8; do
      for family in VAN C4L C8; do submit_cell "${family}_${rung}"; done
    done
    submit_cell CKPT4_32x2
    manifest_publish
    ;;
  spot)
    RUNG="${2:?usage: $0 spot <32x2|16x4|8x8>}"
    case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "spot rung must be one of 32x2/16x4/8x8"; exit 2;; esac
    manifest_begin "$BASE_RUNID"
    submit_cell "C16_${RUNG}"
    submit_cell "C32_${RUNG}"
    manifest_publish
    ;;
  workers)
    CELLFAM="${2:?usage: $0 workers <FAMILY e.g. C4L> <RUNG>}"
    RUNG="${3:?usage: $0 workers <FAMILY> <32x2|16x4|8x8>}"
    case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung must be one of 32x2/16x4/8x8"; exit 2;; esac
    # Both halves carry the SAME cell tag (the config and rung are identical; only
    # --num-workers differs), so they get one manifest each — a single manifest
    # with two rows for one cell is exactly what the collector refuses.
    for W in 0 6; do
      manifest_begin "${BASE_RUNID}-w${W}"
      submit_cell "${CELLFAM}_${RUNG}" "$W"
      manifest_publish
    done
    ;;
  *)
    echo "usage: $0 [matrix | spot <RUNG> | workers <FAMILY> <RUNG>]   (DRYRUN=1 to preview)"; exit 2 ;;
esac

if [ "$FAILURES" -ne 0 ]; then
  echo "${FAILURES} submission(s) FAILED - the manifest records them as SUBMIT_FAILED and collection will refuse to report"
  exit 1
fi
