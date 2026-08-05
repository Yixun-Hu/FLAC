#!/usr/bin/env bash
# ============================================================================
# exp_11 P0 matrix submitter — plan Rev 3 §10 P0.1/P0.2 (Rev 3: round-2 re-review).
#
# ONE 30-step job per (cell, workers); the steady-state rate comes from the
# runner's in-fit step-10/30 marks, so there is no job pairing.
#
#   matrix (default) ... rungs {32x2,16x4,8x8} x {VAN, FA1, C4L, C8} + CKPT4_32x2
#                        = 13 jobs. FA1 (single-angle fa_invariant) is the
#                        attribution baseline: it shares C4L/C8's cylindrical
#                        pose path, so FA1->C4L->C8 isolates the per-orbit-pass
#                        cost, and VAN is the separate vanilla contrast.
#   spot <RUNG> ........ C16 and C32 at one of {32x2,16x4,8x8}
#   workers <FAM> <RUNG> the 0-vs-6-worker pair for one cell, in ONE manifest
#
# Every submission gets a collision-proof RUNID (short sha + epoch NANOseconds +
# random hex) and an ATOMIC, NO-CLOBBER manifest listing runid, mode, commit sha
# and every expected row with its FULL execution shape (maxsteps, jobid, config
# sha256, mb, ngpu, workers, time limit). p0_collect.py admits only rows matching
# those fields exactly. The config path/sha comes from p0_profile.sbatch's own map
# via P0_PRINT_CONFIG (queried env-clean), so submitter and job cannot disagree.
#
# Time limits are per cell (re-review B5): the C16/C32 spots are far slower than
# a matrix cell and a killed spot costs more than a queued hour.
#
# Refuses to run on a dirty tracked tree, and exits nonzero if ANY sbatch fails.
# DRYRUN=1 prints the sbatch commands without submitting.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH_FILE="$EXPDIR/p0_profile.sbatch"
MAXSTEPS=30                      # pinned P0 budget (re-review B3)
MATRIX_WORKERS=6
DRYRUN="${DRYRUN:-0}"
FAILURES=0

[ -f "$SBATCH_FILE" ] || { echo "missing ${SBATCH_FILE} - abort"; exit 3; }

# --- drift gate: a queued job must run reviewed, committed code ---
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces have uncommitted changes - commit first, abort:"; echo "$DRIFT"; exit 2; }
SHA="$(git rev-parse HEAD)"
# Collision-proof: nanoseconds + kernel random hex (epoch seconds alone can
# collide between concurrent submissions — re-review B3).
RUNID="$(git rev-parse --short HEAD)-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
echo "submitting P0 run ${RUNID} at ${SHA}"

# --- per-cell wall-clock limits (re-review B5) ---
time_limit_for() {  # $1 = CELL
  case "$1" in
    C16_32x2) echo "01:30:00" ;;
    C16_*)    echo "01:00:00" ;;
    C32_32x2) echo "04:00:00" ;;
    C32_16x4) echo "02:30:00" ;;
    C32_8x8)  echo "01:30:00" ;;
    *)        echo "00:40:00" ;;
  esac
}

manifest_begin() {  # $1 = mode
  MANIFEST="$EXPDIR/p0_manifest_${RUNID}.txt"
  [ ! -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} already exists - abort (run id collision)"; exit 2; }
  MANIFEST_TMP="$(mktemp "${MANIFEST}.XXXXXX")" || exit 3
  trap 'rm -f "$MANIFEST_TMP"' EXIT
  {
    echo "# exp_11 P0 submission manifest (consumed by p0_collect.py --manifest)"
    echo "# cell <CELL> <MAXSTEPS> <JOBID> <CONFIG_SHA256> <MB> <NGPU> <WORKERS> <TIMELIMIT>"
    echo "runid ${RUNID}"
    echo "sha ${SHA}"
    echo "mode $1"
    echo "submitted_at $(date +%s)"
  } >> "$MANIFEST_TMP"
}

manifest_publish() {
  [ ! -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} appeared during submission - abort"; exit 2; }
  mv -n "$MANIFEST_TMP" "$MANIFEST" || { echo "manifest publication failed - abort"; exit 2; }
  [ -e "$MANIFEST" ] || { echo "manifest ${MANIFEST} was not published - abort"; exit 2; }
  trap - EXIT
  echo "manifest: ${MANIFEST}"
  echo "collect with: python ${EXPDIR}/p0_collect.py --manifest ${MANIFEST}"
}

submit_cell() {  # $1 = CELL, $2 = NUM_WORKERS (explicit, never defaulted)
  local cell="$1" workers="$2" mb ngpu cfg cfg_sha tlimit out jid
  local rung="${cell#*_}"
  mb="${rung%x*}"; ngpu="${rung#*x}"
  [ "$((mb * ngpu))" -eq 64 ] || { echo "cell ${cell}: MB*NGPU != 64 - skip"; FAILURES=$((FAILURES+1)); return 1; }

  # config path from the sbatch's own map, queried env-clean (NIT 7)
  cfg="$(env -u SLURM_JOB_ID P0_PRINT_CONFIG="$cell" bash "$SBATCH_FILE" 2>/dev/null)"
  [ -n "$cfg" ] && [ -f "$cfg" ] || { echo "cell ${cell}: no config from the sbatch map - skip"; FAILURES=$((FAILURES+1)); return 1; }
  cfg_sha="$(sha256sum "$cfg" | awk '{print $1}')"
  tlimit="$(time_limit_for "$cell")"

  local tag="${cell}-w${workers}"
  local args=(
    --job-name="p0-${tag}"
    --gres="gpu:l40:${ngpu}"
    --cpus-per-task="$((8 + 7 * ngpu))"
    --mem="$((12 * ngpu + 12))G"
    --time="$tlimit"
    --export="ALL,EXPECT_SHA=${SHA},RUNID=${RUNID},CELL=${cell},MAXSTEPS=${MAXSTEPS},NUM_WORKERS=${workers}"
    "$SBATCH_FILE"
  )
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN sbatch ${args[*]}"
    echo "cell ${cell} ${MAXSTEPS} DRYRUN ${cfg_sha} ${mb} ${ngpu} ${workers} ${tlimit}" >> "$MANIFEST_TMP"
    return 0
  fi
  out="$(sbatch "${args[@]}" 2>&1)"
  jid="$(echo "$out" | awk '/Submitted batch job/ {print $NF}')"
  if [ -z "$jid" ]; then
    echo "SUBMIT FAILED ${tag}: ${out}"
    echo "cell ${cell} ${MAXSTEPS} SUBMIT_FAILED ${cfg_sha} ${mb} ${ngpu} ${workers} ${tlimit}" >> "$MANIFEST_TMP"
    FAILURES=$((FAILURES + 1))
    return 1
  fi
  echo "submitted ${tag} (${ngpu} GPU x MB ${mb}, workers ${workers}, limit ${tlimit}) -> job ${jid}"
  echo "cell ${cell} ${MAXSTEPS} ${jid} ${cfg_sha} ${mb} ${ngpu} ${workers} ${tlimit}" >> "$MANIFEST_TMP"
}

MODE="${1:-matrix}"
case "$MODE" in
  matrix)
    manifest_begin matrix
    for rung in 32x2 16x4 8x8; do
      for family in VAN FA1 C4L C8; do submit_cell "${family}_${rung}" "$MATRIX_WORKERS"; done
    done
    submit_cell CKPT4_32x2 "$MATRIX_WORKERS"
    manifest_publish
    ;;
  spot)
    RUNG="${2:?usage: $0 spot <32x2|16x4|8x8>}"
    case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "spot rung must be one of 32x2/16x4/8x8"; exit 2;; esac
    manifest_begin spot
    submit_cell "C16_${RUNG}" "$MATRIX_WORKERS"
    submit_cell "C32_${RUNG}" "$MATRIX_WORKERS"
    manifest_publish
    ;;
  workers)
    CELLFAM="${2:?usage: $0 workers <FAMILY e.g. C4L> <RUNG>}"
    RUNG="${3:?usage: $0 workers <FAMILY> <32x2|16x4|8x8>}"
    case "$RUNG" in 32x2|16x4|8x8) ;; *) echo "rung must be one of 32x2/16x4/8x8"; exit 2;; esac
    # ONE manifest: the halves differ only in worker count, and the collector keys
    # rows by (cell, workers), so the pair stays jointly provenance-bound (B4).
    manifest_begin workers
    submit_cell "${CELLFAM}_${RUNG}" 0
    submit_cell "${CELLFAM}_${RUNG}" "$MATRIX_WORKERS"
    manifest_publish
    ;;
  *)
    echo "usage: $0 [matrix | spot <RUNG> | workers <FAMILY> <RUNG>]   (DRYRUN=1 to preview)"; exit 2 ;;
esac

if [ "$FAILURES" -ne 0 ]; then
  echo "${FAILURES} submission(s) FAILED - the manifest records them as SUBMIT_FAILED and collection will refuse to report"
  exit 1
fi
