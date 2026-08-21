#!/usr/bin/env bash
# ============================================================================
# della_chain_submit.sh — submit the whole exp_16 chain (plan Rev 4) as ONE
# transaction: N legs, held, dependency-linked, recorded, then released.
#
# Usage:
#   della_chain_submit.sh chain --time D-HH:MM:SS     # 67,500 in 7,500-step legs
#   della_chain_submit.sh probe                       # the 3+1 leg GPU rehearsal
#   della_chain_submit.sh cont --time D-HH:MM:SS      # the CONTINUOUS A/B arm:
#                                                     # 67,500 steps as ONE job
#
# `cont` is the control for the chunking A/B: the same recipe and budget with no
# resume seams, into a sibling save-dir, submitted as a single job with no
# manifest and no dependencies. Everything else about it — gates, interlock,
# transaction — is identical to a chain submission.
#
# Legs are linked with `--dependency=afterany:<prev>`: AFTERANY, not afterok, is
# the watchdog. A leg that crashes still releases its successor, which discovers
# the last good checkpoint for itself and retries from there. The extra "spare"
# leg exists so the final chunk still has a retry; if everything went well it
# simply finds S >= TOTAL and CHAINDONEs.
#
# GATES (all fail-closed): the chain closure clean; HEAD pushed (ls-remote, so a
# DRYRUN writes nothing); the committed Phase-1 verdict tracked at HEAD; and NO
# exp16-train*/exp16-chain job of this user in ANY state — the chain replaces the
# racers (Yixun-approved swap), it does not run beside them.
#
# TRANSACTION: every leg is submitted HELD, the manifest and the command-record
# block are written under flock and read back, and only then is every id
# released. Any failure in between cancels every id created and removes the
# manifest, so a half-built chain never exists in the queue.
#
# Exit codes: 2 usage, 3 closure dirty, 4 HEAD not pushed, 5 sbatch failed or
# unparsable, 6 record/manifest write or verify failed, 7 release failed,
# 8 Phase-1 verdict missing, 9 a racer/chain/cont job already exists,
# 10 the probe or continuous save-dir is not clean, 11 the squeue query itself
# failed, 12 another submission holds the transaction lock.
# ============================================================================
set -euo pipefail

REPO=/home/yh4742/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"
BRANCH=della-flac-chequity
REMOTE=origin
RECORD="$EXPDIR/della_vanilla_repro_command.md"
MANIFEST="$EXPDIR/chain_manifest_current.txt"
LEG_SBATCH="$EXPDIR/della_chain.sbatch"
PHASE1_PASS_REL="worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/PHASE1_PASS.md"
PROD_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro
PROBE_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro_chainprobe
CONT_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro_cont
RUN_NAME=FLAC_vanilla_repro
JOB_NAME=exp16-chain
EXCLUSIVE_NAMES=exp16-train,exp16-train-smoke,exp16-train-resume,exp16-chain,exp16-cont
DRYRUN="${DRYRUN:-0}"
export PYTHONDONTWRITEBYTECODE=1

CLOSURE=(
  src
  data
  train.py
  eval_FLAC.py
  finetune_cond.py
  eval_pl.py
  eval_VAE.py
  unwrap_model.py
  defaults.ini
  pyproject.toml
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_chain.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_chain_submit.sh
)

SUBMIT_LOCK="$EXPDIR/.chain_submit.lock"

die() { echo "GATEFAIL rc=${2} ${1}" >&2; exit "${2}"; }

# --- ONE LOCK OVER THE WHOLE TRANSACTION (review N2) --------------------------
# Every gate below reads scheduler and filesystem state that a CONCURRENT
# submission is in the middle of changing: two invocations could both see "no
# chain jobs exist", both submit, and the second would overwrite the first's
# manifest — leaving legs whose manifest names another chain's ids. Re-exec once
# under flock so check -> submit -> manifest -> record -> release is indivisible.
# A DRYRUN performs no transaction and takes no lock, which is also what keeps it
# from creating the lockfile.
if [ "${CHAIN_SUBMIT_LOCK_HELD:-0}" != "1" ] && [ "${DRYRUN:-0}" != "1" ]; then
  export CHAIN_SUBMIT_LOCK_HELD=1
  RC=0
  # `bash "$0"` rather than "$0": the kit's files are mode 644 like the rest of
  # the worklog, so they are run through an interpreter, never exec'd directly.
  flock -w 10 -E 99 "$SUBMIT_LOCK" bash "$0" "$@" || RC=$?
  [ "$RC" -ne 99 ] || die "another chain submission is in progress (could not take ${SUBMIT_LOCK} within 10s)" 12
  exit "$RC"
fi

usage() {
  echo "usage: della_chain_submit.sh chain --time D-HH:MM:SS" >&2
  echo "       della_chain_submit.sh probe" >&2
  echo "       della_chain_submit.sh cont --time D-HH:MM:SS" >&2
}

validate_time() {   # D-HH:MM:SS or HH:MM:SS, minutes/seconds <= 59
  local t="$1" days="" hms hh mm ss
  case "$t" in *-*) days="${t%%-*}"; hms="${t#*-}" ;; *) hms="$t" ;; esac
  if [ -n "$days" ]; then case "$days" in ''|*[!0-9]*) return 1 ;; esac; fi
  case "$hms" in
    [0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9]:[0-9][0-9]:[0-9][0-9]|[0-9][0-9][0-9]:[0-9][0-9]:[0-9][0-9]) ;;
    *) return 1 ;;
  esac
  hh="${hms%%:*}"; mm="${hms#*:}"; mm="${mm%%:*}"; ss="${hms##*:}"
  [ "$((10#$mm))" -le 59 ] || return 1
  [ "$((10#$ss))" -le 59 ] || return 1
  [ -z "$days" ] || [ "$((10#$hh))" -le 23 ] || return 1
  return 0
}

# Newest checkpoint step under a save-dir (0 when none); same single-* glob shape
# and same fatal-on-anomaly rule as the leg driver.
newest_step() {   # <save-dir> -> prints the step
  local sd="$1" step max=0 base c
  local -a cands=()
  shopt -s nullglob
  cands=("$sd/$RUN_NAME"/*/checkpoints/epoch=*-step=*.ckpt)
  shopt -u nullglob
  for c in "${cands[@]}"; do
    base="$(basename -- "$c")"
    [[ "$base" =~ ^epoch=[0-9]+-step=([0-9]+)\.ckpt$ ]] || return 1
    step=$((10#${BASH_REMATCH[1]}))
    [ "$step" -le "$max" ] || max="$step"
  done
  printf '%s\n' "$max"
}

# --- A. arguments -------------------------------------------------------------
KIND="${1:-}"; shift || true
TIME_LIMIT=""
CONT=0
case "$KIND" in
  chain)
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --time)   TIME_LIMIT="${2:-}"; shift 2 || die "--time needs a value" 2 ;;
        --time=*) TIME_LIMIT="${1#--time=}"; shift ;;
        *) usage; die "unknown chain option '${1}'" 2 ;;
      esac
    done
    [ -n "$TIME_LIMIT" ] || die "a chain needs an explicit --time D-HH:MM:SS (plan Rev 4: 04:30:00 per gpu-short leg)" 2
    validate_time "$TIME_LIMIT" || die "--time '${TIME_LIMIT}' is not D-HH:MM:SS or HH:MM:SS with minutes/seconds <= 59" 2
    CHAIN_TOTAL=67500; CHAIN_CHUNK=7500; PROBE=0; SAVEDIR="$PROD_SAVEDIR" ;;
  probe)
    [ "$#" -eq 0 ] || die "probe takes no options (its --time is fixed at 00:30:00)" 2
    TIME_LIMIT=00:30:00
    CHAIN_TOTAL=300; CHAIN_CHUNK=100; PROBE=1; SAVEDIR="$PROBE_SAVEDIR" ;;
  cont)
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --time)   TIME_LIMIT="${2:-}"; shift 2 || die "--time needs a value" 2 ;;
        --time=*) TIME_LIMIT="${1#--time=}"; shift ;;
        *) usage; die "unknown cont option '${1}'" 2 ;;
      esac
    done
    # No default: a continuous 67,500-step run is ~20 h of wall clock and the
    # budget is a measured quantity, not something to inherit from a constant.
    [ -n "$TIME_LIMIT" ] || die "the continuous arm needs an explicit --time D-HH:MM:SS" 2
    validate_time "$TIME_LIMIT" || die "--time '${TIME_LIMIT}' is not D-HH:MM:SS or HH:MM:SS with minutes/seconds <= 59" 2
    # CHUNK == TOTAL is what makes it one leg; the driver refuses any other pair.
    CHAIN_TOTAL=67500; CHAIN_CHUNK=67500; PROBE=0; CONT=1
    JOB_NAME=exp16-cont; SAVEDIR="$CONT_SAVEDIR" ;;
  *) usage; die "first argument must be 'chain' or 'probe' (got '${KIND}')" 2 ;;
esac

# --- B. the chain closure must be clean --------------------------------------
CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
  || die "git status FAILED - refusing to treat an error as a clean closure" 3
if [ -n "$CLOSURE_DIRT" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: the chain closure is dirty (a real submit aborts here):"
    echo "$CLOSURE_DIRT"
  else
    echo "$CLOSURE_DIRT" >&2
    die "the chain closure is dirty - commit or stash before submitting" 3
  fi
fi

# --- C. the Phase-1 verdict gates all training compute (plan §5 / Rev 3 §2) ---
# Both kinds, probe included: the file already exists once Phase 1 has passed, so
# requiring it costs nothing and keeps ONE rule about what may consume a training
# GPU in this experiment.
if [ ! -f "$REPO/$PHASE1_PASS_REL" ]; then
  die "the Phase-1 gate (plan §5) has no verdict: ${PHASE1_PASS_REL} does not exist - Phase 1 gates all training compute" 8
fi
git -C "$REPO" cat-file -e "HEAD:${PHASE1_PASS_REL}" 2>/dev/null \
  || die "the Phase-1 verdict ${PHASE1_PASS_REL} is not tracked at HEAD - an uncommitted verdict is not a reviewable gate" 8
echo "Phase-1 gate OK: ${PHASE1_PASS_REL} exists and is tracked at HEAD"

# --- D. no racer or chain job may exist, in ANY state -------------------------
# The chain REPLACES the two-leg race (Yixun 2026-08-13). Two GPU runs writing one
# save-dir would interleave checkpoints and make every leg's resume point
# ambiguous, and a queued racer would do it hours later without anyone watching.
# The QUERY is checked before its OUTPUT is (review B1): a controller outage or a
# bad argument makes squeue print nothing and exit nonzero, and an empty result
# then reads as "no racers exist" — which is precisely the state this gate must
# never invent. Under DRYRUN a failed query is an advisory (a dry run submits
# nothing, so it cannot act on the misreading); a real submission refuses.
# `|| SQ_RC=$?`, not a bare assignment then `$?`: this script runs under `set -e`,
# where a failing command substitution inside an assignment aborts the shell
# BEFORE the next line can inspect the status — the gate would then die silently
# with rc=1 instead of reporting why. (Measured: the first cut of this fix did
# exactly that. The leg driver runs without -e and is not affected.)
SQ_RC=0
EXISTING_RAW="$(squeue -h -u "$(id -un)" -n "$EXCLUSIVE_NAMES" -o '%i %j %T')" || SQ_RC=$?
if [ "$SQ_RC" -ne 0 ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: squeue -h -u $(id -un) -n ${EXCLUSIVE_NAMES} FAILED (rc=${SQ_RC}); a real submit aborts here"
    EXISTING_RAW=""
  else
    die "squeue -h -u $(id -un) -n ${EXCLUSIVE_NAMES} FAILED (rc=${SQ_RC}) - refusing to read a scheduler error as 'no racers exist'" 11
  fi
fi
EXISTING="$(printf '%s' "$EXISTING_RAW" | tr '\n' ';')"
if [ -n "${EXISTING%;}" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: exp16 training jobs already exist (a real submit aborts here): ${EXISTING%;}"
  else
    die "exp16 training jobs already exist (${EXISTING%;}) - scancel the racers first: the chain replaces them, it does not run beside them" 9
  fi
fi

# --- E. a probe, and the continuous arm, must start clean --------------------
# Never removed automatically: deleting checkpoints is an operator decision. A
# probe that silently resumed would prove nothing about seams, and a CONTINUOUS
# run that resumed would contain the very seam it exists to measure against.
CLEAN_DIR=""; CLEAN_WHAT=""
[ "$PROBE" != "1" ] || { CLEAN_DIR="$PROBE_SAVEDIR"; CLEAN_WHAT="probe"; }
[ "$CONT" != "1" ]  || { CLEAN_DIR="$CONT_SAVEDIR";  CLEAN_WHAT="continuous"; }
if [ -n "$CLEAN_DIR" ]; then
  CLEAN_S="$(newest_step "$CLEAN_DIR")" || die "the ${CLEAN_WHAT} save-dir contains a malformed checkpoint name" 10
  if [ "$CLEAN_S" != "0" ]; then
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY-RUN ADVISORY: the ${CLEAN_WHAT} save-dir already holds checkpoints (newest step ${CLEAN_S}); a real submit aborts here"
    else
      die "the ${CLEAN_WHAT} save-dir already holds checkpoints (newest step ${CLEAN_S}) - remove ${CLEAN_DIR} by hand and resubmit" 10
    fi
  fi
fi

# --- F. HEAD must be PUSHED ---------------------------------------------------
EXPECT_SHA="$(git -C "$REPO" rev-parse HEAD)" || die "git rev-parse HEAD FAILED" 4
REMOTE_SHA="$(git -C "$REPO" ls-remote "$REMOTE" "refs/heads/${BRANCH}" | awk '{print $1}')" \
  || die "git ls-remote ${REMOTE} refs/heads/${BRANCH} FAILED - cannot prove HEAD is pushed" 4
if [ "$EXPECT_SHA" != "$REMOTE_SHA" ]; then
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY-RUN ADVISORY: HEAD ${EXPECT_SHA} != ${REMOTE}/${BRANCH} ${REMOTE_SHA:-<none>} (a real submit aborts here)"
  else
    die "HEAD ${EXPECT_SHA} is not pushed to ${REMOTE}/${BRANCH} (${REMOTE_SHA:-<none>})" 4
  fi
fi

# --- G. how many legs ---------------------------------------------------------
# From where training ACTUALLY is, not from zero: resubmitting a chain over a
# half-finished run must not queue nine legs to do two legs' work.
S0="$(newest_step "$SAVEDIR")" || die "the save-dir contains a malformed checkpoint name" 3
REMAIN=$((CHAIN_TOTAL - S0))
[ "$REMAIN" -gt 0 ] || die "training is already at step ${S0} >= CHAIN_TOTAL ${CHAIN_TOTAL} - nothing to submit" 2
if [ "$CONT" = "1" ]; then
  # ONE job, no spare: a spare would be a second attempt at a run that must not
  # resume, and afterany has nothing to chain to.
  NLEGS=1
  echo "continuous plan: 0 -> ${CHAIN_TOTAL} in ONE job, --time ${TIME_LIMIT}, save-dir ${SAVEDIR}"
else
  NLEGS=$(( (REMAIN + CHAIN_CHUNK - 1) / CHAIN_CHUNK + 1 ))   # ceil + one spare
  echo "chain plan: S0=${S0} TOTAL=${CHAIN_TOTAL} CHUNK=${CHAIN_CHUNK} -> ${NLEGS} legs (incl. 1 spare), --time ${TIME_LIMIT} each"
fi

EXPORTS="EXPECT_SHA=${EXPECT_SHA},CHAIN_TOTAL=${CHAIN_TOTAL},CHAIN_CHUNK=${CHAIN_CHUNK}"
[ "$PROBE" = "0" ] || EXPORTS="${EXPORTS},PROBE=1"
[ "$CONT" = "0" ] || EXPORTS="${EXPORTS},CONT=1"

if [ "$DRYRUN" = "1" ]; then
  PREV="<leg-1-id>"
  for i in $(seq 1 "$NLEGS"); do
    if [ "$i" = "1" ]; then
      echo "sbatch --hold --parsable --job-name=${JOB_NAME} --time=${TIME_LIMIT} --export=${EXPORTS} ${LEG_SBATCH}"
    else
      echo "sbatch --hold --parsable --job-name=${JOB_NAME} --time=${TIME_LIMIT} --dependency=afterany:${PREV} --export=${EXPORTS} ${LEG_SBATCH}"
    fi
    PREV="<leg-${i}-id>"
  done
  echo "DRY RUN: nothing submitted, no manifest, command record untouched (EXPECT_SHA would be ${EXPECT_SHA})"
  exit 0
fi

# --- H. submit every leg HELD, linked afterany --------------------------------
JOBIDS=()
# Cleanup may only undo what THIS invocation did (review N2): a manifest that was
# already there belongs to another chain, and deleting it would strand that
# chain's legs (they refuse to run without one). Its bytes are kept so a failed
# overwrite can be put back exactly.
MANIFEST_EXISTED=0
MANIFEST_PREEXISTING=""
MANIFEST_WRITTEN=0
if [ -f "$MANIFEST" ]; then
  MANIFEST_EXISTED=1
  MANIFEST_PREEXISTING="$(cat "$MANIFEST")"
fi
restore_manifest() {
  [ "$MANIFEST_WRITTEN" = "1" ] || return 0     # we never touched it
  if [ "$MANIFEST_EXISTED" = "1" ]; then
    printf '%s\n' "$MANIFEST_PREEXISTING" > "$MANIFEST" \
      && echo "restored the pre-existing manifest ${MANIFEST}" >&2 \
      || echo "could NOT restore the pre-existing manifest ${MANIFEST} - repair it by hand" >&2
  else
    rm -f "$MANIFEST"
  fi
}
abort_transaction() {   # <message>
  if [ "${#JOBIDS[@]}" -gt 0 ]; then
    echo "cancelling the partial chain: ${JOBIDS[*]}" >&2
    scancel "${JOBIDS[@]}" || echo "scancel FAILED - cancel ${JOBIDS[*]} by hand" >&2
  fi
  restore_manifest
  die "$1" 5
}
SBATCH_LINES=()
PREV=""
for i in $(seq 1 "$NLEGS"); do
  ARGV=(--hold --parsable --job-name="$JOB_NAME" --time="$TIME_LIMIT")
  [ -z "$PREV" ] || ARGV+=(--dependency=afterany:"$PREV")
  ARGV+=(--export="$EXPORTS" "$LEG_SBATCH")
  OUT="$(sbatch "${ARGV[@]}")" || abort_transaction "sbatch FAILED for leg ${i}"
  ID="${OUT%%;*}"
  case "$ID" in
    ''|*[!0-9]*) echo "sbatch returned: '${OUT}'" >&2; abort_transaction "leg ${i}: sbatch output is not a numeric job id" ;;
  esac
  JOBIDS+=("$ID")
  SBATCH_LINES+=("sbatch ${ARGV[*]}")
  PREV="$ID"
  echo "leg ${i}/${NLEGS} submitted HELD as ${ID}${PREV:+}"
done

# --- I. manifest + command record, under flock, then read back ---------------
write_records() {
  {
    flock -w 30 200 || return 1
    # NO manifest in CONT mode: one job with no successors has no chain to know
    # about, and writing one would leave a stale file that a later chain leg
    # would read as ITS chain.
    if [ "$CONT" = "0" ]; then
      MANIFEST_WRITTEN=1      # from here on, cleanup owns this file
      printf '%s\n' "${JOBIDS[@]}" > "$MANIFEST" || return 1
    fi
    if [ ! -s "$RECORD" ]; then
      {
        echo "# della_vanilla_repro — submission command record"
        echo
      } >&200 || return 1
    fi
    {
      echo "## $(date -Is) — ${JOB_NAME} (${KIND}) — ${NLEGS} legs — jobs ${JOBIDS[*]}"
      echo
      echo "- EXPECT_SHA: \`${EXPECT_SHA}\`"
      echo "- chain: TOTAL=${CHAIN_TOTAL} CHUNK=${CHAIN_CHUNK} S0=${S0} PROBE=${PROBE} CONT=${CONT}"
      echo "- save-dir: \`${SAVEDIR}\`"
      if [ "$CONT" = "0" ]; then
        echo "- manifest: \`${MANIFEST}\`"
      else
        echo "- manifest: none (continuous single-job arm)"
      fi
      echo "- submitted by: \`$(whoami)@$(hostname)\`"
      echo
      echo '```bash'
      printf '%s\n' "${SBATCH_LINES[@]}"
      echo '```'
      echo
    } >&200 || return 1
  } 200>>"$RECORD"
  local id
  if [ "$CONT" = "0" ]; then
    for id in "${JOBIDS[@]}"; do
      grep -q "^${id}\$" "$MANIFEST" || return 1
    done
  fi
  grep -q "jobs ${JOBIDS[*]}" "$RECORD" || return 1
}
if ! write_records; then
  echo "manifest/command-record write or verify FAILED - cancelling the whole chain" >&2
  scancel "${JOBIDS[@]}" || echo "scancel FAILED - cancel ${JOBIDS[*]} by hand" >&2
  restore_manifest
  die "no released chain may lack its manifest and record" 6
fi
if [ "$CONT" = "0" ]; then
  echo "manifest ${MANIFEST} and record ${RECORD} written and verified"
else
  echo "record ${RECORD} written and verified (no manifest: continuous single-job arm)"
fi

# --- J. release everything ----------------------------------------------------
# Order is irrelevant: the afterany dependencies keep legs 2..N queued until their
# predecessor ends, so releasing them all at once cannot start two legs at once.
if ! scontrol release "${JOBIDS[@]}"; then
  echo "scontrol release FAILED - cancelling the chain; the record block stays as evidence" >&2
  scancel "${JOBIDS[@]}" || echo "scancel FAILED too - jobs ${JOBIDS[*]} are HELD; release or cancel by hand" >&2
  restore_manifest
  die "could not release the chain" 7
fi
echo "released ${NLEGS} legs: ${JOBIDS[*]}"
printf '%s\n' "${JOBIDS[@]}"
