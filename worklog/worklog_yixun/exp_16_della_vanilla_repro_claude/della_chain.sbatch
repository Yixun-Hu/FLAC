#!/bin/bash
# ============================================================================
# exp_16 CHAIN LEG (della, plan Rev 4) — one 7,500-step chunk of the Phase-2
# reproduction, self-locating in the chain and self-terminating when done.
#
# WHY a chain: fairshare exhaustion makes long jobs queue for days while short
# jobs backfill. The 67,500-step budget is therefore cut into gpu-short legs
# submitted `--dependency=afterany`, each of which works out for ITSELF where the
# run has got to. That is what makes the chain a watchdog rather than a schedule:
# a leg that dies is simply retried by its successor from the last good
# checkpoint, and a leg that finds the budget already met cancels the rest.
#
# WHY per-leg reseeding: PL 2.1 restarts the dataloader on a mid-epoch resume
# (training_epoch_loop.py:155) and an empirical probe showed same-seed legs draw
# BYTE-IDENTICAL permutations — a fixed seed would make every leg re-walk the
# same head of the shuffle. SEED = 42 + LEG is deterministic (a retried leg reuses
# its seed) and leg 1 keeps the recipe's seed 42.
#
# Env contract (submitted with an explicit --export list, so nothing else crosses):
#   EXPECT_SHA   (required) the reviewed commit this leg is bound to
#   CHAIN_TOTAL  (required) the budget, e.g. 67500
#   CHAIN_CHUNK  (required) steps per leg, e.g. 7500
#   PROBE=1      (optional) the 3-leg GPU rehearsal: separate save-dir,
#                --checkpoint-every 100, validation off
#
# Leg logic: S = step of the newest checkpoint (0 if none) -> LEG = S/CHUNK,
# SEED = 42+LEG, TARGET = min(S+CHUNK, TOTAL), resume from that checkpoint when
# S > 0. S >= TOTAL means the chain is finished: CHAINDONE, cancel the legs after
# this one in the manifest, exit 0.
#
# Watchdog (half 1, in-leg): an attempts stamp per S. Two consecutive legs that
# made no progress at the same S is a systematic failure, not bad luck — the leg
# CHAINHALTs and cancels the rest of the chain rather than burning nine slots on
# the same crash. (Half 2 is the afterany dependency chain itself.)
#
# Submit through della_chain_submit.sh (it owns the racer check, the manifest and
# the held-submit transaction). DRYRUN=1 outside Slurm runs every gate that does
# not need a GPU, prints the argv, and writes NOTHING (no log, no stamp, no
# scancel — cancellations are echoed instead).
#
# Exit codes: 2 parameters, 3 repo/dirs, 4 closure binding, 5 interpreter
# provenance, 6 GPU, 7 ntasks, 8 log durability, 9 checkpoint discovery,
# 10 mutual exclusion, 11 CHAINHALT / stamp, 12 chain manifest,
# 13 post-run target mismatch, 14 the squeue query itself failed;
# otherwise train.py's own rc.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/slurm_chain_%x_%j.out

set -uo pipefail

# --- SHARED CONSTANTS (copied from della_train.sbatch; that file is frozen while
# --- the racer jobs are queued, so this is a deliberate copy, not a refactor) ---
REPO=/home/yh4742/codespace/FLAC
PYEXE=/scratch/gpfs/BLANCHETTE/yh4742/conda_envs/flac/bin/python
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"

export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb
export PYTHONDONTWRITEBYTECODE=1

# --- THE CHAIN'S MEASUREMENT CLOSURE -----------------------------------------
# The ten core paths plus the two CHAIN files. The three round-B/C kit files are
# deliberately absent: they are frozen for the queued racer jobs, and this chain
# neither reads nor is affected by them.
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

PROD_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro
PROBE_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro_chainprobe
RUN_NAME=FLAC_vanilla_repro
EXPERIMENT_NAME=exp16_della_vanilla_repro
MANIFEST="$EXPDIR/chain_manifest_current.txt"
# Every job name that owns a GPU training run in this experiment. A chain leg
# must never share a GPU-hour window with one of them (they write the SAME
# save-dir), so a running one is a hard stop.
EXCLUSIVE_NAMES=exp16-train,exp16-train-smoke,exp16-train-resume,exp16-chain

DRYRUN="${DRYRUN:-0}"
PROBE="${PROBE:-0}"
TS="$(date +%Y-%m-%d_%H:%M:%S)"

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

# --- checkpoint discovery (fatal on any error; never "start from scratch") ----
# Newest by mtime, exactly as della_train.sbatch does it, and then ASSERTED to be
# the highest step present: if an older-mtime checkpoint carried a higher step we
# would silently redo finished work, so that state stops the chain instead.
NEWEST=""; NEWEST_STEP=0
discover_newest() {
  NEWEST=""; NEWEST_STEP=0
  local -a cands=()
  shopt -s nullglob
  cands=("$SAVEDIR/$RUN_NAME"/*/checkpoints/epoch=*-step=*.ckpt)
  shopt -u nullglob
  [ "${#cands[@]}" -gt 0 ] || return 0
  local sorted first base max=0 step c cmax=""
  # No `| head -1`: pipefail turns head's early exit into a failed pipeline.
  sorted="$(ls -t -- "${cands[@]}")" || return 1
  first="${sorted%%$'\n'*}"
  [ -n "$first" ] || return 1
  for c in "${cands[@]}"; do
    base="$(basename -- "$c")"
    [[ "$base" =~ ^epoch=[0-9]+-step=([0-9]+)\.ckpt$ ]] || return 1
    step=$((10#${BASH_REMATCH[1]}))
    [ "$step" -le "$max" ] || { max="$step"; cmax="$c"; }
  done
  base="$(basename -- "$first")"
  [[ "$base" =~ ^epoch=[0-9]+-step=([0-9]+)\.ckpt$ ]] || return 1
  NEWEST_STEP=$((10#${BASH_REMATCH[1]}))
  NEWEST="$first"
  if [ "$NEWEST_STEP" -ne "$max" ]; then
    echo "newest-by-mtime ${first} is at step ${NEWEST_STEP} but ${cmax} carries step ${max}" >&2
    return 1
  fi
  return 0
}

# --- chain manifest: the ids submitted with this leg, in submission order -----
MANIFEST_LATER=()
read_manifest() {
  MANIFEST_LATER=()
  if [ ! -f "$MANIFEST" ]; then
    [ "$DRYRUN" = "1" ] || return 1
    echo "DRY-RUN ADVISORY: no chain manifest at ${MANIFEST} (a real leg aborts here)"
    return 0
  fi
  local seen_self=0 id
  while read -r id; do
    case "$id" in ''|\#*) continue ;; esac
    if [ "$seen_self" = "1" ]; then
      MANIFEST_LATER+=("$id")
    elif [ -n "${SLURM_JOB_ID:-}" ] && [ "$id" = "$SLURM_JOB_ID" ]; then
      seen_self=1
    fi
  done < "$MANIFEST"
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$seen_self" = "0" ]; then
    return 2
  fi
  return 0
}

# --- the attempts stamp: read / decide / halt (used under one lock, see gate G) --
STAMP_S=-1; STAMP_N=0; STAMP_NEXT=1; STAMP_HALT=0
stamp_read() {     # -> STAMP_S / STAMP_N ( -1 / 0 when there is no stamp yet )
  STAMP_S=-1; STAMP_N=0
  [ -f "$STAMP" ] || return 0
  read -r STAMP_S STAMP_N < "$STAMP" || return 1
  case "${STAMP_S}${STAMP_N}" in ''|*[!0-9]*) return 1 ;; esac
  return 0
}
stamp_decide() {   # -> STAMP_HALT (two strikes at the same S) and STAMP_NEXT
  STAMP_HALT=0
  if [ "$STAMP_S" = "$S" ]; then
    STAMP_NEXT=$((STAMP_N + 1))
    [ "$STAMP_N" -lt 2 ] || STAMP_HALT=1
  else
    STAMP_NEXT=1     # progress since the stamp was written: strikes reset
  fi
}
chainhalt() {
  echo "CHAINHALT: two consecutive legs made no progress at step ${S} - systematic failure, chain halted"
  cancel_later "CHAINHALT"
  die "two consecutive legs made no progress at step ${S} - systematic failure, chain halted" 11
}

cancel_later() {   # <reason>
  if [ "${#MANIFEST_LATER[@]}" -eq 0 ]; then
    echo "no later legs in the manifest to cancel (${1})"
    return 0
  fi
  if [ "$DRYRUN" = "1" ]; then
    echo "DRYRUN would: scancel ${MANIFEST_LATER[*]}   (${1})"
  else
    scancel "${MANIFEST_LATER[@]}" \
      && echo "cancelled later chain legs: ${MANIFEST_LATER[*]}   (${1})" \
      || echo "scancel FAILED for ${MANIFEST_LATER[*]} - cancel them by hand" >&2
  fi
}

main() {
  echo "=== exp_16 chain leg (PROBE=${PROBE}) — ${TS} — host $(hostname) — job ${SLURM_JOB_ID:-<none>} ==="

  # --- A. parameters ----------------------------------------------------------
  [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported" 2
  [ -n "${CHAIN_TOTAL:-}" ] || die "CHAIN_TOTAL must be exported (the budget, e.g. 67500)" 2
  [ -n "${CHAIN_CHUNK:-}" ] || die "CHAIN_CHUNK must be exported (steps per leg, e.g. 7500)" 2
  case "$CHAIN_TOTAL" in ''|*[!0-9]*) die "CHAIN_TOTAL '${CHAIN_TOTAL}' must be a positive integer" 2 ;; esac
  case "$CHAIN_CHUNK" in ''|*[!0-9]*) die "CHAIN_CHUNK '${CHAIN_CHUNK}' must be a positive integer" 2 ;; esac
  [ "$CHAIN_TOTAL" -gt 0 ] && [ "$CHAIN_CHUNK" -gt 0 ] || die "CHAIN_TOTAL/CHAIN_CHUNK must be positive" 2
  [ "$CHAIN_CHUNK" -le "$CHAIN_TOTAL" ] || die "CHAIN_CHUNK ${CHAIN_CHUNK} > CHAIN_TOTAL ${CHAIN_TOTAL}" 2
  case "$PROBE" in 0|1) ;; *) die "PROBE='${PROBE}' must be 0 or 1" 2 ;; esac
  # The recipe knobs are literals below; their presence in the environment is the
  # error (defence against a hand-rolled `sbatch --export=ALL`).
  for VAR in MAXSTEPS VAL_EVERY CHECKPOINT_EVERY SEED; do
    [ -z "${!VAR:-}" ] || die "${VAR} is computed by this script and must not be in the environment" 2
  done

  if [ "$PROBE" = "1" ]; then
    SAVEDIR="$PROBE_SAVEDIR"
    CHECKPOINT_EVERY=100
    # -1 is defaults.ini's sentinel: train.py:215 only builds val_args when
    # val_every > 0, so PL keeps its default check_val_every_n_epoch=1 and
    # validation would run at EPOCH end. A probe leg is 100 steps against a
    # 4,550-step epoch, so no validation pass can occur inside it — which is the
    # intent (the probe measures seams, not metrics).
    VAL_EVERY=-1
  else
    SAVEDIR="$PROD_SAVEDIR"
    CHECKPOINT_EVERY=2500
    VAL_EVERY=2500
  fi

  [ -d "$REPO" ] || die "REPO ${REPO} is not a directory" 3
  cd "$REPO" || die "could not cd ${REPO}" 3
  [ -e weights/FLAC/VAE.safetensors ] || die "VAE pretransform missing: weights/FLAC/VAE.safetensors" 2

  # --- B. closure binding ------------------------------------------------------
  git -C "$REPO" cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null \
    || die "EXPECT_SHA ${EXPECT_SHA} is not a commit in ${REPO}" 4
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED" 4
  CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
    || die "git status FAILED - refusing to treat an error as a clean closure" 4
  [ -z "$CLOSURE_DIRT" ] || die "chain closure is dirty: $(echo "$CLOSURE_DIRT" | tr '\n' ';')" 4
  git -C "$REPO" diff --quiet "$EXPECT_SHA" HEAD -- "${CLOSURE[@]}"
  DIFF_RC=$?
  case "$DIFF_RC" in
    0) ;;
    1) die "chain closure changed between EXPECT_SHA ${EXPECT_SHA} and HEAD ${HEAD_SHA}" 4 ;;
    *) die "git diff FAILED (rc=${DIFF_RC})" 4 ;;
  esac
  echo "closure binding OK: EXPECT_SHA ${EXPECT_SHA} == HEAD ${HEAD_SHA} content over ${#CLOSURE[@]} paths"

  # --- C. interpreter provenance ----------------------------------------------
  "$PYEXE" -B -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task, and NO other training job running --------------------------
  [ "${SLURM_NTASKS:-1}" -le 1 ] || die "SLURM_NTASKS=${SLURM_NTASKS} > 1: this kit is single-task only" 7
  # Two GPU runs writing one save-dir would interleave checkpoints and make every
  # leg's S ambiguous. The racers and the chain must never overlap.
  #
  # The QUERY is checked before its OUTPUT is (review B1): a controller outage, an
  # auth failure or a bad argument makes squeue print nothing and exit nonzero, and
  # an empty result then reads exactly like "no other job is running" — the most
  # dangerous possible misreading of this particular gate. Capture stdout and rc
  # separately, refuse on rc, and only then filter. This stays fatal under DRYRUN:
  # dry runs happen on the login node, where squeue works.
  RUNNING_RAW="$(squeue -h -u "$(id -un)" -n "$EXCLUSIVE_NAMES" -t R -o '%i')"
  SQ_RC=$?
  [ "$SQ_RC" -eq 0 ] \
    || die "squeue -h -u $(id -un) -n ${EXCLUSIVE_NAMES} -t R FAILED (rc=${SQ_RC}) - refusing to read a scheduler error as 'no other job is running'" 14
  # grep exits 1 when everything was filtered out; that is a legitimate empty
  # result HERE because the query itself already proved successful above.
  RUNNING="$(printf '%s\n' "$RUNNING_RAW" | grep -v "^${SLURM_JOB_ID:-__none__}\$" | tr '\n' ' ')" || true
  RUNNING="$(echo "$RUNNING" | tr -s ' ' | sed 's/^ //;s/ $//')"
  [ -z "$RUNNING" ] \
    || die "another exp16 training job is RUNNING (${RUNNING}): a chain leg may not share the save-dir - cancel it first" 10
  echo "mutual exclusion OK: no other ${EXCLUSIVE_NAMES} job is running"
  echo "EFFECTIVE_SEED offset: train.py adds SLURM_PROCID=${SLURM_PROCID:-0} to --seed"

  # --- E. where the chain has got to ------------------------------------------
  discover_newest || die "checkpoint discovery FAILED under ${SAVEDIR}/${RUN_NAME}/*/checkpoints" 9
  S="$NEWEST_STEP"
  read_manifest
  case "$?" in
    0) ;;
    2) die "chain manifest ${MANIFEST} does not list this job (${SLURM_JOB_ID:-<none>}) - a leg must know its chain" 12 ;;
    *) die "chain manifest ${MANIFEST} is missing - a leg must know its chain" 12 ;;
  esac
  echo "chain state: S=${S} TOTAL=${CHAIN_TOTAL} CHUNK=${CHAIN_CHUNK} save-dir ${SAVEDIR}"
  [ -z "$NEWEST" ] || echo "newest checkpoint: ${NEWEST}"
  echo "later legs in the manifest: ${MANIFEST_LATER[*]:-<none>}"

  # --- F. already finished? ----------------------------------------------------
  if [ "$S" -ge "$CHAIN_TOTAL" ]; then
    echo "CHAINDONE at step ${S} (budget ${CHAIN_TOTAL})"
    cancel_later "CHAINDONE"
    return 0
  fi

  LEG=$((S / CHAIN_CHUNK))
  # CHECKPOINT-INDEXED reseeding (Codex round-D N1), not leg-indexed: a leg that
  # crashes mid-chunk is retried from the last CHECKPOINT, not from the chunk
  # boundary, so a leg-indexed seed would hand the retry the same seed as the
  # original attempt and replay the same permutation over the steps that already
  # succeeded. Indexing on the checkpoint makes every distinct resume point a
  # distinct seed (S=60000 -> 66, a retry at S=62500 -> 67), while S=0 still gives
  # the recipe's seed 42 and the probe still walks 42/43/44 at 0/100/200.
  SEED=$((42 + S / CHECKPOINT_EVERY))
  TARGET=$((S + CHAIN_CHUNK))
  [ "$TARGET" -le "$CHAIN_TOTAL" ] || TARGET="$CHAIN_TOTAL"

  # --- G. watchdog half 1: the attempts stamp ---------------------------------
  # "S count": how many legs have STARTED at this S without moving it. A second
  # consecutive no-progress attempt is a systematic failure (bad checkpoint, OOM,
  # a code bug) — nine legs would each reproduce it, so the chain stops instead.
  STAMP="$SAVEDIR/chain_attempts.stamp"
  STAMP_LOCK="$SAVEDIR/.chain_stamp.lock"
  # Read, decide and write happen INSIDE ONE lock, held on a SEPARATE lockfile
  # (review N3). The previous version took the lock on the stamp itself with
  # `200>"$STAMP"`, which truncates before the lock is granted — a second leg
  # could read the emptied file and reset the strike count, defeating the
  # watchdog exactly when two legs are contending, i.e. when it matters.
  if [ "$DRYRUN" = "1" ]; then
    # Read-only preview: a dry run takes no lock and creates no lockfile, so it
    # leaves the filesystem byte-identical.
    stamp_read || die "attempts stamp ${STAMP} is unreadable or malformed" 11
    stamp_decide
    [ "$STAMP_HALT" = "0" ] || chainhalt
    echo "DRYRUN would: write attempts stamp '${S} ${STAMP_NEXT}' to ${STAMP}"
  else
    mkdir -p "$SAVEDIR" || die "could not create save-dir ${SAVEDIR}" 3
    exec 9>"$STAMP_LOCK" || die "could not open the stamp lock ${STAMP_LOCK}" 11
    flock -w 30 9 || die "could not take the stamp lock ${STAMP_LOCK} within 30s" 11
    stamp_read || die "attempts stamp ${STAMP} is unreadable or malformed" 11
    stamp_decide
    [ "$STAMP_HALT" = "0" ] || chainhalt      # exits while still holding the lock
    echo "$S $STAMP_NEXT" > "$STAMP" || die "could not update the attempts stamp ${STAMP}" 11
    exec 9>&-
    echo "attempts stamp: ${S} ${STAMP_NEXT}"
  fi

  # --- H. the argv -------------------------------------------------------------
  RESUME_ARGV=()
  if [ "$S" -gt 0 ]; then
    RESUME_ARGV=(--ckpt-path "$NEWEST")
    echo "RESUME from ${NEWEST} (step ${S}) seed=${SEED} leg=${LEG}   # not bit-exact: no RNG/dataloader state"
  else
    echo "fresh start: no checkpoint under ${SAVEDIR}/${RUN_NAME}/*/checkpoints (leg ${LEG}, seed ${SEED})"
  fi
  TRAIN_ARGV=(
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
    --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors
    --save-dir "$SAVEDIR"
    --name "$RUN_NAME"
    --experiment-name "$EXPERIMENT_NAME"
    --batch-size 64
    --num-gpus 1
    --accum-batches 1
    --num-workers 8
    --checkpoint-every "$CHECKPOINT_EVERY"
    --val-every "$VAL_EVERY"
    --max-steps "$TARGET"
    --seed "$SEED"
    "${RESUME_ARGV[@]}"
  )
  echo "--- chain leg ${LEG}: ${S} -> ${TARGET} (seed ${SEED}) ---"
  echo "${PYEXE} train.py ${TRAIN_ARGV[*]}"

  if [ "$DRYRUN" = "1" ]; then
    echo "DRY RUN complete: gates A-G passed (leg ${LEG}, S=${S}, TARGET=${TARGET}, SEED=${SEED}; GPU gate skipped, nothing executed)"
    return 0
  fi

  # --- I. GPU ------------------------------------------------------------------
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  GPUINFO="$(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)" \
    || die "nvidia-smi --query-gpu FAILED - the GPU could not be disclosed" 6
  echo "GPU: ${GPUINFO}"
  mkdir -p "$WANDB_DIR" || die "could not create WANDB_DIR ${WANDB_DIR}" 3

  # --- J. run ------------------------------------------------------------------
  "$PYEXE" train.py "${TRAIN_ARGV[@]}"
  TRAIN_RC=$?

  # --- K. did it actually get there? -------------------------------------------
  # rc=0 is not evidence: a leg that exits clean without reaching TARGET has not
  # advanced the chain, and treating it as success would let the successor start
  # from the same S and burn the watchdog budget silently.
  discover_newest || die "post-run checkpoint discovery FAILED under ${SAVEDIR}" 9
  S_NEW="$NEWEST_STEP"
  if [ "$TRAIN_RC" -ne 0 ]; then
    echo "CHAINFAIL leg=${LEG} ${S} -> ${S_NEW} (target ${TARGET}) seed=${SEED} rc=${TRAIN_RC}"
    return "$TRAIN_RC"
  fi
  if [ "$S_NEW" -ne "$TARGET" ]; then
    echo "CHAINFAIL leg=${LEG} ${S} -> ${S_NEW} but TARGET was ${TARGET} (train.py exited 0)"
    die "leg ${LEG} exited 0 without reaching TARGET ${TARGET} (newest step ${S_NEW})" 13
  fi
  echo "CHAINRESULT leg=${LEG} ${S} -> ${S_NEW} seed=${SEED} rc=0"
  if [ "$S_NEW" -ge "$CHAIN_TOTAL" ]; then
    echo "CHAINDONE at step ${S_NEW} (budget ${CHAIN_TOTAL})"
    # A spare leg would CHAINDONE by itself; cancelling frees the queue slot.
    cancel_later "CHAINDONE"
  fi
  return 0
}

# --- LOG FIRST (round-B pattern): pre-open, then exec into tee BEFORE the gates.
# The leg index is not known until gate E, so the name carries the job id and the
# timestamp; a dying tee SIGPIPEs the shell, which is the fail-closed behaviour.
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_chain${LOG_JID}.log"
if [ "$DRYRUN" != "1" ]; then
  : > "$LOG" || { echo "GATEFAIL rc=8 could not open the run log ${LOG}"; exit 8; }
  exec > >(tee -- "$LOG") 2>&1
fi

main
exit $?
