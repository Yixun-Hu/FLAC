#!/bin/bash
# ============================================================================
# exp_16 TRAIN DRIVER (della) — the Phase-2 vanilla FLAC reproduction leg
# (plan §5 Phase 2): the paper-literal recipe, batch 64 on ONE GPU, 67,500 steps
# (the budget recovered from the released checkpoint's embedded global_step).
#
# Batch 64 on a single GPU is the recipe, not a convenience: BN sees 64 DiT-side
# and 512 RIR-encoder-side, and gradient accumulation NEVER feeds BN statistics
# (CLAUDE.md) — so the OOM contingency is ViT gradient_checkpointing or a bigger
# card, never --accum-batches (plan §5, N6).
#
# THE RECIPE NUMBERS ARE LITERAL CONSTANTS (review B1). max-steps / val-every /
# checkpoint-every are not tunable from the environment in either direction: they
# define what "the reproduction" means, and an ambient export that silently
# shortened a 67,500-step leg would produce a run that looks like the recipe and
# is not. If any of them arrives in the environment the job REFUSES to start.
#
# Parameters via --export (nothing else crosses; see ENVIRONMENT ISOLATION below):
#   EXPECT_SHA  (required) the reviewed commit this leg is bound to
#   SMOKE=1     (optional) the throughput/ETA measurement: 300 steps, validation
#               every 50, checkpointing effectively off, into a SIBLING save-dir.
#               It prices the full leg's --time; it is not a training result.
#   RESUME=1    (optional) opt-in continuation of an interrupted production leg.
#               Without it, an existing checkpoint is a HARD ERROR rather than an
#               accidental double-start — and a resume is never silent, because a
#               PL resume is NOT bit-exact (no RNG state, no dataloader position).
#
# ENVIRONMENT ISOLATION (review B1): della_submit.sh submits with an explicit
# --export list, so the job inherits NOTHING from the submitting shell except
# SLURM_* and the parameters above. PATH is therefore set here explicitly.
#
# della specifics: NO --partition line (della REJECTS an explicit partition for
# GPU jobs and auto-routes from GRES) and NO --time line — the wall-clock budget
# is a MEASURED quantity (smoke rate x1.5) and is passed at submit
# (`sbatch --time=...`), which is also what selects the QOS. --constraint=gpu80
# excludes the 40GB A100s; H100s are unreachable by the blanchette account, so
# Phase 2 runs on an A100 80GB per the plan's Yixun-approved hardware amendment
# (recipe deviation = GPU architecture only; disclosed in every artifact).
#
# Submit through della_submit.sh. By hand:
#   sbatch --job-name=exp16-train --time=2-12:00:00 \
#          --export=EXPECT_SHA=$(git -C /home/yh4742/codespace/FLAC rev-parse HEAD) \
#          /home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_train.sbatch
#
# DRYRUN=1 (outside Slurm) runs every gate that does not need a GPU, echoes the
# FULL final argv and exits 0 — writing NOTHING (no log, no directories).
#
# Exit codes: 2 parameters, 3 repo/cd, 4 closure binding, 5 interpreter provenance,
# 6 GPU, 7 ntasks, 8 log durability, 9 resume guard; otherwise train.py's own rc.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --output=/home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/slurm_train_%x_%j.out

set -uo pipefail

# --- SHARED CONSTANTS (identical in della_eval.sbatch) ------------------------
REPO=/home/yh4742/codespace/FLAC
PYEXE=/scratch/gpfs/BLANCHETTE/yh4742/conda_envs/flac/bin/python   # no conda activate needed
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"

# The job inherits only SLURM_* + the explicit exports (review B1), so PATH must
# be built here; every tool this script uses lives in /usr/bin on della.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
# HOME is not part of the export list either. git and python tolerate its absence,
# but torch/wandb resolve cache paths through it, so derive it rather than inherit.
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"

export HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache
export HF_HUB_OFFLINE=1          # della compute nodes are OFFLINE: fetches must fail
export TRANSFORMERS_OFFLINE=1    # loudly, not hang, and the ViT resolves from models/
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb
# No .pyc anywhere: a stray __pycache__ under src/ would dirty the measurement
# closure checked below (and a DRYRUN must leave the tree untouched).
export PYTHONDONTWRITEBYTECODE=1

# --- THE MEASUREMENT CLOSURE (review B3; same list verbatim in all three kit files) --
# Content-scoped rather than whole-tree: everything below can change what a job
# COMPUTES, so it must be clean and identical to the submitted commit. The worklog
# deliberately is NOT in it — the command record and the run logs are written
# while jobs are queued, and a record commit must never invalidate a queued job.
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
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_eval.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_train.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_submit.sh
)

# --- THE RECIPE (literal; never read from the environment) -------------------
PROD_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro
# A SIBLING, not a child: a smoke's checkpoints must be outside the production
# glob BY CONSTRUCTION, so they can never be mistaken for a resumable leg.
SMOKE_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro_smoke
BUDGET_STEPS=67500          # the released checkpoint's embedded global_step
RUN_NAME=FLAC_vanilla_repro
EXPERIMENT_NAME=exp16_della_vanilla_repro

DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
JOBKIND=train
[ "$SMOKE" = "1" ] && JOBKIND=train_smoke
TS="$(date +%Y-%m-%d_%H:%M:%S)"
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_${JOBKIND}${LOG_JID}.log"

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

# --- LOG FIRST, then every line below is captured (review B4) -----------------
# Pre-opened so an unwritable path fails BEFORE the gates rather than after a GPU
# has been spent, then stdout/stderr are exec'd into tee so gate output is logged
# too. NOTE: with exec-into-tee there is no pipeline to inspect — if tee dies the
# shell takes SIGPIPE on its next write and the default disposition kills it,
# which is the fail-closed behaviour we want for a multi-day leg.
if [ "$DRYRUN" != "1" ]; then
  : > "$LOG" || { echo "GATEFAIL rc=8 could not open the run log ${LOG}"; exit 8; }
  exec > >(tee -- "$LOG") 2>&1
fi

main() {
  echo "=== exp_16 train (SMOKE=${SMOKE}, RESUME=${RESUME}) — ${TS} — host $(hostname) — job ${SLURM_JOB_ID:-<none>} ==="

  # --- A. parameters -----------------------------------------------------------
  [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported" 2
  # Defence in depth against a hand-rolled `sbatch --export=ALL`: the recipe knobs
  # are constants below, so their PRESENCE in the environment is itself the error.
  for VAR in MAXSTEPS VAL_EVERY CHECKPOINT_EVERY; do
    [ -z "${!VAR:-}" ] \
      || die "${VAR} is a LITERAL recipe constant in this script and must not be in the environment (submit through della_submit.sh)" 2
  done
  case "$SMOKE" in 0|1) ;; *) die "SMOKE='${SMOKE}' must be 0 or 1" 2 ;; esac
  case "$RESUME" in 0|1) ;; *) die "RESUME='${RESUME}' must be 0 or 1" 2 ;; esac
  [ "$SMOKE" != "1" ] || [ "$RESUME" != "1" ] \
    || die "SMOKE=1 with RESUME=1 is meaningless: a smoke always starts fresh" 2

  if [ "$SMOKE" = "1" ]; then
    MAXSTEPS=300
    VAL_EVERY=50
    CHECKPOINT_EVERY=1000000     # effectively off: a smoke writes no periodic checkpoint
    SAVEDIR="$SMOKE_SAVEDIR"
  else
    MAXSTEPS="$BUDGET_STEPS"
    VAL_EVERY=2500
    CHECKPOINT_EVERY=2500
    SAVEDIR="$PROD_SAVEDIR"
  fi

  [ -d "$REPO" ] || die "REPO ${REPO} is not a directory" 3
  cd "$REPO" || die "could not cd ${REPO}" 3
  [ -e weights/FLAC/VAE.safetensors ] || die "VAE pretransform missing: weights/FLAC/VAE.safetensors" 2

  # --- B. closure binding (review B3) -----------------------------------------
  # Two questions, both about CONTENT, neither about the whole tree:
  #   (a) is the closure clean in the worktree right now?
  #   (b) is it byte-identical to the submitted commit, even if HEAD has since
  #       moved for a worklog/record commit?
  # Together they bind a multi-day leg to reviewed code while leaving the worklog
  # writable — the exp_11 lesson, where record commits killed queued legs.
  git -C "$REPO" cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null \
    || die "EXPECT_SHA ${EXPECT_SHA} is not a commit in ${REPO}" 4
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED" 4
  CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
    || die "git status FAILED - refusing to treat an error as a clean closure" 4
  [ -z "$CLOSURE_DIRT" ] \
    || die "measurement closure is dirty: $(echo "$CLOSURE_DIRT" | tr '\n' ';')" 4
  git -C "$REPO" diff --quiet "$EXPECT_SHA" HEAD -- "${CLOSURE[@]}"
  DIFF_RC=$?
  case "$DIFF_RC" in
    0) ;;
    1) die "measurement closure changed between EXPECT_SHA ${EXPECT_SHA} and HEAD ${HEAD_SHA}" 4 ;;
    *) die "git diff FAILED (rc=${DIFF_RC}) - refusing to treat an error as an unchanged closure" 4 ;;
  esac
  echo "closure binding OK: EXPECT_SHA ${EXPECT_SHA} == HEAD ${HEAD_SHA} content over ${#CLOSURE[@]} paths"

  # --- C. interpreter provenance (plan §4d N4) --------------------------------
  "$PYEXE" -B -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task only -------------------------------------------------------
  # train.py ADDS SLURM_PROCID to the seed (train.py:100), so a multi-task launch
  # would silently train N differently-seeded runs over one save-dir (plan N13).
  [ "${SLURM_NTASKS:-1}" -le 1 ] || die "SLURM_NTASKS=${SLURM_NTASKS} > 1: this kit is single-task only" 7
  echo "ntasks OK: ${SLURM_NTASKS:-1}"
  echo "EFFECTIVE_SEED=$((42 + ${SLURM_PROCID:-0}))   # train.py: --seed 42 + SLURM_PROCID"

  # --- E. resume: OPT-IN, and fail-closed both ways (review B2) ----------------
  # ONE glob shape, matching train.py's layout exactly
  # (<save-dir>/<--name>/<run-id>/checkpoints/): the run id varies per launch,
  # hence the single `*` level. `**` is deliberately gone — it would have reached
  # into a nested smoke directory, which is why the smoke dir is now a sibling.
  # nullglob + array, so an empty tree yields an EMPTY LIST rather than a literal
  # pattern or a bare `ls -t` (which lists the CWD and returns a non-checkpoint).
  RESUME_ARGV=()
  if [ "$SMOKE" = "1" ]; then
    echo "smoke: never resumes; save-dir ${SAVEDIR} (sibling of the production tree)"
  else
    local -a CANDS=()
    shopt -s nullglob
    CANDS=("$SAVEDIR/$RUN_NAME"/*/checkpoints/epoch=*-step=*.ckpt)
    shopt -u nullglob
    if [ "${#CANDS[@]}" -eq 0 ]; then
      [ "$RESUME" != "1" ] \
        || die "RESUME=1 but no checkpoint matches ${SAVEDIR}/${RUN_NAME}/*/checkpoints/epoch=*-step=*.ckpt - refusing to silently start a fresh run under a resume request" 9
      echo "no checkpoint under ${SAVEDIR}/${RUN_NAME}/*/checkpoints: starting from scratch"
    else
      [ "$RESUME" = "1" ] \
        || die "${#CANDS[@]} checkpoint(s) exist under ${SAVEDIR}/${RUN_NAME}/*/checkpoints; pass RESUME=1 or clean the save dir" 9
      # No `| head -1`: under `set -o pipefail` head's early exit can SIGPIPE ls
      # and turn a successful discovery into a failed pipeline. Take the first
      # line of the captured output instead.
      local LATEST SORTED
      SORTED="$(ls -t -- "${CANDS[@]}")" \
        || die "checkpoint discovery FAILED (ls -t over ${#CANDS[@]} candidates)" 9
      LATEST="${SORTED%%$'\n'*}"
      [ -n "$LATEST" ] || die "checkpoint discovery produced no winner among ${#CANDS[@]} candidates" 9
      local BASE STEP
      BASE="$(basename -- "$LATEST")"
      [[ "$BASE" =~ ^epoch=[0-9]+-step=([0-9]+)\.ckpt$ ]] \
        || die "resume candidate '${BASE}' is not epoch=<n>-step=<n>.ckpt" 9
      STEP="${BASH_REMATCH[1]}"
      [ "$((10#$STEP))" -lt "$BUDGET_STEPS" ] \
        || die "resume candidate is at step ${STEP} >= the ${BUDGET_STEPS}-step budget: a completed run must not silently resume-restart" 9
      RESUME_ARGV=(--ckpt-path "$LATEST")
      echo "RESUME from ${LATEST} (step ${STEP})   # NOT bit-exact: PL restores no RNG state and no dataloader position"
    fi
  fi

  # --- F. the argv (built once, printed, then executed verbatim) --------------
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
    --max-steps "$MAXSTEPS"
    --seed 42
    "${RESUME_ARGV[@]}"
  )
  echo "--- train argv ---"
  echo "${PYEXE} train.py ${TRAIN_ARGV[*]}"
  echo "save-dir ${SAVEDIR} | checkpoints land in ${SAVEDIR}/${RUN_NAME}/<run-id>/checkpoints"

  if [ "$DRYRUN" = "1" ]; then
    echo "DRY RUN complete: gates A-E passed (SMOKE=${SMOKE}, RESUME=${RESUME}, MAXSTEPS=${MAXSTEPS}; GPU gate skipped, nothing executed)"
    return 0
  fi

  # --- G. GPU + output directories --------------------------------------------
  # Created HERE, after the dry-run return, so a DRYRUN inspects the plan without
  # leaving directories behind on scratch.
  # `nvidia-smi -L` prints its FAILURE MESSAGE ON STDOUT and exits 9 on a node with
  # no driver (measured on della9: two stdout lines, rc=9). A bare `| wc -l` would
  # therefore count error prose as GPUs — fail-open the moment that text is one
  # line long. Require rc=0 AND count only real `GPU <n>:` rows.
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  # Arch disclosure (A100 80GB per the hardware amendment) is a GATE: a leg whose
  # GPU cannot be named cannot be disclosed.
  GPUINFO="$(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)" \
    || die "nvidia-smi --query-gpu FAILED - the GPU could not be disclosed" 6
  echo "GPU: ${GPUINFO}"
  mkdir -p "$SAVEDIR" || die "could not create save-dir ${SAVEDIR}" 3
  mkdir -p "$WANDB_DIR" || die "could not create WANDB_DIR ${WANDB_DIR}" 3

  # --- H. run -----------------------------------------------------------------
  "$PYEXE" train.py "${TRAIN_ARGV[@]}"
  TRAIN_RC=$?
  local LAST="" FINAL_SORTED=""
  shopt -s nullglob
  local -a FINAL=("$SAVEDIR/$RUN_NAME"/*/checkpoints/epoch=*-step=*.ckpt)
  shopt -u nullglob
  if [ "${#FINAL[@]}" -gt 0 ] && FINAL_SORTED="$(ls -t -- "${FINAL[@]}")"; then
    LAST="${FINAL_SORTED%%$'\n'*}"
  fi
  echo "TRAINRESULT rc=${TRAIN_RC} last_ckpt=${LAST:-<none>}"
  return "$TRAIN_RC"
}

main
exit $?
