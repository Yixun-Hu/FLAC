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
# Parameters via --export:
#   EXPECT_SHA  (required) the reviewed commit this leg is bound to
#   MAXSTEPS    (optional, default 67500)
#   SMOKE=1     (optional) the throughput/ETA measurement: 300 steps, validation
#               every 50, checkpointing effectively off, into a /smoke save-dir.
#               It prices the full leg's --time; it is not a training result.
#
# della specifics: NO --partition line (della REJECTS an explicit partition for
# GPU jobs and auto-routes from GRES) and NO --time line — the wall-clock budget
# is a MEASURED quantity (smoke rate x1.5) and is passed at submit
# (`sbatch --time=...`), which is also what selects the QOS. --constraint=gpu80
# excludes the 40GB A100s; H100s are unreachable by the blanchette account, so
# Phase 2 runs on an A100 80GB per the plan's Yixun-approved hardware amendment
# (recipe deviation = GPU architecture only; disclosed in every artifact).
#
# Resume: a PL resume is NOT bit-exact (no RNG or dataloader position is restored)
# — the RESUME log line is the disclosure hook, and a smoke never resumes.
#
# Submit through della_submit.sh. By hand:
#   sbatch --job-name=exp16-train --time=2-12:00:00 \
#          --export=ALL,EXPECT_SHA=$(git -C /home/yh4742/codespace/FLAC rev-parse HEAD) \
#          /home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_train.sbatch
#
# DRYRUN=1 (outside Slurm) runs every gate that does not need a GPU, echoes the
# FULL final argv and exits 0.
#
# Exit codes: 2 parameters, 3 repo/cd, 4 commit binding, 5 interpreter provenance,
# 6 GPU, 7 ntasks, 8 log durability; otherwise train.py's own rc.
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

export HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache
export HF_HUB_OFFLINE=1          # della compute nodes are OFFLINE: fetches must fail
export TRANSFORMERS_OFFLINE=1    # loudly, not hang, and the ViT resolves from models/
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb

DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
JOBKIND=train
[ "$SMOKE" = "1" ] && JOBKIND=train_smoke
TS="$(date +%Y-%m-%d_%H:%M:%S)"
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_${JOBKIND}${LOG_JID}.log"

BASE_SAVEDIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

# train.py puts checkpoints under <save-dir>/<--name>/<--experiment-name>/checkpoints
# (project/run of the offline W&B logger), i.e. three levels down — hence globstar
# rather than a flat glob. nullglob + an array so an EMPTY tree can never degrade
# into bare `ls -t` (which would list the CWD and hand back a non-checkpoint).
latest_ckpt() {
  local cands=()
  shopt -s globstar nullglob
  cands=("$SAVEDIR"/**/*.ckpt)
  shopt -u globstar nullglob
  [ "${#cands[@]}" -gt 0 ] || return 0
  ls -t "${cands[@]}" 2>/dev/null | head -1
}

main() {
  echo "=== exp_16 train${SMOKE:+ (SMOKE=${SMOKE})} — ${TS} — host $(hostname) — job ${SLURM_JOB_ID:-<none>} ==="

  # --- A. parameters ----------------------------------------------------------
  [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported" 2
  SAVEDIR="$BASE_SAVEDIR"
  if [ "$SMOKE" = "1" ]; then
    # A smoke is a measurement of throughput, so its knobs are FIXED, not
    # defaulted: silently honouring an exported MAXSTEPS/VAL_EVERY here would
    # produce a "smoke" that priced a different run than the one it names.
    for VAR in MAXSTEPS VAL_EVERY CHECKPOINT_EVERY; do
      [ -z "${!VAR:-}" ] || die "SMOKE=1 fixes ${VAR}; unset it (or drop SMOKE) rather than overriding a measurement" 2
    done
    MAXSTEPS=300
    VAL_EVERY=50
    CHECKPOINT_EVERY=1000000     # effectively off: a smoke writes no periodic checkpoint
    SAVEDIR="${BASE_SAVEDIR}/smoke"
  else
    MAXSTEPS="${MAXSTEPS:-67500}"
  fi
  case "$MAXSTEPS" in ''|*[!0-9]*) die "MAXSTEPS '${MAXSTEPS}' must be a positive integer" 2 ;; esac
  [ "$MAXSTEPS" -gt 0 ] || die "MAXSTEPS '${MAXSTEPS}' must be a positive integer" 2

  [ -d "$REPO" ] || die "REPO ${REPO} is not a directory" 3
  cd "$REPO" || die "could not cd ${REPO}" 3
  [ -e weights/FLAC/VAE.safetensors ] || die "VAE pretransform missing: weights/FLAC/VAE.safetensors" 2

  # --- B. commit binding + tracked drift --------------------------------------
  # Untracked files are ignored on purpose (this driver's own logs and the Slurm
  # transcripts land in EXPDIR while the job runs); tracked drift never is.
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED" 4
  DRIFT="$(git -C "$REPO" status --porcelain --untracked-files=no)" \
    || die "git status FAILED - refusing to treat an error as a clean tree" 4
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA}" 4
  [ -z "$DRIFT" ] || die "tracked files are dirty: $(echo "$DRIFT" | tr '\n' ';')" 4
  echo "commit binding OK: ${HEAD_SHA}"

  # --- C. interpreter provenance (plan §4d N4) --------------------------------
  "$PYEXE" -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task only -------------------------------------------------------
  # train.py ADDS SLURM_PROCID to the seed (train.py:100), so a multi-task launch
  # would silently train N differently-seeded runs over one save-dir (plan N13).
  [ "${SLURM_NTASKS:-1}" -le 1 ] || die "SLURM_NTASKS=${SLURM_NTASKS} > 1: this kit is single-task only" 7
  echo "ntasks OK: ${SLURM_NTASKS:-1}"
  echo "EFFECTIVE_SEED=$((42 + ${SLURM_PROCID:-0}))   # train.py: --seed 42 + SLURM_PROCID"

  # --- E. resume (never for a smoke) ------------------------------------------
  LATEST="$(latest_ckpt)"
  RESUME_ARGV=()
  if [ -n "$LATEST" ] && [ "$SMOKE" != "1" ]; then
    RESUME_ARGV=(--ckpt-path "$LATEST")
    echo "RESUME from ${LATEST}   # NOT bit-exact: PL restores no RNG state and no dataloader position"
  elif [ -n "$LATEST" ]; then
    echo "smoke: ignoring existing checkpoint ${LATEST} (a smoke always starts fresh)"
  else
    echo "no checkpoint under ${SAVEDIR}: starting from scratch"
  fi

  # --- F. the argv (built once, printed, then executed verbatim) --------------
  TRAIN_ARGV=(
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
    --val-dataset-config src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors
    --save-dir "$SAVEDIR"
    --name FLAC_vanilla_repro
    --experiment-name exp16_della_vanilla_repro
    --batch-size 64
    --num-gpus 1
    --accum-batches 1
    --num-workers 8
    --checkpoint-every "${CHECKPOINT_EVERY:-2500}"
    --val-every "${VAL_EVERY:-2500}"
    --max-steps "$MAXSTEPS"
    --seed 42
    "${RESUME_ARGV[@]}"
  )
  echo "--- train argv ---"
  echo "${PYEXE} train.py ${TRAIN_ARGV[*]}"
  echo "save-dir ${SAVEDIR} | checkpoints land in ${SAVEDIR}/FLAC_vanilla_repro/exp16_della_vanilla_repro/checkpoints"

  if [ "$DRYRUN" = "1" ]; then
    echo "DRY RUN complete: gates A-E passed (SMOKE=${SMOKE}, MAXSTEPS=${MAXSTEPS}; GPU gate skipped, nothing executed)"
    return 0
  fi

  # --- G. GPU + output directories --------------------------------------------
  # Created HERE, after the dry-run return, so a DRYRUN inspects the plan without
  # leaving directories behind on scratch.
  mkdir -p "$SAVEDIR" || die "could not create save-dir ${SAVEDIR}" 3
  mkdir -p "$WANDB_DIR" || die "could not create WANDB_DIR ${WANDB_DIR}" 3
  # (arch disclosure: A100 80GB per the hardware amendment)
  # `nvidia-smi -L` prints its FAILURE MESSAGE ON STDOUT and exits 9 on a node with
  # no driver (measured on della9: two stdout lines, rc=9). A bare `| wc -l` would
  # therefore count error prose as GPUs — fail-open the moment that text is one
  # line long. Require rc=0 AND count only real `GPU <n>:` rows.
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)"

  # --- H. run -----------------------------------------------------------------
  "$PYEXE" train.py "${TRAIN_ARGV[@]}"
  TRAIN_RC=$?
  echo "TRAINRESULT rc=${TRAIN_RC} last_ckpt=$(latest_ckpt)"
  return "$TRAIN_RC"
}

if [ "$DRYRUN" = "1" ]; then
  main
  exit $?
fi
main 2>&1 | tee "$LOG"
RCS=("${PIPESTATUS[@]}")
[ "${RCS[1]}" -eq 0 ] || { echo "GATEFAIL rc=8 tee exited ${RCS[1]} - the run log ${LOG} is not durable"; exit 8; }
exit "${RCS[0]}"
