#!/bin/bash
# ============================================================================
# exp_16 PHASE-3 EVAL DRIVER (della) — ONE evaluation cell on the REPRODUCTION
# checkpoint produced by the 9-leg chain (epoch=14-step=67500.ckpt, the artifact
# that mirrors the release).
#
# Unlike the Phase-1 kit this driver has no chain manifest, and DIFFERENT cells are
# meant to run concurrently — that is the whole point of submitting 14 at once. The
# exclusion that remains is narrow and exact: no second job running THIS cell (they
# would race to write one metrics JSON) and no training job (it writes the
# checkpoint directory this cell reads).
#
# Registered cells (an unregistered CELL is refused; all use the same protocol
# literals, only the split / seed / checkpoint change):
#   u8_s42 … u8_s46      headline, full unseen split K=8, seeds 42-46
#   u1_s42 … u1_s46      the K=1 schedule of the same split and seeds
#   u8_s42_step62500     endpoint-draw screen on the 62,500-step checkpoint
#   u8_s42_step65000     endpoint-draw screen on the 65,000-step checkpoint
#   seen_s42             seen split, descriptive
#   u8_s42_a100          arch spot-check: identical to u8_s42 but a distinct
#                        eval name, so the A100 JSON cannot overwrite the H200 one
#   c8_s42 … c8_s46      the CONTINUOUS arm's headline, K=8, seeds 42-46
#   c1_s42 … c1_s46      the continuous arm at K=1, seeds 42-46
#
# The c-cells are the other half of the chunking A/B: same protocol, same budget,
# a checkpoint from the single-job run. Their checkpoint FILENAME is identical to
# the chunked arm's (epoch=14-step=67500.ckpt) — only the eval name separates the
# artifacts, which is why the c-cells' names carry exp16_cont_ rather than
# exp16_repro_.
#
# The endpoint screens deliberately REUSE u8_s42's eval name: eval_FLAC's
# build_output_paths prefixes every artifact with the checkpoint's own filename
# (epoch=13-step=62500_metrics_1_1.0_…json), so the three cells cannot collide.
# Only the A100 cell needs a name suffix, because it shares the checkpoint too.
#
# Protocol literals (the exp_09 lesson: eval-protocol defaults are the trap):
# --cond-method vanilla --cond-autocast default --record-stream and the
# pre-registered --expected-stream-count. Deliberately absent: --store_predictions,
# --allow-partial-load, and any batch-size/num-workers override.
#
# VENUE: the header targets ailab/H200 (where the chain ran). The A100 spot-check
# is submitted with `--partition= --qos= --gres=gpu:a100:1` — verified with
# `sbatch --test-only` to unset the in-file directives and let della auto-route to
# the gpu partition (an explicit `--partition=gpu` is REJECTED by della).
#
# Submit through della_repro_eval_submit.sh. DRYRUN=1 outside Slurm runs every
# gate that does not need a GPU, prints the argv, and writes NOTHING.
#
# Exit codes: 2 parameters, 3 repo/dirs, 4 closure binding, 5 interpreter
# provenance, 6 GPU, 7 ntasks, 8 log durability, 9 missing artifact,
# 10 checkpoint missing/empty, 11 the squeue query itself failed,
# 12 a duplicate of this cell (or a training job) is running;
# otherwise eval_FLAC's own rc.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/slurm_reval_%x_%j.out

set -uo pipefail

# --- SHARED CONSTANTS ---------------------------------------------------------
REPO=/home/yh4742/codespace/FLAC
PYEXE=/scratch/gpfs/BLANCHETTE/yh4742/conda_envs/flac/bin/python
EXPDIR="$REPO/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude"
CKD=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro/FLAC_vanilla_repro/exp16_della_vanilla_repro/checkpoints
# The CONTINUOUS arm's tree (round F): same layout, same 2500 cadence, same
# headline filename — a different RUN. Its cells carry exp16_cont_ eval names, so
# although the checkpoint basenames collide, the artifact stems never do.
CKD_CONT=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/checkpoints/exp16_vanilla_repro_cont/FLAC_vanilla_repro/exp16_della_vanilla_repro/checkpoints
REPO_OUT="$REPO/outputs_FLAC/exp16_vanilla_repro"
# Job names that own a GPU TRAINING run in this experiment: any of them running
# means the checkpoint directory is being written while this cell reads it.
TRAINING_NAMES=exp16-train,exp16-train-smoke,exp16-train-resume,exp16-chain

export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb
export PYTHONDONTWRITEBYTECODE=1

# --- THE CLOSURE (same list verbatim in della_repro_eval_submit.sh) -----------
# The ten core paths plus this kit's two files. The Phase-1/chain kit files are
# deliberately absent: job 12681303 still pins della_eval.sbatch's closure, and a
# Phase-3 commit must not invalidate it.
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
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_repro_eval.sbatch
  worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_repro_eval_submit.sh
)

DRYRUN="${DRYRUN:-0}"
TS="$(date +%Y-%m-%d_%H:%M:%S)"

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

# --- LOG FIRST: pre-open, then exec into tee BEFORE the gates ----------------
# A dying tee SIGPIPEs the shell (fail-closed). The name carries the cell and the
# job id so 14 concurrent cells cannot collide on one file.
LOG_CELL="${CELL:-nocell}"
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_reval_${LOG_CELL}${LOG_JID}.log"
if [ "$DRYRUN" != "1" ]; then
  : > "$LOG" || { echo "GATEFAIL rc=8 could not open the run log ${LOG}"; exit 8; }
  exec > >(tee -- "$LOG") 2>&1
fi

main() {
  echo "=== exp_16 Phase-3 eval — CELL=${CELL:-<unset>} — ${TS} — host $(hostname) — job ${SLURM_JOB_ID:-<none>} ==="

  # --- A. parameters: the cell table is hardcoded ------------------------------
  [ -n "${CELL:-}" ] || die "CELL must be exported (u8_s4[2-6]|u1_s4[2-6]|c8_s4[2-6]|c1_s4[2-6]|u8_s42_step62500|u8_s42_step65000|seen_s42|u8_s42_a100)" 2
  [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported" 2
  UNSEEN_K8=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
  UNSEEN_K1=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
  SEEN_K8=src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
  CKPT_67500="$CKD/epoch=14-step=67500.ckpt"
  CKPT_CONT_67500="$CKD_CONT/epoch=14-step=67500.ckpt"
  # Longest patterns first: a plain `u8_s42` glob must never swallow `u8_s42_a100`.
  case "$CELL" in
    u8_s42_step62500)
      DSCFG="$UNSEEN_K8"; SEED=42; COUNT=6337
      CKPT="$CKD/epoch=13-step=62500.ckpt"; EVALNAME=exp16_repro_unseen_K8_seed42 ;;
    u8_s42_step65000)
      DSCFG="$UNSEEN_K8"; SEED=42; COUNT=6337
      CKPT="$CKD/epoch=14-step=65000.ckpt"; EVALNAME=exp16_repro_unseen_K8_seed42 ;;
    u8_s42_a100)
      DSCFG="$UNSEEN_K8"; SEED=42; COUNT=6337
      CKPT="$CKPT_67500"; EVALNAME=exp16_repro_unseen_K8_seed42_a100 ;;
    seen_s42)
      DSCFG="$SEEN_K8"; SEED=42; COUNT=6217
      CKPT="$CKPT_67500"; EVALNAME=exp16_repro_seen_K8_seed42 ;;
    u8_s4[2-6])
      DSCFG="$UNSEEN_K8"; SEED="${CELL#u8_s}"; COUNT=6337
      CKPT="$CKPT_67500"; EVALNAME="exp16_repro_unseen_K8_seed${SEED}" ;;
    c8_s4[2-6])
      DSCFG="$UNSEEN_K8"; SEED="${CELL#c8_s}"; COUNT=6337
      CKPT="$CKPT_CONT_67500"; EVALNAME="exp16_cont_unseen_K8_seed${SEED}" ;;
    c1_s4[2-6])
      DSCFG="$UNSEEN_K1"; SEED="${CELL#c1_s}"; COUNT=6337
      CKPT="$CKPT_CONT_67500"; EVALNAME="exp16_cont_unseen_K1_seed${SEED}" ;;
    u1_s4[2-6])
      DSCFG="$UNSEEN_K1"; SEED="${CELL#u1_s}"; COUNT=6337
      CKPT="$CKPT_67500"; EVALNAME="exp16_repro_unseen_K1_seed${SEED}" ;;
    *) die "CELL '${CELL}' is not registered for exp_16 Phase 3" 2 ;;
  esac

  [ -d "$REPO" ] || die "REPO ${REPO} is not a directory" 3
  cd "$REPO" || die "could not cd ${REPO}" 3
  [ -f "$DSCFG" ] || die "dataset config not found: ${DSCFG}" 2

  # --- A2. the checkpoint must be there, and must not be a stub ----------------
  # An interrupted copy leaves a zero-byte file that torch.load reports as a
  # corrupt archive twenty minutes into a queue slot; catch it in the first second.
  [ -f "$CKPT" ] || die "checkpoint not found: ${CKPT}" 10
  [ -s "$CKPT" ] || die "checkpoint is empty: ${CKPT}" 10
  echo "checkpoint: ${CKPT} ($(stat -c '%s' "$CKPT") bytes)"

  # --- B. closure binding ------------------------------------------------------
  git -C "$REPO" cat-file -e "${EXPECT_SHA}^{commit}" 2>/dev/null \
    || die "EXPECT_SHA ${EXPECT_SHA} is not a commit in ${REPO}" 4
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED" 4
  CLOSURE_DIRT="$(git -C "$REPO" status --porcelain -- "${CLOSURE[@]}")" \
    || die "git status FAILED - refusing to treat an error as a clean closure" 4
  [ -z "$CLOSURE_DIRT" ] || die "eval closure is dirty: $(echo "$CLOSURE_DIRT" | tr '\n' ';')" 4
  git -C "$REPO" diff --quiet "$EXPECT_SHA" HEAD -- "${CLOSURE[@]}"
  DIFF_RC=$?
  case "$DIFF_RC" in
    0) ;;
    1) die "eval closure changed between EXPECT_SHA ${EXPECT_SHA} and HEAD ${HEAD_SHA}" 4 ;;
    *) die "git diff FAILED (rc=${DIFF_RC})" 4 ;;
  esac
  echo "closure binding OK: EXPECT_SHA ${EXPECT_SHA} == HEAD ${HEAD_SHA} content over ${#CLOSURE[@]} paths"

  # --- C. interpreter provenance -----------------------------------------------
  "$PYEXE" -B -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task -------------------------------------------------------------
  [ "${SLURM_NTASKS:-1}" -le 1 ] || die "SLURM_NTASKS=${SLURM_NTASKS} > 1: this kit is single-task only" 7
  echo "ntasks OK: ${SLURM_NTASKS:-1}"

  # --- D2. no DUPLICATE of this cell, and no training job, running -------------
  # Cells are concurrent by design — but only DIFFERENT cells. Two jobs running the
  # SAME cell would race to write one metrics JSON and one stream sidecar, and the
  # loser's partial file could be the one that survives. The wrapper refuses a
  # duplicate at submit time; this is the second half of that guarantee, for a
  # duplicate that slipped through a race between two submitters (or a hand-rolled
  # sbatch). A training job is refused too: it writes the checkpoint directory this
  # cell is reading from.
  #
  # The QUERY is checked before its OUTPUT is: a controller outage makes squeue
  # print nothing and exit nonzero, and an empty result then reads exactly like
  # "no duplicate exists". Fatal even under DRYRUN — dry runs happen on the login
  # node, where squeue works.
  MY_NAME="exp16-reval-${CELL}"
  DUP_RAW="$(squeue -h -u "$(id -un)" -n "$MY_NAME" -t R -o '%i')"
  SQ_RC=$?
  [ "$SQ_RC" -eq 0 ] \
    || die "squeue -h -u $(id -un) -n ${MY_NAME} -t R FAILED (rc=${SQ_RC}) - refusing to read a scheduler error as 'no duplicate is running'" 11
  # grep exits 1 when self was the only row; legitimate here because the query
  # itself already proved successful.
  DUP="$(printf '%s\n' "$DUP_RAW" | grep -v "^${SLURM_JOB_ID:-__none__}\$" | tr '\n' ' ')" || true
  DUP="$(echo "$DUP" | tr -s ' ' | sed 's/^ //;s/ $//')"
  [ -z "$DUP" ] \
    || die "another job named ${MY_NAME} is RUNNING (${DUP}): two runs of one cell would race to write the same metrics JSON" 12
  TRAIN_RAW="$(squeue -h -u "$(id -un)" -n "$TRAINING_NAMES" -t R -o '%i')"
  SQ_RC=$?
  [ "$SQ_RC" -eq 0 ] \
    || die "squeue -h -u $(id -un) -n ${TRAINING_NAMES} -t R FAILED (rc=${SQ_RC}) - refusing to read a scheduler error as 'no training job is running'" 11
  TRAIN="$(printf '%s\n' "$TRAIN_RAW" | tr '\n' ' ' | tr -s ' ' | sed 's/^ //;s/ $//')"
  [ -z "$TRAIN" ] \
    || die "an exp16 training job is RUNNING (${TRAIN}): it writes the checkpoint directory this cell reads - wait for it to finish" 12
  echo "exclusion OK: no other ${MY_NAME} and no ${TRAINING_NAMES} job is running"

  # --- E. the argv --------------------------------------------------------------
  EVAL_ARGV=(
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json
    --dataset-config "$DSCFG"
    --ckpt-path "$CKPT"
    --cfg-scale 1.0
    --steps 1
    --eval-name "$EVALNAME"
    --seed "$SEED"
    --cond-method vanilla
    --cond-autocast default
    --record-stream
    --expected-stream-count "$COUNT"
  )
  # Artifacts land BESIDE the checkpoint (build_output_paths uses dirname of
  # --ckpt-path), i.e. on scratch — the checkpoint's own filename is the stem's
  # prefix, which is what keeps the three endpoint cells apart.
  # Derived from the CHECKPOINT's own directory, exactly as build_output_paths
  # does it — not from CKD. With two arms in two trees a hardcoded CKD would send
  # a continuous cell looking for its metrics in the chunked arm's directory.
  CKPT_STEM="$(basename -- "$CKPT" .ckpt)"
  CKPT_DIR="$(dirname -- "$CKPT")"
  METRICS="${CKPT_DIR}/${CKPT_STEM}_metrics_1_1.0_${EVALNAME}.json"
  STREAM="${CKPT_DIR}/${CKPT_STEM}_metrics_1_1.0_${EVALNAME}.stream.json"
  echo "--- eval argv ---"
  echo "${PYEXE} eval_FLAC.py ${EVAL_ARGV[*]}"
  echo "--- expected artifacts ---"
  echo "metrics ${METRICS}"
  echo "stream  ${STREAM}"
  echo "repo copy dir ${REPO_OUT}"

  if [ "$DRYRUN" = "1" ]; then
    echo "DRY RUN complete: gates A-E passed for CELL=${CELL} (GPU gate skipped, nothing executed)"
    return 0
  fi

  # --- F. GPU -------------------------------------------------------------------
  # nvidia-smi -L prints its failure ON STDOUT and exits 9 on a driverless node, so
  # a bare line count would read error prose as GPUs. Check rc, then count real rows.
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  # Arch disclosure is a GATE: the A100/H200 comparison is only meaningful if every
  # cell recorded which card it ran on.
  GPUINFO="$(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)" \
    || die "nvidia-smi --query-gpu FAILED - the GPU could not be disclosed" 6
  echo "GPU: ${GPUINFO}"
  mkdir -p "$WANDB_DIR" || die "could not create WANDB_DIR ${WANDB_DIR}" 3

  # --- G. run --------------------------------------------------------------------
  "$PYEXE" eval_FLAC.py "${EVAL_ARGV[@]}"
  EVAL_RC=$?
  [ "$EVAL_RC" -eq 0 ] || die "eval_FLAC.py exited ${EVAL_RC}" "$EVAL_RC"

  # --- H. artifacts, then mirror them into the repo -----------------------------
  # The metrics JSONs are what model_comparison.md regenerates from, and scratch is
  # not backed up: a cell is not done until its numbers are in the checkout.
  [ -f "$METRICS" ] || die "metrics JSON not written: ${METRICS}" 9
  [ -f "$STREAM" ] || die "stream sidecar not written: ${STREAM}" 9
  mkdir -p "$REPO_OUT" || die "could not create ${REPO_OUT}" 3
  cp -f "$METRICS" "$REPO_OUT/" || die "could not copy the metrics JSON into ${REPO_OUT}" 9
  cp -f "$STREAM" "$REPO_OUT/" || die "could not copy the stream sidecar into ${REPO_OUT}" 9
  echo "EVALRESULT cell=${CELL} json=${METRICS} repo_copy=${REPO_OUT}/$(basename -- "$METRICS") rc=0"
  return 0
}

main
exit $?
