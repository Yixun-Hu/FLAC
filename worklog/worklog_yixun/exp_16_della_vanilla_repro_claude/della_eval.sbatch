#!/bin/bash
# ============================================================================
# exp_16 EVAL DRIVER (della) — ONE Phase-1 calibration cell on the RELEASED
# FLAC_EMA checkpoint. Phase 1 gates all Phase-2 training compute (plan §5), so
# every protocol flag is a LITERAL here: the eval-protocol defaults are the exact
# trap that produced exp_09's retracted numbers (CLAUDE.md: --cond-method must
# match how the checkpoint was trained, and nothing errors when it does not).
#
# Three registered cells (plan §5 Phase 1); an unregistered CELL is refused:
#   CELL=unseen_s42   gating Run B, full unseen split (6,337 items / 17 rooms), seed 42
#   CELL=unseen_s43   the seed-43 diagnostic of the same cell
#   CELL=seen_s42     Run A, seen split (6,217 items), DESCRIPTIVE only — it does not gate
#
# Deliberately NOT passed (each omission is part of the protocol, not an oversight):
#   --store_predictions   ~GB of decoded audio into $HOME; metrics unaffected (plan §2)
#   --allow-partial-load  an unclean load of the released checkpoint must FAIL, not warn
#   --batch-size/--num-workers  eval_FLAC's own defaults (64/4) ARE exp_01's protocol
#   --rotate-deg          default 0.0; Phase 1 is unrotated by contract (announcement 05)
#
# ENVIRONMENT ISOLATION (review B1): della_submit.sh submits with an explicit
# --export list, so the job inherits NOTHING from the submitting shell except
# SLURM_* and the two parameters below. PATH is therefore set here explicitly —
# without it git/nvidia-smi/awk would not resolve. (PYEXE is already absolute.)
#
# della specifics: NO --partition line — della REJECTS an explicit partition for
# GPU jobs and auto-routes from the GRES request. H100s live only in pli*/cryoem/
# ailab (unreachable by the blanchette account), so the kit targets A100 (plan §4d
# hardware amendment). Account/QOS are left to della's defaults (QOS is derived
# from --time).
#
# --no-requeue (plan Rev 3 §3): measured cluster policy is already JobRequeue=0
# and PreemptMode=OFF, so this is belt and braces — it pins determinism against a
# manual `scontrol requeue` and against a future policy change. A silently
# restarted cell would re-run from step 0 and overwrite its own metrics JSON.
#
# Submit through della_submit.sh (it owns the pushed-SHA check, the record
# transaction and the export list). By hand:
#   sbatch --job-name=exp16-eval-unseen_s42 \
#          --export=EXPECT_SHA=$(git -C /home/yh4742/codespace/FLAC rev-parse HEAD),CELL=unseen_s42 \
#          /home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_eval.sbatch
#
# DRYRUN=1 (outside Slurm) runs every gate that does not need a GPU, echoes the
# FULL final argv and exits 0 — writing NOTHING (no log, no directories).
#
# Exit codes: 2 parameters, 3 repo/cd, 4 closure binding, 5 interpreter provenance,
# 6 GPU, 7 ntasks, 8 log durability, 9 missing artifact; otherwise eval_FLAC's own rc.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/slurm_eval_%x_%j.out

set -uo pipefail

# --- SHARED CONSTANTS (identical in della_train.sbatch) -----------------------
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
# closure checked below (and a DRYRUN must leave the tree untouched). Belt and
# braces with the -B flag on every interpreter call.
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

DRYRUN="${DRYRUN:-0}"
JOBKIND=eval
TS="$(date +%Y-%m-%d_%H:%M:%S)"
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_${JOBKIND}${LOG_JID}.log"

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

# --- LOG FIRST, then every line below is captured (review B4) -----------------
# The log is pre-opened so an unwritable path fails BEFORE the gates rather than
# after the GPU is spent, and stdout/stderr are then exec'd into tee, so gate
# output lands in the log too. NOTE: with exec-into-tee there is no pipeline to
# inspect — if tee dies, the shell gets SIGPIPE on its next write and the default
# disposition kills it, which is the fail-closed behaviour we want (a run whose
# log stopped being durable must not keep going).
if [ "$DRYRUN" != "1" ]; then
  : > "$LOG" || { echo "GATEFAIL rc=8 could not open the run log ${LOG}"; exit 8; }
  exec > >(tee -- "$LOG") 2>&1
fi

main() {
  echo "=== exp_16 eval — CELL=${CELL:-<unset>} — ${TS} — host $(hostname) — job ${SLURM_JOB_ID:-<none>} ==="

  # --- A. parameters: the cell table is hardcoded; unknown CELL is refused -----
  # A cell is a protocol choice, so there is deliberately NO default: an omitted
  # export must not silently decide which split (and which pre-registered item
  # count) this job claims to have evaluated.
  [ -n "${CELL:-}" ] || die "CELL must be exported (unseen_s42|unseen_s43|seen_s42)" 2
  case "$CELL" in
    unseen_s42)
      DSCFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
      SEED=42; COUNT=6337; EVALNAME=exp16_calib_unseen_K8_seed42 ;;
    unseen_s43)
      DSCFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
      SEED=43; COUNT=6337; EVALNAME=exp16_calib_unseen_K8_seed43 ;;
    seen_s42)
      DSCFG=src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json
      SEED=42; COUNT=6217; EVALNAME=exp16_calib_seen_K8_seed42 ;;
    *) die "CELL '${CELL}' is not registered for exp_16 (unseen_s42|unseen_s43|seen_s42)" 2 ;;
  esac
  [ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported" 2

  [ -d "$REPO" ] || die "REPO ${REPO} is not a directory" 3
  cd "$REPO" || die "could not cd ${REPO}" 3
  [ -f "$DSCFG" ] || die "dataset config not found: ${DSCFG}" 2
  [ -e weights/FLAC/FLAC_EMA.ckpt ] || die "released checkpoint missing: weights/FLAC/FLAC_EMA.ckpt" 2

  # --- B. closure binding (review B3) -----------------------------------------
  # Two questions, both about CONTENT, neither about the whole tree:
  #   (a) is the closure clean in the worktree right now?
  #   (b) is it byte-identical to the submitted commit, even if HEAD has since
  #       moved for a worklog/record commit?
  # Together they bind the measurement to reviewed code while leaving the worklog
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
  # `pip install .` copied a whole src/ tree into site-packages; if that copy wins
  # the import, this job measures STALE code that looks exactly like the checkout.
  "$PYEXE" -B -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task only -------------------------------------------------------
  # eval under srun -n>1 would run N duplicate evaluations over the same output
  # paths (plan §1, N13).
  [ "${SLURM_NTASKS:-1}" -le 1 ] || die "SLURM_NTASKS=${SLURM_NTASKS} > 1: this kit is single-task only" 7
  echo "ntasks OK: ${SLURM_NTASKS:-1}"

  # --- E. the argv (built once, printed, then executed verbatim) --------------
  EVAL_ARGV=(
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json
    --dataset-config "$DSCFG"
    --ckpt-path weights/FLAC/FLAC_EMA.ckpt
    --cfg-scale 1.0
    --steps 1
    --eval-name "$EVALNAME"
    --seed "$SEED"
    --cond-method vanilla
    --cond-autocast default
    --record-stream
    --expected-stream-count "$COUNT"
  )
  METRICS="weights/FLAC/FLAC_EMA_metrics_1_1.0_${EVALNAME}.json"
  STREAM="weights/FLAC/FLAC_EMA_metrics_1_1.0_${EVALNAME}.stream.json"
  echo "--- eval argv ---"
  echo "${PYEXE} eval_FLAC.py ${EVAL_ARGV[*]}"
  echo "--- expected artifacts ---"
  echo "metrics ${METRICS}"
  echo "stream  ${STREAM}"

  if [ "$DRYRUN" = "1" ]; then
    echo "DRY RUN complete: gates A-E passed for CELL=${CELL} (GPU gate skipped, nothing executed)"
    return 0
  fi

  # --- F. GPU + output dirs ----------------------------------------------------
  # `nvidia-smi -L` prints its FAILURE MESSAGE ON STDOUT and exits 9 on a node with
  # no driver (measured on della9: two stdout lines, rc=9). A bare `| wc -l` would
  # therefore count error prose as GPUs — fail-open the moment that text is one
  # line long. Require rc=0 AND count only real `GPU <n>:` rows.
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  # Arch disclosure: the plan's hardware amendment is A100, not H100. It is a
  # GATE, not a nicety — a cell whose GPU cannot be named cannot be disclosed.
  GPUINFO="$(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)" \
    || die "nvidia-smi --query-gpu FAILED - the GPU could not be disclosed" 6
  echo "GPU: ${GPUINFO}"
  mkdir -p "$WANDB_DIR" || die "could not create WANDB_DIR ${WANDB_DIR}" 3

  # --- G. run -----------------------------------------------------------------
  "$PYEXE" eval_FLAC.py "${EVAL_ARGV[@]}"
  EVAL_RC=$?
  [ "$EVAL_RC" -eq 0 ] || die "eval_FLAC.py exited ${EVAL_RC}" "$EVAL_RC"

  # --- H. artifacts must exist before any result line -------------------------
  # A cell that "looks done" but wrote no metrics is the fail-open case: the
  # stream sidecar is what makes the pre-registered item count auditable.
  [ -f "$METRICS" ] || die "metrics JSON not written: ${METRICS}" 9
  [ -f "$STREAM" ] || die "stream sidecar not written: ${STREAM}" 9
  echo "EVALRESULT cell=${CELL} json=${REPO}/${METRICS} rc=0"
  return 0
}

main
exit $?
