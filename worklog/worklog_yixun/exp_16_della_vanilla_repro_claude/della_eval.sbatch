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
# della specifics: NO --partition line — della REJECTS an explicit partition for
# GPU jobs and auto-routes from the GRES request. H100s live only in pli*/cryoem/
# ailab (unreachable by the blanchette account), so the kit targets A100 (plan §4d
# hardware amendment). Account/QOS are left to della's defaults (QOS is derived
# from --time), matching the round-B spec.
#
# Submit through della_submit.sh (it owns the pushed-SHA check and the command
# record). By hand:
#   sbatch --job-name=exp16-eval-unseen_s42 \
#          --export=ALL,EXPECT_SHA=$(git -C /home/yh4742/codespace/FLAC rev-parse HEAD),CELL=unseen_s42 \
#          /home/yh4742/codespace/FLAC/worklog/worklog_yixun/exp_16_della_vanilla_repro_claude/della_eval.sbatch
#
# DRYRUN=1 (outside Slurm) runs every gate that does not need a GPU, echoes the
# FULL final argv and exits 0.
#
# Exit codes: 2 parameters, 3 repo/cd, 4 commit binding, 5 interpreter provenance,
# 6 GPU, 7 ntasks, 8 log durability, 9 missing artifact; otherwise eval_FLAC's own rc.
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

export HF_HOME=/scratch/gpfs/BLANCHETTE/yh4742/hf_cache
export HF_HUB_OFFLINE=1          # della compute nodes are OFFLINE: fetches must fail
export TRANSFORMERS_OFFLINE=1    # loudly, not hang, and the ViT resolves from models/
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/BLANCHETTE/yh4742/FLAC/wandb

DRYRUN="${DRYRUN:-0}"
JOBKIND=eval
TS="$(date +%Y-%m-%d_%H:%M:%S)"
LOG_JID="${SLURM_JOB_ID:+_jid${SLURM_JOB_ID}}"
LOG="${EXPDIR}/della_vanilla_repro_${TS}_${JOBKIND}${LOG_JID}.log"

die() {   # <one-line GATEFAIL message> <distinct nonzero exit code>
  echo "GATEFAIL rc=${2} ${1}"
  exit "${2}"
}

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

  # --- B. commit binding + tracked drift --------------------------------------
  # Untracked files are ignored on purpose: this driver's own logs and the Slurm
  # transcripts land in EXPDIR while the job runs, and a job must not fail because
  # it wrote its own log. Tracked drift is never tolerated — it would mean the code
  # under measurement is not the code that was reviewed.
  HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED" 4
  DRIFT="$(git -C "$REPO" status --porcelain --untracked-files=no)" \
    || die "git status FAILED - refusing to treat an error as a clean tree" 4
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA}" 4
  [ -z "$DRIFT" ] || die "tracked files are dirty: $(echo "$DRIFT" | tr '\n' ';')" 4
  echo "commit binding OK: ${HEAD_SHA}"

  # --- C. interpreter provenance (plan §4d N4) --------------------------------
  # `pip install .` copied a whole src/ tree into site-packages; if that copy wins
  # the import, this job measures STALE code that looks exactly like the checkout.
  # Assert the resolved module file lives under REPO (CWD is REPO, set above).
  "$PYEXE" -c "import src.models.conditioners as c; assert c.__file__.startswith('$REPO'), c.__file__" \
    || die "interpreter provenance FAILED: src.models.conditioners does not resolve under ${REPO}" 5
  echo "interpreter provenance OK: ${PYEXE} imports src.* from ${REPO}"

  # --- D. one task only -------------------------------------------------------
  # train.py adds SLURM_PROCID to the seed and eval under srun -n>1 would run N
  # duplicate evaluations over the same output paths (plan §1, N13).
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

  # --- F. GPU (arch disclosure: the plan's hardware amendment is A100, not H100) --
  # `nvidia-smi -L` prints its FAILURE MESSAGE ON STDOUT and exits 9 on a node with
  # no driver (measured on della9: two stdout lines, rc=9). A bare `| wc -l` would
  # therefore count error prose as GPUs — fail-open the moment that text is one
  # line long. Require rc=0 AND count only real `GPU <n>:` rows.
  GPU_LIST="$(nvidia-smi -L 2>/dev/null)" || die "nvidia-smi -L FAILED - no usable GPU on this node" 6
  GPU_COUNT="$(printf '%s\n' "$GPU_LIST" | grep -c '^GPU [0-9]')"
  [ "$GPU_COUNT" -eq 1 ] || die "expected exactly 1 visible GPU, got ${GPU_COUNT}" 6
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader)"

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

if [ "$DRYRUN" = "1" ]; then
  main
  exit $?
fi
main 2>&1 | tee "$LOG"
RCS=("${PIPESTATUS[@]}")
[ "${RCS[1]}" -eq 0 ] || { echo "GATEFAIL rc=8 tee exited ${RCS[1]} - the run log ${LOG} is not durable"; exit 8; }
exit "${RCS[0]}"
