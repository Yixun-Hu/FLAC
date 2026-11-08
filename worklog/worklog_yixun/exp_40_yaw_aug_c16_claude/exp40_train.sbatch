#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
# ============================================================================
# exp40_train.sbatch -- exp_40 yaw_aug_c16 on neuronic (plan r2 §4.4). NON-PRIMARY path.
# Submitted ONLY by exp40_submit_neuronic.sh, which builds the immutable run tree, sets
# --job-name/--comment/--gres/--cpus-per-task/--mem/--time/--output and exports
# EXP40_INTENT, EXP40_RUNTREE, EXP40_REPO, EXP40_MODE, EXP40_RUNG, EXP40_RUN_ID,
# EXP40_EXPECT_SHA, EXP40_SAVE_ROOT, EXP40_STAMP. This file is executed FROM the run tree.
# Order: job-name/intent binding -> execution-side O_EXCL claim (exit 3 if another job owns
# the run) -> record + clear SLURM rank vars -> topology -> kit check (no GPU) -> GPU gate
# -> torchrun evidence driver -> start watcher (+ in-job cadence manifest on FULL) ->
# evidence / receipt (+ audit on FULL). Exit 0 PASS | 1 FAIL | 2 gate | 3 claim | 8/9/10.
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan r2 §4.4.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }
PY="/n/fs/gatrdp/envs/flac/bin/python"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
for v in EXP40_INTENT EXP40_RUNTREE EXP40_REPO EXP40_MODE EXP40_RUNG EXP40_RUN_ID EXP40_EXPECT_SHA EXP40_SAVE_ROOT EXP40_STAMP; do
  [ -n "${!v:-}" ] || die "$v must be exported by exp40_submit_neuronic.sh"
done
MODE="$EXP40_MODE"; RUNG="$EXP40_RUNG"; RUN_ID="$EXP40_RUN_ID"; SHA="$EXP40_EXPECT_SHA"
RUNTREE="$EXP40_RUNTREE"; REPO="$EXP40_REPO"; SAVE_ROOT="$EXP40_SAVE_ROOT"; TS="$EXP40_STAMP"
case "$MODE" in PILOT|FULL) ;; *) die "EXP40_MODE must be PILOT or FULL" ;; esac
case "$RUNG" in 2x32) NGPU=2 ;; 8x8) NGPU=8 ;; *) die "EXP40_RUNG must be 2x32 or 8x8" ;; esac
KIT="$RUNTREE/$EXPREL/exp40_kit.py"; DRIVER="$RUNTREE/$EXPREL/exp40_evidence_driver.py"
[ "${SLURM_JOB_NAME:-}" = "exp40-${RUN_ID}-${MODE}-${EXP40_INTENT}" ] \
  || die "job name ${SLURM_JOB_NAME:-<unset>} does not carry intent ${EXP40_INTENT}"
LOGBASE="$REPO/$EXPREL/yaw_aug_c16_${TS}_neuronic_${MODE}_${SLURM_JOB_ID}"
echo "=== exp_40 C16 neuronic ${MODE} ${RUNG} -- job ${SLURM_JOB_ID} -- $(hostname) -- $(date '+%F %T') ==="

# --- the execution-side guarantee: exactly one job trains this run -------------------- #
"$PY" "$KIT" claim --claims-dir "$SAVE_ROOT/claims" --run-id "$RUN_ID" --mode "$MODE" \
  --intent "$EXP40_INTENT" --job-id "$SLURM_JOB_ID" || { echo "CLAIM REFUSED: another job owns this run"; exit 3; }

# --- topology; SLURM rank variables recorded, then cleared (train.py adds SLURM_PROCID to
#     the seed, and PL would elect SLURMEnvironment over TorchElastic) ------------------- #
[ "${SLURM_NNODES:-}" = 1 ] && [ "${SLURM_NTASKS:-}" = 1 ] || die "need exactly 1 node / 1 task (got ${SLURM_NNODES:-?}/${SLURM_NTASKS:-?})"
env | grep -E '^SLURM_' | sort > "${LOGBASE}_slurm_env.txt"
for v in SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_STEP_NUM_TASKS; do unset "$v"; done
unset PYTHONPATH PYTHONSTARTUP PYTHONHOME PYTHONUSERBASE
export PATH="/n/fs/gatrdp/envs/flac/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=/n/fs/gatrdp/hf_cache HF_HUB_OFFLINE=1
NS="$RUN_ID"; [ "$MODE" = PILOT ] && NS="${RUN_ID}_pilot_${TS}"
RT="$SAVE_ROOT/runtime/$NS"
TMPDIR="$(mktemp -d /tmp/exp40.XXXXXX)" || die "mktemp failed"     # short: torch_shm_manager socket path
export WANDB_MODE=offline WANDB_DIR="$RT/wandb" TMPDIR XDG_CACHE_HOME="$RT/xdg" \
  TORCH_HOME="$RT/torch" TRITON_CACHE_DIR="$RT/triton" MPLCONFIGDIR="$RT/mpl"
mkdir -p "$WANDB_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" || die "runtime dirs"
EXP40_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"; export EXP40_DRIVER_VERSION

# --- gates ----------------------------------------------------------------------------- #
LINKS=(); while IFS= read -r l; do LINKS+=(--link "$l"); done < <("$PY" -c "
import json,sys; m=json.load(open(sys.argv[1])); [print(f'{k}={v}') for k,v in sorted(m['links'].items())]" "$RUNTREE/.exp40_runtree_manifest.json")
CHECK_JSON="${LOGBASE}_check.json"
"$PY" "$KIT" check --platform neuronic --rung "$RUNG" --mode "$MODE" --run-id "$RUN_ID" \
  --save-root "$SAVE_ROOT" --stamp "$TS" --sha "$SHA" --repo "$REPO" --tree "$RUNTREE" \
  "${LINKS[@]}" --receipts "$REPO/$EXPREL/receipts" --out "$CHECK_JSON" --in-job \
  || die "gates refused"
LANES="$(seq -s, 0 $((NGPU - 1)))"
"$PY" "$KIT" gpus --platform neuronic --rung "$RUNG" --mode "$MODE" --lanes "$LANES" \
  --policy idle --expect-total "$NGPU" --out "${LOGBASE}_gpus.json" || die "GPU gate refused"
mapfile -t ARGV < <("$PY" "$KIT" argv --platform neuronic --rung "$RUNG" --mode "$MODE" \
  --run-id "$RUN_ID" --save-root "$SAVE_ROOT" --stamp "$TS") || die "argv"
echo "argv: train.py ${ARGV[*]}"

# --- train ------------------------------------------------------------------------------- #
EVDIR="$SAVE_ROOT/evidence/$NS"; mkdir -p "$EVDIR" || die "evidence dir"
EVMODE=digest; [ "$MODE" = PILOT ] && EVMODE=verbatim
TRAIN_LOG="${LOGBASE}_train.log"
( cd "$RUNTREE" && EXP40_EVIDENCE_DIR="$EVDIR" EXP40_EVIDENCE_MODE="$EVMODE" \
    exec setsid torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "$DRIVER" "${ARGV[@]}" ) > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
sleep 2; [ "$(ps -o pgid= -p "$TRAIN_PID" | tr -d ' ')" = "$TRAIN_PID" ] || { kill "$TRAIN_PID"; die "setsid did not give the training its own process group"; }
"$PY" "$KIT" watch --log "$TRAIN_LOG" --members "$NGPU" --pid "$TRAIN_PID" --pgid "$TRAIN_PID" --timeout 3600 &
WATCH_PID=$!
CKPT_DIR="$SAVE_ROOT/FLAC_exp40_C16/$NS/checkpoints"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  [ "$MODE" = FULL ] && [ -d "$CKPT_DIR" ] && "$PY" "$KIT" publish-cadence --src "$CKPT_DIR" --run-id "$RUN_ID" \
      --local-manifest "$CKPT_DIR/../CHECKPOINTS.partial.sha256" >> "${LOGBASE}_cadence.log" 2>&1
  sleep 300
done
wait "$TRAIN_PID"; TRAIN_RC=$?
wait "$WATCH_PID"; WATCH_RC=$?
echo "train rc=${TRAIN_RC} watcher rc=${WATCH_RC}"

# --- verdicts ------------------------------------------------------------------------------ #
FINAL=0
if [ "$MODE" = FULL ]; then
  [ "$TRAIN_RC" = 0 ] && "$PY" "$KIT" publish-cadence --src "$CKPT_DIR" --run-id "$RUN_ID" --exited-ok \
      --local-manifest "$CKPT_DIR/../CHECKPOINTS.partial.sha256" >> "${LOGBASE}_cadence.log" 2>&1
  EVOUT="$REPO/$EXPREL/receipts/full_evidence_${RUN_ID}.json"
  REF="$REPO/$EXPREL/receipts/pilot_neuronic_${RUNG}_${SHA}.json"
  "$PY" "$KIT" evidence --platform neuronic --rung "$RUNG" --mode FULL --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$REPO/$EXPREL/receipts" --train-rc "$TRAIN_RC" \
    --watcher-rc "$WATCH_RC" --reference "$REF" --out "$EVOUT" || FINAL=1
  "$PY" "$KIT" audit --ckpt-dir "$CKPT_DIR" --evidence "$EVOUT" --run-id "$RUN_ID" \
    --out "${LOGBASE}_audit.json" || FINAL=1
else
  "$PY" "$KIT" evidence --platform neuronic --rung "$RUNG" --mode PILOT --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$REPO/$EXPREL/receipts" --train-rc "$TRAIN_RC" \
    --watcher-rc "$WATCH_RC" --out "${LOGBASE}_evidence.json" || FINAL=1
  [ "$(find "$SAVE_ROOT/FLAC_exp40_C16/$NS" -name '*.ckpt' 2>/dev/null | wc -l)" = 0 ] || FINAL=1
fi
[ "$TRAIN_RC" = 0 ] && [ "$WATCH_RC" = 0 ] || FINAL=1
echo "=== exp_40 C16 neuronic ${MODE} finished $(date '+%F %T'): $([ $FINAL = 0 ] && echo PASS || echo FAIL) ==="
[ "$WATCH_RC" -ge 8 ] && exit "$WATCH_RC"
exit "$FINAL"
