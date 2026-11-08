#!/usr/bin/env bash
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
# ============================================================================
# exp40_train.sbatch -- exp_40 yaw_aug_c16 on neuronic (plan §4.4 / §14). NON-PRIMARY path.
# Submitted ONLY by exp40_submit_neuronic.sh, which builds the immutable run tree, sets
# --job-name/--comment/--gres/--cpus-per-task/--mem/--time/--output and exports
# EXP40_INTENT, EXP40_RUNTREE, EXP40_REPO, EXP40_MODE, EXP40_RUNG, EXP40_RUN_ID,
# EXP40_EXPECT_SHA, EXP40_SAVE_ROOT, EXP40_STAMP, EXP40_CONTROL. Executed FROM the run tree.
# Order: job-name/intent binding -> execution-side O_EXCL claim in the ONE control directory
# (exit 3 if another job owns the execution) -> record + clear SLURM rank vars -> topology ->
# kit check (no GPU) -> GPU gate on the ALLOCATED devices by UUID -> torchrun evidence driver
# -> watcher -> evidence / receipt (+ local audit on FULL). A FULL ends in
# LOCAL_COMPLETE_PENDING_PUBLICATION, never PASS: PASS needs the pull, NAS publication and
# admission on the A6000 box (plan §14). EXIT/INT/TERM trap: the training's own process group
# is terminated (SIGTERM, bounded wait, SIGKILL) and reaped.
# Exit 0 PILOT PASS / FULL LOCAL_COMPLETE_PENDING_PUBLICATION | 1 FAIL | 2 gate | 3 claim |
# 8/9/10/11 watcher | 12 cleanup failed | 130 signalled.
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan §4.4 / §14 and code review 1.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }
PY="/n/fs/gatrdp/envs/flac/bin/python"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
for v in EXP40_CONTROL EXP40_INTENT EXP40_EXECUTION_ID EXP40_RUNTREE EXP40_REPO EXP40_MODE EXP40_RUNG EXP40_RUN_ID EXP40_EXPECT_SHA EXP40_SAVE_ROOT EXP40_STAMP; do
  [ -n "${!v:-}" ] || die "$v must be exported by exp40_submit_neuronic.sh"
done
MODE="$EXP40_MODE"; RUNG="$EXP40_RUNG"; RUN_ID="$EXP40_RUN_ID"; SHA="$EXP40_EXPECT_SHA"
RUNTREE="$EXP40_RUNTREE"; REPO="$EXP40_REPO"; SAVE_ROOT="$EXP40_SAVE_ROOT"; TS="$EXP40_STAMP"
case "$MODE" in PILOT|FULL) ;; *) die "EXP40_MODE must be PILOT or FULL" ;; esac
case "$RUNG" in 2x32) NGPU=2 ;; 8x8) NGPU=8 ;; *) die "EXP40_RUNG must be 2x32 or 8x8" ;; esac
KIT="$RUNTREE/$EXPREL/exp40_kit.py"; DRIVER="$RUNTREE/$EXPREL/exp40_evidence_driver.py"
EXEC_ID="$EXP40_EXECUTION_ID"
[ "${SLURM_JOB_NAME:-}" = "exp40-${EXEC_ID}-${MODE}" ] \
  || die "job name ${SLURM_JOB_NAME:-<unset>} is not exp40-${EXEC_ID}-${MODE}"
LOGBASE="$REPO/$EXPREL/yaw_aug_c16_${TS}_neuronic_${MODE}_${SLURM_JOB_ID}"
echo "=== exp_40 C16 neuronic ${MODE} ${RUNG} -- job ${SLURM_JOB_ID} -- $(hostname) -- $(date '+%F %T') ==="

# --- the execution-side guarantee: exactly one job trains this run -------------------- #
"$PY" "$KIT" claim --claims-dir "$EXP40_CONTROL/claims" --execution-id "$EXEC_ID" --mode "$MODE" \
  --intent "$EXP40_INTENT" --job-id "$SLURM_JOB_ID" || { echo "CLAIM REFUSED: another job owns execution $EXEC_ID"; exit 3; }

# --- topology; SLURM rank variables recorded, then cleared (train.py adds SLURM_PROCID to
#     the seed, and PL would elect SLURMEnvironment over TorchElastic) ------------------- #
[ "${SLURM_NNODES:-}" = 1 ] && [ "${SLURM_NTASKS:-}" = 1 ] || die "need exactly 1 node / 1 task (got ${SLURM_NNODES:-?}/${SLURM_NTASKS:-?})"
env | grep -E '^SLURM_|^CUDA_VISIBLE_DEVICES=' | sort > "${LOGBASE}_slurm_env.txt"
for v in SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_STEP_NUM_TASKS; do unset "$v"; done
unset PYTHONPATH PYTHONSTARTUP PYTHONHOME PYTHONUSERBASE
export PATH="/n/fs/gatrdp/envs/flac/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
HF_SEALED="$SAVE_ROOT/hf_sealed"               # built by the submitter from /n/fs/gatrdp/hf_cache/hub
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HOME="$HF_SEALED" HF_HUB_CACHE="$HF_SEALED/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NS="$EXEC_ID"
RT="$SAVE_ROOT/runtime/$NS"
TMPDIR="$(mktemp -d /tmp/exp40.XXXXXX)" || die "mktemp failed"     # short: torch_shm_manager socket path
trap 'rm -rf -- "$TMPDIR"' EXIT                 # replaced by the full cleanup trap below
export WANDB_MODE=offline WANDB_DIR="$RT/wandb" TMPDIR XDG_CACHE_HOME="$RT/xdg" \
  TORCH_HOME="$RT/torch" TRITON_CACHE_DIR="$RT/triton" MPLCONFIGDIR="$RT/mpl"
mkdir -p "$WANDB_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" || die "runtime dirs"
EXP40_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u)"
[ "$(printf '%s\n' "$EXP40_DRIVER_VERSION" | wc -l)" = 1 ] || die "not exactly one NVIDIA driver version: $EXP40_DRIVER_VERSION"
export EXP40_DRIVER_VERSION

# --- gates ----------------------------------------------------------------------------- #
LINKS=(); while IFS= read -r l; do LINKS+=(--link "$l"); done < <("$PY" -c "
import json,sys; m=json.load(open(sys.argv[1])); [print(f'{k}={v}') for k,v in sorted(m['links'].items())]" "$RUNTREE/.exp40_runtree_manifest.json")
CHECK_JSON="${LOGBASE}_check.json"
"$PY" "$KIT" check --platform neuronic --rung "$RUNG" --mode "$MODE" --run-id "$RUN_ID" \
  --save-root "$SAVE_ROOT" --stamp "$TS" --sha "$SHA" --repo "$REPO" --tree "$RUNTREE" \
  "${LINKS[@]}" --receipts "$REPO/$EXPREL/receipts" --hf-sealed "$HF_SEALED" --out "$CHECK_JSON" --in-job \
  || die "gates refused"
LANES="$(seq -s, 0 $((NGPU - 1)))"
# code-r1 H5: the gate inspects the devices Slurm ALLOCATED (resolved by UUID under this job's
# CUDA_VISIBLE_DEVICES, local rank i -> visible device i), never physical indices 0..N-1
"$PY" "$KIT" gpus --platform neuronic --rung "$RUNG" --mode "$MODE" --lanes "$LANES" \
  --policy idle --expect-total "$NGPU" --slurm-allocated --out "${LOGBASE}_gpus.json" || die "GPU gate refused"
mapfile -t ARGV < <("$PY" "$KIT" argv --platform neuronic --rung "$RUNG" --mode "$MODE" \
  --run-id "$RUN_ID" --save-root "$SAVE_ROOT" --stamp "$TS") || die "argv"
echo "argv: train.py ${ARGV[*]}"

# --- signal-safe cleanup (code-r1 B4) ------------------------------------------------------ #
TRAIN_PID=""; WATCH_PID=""; CLEANED=0; INTERRUPTED=""
cleanup() {
  local rc=$? bad=0
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  trap - EXIT INT TERM
  if [ -n "$WATCH_PID" ] && kill -0 "$WATCH_PID" 2>/dev/null; then
    kill -TERM "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null
  fi
  if [ -n "$TRAIN_PID" ]; then          # Slurm's KillWait is short: a 20 s grace, then SIGKILL
    "$PY" "$KIT" stop-group --pgid "$TRAIN_PID" --grace 20 || { echo "CLEANUP: training group $TRAIN_PID survived"; bad=1; }
    wait "$TRAIN_PID" 2>/dev/null
  fi
  [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && rm -rf -- "$TMPDIR"
  if [ -n "$INTERRUPTED" ]; then
    echo "=== exp_40 C16 neuronic ${MODE} INTERRUPTED by SIG${INTERRUPTED} $(date '+%F %T'): FAIL ==="
    exit 130
  fi
  [ "$bad" = 0 ] || exit 12
  exit "$rc"
}
on_signal() { INTERRUPTED="$1"; exit 130; }
trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# --- train ------------------------------------------------------------------------------- #
EVDIR="$SAVE_ROOT/evidence/$NS"; mkdir -p "$EVDIR" || die "evidence dir"
EVMODE=digest; [ "$MODE" = PILOT ] && EVMODE=verbatim
TRAIN_LOG="${LOGBASE}_train.log"
PUBLISH=off; [ "$MODE" = FULL ] && PUBLISH=local-manifest
( cd "$RUNTREE" && EXP40_EVIDENCE_DIR="$EVDIR" EXP40_EVIDENCE_MODE="$EVMODE" EXP40_PUBLISH="$PUBLISH" \
    EXP40_RUN_ID="$RUN_ID" exec setsid torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "$DRIVER" "${ARGV[@]}" ) > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
sleep 2
[ "$(ps -o pgid= -p "$TRAIN_PID" | tr -d ' ')" = "$TRAIN_PID" ] || die "setsid did not give the training its own process group"
"$PY" "$KIT" watch --log "$TRAIN_LOG" --members "$NGPU" --pid "$TRAIN_PID" --pgid "$TRAIN_PID" --timeout 3600 &
WATCH_PID=$!
CKPT_DIR="$SAVE_ROOT/FLAC_exp40_C16/$NS/checkpoints"
wait "$TRAIN_PID"; TRAIN_RC=$?
wait "$WATCH_PID"; WATCH_RC=$?
WATCH_PID=""
"$PY" "$KIT" stop-group --pgid "$TRAIN_PID" --grace 20 || die "training group $TRAIN_PID survived"
TRAIN_PID=""
echo "train rc=${TRAIN_RC} watcher rc=${WATCH_RC}"

# --- verdicts ------------------------------------------------------------------------------ #
FINAL=0; RECEIPTS="$REPO/$EXPREL/receipts"
if [ "$MODE" = FULL ]; then
  EVOUT="$RECEIPTS/full_evidence_${RUN_ID}.json"
  REF="$RECEIPTS/pilot_neuronic_${RUNG}_${SHA}.json"
  AUDIT="$RECEIPTS/audit_${RUN_ID}.json"
  "$PY" "$KIT" evidence --platform neuronic --rung "$RUNG" --mode FULL --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$RECEIPTS" --train-rc "$TRAIN_RC" \
    --watcher-rc "$WATCH_RC" --reference "$REF" --out "$EVOUT" || FINAL=1
  "$PY" "$KIT" audit --ckpt-dir "$CKPT_DIR" --evidence "$EVOUT" --run-id "$RUN_ID" \
    --check "$CHECK_JSON" --reference "$REF" --out "$AUDIT" || FINAL=1
  cp "$CHECK_JSON" "$RECEIPTS/check_${RUN_ID}.json" || FINAL=1
  [ "$TRAIN_RC" = 0 ] && [ "$WATCH_RC" = 0 ] || FINAL=1
  # code-r1 H13: a local completion is NOT a FULL PASS; the durable pending receipt names
  # everything the post-pull publication, NAS audit and admission need
  VERDICT=LOCAL_COMPLETE_PENDING_PUBLICATION; [ "$FINAL" = 0 ] || VERDICT=FAIL
  "$PY" "$KIT" run-result --run-id "$RUN_ID" --sha "$SHA" --verdict "$VERDICT" \
    --check "$RECEIPTS/check_${RUN_ID}.json" --reference "$REF" --evidence "$EVOUT" --audit "$AUDIT" \
    --expected-dirpath "$CKPT_DIR" --ckpt-dir "$CKPT_DIR" --runtree "$RUNTREE" \
    --out "$RECEIPTS/run_result_${RUN_ID}.json" || { FINAL=1; VERDICT=FAIL; }
  echo "=== exp_40 C16 neuronic FULL finished $(date '+%F %T'): ${VERDICT} ==="
  [ "$VERDICT" = FAIL ] || echo "next: pull ${CKPT_DIR} + receipts to the A6000 box; publish-cadence --pulled-manifest; audit --nas-dir; admit (yaw_aug_c16_neuronic_commands.md)"
else
  "$PY" "$KIT" evidence --platform neuronic --rung "$RUNG" --mode PILOT --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$RECEIPTS" --train-rc "$TRAIN_RC" \
    --watcher-rc "$WATCH_RC" --out "${LOGBASE}_evidence.json" || FINAL=1
  [ "$(find "$SAVE_ROOT/FLAC_exp40_C16/$NS" -name '*.ckpt' 2>/dev/null | wc -l)" = 0 ] || FINAL=1
  [ "$TRAIN_RC" = 0 ] && [ "$WATCH_RC" = 0 ] || FINAL=1
  echo "=== exp_40 C16 neuronic PILOT finished $(date '+%F %T'): $([ $FINAL = 0 ] && echo PASS || echo FAIL) ==="
fi
[ "$WATCH_RC" -ge 8 ] && exit "$WATCH_RC"
exit "$FINAL"
