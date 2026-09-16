#!/bin/bash
# exp_14 ladder rung 7: NAS checkpoint write/reload probe with the run contract (cyl arm): 5 steps, checkpoint at step 5, then validate.
# Hardened in round F (codex finding 4) and made invocation-unique in round G (codex
# full-r2 finding 3: the run dir was FIXED, so a step-5 checkpoint an earlier invocation
# had left there satisfied `find … step=5` for a run that wrote none). The run dir is now
# <RUN_BASE>/smoke7_cyl_<ts>-<pid>, it must be empty of checkpoints before training, and
# the checkpoint that is validated must have been written after this run started -- its
# inode and sha256 are recorded. Acceptance: make-contract 0, train.py 0, Lightning's
# max_steps banner + the cyl backbone banner + the 5/1148 batch count + a finite loss, a
# NEW step-5 checkpoint whose size is non-zero and unchanged across STABLE_WAIT, and
# `validate --for-resume` 0 (round H, codex full-r3 finding 2: the batch check existed,
# but only the standalone replay test ever called it). Only
# then the .done marker; otherwise .failed + exit 1.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
RUN_BASE="${RUN_BASE:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
PKG_DIR="${PKG_DIR:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin}"
STABLE_WAIT="${STABLE_WAIT:-60}"
STEPS="${STEPS:-5}"; BATCHES="${BATCHES:-5/1148}"   # the recorded run's own evidence
RUN_GIVEN="${RUN:-}"           # snapshot before this script makes its own (see the guard)
TS=$(date +%Y-%m-%d_%H-%M-%S); LOG=$REC/rung7_nas_cyl_${TS}.log
# Unique per INVOCATION, not per name: nothing an earlier run left behind can be in here.
RUN="$RUN_BASE/smoke7_cyl_${TS}-$$"
fail () { printf 'RUNG7_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung7_nas_${TS}.failed"
  echo "rung7 FAILED: $*" >&2; exit 1; }
[ -z "$RUN_GIVEN" ] || fail "RUN='$RUN_GIVEN' is no longer honoured: rung 7 always makes \
its own invocation-unique run dir under RUN_BASE, because a fixed one can hand the \
validator a step-$STEPS checkpoint this invocation never wrote. Set RUN_BASE instead."
# shellcheck source=/dev/null
source "$REC/ladder_checks.sh" || fail "cannot source $REC/ladder_checks.sh"
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN"
lc_no_checkpoints "$RUN" || fail "$RUN is not a fresh run dir"
STARTED_AT=$(date +%s)
CFG=$KIT/configs/FLAC_AR_exp14_cylS.json; DS=src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json
echo "=== rung7 start $(date -Is) FLAC=$(git rev-parse --short HEAD)" | tee "$LOG"
CONTRACT=$(python -m src.training.run_contract make-contract --run-dir "$RUN" --run-id smoke7_cyl --fraction 0.25 --dataset-config "$DS" --split-json data/AR/train_frac025_s2026.json --model-config "$CFG" --seed 42 --micro-batch 32 --num-gpus 2 --accum-batches 1 --sync-batchnorm true --flac-sha "$(git rev-parse HEAD)" --package-sha "$(git -C "$PKG_DIR" rev-parse HEAD)" 2>>"$LOG") \
  || fail "make-contract exited $? (see $LOG)"
echo "contract: $CONTRACT" | tee -a "$LOG"
RC=0
CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" --dataset-config "$DS" \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps "$STEPS" --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every "$STEPS" \
  --name smoke7_cyl --experiment-name smoke7_cyl --save-dir "$RUN" --run-contract-json "$CONTRACT" >> "$LOG" 2>&1 || RC=$?
echo "=== rung7 train rc=$RC $(date -Is)" | tee -a "$LOG"
[ "$RC" = 0 ] || fail "train.py exited $RC (see $LOG)"
lc_fit_banner "$LOG" "$STEPS" || fail "rc 0 but no Lightning max_steps=$STEPS banner in $LOG"
lc_backbone "$LOG" cyl || fail "the cyl backbone banner is missing from $LOG"
lc_batches "$LOG" "$BATCHES" || fail "the fit never reached $BATCHES in $LOG"
lc_finite_loss "$LOG" || fail "no finite train/loss in $LOG"
CK=$(find "$RUN" -name "*step=$STEPS.ckpt" -print -quit)
[ -n "$CK" ] || fail "no step-$STEPS checkpoint under $RUN"
# It has to be THIS invocation's file: a unique run dir plus a birth no earlier than the
# moment training began. The inode and digest go into the log as the identity of what was
# validated -- the same binding eval_FLAC now stamps into its own artifacts.
INODE=$(stat -c%i "$CK"); MTIME=$(stat -c%Y "$CK")
[ "$MTIME" -ge "$STARTED_AT" ] \
  || fail "$CK (mtime $MTIME) predates this run (started $STARTED_AT): it is not ours"
S1=$(stat -c%s "$CK"); sleep "$STABLE_WAIT"; S2=$(stat -c%s "$CK")
echo "ckpt: $CK inode=$INODE sha256=$(sha256sum "$CK" | cut -d' ' -f1)" | tee -a "$LOG"
echo "ckpt: size=$S1 size after ${STABLE_WAIT}s: $S2" | tee -a "$LOG"
{ [ "$S1" = "$S2" ] && [ "$S1" -gt 0 ]; } || fail "checkpoint size is not stable ($S1 -> $S2)"
VRC=0
python -m src.training.run_contract validate --ckpt "$CK" --contract "$CONTRACT" --model-config "$CFG" --expect-step "$STEPS" --for-resume >> "$LOG" 2>&1 || VRC=$?
echo "=== validate rc=$VRC" | tee -a "$LOG"
[ "$VRC" = 0 ] || fail "validate exited $VRC (see $LOG)"
echo "RUNG7_DONE $(date -Is)" > "$REC/rung7_nas_${TS}.done"
