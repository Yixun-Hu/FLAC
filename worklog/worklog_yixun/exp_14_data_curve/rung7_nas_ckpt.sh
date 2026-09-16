#!/bin/bash
# exp_14 ladder rung 7: NAS checkpoint write/reload probe with the run contract (cyl arm): 5 steps, checkpoint at step 5, then validate.
# Hardened in round F (codex finding 4). Acceptance: make-contract 0, train.py 0, a step-5
# checkpoint exists whose size is non-zero and unchanged across STABLE_WAIT, and `validate
# --for-resume` exits 0. Only then the .done marker; otherwise .failed + exit 1.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
RUN="${RUN:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke/smoke7_cyl}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
PKG_DIR="${PKG_DIR:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin}"
STABLE_WAIT="${STABLE_WAIT:-60}"
TS=$(date +%Y-%m-%d_%H-%M-%S); LOG=$REC/rung7_nas_cyl_${TS}.log
fail () { printf 'RUNG7_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung7_nas_${TS}.failed"
  echo "rung7 FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN"
CFG=$KIT/configs/FLAC_AR_exp14_cylS.json; DS=src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json
echo "=== rung7 start $(date -Is) FLAC=$(git rev-parse --short HEAD)" | tee "$LOG"
CONTRACT=$(python -m src.training.run_contract make-contract --run-dir "$RUN" --run-id smoke7_cyl --fraction 0.25 --dataset-config "$DS" --split-json data/AR/train_frac025_s2026.json --model-config "$CFG" --seed 42 --micro-batch 32 --num-gpus 2 --accum-batches 1 --sync-batchnorm true --flac-sha "$(git rev-parse HEAD)" --package-sha "$(git -C "$PKG_DIR" rev-parse HEAD)" 2>>"$LOG") \
  || fail "make-contract exited $? (see $LOG)"
echo "contract: $CONTRACT" | tee -a "$LOG"
RC=0
CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" --dataset-config "$DS" \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 5 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger none --checkpoint-every 5 \
  --name smoke7_cyl --experiment-name smoke7_cyl --save-dir "$RUN" --run-contract-json "$CONTRACT" >> "$LOG" 2>&1 || RC=$?
echo "=== rung7 train rc=$RC $(date -Is)" | tee -a "$LOG"
[ "$RC" = 0 ] || fail "train.py exited $RC (see $LOG)"
CK=$(find "$RUN" -name '*step=5.ckpt' -print -quit)
[ -n "$CK" ] || fail "no step-5 checkpoint under $RUN"
S1=$(stat -c%s "$CK"); sleep "$STABLE_WAIT"; S2=$(stat -c%s "$CK")
echo "ckpt: $CK size=$S1 size after ${STABLE_WAIT}s: $S2" | tee -a "$LOG"
{ [ "$S1" = "$S2" ] && [ "$S1" -gt 0 ]; } || fail "checkpoint size is not stable ($S1 -> $S2)"
VRC=0
python -m src.training.run_contract validate --ckpt "$CK" --contract "$CONTRACT" --model-config "$CFG" --expect-step 5 --for-resume >> "$LOG" 2>&1 || VRC=$?
echo "=== validate rc=$VRC" | tee -a "$LOG"
[ "$VRC" = 0 ] || fail "validate exited $VRC (see $LOG)"
echo "RUNG7_DONE $(date -Is)" > "$REC/rung7_nas_${TS}.done"
