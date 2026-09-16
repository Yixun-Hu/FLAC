#!/bin/bash
# exp_14 ladder rung 5: smallest-batch, NO-checkpoint smoke per arm, 2-GPU DDP+SyncBN (co-tenant with exp13_vanB by Yixun's decision).
# Planner one-off; hardened in round F (codex finding 4: these scripts used to write their
# .done marker whatever happened). Acceptance, per arm: train.py exits 0 AND its log carries
# the `max_steps=<STEPS> reached` banner. Only then is the .done marker written; otherwise a
# .failed marker names the reason and the script exits non-zero. Never touches other processes.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
NAS="${NAS:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
STEPS="${STEPS:-3}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
fail () { printf 'RUNG5_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung5_smoke_${TS}.failed"
  echo "rung5 FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
for ARM in cyl van; do
  CFG=$KIT/configs/FLAC_AR_exp14_${ARM}S.json; LOG=$REC/rung5_smoke_${ARM}_${TS}.log
  [ -f "$CFG" ] || fail "$ARM: no arm config at $CFG"
  echo "=== rung5 $ARM start $(date -Is) FLAC=$(git rev-parse --short HEAD) cfg=$(sha256sum "$CFG" | cut -c1-16)" | tee "$LOG"
  RC=0
  CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$CFG" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps "$STEPS" --batch-size 2 --accum-batches 1 --num-workers 2 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 100000 \
    --name smoke5_dc_$ARM --experiment-name smoke5_dc_$ARM --save-dir "$NAS/smoke5_$ARM" >> "$LOG" 2>&1 || RC=$?
  echo "=== rung5 $ARM rc=$RC end $(date -Is)" | tee -a "$LOG"
  [ "$RC" = 0 ] || fail "$ARM: train.py exited $RC (see $LOG)"
  grep -qF "max_steps=$STEPS reached" "$LOG" \
    || fail "$ARM: rc 0 but no 'max_steps=$STEPS reached' banner in $LOG"
done
echo "RUNG5_DONE $(date -Is)" > "$REC/rung5_smoke_${TS}.done"
