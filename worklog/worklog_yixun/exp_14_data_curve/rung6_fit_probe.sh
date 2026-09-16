#!/bin/bash
# exp_14 ladder rung 6: fit probe at the real micro-batch (32/GPU), BOTH arms concurrently, 5 steps, no checkpoints; GPU memory sampled every 5 s.
# Hardened in round F (codex finding 4). Acceptance: BOTH arms exit 0 (each arm's rc is
# recorded in its OWN log, which the previous version mixed into the cyl log). Only then the
# .done marker; otherwise .failed naming the arms that failed, and exit 1.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
NAS="${NAS:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
fail () { printf 'RUNG6_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung6_fit_${TS}.failed"
  echo "rung6 FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH" || fail "cannot source $CONDA_SH"
conda activate flac || fail "cannot activate the flac env"
cd "$WT" || fail "cannot cd to $WT"
export PYTHONPATH="$PKG_SRC" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PIDS=(); LOGS=(); RUNARMS=()
for ARM in cyl van; do
  LOG=$REC/rung6_fit_${ARM}_${TS}.log
  echo "=== rung6 $ARM start $(date -Is) FLAC=$(git rev-parse --short HEAD)" > "$LOG"
  CUDA_VISIBLE_DEVICES=0,1 python train.py --model-config "$KIT/configs/FLAC_AR_exp14_${ARM}S.json" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train_frac025.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps 5 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 100000 \
    --name smoke6_dc_$ARM --experiment-name smoke6_dc_$ARM --save-dir "$NAS/smoke6_$ARM" >> "$LOG" 2>&1 &
  PIDS+=("$!"); LOGS+=("$LOG"); RUNARMS+=("$ARM")
done
bash "$REC/gpu_sampler.sh" "$REC/rung6_gpu_samples_${TS}.txt" "${PIDS[0]}" &
SAMPLER=$!
BAD=()
for i in "${!PIDS[@]}"; do
  RC=0; wait "${PIDS[$i]}" || RC=$?
  echo "=== rung6 ${RUNARMS[$i]} pid ${PIDS[$i]} rc=$RC $(date -Is)" >> "${LOGS[$i]}"
  [ "$RC" = 0 ] || BAD+=("${RUNARMS[$i]}(rc=$RC)")
done
wait "$SAMPLER" || true                     # it exits on its own when the watched pid is gone
[ "${#BAD[@]}" -eq 0 ] || fail "arm(s) exited non-zero: ${BAD[*]}"
echo "RUNG6_DONE $(date -Is)" > "$REC/rung6_fit_${TS}.done"
