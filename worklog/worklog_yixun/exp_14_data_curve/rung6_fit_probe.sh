#!/bin/bash
# exp_14 ladder rung 6: fit probe at the real micro-batch (32/GPU), BOTH arms concurrently, 5 steps, no checkpoints; GPU memory sampled every 5 s.
# Hardened in round F (codex finding 4) and given real acceptance evidence in round G
# (codex full-r2 finding 3: round F accepted a process exit code and nothing else, so a run
# that crashed after one batch, loaded the wrong backbone, diverged, or wrote checkpoints
# it must not write, all passed). Acceptance, ALL of: BOTH arms exit 0 (each arm's rc in
# its OWN log, which the pre-round-F version mixed into the cyl log), each log carries
# Lightning's real max_steps=5 banner, that arm's backbone banner, the 5/1148 batch count
# the recorded run reached, and a finite train/loss -- and the two save dirs hold ZERO
# checkpoints. Only then the .done marker; otherwise .failed naming the reason, exit 1.
set -euo pipefail
WT="${WT:-/home/yixunhu/codespace/exp-14-data-curve}"
KIT="${KIT:-/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_14_data_curve_claude}"
REC="${REC:-$WT/worklog/worklog_yixun/exp_14_data_curve}"
NAS="${NAS:-/media/diskstation/yixunhu/FLAC/checkpoints/exp14_data_curve/smoke}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PKG_SRC="${PKG_SRC:-/home/yixunhu/codespace/cylindrical-dinov3-exp13pin/src}"
STEPS="${STEPS:-5}"; BATCHES="${BATCHES:-5/1148}"   # the recorded run's own evidence
TS=$(date +%Y-%m-%d_%H-%M-%S)
fail () { printf 'RUNG6_FAILED %s | %s\n' "$*" "$(date -Is)" > "$REC/rung6_fit_${TS}.failed"
  echo "rung6 FAILED: $*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$REC/ladder_checks.sh" || fail "cannot source $REC/ladder_checks.sh"
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
    --max-steps "$STEPS" --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 100000 \
    --name smoke6_dc_$ARM --experiment-name smoke6_dc_$ARM --save-dir "$NAS/smoke6_$ARM" >> "$LOG" 2>&1 &
  PIDS+=("$!"); LOGS+=("$LOG"); RUNARMS+=("$ARM")
done
bash "$REC/gpu_sampler.sh" "$REC/rung6_gpu_samples_${TS}.txt" "${PIDS[0]}" &
SAMPLER=$!
BAD=()
for i in "${!PIDS[@]}"; do
  ARM="${RUNARMS[$i]}"; LOG="${LOGS[$i]}"
  RC=0; wait "${PIDS[$i]}" || RC=$?
  echo "=== rung6 $ARM pid ${PIDS[$i]} rc=$RC $(date -Is)" >> "$LOG"
  [ "$RC" = 0 ] || { BAD+=("$ARM(rc=$RC)"); continue; }
  lc_fit_banner "$LOG" "$STEPS" || BAD+=("$ARM(no max_steps=$STEPS banner)")
  lc_backbone "$LOG" "$ARM" || BAD+=("$ARM(no backbone banner)")
  lc_batches "$LOG" "$BATCHES" || BAD+=("$ARM(never reached $BATCHES)")
  lc_finite_loss "$LOG" || BAD+=("$ARM(no finite train/loss)")
done
wait "$SAMPLER" || true                     # it exits on its own when the watched pid is gone
# A 5-step probe with --checkpoint-every 100000 must leave NOTHING behind.
lc_no_checkpoints "$NAS/smoke6_cyl" "$NAS/smoke6_van" \
  || BAD+=("checkpoint(s) written by a probe that must write none")
[ "${#BAD[@]}" -eq 0 ] || fail "acceptance not met: ${BAD[*]}"
echo "RUNG6_DONE $(date -Is)" > "$REC/rung6_fit_${TS}.done"
