#!/usr/bin/env bash
# ============================================================================
# p1a_fit_probe.sh - exp_07 phase 2 (B-V parity), P1a fit/throughput probe.
#
# Finds the LARGEST micro-batch at which the VANILLA arm (B-V) fits on GPU 1
# (48 GiB), over the eff-batch-64 ladder 64x1 -> 32x2 -> 16x4, first-fit-wins.
# (M0 in phase 1 only proved the HEAVIER B-F arm OOMs at all three rungs; B-V
# -- no C4 frame-averaging -- was never measured.) The largest fitting micro
# gives BN statistics closest to the released micro-64; that rung feeds P1b.
#
# Per rung: 15 optimizer steps, EMA on (config), HF_HUB_OFFLINE=1, seed 42.
#   FIT is declared ONLY when rc=0 AND Lightning reports "max_steps=15 reached"
#   AND loss is finite (rc=0 alone is insufficient). A nonzero exit descends the
#   ladder ONLY on a confirmed CUDA OOM; any other failure (missing data/VAE,
#   driver fault) HARD-ABORTS -- it must not be mistaken for "does not fit".
#   Peak VRAM via a 1 s nvidia-smi sampler on physical GPU 1 (trap-cleaned).
#   15-step wall -> a warmup-inflated LOWER-BOUND rate; the precise steady-state
#   rate + 67.5k ETA are re-anchored from P1b's first ~200 steps at launch.
#
# Does NOT run P1b; prints a VERDICT (winning rung, peak VRAM, crude ETA).
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_p1a_fit_probe.log"
SCRATCH="${TMPDIR:-/tmp}/exp07_p1a_probe"
mkdir -p "$SCRATCH"

SAMPLER=""
trap 'kill "${SAMPLER:-}" 2>/dev/null' EXIT INT TERM

exec > >(tee -a "$LOG") 2>&1
echo "=== P1a fit/throughput probe (B-V) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="

python "${EXPDIR}/assert_arm_configs.py"
gate=$?
if [ "$gate" -ne 0 ]; then echo "GATE FAILED (exit ${gate}) - abort probe"; exit 1; fi

WIN_MB=""; WIN_ACC=""; WIN_PEAK=""; WIN_WALL=""
for pair in "64 1" "32 2" "16 4"; do
  MB="${pair% *}"; ACC="${pair#* }"
  RUNLOG="${SCRATCH}/rung_${MB}x${ACC}.out"
  echo "=== [$(date '+%H:%M:%S')] attempt bv_${MB}x${ACC} (micro ${MB} x accum ${ACC}, eff $((MB*ACC))) ==="

  PEAKF="${SCRATCH}/peak_${MB}x${ACC}.txt"; echo 0 > "$PEAKF"
  ( peak=0
    while :; do
      u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -dc '0-9')"
      if [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null; then peak="$u"; echo "$peak" > "$PEAKF"; fi
      sleep 1
    done ) &
  SAMPLER=$!

  start="$(date +%s)"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py \
    --model-config "${EXPDIR}/FLAC_AR_BV.json" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps 15 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
    --precision bf16-mixed --logger none --checkpoint-every 1000000 \
    --name exp07_p1a_bv_${MB}x${ACC} --experiment-name exp07_p1a --save-dir "${SCRATCH}/run" > "$RUNLOG" 2>&1
  rc=$?
  end="$(date +%s)"
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=""
  cat "$RUNLOG"   # surface the rung output into the tee'd probe log
  peak="$(cat "$PEAKF" 2>/dev/null || echo 0)"; wall=$((end - start))

  reached="$(grep -ciE "max_steps=15.*reached|stopped:.*max_steps=15" "$RUNLOG" 2>/dev/null)"; reached="${reached:-0}"
  oom="$(grep -ciE "CUDA out of memory|OutOfMemoryError" "$RUNLOG" 2>/dev/null)"; oom="${oom:-0}"
  nanloss="$(grep -ciE "loss=nan|loss=inf" "$RUNLOG" 2>/dev/null)"; nanloss="${nanloss:-0}"
  echo "=== bv_${MB}x${ACC}: exit=${rc} reached15=${reached} oom=${oom} nanloss=${nanloss} peakVRAM=${peak}MiB wall=${wall}s ==="

  if [ "$rc" -eq 0 ] && [ "$reached" -ge 1 ] && [ "$nanloss" -eq 0 ]; then
    [ "$peak" -le 0 ] 2>/dev/null && echo "WARN: no positive VRAM sample (peak=${peak}); accepting fit, VRAM unknown."
    WIN_MB="$MB"; WIN_ACC="$ACC"; WIN_PEAK="$peak"; WIN_WALL="$wall"
    echo ">>> FIT: bv_${MB}x${ACC} reached 15 steps with finite loss - largest fitting rung, stop ladder."
    break
  fi
  if [ "$oom" -ge 1 ]; then echo "--- bv_${MB}x${ACC}: CUDA OOM -> descend ladder ---"; continue; fi
  echo "HARD ABORT: bv_${MB}x${ACC} did not fit-confirm and was NOT a CUDA OOM (rc=${rc}, reached15=${reached}, nanloss=${nanloss}). Investigate data/VAE/driver; not descending."; exit 4
done

if [ -z "$WIN_MB" ]; then echo "P1a VERDICT: NO RUNG FITS B-V (all rungs OOM) - STOP, ask Yixun."; exit 2; fi

rate="$(python -c "w=max($WIN_WALL,1); print(round(960.0/w,2))" 2>/dev/null || echo '?')"
eta_h="$(python -c "w=max($WIN_WALL,1); print(round(4500.0*w/3600.0,1))" 2>/dev/null || echo '?')"
echo "=== P1a VERDICT ==="
echo "winner: bv_${WIN_MB}x${WIN_ACC} (eff 64) | peakVRAM=${WIN_PEAK}MiB | 15-step wall=${WIN_WALL}s"
echo "rough samples/s (warmup-inflated LOWER bound): ${rate}  => crude 67.5k ETA ~${eta_h} h (over-estimate)"
echo "NOTE: precise steady-state rate + ETA re-anchored from P1b's first ~200 steps at launch."
