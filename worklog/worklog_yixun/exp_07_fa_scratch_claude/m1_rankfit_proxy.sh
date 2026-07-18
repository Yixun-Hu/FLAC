#!/usr/bin/env bash
# ============================================================================
# m1_rankfit_proxy.sh - exp_07: SINGLE-RANK fit proxy for the M1 rung, run
# co-tenant with aug291k (Yixun policy 2026-07-18: don't wait for empty GPUs;
# use available headroom).
#
# Why a proxy: a micro-32 B-F rank is ~40-47 GB-class (M0 measured: exhausted a
# full 47.3 GiB card; even micro-8 B-F peaked 36.8 GB), so NO 2-rank DDP probe
# fits while aug291k holds ~21 GB of GPU 0 (27.8 GB free). GPU 1 is empty ->
# probe ONE rank there: per-rank activations/weights are identical to a DDP
# rank (DDP adds only ~1-1.5 GB buckets/NCCL). SyncBatchNorm needs >=2 ranks,
# so the proxy runs WITHOUT --sync-batchnorm (DISCLOSED - SyncBN adds small
# comm buffers, not activation memory). Interpretation:
#   - OOM here (48.4 GB free, flash-DiT active)  -> the DDP rung will OOM too.
#   - FIT with >=2 GB headroom                    -> the DDP rung very likely fits.
#   - FIT within <2 GB                            -> marginal; 2-rank confirm required.
# 15 opt steps, micro 32, accum 1 (matched to the rung), seed 42, env: caller's
# (expected conda `flac`). Throughput here is NOT representative (single rank,
# co-tenant CPU); the fit bit is the signal. DOES NOT LAUNCH TRAINING.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_m1_rankfit_proxy.log"
SCRATCH="${TMPDIR:-/tmp}/exp07_m1_proxy"
mkdir -p "$SCRATCH"

S1=""
cleanup() { kill "${S1:-}" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec > >(tee -a "$LOG") 2>&1
echo "=== M1 single-rank fit PROXY (micro 32, no SyncBN, GPU 1) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="

# GPU 1 must have full-rung headroom (proxy needs what a rank needs); GPU 0 untouched.
MIN_FREE_MB="${MIN_FREE_MB:-44000}"
FREE="$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
rc_q=$?
[ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi query failed (rc=${rc_q}) - abort"; exit 2; }
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU 1 free ${FREE} MiB < ${MIN_FREE_MB} MiB - abort"; exit 2; }
echo "--- co-tenancy disclosure: compute apps at proxy start ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# flash-DiT is part of the verdict's premise -> enforce the env, don't assume it
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "conda env 'flac' required (got '${CONDA_DEFAULT_ENV:-none}') - abort"; exit 2; }

HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

RUNLOG="${SCRATCH}/rankfit_32.out"
PF="${SCRATCH}/peak1.txt"; echo 0 > "$PF"
( peak=0; while :; do u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -dc '0-9')"; if [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null; then peak="$u"; echo "$peak" > "$PF"; fi; sleep 1; done ) & S1=$!

start="$(date +%s)"
timeout -k 30s 900s env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_BF.json" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 15 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42 \
  --logger none --checkpoint-every 1000000 \
  --name exp07_m1proxy_bf_32 --experiment-name exp07_m1proxy --save-dir "${SCRATCH}/run" > "$RUNLOG" 2>&1
rc=$?
end="$(date +%s)"
kill "$S1" 2>/dev/null; wait "$S1" 2>/dev/null; S1=""
cat "$RUNLOG"
peak="$(cat "$PF" 2>/dev/null || echo 0)"; wall=$((end - start))

reached="$(grep -ciE "max_steps=15.*reached|stopped:.*max_steps=15" "$RUNLOG" 2>/dev/null)"; reached="${reached:-0}"
oom="$(grep -ciE "CUDA out of memory|torch\.cuda\.OutOfMemoryError" "$RUNLOG" 2>/dev/null)"; oom="${oom:-0}"
nanloss="$(grep -ciE "loss[[:space:]]*=[[:space:]]*[+-]?(nan|inf(inity)?)" "$RUNLOG" 2>/dev/null)"; nanloss="${nanloss:-0}"
echo "=== proxy_32: exit=${rc} reached15=${reached} oom=${oom} nanloss=${nanloss} peakVRAM(GPU1)=${peak}MiB wall=${wall}s ==="

if [ "$rc" -eq 0 ] && [ "$reached" -ge 1 ] && [ "$nanloss" -eq 0 ]; then
  [ "$peak" -gt 0 ] 2>/dev/null || { echo "HARD ABORT: VRAM sampler failed (peak=${peak}) - fit observed but headroom unmeasurable; rerun."; exit 4; }
  head_mb=$((49140 - peak))
  echo "=== PROXY VERDICT: micro-32 rank FITS on an empty A6000 (peak ${peak} MiB; headroom ${head_mb} MiB vs 49,140 total). ==="
  if [ "$head_mb" -ge 2000 ]; then
    echo "CLASSIFICATION: headroom ${head_mb} MiB >= 2000 -> DDP+SyncBN rung (est. +0.35-1 GiB/rank per review) VERY LIKELY FITS."
  else
    echo "CLASSIFICATION: headroom ${head_mb} MiB < 2000 -> MARGINAL; 2-rank confirm REQUIRED before any go."
  fi
  echo "REPORT TO YIXUN AND HOLD - no training, and the 2-rank confirm still needs a both-GPU window."
  exit 0
fi
if [ "$oom" -ge 1 ]; then
  echo "=== PROXY VERDICT: micro-32 rank OOMs even on an empty card with flash-DiT -> STRONGLY RULES OUT the BN=64 DDP rung (no formal guarantee, but operationally decisive). STOP; options to Yixun (expandable_segments / revisit mandate). ==="
  exit 2
fi
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then echo "HARD ABORT: proxy timed out (rc=${rc}; 124=TERM, 137=KILL after -k), no OOM signature - investigate ${RUNLOG}."; exit 4; fi
echo "HARD ABORT: proxy failed without CUDA-OOM (rc=${rc}, reached15=${reached}, nanloss=${nanloss}) - investigate."; exit 4
