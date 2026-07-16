#!/usr/bin/env bash
# ============================================================================
# m1_ddp_fit_probe.sh - exp_07: M1 fit/throughput probe for B-F, 2-GPU DDP +
# SyncBatchNorm (Yixun mandate: BN eff batch 64 = paper).
#
# SINGLE compliant rung (review finding: gradient ACCUMULATION never contributes
# to BatchNorm statistics, so 16x2x2 / 8x2x4 would silently violate the BN=64
# mandate - SyncBN batch is micro/GPU x world_size only):
#   MB 32 x 2 GPUs x accum 1  ->  SyncBN batch 64 = paper. The ONLY compliant pair.
# Single-GPU B-F OOM'd at micro 32 with ~8 GiB fragmentation waste, so DDP fit
# is genuinely open. Plain allocator only - if it does NOT fit, STOP and report
# to Yixun (options then: expandable_segments escalation / revisit the mandate).
#
# Per rung: 15 opt steps, seed 42, logger none, scratch save-dir. FIT requires
# rc=0 AND "max_steps=15 reached" AND no nan/inf loss. A nonzero exit descends
# ONLY on a CUDA-OOM signature in the rung log (a dying rank's OOM lands there
# even when the peer rank exits with an NCCL error); anything else HARD-ABORTS.
# `timeout 900` caps each rung so an NCCL hang (default timeout 30 min) cannot
# stall the ladder. Per-GPU 1 s VRAM samplers on BOTH cards (trap-cleaned).
#
# VERDICT: winning rung + per-GPU peak VRAM + 15-step wall + crude ETA.
# Does NOT launch training - Yixun's explicit go required on the reported rung.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/fa_scratch_${TS}_m1_ddp_fit_probe.log"
SCRATCH="${TMPDIR:-/tmp}/exp07_m1_probe"
mkdir -p "$SCRATCH"

S0=""; S1=""
cleanup() { kill "${S0:-}" "${S1:-}" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT     # convert to exit so the EXIT cleanup trap actually runs
trap 'exit 143' TERM

exec > >(tee -a "$LOG") 2>&1
echo "=== M1 DDP+SyncBN fit probe (B-F) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="

# --- BOTH GPUs must be free (fail-CLOSED) ---
for G in 0 1; do
  BUSY="$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] || { echo "nvidia-smi query failed on GPU ${G} (rc=${rc_q}) - abort"; exit 2; }
  [ -z "$BUSY" ] || { echo "GPU ${G} busy (pids: ${BUSY}) - abort (aug291k still running?)"; exit 2; }
done

# --- pre-launch gate: DINOv3 pin + arm init-identity (offline, fail-closed) ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

WIN_MB=""; WIN_ACC=""; WIN_P0=""; WIN_P1=""; WIN_WALL=""
for pair in "32 1"; do   # single BN-compliant rung (see header)
  MB="${pair% *}"; ACC="${pair#* }"
  RUNLOG="${SCRATCH}/rung_${MB}x2x${ACC}.out"
  echo "=== [$(date '+%H:%M:%S')] attempt bf_ddp_${MB}x2x${ACC} (micro ${MB}/GPU x 2 x accum ${ACC}, eff 64, SyncBN) ==="

  P0F="${SCRATCH}/peak0_${MB}.txt"; echo 0 > "$P0F"
  P1F="${SCRATCH}/peak1_${MB}.txt"; echo 0 > "$P1F"
  ( peak=0; while :; do u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -dc '0-9')"; if [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null; then peak="$u"; echo "$peak" > "$P0F"; fi; sleep 1; done ) & S0=$!
  ( peak=0; while :; do u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -dc '0-9')"; if [ -n "$u" ] && [ "$u" -gt "$peak" ] 2>/dev/null; then peak="$u"; echo "$peak" > "$P1F"; fi; sleep 1; done ) & S1=$!

  start="$(date +%s)"
  # -k 30s: KILL 30 s after TERM in case NCCL ignores the TERM (review finding)
  timeout -k 30s 900s env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model-config "${EXPDIR}/FLAC_AR_BF.json" \
    --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
    --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
    --max-steps 15 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
    --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
    --logger none --checkpoint-every 1000000 \
    --name exp07_m1_bf_${MB}x2x${ACC} --experiment-name exp07_m1 --save-dir "${SCRATCH}/run" > "$RUNLOG" 2>&1
  rc=$?
  end="$(date +%s)"
  kill "$S0" "$S1" 2>/dev/null; wait "$S0" "$S1" 2>/dev/null; S0=""; S1=""
  cat "$RUNLOG"
  p0="$(cat "$P0F" 2>/dev/null || echo 0)"; p1="$(cat "$P1F" 2>/dev/null || echo 0)"; wall=$((end - start))

  reached="$(grep -ciE "max_steps=15.*reached|stopped:.*max_steps=15" "$RUNLOG" 2>/dev/null)"; reached="${reached:-0}"
  oom="$(grep -ciE "CUDA out of memory|torch\.cuda\.OutOfMemoryError" "$RUNLOG" 2>/dev/null)"; oom="${oom:-0}"
  nanloss="$(grep -ciE "loss[[:space:]]*=[[:space:]]*[+-]?(nan|inf(inity)?)" "$RUNLOG" 2>/dev/null)"; nanloss="${nanloss:-0}"
  echo "=== bf_ddp_${MB}x2x${ACC}: exit=${rc} reached15=${reached} oom=${oom} nanloss=${nanloss} peakVRAM GPU0=${p0}MiB GPU1=${p1}MiB wall=${wall}s ==="

  if [ "$rc" -eq 0 ] && [ "$reached" -ge 1 ] && [ "$nanloss" -eq 0 ]; then
    { [ "$p0" -le 0 ] || [ "$p1" -le 0 ]; } 2>/dev/null && echo "WARN: a VRAM sampler saw no positive sample (p0=${p0}, p1=${p1})."
    WIN_MB="$MB"; WIN_ACC="$ACC"; WIN_P0="$p0"; WIN_P1="$p1"; WIN_WALL="$wall"
    echo ">>> FIT: bf_ddp_${MB}x2x${ACC} reached 15 steps, finite loss - largest fitting rung, stop ladder."
    break
  fi
  if [ "$oom" -ge 1 ]; then echo "--- bf_ddp_${MB}x2x${ACC}: CUDA OOM on the only BN-compliant rung ---"; continue; fi
  if [ "$rc" -eq 124 ]; then echo "HARD ABORT: rung timed out (900 s) with NO CUDA-OOM signature - NCCL/dataloader hang? Investigate ${RUNLOG}."; exit 4; fi
  echo "HARD ABORT: bf_ddp_${MB}x2x${ACC} failed without a CUDA-OOM signature (rc=${rc}, reached15=${reached}, nanloss=${nanloss}). Investigate; not descending."; exit 4
done

if [ -z "$WIN_MB" ]; then echo "M1 VERDICT: the only BN-compliant rung (32x2x1) does NOT fit under the plain allocator - STOP; report to Yixun (options: expandable_segments escalation / revisit the BN mandate)."; exit 2; fi

rate="$(python -c "w=max($WIN_WALL,1); print(round(960.0/w,2))" 2>/dev/null || echo '?')"
eta_d="$(python -c "w=max($WIN_WALL,1); print(round(4500.0*w/86400.0,2))" 2>/dev/null || echo '?')"
echo "=== M1 VERDICT (REPORT TO YIXUN - DO NOT LAUNCH WITHOUT HIS GO) ==="
echo "winner: bf_ddp_${WIN_MB}x2x${WIN_ACC} (eff 64, SyncBN batch $((WIN_MB*2)) per update) | peak GPU0=${WIN_P0}MiB GPU1=${WIN_P1}MiB | 15-step wall=${WIN_WALL}s"
echo "rough samples/s (warmup-inflated LOWER bound): ${rate}  => crude 67.5k ETA ~${eta_d} d (over-estimate; re-anchor at launch)"
