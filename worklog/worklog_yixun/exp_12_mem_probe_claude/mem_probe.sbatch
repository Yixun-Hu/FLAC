#!/bin/bash
# exp_12 mem_probe — paper-configuration VRAM probe (Yixun commission 2026-08-05).
# Rev 2: all 5 BLOCKING + 2 NIT findings of mem_probe_codex_code_probe_review.md
# applied (UUID-bound fail-closed measurement, literal-SHA config pin, commit-
# bound execution + fully explicit CLI, no wandb fallback, synchronous train
# log + completion marker, job-unique artifacts, duplicate-run guard).
#
# 1× L40, micro-batch 64, accum 1, bf16-mixed, single process (no DDP), no
# SyncBN, no ViT gradient checkpointing, canonical FLAC_AR.json verbatim.
# 5 optimizer steps; peak GPU memory via 0.5 s UUID-bound nvidia-smi poll.
# OOM policy (verbatim mandate): report the exact error and peak memory; NO
# fallback — no batch reduction, no grad-ckpt, no accumulation, no extra GPUs.
# Does not touch any other job (notably P1 job 3637217).
#
# Submit: sbatch --export=ALL,EXPECT_SHA=<full reviewed HEAD sha> mem_probe.sbatch
#SBATCH --job-name=exp12-mem-probe
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_12_mem_probe_claude/slurm_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_12_mem_probe_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

JID="${SLURM_JOB_ID:?must run under sbatch}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

echo "=== exp_12 mem_probe Rev2 - ${TS} - job ${JID} - host $(hostname) ==="

# --- commit binding (review B3): the job runs exactly the reviewed SHA, with
# --- the measurement-relevant tracked surfaces unmodified ---
: "${EXPECT_SHA:?EXPECT_SHA (full reviewed commit sha) must be exported at submission - abort}"
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || { echo "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"; exit 2; }
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
echo "commit binding OK: ${HEAD_SHA}"

# --- Slurm shape asserts (review B3) ---
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || { echo "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"; exit 2; }
[ "${SLURM_NTASKS:-1}" = "1" ] || { echo "expected 1 task, got ${SLURM_NTASKS} - abort"; exit 2; }

# --- exactly ONE visible GPU, an L40, addressed by UUID from here on (review B1) ---
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq 1 ] || { echo "expected exactly 1 visible GPU, got ${#GPU_ROWS[@]} - abort"; exit 2; }
GPU_UUID="$(echo "${GPU_ROWS[0]}" | cut -d, -f1 | tr -d ' ')"
GPU_NAME="$(echo "${GPU_ROWS[0]}" | cut -d, -f2 | sed 's/^ *//')"
TOTAL_MIB="$(echo "${GPU_ROWS[0]}" | cut -d, -f3 | tr -d ' ')"
case "$GPU_NAME" in *L40*) ;; *) echo "allocated GPU '${GPU_NAME}' is not an L40 - abort"; exit 2;; esac
case "$TOTAL_MIB" in ''|*[!0-9]*) echo "non-numeric memory.total '${TOTAL_MIB}' - abort"; exit 2;; esac
echo "GPU: ${GPU_NAME} ${GPU_UUID} total ${TOTAL_MIB} MiB"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- config identity (review B2): BOTH files must equal the reviewed literal sha ---
PIN_SHA="f3eafef4456666e4705ddaf35540f6b9f1f746189814cec000bac794ba2a7ec9"
CANON="src/configs/model_configs/FLAC/AR/FLAC_AR.json"
PROBE="$EXPDIR/FLAC_AR_exp12_memprobe.json"
for F in "$CANON" "$PROBE"; do
  GOT="$(sha256sum "$F" | awk '{print $1}')"
  [ "$GOT" = "$PIN_SHA" ] || { echo "sha256(${F}) = ${GOT} != reviewed pin ${PIN_SHA} - abort"; exit 2; }
done
echo "config pin OK: ${PIN_SHA} (canonical == probe == reviewed)"

# --- paper-config semantic gate (review B2: no bare asserts; + use_ema) ---
python3 - "$PROBE" <<'PY' || { echo "paper-config gate FAILED - abort"; exit 2; }
import json, sys
cfg = json.load(open(sys.argv[1])); t = cfg["training"]; bad = []
if "cond_method" in t or "frame_avg_angles" in t: bad.append("non-vanilla training keys present")
if t.get("use_ema") is not True: bad.append(f"use_ema={t.get('use_ema')!r} (paper: True)")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if len(vits) != 2: bad.append(f"expected 2 ViT conditioners, got {len(vits)}")
for c in vits:
    gc = c["config"].get("gradient_checkpointing", False)
    if gc is not False: bad.append(f"{c['id']}: gradient_checkpointing={gc!r} (must be absent/False)")
if bad: sys.exit("PAPER-CONFIG GATE: " + "; ".join(bad))
print("gate OK: vanilla cond; use_ema True; ViT grad-ckpt absent/False in", [c["id"] for c in vits])
PY

# --- wandb: fixed logger, fail-closed identity, isolated dir (review B4) ---
mkdir -p "$REPO/outputs_FLAC/exp12_memprobe"
export WANDB_DIR="$REPO/outputs_FLAC/exp12_memprobe"
eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
python3 - <<'PY' || { echo "wandb identity gate FAILED (no logger fallback per review B4) - abort"; exit 2; }
import sys
try:
    import wandb; email = wandb.Api().viewer.email
except Exception as e:
    sys.exit(f"wandb identity check FAILED: {e}")
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY

# --- duplicate-run guard (review B6) ---
RUNDIR="$REPO/outputs_FLAC/exp12_memprobe/FLAC_exp12_memprobe"
[ ! -e "$RUNDIR" ] || { echo "run dir ${RUNDIR} already exists (duplicate submission?) - abort"; exit 2; }

# --- UUID-bound peak-VRAM poller, job-unique truncated CSV (reviews B1, B6) ---
PEAKCSV="$EXPDIR/mem_probe_${TS}_jid${JID}_vram.csv"
: > "$PEAKCSV"
( while true; do
    nvidia-smi --id="$GPU_UUID" --query-gpu=uuid,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "POLLFAIL"
    sleep 0.5
  done >> "$PEAKCSV" ) &
POLLER=$!

# --- training: synchronous tee, PIPESTATUS-preserved rc (review B5);
# --- every defining setting explicit on the CLI (review B3) ---
TRAINLOG="$EXPDIR/mem_probe_${TS}_jid${JID}_S5_train.log"
python train.py \
  --model-config "$PROBE" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 5 --batch-size 64 --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus 1 --num-nodes 1 --strategy auto --precision bf16-mixed \
  --sync-batchnorm false --val-every -1 --val-dataset-config '' \
  --ckpt-path '' --pretrained-ckpt-path '' --gradient-clip-val 0.0 \
  --logger wandb --checkpoint-every 10000 \
  --name FLAC_exp12_memprobe --experiment-name exp12_memprobe \
  --save-dir outputs_FLAC/exp12_memprobe 2>&1 | tee "$TRAINLOG"
rc="${PIPESTATUS[0]}"
kill "$POLLER" 2>/dev/null
wait "$POLLER" 2>/dev/null

# --- measurement validation: fail-closed even when rc==0 (review B1) ---
if grep -q "POLLFAIL" "$PEAKCSV"; then echo "VRAM poller reported query failure - measurement INVALID"; MEAS_OK=0
else
  NROWS="$(wc -l < "$PEAKCSV")"
  BADROWS="$(awk -F', *' -v u="$GPU_UUID" '$1 != u || $2 !~ /^[0-9]+$/ {c++} END {print c+0}' "$PEAKCSV")"
  if [ "$NROWS" -ge 1 ] && [ "$BADROWS" -eq 0 ]; then MEAS_OK=1; else echo "VRAM CSV invalid (rows=${NROWS}, bad=${BADROWS})"; MEAS_OK=0; fi
fi
PEAK="$(awk -F', *' 'BEGIN{m=-1} $2 ~ /^[0-9]+$/ {if ($2+0>m) m=$2+0} END{print m}' "$PEAKCSV")"

# --- outcome classification (review B5) ---
OOM_RE="CUDA out of memory|OutOfMemoryError"
DONE_RE="max_steps=5.*reached"
echo "=== RESULT: train.py rc=${rc} | peak ${PEAK} MiB / ${TOTAL_MIB} MiB on ${GPU_UUID} (0.5s poll, ${GPU_NAME}) | measurement_valid=${MEAS_OK} ==="
if [ "$rc" -eq 0 ]; then
  grep -Eq "$DONE_RE" "$TRAINLOG" || { echo "FAIL: rc=0 but no 'max_steps=5 reached' completion marker in train log"; exit 4; }
  [ "$MEAS_OK" -eq 1 ] || { echo "FAIL: training completed but peak-VRAM measurement invalid"; exit 5; }
  echo "SUCCESS: 5 optimizer steps completed; peak ${PEAK} MiB of ${TOTAL_MIB} MiB"
  exit 0
fi
if grep -Eq "$OOM_RE" "$TRAINLOG"; then
  echo "--- OOM (exact error excerpt; fail-honest per mandate, no fallback) ---"
  grep -m1 -B2 -A10 -E "$OOM_RE" "$TRAINLOG"
  echo "OOM: peak before failure ${PEAK} MiB / ${TOTAL_MIB} MiB (measurement_valid=${MEAS_OK})"
  exit 3
fi
echo "NON-OOM FAILURE: rc=${rc} (see ${TRAINLOG}); peak ${PEAK} MiB (measurement_valid=${MEAS_OK})"
exit "$rc"
