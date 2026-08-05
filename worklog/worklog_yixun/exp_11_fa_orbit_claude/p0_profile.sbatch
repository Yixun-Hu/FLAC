#!/bin/bash
# ============================================================================
# exp_11 P0 profiling cell — plan Rev 3 §10 (throughput/fit matrix + component
# attribution). ONE cell, ONE step budget; the caller submits the SAME cell at
# MAXSTEPS=10 and MAXSTEPS=30 so the collector can cancel startup cost:
#     steps/s = 20 / (wall_fit_30 - wall_fit_10)
# No checkpoints (checkpoint-every 100000 > max steps), LOGGER none, isolated
# save-dir outputs_FLAC/exp11_p0. Measurement machinery (commit binding, drift
# gate, UUID-bound fail-closed VRAM poll, synchronous tee + completion marker,
# job-unique artifacts) is cloned from the reviewed exp_12 mem_probe.sbatch Rev 2.
#
# Resource flags CANNOT be read from the environment in #SBATCH lines, so the
# SUBMITTER (p0_submit_matrix.sh) must pass them on the command line:
#
#   sbatch --job-name=p0-<CELL>-s<MAXSTEPS> \
#          --gres=gpu:l40:$NGPU --cpus-per-task=$((8+7*NGPU)) \
#          --mem=$((12*NGPU+12))G --time=00:40:00 \
#          --export=ALL,EXPECT_SHA=<full sha>,CELL=<tag>,MODEL_CONFIG=<path>,\
# NGPU=<1|2|4|8>,MB=<micro>,MAXSTEPS=<10|30> p0_profile.sbatch
#
# Emits exactly one machine-parseable final line:
#   P0RESULT cell=.. maxsteps=.. ngpu=.. mb=.. rc=.. wall_fit=.. \
#            peak_overall_mib=.. peak_per_uuid=uuid:mib,.. valid=<0|1>
# OOM cells still emit it (rc!=0) and exit 3, like mem_probe.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_p0_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

JID="${SLURM_JOB_ID:?must run under sbatch}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- required cell parameters (no defaults: an unset knob must abort) ---
: "${CELL:?CELL (cell tag, e.g. C4L_32x2) must be exported - abort}"
: "${MODEL_CONFIG:?MODEL_CONFIG (model config path) must be exported - abort}"
: "${NGPU:?NGPU must be exported - abort}"
: "${MB:?MB (micro-batch per GPU) must be exported - abort}"
: "${MAXSTEPS:?MAXSTEPS (10 or 30) must be exported - abort}"

echo "=== exp_11 P0 cell ${CELL} s${MAXSTEPS} - ${TS} - job ${JID} - host $(hostname) ==="

case "$CELL" in *[!A-Za-z0-9_x]*) echo "CELL '${CELL}' has unexpected characters - abort"; exit 2;; esac
case "$NGPU" in 1|2|4|8) ;; *) echo "NGPU '${NGPU}' not in {1,2,4,8} - abort"; exit 2;; esac
case "$MAXSTEPS" in 10|30) ;; *) echo "MAXSTEPS '${MAXSTEPS}' not in {10,30} - abort"; exit 2;; esac
case "$MB" in ''|*[!0-9]*) echo "MB '${MB}' not numeric - abort"; exit 2;; esac
[ "$((MB * NGPU))" -eq 64 ] || { echo "MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"; exit 2; }

# --- commit binding + drift gate (mem_probe review B3) ---
: "${EXPECT_SHA:?EXPECT_SHA (full reviewed commit sha) must be exported at submission - abort}"
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || { echo "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"; exit 2; }
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
echo "commit binding OK: ${HEAD_SHA}"

# --- Slurm shape asserts ---
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || { echo "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"; exit 2; }
[ "${SLURM_NTASKS:-1}" = "1" ] || { echo "expected 1 task, got ${SLURM_NTASKS} - abort"; exit 2; }

# --- exactly NGPU visible GPUs, all L40, addressed by UUID from here on (B1) ---
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || { echo "expected exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"; exit 2; }
UUIDS=()
for ROW in "${GPU_ROWS[@]}"; do
  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"
  N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
  T="$(echo "$ROW" | cut -d, -f3 | tr -d ' ')"
  case "$N" in *L40*) ;; *) echo "allocated GPU '${N}' is not an L40 - abort"; exit 2;; esac
  case "$T" in ''|*[!0-9]*) echo "non-numeric memory.total '${T}' - abort"; exit 2;; esac
  UUIDS+=("$U")
  echo "GPU: ${N} ${U} total ${T} MiB"
  TOTAL_MIB="$T"
done
UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- config gate: resolved path must be one of the five allowed manifests ---
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || { echo "MODEL_CONFIG '${MODEL_CONFIG}' does not resolve to a file - abort"; exit 2; }
ALLOWED=(
  "$EXPDIR/FLAC_AR_BF_C4L.json"
  "$EXPDIR/FLAC_AR_BF_C8.json"
  "$EXPDIR/FLAC_AR_BF_C16.json"
  "$EXPDIR/FLAC_AR_BF_C32.json"
  "$REPO/src/configs/model_configs/FLAC/AR/FLAC_AR.json"
  "$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
)
OK=0
for A in "${ALLOWED[@]}"; do [ "$MODEL_CONFIG_ABS" = "$(readlink -f "$A")" ] && OK=1; done
[ "$OK" -eq 1 ] || { echo "MODEL_CONFIG '${MODEL_CONFIG_ABS}' is not in the P0 allow-list - abort"; exit 2; }
echo "config gate OK: ${MODEL_CONFIG_ABS}"

# --- strategy: mirror exp_07 conventions (explicit, never the implicit default) ---
if [ "$NGPU" -eq 1 ]; then
  STRAT_ARGS=(--strategy auto --sync-batchnorm false)
else
  STRAT_ARGS=(--strategy ddp_find_unused_parameters_true --sync-batchnorm true)
fi

# --- UUID-bound peak-VRAM poller, job-unique truncated CSV (B1, B6) ---
PEAKCSV="$EXPDIR/p0_${CELL}_s${MAXSTEPS}_${TS}_jid${JID}_vram.csv"
: > "$PEAKCSV"
( while true; do
    nvidia-smi --id="$UUID_CSV" --query-gpu=uuid,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "POLLFAIL"
    sleep 0.5
  done >> "$PEAKCSV" ) &
POLLER=$!

# --- training: ONLY this invocation is wall-timed; synchronous tee, PIPESTATUS rc ---
TRAINLOG="$EXPDIR/p0_${CELL}_s${MAXSTEPS}_${TS}_jid${JID}_train.log"
T0="$(date +%s.%N)"
python train.py \
  --model-config "$MODEL_CONFIG_ABS" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42 \
  --num-gpus "$NGPU" --num-nodes 1 "${STRAT_ARGS[@]}" --precision bf16-mixed \
  --val-every -1 --val-dataset-config '' \
  --ckpt-path '' --pretrained-ckpt-path '' --gradient-clip-val 0.0 \
  --logger none --checkpoint-every 100000 \
  --name "FLAC_exp11_p0_${CELL}" --experiment-name "exp11_p0_${CELL}_s${MAXSTEPS}" \
  --save-dir outputs_FLAC/exp11_p0 2>&1 | tee "$TRAINLOG"
rc="${PIPESTATUS[0]}"
T1="$(date +%s.%N)"
WALL_FIT="$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')"
kill "$POLLER" 2>/dev/null
wait "$POLLER" 2>/dev/null

# --- measurement validation: fail-closed even when rc==0 (B1) ---
if grep -q "POLLFAIL" "$PEAKCSV"; then
  echo "VRAM poller reported query failure - measurement INVALID"; MEAS_OK=0
else
  NROWS="$(wc -l < "$PEAKCSV")"
  BADROWS="$(awk -F', *' -v us=",${UUID_CSV}," '{if (index(us, ","$1",")==0 || $2 !~ /^[0-9]+$/) c++} END {print c+0}' "$PEAKCSV")"
  NSEEN="$(awk -F', *' '{print $1}' "$PEAKCSV" | sort -u | wc -l)"
  if [ "$NROWS" -ge 1 ] && [ "$BADROWS" -eq 0 ] && [ "$NSEEN" -eq "$NGPU" ]; then MEAS_OK=1
  else echo "VRAM CSV invalid (rows=${NROWS}, bad=${BADROWS}, uuids_seen=${NSEEN}/${NGPU})"; MEAS_OK=0; fi
fi
PEAK="$(awk -F', *' 'BEGIN{m=-1} $2 ~ /^[0-9]+$/ {if ($2+0>m) m=$2+0} END{print m}' "$PEAKCSV")"
# per-UUID maxima, sorted so the emitted line is byte-deterministic
PER_UUID="$(awk -F', *' '$2 ~ /^[0-9]+$/ {if ($2+0 > m[$1]) m[$1]=$2+0} END {for (u in m) print u ":" m[u]}' "$PEAKCSV" | tr -d ' ' | sort | paste -sd, -)"
[ -n "$PER_UUID" ] || PER_UUID="none"

# --- outcome classification (B5) ---
OOM_RE="CUDA out of memory|OutOfMemoryError"
DONE_RE="max_steps=${MAXSTEPS}[^0-9].*reached"
echo "=== RESULT: cell ${CELL} s${MAXSTEPS} rc=${rc} | wall_fit ${WALL_FIT}s | peak ${PEAK} MiB / ${TOTAL_MIB} MiB per GPU | measurement_valid=${MEAS_OK} ==="
if [ "$rc" -eq 0 ] && ! grep -Eq "$DONE_RE" "$TRAINLOG"; then
  echo "FAIL: rc=0 but no 'max_steps=${MAXSTEPS} reached' completion marker in train log"
  rc=4
fi
if [ "$rc" -ne 0 ] && grep -Eq "$OOM_RE" "$TRAINLOG"; then
  echo "--- OOM (exact error excerpt; fail-honest, no fallback) ---"
  grep -m1 -B2 -A10 -E "$OOM_RE" "$TRAINLOG"
  rc=3
fi
if [ "$rc" -eq 0 ] && [ "$MEAS_OK" -ne 1 ]; then
  echo "FAIL: training completed but the peak-VRAM measurement is invalid"
  rc=5   # fail-closed (mem_probe convention): a clean run with no valid peak is not a result
fi

# --- the one machine-parseable line the collector reads (always emitted) ---
echo "P0RESULT cell=${CELL} maxsteps=${MAXSTEPS} ngpu=${NGPU} mb=${MB} rc=${rc} wall_fit=${WALL_FIT} peak_overall_mib=${PEAK} peak_per_uuid=${PER_UUID} valid=${MEAS_OK}"

[ "$rc" -eq 0 ] && [ "$MEAS_OK" -eq 1 ] || echo "cell ${CELL} s${MAXSTEPS} NOT clean (rc=${rc}, measurement_valid=${MEAS_OK}); see ${TRAINLOG}"
exit "$rc"
