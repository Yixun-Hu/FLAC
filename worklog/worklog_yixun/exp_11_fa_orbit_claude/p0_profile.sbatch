#!/bin/bash
# ============================================================================
# exp_11 P0 profiling cell — plan Rev 3 §10 (throughput/fit matrix + component
# attribution). ONE cell, ONE 30-step job. Rev 2 of this script applies all
# round-2 review findings.
#
# Rate measurement (review B2): NOT job wall time. p0_runner.py stamps completed
# optimizer steps 10 and 30 inside the single fit (after cuda.synchronize, rank
# zero) and prints P0STEP lines; this script extracts them and the collector
# computes steps/s = 20 / (t30_mono - t10_mono). Job wall time is emitted only
# as a startup diagnostic.
#
# DDP launch (review B1, CRITICAL): under sbatch, Lightning 2.1.0 detects
# SLURM_NTASKS -> SLURMEnvironment, whose creates_processes_externally=True means
# it NEVER spawns the missing local ranks. With --ntasks=1 that silently yields
# world size 1, i.e. BN over the micro-batch instead of micro x N = 64 and fake
# "multi-GPU" numbers. So: the SLURM rank variables are unset (TorchElastic then
# wins the cluster-environment election, which is also its documented priority)
# and the N ranks are launched explicitly with torchrun inside the one task.
# A fail-closed WORLD-SIZE GATE afterwards requires Lightning's own literal
# "All distributed processes registered. Starting with N processes"
# (lightning_fabric/utilities/distributed.py:296) plus in-window nonzero memory
# on EVERY allocated UUID (rank-placement evidence).
#
# Resource flags cannot be read from the environment in #SBATCH lines, so the
# SUBMITTER (p0_submit_matrix.sh) passes them on the command line:
#
#   sbatch --job-name=p0-<CELL> \
#          --gres=gpu:l40:$NGPU --cpus-per-task=$((8+7*NGPU)) \
#          --mem=$((12*NGPU+12))G --time=00:40:00 \
#          --export=ALL,EXPECT_SHA=<sha>,RUNID=<runid>,CELL=<tag>,MAXSTEPS=30[,NUM_WORKERS=N] \
#          p0_profile.sbatch
#
# There is NO MODEL_CONFIG knob (review B5): the config is DERIVED from the cell
# family by the map below — the single source of truth, which the submitter reads
# through `P0_PRINT_CONFIG=<CELL> bash p0_profile.sbatch` so both agree by
# construction — and its sha256 is emitted for the collector to verify.
#
# Emits exactly one machine-parseable final line:
#   P0RESULT runid= sha= jobid= cell= maxsteps= ngpu= mb= workers= rc= config_sha= \
#            wall_fit= t10= t30= t10_mono= t30_mono= peak_overall_mib= \
#            peak_per_uuid= vram_csv= pollcsv_sha256= valid=
# OOM cells still emit it (rc=3) and exit 3.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_p0_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
CANON="$REPO/src/configs/model_configs/FLAC/AR/FLAC_AR.json"

# --- cell family -> model config: THE single source of truth (review B5) ---
p0_config_for() {
  case "$1" in
    VAN)                echo "$CANON" ;;
    FA1|C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
    CKPT4)          echo "$EXP07/FLAC_AR_BF.json" ;;
    *)              return 1 ;;
  esac
}
# The submitter queries the map with this hook (prints the path, exits; no gates,
# no Slurm needed) so it cannot drift from what the job will actually run.
# N7: honoured ONLY outside a Slurm job — inside one it is inherited state that
# could turn a scheduled cell into a misleading successful no-op, so it is ignored
# and scrubbed from the environment.
if [ -n "${P0_PRINT_CONFIG:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  p0_config_for "${P0_PRINT_CONFIG%%_*}" || { echo "unknown cell family in '${P0_PRINT_CONFIG}'" >&2; exit 2; }
  exit 0
fi
unset P0_PRINT_CONFIG

cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

JID="${SLURM_JOB_ID:?must run under sbatch}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- required cell parameters (no defaults: an unset knob must abort) ---
: "${CELL:?CELL (e.g. C4L_32x2) must be exported - abort}"
: "${MAXSTEPS:?MAXSTEPS must be exported - abort}"
: "${RUNID:?RUNID (submission run id) must be exported - abort}"
: "${EXPECT_SHA:?EXPECT_SHA (full reviewed commit sha) must be exported at submission - abort}"
# Never defaulted (re-review B4): an ambient NUM_WORKERS must not silently change
# a matrix cell, so the submitter always exports it explicitly.
: "${NUM_WORKERS:?NUM_WORKERS must be exported explicitly (6 for matrix/spot cells) - abort}"
case "$NUM_WORKERS" in ''|*[!0-9]*) echo "NUM_WORKERS '${NUM_WORKERS}' not numeric - abort"; exit 2;; esac

echo "=== exp_11 P0 cell ${CELL} - ${TS} - job ${JID} - run ${RUNID} - host $(hostname) ==="

# --- cell grammar + rung as the single source of MB/NGPU (review B5) ---
case "$CELL" in
  VAN_32x2|VAN_16x4|VAN_8x8|FA1_32x2|FA1_16x4|FA1_8x8) ;;
  C4L_32x2|C4L_16x4|C4L_8x8|C8_32x2|C8_16x4|C8_8x8) ;;
  C16_32x2|C16_16x4|C16_8x8|C32_32x2|C32_16x4|C32_8x8|CKPT4_32x2|CKPT4_16x4|CKPT4_8x8) ;;
  *) echo "CELL '${CELL}' is not a legal <FAMILY>_<RUNG> cell - abort"; exit 2 ;;
esac
FAMILY="${CELL%%_*}"
RUNG="${CELL#*_}"
MB="${RUNG%x*}"
NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"; exit 2; }
# The approved P0 contract is EXACTLY 30 steps (re-review B3): a different budget
# measures a different workload, and peak VRAM would not be comparable.
[ "$MAXSTEPS" = "30" ] || { echo "MAXSTEPS '${MAXSTEPS}' != 30 (the pinned P0 budget) - abort"; exit 2; }

# --- commit binding + drift gate (mem_probe review B3) ---
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || { echo "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"; exit 2; }
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
echo "commit binding OK: ${HEAD_SHA}"

# --- Slurm shape asserts ---
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || { echo "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"; exit 2; }
[ "${SLURM_NTASKS:-1}" = "1" ] || { echo "expected 1 task, got ${SLURM_NTASKS} - abort"; exit 2; }

# --- exactly NGPU visible GPUs, all L40, addressed by UUID from here on ---
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || { echo "expected exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"; exit 2; }
UUIDS=()
for ROW in "${GPU_ROWS[@]}"; do
  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"
  N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
  T="$(echo "$ROW" | cut -d, -f3 | tr -d ' ')"
  case "$N" in *L40*) ;; *) echo "allocated GPU '${N}' is not an L40 - abort"; exit 2;; esac
  case "$T" in ''|*[!0-9]*) echo "non-numeric memory.total '${T}' - abort"; exit 2;; esac
  UUIDS+=("$U"); TOTAL_MIB="$T"
  echo "GPU: ${N} ${U} total ${T} MiB"
done
UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- config: derived, existing, hashed (review B5) ---
MODEL_CONFIG="$(p0_config_for "$FAMILY")" || { echo "no config mapped for family '${FAMILY}' - abort"; exit 2; }
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || { echo "derived config '${MODEL_CONFIG}' does not exist - abort"; exit 2; }
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${FAMILY}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

# --- semantic gate: the config must carry the orbit/checkpointing the cell claims ---
python3 - "$MODEL_CONFIG_ABS" "$FAMILY" <<'PY' || { echo "cell/config semantic gate FAILED - abort"; exit 2; }
import json, sys
cfg = json.load(open(sys.argv[1])); fam = sys.argv[2]
t = cfg.get("training", {}); bad = []
# FA1: fa_invariant with a SINGLE-angle orbit -> cylindrical pose path + exactly
# one ViT pass (yaw_rotation.invariant_conditioning returns the base pass when
# len(angles) == 1), the attribution baseline the orbit slope is fitted against.
want = {"VAN": None, "FA1": 1, "C4L": 4, "C8": 8, "C16": 16, "C32": 32, "CKPT4": 4}[fam]
angles = t.get("frame_avg_angles")
if want is None:
    if "cond_method" in t or angles is not None:
        bad.append("VAN config carries frame-averaging keys")
else:
    if t.get("cond_method") != "fa_invariant":
        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
    if not isinstance(angles, list) or len(angles) != want:
        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
    elif angles != [k * 360.0 / want for k in range(want)]:
        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if len(vits) != 2:
    bad.append(f"expected 2 ViT conditioners, got {len(vits)}")
want_gc = (fam == "CKPT4")
for c in vits:
    gc = c["config"].get("gradient_checkpointing", False)
    if gc is not want_gc:                      # strict identity: 0/1 are not booleans
        bad.append(f"{c['id']}: gradient_checkpointing={gc!r} (want {want_gc})")
if bad:
    sys.exit("CELL/CONFIG GATE: " + "; ".join(bad))
print(f"gate OK: {fam} config has the expected orbit and grad-ckpt={want_gc}")
PY

# --- Lightning must NOT see a SLURM cluster environment (review B1) ---
# With SLURM_NTASKS set, SLURMEnvironment.creates_processes_externally is True and
# Lightning waits for ranks srun never launched. torchrun provides the ranks here.
unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
if [ "$NGPU" -eq 1 ]; then
  LAUNCH=(python "$EXPDIR/p0_runner.py")
  STRAT_ARGS=(--strategy auto --sync-batchnorm false)
else
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "$EXPDIR/p0_runner.py")
  STRAT_ARGS=(--strategy ddp_find_unused_parameters_true --sync-batchnorm true)
fi

# --- UUID-bound poller: sequenced ticks with mem/util/power (reviews B3, B7) ---
PEAKCSV="$EXPDIR/p0_${CELL}_${TS}_jid${JID}_vram.csv"
STOPFILE="$EXPDIR/.p0_stop_${JID}"
rm -f "$STOPFILE"; : > "$PEAKCSV"
(
  TICK=0
  while [ ! -e "$STOPFILE" ]; do
    NOW="$(date +%s.%N)"
    SAMPLE="$(nvidia-smi --id="$UUID_CSV" --query-gpu=uuid,memory.used,utilization.gpu,power.draw \
                --format=csv,noheader,nounits 2>/dev/null)"
    if [ -z "$SAMPLE" ]; then
      echo "POLLFAIL tick=${TICK} ts=${NOW}"
    else
      echo "$SAMPLE" | awk -F', *' -v tick="$TICK" -v ts="$NOW" \
        '{gsub(/ /,""); printf "tick=%s uuid=%s mem=%s util=%s power=%s ts=%s\n", tick, $1, $2, $3, $4, ts}'
    fi
    TICK=$((TICK + 1))
    sleep 0.5
  done
) >> "$PEAKCSV" &
POLLER=$!

# --- training ---
TRAINLOG="$EXPDIR/p0_${CELL}_${TS}_jid${JID}_train.log"
T0="$(date +%s.%N)"
"${LAUNCH[@]}" \
  --model-config "$MODEL_CONFIG_ABS" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 \
  --num-workers "$NUM_WORKERS" --seed 42 \
  --num-gpus "$NGPU" --num-nodes 1 "${STRAT_ARGS[@]}" --precision bf16-mixed \
  --val-every -1 --val-dataset-config '' \
  --ckpt-path '' --pretrained-ckpt-path '' --gradient-clip-val 0.0 \
  --logger none --checkpoint-every 100000 \
  --name "FLAC_exp11_p0_${CELL}" --experiment-name "exp11_p0_${CELL}" \
  --save-dir outputs_FLAC/exp11_p0 2>&1 | tee "$TRAINLOG"
rc="${PIPESTATUS[0]}"
T1="$(date +%s.%N)"
WALL_FIT="$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')"

# --- stop the poller cleanly and check that it lived to the end (review B7) ---
touch "$STOPFILE"
wait "$POLLER"; POLLER_RC=$?
rm -f "$STOPFILE"

# --- in-fit window marks from the runner's callback (review B2) ---
# Matched as a SUBSTRING, not an anchored line: Lightning's progress bar writes
# carriage-return updates on the same stream, so a mark can share a line with it.
p0_mark() {  # $1 = step, $2 = key (t | ts)
  awk -v st="$1" -v key="$2" '
    { pat = "P0STEP step=" st " t=[0-9.]+ ts=[0-9.]+"
      if (match($0, pat)) {
        n = split(substr($0, RSTART, RLENGTH), f, " ")
        for (i = 1; i <= n; i++) { split(f[i], kv, "="); if (kv[1] == key) v = kv[2] } } }
    END { print v }' "$TRAINLOG"
}
T10_MONO="$(p0_mark 10 t)"; T30_MONO="$(p0_mark 30 t)"
T10_TS="$(p0_mark 10 ts)"; T30_TS="$(p0_mark 30 ts)"
MARKS_OK=1
for V in "$T10_MONO" "$T30_MONO" "$T10_TS" "$T30_TS"; do
  case "$V" in ''|*[!0-9.]*) MARKS_OK=0;; esac
done
[ "$MARKS_OK" -eq 1 ] || { echo "missing/blank P0STEP window marks in the train log"; T10_MONO=-1; T30_MONO=-1; T10_TS=-1; T30_TS=-1; }

# --- measurement validation: every tick must be complete (reviews B1, B7) ---
MEAS_OK=1
if [ "$POLLER_RC" -ne 0 ]; then echo "poller exited ${POLLER_RC} - measurement INVALID"; MEAS_OK=0; fi
if grep -q "POLLFAIL" "$PEAKCSV"; then echo "poller reported a query failure - measurement INVALID"; MEAS_OK=0; fi
NTICKS="$(awk '{print $1}' "$PEAKCSV" | sort -u | wc -l)"
BADTICKS="$(awk -v want="$NGPU" -v us=",${UUID_CSV}," '
  { split($1,ka,"="); split($2,kb,"="); split($3,kc,"="); split($4,kd,"="); split($5,ke,"=");
    t=ka[2]; u=kb[2]; m=kc[2]; ut=kd[2]; pw=ke[2];
    if (index(us, ","u",")==0 || m !~ /^[0-9]+$/) { bad[t]=1; next }
    # utilisation/power are mandatory evidence (re-review B2), not decoration
    if (ut !~ /^[0-9]+(\.[0-9]+)?$/ || pw !~ /^[0-9]+(\.[0-9]+)?$/) { bad[t]=1; next }
    if ((t SUBSEP u) in seen) { bad[t]=1; next }        # duplicate UUID in one tick
    seen[t SUBSEP u]=1; n[t]++ }
  END { nbad=0; for (t in n) if (n[t] != want && !(t in bad)) nbad++; for (t in bad) nbad++; print nbad+0 }' "$PEAKCSV")"
if [ "$NTICKS" -lt 2 ] || [ "$BADTICKS" -ne 0 ]; then
  echo "poller CSV invalid (ticks=${NTICKS}, incomplete/bad ticks=${BADTICKS}, want ${NGPU} uuid rows per tick)"
  MEAS_OK=0
fi
LAST_TS="$(awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="ts")v=a[2]}} END{print v+0}' "$PEAKCSV")"
LATE="$(awk -v a="$LAST_TS" -v b="$T1" 'BEGIN{print (b-a > 2.0) ? 1 : 0}')"
if [ "$LATE" -eq 1 ]; then echo "poller stopped ${LAST_TS} > 2s before train end ${T1} - measurement INVALID"; MEAS_OK=0; fi

PEAK="$(awk 'BEGIN{m=-1} {for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="mem" && a[2] ~ /^[0-9]+$/ && a[2]+0>m) m=a[2]+0}} END{print m}' "$PEAKCSV")"
PER_UUID="$(awk '{u="";m="";for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="uuid")u=a[2];if(a[1]=="mem")m=a[2]}
  if (u!="" && m ~ /^[0-9]+$/ && m+0 > p[u]) p[u]=m+0} END{for (u in p) print u ":" p[u]}' "$PEAKCSV" | sort | paste -sd, -)"
[ -n "$PER_UUID" ] || PER_UUID="none"

# --- WORLD-SIZE / rank-placement gate (review B1) ---
# Literal from lightning_fabric/utilities/distributed.py:296 (PL 2.1.0).
WORLD_MSG="All distributed processes registered. Starting with ${NGPU} processes"
if [ "$NGPU" -ge 2 ]; then
  if ! grep -qF "$WORLD_MSG" "$TRAINLOG"; then
    echo "WORLD-SIZE GATE FAILED: '${WORLD_MSG}' not in the train log (DDP did not start with ${NGPU} ranks)"
    MEAS_OK=0; [ "$rc" -eq 0 ] && rc=6
  fi
else
  if grep -Eq "All distributed processes registered\. Starting with [2-9][0-9]* processes" "$TRAINLOG"; then
    echo "WORLD-SIZE GATE FAILED: a 1-GPU cell started a multi-rank world"
    MEAS_OK=0; [ "$rc" -eq 0 ] && rc=6
  fi
fi
if [ "$MARKS_OK" -eq 1 ]; then
  IDLE="$(awk -v lo="$T10_TS" -v hi="$T30_TS" '
    {u="";m="";ts="";for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="uuid")u=a[2];if(a[1]=="mem")m=a[2];if(a[1]=="ts")ts=a[2]}
     if (u!="" && ts+0>=lo+0 && ts+0<=hi+0 && m+0>0) busy[u]=1}
    END{n=0; for (u in busy) n++; print n}' "$PEAKCSV")"
  if [ "${IDLE:-0}" -ne "$NGPU" ]; then
    echo "RANK-PLACEMENT GATE FAILED: only ${IDLE:-0}/${NGPU} allocated GPU(s) held memory during the measured window"
    MEAS_OK=0; [ "$rc" -eq 0 ] && rc=6
  fi
fi

# --- outcome classification ---
# Literal from pytorch_lightning/loops/fit_loop.py:167 (PL 2.1.0).
DONE_MSG="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."
OOM_RE="CUDA out of memory|OutOfMemoryError"
echo "=== RESULT: cell ${CELL} rc=${rc} | wall ${WALL_FIT}s (startup diagnostic) | window ${T10_MONO} -> ${T30_MONO} | peak ${PEAK} MiB / ${TOTAL_MIB} MiB | measurement_valid=${MEAS_OK} ==="
if [ "$rc" -eq 0 ] && ! grep -qF "$DONE_MSG" "$TRAINLOG"; then
  echo "FAIL: rc=0 but the exact completion marker is absent: ${DONE_MSG}"
  rc=4
fi
if [ "$rc" -eq 0 ] && [ "$MARKS_OK" -ne 1 ]; then
  echo "FAIL: rc=0 but the in-fit window marks are missing"
  rc=4
fi
if [ "$rc" -ne 0 ] && grep -Eq "$OOM_RE" "$TRAINLOG"; then
  echo "--- OOM (exact error excerpt; fail-honest, no fallback) ---"
  grep -m1 -B2 -A10 -E "$OOM_RE" "$TRAINLOG"
  rc=3
fi
if [ "$rc" -eq 0 ] && [ "$MEAS_OK" -ne 1 ]; then
  echo "FAIL: training completed but the measurement is invalid"
  rc=5
fi

# --- the one machine-parseable line the collector reads (always emitted) ---
POLLCSV_SHA="$(sha256sum "$PEAKCSV" | awk '{print $1}')"
echo "P0RESULT runid=${RUNID} sha=${HEAD_SHA} jobid=${JID} cell=${CELL} maxsteps=${MAXSTEPS} ngpu=${NGPU} mb=${MB} workers=${NUM_WORKERS} rc=${rc} config_sha=${CONFIG_SHA} wall_fit=${WALL_FIT} t10=${T10_TS} t30=${T30_TS} t10_mono=${T10_MONO} t30_mono=${T30_MONO} peak_overall_mib=${PEAK} peak_per_uuid=${PER_UUID} vram_csv=$(basename "$PEAKCSV") pollcsv_sha256=${POLLCSV_SHA} valid=${MEAS_OK}"

[ "$rc" -eq 0 ] && [ "$MEAS_OK" -eq 1 ] || echo "cell ${CELL} NOT clean (rc=${rc}, measurement_valid=${MEAS_OK}); see ${TRAINLOG}"
exit "$rc"
