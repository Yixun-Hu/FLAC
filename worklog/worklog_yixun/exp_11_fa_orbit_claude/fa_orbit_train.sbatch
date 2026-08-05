#!/bin/bash
# ============================================================================
# exp_11 ARM TRAINING LAUNCHER — one orbit arm, one rung, plan §5 Round 2 + §10.
#
# Reuses the machinery already reviewed elsewhere: torchrun + SLURM-env
# neutralization + world-size gate + UUID poller provenance from
# p0_profile.sbatch; wandb identity gate, DINOv3 pin and init-identity gate from
# exp_07's bf_scratch_launch.sh; restart-lineage gating from exp_10's
# bf_resume_launch.sh. The arms train with train.py itself (parity: p0_runner.py
# is profiling-only) — verified rank-safe under torchrun, see "torchrun" below.
#
# SUBMIT CONTRACT (the operator passes the resources; #SBATCH lines cannot read
# the environment, and the wall limit comes from the P0-measured rate):
#
#   sbatch --job-name=exp11-<ARM>-train \
#          --gres=gpu:l40:<N> --cpus-per-task=$((8+7*N)) --mem=$((12*N+12))G \
#          --time=<from the P0 report: 40000 / steps_s + startup margin> \
#          --export=ALL,ARM=<C4L|C8|C16|C32>,RUNG=<16x4|8x8>,EXPECT_SHA=<full sha>,\
# MIN_FREE_MB=<from the P0 per-GPU peak + margin>[,MAXSTEPS=40000][,RESUME_CKPT=..,EXPECTED_STEP=..] \
#          worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch
#
#   ARM          C4L | C8 | C16 | C32          (FA1/VAN are P0 cells, never arms)
#   RUNG         16x4 | 8x8                    (32x2 is BARRED: C4L OOMed at that
#                                               rung without grad-ckpt in the P0
#                                               smoke — a measured bound)
#   MAXSTEPS     default 40000 (the plan's primary matched-step budget)
#   MIN_FREE_MB  OPERATOR INPUT from the P0 per-GPU peak of this arm+rung plus
#                margin; no default, because guessing it would defeat the gate
#   TIME LIMIT   OPERATOR INPUT computed from the P0 steps/s at launch time
#   RESUME_CKPT / EXPECTED_STEP  crash restart only (exp_10 rules, below)
#
# LINEAGE (fail-closed, exactly two admissible stories):
#   INITIAL  EXPECTED_STEP unset/0 and no RESUME_CKPT; the arm's run directory
#            must not already exist (no clobber of a live or finished run).
#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN save-dir
#            (only this run writes there) AND MAXSTEPS > EXPECTED_STEP.
#
# WORLD-SIZE PROTECTION (two layers, because a silent world-size-1 run would burn
# days and produce BN-16/BN-8 statistics instead of BN-64):
#   1. an early-abort WATCHDOG: a background timer scancels the job if Lightning's
#      literal "All distributed processes registered. Starting with <N> processes"
#      has not appeared in the training log within WATCHDOG_SEC (default 300);
#   2. a POST-HOC gate on the same literal after training exits (rc 6 if absent).
#   Training itself runs in the FOREGROUND through a synchronous tee, so the log
#   both layers read is flushed by the time they read it.
#
# torchrun: Lightning 2.1.0 elects TorchElastic before SLURMEnvironment
# (accelerator_connector.py:420), so the ranks torchrun starts are used as-is;
# the SLURM rank variables are unset so SLURMEnvironment cannot claim the job and
# stall waiting for ranks srun never launched. train.py needs NO modification
# under this launcher: WandbLogger.experiment is @rank_zero_experiment (ranks > 0
# get a _DummyExperiment, so train.py's `isinstance(logger.experiment.id, str)`
# is simply False there and no rank-1 wandb run is created), logger.watch is
# @rank_zero_only, and ModelCheckpoint.setup broadcasts rank 0's resolved dirpath
# to every rank (model_checkpoint.py:265-267), so all ranks write one checkpoint
# tree. rank_zero_only.rank reads RANK first, which torchrun sets.
#
# DRYRUN=1 runs every cheap gate, prints the exact train.py argv and its parity
# diff against the exp_07 reference argv, and exits WITHOUT touching Slurm, GPUs,
# wandb or torchrun (the commit/drift gate is advisory in that mode only).
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

DRYRUN="${DRYRUN:-0}"
MAXSTEPS="${MAXSTEPS:-40000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
WATCHDOG_SEC="${WATCHDOG_SEC:-300}"
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- A. parameters -----------------------------------------------------------
[ -n "${ARM:-}" ] || { echo "ARM must be exported (C4L|C8|C16|C32) - abort"; exit 2; }
[ -n "${RUNG:-}" ] || { echo "RUNG must be exported (16x4|8x8) - abort"; exit 2; }
[ -n "${EXPECT_SHA:-}" ] || { echo "EXPECT_SHA (full reviewed commit sha) must be exported - abort"; exit 2; }

case "$ARM" in
  C4L|C8|C16|C32) ;;
  *) echo "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN are P0 profiling cells, never arms) - abort"; exit 2 ;;
esac
case "$RUNG" in
  16x4|8x8) ;;
  32x2) echo "RUNG 32x2 is BARRED for the arms: the P0 smoke measured an OOM for C4L at 32x2 without ViT grad-ckpt, so every richer orbit OOMs too - use RUNG 16x4 or 8x8 - abort"; exit 2 ;;
  *) echo "RUNG '${RUNG}' must be 16x4 or 8x8 - abort"; exit 2 ;;
esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || { echo "RUNG ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"; exit 2; }
case "$MAXSTEPS" in ''|*[!0-9]*) echo "MAXSTEPS '${MAXSTEPS}' must be a positive integer - abort"; exit 2;; esac
[ "$MAXSTEPS" -gt 0 ] || { echo "MAXSTEPS '${MAXSTEPS}' must be > 0 - abort"; exit 2; }
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort"; exit 2;; esac
case "$CHECKPOINT_EVERY" in ''|*[!0-9]*) echo "CHECKPOINT_EVERY '${CHECKPOINT_EVERY}' must be a positive integer - abort"; exit 2;; esac

NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="outputs_FLAC/exp11_${ARM}"
RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU) — ${TS} — host $(hostname) ==="

# --- B. arm -> config (single source of truth) + semantic gate ----------------
arm_config_for() {
  case "$1" in
    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
    *) return 1 ;;
  esac
}
MODEL_CONFIG="$(arm_config_for "$ARM")" || { echo "no config mapped for arm '${ARM}' - abort"; exit 2; }
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || { echo "arm config '${MODEL_CONFIG}' does not exist - abort"; exit 2; }
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || { echo "arm/config semantic gate FAILED - abort"; exit 2; }
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
t = cfg.get("training", {}); bad = []
want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
angles = t.get("frame_avg_angles")
if t.get("cond_method") != "fa_invariant":
    bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
if not isinstance(angles, list) or len(angles) != want:
    bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
elif angles != [k * 360.0 / want for k in range(want)]:
    bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if len(vits) != 2:
    bad.append(f"expected 2 ViT conditioners, got {len(vits)}")
for c in vits:                                  # the fast recipe: grad-ckpt OFF, strictly False
    gc = c["config"].get("gradient_checkpointing", False)
    if gc is not False:
        bad.append(f"{c['id']}: gradient_checkpointing={gc!r} (want False)")
if t.get("use_ema") is not True:
    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
print(f"gate OK: {arm} config carries the uniform C{want} orbit, grad-ckpt False, EMA on")
PY

# --- C. commit binding + tracked-surface drift -------------------------------
# Fail-closed in real mode; advisory under DRYRUN so the parity report can be run
# on a working tree (a dry run cannot start training).
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" "$EXP07/FLAC_AR_BF.json" 2>/dev/null)"
if [ "$DRYRUN" = "1" ]; then
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
    || echo "DRY-RUN ADVISORY: HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here)"
  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"; exit 2; }
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || { echo "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"; exit 2; }
  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
  echo "commit binding OK: ${HEAD_SHA}"
fi

# --- D. lineage: INITIAL vs RESTART (exp_10 rules) ----------------------------
SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
if [ "$EXPECTED_STEP" -eq 0 ]; then
  MODE="INITIAL"
  [ -z "$RESUME_CKPT" ] || { echo "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"; exit 2; }
  [ ! -e "$RUNDIR" ] || { echo "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run (restart with EXPECTED_STEP/RESUME_CKPT, or move it aside) - abort"; exit 2; }
else
  MODE="RESTART"
  [ -n "$RESUME_CKPT" ] || { echo "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"; exit 2; }
  [ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  case "$RESUME_REAL" in
    "${SAVEDIR_REAL}"/*) ;;
    *) echo "a RESTART may only resume a checkpoint written by this arm, i.e. one inside ${SAVEDIR_REAL}/"
       echo "  got: ${RESUME_REAL} - abort"; exit 2 ;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }
fi
echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY})"

# --- E. the exact train.py argv ----------------------------------------------
ARGV=(
  --model-config "$MODEL_CONFIG_ABS"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus "$NGPU" --num-nodes 1
  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
  --val-every -1 --val-dataset-config ''
  --gradient-clip-val 0.0
  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
)
[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")

# --- F. argv-parity dry run (plan N13) ---------------------------------------
ARGV_FILE="$(mktemp)"; printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE"
python3 - "$ARGV_FILE" "$MODE" <<'PY'
import sys
# The exp_07 B-F reference argv (bf_scratch_launch.sh), the lineage this sweep
# continues. Anything differing outside the enumerated allowances is a silent
# recipe change and aborts the launch.
REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
--pretransform-ckpt-path weights/FLAC/VAE.safetensors
--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
--logger wandb --checkpoint-every 2500
--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
ALLOWED = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
           "--num-gpus", "--batch-size", "--logger", "--ckpt-path"}

new_tokens = open(sys.argv[1]).read().split("\n")
if new_tokens and new_tokens[-1] == "":
    new_tokens.pop()
mode = sys.argv[2]

def as_map(tokens):
    out, i = {}, 0
    while i < len(tokens):
        flag = tokens[i]
        if not flag.startswith("--"):
            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
        val = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else ""
        out[flag] = val
        i += 2 if val != "" or (i + 1 < len(tokens) and not tokens[i + 1].startswith("--")) else 1
    return out

def defaults_ini():
    vals = {}
    for line in open("defaults.ini"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        vals["--" + k.strip().replace("_", "-")] = v.strip().strip("'\"")
    return vals

ref, new, ini = as_map(REF), as_map(new_tokens), defaults_ini()
violations, allowed_diffs, explicit_defaults = [], [], []
for flag in sorted(set(ref) | set(new)):
    if flag in ref and flag in new:
        if ref[flag] != new[flag]:
            (allowed_diffs if flag in ALLOWED else violations).append(
                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
    elif flag in new:                       # explicit here, defaulted in exp_07
        if flag in ALLOWED:
            allowed_diffs.append(f"{flag}: added, {new[flag]!r}")
        elif flag in ini and ini[flag] == new[flag]:
            explicit_defaults.append(f"{flag}={new[flag]!r} (== defaults.ini)")
        else:
            violations.append(f"{flag}: added with {new[flag]!r}, defaults.ini has "
                              f"{ini.get(flag, '<absent>')!r}")
    else:
        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")

print("--- train.py argv ---")
print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
print("--- argv parity vs exp_07 B-F ---")
for d in allowed_diffs:
    print(f"  allowed  {d}")
for d in explicit_defaults:
    print(f"  explicit {d}")
if violations:
    print("ARGV PARITY VIOLATIONS:")
    for v in violations:
        print(f"  !! {v}")
    raise SystemExit(2)
print(f"ARGV PARITY OK ({mode}): only the enumerated differences, everything else matches "
      "exp_07 or restates a defaults.ini value")
PY
parity=$?
rm -f "$ARGV_FILE"
[ "$parity" -eq 0 ] || { echo "argv parity check FAILED - abort"; exit 2; }
if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN complete: gates A–F passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE}; "
  echo "  (Slurm/GPU/VRAM/wandb/ViT gates and training are skipped in DRYRUN)"
  exit 0
fi

# --- G. Slurm shape + GPUs ----------------------------------------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || { echo "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"; exit 2; }
[ "${SLURM_NTASKS:-1}" = "1" ] || { echo "expected 1 task, got ${SLURM_NTASKS} - abort"; exit 2; }
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || { echo "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"; exit 2; }
UUIDS=()
for ROW in "${GPU_ROWS[@]}"; do
  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"
  N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
  case "$N" in *L40*) ;; *) echo "allocated GPU '${N}' is not an L40 - abort"; exit 2;; esac
  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
done
UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"

# --- H. per-GPU free-VRAM gate (MIN_FREE_MB is operator input from P0) -------
[ -n "$MIN_FREE_MB" ] || { echo "MIN_FREE_MB must be exported (per-GPU floor from this arm+rung's P0 peak) - abort"; exit 2; }
case "$MIN_FREE_MB" in ''|*[!0-9]*) echo "MIN_FREE_MB '${MIN_FREE_MB}' must be an integer - abort"; exit 2;; esac
for U in "${UUIDS[@]}"; do
  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

# --- I. wandb identity gate (fail-closed; the arms train WITH wandb) ----------
mkdir -p "$SAVEDIR"
export WANDB_DIR="$REPO/$SAVEDIR"
eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
python3 - <<'PY' || { echo "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"; exit 2; }
import sys
try:
    import wandb; email = wandb.Api().viewer.email
except Exception as e:
    sys.exit(f"wandb identity check FAILED: {e}")
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY

# --- J. DINOv3 pin + init-identity gate (exp_07's checks, exp_11 arms) --------
HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || { echo "ViT-pin / init-identity GATE FAILED - abort"; exit 1; }

# --- K. atomic pre-launch manifest (plan N12) --------------------------------
TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
ENV_SHA="$(pip freeze 2>/dev/null | sha256sum | awk '{print $1}')"
{
  echo "# exp_11 arm launch manifest"
  echo "timestamp ${TS}"
  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE}"
  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
  echo "commit ${HEAD_SHA}"
  echo "model_config ${MODEL_CONFIG_ABS}"
  echo "config_sha256 ${CONFIG_SHA}"
  echo "env_pip_freeze_sha256 ${ENV_SHA}"
  echo "gpu_uuids ${UUID_CSV}"
  echo "time_limit $(squeue -h -j "${SLURM_JOB_ID}" -o %l 2>/dev/null || echo unknown)"
  echo "min_free_mb ${MIN_FREE_MB}"
  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP}"
  echo "save_dir ${SAVEDIR}"
  echo "train_log ${TRAINLOG}"
  echo "train_log_copy ${SAVEDIR_LOG}"
  echo "wandb ${NAME}/${EXPNAME}"
  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
} > "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
echo "manifest: ${MANIFEST}"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- L. world-size early-abort watchdog --------------------------------------
# Literal from lightning_fabric/utilities/distributed.py:296 (PL 2.1.0).
WORLD_MSG="All distributed processes registered. Starting with ${NGPU} processes"
: > "$TRAINLOG"
(
  sleep "$WATCHDOG_SEC"
  if ! grep -qF "$WORLD_MSG" "$TRAINLOG" 2>/dev/null; then
    echo "WATCHDOG: '${WORLD_MSG}' absent after ${WATCHDOG_SEC}s — cancelling job ${SLURM_JOB_ID} rather than burn days at world size 1"
    scancel "${SLURM_JOB_ID}"
  fi
) &
WATCHDOG=$!

# --- M. training: torchrun in the foreground, dual-tee, PIPESTATUS rc --------
# The SLURM rank variables must not be visible: with them set, Lightning elects
# SLURMEnvironment (creates_processes_externally) and never starts the ranks.
unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" 2>&1 \
  | tee "$TRAINLOG" "$SAVEDIR_LOG"
rc="${PIPESTATUS[0]}"
kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null

# --- N. post-hoc gates + classification --------------------------------------
DONE_MSG="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."   # fit_loop.py:167
if ! grep -qF "$WORLD_MSG" "$TRAINLOG"; then
  echo "WORLD-SIZE GATE FAILED: '${WORLD_MSG}' never appeared — this run did NOT train with ${NGPU} ranks"
  [ "$rc" -eq 0 ] && rc=6
fi
if [ "$rc" -eq 0 ] && ! grep -qF "$DONE_MSG" "$TRAINLOG"; then
  echo "NOTE: rc=0 without the completion marker (${DONE_MSG}) — treat as an early exit, not a finished budget"
  rc=4
fi
if grep -Eq "CUDA out of memory|OutOfMemoryError" "$TRAINLOG"; then
  echo "--- OOM excerpt (fail-honest, no fallback) ---"
  grep -m1 -B2 -A10 -E "CUDA out of memory|OutOfMemoryError" "$TRAINLOG"
  [ "$rc" -eq 0 ] && rc=3
fi
LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"
echo "=== exp_11 ${ARM} ${RUNG} exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') | last ckpt: ${LAST_CKPT:-<none>} ==="
echo "restart with: --export=ALL,ARM=${ARM},RUNG=${RUNG},EXPECT_SHA=${HEAD_SHA},MIN_FREE_MB=${MIN_FREE_MB},MAXSTEPS=${MAXSTEPS},EXPECTED_STEP=<step of that ckpt>,RESUME_CKPT=${LAST_CKPT:-<path>}"
exit "$rc"
