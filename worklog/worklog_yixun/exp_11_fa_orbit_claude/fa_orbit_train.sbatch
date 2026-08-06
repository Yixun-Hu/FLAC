#!/bin/bash
# ============================================================================
# exp_11 ARM TRAINING LAUNCHER — one orbit arm at the ONE pinned P0 rung.
#
# Round-3 review (fa_orbit_codex_code_r3_review.md) rebuilt this script: the
# recipe is now literally pinned rather than operator-supplied, restarts get
# exp_10's torch.load depth, the run directory is locked, the exit taxonomy is a
# tested function, and the environment/W&B destinations are gated.
#
# RECIPE (post-P0, Yixun 2026-08-05): every arm runs WITH ViT gradient
# checkpointing. P0 measured the no-ckpt recipe INFEASIBLE for C8 and richer
# orbits (OOM at micro-8, 45,457 MiB) while checkpointed peaks at ~9.4 GB, so all
# four arms share one checkpointed rung and the ONLY delta between them is the
# averaging orbit.
#
# SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
# from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
#   ARM                C4L | C8 | C16 | C32
#   EXPECT_SHA         full reviewed commit sha (required)
#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
#
# LINEAGE (fail-closed, exactly two stories):
#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
#            <RUNDIR>/checkpoints/ AND the checkpoint passes
#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
#            scheduler/EMA + binding to the original launch manifest).
#
# WORLD SIZE: no absence timer (round-3 B4 — a cold start with W&B has no
# measured bound, and `scancel` bypassed classification). Instead: a watcher that
# terminates the torchrun process group the moment Lightning reports the WRONG
# rank count, plus the post-hoc classification in fa_orbit_classify.py.
#
# torchrun: PL 2.1.0 elects TorchElastic before SLURMEnvironment, so the ranks
# torchrun starts are used as-is; the SLURM rank variables are unset so
# SLURMEnvironment cannot claim the job. train.py is unmodified and rank-safe:
# WandbLogger.experiment is @rank_zero_experiment, and ModelCheckpoint.setup
# broadcasts rank 0's dirpath to every rank.
#
# SMOKE MODE (SMOKE=1): the reviewed pre-launch smoke. Bypasses ONLY the "pins
# must be pinned" gate; every other gate still runs. Uses SMOKE_RUNG,
# SMOKE_MAXSTEPS (small), SMOKE_MIN_FREE_MB, its own identity
# (FLAC_exp11_smoke_<ARM> / exp11_smoke_<ARM>) and its own save-dir prefix, so a
# smoke can never touch or resume an arm's real lineage.
#
# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
# the guard tests never write under a production prefix. It changes no gate.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out

set -uo pipefail

# ============================ PINNED RECIPE =================================
# Filled from the reviewed P0 report; until then every value is the literal
# placeholder and the launcher refuses to run (except under SMOKE=1).
PIN_PLACEHOLDER="TO-PIN-AFTER-P0"
PINNED_RUNG="8x8"                          # P0 run 1334933 + spot 9bf1936: fastest uniform rung where ALL arms fit (C32 peak 30,817 MiB)
PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
PINNED_NGPU="8"                            # ranks
PINNED_MAXSTEPS=40000                      # plan §2 primary matched-step budget
PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
PINNED_MIN_FREE_MB="35500"                 # single floor = max across arms: C32 peak 30,817 + ~4.7 GB margin (reviewer-allowed max-floor form)
PINNED_TIME_LIMIT_C4L="42:00:00"           # 40k/0.3604 = 30.8 h x1.3 + startup
PINNED_TIME_LIMIT_C8="80:00:00"            # 40k/0.1863 = 59.6 h x1.3 + startup
PINNED_TIME_LIMIT_C16="150:00:00"          # 40k/0.0982 = 113.2 h x1.3 + startup (< 7 d partition cap)
PINNED_TIME_LIMIT_C32="167:00:00"          # partition cap 7 d; 40k needs ~215 h => PRE-REGISTERED 2-segment run: wall-stop ~step 30-31k, RESTART to 40k (~2.5 d)
PINNED_P0_MANIFEST_SHA256="b2aeaf9c1e797d5268f02faa594cc416cb59c113e8c3a8be70ad7f34e242208e"  # p0_manifest_1334933 (official matrix); spot manifest 9bf1936 sha256 recorded in the pin commit message
# Environment pins (round-3 B6) — measured on the reviewed environment:
PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
PINNED_PL_VERSION="2.1.0"
PINNED_TORCH_VERSION="2.7.0+cu126"
PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
# ============================================================================

REPO=/n/fs/gatrdp/codespace/FLAC
# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
# repo path must be absolute; FA_ORBIT_REPO_OVERRIDE lets the guard suite point a
# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
# immediately, so it can never influence a real launch.
if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$FA_ORBIT_REPO_OVERRIDE"
fi
unset FA_ORBIT_REPO_OVERRIDE
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

DRYRUN="${DRYRUN:-0}"
SMOKE="${SMOKE:-0}"
# NEW-2: the production output namespace is not operator state. Inside a Slurm
# job it is the literal below; an ambient value that disagrees aborts. The
# override exists only for non-Slurm guard dry runs.
PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != the production literal '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PRODUCTION_OUTPUT_ROOT}"
fi
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32) - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  C4L|C8|C16|C32) ;;
  *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
esac
case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac

# --- B. the pins decide the recipe (round-3 B1) -------------------------------
if [ "$SMOKE" = "1" ]; then
  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"
  NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
else
  for PIN_NAME in PINNED_RUNG PINNED_MB PINNED_NGPU PINNED_MIN_FREE_MB PINNED_P0_MANIFEST_SHA256 \
                  "PINNED_TIME_LIMIT_${ARM}"; do
    eval "PIN_VAL=\${$PIN_NAME}"
    [ "$PIN_VAL" != "$PIN_PLACEHOLDER" ] || die "${PIN_NAME} is still '${PIN_PLACEHOLDER}': the P0 report has not been pinned into this launcher yet — no arm may launch (use SMOKE=1 for the pre-launch smoke) - abort"
  done
  RUNG="$PINNED_RUNG"; MAXSTEPS="$PINNED_MAXSTEPS"; MIN_FREE_MB="$PINNED_MIN_FREE_MB"
  CHECKPOINT_EVERY="$PINNED_CHECKPOINT_EVERY"
  eval "TIME_LIMIT=\${PINNED_TIME_LIMIT_${ARM}}"
  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
fi

case "$RUNG" in
  32x2|16x4|8x8) ;;
  *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
if [ "$SMOKE" != "1" ]; then
  [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
  [ "$MAXSTEPS" = "40000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 40000 - abort"
fi
RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="

# --- C. commit binding + tracked-surface drift --------------------------------
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" "$EXP07/FLAC_AR_BF.json" 2>/dev/null)"
if [ "$DRYRUN" = "1" ]; then
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
    || echo "DRY-RUN ADVISORY: HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} (a real launch aborts here)"
  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
  echo "commit binding OK: ${HEAD_SHA}"
fi

# --- D. arm -> config (single source) + semantic gate -------------------------
arm_config_for() { case "$1" in C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;; *) return 1 ;; esac; }
MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
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
if t.get("use_ema") is not True:
    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
for c in vits:
    if "gradient_checkpointing" not in c["config"]:
        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
    elif c["config"]["gradient_checkpointing"] is not True:
        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
PY

# --- E. lineage: INITIAL vs RESTART -------------------------------------------
SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
CKPT_DIR_REAL="$(realpath -m "${RUNDIR}/checkpoints")"
LAUNCH_MANIFEST_LINK="${SAVEDIR}/launch_manifest.txt"     # written by the INITIAL launch
if [ "$EXPECTED_STEP" -eq 0 ]; then
  MODE="INITIAL"
  [ -z "$RESUME_CKPT" ] || die "INITIAL launch must not carry RESUME_CKPT (set EXPECTED_STEP > 0 to declare a RESTART) - abort"
  [ ! -e "$RUNDIR" ] || die "run directory ${RUNDIR} already exists — an INITIAL launch never clobbers a previous run - abort"
else
  MODE="RESTART"
  [ -n "$RESUME_CKPT" ] || die "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART, but RESTART requires RESUME_CKPT - abort"
  [ -f "$RESUME_CKPT" ] || die "RESUME_CKPT not found: ${RESUME_CKPT} - abort"
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  # exactly this arm's own checkpoints directory — not merely somewhere below the save root
  case "$RESUME_REAL" in
    "${CKPT_DIR_REAL}"/*.ckpt) ;;
    *) die "a RESTART may only resume a checkpoint from ${CKPT_DIR_REAL}/ (got ${RESUME_REAL}) - abort" ;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
fi
echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, ckpt every ${CHECKPOINT_EVERY})"

# --- F. the exact train.py argv ----------------------------------------------
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

# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
python3 - "$ARGV_FILE" "$MODE" <<'PY'
import sys
# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
REF = """--model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
--pretransform-ckpt-path weights/FLAC/VAE.safetensors
--max-steps 67500 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
--num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
--logger wandb --checkpoint-every 2500
--name FLAC_exp07_BF --experiment-name exp07_BF --save-dir outputs_FLAC/exp07_BF""".split()
# Flags whose VALUE may differ from exp_07 (identity, budget, rung, resume):
ALLOWED_DIFF = {"--model-config", "--name", "--experiment-name", "--save-dir", "--max-steps",
                "--num-gpus", "--batch-size", "--logger", "--checkpoint-every", "--ckpt-path"}
# Flags exp_07 left to defaults.ini and we state explicitly — whitelisted with their
# EXACT expected values (round-3 N9: no "equals the mutable ini" escape hatch):
ALLOWED_ADD = {"--num-nodes": "1", "--precision": "bf16-mixed", "--val-every": "-1",
               "--val-dataset-config": "", "--gradient-clip-val": "0.0", "--ckpt-path": None}
tokens = [t for t in open(sys.argv[1]).read().split("\n")]
if tokens and tokens[-1] == "":
    tokens.pop()
mode = sys.argv[2]

def as_map(toks):
    out, i = {}, 0
    while i < len(toks):
        flag = toks[i]
        if not flag.startswith("--"):
            raise SystemExit(f"ARGV PARITY: stray token {flag!r}")
        val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
        if flag in out:
            raise SystemExit(f"ARGV PARITY: duplicate flag {flag}")
        out[flag] = val
        i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("--")) else 1
    return out

ref, new = as_map(REF), as_map(tokens)
violations, allowed, explicit = [], [], []
for flag in sorted(set(ref) | set(new)):
    if flag in ref and flag in new:
        if ref[flag] != new[flag]:
            (allowed if flag in ALLOWED_DIFF else violations).append(
                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
    elif flag in new:
        if flag == "--ckpt-path":
            (allowed if mode == "RESTART" else violations).append(
                f"--ckpt-path: {new[flag]!r} (RESTART only)")
        elif flag in ALLOWED_ADD and ALLOWED_ADD[flag] == new[flag]:
            explicit.append(f"{flag}={new[flag]!r} (whitelisted explicit default)")
        else:
            violations.append(f"{flag}: added with {new[flag]!r}, not a whitelisted addition "
                              f"(expected {ALLOWED_ADD.get(flag, '<not allowed>')!r})")
    else:
        violations.append(f"{flag}: present in exp_07 ({ref[flag]!r}), MISSING here")

print("--- train.py argv ---")
print(" ".join(f"{k} {v!r}" if v == "" else f"{k} {v}" for k, v in new.items()))
print("--- argv parity vs exp_07 B-F ---")
for d in allowed:
    print(f"  allowed  {d}")
for d in explicit:
    print(f"  explicit {d}")
if violations:
    print("ARGV PARITY VIOLATIONS:")
    for v in violations:
        print(f"  !! {v}")
    raise SystemExit(2)
print(f"ARGV PARITY OK ({mode}): only whitelisted differences and additions")
PY
parity=$?
rm -f "$ARGV_FILE"
[ "$parity" -eq 0 ] || die "argv parity check FAILED - abort"
if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN complete: gates A–G passed for ARM=${ARM} RUNG=${RUNG} MODE=${MODE} SMOKE=${SMOKE}"
  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
  exit 0
fi

# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the pinned ${TIME_LIMIT} — submit via fa_orbit_submit.sh - abort"
echo "allocation matches the pins: ${GOT_CPUS} cpus, ${GOT_MEM_MB} MB, ${GOT_TIME}"

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq "$NGPU" ] || die "rung ${RUNG} needs exactly ${NGPU} visible GPU(s), got ${#GPU_ROWS[@]} - abort"
UUIDS=()
for ROW in "${GPU_ROWS[@]}"; do
  U="$(echo "$ROW" | cut -d, -f1 | tr -d ' ')"; N="$(echo "$ROW" | cut -d, -f2 | sed 's/^ *//')"
  case "$N" in *L40*) ;; *) die "allocated GPU '${N}' is not an L40 - abort";; esac
  UUIDS+=("$U"); echo "GPU: ${N} ${U}"
done
UUID_CSV="$(IFS=,; echo "${UUIDS[*]}")"

# --- I. environment identity gates (round-3 B6) -------------------------------
REAL_PY="$(command -v python)"
[ "$(readlink -f "$REAL_PY")" = "$(readlink -f "$PINNED_PYTHON")" ] || die "python resolves to ${REAL_PY}, pinned ${PINNED_PYTHON} - abort"
python3 - "$PINNED_PL_VERSION" "$PINNED_TORCH_VERSION" <<'PY' || die "runtime version gate FAILED - abort"
import sys
import pytorch_lightning as pl, torch
want_pl, want_torch = sys.argv[1], sys.argv[2]
bad = []
if pl.__version__ != want_pl:
    bad.append(f"pytorch_lightning {pl.__version__} != pinned {want_pl}")
if torch.__version__ != want_torch:
    bad.append(f"torch {torch.__version__} != pinned {want_torch}")
if bad:
    sys.exit("; ".join(bad))
print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
PY
VAE_SHA="$(sha256sum weights/FLAC/VAE.safetensors | awk '{print $1}')"
[ "$VAE_SHA" = "$PINNED_VAE_SHA256" ] || die "weights/FLAC/VAE.safetensors sha256 ${VAE_SHA} != pinned ${PINNED_VAE_SHA256} - abort"
echo "VAE pin OK: ${VAE_SHA}"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_VER="$(python3 -c 'import torch; print(torch.version.cuda)')"

# --- J. per-GPU free-VRAM gate (floor from the pins) --------------------------
for U in "${UUIDS[@]}"; do
  FREE="$(nvidia-smi --id="$U" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  [ -n "$FREE" ] || die "nvidia-smi free-mem query failed on ${U} - refusing to launch blind"
  [ "$FREE" -ge "$MIN_FREE_MB" ] || die "GPU ${U} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"
done
echo "--- co-tenancy disclosure at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

# --- K. exclusive run ownership via flock (round-3 B3 residual) --------------
# mkdir + stale recovery had two races: a contender could arrive between mkdir
# and the owner write, and release removed the directory without checking whose
# it was. flock has neither: the kernel holds the lock while the fd is open and
# releases it on close (including on kill), so there is no stale state to
# recover and no recovery path to get wrong.
mkdir -p "$OUTPUT_ROOT" || die "could not create ${OUTPUT_ROOT} - abort" 3
LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
exec 9>"$LOCKFILE" || die "could not open the lock file ${LOCKFILE} - abort" 3
if ! flock -n 9; then
  OWNER="$(tr '\n' ' ' < "$LOCKFILE" 2>/dev/null)"
  die "arm ${ARM} is locked by another live job (${OWNER:-<no metadata>}) - refusing a concurrent writer - abort"
fi
LAUNCH_UUID="$(cat /proc/sys/kernel/random/uuid)"
{ echo "job ${SLURM_JOB_ID}"; echo "uuid ${LAUNCH_UUID}"; echo "arm ${ARM}"; echo "mode ${MODE}"; echo "acquired ${TS}"; } >&9 \
  || die "could not write the lock owner metadata - abort" 3
echo "lock acquired: ${LOCKFILE} (flock on fd 9, released on exit)"
mkdir -p "$SAVEDIR" || die "could not create ${SAVEDIR} - abort" 3

# --- L. RESTART preflight (round-3 B2) ---------------------------------------
CKPT_SHA=""
if [ "$MODE" = "RESTART" ]; then
  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" --ckpt "$RESUME_CKPT" \
      --expected-step "$EXPECTED_STEP" --config "$MODEL_CONFIG_ABS" --max-steps "$MAXSTEPS" \
      --arm "$ARM" --rung "$RUNG" --commit "$HEAD_SHA" \
      ${LAUNCH_MANIFEST_LINK:+--launch-manifest "$LAUNCH_MANIFEST_LINK"} 2>&1)"
  echo "$PRE_OUT"
  echo "$PRE_OUT" | grep -q "^CKPT_SHA256 " || die "restart preflight FAILED - abort"
  CKPT_SHA="$(echo "$PRE_OUT" | awk '/^CKPT_SHA256 /{print $2}')"
fi

# --- M. wandb: scrub, pin the destination, fix the run id (round-3 B7) --------
unset WANDB_MODE WANDB_DISABLED WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME WANDB_DIR WANDB_PROJECT WANDB_NAME
export WANDB_DIR="$REPO/$SAVEDIR"
export WANDB_MODE=online
eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
WANDB_ENTITY_SEEN="$(python3 - <<'PY'
import sys
try:
    import wandb
    v = wandb.Api().viewer
except Exception as e:
    sys.exit(f"wandb identity check FAILED: {e}")
if v.email != "yh4742@princeton.edu":
    sys.exit(f"wandb identity {v.email} != yh4742@princeton.edu")
print(v.entity)
PY
)" || die "wandb identity gate FAILED (no logger fallback: the arms train with wandb) - abort"
[ -n "$WANDB_ENTITY_SEEN" ] || die "wandb returned an empty entity - abort"
# B7 residual: pin the destination account explicitly instead of leaving it implicit
export WANDB_ENTITY="$WANDB_ENTITY_SEEN"
echo "wandb identity OK: entity ${WANDB_ENTITY} (exported), mode ${WANDB_MODE}"
if [ "$MODE" = "RESTART" ]; then
  WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
  [ -n "$WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
  export WANDB_RUN_ID WANDB_RESUME=must
  echo "wandb: resuming run ${WANDB_RUN_ID} (resume=must)"
else
  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
  export WANDB_RUN_ID
  echo "wandb: new run id ${WANDB_RUN_ID}"
fi

# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1

# --- O. atomic manifest, duplicated to the save-dir (round-3 B5) --------------
TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
# B5 residual: a failed environment dump must not be silently hashed into the record
PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
rm -f "$PIPFREEZE_FILE"
{
  echo "# exp_11 arm launch manifest"
  echo "timestamp ${TS}"
  echo "job ${SLURM_JOB_ID} host $(hostname) mode ${MODE} launch_uuid ${LAUNCH_UUID}"
  echo "arm ${ARM} rung ${RUNG} micro ${MB} ngpu ${NGPU} max_steps ${MAXSTEPS} ckpt_every ${CHECKPOINT_EVERY}"
  echo "commit ${HEAD_SHA}"
  echo "p0_manifest_sha256 ${PINNED_P0_MANIFEST_SHA256}"
  echo "model_config ${MODEL_CONFIG_ABS}"
  echo "config_sha256 ${CONFIG_SHA}"
  echo "vae_sha256 ${VAE_SHA}"
  echo "env_pip_freeze_sha256 ${ENV_SHA}"
  echo "torch_version $(python3 -c 'import torch; print(torch.__version__)') cuda ${CUDA_VER} driver ${DRIVER}"
  echo "gpu_uuids ${UUID_CSV}"
  echo "time_limit ${TIME_LIMIT} min_free_mb ${MIN_FREE_MB}"
  echo "resume_ckpt ${RESUME_CKPT:-<none>} expected_step ${EXPECTED_STEP} resume_ckpt_sha256 ${CKPT_SHA:-<none>}"
  echo "save_dir ${SAVEDIR}"
  echo "train_log ${TRAINLOG}"
  echo "train_log_copy ${SAVEDIR_LOG}"
  echo "wandb_entity ${WANDB_ENTITY_SEEN} wandb_project ${NAME} wandb_name ${EXPNAME}"
  echo "wandb_run_id ${WANDB_RUN_ID}"
  echo "command torchrun --standalone --nnodes=1 --nproc_per_node=${NGPU} train.py ${ARGV[*]}"
} > "${MANIFEST}.tmp" || die "manifest write FAILED - abort" 3
mv "${MANIFEST}.tmp" "$MANIFEST" || die "manifest publication FAILED - abort" 3
cp "$MANIFEST" "${SAVEDIR}/$(basename "$MANIFEST")" || die "manifest copy to the save-dir FAILED - abort" 3
[ "$MODE" = "INITIAL" ] && { cp "$MANIFEST" "$LAUNCH_MANIFEST_LINK" || die "launch-manifest link write FAILED - abort" 3; }
echo "manifest: ${MANIFEST} (copied to ${SAVEDIR})"

# --- P. training: torchrun + FIFO tee (both statuses captured) ----------------
unset SLURM_NTASKS SLURM_JOB_NAME SLURM_PROCID SLURM_LOCALID SLURM_NODEID
: > "$TRAINLOG" || die "could not create ${TRAINLOG} - abort" 3
: > "$SAVEDIR_LOG" || die "could not create ${SAVEDIR_LOG} - abort" 3
FIFO="$(mktemp)" || die "mktemp failed - abort" 3
rm -f "$FIFO" && mkfifo "$FIFO" || die "mkfifo failed - abort" 3
trap 'rm -f "$FIFO"' EXIT
tee -a "$TRAINLOG" "$SAVEDIR_LOG" < "$FIFO" &
TEE_PID=$!
echo "=== launching ${ARM} ${RUNG}: torchrun --standalone --nproc_per_node=${NGPU} train.py (${MODE}) ==="
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" train.py "${ARGV[@]}" > "$FIFO" 2>&1 &
TR_PID=$!

# world-size watcher: terminate the moment Lightning reports the WRONG rank count
WORLD_RE="All distributed processes registered\. Starting with [0-9]+ processes"
(
  while kill -0 "$TR_PID" 2>/dev/null; do
    if grep -Eq "$WORLD_RE" "$TRAINLOG" 2>/dev/null; then
      GOT="$(grep -Eo "$WORLD_RE" "$TRAINLOG" | head -1 | grep -Eo '[0-9]+')"
      if [ "$GOT" != "$NGPU" ]; then
        echo "WORLD-SIZE WATCHER: Lightning started with ${GOT} processes, expected ${NGPU} — terminating torchrun"
        pkill -TERM -P "$TR_PID" 2>/dev/null; kill -TERM "$TR_PID" 2>/dev/null
      fi
      exit 0
    fi
    sleep 5
  done
) &
WATCHER=$!

wait "$TR_PID"; rc=$?
kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
wait "$TEE_PID"; tee_rc=$?
rm -f "$FIFO"

# --- Q. W&B run identity verification (round-3 B7 residual) ------------------
# The gate above proves WHO we are; this proves WHERE the run actually landed.
# wandb writes <WANDB_DIR>/wandb/run-*/files/wandb-metadata.json plus the run
# directory name; both are compared against the manifest, and a mismatch is a
# provenance failure (class 7), not a footnote.
WANDB_CHECK_RC=0
python3 - "$WANDB_DIR" "$WANDB_RUN_ID" "$WANDB_ENTITY" "$NAME" "$EXPNAME" <<'PY' || WANDB_CHECK_RC=$?
import glob, json, os, sys
wandb_dir, want_id, want_entity, want_project, want_name = sys.argv[1:6]
runs = sorted(glob.glob(os.path.join(wandb_dir, "wandb", f"run-*-{want_id}")))
if not runs:
    print(f"WANDB IDENTITY: no run directory for id {want_id} under {wandb_dir}/wandb")
    sys.exit(1)
meta_path = os.path.join(runs[-1], "files", "wandb-metadata.json")
meta = {}
if os.path.isfile(meta_path):
    try:
        meta = json.load(open(meta_path))
    except Exception as exc:
        print(f"WANDB IDENTITY: unreadable {meta_path}: {exc}")
        sys.exit(1)
bad = []
for key, want in (("entity", want_entity), ("project", want_project), ("name", want_name)):
    got = meta.get(key)
    if got is not None and got != want:
        bad.append(f"{key}={got!r} != manifest {want!r}")
if bad:
    print("WANDB IDENTITY MISMATCH: " + "; ".join(bad))
    sys.exit(1)
print(f"wandb run identity OK: id {want_id} at {runs[-1]} "
      f"(entity {want_entity}, project {want_project}, name {want_name})")
PY
if [ "$WANDB_CHECK_RC" -ne 0 ]; then
  echo "W&B run identity could not be verified against the manifest - provenance failure"
fi

# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
--------
CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
  --maxsteps "$MAXSTEPS" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
final_rc=$?
echo "$CLASSIFY_OUT"
LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"

# --- S. durable final record (round-3 B5 residual) ---------------------------
# The classifier verdict and this summary must survive in BOTH copies; the tee
# status is captured (it used to be discarded) and a failure is class 7. The
# whole preflight transcript (this job's Slurm output so far) is copied into the
# save-dir too, so the durable record is not just torchrun's stdout.
FINAL_RECORD="$(
  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
  echo "$CLASSIFY_OUT"
  echo "last checkpoint: ${LAST_CKPT:-<none>}"
  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
)"
printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
final_tee_rc="${PIPESTATUS[1]}"
SLURM_OUT="${EXPDIR}/slurm_train_${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID}.out"
PREFLIGHT_COPY_RC=0
if [ -f "$SLURM_OUT" ]; then
  cp "$SLURM_OUT" "${SAVEDIR}/preflight_${TS}_jid${SLURM_JOB_ID}.out" || PREFLIGHT_COPY_RC=$?
else
  echo "NOTE: ${SLURM_OUT} not found; preflight transcript not duplicated"
fi
if [ "$final_tee_rc" -ne 0 ] || [ "$PREFLIGHT_COPY_RC" -ne 0 ] || [ "$WANDB_CHECK_RC" -ne 0 ]; then
  echo "LOG-PROVENANCE: final record tee rc=${final_tee_rc}, preflight copy rc=${PREFLIGHT_COPY_RC}, wandb identity rc=${WANDB_CHECK_RC}"
  [ "$final_rc" -eq 0 ] && final_rc=7
fi
printf '%s\n' "$FINAL_RECORD"
exit "$final_rc"
