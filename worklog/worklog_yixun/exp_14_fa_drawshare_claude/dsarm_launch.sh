#!/usr/bin/env bash
# ============================================================================
# dsarm_launch.sh - exp_14 (fa_drawshare): FROM-SCRATCH training of ONE
# draw-sharing arm, 2-GPU DDP + SyncBN.
#
# Seats this round: main session claude-fable-5 (xhigh) per the model-change
# flag; code by the Opus 5 Coder seat; codex MCP reviews (SOP §Roles).
#
# THE TREATMENT IS THE CHUNK PLAN, AND IT IS A CONFIG KEY.
# src/data/yaw_rotation.py derives the orbit partition as
#     angles_per_chunk = max(1, cap // per_rank_micro_batch)
# and under train-mode DINOv3 the angles inside one chunk SHARE a single random
# RoPE rescale draw (announcement 06). exp_14 therefore declares the cap per arm
# as `training.frame_avg_max_fwd_samples`, which train.py embeds into every
# checkpoint, instead of inheriting the module constant:
#
#   ARM=DSPA   cap 32  -> 32 // 32 = 1 angle  per chunk -> per-angle draws
#                         (= exp_07 / exp_10's B-F, the July path)
#   ARM=DSCS3  cap 96  -> 96 // 32 = 3 angles per chunk -> all three non-identity
#                         angles share one draw (= exp_11's C4L TOPOLOGY, not its
#                         configuration: micro-8/cap-64 issued a 24-row GEMM,
#                         micro-32/cap-96 issues a 96-row one)
#
# Everything else is pinned to the exp_07 B-F recipe and asserted below as a
# parsed-object identity: the arm config must be FLAC_AR_BF.json PLUS EXACTLY
# that one key. 32/GPU x 2 GPUs x accum 1 (eff 64), SyncBN (BN=64), ViT
# grad-ckpt via config, seed 42, bf16-mixed (ini default), ckpt/2500, wandb,
# env flac, 40,000 steps, from scratch.
#
# Based on the reviewed exp_07 bf_scratch_launch.sh (from-scratch; f_arm_launch.sh
# is resume-REQUIRED and wrong for this) with exp_13's gate ordering.
#
# RESUME (crash restart only - exp_14 arms are scratch runs):
#   RESUME_CKPT + EXPECTED_STEP>0 both required, together. train.py rebuilds the
#   wrapper from the CURRENT json (train.py:160) BEFORE PL loads the checkpoint
#   (:230), so a changed cap would take effect silently on resume. The gate below
#   therefore compares the checkpoint's EMBEDDED model_config against this arm's
#   current JSON as parsed objects and aborts on ANY mismatch, and the checkpoint
#   must live inside this arm's own save-dir.
#
# PROBE mode: a SHORT window, MAXSTEPS - EXPECTED_STEP <= 15 (so MAXSTEPS <= 15
# from scratch, and MAXSTEPS <= 40015 when restarting a finished 40,000-step arm).
# It makes the post-run embedded-cap gate MANDATORY and requires CHECKPOINT_EVERY
# to be small enough that PL actually saves inside the window. That gate - reload
# the newest checkpoint, assert its EMBEDDED cap is this arm's cap - also runs
# after any ordinary successful run; it is only in probe mode that a MISSING
# checkpoint is an abort rather than a warning.
#
# NOT the fit probe. Plan §3.2 requires a real 15-step DDP fit + VRAM measurement
# before the cap-96 arm is committed to; the VRAM floor here is the standing
# co-tenancy policy number (21,900 MiB), which was measured at cap 64 and has NOT
# been requalified for cap 96. A loud NOTE is printed for DSCS3.
#
# Knobs (env): ARM MAXSTEPS CHECKPOINT_EVERY EXPECTED_STEP RESUME_CKPT LOGGER
#              MB ACC MIN_FREE_MB MIN_FREE_DISK_MB
#
# Usage (arm 1, full 40k budget):
#   ARM=DSPA bash worklog/worklog_yixun/exp_14_fa_drawshare_claude/dsarm_launch.sh
#
# Usage (probe: the embedded-cap read-back is the point of it):
#   ARM=DSCS3 MAXSTEPS=15 CHECKPOINT_EVERY=5 \
#     bash worklog/worklog_yixun/exp_14_fa_drawshare_claude/dsarm_launch.sh
#
# Usage (restart after a crash, from this arm's own 20k checkpoint):
#   ARM=DSPA EXPECTED_STEP=20000 \
#   RESUME_CKPT=outputs_FLAC/exp14_DSPA/FLAC_exp14_DSPA/exp14_DSPA/checkpoints/epoch=4-step=20000.ckpt \
#     bash worklog/worklog_yixun/exp_14_fa_drawshare_claude/dsarm_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR07="worklog/worklog_yixun/exp_07_fa_scratch_claude"       # BF reference + pin gate
EXPDIR14="worklog/worklog_yixun/exp_14_fa_drawshare_claude"     # arm configs, logs
BF_PATH="${EXPDIR07}/FLAC_AR_BF.json"                           # the single-delta base
CAP_KEY="frame_avg_max_fwd_samples"
ORBIT_ANGLES=4                                                  # C4: 1 identity + 3 rotated
PROBE_WINDOW=15                                                 # MAXSTEPS-EXPECTED_STEP <= this => probe
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"

_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

# --- arm selection (fail-closed: there is no default arm; picking one silently
# --- would be the single easiest way to spend six days on the wrong treatment) ---
ARM="${ARM:-}"
case "$ARM" in
  DSPA)  MODEL_CONFIG_PATH="${EXPDIR14}/FLAC_AR_BF_DSPA.json";  WANT_CAP=32
         NAME="FLAC_exp14_DSPA";  EXPNAME="exp14_DSPA";  SAVEDIR="outputs_FLAC/exp14_DSPA";;
  DSCS3) MODEL_CONFIG_PATH="${EXPDIR14}/FLAC_AR_BF_DSCS3.json"; WANT_CAP=96
         NAME="FLAC_exp14_DSCS3"; EXPNAME="exp14_DSCS3"; SAVEDIR="outputs_FLAC/exp14_DSCS3";;
  *) echo "ARM must be exactly one of DSPA (cap 32, per-angle draws) or DSCS3 (cap 96, 3/3 shared)"
     echo "  got ARM='${ARM}' - abort"; exit 2;;
esac

# --- environment gate (as exp_13/exp_10: plain `python` must not resolve to
# --- another env, and the PL version decides checkpoint/restore semantics) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- knob validation ---
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2
MAXSTEPS="${MAXSTEPS:-40000}";                _posint MAXSTEPS "$MAXSTEPS" || exit 2

# invariant (exp_07 review): accumulation never feeds BN statistics, so the BN=64
# mandate leaves exactly ONE legal rung - pinned literally (string equality;
# arithmetic like MB*2*ACC==64 is bypassable via bash integer overflow).
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

[ -f "$MODEL_CONFIG_PATH" ] || { echo "arm config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }
[ -f "$BF_PATH" ]           || { echo "BF reference config not found: ${BF_PATH} - abort"; exit 2; }

# --- resume mode (exp_14 arms train from SCRATCH; a resume is a crash restart and
# --- must declare BOTH the file and the step it claims to be at) ---
RESUME_CKPT="${RESUME_CKPT:-}"
EXPECTED_STEP="${EXPECTED_STEP:-0}"
case "$EXPECTED_STEP" in ''|*[!0-9]*) echo "EXPECTED_STEP must be a non-negative integer (got '${EXPECTED_STEP}') - abort"; exit 2;; esac
MODE="SCRATCH"
if [ -n "$RESUME_CKPT" ] || [ "$EXPECTED_STEP" -gt 0 ]; then
  MODE="RESTART"
  [ -n "$RESUME_CKPT" ] || { echo "EXPECTED_STEP ${EXPECTED_STEP} declares a RESTART but RESUME_CKPT is unset - abort"; exit 2; }
  [ "$EXPECTED_STEP" -gt 0 ] || { echo "RESUME_CKPT was given but EXPECTED_STEP is 0/unset: a resume must state the step it claims to be at - abort"; exit 2; }
  [ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
  case "$RESUME_REAL" in
    "${SAVEDIR_REAL}"/*) ;;
    *) echo "a ${ARM} RESTART may only resume a checkpoint written by THIS arm, i.e. one inside"
       echo "  ${SAVEDIR_REAL}/"
       echo "  got: ${RESUME_REAL}"
       echo "  -> the two arms have different chunk plans; crossing them would train a third method. abort"
       exit 2;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }
fi

# --- probe-mode cadence sanity: the post-run embedded-cap gate needs a checkpoint ---
PROBE=0; [ $(( MAXSTEPS - EXPECTED_STEP )) -le "$PROBE_WINDOW" ] && PROBE=1
if [ "$PROBE" -eq 1 ]; then
  FIRST_SAVE=$(( (EXPECTED_STEP / CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY ))
  [ "$FIRST_SAVE" -le "$MAXSTEPS" ] || {
    echo "PROBE mode (${MAXSTEPS} - ${EXPECTED_STEP} <= ${PROBE_WINDOW} steps) but CHECKPOINT_EVERY=${CHECKPOINT_EVERY} would not save"
    echo "  before ${MAXSTEPS} (first save at global_step ${FIRST_SAVE}); the post-run embedded-cap gate reads"
    echo "  the probe's final checkpoint, so it would have nothing to read. Lower CHECKPOINT_EVERY. abort"
    exit 2; }
fi

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR14}/fa_drawshare_${TS}_${EXPNAME}_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_14 fa_drawshare ${ARM} (${MODE}) DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER} | probe=${PROBE}"
echo "arm config: ${MODEL_CONFIG_PATH} (single delta vs ${BF_PATH}: training.${CAP_KEY}=${WANT_CAP})"
echo "resume: '${RESUME_CKPT}' | expected_step=${EXPECTED_STEP} | mode=${MODE}"

# --- config contract (fail-closed, parsed objects, cheap so it runs FIRST):
# --- (1) the arm config IS FLAC_AR_BF.json plus EXACTLY the one cap key;
# --- (2) the declared chunk plan is echoed per announcement 06;
# --- (3) the factory really parses the key into the wrapper kwarg. ---
dsarm_config_gate() {
  ARM_CFG="$MODEL_CONFIG_PATH" BF_CFG="$BF_PATH" WANT_CAP="$WANT_CAP" MB="$MB" \
  ORBIT_ANGLES="$ORBIT_ANGLES" ARM_LABEL="$ARM" python3 - <<'PY'
import copy, json, os, sys
sys.path.insert(0, os.getcwd())          # beat any stale pip-installed src copy
from src.training.factory import _parse_frame_avg_cap_config

arm_p, bf_p = os.environ["ARM_CFG"], os.environ["BF_CFG"]
want_cap = int(os.environ["WANT_CAP"]); micro = int(os.environ["MB"])
n_angles = int(os.environ["ORBIT_ANGLES"]); label = os.environ["ARM_LABEL"]
KEY = "frame_avg_max_fwd_samples"

arm, bf = json.load(open(arm_p)), json.load(open(bf_p))

# (1) POSITIVE: the declared cap is this arm's cap, and nothing else moved.
got = arm["training"].get(KEY)
if got != want_cap or isinstance(got, bool):
    sys.exit(f"{arm_p}: training.{KEY} is {got!r}, expected {want_cap!r}")
stripped = copy.deepcopy(arm)
del stripped["training"][KEY]
if stripped != bf:
    sys.exit(f"{arm_p} differs from {bf_p} somewhere OTHER than training.{KEY} "
             "(parsed-object mismatch) -> this is no longer a single-delta arm")

# (2) the rest of the pinned arm identity (announcement 05: the eval flag must
#     match how the checkpoint was trained, so the training method is asserted).
if arm["training"].get("cond_method") != "fa_invariant":
    sys.exit(f"{arm_p}: cond_method {arm['training'].get('cond_method')!r} != 'fa_invariant'")
angles = arm["training"].get("frame_avg_angles")
if angles != [0.0, 90.0, 180.0, 270.0]:
    sys.exit(f"{arm_p}: frame_avg_angles {angles!r} != C4 [0.0, 90.0, 180.0, 270.0]")
if len(angles) != n_angles:
    sys.exit(f"{arm_p}: orbit size {len(angles)} != {n_angles}")
if arm["training"].get("use_ema") is not True:
    sys.exit(f"{arm_p}: use_ema must be true")
n_gc = sum(1 for c in arm["model"]["conditioning"]["configs"]
           if c.get("type") == "ViTCoordinates" and c["config"].get("gradient_checkpointing") is True)
if n_gc != 2:
    sys.exit(f"{arm_p}: expected 2 grad-checkpointed ViT conditioners, found {n_gc}")

# (3) the key really reaches the wrapper kwarg (the config path, not a re-read).
kwargs = _parse_frame_avg_cap_config(arm["training"])
if kwargs != {KEY: want_cap}:
    sys.exit(f"factory parse produced {kwargs!r}, expected {{{KEY!r}: {want_cap}}}")

# (4) THE CHUNK PLAN, declared per announcement 06 rule 1.
rotated = len(angles) - 1                       # C4 chunks the non-identity angles
per_chunk = max(1, want_cap // micro)
n_chunks = -(-rotated // per_chunk)             # ceil
chunk_sizes = [min(per_chunk, rotated - i * per_chunk) for i in range(n_chunks)]
shared = max(chunk_sizes)
print(f"config contract OK ({label}): {arm_p} == {bf_p} + training.{KEY}={want_cap}")
print(f"  CHUNK PLAN (announcement 06): cap={want_cap} per-rank micro-batch={micro} "
      f"orbit={len(angles)} angles ({rotated} rotated)")
print(f"    angles_per_chunk = max(1, {want_cap} // {micro}) = {per_chunk}")
print(f"    rotated-angle chunks = {chunk_sizes} ({n_chunks} conditioner forward(s) "
      f"of {[c * micro for c in chunk_sizes]} samples)")
print(f"    shared-angle count = {shared}/{rotated}  "
      f"({'per-angle draws (the July path)' if shared == 1 else f'{shared} angles share one RoPE draw'})")
PY
}
dsarm_config_gate || { echo "config contract FAILED - abort"; exit 2; }

# --- disk floor on the outputs volume (each 2,500-step checkpoint is ~690 MB;
# --- 16 of them per arm). Bypass for guard-testing only: MIN_FREE_DISK_MB=1. ---
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"; _posint MIN_FREE_DISK_MB "$MIN_FREE_DISK_MB" || exit 2
DF_TARGET="outputs_FLAC"; [ -d "$DF_TARGET" ] || DF_TARGET="."
DISK_FREE_MB="$(df -P -B1M "$DF_TARGET" 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')"
[ -n "$DISK_FREE_MB" ] || { echo "df query failed on ${DF_TARGET} - refusing to launch blind - abort"; exit 2; }
echo "disk: ${DISK_FREE_MB} MiB free on the volume holding ${DF_TARGET} (floor ${MIN_FREE_DISK_MB} MiB)"
[ "$DISK_FREE_MB" -ge "$MIN_FREE_DISK_MB" ] || { echo "free disk ${DISK_FREE_MB} MiB < required ${MIN_FREE_DISK_MB} MiB - refusing to launch"; exit 2; }

# --- RESTART only: the resume-safety gate the plan requires (re-review 4).
# --- train.py:160 rebuilds the wrapper from the CURRENT json BEFORE PL loads the
# --- checkpoint at :230, so a cap edited between the crash and the restart would
# --- silently take effect. The embedded model_config must therefore equal this
# --- arm's current JSON as parsed objects - ANY mismatch aborts. ---
if [ "$MODE" = "RESTART" ]; then
  RESUME_CKPT="$RESUME_CKPT" EXPECTED_STEP="$EXPECTED_STEP" ARM_CFG="$MODEL_CONFIG_PATH" \
  WANT_CAP="$WANT_CAP" python3 - <<'PY' || { echo "resume-lineage check FAILED - abort"; exit 2; }
import json, os, sys, torch
p = os.environ["RESUME_CKPT"]; want = int(os.environ["EXPECTED_STEP"])
cfg_path = os.environ["ARM_CFG"]; want_cap = int(os.environ["WANT_CAP"])
KEY = "frame_avg_max_fwd_samples"
ck = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ck, dict):
    sys.exit(f"not a Lightning checkpoint: {p}")
gs = ck.get("global_step")
if gs != want:
    sys.exit(f"global_step {gs} != expected {want} (set EXPECTED_STEP to resume a different step)")
mc = ck.get("model_config")
if not isinstance(mc, dict):
    sys.exit("checkpoint carries no embedded 'model_config' dict -> cannot prove which chunk plan "
             "trained it; refusing to resume it under an assumed one")
want_cfg = json.load(open(cfg_path))
if mc != want_cfg:
    embedded_cap = (mc.get("training") or {}).get(KEY, "<absent>")
    sys.exit(f"embedded model_config != {cfg_path} (parsed-object mismatch). Embedded "
             f"training.{KEY}={embedded_cap!r}, current={want_cap!r}. train.py rebuilds the "
             "wrapper from the CURRENT json before loading the checkpoint, so resuming under a "
             "different config would silently change the treatment mid-run")
if "optimizer_states" not in ck:
    sys.exit("no 'optimizer_states' key -> weights-only ckpt; PL 2.1 KeyErrors on resume")
sd = ck.get("state_dict") or {}
if not any(k.startswith("diffusion_ema.") for k in sd):
    sys.exit("no EMA weights in state_dict (the arm config has use_ema true)")
print(f"resume lineage OK: {p}\n  global_step={gs} epoch={ck.get('epoch')} "
      f"embedded model_config == {cfg_path} (training.{KEY}={want_cap})")
PY
fi

# --- per-GPU FREE-VRAM gate (Yixun co-tenancy policy; the exp_07 M1-measured
# --- checkpointed rank 15,712 MiB + DDP/SyncBN ~1,000 + safety ~5,200). ---
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
if [ "$ARM" = "DSCS3" ]; then
  cat <<'NOTE'
################################################################################
## NOTE - the VRAM floor below (21,900 MiB) was measured at cap 64. DSCS3 puts ##
## 96 samples in ONE ViT chunk (~1.5x the activation footprint of 64), so the  ##
## floor is NOT requalified for this arm. Plan §3.2 gates DSCS3 on a REAL      ##
## 15-step DDP fit probe with VRAM sampled; if cap 96 does not fit, STOP and   ##
## report - do not silently fall back to cap 64, which would no longer test    ##
## exp_11's topology.                                                          ##
################################################################################
NOTE
fi
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  # ~/.bashrc's interactive guard blocks non-interactive sourcing, so extract the
  # newest exported key directly; the gate below re-verifies at every launch.
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
  python - <<'PY'
import sys
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY
  gate=$?
  [ "$gate" -eq 0 ] || { echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - set the right WANDB_API_KEY or use LOGGER=none - abort"; exit 2; }
fi

# --- DINOv3 pin + arm init-identity (offline, fail-closed) ---
HF_HUB_OFFLINE=1 python "${EXPDIR07}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

cat <<'NOTE'
################################################################################
## NOTE - READ THE LINES ABOVE CAREFULLY.                                     ##
## assert_arm_configs.py is an exp_07 artifact. It builds FLAC_AR_BV.json and  ##
## FLAC_AR_BF.json, and its printed schedule assertion                        ##
##     "InverseLR(1e6,0.5,0.99)"                                              ##
## is about THOSE exp_07 configs - not about the exp_14 arm launched here.    ##
## It is kept for its DINOv3 initializer PIN + seeded init-identity check,    ##
## which do apply: each exp_14 arm is FLAC_AR_BF.json plus one TRAINING key,  ##
## so the module tree and its initialization are B-F's exactly.               ##
## The authoritative exp_14 assertion is the config contract re-run below.    ##
################################################################################
NOTE
dsarm_config_gate || { echo "config contract FAILED - abort"; exit 2; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true (fail-closed in train.py below num_gpus 2) | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | grad-ckpt: config"
echo "arm: ${ARM} | mode: ${MODE} | model-config: ${MODEL_CONFIG_PATH} | cap: ${WANT_CAP}"

RESUME_ARGS=()
[ "$MODE" = "RESTART" ] && RESUME_ARGS=(--ckpt-path "$RESUME_CKPT")

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$MODEL_CONFIG_PATH" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  "${RESUME_ARGS[@]}" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every "$CHECKPOINT_EVERY" \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
echo "=== exp_14 ${ARM} (${MODE}) exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="

# ============================================================================
# Embedded-cap gate. The treatment is only real if it is IN THE ARTIFACT:
# train.py's ModelConfigEmbedderCallback writes model_config into every
# checkpoint, so the newest checkpoint is read back and its embedded cap compared
# against this arm's. An analytic assertion cannot see a threading bug; this
# reads what the run actually saved. Mandatory in PROBE mode (where a missing
# checkpoint is itself the failure); after a long run it is the closing audit.
# ============================================================================
if [ "$rc" -eq 0 ]; then
  echo "--- embedded-cap gate (probe=${PROBE}) ---"
  SAVEDIR="$SAVEDIR" WANT_CAP="$WANT_CAP" ARM_CFG="$MODEL_CONFIG_PATH" \
  EXPECTED_STEP="$EXPECTED_STEP" MAXSTEPS="$MAXSTEPS" ARM_LABEL="$ARM" PROBE="$PROBE" \
  python3 - <<'PY' || { echo "embedded-cap gate FAILED - abort"; exit 2; }
import glob, json, os, sys, torch
savedir = os.environ["SAVEDIR"]; want_cap = int(os.environ["WANT_CAP"])
start = int(os.environ["EXPECTED_STEP"]); maxsteps = int(os.environ["MAXSTEPS"])
label = os.environ["ARM_LABEL"]; cfg_path = os.environ["ARM_CFG"]
probe = os.environ["PROBE"] == "1"
KEY = "frame_avg_max_fwd_samples"

cands = glob.glob(os.path.join(savedir, "**", "*.ckpt"), recursive=True)
if not cands:
    msg = (f"no checkpoint under {savedir} -> the embedded-cap gate has nothing to read "
           "(lower CHECKPOINT_EVERY so PL saves inside the window)")
    if probe:
        sys.exit(msg)
    print("WARNING: " + msg)
    raise SystemExit(0)
p = max(cands, key=os.path.getmtime)
ck = torch.load(p, map_location="cpu", weights_only=False)
gs = ck.get("global_step")
mc = ck.get("model_config")
print(f"probe checkpoint: {p}\n  global_step={gs}")
fails = []
if not (start < gs <= maxsteps):
    fails.append(f"global_step {gs} outside the run window ({start}, {maxsteps}]")
if not isinstance(mc, dict):
    fails.append("checkpoint carries no embedded 'model_config' dict")
else:
    got = (mc.get("training") or {}).get(KEY, "<absent>")
    print(f"  embedded training.{KEY} = {got!r}  (arm {label} declares {want_cap})")
    if got != want_cap or isinstance(got, bool):
        fails.append(f"embedded training.{KEY} {got!r} != {want_cap!r} -> the treatment did NOT "
                     "reach the artifact")
    if mc != json.load(open(cfg_path)):
        fails.append(f"embedded model_config != {cfg_path} (parsed-object mismatch)")
if fails:
    sys.exit("PROBE embedded-cap gate FAILED:\n  - " + "\n  - ".join(fails))
print(f"PROBE embedded-cap gate PASSED: the saved checkpoint declares "
      f"training.{KEY}={want_cap} for arm {label}.")
PY
  echo "PROBE embedded-cap gate OK"
fi

exit $rc
