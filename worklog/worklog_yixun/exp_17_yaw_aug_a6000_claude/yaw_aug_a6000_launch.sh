#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_launch.sh — exp_17: Yaw-Aug FROM-SCRATCH training, 2×A6000
# DDP + SyncBN. Vanilla conditioning + training-time random yaw augmentation.
#
# Plan Rev 2 (worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/plan_yaw_aug_a6000.md).
# The arm is P1's config plus exactly one block (training.yaw_aug); P1@40k is the
# control, so ANY second difference silently makes this a two-factor experiment.
# Every gate below exists to make one specific silent failure impossible.
#
# MODE=SMOKE  — short rate/fit measurement. Its own W&B + output namespace, so it
#               can never write into or resume from the FULL namespace. Ends by
#               projecting the 40,000-step wall clock and REFUSING to bless the
#               run if the projection exceeds MAX_PROJECTED_HOURS (plan R3).
# MODE=FULL   — the registered 40,000-step endpoint. Step count and cadence are
#               PINNED, not defaults: a FULL run at any other budget is not the
#               registered experiment.
#
# Written by the main session seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
CONTROL_CFG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARM_CFG="${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
DATASET_CFG="src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE="weights/FLAC/VAE.safetensors"

MODE="${MODE:-}"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"          # micro-batch per GPU
ACC="${ACC:-1}"         # accumulation
SMOKE_STEPS="${SMOKE_STEPS:-25}"
MAX_PROJECTED_HOURS="${MAX_PROJECTED_HOURS:-55}"   # plan R3 abort threshold
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"
ENDPOINT_STEPS=40000    # PINNED — the registered endpoint
FULL_CADENCE=2500       # PINNED — the registered checkpoint cadence

TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- MODE is explicit; no step-count inference (exp_16 review finding) --------
case "$MODE" in
  SMOKE) NAME="FLAC_exp17_YAWAUG_smoke"; EXPNAME="exp17_YAWAUG_smoke"
         SAVEDIR="outputs_FLAC/exp17_YAWAUG_smoke"
         MAXSTEPS="$SMOKE_STEPS"; CADENCE="$SMOKE_STEPS" ;;
  FULL)  NAME="FLAC_exp17_YAWAUG";       EXPNAME="exp17_YAWAUG"
         SAVEDIR="outputs_FLAC/exp17_YAWAUG"
         MAXSTEPS="$ENDPOINT_STEPS";  CADENCE="$FULL_CADENCE" ;;
  *) echo "MODE must be exactly SMOKE or FULL (got '${MODE}') - abort"; exit 2 ;;
esac
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_${EXPNAME}_train.log"

# --- the rung is pinned by STRING equality --------------------------------- #
# Accumulation never feeds BatchNorm statistics (standing repo lesson), so
# BN=64 admits exactly one rung. Arithmetic like MB*2*ACC==64 is bypassable via
# bash integer overflow; literal comparison is not.
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || {
  echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 Yaw-Aug ${MODE} — ${TS} — $(git rev-parse --short HEAD 2>/dev/null) — ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} — logger=${LOGGER} ==="
echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) | tree clean: $(git diff --quiet && echo yes || echo NO)"
# Declare what this invocation intends BEFORE any gate runs: a run that aborts
# at a gate must still leave a record of which namespace and budget it was for.
echo "identity: name=${NAME} | experiment=${EXPNAME} | save-dir=${SAVEDIR}"
echo "budget: mode=${MODE} | endpoint=${MAXSTEPS} | cadence=${CADENCE}"

# --- FULL refuses to start into a namespace that already holds checkpoints --- #
if [ "$MODE" = "FULL" ] && [ -d "$SAVEDIR" ] && [ -n "$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | head -1)" ]; then
  echo "${SAVEDIR} already contains checkpoints - refusing to overwrite a run; move it aside or resume deliberately - abort"; exit 2
fi

# --- config contract: the arm is the control plus exactly one block --------- #
python - "$ARM_CFG" "$CONTROL_CFG" <<'PY' || { echo "CONFIG CONTRACT FAILED - abort"; exit 2; }
import json, sys
arm_p, ctl_p = sys.argv[1], sys.argv[2]
arm, ctl = json.load(open(arm_p)), json.load(open(ctl_p))

def strict(a, b, path="root"):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: key sets differ ({set(a) ^ set(b)})"
        for k in a:
            r = strict(a[k], b[k], f"{path}.{k}")
            if r: return r
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            r = strict(x, y, f"{path}[{i}]")
            if r: return r
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"

t = arm["training"]
block = t.get("yaw_aug")
if block != {"enabled": True, "img_w": 512, "seed": 42}:
    sys.exit(f"yaw_aug block is not the registered treatment: {block!r}")
if not isinstance(block["enabled"], bool) or isinstance(block["img_w"], bool) \
   or not isinstance(block["img_w"], int) or not isinstance(block["seed"], int):
    sys.exit(f"yaw_aug block has the wrong TYPES: {block!r}")
if "cond_method" in t:
    sys.exit("this arm must be vanilla-conditioned; cond_method is present")
if t.get("use_ema") is not True:
    sys.exit("use_ema must be true (matched to P1)")

vits = [c for c in arm["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if len(vits) != 2:
    sys.exit(f"expected 2 ViT conditioners, found {len(vits)}")
for c in vits:
    if c["config"].get("gradient_checkpointing") is not True:
        sys.exit(f"gradient_checkpointing must be true on {c['id']}")
widths = {c["config"]["ViT"]["img_w"] for c in vits}
if widths != {block["img_w"]}:
    sys.exit(f"yaw_aug.img_w={block['img_w']} but ViT widths are {widths} - the "
             "augmentation would rotate by the wrong angle, silently")

stripped = json.loads(json.dumps(arm)); stripped["training"].pop("yaw_aug")
d = strict(stripped, ctl)
if d:
    sys.exit(f"arm config is NOT the control plus one block - {d}")
print(f"config contract OK: {arm_p} == {ctl_p} + training.yaw_aug{block}")
print(f"chunk/aug plan: img_w={block['img_w']} seed={block['seed']} enabled={block['enabled']}")
PY

# --- data + VAE pins -------------------------------------------------------- #
[ -f "$DATASET_CFG" ] || { echo "dataset config missing: ${DATASET_CFG} - abort"; exit 2; }
[ -f "$VAE" ] || { echo "VAE checkpoint missing: ${VAE} - abort"; exit 2; }
echo "VAE sha256: $(sha256sum "$VAE" | cut -d' ' -f1)"
echo "dataset config sha256: $(sha256sum "$DATASET_CFG" | cut -d' ' -f1)"
echo "arm config sha256: $(sha256sum "$ARM_CFG" | cut -d' ' -f1)"

# --- disk floor ------------------------------------------------------------- #
FREE_DISK="$(df -Pm . | awk 'NR==2{print $4}')"
[ -n "$FREE_DISK" ] || { echo "df query failed - refusing to launch blind"; exit 2; }
[ "$FREE_DISK" -ge "$MIN_FREE_DISK_MB" ] || {
  echo "free disk ${FREE_DISK} MiB < required ${MIN_FREE_DISK_MB} MiB - abort"; exit 2; }

# --- per-GPU free-VRAM gate + co-tenancy disclosure (plan R4) ---------------- #
# Standing policy: co-tenancy is allowed with an explicit floor and disclosure;
# an occupied card is not by itself a reason to refuse. Fail-CLOSED on query error.
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - abort"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate ---------------------------------------- #
if [ "$LOGGER" = "wandb" ]; then
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
  [ $? -eq 0 ] || { echo "wandb identity != yh4742@princeton.edu - set the right key or use LOGGER=none - abort"; exit 2; }
fi

# --- DINOv3 pin (offline, so the gate cannot mutate the cache it validates) -- #
HF_HUB_OFFLINE=1 python "worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py" \
  || { echo "DINOv3/init-identity GATE FAILED - abort"; exit 1; }
echo "NOTE: assert_arm_configs.py validates the exp_07 configs' scheduler/init pins; the exp_17 arm contract was checked above."

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "mode=${MODE} | endpoint=${MAXSTEPS} | cadence=${CADENCE} | sync_batchnorm=true | strategy=ddp_find_unused_parameters_true"

START_EPOCH="$(date +%s)"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$ARM_CFG" \
  --dataset-config "$DATASET_CFG" \
  --pretransform-ckpt-path "$VAE" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --precision bf16-mixed \
  --logger "$LOGGER" --checkpoint-every "$CADENCE" \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
END_EPOCH="$(date +%s)"
echo "=== exp_17 Yaw-Aug ${MODE} exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- the augmentation must have been ACTIVE, in every mode ------------------ #
# A silently disabled treatment is the failure this whole experiment cannot
# survive, and it looks exactly like a successful run.
if grep -qiE "yaw[_ -]?aug.*(enabled|active)" "$LOG"; then
  echo "treatment banner: FOUND (augmentation was active)"
else
  echo "TREATMENT BANNER NOT FOUND in ${LOG} - the run may have trained WITHOUT augmentation - treat this run as invalid"
  [ "$rc" -eq 0 ] && rc=3
fi

# --- SMOKE: project the full wall clock and apply the plan R3 threshold ----- #
if [ "$MODE" = "SMOKE" ] && [ "$rc" -eq 0 ]; then
  ELAPSED=$(( END_EPOCH - START_EPOCH ))
  PROJ_H="$(python - "$ELAPSED" "$SMOKE_STEPS" "$ENDPOINT_STEPS" <<'PY'
import sys
elapsed, steps, endpoint = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(f"{elapsed/steps*endpoint/3600:.1f}")
PY
)"
  echo "SMOKE: ${SMOKE_STEPS} steps in ${ELAPSED}s (includes startup) -> projected ${PROJ_H} h for ${ENDPOINT_STEPS} steps"
  echo "NOTE: this projection INCLUDES process startup/compile, so it OVER-estimates; treat it as an upper bound."
  OVER="$(python -c "print(1 if float('${PROJ_H}') > float('${MAX_PROJECTED_HOURS}') else 0)")"
  if [ "$OVER" = "1" ]; then
    echo "SMOKE VERDICT: FAIL - projected ${PROJ_H} h exceeds MAX_PROJECTED_HOURS=${MAX_PROJECTED_HOURS} (plan R3). Do NOT launch FULL; report and re-plan."
    rc=4
  else
    echo "SMOKE VERDICT: PASS - projected ${PROJ_H} h <= ${MAX_PROJECTED_HOURS} h. FULL launch is within the registered budget."
  fi
fi

exit $rc
