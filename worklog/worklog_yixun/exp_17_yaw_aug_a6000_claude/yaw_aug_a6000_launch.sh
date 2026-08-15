#!/usr/bin/env bash
# ============================================================================
# yaw_aug_a6000_launch.sh — exp_17: Yaw-Aug FROM-SCRATCH training, 2×A6000
# DDP + SyncBN. Vanilla conditioning + training-time random yaw augmentation.
#
# Plan Rev 2 (worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/plan_yaw_aug_a6000.md).
# The arm is P1's config plus exactly THREE registered deltas:
#   1. training.yaw_aug            — the treatment
#   2/3. gradient_checkpointing     true -> false on both ViT conditioners
# Deltas 2/3 are a memory/time knob only: exp_07 measured ON-vs-OFF parameter
# gradients as bitwise identical (210 tensors, max abs diff 0.0), pinned by
# src/tests/test_vit_gradient_checkpointing.py. P1@40k stays the control.
# ANY fourth difference silently makes this a two-factor experiment.
#
# MODE=SMOKE  — short rate/fit measurement, in its OWN W&B + output namespace so
#               it can never write into or resume from the FULL namespace. Never
#               writes a checkpoint (cadence >> its endpoint, asserted after).
#               Ends by projecting the 40,000-step wall clock and REFUSING to
#               bless the run above MAX_PROJECTED_HOURS (plan R3).
# MODE=FULL   — the registered 40,000-step endpoint. Step count and cadence are
#               PINNED. FULL additionally REQUIRES a passing SMOKE on record.
# DRY_RUN=1   — run every gate, print the exact train.py argv, exit without
#               training. This is the boundary the guard suite asserts against:
#               it inspects the real argv, not a preflight paraphrase.
#
# Written by the main session seat (Claude Opus 5, max effort).
# Rev 2: applies the Codex r1 review (source pins, exact banner, R3 hardening,
#        no smoke checkpoints, real dry-run boundary).
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude"
CONTROL_CFG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARM_CFG="${EXPDIR}/FLAC_AR_YAWAUG_A6000.json"
DATASET_CFG="src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE="weights/FLAC/VAE.safetensors"

MODE="${MODE:-}"
DRY_RUN="${DRY_RUN:-0}"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"          # micro-batch per GPU
ACC="${ACC:-1}"         # accumulation
SMOKE_STEPS="${SMOKE_STEPS:-25}"
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"

# --- PINNED constants: not overridable from the environment ----------------- #
# A threshold you can raise from the shell is not a threshold. R3's whole job is
# to stop a run whose projection says it will not fit; making it an env var
# hands the abort decision to the same command line that wants to launch.
ENDPOINT_STEPS=40000          # the registered endpoint
FULL_CADENCE=2500             # the registered checkpoint cadence
SMOKE_CADENCE=1000000         # >> SMOKE_STEPS: smoke must never write a ckpt
MAX_PROJECTED_HOURS=55        # plan R3 abort threshold
BANNER="yaw_aug ENABLED img_w=512 seed=42"   # EXACT text of diffusion.py:407

# Content pins for the code that defines the treatment. A clean-tree check only
# says "nothing is modified"; these say "it is THIS code" even after a checkout,
# a rebase, or a stash pop.
PIN_yaw_rotation="bf8dd38f62dbd88461e9e215c9f639a57c6fefe673d1a9a4185df32ab5f848a1  src/data/yaw_rotation.py"
PIN_diffusion="ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb  src/training/diffusion.py"
PIN_factory="6967ec9fd800bb991d6f2ee2aee890bb73c093bfc5f676617f590f1dbd9d330f  src/training/factory.py"
PIN_train="bce1c94e648138459c056d82ac3e5f385e413b99f819b71bbbcd6d470d5f13ea  train.py"
PIN_dataset="71f11e80b9b09db754e2ce42f517480a9fb85977f27990f7f01cdbdebfc9b242  src/configs/dataset_configs/AR/train/acousticroom_train.json"
PIN_vae="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9  weights/FLAC/VAE.safetensors"
PIN_control="733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8  worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"

# --- MODE is explicit; no step-count inference (exp_16 review finding) -------- #
case "$MODE" in
  SMOKE) NAME="FLAC_exp17_YAWAUG_smoke"; EXPNAME="exp17_YAWAUG_smoke"
         SAVEDIR="outputs_FLAC/exp17_YAWAUG_smoke"
         MAXSTEPS="$SMOKE_STEPS"; CADENCE="$SMOKE_CADENCE" ;;
  FULL)  NAME="FLAC_exp17_YAWAUG";       EXPNAME="exp17_YAWAUG"
         SAVEDIR="outputs_FLAC/exp17_YAWAUG"
         MAXSTEPS="$ENDPOINT_STEPS";  CADENCE="$FULL_CADENCE" ;;
  *) echo "MODE must be exactly SMOKE or FULL (got '${MODE}') - abort"; exit 2 ;;
esac
LOG="${EXPDIR}/yaw_aug_a6000_${TS}_${EXPNAME}_train.log"

# --- the rung is pinned by STRING equality ---------------------------------- #
# Accumulation never feeds BatchNorm statistics (standing repo lesson), so
# BN=64 admits exactly one rung. Arithmetic like MB*2*ACC==64 is bypassable via
# bash integer overflow; literal comparison is not.
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || {
  echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_17 Yaw-Aug ${MODE} — ${TS} — $(git rev-parse --short HEAD 2>/dev/null) — ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} — logger=${LOGGER} dry_run=${DRY_RUN} ==="
# Declare what this invocation intends BEFORE any gate runs: a run that aborts
# at a gate must still leave a record of which namespace and budget it was for.
echo "identity: name=${NAME} | experiment=${EXPNAME} | save-dir=${SAVEDIR}"
echo "budget: mode=${MODE} | endpoint=${MAXSTEPS} | cadence=${CADENCE}"

# --- source pins: the reviewed code, byte for byte --------------------------- #
for P in "$PIN_yaw_rotation" "$PIN_diffusion" "$PIN_factory" "$PIN_train" \
         "$PIN_dataset" "$PIN_vae" "$PIN_control"; do
  echo "$P" | sha256sum -c --status - || {
    echo "SOURCE PIN FAILED for '${P##*  }' - the reviewed code/config/weights moved under this experiment - abort"; exit 2; }
done
echo "source pins OK (7 files match the reviewed revision)"

# A dirty tracked tree under the code paths means the pins above were computed
# from something that is not committed anywhere. Bookkeeping under EXPDIR is
# exempt: logs and notes are written by this very script.
DIRTY="$(git status --porcelain -- src train.py baselines 2>/dev/null | head -5)"
[ -z "$DIRTY" ] || {
  echo "tracked code tree is dirty - commit or stash before launching:"; echo "$DIRTY"; exit 2; }
echo "tree clean under src/ train.py baselines/ | branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

# --- FULL refuses to start into a namespace that already holds checkpoints ---- #
if [ "$MODE" = "FULL" ] && [ -d "$SAVEDIR" ] && [ -n "$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | head -1)" ]; then
  echo "${SAVEDIR} already contains checkpoints - refusing to overwrite a run; move it aside or resume deliberately - abort"; exit 2
fi

# --- config contract: the arm is the control plus exactly three deltas ------- #
# Deliberately BEFORE the FULL-only prerequisites below: a malformed treatment
# should be reported as such, not masked by "you have no smoke on record".
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
# Registered deltas 2/3: OFF here, ON in the control. Numerically inert
# (exp_07 bitwise-identity evidence); asserted so it stays a DECISION.
for c in vits:
    if c["config"].get("gradient_checkpointing") is not False:
        sys.exit(f"gradient_checkpointing must be false on {c['id']} (registered delta)")
widths = {c["config"]["ViT"]["img_w"] for c in vits}
if widths != {block["img_w"]}:
    sys.exit(f"yaw_aug.img_w={block['img_w']} but ViT widths are {widths} - the "
             "augmentation would rotate by the wrong angle, silently")

# Undo all three registered deltas; what remains must be the control exactly.
stripped = json.loads(json.dumps(arm)); stripped["training"].pop("yaw_aug")
for c in stripped["model"]["conditioning"]["configs"]:
    if c["type"] == "ViTCoordinates":
        c["config"]["gradient_checkpointing"] = True
d = strict(stripped, ctl)
if d:
    sys.exit(f"arm config is NOT the control plus the three registered deltas - {d}")
print(f"config contract OK: {arm_p} == {ctl_p} + training.yaw_aug{block} + grad-ckpt off x2")
# NOTE: deliberately does NOT echo the words of the treatment banner. The
# post-run banner check greps this same log, and a preflight paraphrase that
# matched it would make that check self-satisfying (Codex r1 finding).
print(f"preflight treatment plan: width={block['img_w']} rng-seed={block['seed']} on={block['enabled']}")
PY

# --- FULL requires a passing SMOKE on record (plan R3 is a prerequisite) ----- #
# Without this, R3 is advisory: nothing stops a FULL launch that never measured
# its own rate. The evidence must be a smoke log that BOTH saw the exact banner
# AND recorded a PASS verdict.
if [ "$MODE" = "FULL" ]; then
  SMOKE_EVIDENCE=""
  for F in $(ls -t "${EXPDIR}"/*_exp17_YAWAUG_smoke_train.log 2>/dev/null); do
    if tr '\r' '\n' < "$F" | grep -qxF "$BANNER" && grep -qF "SMOKE VERDICT: PASS" "$F"; then
      SMOKE_EVIDENCE="$F"; break
    fi
  done
  [ -n "$SMOKE_EVIDENCE" ] || {
    echo "FULL requires a SMOKE log in ${EXPDIR} containing the exact treatment banner AND 'SMOKE VERDICT: PASS' - none found - run MODE=SMOKE first - abort"; exit 2; }
  echo "smoke evidence: ${SMOKE_EVIDENCE}"
fi

# --- disk floor -------------------------------------------------------------- #
FREE_DISK="$(df -Pm . | awk 'NR==2{print $4}')"
[ -n "$FREE_DISK" ] || { echo "df query failed - refusing to launch blind"; exit 2; }
[ "$FREE_DISK" -ge "$MIN_FREE_DISK_MB" ] || {
  echo "free disk ${FREE_DISK} MiB < required ${MIN_FREE_DISK_MB} MiB - abort"; exit 2; }

# --- per-GPU free-VRAM gate + co-tenancy disclosure (plan R4) ----------------- #
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

# --- fail-closed wandb identity gate ----------------------------------------- #
if [ "$LOGGER" = "wandb" ] && [ "$DRY_RUN" != "1" ]; then
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

# --- DINOv3 pin (offline, so the gate cannot mutate the cache it validates) --- #
HF_HUB_OFFLINE=1 python "worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py" \
  || { echo "DINOv3/init-identity GATE FAILED - abort"; exit 1; }
echo "NOTE: assert_arm_configs.py validates the exp_07 configs' scheduler/init pins; the exp_17 arm contract was checked above."

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "mode=${MODE} | endpoint=${MAXSTEPS} | cadence=${CADENCE} | sync_batchnorm=true | strategy=ddp_find_unused_parameters_true"

# --- the exact argv, as one line, before it is used -------------------------- #
# The guard suite asserts against THIS array, so a wrong --max-steps or
# --save-dir cannot hide behind an agreeing preflight message.
ARGV=(python train.py
  --model-config "$ARM_CFG"
  --dataset-config "$DATASET_CFG"
  --pretransform-ckpt-path "$VAE"
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --precision bf16-mixed
  --logger "$LOGGER" --checkpoint-every "$CADENCE"
  --name "$NAME" --experiment-name "$EXPNAME"
  --save-dir "$SAVEDIR")
echo "ARGV: ${ARGV[*]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; train.py NOT launched"; exit 0
fi

START_EPOCH="$(date +%s)"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 "${ARGV[@]}"
rc=$?
END_EPOCH="$(date +%s)"
echo "=== exp_17 Yaw-Aug ${MODE} exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- the augmentation must have been ACTIVE, in every mode ------------------- #
# A silently disabled treatment is the failure this whole experiment cannot
# survive, and it looks exactly like a successful run. Matched as a WHOLE LINE
# against the literal print at src/training/diffusion.py:407 — a substring grep
# would also match this script's own preflight output in the same log.
if tr '\r' '\n' < "$LOG" | grep -qxF "$BANNER"; then
  echo "treatment banner: FOUND (exact match: '${BANNER}')"
else
  echo "TREATMENT BANNER '${BANNER}' NOT FOUND in ${LOG} - the run may have trained WITHOUT augmentation - treat this run as invalid"
  [ "$rc" -eq 0 ] && rc=3
fi

# --- SMOKE must not have written a checkpoint -------------------------------- #
if [ "$MODE" = "SMOKE" ]; then
  NCKPT="$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | wc -l)"
  if [ "$NCKPT" -ne 0 ]; then
    echo "SMOKE wrote ${NCKPT} checkpoint(s) under ${SAVEDIR} - cadence pin failed; a smoke artifact could be mistaken for a run checkpoint"
    [ "$rc" -eq 0 ] && rc=5
  else
    echo "smoke checkpoints: 0 (cadence ${SMOKE_CADENCE} >> endpoint ${SMOKE_STEPS}, as pinned)"
  fi
fi

# --- SMOKE: project the full wall clock and apply the plan R3 threshold ------ #
if [ "$MODE" = "SMOKE" ] && [ "$rc" -eq 0 ]; then
  if tr '\r' '\n' < "$LOG" | grep -qiE '\b(nan|inf)\b'; then
    echo "SMOKE VERDICT: FAIL - the log contains nan/inf; the fit is not healthy regardless of rate"
    rc=4
  else
  ELAPSED=$(( END_EPOCH - START_EPOCH ))
  PROJ_H="$(python - "$ELAPSED" "$SMOKE_STEPS" "$ENDPOINT_STEPS" <<'PY'
import sys
elapsed, steps, endpoint = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(f"{elapsed/steps*endpoint/3600:.1f}")
PY
)"
  echo "SMOKE: ${SMOKE_STEPS} steps in ${ELAPSED}s (includes startup) -> projected ${PROJ_H} h for ${ENDPOINT_STEPS} steps"
  echo "NOTE: this projection INCLUDES process startup/compile, so it OVER-estimates. It is an UPPER bound, which is the safe direction: a PASS here cannot be an under-estimate."
  OVER="$(python -c "print(1 if float('${PROJ_H}') > float('${MAX_PROJECTED_HOURS}') else 0)")"
  if [ "$OVER" = "1" ]; then
    echo "SMOKE VERDICT: FAIL - projected ${PROJ_H} h exceeds MAX_PROJECTED_HOURS=${MAX_PROJECTED_HOURS} (plan R3). Do NOT launch FULL; report and re-plan."
    rc=4
  else
    echo "SMOKE VERDICT: PASS - projected ${PROJ_H} h <= ${MAX_PROJECTED_HOURS} h. FULL launch is within the registered budget."
  fi
  fi
fi

exit $rc
