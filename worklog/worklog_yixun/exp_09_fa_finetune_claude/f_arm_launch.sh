#!/usr/bin/env bash
# ============================================================================
# f_arm_launch.sh - exp_09: fa fine-tune from the 87.5k full-parity anchor.
#
# MODEL_CONFIG-parameterized mirror of exp_07's reviewed p1_ddp_launch.sh. Two
# arms, one script (plan_fa_finetune.md §1/§4):
#   MODEL_CONFIG=FLAC_AR_BF.json   -> F arm (fa_invariant fine-tune)
#   MODEL_CONFIG=FLAC_AR_BVp1.json -> V arm (continued-vanilla control)
# Both configs live in exp_07's folder (the anchor's own lineage); nothing else
# is accepted (fail-closed, exit 2).
#
# exp_09 NEVER trains from scratch: RESUME_CKPT is REQUIRED and MAXSTEPS must
# exceed the anchor's 87,500 steps. OPT_RESET=1 selects the F-reset variant -
# a state-stripped COPY of RESUME_CKPT is written into the arm's save-dir and
# resumed from instead; the anchor file itself is never modified.
#
# Recipe is byte-identical to exp_07 P1: 32/GPU x 2 GPUs x accum 1 (eff 64),
# SyncBN (BN=64), ViT grad-ckpt via config, env flac, seed 42, ckpt/2500, wandb.
#
# Usage:
#   MODEL_CONFIG=FLAC_AR_BF.json \
#   RESUME_CKPT=outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt \
#   MAXSTEPS=88750 [OPT_RESET=1] bash worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"      # configs + arm-audit gate live here
EXPDIR09="worklog/worklog_yixun/exp_09_fa_finetune_claude"   # logs land here
ANCHOR_STEPS=87500
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"

# --- arm selection (fail-closed allow-list; no path traversal, no other config) ---
MODEL_CONFIG="${MODEL_CONFIG:-}"
case "$MODEL_CONFIG" in
  FLAC_AR_BF.json)   ARM="F"; NAME="FLAC_exp09_F"; EXPNAME="exp09_F"; SAVEDIR="outputs_FLAC/exp09_F";;
  FLAC_AR_BVp1.json) ARM="V"; NAME="FLAC_exp09_V"; EXPNAME="exp09_V"; SAVEDIR="outputs_FLAC/exp09_V";;
  *) echo "MODEL_CONFIG must be exactly 'FLAC_AR_BF.json' or 'FLAC_AR_BVp1.json' (got '${MODEL_CONFIG}') - abort"; exit 2;;
esac
MODEL_CONFIG_PATH="${EXPDIR}/${MODEL_CONFIG}"
[ -f "$MODEL_CONFIG_PATH" ] || { echo "model config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }

# --- full-state resume is MANDATORY for exp_09 (fine-tune only, never scratch) ---
RESUME_CKPT="${RESUME_CKPT:-}"
[ -n "$RESUME_CKPT" ] || { echo "RESUME_CKPT is REQUIRED (exp_09 never trains from scratch) - abort"; exit 2; }
[ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }

# --- training budget is MANDATORY and must extend past the anchor ---
MAXSTEPS="${MAXSTEPS:-}"
[ -n "$MAXSTEPS" ] || { echo "MAXSTEPS is REQUIRED (integer > ${ANCHOR_STEPS}) - abort"; exit 2; }
case "$MAXSTEPS" in ''|*[!0-9]*) echo "MAXSTEPS must be a positive integer (got '${MAXSTEPS}') - abort"; exit 2;; esac
[ "$MAXSTEPS" -gt "$ANCHOR_STEPS" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the anchor's ${ANCHOR_STEPS} steps - abort"; exit 2; }

OPT_RESET="${OPT_RESET:-0}"
case "$OPT_RESET" in 0|1) ;; *) echo "OPT_RESET must be 0 or 1 (got '${OPT_RESET}') - abort"; exit 2;; esac

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR09}/fa_finetune_${TS}_${EXPNAME}$([ "$OPT_RESET" = "1" ] && echo "_optreset")_train.log"

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_09 arm ${ARM} (${MODEL_CONFIG}) DDP+SyncBN+ckpt - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} - logger=${LOGGER} - resume='${RESUME_CKPT}' - opt_reset=${OPT_RESET} ==="

# --- config-contract pre-flight (fail-closed; VERBATIM from p1_ddp_launch.sh - it
# --- validates the BV/BVp1/BF triangle, i.e. both arms, whichever one launches) ---
python3 - <<'PY' || { echo "config-contract FAILED - abort"; exit 2; }
import json, copy
base = "worklog/worklog_yixun/exp_07_fa_scratch_claude/"
bv  = json.load(open(base + "FLAC_AR_BV.json"))
bvp = json.load(open(base + "FLAC_AR_BVp1.json"))
bf  = json.load(open(base + "FLAC_AR_BF.json"))
exp = copy.deepcopy(bv)
n = 0
for c in exp["model"]["conditioning"]["configs"]:
    if c.get("id") in ("source_vit", "context_poses_vit"):
        c["config"]["gradient_checkpointing"] = True; n += 1
assert n == 2, f"expected exactly 2 ViT conditioner blocks, found {n}"
assert bvp == exp, "BVp1 != BV + the 2 gradient_checkpointing keys (parsed-object mismatch)"
exp2 = copy.deepcopy(bvp)
exp2["training"]["cond_method"] = "fa_invariant"
exp2["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
assert bf == exp2, "BF != BVp1 + cond_method/frame_avg_angles (parsed-object mismatch)"
print("config contract OK (exact parsed-object assertions)")
PY

# --- optional Adam-state reset (F-reset variant): strip a COPY, resume from it ---
if [ "$OPT_RESET" = "1" ]; then
  [ "$ARM" = "F" ] || echo "NOTE: OPT_RESET=1 on the ${ARM} arm - the plan's F-reset variant is an F-arm treatment; proceeding as explicitly requested."
  STEP_TAG="$(basename "$RESUME_CKPT" | sed -n 's/.*step=\([0-9]\+\).*/\1/p')"
  [ -n "$STEP_TAG" ] || STEP_TAG="anchor"
  RESET_CKPT="${SAVEDIR}/optreset_from_${STEP_TAG}.ckpt"
  mkdir -p "$SAVEDIR" || { echo "cannot create ${SAVEDIR} - abort"; exit 2; }
  if [ -e "$RESET_CKPT" ] && [ "${OPT_RESET_FORCE:-0}" != "1" ]; then
    echo "stripped copy already exists: ${RESET_CKPT} - refusing to overwrite (a live run may be resuming from it)."
    echo "  -> pass OPT_RESET_FORCE=1 to regenerate, or set RESUME_CKPT=${RESET_CKPT} with OPT_RESET=0 to reuse it. abort"
    exit 2
  fi
  echo "--- OPT_RESET: writing state-stripped copy (anchor untouched) ---"
  python -m src.tools.strip_optimizer_state --in "$RESUME_CKPT" --out "$RESET_CKPT" \
    $([ "${OPT_RESET_FORCE:-0}" = "1" ] && echo --force) \
    || { echo "strip_optimizer_state FAILED - abort"; exit 2; }
  # post-condition gate: empty (not absent) optimizer_states, resume position intact
  RESET_CKPT="$RESET_CKPT" ANCHOR_CKPT="$RESUME_CKPT" python3 - <<'PY' || { echo "stripped-ckpt verification FAILED - abort"; exit 2; }
import os, sys, torch
p, a = os.environ["RESET_CKPT"], os.environ["ANCHOR_CKPT"]
assert os.path.realpath(p) != os.path.realpath(a), "stripped copy resolves to the anchor!"
c = torch.load(p, map_location="cpu", weights_only=False)
assert "optimizer_states" in c, "optimizer_states absent -> PL 2.1 KeyErrors on resume"
assert c["optimizer_states"] == [], f"optimizer_states not empty: {len(c['optimizer_states'])}"
assert c.get("lr_schedulers"), "lr_schedulers missing -> PL 2.1 KeyErrors on resume"
assert c.get("global_step"), "global_step missing"
assert any(k.startswith("diffusion_ema.") for k in c["state_dict"]), "EMA weights missing"
# the anchor must be byte-untouched
b = torch.load(a, map_location="cpu", weights_only=False)
assert b["optimizer_states"] and b["optimizer_states"][0]["state"], "ANCHOR WAS MUTATED"
print(f"stripped ckpt OK: global_step={c['global_step']} epoch={c.get('epoch')} "
      f"sched_last_epoch={c['lr_schedulers'][0].get('last_epoch')} | anchor intact")
PY
  RESUME_CKPT="$RESET_CKPT"
  echo "resuming from stripped copy: ${RESUME_CKPT}"
fi

# --- per-GPU FREE-VRAM gate (co-tenancy policy; M1-measured rank ~15.9 GB) ---
MIN_FREE_MB="${MIN_FREE_MB:-21900}"
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate ---
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
  gate=$?
  [ "$gate" -eq 0 ] || { echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - abort"; exit 2; }
fi

# --- DINOv3 pin + arm init-identity (offline, fail-closed) ---
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | grad-ckpt: config"
echo "arm: ${ARM} | model-config: ${MODEL_CONFIG_PATH} | name: ${NAME} | experiment: ${EXPNAME} | save-dir: ${SAVEDIR}"

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$MODEL_CONFIG_PATH" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --ckpt-path "$RESUME_CKPT" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
echo "=== exp_09 arm ${ARM} exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
