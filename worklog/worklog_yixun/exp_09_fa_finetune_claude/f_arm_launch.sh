#!/usr/bin/env bash
# ============================================================================
# f_arm_launch.sh - exp_09: fa fine-tune from the 87.5k full-parity anchor.
#
# MODEL_CONFIG-parameterized mirror of exp_07's reviewed p1_ddp_launch.sh.
# Three run identities, one script (plan_fa_finetune.md §1/§4):
#
#   MODEL_CONFIG=FLAC_AR_BF.json                 -> F-warm : FLAC_exp09_Fw / exp09_Fw
#   MODEL_CONFIG=FLAC_AR_BF.json   OPT_RESET=1   -> F-reset: FLAC_exp09_Fr / exp09_Fr
#   MODEL_CONFIG=FLAC_AR_BVp1.json               -> V ctrl : FLAC_exp09_V  / exp09_V
#
# (codex r1 BLOCKER: warm and reset previously shared one W&B/checkpoint
# namespace - train.py derives the ckpt dir from --name/--experiment-name.)
# Both configs live in exp_07's folder (the anchor's own lineage); nothing else
# is accepted (fail-closed, exit 2). OPT_RESET on the V arm is rejected: it is
# outside the approved treatment scope (codex r1 HIGH).
#
# exp_09 NEVER trains from scratch: RESUME_CKPT is REQUIRED, is lineage-checked
# (global_step, optimizer-state fullness, scheduler/EMA presence) and MAXSTEPS
# must exceed the anchor's 87,500 steps. OPT_RESET=1 writes a state-stripped
# COPY of RESUME_CKPT into the arm's save-dir and resumes from that; the anchor
# is SHA-256-verified unchanged across the operation.
#
# Recipe is byte-identical to exp_07 P1: 32/GPU x 2 GPUs x accum 1 (eff 64),
# SyncBN (BN=64), ViT grad-ckpt via config, env flac, seed 42, wandb.
# CHECKPOINT_EVERY defaults to 2500 but MUST be lowered for the 625/1,250-step
# probe screens (PL only saves when global_step % cadence == 0).
#
# Usage (F-reset probe with screens at 88,125 / 88,750):
#   MODEL_CONFIG=FLAC_AR_BF.json OPT_RESET=1 CHECKPOINT_EVERY=625 MAXSTEPS=88750 \
#   RESUME_CKPT=outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt \
#   bash worklog/worklog_yixun/exp_09_fa_finetune_claude/f_arm_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"      # configs + arm-audit gate live here
EXPDIR09="worklog/worklog_yixun/exp_09_fa_finetune_claude"   # logs land here
ANCHOR_STEPS=87500                                           # the full-parity anchor's step
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"

_bool() { # $1=name $2=value -> must be 0 or 1
  case "$2" in 0|1) return 0;; *) echo "$1 must be 0 or 1 (got '$2') - abort"; return 1;; esac
}
_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

# --- environment gate (codex r1 HIGH: the optimizer-restore semantics are
# --- PL-version dependent, and plain `python` must not resolve to another env) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- knob validation ---
OPT_RESET="${OPT_RESET:-0}";                  _bool OPT_RESET "$OPT_RESET" || exit 2
OPT_RESET_FORCE="${OPT_RESET_FORCE:-0}";      _bool OPT_RESET_FORCE "$OPT_RESET_FORCE" || exit 2
RESET_LINEAGE="${RESET_LINEAGE:-0}";          _bool RESET_LINEAGE "$RESET_LINEAGE" || exit 2
EXPECTED_STEP_SET=0; [ -n "${EXPECTED_STEP+x}" ] && EXPECTED_STEP_SET=1
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2

# --- arm/variant selection (fail-closed allow-list; no path traversal, no other config) ---
MODEL_CONFIG="${MODEL_CONFIG:-}"
case "$MODEL_CONFIG" in
  FLAC_AR_BF.json)   ARM="F";;
  FLAC_AR_BVp1.json) ARM="V";;
  *) echo "MODEL_CONFIG must be exactly 'FLAC_AR_BF.json' or 'FLAC_AR_BVp1.json' (got '${MODEL_CONFIG}') - abort"; exit 2;;
esac
if [ "$ARM" = "V" ]; then
  # codex r1 HIGH: the continued-vanilla control is a WARM continuation by definition.
  [ "$OPT_RESET" = "0" ] || { echo "OPT_RESET=1 is not an approved treatment for the V control arm (plan §1: V = continued vanilla, warm) - abort"; exit 2; }
  [ "$RESET_LINEAGE" = "0" ] || { echo "RESET_LINEAGE=1 is meaningless on the V arm - abort"; exit 2; }
  VARIANT="V"
elif [ "$OPT_RESET" = "1" ]; then
  [ "$RESET_LINEAGE" = "0" ] || { echo "RESET_LINEAGE=1 with OPT_RESET=1 is contradictory (OPT_RESET already selects the reset lineage) - abort"; exit 2; }
  VARIANT="Fr"                       # F-reset: strip now, then resume from the copy
elif [ "$RESET_LINEAGE" = "1" ]; then
  VARIANT="Fr"                       # F-reset continuation/restart: no strip, reset identity
else
  VARIANT="Fw"                       # F-warm
fi
NAME="FLAC_exp09_${VARIANT}"; EXPNAME="exp09_${VARIANT}"; SAVEDIR="outputs_FLAC/exp09_${VARIANT}"
MODEL_CONFIG_PATH="${EXPDIR}/${MODEL_CONFIG}"
[ -f "$MODEL_CONFIG_PATH" ] || { echo "model config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }

# --- full-state resume is MANDATORY for exp_09 (fine-tune only, never scratch) ---
RESUME_CKPT="${RESUME_CKPT:-}"
[ -n "$RESUME_CKPT" ] || { echo "RESUME_CKPT is REQUIRED (exp_09 never trains from scratch) - abort"; exit 2; }
[ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }

# --- training budget is MANDATORY and must extend past the anchor ---
MAXSTEPS="${MAXSTEPS:-}"
[ -n "$MAXSTEPS" ] || { echo "MAXSTEPS is REQUIRED (integer > ${ANCHOR_STEPS}) - abort"; exit 2; }
_posint MAXSTEPS "$MAXSTEPS" || exit 2
[ "$MAXSTEPS" -gt "$ANCHOR_STEPS" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the anchor's ${ANCHOR_STEPS} steps - abort"; exit 2; }

# EXPECTED_STEP: the global_step the resume ckpt must carry. Defaults to the anchor's
# 87,500 (fail-closed lineage pin); set explicitly only when restarting a crashed exp_09
# run from one of its own later checkpoints. exp_09 never resumes a PRE-anchor step, so
# values below 87,500 are rejected outright (reverify knob assessment). Always logged.
EXPECTED_STEP="${EXPECTED_STEP:-$ANCHOR_STEPS}"; _posint EXPECTED_STEP "$EXPECTED_STEP" || exit 2
[ "$EXPECTED_STEP" -ge "$ANCHOR_STEPS" ] || { echo "EXPECTED_STEP ${EXPECTED_STEP} is pre-anchor: exp_09 only ever resumes at or after ${ANCHOR_STEPS} - abort"; exit 2; }
[ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }

# --- RESET_LINEAGE provenance gate (reverify REMAINING: the label was self-attested, so
# --- RESET_LINEAGE=1 with a warm anchor could launch as Fr WITHOUT resetting Adam).
# --- Two independent, non-attestable conditions are now required:
# ---   (a) PATH provenance - the resume ckpt must physically live inside the Fr save-dir,
# ---       which only ever contains the stripped copy and Fr's own checkpoints;
# ---   (b) STEP provenance - an Fr restart is by definition past the anchor, so
# ---       EXPECTED_STEP must be given explicitly and be strictly > the anchor step.
# --- (The OPT_RESET=1 and warm paths are untouched and keep the ==87,500 default.)
if [ "$RESET_LINEAGE" = "1" ]; then
  FR_DIR="$(realpath -m "outputs_FLAC/exp09_Fr")"
  RESUME_REAL="$(realpath -m "$RESUME_CKPT")"
  case "$RESUME_REAL" in
    "${FR_DIR}"/*) ;;
    *) echo "RESET_LINEAGE=1 requires RESUME_CKPT to live inside the F-reset namespace ${FR_DIR}/"
       echo "  got: ${RESUME_REAL}"
       echo "  -> a warm anchor / Fw checkpoint is NOT an F-reset lineage; use OPT_RESET=1 to create the reset lineage. abort"
       exit 2;;
  esac
  [ "$EXPECTED_STEP_SET" = "1" ] || { echo "RESET_LINEAGE=1 requires EXPECTED_STEP to be set explicitly (an Fr restart resumes past the anchor) - abort"; exit 2; }
  [ "$EXPECTED_STEP" -gt "$ANCHOR_STEPS" ] || { echo "RESET_LINEAGE=1 requires EXPECTED_STEP > ${ANCHOR_STEPS} (got ${EXPECTED_STEP}); the un-advanced stripped copy is resumed with OPT_RESET=1, not RESET_LINEAGE=1 - abort"; exit 2; }
fi

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR09}/fa_finetune_${TS}_${EXPNAME}_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_09 variant ${VARIANT} (arm ${ARM}, ${MODEL_CONFIG}) DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER}"
echo "resume: '${RESUME_CKPT}' | expected_step=${EXPECTED_STEP} | opt_reset=${OPT_RESET} | reset_lineage=${RESET_LINEAGE}"

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

# --- RESUME_CKPT lineage pre-flight (codex r1 MEDIUM: existence was the only check).
# --- One torch.load of the resume ckpt; asserts the step, the optimizer-state fullness
# --- against the declared variant, and scheduler/EMA presence. ---
RESUME_CKPT="$RESUME_CKPT" EXPECTED_STEP="$EXPECTED_STEP" VARIANT="$VARIANT" \
OPT_RESET="$OPT_RESET" python3 - <<'PY' || { echo "resume-lineage check FAILED - abort"; exit 2; }
import os, sys, torch
p = os.environ["RESUME_CKPT"]; want = int(os.environ["EXPECTED_STEP"])
variant = os.environ["VARIANT"]; opt_reset = os.environ["OPT_RESET"] == "1"
ck = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ck, dict):
    sys.exit(f"not a Lightning checkpoint: {p}")
gs = ck.get("global_step")
if gs != want:
    sys.exit(f"global_step {gs} != expected {want} (set EXPECTED_STEP to resume a different step)")
if "optimizer_states" not in ck:
    sys.exit("no 'optimizer_states' key -> weights-only ckpt; PL 2.1 KeyErrors on resume")
opts = ck["optimizer_states"]
if len(opts) != 1:
    sys.exit(f"expected exactly 1 optimizer entry, found {len(opts)}")
n_state = len(opts[0].get("state", {}))
n_groups = len(opts[0].get("param_groups", []))
if n_groups != 1:
    sys.exit(f"expected exactly 1 param_group, found {n_groups}")
lr = opts[0]["param_groups"][0].get("lr")
if not ck.get("lr_schedulers"):
    sys.exit("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
sd = ck.get("state_dict") or {}
if not any(k.startswith("diffusion_ema.") for k in sd):
    sys.exit("no EMA weights in state_dict")
kind = "FULL/warm" if n_state else "CLEARED/stripped"
print(f"resume lineage: {p}\n  global_step={gs} epoch={ck.get('epoch')} optimizer_state={kind} "
      f"({n_state} entries) lr={lr} sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} "
      f"ema_entries={sum(1 for k in sd if k.startswith('diffusion_ema.'))}")
if opt_reset and not n_state:
    sys.exit("OPT_RESET=1 but the resume ckpt ALREADY has cleared optimizer state: strip from the "
             "warm anchor instead, or reuse the existing stripped copy with OPT_RESET=0 RESET_LINEAGE=1")
if variant == "Fw" and not n_state:
    sys.exit("F-warm declared but the resume ckpt has CLEARED optimizer state (that is a reset-lineage "
             "file): the F-reset arm is launched with OPT_RESET=1 from the warm anchor, never as F-warm")
if variant == "V" and not n_state:
    sys.exit("V control declared but the resume ckpt has CLEARED optimizer state - abort")
print(f"lineage OK for variant {variant}")
PY

# --- optional Adam-state reset (F-reset variant): strip a COPY, resume from it ---
if [ "$OPT_RESET" = "1" ]; then
  STEP_TAG="$(basename "$RESUME_CKPT" | sed -n 's/.*step=\([0-9]\+\).*/\1/p')"
  [ -n "$STEP_TAG" ] || STEP_TAG="anchor"
  RESET_CKPT="${SAVEDIR}/optreset_from_${STEP_TAG}.ckpt"
  mkdir -p "$SAVEDIR" || { echo "cannot create ${SAVEDIR} - abort"; exit 2; }
  if [ -e "$RESET_CKPT" ] && [ "$OPT_RESET_FORCE" != "1" ]; then
    echo "stripped copy already exists: ${RESET_CKPT} - refusing to overwrite (a live run may be resuming from it)."
    echo "  -> pass OPT_RESET_FORCE=1 to regenerate it (the strip is deterministic). RESET_LINEAGE=1 is NOT a reuse route:"
    echo "     it requires EXPECTED_STEP > ${ANCHOR_STEPS}, i.e. an Fr checkpoint that has already advanced. abort"
    exit 2
  fi
  echo "--- OPT_RESET: writing state-stripped copy (anchor untouched) ---"
  ANCHOR_SHA_BEFORE="$(sha256sum "$RESUME_CKPT" | awk '{print $1}')"
  echo "anchor sha256 (before): ${ANCHOR_SHA_BEFORE}"
  if [ "$OPT_RESET_FORCE" = "1" ]; then
    python -m src.tools.strip_optimizer_state --in "$RESUME_CKPT" --out "$RESET_CKPT" --force \
      || { echo "strip_optimizer_state FAILED - abort"; exit 2; }
  else
    python -m src.tools.strip_optimizer_state --in "$RESUME_CKPT" --out "$RESET_CKPT" \
      || { echo "strip_optimizer_state FAILED - abort"; exit 2; }
  fi
  ANCHOR_SHA_AFTER="$(sha256sum "$RESUME_CKPT" | awk '{print $1}')"
  echo "anchor sha256 (after) : ${ANCHOR_SHA_AFTER}"
  [ "$ANCHOR_SHA_BEFORE" = "$ANCHOR_SHA_AFTER" ] || { echo "ANCHOR WAS MODIFIED (sha256 mismatch) - abort"; exit 2; }
  # post-condition gate on the copy: entry kept, state cleared, param_groups identical
  RESET_CKPT="$RESET_CKPT" ANCHOR_CKPT="$RESUME_CKPT" python3 - <<'PY' || { echo "stripped-ckpt verification FAILED - abort"; exit 2; }
import os, sys, torch
p, a = os.environ["RESET_CKPT"], os.environ["ANCHOR_CKPT"]
if os.path.realpath(p) == os.path.realpath(a):
    sys.exit("stripped copy resolves to the anchor!")
c = torch.load(p, map_location="cpu", weights_only=False)
if "optimizer_states" not in c:
    sys.exit("optimizer_states absent -> PL 2.1 KeyErrors on resume")
if len(c["optimizer_states"]) != 1:
    sys.exit(f"expected the optimizer entry to be RETAINED, found {len(c['optimizer_states'])} entries")
e = c["optimizer_states"][0]
if e["state"] != {}:
    sys.exit(f"optimizer state not cleared ({len(e['state'])} entries remain)")
if len(e["param_groups"]) != 1:
    sys.exit("param_groups were not preserved")
b = torch.load(a, map_location="cpu", weights_only=False)
if e["param_groups"] != b["optimizer_states"][0]["param_groups"]:
    sys.exit("param_groups differ from the anchor's -> the scheduled LR would change")
if not c.get("lr_schedulers"):
    sys.exit("lr_schedulers missing -> PL 2.1 KeyErrors on resume")
if not c.get("global_step"):
    sys.exit("global_step missing")
if not any(k.startswith("diffusion_ema.") for k in c["state_dict"]):
    sys.exit("EMA weights missing")
if not b["optimizer_states"][0]["state"]:
    sys.exit("ANCHOR WAS MUTATED (its optimizer state is gone)")
print(f"stripped ckpt OK: global_step={c['global_step']} epoch={c.get('epoch')} "
      f"adam_state=CLEARED param_groups=KEPT lr={e['param_groups'][0].get('lr')} "
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
echo "variant: ${VARIANT} | model-config: ${MODEL_CONFIG_PATH} | name: ${NAME} | experiment: ${EXPNAME} | save-dir: ${SAVEDIR}"

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$MODEL_CONFIG_PATH" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --ckpt-path "$RESUME_CKPT" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every "$CHECKPOINT_EVERY" \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
echo "=== exp_09 variant ${VARIANT} exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
