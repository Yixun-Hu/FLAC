#!/usr/bin/env bash
# ============================================================================
# bf_resume_launch.sh - exp_10: resume B-F (fa_invariant, from scratch) from the
# exp_07 40,000-step futility-stop checkpoint under the correct fa protocol.
#
# Single arm, single identity (plan_fa_scratch_resume.md §3):
#   model config : worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json
#   identity     : FLAC_exp10_BF / exp10_BF / outputs_FLAC/exp10_BF
#   budget       : --max-steps 67500 (the exp_07 matched budget), ckpt every 2500
#
# Derived from the reviewed exp_09 f_arm_launch.sh. The exp_09-specific machinery
# is DELETED, not disabled: no MODEL_CONFIG selection/allow-list (the config is
# hardcoded), no OPT_RESET/OPT_RESET_FORCE/RESET_LINEAGE, no optimizer stripping,
# no Fw/Fr/V identities, no >=87,500 step floor. exp_10 is a plain WARM
# continuation of one lineage.
#
# RESUME_CKPT is REQUIRED (exp_10 never trains from scratch) and is admitted by
# exactly TWO lineage modes (codex plan review, Blocker 2):
#
#   (a) INITIAL  - EXPECTED_STEP unset or == 40000. RESUME_CKPT must resolve to
#       THE exact B-F 40k anchor path, its sha256 must equal the pin in
#       bf40k_anchor.sha256 (read at runtime; missing pin file => abort), and the
#       checkpoint's EMBEDDED model_config must be the fa_invariant B-F config
#       (cond_method == 'fa_invariant', and equal to FLAC_AR_BF.json as parsed
#       objects) at global_step 40000.
#   (b) RESTART  - EXPECTED_STEP > 40000, set explicitly. RESUME_CKPT must live
#       INSIDE outputs_FLAC/exp10_BF/ (only this run writes there), carry
#       global_step == EXPECTED_STEP, the same embedded fa config, and MAXSTEPS
#       must exceed EXPECTED_STEP. No sha pin (the file is produced by this run).
#
# Recipe is byte-identical to exp_07 B-F/P1: 32/GPU x 2 GPUs x accum 1 (eff 64),
# SyncBN (BN=64), ViT grad-ckpt via config, env flac, seed 42, wandb.
# CHECKPOINT_EVERY defaults to 2500; lower it for short probe screens (PL only
# saves when global_step % cadence == 0).
#
# NOT bit-exact across the resume boundary: Lightning restores no RNG state and
# no dataloader position, and the 249 unsaved exp_07 steps past 40,000 are
# intentionally discarded (see bf_stop_record_and_p1_amendment.md).
#
# Usage (initial launch, full 67.5k budget):
#   RESUME_CKPT=outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt \
#   bash worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch.sh
#
# Usage (restart after a crash, from exp_10's own 47.5k checkpoint):
#   EXPECTED_STEP=47500 MAXSTEPS=67500 \
#   RESUME_CKPT=outputs_FLAC/exp10_BF/FLAC_exp10_BF/exp10_BF/checkpoints/epoch=10-step=47500.ckpt \
#   bash worklog/worklog_yixun/exp_10_fa_scratch_resume_claude/bf_resume_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_07_fa_scratch_claude"        # configs + arm-audit gate live here
EXPDIR10="worklog/worklog_yixun/exp_10_fa_scratch_resume_claude" # logs + sha pin land here
ANCHOR_STEPS=40000                                             # the B-F futility-stop step
ANCHOR_CKPT="outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt"
SHA_PIN_FILE="${EXPDIR10}/bf40k_anchor.sha256"
MODEL_CONFIG="FLAC_AR_BF.json"                                 # hardcoded: exp_10 has ONE arm
MODEL_CONFIG_PATH="${EXPDIR}/${MODEL_CONFIG}"
NAME="FLAC_exp10_BF"; EXPNAME="exp10_BF"; SAVEDIR="outputs_FLAC/exp10_BF"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"

_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

# --- environment gate (codex r1 HIGH on exp_09: the optimizer-restore semantics are
# --- PL-version dependent, and plain `python` must not resolve to another env) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- knob validation ---
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2
MAXSTEPS="${MAXSTEPS:-67500}";                _posint MAXSTEPS "$MAXSTEPS" || exit 2
EXPECTED_STEP_SET=0; [ -n "${EXPECTED_STEP:-}" ] && EXPECTED_STEP_SET=1
EXPECTED_STEP="${EXPECTED_STEP:-$ANCHOR_STEPS}"; _posint EXPECTED_STEP "$EXPECTED_STEP" || exit 2

[ -f "$MODEL_CONFIG_PATH" ] || { echo "model config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }

# --- full-state resume is MANDATORY for exp_10 (continuation only, never scratch) ---
RESUME_CKPT="${RESUME_CKPT:-}"
[ -n "$RESUME_CKPT" ] || { echo "RESUME_CKPT is REQUIRED (exp_10 never trains from scratch) - abort"; exit 2; }
[ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }
RESUME_REAL="$(realpath -m "$RESUME_CKPT")"

# --- lineage mode (fail-closed; there are exactly two admissible resume stories) ---
if [ "$EXPECTED_STEP" -lt "$ANCHOR_STEPS" ]; then
  echo "EXPECTED_STEP ${EXPECTED_STEP} is below the B-F anchor step ${ANCHOR_STEPS}: exp_10 resumes the 40k anchor"
  echo "  (EXPECTED_STEP unset or ${ANCHOR_STEPS}) or one of its OWN later checkpoints (EXPECTED_STEP > ${ANCHOR_STEPS}) - abort"
  exit 2
elif [ "$EXPECTED_STEP" -eq "$ANCHOR_STEPS" ]; then
  MODE="INITIAL"
  ANCHOR_REAL="$(realpath -m "$ANCHOR_CKPT")"
  [ -f "$ANCHOR_CKPT" ] || { echo "the B-F 40k anchor is missing: ${ANCHOR_CKPT} - abort"; exit 2; }
  [ "$RESUME_REAL" = "$ANCHOR_REAL" ] || {
    echo "INITIAL launch requires RESUME_CKPT to BE the B-F 40k anchor"
    echo "  want: ${ANCHOR_REAL}"
    echo "  got : ${RESUME_REAL}"
    echo "  -> to restart exp_10 from one of its own later checkpoints, set EXPECTED_STEP > ${ANCHOR_STEPS}. abort"
    exit 2; }
  [ "$MAXSTEPS" -gt "$ANCHOR_STEPS" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the anchor's ${ANCHOR_STEPS} steps - abort"; exit 2; }
else
  MODE="RESTART"
  SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
  case "$RESUME_REAL" in
    "${SAVEDIR_REAL}"/*) ;;
    *) echo "EXPECTED_STEP ${EXPECTED_STEP} > ${ANCHOR_STEPS} declares an exp_10 RESTART, which may only resume a"
       echo "  checkpoint written by this run, i.e. one inside ${SAVEDIR_REAL}/"
       echo "  got: ${RESUME_REAL}"
       echo "  -> the 40k anchor is resumed with EXPECTED_STEP unset (or == ${ANCHOR_STEPS}). abort"
       exit 2;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }
fi

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR10}/fa_scratch_resume_${TS}_${EXPNAME}_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_10 B-F resume (${MODE}) DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER}"
echo "resume: '${RESUME_CKPT}' | expected_step=${EXPECTED_STEP} (explicit=${EXPECTED_STEP_SET}) | mode=${MODE}"

# --- config-contract pre-flight (fail-closed; VERBATIM from p1_ddp_launch.sh /
# --- f_arm_launch.sh - it validates the BV/BVp1/BF triangle, i.e. arm-agnostic) ---
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

# --- INITIAL only: sha256 pin of the B-F 40k anchor (codex plan review Blocker 2).
# --- The expected digest is read from ${SHA_PIN_FILE} at runtime; a missing or
# --- malformed pin file aborts (fail-closed). The anchor is only ever READ. ---
if [ "$MODE" = "INITIAL" ]; then
  [ -f "$SHA_PIN_FILE" ] || { echo "sha256 pin file missing: ${SHA_PIN_FILE} - refusing to resume the anchor unverified - abort"; exit 2; }
  SHA_WANT="$(awk 'NR==1{print $1}' "$SHA_PIN_FILE")"
  printf '%s' "$SHA_WANT" | grep -Eq '^[0-9a-f]{64}$' || { echo "sha256 pin file ${SHA_PIN_FILE} does not contain a 64-hex digest (got '${SHA_WANT}') - abort"; exit 2; }
  echo "--- verifying anchor sha256 against the pin (read-only) ---"
  SHA_HAVE="$(sha256sum "$RESUME_CKPT" | awk '{print $1}')"
  echo "anchor sha256 want: ${SHA_WANT}"
  echo "anchor sha256 have: ${SHA_HAVE}"
  [ "$SHA_HAVE" = "$SHA_WANT" ] || { echo "ANCHOR SHA-256 MISMATCH - the 40k checkpoint is not the pinned B-F file - abort"; exit 2; }
  echo "anchor sha256 OK"
fi

# --- RESUME_CKPT lineage pre-flight. One torch.load; asserts the step, that the
# --- checkpoint's EMBEDDED model_config is the fa_invariant B-F config (equal to
# --- FLAC_AR_BF.json as parsed objects), full (warm) optimizer state, and
# --- scheduler/EMA presence. exp_10 has no reset lineage: a cleared optimizer
# --- state means the wrong file was passed. ---
RESUME_CKPT="$RESUME_CKPT" EXPECTED_STEP="$EXPECTED_STEP" MODE="$MODE" \
MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH" python3 - <<'PY' || { echo "resume-lineage check FAILED - abort"; exit 2; }
import json, os, sys, torch
p = os.environ["RESUME_CKPT"]; want = int(os.environ["EXPECTED_STEP"])
mode = os.environ["MODE"]; cfg_path = os.environ["MODEL_CONFIG_PATH"]
ck = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ck, dict):
    sys.exit(f"not a Lightning checkpoint: {p}")
gs = ck.get("global_step")
if gs != want:
    sys.exit(f"global_step {gs} != expected {want} (set EXPECTED_STEP to resume a different step)")
# --- embedded-config lineage: this is what makes it a B-F (fa_invariant) checkpoint ---
mc = ck.get("model_config")
if not isinstance(mc, dict):
    sys.exit("checkpoint carries no embedded 'model_config' dict -> cannot prove fa lineage")
tr = mc.get("training")
if not isinstance(tr, dict):
    sys.exit("embedded model_config has no 'training' block -> cannot prove fa lineage")
cond = tr.get("cond_method")
if cond != "fa_invariant":
    sys.exit(f"embedded model_config['training']['cond_method'] == {cond!r}, expected 'fa_invariant' "
             "-> this is NOT a B-F (frame-averaged) checkpoint; exp_10 resumes the fa lineage only")
want_cfg = json.load(open(cfg_path))
if mc != want_cfg:
    sys.exit(f"embedded model_config != {cfg_path} (parsed-object mismatch) -> the checkpoint was trained "
             "under a different architecture/training contract than the one this run would use")
if "optimizer_states" not in ck:
    sys.exit("no 'optimizer_states' key -> weights-only ckpt; PL 2.1 KeyErrors on resume")
opts = ck["optimizer_states"]
if len(opts) != 1:
    sys.exit(f"expected exactly 1 optimizer entry, found {len(opts)}")
n_state = len(opts[0].get("state", {}))
n_groups = len(opts[0].get("param_groups", []))
if n_groups != 1:
    sys.exit(f"expected exactly 1 param_group, found {n_groups}")
if not n_state:
    sys.exit("optimizer state is CLEARED (stripped checkpoint) -> exp_10 is a WARM continuation; "
             "there is no optimizer-reset arm in this experiment")
lr = opts[0]["param_groups"][0].get("lr")
if not ck.get("lr_schedulers"):
    sys.exit("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
sd = ck.get("state_dict") or {}
n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
if not n_ema:
    sys.exit("no EMA weights in state_dict")
print(f"resume lineage: {p}\n  global_step={gs} epoch={ck.get('epoch')} cond_method={cond} "
      f"frame_avg_angles={tr.get('frame_avg_angles')} optimizer_state=FULL ({n_state} entries) lr={lr} "
      f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
print(f"lineage OK for mode {mode} (embedded model_config == {cfg_path})")
PY

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
echo "mode: ${MODE} | model-config: ${MODEL_CONFIG_PATH} | name: ${NAME} | experiment: ${EXPNAME} | save-dir: ${SAVEDIR}"

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
echo "=== exp_10 B-F resume (${MODE}) exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
