#!/usr/bin/env bash
# ============================================================================
# dtail_launch.sh - exp_13 (decay_tail): continue the exp_07 P1 87,500-step
# anchor on a DECAYING InverseLR tail.
#
# Single arm, single identity (plan_decay_tail.md §3):
#   model config : worklog/worklog_yixun/exp_13_decay_tail_claude/FLAC_AR_BVp1_dtail.json
#   identity     : FLAC_exp13_DT / exp13_DT / outputs_FLAC/exp13_DT
#   budget       : --max-steps 97500 (87.5k -> 97.5k), ckpt every 1250
#
# THE TREATMENT IS A CHECKPOINT REWRITE, NOT A CONFIG CHANGE (review B1).
# InverseLR keeps inv_gamma/power/warmup/final_lr as instance attributes; torch
# serialises them into the checkpoint and `load_state_dict` is
# `self.__dict__.update(state_dict)`, so a warm resume CLOBBERS whatever the new
# config asked for. Measured: config-only swap -> inv_gamma reverts to 1e6,
# power to 0.5, lr at 97,500 = 4.7727e-05 (a silent no-op tail). This launcher
# therefore runs `src/tools/retune_lr_state.py` on the anchor to produce
#   outputs_FLAC/exp13_DT/retuned_from_87500.ckpt
# with lr_schedulers[0].{inv_gamma:30000, power:1.0} and
# optimizer_states[0].param_groups[0].lr = 1.2765957446808513e-05, and resumes
# from THAT COPY. The anchor is only ever READ (sha256sum / torch.load).
# FLAC_AR_BVp1_dtail.json is still passed as --model-config: it is the
# fresh-build source of truth and the documentation of the intended schedule.
#
# Derived from the reviewed exp_10 bf_resume_launch.sh. RESUME_CKPT is REQUIRED
# and is admitted by exactly TWO lineage modes:
#
#   (a) INITIAL  - EXPECTED_STEP unset or == 87500. RESUME_CKPT must resolve to
#       THE exact P1 87.5k anchor path, its sha256 must equal the pin in
#       anchor_87500.sha256 (read at runtime; missing pin file => abort), and the
#       checkpoint's EMBEDDED model_config must equal FLAC_AR_BVp1.json as parsed
#       objects (the anchor was trained under BVp1, NOT under the dtail config).
#       The retune tool then runs and training resumes from the retuned copy.
#   (b) RESTART  - EXPECTED_STEP > 87500, set explicitly. RESUME_CKPT must live
#       INSIDE outputs_FLAC/exp13_DT/, carry global_step == EXPECTED_STEP, and its
#       EMBEDDED model_config must equal the DTAIL config (exp_13's own checkpoints
#       embed the config this run passes). MAXSTEPS must exceed EXPECTED_STEP.
#       No sha pin and no retune (the schedule is already inside those files).
#       NB: retuned_from_87500.ckpt itself carries global_step 87500 and embeds
#       BVp1, so it is NOT a RESTART candidate - re-run INITIAL with RETUNE_FORCE=1.
#
# Recipe is byte-identical to exp_07 P1: 32/GPU x 2 GPUs x accum 1 (eff 64),
# SyncBN (BN=64), ViT grad-ckpt via config, env flac, seed 42, wandb.
#
# PROBE mode: MAXSTEPS <= 87520 switches on the post-run live-lr gate (see the
# bottom of this file). CHECKPOINT_EVERY must then be small enough that PL
# actually saves inside the probe window.
#
# NOT bit-exact across the resume boundary: Lightning restores no RNG state and
# no dataloader position.
#
# Knobs (env): MAXSTEPS CHECKPOINT_EVERY EXPECTED_STEP RESUME_CKPT LOGGER MB ACC
#              MIN_FREE_MB MIN_FREE_DISK_MB RETUNE_FORCE
#
# Usage (initial launch, full 97.5k budget):
#   RESUME_CKPT=outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt \
#   bash worklog/worklog_yixun/exp_13_decay_tail_claude/dtail_launch.sh
#
# Usage (15-step probe: the live-lr gate is the point of it):
#   MAXSTEPS=87515 CHECKPOINT_EVERY=5 \
#   RESUME_CKPT=outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt \
#   bash worklog/worklog_yixun/exp_13_decay_tail_claude/dtail_launch.sh
#
# Usage (restart after a crash, from exp_13's own 90k checkpoint):
#   EXPECTED_STEP=90000 MAXSTEPS=97500 \
#   RESUME_CKPT=outputs_FLAC/exp13_DT/FLAC_exp13_DT/exp13_DT/checkpoints/epoch=20-step=90000.ckpt \
#   bash worklog/worklog_yixun/exp_13_decay_tail_claude/dtail_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR07="worklog/worklog_yixun/exp_07_fa_scratch_claude"       # BVp1/BV/BF + arm-audit gate
EXPDIR13="worklog/worklog_yixun/exp_13_decay_tail_claude"       # dtail config, sha pin, logs
ANCHOR_STEPS=87500                                              # the P1 anchor step
ANCHOR_CKPT="outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt"
SHA_PIN_FILE="${EXPDIR13}/anchor_87500.sha256"
BVP1_PATH="${EXPDIR07}/FLAC_AR_BVp1.json"                       # the anchor's embedded config
MODEL_CONFIG_PATH="${EXPDIR13}/FLAC_AR_BVp1_dtail.json"         # hardcoded: exp_13 has ONE arm
NAME="FLAC_exp13_DT"; EXPNAME="exp13_DT"; SAVEDIR="outputs_FLAC/exp13_DT"
RETUNED_CKPT="${SAVEDIR}/retuned_from_87500.ckpt"
TAIL_LR="1.2765957446808513e-05"                                # InverseLR(30000,1.0,0.99) @ 87500
TAIL_INV_GAMMA=30000; TAIL_POWER=1.0; TAIL_WARMUP=0.99
PROBE_MAX=87520                                                 # MAXSTEPS <= this => probe mode
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"; ACC="${ACC:-1}"
RETUNE_FORCE="${RETUNE_FORCE:-0}"

_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

# --- environment gate (VERBATIM from bf_resume_launch.sh: the optimizer/scheduler
# --- restore semantics this whole experiment turns on are PL-version dependent,
# --- and plain `python` must not resolve to another env) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- knob validation ---
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1250}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2
MAXSTEPS="${MAXSTEPS:-97500}";                _posint MAXSTEPS "$MAXSTEPS" || exit 2
EXPECTED_STEP_SET=0; [ -n "${EXPECTED_STEP:-}" ] && EXPECTED_STEP_SET=1
EXPECTED_STEP="${EXPECTED_STEP:-$ANCHOR_STEPS}"; _posint EXPECTED_STEP "$EXPECTED_STEP" || exit 2

[ -f "$MODEL_CONFIG_PATH" ] || { echo "model config not found: ${MODEL_CONFIG_PATH} - abort"; exit 2; }
[ -f "$BVP1_PATH" ]        || { echo "BVp1 reference config not found: ${BVP1_PATH} - abort"; exit 2; }

# --- full-state resume is MANDATORY for exp_13 (continuation only, never scratch) ---
RESUME_CKPT="${RESUME_CKPT:-}"
[ -n "$RESUME_CKPT" ] || { echo "RESUME_CKPT is REQUIRED (exp_13 never trains from scratch) - abort"; exit 2; }
[ -f "$RESUME_CKPT" ] || { echo "RESUME_CKPT not found: ${RESUME_CKPT} - abort"; exit 2; }
RESUME_REAL="$(realpath -m "$RESUME_CKPT")"

# --- lineage mode (fail-closed; there are exactly two admissible resume stories) ---
if [ "$EXPECTED_STEP" -lt "$ANCHOR_STEPS" ]; then
  echo "EXPECTED_STEP ${EXPECTED_STEP} is below the P1 anchor step ${ANCHOR_STEPS}: exp_13 resumes the 87.5k anchor"
  echo "  (EXPECTED_STEP unset or ${ANCHOR_STEPS}) or one of its OWN later checkpoints (EXPECTED_STEP > ${ANCHOR_STEPS}) - abort"
  exit 2
elif [ "$EXPECTED_STEP" -eq "$ANCHOR_STEPS" ]; then
  MODE="INITIAL"
  WANT_EMBEDDED_CFG="$BVP1_PATH"     # the anchor was trained under BVp1
  ANCHOR_REAL="$(realpath -m "$ANCHOR_CKPT")"
  [ -f "$ANCHOR_CKPT" ] || { echo "the P1 87.5k anchor is missing: ${ANCHOR_CKPT} - abort"; exit 2; }
  [ "$RESUME_REAL" = "$ANCHOR_REAL" ] || {
    echo "INITIAL launch requires RESUME_CKPT to BE the P1 87.5k anchor"
    echo "  want: ${ANCHOR_REAL}"
    echo "  got : ${RESUME_REAL}"
    echo "  -> to restart exp_13 from one of its own later checkpoints, set EXPECTED_STEP > ${ANCHOR_STEPS}. abort"
    exit 2; }
  [ "$MAXSTEPS" -gt "$ANCHOR_STEPS" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the anchor's ${ANCHOR_STEPS} steps - abort"; exit 2; }
else
  MODE="RESTART"
  WANT_EMBEDDED_CFG="$MODEL_CONFIG_PATH"   # exp_13's own ckpts embed the dtail config
  SAVEDIR_REAL="$(realpath -m "$SAVEDIR")"
  case "$RESUME_REAL" in
    "${SAVEDIR_REAL}"/*) ;;
    *) echo "EXPECTED_STEP ${EXPECTED_STEP} > ${ANCHOR_STEPS} declares an exp_13 RESTART, which may only resume a"
       echo "  checkpoint written by this run, i.e. one inside ${SAVEDIR_REAL}/"
       echo "  got: ${RESUME_REAL}"
       echo "  -> the 87.5k anchor is resumed with EXPECTED_STEP unset (or == ${ANCHOR_STEPS}). abort"
       exit 2;;
  esac
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || { echo "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"; exit 2; }
fi

[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

# --- probe-mode cadence sanity: the live-lr gate needs at least one saved ckpt ---
PROBE=0; [ "$MAXSTEPS" -le "$PROBE_MAX" ] && PROBE=1
if [ "$PROBE" -eq 1 ]; then
  FIRST_SAVE=$(( (ANCHOR_STEPS / CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY ))
  [ "$FIRST_SAVE" -le "$MAXSTEPS" ] || {
    echo "PROBE mode (MAXSTEPS ${MAXSTEPS} <= ${PROBE_MAX}) but CHECKPOINT_EVERY=${CHECKPOINT_EVERY} would not save"
    echo "  before ${MAXSTEPS} (first save at global_step ${FIRST_SAVE}); the post-run live-lr gate reads the"
    echo "  probe's final checkpoint, so it would have nothing to read. Lower CHECKPOINT_EVERY. abort"
    exit 2; }
fi

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR13}/decay_tail_${TS}_${EXPNAME}_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_13 decay_tail (${MODE}) DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x2x${ACC} eff64 seed42 -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER}"
echo "resume: '${RESUME_CKPT}' | expected_step=${EXPECTED_STEP} (explicit=${EXPECTED_STEP_SET}) | mode=${MODE}"
echo "treatment: InverseLR(${TAIL_INV_GAMMA}, ${TAIL_POWER}, ${TAIL_WARMUP}) via CHECKPOINT REWRITE | boundary lr ${TAIL_LR} | probe=${PROBE}"

# --- config-contract pre-flight (fail-closed; VERBATIM from bf_resume_launch.sh /
# --- p1_ddp_launch.sh - it validates the BV/BVp1/BF triangle, i.e. arm-agnostic) ---
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

# --- exp_13-specific config gate (review B4.3 + B5). Cheap, so it runs TWICE: here
# --- (before any expensive gate, so a wrong config aborts in seconds) and again
# --- immediately after assert_arm_configs.py, whose printed scheduler assertion is
# --- about the exp_07 configs and would otherwise be the last word in the log. ---
exp13_config_gate() {
  DTAIL_CFG="$MODEL_CONFIG_PATH" BVP1_CFG="$BVP1_PATH" \
  TAIL_INV_GAMMA="$TAIL_INV_GAMMA" TAIL_POWER="$TAIL_POWER" TAIL_WARMUP="$TAIL_WARMUP" \
  python3 - <<'PY'
import copy, json, os, sys
sys.path.insert(0, os.getcwd())          # beat any stale pip-installed src copy
import torch
from src.training.utils import InverseLR, create_scheduler_from_config

dtail_p, bvp1_p = os.environ["DTAIL_CFG"], os.environ["BVP1_CFG"]
want = {"type": "InverseLR", "config": {"inv_gamma": int(os.environ["TAIL_INV_GAMMA"]),
                                        "power": float(os.environ["TAIL_POWER"]),
                                        "warmup": float(os.environ["TAIL_WARMUP"])}}
SCHED_PATH = ("training", "optimizer_configs", "diffusion", "scheduler")

def at(d, path):
    for k in path:
        d = d[k]
    return d

dtail, bvp1 = json.load(open(dtail_p)), json.load(open(bvp1_p))

# (1) POSITIVE, byte-exact: the registered scheduler block, key-for-key.
block = at(dtail, SCHED_PATH)
if block != want:
    sys.exit(f"dtail scheduler block {block!r} != registered {want!r}")
if sorted(block["config"]) != ["inv_gamma", "power", "warmup"]:
    sys.exit(f"dtail scheduler config has unexpected keys: {sorted(block['config'])}")
if "final_lr_ratio" in json.dumps(dtail):
    sys.exit("'final_lr_ratio' is not an InverseLR kwarg -> would TypeError in configure_optimizers")

# (2) NEGATIVE: dtail == BVp1 once BOTH scheduler subtrees are deleted.
a, b = copy.deepcopy(dtail), copy.deepcopy(bvp1)
for d in (a, b):
    del at(d, SCHED_PATH[:-1])[SCHED_PATH[-1]]
if a != b:
    sys.exit("dtail differs from BVp1 somewhere OTHER than the scheduler block (parsed-object mismatch)")
old = at(bvp1, SCHED_PATH)["config"]

# (3) RUNTIME: build the repo's scheduler from that very dict on a dummy optimizer.
p = torch.nn.Parameter(torch.zeros(2))
opt = torch.optim.AdamW([p], lr=5e-05)
sched = create_scheduler_from_config(block, opt)
if not isinstance(sched, InverseLR):
    sys.exit(f"create_scheduler_from_config returned {type(sched).__name__}, expected InverseLR")
got = (sched.inv_gamma, sched.power, sched.warmup)
if got != (want["config"]["inv_gamma"], want["config"]["power"], want["config"]["warmup"]):
    sys.exit(f"instantiated InverseLR{got} != InverseLR(30000, 1.0, 0.99)")

print(f"exp_13 config gate OK: {dtail_p}")
print(f"  scheduler block (byte-exact) : {json.dumps(block)}")
print(f"  BVp1 said                    : {json.dumps(old)}  -> exp_13 decays, BVp1 does not")
print(f"  dtail == BVp1 outside the scheduler block: yes")
print(f"  create_scheduler_from_config -> InverseLR(inv_gamma={got[0]}, power={got[1]}, warmup={got[2]})")
PY
}
exp13_config_gate || { echo "exp_13 config gate FAILED - abort"; exit 2; }

# --- disk floor on the outputs volume (review B4: it sits at 95%; the retuned copy
# --- is ~690 MB and each 1250-step checkpoint another ~690 MB).
# --- Bypass for guard-testing only: MIN_FREE_DISK_MB=1. ---
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"; _posint MIN_FREE_DISK_MB "$MIN_FREE_DISK_MB" || exit 2
DF_TARGET="outputs_FLAC"; [ -d "$DF_TARGET" ] || DF_TARGET="."
DISK_FREE_MB="$(df -P -B1M "$DF_TARGET" 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')"
[ -n "$DISK_FREE_MB" ] || { echo "df query failed on ${DF_TARGET} - refusing to launch blind - abort"; exit 2; }
echo "disk: ${DISK_FREE_MB} MiB free on the volume holding ${DF_TARGET} (floor ${MIN_FREE_DISK_MB} MiB)"
[ "$DISK_FREE_MB" -ge "$MIN_FREE_DISK_MB" ] || { echo "free disk ${DISK_FREE_MB} MiB < required ${MIN_FREE_DISK_MB} MiB - refusing to launch"; exit 2; }

# --- INITIAL only: sha256 pin of the P1 87.5k anchor. The expected digest is read
# --- from ${SHA_PIN_FILE} at runtime; a missing or malformed pin file aborts
# --- (fail-closed). The anchor is only ever READ. ---
if [ "$MODE" = "INITIAL" ]; then
  [ -f "$SHA_PIN_FILE" ] || { echo "sha256 pin file missing: ${SHA_PIN_FILE} - refusing to resume the anchor unverified - abort"; exit 2; }
  SHA_WANT="$(awk 'NR==1{print $1}' "$SHA_PIN_FILE")"
  printf '%s' "$SHA_WANT" | grep -Eq '^[0-9a-f]{64}$' || { echo "sha256 pin file ${SHA_PIN_FILE} does not contain a 64-hex digest (got '${SHA_WANT}') - abort"; exit 2; }
  echo "--- verifying anchor sha256 against the pin (read-only) ---"
  SHA_HAVE="$(sha256sum "$RESUME_CKPT" | awk '{print $1}')"
  echo "anchor sha256 want: ${SHA_WANT}"
  echo "anchor sha256 have: ${SHA_HAVE}"
  [ "$SHA_HAVE" = "$SHA_WANT" ] || { echo "ANCHOR SHA-256 MISMATCH - the 87.5k checkpoint is not the pinned P1 file - abort"; exit 2; }
  echo "anchor sha256 OK"

  # cheap pre-check of the retune destination, so a stale copy aborts in seconds
  # rather than after every expensive gate has run.
  if [ -e "$RETUNED_CKPT" ] && [ "$RETUNE_FORCE" != "1" ]; then
    echo "the retuned checkpoint already exists: ${RETUNED_CKPT}"
    echo "  refusing to overwrite it (a live or previous exp_13 run may be resuming from it)."
    echo "  -> to rebuild it from the anchor, re-run with RETUNE_FORCE=1. abort"
    exit 2
  fi
fi

# --- RESUME_CKPT lineage pre-flight. One torch.load; asserts the step, that the
# --- checkpoint's EMBEDDED model_config equals the config appropriate to the mode
# --- (INITIAL -> BVp1, the anchor's own; RESTART -> the dtail config, which exp_13's
# --- own checkpoints embed), full (warm) optimizer state, and scheduler/EMA presence.
# --- exp_13 has no reset lineage: a cleared optimizer state means the wrong file. ---
RESUME_CKPT="$RESUME_CKPT" EXPECTED_STEP="$EXPECTED_STEP" MODE="$MODE" \
WANT_EMBEDDED_CFG="$WANT_EMBEDDED_CFG" python3 - <<'PY' || { echo "resume-lineage check FAILED - abort"; exit 2; }
import json, os, sys, torch
p = os.environ["RESUME_CKPT"]; want = int(os.environ["EXPECTED_STEP"])
mode = os.environ["MODE"]; cfg_path = os.environ["WANT_EMBEDDED_CFG"]
ck = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(ck, dict):
    sys.exit(f"not a Lightning checkpoint: {p}")
gs = ck.get("global_step")
if gs != want:
    sys.exit(f"global_step {gs} != expected {want} (set EXPECTED_STEP to resume a different step)")
mc = ck.get("model_config")
if not isinstance(mc, dict):
    sys.exit("checkpoint carries no embedded 'model_config' dict -> cannot prove lineage")
want_cfg = json.load(open(cfg_path))
if mc != want_cfg:
    sys.exit(f"embedded model_config != {cfg_path} (parsed-object mismatch) -> in mode {mode} the "
             "checkpoint must have been trained under exactly that config "
             "(INITIAL: the anchor's BVp1; RESTART: exp_13's own dtail config)")
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
    sys.exit("optimizer state is CLEARED (stripped checkpoint) -> exp_13 is a WARM continuation; "
             "there is no optimizer-reset arm in this experiment")
lr = opts[0]["param_groups"][0].get("lr")
scheds = ck.get("lr_schedulers")
if not scheds:
    sys.exit("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
s0 = scheds[0]
sd = ck.get("state_dict") or {}
n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
if not n_ema:
    sys.exit("no EMA weights in state_dict")
print(f"resume lineage: {p}\n  global_step={gs} epoch={ck.get('epoch')} optimizer_state=FULL "
      f"({n_state} entries) lr={lr} ema_entries={n_ema}")
print(f"  embedded scheduler state: inv_gamma={s0.get('inv_gamma')} power={s0.get('power')} "
      f"warmup={s0.get('warmup')} last_epoch={s0.get('last_epoch')} _last_lr={s0.get('_last_lr')}")
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
HF_HUB_OFFLINE=1 python "${EXPDIR07}/assert_arm_configs.py" || { echo "GATE FAILED - abort"; exit 1; }

cat <<'NOTE'
################################################################################
## NOTE - READ THE LINE ABOVE CAREFULLY.                                      ##
## assert_arm_configs.py is an exp_07 artifact. Its                           ##
##     "InverseLR(1e6,0.5,0.99)"                                              ##
## assertion is about FLAC_AR_BV.json / FLAC_AR_BF.json - the exp_07 arms.    ##
## It says NOTHING about this run, which uses FLAC_AR_BVp1_dtail.json and     ##
## whose schedule is delivered by a CHECKPOINT REWRITE, not by the config.    ##
## The gate is kept ONLY for its DINOv3 initializer pin + init-identity check.##
## The authoritative exp_13 scheduler assertion is re-run immediately below.  ##
################################################################################
NOTE
exp13_config_gate || { echo "exp_13 config gate FAILED - abort"; exit 2; }

# --- INITIAL only: BUILD THE TREATMENT. Copy the anchor with the InverseLR tail
# --- installed, then verify the copy before handing it to train.py. The anchor is
# --- re-hashed afterwards to prove it was not touched. ---
TRAIN_CKPT="$RESUME_CKPT"
if [ "$MODE" = "INITIAL" ]; then
  echo "--- building the retuned checkpoint (the treatment) ---"
  mkdir -p "$SAVEDIR" || { echo "could not create ${SAVEDIR} - abort"; exit 2; }
  RETUNE_ARGS=(--in "$RESUME_CKPT" --out "$RETUNED_CKPT")
  [ "$RETUNE_FORCE" = "1" ] && RETUNE_ARGS+=(--force)
  python -m src.tools.retune_lr_state "${RETUNE_ARGS[@]}" || { echo "retune_lr_state FAILED - abort"; exit 2; }
  [ -f "$RETUNED_CKPT" ] || { echo "retune produced no file at ${RETUNED_CKPT} - abort"; exit 2; }

  echo "--- post-condition check on ${RETUNED_CKPT} ---"
  RETUNED_CKPT="$RETUNED_CKPT" EXPECTED_STEP="$ANCHOR_STEPS" TAIL_LR="$TAIL_LR" \
  TAIL_INV_GAMMA="$TAIL_INV_GAMMA" TAIL_POWER="$TAIL_POWER" TAIL_WARMUP="$TAIL_WARMUP" \
  BVP1_PATH="$BVP1_PATH" python3 - <<'PY' || { echo "retuned-checkpoint post-condition FAILED - abort"; exit 2; }
import json, os, sys, torch
p = os.environ["RETUNED_CKPT"]
want_lr = float(os.environ["TAIL_LR"])
want_ig, want_pw = int(os.environ["TAIL_INV_GAMMA"]), float(os.environ["TAIL_POWER"])
want_wu = float(os.environ["TAIL_WARMUP"]); want_step = int(os.environ["EXPECTED_STEP"])
ck = torch.load(p, map_location="cpu", weights_only=False)
s = ck["lr_schedulers"][0]; g = ck["optimizer_states"][0]["param_groups"][0]
fails = []
if ck.get("global_step") != want_step:
    fails.append(f"global_step {ck.get('global_step')} != {want_step}")
if s.get("inv_gamma") != want_ig:
    fails.append(f"lr_schedulers[0]['inv_gamma'] {s.get('inv_gamma')!r} != {want_ig!r}")
if s.get("power") != want_pw:
    fails.append(f"lr_schedulers[0]['power'] {s.get('power')!r} != {want_pw!r}")
if s.get("warmup") != want_wu:
    fails.append(f"lr_schedulers[0]['warmup'] {s.get('warmup')!r} != {want_wu!r} (must be preserved)")
if s.get("last_epoch") != want_step:
    fails.append(f"lr_schedulers[0]['last_epoch'] {s.get('last_epoch')!r} != {want_step} (counter must be preserved)")
if g.get("lr") != want_lr:
    fails.append(f"param_groups[0]['lr'] {g.get('lr')!r} != {want_lr!r}")
if g.get("initial_lr") != 5e-05:
    fails.append(f"param_groups[0]['initial_lr'] {g.get('initial_lr')!r} != 5e-05 (must be preserved)")
if not len(ck["optimizer_states"][0].get("state", {})):
    fails.append("Adam moments were dropped (exp_13 is a WARM continuation)")
if ck.get("model_config") != json.load(open(os.environ["BVP1_PATH"])):
    fails.append("embedded model_config was altered (it must stay the anchor's BVp1)")
if fails:
    sys.exit("retuned checkpoint post-condition FAILED:\n  - " + "\n  - ".join(fails))
print(f"retuned OK: {p}")
print(f"  lr_schedulers[0]: inv_gamma={s['inv_gamma']} power={s['power']} warmup={s['warmup']} "
      f"final_lr={s.get('final_lr')} base_lrs={s.get('base_lrs')} last_epoch={s['last_epoch']} "
      f"_step_count={s.get('_step_count')}")
print(f"  optimizer param_groups[0]: lr={g['lr']!r} initial_lr={g['initial_lr']!r} "
      f"({len(ck['optimizer_states'][0]['state'])} Adam moment entries kept)")
print(f"  embedded model_config: unchanged (== BVp1)")
PY

  echo "--- anchor integrity re-check (must still equal the pin) ---"
  SHA_AFTER="$(sha256sum "$ANCHOR_CKPT" | awk '{print $1}')"
  echo "anchor sha256 after retune: ${SHA_AFTER}"
  [ "$SHA_AFTER" = "$SHA_WANT" ] || { echo "ANCHOR SHA-256 CHANGED DURING THE RETUNE - abort"; exit 2; }
  echo "anchor unchanged"
  TRAIN_CKPT="$RETUNED_CKPT"
fi

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | grad-ckpt: config"
echo "mode: ${MODE} | model-config: ${MODEL_CONFIG_PATH} | name: ${NAME} | experiment: ${EXPNAME} | save-dir: ${SAVEDIR}"
echo "resuming from: ${TRAIN_CKPT}"

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "$MODEL_CONFIG_PATH" \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --ckpt-path "$TRAIN_CKPT" \
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every "$CHECKPOINT_EVERY" \
  --name "$NAME" --experiment-name "$EXPNAME" \
  --save-dir "$SAVEDIR"
rc=$?
echo "=== exp_13 decay_tail (${MODE}) exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="

# ============================================================================
# PROBE live-lr gate (review B1: "analytic assertion is exactly what cannot see
# this bug" - so the gate reads REAL post-restore state).
#
# WHY THE CHECKPOINT AND NOT THE LOG: `train/lr` IS logged
# (src/training/diffusion.py:353 -> log_dict(..., prog_bar=True, on_step=True)),
# but its only textual sink here is the tqdm progress bar, which renders metrics
# at ~3 significant figures ("1.28e-5"). That cannot support a 1e-12 comparison,
# and with `--logger wandb` the full-precision value goes to wandb, not to this
# tee'd file. The robust reading is therefore the probe's own final checkpoint:
# PL writes `optimizer_states[0]['param_groups'][0]['lr']` and
# `lr_schedulers[0]` at full float precision, after a real restore and real
# optimizer/scheduler steps. The bar's rounded value is still grepped and
# printed as an independent, non-gating cross-check.
# ============================================================================
if [ "$PROBE" -eq 1 ]; then
  echo "--- PROBE live-lr gate (MAXSTEPS ${MAXSTEPS} <= ${PROBE_MAX}) ---"
  if [ "$rc" -ne 0 ]; then
    echo "probe run exited rc=${rc}; live-lr gate NOT run (nothing to verify) - abort"
    exit "$rc"
  fi
  echo "progress-bar train/lr occurrences (informational, ~3 s.f. - NOT the gate):"
  grep -o 'train/lr=[^ ,]*' "$LOG" | tail -3 || echo "  (none captured in the tee'd log)"

  SAVEDIR="$SAVEDIR" ANCHOR_STEPS="$ANCHOR_STEPS" MAXSTEPS="$MAXSTEPS" TAIL_LR="$TAIL_LR" \
  TAIL_INV_GAMMA="$TAIL_INV_GAMMA" TAIL_POWER="$TAIL_POWER" TAIL_WARMUP="$TAIL_WARMUP" \
  python3 - <<'PY' || { echo "PROBE live-lr gate FAILED - abort"; exit 2; }
import glob, os, sys, torch

savedir = os.environ["SAVEDIR"]
anchor = int(os.environ["ANCHOR_STEPS"]); maxsteps = int(os.environ["MAXSTEPS"])
want_lr = float(os.environ["TAIL_LR"])
want_ig, want_pw = int(os.environ["TAIL_INV_GAMMA"]), float(os.environ["TAIL_POWER"])
want_wu = float(os.environ["TAIL_WARMUP"])
TOL = 1e-12

cands = [p for p in glob.glob(os.path.join(savedir, "**", "*.ckpt"), recursive=True)
         if os.path.basename(p) != "retuned_from_87500.ckpt"]
if not cands:
    sys.exit(f"no probe checkpoint under {savedir} -> the live-lr gate has nothing to read "
             "(lower CHECKPOINT_EVERY so PL saves inside the probe window)")
p = max(cands, key=os.path.getmtime)
ck = torch.load(p, map_location="cpu", weights_only=False)
gs = ck.get("global_step")
s = ck["lr_schedulers"][0]
lr = ck["optimizer_states"][0]["param_groups"][0]["lr"]

def closed_form(last_epoch, inv_gamma, power, warmup, final_lr, base_lr):
    """src/training/utils.py::InverseLR._get_closed_form_lr"""
    return (1 - warmup ** (last_epoch + 1)) * max(final_lr, base_lr * (1 + last_epoch / inv_gamma) ** -power)

print(f"probe checkpoint: {p}")
print(f"  global_step={gs}")
print(f"  lr_schedulers[0]: inv_gamma={s.get('inv_gamma')} power={s.get('power')} "
      f"warmup={s.get('warmup')} last_epoch={s.get('last_epoch')} base_lrs={s.get('base_lrs')}")
print(f"  optimizer param_groups[0]['lr'] = {lr!r}")

fails = []
if not (anchor < gs <= maxsteps):
    fails.append(f"global_step {gs} outside the probe window ({anchor}, {maxsteps}]")
# 1. the scheduler survived the restore with the TAIL hyperparameters (the B1 bug
#    would show up right here as 1000000 / 0.5)
if s.get("inv_gamma") != want_ig:
    fails.append(f"POST-RESTORE inv_gamma {s.get('inv_gamma')!r} != {want_ig!r} -> the tail was CLOBBERED")
if s.get("power") != want_pw:
    fails.append(f"POST-RESTORE power {s.get('power')!r} != {want_pw!r} -> the tail was CLOBBERED")
if s.get("warmup") != want_wu:
    fails.append(f"POST-RESTORE warmup {s.get('warmup')!r} != {want_wu!r}")
# 2. the live lr sits on the tail curve, not the untreated one
base = (s.get("base_lrs") or [None])[0]
if base is None:
    fails.append("scheduler state has no base_lrs")
else:
    expect = closed_form(s.get("last_epoch"), want_ig, want_pw, want_wu, s.get("final_lr", 0.0), base)
    print(f"  expected on the tail curve at last_epoch={s.get('last_epoch')}: {expect!r}")
    print(f"  boundary tail lr (step {anchor}):                    {want_lr!r}")
    if abs(lr - expect) > TOL:
        fails.append(f"live lr {lr!r} != tail-curve {expect!r} (|d|={abs(lr - expect):.3e} > {TOL:.0e})")
    # a probe is <=20 steps past the boundary; the tail curve moves ~1.1e-10/step there,
    # so the live lr must still be within 1e-8 of the boundary value. The untreated
    # schedule would sit at ~4.79e-05, i.e. 3.5e-05 away.
    if abs(lr - want_lr) > 1e-8:
        fails.append(f"live lr {lr!r} is not within 1e-08 of the boundary tail lr {want_lr!r} "
                     f"(|d|={abs(lr - want_lr):.3e}) -> NOT on the decay tail")
if fails:
    sys.exit("PROBE live-lr gate FAILED:\n  - " + "\n  - ".join(fails))
print(f"PROBE live-lr gate PASSED: post-restore InverseLR({want_ig}, {want_pw}, {want_wu}) and the live "
      f"optimizer lr is on the decay tail (within {TOL:.0e} of the closed form, "
      f"{abs(lr - want_lr):.3e} from the boundary value).")
PY
  echo "PROBE live-lr gate OK"
fi

exit $rc
