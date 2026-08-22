#!/usr/bin/env bash
# ============================================================================
# bfc_launch.sh - exp_21 (bf_fa_cartesian): the BFC arm, FROM SCRATCH,
# 2-GPU DDP + SyncBN, 40,000 steps, seed 42.
#
# WHAT THIS ARM IS. B-F (exp_07 `fa_invariant`) with exactly ONE mechanism
# swapped: the pose branch's symmetrization. Cylindrical pose invariants become
# a full C4 frame average over all four pose/geometry conditioners, poses left
# raw Cartesian (`cond_method: fa_cartesian`). Everything else - architecture,
# optimizer, schedule, EMA, metrics, data - is B-F's.
#
# THE RECIPE IS B-F's OWN LAUNCHER, worklog/worklog_yixun/exp_07_fa_scratch_claude/
# bf_scratch_launch.sh (plan §3f/§4.7). Flag-for-flag, the training closure here
# differs from that file in exactly three places, all declared:
#   1. --model-config  -> FLAC_AR_BFC.json (this experiment's arm; it is
#      FLAC_AR_BF.json + cond_method + the declared training cap, nothing else -
#      asserted below as a parsed-object diff, not a diff'd file);
#   2. --max-steps     -> 40000 (plan §5: BFC's registered endpoint. B-F ran to
#      67500; its 40k checkpoint is the comparator);
#   3. the identity triple --name / --experiment-name / --save-dir.
# Two flags are stated here that bf_scratch_launch.sh let defaults.ini supply:
#   * --precision bf16-mixed. defaults.ini:33 says "bf16-mixed" and prefigure
#     literal_eval's it, so this is the SAME value B-F trained under - stated
#     because defaults.ini has already drifted once (strategy is now "auto",
#     which is why B-F's launcher pinned that flag explicitly after the fact).
#   * --logger wandb (bf_scratch_launch.sh passed --logger "$LOGGER", default
#     wandb; same thing with the knob named).
# NOTHING ELSE from that launcher is dropped. Its BN-64 rung pin (MB=32 ACC=1,
# string equality), its per-GPU free-VRAM gate, its wandb identity gate, its
# DINOv3 pin + init-identity gate, its env manifest and its tee'd timestamped
# log are all here; the modern gates transplanted from exp_13's dtail_launch.sh
# (conda + PL assert, parsed-config contract, df floor, knob validation) are
# added AROUND that recipe, never inside it.
#
# NO VALIDATION LOADER, DELIBERATELY (plan §3f, r2 plan-review blocker 1).
# B-F trained without one (bf_scratch_launch.sh:88-96 passes no val config), and
# a validation pass draws RNG noise that would shift every subsequent training
# draw - so adding one would break the single-delta parity this whole arm exists
# to measure. The flag is not merely omitted: the assembled command is asserted
# to be free of it.
#
# CHUNK PLAN (announcement 06, declared BEFORE the run):
#   training cap 32 (FLAC_AR_BFC.json training.frame_avg_max_fwd_samples)
#   micro-batch 32/rank, C4 orbit -> angles_per_chunk = max(1, 32//32) = 1,
#   i.e. ONE angle per chunk = per-angle DINOv3 RoPE draws = B-F's training-era
#   schedule. The gate below recomputes this from the config and the live rung
#   and aborts if it is not 1. EVALUATION is a separate declaration: cap 64 at
#   batch 64 (plan §5 templates), which the eval driver pins, not this file.
#
# INIT-IDENTITY AUDIT (plan §3f/§4.7). Both arms are built on CPU under the same
# seed and their full state_dicts must hash identically. The architectures are
# identical by construction, so this MUST pass; a mismatch means the configs have
# diverged somewhere real and the arm is no longer a single-delta comparison.
# What it covers: every module create_model_from_config builds, the VAE
# pretransform module included (randomly initialised - the trained VAE weights
# arrive via --pretransform-ckpt-path at train time, from the same file for both
# arms, and are never loaded here). It also asserts the DINOv3 cache pin (the ViT
# is trainable, so its initialiser is lineage-relevant) and the wrapper wiring
# each config actually produces.
#
# CO-TENANCY. Yixun's own runs hold both A6000s (plan §5, D3). The VRAM floor is
# therefore a per-GPU FREE-memory floor, not an exclusivity check: 20,480 MiB
# per GPU by default, above B-F's measured checkpointed rank (~15.7 GiB) plus a
# DDP/SyncBN allowance, with room left for the co-tenant.
#
# NOT bit-equivalent to anything resumed: this is a from-scratch run and there is
# no resume path here at all. RESUMING IS OUT OF SCOPE - no --ckpt-path, no
# --pretrained-ckpt-path; a crashed run is relaunched by decision, not by this
# script guessing.
#
# TWO MODES, AND THE MANIFEST IS NOT A DEFAULT (r4 review BLOCKING 1).
#   REGISTERED (the default): MAXSTEPS=40000, CHECKPOINT_EVERY=2500,
#     LOGGER=wandb and the MB=32/ACC=1 rung are PINNED. Passing any other value
#     aborts - it does not override. Before this, each of them merely defaulted,
#     so `MAXSTEPS=50000 CHECKPOINT_EVERY=1 LOGGER=none bash bfc_launch.sh`
#     passed every gate and trained an unapproved recipe under the approved run's
#     name and save-dir. (Confirmed live in the round-5 red phase: that command
#     reached a real training start before it was killed.)
#   SMOKE=1: the ONE sanctioned short run (ladder rung 5). Steps default to 25
#     and may not exceed 50, the logger is forced off, checkpointing is pushed
#     past the budget, and the run gets its OWN identity and save-dir
#     (outputs_FLAC/exp21_BFC_smoke). A smoke run therefore cannot write into,
#     log to, or be mistaken for the registered run - which is what makes it safe
#     to have a short mode at all.
#
# Knobs (env): DRY_RUN{0,1} SMOKE{0,1} MB ACC MIN_FREE_MB MIN_FREE_DISK_MB, plus
#              MAXSTEPS/CHECKPOINT_EVERY/LOGGER, which are ACCEPTED ONLY where
#              the mode above says so. DRY_RUN and SMOKE fail closed on anything
#              but 0/1 - "not 1, therefore train" is how a typo becomes a run.
#
# Usage (the registered launch - no knobs, on purpose):
#   conda activate flac
#   bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
#
# Usage (gate rehearsal - every gate runs, train.py does not):
#   DRY_RUN=1 bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
#
# Usage (ladder rung 5, ~25 steps, own namespace):
#   SMOKE=1 bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch.sh
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR21="worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude"
EXPDIR07="worklog/worklog_yixun/exp_07_fa_scratch_claude"
MODEL_CONFIG_PATH="${EXPDIR21}/FLAC_AR_BFC.json"      # hardcoded: exp_21 has ONE arm
BF_CONFIG_PATH="${EXPDIR07}/FLAC_AR_BF.json"          # the parity reference (B-F)
DATASET_CONFIG="src/configs/dataset_configs/AR/train/acousticroom_train.json"
PRETRANSFORM="weights/FLAC/VAE.safetensors"
NAME="FLAC_exp21_BFC"; EXPNAME="exp21_BFC"; SAVEDIR="outputs_FLAC/exp21_BFC"
NUM_GPUS=2; STRATEGY="ddp_find_unused_parameters_true"; PRECISION="bf16-mixed"
NUM_WORKERS=6; SEED=42
WANT_TRAIN_CAP=32            # the DECLARED training chunk cap (plan §2, D5)
WANT_ANGLES_PER_CHUNK=1      # ...and the partition it must produce at this rung

# --- THE REGISTERED MANIFEST (plan §5). These are not defaults, they are the
# --- approved recipe: in the registered mode the launcher trains exactly this or
# --- it trains nothing (r4 review BLOCKING 1). MB/ACC were already pinned; these
# --- three were merely defaulted, so `MAXSTEPS=50000 CHECKPOINT_EVERY=1
# --- LOGGER=none bash bfc_launch.sh` passed every gate and trained an unapproved
# --- recipe under the approved run's name. Verified live during the round-5 red
# --- phase: that command reached a real training start.
REG_LOGGER="wandb"; REG_MAXSTEPS=40000; REG_CHECKPOINT_EVERY=2500
# --- SMOKE mode is the ONE sanctioned short run (ladder rung 5). It is a
# --- different mode, not a loosened manifest: its own save-dir and run identity,
# --- no logger, no checkpoint cadence inside the window, its own *_smoke.log.
SMOKE_MAXSTEPS_DEFAULT=25; SMOKE_MAXSTEPS_CAP=50
SMOKE_NAME="FLAC_exp21_BFC_smoke"; SMOKE_EXPNAME="exp21_BFC_smoke"
SMOKE_SAVEDIR="outputs_FLAC/exp21_BFC_smoke"

# did the CALLER set these, as opposed to inheriting the manifest? Captured
# BEFORE defaulting, because "equals the approved value" and "was never asked
# for" must stay distinguishable in smoke mode.
LOGGER_SET=0;           [ -n "${LOGGER+x}" ]           && LOGGER_SET=1
MAXSTEPS_SET=0;         [ -n "${MAXSTEPS+x}" ]         && MAXSTEPS_SET=1
CHECKPOINT_EVERY_SET=0; [ -n "${CHECKPOINT_EVERY+x}" ] && CHECKPOINT_EVERY_SET=1
LOGGER="${LOGGER:-$REG_LOGGER}"
MB="${MB:-32}"; ACC="${ACC:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

_posint() { # $1=name $2=value -> must be a positive integer
  case "$2" in ''|*[!0-9]*) echo "$1 must be a positive integer (got '$2') - abort"; return 1;; esac
  [ "$2" -gt 0 ] || { echo "$1 must be > 0 (got '$2') - abort"; return 1; }
}

_bool01() { # $1=name $2=value -> must be exactly 0 or 1 (never "not 1 => live run")
  case "$2" in 0|1) return 0;; esac
  echo "$1 must be 0 or 1 (got '$2') - abort"; return 1
}

# --- environment gate (VERBATIM from dtail_launch.sh: plain `python` must not
# --- resolve to another env, and the training closure is PL-version dependent) ---
[ "${CONDA_DEFAULT_ENV:-}" = "flac" ] || { echo "CONDA_DEFAULT_ENV must be 'flac' (got '${CONDA_DEFAULT_ENV:-<unset>}') - run 'conda activate flac' - abort"; exit 2; }
python - <<'PY' || { echo "environment gate FAILED (need pytorch_lightning 2.1.0) - abort"; exit 2; }
import sys
import pytorch_lightning as pl, torch
print(f"env gate: python {sys.version.split()[0]} | pytorch_lightning {pl.__version__} | torch {torch.__version__}")
sys.exit(0 if pl.__version__ == "2.1.0" else 2)
PY

# --- mode + knob validation (BEFORE every expensive gate, so a recipe this
# --- launcher may not train aborts in under a second) ---
_bool01 DRY_RUN "$DRY_RUN" || exit 2
_bool01 SMOKE "$SMOKE" || exit 2

CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$REG_CHECKPOINT_EVERY}"; _posint CHECKPOINT_EVERY "$CHECKPOINT_EVERY" || exit 2
MAXSTEPS="${MAXSTEPS:-$REG_MAXSTEPS}";                         _posint MAXSTEPS "$MAXSTEPS" || exit 2

if [ "$SMOKE" = "1" ]; then
  # --- SMOKE: short, logger-less, in its own namespace. ---------------------
  MODE="SMOKE"
  [ "$MAXSTEPS_SET" = "1" ] || MAXSTEPS="$SMOKE_MAXSTEPS_DEFAULT"
  if [ "$CHECKPOINT_EVERY_SET" = "1" ]; then
    echo "SMOKE=1 disables checkpointing (got CHECKPOINT_EVERY='${CHECKPOINT_EVERY}'): a rehearsal"
    echo "  must not leave checkpoints anywhere. Drop CHECKPOINT_EVERY. abort"; exit 2
  fi
  [ "$MAXSTEPS" -le "$SMOKE_MAXSTEPS_CAP" ] || {
    echo "SMOKE=1 caps MAXSTEPS at ${SMOKE_MAXSTEPS_CAP} (got ${MAXSTEPS}): a smoke run is a"
    echo "  ladder rung, not a short training run. For the approved training run use the"
    echo "  registered manifest (no MAXSTEPS at all). abort"; exit 2; }
  if [ "$LOGGER_SET" = "1" ] && [ "$LOGGER" != "none" ]; then
    echo "SMOKE=1 runs without a logger (got LOGGER='${LOGGER}'): a rehearsal must not create a"
    echo "  run in the project the registered training run publishes to. Drop LOGGER, or pass"
    echo "  LOGGER=none explicitly. abort"; exit 2
  fi
  LOGGER="none"
  # checkpointing OFF inside the window: the cadence is pushed past the budget,
  # AND the run writes to its own save-dir, so nothing a smoke run produces can
  # ever be mistaken for - or land beside - a registered checkpoint.
  CHECKPOINT_EVERY=$((MAXSTEPS + 1000000))
  NAME="$SMOKE_NAME"; EXPNAME="$SMOKE_EXPNAME"; SAVEDIR="$SMOKE_SAVEDIR"
else
  # --- REGISTERED: the approved manifest, or nothing. -----------------------
  MODE="REGISTERED"
  _pinned() { # $1=knob $2=value $3=approved
    [ "$2" = "$3" ] && return 0
    echo "${1}='${2}' is not the registered BFC manifest (${1}=${3}) - refusing to train an"
    echo "  unapproved recipe under the approved run's identity. The manifest is fixed by"
    echo "  plan §5 and the r4 review; for a short rehearsal use SMOKE=1, which runs in its"
    echo "  own save-dir with its own run name. abort"
    return 1
  }
  _pinned MAXSTEPS "$MAXSTEPS" "$REG_MAXSTEPS" || exit 2
  _pinned CHECKPOINT_EVERY "$CHECKPOINT_EVERY" "$REG_CHECKPOINT_EVERY" || exit 2
  _pinned LOGGER "$LOGGER" "$REG_LOGGER" || exit 2
fi

# The BN-64 mandate leaves exactly ONE legal rung, and accumulation never feeds
# BN statistics, so it is pinned by STRING equality - arithmetic like MB*2*ACC==64
# is bypassable via bash integer overflow (bf_scratch_launch.sh:34-37).
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

for f in "$MODEL_CONFIG_PATH" "$BF_CONFIG_PATH" "$DATASET_CONFIG" "$PRETRANSFORM"; do
  [ -f "$f" ] || { echo "required input not found: ${f} - abort"; exit 2; }
done

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
# A rehearsal is not a training run and must not leave a log that reads like one
# (the SOP's per-run log is evidence; a *_train.log that trained nothing is a
# forged one). Same folder, same timestamp convention, different noun.
if [ "$DRY_RUN" = "1" ]; then
  LOG="${EXPDIR21}/bf_fa_cartesian_${TS}_dryrun.log"
elif [ "$MODE" = "SMOKE" ]; then
  LOG="${EXPDIR21}/bf_fa_cartesian_${TS}_smoke.log"
else
  LOG="${EXPDIR21}/bf_fa_cartesian_${TS}_train.log"
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_21 BFC from-scratch DDP+SyncBN (${MODE}) - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "mode: ${MODE} $([ "$MODE" = "SMOKE" ] && echo "(ladder rung: short, logger-less, own save-dir - NEVER the registered run)" || echo "(the approved manifest; MAXSTEPS/CHECKPOINT_EVERY/LOGGER are pinned, not defaulted)")"
echo "identity: --name ${NAME} --experiment-name ${EXPNAME} --save-dir ${SAVEDIR}"
echo "recipe: ${MB}x${NUM_GPUS}x${ACC} eff64 seed${SEED} -> ${MAXSTEPS} | ckpt-every ${CHECKPOINT_EVERY} | logger=${LOGGER} | precision=${PRECISION}"
echo "arm: fa_cartesian (single-delta vs exp_07 B-F) | validation loader: NONE (parity: B-F trained without one)"
echo "resume: NONE (from scratch; this launcher has no resume path) | dry_run=${DRY_RUN}"

# --- config contract (fail-closed, parsed objects - never a text diff) ---------
# The arm is a ONE-MECHANISM change, so the diff against B-F is enumerable: the
# conditioning method and the declared training cap, and nothing else. This gate
# is what makes that claim checkable at launch time rather than at review time.
MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH" BF_CONFIG_PATH="$BF_CONFIG_PATH" \
WANT_TRAIN_CAP="$WANT_TRAIN_CAP" MB="$MB" WANT_ANGLES_PER_CHUNK="$WANT_ANGLES_PER_CHUNK" \
python3 - <<'PY' || { echo "config contract FAILED - abort"; exit 2; }
import copy, json, os, sys

bfc = json.load(open(os.environ["MODEL_CONFIG_PATH"]))
bf = json.load(open(os.environ["BF_CONFIG_PATH"]))
cap = int(os.environ["WANT_TRAIN_CAP"]); mb = int(os.environ["MB"])
want_apc = int(os.environ["WANT_ANGLES_PER_CHUNK"])

# (1) the DECLARED edits, byte-exact.
if bfc["training"].get("cond_method") != "fa_cartesian":
    sys.exit(f"BFC training.cond_method is {bfc['training'].get('cond_method')!r}, not 'fa_cartesian'")
if bfc["training"].get("frame_avg_max_fwd_samples") != cap:
    sys.exit(f"BFC training.frame_avg_max_fwd_samples is "
             f"{bfc['training'].get('frame_avg_max_fwd_samples')!r}, not the declared {cap}")
if bfc["training"].get("frame_avg_angles") != [0.0, 90.0, 180.0, 270.0]:
    sys.exit(f"BFC frame_avg_angles {bfc['training'].get('frame_avg_angles')!r} is not the C4 orbit")
if "yaw_aug" in json.dumps(bfc):
    sys.exit("BFC declares a yaw-augmentation key: yaw_aug and a frame-averaged "
             "method are mutually exclusive (rejected by the wrapper AND the factory)")

# (2) the ViT gradient-checkpointing the B-F recipe runs under lives in the
# CONFIG (bf_scratch_launch.sh passes no flag for it), so it is asserted here --
# and BEFORE the equality below, which would also catch its absence but would
# report it as an anonymous "something drifted". The specific diagnosis wins.
n = sum(1 for c in bfc["model"]["conditioning"]["configs"]
        if c.get("id") in ("source_vit", "context_poses_vit")
        and c["config"].get("gradient_checkpointing") is True)
if n != 2:
    sys.exit(f"expected ViT gradient_checkpointing on BOTH ViT conditioners, found {n}")

# (3) NEGATIVE: BFC equals B-F once the declared keys are accounted for. Anything
# else that drifted - optimizer, schedule, EMA, metrics, conditioning topology -
# would make this a two-mechanism experiment wearing a one-mechanism label.
exp = copy.deepcopy(bf)
exp["training"]["cond_method"] = "fa_cartesian"
exp["training"]["frame_avg_max_fwd_samples"] = cap
if bfc != exp:
    sys.exit("BFC != BF + {cond_method: fa_cartesian, frame_avg_max_fwd_samples: "
             f"{cap}}} (parsed-object mismatch): the arm is not a single-delta change")

# (4) announcement 06: the chunk plan is DERIVED, so derive it and pin it.
apc = max(1, cap // mb)
if apc != want_apc:
    sys.exit(f"chunk plan: angles_per_chunk = max(1, {cap}//{mb}) = {apc}, not the "
             f"declared {want_apc} - the same config at a different rung is a "
             "different method (announcement 06)")
print(f"config contract OK: BFC == BF + cond_method 'fa_cartesian' + training cap {cap}")
print(f"  ViT gradient_checkpointing: True on both ViT conditioners (config, as in the B-F recipe)")
print(f"  chunk plan: cap {cap} / micro-batch {mb} / C4 -> angles_per_chunk {apc} "
      f"(per-angle RoPE draws, matching B-F's training-era schedule)")
PY

# --- the two resource floors are PINNED in the registered run (r5 review nit) --
# The 20 GiB per-GPU floor is this launch's co-tenancy policy, and the disk floor
# sizes 16 boundary checkpoints. Both were env-overridable in every mode, which
# is exactly how a registered launch quietly proceeds onto a GPU or a volume that
# cannot hold it -- the same defect r4 fixed for MAXSTEPS/CHECKPOINT_EVERY/LOGGER,
# and it gets the same treatment: in REGISTERED live mode an override ABORTS
# rather than being silently accepted or silently ignored.
#
# The bypass survives only where it is needed and harmless: a rehearsal
# (DRY_RUN=1 / SMOKE=1) and GUARDTEST=1, which is what bfc_launch_guardtests.sh
# exports to drive these gates to their failure branches on purpose.
_floor_guard() { # $1=knob name ; refuse an override outside a rehearsal
  local name="$1" isset
  eval "isset=\${${name}+set}"
  [ "${isset:-}" = "set" ] || return 0
  [ "$DRY_RUN" = "1" ] || [ "$MODE" = "SMOKE" ] || [ "${GUARDTEST:-0}" = "1" ] || {
    eval "local val=\$${name}"
    echo "${name}='${val}' may not be overridden in the ${MODE} run: it is a resource"
    echo "  FLOOR, and lowering it is how a launch proceeds onto a GPU or volume that"
    echo "  cannot hold it. Rehearse with DRY_RUN=1 or SMOKE=1, or exercise the gate"
    echo "  itself with GUARDTEST=1. abort"
    return 1; }
  return 0
}
_floor_guard MIN_FREE_DISK_MB || exit 2
_floor_guard MIN_FREE_MB      || exit 2

# --- disk floor on the outputs volume ----------------------------------------
# 40,000 steps at one checkpoint per 2,500 = 16 boundary checkpoints of ~690 MB
# each (~11 GB) plus the final. Bypass for guard-testing only: MIN_FREE_DISK_MB=1.
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-20480}"; _posint MIN_FREE_DISK_MB "$MIN_FREE_DISK_MB" || exit 2
DF_TARGET="outputs_FLAC"; [ -d "$DF_TARGET" ] || DF_TARGET="."
DISK_FREE_MB="$(df -P -B1M "$DF_TARGET" 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')"
[ -n "$DISK_FREE_MB" ] || { echo "df query failed on ${DF_TARGET} - refusing to launch blind - abort"; exit 2; }
echo "disk: ${DISK_FREE_MB} MiB free on the volume holding ${DF_TARGET} (floor ${MIN_FREE_DISK_MB} MiB)"
[ "$DISK_FREE_MB" -ge "$MIN_FREE_DISK_MB" ] || { echo "free disk ${DISK_FREE_MB} MiB < required ${MIN_FREE_DISK_MB} MiB - refusing to launch"; exit 2; }

# --- per-GPU FREE-VRAM gate (co-tenancy policy: a FLOOR, not exclusivity) -----
MIN_FREE_MB="${MIN_FREE_MB:-20480}"; _posint MIN_FREE_MB "$MIN_FREE_MB" || exit 2
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate (only when wandb is requested) -----------
if [ "$LOGGER" = "wandb" ]; then
  # ~/.bashrc's interactive guard blocks non-interactive sourcing, so extract the
  # newest exported key directly; the gate below re-verifies it at every launch.
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

# --- DINOv3 pin + INIT-IDENTITY AUDIT (CPU-only, offline, fail-closed) --------
# Offline so the gate's own model construction cannot contact the Hub and mutate
# the cache it just validated. The pinned revision/digest are exp_07's
# (assert_arm_configs.py:56-57): the SAME initialiser B-F was trained from.
MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH" BF_CONFIG_PATH="$BF_CONFIG_PATH" \
SEED="$SEED" WANT_TRAIN_CAP="$WANT_TRAIN_CAP" \
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python3 - <<'PY' || { echo "INIT-IDENTITY AUDIT FAILED - abort"; exit 1; }
import hashlib, json, os, random, sys
sys.path.insert(0, os.getcwd())          # beat any stale pip-installed src copy
import numpy as np
import torch

VIT_REV = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
VIT_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
SEED = int(os.environ["SEED"])


def assert_vit_pin():
    """Fail-closed DINOv3 initializer pin (exp_07 audit §3/§5.iii). The ViT is
    trainable, so its init weights are lineage-relevant; with HF_HUB_OFFLINE=1 the
    run can never silently pick up a different revision. Explicit raises, and the
    cache root resolved by huggingface_hub itself (honours HF_HOME/HF_HUB_CACHE
    exactly as the transformers loader does)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    snap_dir = os.path.join(HF_HUB_CACHE,
                            "models--facebook--dinov3-vits16-pretrain-lvd1689m", "snapshots")
    if not os.path.isdir(snap_dir):
        raise RuntimeError(f"DINOv3 cache missing at {snap_dir} - refuse to launch")
    snaps = sorted(os.listdir(snap_dir))
    if snaps != [VIT_REV]:
        raise RuntimeError(f"DINOv3 cache snapshots {snaps} != pinned [{VIT_REV!r}] - refuse to launch")
    h = hashlib.sha256()
    with open(os.path.join(snap_dir, VIT_REV, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != VIT_SHA256:
        raise RuntimeError(f"DINOv3 model.safetensors sha256 {h.hexdigest()} != pinned {VIT_SHA256}")
    print(f"ViT pin OK: single snapshot {VIT_REV[:12]}..., sha256 {VIT_SHA256[:12]}... "
          "(launched with HF_HUB_OFFLINE=1)")


from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config


def build(path):
    # identical RNG state before EACH build -> init weights must match across arms
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    cfg = json.load(open(path))
    model = create_model_from_config(cfg)
    return model, create_training_wrapper_from_config(cfg, model)


def state_hash(model):
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


assert_vit_pin()
bfc_path, bf_path = os.environ["MODEL_CONFIG_PATH"], os.environ["BF_CONFIG_PATH"]
print(f"building BFC (fa_cartesian) from {bfc_path} ...")
model_c, wrap_c = build(bfc_path)
print(f"building BF  (fa_invariant) from {bf_path} ...")
model_f, wrap_f = build(bf_path)

# --- the wiring each config actually produces (factory output, not a JSON re-read)
if wrap_c.cond_method != "fa_cartesian":
    sys.exit(f"BFC wrapper cond_method is {wrap_c.cond_method!r}")
if wrap_f.cond_method != "fa_invariant":
    sys.exit(f"BF wrapper cond_method is {wrap_f.cond_method!r} - the reference arm changed")
if tuple(wrap_c.frame_avg_angles) != (0.0, 90.0, 180.0, 270.0):
    sys.exit(f"BFC frame_avg_angles {wrap_c.frame_avg_angles} is not the C4 orbit")
cap = int(os.environ["WANT_TRAIN_CAP"])
if getattr(wrap_c, "frame_avg_max_fwd_samples", None) != cap:
    sys.exit(f"BFC wrapper cap {getattr(wrap_c, 'frame_avg_max_fwd_samples', None)!r} != {cap}")
print(f"wiring: BFC cond_method={wrap_c.cond_method!r} angles={tuple(wrap_c.frame_avg_angles)} "
      f"cap={getattr(wrap_c, 'frame_avg_max_fwd_samples', None)} | "
      f"BF cond_method={wrap_f.cond_method!r} cap={getattr(wrap_f, 'frame_avg_max_fwd_samples', None)} "
      "(B-F declared no cap: its training-era schedule was the legacy per-angle loop)")

# --- identical architecture, then identical INITIAL WEIGHTS under one seed -----
names_c = [n for n, _ in model_c.named_parameters()]
names_f = [n for n, _ in model_f.named_parameters()]
count_c = sum(p.numel() for p in model_c.parameters())
count_f = sum(p.numel() for p in model_f.parameters())
if names_c != names_f:
    sys.exit("parameter-name sets differ between BFC and BF")
if count_c != count_f:
    sys.exit(f"parameter counts differ: BFC {count_c} vs BF {count_f}")
hc, hf = state_hash(model_c), state_hash(model_f)
if hc != hf:
    sys.exit(f"INIT STATE HASHES DIFFER:\n  BFC {hc}\n  BF  {hf}\n"
             "  the two configs no longer describe the same initialisation, so this "
             "is not a single-delta comparison - abort")
print(f"architecture: identical param names ({len(names_c)} tensors) and count "
      f"({count_c/1e6:.2f}M) in both arms")
print(f"init identity: state_dict sha256 match under seed {SEED}: {hc[:16]}...")
print("  (covers every module create_model_from_config builds, the VAE pretransform "
      "module included - randomly initialised here; the TRAINED VAE weights arrive at "
      "train time via --pretransform-ckpt-path, from the same file for both arms)")
PY

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader 2>/dev/null || true
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true (fail-closed in train.py below num_gpus 2) | strategy: ${STRATEGY} | rung: ${MB}x${NUM_GPUS}x${ACC} | grad-ckpt: config"

# --- the training command, assembled ONCE and printed verbatim ----------------
# One array, one print, one execution: what the log shows is what runs. The
# LAUNCH-CMD line is the machine-readable form the guardtests assert against.
CMD=(env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py
  --model-config "$MODEL_CONFIG_PATH"
  --dataset-config "$DATASET_CONFIG"
  --pretransform-ckpt-path "$PRETRANSFORM"
  --max-steps "$MAXSTEPS" --batch-size "$MB" --accum-batches "$ACC"
  --num-workers "$NUM_WORKERS" --seed "$SEED"
  --num-gpus "$NUM_GPUS" --strategy "$STRATEGY" --sync-batchnorm true
  --precision "$PRECISION"
  --logger "$LOGGER" --checkpoint-every "$CHECKPOINT_EVERY"
  --name "$NAME" --experiment-name "$EXPNAME"
  --save-dir "$SAVEDIR")
echo "LAUNCH-CMD: ${CMD[*]}"
# ...and the same argv one token per line, which is what the guardtests compare
# against the approved manifest. A space-joined line cannot distinguish a token
# that was split, merged or quoted differently; this can.
echo "LAUNCH-ARGV-BEGIN"
printf '%s\n' "${CMD[@]}"
echo "LAUNCH-ARGV-END"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN (${MODE}): every gate above ran; train.py was NOT executed and no checkpoint directory was created."
  exit 0
fi

"${CMD[@]}"
rc=$?
echo "=== exp_21 BFC (${MODE}) training exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="
exit $rc
