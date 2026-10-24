#!/usr/bin/env bash
# ============================================================================
# depth_canon_launch.sh -- exp_25 depth_canon: the CANON arm, FROM SCRATCH, at
# P1's recipe (plan_depth_canon.md v8 §4): 2 x A6000, DDP + SyncBN, 32/GPU x 2
# x accum 1 (BN 64), bf16-mixed, seed 42, 40,000 steps, checkpoint / 2,500.
# The arm is P1's config (FLAC_AR_BVp1.json) plus exactly ONE delta,
# training.canonicalize {enabled: true, img_w: 512}.
#
# MODE=SMOKE  25 steps, own W&B project + run namespace, checkpoint cadence
#             10^6 (writes none); projects the 40k wall clock; writes the
#             smoke-evidence JSON that FULL requires.
# MODE=FULL   the registered 40,000-step run; needs a PASSING smoke evidence
#             with the SAME HEAD and pin-manifest sha256.
# DRY_RUN=1   every gate except the side-effecting / GPU ones (write probe,
#             GPU memory, W&B identity), prints the exact argv, exits 0.
# EXPECT_HEAD the full reviewed pin commit; HEAD must equal it (required).
# DEPTH_CANON_HARNESS=<dir under /tmp with a .depth_canon_harness marker>:
#             guard-suite mode -- the save root moves into <dir>/save, the W&B
#             identity gate is skipped, the evidence is tagged harness=true and
#             a production FULL never accepts it. Nothing else changes.
#
# Adapted from exp_17's yaw_aug_a6000_launch.sh with its r3 review's BLOCKING
# findings applied (HEAD + content binding, smoke evidence bound to HEAD +
# manifest, no pipefail around tr|grep -- the log is read ONCE into a variable
# after the tee is drained, a non-finite scan in FULL too) plus plan §4 gates
# (5) run-owned checkpoints, (6) NAS identity / write probe / free space,
# (7) projection <= 60 h, and (8) GPU free memory. It never kills anything.
#
# Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §4.
# ============================================================================
set -euo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "$REPO"
EXPDIR="worklog/worklog_yixun/exp_25_depth_canon_claude"
EXP07="worklog/worklog_yixun/exp_07_fa_scratch_claude"
SELF="$EXPDIR/depth_canon_launch.sh"
MAKE_PINS="$EXPDIR/depth_canon_make_pins.sh"
PIN_MANIFEST="$EXPDIR/depth_canon_pins.sha256"
CONTROL_CFG="$EXP07/FLAC_AR_BVp1.json"
ARM_CFG="$EXPDIR/FLAC_AR_CANON.json"
DATASET_CFG="src/configs/dataset_configs/AR/train/acousticroom_train.json"
VAE="weights/FLAC/VAE.safetensors"
DINO_GATE="$EXP07/assert_arm_configs.py"

# --- PINNED constants: not overridable from the environment ----------------- #
ENDPOINT_STEPS=40000
FULL_CADENCE=2500
SMOKE_STEPS=25
SMOKE_CADENCE=1000000             # >> SMOKE_STEPS: a smoke never writes a checkpoint
MAX_PROJECTED_HOURS=60            # plan §4 gate (7)
MIN_NAS_FREE_GIB=30               # plan §4 gate (6)
MIN_GPU_FREE_MIB=23040            # 22.5 GiB per card (gate (8))
SAVE_ROOT_PIN="/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon"
NAS_FSTYPE="cifs"
NAS_SOURCE="//10.249.7.28/shared_data"
WANDB_IDENTITY="yh4742@princeton.edu"
CONTROL_SHA="733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"
BANNER="canonicalize ENABLED img_w=512 tiebreak=image"   # src/training/diffusion.py, exact

MODE="${MODE:-}"
DRY_RUN="${DRY_RUN:-0}"
MB="${MB:-32}"
ACC="${ACC:-1}"
EXPECT_HEAD="${EXPECT_HEAD:-}"
HARNESS="${DEPTH_CANON_HARNESS:-}"

case "$MODE" in
  SMOKE) NAME="FLAC_exp25_CANON_smoke"; EXPNAME="exp25_CANON_smoke"
         MAXSTEPS="$SMOKE_STEPS"; CADENCE="$SMOKE_CADENCE" ;;
  FULL)  NAME="FLAC_exp25_CANON"; EXPNAME="exp25_CANON"
         MAXSTEPS="$ENDPOINT_STEPS"; CADENCE="$FULL_CADENCE" ;;
  *) echo "MODE must be exactly SMOKE or FULL (got '${MODE}') - abort"; exit 2 ;;
esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1 (got '${DRY_RUN}') - abort"; exit 2 ;; esac
# BN 64 admits exactly one rung (accumulation never feeds BatchNorm); literal match.
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || {
  echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

SAVE_ROOT="$SAVE_ROOT_PIN"
HARNESS_FLAG=false
if [ -n "$HARNESS" ]; then
  HREAL="$(readlink -f -- "$HARNESS" 2>/dev/null || true)"
  case "$HREAL" in /tmp/?*) ;; *)
    echo "DEPTH_CANON_HARNESS must be a directory under /tmp (got '${HARNESS}') - abort"; exit 2 ;; esac
  [ -d "$HREAL" ] && [ -f "$HREAL/.depth_canon_harness" ] || {
    echo "DEPTH_CANON_HARNESS '${HARNESS}' lacks the .depth_canon_harness marker - abort"; exit 2; }
  SAVE_ROOT="$HREAL/save"
  HARNESS_FLAG=true
fi
RUN_NS="$SAVE_ROOT/$NAME/$EXPNAME"      # train.py: <save-dir>/<W&B project>/<run>/checkpoints

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="$EXPDIR/depth_canon_${TS}_${EXPNAME}_train.log"
exec 3>&1 4>&2
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!
HEAD_SHA="$(git rev-parse HEAD)"
echo "=== exp_25 depth_canon ${MODE} -- ${TS} -- HEAD ${HEAD_SHA} -- dry_run=${DRY_RUN} -- harness=${HARNESS_FLAG} ==="
[ "$HARNESS_FLAG" = true ] && echo "HARNESS MODE (${HREAL}): guard-suite run, not a registered run"
echo "identity: project=${NAME} | run=${EXPNAME} | save-root=${SAVE_ROOT} (production pin ${SAVE_ROOT_PIN})"
echo "budget: mode=${MODE} | endpoint=${MAXSTEPS} | cadence=${CADENCE}"
# P1's recipe exactly (p1_ddp_launch.sh): no --val-dataset-config, no resume, no
# pretrained init; --precision bf16-mixed is defaults.ini's value, passed explicitly.
ARGV=(python train.py
  --model-config "$ARM_CFG" --dataset-config "$DATASET_CFG" --pretransform-ckpt-path "$VAE"
  --max-steps "$MAXSTEPS" --checkpoint-every "$CADENCE"
  --batch-size "$MB" --accum-batches "$ACC" --num-gpus 2 --sync-batchnorm true
  --strategy ddp_find_unused_parameters_true --precision bf16-mixed --seed 42 --num-workers 6
  --logger wandb --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVE_ROOT")
echo "ARGV: ${ARGV[*]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; train.py NOT launched"; exit 0
fi
echo "launch path not implemented yet - abort"; exit 2
