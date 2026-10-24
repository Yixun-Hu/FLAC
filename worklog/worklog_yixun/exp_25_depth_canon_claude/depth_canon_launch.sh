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

# --- (1) reviewed-tree binding: HEAD ----------------------------------------- #
if ! [[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECT_HEAD must be the full 40-hex reviewed pin commit (got '${EXPECT_HEAD}') - abort"; exit 2
fi
[ "$HEAD_SHA" = "$EXPECT_HEAD" ] || {
  echo "HEAD PIN FAILED: HEAD ${HEAD_SHA} != EXPECT_HEAD ${EXPECT_HEAD} - abort"; exit 2; }
echo "HEAD pin OK: ${HEAD_SHA}"

# --- (1) content pins: coverage, bytes, committed ---------------------------- #
# The manifest's path column must equal the generator's list (so a new module on
# the training path cannot go unpinned), every byte must match (sha256sum -c also
# covers the gitignored VAE), and every pinned path plus the launcher kit must be
# committed at HEAD (so HEAD names exactly this content).
EXPECTED_LIST="$(bash "$MAKE_PINS" --list)" || {
  echo "PIN MANIFEST FAILED: the generator could not list the training path - abort"; exit 2; }
[ -f "$PIN_MANIFEST" ] || { echo "PIN MANIFEST FAILED: ${PIN_MANIFEST} is missing - abort"; exit 2; }
MANIFEST_LIST="$(awk '{print $2}' "$PIN_MANIFEST")"
if [ "$EXPECTED_LIST" != "$MANIFEST_LIST" ]; then
  echo "PIN MANIFEST FAILED: coverage -- the manifest's paths differ from the training path (regenerate it at the pin commit):"
  diff <(printf '%s\n' "$MANIFEST_LIST") <(printf '%s\n' "$EXPECTED_LIST") | head -20 || true
  exit 2
fi
sha256sum -c --strict --quiet "$PIN_MANIFEST" || {
  echo "PIN MANIFEST FAILED: content -- a pinned file differs from ${PIN_MANIFEST} - abort"; exit 2; }
mapfile -t PINNED <<<"$EXPECTED_LIST"
DIRTY="$(git status --porcelain --untracked-files=all -- "${PINNED[@]}" "$PIN_MANIFEST" "$MAKE_PINS" "$SELF")"
if [ -n "$DIRTY" ]; then
  echo "PINNED TREE NOT COMMITTED at HEAD (commit or restore, then regenerate the manifest):"
  printf '%s\n' "$DIRTY" | head -20 || true
  exit 2
fi
MANIFEST_SHA="$(sha256sum "$PIN_MANIFEST" | awk '{print $1}')"
echo "pin manifest OK: ${#PINNED[@]} files match ${PIN_MANIFEST} (sha256 ${MANIFEST_SHA}); pinned paths + launcher kit committed at HEAD"

# --- config contract: the P1 control plus exactly training.canonicalize -------- #
python - "$ARM_CFG" "$CONTROL_CFG" "$CONTROL_SHA" <<'PY' || { echo "CONFIG CONTRACT FAILED - abort"; exit 2; }
import hashlib, json, sys
arm_p, ctl_p, ctl_sha = sys.argv[1:4]
ctl_raw, arm_raw = open(ctl_p, "rb").read(), open(arm_p, "rb").read()
got = hashlib.sha256(ctl_raw).hexdigest()
if got != ctl_sha:
    sys.exit(f"control config sha256 {got} != the pinned P1 config {ctl_sha}")
arm, ctl = json.loads(arm_raw), json.loads(ctl_raw)

def strict(a, b, path="root"):   # type-strict, key ORDER included (1 == True, 0 == 0.0 in Python)
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys {list(a)} != {list(b)}"
        return next((r for k in a for r in [strict(a[k], b[k], f"{path}.{k}")] if r), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        return next((r for i, (x, y) in enumerate(zip(a, b)) for r in [strict(x, y, f"{path}[{i}]")] if r), None)
    return None if a == b else f"{path}: {a!r} != {b!r}"

t = arm["training"]
block = t.get("canonicalize")
if block != {"enabled": True, "img_w": 512}:
    sys.exit(f"canonicalize block is not the registered treatment: {block!r}")
if type(block["enabled"]) is not bool or type(block["img_w"]) is not int:
    sys.exit(f"canonicalize block has the wrong TYPES: {block!r}")
if "cond_method" in t:
    sys.exit("this arm must be vanilla-conditioned; cond_method is present")
if "yaw_aug" in t:
    sys.exit("yaw_aug must be absent (a second treatment)")
if t.get("use_ema") is not True:
    sys.exit("use_ema must be true (matched to P1)")
if list(t)[-1] != "canonicalize":
    sys.exit(f"canonicalize must be the last key of training (got {list(t)[-1]!r})")
vits = [c for c in arm["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if len(vits) != 2:
    sys.exit(f"expected 2 ViT conditioners, found {len(vits)}")
for c in vits:
    if c["config"].get("gradient_checkpointing") is not True:
        sys.exit(f"gradient_checkpointing must be true on {c['id']} (P1's recipe)")
widths = {c["config"]["ViT"]["img_w"] for c in vits}
if widths != {block["img_w"]}:
    sys.exit(f"ViT widths {widths} != canonicalize.img_w {block['img_w']}: the canonical shift "
             "would be computed on the wrong column grid")
stripped = json.loads(arm_raw)
stripped["training"].pop("canonicalize")
d = strict(stripped, ctl)
if d:
    sys.exit(f"arm config is NOT the control plus the one registered delta - {d}")
INSERT = (b',\n        "canonicalize": {\n            "enabled": true,\n            "img_w": 512\n        }')
TRAILER = b'\n    }\n}'
if not (ctl_raw.endswith(TRAILER) and ctl_raw.count(TRAILER) == 1
        and arm_raw == ctl_raw[:-len(TRAILER)] + INSERT + TRAILER):
    sys.exit("arm config is not the control's bytes plus the literal insertion (formatting / newline drift)")
print(f"config contract OK: {arm_p} == {ctl_p} (sha {ctl_sha[:12]}...) + training.canonicalize {block}")
PY

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; train.py NOT launched"; exit 0
fi
echo "launch path not implemented yet - abort"; exit 2
