#!/usr/bin/env bash
# ============================================================================
# haa_ft_launch.sh — exp_19: HAA finetuning of three 40k AR inits.
#
#   ARM={P1|BF|YAW} GPU={0|1} MODE={SMOKE|FULL} EXPECT_SHA=<HEAD> bash haa_ft_launch.sh
#
# One arm, one card, the released HAA recipe (README "Finetuning on HAA"):
# 1,000 steps, batch 16 x accum 4 = eff 64, AdamW 5e-6 + InverseLR, VAE frozen,
# init from EMA weights via --pretrained-ckpt-path (a WEIGHTS-ONLY load into a
# fresh wrapper — which is why this sidesteps the repo's scheduler-clobber trap).
# The three arms differ ONLY in (a) which 40k checkpoint they start from and
# (b) the matching training-time treatment their config carries:
#
#   P1  vanilla        stock config, used DIRECTLY (no copy can drift from it)
#   BF  fa_invariant   stock + exactly B-F's two AR deltas
#   YAW yaw_aug        stock + exactly exp_17's treatment block
#
# MODE=SMOKE — 20 steps in its OWN namespace, cadence >> endpoint so it can never
#              write a checkpoint (asserted afterwards).
# MODE=FULL  — the registered 1,000-step endpoint, checkpoints every 10 so BOTH
#              the step-410 and step-1000 readings exist (plan B1).
#
# DRY_RUN=0  — production. EXPECT_SHA is required for FULL, every test override is
#              refused, and the resource floors cannot be lowered.
# DRY_RUN=1  — run every gate, print the exact train.py argv, exit 0 before any
#              side effect beyond the log. The guard suite asserts against the
#              REAL argv, never a preflight paraphrase.
# DRY_RUN=2  — REHEARSAL: every gate, then TRAIN_CMD in place of train.py, then
#              the complete post-run verdict block. Exists so the post-run gates
#              (endpoint, banner, NaN, 410/1000, smoke-ckpt, rc propagation) are
#              EXECUTED by the guard suite instead of grepped for. Requires
#              TRAIN_CMD and REHEARSAL_DIR, so its checkpoints and logs can never
#              land in a production namespace.
#
# Lessons carried from the exp_17 lineage and the exp_19 Codex reviews:
#  * post-run log checks never use `tr | grep -q` — under `pipefail` a `-q` early
#    exit SIGPIPEs `tr` and the pipeline reports failure after a successful match
#    (exp_17 r3). The log is normalised to a file by REDIRECTION and grepped.
#  * the endpoint marker uses Lightning's REAL framing: tqdm leaves the progress
#    line unterminated, so the marker is APPENDED to it — matched as
#    line-ENDS-WITH (src/tools/exp17_full_audit.py), not whole-line, not substring.
#  * the treatment banner is matched as a WHOLE line, and the preflight never
#    prints those words, so the check cannot satisfy itself.
#  * NaN/Inf is checked in BOTH modes (exp_17 r3).
#  * train.py's output goes through a SYNCHRONOUS pipe to its own run log, so the
#    post-run reader cannot race an undrained async `tee`.
#  * r2-B1 revision + split binding: EXPECT_SHA, and the two data/HAA split files
#    are pinned AND inside the clean-tree closure.
#  * r2-B2 the per-arm and per-GPU locks are held for the WHOLE run.
#  * r2-B6 the init sha is re-validated immediately before exec.
#
# Written by the exp_19 coder seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || { echo "cannot reach repo root - abort"; exit 2; }
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"

# --- inputs ------------------------------------------------------------------ #
ARM="${ARM:-}"
GPU="${GPU:-}"
MODE="${MODE:-}"
DRY_RUN="${DRY_RUN:-0}"
LOGGER="${LOGGER:-wandb}"
EXPECT_SHA="${EXPECT_SHA:-}"

# Floors. Raising them from the environment is allowed; LOWERING them is not
# (r2 finding: they were overridable down to zero, so the floors were bypassable).
# Clamped and disclosed below; DRY_RUN is exempt because it never allocates.
DEFAULT_MIN_FREE_MB=20000
MIN_FREE_MB="${MIN_FREE_MB:-$DEFAULT_MIN_FREE_MB}"
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-}"

# Test-only overrides. Each substitutes EVIDENCE for a gate, so each is refused
# in production (DRY_RUN=0) and disclosed loudly in the log when used.
MANIFEST_DEFAULT="${EXPDIR}/exp19_init_shas.txt"
INIT_DIR_DEFAULT="outputs_FLAC/exp19_inits"
MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
INIT_DIR="${INIT_DIR:-$INIT_DIR_DEFAULT}"
PROBE_CMD="${PROBE_CMD:-}"
# Substitutes gate 2's ARM-CONFIG pin only. Without it the config-contract gate
# is unreachable by mutation — every edit to an arm config trips the byte pin
# first — and an unreachable gate is a gate nobody has ever tested.
ARM_CFG_SHA="${ARM_CFG_SHA:-}"
# Substitutes the dataset ROOT for the inventory gate, so the guard suite can
# exercise it without a prepared 50 GB HAA tree.
HAA_ROOT="${HAA_ROOT:-}"
# DRY_RUN=2 only: what runs in place of train.py, and where its artifacts go.
TRAIN_CMD="${TRAIN_CMD:-}"
REHEARSAL_DIR="${REHEARSAL_DIR:-}"
# The guard suite holds the gate lock for its whole duration so that no
# production launcher can be mid-gates while an arm config is momentarily
# mutated (r2-B3). Its own launcher children must therefore be told the caller
# already holds it — otherwise they would deadlock against their own suite.
GATE_LOCK_HELD_BY_CALLER="${GATE_LOCK_HELD_BY_CALLER:-}"

# --- pinned constants: not overridable from the environment ------------------ #
FULL_STEPS=1000               # the released HAA recipe's endpoint
FULL_CADENCE=10               # README; also what makes step-410 exist (plan B1)
FULL_VAL_EVERY=10             # README
MID_STEPS=410                 # the second registered reading (plan B1)
SMOKE_STEPS=20
SMOKE_CADENCE=1000000         # >> SMOKE_STEPS: smoke must never write a ckpt
SMOKE_VAL_EVERY=1000000
BATCH=16                      # 16 x accum 4 = eff 64 (README)
ACCUM=4
NUM_WORKERS=8
SEED=42
PROBE_THRESHOLD=1e-7   # R1 gate re-parameterized per Yixun 2026-08-18 adjudication:
                       # fp32@1e-5 measured precision noise (3e-5 on three unrelated
                       # inits); fp64 measures 5.8e-14. float64@1e-7 is the calibrated,
                       # stricter-in-meaning gate (exactness to fp64 rounding).
BANNER="yaw_aug ENABLED img_w=512 seed=42"   # EXACT text of diffusion.py:406-408
WANDB_IDENTITY="yh4742@princeton.edu"

DATASET_CFG="src/configs/dataset_configs/HAA/train/haa_train.json"
VAL_DATASET_CFG="src/configs/dataset_configs/HAA/eval/haa_val.json"
STOCK_CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
VAE="weights/FLAC/VAE.safetensors"
PROBE_SCRIPT="${EXPDIR}/probe_haa_fa_invariance.py"

# The invariants the cadence pins depend on, asserted rather than assumed.
[ "$SMOKE_STEPS" -ge 1 ] && [ "$SMOKE_STEPS" -lt "$SMOKE_CADENCE" ] || {
  echo "SMOKE_STEPS=${SMOKE_STEPS} must satisfy 1 <= steps < cadence ${SMOKE_CADENCE} - abort"; exit 2; }
[ $(( MID_STEPS % FULL_CADENCE )) -eq 0 ] && [ "$MID_STEPS" -lt "$FULL_STEPS" ] || {
  echo "step ${MID_STEPS} is not on the cadence-${FULL_CADENCE} grid below ${FULL_STEPS} - abort"; exit 2; }

# --- gate 1: ARM / GPU / MODE / DRY_RUN are matched EXACTLY, never inferred --- #
case "$ARM" in
  P1|BF|YAW|YNA|BNA) ;;
  *) echo "ARM must be exactly P1, BF, YAW, YNA or BNA (got '${ARM}') - abort"; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "GPU must be exactly 0 or 1 (got '${GPU}') - abort"; exit 2 ;;
esac
case "$MODE" in
  SMOKE|FULL) ;;
  *) echo "MODE must be exactly SMOKE or FULL (got '${MODE}') - abort"; exit 2 ;;
esac
case "$DRY_RUN" in
  0|1|2) ;;
  *) echo "DRY_RUN must be exactly 0 (production), 1 (gate boundary) or 2 (post-run rehearsal), got '${DRY_RUN}' - abort"; exit 2 ;;
esac

# --- gate 1b: the test overrides are refused in PRODUCTION -------------------- #
# Checked BEFORE the EXPECT_SHA gate on purpose: the guard suite asserts these
# messages, and EXPECT_SHA is what independently stops such a case from ever
# reaching training if one of these refusals were deleted (r2-B5).
if [ "$DRY_RUN" = "0" ]; then
  [ -z "$PROBE_CMD" ] || { echo "PROBE_CMD is a DRY_RUN-only test override and is set ('${PROBE_CMD}') - a real launch must run the REAL R1 probe - abort"; exit 2; }
  [ -z "$ARM_CFG_SHA" ] || { echo "ARM_CFG_SHA is a DRY_RUN-only test override and is set ('${ARM_CFG_SHA}') - a real launch must match the hard-coded arm-config pin - abort"; exit 2; }
  [ -z "$HAA_ROOT" ] || { echo "HAA_ROOT is a DRY_RUN-only test override and is set ('${HAA_ROOT}') - a real launch must validate the REAL dataset root - abort"; exit 2; }
  [ -z "$TRAIN_CMD" ] || { echo "TRAIN_CMD is a DRY_RUN-only test override and is set ('${TRAIN_CMD}') - a real launch must run train.py - abort"; exit 2; }
  [ -z "$REHEARSAL_DIR" ] || { echo "REHEARSAL_DIR is a DRY_RUN-only test override and is set ('${REHEARSAL_DIR}') - abort"; exit 2; }
  [ -z "$GATE_LOCK_HELD_BY_CALLER" ] || { echo "GATE_LOCK_HELD_BY_CALLER is a DRY_RUN-only test override and is set - a real launch must acquire the gate lock itself - abort"; exit 2; }
  [ "$MANIFEST" = "$MANIFEST_DEFAULT" ] || { echo "MANIFEST is a DRY_RUN-only test override and points at '${MANIFEST}' instead of ${MANIFEST_DEFAULT} - abort"; exit 2; }
  [ "$INIT_DIR" = "$INIT_DIR_DEFAULT" ] || { echo "INIT_DIR is a DRY_RUN-only test override and points at '${INIT_DIR}' instead of ${INIT_DIR_DEFAULT} - abort"; exit 2; }
fi
if [ "$DRY_RUN" = "2" ]; then
  [ -n "$TRAIN_CMD" ] || { echo "DRY_RUN=2 (rehearsal) requires TRAIN_CMD - abort"; exit 2; }
  [ -d "$REHEARSAL_DIR" ] || { echo "DRY_RUN=2 (rehearsal) requires REHEARSAL_DIR to be an existing directory (got '${REHEARSAL_DIR}') - its artifacts must never land in a production namespace - abort"; exit 2; }
fi

# --- gate 1c: revision binding (r2-B1) --------------------------------------- #
# HEAD is RECORDED in every mode and BOUND in production FULL. The exp_17 r3 debt
# was that a clean commit carrying changes to unpinned closure code passed every
# gate; requiring the operator to name the revision they reviewed closes it, and
# the closure-clean check below is what makes the name mean anything.
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
if [ "$DRY_RUN" = "0" ] && [ "$MODE" = "FULL" ]; then
  [ -n "$EXPECT_SHA" ] || {
    echo "EXPECT_SHA is REQUIRED for a production FULL launch: name the reviewed revision (git rev-parse HEAD) so the run is bound to it - abort"; exit 2; }
  [ -n "$HEAD_SHA" ] || { echo "git rev-parse HEAD failed - refusing to launch an unidentifiable revision - abort"; exit 2; }
  [ "$EXPECT_SHA" = "$HEAD_SHA" ] || {
    echo "EXPECT_SHA=${EXPECT_SHA} != HEAD ${HEAD_SHA} - this checkout is not the revision you reviewed - abort"; exit 2; }
fi

case "$ARM" in
  P1)  ARM_CFG="$STOCK_CFG" ;;
  YNA) ARM_CFG="$STOCK_CFG" ;;  # ablation arm (Yixun 2026-08-19): YAW's INIT, stock finetune — separates "aug-during-FT burden" from "YAW representation transfers worse"
  BNA) ARM_CFG="$STOCK_CFG" ;;  # ablation arm (Yixun 2026-08-20): B-F's INIT, stock (vanilla) finetune — can FA shed its architectural tax at adaptation, mirroring YNA? Risk disclosed: BF's DiT was trained on ORBIT-AVERAGED conditioning, so single-orientation conditioning is an interface shift YNA never faced
  BF)  ARM_CFG="${EXPDIR}/FLAC_HAA_finetune_BF.json" ;;
  YAW) ARM_CFG="${EXPDIR}/FLAC_HAA_finetune_YAW.json" ;;
esac
INIT_ARM="$ARM"
[ "$ARM" = "YNA" ] && INIT_ARM="YAW"   # YNA = YAW's init + stock finetune
[ "$ARM" = "BNA" ] && INIT_ARM="BF"    # BNA = B-F's init + stock finetune
INIT="${INIT_DIR}/HAA_init_${INIT_ARM}.ckpt"

case "$MODE" in
  SMOKE) SUFFIX="_smoke"; MAXSTEPS="$SMOKE_STEPS"; CADENCE="$SMOKE_CADENCE"; VALEVERY="$SMOKE_VAL_EVERY"
         DEFAULT_MIN_FREE_DISK_MB=4096 ;;
  FULL)  SUFFIX="";       MAXSTEPS="$FULL_STEPS";  CADENCE="$FULL_CADENCE";  VALEVERY="$FULL_VAL_EVERY"
         # 100 checkpoints x ~690 MiB each (measured on the 40k artifacts) is
         # ~69 GiB PER ARM. This floor is not decoration: at cadence 10 the
         # deliverable is two orders of magnitude larger than a normal run's.
         DEFAULT_MIN_FREE_DISK_MB=72000 ;;
esac
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-$DEFAULT_MIN_FREE_DISK_MB}"

NAME="FLAC_exp19_HAA_${ARM}${SUFFIX}"
EXPNAME="exp19_HAA_${ARM}${SUFFIX}"
if [ "$DRY_RUN" = "2" ]; then
  SAVEDIR="${REHEARSAL_DIR}/exp19_HAA_${ARM}${SUFFIX}"      # never a production namespace
else
  SAVEDIR="outputs_FLAC/exp19_HAA_${ARM}${SUFFIX}"
fi

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
# DRY output is NOT production evidence and never lands in the evidence
# directory. A PER-INVOCATION mktemp dir, not a shared one: the guard suite used
# to delete a shared `.dryrun_logs`, which could erase a concurrent dry run's log.
case "$DRY_RUN" in
  0) LOGDIR="$EXPDIR"; LOG="${EXPDIR}/haa_ft_${TS}_${ARM}_${MODE}.log" ;;
  1) LOGDIR="$(mktemp -d -t haa_ft_dryrun.XXXXXXXX)" || { echo "mktemp failed - abort"; exit 2; }
     LOG="${LOGDIR}/haa_ft_${TS}_${ARM}_${MODE}_dryrun.log" ;;
  2) LOGDIR="${REHEARSAL_DIR}"; LOG="${LOGDIR}/haa_ft_${TS}_${ARM}_${MODE}_rehearsal.log" ;;
esac
RUNLOG="${LOGDIR}/haa_ft_${TS}_${ARM}_${MODE}_train.log"

# --- locks ------------------------------------------------------------------- #
# (r2-B2) The per-arm and per-GPU locks are held for the WHOLE RUN, not just the
# gate phase: before the first checkpoint exists, the namespace-occupancy check
# cannot see a sibling launch, so two same-arm FULL runs could both pass and then
# write into one SAVEDIR — and two arms could both claim one card during the
# other's startup window. These fds are deliberately NOT closed before train.py:
# bash keeps them open across the child, the child inherits them, and the locks
# therefore stay held for as long as the run lives. That inheritance is the
# mechanism, not an accident.
ARM_LOCK="${EXPDIR}/.haa_ft_${ARM}.lock"
GPU_LOCK="${EXPDIR}/.haa_gpu${GPU}.lock"
exec 8>"$ARM_LOCK" || { echo "cannot open ${ARM_LOCK} - abort"; exit 2; }
flock -n 8 || { echo "arm ${ARM} is already running or gating (${ARM_LOCK} held) - abort"; exit 2; }
exec 7>"$GPU_LOCK" || { echo "cannot open ${GPU_LOCK} - abort"; exit 2; }
flock -n 7 || { echo "GPU ${GPU} is already reserved by another exp_19 run (${GPU_LOCK} held) - abort"; exit 2; }

# The shared gate lock stays GATE-PHASE-SCOPED: the gates read shared state (the
# manifest, the namespace, one free-VRAM snapshot per card) and two launchers
# interleaving through them could both pass a floor only one can satisfy — but
# the arms are MEANT to train concurrently, one per card, so it is released
# before train.py. It is also the lock the guard suite holds while an arm config
# is momentarily mutated, which is why a caller may legitimately already own it.
LOCK="${EXPDIR}/.haa_ft.lock"
if [ -n "$GATE_LOCK_HELD_BY_CALLER" ]; then
  GATE_LOCK_OWNED=0
else
  exec 9>"$LOCK" || { echo "cannot open ${LOCK} - abort"; exit 2; }
  flock -n 9 || { echo "another exp_19 launcher holds ${LOCK} (gate phase) - abort"; exit 2; }
  GATE_LOCK_OWNED=1
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_19 HAA finetune | ARM=${ARM} MODE=${MODE} GPU=${GPU} | ${TS} | HEAD ${HEAD_SHA:-unknown} | dry_run=${DRY_RUN} logger=${LOGGER} ==="
echo "identity: name=${NAME} | experiment=${EXPNAME} | save-dir=${SAVEDIR}"
echo "budget:   endpoint=${MAXSTEPS} | cadence=${CADENCE} | val-every=${VALEVERY} | batch=${BATCH}x accum ${ACCUM} = eff $((BATCH*ACCUM))"
echo "config:   ${ARM_CFG}"
echo "init:     ${INIT}"
echo "locks:    ${ARM_LOCK} + ${GPU_LOCK} held for the whole run; gate lock owned=${GATE_LOCK_OWNED}"
[ "$DRY_RUN" = "0" ] && [ "$MODE" = "FULL" ] && echo "revision: EXPECT_SHA=${EXPECT_SHA} matches HEAD"
if [ "$DRY_RUN" != "0" ]; then
  echo "dry-run log dir: ${LOGDIR}"
  [ -n "$PROBE_CMD" ] && echo "!! TEST OVERRIDE ACTIVE: PROBE_CMD='${PROBE_CMD}' (the R1 gate is STUBBED; this run proves nothing about invariance)"
  [ "$MANIFEST" != "$MANIFEST_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: MANIFEST='${MANIFEST}'"
  [ "$INIT_DIR" != "$INIT_DIR_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: INIT_DIR='${INIT_DIR}'"
  [ -n "$ARM_CFG_SHA" ] && echo "!! TEST OVERRIDE ACTIVE: ARM_CFG_SHA='${ARM_CFG_SHA}' (gate 2's arm-config pin is SUBSTITUTED)"
  [ -n "$HAA_ROOT" ] && echo "!! TEST OVERRIDE ACTIVE: HAA_ROOT='${HAA_ROOT}' (the dataset inventory is checked against a FIXTURE)"
  [ -n "$TRAIN_CMD" ] && echo "!! TEST OVERRIDE ACTIVE: TRAIN_CMD='${TRAIN_CMD}' (train.py is NOT run; the post-run gates are being rehearsed)"
  [ -n "$GATE_LOCK_HELD_BY_CALLER" ] && echo "!! TEST OVERRIDE ACTIVE: the caller holds ${LOCK}"
  true
fi

# --- gate 2: source pins — the reviewed code/config/weights/SPLITS, byte for byte #
# (r2-B1) The dataset configs only NAME a split; the actual sample list lives in
# data/HAA/*_base.json, so an edited split could train successfully while every
# other gate reported success. Both are pinned here AND in the closure below.
# ⚠️ PIN_probe changes whenever the probe does; update it deliberately and re-run
# the guard suite then.
PIN_train="bce1c94e648138459c056d82ac3e5f385e413b99f819b71bbbcd6d470d5f13ea  train.py"
PIN_defaults="09fe9f28ca78e6bc741797e15eeb6632259760d6efe58ffbb626d2ef9383a612  defaults.ini"
PIN_haamd="7a0906c34b9bccac3d6db198bd1bdac75688b54724292563968f53b088ad91a6  src/configs/dataset_configs/custom_metadata/HAA_md.py"
PIN_traincfg="5a530327eb89c2745086fe777c9f9c179b40a419c6fe1baf8473d5ef8cb468c4  ${DATASET_CFG}"
PIN_valcfg="8f00393f49970448e3d87051265787a3ff3c2b819a7263c0398c789bb28b5d47  ${VAL_DATASET_CFG}"
PIN_trainsplit="4ce6b46d5903b9a26b008c6996a1ae2913b49b8097ba745007bcc2fed32effe2  data/HAA/train_base.json"
PIN_valsplit="445fc856d4bf3aca3cd772da9991713cc1e42245daf8f2541f71c4f9e89f2152  data/HAA/val_base.json"
PIN_vae="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9  weights/FLAC/VAE.safetensors"
PIN_stock="3639a9face84d13bcbb8f4472e78970c8e045952337f11b4f77d8798f786ba80  ${STOCK_CFG}"
PIN_yaw_rotation="bf8dd38f62dbd88461e9e215c9f639a57c6fefe673d1a9a4185df32ab5f848a1  src/data/yaw_rotation.py"
PIN_diffusion="ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb  src/training/diffusion.py"
PIN_factory="6967ec9fd800bb991d6f2ee2aee890bb73c093bfc5f676617f590f1dbd9d330f  src/training/factory.py"
PIN_probe="24110322095e125f844b6793cad5eda4b2cf42128e380d551ff22653cb921de0  ${PROBE_SCRIPT}"
case "$ARM" in
  P1)  PIN_armcfg="" ;;   # P1's config IS the stock file, already pinned above
  YNA) PIN_armcfg="" ;;   # same stock file
  BNA) PIN_armcfg="" ;;   # same stock file
  BF)  PIN_armcfg="834e4933f2f5c8050f196043e11260e00023a7c31205a55961e0a77ca910c1dc  ${ARM_CFG}" ;;
  YAW) PIN_armcfg="a03d106cd72744df40187b5c493010ecc996275b2afa32a4811d7c962c77cb53  ${ARM_CFG}" ;;
esac
[ -z "$ARM_CFG_SHA" ] || PIN_armcfg="${ARM_CFG_SHA}  ${ARM_CFG}"

NPINS=0
for P in "$PIN_train" "$PIN_defaults" "$PIN_haamd" "$PIN_traincfg" "$PIN_valcfg" \
         "$PIN_trainsplit" "$PIN_valsplit" "$PIN_vae" "$PIN_stock" \
         "$PIN_yaw_rotation" "$PIN_diffusion" "$PIN_factory" "$PIN_probe" "$PIN_armcfg"; do
  [ -n "$P" ] || continue
  echo "$P" | sha256sum -c --status - || {
    echo "SOURCE PIN FAILED for '${P##*  }' - the reviewed code/config/weights/split moved under this experiment - abort"; exit 2; }
  NPINS=$((NPINS+1))
done
echo "source pins OK (${NPINS} files match the reviewed revision, incl. both HAA split inventories)"

# A dirty tracked tree under the TRAINING CLOSURE means the pins were computed
# from something committed nowhere, and it is what makes EXPECT_SHA meaningful.
# The closure is enumerated rather than taken as "all of src/", and the
# exclusions are arguments, not conveniences:
#   * src/tests/ — no training process imports it; it cannot change a run.
#   * src/tools/ — likewise not imported. Its one output that DOES reach a run is
#     the init, and that is bound by sha in gate 3 (and again before exec), which
#     is strictly stronger than the cleanliness of the script that produced it.
#   * worklog/ — this script writes there while running.
CLOSURE=(train.py defaults.ini baselines
         src/models src/data src/training src/configs src/inference src/metrics src/interface
         src/__init__.py data/HAA/train_base.json data/HAA/val_base.json)
DIRTY="$(git status --porcelain -- "${CLOSURE[@]}" 2>/dev/null | head -5)"
[ -z "$DIRTY" ] || {
  echo "tracked training closure is dirty - commit or stash before launching:"; echo "$DIRTY"; exit 2; }
echo "tree clean across the training closure (${CLOSURE[*]}) | branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

# --- gate 3: the initial weights, pinned by the sibling manifest -------------- #
# The init is the ONLY thing that differs between two otherwise identical arms,
# so it is the one artifact a mix-up would make invisible. The manifest pins sha
# AND path because they are two independent facts: WHICH BYTES (the extractor
# writes a content-only sha) and WHICH ARM they belong to. Re-validated
# immediately before exec (r2-B6) — this check alone leaves a window.
[ -f "$MANIFEST" ] || {
  echo "init manifest ${MANIFEST} not found - extract the EMA inits first (python -m src.tools.extract_ema_weights) and record '<sha256>  <path>' lines there - abort"; exit 2; }
MATCHES="$(awk -v p="$INIT" '$2 == p' "$MANIFEST")"
NMATCH="$(printf '%s' "$MATCHES" | grep -c . )"
[ "$NMATCH" = "1" ] || {
  echo "the manifest ${MANIFEST} has ${NMATCH} line(s) for '${INIT}' (need exactly 1) - refusing to launch an arm whose init is unpinned or ambiguous - abort"; exit 2; }
[ -f "$INIT" ] || {
  echo "init checkpoint ${INIT} does not exist (the manifest pins it, but the file is missing) - abort"; exit 2; }
printf '%s\n' "$MATCHES" | sha256sum -c --status - || {
  echo "INIT SHA MISMATCH for ${INIT} - this is NOT the extracted EMA checkpoint the manifest pins - abort"; exit 2; }
echo "init gate OK: ${INIT} matches its manifest line (${MATCHES%% *})"

# --- gate 4: config contract — the arm is the stock plus exactly its deltas --- #
# Deliberately BEFORE the FULL-only namespace check: a malformed treatment must
# be reported as such, not masked by "that directory is busy" (exp_17 r2 finding).
python - "$ARM" "$ARM_CFG" "$STOCK_CFG" <<'PY' || { echo "CONFIG CONTRACT FAILED - abort"; exit 2; }
import json, sys
arm_id, arm_p, stock_p = sys.argv[1], sys.argv[2], sys.argv[3]

def strict(a, b, path="root"):
    """Type-then-value diff: 1 == True and 0 == 0.0 in Python, so plain equality
    cannot see a type drift that the trainer WOULD see."""
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
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = strict(x, y, f"{path}[{i}]")
            if r: return r
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"

stock_bytes = open(stock_p, "rb").read()
arm_bytes = open(arm_p, "rb").read()
arm, stock = json.loads(arm_bytes), json.loads(stock_bytes)
t = arm["training"]

# Checked FIRST, and by name. Placed after the strict comparison it was
# unreachable: `use_ema: false` also breaks equality against the stock, so the
# operator would be told "not the stock plus its deltas" about the one setting
# that decides whether the HAA rows are EMA rows at all.
if t.get("use_ema") is not True:
    sys.exit("use_ema must be true: the inits are EMA weights and the HAA rows will be EMA rows")

if arm_id in ("P1", "YNA", "BNA"):
    # P1 consumes the stock file itself; anything else is a copy that can drift.
    if arm_bytes != stock_bytes:
        sys.exit(f"{arm_id} must run the stock config bytes; {arm_p} differs from {stock_p}")
    if "cond_method" in t or "yaw_aug" in t:
        sys.exit(f"{arm_id} is a vanilla-finetune arm; a treatment key is present")
    delta = "none (stock config used directly)"

elif arm_id == "BF":
    if "yaw_aug" in t:
        sys.exit("the BF arm must NOT carry yaw_aug: one treatment per arm")
    if t.get("cond_method") != "fa_invariant":
        sys.exit(f"cond_method must be 'fa_invariant', got {t.get('cond_method')!r}")
    angles = t.get("frame_avg_angles")
    if angles != [0.0, 90.0, 180.0, 270.0] or not all(type(a) is float for a in angles or []):
        sys.exit(f"frame_avg_angles must be B-F's float C4 orbit, got {angles!r}")
    if angles[0] != 0.0:
        sys.exit("frame_avg_angles[0] must be 0.0 (invariant_conditioning's contract)")
    stripped = json.loads(json.dumps(arm))
    stripped["training"].pop("cond_method"); stripped["training"].pop("frame_avg_angles")
    # Registered deltas 3/4 (Yixun 2026-08-18): grad-ckpt restored on both ViT
    # conditioners -- the fa 4-angle training peak measured 47.37/47.40 GiB on
    # an EMPTY A6000, and FLAC_AR_BF itself trained with checkpointing ON.
    reverted = 0
    for c in stripped["model"]["conditioning"]["configs"]:
        if c.get("type") == "ViTCoordinates":
            if c["config"].pop("gradient_checkpointing", None) is not True:
                sys.exit(f"BF requires gradient_checkpointing true on {c.get('id')} (registered delta 3/4)")
            reverted += 1
    if reverted != 2:
        sys.exit(f"expected exactly 2 ViT conditioners carrying grad-ckpt, reverted {reverted}")
    d = strict(stripped, stock)
    if d:
        sys.exit(f"BF config is NOT the stock plus exactly its two registered deltas - {d}")
    delta = f"cond_method=fa_invariant + frame_avg_angles={angles} + grad-ckpt x2"

else:  # YAW
    if "cond_method" in t:
        sys.exit("the YAW arm must NOT carry cond_method: one treatment per arm")
    if "frame_avg_angles" in t:
        sys.exit("the YAW arm must NOT carry frame_avg_angles")
    block = t.get("yaw_aug")
    if block != {"enabled": True, "img_w": 512, "seed": 42}:
        sys.exit(f"yaw_aug block is not the registered treatment: {block!r}")
    if not isinstance(block["enabled"], bool) or isinstance(block["img_w"], bool) \
       or not isinstance(block["img_w"], int) or isinstance(block["seed"], bool) \
       or not isinstance(block["seed"], int):
        sys.exit(f"yaw_aug block has the wrong TYPES: {block!r}")
    vits = [c for c in arm["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    if len(vits) != 2:
        sys.exit(f"expected 2 ViT conditioners, found {len(vits)}")
    widths = {c["config"]["ViT"]["img_w"] for c in vits}
    if widths != {block["img_w"]}:
        sys.exit(f"yaw_aug.img_w={block['img_w']} but ViT widths are {widths} - the "
                 "augmentation would roll the panorama by the wrong number of "
                 "columns and training would proceed without error")
    stripped = json.loads(json.dumps(arm))
    stripped["training"].pop("yaw_aug")
    d = strict(stripped, stock)
    if d:
        sys.exit(f"YAW config is NOT the stock plus exactly training.yaw_aug - {d}")
    delta = f"yaw_aug={block}"

print(f"config contract OK: {arm_p} == {stock_p} + {delta}")
# Deliberately does NOT echo the words of the treatment banner: the post-run
# banner check greps this same log, and a preflight paraphrase that matched it
# would make that check self-satisfying (exp_17 Codex r1 finding).
print(f"preflight treatment plan: arm={arm_id} deltas_registered=1-per-arm use_ema=True")
PY

# --- gate 5: the dataset root and both split inventories --------------------- #
# HAA is being relocated to /media/diskstation and reached through a symlink. The
# dataloader joins paths LEXICALLY (src/data/dataset.py:20), so a symlink works —
# and so does a half-copied target, an empty mount, or a link pointing somewhere
# else entirely. The resolved root is logged, and the FIRST and LAST audio file
# each split actually references must exist and be non-empty. That is two stat
# calls per split, and it is the difference between "the recipe named a split"
# and "the split is on this disk right now".
python - "${HAA_ROOT}" "$DATASET_CFG" "$VAL_DATASET_CFG" <<'PY' || { echo "DATASET INVENTORY GATE FAILED - abort"; exit 2; }
import json, os, sys
override = sys.argv[1] or None
problems = []
for cfg_path in sys.argv[2:]:
    cfg = json.load(open(cfg_path))
    ds = cfg["datasets"][0]
    root = override or ds["path"]
    folder = ds.get("folder_name", "binaural_rirs")
    split_path = ds["json_file_path"]
    split = json.load(open(split_path))
    scenes = [s for s in split if split[s]]
    if not scenes:
        problems.append(f"{split_path}: no scene has any file")
        continue
    first = os.path.join(root, scenes[0], folder, split[scenes[0]][0])
    last = os.path.join(root, scenes[-1], folder, split[scenes[-1]][-1])
    real = os.path.realpath(root)
    kind = "symlink" if os.path.islink(root) else ("dir" if os.path.isdir(root) else "MISSING")
    print(f"  {os.path.basename(split_path)}: root '{root}' ({kind}) -> {real}; "
          f"{len(scenes)} scenes, {sum(len(split[s]) for s in scenes)} files")
    for role, p in (("first", first), ("last", last)):
        if not os.path.isfile(p):
            problems.append(f"{role} referenced file missing: {p}")
        elif os.path.getsize(p) == 0:
            problems.append(f"{role} referenced file is EMPTY: {p}")
        else:
            print(f"    {role}: {p} ({os.path.getsize(p)} bytes) OK")
if problems:
    print("the dataset is not ready on this machine:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("  dataset inventory OK (both splits' first and last files present and non-empty)")
PY

# --- gate 6: R1 — fa/rotate machinery must be C4-invariant on HAA metadata ---- #
# BF and YAW BOTH ride src/data/yaw_rotation.py: BF averages the conditioner over
# the C4 orbit, YAW rolls the panorama by a drawn offset. Plan §3 R1 makes this a
# HARD gate — a failure STOPS the arm and is reported to Yixun; the sign
# convention is NOT silently "fixed" here. P1 rotates nothing, so it is skipped,
# explicitly and by name.
if [ "$ARM" = "P1" ] || [ "$ARM" = "YNA" ] || [ "$ARM" = "BNA" ]; then
  echo "R1 probe: SKIPPED for the ${ARM} arm (vanilla conditioning rotates nothing)"
else
  echo "R1 probe: required for the ${ARM} arm (it drives src/data/yaw_rotation.py)"
  # The probe loads THIS ARM'S INIT through train.py's own consumer path (Codex
  # exp_19 r1 finding 5, closed), so the tensors under test are the ones the
  # finetune starts from.
  # ⚠️ STANDING LIMITATION (r1 finding 4, accepted as demoted): the probe's orbit
  # and the frame average it measures both call rotate_scene_metadata, so a gauge
  # error inside that primitive moves both sides together and is invisible here.
  # A PASS certifies pipeline/shape/mask consistency on real HAA data with the
  # arm's weights; gauge correctness rests on plan_haa_finetune.md §3 R1's code
  # reading plus FA's sign-closure argument. Stated in every launch log.
  echo "R1 probe DISCLOSURE: loads ${INIT} through train.py's consumer path and measures the arm's own conditioner weights; subject and oracle share rotate_scene_metadata, so a shared gauge error is not detectable here (plan §3 R1)"
  if [ -n "$PROBE_CMD" ]; then
    PROBE=(bash -c "$PROBE_CMD")
  else
    PROBE=(env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" python "$PROBE_SCRIPT"
           --model-config "$ARM_CFG" --dataset-config "$VAL_DATASET_CFG"
           --ckpt-path "$INIT"
           --dtype float64
           --threshold "$PROBE_THRESHOLD" --num-samples 4
           --out "${LOGDIR}/probe_haa_fa_invariance_${ARM}_${TS}.json")
  fi
  PROBE_OUT="$("${PROBE[@]}" 2>&1)"; PRC=$?
  printf '%s\n' "$PROBE_OUT"
  # rc AND the verdict line: a probe that cannot build the stack or load the HAA
  # dataset exits non-zero, and that is a REFUSAL, never a skip.
  if [ "$PRC" -ne 0 ]; then
    echo "R1 GATE REFUSED (probe rc=${PRC}): fa/rotate conditioning is not proven C4-invariant on HAA - do NOT launch ${ARM}; report to Yixun (plan §3 R1)"; exit 2
  fi
  grep -qF "R1 GATE PASS" <<<"$PROBE_OUT" || {
    echo "R1 GATE REFUSED: the probe exited 0 but printed no 'R1 GATE PASS' verdict - refusing to read silence as invariance"; exit 2; }
  echo "R1 gate OK: threshold ${PROBE_THRESHOLD}"
fi

# --- gate 7: namespace occupancy, then the logging identity ------------------- #
if [ "$MODE" = "FULL" ] && [ -d "$SAVEDIR" ] && [ -n "$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | head -1)" ]; then
  echo "${SAVEDIR} already contains checkpoints - refusing to overwrite or interleave with a run; move it aside or resume deliberately - abort"; exit 2
fi
echo "namespace OK: ${SAVEDIR} holds no checkpoints"

if [ "$LOGGER" = "wandb" ] && [ "$DRY_RUN" = "0" ]; then
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1)"
  python - "$WANDB_IDENTITY" <<'PY'
import sys
want = sys.argv[1]
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == want else 2)
PY
  [ $? -eq 0 ] || { echo "wandb identity != ${WANDB_IDENTITY} - set the right key or use LOGGER=none - abort"; exit 2; }
fi

# --- gate 8: resource floors on the CHOSEN card, plus co-tenancy disclosure --- #
# The floors may be RAISED from the environment but never lowered below the
# registered defaults: a floor the launching command line can set to zero is not
# a floor (r2 finding). Clamped in EVERY mode, not just production — the rule is
# the same everywhere, and a rule that only exists on the path no test can reach
# is a rule nobody has ever seen work (the guard suite drives this in DRY).
if [ "$MIN_FREE_MB" -lt "$DEFAULT_MIN_FREE_MB" ] 2>/dev/null; then
  echo "MIN_FREE_MB=${MIN_FREE_MB} is below the registered floor ${DEFAULT_MIN_FREE_MB}; CLAMPED (floors may be raised, never lowered)"
  MIN_FREE_MB="$DEFAULT_MIN_FREE_MB"
fi
if [ "$MIN_FREE_DISK_MB" -lt "$DEFAULT_MIN_FREE_DISK_MB" ] 2>/dev/null; then
  echo "MIN_FREE_DISK_MB=${MIN_FREE_DISK_MB} is below the registered floor ${DEFAULT_MIN_FREE_DISK_MB}; CLAMPED (floors may be raised, never lowered)"
  MIN_FREE_DISK_MB="$DEFAULT_MIN_FREE_DISK_MB"
fi

FREE="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
[ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${GPU} - refusing to launch blind - abort"; exit 2; }
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${GPU} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - abort"; exit 2; }
echo "GPU ${GPU} free ${FREE} MiB >= ${MIN_FREE_MB} MiB floor"

FREE_DISK="$(df -Pm . | awk 'NR==2{print $4}')"
[ -n "$FREE_DISK" ] || { echo "df query failed - refusing to launch blind - abort"; exit 2; }
[ "$FREE_DISK" -ge "$MIN_FREE_DISK_MB" ] || {
  echo "free disk ${FREE_DISK} MiB < required ${MIN_FREE_DISK_MB} MiB - at cadence ${CADENCE} this arm writes $(( MAXSTEPS / CADENCE )) checkpoints of ~690 MiB - abort"; exit 2; }
echo "disk: ${FREE_DISK} MiB free >= ${MIN_FREE_DISK_MB} MiB floor (this arm projects $(( MAXSTEPS / CADENCE )) x ~690 MiB)"

echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'

# --- gate 9: the exact argv, then the dry-run boundary ----------------------- #
# The guard suite asserts against THIS line, so a wrong --max-steps, --save-dir
# or --pretrained-ckpt-path cannot hide behind an agreeing preflight message.
ARGV=(python train.py
  --dataset-config "$DATASET_CFG"
  --val-dataset-config "$VAL_DATASET_CFG"
  --model-config "$ARM_CFG"
  --pretransform-ckpt-path "$VAE"
  --pretrained-ckpt-path "$INIT"
  --max-steps "$MAXSTEPS" --batch-size "$BATCH" --accum-batches "$ACCUM"
  --num-workers "$NUM_WORKERS" --seed "$SEED" --num-gpus 1 --precision bf16-mixed
  --val-every "$VALEVERY" --checkpoint-every "$CADENCE"
  --logger "$LOGGER" --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR")
echo "ARGV: ${ARGV[*]}"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; train.py NOT launched"; exit 0
fi

# --- gate 10 (r2-B6): the init, re-validated at the point of consumption ----- #
# Gate 3 ran before the probe, the resource gates and the identity check; between
# then and now the file could have been replaced, relocated, or completed by a
# concurrent extraction. The manifest line is re-hashed here, last, so the bytes
# train.py is about to open are the bytes that were pinned and probed.
[ -f "$INIT" ] || { echo "INIT DISAPPEARED between gate 3 and launch: ${INIT} - abort"; exit 2; }
printf '%s\n' "$MATCHES" | sha256sum -c --status - || {
  echo "INIT SHA CHANGED between gate 3 and launch (${INIT}) - the bytes train.py would load are not the bytes that were pinned and probed - abort"; exit 2; }
echo "init re-validated at the point of consumption: ${INIT}"

# The GATE phase is over: release the shared lock so a sibling arm can gate and
# start on the other card. The per-arm (fd 8) and per-GPU (fd 7) locks are
# deliberately left OPEN — the child inherits them and they stay held for the
# whole run, which is what stops a same-arm or same-card second launch.
[ "$GATE_LOCK_OWNED" = "1" ] && exec 9>&-

if [ "$DRY_RUN" = "2" ]; then
  echo "REHEARSAL: running TRAIN_CMD in place of train.py; the post-run gates below are the real ones"
  export SAVEDIR MAXSTEPS ARM MODE MID_STEPS FULL_STEPS
  RUNNER=(bash -c "$TRAIN_CMD")
else
  RUNNER=(env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}")
fi

START_EPOCH="$(date +%s)"
"${RUNNER[@]}" 2>&1 | tee -a "$RUNLOG"
rc="${PIPESTATUS[0]}"
END_EPOCH="$(date +%s)"
echo "=== exp_19 HAA ${ARM} ${MODE} exit rc=${rc} after $((END_EPOCH-START_EPOCH))s at $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- post-run: read a NORMALISED COPY of the run log ------------------------- #
# Redirection, not `tr ... | grep -q`: under `pipefail` a `-q` early exit sends
# SIGPIPE to `tr`, and the pipeline then reports failure AFTER a successful match
# (exp_17 Codex r3 blocking finding). tqdm writes \r, ordinary prints write \n.
NORM="$(mktemp)"
trap 'rm -f "$NORM"' EXIT
tr '\r' '\n' < "$RUNLOG" > "$NORM"

# The augmentation must have been ACTIVE — YAW only; it is the only arm whose
# treatment announces itself, and a silently disabled treatment looks exactly
# like a successful run. WHOLE-LINE match against diffusion.py:406-408.
if [ "$ARM" = "YAW" ]; then
  if grep -qxF -- "$BANNER" "$NORM"; then
    echo "treatment banner: FOUND (exact whole-line match: '${BANNER}')"
  else
    echo "TREATMENT BANNER '${BANNER}' NOT FOUND in ${RUNLOG} - the run may have trained WITHOUT augmentation - treat this run as invalid"
    [ "$rc" -eq 0 ] && rc=3
  fi
fi

# The run must have REACHED its endpoint. Lightning catches KeyboardInterrupt
# without re-raising, so an interrupted run still exits 0. Framing matters: tqdm
# leaves its progress line unterminated, so the marker is APPENDED to it —
# matched as line-ENDS-WITH, which still rejects a diagnostic that merely QUOTES
# the marker (quoted text has words after it).
MARKER="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."
if ! awk -v m="$MARKER" 'substr($0, length($0)-length(m)+1) == m { found=1 }
                         END { exit found ? 0 : 1 }' "$NORM"; then
  echo "ENDPOINT NOT REACHED: no line ENDS WITH Lightning's '${MARKER}' in ${RUNLOG} - this run did NOT complete its registered budget (interrupted runs can still exit 0) - treat as invalid"
  [ "$rc" -eq 0 ] && rc=6
else
  echo "endpoint marker: FOUND (line ends with '${MARKER}')"
fi

# Fit health, in BOTH modes (exp_17 r3: NaN checking was SMOKE-only, so a FULL
# run could go non-finite, still reach its endpoint, write checkpoints, exit 0).
if grep -qiE 'train/loss=(nan|-?inf(inity)?)' "$NORM"; then
  echo "NON-FINITE LOSS observed in ${RUNLOG} - the fit is not healthy regardless of the endpoint - treat as invalid"
  [ "$rc" -eq 0 ] && rc=4
else
  echo "fit health: no non-finite train/loss observed"
fi

if [ "$MODE" = "FULL" ]; then
  NCK="$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | wc -l)"
  MISSING=""
  for S in "$MID_STEPS" "$FULL_STEPS"; do
    [ -n "$(find "$SAVEDIR" -name "*step=${S}.ckpt" 2>/dev/null | head -1)" ] || MISSING="${MISSING} ${S}"
  done
  if [ -n "$MISSING" ]; then
    echo "REQUIRED CHECKPOINT(S) MISSING at step(s)${MISSING} under ${SAVEDIR} - both registered readings (step ${MID_STEPS} and step ${FULL_STEPS}, plan B1) must exist - treat as invalid"
    [ "$rc" -eq 0 ] && rc=6
  else
    echo "both registered readings present: step=${MID_STEPS} and step=${FULL_STEPS}"
  fi
  # NOT an equality check: --val-every can add best-checkpoints on top of the
  # cadence series, so a count that is not exactly MAXSTEPS/CADENCE is normal.
  echo "checkpoints under ${SAVEDIR}: ${NCK} (cadence ${CADENCE} alone would give $(( MAXSTEPS / CADENCE )))"
fi

if [ "$MODE" = "SMOKE" ]; then
  NCKPT="$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | wc -l)"
  if [ "$NCKPT" -ne 0 ]; then
    echo "SMOKE wrote ${NCKPT} checkpoint(s) under ${SAVEDIR} - the cadence pin failed; a smoke artifact could be mistaken for a run checkpoint"
    [ "$rc" -eq 0 ] && rc=5
  else
    echo "smoke checkpoints: 0 (cadence ${SMOKE_CADENCE} >> endpoint ${SMOKE_STEPS}, as pinned)"
  fi
  echo "SMOKE: ${SMOKE_STEPS} steps in $((END_EPOCH-START_EPOCH))s (includes startup) -> FULL projects ~$(( (END_EPOCH-START_EPOCH) * FULL_STEPS / SMOKE_STEPS / 60 )) min (upper bound: startup is amortised over ${SMOKE_STEPS} steps here)"
fi

echo "=== exp_19 HAA ${ARM} ${MODE} final rc=${rc} ==="
exit $rc
