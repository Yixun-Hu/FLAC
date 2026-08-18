#!/usr/bin/env bash
# ============================================================================
# haa_ft_launch.sh — exp_19: HAA finetuning of three 40k AR inits.
#
#   ARM={P1|BF|YAW} GPU={0|1} MODE={SMOKE|FULL} [DRY_RUN=1] bash haa_ft_launch.sh
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
#              write a checkpoint (asserted afterwards). Nothing it produces can
#              be mistaken for, or resumed into, a FULL artifact.
# MODE=FULL  — the registered 1,000-step endpoint, checkpoints every 10 so BOTH
#              the step-410 and step-1000 readings exist (plan B1).
# DRY_RUN=1  — run every gate, print the exact train.py argv, exit 0 before any
#              side effect beyond this log. This is the boundary the guard suite
#              asserts against: it inspects the REAL argv, never a paraphrase.
#
# Lessons carried from the exp_17 lineage and its Codex r3 debts:
#  * post-run log checks never use `tr | grep -q` — under `pipefail` a `-q` early
#    exit SIGPIPEs `tr` and the pipeline reports failure after a successful match
#    (r3 blocking). The log is normalised to a file by REDIRECTION and grepped
#    from the file.
#  * the endpoint marker is matched with Lightning's REAL framing: tqdm leaves
#    the progress line unterminated, so the marker is APPENDED to it. Matched as
#    line-ENDS-WITH (src/tools/exp17_full_audit.py), not whole-line, not
#    substring — a substring also matches a diagnostic that quotes it.
#  * the treatment banner is matched as a WHOLE line, and the preflight
#    deliberately never prints those words, so the check cannot satisfy itself.
#  * NaN/Inf is checked in BOTH modes (r3: FULL could reach its endpoint,
#    write a checkpoint and exit 0 with non-finite loss).
#  * train.py's output goes through a SYNCHRONOUS pipe to its own run log, so the
#    post-run reader cannot race an undrained async `tee` (r3 non-blocking).
#  * test overrides (PROBE_CMD / MANIFEST / INIT_DIR) are REFUSED unless
#    DRY_RUN=1: a gate you can redirect from the environment is not a gate.
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
MIN_FREE_MB="${MIN_FREE_MB:-20000}"     # free VRAM floor on the CHOSEN card
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-}"   # resolved per MODE below

# Test-only overrides. Each one substitutes EVIDENCE for a gate, so each is
# refused outside DRY_RUN (checked immediately after MODE validation) and each
# is disclosed loudly in the log when used.
MANIFEST_DEFAULT="${EXPDIR}/exp19_init_shas.txt"
INIT_DIR_DEFAULT="outputs_FLAC/exp19_inits"
MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"
INIT_DIR="${INIT_DIR:-$INIT_DIR_DEFAULT}"
PROBE_CMD="${PROBE_CMD:-}"
# Substitutes gate 2's ARM-CONFIG pin only. Without it the config-contract gate
# (gate 4) is unreachable by mutation — every edit to an arm config trips the
# byte pin first — and an unreachable gate is a gate nobody has ever tested.
ARM_CFG_SHA="${ARM_CFG_SHA:-}"

# --- pinned constants: not overridable from the environment ------------------ #
# A budget you can raise from the shell is not a budget: the same command line
# that wants to launch would decide what the endpoint is.
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
PROBE_THRESHOLD="1e-5"        # plan R1
BANNER="yaw_aug ENABLED img_w=512 seed=42"   # EXACT text of diffusion.py:406-408
WANDB_IDENTITY="yh4742@princeton.edu"

DATASET_CFG="src/configs/dataset_configs/HAA/train/haa_train.json"
VAL_DATASET_CFG="src/configs/dataset_configs/HAA/eval/haa_val.json"
STOCK_CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
VAE="weights/FLAC/VAE.safetensors"
PROBE_SCRIPT="${EXPDIR}/probe_haa_fa_invariance.py"

# The invariant the smoke cadence pin depends on, asserted rather than assumed.
[ "$SMOKE_STEPS" -ge 1 ] && [ "$SMOKE_STEPS" -lt "$SMOKE_CADENCE" ] || {
  echo "SMOKE_STEPS=${SMOKE_STEPS} must satisfy 1 <= steps < cadence ${SMOKE_CADENCE} - abort"; exit 2; }
# ...and the one the step-410 reading depends on.
[ $(( MID_STEPS % FULL_CADENCE )) -eq 0 ] && [ "$MID_STEPS" -lt "$FULL_STEPS" ] || {
  echo "step ${MID_STEPS} is not on the cadence-${FULL_CADENCE} grid below ${FULL_STEPS} - abort"; exit 2; }

# --- gate 1: ARM / GPU / MODE are matched EXACTLY, never inferred ------------ #
case "$ARM" in
  P1|BF|YAW) ;;
  *) echo "ARM must be exactly P1, BF or YAW (got '${ARM}') - abort"; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "GPU must be exactly 0 or 1 (got '${GPU}') - abort"; exit 2 ;;
esac
case "$MODE" in
  SMOKE|FULL) ;;
  *) echo "MODE must be exactly SMOKE or FULL (got '${MODE}') - abort"; exit 2 ;;
esac

# Evidence-substituting overrides are a DRY_RUN-only facility.
if [ "$DRY_RUN" != "1" ]; then
  [ -z "$PROBE_CMD" ] || { echo "PROBE_CMD is a DRY_RUN-only test override and is set ('${PROBE_CMD}') - a real launch must run the REAL R1 probe - abort"; exit 2; }
  [ -z "$ARM_CFG_SHA" ] || { echo "ARM_CFG_SHA is a DRY_RUN-only test override and is set ('${ARM_CFG_SHA}') - a real launch must match the hard-coded arm-config pin - abort"; exit 2; }
  [ "$MANIFEST" = "$MANIFEST_DEFAULT" ] || { echo "MANIFEST is a DRY_RUN-only test override and points at '${MANIFEST}' instead of ${MANIFEST_DEFAULT} - abort"; exit 2; }
  [ "$INIT_DIR" = "$INIT_DIR_DEFAULT" ] || { echo "INIT_DIR is a DRY_RUN-only test override and points at '${INIT_DIR}' instead of ${INIT_DIR_DEFAULT} - abort"; exit 2; }
fi

case "$ARM" in
  P1)  ARM_CFG="$STOCK_CFG" ;;
  BF)  ARM_CFG="${EXPDIR}/FLAC_HAA_finetune_BF.json" ;;
  YAW) ARM_CFG="${EXPDIR}/FLAC_HAA_finetune_YAW.json" ;;
esac
INIT="${INIT_DIR}/HAA_init_${ARM}.ckpt"

case "$MODE" in
  SMOKE) SUFFIX="_smoke"; MAXSTEPS="$SMOKE_STEPS"; CADENCE="$SMOKE_CADENCE"; VALEVERY="$SMOKE_VAL_EVERY"
         MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-4096}" ;;
  FULL)  SUFFIX="";       MAXSTEPS="$FULL_STEPS";  CADENCE="$FULL_CADENCE";  VALEVERY="$FULL_VAL_EVERY"
         # 100 checkpoints x ~690 MiB each (measured on the 40k artifacts) is
         # ~69 GiB PER ARM. This floor is not decoration: at cadence 10 the
         # deliverable is two orders of magnitude larger than a normal run's.
         MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-72000}" ;;
esac
NAME="FLAC_exp19_HAA_${ARM}${SUFFIX}"
EXPNAME="exp19_HAA_${ARM}${SUFFIX}"
SAVEDIR="outputs_FLAC/exp19_HAA_${ARM}${SUFFIX}"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
# DRY_RUN output is NOT production evidence and never lands in the evidence
# directory: the exp_17 guard suite had to delete new logs out of EXPDIR at exit,
# which its own review flagged as able to destroy a CONCURRENT real run's log.
if [ "$DRY_RUN" = "1" ]; then
  LOGDIR="${EXPDIR}/.dryrun_logs"; mkdir -p "$LOGDIR"
  LOG="${LOGDIR}/haa_ft_${TS}_${ARM}_${MODE}_dryrun.log"
else
  LOGDIR="$EXPDIR"
  LOG="${EXPDIR}/haa_ft_${TS}_${ARM}_${MODE}.log"
fi
RUNLOG="${LOGDIR}/haa_ft_${TS}_${ARM}_${MODE}_train.log"

# --- single instance through the GATE phase ---------------------------------- #
# Shared across arms on purpose: the gates read shared state (the manifest, the
# namespace, one free-VRAM snapshot per card), and two launchers interleaving
# through them could both pass a floor only one of them can satisfy.
# RELEASED before train.py starts — the arms are MEANT to train concurrently,
# one per card, so a lock held across training would serialise the experiment.
LOCK="${EXPDIR}/.haa_ft.lock"
exec 9>"$LOCK" || { echo "cannot open ${LOCK} - abort"; exit 2; }
flock -n 9 || { echo "another exp_19 launcher holds ${LOCK} (gate phase) - abort"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_19 HAA finetune | ARM=${ARM} MODE=${MODE} GPU=${GPU} | ${TS} | HEAD $(git rev-parse --short HEAD 2>/dev/null) | dry_run=${DRY_RUN} logger=${LOGGER} ==="
echo "identity: name=${NAME} | experiment=${EXPNAME} | save-dir=${SAVEDIR}"
echo "budget:   endpoint=${MAXSTEPS} | cadence=${CADENCE} | val-every=${VALEVERY} | batch=${BATCH}x accum ${ACCUM} = eff $((BATCH*ACCUM))"
echo "config:   ${ARM_CFG}"
echo "init:     ${INIT}"
[ "$DRY_RUN" = "1" ] && {
  [ -n "$PROBE_CMD" ] && echo "!! TEST OVERRIDE ACTIVE: PROBE_CMD='${PROBE_CMD}' (the R1 gate is STUBBED; this run proves nothing about invariance)"
  [ "$MANIFEST" != "$MANIFEST_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: MANIFEST='${MANIFEST}'"
  [ "$INIT_DIR" != "$INIT_DIR_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: INIT_DIR='${INIT_DIR}'"
  [ -n "$ARM_CFG_SHA" ] && echo "!! TEST OVERRIDE ACTIVE: ARM_CFG_SHA='${ARM_CFG_SHA}' (gate 2's arm-config pin is SUBSTITUTED)"
  true; }

# --- gate 2: source pins — the reviewed code/config/weights, byte for byte ---- #
# The brief's eight, plus the exp_17-lineage additions: the treatment code
# itself, and the R1 gate script (r3 blocking finding: the executed gate script
# lived under the exempt worklog tree and was not content-pinned).
# ⚠️ PIN_probe is EXPECTED to change when the exp_19 r1 review's fix batch lands;
# update it deliberately and re-run the guard suite then.
PIN_train="bce1c94e648138459c056d82ac3e5f385e413b99f819b71bbbcd6d470d5f13ea  train.py"
PIN_defaults="09fe9f28ca78e6bc741797e15eeb6632259760d6efe58ffbb626d2ef9383a612  defaults.ini"
PIN_haamd="7a0906c34b9bccac3d6db198bd1bdac75688b54724292563968f53b088ad91a6  src/configs/dataset_configs/custom_metadata/HAA_md.py"
PIN_traincfg="5a530327eb89c2745086fe777c9f9c179b40a419c6fe1baf8473d5ef8cb468c4  src/configs/dataset_configs/HAA/train/haa_train.json"
PIN_valcfg="8f00393f49970448e3d87051265787a3ff3c2b819a7263c0398c789bb28b5d47  src/configs/dataset_configs/HAA/eval/haa_val.json"
PIN_vae="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9  weights/FLAC/VAE.safetensors"
PIN_stock="3639a9face84d13bcbb8f4472e78970c8e045952337f11b4f77d8798f786ba80  ${STOCK_CFG}"
PIN_yaw_rotation="bf8dd38f62dbd88461e9e215c9f639a57c6fefe673d1a9a4185df32ab5f848a1  src/data/yaw_rotation.py"
PIN_diffusion="ef6a1f69459eabd77595bade192d269a0ce8a7ade2c8b4d8e50bb695c6e0f5fb  src/training/diffusion.py"
PIN_factory="6967ec9fd800bb991d6f2ee2aee890bb73c093bfc5f676617f590f1dbd9d330f  src/training/factory.py"
PIN_probe="bb3b1ae96ddd0d1e410f94a75db52efacf5a199807b266feec7d0eed80cd4cd6  ${PROBE_SCRIPT}"
case "$ARM" in
  P1)  PIN_armcfg="" ;;   # P1's config IS the stock file, already pinned above
  BF)  PIN_armcfg="eb64e6c24e9aba58d984bf9088b4248e2abffea48f180b90602c342a60dd66cd  ${ARM_CFG}" ;;
  YAW) PIN_armcfg="a03d106cd72744df40187b5c493010ecc996275b2afa32a4811d7c962c77cb53  ${ARM_CFG}" ;;
esac
[ -z "$ARM_CFG_SHA" ] || PIN_armcfg="${ARM_CFG_SHA}  ${ARM_CFG}"

NPINS=0
for P in "$PIN_train" "$PIN_defaults" "$PIN_haamd" "$PIN_traincfg" "$PIN_valcfg" \
         "$PIN_vae" "$PIN_stock" "$PIN_yaw_rotation" "$PIN_diffusion" \
         "$PIN_factory" "$PIN_probe" "$PIN_armcfg"; do
  [ -n "$P" ] || continue
  echo "$P" | sha256sum -c --status - || {
    echo "SOURCE PIN FAILED for '${P##*  }' - the reviewed code/config/weights moved under this experiment - abort"; exit 2; }
  NPINS=$((NPINS+1))
done
echo "source pins OK (${NPINS} files match the reviewed revision)"

# A dirty tracked tree under the TRAINING CLOSURE means the pins were computed
# from something committed nowhere. The closure is enumerated rather than taken
# as "all of src/", and the exclusions are arguments, not conveniences:
#   * src/tests/ — no training process imports it; it cannot change a run.
#   * src/tools/ — likewise not imported. Its one output that DOES reach a run is
#     the init, and that is bound by sha in gate 3, which is strictly stronger
#     than the cleanliness of the script that produced it.
#   * worklog/ — this script writes there while running.
# (Known-insufficient on its own — clean COMMITTED drift in unpinned closure code
# still passes, which is why HEAD is recorded in the banner above and the
# behaviour-defining files are content-pinned in gate 2.)
CLOSURE=(train.py defaults.ini baselines
         src/models src/data src/training src/configs src/inference src/metrics src/interface
         src/__init__.py)
DIRTY="$(git status --porcelain -- "${CLOSURE[@]}" 2>/dev/null | head -5)"
[ -z "$DIRTY" ] || {
  echo "tracked training closure is dirty - commit or stash before launching:"; echo "$DIRTY"; exit 2; }
echo "tree clean across the training closure (${CLOSURE[*]}) | branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

# --- gate 3: the initial weights, pinned by the sibling manifest -------------- #
# The init is the ONLY thing that differs between two otherwise identical arms,
# so it is the one artifact a mix-up would make invisible. The manifest pins sha
# AND path because they are two independent facts: WHICH BYTES (the extractor
# writes a content-only sha — it serialises through a file object, so the hash
# does not depend on the filename) and WHICH ARM they belong to. Matched on both.
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
# that decides whether the HAA rows are EMA rows at all (guard suite D7).
if t.get("use_ema") is not True:
    sys.exit("use_ema must be true: the inits are EMA weights and the HAA rows will be EMA rows")

if arm_id == "P1":
    # P1 consumes the stock file itself; anything else is a copy that can drift.
    if arm_bytes != stock_bytes:
        sys.exit(f"P1 must run the stock config bytes; {arm_p} differs from {stock_p}")
    if "cond_method" in t or "yaw_aug" in t:
        sys.exit("P1 is the vanilla arm; a treatment key is present")
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
    d = strict(stripped, stock)
    if d:
        sys.exit(f"BF config is NOT the stock plus exactly its two registered deltas - {d}")
    delta = f"cond_method=fa_invariant + frame_avg_angles={angles}"

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

# --- gate 5: R1 — fa/rotate machinery must be C4-invariant on HAA metadata ---- #
# BF and YAW BOTH ride src/data/yaw_rotation.py: BF averages the conditioner over
# the C4 orbit, YAW rolls the panorama by a drawn offset. The machinery was only
# ever validated on AR (listener-position panoramas); HAA renders at the SOURCE
# and flips the map vertically. Plan §3 R1 makes this a HARD gate — a failure
# STOPS the arm and is reported to Yixun; the sign convention is NOT silently
# "fixed" here. P1 rotates nothing, so it is skipped, explicitly and by name.
if [ "$ARM" = "P1" ]; then
  echo "R1 probe: SKIPPED for the P1 arm (vanilla conditioning rotates nothing)"
else
  echo "R1 probe: required for the ${ARM} arm (it drives src/data/yaw_rotation.py)"
  # The probe loads THIS ARM'S INIT through train.py's own consumer path before
  # measuring (Codex exp_19 r1 finding 5, closed in the r1 fix batch), so the
  # tensors under test are the ones the finetune starts from.
  # ⚠️ STANDING LIMITATION (r1 finding 4, not closable by code): the probe's orbit
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

# --- gate 6: namespace occupancy, then the logging identity ------------------- #
if [ "$MODE" = "FULL" ] && [ -d "$SAVEDIR" ] && [ -n "$(find "$SAVEDIR" -name '*.ckpt' 2>/dev/null | head -1)" ]; then
  echo "${SAVEDIR} already contains checkpoints - refusing to overwrite or interleave with a run; move it aside or resume deliberately - abort"; exit 2
fi
echo "namespace OK: ${SAVEDIR} holds no checkpoints"

if [ "$LOGGER" = "wandb" ] && [ "$DRY_RUN" != "1" ]; then
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

# --- gate 7: resource floors on the CHOSEN card, plus co-tenancy disclosure --- #
# Standing policy: co-tenancy is allowed with an explicit floor and disclosure —
# the other A6000 is expected to be running a sibling arm. Fail-CLOSED on a query
# error: a card we cannot measure is not a card we launch onto.
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

# --- gate 8: the exact argv, then the dry-run boundary ----------------------- #
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

# The gate phase is over: release the shared lock so a sibling arm can gate and
# start on the other card while this one trains.
exec 9>&-

START_EPOCH="$(date +%s)"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" "${ARGV[@]}" 2>&1 | tee -a "$RUNLOG"
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
# like a successful run. WHOLE-LINE match against diffusion.py:406-408; a
# substring grep would also match this script's own preflight output.
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
# matched as line-ENDS-WITH (src/tools/exp17_full_audit.py), which still rejects
# a diagnostic that merely QUOTES the marker (quoted text has words after it).
MARKER="\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached."
if ! awk -v m="$MARKER" 'substr($0, length($0)-length(m)+1) == m { found=1 }
                         END { exit found ? 0 : 1 }' "$NORM"; then
  echo "ENDPOINT NOT REACHED: no line ENDS WITH Lightning's '${MARKER}' in ${RUNLOG} - this run did NOT complete its registered budget (interrupted runs can still exit 0) - treat as invalid"
  [ "$rc" -eq 0 ] && rc=6
else
  echo "endpoint marker: FOUND (line ends with '${MARKER}')"
fi

# Fit health, in BOTH modes (exp_17 r3: NaN checking was SMOKE-only, so a FULL
# run could go non-finite, still reach its endpoint, write checkpoints and exit 0).
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
  echo "SMOKE: ${SMOKE_STEPS} steps in $((END_EPOCH-START_EPOCH))s (includes startup) -> FULL projects ~$(( (END_EPOCH-START_EPOCH) * FULL_STEPS / SMOKE_STEPS / 60 )) min (upper bound: startup is amortised over 20 steps here)"
fi

echo "=== exp_19 HAA ${ARM} ${MODE} final rc=${rc} ==="
exit $rc
