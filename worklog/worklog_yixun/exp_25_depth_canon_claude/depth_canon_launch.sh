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
#             with the SAME HEAD and pin-manifest sha256 (+ the schema-2 GPU
#             provenance under this launcher's literals, rC-F2).
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
# (7) projection <= 120 h (v14; was 60 h), and (8) the GPU gate: >= 12 GiB free on each card (v14;
# was 22.5 GiB) with compute processes on GPU 0/1 handled per GPU_OCCUPANCY_POLICY -- `shared`
# (pinned, plan v14, ordered by Yixun 2026-10-25): co-tenants DISCLOSED (pid, card, used MiB, cwd)
# in the log and the SMOKE evidence, never refused; `idle`: refused (v10). It never kills anything.
# Exit codes: 0 ok (DRY_RUN: every gate passed) | 2 a gate refused (CONFIG ERROR included) | 3 treatment
#   banner missing | 4 non-finite values / the smoke verdict FAILED | 5 the checkpoint set is wrong | 6 endpoint
#   or two-rank start missing | 7 a GPU-gate refusal (the memory floor, or an nvidia-smi answer that is not
#   clean; occupancy only under GPU_OCCUPANCY_POLICY=idle -- under `shared` co-tenants are disclosed, not
#   refused), RESERVED: only gpu_refuse emits it -- the ONE transient class the watcher retries (rA A6e) |
#   8 any OTHER exit 7, e.g. train.py's own status 7, remapped by the EXIT trap so a trainer failure is
#   never retried (rB-F2 F2-1, Codex A2 HIGH 1) | anything else: train.py's own nonzero status, passed
#   through (terminal).
#
# Write targets: the NAS save root (checkpoints, launch marker, write probe), this experiment
# folder (logs, smoke evidence, checkpoint manifest) and -- the one expected extra target --
# W&B's OFFLINE run record in the checkout's repo-local `wandb/offline-run-*` (the W&B mode is
# pinned offline, rC-F3, to be synced later with a newer SDK; train.py's WandbLogger is built
# without a save_dir). r3b-F1 (Codex r3b review) added: the flac interpreter pin, the tie-audit
# gate, the GPU-occupancy gate, finite-loss + latency smoke evidence, content-scoped binding.
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
AUDIT_GATE="$EXPDIR/depth_canon_audit_gate.py"   # the ONE tie-audit predicate (Codex rB B1 / rA A1)

# --- PINNED constants: not overridable from the environment ----------------- #
ENDPOINT_STEPS=40000
FULL_CADENCE=2500
SMOKE_STEPS=25
SMOKE_CADENCE=1000000             # >> SMOKE_STEPS: a smoke never writes a checkpoint
MAX_PROJECTED_HOURS=120           # plan §4 gate (7), v14: a shared card roughly halves the uncontended 0.259 opt-steps/s; a sanity cap only
MIN_NAS_FREE_GIB=30               # plan §4 gate (6)
# gate (8), v14: 12 GiB per card = the measured need of the identical --batch-size 32 --num-gpus 2 recipe on this
# box (5.9-6.6 GiB per rank in exp_14's smoke, ~10.3 GiB per card for the exp_13 vanilla-B arm); was 22.5 GiB
MIN_GPU_FREE_MIB=12288            # 12 GiB per card (gate (8), v14)
# gate (8) occupancy, v14 (Yixun 2026-10-25: "Relax the gates and launch shared now"): shared = compute processes
# on GPU 0/1 are co-tenants, disclosed and never refused (the memory floor decides); idle = they refuse (rc 7, v10)
GPU_OCCUPANCY_POLICY=shared
SAVE_ROOT_PIN="/media/diskstation/yixunhu/FLAC/checkpoints/exp25_depth_canon"
PY_PIN="/home/yixunhu/miniconda3/envs/flac/bin/python"   # the flac env, REQUIRED (r3b-F1 fix 6)
NAS_FSTYPE="cifs"
NAS_SOURCE="//10.249.7.28/shared_data"
WANDB_IDENTITY="yh4742@princeton.edu"
# rC-F3: W&B runs OFFLINE, always. The flac env's wandb SDK 0.15.4 cannot create an online run against the current
# W&B server -- upsert_run -> server_settings_introspection() gets None back ("'NoneType' object has no attribute
# 'get'") -- so the first real SMOKE (2026-10-26 03:45 EDT) died at train.py's logger.watch -> wandb.init. The env is
# pinned (no upgrade): train.py gets WANDB_MODE exported from this literal right before it starts (a caller's own value
# is overridden, never honoured); its run lands in the checkout's repo-local wandb/offline-run-*, to be synced later
# with a newer SDK. The W&B identity gate below still runs: it proves the key / account that later sync will use.
WANDB_MODE_PIN="offline"
CONTROL_SHA="733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"
BANNER="canonicalize ENABLED img_w=512 tiebreak=image"   # src/training/diffusion.py, exact
RC_GPU_REFUSAL=7                  # the watcher's one transient class: emitted ONLY by gpu_refuse
RC_TRAINER7=8                     # any other exit 7 (train.py's own status 7) becomes this terminal code
case "$GPU_OCCUPANCY_POLICY" in idle|shared) ;;   # self-check of the pinned literal (rC-F1)
  *) echo "CONFIG ERROR: GPU_OCCUPANCY_POLICY must be idle or shared"; exit 2 ;; esac

GPU_REFUSED=0
gpu_refuse() {   # gpu_refuse <message>: the ONE emitter of exit 7 (RC_GPU_REFUSAL)
  echo "$1"; GPU_REFUSED=1; exit "$RC_GPU_REFUSAL"
}
rc7_guard() {    # EXIT trap: an exit 7 that gpu_refuse did not emit is remapped (rB-F2 F2-1)
  local st=$?
  if [ "$st" -eq "$RC_GPU_REFUSAL" ] && [ "$GPU_REFUSED" != 1 ]; then
    echo "exit ${RC_GPU_REFUSAL} is reserved for the GPU gate: this exit ${RC_GPU_REFUSAL} (train.py's own status or another step) is remapped to the terminal ${RC_TRAINER7}"
    exit "$RC_TRAINER7"
  fi
}
trap rc7_guard EXIT

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
# --- the interpreter (r3b-F1 fix 6): the flac env runs every gate AND train.py -------- #
# Production: the pinned interpreter, and 'python' on PATH must BE it (an activated other env
# fails closed). Harness: the guard suite's stub, which must live inside the harness dir.
PATH_PY="$(command -v python 2>/dev/null || true)"
if [ "$HARNESS_FLAG" = true ]; then
  case "$(readlink -f -- "$PATH_PY" 2>/dev/null)" in
    "$HREAL"/?*) PY="$PATH_PY" ;;
    *) echo "INTERPRETER GATE FAILED: harness mode needs the harness's python stub first on PATH (got '${PATH_PY}') - abort"; exit 2 ;;
  esac
else
  [ -x "$PY_PIN" ] || { echo "INTERPRETER GATE FAILED: ${PY_PIN} is missing - abort"; exit 2; }
  if [ -z "$PATH_PY" ] || [ "$(readlink -f -- "$PATH_PY")" != "$(readlink -f -- "$PY_PIN")" ]; then
    echo "INTERPRETER GATE FAILED: 'python' on PATH is '${PATH_PY:-absent}', not the flac env ${PY_PIN} (activate it) - abort"; exit 2
  fi
  PY="$PY_PIN"
fi
echo "interpreter OK: ${PY}"
# rC-F3: the SDK version the W&B mode line names -- an import with the pinned interpreter, no server contact
WANDB_SDK="$("$PY" -c 'import wandb; print(wandb.__version__)')" || {
  echo "W&B SDK CHECK FAILED: ${PY} cannot import wandb (its error is above) - abort"; exit 2; }
# P1's recipe exactly (p1_ddp_launch.sh): no --val-dataset-config, no resume, no
# pretrained init; --precision bf16-mixed is defaults.ini's value, passed explicitly.
ARGV=("$PY" train.py
  --model-config "$ARM_CFG" --dataset-config "$DATASET_CFG" --pretransform-ckpt-path "$VAE"
  --max-steps "$MAXSTEPS" --checkpoint-every "$CADENCE"
  --batch-size "$MB" --accum-batches "$ACC" --num-gpus 2 --sync-batchnorm true
  --strategy ddp_find_unused_parameters_true --precision bf16-mixed --seed 42 --num-workers 6
  --logger wandb --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVE_ROOT")
echo "W&B mode: ${WANDB_MODE_PIN} (pinned WANDB_MODE_PIN; SDK ${WANDB_SDK} cannot create online runs against the current server - offline runs in ${REPO}/wandb/offline-run-*, sync later)"
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
DIRTY="$(git status --porcelain --untracked-files=all -- "${PINNED[@]}" "$PIN_MANIFEST" "$MAKE_PINS" "$SELF" \
  "$AUDIT_GATE")"
if [ -n "$DIRTY" ]; then
  echo "PINNED TREE NOT COMMITTED at HEAD (commit or restore, then regenerate the manifest):"
  printf '%s\n' "$DIRTY" | head -20 || true
  exit 2
fi
MANIFEST_SHA="$(sha256sum "$PIN_MANIFEST" | awk '{print $1}')"
echo "pin manifest OK: ${#PINNED[@]} files match ${PIN_MANIFEST} (sha256 ${MANIFEST_SHA}); pinned paths + launcher kit committed at HEAD"

# --- config contract: the P1 control plus exactly training.canonicalize -------- #
"$PY" - "$ARM_CFG" "$CONTROL_CFG" "$CONTROL_SHA" <<'PY' || { echo "CONFIG CONTRACT FAILED - abort"; exit 2; }
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

# --- (rung 3) the complete PASSING tie audit, through ONE gate (rB B1 / rA A1) ------------ #
# SMOKE and FULL both need it (the smoke evidence binds its sha256). The report + sidecar are
# PINNED files (bytes bound by the manifest, committed at HEAD). depth_canon_audit_gate.py --
# the predicate the collector applies too -- requires the exact report the audit script writes
# for a complete PASS over the three registered splits, re-derives the census with the audit's
# own code, checks every current hash and re-hashes all 9,650 depth maps (minutes over CIFS).
# Harness mode: a harness-tagged report against the harness's contract only (never production).
TIE_AUDIT="$EXPDIR/depth_canon_tie_audit.json"
GATE_ARGS=(--audit "$TIE_AUDIT" --repo "$REPO" --print-sha)
[ "$HARNESS_FLAG" = true ] && GATE_ARGS+=(--harness-contract "$HREAL/audit_contract.json")
AUDIT_SHA="$("$PY" "$AUDIT_GATE" "${GATE_ARGS[@]}")" || exit 2
[[ "$AUDIT_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "TIE AUDIT GATE FAILED: the gate printed no sha256 - abort"; exit 2; }
echo "tie audit OK: ${TIE_AUDIT} (sha256 ${AUDIT_SHA}) via ${AUDIT_GATE}: the registered census, 0 survivors, 0 errors, hashes current, map bytes re-hashed"

# --- the content the SMOKE evidence binds (r3b-F1 fix 9): HEAD is provenance only -------- #
# A record-only commit between SMOKE and FULL keeps every one of these; any change to the
# training closure (pin manifest), the launcher, the pin generator, the config contract's
# inputs or the tie audit changes one of them and needs a new SMOKE.
LAUNCHER_SHA="$(sha256sum "$SELF" | awk '{print $1}')"
MAKE_PINS_SHA="$(sha256sum "$MAKE_PINS" | awk '{print $1}')"
CONTRACT_SHA="$(cat "$ARM_CFG" "$CONTROL_CFG" | sha256sum | awk '{print $1}')"
AUDIT_GATE_SHA="$(sha256sum "$AUDIT_GATE" | awk '{print $1}')"
echo "evidence binding: manifest ${MANIFEST_SHA} launcher ${LAUNCHER_SHA} pins-generator ${MAKE_PINS_SHA} contract ${CONTRACT_SHA} audit ${AUDIT_SHA} audit-gate ${AUDIT_GATE_SHA}"

# --- (2) FULL requires a PASSING smoke evidence of this HEAD + manifest ------- #
# rC-F2 (Codex rC HIGH, C3): the schema-2 GPU provenance is part of the predicate -- gpu_policy, min_gpu_free_mib
# and max_projected_hours must be THIS launcher's literals (argv), free_mib_at_launch exactly {"0": int, "1": int}
# both >= the floor (the writer's `null` no-gate form -- unreachable, DRY_RUN exits before the writer -- is refused),
# every co-tenant record exactly {gpu: 0|1, pid: int > 0, used_mib: int >= 0, cwd: non-empty str} (a bool is not an
# int). The checker prints ONE line, "OK <path>" or "NO <why>"; a refusal prints the headline, then
# "evidence check: <why>" = the newest candidate's first offending key and the value seen.
if [ "$MODE" = "FULL" ]; then
  EV_CHECK="$("$PY" - "$EXPDIR" "$MANIFEST_SHA" "$HARNESS_FLAG" "$LAUNCHER_SHA" "$MAKE_PINS_SHA" \
      "$CONTRACT_SHA" "$AUDIT_SHA" "$AUDIT_GATE_SHA" "$GPU_OCCUPANCY_POLICY" "$MIN_GPU_FREE_MIB" \
      "$MAX_PROJECTED_HOURS" <<'PY' || true
import glob, hashlib, json, math, os, sys
exp, manifest, harness, launcher, make_pins, contract, audit, audit_gate, policy, floor, max_h = sys.argv[1:12]
floor, max_h = int(floor), int(max_h)
NEED = {"schema_version": 2, "kind": "depth_canon_smoke_evidence", "verdict": "PASS",
        "manifest_sha256": manifest, "launcher_sha256": launcher, "make_pins_sha256": make_pins,
        "contract_sha256": contract, "audit_sha256": audit, "audit_gate_sha256": audit_gate,
        "harness": harness == "true",
        "config_contract_ok": True,
        "banner_seen": True, "endpoint_seen": True, "ranks_seen": True, "nonfinite_seen": False,
        "smoke_checkpoints": 0, "finite_loss_seen": True,
        "gpu_policy": policy, "min_gpu_free_mib": floor, "max_projected_hours": max_h}   # rC-F2
ABSENT = object()
INTS = (("gpu", lambda v: v in (0, 1), "an int, 0 or 1"), ("pid", lambda v: v > 0, "an int > 0"),
        ("used_mib", lambda v: v >= 0, "an int >= 0"))
REC = "exactly {gpu: 0|1, pid: int > 0, used_mib: int >= 0, cwd: non-empty str}"
def show(v):          # the value seen, as JSON (one line)
    return "<missing>" if v is ABSENT else json.dumps(v)
def gpu_why(ev):      # rC-F2: the typed GPU disclosure; None = clean
    fm = ev.get("free_mib_at_launch", ABSENT)
    if type(fm) is not dict or sorted(fm) != ["0", "1"]:
        return f'free_mib_at_launch = {show(fm)} (want exactly {{"0": int >= {floor}, "1": int >= {floor}}})'
    for g in ("0", "1"):
        if type(fm[g]) is not int or fm[g] < floor:
            return f'free_mib_at_launch["{g}"] = {show(fm[g])} (want an int >= {floor})'
    cot = ev.get("co_tenants_at_launch", ABSENT)
    if type(cot) is not list:
        return f"co_tenants_at_launch = {show(cot)} (want a list of records {REC})"
    for i, rec in enumerate(cot):
        at = f"co_tenants_at_launch[{i}]"
        if type(rec) is not dict:
            return f"{at} = {show(rec)} (want {REC})"
        extra = [k for k in rec if k not in ("gpu", "pid", "used_mib", "cwd")]
        if extra:
            return f"{at}.{extra[0]} = {show(rec[extra[0]])} (want {REC}; unexpected key)"
        for k, good, want in INTS:
            v = rec.get(k, ABSENT)
            if type(v) is not int or not good(v):
                return f"{at}.{k} = {show(v)} (want {want})"
        v = rec.get("cwd", ABSENT)
        if type(v) is not str or not v:
            return f"{at}.cwd = {show(v)} (want a non-empty str)"
    return None
def why_not(path):    # None = qualifies; else the first offending key and the value seen
    try:
        ev = json.load(open(path))
    except (OSError, ValueError) as e:
        return f"not a JSON object ({type(e).__name__}: {e})"
    if type(ev) is not dict:
        return f"not a JSON object (top level {type(ev).__name__})"
    for k, v in NEED.items():
        got = ev.get(k, ABSENT)
        if type(got) is not type(v) or got != v:
            return f"{k} = {show(got)} (want {json.dumps(v)})"
    lat = ev.get("canonicalization_latency_ms", ABSENT)      # r3b-F1 fix 2: measured, positive, finite
    if type(lat) is not float or not math.isfinite(lat) or lat <= 0:
        return f"canonicalization_latency_ms = {show(lat)} (want a positive finite float)"
    why = gpu_why(ev)
    if why:
        return why
    log = ev.get("log", ABSENT)      # the evidence is bound to the bytes of the log it summarizes
    if not (isinstance(log, str) and os.path.isfile(log)):
        return f"log = {show(log)} (want an existing log file)"
    try:
        digest = hashlib.sha256(open(log, "rb").read()).hexdigest()
    except OSError as e:
        return f"log = {show(log)} (unreadable: {type(e).__name__})"
    if digest != ev.get("log_sha256"):
        return f"log_sha256 = {show(ev.get('log_sha256', ABSENT))} (want {digest}, the log's current sha256)"
    return None
first = None
for p in sorted(glob.glob(os.path.join(exp, "depth_canon_smoke_evidence_*.json")), reverse=True):
    why = why_not(p)
    if why is None:
        print(f"OK {p}")
        sys.exit(0)
    first = first or f"newest candidate {os.path.basename(p)} refused: {why}"
print("NO " + (first or f"no depth_canon_smoke_evidence_*.json in {exp}").replace("\r", " ").replace("\n", " "))
PY
)"
  SMOKE_EVIDENCE=""
  case "$EV_CHECK" in "OK "?*) SMOKE_EVIDENCE="${EV_CHECK#OK }" ;; esac
  [ -n "$SMOKE_EVIDENCE" ] || {
    echo "FULL requires a PASSING smoke evidence bound to this content (manifest ${MANIFEST_SHA}, launcher ${LAUNCHER_SHA}, pins generator ${MAKE_PINS_SHA}, contract ${CONTRACT_SHA}, audit ${AUDIT_SHA}, audit gate ${AUDIT_GATE_SHA}; harness=${HARNESS_FLAG}) in ${EXPDIR} - none qualifies - run MODE=SMOKE first - abort"
    case "$EV_CHECK" in
      "NO "?*) echo "evidence check: ${EV_CHECK#NO }" ;;
      *) echo "evidence check: the checker printed no verdict ('${EV_CHECK//$'\n'/|}'; its error is above)" ;;
    esac
    exit 2; }
  echo "smoke evidence: ${SMOKE_EVIDENCE}"
fi

# --- (6) NAS gate on the save root: identity, free space, write probe --------- #
nas_identity_ok() {   # the CIFS share must be what holds <path>
  local line
  line="$(findmnt -n -T "$1" -o FSTYPE,SOURCE 2>/dev/null || true)"
  grep -qE "^${NAS_FSTYPE}[[:space:]]+${NAS_SOURCE//./\\.}\$" <<<"$line" && return 0
  echo "NAS GATE FAILED: identity -- ${1} is on '$(tr '\n' ';' <<<"$line")', not '${NAS_FSTYPE} ${NAS_SOURCE}' - abort"
  return 1
}
nas_gate() {
  local anc="$SAVE_ROOT" avail probe
  while [ ! -e "$anc" ] && [ "$anc" != "/" ]; do anc="$(dirname "$anc")"; done
  nas_identity_ok "$anc" || return 1
  avail="$(df -P -B1 "$anc" 2>/dev/null | awk 'NR==2{print $4}' || true)"
  [[ "$avail" =~ ^[0-9]+$ ]] || { echo "NAS GATE FAILED: free space -- df failed on ${anc} - abort"; return 1; }
  [ "$avail" -ge $((MIN_NAS_FREE_GIB * 1024 * 1024 * 1024)) ] || {
    echo "NAS GATE FAILED: free space -- ${avail} bytes available on ${anc} < ${MIN_NAS_FREE_GIB} GiB - abort"; return 1; }
  if [ "$DRY_RUN" = "1" ]; then
    echo "NAS gate OK: ${anc} is ${NAS_FSTYPE} ${NAS_SOURCE}, ${avail} bytes free; write probe SKIPPED under DRY_RUN"
    return 0
  fi
  mkdir -p "$SAVE_ROOT" || { echo "NAS GATE FAILED: write probe -- cannot create ${SAVE_ROOT} - abort"; return 1; }
  nas_identity_ok "$SAVE_ROOT" || return 1
  probe="$SAVE_ROOT/.depth_canon_write_probe_$$"
  if ! { printf 'probe\n' > "$probe" && [ "$(cat "$probe")" = probe ] && rm -f "$probe" && [ ! -e "$probe" ]; } 2>/dev/null; then
    rm -f "$probe" 2>/dev/null || true
    echo "NAS GATE FAILED: write probe -- could not create, read back and remove a file in ${SAVE_ROOT} - abort"; return 1
  fi
  echo "NAS gate OK: ${SAVE_ROOT} is ${NAS_FSTYPE} ${NAS_SOURCE}, ${avail} bytes free, write probe created + removed"
}
nas_gate || exit 2

# --- (5) run-owned namespaces ---------------------------------------------------- #
if [ "$MODE" = "FULL" ]; then
  if [ -e "$RUN_NS" ] && [ -n "$(find "$RUN_NS" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "RUN NAMESPACE NOT EMPTY: ${RUN_NS} already holds files - a FULL run must own its namespace; move it aside deliberately - abort"; exit 2
  fi
elif [ -n "$(find "$RUN_NS" -name '*.ckpt' -print -quit 2>/dev/null)" ]; then
  echo "SMOKE NAMESPACE HOLDS CHECKPOINTS under ${RUN_NS} - a smoke must never leave one; move them aside - abort"; exit 2
fi
echo "run namespace OK: ${RUN_NS}"

# --- (8) GPU free memory on both cards + occupancy per GPU_OCCUPANCY_POLICY (never kills anything) --- #
declare -A FREE_OF=()
COT_N=(0 0)                       # co-tenant compute processes on GPU 0 / GPU 1
COTENANTS=()                      # gpu pid used_mib cwd -- four words per co-tenant, for the SMOKE evidence
if [ "$DRY_RUN" = "1" ]; then
  echo "GPU memory gate SKIPPED under DRY_RUN"
else
  # rA A3: every query's exit status is checked (a nonzero status refuses even after plausible
  # output), memory is exactly one number per card, every compute-app row parses and names a
  # known uuid. rA A6e: EVERY GPU-gate refusal exits 7 (gpu_refuse) -- the watcher's one transient class.
  gpu_query() {   # gpu_query <nvidia-smi args...> -> GPU_OUT; refuses (gpu_refuse) on any nonzero status
    local rc=0
    GPU_OUT="$(nvidia-smi "$@" 2>/dev/null)" || rc=$?
    [ "$rc" -eq 0 ] || gpu_refuse "GPU GATE FAILED: nvidia-smi $* exited ${rc} (stdout '${GPU_OUT//$'\n'/|}') - refusing to launch blind"
  }
  for G in 0 1; do
    gpu_query -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits
    [[ "$GPU_OUT" =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]] || gpu_refuse \
      "GPU GATE FAILED: the free memory of GPU ${G} is not exactly one number ('${GPU_OUT//$'\n'/|}') - refusing to launch blind"
    FREE="${BASH_REMATCH[1]}"; FREE_OF[$G]="$FREE"
    [ "$FREE" -ge "$MIN_GPU_FREE_MIB" ] || gpu_refuse \
      "GPU GATE FAILED: GPU ${G} free ${FREE} MiB < ${MIN_GPU_FREE_MIB} MiB (12 GiB) - abort; wait for the card, this launcher kills nothing"
  done
  echo "GPU memory gate OK: GPU 0 and GPU 1 each >= ${MIN_GPU_FREE_MIB} MiB free"
  # r3b-F1 fix 7 (Codex r3b HIGH 7) made ANY compute process on GPU 0 / GPU 1 refuse -- now policy `idle`.
  # rC-F1 (plan v14): under the pinned `shared` they are co-tenants, disclosed (pid, card, used MiB, cwd) and
  # never refused; the strict parsing below refuses either way. Read-only: nothing is signalled.
  declare -A UUID_OF=() IDX_OF=()
  gpu_query --query-gpu=index,uuid --format=csv,noheader
  while IFS= read -r line; do
    [[ "$line" =~ ^([0-9]+),[[:space:]]*(GPU-[0-9A-Za-z-]+)$ ]] || gpu_refuse \
      "GPU GATE FAILED: malformed index,uuid row '${line}' - refusing to launch blind"
    [ -z "${UUID_OF[${BASH_REMATCH[1]}]:-}" ] && [ -z "${IDX_OF[${BASH_REMATCH[2]}]:-}" ] || gpu_refuse \
      "GPU GATE FAILED: a duplicate index or uuid in '${line}' - refusing to launch blind"
    UUID_OF[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"; IDX_OF[${BASH_REMATCH[2]}]="${BASH_REMATCH[1]}"
  done <<<"$GPU_OUT"
  for G in 0 1; do
    [ -n "${UUID_OF[$G]:-}" ] || gpu_refuse "GPU GATE FAILED: no uuid for GPU ${G} - refusing to launch blind"
  done
  gpu_query --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits
  BUSY=""; OTHER=""
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    [[ "$line" =~ ^([0-9]+),[[:space:]]*(GPU-[0-9A-Za-z-]+),[[:space:]]*([0-9]+)$ ]] || gpu_refuse \
      "GPU GATE FAILED: malformed compute-app row '${line}' (want 'pid, uuid, MiB') - refusing to launch blind"
    APID="${BASH_REMATCH[1]}"; AUUID="${BASH_REMATCH[2]}"; AMIB="${BASH_REMATCH[3]}"
    [ -n "${IDX_OF[$AUUID]:-}" ] || gpu_refuse \
      "GPU GATE FAILED: compute process ${APID} is on an unknown uuid ${AUUID} - refusing to launch blind"
    AIDX="${IDX_OF[$AUUID]}"
    case "$AIDX" in
      0|1) ACWD="$(readlink "/proc/${APID}/cwd" 2>/dev/null || echo '?')"
           BUSY="${BUSY}  pid ${APID} on GPU ${AIDX} (${AUUID}): ${AMIB} MiB, cwd ${ACWD}"$'\n'
           COT_N[AIDX]=$((COT_N[AIDX] + 1)); COTENANTS+=("$AIDX" "$APID" "$AMIB" "$ACWD") ;;
      *) OTHER="${OTHER}  pid ${APID} on GPU ${AIDX} (${AUUID})"$'\n' ;;
    esac
  done <<<"$GPU_OUT"
  if [ "$GPU_OCCUPANCY_POLICY" = idle ]; then
    if [ -n "$BUSY" ]; then
      echo "GPU OCCUPIED: foreign compute process(es) on GPU 0/1 - this launcher never signals another process; wait for them to end:"
      gpu_refuse "$(printf '%s' "$BUSY")"
    fi
    echo "GPU occupancy gate OK: no compute process on GPU 0 or GPU 1"
  elif [ -n "$BUSY" ]; then
    echo "GPU occupancy gate: policy=shared - $((COT_N[0] + COT_N[1])) co-tenant compute process(es) on GPU 0/1 disclosed, not refused:"
    printf '%s' "$BUSY"
  else
    echo "GPU occupancy gate: policy=shared - no compute process on GPU 0 or GPU 1"
  fi
  # rB-F2 F2-2 (Codex A2 HIGH 2): no further, best-effort nvidia-smi query -- the disclosure is the
  # strict gate's own validated answers
  echo "--- co-tenancy disclosure at launch (the validated gate answers above) ---"
  for G in 0 1; do echo "GPU ${G} ${UUID_OF[$G]}: ${FREE_OF[$G]} MiB free, ${COT_N[G]} co-tenant process(es)"; done
  if [ -n "$OTHER" ]; then echo "compute processes on other cards:"; printf '%s' "$OTHER"
  else echo "compute processes on other cards: none"; fi
fi

# --- W&B identity (exp_17 gate; not under DRY_RUN, not in harness mode) --------- #
if [ "$DRY_RUN" = "1" ]; then
  echo "W&B identity gate SKIPPED under DRY_RUN"
elif [ "$HARNESS_FLAG" = true ]; then
  echo "W&B identity gate SKIPPED (harness)"
else
  eval "$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1 || true)"
  "$PY" - "$WANDB_IDENTITY" <<'PY' || { echo "W&B IDENTITY GATE FAILED: not ${WANDB_IDENTITY} - abort"; exit 2; }
import sys
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:  # noqa: BLE001
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == sys.argv[1] else 2)
PY
fi

# --- DINOv3 pin + exp_07 wiring / init identity, offline ------------------------ #
# src is imported from THIS checkout before the gate runs: the gate's own .git
# marker walk stops at a parent directory's .git when run from a git worktree.
HF_HUB_OFFLINE=1 "$PY" - "$REPO" "$DINO_GATE" <<'PY' || { echo "DINOv3/init-identity GATE FAILED - abort"; exit 2; }
import os, runpy, sys
repo, gate = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import src
origin = os.path.realpath(os.path.dirname(src.__file__))
if origin != os.path.realpath(os.path.join(repo, "src")):
    sys.exit(f"src would be imported from {origin}, not {repo}/src")
print(f"src imported from {origin}")
runpy.run_path(gate, run_name="__main__")
PY

echo "--- env manifest ---"
echo "python: ${PY}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
"$PY" -m pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}' || true

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; train.py NOT launched"; exit 0
fi
# --- launch -------------------------------------------------------------------- #
LAUNCH_MARK="$SAVE_ROOT/.depth_canon_launch_${EXPNAME}_${TS}"
: > "$LAUNCH_MARK" || { echo "cannot write the launch marker ${LAUNCH_MARK} - abort"; exit 2; }
echo "launch marker: ${LAUNCH_MARK} (its filesystem mtime is this run's ownership epoch)"
START_EPOCH="$(date +%s)"
rc=0
export WANDB_MODE="$WANDB_MODE_PIN"   # rC-F3: the pin, overriding any caller value (see the PINNED block)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 "${ARGV[@]}" || rc=$?
END_EPOCH="$(date +%s)"
echo "=== exp_25 depth_canon ${MODE}: train.py exited rc=${rc} after $((END_EPOCH - START_EPOCH)) s ==="

# --- (3) drain the tee, then read the log ONCE (no tr|grep pipeline under pipefail)
drain_log() {
  exec 1>&3 2>&4                       # our ends of the tee pipe close; tee sees EOF
  local i
  for ((i = 0; i < 600; i++)); do kill -0 "$TEE_PID" 2>/dev/null || break; sleep 0.2; done
  if kill -0 "$TEE_PID" 2>/dev/null; then echo "WARNING: the log tee is still open after 120 s; reading ${LOG} as it stands"; fi
  LOGTEXT="$(tr '\r' '\n' < "$LOG")"
}
drain_log
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!
fail() { [ "$rc" -ne 0 ] || rc="$1"; }   # the first failure decides the exit code

if grep -qxF "$BANNER" <<<"$LOGTEXT"; then
  BANNER_SEEN=true; echo "treatment banner: FOUND (whole line '${BANNER}')"
else
  BANNER_SEEN=false; echo "TREATMENT BANNER NOT FOUND in ${LOG} - the run may have trained WITHOUT canonicalization - invalid"; fail 3
fi
if grep -qF "All distributed processes registered. Starting with 2 processes" <<<"$LOGTEXT"; then
  RANKS_SEEN=true
else
  RANKS_SEEN=false; echo "TWO-RANK DDP START NOT FOUND in ${LOG} - not the registered recipe - invalid"; fail 6
fi
if grep -qF "\`Trainer.fit\` stopped: \`max_steps=${MAXSTEPS}\` reached." <<<"$LOGTEXT"; then
  ENDPOINT_SEEN=true
else
  ENDPOINT_SEEN=false; echo "ENDPOINT NOT REACHED: Lightning's max_steps=${MAXSTEPS} termination marker is absent - invalid"; fail 6
fi
if grep -qiE '\b(nan|inf)\b' <<<"$LOGTEXT"; then
  NONFINITE=true
  echo "NON-FINITE VALUES in ${LOG}: $(grep -m1 -iE '\b(nan|inf)\b' <<<"$LOGTEXT" | cut -c1-200) - the fit is not healthy"; fail 4
else
  NONFINITE=false; echo "non-finite scan: clean"
fi

# --- (5) run-owned checkpoints (FULL) / none at all (SMOKE) ---------------------- #
CKPT_MANIFEST="$EXPDIR/depth_canon_${TS}_${EXPNAME}_checkpoints.sha256"
ckpt_rc=0
"$PY" - "$RUN_NS" "$LAUNCH_MARK" "$MODE" "$MAXSTEPS" "$CADENCE" "$CKPT_MANIFEST" <<'PY' || ckpt_rc=$?
import hashlib, os, re, sys
ns, mark, mode, endpoint, cadence, manifest = sys.argv[1:7]
endpoint, cadence = int(endpoint), int(cadence)
files = sorted(os.path.relpath(os.path.join(d, f), ns) for d, _, fs in os.walk(ns) for f in fs)
if mode == "SMOKE":
    ckpts = [f for f in files if f.endswith(".ckpt")]
    if ckpts:
        print(f"SMOKE WROTE {len(ckpts)} CHECKPOINT(S) under {ns}: {ckpts[:5]} - the cadence pin failed")
        sys.exit(5)
    print(f"smoke checkpoints: 0 under {ns} (cadence {cadence} >> {endpoint} steps)")
    sys.exit(0)
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
with open(manifest + ".tmp", "w") as out:            # every file found, for the record
    for f in files:
        out.write(f"{sha(os.path.join(ns, f))}  {os.path.join(ns, f)}\n")
os.replace(manifest + ".tmp", manifest)
epoch = os.stat(mark).st_mtime                        # same filesystem clock as the checkpoints
pat = re.compile(r"checkpoints/epoch=\d+-step=(\d+)\.ckpt")
steps, foreign, stale = [], [], []
for f in files:
    m = pat.fullmatch(f)
    if m is None:
        foreign.append(f)
        continue
    steps.append(int(m.group(1)))
    if os.stat(os.path.join(ns, f)).st_mtime < epoch:
        stale.append(f)
print(f"checkpoint manifest: {manifest} ({len(files)} files)")
if endpoint not in steps:
    print(f"ENDPOINT CHECKPOINT MISSING: no checkpoints/epoch=*-step={endpoint}.ckpt under {ns}")
    sys.exit(6)
want = list(range(cadence, endpoint + 1, cadence))
problems = []
if foreign:
    problems.append(f"foreign files {foreign[:5]}")
if stale:
    problems.append(f"files older than the launch marker {stale[:5]}")
if sorted(steps) != want:
    problems.append(f"cadence set {sorted(steps)} != {cadence}, {2 * cadence}, ..., {endpoint}")
if problems:
    print("CHECKPOINT SET REJECTED: " + "; ".join(problems))
    sys.exit(5)
print(f"checkpoint set OK: exactly the {len(want)} cadence checkpoints, all newer than the launch marker")
PY
[ "$ckpt_rc" -eq 0 ] || fail "$ckpt_rc"

# --- (7) SMOKE: steady-state projection <= 120 h (v14), then the evidence --------- #
if [ "$MODE" = "SMOKE" ]; then
  EVIDENCE="$EXPDIR/depth_canon_smoke_evidence_${TS}.json"
  "$PY" - "$EVIDENCE.partial" "$LOG" "$rc" "$((END_EPOCH - START_EPOCH))" "$SMOKE_STEPS" "$ENDPOINT_STEPS" \
      "$MAX_PROJECTED_HOURS" "$HEAD_SHA" "$MANIFEST_SHA" "$HARNESS_FLAG" "$BANNER_SEEN" "$ENDPOINT_SEEN" \
      "$RANKS_SEEN" "$NONFINITE" "$ckpt_rc" "${ARGV[*]}" "$LAUNCHER_SHA" "$MAKE_PINS_SHA" "$CONTRACT_SHA" \
      "$AUDIT_SHA" "$AUDIT_GATE_SHA" "$GPU_OCCUPANCY_POLICY" "$MIN_GPU_FREE_MIB" "${FREE_OF[0]:-}" "${FREE_OF[1]:-}" \
      "${COTENANTS[@]}" <<'PY' || fail 4
import json, math, re, sys
(out, log, rc, elapsed, steps, endpoint, max_h, head, manifest, harness, banner, endpoint_seen,
 ranks, nonfinite, ckpt_rc, argv, launcher, make_pins, contract, audit, audit_gate,
 gpu_policy, min_free, free0, free1) = sys.argv[1:26]
cot = sys.argv[26:]               # rC-F1: the gate's validated co-tenants, four words each (none under DRY_RUN)
if len(cot) % 4:
    sys.exit(f"the co-tenant record has {len(cot)} words, not four per process")
co_tenants = [{"gpu": int(g), "pid": int(p), "used_mib": int(m), "cwd": c}
              for g, p, m, c in zip(cot[0::4], cot[1::4], cot[2::4], cot[3::4])]
free_at_launch = None if free0 == free1 == "" else {"0": int(free0), "1": int(free1)}
rc, elapsed, steps, endpoint, max_h = int(rc), float(elapsed), int(steps), int(endpoint), float(max_h)
text = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
stamps = {}   # tqdm '<n>/<total> [<elapsed><' stamps of the training progress bar
for m in re.finditer(r"(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<", text):
    stamps[int(m.group(1))] = int(m.group(3) or 0) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
late = [n for n in stamps if n >= 10]
rate = None
if late and max(stamps) > min(late) and stamps[max(stamps)] > stamps[min(late)]:
    n1, n2 = min(late), max(stamps)
    rate = (stamps[n2] - stamps[n1]) / (n2 - n1)
proj = None if rate is None else rate * endpoint / 3600
upper = elapsed / steps * endpoint / 3600
print(f"SMOKE: {steps} steps in {elapsed:.0f} s wall (startup included -> {upper:.1f} h, descriptive); "
      f"steady state {'n/a' if rate is None else f'{rate:.3f} s/step'} -> "
      f"{'n/a' if proj is None else f'{proj:.1f} h'} for {endpoint} steps")
def num(v):
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None
# r3b-F1 fix 2: a finite train/loss after a completed step (tqdm '| n/total [' with n >= 1; the
# value is bounded by [\s,[] on the left so '/pretrain/loss=' and 'mse_loss=' never count) ...
LOSS, STEP = re.compile(r"(?:^|[\s,\[])train/loss=([^,\]\s]+)"), re.compile(r"\| (\d+)/\d+ \[")
losses = [v for line in text.split("\n") for m in [STEP.search(line)] if m and int(m.group(1)) >= 1
          for v in LOSS.findall(line)]
parsed = [num(v) for v in losses]
finite_loss = bool(parsed) and all(p is not None for p in parsed)
# ... and rank 0's measured canonicalization latency (src/training/diffusion.py on_fit_end)
LAT = re.compile(r"^canonicalize latency rank0 batches=(\d+) samples=(\d+) total_s=(\S+) "
                 r"mean_ms_per_batch=(\S+)$", re.M)
found = LAT.findall(text)
latency = None
if len(found) == 1:
    b, n, t, ms = int(found[0][0]), int(found[0][1]), num(found[0][2]), num(found[0][3])
    if b >= 1 and n >= b and t and t > 0 and ms and ms > 0:
        latency = {"batches": b, "samples": n, "total_s": t, "mean_ms_per_batch": ms}
print(f"SMOKE: {len(losses)} train/loss value(s) after completed steps, all finite: {finite_loss}; "
      f"canonicalization latency: {latency}")
if rc != 0:
    verdict, why = "FAIL", f"an earlier check or train.py failed (rc {rc})"
elif proj is None:
    verdict, why = "FAIL", "the projection could not be computed (no tqdm rate stamps at n >= 10)"
elif proj > max_h:
    verdict, why = "FAIL", f"projected {proj:.1f} h > MAX_PROJECTED_HOURS={max_h:.0f}"
elif not finite_loss:
    verdict, why = "FAIL", f"no parseable finite training loss after a completed step (values {losses[:5]})"
elif latency is None:
    verdict, why = "FAIL", (f"the 'canonicalize latency rank0 ...' line is missing or invalid "
                            f"({len(found)} found)")
else:
    verdict, why = "PASS", f"projected {proj:.1f} h <= {max_h:.0f} h"
print(f"SMOKE VERDICT: {verdict} - {why}")
b = lambda s: s == "true"  # noqa: E731
ev = {"schema_version": 2, "kind": "depth_canon_smoke_evidence", "verdict": verdict, "why": why,
      "head": head, "manifest_sha256": manifest, "launcher_sha256": launcher,
      "make_pins_sha256": make_pins, "contract_sha256": contract, "audit_sha256": audit,
      "audit_gate_sha256": audit_gate, "harness": b(harness), "config_contract_ok": True,
      "banner_seen": b(banner), "endpoint_seen": b(endpoint_seen), "ranks_seen": b(ranks),
      "nonfinite_seen": b(nonfinite), "smoke_checkpoints": 0 if ckpt_rc == "0" else -1,
      "elapsed_s": elapsed, "startup_inclusive_projection_h": upper,
      "steady_state_s_per_step": rate, "steps_per_s": None if not rate else 1.0 / rate,
      "projection_h": proj, "max_projected_hours": int(max_h), "finite_loss_seen": finite_loss,
      "train_loss_values_seen": len(losses), "last_train_loss": parsed[-1] if parsed else None,
      "canonicalization_latency": latency,
      "canonicalization_latency_ms": latency["mean_ms_per_batch"] if latency else None,
      "gpu_policy": gpu_policy, "min_gpu_free_mib": int(min_free), "free_mib_at_launch": free_at_launch,
      "co_tenants_at_launch": co_tenants, "train_rc": rc, "argv": argv, "log": log}
json.dump(ev, open(out, "w"), indent=1)
sys.exit(0 if verdict == "PASS" else 4)
PY
fi

echo "=== exp_25 depth_canon ${MODE} finished: exit ${rc} ==="
if [ "$MODE" = "SMOKE" ]; then
  drain_log                             # the evidence binds the FINAL bytes of the log
  "$PY" - "$EVIDENCE.partial" "$EVIDENCE" "$LOG" <<'PY' || { echo "could not publish the smoke evidence"; exit 2; }
import hashlib, json, os, sys
partial, final, log = sys.argv[1:4]
ev = json.load(open(partial))
ev["log_sha256"] = hashlib.sha256(open(log, "rb").read()).hexdigest()
json.dump(ev, open(partial, "w"), indent=1)
os.replace(partial, final)
print(f"smoke evidence ({ev['verdict']}): {final}")
PY
fi
exit "$rc"
