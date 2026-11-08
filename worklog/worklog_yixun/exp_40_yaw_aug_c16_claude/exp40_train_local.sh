#!/usr/bin/env bash
# ============================================================================
# exp40_train_local.sh -- exp_40 yaw_aug_c16 training on the 2xRTX A6000 box (plan §4, §14).
# Thin sequencing over exp40_kit.py; every decision is made (and unit-tested) there.
#
#   MODE=SMOKE|DDP_SMOKE|FULL  PLATFORM=a6000  RUNG=2x32  RUN_ID=exp40_C16_a6000_2x32[_r<n>]
#   EXPECT_SHA=<40-hex>  [DRY_RUN=0|1]  [GPU_POLICY=idle|shared (smokes only)]  [SMOKE_GPU=0]
#   DDP_SMOKE / FULL also need GPU_RESERVATION_OK=1 (cards sequenced through the
#   ~/codespace/claude_gpu_yield/ handshake); every mode takes the cooperative per-card
#   locks claude_gpu_yield/GPU_LOCK_<uuid> (O_EXCL, owner token) until it ends.
#   bash worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh
#
# Order: exp_40 card flock -> sealed HF hub -> immutable run tree (git archive EXPECT_SHA +
# 2 allowlisted links) -> every no-GPU gate (the run tree's own kit) -> GPU gate ->
# [DRY_RUN stops] -> per-card locks -> GPU re-check -> launch record -> setsid evidence
# driver (cwd = run tree; FULL publishes + NAS-acknowledges every checkpoint right after
# its save, inside the training) -> start watcher -> wait -> evidence (+ FULL audit).
# Never signals a process it did not start; the watcher and the EXIT/INT/TERM cleanup trap
# terminate only the training's own session (SIGTERM, bounded wait, SIGKILL of that group).
# FULL ends with a stable run-result manifest (receipts/run_result_<RUN_ID>.json) from which
# `exp40_kit.py admit` derives the admission. Exit: 0 PASS | 1 a verdict FAILED | 2 a gate
# refused | 8/9/10/11 watcher | 12 cleanup failed | 130 interrupted.
# Written by the exp_40 Coder seat (Claude Opus 5.5) per plan_yaw_aug_c16.md §4 / §14.
# ============================================================================
set -uo pipefail
die() { echo "REFUSED: $*"; exit 2; }

SELF_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(git -C "$SELF_DIR" rev-parse --show-toplevel)" || die "not inside a git checkout"
EXPREL="worklog/worklog_yixun/exp_40_yaw_aug_c16_claude"
PY="/home/yixunhu/miniconda3/envs/flac/bin/python"
MODE="${MODE:-}"; PLATFORM="${PLATFORM:-}"; RUNG="${RUNG:-}"; RUN_ID="${RUN_ID:-}"
EXPECT_SHA="${EXPECT_SHA:-}"; DRY_RUN="${DRY_RUN:-0}"; GPU_POLICY="${GPU_POLICY:-idle}"
SMOKE_GPU="${SMOKE_GPU:-0}"; GPU_RESERVATION_OK="${GPU_RESERVATION_OK:-0}"
case "$MODE" in SMOKE|DDP_SMOKE|FULL) ;; *) die "MODE must be SMOKE, DDP_SMOKE or FULL (got '${MODE}')" ;; esac
[ "$PLATFORM" = a6000 ] || die "PLATFORM must be a6000 on this box (got '${PLATFORM}')"
[ "$RUNG" = 2x32 ] || die "RUNG must be 2x32 on this box (got '${RUNG}')"
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1" ;; esac
[[ "$EXPECT_SHA" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_SHA must be the full 40-hex commit"
[[ "$SMOKE_GPU" =~ ^[0-9]$ ]] || die "SMOKE_GPU must be a GPU index"
[ -x "$PY" ] || die "pinned interpreter $PY missing"

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
BASE="$REPO/outputs_FLAC/exp40_yaw_aug_c16"
if [ "$MODE" = FULL ]; then
  SAVE_ROOT="$BASE"; NS="$RUN_ID"; LANES="0,1"; MEMBERS=2; PUBLISH=nas
else
  SAVE_ROOT="$REPO/outputs_FLAC/exp40_smoke"; NS="${RUN_ID}_$(echo "$MODE" | tr 'A-Z' 'a-z')_${TS}"; PUBLISH=off
  if [ "$MODE" = SMOKE ]; then LANES="$SMOKE_GPU"; MEMBERS=1; else LANES="0,1"; MEMBERS=2; fi
fi
RUNTREE="$BASE/runtrees/${NS}__${EXPECT_SHA:0:12}"
[ "$DRY_RUN" = 1 ] && RUNTREE="${RUNTREE}__dryrun_${TS}"
LOGBASE="$REPO/$EXPREL/yaw_aug_c16_${TS}_${MODE}"; [ "$DRY_RUN" = 1 ] && LOGBASE="${LOGBASE}_DRYRUN"
RECEIPTS="$REPO/$EXPREL/receipts"; EVDIR="$SAVE_ROOT/evidence/$NS"
CKPT_DIR="$SAVE_ROOT/FLAC_exp40_C16/$NS/checkpoints"
NAS_DIR="/media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/$RUN_ID"
HF_SEALED="$BASE/hf_sealed"
mkdir -p "$BASE/runtrees" "$RECEIPTS" || die "cannot create $BASE/runtrees / $RECEIPTS"
exec > >(tee -a "${LOGBASE}_launcher.log") 2>&1
echo "=== exp_40 C16 ${MODE} (dry_run=${DRY_RUN}) -- ${TS} -- host $(hostname) -- source HEAD $(git -C "$REPO" rev-parse HEAD) ==="

# --- environment: nothing injected; sealed offline HF; offline W&B; no bytecode --------- #
unset PYTHONPATH PYTHONSTARTUP PYTHONHOME PYTHONUSERBASE TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$HF_SEALED" HF_HUB_CACHE="$HF_SEALED/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RT="$SAVE_ROOT/runtime/$NS"
# TMPDIR stays SHORT: torch_shm_manager (file_system sharing, train.py) binds a UNIX socket
# under it, and a socket path over 108 bytes fails with EINVAL (smoke 2026-11-07 22:22).
TMPDIR="$(mktemp -d /tmp/exp40.XXXXXX)" || die "mktemp failed"
export WANDB_MODE=offline WANDB_DIR="$RT/wandb" TMPDIR XDG_CACHE_HOME="$RT/xdg" \
  TORCH_HOME="$RT/torch" TRITON_CACHE_DIR="$RT/triton" MPLCONFIGDIR="$RT/mpl"
EXP40_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u)"
[ "$(printf '%s\n' "$EXP40_DRIVER_VERSION" | wc -l)" = 1 ] || die "not exactly one NVIDIA driver version: $EXP40_DRIVER_VERSION"
export EXP40_DRIVER_VERSION

# --- one exp_40 user of the cards at a time (training and evaluation share it) --------- #
CARDS_LOCK="$HOME/.exp40_cards.lock"          # checkout-independent; the eval driver takes it too
exec 8>"$CARDS_LOCK" || die "cannot open the card lock"
flock -n 8 || die "another exp_40 training/evaluation holds $CARDS_LOCK"

# --- sealed HF hub (built once), immutable run tree -------------------------------------- #
[ -d "$HF_SEALED" ] || "$PY" "$REPO/$EXPREL/exp40_kit.py" build-hf-sealed \
  --src-hub /home/yixunhu/.cache/huggingface/hub --dest "$HF_SEALED" || die "sealed HF hub build failed"
AR_REAL="$(readlink -f "$REPO/AcousticRooms")" || die "AcousticRooms link missing"
VAE_REAL="$(readlink -f "$REPO/weights/FLAC/VAE.safetensors")" || die "VAE link missing"
LINKS=(--link "AcousticRooms=$AR_REAL" --link "weights/FLAC/VAE.safetensors=$VAE_REAL")
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECT_SHA" ] || die "source HEAD != EXPECT_SHA"
"$PY" "$REPO/$EXPREL/exp40_kit.py" build-runtree --repo "$REPO" --sha "$EXPECT_SHA" \
  --dest "$RUNTREE" "${LINKS[@]}" || die "run tree build failed"
KIT="$RUNTREE/$EXPREL/exp40_kit.py"; DRIVER="$RUNTREE/$EXPREL/exp40_evidence_driver.py"
echo "run tree: $RUNTREE"

# --- every no-GPU gate, from the run tree's own kit ------------------------------------ #
mkdir -p "$WANDB_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" \
  || die "cannot create the runtime dirs"
CHECK_JSON="${LOGBASE}_check.json"
"$PY" "$KIT" check --platform "$PLATFORM" --rung "$RUNG" --mode "$MODE" --run-id "$RUN_ID" \
  --save-root "$SAVE_ROOT" --stamp "$TS" --sha "$EXPECT_SHA" --repo "$REPO" --tree "$RUNTREE" \
  "${LINKS[@]}" --receipts "$RECEIPTS" --hf-sealed "$HF_SEALED" --out "$CHECK_JSON" \
  || die "gates refused (see $CHECK_JSON)"
GPUS_JSON="${LOGBASE}_gpus.json"
"$PY" "$KIT" gpus --platform "$PLATFORM" --rung "$RUNG" --mode "$MODE" --lanes "$LANES" \
  --policy "$GPU_POLICY" --expect-total 2 --out "$GPUS_JSON" || die "GPU gate refused (see $GPUS_JSON)"
mapfile -t ARGV < <("$PY" "$KIT" argv --platform "$PLATFORM" --rung "$RUNG" --mode "$MODE" \
  --run-id "$RUN_ID" --save-root "$SAVE_ROOT" --stamp "$TS") || die "argv failed"
echo "argv: train.py ${ARGV[*]}"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY_RUN: every gate passed; nothing spawned, no lock taken. Run tree kept: $RUNTREE"
  exit 0
fi

# --- cooperative per-card locks, then the GPU re-check ------------------------------------ #
[ "$MODE" = SMOKE ] || [ "$GPU_RESERVATION_OK" = 1 ] \
  || die "DDP_SMOKE / FULL need GPU_RESERVATION_OK=1 after sequencing the cards via ~/codespace/claude_gpu_yield/"
UUIDS="$("$PY" -c "import json,sys; g=json.load(open(sys.argv[1])); l=set(g['lanes']); print(','.join(x['uuid'] for x in g['gpus'] if x['index'] in l))" "$GPUS_JSON")"
LOCK_JSON="${LOGBASE}_gpu_locks.json"

# --- signal-safe cleanup (code-r1 B4), installed BEFORE any resource is taken ------------- #
# Tracks what this launcher owns; on EXIT / INT / TERM it stops the watcher, terminates ONLY
# the training's own process group (SIGTERM, bounded wait, SIGKILL of that group), reaps both
# children and releases ONLY the locks recorded in LOCK_JSON. Any cleanup failure makes the
# launcher fail.
TRAIN_PID=""; WATCH_PID=""; CLEANED=0; INTERRUPTED=""
cleanup() {
  local rc=$? bad=0
  [ "$CLEANED" = 1 ] && return
  CLEANED=1
  trap - EXIT INT TERM
  if [ -n "$WATCH_PID" ] && kill -0 "$WATCH_PID" 2>/dev/null; then
    kill -TERM "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null
  fi
  if [ -n "$TRAIN_PID" ]; then
    "$PY" "$KIT" stop-group --pgid "$TRAIN_PID" --grace 120 || { echo "CLEANUP: training group $TRAIN_PID survived"; bad=1; }
    wait "$TRAIN_PID" 2>/dev/null
  fi
  if [ -f "$LOCK_JSON" ]; then
    if "$PY" "$KIT" gpu-unlock --lock-json "$LOCK_JSON"; then
      mv "$LOCK_JSON" "${LOCK_JSON%.json}_released.json" || bad=1
    else
      echo "CLEANUP: could not release the per-card locks in $LOCK_JSON"; bad=1
    fi
  fi
  [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && rm -rf -- "$TMPDIR"
  if [ -n "$INTERRUPTED" ]; then
    echo "=== exp_40 C16 ${MODE} INTERRUPTED by SIG${INTERRUPTED} $(date '+%Y-%m-%d %H:%M:%S'): FAIL ==="
    exit 130
  fi
  [ "$bad" = 0 ] || { echo "=== cleanup FAILED: exit 12 ==="; exit 12; }
  exit "$rc"
}
on_signal() { INTERRUPTED="$1"; exit 130; }
trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

"$PY" "$KIT" gpu-lock --uuids "$UUIDS" --out "$LOCK_JSON" \
  --payload-json "{\"experiment\": \"exp_40\", \"namespace\": \"$NS\", \"host\": \"$(hostname)\", \"pid\": $$}" \
  || die "per-card locks refused"
"$PY" "$KIT" gpus --platform "$PLATFORM" --rung "$RUNG" --mode "$MODE" --lanes "$LANES" \
  --policy "$GPU_POLICY" --expect-total 2 --out "${LOGBASE}_gpus_prespawn.json" \
  || die "GPU re-check after taking the locks refused"

# --- launch record, written before the fork ---------------------------------------------- #
"$PY" - "${LOGBASE}_launch.json" "$CHECK_JSON" "${LOGBASE}_gpus_prespawn.json" "$LOCK_JSON" "$RUNTREE" "$NS" "$LANES" "$GPU_POLICY" <<'PYEOF' || die "launch record failed"
import glob, hashlib, json, os, socket, sys, time
out, check, gpus, locks, tree, ns, lanes, policy = sys.argv[1:]
hs = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()
      for p in sorted(glob.glob(os.path.expanduser("~/codespace/claude_gpu_yield/*"))) if os.path.isfile(p)}
json.dump({"host": socket.gethostname(), "t": time.time(), "runtree": tree, "namespace": ns,
           "lanes": lanes, "gpu_policy": policy, "check": json.load(open(check)),
           "gpus": json.load(open(gpus)), "gpu_locks": json.load(open(locks))["paths"],
           "gpu_yield_handshake_sha256": hs}, open(out, "w"), indent=1, sort_keys=True)
PYEOF

# --- spawn: own session, cwd = run tree, the evidence driver runs train.py ------------ #
mkdir -p "$EVDIR" || die "cannot create $EVDIR"
TRAIN_LOG="${LOGBASE}_train.log"
EVMODE=verbatim; [ "$MODE" = FULL ] && EVMODE=digest
( cd "$RUNTREE" && CUDA_VISIBLE_DEVICES="$LANES" EXP40_EVIDENCE_DIR="$EVDIR" EXP40_EVIDENCE_MODE="$EVMODE" \
    EXP40_PUBLISH="$PUBLISH" EXP40_RUN_ID="$RUN_ID" exec setsid "$PY" "$DRIVER" "${ARGV[@]}" ) > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
sleep 2
[ "$(ps -o pgid= -p "$TRAIN_PID" | tr -d ' ')" = "$TRAIN_PID" ] \
  || die "setsid did not give the training its own process group"
echo "spawned pid/pgid $TRAIN_PID; log $TRAIN_LOG"
"$PY" "$KIT" watch --log "$TRAIN_LOG" --members "$MEMBERS" --pid "$TRAIN_PID" --pgid "$TRAIN_PID" \
  --timeout 3600 &
WATCH_PID=$!
wait "$TRAIN_PID"; TRAIN_RC=$?
wait "$WATCH_PID"; WATCH_RC=$?
WATCH_PID=""
# stragglers of the training's own group (e.g. an orphaned DDP rank) are terminated too
"$PY" "$KIT" stop-group --pgid "$TRAIN_PID" --grace 120 || die "training group $TRAIN_PID survived"
TRAIN_PID=""
echo "train rc=${TRAIN_RC} watcher rc=${WATCH_RC}"

# --- verdicts ------------------------------------------------------------------------------ #
FINAL=0
if [ "$MODE" = FULL ]; then
  EVOUT="$RECEIPTS/full_evidence_${RUN_ID}.json"
  REF="$RECEIPTS/ddp_smoke_${PLATFORM}_${RUNG}_${EXPECT_SHA}.json"
  AUDIT="$RECEIPTS/audit_${RUN_ID}.json"
  "$PY" "$KIT" evidence --platform "$PLATFORM" --rung "$RUNG" --mode FULL --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$RECEIPTS" --train-rc "$TRAIN_RC" --watcher-rc "$WATCH_RC" \
    --reference "$REF" --out "$EVOUT" || FINAL=1
  "$PY" "$KIT" audit --ckpt-dir "$CKPT_DIR" --nas-dir "$NAS_DIR" --evidence "$EVOUT" --run-id "$RUN_ID" \
    --check "$CHECK_JSON" --reference "$REF" --out "$AUDIT" || FINAL=1
  # the stable run-result manifest (code-r1 M6): admission resolves every record from it
  cp "$CHECK_JSON" "$RECEIPTS/check_${RUN_ID}.json" || FINAL=1
  VERDICT=PASS; [ "$FINAL" = 0 ] && [ "$TRAIN_RC" = 0 ] && [ "$WATCH_RC" = 0 ] || VERDICT=FAIL
  "$PY" "$KIT" run-result --run-id "$RUN_ID" --sha "$EXPECT_SHA" --verdict "$VERDICT" \
    --check "$RECEIPTS/check_${RUN_ID}.json" --reference "$REF" --evidence "$EVOUT" --audit "$AUDIT" \
    --expected-dirpath "$CKPT_DIR" --ckpt-dir "$CKPT_DIR" --runtree "$RUNTREE" \
    --out "$RECEIPTS/run_result_${RUN_ID}.json" || FINAL=1
  echo "next (admission): $PY $REPO/$EXPREL/exp40_kit.py admit --run-result $RECEIPTS/run_result_${RUN_ID}.json --receipts-dir $RECEIPTS --out $REPO/$EXPREL/exp40_C16_admission.json"
else
  "$PY" "$KIT" evidence --platform "$PLATFORM" --rung "$RUNG" --mode "$MODE" --run-id "$RUN_ID" \
    --save-root "$SAVE_ROOT" --stamp "$TS" --evidence-dir "$EVDIR" --log "$TRAIN_LOG" \
    --check-json "$CHECK_JSON" --receipts "$RECEIPTS" --train-rc "$TRAIN_RC" --watcher-rc "$WATCH_RC" \
    --out "${LOGBASE}_evidence.json" || FINAL=1
  n_ckpt="$(find "$SAVE_ROOT/FLAC_exp40_C16/$NS" -name '*.ckpt' 2>/dev/null | wc -l)"
  [ "$n_ckpt" = 0 ] || { echo "a smoke wrote ${n_ckpt} checkpoint(s)"; FINAL=1; }
fi
[ "$TRAIN_RC" = 0 ] && [ "$WATCH_RC" = 0 ] || FINAL=1
echo "=== exp_40 C16 ${MODE} training finished $(date '+%Y-%m-%d %H:%M:%S'): $([ $FINAL = 0 ] && echo PASS || echo FAIL) (locks released at exit) ==="
[ "$WATCH_RC" -ge 8 ] && exit "$WATCH_RC"
exit "$FINAL"
