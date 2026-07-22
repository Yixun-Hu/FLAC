#!/usr/bin/env bash
# ============================================================================
# d_eval_driver.sh <CKPT> <CONFIG> <EVAL_NAME> <EXTERNAL_LOG_DIR> - exp-09 Stage D
# eval launcher TEMPLATE. Builds ONE eval_FLAC.py invocation on a SAVED checkpoint and
# runs it under the frozen free-VRAM gate. It NEVER auto-runs during C2: D evals run on
# saved checkpoints (C2 records prohibit co-tenant screening), and each eval is a GPU job.
#
# MANDATORY flags (Codex Stage-B blocker 1): --cond-method fa_invariant --frame-avg-angles 0
# are passed EXPLICITLY. eval_FLAC.py defaults cond_method to 'vanilla' and does NOT read it
# from the JSON training block (unlike train.py), so omitting them would silently evaluate
# the raw-pose vanilla trap instead of the native-invariant path.
#
# --rotate-deg <ROTATE_DEG> (env, default 0) is the D2 sweep parameter: set it per angle
# {0,45,90,180,270} to produce the rotated-input predictions H-A3 / the equivariance gates
# compare. eval_FLAC suffixes the output name by method/rotation, so sweep angles never
# collide.
#
# Eval ENV (plan §7): PYTHONPATH-FIRST. PYTHONPATH=<cylindrical-dinov3>/src is PREPENDED so
# the process-local package src wins with ZERO mutation of the eval env, and its git SHA is
# recorded for provenance. EVAL_PYTHON is parameterized (default the flac python): the
# rir2rir-vs-flac env decision is pinned in the D records, and BOTH are supported here -
# point EVAL_PYTHON at the rir2rir python to use that env (its torch/transformers pins match).
#
# Frozen VRAM gate: binds MIN_FREE_MB to the EXACT value the C1 fit probe froze into
# c1_frozen_min_free.txt (like exp09_launch.sh) - absent/non-numeric/override-mismatch all
# REFUSE. Refuses missing args with exit 3 + usage (guarded, no `set -u` unbound trips). The
# EXTERNAL log dir ($4) must be OUTSIDE the worktree (a teed log inside would dirty the
# clean-tree pin gate). DRY_RUN=1 prints the assembled command and exits 0 WITHOUT running.
# ============================================================================
set -o pipefail    # verbatim (records-referenced)
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
WORKTREE="$(readlink -f .)"
USAGE="usage: d_eval_driver.sh <CKPT> <CONFIG> <EVAL_NAME> <EXTERNAL_LOG_DIR>  (env: ROTATE_DEG EVAL_GPU EVAL_SEED EVAL_PYTHON CYL_DINOV3_SRC EVAL_DATASET_CONFIG DRY_RUN)"

# --- required positional args (guarded with ${N:-} so an absent arg hits the intended
# exit-3 REFUSE, never a `set -u` 'unbound variable' trip; messages use plain words) ---
CKPT="${1:-}"
[ -n "$CKPT" ] || { echo "$USAGE"; echo "  (arg 1 CKPT is required)"; exit 3; }
CONFIG="${2:-}"
[ -n "$CONFIG" ] || { echo "$USAGE"; echo "  (arg 2 CONFIG is required)"; exit 3; }
EVAL_NAME="${3:-}"
[ -n "$EVAL_NAME" ] || { echo "$USAGE"; echo "  (arg 3 EVAL_NAME is required)"; exit 3; }
LOGDIR="${4:-}"
[ -n "$LOGDIR" ] || { echo "$USAGE"; echo "  (arg 4 EXTERNAL_LOG_DIR is required)"; exit 3; }

[ -f "$CKPT" ] || { echo "REFUSING: checkpoint '${CKPT}' not found."; exit 3; }
[ -f "$CONFIG" ] || { echo "REFUSING: model-config '${CONFIG}' not found."; exit 3; }

# --- EXTERNAL log dir (must be OUTSIDE the worktree) ---
mkdir -p "$LOGDIR" 2>/dev/null || { echo "cannot create log dir '${LOGDIR}' - abort"; exit 3; }
LOGDIR_ABS="$(readlink -f "$LOGDIR")"
case "${LOGDIR_ABS}/" in
  "${WORKTREE}"/*)
    echo "REFUSING: log dir '${LOGDIR_ABS}' is INSIDE the worktree '${WORKTREE}' - a teed eval"
    echo "log there would dirty the clean-tree pin gate. Pass an EXTERNAL directory."
    exit 3;;
esac

# --- parameters (all overridable; defaults match the exp09_screen.sh eval shape) ---
EVAL_PYTHON="${EVAL_PYTHON:-/home/yixunhu/miniconda3/envs/flac/bin/python}"
CYL_DINOV3_SRC="${CYL_DINOV3_SRC:-/home/yixunhu/codespace/cylindrical-dinov3/src}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json}"
ROTATE_DEG="${ROTATE_DEG:-0}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-1}"
EVAL_CFG_SCALE="${EVAL_CFG_SCALE:-1.0}"
COND_AUTOCAST="${COND_AUTOCAST:-bf16}"

PYPATH="${CYL_DINOV3_SRC}:${PYTHONPATH:-}"
# The eval_FLAC.py argument vector. The two --cond-method / --frame-avg-angles flags are
# MANDATORY and must never be dropped (would evaluate the vanilla trap).
ARGS=(--model-config "$CONFIG" --dataset-config "$EVAL_DATASET_CONFIG" --ckpt-path "$CKPT"
      --cond-method fa_invariant --frame-avg-angles 0 --rotate-deg "$ROTATE_DEG"
      --cond-autocast "$COND_AUTOCAST" --seed "$EVAL_SEED" --steps "$EVAL_STEPS"
      --cfg-scale "$EVAL_CFG_SCALE" --eval-name "$EVAL_NAME")
CMD_DISPLAY="PYTHONPATH=${PYPATH} CUDA_VISIBLE_DEVICES=${EVAL_GPU} ${EVAL_PYTHON} eval_FLAC.py ${ARGS[*]}"

# --- DRY_RUN: emit the assembled command and exit WITHOUT touching the GPU or a log ---
if [ -n "${DRY_RUN:-}" ]; then
  echo "DRY_RUN (not executed):"
  echo "$CMD_DISPLAY"
  exit 0
fi

# --- EXACT-MATCH frozen-threshold binding (mirrors exp09_launch.sh) ---
FROZEN_FILE="${C1_FROZEN_MIN_FREE_FILE:-${EXPDIR}/c1_frozen_min_free.txt}"
[ -f "$FROZEN_FILE" ] || { echo "REFUSING TO LAUNCH: frozen-records file '${FROZEN_FILE}' is absent - the C1 fit probe (plan §3/§6) has not FROZEN the exp-09 free-VRAM threshold yet."; exit 3; }
FROZEN_RAW="$(cat "$FROZEN_FILE")"
FROZEN="$(printf '%s' "$FROZEN_RAW" | tr -d '[:space:]')"
case "$FROZEN" in
  ''|*[!0-9]*) echo "REFUSING TO LAUNCH: frozen value '${FROZEN_RAW}' in ${FROZEN_FILE} is not a plain integer."; exit 3;;
esac
MIN_FREE_MB="${MIN_FREE_MB:-$FROZEN}"
[ "$MIN_FREE_MB" = "$FROZEN" ] || { echo "REFUSING TO LAUNCH: MIN_FREE_MB=${MIN_FREE_MB} does not match the FROZEN C1 threshold ${FROZEN} (arbitrary override rejected)."; exit 3; }

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${LOGDIR_ABS}/exp09_${TS}_D_eval_${EVAL_NAME}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== exp-09 Stage D eval - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${EVAL_NAME} rot=${ROTATE_DEG} gpu=${EVAL_GPU} MIN_FREE_MB=${MIN_FREE_MB} ==="
echo "--- eval-env provenance (plan §7 PYTHONPATH-first) ---"
echo "EVAL_PYTHON=${EVAL_PYTHON}"
echo "CYL_DINOV3_SRC=${CYL_DINOV3_SRC} sha=$(git -C "$CYL_DINOV3_SRC" rev-parse --short HEAD 2>/dev/null || echo '(not a git repo)')"

# --- per-GPU FREE-VRAM gate on the FROZEN value (fail-CLOSED on query errors) ---
FREE="$(nvidia-smi -i "$EVAL_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
rc_q=$?
[ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on eval GPU ${EVAL_GPU} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
[ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "eval GPU ${EVAL_GPU} free ${FREE} MiB < frozen ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
echo "--- co-tenancy disclosure ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

echo "--- launching: ${CMD_DISPLAY} ---"
PYTHONPATH="$PYPATH" CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$EVAL_PYTHON" eval_FLAC.py "${ARGS[@]}"
rc=${PIPESTATUS[0]}
echo "=== exp-09 Stage D eval '${EVAL_NAME}' exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') - log ${LOG} ==="
exit "$rc"
