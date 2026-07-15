#!/usr/bin/env bash
# Run one exp06 full-split evaluation condition.
#
# Usage:
#   bash run_eval_one.sh <linear|cnn> <K:1|8> <eval-seed> <yaw>
#
# The default checkpoint is the matching train-seed-42 step-30000 artifact.
# Resolution order is: CKPT override, BARE_CKPT override, then the exact
# Lightning milestone.  Bare exports are never guessed because their filenames
# do not reliably prove which training step they contain.  No checkpoint is
# partially loaded: eval_FLAC.py's normal integrity checks remain in force.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAC_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$FLAC_ROOT"

usage() {
  cat <<'EOF'
Usage: bash run_eval_one.sh <linear|cnn> <K:1|8> <eval-seed> <yaw>

Standard C16 yaw values:
  0 22.5 45 67.5 90 112.5 135 157.5
  180 202.5 225 247.5 270 292.5 315 337.5

Useful environment overrides:
  GPU=0                    CUDA device exposed to this single-GPU eval
  TRAIN_SEED=42            seed of the trained checkpoint
  TRAIN_STEP=30000         training milestone encoded in every output name
  CKPT=/path/model.ckpt    authoritative checkpoint override
  BARE_CKPT=/path/x.ckpt   authoritative bare-export override; never auto-guessed
  SAVE_DIR=/path/to/run    checkpoint directory used by auto-resolution
  MODEL_CONFIG=/path.json  model-config override
  PYTHON=../venv/bin/python
  EVAL_BATCH_SIZE=64 EVAL_NUM_WORKERS=4
  FORCE=1                  rerun even when the exact metrics JSON exists
  DRY_RUN=1                print the resolved command without executing it
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VARIANT="${1:-${VARIANT:-}}"
K="${2:-${K:-}}"
EVAL_SEED="${3:-${EVAL_SEED:-}}"
YAW="${4:-${YAW:-}}"

if [[ -z "$VARIANT" || -z "$K" || -z "$EVAL_SEED" || -z "$YAW" ]]; then
  usage >&2
  exit 2
fi

GPU="${GPU:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TRAIN_STEP="${TRAIN_STEP:-30000}"
PYTHON="${PYTHON:-../venv/bin/python}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! "$TRAIN_SEED" =~ ^[0-9]+$ || ! "$TRAIN_STEP" =~ ^[0-9]+$ || ! "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_SEED, TRAIN_STEP, and EVAL_SEED must be non-negative integers." >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ || ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "GPU/NUM_WORKERS must be non-negative integers and EVAL_BATCH_SIZE must be positive." >&2
  exit 2
fi

case "$VARIANT" in
  linear)
    DEFAULT_MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json"
    DEFAULT_SAVE_DIR="outputs_FLAC/exp06_cylvit_pe_linear_trainS${TRAIN_SEED}"
    ;;
  cnn)
    DEFAULT_MODEL_CONFIG="src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json"
    DEFAULT_SAVE_DIR="outputs_FLAC/exp06_cylvit_pe_cnn_trainS${TRAIN_SEED}"
    ;;
  *)
    echo "VARIANT must be 'linear' or 'cnn', got '$VARIANT'." >&2
    exit 2
    ;;
esac

MODEL_CONFIG="${MODEL_CONFIG:-$DEFAULT_MODEL_CONFIG}"
SAVE_DIR="${SAVE_DIR:-$DEFAULT_SAVE_DIR}"

case "$K" in
  1) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json" ;;
  8) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json" ;;
  *) echo "K must be 1 or 8, got '$K'." >&2; exit 2 ;;
esac

# The yaw tag is lossless for the complete C16 grid.  In particular, 22.5 is
# encoded as yaw22p5 rather than being truncated to yaw22 by eval_FLAC.py's
# legacy rotation suffix.
case "$YAW" in
  0)     YAW_TAG="0";     ROT_SUFFIX="" ;;
  22.5)  YAW_TAG="22p5";  ROT_SUFFIX="_rot22" ;;
  45)    YAW_TAG="45";    ROT_SUFFIX="_rot45" ;;
  67.5)  YAW_TAG="67p5";  ROT_SUFFIX="_rot67" ;;
  90)    YAW_TAG="90";    ROT_SUFFIX="_rot90" ;;
  112.5) YAW_TAG="112p5"; ROT_SUFFIX="_rot112" ;;
  135)   YAW_TAG="135";   ROT_SUFFIX="_rot135" ;;
  157.5) YAW_TAG="157p5"; ROT_SUFFIX="_rot157" ;;
  180)   YAW_TAG="180";   ROT_SUFFIX="_rot180" ;;
  202.5) YAW_TAG="202p5"; ROT_SUFFIX="_rot202" ;;
  225)   YAW_TAG="225";   ROT_SUFFIX="_rot225" ;;
  247.5) YAW_TAG="247p5"; ROT_SUFFIX="_rot247" ;;
  270)   YAW_TAG="270";   ROT_SUFFIX="_rot270" ;;
  292.5) YAW_TAG="292p5"; ROT_SUFFIX="_rot292" ;;
  315)   YAW_TAG="315";   ROT_SUFFIX="_rot315" ;;
  337.5) YAW_TAG="337p5"; ROT_SUFFIX="_rot337" ;;
  *)
    echo "Unsupported yaw '$YAW'; use one of the 16 C16 angles listed by --help." >&2
    exit 2
    ;;
esac

resolve_checkpoint() {
  local padded_step
  local -a candidates=()
  local -a matches=()

  if [[ -n "${CKPT:-}" ]]; then
    [[ -f "$CKPT" ]] || { echo "Explicit CKPT does not exist: $CKPT" >&2; return 2; }
    printf '%s\n' "$CKPT"
    return 0
  fi
  if [[ -n "${BARE_CKPT:-}" ]]; then
    [[ -f "$BARE_CKPT" ]] || { echo "Explicit BARE_CKPT does not exist: $BARE_CKPT" >&2; return 2; }
    printf '%s\n' "$BARE_CKPT"
    return 0
  fi

  # Resolve only an exact step-bearing Lightning milestone.  Silently picking a
  # generic bare export could label a later 100k export as the requested 30k run.
  printf -v padded_step '%09d' "$TRAIN_STEP"
  candidates=(
    "$SAVE_DIR/step=${padded_step}.ckpt"
    "$SAVE_DIR/step=${TRAIN_STEP}.ckpt"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ -d "$SAVE_DIR" ]]; then
    mapfile -t matches < <(
      find "$SAVE_DIR" -maxdepth 1 -type f -name "epoch=*-step=${TRAIN_STEP}.ckpt" -print | sort
    )
    if (( ${#matches[@]} == 1 )); then
      printf '%s\n' "${matches[0]}"
      return 0
    elif (( ${#matches[@]} > 1 )); then
      echo "Multiple step-${TRAIN_STEP} checkpoints in $SAVE_DIR; set CKPT explicitly:" >&2
      printf '  %s\n' "${matches[@]}" >&2
      return 2
    fi
  fi

  echo "No exact step-${TRAIN_STEP} checkpoint found for variant=$VARIANT in $SAVE_DIR." >&2
  echo "Set CKPT=/exact/path.ckpt or BARE_CKPT=/explicit/export.ckpt to override." >&2
  return 2
}

CKPT_PATH="$(resolve_checkpoint)"

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$MODEL_CONFIG" ]] || { echo "Missing model config: $MODEL_CONFIG" >&2; exit 2; }
  [[ -f "$DATASET_CONFIG" ]] || { echo "Missing dataset config: $DATASET_CONFIG" >&2; exit 2; }
  if [[ "$PYTHON" == */* ]]; then
    [[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
  else
    command -v "$PYTHON" >/dev/null || { echo "Python not found on PATH: $PYTHON" >&2; exit 2; }
  fi
fi

EVAL_NAME="exp06_cylvit_pe_${VARIANT}_trainS${TRAIN_SEED}_step${TRAIN_STEP}_K${K}_evalS${EVAL_SEED}_yaw${YAW_TAG}"
CKPT_STEM="${CKPT_PATH%.ckpt}"
METRICS_PATH="${CKPT_STEM}_metrics_1_1.0_${EVAL_NAME}${ROT_SUFFIX}.json"

if [[ -f "$METRICS_PATH" && "$FORCE" != "1" ]]; then
  echo "[exp06-eval] exists, skip: $METRICS_PATH"
  exit 0
fi

echo "[exp06-eval] variant=$VARIANT train_seed=$TRAIN_SEED train_step=$TRAIN_STEP K=$K eval_seed=$EVAL_SEED yaw=$YAW gpu=$GPU"
echo "[exp06-eval] model_config=$MODEL_CONFIG"
echo "[exp06-eval] checkpoint=$CKPT_PATH"
echo "[exp06-eval] metrics=$METRICS_PATH"

CMD=(
  "$PYTHON" eval_FLAC.py
  --model-config "$MODEL_CONFIG"
  --dataset-config "$DATASET_CONFIG"
  --ckpt-path "$CKPT_PATH"
  --cfg-scale 1.0
  --steps 1
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$EVAL_SEED"
  --rotate-deg "$YAW"
  --cond-method vanilla
  --cond-autocast default
  --eval-name "$EVAL_NAME"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

if [[ ! -f "$METRICS_PATH" ]]; then
  echo "Evaluation exited without writing the expected metrics file: $METRICS_PATH" >&2
  exit 1
fi
echo "[exp06-eval] complete: $METRICS_PATH"
