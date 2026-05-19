#!/usr/bin/env bash
# 3-way context-ablation control (baseline FLAC_AR.json @ step=145000, K=1).
#
#   correct   = baseline_A_{split}_K1   (ALREADY in the matrix; not rerun here)
#   wrongroom = same poses, context AUDIO from a different room
#   zeroctx   = same poses, context AUDIO zeroed
#
# Tests advisor H1 (context = material proxy) vs H2 (context ignored):
#   correct ≈ wrongroom ≈ zeroctx  -> H2 (audio channel unused)
#   correct ≪ wrongroom ≈ zeroctx  -> H1 (room-specific material info)
#
# GPU 1, resumable (skips a cell whose metrics JSON exists).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac

GPU=1
STEP=145000
CKPT="outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints/epoch=15-step=${STEP}.ckpt"
MC="src/configs/model_configs/FLAC/AR/FLAC_AR.json"
DC="src/configs/dataset_configs/AR/eval"
LOG="outputs_FLAC/arbRIR_eval_logs"
mkdir -p "$LOG"
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 2; }

# split | variant
CELLS=(
 "seen|wrongroom"
 "seen|zeroctx"
 "unseen|wrongroom"
 "unseen|zeroctx"
)
ckpt_dir="$(dirname "$CKPT")"; ckpt_base="$(basename "$CKPT" .ckpt)"

for c in "${CELLS[@]}"; do
  IFS='|' read -r SPLIT VAR <<< "$c"
  DCN="acousticroom_${SPLIT}eval_arbRIR_v0eval${VAR}_1.json"
  NAME="baseline_${VAR}_${SPLIT}_K1"
  out="${ckpt_dir}/${ckpt_base}_metrics_1_1.0_${NAME}.json"
  lg="${LOG}/ctxabl_${NAME}.log"
  if [ -f "$out" ]; then echo "[$NAME] SKIP (exists)"; continue; fi
  echo "[$NAME] START $(date '+%F %T')  DC=$DCN"
  CUDA_VISIBLE_DEVICES=$GPU python -u eval_FLAC.py \
    --model-config "$MC" \
    --dataset-config "$DC/$DCN" \
    --ckpt-path "$CKPT" --cfg-scale 1.0 --steps 1 \
    --batch-size 32 --num-workers 4 --seed 42 \
    --eval-name "$NAME" > "$lg" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
    echo "[$NAME] FAIL rc=$rc (see $lg)"; tail -8 "$lg"; exit 1
  fi
  echo "[$NAME] DONE $(date '+%F %T') -> $out"
done
echo "ALL CONTEXT-ABLATION CELLS DONE"
