#!/usr/bin/env bash
# 12-cell eval matrix driver (plan/eval_arbRIR_v0_vs_baseline_K1_K8.md).
#
# Pinned to GPU 1, locked checkpoint step=145000 (latest common to both runs).
# Resumable: a cell whose metrics JSON already exists is skipped.
#
# Usage:
#   tools/run_arbRIR_eval_matrix.sh            # all 12 cells, in order
#   tools/run_arbRIR_eval_matrix.sh 1          # only cell 1 (smoke test)
#   tools/run_arbRIR_eval_matrix.sh 2 3 4 ...  # listed cells
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac

GPU=1
STEP=145000
CKPT_ABL="outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints/epoch=15-step=${STEP}.ckpt"
CKPT_BASE="outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints/epoch=15-step=${STEP}.ckpt"
MC_ABL="src/configs/model_configs/FLAC/AR/FLAC_AR_arbRIR_v0.json"
MC_BASE="src/configs/model_configs/FLAC/AR/FLAC_AR.json"
DC="src/configs/dataset_configs/AR/eval"
LOGDIR="outputs_FLAC/arbRIR_eval_logs"
mkdir -p "$LOGDIR"

[ -f "$CKPT_ABL" ]  || { echo "missing $CKPT_ABL"; exit 2; }
[ -f "$CKPT_BASE" ] || { echo "missing $CKPT_BASE"; exit 2; }

# cell: MODEL_CONFIG | DATASET_CONFIG | CKPT | EVAL_NAME
CELLS=(
 "$MC_ABL|acousticroom_seeneval_arbRIR_v0evalA_8.json|$CKPT_ABL|arbRIR_v0_seen_K8"
 "$MC_ABL|acousticroom_seeneval_arbRIR_v0evalA_1.json|$CKPT_ABL|arbRIR_v0_seen_K1"
 "$MC_ABL|acousticroom_unseeneval_arbRIR_v0evalA_8.json|$CKPT_ABL|arbRIR_v0_unseen_K8"
 "$MC_ABL|acousticroom_unseeneval_arbRIR_v0evalA_1.json|$CKPT_ABL|arbRIR_v0_unseen_K1"
 "$MC_BASE|acousticroom_seeneval_arbRIR_v0evalA_8.json|$CKPT_BASE|baseline_A_seen_K8"
 "$MC_BASE|acousticroom_seeneval_arbRIR_v0evalA_1.json|$CKPT_BASE|baseline_A_seen_K1"
 "$MC_BASE|acousticroom_unseeneval_arbRIR_v0evalA_8.json|$CKPT_BASE|baseline_A_unseen_K8"
 "$MC_BASE|acousticroom_unseeneval_arbRIR_v0evalA_1.json|$CKPT_BASE|baseline_A_unseen_K1"
 "$MC_BASE|acousticroom_seeneval_arbRIR_v0evalB_8.json|$CKPT_BASE|baseline_B_seen_K8"
 "$MC_BASE|acousticroom_seeneval_arbRIR_v0evalB_1.json|$CKPT_BASE|baseline_B_seen_K1"
 "$MC_BASE|acousticroom_unseeneval_arbRIR_v0evalB_8.json|$CKPT_BASE|baseline_B_unseen_K8"
 "$MC_BASE|acousticroom_unseeneval_arbRIR_v0evalB_1.json|$CKPT_BASE|baseline_B_unseen_K1"
)

if [ "$#" -gt 0 ]; then SEL=("$@"); else SEL=($(seq 1 ${#CELLS[@]})); fi

for i in "${SEL[@]}"; do
  IFS='|' read -r MC DCN CKPT NAME <<< "${CELLS[$((i-1))]}"
  ckpt_dir="$(dirname "$CKPT")"; ckpt_base="$(basename "$CKPT" .ckpt)"
  out_json="${ckpt_dir}/${ckpt_base}_metrics_1_1.0_${NAME}.json"
  log="${LOGDIR}/cell${i}_${NAME}.log"
  if [ -f "$out_json" ]; then
    echo "[cell $i/$((${#CELLS[@]})) $NAME] SKIP — metrics exist: $out_json"
    continue
  fi
  echo "[cell $i $NAME] START $(date '+%F %T')  MC=$(basename "$MC")  DC=$DCN"
  CUDA_VISIBLE_DEVICES=$GPU python -u eval_FLAC.py \
    --model-config "$MC" \
    --dataset-config "$DC/$DCN" \
    --ckpt-path "$CKPT" --cfg-scale 1.0 --steps 1 \
    --batch-size 32 --num-workers 4 --seed 42 \
    --eval-name "$NAME" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out_json" ]; then
    echo "[cell $i $NAME] FAIL rc=$rc  (see $log)"; tail -5 "$log"
    exit 1
  fi
  echo "[cell $i $NAME] DONE $(date '+%F %T')  -> $out_json"
done
echo "ALL REQUESTED CELLS DONE"
