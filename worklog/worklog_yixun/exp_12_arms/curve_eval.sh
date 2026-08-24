#!/bin/bash
# exp_12: REGISTERED curve-eval protocol (Yixun, 2026-08-23): K in {1,8} x seed 42, one
# seed per cell, over every retained checkpoint of a run. Steps 2,500..65,000 run under
# the CURVE naming; 67,500 is NOT duplicated -- the endpoint D1 cells (s42, same
# fa_invariant [0] bf16 protocol) already cover it and the aggregator reads them.
#
#   bash curve_eval.sh <run_name> [gpu0] [gpu1]     # cells alternate across the two GPUs
set -uo pipefail
RUN="${1:?run name}"; G0="${2:-0}"; G1="${3:-1}"
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
REC=worklog/worklog_yixun/exp_12_arms
case "$RUN" in
  exp12B_curve_ddp|exp12B_ssl_cond_ddp) CFG=$REC/FLAC_AR_exp12B.json ;;
  exp12A_c3c4_ddp) CFG=$REC/FLAC_AR_exp12A.json ;;
  *) echo "REFUSE: unknown run '$RUN'"; exit 2 ;;
esac
LOG=$REC/curve_eval_$RUN.log
say () { echo "[curve-eval $RUN] $* | $(date -Is)" | tee -a "$LOG"; }

mapfile -t CKPTS < <(ls outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null \
  | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' | sort -n | awk '$1 < 67500 {print $2}')
say "protocol K={1,8} x s42; ${#CKPTS[@]} checkpoints (<67500; endpoint covered by D1 cells)"
[ "${#CKPTS[@]}" -gt 0 ] || { say "REFUSE: no checkpoints"; exit 2; }

i=0
run_cell () {  # ckpt K gpu
  local CK="$1" K="$2" GPU="$3"
  local STEP; STEP=$(echo "$CK" | grep -oP 'step=\K[0-9]+')
  local DS; [ "$K" = "8" ] && DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
                           || DS=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
  local NAME="${RUN}_CURVE_S${STEP}_K${K}_s42"
  local OUT="$(dirname "$CK")/$(basename "$CK" .ckpt)_metrics_1_1.0_${NAME}_fa_invariant_a1.json"
  if [ -s "$OUT" ]; then say "SKIP $NAME (exists)"; return 0; fi
  say "RUN  $NAME (gpu$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python eval_FLAC.py \
    --model-config "$CFG" --dataset-config "$DS" --ckpt-path "$CK" \
    --cond-method fa_invariant --frame-avg-angles 0 \
    --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 \
    --eval-name "$NAME" > "$REC/curve_cell_${NAME}.log" 2>&1
  say "DONE $NAME rc=$?"
}
# two workers, one per GPU, splitting the (ckpt,K) cell list evenly
CELLS=()
for CK in "${CKPTS[@]}"; do for K in 1 8; do CELLS+=("$CK|$K"); done; done
say "total cells: ${#CELLS[@]}"
half=$(( (${#CELLS[@]} + 1) / 2 ))
( for c in "${CELLS[@]:0:$half}";  do run_cell "${c%|*}" "${c#*|}" "$G0"; done ) &
W0=$!
( for c in "${CELLS[@]:$half}";    do run_cell "${c%|*}" "${c#*|}" "$G1"; done ) &
W1=$!
wait $W0; wait $W1
N=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*_metrics_*CURVE_S*_s42_fa_invariant_a1.json 2>/dev/null | wc -l)
say "CURVE EVAL COMPLETE: $N curve cells on disk"
