#!/bin/bash
# exp_24 eval: the exp_19/exp_23 HAA protocol for a vanilla-path arm (--cond-method vanilla, bf16, cfg 1.0, one step,
# per-scene records), test split K=8 (haa_test_ori) and K=1 (haa_test_1_ori), ckpt-1000 x seeds 42-46, plus the K=8
# seed-42 steps curve (100..900). Two cells in parallel.   GPU=<g> ARM=<P1ORI27|YAWORI27> bash haa_ft_ori_eval.sh [all|endpoints|curve]
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
GPU="${GPU:-0}"; WHAT="${1:-all}"; ARM="${ARM:-P1ORI27}"; ENDPOINTS="${ENDPOINTS:-1000}"; TAG="${TAG:-}"
case "$ARM" in P1ORI27|YAWORI27) ;; *) echo "ARM must be P1ORI27 or YAWORI27"; exit 2 ;; esac
E=worklog/worklog_yixun/exp_24_haa_orientation_fairness_claude; E23=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
CFG=$E/FLAC_HAA_finetune_${ARM}.json; K8=$E23/haa_test_ori.json; K1=$E23/haa_test_1_ori.json; PROT=(--cond-method vanilla)
TS="$(date '+%Y-%m-%d_%H-%M-%S')"; exec > >(tee -a "$E/haa_ft_${ARM}${TAG}_eval_${TS}.log") 2>&1
echo "=== exp_24 ${ARM}${TAG} eval ${WHAT} endpoints=${ENDPOINTS} | ${TS} | FLAC $(git rev-parse --short HEAD) ==="
sha256sum eval_FLAC.py "$CFG" "$K8" "$K1" "$E23/HAA_md_ori.py"
CELLS=()
add() { # step K seed
  local CKS=( outputs_FLAC/exp24_HAA_${ARM}${TAG}/*/*/checkpoints/epoch=*-step=$1.ckpt ); [ -e "${CKS[0]}" ] || { echo "!! no ckpt for step $1"; return; }
  local NAME="exp24_HAA_${ARM}${TAG}_S$1_K$2_s$3"; local DC; [ "$2" = 8 ] && DC=$K8 || DC=$K1
  if ls "$(dirname "${CKS[0]}")"/*"${NAME}"*.json >/dev/null 2>&1; then echo "skip: $NAME"; return; fi
  CELLS+=("${CKS[0]}|$DC|$3|$NAME")
}
if [ "$WHAT" = all ] || [ "$WHAT" = endpoints ]; then for S in $ENDPOINTS; do for K in 8 1; do for s in 42 43 44 45 46; do add $S $K $s; done; done; done; fi
if [ "$WHAT" = all ] || [ "$WHAT" = curve ]; then for S in 100 200 300 400 500 600 700 800 900; do add $S 8 42; done; fi
echo "cells to run: ${#CELLS[@]}"
run_cell() { IFS='|' read -r CK DC SEED NAME <<< "$1"
  echo "[$(date '+%T')] start $NAME"
  HF_HUB_OFFLINE=1 PYTHONPATH="" CUDA_VISIBLE_DEVICES="$GPU" $PY eval_FLAC.py --model-config "$CFG" --dataset-config "$DC" --ckpt-path "$CK" \
    "${PROT[@]}" --cond-autocast bf16 --record-per-scene --cfg-scale 1.0 --steps 1 --seed "$SEED" --eval-name "$NAME" > "$E/eval_${NAME}.log" 2>&1
  local rc=$?; echo "[$(date '+%T')] end   $NAME rc=$rc"; [ $rc -eq 0 ] || echo "!! FAILED $NAME (see $E/eval_${NAME}.log)"; return $rc; }
FAILED=0; i=0
while [ $i -lt ${#CELLS[@]} ]; do
  run_cell "${CELLS[$i]}" & P1=$!; i=$((i+1))
  if [ $i -lt ${#CELLS[@]} ]; then run_cell "${CELLS[$i]}" & P2=$!; i=$((i+1)); wait $P2 || FAILED=$((FAILED+1)); fi
  wait $P1 || FAILED=$((FAILED+1))
done
echo "=== eval ${WHAT} done: failed=${FAILED} at $(date '+%F %T') ==="; [ $FAILED -eq 0 ]
