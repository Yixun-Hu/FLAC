#!/usr/bin/env bash
# exp_23 eval: arm CYLORI under its OWN protocol = the CYL protocol (fa_invariant, trivial orbit
# "0", cap 64, bf16, cfg 1.0, steps 1, per-scene recorded), K in {8,1} x seeds 42-46 at the
# registered endpoints {1000, 410}, then the K=8 seed-42 steps curve (100..900). Two cells run
# concurrently on one card, as exp_19 did. Resume-safe (skips existing records).
#   GPU=0 bash haa_ft_cylori_eval.sh [endpoints|curve|all]
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
GPU="${GPU:-0}"; WHAT="${1:-all}"; ARM="${ARM:-CYLORI}"; ENDPOINTS="${ENDPOINTS:-1000 410}"
case "$ARM" in CYLORI|CYLORI27) ;; *) echo "ARM must be CYLORI or CYLORI27"; exit 2 ;; esac
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python; CYL_PKG=/home/yixunhu/codespace/cylindrical-dinov3/src
CFG=$E/FLAC_HAA_finetune_${ARM}.json; K8=$E/haa_test_ori.json; K1=$E/haa_test_1_ori.json
TS="$(date '+%Y-%m-%d_%H-%M-%S')"; exec > >(tee -a "$E/haa_ft_${ARM}_eval_${TS}.log") 2>&1
echo "=== exp_23 ${ARM} eval ${WHAT} endpoints=${ENDPOINTS} | ${TS} | FLAC $(git rev-parse --short HEAD) | pkg $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse --short HEAD) ==="
sha256sum eval_FLAC.py "$CFG" "$K8" "$K1" "$E/HAA_md_ori.py"
CELLS=()
add() { # step K seed
  local CKS=( outputs_FLAC/exp23_HAA_${ARM}/*/*/checkpoints/epoch=*-step=$1.ckpt ); [ -e "${CKS[0]}" ] || { echo "!! no ckpt for step $1"; return; }
  local NAME="exp23_HAA_${ARM}_S$1_K$2_s$3"; local DC; [ "$2" = 8 ] && DC=$K8 || DC=$K1
  if ls "$(dirname "${CKS[0]}")"/*"${NAME}"*.json >/dev/null 2>&1; then echo "skip: $NAME"; return; fi
  CELLS+=("${CKS[0]}|$DC|$3|$NAME")
}
if [ "$WHAT" = all ] || [ "$WHAT" = endpoints ]; then for S in $ENDPOINTS; do for K in 8 1; do for s in 42 43 44 45 46; do add $S $K $s; done; done; done; fi
if [ "$WHAT" = all ] || [ "$WHAT" = curve ]; then for S in 100 200 300 500 600 700 800 900; do add $S 8 42; done; fi
echo "cells to run: ${#CELLS[@]}"
run_cell() { IFS='|' read -r CK DC SEED NAME <<< "$1"
  echo "[$(date '+%T')] start $NAME"
  HF_HUB_OFFLINE=1 PYTHONPATH="$CYL_PKG" CUDA_VISIBLE_DEVICES="$GPU" $PY eval_FLAC.py --model-config "$CFG" --dataset-config "$DC" --ckpt-path "$CK" \
    --cond-method fa_invariant --frame-avg-angles 0 --frame-avg-max-fwd-samples 64 --cond-autocast bf16 --record-per-scene \
    --cfg-scale 1.0 --steps 1 --seed "$SEED" --eval-name "$NAME" > "$E/eval_${NAME}.log" 2>&1
  local rc=$?; echo "[$(date '+%T')] end   $NAME rc=$rc"; [ $rc -eq 0 ] || echo "!! FAILED $NAME (see $E/eval_${NAME}.log)"; return $rc; }
FAILED=0; i=0
while [ $i -lt ${#CELLS[@]} ]; do
  run_cell "${CELLS[$i]}" & P1=$!; i=$((i+1))
  if [ $i -lt ${#CELLS[@]} ]; then run_cell "${CELLS[$i]}" & P2=$!; i=$((i+1)); wait $P2 || FAILED=$((FAILED+1)); fi
  wait $P1 || FAILED=$((FAILED+1))
done
echo "=== eval ${WHAT} done: failed=${FAILED} at $(date '+%F %T') ==="; [ $FAILED -eq 0 ]
