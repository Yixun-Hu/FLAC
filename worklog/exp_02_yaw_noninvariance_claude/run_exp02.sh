#!/usr/bin/env bash
# exp_02_yaw_noninvariance — yaw-rotated conditioning sweep, unseen K=1, seed 42.
set -uo pipefail
cd /home/yixunhu/codespace/FLAC

COMMON="--model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
  --steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed 42 --store_predictions"

run () {
  echo "=== [$(date '+%F %T')] $* start ==="
  CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py $COMMON "$@"
  echo "=== [$(date '+%F %T')] $* exit=$? ==="
}

run --eval-name yaw_baseline
run --eval-name yaw_rot0   --rotate-deg 0
run --eval-name yaw_rot90  --rotate-deg 90
run --eval-name yaw_rot180 --rotate-deg 180
run --eval-name yaw_rot270 --rotate-deg 270
echo "=== [$(date '+%F %T')] exp02 all runs finished ==="
