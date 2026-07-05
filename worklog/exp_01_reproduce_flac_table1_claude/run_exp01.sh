#!/usr/bin/env bash
# exp_01_reproduce_flac_table1 — 5 seeds x K in {1, 8}, full unseen split.
set -uo pipefail
cd /home/yixunhu/codespace/FLAC

for K in 1 8; do
  if [ "$K" = "1" ]; then
    CFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
  else
    CFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
  fi
  for SEED in 42 43 44 45 46; do
    echo "=== [$(date '+%F %T')] K=${K} seed=${SEED} start ==="
    CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
      --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
      --dataset-config "$CFG" \
      --ckpt-path weights/FLAC/FLAC_EMA.ckpt \
      --cfg-scale 1.0 --steps 1 --seed "$SEED" \
      --eval-name "exp01_unseen_K${K}_seed${SEED}"
    echo "=== [$(date '+%F %T')] K=${K} seed=${SEED} exit=$? ==="
  done
done
echo "=== [$(date '+%F %T')] exp01 all runs finished ==="
