#!/usr/bin/env bash
# exp_03 R1 gate evals: vanilla control, exp_01 protocol, K in {1,8} x seeds 42-46.
set -uo pipefail
cd /home/yixunhu/codespace/FLAC
for K in 1 8; do
  CFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json
  [ "$K" = 8 ] && CFG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json
  for SEED in 42 43 44 45 46; do
    echo "=== [$(date '+%F %T')] R1 eval K=${K} seed=${SEED} start ==="
    CUDA_VISIBLE_DEVICES=0 python eval_FLAC.py \
      --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
      --dataset-config "$CFG" \
      --ckpt-path outputs_FLAC/exp03_R1_vanilla_ft/FLAC_exp03_R1_vanilla.ckpt \
      --steps 1 --cfg-scale 1.0 --batch-size 32 --num-workers 4 --seed "$SEED" \
      --eval-name "exp03_R1_K${K}_seed${SEED}"
    echo "=== [$(date '+%F %T')] R1 eval K=${K} seed=${SEED} exit=$? ==="
  done
done
echo "=== [$(date '+%F %T')] R1 evals all finished ==="
