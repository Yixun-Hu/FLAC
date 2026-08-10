#!/bin/bash
# exp_12 arm B -- launch the native cylindrical DINO+iBOT+Gram SSL stage.
#   bash launch_ssl.sh <gpu>
set -euo pipefail

GPU="${1:?gpu index}"
cd /home/yixunhu/codespace/exp-12-arms
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
export PYTHONPATH=/home/yixunhu/codespace/cylindrical-dinov3/src:/home/yixunhu/codespace/exp-12-arms/worklog/worklog_yixun/exp_12_arms/ssl
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

REC=worklog/worklog_yixun/exp_12_arms
OUT=outputs_FLAC/exp12B_ssl
mkdir -p "$OUT"

RESUME=""
[ -f "$OUT/last.pt" ] && RESUME="--resume $OUT/last.pt"

{
  echo "launched_at: $(date -Is)"
  echo "gpu: $GPU  arm: B (native SSL)"
  echo "exp12_sha: $(git rev-parse HEAD)"
  echo "package_sha: $(git -C /home/yixunhu/codespace/cylindrical-dinov3 rev-parse HEAD)"
  echo "index_sha256: $(sha256sum $OUT/ssl_index.json 2>/dev/null | cut -d' ' -f1)"
  echo "train_manifest_sha256: $(sha256sum data/AR/train.json | cut -d' ' -f1)"
  echo "resume: ${RESUME:-<none>}"
} > "$REC/at_launch_exp12B_ssl.txt"

CUDA_VISIBLE_DEVICES=$GPU nohup python $REC/ssl/ssl_train.py \
  --out-dir "$OUT" --gpu 0 --steps 30000 --batch-size 32 --n-local 4 \
  --num-workers 8 --seed 42 --ckpt-every 2500 --log-every 25 \
  --azimuth-mode lowband --prefix-mode m0_registers \
  --index-cache "$OUT/ssl_index.json" $RESUME \
  > "$REC/train_exp12B_ssl.log" 2>&1 &

echo "pid: $!" >> "$REC/at_launch_exp12B_ssl.txt"
echo "arm B SSL -> GPU$GPU pid $!"
