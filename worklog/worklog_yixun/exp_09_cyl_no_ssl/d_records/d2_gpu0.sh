#!/usr/bin/env bash
# D2 K=1 arm on GPU0: seed 42, rot {0,45,90,180,270}, EVAL_STORE_PREDS=1. Stops on first failure.
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
: "${EXPECT_PACKAGE_SHA:?}" "${EXPECT_EXP09_SHA:?}"
CKPT=outputs_FLAC/exp09_cylNoSSL/FLAC_exp09_cylNoSSL/exp09_cylNoSSL/checkpoints/epoch=14-step=67500.ckpt
CFG=worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp09_online_eval.json
LOGS=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude/d_eval_logs
for ROT in 0 45 90 180 270; do
  EVAL_GPU=0 EVAL_SEED=42 ROTATE_DEG=$ROT EVAL_STORE_PREDS=1 \
  EVAL_DATASET_CONFIG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  bash worklog/worklog_yixun/exp_09_cyl_no_ssl/d_eval_driver.sh "$CKPT" "$CFG" "exp09_D2_K1" "$LOGS" \
    || { echo "D2_GPU0 ABORT at rot ${ROT} rc=$?"; exit 1; }
done
echo "D2_GPU0 ALL 5 ROTS DONE rc=0"
