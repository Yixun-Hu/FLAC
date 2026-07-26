#!/usr/bin/env bash
# D1 K=1 arm on GPU0: seeds 42..46, rot0, no predictions. Stops on first failure.
set -u
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
: "${EXPECT_PACKAGE_SHA:?}" "${EXPECT_EXP09_SHA:?}"   # pins mandatory (records freeze)
CKPT=outputs_FLAC/exp09_cylNoSSL/FLAC_exp09_cylNoSSL/exp09_cylNoSSL/checkpoints/epoch=14-step=67500.ckpt
CFG=worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp09_online_eval.json
LOGS=/home/yixunhu/codespace/cylindrical-dinov3/worklog/worklog_yixun/exp_06_flac_no_ssl_claude/d_eval_logs
for SEED in 42 43 44 45 46; do
  EVAL_GPU=0 EVAL_SEED=$SEED ROTATE_DEG=0 \
  EVAL_DATASET_CONFIG=src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json \
  bash worklog/worklog_yixun/exp_09_cyl_no_ssl/d_eval_driver.sh "$CKPT" "$CFG" "exp09_D1_K1_s${SEED}" "$LOGS" \
    || { echo "D1_GPU0 ABORT at seed ${SEED} rc=$?"; exit 1; }
done
echo "D1_GPU0 ALL 5 SEEDS DONE rc=0"
