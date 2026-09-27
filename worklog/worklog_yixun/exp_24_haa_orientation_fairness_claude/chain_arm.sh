#!/bin/bash
# exp_24 per-arm chain: FULL fine-tune -> all eval cells -> results table (idempotent writer). Detach with setsid nohup.
#   GPU=<g> ARM=<P1ORI27|YAWORI27|P1ZUP27|CYLZUP27> bash chain_arm.sh
set -uo pipefail
cd /home/yixunhu/codespace/FLAC; E=worklog/worklog_yixun/exp_24_haa_orientation_fairness_claude
echo "[chain $ARM] start $(date -Is)" | tee -a $E/chain_$ARM.log
MODE=FULL GPU=$GPU ARM=$ARM bash $E/haa_ft_ori_launch.sh; rc=$?; echo "[chain $ARM] FULL rc=$rc $(date -Is)" | tee -a $E/chain_$ARM.log
[ $rc -eq 0 ] || { echo "[chain $ARM] ABORT: finetune failed" | tee -a $E/chain_$ARM.log; exit $rc; }
GPU=$GPU ARM=$ARM bash $E/haa_ft_ori_eval.sh all; rc=$?; echo "[chain $ARM] EVAL rc=$rc $(date -Is)" | tee -a $E/chain_$ARM.log
/home/yixunhu/miniconda3/envs/flac/bin/python $E/exp24_write_results.py > $E/write_results_$ARM.log 2>&1; echo "[chain $ARM] RESULTS rc=$? $(date -Is)" | tee -a $E/chain_$ARM.log
echo "[chain $ARM] DONE $(date -Is)" | tee -a $E/chain_$ARM.log
