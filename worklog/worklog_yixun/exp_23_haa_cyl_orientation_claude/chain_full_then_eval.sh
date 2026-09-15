#!/usr/bin/env bash
# exp_23 unattended chain: FULL finetune -> endpoint grid (20 cells) -> K=8 s42 curve (8 cells).
# Every stage hard-stops on a non-zero rc and leaves the card free.
cd /home/yixunhu/codespace/FLAC
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
echo "chain start $(date -Is) pid $$"
MODE=FULL GPU=0 bash $E/haa_ft_cylori_launch.sh; rc=$?; echo "FULL rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "CHAIN STOPPED after FULL"; exit $rc; }
GPU=0 bash $E/haa_ft_cylori_eval.sh endpoints; rc=$?; echo "ENDPOINTS rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "CHAIN STOPPED after endpoints"; exit $rc; }
GPU=0 bash $E/haa_ft_cylori_eval.sh curve; rc=$?; echo "CURVE rc=$rc $(date -Is)"
echo "CHAIN DONE rc=$rc $(date -Is)"
