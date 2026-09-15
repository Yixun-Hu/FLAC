#!/usr/bin/env bash
# exp_23 robustness: a SECOND finetune seed for both arms of the headline comparison —
# CYLORI27 (facing field, s=27) and P1 (vanilla) — same recipe, seed 43, ckpt every 100 (disk),
# ckpt-1000 grid K8/K1 x 5 eval seeds. NOT auto-launched (both GPUs are occupied by the xRIR
# session as of 2026-09-15 06:05 EDT). Launch when a card has >= 8 GB free:
#   GPU=<0|1> nohup setsid bash chain_seed43.sh > chain_seed43.log 2>&1 &
cd /home/yixunhu/codespace/FLAC
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude; GPU="${GPU:-0}"
echo "chain43 start $(date -Is) pid $$ gpu $GPU"
for ARM in CYLORI27 P1; do
  ARM=$ARM SEED=43 CADENCE=100 MODE=FULL GPU=$GPU VRAM_FLOOR=8000 DISK_FLOOR=60000 bash $E/haa_ft_cylori_launch.sh; rc=$?; echo "FULL ${ARM}_s43 rc=$rc $(date -Is)"
  [ $rc -eq 0 ] || { echo "CHAIN43 STOPPED after FULL $ARM"; exit $rc; }
  ARM=$ARM TAG=_s43 ENDPOINTS="1000" GPU=$GPU bash $E/haa_ft_cylori_eval_v2.sh endpoints; rc=$?; echo "ENDPOINTS ${ARM}_s43 rc=$rc $(date -Is)"
  [ $rc -eq 0 ] || { echo "CHAIN43 STOPPED after endpoints $ARM"; exit $rc; }
done
echo "CHAIN43 DONE rc=0 $(date -Is)"
