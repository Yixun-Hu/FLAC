#!/usr/bin/env bash
# exp_23 follow-on: once the CYLORI chain has released GPU 0, train CYLORI27 (field scale 27 =
# 10x; ckpt every 50 to respect disk) -> ckpt-1000 grid (K8/K1 x 5 seeds) -> K=8 s42 curve.
cd /home/yixunhu/codespace/FLAC
E=worklog/worklog_yixun/exp_23_haa_cyl_orientation_claude
echo "chain27 armed $(date -Is) pid $$ — waiting for the CYLORI chain"
until grep -qE "CHAIN DONE|CHAIN STOPPED" $E/chain_full_then_eval.log; do sleep 60; done
echo "CYLORI chain finished: $(grep -E 'CHAIN DONE|CHAIN STOPPED' $E/chain_full_then_eval.log | tail -1) — starting CYLORI27 $(date -Is)"
ARM=CYLORI27 CADENCE=50 MODE=FULL GPU=0 bash $E/haa_ft_cylori_launch.sh; rc=$?; echo "FULL27 rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "CHAIN27 STOPPED after FULL"; exit $rc; }
ARM=CYLORI27 ENDPOINTS="1000" GPU=0 bash $E/haa_ft_cylori_eval_v2.sh endpoints; rc=$?; echo "ENDPOINTS27 rc=$rc $(date -Is)"; [ $rc -eq 0 ] || { echo "CHAIN27 STOPPED after endpoints"; exit $rc; }
ARM=CYLORI27 GPU=0 bash $E/haa_ft_cylori_eval_v2.sh curve; rc=$?; echo "CURVE27 rc=$rc $(date -Is)"
echo "CHAIN27 DONE rc=$rc $(date -Is)"
