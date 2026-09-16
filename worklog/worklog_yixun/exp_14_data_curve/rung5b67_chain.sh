#!/bin/bash
REC=/home/yixunhu/codespace/exp-14-data-curve/worklog/worklog_yixun/exp_14_data_curve
bash $REC/rung5b_probe.sh; bash $REC/rung6_fit_probe.sh; bash $REC/rung7_nas_ckpt.sh
echo "CHAIN_DONE $(date -Is)" > $REC/rung5b67_chain.done
