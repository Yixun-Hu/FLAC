#!/bin/bash
# Fires the registered curve evals for exp12B_curve_ddp once the pipeline logs its
# endpoint eval done (so the GPUs are free and all checkpoints exist).
set -uo pipefail
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
say () { echo "[curve-after-pipeline] $* | $(date -Is)" >> $REC/curve_eval_exp12B_curve_ddp.log; }
say "waiting for exp12B_curve_ddp endpoint eval in chain_full.log"
until grep -q "exp12B_curve_ddp eval done" $REC/chain_full.log 2>/dev/null; do sleep 300; done
say "endpoint eval done -> starting registered curve evals"
bash $REC/curve_eval.sh exp12B_curve_ddp 0 1
