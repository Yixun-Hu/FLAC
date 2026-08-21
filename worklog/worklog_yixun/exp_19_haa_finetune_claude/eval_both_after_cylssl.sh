#!/bin/bash
# exp_19: run the eval grid for CYL + CYLSSL once the CYLSSL launcher exits.
set -uo pipefail
PID="${1:?cylssl launcher pid}"
cd /home/yixunhu/codespace/FLAC
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
EXPDIR=worklog/worklog_yixun/exp_19_haa_finetune_claude
LOG=$EXPDIR/eval_both_after_cylssl.log
say () { echo "[eval-after-cylssl] $* | $(date -Is)" >> "$LOG"; }
say "waiting on CYLSSL launcher pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
N=$(ls outputs_FLAC/exp19_HAA_CYLSSL/*/*/checkpoints/epoch=*-step=1000.ckpt 2>/dev/null | wc -l)
say "CYLSSL launcher exited; step-1000 checkpoints: $N"
if [ "$N" -lt 1 ]; then say "STOP: CYLSSL did not complete -- eval skipped"; exit 3; fi
say "starting eval grid ARMS='CYL CYLSSL'"
ARMS="CYL CYLSSL" EXPECT_SHA=$(git rev-parse HEAD) bash $EXPDIR/haa_ft_eval.sh >> "$LOG" 2>&1
say "eval grid exited rc=$?"
