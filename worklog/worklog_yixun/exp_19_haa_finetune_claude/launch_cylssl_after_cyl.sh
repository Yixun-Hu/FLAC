#!/bin/bash
# exp_19: launch the CYLSSL arm as soon as the CYL launcher (which holds the exp_19
# gate flock for its whole run, by design) releases it. Waiting on the PID rather than
# passing GATE_LOCK_HELD_BY_CALLER=1 keeps the launcher's own exclusion semantics
# honest -- that flag is for callers that genuinely hold the lock.
set -uo pipefail
CYL_PID="${1:?CYL launcher pid}"
cd /home/yixunhu/codespace/FLAC
export PATH=/home/yixunhu/miniconda3/envs/flac/bin:$PATH
EXPDIR=worklog/worklog_yixun/exp_19_haa_finetune_claude
LOG=$EXPDIR/launch_cylssl_after_cyl.log
say () { echo "[cylssl-after-cyl] $* | $(date -Is)" >> "$LOG"; }

say "waiting on CYL launcher pid $CYL_PID"
while kill -0 "$CYL_PID" 2>/dev/null; do sleep 60; done
say "CYL launcher exited; verifying its training completed"
N=$(ls outputs_FLAC/exp19_HAA_CYL/*/*/checkpoints/epoch=*-step=1000.ckpt 2>/dev/null | wc -l)
say "CYL step-1000 checkpoints found: $N"

sleep 10   # let the flock fd fully close
say "launching CYLSSL on GPU1"
ARM=CYLSSL GPU=1 MODE=FULL EXPECT_SHA=$(git -C /home/yixunhu/codespace/FLAC rev-parse HEAD) \
  bash $EXPDIR/haa_ft_launch.sh >> "$LOG" 2>&1
say "CYLSSL launcher exited rc=$?"
