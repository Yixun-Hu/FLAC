#!/bin/bash
# exp_12 arm B chain: wait for the conditioning run to reach 67,500 -> evaluate it with
# K=1 and K=8 running CONCURRENTLY on the two cards.
#
#   setsid nohup bash chain_B.sh <pid> &
#
# The split is free: the ten D1 cells are independent and each is seeded, so which GPU runs
# which cell changes no number -- only wall clock (~70 min serial -> ~35 min). It does NOT
# touch the training recipe.
#
# HARD STOP: no step=67500 checkpoint means the run did not finish; eval is skipped and both
# cards are left free rather than burning GPU time on a partial model.
set -uo pipefail

PID="${1:?pid}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/chain_B.log
say () { echo "[chain B] $* | $(date -Is)" >> "$LOG"; }

say "waiting on arm B conditioning pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
say "arm B pid $PID exited"

CKPT=$(ls outputs_FLAC/${RUN_NAME:-exp12B_ssl_cond}/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  say "STOP: no step=67500 checkpoint -- arm B did not finish. Eval skipped, GPUs left free."
  exit 3
fi
say "final checkpoint present: $CKPT"

# Wait for a GPU to be free before claiming it: another session may have taken one.
free_gpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
              | awk -F', ' '$2+0 < 2000 {print $1}' | tr '\n' ' '; }
say "free gpus at eval time: $(free_gpu)"

say "launching K=1 on GPU0 and K=8 on GPU1 concurrently"
K_VALUES=1 bash $REC/eval_arm.sh ${RUN_NAME:-exp12B_ssl_cond} $REC/FLAC_AR_exp12B.json 0 "$CKPT" >> $LOG 2>&1 &
E1=$!
K_VALUES=8 bash $REC/eval_arm.sh ${RUN_NAME:-exp12B_ssl_cond} $REC/FLAC_AR_exp12B.json 1 "$CKPT" >> $LOG 2>&1 &
E8=$!
wait $E1; R1=$?
wait $E8; R8=$?
say "eval finished: K=1 rc=$R1  K=8 rc=$R8"

N=$(find outputs_FLAC/${RUN_NAME:-exp12B_ssl_cond} -name "*step=67500_metrics*D1*.json" | wc -l)
say "arm B D1 cells on disk: $N/10"
say "chain complete"
