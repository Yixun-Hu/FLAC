#!/bin/bash
# exp_12 per-GPU chain: wait for a training to exit -> D1 eval it -> (optionally) start
# arm B's SSL on the freed card. Detached, so no GPU-day is lost if the agent session is
# not live at the moment a run finishes.
#
#   bash chain.sh <pid> <run> <model_config> <gpu> <then_ssl:0|1>
#
# HARD STOP: if the 67.5k checkpoint is absent the training did NOT finish. The eval is
# skipped AND the SSL launch is suppressed -- the card must stay free to restart the arm,
# and a human decision is wanted, so this exits 3 rather than helpfully filling the GPU.
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; CFG="${3:?config}"; GPU="${4:?gpu}"; THEN_SSL="${5:-0}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/chain_gpu$GPU.log

echo "[chain gpu$GPU] waiting on $RUN pid $PID | $(date -Is)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
echo "[chain gpu$GPU] $RUN pid $PID exited | $(date -Is)" >> $LOG

CKPT=$(ls outputs_FLAC/$RUN/*/*/checkpoints/*step=67500.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "[chain gpu$GPU] STOP: $RUN has NO step=67500 checkpoint -- it did not finish." >> $LOG
  echo "[chain gpu$GPU] eval skipped; SSL launch suppressed; GPU$GPU left free. | $(date -Is)" >> $LOG
  exit 3
fi

echo "[chain gpu$GPU] $RUN eval start ($CKPT) | $(date -Is)" >> $LOG
bash $REC/eval_arm.sh "$RUN" "$CFG" "$GPU" "$CKPT" >> $LOG 2>&1
echo "[chain gpu$GPU] $RUN eval done rc=$? | $(date -Is)" >> $LOG

if [ "$THEN_SSL" = "1" ]; then
  echo "[chain gpu$GPU] arm B SSL launch | $(date -Is)" >> $LOG
  bash $REC/ssl/launch_ssl.sh "$GPU" >> $LOG 2>&1
  echo "[chain gpu$GPU] arm B SSL launched rc=$? | $(date -Is)" >> $LOG
fi
