#!/bin/bash
# exp_12: stop a running training the moment a given checkpoint is COMPLETE on disk, so the
# resume point loses no work. Fully detached (setsid+nohup) so no session or task timeout
# can kill it mid-vigil -- an earlier in-session waiter was killed by a task timeout and
# would silently have left the training running.
#
#   setsid nohup bash stop_at_ckpt.sh <pid> <run> <target_step> >> <log> 2>&1 &
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; TARGET="${3:?target step}"
cd /home/yixunhu/codespace/exp-12-arms
REC=worklog/worklog_yixun/exp_12_arms
LOG=$REC/stop_at_ckpt_$RUN.log
say () { echo "[stop@$TARGET $RUN] $* | $(date -Is)" >> "$LOG"; }

say "vigil start (pid $PID, target step $TARGET)"

ckpt () { ls outputs_FLAC/$RUN/*/*/checkpoints/*step=$TARGET.ckpt 2>/dev/null | head -1; }

while [ -z "$(ckpt)" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    say "training exited on its own before step $TARGET -- nothing to stop"
    exit 0
  fi
  sleep 60
done

F=$(ckpt)
# Never signal mid-write: hold until the size stops changing AND reaches full size.
PREV=-1
while :; do
  S=$(stat -c %s "$F" 2>/dev/null || echo 0)
  [ "$S" = "$PREV" ] && [ "$S" -ge 723000000 ] && break
  PREV=$S; sleep 10
done
say "checkpoint complete: $F ($S bytes)"

kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 5; done
if kill -0 "$PID" 2>/dev/null; then
  say "still alive after 300 s -> SIGINT"
  kill -INT "$PID" 2>/dev/null
  for _ in $(seq 1 12); do kill -0 "$PID" 2>/dev/null || break; sleep 5; done
fi

if kill -0 "$PID" 2>/dev/null; then
  say "REFUSE: pid $PID still alive after SIGTERM+SIGINT -- NOT escalating to SIGKILL (a hard"
  say "kill during a checkpoint write is exactly what this script exists to avoid). Needs a look."
  exit 4
fi

say "stopped cleanly at step $TARGET"
leftover=$(pgrep -af "$RUN" | grep -v stop_at_ckpt | grep -v grep | wc -l)
say "leftover processes matching $RUN: $leftover"
say "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
