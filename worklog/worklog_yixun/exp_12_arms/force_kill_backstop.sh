#!/bin/bash
# exp_12 arm B: force-kill backstop, per Yixun's explicit "once it reached 7500 please kill it".
#
# stop_at_ckpt.sh signals SIGTERM -> SIGINT and REFUSES SIGKILL. That refusal is right in
# general (a hard kill during a checkpoint write is what it exists to prevent), but here it
# is provably safe to escalate: this backstop only fires AFTER the target checkpoint is
# complete and size-stable on disk, and the NEXT checkpoint is 2,500 steps (~4 h) away, so
# there is no write in flight during the escalation window.
#
#   setsid nohup bash force_kill_backstop.sh <pid> <run> <target_step> &
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; TARGET="${3:?target step}"
cd /home/yixunhu/codespace/exp-12-arms
LOG=worklog/worklog_yixun/exp_12_arms/stop_at_ckpt_$RUN.log
say () { echo "[backstop $RUN] $* | $(date -Is)" >> "$LOG"; }

ckpt () { ls outputs_FLAC/$RUN/*/*/checkpoints/*step=$TARGET.ckpt 2>/dev/null | head -1; }

say "backstop armed (pid $PID, target $TARGET)"

while [ -z "$(ckpt)" ]; do
  kill -0 "$PID" 2>/dev/null || { say "process gone before $TARGET; nothing to do"; exit 0; }
  sleep 30
done

F=$(ckpt); PREV=-1
while :; do
  S=$(stat -c %s "$F" 2>/dev/null || echo 0)
  [ "$S" = "$PREV" ] && [ "$S" -ge 723000000 ] && break
  PREV=$S; sleep 10
done
say "backstop: checkpoint $TARGET complete ($S bytes); giving the graceful watcher 420 s"

# Let stop_at_ckpt.sh's SIGTERM (300 s) + SIGINT (60 s) run their course first.
for _ in $(seq 1 84); do kill -0 "$PID" 2>/dev/null || break; sleep 5; done

if kill -0 "$PID" 2>/dev/null; then
  say "backstop: still alive after graceful window -> SIGKILL (checkpoint $TARGET is safely on disk)"
  kill -9 "$PID" 2>/dev/null
  sleep 10
  pkill -9 -f "FLAC_AR_exp12B" 2>/dev/null
  sleep 5
  kill -0 "$PID" 2>/dev/null && say "backstop: STILL alive after SIGKILL -- unkillable, needs a look" \
                             || say "backstop: SIGKILLed"
else
  say "backstop: graceful stop already succeeded; no SIGKILL needed"
fi

say "leftover exp12B processes: $(pgrep -f 'FLAC_AR_exp12B' | grep -v $$ | wc -l)"
say "gpu: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
