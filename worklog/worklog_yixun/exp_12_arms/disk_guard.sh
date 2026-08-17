#!/bin/bash
# exp_12: disk guard for a long training run.
#
#   setsid nohup bash disk_guard.sh <pid> <run> <min_free_gb> &
#
# Arm B needs ~17 GB of checkpoints to reach 67,500 while the box as a whole has been
# burning ~11 GB/day from other sessions, against ~52 GB free at resume. If the filesystem
# fills, Lightning dies mid-write and can leave a TRUNCATED checkpoint -- the worst outcome,
# because the newest resume point would then be unusable.
#
# This guard stops the run GRACEFULLY (SIGTERM, right after a checkpoint has finished
# writing) while there is still room, so the latest checkpoint is complete and the run stays
# resumable. It never deletes anything -- pruning is the owner's decision, not a script's.
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; MIN_GB="${3:-20}"
cd /home/yixunhu/codespace/exp-12-arms
LOG=worklog/worklog_yixun/exp_12_arms/disk_guard_$RUN.log
say () { echo "[disk-guard $RUN] $* | $(date -Is)" >> "$LOG"; }

free_gb () { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

say "armed on pid $PID, threshold ${MIN_GB} GB, currently $(free_gb) GB free"
WARNED=""

while kill -0 "$PID" 2>/dev/null; do
  F=$(free_gb)
  if [ "$F" -lt "$MIN_GB" ]; then
    say "THRESHOLD BREACHED: ${F} GB free < ${MIN_GB} GB -- stopping $RUN gracefully"
    # Wait for any in-flight checkpoint write to settle, so we never signal mid-write.
    NEW=$(ls -t outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -n "$NEW" ]; then
      P=-1; while :; do S=$(stat -c %s "$NEW" 2>/dev/null || echo 0)
        [ "$S" = "$P" ] && break; P=$S; sleep 10; done
      say "newest checkpoint settled: $NEW ($S bytes)"
    fi
    kill -TERM "$PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 5; done
    kill -0 "$PID" 2>/dev/null && say "still alive after SIGTERM -- needs a look" \
                               || say "stopped gracefully; resume with resume_arm.sh once space is freed"
    exit 0
  fi
  if [ "$F" -lt $((MIN_GB + 15)) ] && [ -z "$WARNED" ]; then
    say "WARNING: only ${F} GB free (threshold ${MIN_GB} GB) -- pruning may be needed soon"
    WARNED=1
  fi
  sleep 300
done
say "run exited on its own; guard standing down (${F:-?} GB free)"
