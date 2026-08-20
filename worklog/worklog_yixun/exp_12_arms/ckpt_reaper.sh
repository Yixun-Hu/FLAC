#!/bin/bash
# exp_12: cap a live run's checkpoint footprint at the newest N (default 2).
#
#   setsid nohup bash ckpt_reaper.sh <pid> <run> [keep_n] &
#
# Lightning here runs save_top_k=-1, so a full 67,500-step run accumulates 27 checkpoints
# (~19.5 GB). Only the 67,500 endpoint is actually needed (it is what gets evaluated), plus
# one or two recent ones for resume safety. Checkpoint-curve sweeps over intermediates were
# already given up for arms A and C, so keeping them for B is paying 18 GB for nothing.
#
# Safety: never touches the newest N, never touches step=67500, and only deletes a file whose
# size has stopped changing -- so a checkpoint mid-write is never removed.
set -uo pipefail

PID="${1:?pid}"; RUN="${2:?run}"; KEEP="${3:-2}"
cd /home/yixunhu/codespace/exp-12-arms
LOG=worklog/worklog_yixun/exp_12_arms/ckpt_reaper_$RUN.log
say () { echo "[reaper $RUN] $* | $(date -Is)" >> "$LOG"; }

say "armed on pid $PID, keeping newest $KEEP (+ always step=67500)"

while kill -0 "$PID" 2>/dev/null; do
  mapfile -t all < <(ls outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null \
                     | sed -E 's/.*step=([0-9]+)\.ckpt/\1 &/' | sort -rn | cut -d' ' -f2-)
  if [ "${#all[@]}" -gt "$KEEP" ]; then
    for f in "${all[@]:$KEEP}"; do
      case "$f" in *step=67500.ckpt) continue ;; esac
      s1=$(stat -c %s "$f" 2>/dev/null || echo 0); sleep 3
      s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
      if [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then
        rm -f "$f" && say "reaped $(basename "$f")  (free now $(df --output=avail -BG / | tail -1 | tr -dc '0-9') GB)"
      fi
    done
  fi
  sleep 120
done
say "run exited; reaper standing down. remaining: $(ls outputs_FLAC/$RUN/*/*/checkpoints/*.ckpt 2>/dev/null | wc -l) checkpoints"
