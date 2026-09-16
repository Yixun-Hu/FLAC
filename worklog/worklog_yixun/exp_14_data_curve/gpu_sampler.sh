#!/bin/bash
# Ladder rung 6 helper: sample per-process GPU memory every 5 s while a probe runs.
# Usage: gpu_sampler.sh <outfile> <pid-to-watch>   (exits when the pid is gone)
OUT="$1"; PID="$2"
while kill -0 "$PID" 2>/dev/null; do
  date -Is >> "$OUT"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> "$OUT"
  sleep 5
done
