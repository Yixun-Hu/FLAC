#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp"
OUTPUT_DIR="$REPO_ROOT/worklog/worklog_yixun/exp_18_depth_aabb_oversized_benchmark"
PYTHON_BIN="/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python"
QUEUE_PARENT_PID=243243

while [[ ! -f "$OUTPUT_DIR/query_03685_depth_aabb_result.json" || \
         ! -f "$OUTPUT_DIR/query_03695_depth_aabb_result.json" ]]; do
  sleep 30
done

"$PYTHON_BIN" "$REPO_ROOT/tools/aggregate_fem_omp_112.py" \
  --selection "$REPO_ROOT/worklog/worklog_yixun/exp_14_depth_aabb_matched_protocol/depth_aabb_matched_16room_112.json" \
  --primary-dir "$REPO_ROOT/worklog/worklog_yixun/exp_16_depth_aabb_matched_97/results" \
  --oversized-dir "$OUTPUT_DIR" \
  --external-summary "$OUTPUT_DIR/external_server_7query_omp_summary.json" \
  --output-json "$OUTPUT_DIR/fem_omp_112_merged_summary.json" \
  --output-md "$OUTPUT_DIR/fem_omp_112_merged_summary.md"

# Result JSONs are atomically renamed after their arrays and payloads are
# complete. Terminate the stopped parent only after the aggregate validates;
# SIGKILL is required because a SIGSTOPped process cannot handle SIGTERM.
kill -KILL "$QUEUE_PARENT_PID" 2>/dev/null || true
