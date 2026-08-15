#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_ROOT=/home/yixunhu/codespace/cylindrical-dinov3/analysis/yaw_robustness
RUN_ROOT="$ANALYSIS_ROOT/runs/a6c4_40k"
GPU0_UUID=GPU-9c72ac69-836a-c2ee-89b7-4b51b68317a8
ARM_B_PID=3661509
WATCH_LOG="$RUN_ROOT/gpu0_watcher.log"

mkdir -p "$RUN_ROOT"
exec >>"$WATCH_LOG" 2>&1

timestamp() {
    date --iso-8601=seconds
}

echo "$(timestamp) | watcher start: waiting for Arm B pid=$ARM_B_PID to release GPU0"
while true; do
    gpu0_rows=$(nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
        --format=csv,noheader 2>/dev/null | awk -F', ' -v uuid="$GPU0_UUID" '$1 == uuid {print}')
    if [[ -z "$gpu0_rows" ]]; then
        break
    fi
    echo "$(timestamp) | GPU0 still occupied: $gpu0_rows"
    sleep 30
done

# Require a second empty observation so a transient process-table gap cannot
# trigger the evaluator while the training job is still tearing down.
sleep 30
gpu0_rows=$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null | awk -F', ' -v uuid="$GPU0_UUID" '$1 == uuid {print}')
if [[ -n "$gpu0_rows" ]]; then
    echo "$(timestamp) | GPU0 was reclaimed during stability check; returning to wait loop"
    exec "$0"
fi

echo "$(timestamp) | GPU0 stably free; launching disjoint K=1, 180-degree worker"
CUDA_PHYSICAL=0 \
K_VALUES=1 \
ANGLE_VALUES=180 \
ARM_VALUES="FA40 VAN40" \
SEED_VALUES="42 43 44 45 46" \
PLOT_ON_COMPLETE=0 \
WORKER_NAME=gpu0_k1_rot180 \
    bash "$ANALYSIS_ROOT/run_missing_c4_evals.sh"
echo "$(timestamp) | GPU0 worker finished"
