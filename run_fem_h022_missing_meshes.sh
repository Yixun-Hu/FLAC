#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_output=${task_exp10}/fem_meshes_h022_optimized
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_ftetwild=/tmp/exp10-fTetWild/build/FloatTetwild_bin

run_rooms() {
    "${task_python}" "${task_repo}/generate_fem_meshes.py" \
        --ftetwild-bin "${task_ftetwild}" \
        --geometry-audit "${task_exp9}/geometry_audit.json" \
        --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
        --output-dir "${task_output}" \
        --ideal-edge-m 0.10 \
        --maximum-edge-m 0.22 \
        --minimum-edge-utilization 0.80 \
        --target-edge-utilization 0.90 \
        --maximum-threads 8 \
        --rooms "$@" \
        --continue-on-error
}

cd "${task_repo}"

# Ordinary rooms use two independent room workers. Manifest updates remain
# serialized by generate_fem_meshes.py's advisory lock.
run_rooms MeetingRoom_idx_20 Office_idx_10 Restaurants_idx_24 &
task_worker_a=$!
run_rooms MeetingRoom_idx_32 Office_idx_11 &
task_worker_b=$!

task_worker_status=0
wait "${task_worker_a}" || task_worker_status=$?
wait "${task_worker_b}" || task_worker_status=$?
if [[ "${task_worker_status}" -ne 0 ]]; then
    exit "${task_worker_status}"
fi

# The two largest geometries run sequentially to bound peak memory and disk use.
run_rooms Auditorium_idx_1
run_rooms Cafe_idx_1
