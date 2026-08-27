#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_output=${task_exp10}/fem_meshes_h022_optimized
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_ftetwild=/tmp/exp10-fTetWild/build/FloatTetwild_bin

cd "${task_repo}"
exec "${task_python}" generate_fem_meshes.py \
    --ftetwild-bin "${task_ftetwild}" \
    --geometry-audit "${task_exp9}/geometry_audit.json" \
    --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
    --output-dir "${task_output}" \
    --ideal-edge-m 0.10 \
    --maximum-edge-m 0.22 \
    --minimum-edge-utilization 0.80 \
    --target-edge-utilization 0.90 \
    --maximum-threads 8 \
    --rooms \
        Bathrooms_idx_14 Bathrooms_idx_18 Bedrooms_idx_33 Bedrooms_idx_18 \
        LivingRoomsWithHallway_idx_25 Apartments_idx_50 Apartments_idx_42 \
        Restaurants_idx_22 LivingRoomsWithHallway_idx_30 \
    --continue-on-error
