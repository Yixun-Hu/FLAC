#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_mkl_runtime=/opt/anaconda3/lib/libmkl_rt.so

cd "${task_repo}"
exec env MKL_RT="${task_mkl_runtime}" "${task_python}" probe_fem_small_room.py \
    --room Bathrooms_idx_14 \
    --production-mesh "${task_exp10}/fem_meshes_h022_optimized/rooms/Bathrooms_idx_14.npz" \
    --reference-mesh "${task_exp10}/fem_meshes_reference_h014/rooms/Bathrooms_idx_14.npz" \
    --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
    --geometry-audit "${task_exp9}/geometry_audit.json" \
    --pilot-manifest "${task_exp9}/pilot_manifest_seed42_4_per_room.json" \
    --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
    --output "${task_exp10}/fem_h022_mkl_preflight_Bathrooms_idx_14.json" \
    --solver-backend mkl_pardiso \
    --solver-threads 24
