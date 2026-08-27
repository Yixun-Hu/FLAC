#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_preflight=${task_exp10}/fem_h022_mkl_preflight_Bathrooms_idx_14.json
task_mesh_manifest=${task_exp10}/fem_meshes_h022_optimized/tetra_mesh_manifest.json
task_output=${task_exp10}/fem_nine_room_threefreq_seed42_h022_mkl_parallel2
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_mkl_runtime=/opt/anaconda3/lib/libmkl_rt.so

while [[ ! -f "${task_preflight}" ]]; do
    sleep 30
done

"${task_python}" -c 'import json, sys; from pathlib import Path; payload=json.loads(Path(sys.argv[1]).read_text()); raise SystemExit(0 if payload.get("passed") is True else 2)' "${task_preflight}"

cd "${task_repo}"
exec /usr/bin/time -v env MPLCONFIGDIR=/tmp/matplotlib-exp10-fem-nine-threefreq \
    MKL_RT="${task_mkl_runtime}" \
    "${task_python}" probe_fem_rooms.py \
    --tetra-mesh-manifest "${task_mesh_manifest}" \
    --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
    --geometry-audit "${task_exp9}/geometry_audit.json" \
    --pilot-manifest "${task_exp10}/fem_nine_room_pilot_seed42_1_per_room.json" \
    --dataset-root /home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms \
    --output-dir "${task_output}" \
    --solver-backend mkl_pardiso \
    --solver-threads 12 \
    --room-workers 2 \
    --minimum-memory-gib-per-worker 56 \
    --continue-on-error
