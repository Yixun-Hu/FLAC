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
        --no-smooth-open-boundary \
        --rooms "$@" \
        --continue-on-error
}

cd "${task_repo}"

run_rooms MeetingRoom_idx_20 Office_idx_10 &
task_worker_a=$!
run_rooms MeetingRoom_idx_32 Office_idx_11 &
task_worker_b=$!

task_worker_status=0
wait "${task_worker_a}" || task_worker_status=$?
wait "${task_worker_b}" || task_worker_status=$?
if [[ "${task_worker_status}" -ne 0 ]]; then
    exit "${task_worker_status}"
fi

# Do not overlap a second oversized attempt with the active Cafe build. A room
# is terminal only after it appears in either the hashed manifest or failures.
while ! "${task_python}" -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ready = json.loads((root / "tetra_mesh_manifest.json").read_text())["rooms"]
failed = json.loads((root / "mesh_generation_failures.json").read_text())["rooms"]
raise SystemExit(0 if "Cafe_idx_1" in ready or "Cafe_idx_1" in failed else 1)
' "${task_output}"; do
    sleep 30
done

# Auditorium is always retried without boundary smoothing. Cafe is resumed if
# its original run passed, otherwise it receives the same geometry-preserving retry.
run_rooms Auditorium_idx_1
run_rooms Cafe_idx_1
