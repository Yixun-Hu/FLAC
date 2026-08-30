#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_mesh_manifest=${task_exp10}/fem_meshes_h022_optimized/tetra_mesh_manifest.json
task_output_root=${task_exp10}/fem_128_query_fullband_h022_mkl24
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_mkl_runtime=/opt/anaconda3/lib/libmkl_rt.so
task_dataset_root=/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms

run_batch() {
    local task_label=$1
    local task_seed=$2
    local task_pilot_manifest=$3

    echo "START ${task_label} non-oversized $(date --iso-8601=seconds)"
    /usr/bin/time -v env \
        MPLCONFIGDIR=/tmp/matplotlib-exp10-fem-112 \
        MKL_RT="${task_mkl_runtime}" \
        "${task_python}" "${task_repo}/localize_baseline.py" \
        --method fem_sabine \
        --context-manifest "${task_exp9}/context_manifest_exp01_seed42.json" \
        --geometry-audit "${task_exp9}/geometry_audit.json" \
        --pilot-manifest "${task_pilot_manifest}" \
        --dataset-root "${task_dataset_root}" \
        --output-dir "${task_output_root}/${task_label}" \
        --device cpu \
        --candidate-batch-size 64 \
        --random-seed "${task_seed}" \
        --tetra-mesh-manifest "${task_mesh_manifest}" \
        --fem-solver-backend mkl_pardiso \
        --fem-solver-threads 24 \
        --mkl-runtime "${task_mkl_runtime}" \
        --skip-rooms Auditorium_idx_1 Cafe_idx_1
    echo "DONE ${task_label} non-oversized $(date --iso-8601=seconds)"
}

cd "${task_repo}"
mkdir -p "${task_output_root}"
printf '%s\n' "$$" > "${task_output_root}/non_oversized_launcher.pid"

run_batch \
    batch1_seed42 \
    42 \
    "${task_exp9}/pilot_manifest_seed42_4_per_room.json"
run_batch \
    batch2_seed43 \
    43 \
    "${task_exp9}/pilot_manifest_seed43_batch2_4_per_room.json"

"${task_python}" - "${task_output_root}" "${task_exp9}" <<'PY'
import json
import os
import sys
from pathlib import Path

from src.localization.pilot import canonical_sha256, load_pilot_manifest
from src.localization.runner import verify_hashed_payload


output_root = Path(sys.argv[1])
exp9 = Path(sys.argv[2])
specifications = (
    (
        "batch1_seed42",
        exp9 / "pilot_manifest_seed42_4_per_room.json",
    ),
    (
        "batch2_seed43",
        exp9 / "pilot_manifest_seed43_batch2_4_per_room.json",
    ),
)
skipped_rooms = {"Auditorium_idx_1", "Cafe_idx_1"}
batches = {}
total = 0
for label, pilot_path in specifications:
    pilot = load_pilot_manifest(pilot_path)
    expected = [
        record for record in pilot["records"] if record["room"] not in skipped_rooms
    ]
    if len(expected) != 56:
        raise RuntimeError(f"{label} non-oversized scope is not 56 queries")
    batch_root = output_root / label
    run_manifest = json.loads((batch_root / "run_manifest.json").read_text())
    verify_hashed_payload(run_manifest, f"{label} run manifest")
    if run_manifest["identity"]["pilot_manifest_sha256"] != pilot["sha256"]:
        raise RuntimeError(f"{label} pilot/run identity mismatch")
    completed = 0
    for record in expected:
        index = int(record["index"])
        result_path = batch_root / "queries" / f"query_{index:05d}.json"
        result = json.loads(result_path.read_text())
        verify_hashed_payload(result, f"{label} query {index}")
        if result.get("run_manifest_sha256") != run_manifest["sha256"]:
            raise RuntimeError(f"run identity mismatch: {result_path}")
        if result.get("room") in skipped_rooms:
            raise RuntimeError(f"oversized room entered non-oversized scope: {result_path}")
        completed += 1
    batches[label] = {
        "completed_queries": completed,
        "run_manifest_sha256": run_manifest["sha256"],
    }
    total += completed

payload = {
    "schema_version": 1,
    "status": "non_oversized_complete",
    "completed_base_queries": total,
    "full_base_query_count": 128,
    "skipped_rooms": sorted(skipped_rooms),
    "skipped_queries": 16,
    "fem_context_counts": [1, 8],
    "frequency_band_hz": [80.0, 300.0],
    "frequency_count": 102,
    "batches": batches,
}
payload["sha256"] = canonical_sha256(payload)
temporary = output_root / "non_oversized_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_root / "non_oversized_summary.json")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
