#!/usr/bin/env bash
set -euo pipefail

task_repo=/home/zhixuanzhao/projects/Frame_Average/NeuriPs_Workshop/localization-exp
task_exp9=${task_repo}/worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
task_exp10=${task_repo}/worklog/worklog_yixun/exp_10_room_helps_baselines_claude
task_mesh_root=${task_exp10}/fem_meshes_h022_optimized
task_mesh_manifest=${task_mesh_root}/tetra_mesh_manifest.json
task_output_root=${task_exp10}/fem_128_query_fullband_h022_mkl24
task_python=/home/zhixuanzhao/projects/Frame_Average/FLAC-vanilla/.venv/bin/python
task_mkl_runtime=/opt/anaconda3/lib/libmkl_rt.so
task_dataset_root=/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms
task_poll_seconds=60

cd "${task_repo}"
for task_required_path in \
    "${task_python}" \
    "${task_mkl_runtime}" \
    "${task_dataset_root}"; do
    if [[ ! -e "${task_required_path}" ]]; then
        echo "missing FEM runtime input: ${task_required_path}" >&2
        exit 2
    fi
done

mkdir -p "${task_output_root}"
printf '%s\n' "$$" > "${task_output_root}/launcher.pid"

check_completion_gate() {
    "${task_python}" - "${task_exp9}" "${task_mesh_root}" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

exp9 = Path(sys.argv[1])
mesh_root = Path(sys.argv[2])


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.pop("sha256", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    payload["sha256"] = expected
    return payload


pilots = [
    load_hashed(exp9 / "pilot_manifest_seed42_4_per_room.json"),
    load_hashed(exp9 / "pilot_manifest_seed43_batch2_4_per_room.json"),
]

indices = []
expected_rooms = set()
for pilot in pilots:
    if int(pilot["query_count"]) != 64 or int(pilot["room_count"]) != 16:
        raise RuntimeError("each frozen pilot must contain 64 queries in 16 rooms")
    indices.extend(int(selected["index"]) for selected in pilot["records"])
    expected_rooms.update(selected["room"] for selected in pilot["records"])
if len(indices) != 128 or len(set(indices)) != 128:
    raise RuntimeError("the two frozen pilots must contain 128 disjoint base queries")
if len(expected_rooms) != 16:
    raise RuntimeError("the 128-query scope must cover exactly 16 rooms")

manifest_path = mesh_root / "tetra_mesh_manifest.json"
audit_path = mesh_root / "mesh_generation_audit.json"
failures_path = mesh_root / "mesh_generation_failures.json"
if not all(path.is_file() for path in (manifest_path, audit_path, failures_path)):
    print("WAIT mesh manifest/audit/failure ledger is not complete", flush=True)
    raise SystemExit(75)

manifest = load_hashed(manifest_path)
audit = load_hashed(audit_path)
failures = load_hashed(failures_path)
ready_rooms = set(manifest.get("rooms", {}))
audited_rooms = set(audit.get("rooms", {}))
failed_rooms = set(failures.get("rooms", {}))

if ready_rooms - expected_rooms or audited_rooms - expected_rooms:
    raise RuntimeError("mesh state contains a room outside the frozen 16-room scope")
if ready_rooms != expected_rooms or audited_rooms != expected_rooms or failed_rooms:
    print(
        "WAIT FEM meshes "
        f"ready={len(ready_rooms)}/16 audited={len(audited_rooms)}/16 "
        f"failures={sorted(failed_rooms)}",
        flush=True,
    )
    raise SystemExit(75)

# Load and join the large frozen manifests only after the inexpensive completion
# poll has reached 16/16, then hash every production NPZ exactly once.
from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import resolve_pilot_records

context = load_context_manifest(exp9 / "context_manifest_exp01_seed42.json")
geometry = load_hashed(exp9 / "geometry_audit.json")
for pilot in pilots:
    resolve_pilot_records(pilot, context, geometry)

if float(manifest.get("maximum_edge_m", -1.0)) != 0.22:
    raise RuntimeError("mesh manifest is not the frozen h_max=0.22 m artifact")
if manifest.get("geometry_audit_sha256") != geometry["sha256"]:
    raise RuntimeError("mesh manifest/geometry audit identity mismatch")
if audit.get("geometry_audit_sha256") != geometry["sha256"]:
    raise RuntimeError("mesh generation audit/geometry identity mismatch")
if audit.get("context_manifest_sha256") != context["sha256"]:
    raise RuntimeError("mesh generation audit/context identity mismatch")

for ordinal, room in enumerate(sorted(expected_rooms), start=1):
    manifest_entry = manifest["rooms"][room]
    audit_entry = audit["rooms"][room]
    mesh_path = Path(manifest_entry["path"])
    if not mesh_path.is_absolute():
        mesh_path = mesh_root / mesh_path
    expected_npz_sha = manifest_entry["npz_sha256"]
    if audit_entry.get("tetra_npz_sha256") != expected_npz_sha:
        raise RuntimeError(f"manifest/generation audit NPZ mismatch for {room}")
    if audit_entry.get("source_obj_sha256") != geometry["rooms"][room]["mesh_sha256"]:
        raise RuntimeError(f"source OBJ provenance mismatch for {room}")
    if not mesh_path.is_file() or file_sha256(mesh_path) != expected_npz_sha:
        raise RuntimeError(f"tetrahedral NPZ hash mismatch for {room}")
    print(f"HASH {ordinal:02d}/16 {room}", flush=True)

print("PASS 16/16 audited meshes and 128 disjoint frozen queries", flush=True)
PY
}

while true; do
    set +e
    check_completion_gate
    task_gate_status=$?
    set -e
    if [[ "${task_gate_status}" -eq 0 ]]; then
        break
    fi
    if [[ "${task_gate_status}" -ne 75 ]]; then
        exit "${task_gate_status}"
    fi
    sleep "${task_poll_seconds}"
done

run_batch() {
    local task_label=$1
    local task_seed=$2
    local task_pilot_manifest=$3

    echo "START ${task_label} $(date --iso-8601=seconds)"
    /usr/bin/time -v env \
        MPLCONFIGDIR=/tmp/matplotlib-exp10-fem-128 \
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
        --mkl-runtime "${task_mkl_runtime}"
    echo "DONE ${task_label} $(date --iso-8601=seconds)"
}

run_batch \
    batch1_seed42 \
    42 \
    "${task_exp9}/pilot_manifest_seed42_4_per_room.json"
run_batch \
    batch2_seed43 \
    43 \
    "${task_exp9}/pilot_manifest_seed43_batch2_4_per_room.json"

"${task_python}" - "${task_output_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

from src.localization.pilot import canonical_sha256
from src.localization.runner import verify_hashed_payload


output_root = Path(sys.argv[1])
batches = {}
total = 0
for label in ("batch1_seed42", "batch2_seed43"):
    batch_root = output_root / label
    run_manifest = json.loads((batch_root / "run_manifest.json").read_text())
    verify_hashed_payload(run_manifest, f"{label} run manifest")
    expected_indices = run_manifest["identity"]["query_indices"]
    completed = 0
    for index in expected_indices:
        result_path = batch_root / "queries" / f"query_{int(index):05d}.json"
        result = json.loads(result_path.read_text())
        verify_hashed_payload(result, f"{label} query {index}")
        if result.get("run_manifest_sha256") != run_manifest["sha256"]:
            raise RuntimeError(f"run identity mismatch: {result_path}")
        completed += 1
    if completed != 64:
        raise RuntimeError(f"{label} completed {completed}/64 queries")
    batches[label] = {
        "completed_queries": completed,
        "run_manifest_sha256": run_manifest["sha256"],
    }
    total += completed

payload = {
    "schema_version": 1,
    "base_query_count": total,
    "flac_k_gen_readout_equivalent": total * 3,
    "fem_context_counts": [1, 8],
    "fem_result_readouts": total * 2,
    "frequency_band_hz": [80.0, 300.0],
    "frequency_count": 102,
    "batches": batches,
}
payload["sha256"] = canonical_sha256(payload)
temporary = output_root / "summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output_root / "summary.json")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
