#!/usr/bin/env python3
"""Run one full-band FEM-Sabine localization query on a depth-only AABB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.depth_aabb import (
    depth_panorama_aabb,
    depth_reprojection_audit,
    points_in_aabb,
    structured_aabb_tetrahedral_mesh,
)
from src.baselines.fem_pipeline import (
    prepare_fem_query_interpolation,
    prepare_fem_room_operators,
    run_fem_sabine_forward,
    save_tetrahedral_mesh_npz,
)
from src.baselines.fem_solver import SparseDirectSolveSession, SparseDirectSolverOptions
from src.localization.baseline_runner import score_fem_room_helps_candidates
from src.localization.engine import (
    filter_frozen_query_candidates,
    load_frozen_query,
    reconstruct_room_base_candidates,
)
from src.localization.pilot import canonical_sha256
from src.localization.scoring import localization_metrics, stable_argmax


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_record(path: Path, query_index: int) -> dict:
    payload = json.loads(path.read_text())
    records = [record for record in payload["records"] if int(record["index"]) == query_index]
    if len(records) != 1:
        raise ValueError(f"query index {query_index} is not unique in {path}")
    return records[0]


def compact_solver_profile(profile: list[dict]) -> dict:
    keys = (
        "symbolic_analysis_seconds",
        "factorization_seconds",
        "solve_seconds",
        "system_construction_seconds",
        "residual_and_sampling_seconds",
    )
    return {
        "frequency_count": len(profile),
        "backend": profile[0]["backend"],
        "threads": profile[0]["threads"],
        **{
            f"total_{key}": float(sum(float(item[key]) for item in profile))
            for key in keys
        },
        "maximum_factor_nonzeros": int(max(item["factor_nonzeros"] for item in profile)),
        "maximum_system_nonzeros": int(max(item["system_nonzeros"] for item in profile)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding-m", type=float, default=0.05)
    parser.add_argument("--maximum-edge-m", type=float, default=0.22)
    parser.add_argument("--solver-backend", choices=("superlu", "mkl_pardiso"), default="mkl_pardiso")
    parser.add_argument("--solver-threads", type=int, default=24)
    parser.add_argument("--mkl-runtime", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--latency-protocol",
        choices=("legacy", "kctx8_kgen1"),
        default="legacy",
    )
    args = parser.parse_args()

    if args.mkl_runtime is not None:
        runtime = args.mkl_runtime.resolve()
        if not runtime.is_file():
            raise FileNotFoundError(runtime)
        os.environ["MKL_RT"] = str(runtime)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    record = load_record(args.context_manifest, args.query_index)
    geometry_audit = json.loads(args.geometry_audit.read_text())
    room_base = reconstruct_room_base_candidates(record["room"], geometry_audit)
    localization_started = time.perf_counter()
    candidate_started = time.perf_counter()
    candidates = filter_frozen_query_candidates(record, geometry_audit, room_base)
    receiver = np.asarray(record["receiver_global"], dtype=np.float64)
    truth = np.asarray(record["source_global"], dtype=np.float64)
    context_sources = np.asarray(record["context_sources_global"], dtype=np.float64)
    candidate_seconds = time.perf_counter() - candidate_started

    geometry_started = time.perf_counter()
    receiver_id = int(record["filename"].split("_")[1][1:])
    depth_path = (
        args.dataset_root
        / "depth_map"
        / record["scene"]
        / record["room"]
        / f"{receiver_id}.npy"
    )
    depth = np.load(depth_path)
    lower_local, upper_local, envelope_audit = depth_panorama_aabb(
        depth, padding_m=args.padding_m
    )
    reprojection_audit = depth_reprojection_audit(depth, lower_local, upper_local)
    candidate_inside = points_in_aabb(candidates - receiver, lower_local, upper_local)
    source_inside = points_in_aabb(
        (truth - receiver)[None, :], lower_local, upper_local
    )
    context_inside = points_in_aabb(
        context_sources - receiver, lower_local, upper_local
    )
    if not candidate_inside.all():
        raise RuntimeError("depth AABB does not contain the frozen candidate set")
    if not source_inside[0] or not context_inside.all():
        raise RuntimeError("depth AABB does not contain all observed source anchors")

    mesh_started = time.perf_counter()
    mesh, lattice_audit = structured_aabb_tetrahedral_mesh(
        lower_local + receiver,
        upper_local + receiver,
        maximum_edge_m=args.maximum_edge_m,
    )
    mesh_seconds = time.perf_counter() - mesh_started
    geometry_seconds = time.perf_counter() - geometry_started
    mesh_path = output_dir / f"query_{args.query_index:05d}_depth_aabb_mesh.npz"
    input_seconds = 0.0
    operator_seconds = 0.0
    solve_seconds = 0.0
    omp_seconds = 0.0
    if not args.geometry_only:
        input_started = time.perf_counter()
        observed, metadata = load_frozen_query(record, args.dataset_root)
        input_seconds = time.perf_counter() - input_started
        operators_started = time.perf_counter()
        room_operators = prepare_fem_room_operators(mesh)
        receiver_load, candidate_interpolation = prepare_fem_query_interpolation(
            room_operators,
            receiver,
            candidates,
        )
        operator_seconds = time.perf_counter() - operators_started
        options = SparseDirectSolverOptions(
            backend=args.solver_backend,
            threads=args.solver_threads,
        )
        solve_started = time.perf_counter()
        with SparseDirectSolveSession(options) as session:
            forward = run_fem_sabine_forward(
                mesh,
                receiver_point=receiver,
                candidate_points=candidates,
                context_waveforms=metadata["context_audio"][:8],
                construct_waveforms=False,
                room_operators=room_operators,
                receiver_load=receiver_load,
                candidate_interpolation=candidate_interpolation,
                solver_options=options,
                solver_session=session,
                maximum_allowed_edge_m=args.maximum_edge_m,
            )
        solve_seconds = time.perf_counter() - solve_started
        omp_started = time.perf_counter()
        scores, recovery = score_fem_room_helps_candidates(
            forward.response,
            forward.bin_indices,
            observed,
            sample_count=observed.shape[-1],
        )
        omp_seconds = time.perf_counter() - omp_started
    localization_seconds = time.perf_counter() - localization_started

    # Artifact hashing and serialization are deliberately outside the localization timer.
    depth_hash = file_sha256(depth_path)
    save_tetrahedral_mesh_npz(mesh, mesh_path, source_mesh_sha256=depth_hash)
    base_payload = {
        "schema_version": 2,
        "method": "fem_sabine_depth_aabb",
        "interpretation": (
            "depth-panorama-only axis-aligned matched-input FEM baseline; "
            "no official unseen-room mesh"
        ),
        "query_index": int(args.query_index),
        "query_id": record["query_id"],
        "scene": record["scene"],
        "room": record["room"],
        "receiver_id": record["filename"].split("_")[1],
        "receiver_global": receiver.tolist(),
        "source_global": truth.tolist(),
        "candidate_count": int(len(candidates)),
        "context_count": 8,
        "coverage_protocol": {
            "candidate_set_policy": "frozen and identical; no filtering or score substitution",
            "required_candidate_coverage_fraction": 1.0,
            "receiver_required_inside": True,
            "target_source_required_inside": True,
            "all_context_sources_required_inside": True,
            "strict_gate_passed": True,
        },
        "depth_path": str(depth_path),
        "depth_sha256": depth_hash,
        "mesh_file": mesh_path.name,
        "mesh_sha256": file_sha256(mesh_path),
        "envelope_audit": envelope_audit,
        "depth_reprojection_audit": reprojection_audit,
        "point_audit": {
            "candidate_inside_count": int(candidate_inside.sum()),
            "candidate_count": int(len(candidate_inside)),
            "source_inside": bool(source_inside[0]),
            "context_inside_count": int(context_inside.sum()),
            "context_count": int(len(context_inside)),
        },
        "lattice_audit": lattice_audit,
        "runtime_seconds": {"mesh_construction": mesh_seconds},
    }
    if args.latency_protocol == "kctx8_kgen1" and not args.geometry_only:
        base_payload["latency_protocol"] = {
            "name": "kctx8_kgen1",
            "context_count": 8,
            "generated_rirs_per_candidate": 1,
            "selector": "room_helps_pulse_stacked_omp",
            "model_and_checkpoint_loading_included": False,
            "result_serialization_included": False,
            "room_base_grid_reconstruction_included": False,
        }
        base_payload["latency_seconds"] = {
            "candidate_preparation": candidate_seconds,
            "depth_aabb_and_mesh": geometry_seconds,
            "query_input_loading": input_seconds,
            "operator_construction": operator_seconds,
            "fullband_solve": solve_seconds,
            "omp_scoring": omp_seconds,
            "localization_total": localization_seconds,
        }
    if args.geometry_only:
        base_payload["sha256"] = canonical_sha256(base_payload)
        geometry_path = output_dir / f"query_{args.query_index:05d}_depth_aabb_geometry.json"
        temporary = geometry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(base_payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(geometry_path)
        print(json.dumps(base_payload, indent=2, sort_keys=True))
        return

    prediction_index = stable_argmax(scores)
    metrics = localization_metrics(candidates, truth, prediction_index)
    metrics["prediction_global"] = candidates[prediction_index].tolist()
    metrics["winning_score"] = float(scores[prediction_index])
    metrics["mean_candidate_score"] = float(scores.mean())

    arrays_path = output_dir / f"query_{args.query_index:05d}_depth_aabb_scores.npz"
    np.savez_compressed(
        arrays_path,
        candidates=candidates.astype(np.float32),
        scores=scores.numpy().astype(np.float32),
        frequencies_hz=forward.frequencies_hz,
    )
    fem_audit = dict(forward.audit)
    fem_audit["solver_profile"] = compact_solver_profile(fem_audit["solver_profile"])
    payload = {
        **base_payload,
        "arrays_file": arrays_path.name,
        "arrays_sha256": file_sha256(arrays_path),
        "fem_audit": fem_audit,
        "sparse_recovery": {
            "support": list(recovery.support),
            "relative_residual_norm": recovery.relative_residual_norm,
        },
        "metrics": metrics,
        "runtime_seconds": {
            **base_payload["runtime_seconds"],
            "operator_construction": operator_seconds,
            "fullband_solve": solve_seconds,
            "total": mesh_seconds + operator_seconds + solve_seconds,
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    result_path = output_dir / f"query_{args.query_index:05d}_depth_aabb_result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
