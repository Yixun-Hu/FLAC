#!/usr/bin/env python3
"""Run one FEM-Sabine query on an LGT-inspired depth-polar room layout."""

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

from src.baselines.depth_aabb import depth_panorama_aabb
from src.baselines.depth_polar_layout import (
    complete_polar_layout_toward_depth_aabb,
    depth_panorama_polar_layout,
    depth_polar_reprojection_audit,
    points_in_polar_layout,
    points_in_tetrahedral_mesh,
    structured_polar_tetrahedral_mesh,
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


def atomic_json(path: Path, payload: dict) -> None:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    content["sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-quantile", type=float, default=0.95)
    parser.add_argument("--padding-m", type=float, default=0.05)
    parser.add_argument("--simplification-tolerance-m", type=float, default=0.03)
    parser.add_argument("--maximum-edge-m", type=float, default=0.22)
    parser.add_argument(
        "--completion-calibration",
        type=Path,
        help="train-only calibration JSON for bounded radial depth completion",
    )
    parser.add_argument(
        "--solver-backend", choices=("superlu", "mkl_pardiso"), default="mkl_pardiso"
    )
    parser.add_argument("--solver-threads", type=int, default=24)
    parser.add_argument("--mkl-runtime", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
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
    candidates = filter_frozen_query_candidates(record, geometry_audit, room_base)
    receiver = np.asarray(record["receiver_global"], dtype=np.float64)
    truth = np.asarray(record["source_global"], dtype=np.float64)
    context_sources = np.asarray(record["context_sources_global"], dtype=np.float64)

    receiver_id = int(record["filename"].split("_")[1][1:])
    depth_path = (
        args.dataset_root
        / "depth_map"
        / record["scene"]
        / record["room"]
        / f"{receiver_id}.npy"
    )
    depth = np.load(depth_path)
    layout_started = time.perf_counter()
    layout, layout_audit = depth_panorama_polar_layout(
        depth,
        wall_quantile=args.wall_quantile,
        simplification_tolerance_m=args.simplification_tolerance_m,
        padding_m=args.padding_m,
    )
    completion_audit = None
    calibration_audit = None
    output_tag = "depth_polar"
    if args.completion_calibration is not None:
        calibration = json.loads(args.completion_calibration.read_text())
        if calibration.get("method") != "depth_bounded_radial_completion_train_calibration":
            raise ValueError("completion calibration has the wrong method contract")
        completion_distance = float(calibration["calibrated_completion_distance_m"])
        lower, upper, aabb_audit = depth_panorama_aabb(
            depth, padding_m=args.padding_m
        )
        layout, completion_audit = complete_polar_layout_toward_depth_aabb(
            layout,
            lower,
            upper,
            completion_distance_m=completion_distance,
        )
        calibration_audit = {
            "path": str(args.completion_calibration.resolve()),
            "sha256": file_sha256(args.completion_calibration),
            "train_split_sha256": calibration["train_split_sha256"],
            "calibration_quantile": calibration["calibration_quantile"],
            "calibrated_completion_distance_m": completion_distance,
            "training_room_count": calibration["room_count"],
            "training_receiver_view_count": calibration["selected_receiver_view_count"],
            "same_view_depth_aabb_audit": aabb_audit,
        }
        output_tag = "depth_completion"
    reprojection_audit = depth_polar_reprojection_audit(depth, layout)
    layout_seconds = time.perf_counter() - layout_started

    continuous_candidate_inside = points_in_polar_layout(
        candidates - receiver, layout, tolerance_m=1e-7
    )
    continuous_source_inside = points_in_polar_layout(
        (truth - receiver)[None, :], layout, tolerance_m=1e-7
    )
    continuous_context_inside = points_in_polar_layout(
        context_sources - receiver, layout, tolerance_m=1e-7
    )

    mesh_started = time.perf_counter()
    mesh, lattice_audit = structured_polar_tetrahedral_mesh(
        layout,
        receiver,
        maximum_edge_m=args.maximum_edge_m,
    )
    mesh_seconds = time.perf_counter() - mesh_started
    mesh_path = output_dir / f"query_{args.query_index:05d}_{output_tag}_mesh.npz"
    depth_hash = file_sha256(depth_path)
    save_tetrahedral_mesh_npz(mesh, mesh_path, source_mesh_sha256=depth_hash)
    layout_path = output_dir / f"query_{args.query_index:05d}_{output_tag}_layout.npz"
    np.savez_compressed(
        layout_path,
        theta_rad=layout.theta_rad.astype("<f8"),
        wall_radius_m=layout.wall_radius_m.astype("<f8"),
        floor_z_m=np.asarray(layout.floor_z_m, dtype="<f8"),
        ceiling_z_m=np.asarray(layout.ceiling_z_m, dtype="<f8"),
        polygon_vertices_xy_m=layout.polygon_vertices_xy_m.astype("<f8"),
    )

    all_audit_points = np.concatenate(
        (receiver[None, :], truth[None, :], context_sources, candidates), axis=0
    )
    mesh_inside = points_in_tetrahedral_mesh(mesh, all_audit_points)
    receiver_mesh_inside = bool(mesh_inside[0])
    source_mesh_inside = bool(mesh_inside[1])
    context_mesh_inside = mesh_inside[2 : 2 + len(context_sources)]
    candidate_mesh_inside = mesh_inside[2 + len(context_sources) :]
    if not receiver_mesh_inside:
        raise RuntimeError("depth-polar tetrahedral mesh does not contain the receiver")

    point_audit = {
        "continuous_layout": {
            "candidate_inside_count": int(continuous_candidate_inside.sum()),
            "candidate_count": int(len(candidates)),
            "source_inside": bool(continuous_source_inside[0]),
            "context_inside_count": int(continuous_context_inside.sum()),
            "context_count": int(len(context_sources)),
        },
        "voxelized_tetrahedral_mesh": {
            "receiver_inside": receiver_mesh_inside,
            "candidate_inside_count": int(candidate_mesh_inside.sum()),
            "candidate_excluded_count": int((~candidate_mesh_inside).sum()),
            "candidate_count": int(len(candidates)),
            "source_inside": source_mesh_inside,
            "context_inside_count": int(context_mesh_inside.sum()),
            "context_count": int(len(context_sources)),
        },
    }
    continuous_gate_passed = bool(
        continuous_candidate_inside.all()
        and continuous_source_inside[0]
        and continuous_context_inside.all()
    )
    tetrahedral_gate_passed = bool(
        candidate_mesh_inside.all()
        and source_mesh_inside
        and context_mesh_inside.all()
        and receiver_mesh_inside
    )
    base_payload = {
        "schema_version": 2,
        "method": (
            "fem_sabine_depth_bounded_radial_completion"
            if completion_audit is not None
            else "fem_sabine_depth_polar_layout"
        ),
        "interpretation": (
            "metric-depth, LGT-inspired horizon-depth layout with train-calibrated bounded "
            "radial hidden-space completion; no official unseen-room mesh"
            if completion_audit is not None
            else "metric-depth, LGT-inspired horizon-depth layout; deterministic plane/polar "
            "recovery without the LGT-Net learned RGB predictor or official room mesh"
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
            "continuous_layout_gate_passed": continuous_gate_passed,
            "tetrahedral_mesh_gate_passed": tetrahedral_gate_passed,
        },
        "depth_path": str(depth_path),
        "depth_sha256": depth_hash,
        "mesh_file": mesh_path.name,
        "mesh_sha256": file_sha256(mesh_path),
        "layout_file": layout_path.name,
        "layout_sha256": file_sha256(layout_path),
        "layout_audit": layout_audit,
        "completion_audit": completion_audit,
        "calibration_audit": calibration_audit,
        "depth_reprojection_audit": reprojection_audit,
        "point_audit": point_audit,
        "lattice_audit": lattice_audit,
        "runtime_seconds": {
            "layout_recovery": layout_seconds,
            "mesh_construction": mesh_seconds,
        },
    }
    if args.geometry_only:
        geometry_path = output_dir / f"query_{args.query_index:05d}_{output_tag}_geometry.json"
        atomic_json(geometry_path, base_payload)
        print(json.dumps(json.loads(geometry_path.read_text()), indent=2, sort_keys=True))
        return

    if not continuous_gate_passed or not tetrahedral_gate_passed:
        raise RuntimeError(
            "strict shared-candidate coverage gate failed: "
            f"continuous_candidates={int(continuous_candidate_inside.sum())}/{len(candidates)}, "
            f"mesh_candidates={int(candidate_mesh_inside.sum())}/{len(candidates)}, "
            f"continuous_source={bool(continuous_source_inside[0])}, "
            f"mesh_source={source_mesh_inside}, "
            f"continuous_contexts={int(continuous_context_inside.sum())}/{len(context_sources)}, "
            f"mesh_contexts={int(context_mesh_inside.sum())}/{len(context_sources)}"
        )

    observed, metadata = load_frozen_query(record, args.dataset_root)
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
    scores, recovery = score_fem_room_helps_candidates(
        forward.response,
        forward.bin_indices,
        observed,
        sample_count=observed.shape[-1],
    )
    prediction_index = stable_argmax(scores)
    metrics = localization_metrics(candidates, truth, prediction_index)
    metrics["prediction_global"] = candidates[prediction_index].tolist()
    metrics["winning_score"] = float(scores[prediction_index])
    metrics["mean_candidate_score"] = float(scores.mean())

    arrays_path = output_dir / f"query_{args.query_index:05d}_{output_tag}_scores.npz"
    np.savez_compressed(
        arrays_path,
        candidates=candidates.astype(np.float32),
        candidate_inside=candidate_mesh_inside,
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
            "support_full_candidate_indices": list(recovery.support),
            "relative_residual_norm": recovery.relative_residual_norm,
        },
        "metrics": metrics,
        "runtime_seconds": {
            **base_payload["runtime_seconds"],
            "operator_construction": operator_seconds,
            "fullband_solve": solve_seconds,
            "total": layout_seconds + mesh_seconds + operator_seconds + solve_seconds,
        },
    }
    result_path = output_dir / f"query_{args.query_index:05d}_{output_tag}_result.json"
    atomic_json(result_path, payload)
    print(json.dumps(json.loads(result_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
