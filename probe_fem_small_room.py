#!/usr/bin/env python3
"""Gate one production/reference FEM pair before multi-room generation."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.fem_pipeline import load_tetrahedral_mesh_npz
from src.baselines.fem_sabine import dft_frequency_bins, estimate_context_t60, sabine_boundary
from src.baselines.fem_solver import (
    SparseDirectSolveSession,
    SparseDirectSolverOptions,
    assemble_p1_matrices,
    barycentric_interpolation_matrix,
    build_barycentric_point_locator,
)
from src.baselines.room_helps_sparse import extract_rir_frequency_response, room_helps_pulse_omp
from src.localization.ar_queries import load_context_manifest
from src.localization.engine import load_frozen_query, reconstruct_query_candidates
from src.localization.pilot import canonical_sha256, load_pilot_manifest, resolve_pilot_records


EXP09_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
EXP10_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_10_room_helps_baselines_claude"


def _atomic_hashed_json(path: Path, payload: dict) -> None:
    content = dict(payload)
    content["sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_geometry_audit(path: Path) -> dict:
    audit = json.loads(path.read_text())
    content = {key: value for key, value in audit.items() if key != "sha256"}
    if audit.get("sha256") != canonical_sha256(content):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    return audit


def _three_probe_bins(sample_rate: int, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    all_bins, all_frequencies = dft_frequency_bins(sample_rate, sample_count, 80.0, 300.0)
    positions = np.asarray([0, len(all_bins) // 2, len(all_bins) - 1], dtype=np.int64)
    return all_bins[positions], all_frequencies[positions]


def _solve_dictionary(
    mesh,
    matrices,
    receiver,
    candidates,
    frequencies,
    impedance,
    *,
    diagnostic_frequency_index: int | None,
    speed_of_sound: float = 343.0,
    receiver_interpolation=None,
    candidate_interpolation=None,
    point_locator=None,
    solver_options: SparseDirectSolverOptions | None = None,
    solver_session: SparseDirectSolveSession | None = None,
):
    if point_locator is None:
        point_locator = build_barycentric_point_locator(mesh)
    if receiver_interpolation is None:
        receiver_interpolation = barycentric_interpolation_matrix(
            mesh,
            np.asarray(receiver, dtype=np.float64).reshape(1, 3),
            locator=point_locator,
        ).toarray()[0]
    if candidate_interpolation is None:
        candidate_interpolation = barycentric_interpolation_matrix(
            mesh, candidates, locator=point_locator
        )
    response = np.empty((len(candidates), len(frequencies)), dtype=np.complex128)
    residuals = np.empty(len(frequencies), dtype=np.float64)
    diagnostics = None
    solver_profiles = []
    stiffness = matrices.stiffness.astype(np.complex128)
    owns_session = solver_session is None
    session = solver_session or SparseDirectSolveSession(solver_options)
    try:
        for frequency_index, frequency in enumerate(frequencies):
            system_started = time.perf_counter()
            wavenumber = 2.0 * np.pi * float(frequency) / speed_of_sound
            system = (
                stiffness
                - wavenumber**2 * matrices.mass
                + 1j * wavenumber / impedance * matrices.boundary_mass
            ).tocsc()
            system_seconds = time.perf_counter() - system_started
            if frequency_index == diagnostic_frequency_index:
                candidate_load = candidate_interpolation.getrow(0).toarray()[0]
                right_hand_side = np.column_stack(
                    (receiver_interpolation, candidate_load)
                )
                pressure, solver_profile = session.solve(system, right_hand_side)
                receiver_pressure = pressure[:, 0]
                candidate_pressure = pressure[:, 1]
                direct_value = np.vdot(receiver_interpolation, candidate_pressure)
                reciprocal_value = candidate_interpolation.getrow(0) @ receiver_pressure
                reciprocal_value = complex(np.asarray(reciprocal_value).item())
                driving_value = np.vdot(receiver_interpolation, receiver_pressure)
                diagnostics = {
                    "frequency_hz": float(frequency),
                    "relative_difference": float(
                        abs(direct_value - reciprocal_value)
                        / max(
                            abs(direct_value),
                            abs(reciprocal_value),
                            np.finfo(float).eps,
                        )
                    ),
                    "driving_point_real": float(driving_value.real),
                    "driving_point_imag": float(driving_value.imag),
                    "passive_dft_sign": bool(driving_value.imag < 0.0),
                }
                right_hand_sides = 2
            else:
                receiver_pressure, solver_profile = session.solve(
                    system, receiver_interpolation
                )
                right_hand_sides = 1
            residual_started = time.perf_counter()
            residual = system @ receiver_pressure - receiver_interpolation
            residuals[frequency_index] = np.linalg.norm(residual) / np.linalg.norm(
                receiver_interpolation
            )
            response[:, frequency_index] = candidate_interpolation @ receiver_pressure
            solver_profile.update(
                {
                    "frequency_hz": float(frequency),
                    "system_construction_seconds": float(system_seconds),
                    "residual_and_sampling_seconds": float(
                        time.perf_counter() - residual_started
                    ),
                    "relative_residual": float(residuals[frequency_index]),
                    "system_nonzeros": int(system.nnz),
                    "right_hand_sides": right_hand_sides,
                }
            )
            solver_profiles.append(solver_profile)
    finally:
        if owns_session:
            session.close()
    return response, residuals, diagnostics, solver_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", default="Bathrooms_idx_14")
    parser.add_argument(
        "--production-mesh",
        type=Path,
        default=EXP10_DIR / "fem_meshes/rooms/Bathrooms_idx_14.npz",
    )
    parser.add_argument(
        "--reference-mesh",
        type=Path,
        default=EXP10_DIR / "fem_meshes_reference_h014/rooms/Bathrooms_idx_14.npz",
    )
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=EXP09_DIR / "context_manifest_exp01_seed42.json",
    )
    parser.add_argument(
        "--geometry-audit", type=Path, default=EXP09_DIR / "geometry_audit.json"
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=EXP09_DIR / "pilot_manifest_seed42_4_per_room.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms"),
    )
    parser.add_argument(
        "--output", type=Path, default=EXP10_DIR / "fem_small_room_probe.json"
    )
    parser.add_argument("--maximum-convergence-error", type=float, default=0.35)
    parser.add_argument("--maximum-residual", type=float, default=1e-8)
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "superlu", "mkl_pardiso"),
        default="auto",
    )
    parser.add_argument(
        "--superlu-ordering",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default="MMD_AT_PLUS_A",
    )
    parser.add_argument("--solver-threads", type=int, default=1)
    args = parser.parse_args()

    context_manifest = load_context_manifest(args.context_manifest)
    geometry_audit = _load_geometry_audit(args.geometry_audit)
    pilot_manifest = load_pilot_manifest(args.pilot_manifest)
    joined = resolve_pilot_records(pilot_manifest, context_manifest, geometry_audit)
    matching = [item for item in joined if item[0]["room"] == args.room]
    if not matching:
        raise RuntimeError(f"pilot contains no query for room {args.room!r}")
    selected, record, _geometry = matching[0]
    observed, metadata = load_frozen_query(record, args.dataset_root)
    candidates = reconstruct_query_candidates(record, geometry_audit)
    truth = np.asarray(record["source_global"], dtype=np.float64)
    oracle_index = int(np.argmin(np.linalg.norm(candidates - truth, axis=1)))
    bin_indices, production_frequencies = _three_probe_bins(22050, observed.shape[-1])
    middle = len(production_frequencies) // 2
    t60 = estimate_context_t60(metadata["context_audio"][:8])
    solver_options = SparseDirectSolverOptions(
        backend=args.solver_backend,
        superlu_ordering=args.superlu_ordering,
        threads=args.solver_threads,
    )

    products = {}
    total_start = time.perf_counter()
    for name, path in (
        ("production", args.production_mesh),
        ("reference", args.reference_mesh),
    ):
        stage_start = time.perf_counter()
        mesh, mesh_metadata = load_tetrahedral_mesh_npz(path)
        matrices = assemble_p1_matrices(mesh)
        point_locator = build_barycentric_point_locator(mesh)
        receiver_interpolation = barycentric_interpolation_matrix(
            mesh,
            np.asarray(record["receiver_global"], dtype=np.float64).reshape(1, 3),
            locator=point_locator,
        ).toarray()[0]
        candidate_interpolation = barycentric_interpolation_matrix(
            mesh, candidates, locator=point_locator
        )
        boundary = sabine_boundary(
            matrices.volume_m3, matrices.surface_area_m2, t60.t60_seconds
        )
        frequencies = (
            production_frequencies
            if name == "production"
            else production_frequencies[middle : middle + 1]
        )
        diagnostic_frequency_index = middle if name == "production" else 0
        with SparseDirectSolveSession(solver_options) as solver_session:
            response, residuals, reciprocity, solver_profile = _solve_dictionary(
                mesh,
                matrices,
                np.asarray(record["receiver_global"], dtype=np.float64),
                candidates,
                frequencies,
                boundary.normalized_impedance,
                diagnostic_frequency_index=diagnostic_frequency_index,
                receiver_interpolation=receiver_interpolation,
                candidate_interpolation=candidate_interpolation,
                point_locator=point_locator,
                solver_options=solver_options,
                solver_session=solver_session,
            )
        products[name] = {
            "response": response,
            "audit": {
                "mesh_path": str(path.resolve()),
                "mesh_metadata": mesh_metadata,
                "node_count": len(mesh.nodes),
                "element_count": len(mesh.elements),
                "volume_m3": matrices.volume_m3,
                "surface_area_m2": matrices.surface_area_m2,
                "maximum_element_edge_m": matrices.maximum_element_edge_m,
                "minimum_element_mean_ratio": matrices.minimum_element_mean_ratio,
                "t60_seconds": t60.t60_seconds,
                "sabine_absorption": boundary.absorption,
                "normalized_impedance": boundary.normalized_impedance,
                "frequency_count": len(frequencies),
                "frequencies_hz": frequencies.tolist(),
                "maximum_relative_solver_residual": float(residuals.max()),
                "reciprocity": reciprocity,
                "solver_profile": solver_profile,
                "stage_seconds": time.perf_counter() - stage_start,
            },
        }
        print(
            f"{name}: {len(mesh.nodes)} nodes, residual={residuals.max():.3e}, "
            f"seconds={products[name]['audit']['stage_seconds']:.1f}",
            flush=True,
        )

    production_response = products["production"]["response"]
    reference_response = products["reference"]["response"]
    production_middle = production_response[:, middle : middle + 1]
    denominator = float(np.vdot(production_middle, production_middle).real)
    complex_alignment = np.vdot(production_middle, reference_response) / denominator
    convergence_error = float(
        np.linalg.norm(complex_alignment * production_middle - reference_response)
        / np.linalg.norm(reference_response)
    )
    synthetic_gain = 0.7 - 0.2j
    synthetic = room_helps_pulse_omp(
        production_response,
        synthetic_gain * production_response[oracle_index],
        source_count=1,
    )
    observed_bins = extract_rir_frequency_response(
        observed, bin_indices, sample_count=observed.shape[-1]
    )
    real_recovery = room_helps_pulse_omp(production_response, observed_bins, source_count=1)

    production_audit = products["production"]["audit"]
    reference_audit = products["reference"]["audit"]
    gates = {
        "production_solver_residual": production_audit[
            "maximum_relative_solver_residual"
        ]
        <= args.maximum_residual,
        "reference_solver_residual": reference_audit["maximum_relative_solver_residual"]
        <= args.maximum_residual,
        "production_reciprocity": production_audit["reciprocity"]["relative_difference"]
        <= 1e-10,
        "reference_reciprocity": reference_audit["reciprocity"]["relative_difference"]
        <= 1e-10,
        "production_passive_dft_sign": production_audit["reciprocity"][
            "passive_dft_sign"
        ],
        "reference_passive_dft_sign": reference_audit["reciprocity"][
            "passive_dft_sign"
        ],
        "mesh_convergence": convergence_error <= args.maximum_convergence_error,
        "synthetic_omp_support": synthetic.support == (oracle_index,),
        "synthetic_omp_residual": synthetic.relative_residual_norm <= 1e-10,
    }
    payload = {
        "schema_version": 1,
        "room": args.room,
        "query_index": int(selected["index"]),
        "query_id": selected["query_id"],
        "candidate_count": len(candidates),
        "oracle_candidate_index": oracle_index,
        "probe_bin_indices": bin_indices.tolist(),
        "probe_frequencies_hz": production_frequencies.tolist(),
        "t60_valid_contexts": t60.valid_count,
        "t60_invalid_contexts": t60.invalid_count,
        "production": production_audit,
        "reference": reference_audit,
        "convergence": {
            "common_complex_alignment_real": float(complex_alignment.real),
            "common_complex_alignment_imag": float(complex_alignment.imag),
            "relative_response_error": convergence_error,
            "maximum_allowed_error": float(args.maximum_convergence_error),
        },
        "synthetic_omp": {
            "support": list(synthetic.support),
            "relative_residual_norm": synthetic.relative_residual_norm,
        },
        "real_three_frequency_omp": {
            "support": list(real_recovery.support),
            "relative_residual_norm": real_recovery.relative_residual_norm,
            "localization_error_m": float(
                np.linalg.norm(candidates[real_recovery.support[0]] - truth)
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "total_seconds": time.perf_counter() - total_start,
        "maximum_resident_memory_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_hashed_json(args.output, payload)
    print(json.dumps({"passed": payload["passed"], "gates": gates}, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
