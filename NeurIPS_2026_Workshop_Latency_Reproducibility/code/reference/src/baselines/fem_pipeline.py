"""End-to-end FEM-Sabine forward path over an audited tetrahedral air mesh."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .fem_sabine import (
    SabineBoundary,
    bandlimited_response_to_waveform,
    dft_frequency_bins,
    estimate_context_t60,
    sabine_boundary,
)
from .fem_solver import (
    BarycentricPointLocator,
    P1Matrices,
    SparseDirectSolveSession,
    SparseDirectSolverOptions,
    TetrahedralMesh,
    assemble_p1_matrices,
    barycentric_interpolation_matrix,
    build_barycentric_point_locator,
    solve_receiver_transfer_functions,
)
from .protocol import SAMPLE_RATE, TARGET_SAMPLES


TETRA_MESH_SCHEMA_VERSION = 1
FEM_MAXIMUM_EDGE_M = 0.22


@dataclass(frozen=True)
class FEMSabineForwardResult:
    waveforms: torch.Tensor | None
    response: torch.Tensor
    bin_indices: np.ndarray
    frequencies_hz: np.ndarray
    boundary: SabineBoundary
    audit: dict


@dataclass(frozen=True)
class FEMRoomOperators:
    """Room-invariant FEM matrices and spatial search index."""

    mesh: TetrahedralMesh
    matrices: P1Matrices
    locator: BarycentricPointLocator


def prepare_fem_room_operators(mesh: TetrahedralMesh) -> FEMRoomOperators:
    """Prepare expensive room-invariant operators once for repeated queries."""

    return FEMRoomOperators(
        mesh=mesh,
        matrices=assemble_p1_matrices(mesh),
        locator=build_barycentric_point_locator(mesh),
    )


def prepare_fem_query_interpolation(
    room_operators: FEMRoomOperators,
    receiver_point: np.ndarray,
    candidate_points: np.ndarray,
):
    """Prepare query-dependent loads once for every requested context count."""

    receiver = np.asarray(receiver_point, dtype=np.float64)
    candidates = np.asarray(candidate_points, dtype=np.float64)
    if receiver.shape != (3,) or candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("receiver and candidate points must have shapes [3] and [M,3]")
    receiver_load = barycentric_interpolation_matrix(
        room_operators.mesh,
        receiver.reshape(1, 3),
        locator=room_operators.locator,
    ).toarray()[0]
    candidate_interpolation = barycentric_interpolation_matrix(
        room_operators.mesh,
        candidates,
        locator=room_operators.locator,
    )
    return receiver_load, candidate_interpolation


def save_tetrahedral_mesh_npz(
    mesh: TetrahedralMesh,
    path: Path | str,
    *,
    source_mesh_sha256: str,
) -> None:
    """Persist an externally generated/audited tetrahedral mesh with provenance."""

    if not isinstance(source_mesh_sha256, str) or not source_mesh_sha256:
        raise ValueError("source_mesh_sha256 must be a nonempty provenance string")
    metadata = {
        "schema_version": TETRA_MESH_SCHEMA_VERSION,
        "source_mesh_sha256": source_mesh_sha256,
        "node_count": len(mesh.nodes),
        "element_count": len(mesh.elements),
    }
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    np.savez_compressed(
        destination,
        nodes=mesh.nodes.astype("<f8"),
        elements=mesh.elements.astype("<i8"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_tetrahedral_mesh_npz(
    path: Path | str,
) -> tuple[TetrahedralMesh, dict]:
    """Load only the versioned tetrahedral schema; reject unlabeled point clouds."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"nodes", "elements", "metadata_json"}:
            raise ValueError("tetrahedral mesh schema keys are missing or unexpected")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid tetrahedral mesh schema metadata") from error
        if metadata.get("schema_version") != TETRA_MESH_SCHEMA_VERSION:
            raise ValueError("unsupported tetrahedral mesh schema version")
        mesh = TetrahedralMesh(nodes=archive["nodes"], elements=archive["elements"])
    if metadata.get("node_count") != len(mesh.nodes) or metadata.get("element_count") != len(
        mesh.elements
    ):
        raise ValueError("tetrahedral mesh schema counts do not match arrays")
    if not metadata.get("source_mesh_sha256"):
        raise ValueError("tetrahedral mesh schema lacks source provenance")
    return mesh, metadata


def run_fem_sabine_forward(
    mesh: TetrahedralMesh,
    *,
    receiver_point: np.ndarray,
    candidate_points: np.ndarray,
    context_waveforms: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    sample_count: int = TARGET_SAMPLES,
    minimum_hz: float = 80.0,
    maximum_hz: float = 300.0,
    unit_gain: float = 1.0,
    speed_of_sound: float = 343.0,
    maximum_allowed_edge_m: float | None = FEM_MAXIMUM_EDGE_M,
    construct_waveforms: bool = True,
    room_operators: FEMRoomOperators | None = None,
    receiver_load: np.ndarray | None = None,
    candidate_interpolation=None,
    solver_options: SparseDirectSolverOptions | None = None,
    solver_session: SparseDirectSolveSession | None = None,
) -> FEMSabineForwardResult:
    """Estimate one Sabine boundary and solve its multifrequency FEM dictionary."""

    if not math.isfinite(speed_of_sound) or speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be finite and positive")
    receiver = np.asarray(receiver_point, dtype=np.float64)
    candidates = np.asarray(candidate_points, dtype=np.float64)
    if receiver.shape != (3,) or candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("receiver and candidate points must have shapes [3] and [M,3]")
    if len(candidates) == 0:
        raise ValueError("at least one FEM candidate is required")
    if room_operators is None:
        room_operators = prepare_fem_room_operators(mesh)
    elif room_operators.mesh is not mesh:
        raise ValueError("room operators were prepared for a different FEM mesh")
    matrices = room_operators.matrices
    if maximum_allowed_edge_m is not None:
        maximum_allowed_edge_m = float(maximum_allowed_edge_m)
        if not math.isfinite(maximum_allowed_edge_m) or maximum_allowed_edge_m <= 0:
            raise ValueError("maximum_allowed_edge_m must be finite and positive")
        if matrices.maximum_element_edge_m > maximum_allowed_edge_m + 1e-12:
            raise ValueError(
                "tetrahedral mesh maximum edge exceeds the FEM frequency-resolution gate"
            )
    t60 = estimate_context_t60(context_waveforms, sample_rate=sample_rate)
    boundary = sabine_boundary(
        matrices.volume_m3, matrices.surface_area_m2, t60.t60_seconds
    )
    bin_indices, frequencies = dft_frequency_bins(
        sample_rate, sample_count, minimum_hz, maximum_hz
    )
    if receiver_load is None:
        receiver_load = barycentric_interpolation_matrix(
            mesh, receiver.reshape(1, 3), locator=room_operators.locator
        ).toarray()[0]
    if candidate_interpolation is None:
        candidate_interpolation = barycentric_interpolation_matrix(
            mesh, candidates, locator=room_operators.locator
        )
    response_np, relative_residuals, solver_profile = solve_receiver_transfer_functions(
        matrices,
        receiver_load=receiver_load,
        candidate_interpolation=candidate_interpolation,
        frequencies_hz=frequencies,
        normalized_impedance=boundary.normalized_impedance,
        speed_of_sound=speed_of_sound,
        return_relative_residuals=True,
        solver_options=solver_options,
        solver_session=solver_session,
        return_solver_profile=True,
    )
    response = torch.from_numpy(response_np).to(torch.complex64)
    waveforms = (
        bandlimited_response_to_waveform(
            response, bin_indices, sample_count=sample_count, unit_gain=unit_gain
        )
        if construct_waveforms
        else None
    )
    audit = {
        "node_count": len(mesh.nodes),
        "element_count": len(mesh.elements),
        "candidate_count": len(candidates),
        "frequency_count": len(frequencies),
        "frequency_min_hz": float(frequencies[0]),
        "frequency_max_hz": float(frequencies[-1]),
        "volume_m3": matrices.volume_m3,
        "surface_area_m2": matrices.surface_area_m2,
        "maximum_element_edge_m": matrices.maximum_element_edge_m,
        "minimum_element_volume_m3": matrices.minimum_element_volume_m3,
        "minimum_element_mean_ratio": matrices.minimum_element_mean_ratio,
        "maximum_allowed_edge_m": maximum_allowed_edge_m,
        "t60_seconds": t60.t60_seconds,
        "t60_valid_contexts": t60.valid_count,
        "t60_invalid_contexts": t60.invalid_count,
        "sabine_raw_absorption": boundary.raw_absorption,
        "sabine_absorption": boundary.absorption,
        "sabine_was_clipped": boundary.was_clipped,
        "normalized_impedance": boundary.normalized_impedance,
        "diagnostic_waveform_gain": float(unit_gain) if construct_waveforms else None,
        "waveform_constructed": bool(construct_waveforms),
        "speed_of_sound_m_s": float(speed_of_sound),
        "maximum_relative_solver_residual": float(relative_residuals.max()),
        "median_relative_solver_residual": float(np.median(relative_residuals)),
        "solver_profile": solver_profile,
    }
    return FEMSabineForwardResult(
        waveforms=waveforms,
        response=response,
        bin_indices=bin_indices,
        frequencies_hz=frequencies,
        boundary=boundary,
        audit=audit,
    )
