import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from src.baselines.fem_solver import (
    SparseDirectSolveSession,
    SparseDirectSolverOptions,
    TetrahedralMesh,
    assemble_p1_matrices,
    barycentric_interpolation_matrix,
    build_barycentric_point_locator,
    solve_receiver_transfer_functions,
)


def _unit_tetrahedron():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return TetrahedralMesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]))


def test_p1_unit_tetrahedron_matrices_recover_volume_surface_and_symmetry():
    matrices = assemble_p1_matrices(_unit_tetrahedron())
    expected_surface = 1.5 + math.sqrt(3.0) / 2.0

    assert matrices.volume_m3 == pytest.approx(1.0 / 6.0)
    assert matrices.surface_area_m2 == pytest.approx(expected_surface)
    assert np.allclose(matrices.stiffness.toarray(), matrices.stiffness.toarray().T)
    assert np.allclose(matrices.mass.toarray(), matrices.mass.toarray().T)
    assert np.allclose(matrices.boundary_mass.toarray(), matrices.boundary_mass.toarray().T)
    assert matrices.mass.sum() == pytest.approx(1.0 / 6.0)
    assert matrices.boundary_mass.sum() == pytest.approx(expected_surface)
    assert np.allclose(matrices.stiffness.toarray().sum(axis=1), 0.0)


def test_shared_internal_tetrahedral_face_is_excluded_from_sabine_surface_area():
    mesh = TetrahedralMesh(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3], [0, 1, 2, 4]]),
    )
    matrices = assemble_p1_matrices(mesh)

    assert matrices.volume_m3 == pytest.approx(1.0 / 3.0)
    assert matrices.surface_area_m2 == pytest.approx(2.0 + math.sqrt(3.0))
    assert matrices.boundary_mass.sum() == pytest.approx(matrices.surface_area_m2)


def test_tetrahedral_mesh_rejects_degenerate_or_invalid_cells():
    nodes = np.zeros((4, 3))
    with pytest.raises(ValueError):
        TetrahedralMesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError):
        TetrahedralMesh(nodes=np.eye(4, 3), elements=np.array([[0, 1, 2, 4]]))


def test_tetrahedral_mesh_rejects_disconnected_or_nonmanifold_air_domains():
    disconnected_nodes = np.vstack(
        (
            _unit_tetrahedron().nodes,
            _unit_tetrahedron().nodes + np.array([3.0, 0.0, 0.0]),
        )
    )
    with pytest.raises(ValueError, match="connected"):
        TetrahedralMesh(
            nodes=disconnected_nodes,
            elements=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        )

    nodes = np.vstack((_unit_tetrahedron().nodes, [[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]))
    with pytest.raises(ValueError, match="non-manifold"):
        TetrahedralMesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]]),
        )


def test_barycentric_interpolation_is_exact_for_vertices_and_centroid():
    mesh = _unit_tetrahedron()
    points = np.vstack((mesh.nodes, np.full((1, 3), 0.25)))
    locator = build_barycentric_point_locator(mesh)
    interpolation = barycentric_interpolation_matrix(mesh, points, locator=locator)
    assert sparse.isspmatrix_csr(interpolation)
    assert np.allclose(interpolation[:4].toarray(), np.eye(4))
    assert np.allclose(interpolation[4].toarray(), np.full((1, 4), 0.25))
    assert np.allclose(interpolation.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="outside"):
        barycentric_interpolation_matrix(mesh, np.array([[2.0, 2.0, 2.0]]))


def test_superlu_session_profiles_factorization_ordering_and_residual():
    matrices = assemble_p1_matrices(_unit_tetrahedron())
    transfer, residuals, profiles = solve_receiver_transfer_functions(
        matrices,
        receiver_load=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_interpolation=sparse.identity(4, format="csr"),
        frequencies_hz=np.array([100.0, 200.0]),
        normalized_impedance=8.0,
        return_relative_residuals=True,
        return_solver_profile=True,
        solver_options=SparseDirectSolverOptions(
            backend="superlu", superlu_ordering="MMD_AT_PLUS_A"
        ),
    )

    assert transfer.shape == (4, 2)
    assert np.max(residuals) < 1e-10
    assert len(profiles) == 2
    assert all(profile["backend"] == "superlu" for profile in profiles)
    assert all(profile["superlu_ordering"] == "MMD_AT_PLUS_A" for profile in profiles)
    assert all(profile["factorization_seconds"] >= 0 for profile in profiles)
    assert all(profile["factor_nonzeros"] > 0 for profile in profiles)


def test_mkl_pardiso_complex_symmetric_session_reuses_symbolic_analysis(monkeypatch):
    runtime = Path("/opt/anaconda3/lib/libmkl_rt.so")
    if not runtime.is_file():
        pytest.skip("oneMKL runtime is unavailable")
    monkeypatch.setenv("MKL_RT", str(runtime))
    system = sparse.csc_matrix(
        np.array([[4.0 + 1.0j, 1.0 + 2.0j], [1.0 + 2.0j, 3.0 + 0.5j]])
    )
    right_hand_side = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with SparseDirectSolveSession(
        SparseDirectSolverOptions(backend="mkl_pardiso", threads=2)
    ) as session:
        first, first_profile = session.solve(system, right_hand_side)
        second, second_profile = session.solve(1.1 * system, right_hand_side)

    assert np.linalg.norm(system @ first - right_hand_side) < 1e-12
    assert np.linalg.norm((1.1 * system) @ second - right_hand_side) < 1e-12
    assert first_profile["symbolic_analysis_reused"] is False
    assert second_profile["symbolic_analysis_reused"] is True
    assert first_profile["backend"] == "mkl_pardiso"
    assert first_profile["matrix_type"] == "complex_symmetric"


def test_mkl_pardiso_multi_rhs_preserves_column_layout(monkeypatch):
    runtime = Path("/opt/anaconda3/lib/libmkl_rt.so")
    if not runtime.is_file():
        pytest.skip("oneMKL runtime is unavailable")
    monkeypatch.setenv("MKL_RT", str(runtime))
    system = sparse.csc_matrix(
        np.array(
            [
                [4.0 + 1.0j, 1.0 + 2.0j, 0.0, 0.0],
                [1.0 + 2.0j, 5.0 + 0.3j, 1.0, 0.0],
                [0.0, 1.0, 3.0 + 0.2j, 1.0j],
                [0.0, 0.0, 1.0j, 2.0 + 0.1j],
            ]
        )
    )
    right_hand_side = np.column_stack((np.arange(1, 5), np.arange(5, 9))).astype(
        np.complex128
    )

    with SparseDirectSolveSession(
        SparseDirectSolverOptions(backend="mkl_pardiso", threads=2)
    ) as session:
        solution, profile = session.solve(system, right_hand_side)

    relative_residuals = np.linalg.norm(
        system @ solution - right_hand_side, axis=0
    ) / np.linalg.norm(right_hand_side, axis=0)
    assert solution.shape == right_hand_side.shape
    assert np.max(relative_residuals) < 1e-12
    assert profile["backend"] == "mkl_pardiso"


def test_receiver_solve_is_reciprocal_with_direct_candidate_loads():
    matrices = assemble_p1_matrices(_unit_tetrahedron())
    receiver = np.array([1.0, 0.0, 0.0, 0.0])
    candidates = sparse.identity(4, format="csr")
    frequencies = np.array([100.0, 200.0])
    transfer = solve_receiver_transfer_functions(
        matrices,
        receiver_load=receiver,
        candidate_interpolation=candidates,
        frequencies_hz=frequencies,
        normalized_impedance=8.0,
        speed_of_sound=343.0,
    )

    assert transfer.shape == (4, 2)
    assert np.isfinite(transfer).all()
    for column, frequency in enumerate(frequencies):
        omega = 2.0 * np.pi * frequency
        wavenumber = omega / 343.0
        system = (
            matrices.stiffness.astype(complex)
            - wavenumber**2 * matrices.mass
            + 1j * wavenumber / 8.0 * matrices.boundary_mass
        )
        direct = []
        for source_node in range(4):
            source = np.zeros(4)
            source[source_node] = 1.0
            pressure = sparse.linalg.spsolve(system.tocsc(), source)
            direct.append(pressure[0])
        assert np.allclose(transfer[:, column], direct)


def test_receiver_solve_reports_small_relative_linear_residuals():
    matrices = assemble_p1_matrices(_unit_tetrahedron())
    transfer, residuals = solve_receiver_transfer_functions(
        matrices,
        receiver_load=np.array([1.0, 0.0, 0.0, 0.0]),
        candidate_interpolation=sparse.identity(4, format="csr"),
        frequencies_hz=np.array([100.0, 200.0]),
        normalized_impedance=8.0,
        return_relative_residuals=True,
    )

    assert transfer.shape == (4, 2)
    assert residuals.shape == (2,)
    assert np.max(residuals) < 1e-10
