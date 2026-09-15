"""Dependency-light first-order tetrahedral Helmholtz FEM primitives."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def _tetrahedral_faces(elements: np.ndarray) -> np.ndarray:
    """Return the four sorted triangular faces of every tetrahedron."""

    return np.sort(
        elements[:, ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))],
        axis=2,
    )


@dataclass(frozen=True)
class TetrahedralMesh:
    nodes: np.ndarray
    elements: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4:
            raise ValueError("nodes must have shape [N, 3]")
        if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
            raise ValueError("elements must have shape [E, 4]")
        if not np.isfinite(nodes).all() or elements.min() < 0 or elements.max() >= len(nodes):
            raise ValueError("mesh contains non-finite nodes or invalid element indices")
        sorted_elements = np.sort(elements, axis=1)
        if np.any(np.diff(sorted_elements, axis=1) == 0):
            raise ValueError("tetrahedra must contain four distinct nodes")
        if len(np.unique(sorted_elements, axis=0)) != len(elements):
            raise ValueError("mesh contains duplicate tetrahedra")
        coordinates = nodes[elements]
        determinants = np.linalg.det(
            np.transpose(coordinates[:, 1:] - coordinates[:, :1], (0, 2, 1))
        )
        if not np.isfinite(determinants).all() or np.any(
            np.abs(determinants) <= 1e-14
        ):
            raise ValueError("mesh contains a degenerate tetrahedron")

        faces = _tetrahedral_faces(elements).reshape(-1, 3)
        _unique_faces, inverse_faces, face_counts = np.unique(
            faces, axis=0, return_inverse=True, return_counts=True
        )
        if np.any(face_counts > 2):
            raise ValueError("tetrahedral air domain contains a non-manifold face")
        if len(elements) > 1:
            occurrence_order = np.argsort(inverse_faces, kind="stable")
            occurrence_starts = np.cumsum(face_counts) - face_counts
            shared_faces = np.flatnonzero(face_counts == 2)
            first_owners = occurrence_order[occurrence_starts[shared_faces]] // 4
            second_owners = occurrence_order[occurrence_starts[shared_faces] + 1] // 4
            adjacency = sparse.coo_matrix(
                (
                    np.ones(2 * len(shared_faces), dtype=np.int8),
                    (
                        np.concatenate((first_owners, second_owners)),
                        np.concatenate((second_owners, first_owners)),
                    ),
                ),
                shape=(len(elements), len(elements)),
            ).tocsr()
            component_count = connected_components(
                adjacency, directed=False, return_labels=False
            )
        else:
            component_count = 1
        if component_count != 1:
            raise ValueError("tetrahedral air domain must be face-connected")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", elements)


@dataclass(frozen=True)
class P1Matrices:
    stiffness: sparse.csr_matrix
    mass: sparse.csr_matrix
    boundary_mass: sparse.csr_matrix
    volume_m3: float
    surface_area_m2: float
    maximum_element_edge_m: float
    minimum_element_volume_m3: float
    minimum_element_mean_ratio: float


SUPERLU_ORDERINGS = ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
SPARSE_DIRECT_BACKENDS = ("auto", "superlu", "mkl_pardiso")


@dataclass(frozen=True)
class SparseDirectSolverOptions:
    """Runtime controls for exact sparse direct Helmholtz solves."""

    backend: str = "superlu"
    superlu_ordering: str = "MMD_AT_PLUS_A"
    threads: int = 1

    def __post_init__(self) -> None:
        if self.backend not in SPARSE_DIRECT_BACKENDS:
            raise ValueError(f"unsupported sparse direct backend: {self.backend}")
        if self.superlu_ordering not in SUPERLU_ORDERINGS:
            raise ValueError(
                f"unsupported SuperLU ordering: {self.superlu_ordering}"
            )
        if not isinstance(self.threads, int) or isinstance(self.threads, bool):
            raise ValueError("sparse direct solver threads must be an integer")
        if self.threads <= 0:
            raise ValueError("sparse direct solver threads must be positive")


@dataclass(frozen=True)
class BarycentricPointLocator:
    """Reusable tetrahedron search index for one frozen FEM mesh."""

    centroids: np.ndarray
    radii: np.ndarray
    tree: cKDTree


def build_barycentric_point_locator(mesh: TetrahedralMesh) -> BarycentricPointLocator:
    """Build the expensive spatial index once and reuse it across all query points."""

    element_coordinates = mesh.nodes[mesh.elements]
    centroids = element_coordinates.mean(axis=1)
    radii = np.linalg.norm(
        element_coordinates - centroids[:, None, :], axis=2
    ).max(axis=1)
    return BarycentricPointLocator(
        centroids=centroids,
        radii=radii,
        tree=cKDTree(centroids),
    )


def _mkl_pardiso_is_available() -> bool:
    try:
        import sparse_dot_mkl  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_sparse_direct_backend(requested: str) -> str:
    """Resolve ``auto`` without silently falling back from an explicit backend."""

    if requested not in SPARSE_DIRECT_BACKENDS:
        raise ValueError(f"unsupported sparse direct backend: {requested}")
    if requested == "auto":
        return "mkl_pardiso" if _mkl_pardiso_is_available() else "superlu"
    if requested == "mkl_pardiso" and not _mkl_pardiso_is_available():
        raise RuntimeError(
            "MKL PARDISO backend is unavailable; install sparse-dot-mkl and expose "
            "libmkl_rt with MKL_RT or LD_LIBRARY_PATH"
        )
    return requested


def _sparse_pattern_sha256(matrix: sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    return digest.hexdigest()


class SparseDirectSolveSession:
    """Profiled direct solver with reusable PARDISO symbolic analysis."""

    def __init__(self, options: SparseDirectSolverOptions | None = None):
        requested = options or SparseDirectSolverOptions()
        backend = resolve_sparse_direct_backend(requested.backend)
        self.options = SparseDirectSolverOptions(
            backend=backend,
            superlu_ordering=requested.superlu_ordering,
            threads=requested.threads,
        )
        self._pattern_sha256: str | None = None
        self._mkl_state: dict[str, Any] | None = None

    def __enter__(self) -> "SparseDirectSolveSession":
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.close()

    def _release_mkl(self) -> None:
        state = self._mkl_state
        if state is None:
            return
        from sparse_dot_mkl import pardiso

        _solution, _pt, _perm, error = pardiso(
            state["matrix"],
            state["rhs"],
            state["pt"],
            6,
            state["iparm"],
            phase=-1,
            perm=state["perm"],
            quiet=True,
        )
        self._mkl_state = None
        self._pattern_sha256 = None
        if error not in (0,):
            raise RuntimeError(f"MKL PARDISO memory release failed with error {error}")

    def close(self) -> None:
        if self.options.backend == "mkl_pardiso":
            self._release_mkl()

    def _solve_superlu(
        self, matrix: sparse.csc_matrix, right_hand_side: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        factor_started = time.perf_counter()
        factor = sparse_linalg.splu(
            matrix,
            permc_spec=self.options.superlu_ordering,
        )
        factor_seconds = time.perf_counter() - factor_started
        solve_started = time.perf_counter()
        solution = factor.solve(right_hand_side)
        solve_seconds = time.perf_counter() - solve_started
        return solution, {
            "backend": "superlu",
            "superlu_ordering": self.options.superlu_ordering,
            "threads": 1,
            "symbolic_analysis_seconds": None,
            "symbolic_analysis_reused": False,
            "factorization_seconds": float(factor_seconds),
            "solve_seconds": float(solve_seconds),
            "factor_nonzeros": int(factor.L.nnz + factor.U.nnz),
        }

    def _solve_mkl_pardiso(
        self, matrix: sparse.csc_matrix, right_hand_side: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        from sparse_dot_mkl import (
            mkl_set_num_threads_local,
            pardiso,
            pardisoinit,
        )

        mkl_set_num_threads_local(self.options.threads)
        upper = sparse.triu(matrix, format="csr")
        upper.sort_indices()
        pattern_sha256 = _sparse_pattern_sha256(upper)
        symbolic_seconds = 0.0
        symbolic_reused = pattern_sha256 == self._pattern_sha256
        if not symbolic_reused:
            self._release_mkl()
            pt, iparm = pardisoinit(6, single_precision=False, zero_indexing=True)
            symbolic_started = time.perf_counter()
            _unused, pt, perm, error = pardiso(
                upper,
                np.zeros(upper.shape[0], dtype=np.complex128),
                pt,
                6,
                iparm,
                phase=11,
            )
            symbolic_seconds = time.perf_counter() - symbolic_started
            if error != 0:
                raise RuntimeError(
                    f"MKL PARDISO symbolic analysis failed with error {error}"
                )
            self._pattern_sha256 = pattern_sha256
            self._mkl_state = {
                "pt": pt,
                "iparm": iparm,
                "perm": perm,
                "matrix": upper,
                "rhs": np.zeros(upper.shape[0], dtype=np.complex128),
            }
        state = self._mkl_state
        if state is None:
            raise RuntimeError("MKL PARDISO state was not initialized")
        factor_started = time.perf_counter()
        _unused, pt, perm, error = pardiso(
            upper,
            np.zeros(upper.shape[0], dtype=np.complex128),
            state["pt"],
            6,
            state["iparm"],
            phase=22,
            perm=state["perm"],
        )
        factor_seconds = time.perf_counter() - factor_started
        if error != 0:
            raise RuntimeError(f"MKL PARDISO factorization failed with error {error}")
        rhs_array = np.asarray(right_hand_side, dtype=np.complex128)
        if rhs_array.ndim == 2:
            # PARDISO consumes dense multi-RHS buffers in column-major order,
            # while sparse-dot-mkl's ctypes boundary requires C-contiguous
            # arrays.  Pack the Fortran-order byte stream into a C-contiguous
            # array and undo that representation on the returned solution.
            rhs = np.ascontiguousarray(
                rhs_array.ravel(order="F").reshape(rhs_array.shape)
            )
        else:
            rhs = np.ascontiguousarray(rhs_array)
        solve_started = time.perf_counter()
        solution, pt, perm, error = pardiso(
            upper,
            rhs,
            pt,
            6,
            state["iparm"],
            phase=33,
            perm=perm,
        )
        solve_seconds = time.perf_counter() - solve_started
        if error != 0:
            raise RuntimeError(f"MKL PARDISO solve failed with error {error}")
        if rhs_array.ndim == 2:
            solution = solution.ravel(order="C").reshape(
                rhs_array.shape, order="F"
            )
        state.update(
            {
                "pt": pt,
                "perm": perm,
                "matrix": upper,
                "rhs": np.zeros(upper.shape[0], dtype=np.complex128),
            }
        )
        factor_nonzeros = int(state["iparm"][17])
        return solution, {
            "backend": "mkl_pardiso",
            "matrix_type": "complex_symmetric",
            "threads": self.options.threads,
            "symbolic_analysis_seconds": float(symbolic_seconds),
            "symbolic_analysis_reused": bool(symbolic_reused),
            "factorization_seconds": float(factor_seconds),
            "solve_seconds": float(solve_seconds),
            "factor_nonzeros": factor_nonzeros if factor_nonzeros >= 0 else None,
        }

    def solve(
        self, matrix: sparse.spmatrix, right_hand_side: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Solve one complex-symmetric system and return phase-level profiling."""

        system = sparse.csc_matrix(matrix, dtype=np.complex128)
        rhs = np.asarray(right_hand_side, dtype=np.complex128)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != system.shape[0]:
            raise ValueError("right-hand side must have shape [N] or [N, R]")
        if self.options.backend == "superlu":
            return self._solve_superlu(system, rhs)
        return self._solve_mkl_pardiso(system, rhs)


def _assemble_sparse(
    size: int,
    rows: np.ndarray,
    columns: np.ndarray,
    data: np.ndarray,
) -> sparse.csr_matrix:
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(size, size),
    ).tocsr()


def assemble_p1_matrices(mesh: TetrahedralMesh) -> P1Matrices:
    """Assemble P1 stiffness, consistent mass, and exterior-boundary mass."""

    elements = mesh.elements
    coordinates = mesh.nodes[elements]
    element_count = len(elements)
    interpolation = np.concatenate(
        (np.ones((element_count, 4, 1), dtype=np.float64), coordinates), axis=2
    )
    # Thousands of independent 4x4 operations are faster single-threaded;
    # letting OpenBLAS launch a full team here badly oversubscribes the host.
    with threadpool_limits(limits=1, user_api="blas"):
        inverse = np.linalg.inv(interpolation)
        determinants = np.linalg.det(
            np.transpose(coordinates[:, 1:] - coordinates[:, :1], (0, 2, 1))
        )
    gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))
    volumes = np.abs(determinants) / 6.0

    edge_first = np.asarray((1, 2, 2, 3, 3, 3), dtype=np.int64)
    edge_second = np.asarray((0, 0, 1, 0, 1, 2), dtype=np.int64)
    edge_vectors = coordinates[:, edge_first] - coordinates[:, edge_second]
    squared_edge_lengths = np.einsum("eij,eij->ei", edge_vectors, edge_vectors)
    mean_ratios = (
        12.0
        * np.power(3.0 * volumes, 2.0 / 3.0)
        / squared_edge_lengths.sum(axis=1)
    )

    local_stiffness = volumes[:, None, None] * np.einsum(
        "eik,ejk->eij", gradients, gradients
    )
    mass_template = np.ones((4, 4), dtype=np.float64) + np.eye(4)
    local_mass = volumes[:, None, None] / 20.0 * mass_template
    volume_rows = np.repeat(elements, 4, axis=1).reshape(-1)
    volume_columns = np.tile(elements, (1, 4)).reshape(-1)

    faces = _tetrahedral_faces(elements).reshape(-1, 3)
    unique_faces, face_counts = np.unique(faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[face_counts == 1]
    boundary_coordinates = mesh.nodes[boundary_faces]
    boundary_areas = np.linalg.norm(
        np.cross(
            boundary_coordinates[:, 1] - boundary_coordinates[:, 0],
            boundary_coordinates[:, 2] - boundary_coordinates[:, 0],
        ),
        axis=1,
    ) / 2.0
    boundary_template = np.ones((3, 3), dtype=np.float64) + np.eye(3)
    local_boundary_mass = (
        boundary_areas[:, None, None] / 12.0 * boundary_template
    )
    boundary_rows = np.repeat(boundary_faces, 3, axis=1).reshape(-1)
    boundary_columns = np.tile(boundary_faces, (1, 3)).reshape(-1)

    node_count = len(mesh.nodes)
    return P1Matrices(
        stiffness=_assemble_sparse(
            node_count,
            volume_rows,
            volume_columns,
            local_stiffness.reshape(-1),
        ),
        mass=_assemble_sparse(
            node_count, volume_rows, volume_columns, local_mass.reshape(-1)
        ),
        boundary_mass=_assemble_sparse(
            node_count,
            boundary_rows,
            boundary_columns,
            local_boundary_mass.reshape(-1),
        ),
        volume_m3=float(volumes.sum()),
        surface_area_m2=float(boundary_areas.sum()),
        maximum_element_edge_m=float(np.sqrt(squared_edge_lengths.max())),
        minimum_element_volume_m3=float(volumes.min()),
        minimum_element_mean_ratio=float(mean_ratios.min()),
    )


def barycentric_interpolation_matrix(
    mesh: TetrahedralMesh,
    points: np.ndarray,
    *,
    tolerance: float = 1e-9,
    locator: BarycentricPointLocator | None = None,
) -> sparse.csr_matrix:
    """Build an exact P1 point interpolation matrix for points inside the mesh."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape [P, 3]")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    locator = locator or build_barycentric_point_locator(mesh)
    centroids = locator.centroids
    radii = locator.radii
    tree = locator.tree
    if len(centroids) != len(mesh.elements) or radii.shape != (len(mesh.elements),):
        raise ValueError("barycentric point locator does not match the FEM mesh")
    maximum_radius = float(radii.max()) + tolerance
    for point_index, point in enumerate(points):
        located = False
        possible = tree.query_ball_point(point, maximum_radius)
        possible.sort()
        for element_index in possible:
            if np.linalg.norm(point - centroids[element_index]) > radii[element_index] + tolerance:
                continue
            element = mesh.elements[element_index]
            coordinates = mesh.nodes[element]
            if np.any(point < coordinates.min(axis=0) - tolerance) or np.any(
                point > coordinates.max(axis=0) + tolerance
            ):
                continue
            system = np.vstack((coordinates.T, np.ones(4)))
            weights = np.linalg.solve(system, np.append(point, 1.0))
            if np.all(weights >= -tolerance) and np.all(weights <= 1.0 + tolerance):
                weights = np.clip(weights, 0.0, 1.0)
                weights /= weights.sum()
                rows.extend([point_index] * 4)
                columns.extend(map(int, element))
                values.extend(map(float, weights))
                located = True
                break
        if not located:
            raise ValueError(f"point {point_index} lies outside the tetrahedral mesh")
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(points), len(mesh.nodes))
    ).tocsr()


def solve_receiver_transfer_functions(
    matrices: P1Matrices,
    *,
    receiver_load: np.ndarray,
    candidate_interpolation: sparse.spmatrix,
    frequencies_hz: np.ndarray,
    normalized_impedance: float,
    speed_of_sound: float = 343.0,
    return_relative_residuals: bool = False,
    solver_options: SparseDirectSolverOptions | None = None,
    solver_session: SparseDirectSolveSession | None = None,
    return_solver_profile: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Use reciprocity to evaluate all candidate transfer functions per frequency."""

    receiver = np.asarray(receiver_load, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    candidates = sparse.csr_matrix(candidate_interpolation)
    node_count = matrices.stiffness.shape[0]
    matrix_shapes = {
        matrices.stiffness.shape, matrices.mass.shape, matrices.boundary_mass.shape
    }
    if matrix_shapes != {(node_count, node_count)}:
        raise ValueError("all FEM matrices must be square and share a shape")
    if receiver.shape != (node_count,) or candidates.shape[1] != node_count:
        raise ValueError("receiver and candidate interpolation must match FEM nodes")
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be a nonempty finite vector")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies must be positive")
    if not math.isfinite(normalized_impedance) or normalized_impedance <= 0:
        raise ValueError("normalized_impedance must be finite and positive")
    if not math.isfinite(speed_of_sound) or speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be finite and positive")
    transfer = np.empty((candidates.shape[0], len(frequencies)), dtype=np.complex128)
    residuals = np.empty(len(frequencies), dtype=np.float64)
    profiles = []
    stiffness = matrices.stiffness.astype(np.complex128)
    owns_session = solver_session is None
    session = solver_session or SparseDirectSolveSession(solver_options)
    try:
        for frequency_index, frequency in enumerate(frequencies):
            system_started = time.perf_counter()
            wavenumber = 2.0 * np.pi * frequency / speed_of_sound
            system = (
                stiffness
                - wavenumber**2 * matrices.mass
                + 1j * wavenumber / normalized_impedance * matrices.boundary_mass
            ).tocsc()
            system_seconds = time.perf_counter() - system_started
            pressure, profile = session.solve(system, receiver)
            if not np.isfinite(pressure).all():
                raise RuntimeError(f"non-finite FEM solution at {frequency:g} Hz")
            residual_started = time.perf_counter()
            residual = system @ pressure - receiver
            residuals[frequency_index] = np.linalg.norm(residual) / max(
                np.linalg.norm(receiver), np.finfo(np.float64).eps
            )
            transfer[:, frequency_index] = candidates @ pressure
            profile.update(
                {
                    "frequency_hz": float(frequency),
                    "system_construction_seconds": float(system_seconds),
                    "residual_and_sampling_seconds": float(
                        time.perf_counter() - residual_started
                    ),
                    "relative_residual": float(residuals[frequency_index]),
                    "system_nonzeros": int(system.nnz),
                    "right_hand_sides": 1,
                }
            )
            profiles.append(profile)
    finally:
        if owns_session:
            session.close()
    if not np.isfinite(residuals).all():
        raise RuntimeError("non-finite FEM linear-system residual")
    outputs: list[Any] = [transfer]
    if return_relative_residuals:
        outputs.append(residuals)
    if return_solver_profile:
        outputs.append(profiles)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
