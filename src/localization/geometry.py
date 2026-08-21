"""Mesh-valid global 3-D lattice and query filtering primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d

EPSILON_METERS = 1e-4


@dataclass
class MeshScene:
    path: Path
    sha256: str
    scene: o3d.t.geometry.RaycastingScene
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    vertex_count: int
    triangle_count: int
    diagnostics: dict[str, bool]


def snap_axis_to_lattice(bounds: Sequence[float], spacing: float) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError("axis bounds must be finite ordered scalars")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")
    start = np.ceil(bounds[0] / spacing) * spacing
    stop = np.floor(bounds[1] / spacing) * spacing
    if start > stop + EPSILON_METERS:
        return np.empty(0, dtype=np.float64)
    count = int(np.floor((stop - start) / spacing + EPSILON_METERS)) + 1
    return start + np.arange(count, dtype=np.float64) * spacing


def build_lattice(aabb_min, aabb_max, spacing: float | Sequence[float]) -> np.ndarray:
    lower = np.asarray(aabb_min, dtype=np.float64)
    upper = np.asarray(aabb_max, dtype=np.float64)
    delta = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("AABB endpoints must have shape (3,)")
    axes = [snap_axis_to_lattice((lower[i], upper[i]), float(delta[i])) for i in range(3)]
    if any(len(axis) == 0 for axis in axes):
        return np.empty((0, 3), dtype=np.float64)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def load_raycast_scene(mesh_path: Path | str, *, compute_topology: bool = True) -> MeshScene:
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    legacy = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(legacy.vertices)
    triangles = np.asarray(legacy.triangles)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
        raise ValueError(f"mesh has no valid vertices: {path}")
    if triangles.ndim != 2 or triangles.shape[0] == 0 or triangles.shape[1] != 3:
        raise ValueError(f"mesh has no valid triangles: {path}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"mesh contains non-finite vertices: {path}")
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(tensor_mesh)
    return MeshScene(
        path=path.resolve(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scene=ray_scene,
        aabb_min=vertices.min(axis=0).astype(np.float64),
        aabb_max=vertices.max(axis=0).astype(np.float64),
        vertex_count=len(vertices),
        triangle_count=len(triangles),
        diagnostics=(
            {
                "edge_manifold": bool(legacy.is_edge_manifold()),
                "vertex_manifold": bool(legacy.is_vertex_manifold()),
                "self_intersecting": bool(legacy.is_self_intersecting()),
                "watertight": bool(legacy.is_watertight()),
                "orientable": bool(legacy.is_orientable()),
            }
            if compute_topology
            else {"topology_deferred_for_fast_cost_audit": True}
        ),
    )


def _points(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite array with shape (N, 3)")
    return points


def classify_mesh_candidates(
    mesh: MeshScene,
    points,
    surface_clearance: float,
    *,
    eps: float = EPSILON_METERS,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    points = _points(points)
    if surface_clearance < 0 or eps < 0 or chunk_size <= 0:
        raise ValueError("clearance/eps/chunk_size must be nonnegative, nonnegative, positive")
    mask = np.zeros(len(points), dtype=bool)
    distances = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        tensor = o3d.core.Tensor(points[start:stop].astype(np.float32))
        occupancy = mesh.scene.compute_occupancy(tensor).numpy()
        distance = mesh.scene.compute_distance(tensor).numpy()
        distances[start:stop] = distance
        mask[start:stop] = (occupancy >= 0.5) & np.isfinite(distance) & (
            distance + eps >= surface_clearance
        )
    return mask, distances


def filter_query_candidates(
    points,
    receiver,
    context_sources,
    *,
    receiver_clearance: float = 0.5,
    context_clearance: float = 0.25,
    z_band: tuple[float, float] | None = None,
    eps: float = EPSILON_METERS,
) -> np.ndarray:
    points = _points(points)
    receiver = np.asarray(receiver, dtype=np.float64)
    contexts = np.asarray(context_sources, dtype=np.float64)
    if receiver.shape != (3,) or not np.all(np.isfinite(receiver)):
        raise ValueError("receiver must be finite with shape (3,)")
    if contexts.ndim != 2 or contexts.shape[1] != 3 or not np.all(np.isfinite(contexts)):
        raise ValueError("context sources must be finite with shape (K, 3)")
    if receiver_clearance < 0 or context_clearance < 0 or eps < 0:
        raise ValueError("clearances and eps must be nonnegative")
    mask = np.linalg.norm(points - receiver, axis=1) + eps >= receiver_clearance
    if len(contexts):
        nearest_context = np.linalg.norm(points[:, None, :] - contexts[None, :, :], axis=2).min(axis=1)
        mask &= nearest_context + eps >= context_clearance
    if z_band is not None:
        low, high = map(float, z_band)
        if not np.isfinite(low + high) or low > high:
            raise ValueError("z band must be finite and ordered")
        mask &= (points[:, 2] + eps >= low) & (points[:, 2] - eps <= high)
    return mask


def grid_oracle_error(points, truth) -> float:
    points = _points(points)
    truth = np.asarray(truth, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("candidate grid must be nonempty")
    if truth.shape != (3,) or not np.all(np.isfinite(truth)):
        raise ValueError("truth must be finite with shape (3,)")
    value = float(np.linalg.norm(points - truth, axis=1).min())
    if not np.isfinite(value):
        raise ValueError("oracle error is not finite")
    return value


def choose_z_band_branch(full_errors, z_errors, z_counts, threshold: float = 0.5) -> str:
    full = np.asarray(full_errors, dtype=np.float64)
    z = np.asarray(z_errors, dtype=np.float64)
    counts = np.asarray(z_counts)
    if full.shape != z.shape or full.shape != counts.shape or full.ndim != 1:
        raise ValueError("z-band audit arrays must be aligned vectors")
    if np.any(counts <= 0) or not np.all(np.isfinite(z)):
        return "full_height"
    newly_unwinnable = (z > threshold) & (full <= threshold)
    return "full_height" if np.any(newly_unwinnable) else "z_band"
