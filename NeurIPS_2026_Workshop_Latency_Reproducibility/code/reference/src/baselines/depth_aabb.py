"""Depth-panorama-only axis-aligned FEM envelope primitives.

This module deliberately implements a small, auditable pilot rather than a
general single-view surface-completion system.  The envelope is determined only
by the receiver-centred equirectangular radial-depth panorama and a fixed pad.
"""

from __future__ import annotations

import math

import numpy as np

from .fem_solver import TetrahedralMesh


def equirectangular_directions(height: int, width: int) -> np.ndarray:
    """Return FLAC-parity unit rays with shape ``[height, width, 3]``."""

    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("panorama dimensions must be integers")
    if height < 2 or width < 4:
        raise ValueError("panorama dimensions are too small")
    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    theta = (columns + 0.5) * (2.0 * math.pi / width) - math.pi
    phi = (rows + 0.5) * (math.pi / height) - math.pi / 2.0
    cos_phi = np.cos(phi)
    return np.stack(
        (
            cos_phi * np.cos(theta),
            cos_phi * np.sin(theta),
            -np.sin(phi),
        ),
        axis=-1,
    )


def depth_panorama_aabb(
    depth_map,
    *,
    padding_m: float = 0.05,
    minimum_valid_depth_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a padded receiver-local AABB to valid radial-depth endpoints."""

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("depth panorama must be a two-dimensional array")
    padding_m = float(padding_m)
    minimum_valid_depth_m = float(minimum_valid_depth_m)
    if not math.isfinite(padding_m) or padding_m < 0:
        raise ValueError("padding_m must be finite and nonnegative")
    if not math.isfinite(minimum_valid_depth_m) or minimum_valid_depth_m <= 0:
        raise ValueError("minimum_valid_depth_m must be finite and positive")
    valid = np.isfinite(depth) & (depth > minimum_valid_depth_m)
    if int(valid.sum()) < 8:
        raise ValueError("depth panorama contains too few valid samples")
    directions = equirectangular_directions(*depth.shape)
    points = directions[valid] * depth[valid, None]
    lower = points.min(axis=0) - padding_m
    upper = points.max(axis=0) + padding_m
    if np.any(lower >= 0.0) or np.any(upper <= 0.0):
        raise ValueError("depth-derived AABB must strictly contain the receiver origin")
    audit = {
        "valid_depth_count": int(valid.sum()),
        "invalid_depth_count": int(valid.size - valid.sum()),
        "valid_depth_fraction": float(valid.mean()),
        "minimum_valid_depth_m": float(depth[valid].min()),
        "maximum_valid_depth_m": float(depth[valid].max()),
        "padding_m": padding_m,
        "lower_receiver_local_m": lower.tolist(),
        "upper_receiver_local_m": upper.tolist(),
        "dimensions_m": (upper - lower).tolist(),
        "volume_m3": float(np.prod(upper - lower)),
    }
    return lower, upper, audit


def points_in_aabb(points, lower, upper, *, tolerance_m: float = 1e-9) -> np.ndarray:
    """Classify points against a closed axis-aligned box."""

    values = np.asarray(points, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    tolerance_m = float(tolerance_m)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("AABB bounds must have shape [3] and positive extent")
    if not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    return np.all(
        (values >= lower[None, :] - tolerance_m)
        & (values <= upper[None, :] + tolerance_m),
        axis=1,
    )


def ray_aabb_depth(directions, lower, upper) -> np.ndarray:
    """Return the positive exit distance from an origin-containing AABB."""

    rays = np.asarray(directions, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if rays.ndim < 2 or rays.shape[-1] != 3:
        raise ValueError("directions must end in an xyz axis")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= 0) or np.any(upper <= 0):
        raise ValueError("AABB must strictly contain the ray origin")
    norms = np.linalg.norm(rays, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("directions must be unit length")
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(rays > 0, upper / rays, np.where(rays < 0, lower / rays, np.inf))
    result = exits.min(axis=-1)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError("ray/AABB intersection produced an invalid exit distance")
    return result


def depth_reprojection_audit(
    depth_map,
    lower,
    upper,
    *,
    minimum_valid_depth_m: float = 0.05,
) -> dict:
    """Compare the AABB ray depth with the panorama that determined it."""

    depth = np.asarray(depth_map, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > float(minimum_valid_depth_m))
    predicted = ray_aabb_depth(equirectangular_directions(*depth.shape), lower, upper)
    signed = predicted[valid] - depth[valid]
    absolute = np.abs(signed)
    relative = absolute / depth[valid]
    return {
        "evaluated_ray_count": int(valid.sum()),
        "median_absolute_error_m": float(np.median(absolute)),
        "p95_absolute_error_m": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error_m": float(absolute.max()),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "fraction_aabb_behind_observation": float(np.mean(signed >= -1e-9)),
    }


def structured_aabb_tetrahedral_mesh(
    lower,
    upper,
    *,
    maximum_edge_m: float = 0.22,
    edge_safety_factor: float = 0.95,
) -> tuple[TetrahedralMesh, dict]:
    """Fill an AABB with a conforming six-tetrahedra-per-cell lattice."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    maximum_edge_m = float(maximum_edge_m)
    edge_safety_factor = float(edge_safety_factor)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("AABB bounds must have shape [3] and positive extent")
    if not math.isfinite(maximum_edge_m) or maximum_edge_m <= 0:
        raise ValueError("maximum_edge_m must be finite and positive")
    if not math.isfinite(edge_safety_factor) or not 0 < edge_safety_factor < 1:
        raise ValueError("edge_safety_factor must lie in (0, 1)")

    # The longest edge in the Freudenthal split is the cell body diagonal.
    target_axis_step = maximum_edge_m * edge_safety_factor / math.sqrt(3.0)
    cell_counts = np.ceil((upper - lower) / target_axis_step).astype(np.int64)
    axes = [
        np.linspace(lower[axis], upper[axis], int(cell_counts[axis]) + 1)
        for axis in range(3)
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    nodes = np.stack(grid, axis=-1).reshape(-1, 3)
    nx, ny, nz = (len(axis) for axis in axes)
    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1, dtype=np.int64),
        np.arange(ny - 1, dtype=np.int64),
        np.arange(nz - 1, dtype=np.int64),
        indexing="ij",
    )

    def index(di: int, dj: int, dk: int) -> np.ndarray:
        return (((ii + di) * ny + (jj + dj)) * nz + (kk + dk)).reshape(-1)

    v000 = index(0, 0, 0)
    v100 = index(1, 0, 0)
    v010 = index(0, 1, 0)
    v110 = index(1, 1, 0)
    v001 = index(0, 0, 1)
    v101 = index(1, 0, 1)
    v011 = index(0, 1, 1)
    v111 = index(1, 1, 1)
    elements = np.concatenate(
        (
            np.stack((v000, v100, v110, v111), axis=1),
            np.stack((v000, v110, v010, v111), axis=1),
            np.stack((v000, v010, v011, v111), axis=1),
            np.stack((v000, v011, v001, v111), axis=1),
            np.stack((v000, v001, v101, v111), axis=1),
            np.stack((v000, v101, v100, v111), axis=1),
        ),
        axis=0,
    )
    mesh = TetrahedralMesh(nodes=nodes, elements=elements)
    axis_steps = (upper - lower) / cell_counts
    observed_maximum_edge = float(np.linalg.norm(axis_steps))
    if observed_maximum_edge > maximum_edge_m + 1e-12:
        raise RuntimeError("structured depth AABB mesh exceeded its edge-length gate")
    return mesh, {
        "cell_counts": cell_counts.tolist(),
        "axis_steps_m": axis_steps.tolist(),
        "node_count": int(len(nodes)),
        "element_count": int(len(elements)),
        "maximum_element_edge_m": observed_maximum_edge,
        "maximum_allowed_edge_m": maximum_edge_m,
        "edge_safety_factor": edge_safety_factor,
    }
