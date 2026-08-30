"""LGT-inspired structural layout recovery from a metric depth panorama.

The original LGT-Net predicts a horizon-depth signal and room height from RGB.
For the matched-observation FEM baseline we already have metric radial depth, so
this module estimates the same representation deterministically: horizontal
floor/ceiling planes plus one wall radius per panorama column.  A closed
piecewise-linear polar footprint is then voxelized into a conforming tetrahedral
air mesh.  No official room mesh or learned weights are read here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .depth_aabb import equirectangular_directions
from .fem_solver import (
    TetrahedralMesh,
    barycentric_interpolation_matrix,
    build_barycentric_point_locator,
)


@dataclass(frozen=True)
class DepthPolarLayout:
    """Receiver-local star-convex room layout in metric coordinates."""

    theta_rad: np.ndarray
    wall_radius_m: np.ndarray
    floor_z_m: float
    ceiling_z_m: float
    polygon_vertices_xy_m: np.ndarray

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta_rad, dtype=np.float64)
        radius = np.asarray(self.wall_radius_m, dtype=np.float64)
        vertices = np.asarray(self.polygon_vertices_xy_m, dtype=np.float64)
        if theta.ndim != 1 or len(theta) < 4 or radius.shape != theta.shape:
            raise ValueError("layout theta and wall radius must be matching vectors")
        if not np.isfinite(theta).all() or not np.isfinite(radius).all():
            raise ValueError("layout polar profile must be finite")
        if np.any(np.diff(theta) <= 0.0) or np.any(radius <= 0.0):
            raise ValueError("layout angles must increase and radii must be positive")
        if (
            not math.isfinite(self.floor_z_m)
            or not math.isfinite(self.ceiling_z_m)
            or self.floor_z_m >= 0.0
            or self.ceiling_z_m <= 0.0
        ):
            raise ValueError("layout must vertically contain the receiver origin")
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("layout polygon must contain at least three xy vertices")
        if not np.isfinite(vertices).all():
            raise ValueError("layout polygon vertices must be finite")
        object.__setattr__(self, "theta_rad", theta)
        object.__setattr__(self, "wall_radius_m", radius)
        object.__setattr__(self, "polygon_vertices_xy_m", vertices)


def _rdp_open(points: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for one open polyline."""

    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length <= np.finfo(np.float64).eps:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        offsets = points - start
        distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / length
    split = int(np.argmax(distances))
    if float(distances[split]) <= tolerance_m:
        return points[[0, -1]]
    first = _rdp_open(points[: split + 1], tolerance_m)
    second = _rdp_open(points[split:], tolerance_m)
    return np.concatenate((first[:-1], second), axis=0)


def simplify_closed_polygon(points, *, tolerance_m: float) -> np.ndarray:
    """Simplify an ordered closed boundary without privileging its seam."""

    values = np.asarray(points, dtype=np.float64)
    tolerance_m = float(tolerance_m)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 4:
        raise ValueError("closed polygon samples must have shape [N, 2], N >= 4")
    if not np.isfinite(values).all() or not math.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("polygon samples and tolerance must be finite")
    first = 0
    second = int(np.argmax(np.linalg.norm(values - values[first], axis=1)))
    first = int(np.argmax(np.linalg.norm(values - values[second], axis=1)))
    second = int(np.argmax(np.linalg.norm(values - values[first], axis=1)))
    if first > second:
        first, second = second, first
    forward = _rdp_open(values[first : second + 1], tolerance_m)
    wrapped = np.concatenate((values[second:], values[: first + 1]), axis=0)
    backward = _rdp_open(wrapped, tolerance_m)
    simplified = np.concatenate((forward[:-1], backward[:-1]), axis=0)
    if len(simplified) < 3:
        raise RuntimeError("layout simplification collapsed the room polygon")
    return simplified


def ray_polygon_radius(polygon_vertices, theta_rad) -> np.ndarray:
    """Return the first positive intersection of origin rays with a polygon."""

    vertices = np.asarray(polygon_vertices, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("polygon vertices must have shape [V, 2], V >= 3")
    if theta.ndim != 1 or not np.isfinite(vertices).all() or not np.isfinite(theta).all():
        raise ValueError("polygon vertices and ray angles must be finite vectors")
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    edges = np.roll(vertices, -1, axis=0) - vertices

    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    denominator = cross(directions[:, None, :], edges[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = cross(vertices[None, :, :], edges[None, :, :]) / denominator
        fraction = cross(vertices[None, :, :], directions[:, None, :]) / denominator
    valid = (
        (np.abs(denominator) > 1e-12)
        & (distance > 0.0)
        & (fraction >= -1e-9)
        & (fraction <= 1.0 + 1e-9)
    )
    radius = np.where(valid, distance, np.inf).min(axis=1)
    if not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("simplified footprint is not star-convex about the receiver")
    return radius


def _periodic_interpolate(theta, sample_theta, sample_values) -> np.ndarray:
    angles = np.asarray(theta, dtype=np.float64)
    samples = np.asarray(sample_theta, dtype=np.float64)
    values = np.asarray(sample_values, dtype=np.float64)
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi
    extended_theta = np.concatenate(
        ((samples[-1:] - 2.0 * math.pi), samples, (samples[:1] + 2.0 * math.pi))
    )
    extended_values = np.concatenate((values[-1:], values, values[:1]))
    return np.interp(wrapped, extended_theta, extended_values)


def depth_panorama_polar_layout(
    depth_map,
    *,
    wall_quantile: float = 0.95,
    cap_fraction: float = 0.125,
    vertical_wall_margin_m: float = 0.05,
    circular_median_columns: int = 5,
    simplification_tolerance_m: float = 0.03,
    padding_m: float = 0.05,
    minimum_valid_depth_m: float = 0.05,
) -> tuple[DepthPolarLayout, dict]:
    """Recover a padded LGT-style horizon-depth layout from radial depth."""

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) < 8:
        raise ValueError("depth panorama must be a two-dimensional image")
    scalar_values = (
        wall_quantile,
        cap_fraction,
        vertical_wall_margin_m,
        simplification_tolerance_m,
        padding_m,
        minimum_valid_depth_m,
    )
    if any(not math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("layout recovery parameters must be finite")
    if not 0.5 <= wall_quantile <= 1.0:
        raise ValueError("wall quantile must lie in [0.5, 1]")
    if not 0.0 < cap_fraction <= 0.25:
        raise ValueError("cap fraction must lie in (0, 0.25]")
    if vertical_wall_margin_m < 0.0 or simplification_tolerance_m < 0.0 or padding_m < 0.0:
        raise ValueError("metric layout tolerances must be nonnegative")
    if minimum_valid_depth_m <= 0.0:
        raise ValueError("minimum valid depth must be positive")
    if (
        not isinstance(circular_median_columns, int)
        or isinstance(circular_median_columns, bool)
        or circular_median_columns <= 0
        or circular_median_columns % 2 == 0
    ):
        raise ValueError("circular median width must be a positive odd integer")

    valid = np.isfinite(depth) & (depth > minimum_valid_depth_m)
    if int(valid.sum()) < 32:
        raise ValueError("depth panorama contains too few valid samples")
    directions = equirectangular_directions(*depth.shape)
    points = directions * np.where(valid, depth, 0.0)[..., None]
    horizontal_radius = np.linalg.norm(points[..., :2], axis=-1)
    height = points[..., 2]
    cap_rows = max(4, int(round(depth.shape[0] * cap_fraction)))
    ceiling_samples = height[:cap_rows][valid[:cap_rows]]
    floor_samples = height[-cap_rows:][valid[-cap_rows:]]
    ceiling_unpadded = float(np.median(ceiling_samples))
    floor_unpadded = float(np.median(floor_samples))
    if floor_unpadded >= 0.0 or ceiling_unpadded <= 0.0:
        raise ValueError("depth cap rows do not establish floor and ceiling around receiver")

    vertical_margin = min(
        float(vertical_wall_margin_m),
        0.2 * (ceiling_unpadded - floor_unpadded),
    )
    wall_support = (
        valid
        & (height > floor_unpadded + vertical_margin)
        & (height < ceiling_unpadded - vertical_margin)
    )
    supported_columns = wall_support.sum(axis=0) > 0
    if int(supported_columns.sum()) < depth.shape[1] // 2:
        raise ValueError("too few panorama columns contain vertical wall support")
    masked_radius = np.where(wall_support, horizontal_radius, np.nan)
    with np.errstate(invalid="ignore"):
        raw_radius = np.nanquantile(masked_radius, wall_quantile, axis=0)
    column_indices = np.arange(depth.shape[1], dtype=np.float64)
    finite_columns = np.isfinite(raw_radius) & (raw_radius > minimum_valid_depth_m)
    if not finite_columns.all():
        good = column_indices[finite_columns]
        extended_good = np.concatenate((good - depth.shape[1], good, good + depth.shape[1]))
        extended_values = np.tile(raw_radius[finite_columns], 3)
        raw_radius = np.interp(column_indices, extended_good, extended_values)
    filtered_radius = ndimage.median_filter(
        raw_radius, size=circular_median_columns, mode="wrap"
    )
    theta = (
        (np.arange(depth.shape[1], dtype=np.float64) + 0.5)
        * (2.0 * math.pi / depth.shape[1])
        - math.pi
    )
    dense_polygon = np.stack(
        (filtered_radius * np.cos(theta), filtered_radius * np.sin(theta)), axis=1
    )
    simplified_polygon = simplify_closed_polygon(
        dense_polygon, tolerance_m=simplification_tolerance_m
    )
    simplified_radius = ray_polygon_radius(simplified_polygon, theta)
    padded_radius = simplified_radius + padding_m
    padded_polygon = np.stack(
        (padded_radius * np.cos(theta), padded_radius * np.sin(theta)), axis=1
    )
    final_polygon = simplify_closed_polygon(
        padded_polygon, tolerance_m=simplification_tolerance_m
    )
    final_radius = ray_polygon_radius(final_polygon, theta)
    floor = floor_unpadded - padding_m
    ceiling = ceiling_unpadded + padding_m
    layout = DepthPolarLayout(
        theta_rad=theta,
        wall_radius_m=final_radius,
        floor_z_m=floor,
        ceiling_z_m=ceiling,
        polygon_vertices_xy_m=final_polygon,
    )
    area = 0.5 * abs(
        float(
            np.dot(final_polygon[:, 0], np.roll(final_polygon[:, 1], -1))
            - np.dot(final_polygon[:, 1], np.roll(final_polygon[:, 0], -1))
        )
    )
    simplification_error = np.abs(simplified_radius - filtered_radius)
    audit = {
        "representation": "metric horizon-depth plus floor/ceiling; LGT-inspired, no learned network",
        "valid_depth_count": int(valid.sum()),
        "invalid_depth_count": int(valid.size - valid.sum()),
        "valid_depth_fraction": float(valid.mean()),
        "cap_rows": cap_rows,
        "floor_z_unpadded_m": floor_unpadded,
        "ceiling_z_unpadded_m": ceiling_unpadded,
        "floor_z_m": floor,
        "ceiling_z_m": ceiling,
        "height_m": ceiling - floor,
        "wall_quantile": float(wall_quantile),
        "vertical_wall_margin_m": vertical_margin,
        "circular_median_columns": circular_median_columns,
        "supported_column_count": int(supported_columns.sum()),
        "interpolated_column_count": int((~supported_columns).sum()),
        "raw_wall_radius_range_m": [float(raw_radius.min()), float(raw_radius.max())],
        "wall_radius_range_m": [float(final_radius.min()), float(final_radius.max())],
        "simplification_tolerance_m": float(simplification_tolerance_m),
        "corner_count_before_padding": int(len(simplified_polygon)),
        "corner_count_after_padding": int(len(final_polygon)),
        "simplification_median_absolute_error_m": float(np.median(simplification_error)),
        "simplification_p95_absolute_error_m": float(np.quantile(simplification_error, 0.95)),
        "simplification_maximum_absolute_error_m": float(simplification_error.max()),
        "padding_m": float(padding_m),
        "footprint_area_m2": area,
        "extruded_volume_m3": area * (ceiling - floor),
    }
    return layout, audit


def ray_polar_layout_depth(layout: DepthPolarLayout, directions) -> np.ndarray:
    """Intersect receiver-origin rays with the extruded polar layout."""

    rays = np.asarray(directions, dtype=np.float64)
    if rays.ndim < 2 or rays.shape[-1] != 3:
        raise ValueError("directions must end in an xyz axis")
    norms = np.linalg.norm(rays, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("directions must be unit length")
    theta = np.arctan2(rays[..., 1], rays[..., 0])
    radius = _periodic_interpolate(theta, layout.theta_rad, layout.wall_radius_m)
    horizontal_norm = np.linalg.norm(rays[..., :2], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        wall_distance = np.where(horizontal_norm > 0.0, radius / horizontal_norm, np.inf)
        cap_distance = np.where(
            rays[..., 2] > 0.0,
            layout.ceiling_z_m / rays[..., 2],
            np.where(rays[..., 2] < 0.0, layout.floor_z_m / rays[..., 2], np.inf),
        )
    result = np.minimum(wall_distance, cap_distance)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("polar layout ray intersection produced invalid depth")
    return result


def depth_polar_reprojection_audit(
    depth_map,
    layout: DepthPolarLayout,
    *,
    minimum_valid_depth_m: float = 0.05,
) -> dict:
    """Compare recovered layout ray depth with the determining panorama."""

    depth = np.asarray(depth_map, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > float(minimum_valid_depth_m))
    predicted = ray_polar_layout_depth(
        layout, equirectangular_directions(*depth.shape)
    )
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
        "fraction_layout_behind_observation": float(np.mean(signed >= -1e-9)),
    }


def points_in_polar_layout(
    points,
    layout: DepthPolarLayout,
    *,
    tolerance_m: float = 1e-9,
) -> np.ndarray:
    """Classify receiver-local points against the continuous layout prism."""

    values = np.asarray(points, dtype=np.float64)
    tolerance_m = float(tolerance_m)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be finite with shape [N, 3]")
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("point tolerance must be finite and nonnegative")
    theta = np.arctan2(values[:, 1], values[:, 0])
    allowed_radius = _periodic_interpolate(
        theta, layout.theta_rad, layout.wall_radius_m
    )
    radius = np.linalg.norm(values[:, :2], axis=1)
    return (
        (radius <= allowed_radius + tolerance_m)
        & (values[:, 2] >= layout.floor_z_m - tolerance_m)
        & (values[:, 2] <= layout.ceiling_z_m + tolerance_m)
    )


def required_horizontal_completion_distance(points, layout: DepthPolarLayout) -> np.ndarray:
    """Return per-point radial growth needed to enter a polar footprint."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be finite with shape [N, 3]")
    theta = np.arctan2(values[:, 1], values[:, 0])
    allowed_radius = _periodic_interpolate(
        theta, layout.theta_rad, layout.wall_radius_m
    )
    return np.maximum(0.0, np.linalg.norm(values[:, :2], axis=1) - allowed_radius)


def complete_polar_layout_toward_depth_aabb(
    layout: DepthPolarLayout,
    aabb_lower,
    aabb_upper,
    *,
    completion_distance_m: float,
) -> tuple[DepthPolarLayout, dict]:
    """Conservatively complete hidden floor area toward the depth AABB.

    The operation is receiver-view-only and candidate independent.  Every
    horizontal layout ray may grow by at most one globally calibrated metric
    distance, and never beyond the exit of the AABB recovered from the same
    depth panorama.  A distance of zero reproduces the polar layout; infinity
    reaches its depth-AABB envelope.  The vertical extent is the union of the
    polar cap estimate and the same-view AABB because this completion targets
    horizontal single-view occlusion rather than floor/ceiling estimation.
    """

    lower = np.asarray(aabb_lower, dtype=np.float64)
    upper = np.asarray(aabb_upper, dtype=np.float64)
    distance = float(completion_distance_m)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("AABB bounds must be finite ordered xyz vectors")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("AABB bounds must be finite ordered xyz vectors")
    if np.any(lower >= 0.0) or np.any(upper <= 0.0):
        raise ValueError("receiver-local AABB must strictly contain the origin")
    if math.isnan(distance) or distance < 0.0:
        raise ValueError("completion distance must be nonnegative")

    directions = np.stack(
        (np.cos(layout.theta_rad), np.sin(layout.theta_rad)), axis=1
    )
    lower_xy = lower[:2]
    upper_xy = upper[:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(
            directions > 0.0,
            upper_xy[None, :] / directions,
            np.where(directions < 0.0, lower_xy[None, :] / directions, np.inf),
        )
    aabb_radius = exits.min(axis=1)
    if not np.isfinite(aabb_radius).all() or np.any(aabb_radius <= 0.0):
        raise RuntimeError("horizontal ray/AABB intersection is invalid")

    if math.isinf(distance):
        proposed = aabb_radius
    else:
        proposed = np.minimum(aabb_radius, layout.wall_radius_m + distance)
    completed_radius = np.maximum(layout.wall_radius_m, proposed)
    polygon = np.stack(
        (
            completed_radius * np.cos(layout.theta_rad),
            completed_radius * np.sin(layout.theta_rad),
        ),
        axis=1,
    )
    completed = DepthPolarLayout(
        theta_rad=layout.theta_rad,
        wall_radius_m=completed_radius,
        floor_z_m=min(layout.floor_z_m, float(lower[2])),
        ceiling_z_m=max(layout.ceiling_z_m, float(upper[2])),
        polygon_vertices_xy_m=polygon,
    )
    original_polygon = np.stack(
        (
            layout.wall_radius_m * np.cos(layout.theta_rad),
            layout.wall_radius_m * np.sin(layout.theta_rad),
        ),
        axis=1,
    )
    original_area = 0.5 * abs(
        float(
            np.dot(original_polygon[:, 0], np.roll(original_polygon[:, 1], -1))
            - np.dot(original_polygon[:, 1], np.roll(original_polygon[:, 0], -1))
        )
    )
    completed_area = 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )
    expansion = completed_radius - layout.wall_radius_m
    return completed, {
        "method": "bounded radial completion toward same-view depth AABB",
        "candidate_independent": True,
        "completion_distance_m": distance,
        "expanded_ray_count": int(np.count_nonzero(expansion > 1e-12)),
        "ray_count": int(len(expansion)),
        "median_radial_expansion_m": float(np.median(expansion)),
        "p95_radial_expansion_m": float(np.quantile(expansion, 0.95)),
        "maximum_radial_expansion_m": float(expansion.max()),
        "original_footprint_area_m2": original_area,
        "completed_footprint_area_m2": completed_area,
        "area_ratio": completed_area / original_area,
        "floor_z_m": completed.floor_z_m,
        "ceiling_z_m": completed.ceiling_z_m,
    }


def structured_polar_tetrahedral_mesh(
    layout: DepthPolarLayout,
    receiver_global,
    *,
    maximum_edge_m: float = 0.22,
    edge_safety_factor: float = 0.95,
) -> tuple[TetrahedralMesh, dict]:
    """Voxelize a non-convex polar prism into conforming tetrahedra."""

    receiver = np.asarray(receiver_global, dtype=np.float64)
    maximum_edge_m = float(maximum_edge_m)
    edge_safety_factor = float(edge_safety_factor)
    if receiver.shape != (3,) or not np.isfinite(receiver).all():
        raise ValueError("receiver global coordinate must be a finite xyz vector")
    if not math.isfinite(maximum_edge_m) or maximum_edge_m <= 0.0:
        raise ValueError("maximum edge must be finite and positive")
    if not math.isfinite(edge_safety_factor) or not 0.0 < edge_safety_factor < 1.0:
        raise ValueError("edge safety factor must lie in (0, 1)")

    boundary = np.stack(
        (
            layout.wall_radius_m * np.cos(layout.theta_rad),
            layout.wall_radius_m * np.sin(layout.theta_rad),
        ),
        axis=1,
    )
    lower_local = np.array(
        [boundary[:, 0].min(), boundary[:, 1].min(), layout.floor_z_m],
        dtype=np.float64,
    )
    upper_local = np.array(
        [boundary[:, 0].max(), boundary[:, 1].max(), layout.ceiling_z_m],
        dtype=np.float64,
    )
    target_axis_step = maximum_edge_m * edge_safety_factor / math.sqrt(3.0)
    cell_counts = np.ceil((upper_local - lower_local) / target_axis_step).astype(np.int64)
    local_axes = [
        np.linspace(lower_local[axis], upper_local[axis], int(cell_counts[axis]) + 1)
        for axis in range(3)
    ]
    center_x = (local_axes[0][:-1] + local_axes[0][1:]) / 2.0
    center_y = (local_axes[1][:-1] + local_axes[1][1:]) / 2.0
    xx, yy = np.meshgrid(center_x, center_y, indexing="ij")
    center_theta = np.arctan2(yy, xx)
    center_radius = np.hypot(xx, yy)
    allowed_radius = _periodic_interpolate(
        center_theta, layout.theta_rad, layout.wall_radius_m
    )
    footprint_cells = center_radius <= allowed_radius
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    labels, component_count = ndimage.label(footprint_cells, structure=structure)
    receiver_i = int(np.clip(np.searchsorted(local_axes[0], 0.0) - 1, 0, len(center_x) - 1))
    receiver_j = int(np.clip(np.searchsorted(local_axes[1], 0.0) - 1, 0, len(center_y) - 1))
    selected_label = int(labels[receiver_i, receiver_j])
    if selected_label == 0:
        raise RuntimeError("voxelized layout does not contain the receiver cell")
    selected_footprint = labels == selected_label
    removed_footprint_cells = int(footprint_cells.sum() - selected_footprint.sum())

    global_axes = [axis + receiver[index] for index, axis in enumerate(local_axes)]
    grid = np.meshgrid(*global_axes, indexing="ij")
    nodes = np.stack(grid, axis=-1).reshape(-1, 3)
    nx, ny, nz = (len(axis) for axis in global_axes)
    cell_i, cell_j = np.nonzero(selected_footprint)
    cell_k = np.arange(nz - 1, dtype=np.int64)
    ii = np.repeat(cell_i, len(cell_k))
    jj = np.repeat(cell_j, len(cell_k))
    kk = np.tile(cell_k, len(cell_i))

    def index(di: int, dj: int, dk: int) -> np.ndarray:
        return ((ii + di) * ny + (jj + dj)) * nz + (kk + dk)

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
    used_nodes = np.unique(elements)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
    mesh = TetrahedralMesh(nodes=nodes[used_nodes], elements=remap[elements])
    axis_steps = (upper_local - lower_local) / cell_counts
    observed_maximum_edge = float(np.linalg.norm(axis_steps))
    if observed_maximum_edge > maximum_edge_m + 1e-12:
        raise RuntimeError("structured polar mesh exceeded its edge-length gate")
    return mesh, {
        "meshing_method": "Cartesian voxelization of depth-derived polar prism",
        "cell_counts_aabb": cell_counts.tolist(),
        "axis_steps_m": axis_steps.tolist(),
        "footprint_cell_count": int(selected_footprint.sum()),
        "footprint_component_count": int(component_count),
        "selected_footprint_component": selected_label,
        "removed_footprint_cell_count": removed_footprint_cells,
        "node_count": int(len(mesh.nodes)),
        "element_count": int(len(mesh.elements)),
        "maximum_element_edge_m": observed_maximum_edge,
        "maximum_allowed_edge_m": maximum_edge_m,
        "edge_safety_factor": edge_safety_factor,
        "voxelized_volume_m3": float(
            selected_footprint.sum() * (nz - 1) * np.prod(axis_steps)
        ),
    }


def points_in_tetrahedral_mesh(
    mesh: TetrahedralMesh,
    points,
    *,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Audit point inclusion one-by-one without aborting at the first miss."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be finite with shape [N, 3]")
    locator = build_barycentric_point_locator(mesh)
    included = np.zeros(len(values), dtype=bool)
    for index, point in enumerate(values):
        try:
            barycentric_interpolation_matrix(
                mesh, point[None, :], tolerance=tolerance, locator=locator
            )
        except ValueError as error:
            if "lies outside the tetrahedral mesh" not in str(error):
                raise
        else:
            included[index] = True
    return included
