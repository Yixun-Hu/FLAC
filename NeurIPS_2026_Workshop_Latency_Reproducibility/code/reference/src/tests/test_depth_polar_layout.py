import numpy as np

from src.baselines.depth_aabb import (
    equirectangular_directions,
    ray_aabb_depth,
)
from src.baselines.depth_polar_layout import (
    DepthPolarLayout,
    complete_polar_layout_toward_depth_aabb,
    depth_panorama_polar_layout,
    depth_polar_reprojection_audit,
    points_in_polar_layout,
    points_in_tetrahedral_mesh,
    structured_polar_tetrahedral_mesh,
)


def _box_depth(height=128, width=256):
    lower = np.array([-1.0, -1.5, -0.8])
    upper = np.array([2.0, 1.2, 1.4])
    directions = equirectangular_directions(height, width)
    return ray_aabb_depth(directions, lower, upper), lower, upper


def test_depth_polar_layout_recovers_metric_box_planes_and_wall_profile():
    depth, lower, upper = _box_depth()
    layout, audit = depth_panorama_polar_layout(
        depth,
        padding_m=0.02,
        simplification_tolerance_m=0.01,
    )
    assert abs(layout.floor_z_m - (lower[2] - 0.02)) < 1e-3
    assert abs(layout.ceiling_z_m - (upper[2] + 0.02)) < 1e-3
    assert audit["corner_count_before_padding"] >= 4
    assert audit["footprint_area_m2"] > 0.0
    reprojection = depth_polar_reprojection_audit(depth, layout)
    assert reprojection["median_absolute_error_m"] < 0.03
    assert reprojection["fraction_layout_behind_observation"] > 0.99


def test_polar_layout_point_and_tetrahedral_mesh_audits():
    depth, _lower, _upper = _box_depth(64, 128)
    layout, _audit = depth_panorama_polar_layout(
        depth,
        padding_m=0.03,
        simplification_tolerance_m=0.01,
    )
    local_points = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.2], [3.0, 0.0, 0.0]])
    assert points_in_polar_layout(local_points, layout).tolist() == [True, True, False]
    receiver = np.array([4.0, 5.0, 1.0])
    mesh, mesh_audit = structured_polar_tetrahedral_mesh(
        layout, receiver, maximum_edge_m=0.22
    )
    assert mesh_audit["maximum_element_edge_m"] <= 0.22
    coordinates = mesh.nodes[mesh.elements]
    volume = np.abs(
        np.linalg.det(
            np.transpose(coordinates[:, 1:] - coordinates[:, :1], (0, 2, 1))
        )
    ).sum() / 6.0
    assert np.isclose(mesh_audit["voxelized_volume_m3"], volume)
    global_points = local_points + receiver
    assert points_in_tetrahedral_mesh(mesh, global_points).tolist() == [True, True, False]


def test_bounded_completion_is_monotone_and_candidate_independent():
    theta = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    radius = np.ones_like(theta)
    polygon = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    layout = DepthPolarLayout(theta, radius, -1.0, 1.0, polygon)
    lower = np.array([-2.0, -3.0, -1.2])
    upper = np.array([4.0, 5.0, 1.4])

    unchanged, audit_zero = complete_polar_layout_toward_depth_aabb(
        layout, lower, upper, completion_distance_m=0.0
    )
    completed, audit = complete_polar_layout_toward_depth_aabb(
        layout, lower, upper, completion_distance_m=0.5
    )

    np.testing.assert_allclose(unchanged.wall_radius_m, radius)
    assert np.all(completed.wall_radius_m >= unchanged.wall_radius_m)
    assert np.all(completed.wall_radius_m <= radius + 0.5 + 1e-12)
    assert completed.floor_z_m == -1.2
    assert completed.ceiling_z_m == 1.4
    assert audit_zero["expanded_ray_count"] == 0
    assert audit["candidate_independent"] is True
    assert audit["area_ratio"] > 1.0
