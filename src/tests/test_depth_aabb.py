import numpy as np

from src.baselines.depth_aabb import (
    depth_panorama_aabb,
    depth_reprojection_audit,
    equirectangular_directions,
    points_in_aabb,
    ray_aabb_depth,
    structured_aabb_tetrahedral_mesh,
)


def test_equirectangular_directions_are_unit_length():
    directions = equirectangular_directions(16, 32)
    assert directions.shape == (16, 32, 3)
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)


def test_depth_aabb_contains_every_radial_endpoint_and_reprojects_behind_it():
    depth = np.full((16, 32), 2.0)
    lower, upper, audit = depth_panorama_aabb(depth, padding_m=0.05)
    points = equirectangular_directions(*depth.shape).reshape(-1, 3) * 2.0
    assert points_in_aabb(points, lower, upper).all()
    exits = ray_aabb_depth(equirectangular_directions(*depth.shape), lower, upper)
    assert np.all(exits >= depth - 1e-9)
    reprojection = depth_reprojection_audit(depth, lower, upper)
    assert audit["valid_depth_fraction"] == 1.0
    assert reprojection["fraction_aabb_behind_observation"] == 1.0


def test_structured_aabb_mesh_is_connected_and_respects_edge_gate():
    lower = np.array([-0.2, -0.15, -0.1])
    upper = np.array([0.2, 0.15, 0.1])
    mesh, audit = structured_aabb_tetrahedral_mesh(
        lower, upper, maximum_edge_m=0.22
    )
    assert len(mesh.nodes) == audit["node_count"]
    assert len(mesh.elements) == audit["element_count"]
    assert audit["maximum_element_edge_m"] <= 0.22
    assert points_in_aabb(mesh.nodes, lower, upper).all()
