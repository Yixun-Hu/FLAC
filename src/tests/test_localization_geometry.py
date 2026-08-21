from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src.localization.geometry import (
    SURFACE_CLEARANCE_METERS,
    build_lattice,
    choose_z_band_branch,
    classify_free_space,
    classify_mesh_candidates,
    filter_query_candidates,
    grid_oracle_error,
    load_raycast_scene,
    snap_axis_to_lattice,
)


def _box_mesh(tmp_path: Path) -> Path:
    path = tmp_path / "box.obj"
    mesh = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 2.0)
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=True)
    return path


def _room_with_obstacle_mesh(tmp_path: Path) -> Path:
    path = tmp_path / "room_with_obstacle.obj"
    room_shell = o3d.geometry.TriangleMesh.create_box(4.0, 4.0, 4.0)
    obstacle = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).translate((1.5, 1.5, 1.5))
    assert o3d.io.write_triangle_mesh(str(path), room_shell + obstacle, write_ascii=True)
    return path


@pytest.mark.parametrize("spacing", [0, -0.5, np.nan, np.inf])
def test_spacing_rejected(spacing):
    with pytest.raises(ValueError):
        snap_axis_to_lattice((-1, 1), spacing)


def test_negative_bounds_and_lexicographic_lattice():
    axis = snap_axis_to_lattice((-1.1, 0.9), 0.5)
    assert np.array_equal(axis, np.array([-1.0, -0.5, 0.0, 0.5]))
    points = build_lattice((-0.6, -0.1, 0.1), (0.6, 0.9, 0.6), 0.5)
    expected = np.array(
        [
            [-0.5, 0.0, 0.5],
            [-0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    assert np.array_equal(points, expected)


def test_box_occupancy_surface_clearance_and_chunk_identity(tmp_path):
    mesh = load_raycast_scene(_box_mesh(tmp_path))
    assert SURFACE_CLEARANCE_METERS == 0.2
    points = build_lattice(mesh.aabb_min, mesh.aabb_max, 0.5)
    mask_a, distance_a = classify_mesh_candidates(
        mesh, points, SURFACE_CLEARANCE_METERS, chunk_size=7
    )
    mask_b, distance_b = classify_mesh_candidates(
        mesh, points, SURFACE_CLEARANCE_METERS, chunk_size=1000
    )
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(distance_a, distance_b)
    assert mask_a.sum() == 27
    assert np.all(distance_a[mask_a] + 1e-4 >= SURFACE_CLEARANCE_METERS)


def test_ray_parity_majority_separates_room_air_obstacle_and_outside(tmp_path):
    mesh = load_raycast_scene(_room_with_obstacle_mesh(tmp_path))
    points = np.array([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [5.0, 2.0, 2.0]])
    mask_a, votes_a = classify_free_space(mesh, points, chunk_size=1)
    mask_b, votes_b = classify_free_space(mesh, points, chunk_size=100)
    assert np.array_equal(mask_a, np.array([True, False, False]))
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(votes_a, votes_b)


def test_surface_clearance_prior_uses_0_20m_with_eps(tmp_path):
    mesh = load_raycast_scene(_box_mesh(tmp_path))
    points = np.array([[0.199, 1.0, 1.0], [0.2, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mask, distance = classify_mesh_candidates(mesh, points, SURFACE_CLEARANCE_METERS)
    assert np.array_equal(mask, np.array([False, True, True]))
    assert distance.tolist() == pytest.approx([0.199, 0.2, 1.0], abs=1e-6)


def test_receiver_context_and_z_boundaries():
    points = np.array(
        [
            [0.0, 0.0, 0.5],   # receiver: reject
            [0.5, 0.0, 0.5],   # receiver boundary: keep
            [1.0, 0.0, 0.5],   # context: reject
            [1.25, 0.0, 0.5],  # context boundary: keep
            [2.0, 0.0, 1.5],   # z upper boundary: keep
            [2.0, 0.0, 2.0],   # outside z: reject
        ]
    )
    mask = filter_query_candidates(
        points,
        receiver=np.array([0.0, 0.0, 0.5]),
        context_sources=np.array([[1.0, 0.0, 0.5]]),
        receiver_clearance=0.5,
        context_clearance=0.25,
        z_band=(0.5, 1.5),
    )
    assert np.array_equal(mask, np.array([False, True, False, True, True, False]))


def test_oracle_and_global_z_branch_gate():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert grid_oracle_error(points, np.array([0.2, 0.0, 0.0])) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="nonempty"):
        grid_oracle_error(np.empty((0, 3)), np.zeros(3))
    assert choose_z_band_branch([0.2, 0.4], [0.3, 0.4], [3, 2]) == "z_band"
    assert choose_z_band_branch([0.2, 0.4], [0.6, 0.4], [3, 2]) == "full_height"
    assert choose_z_band_branch([0.2, 0.4], [0.3, np.inf], [3, 0]) == "full_height"


def test_missing_and_malformed_mesh_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raycast_scene(tmp_path / "missing.obj")
    malformed = tmp_path / "bad.obj"
    malformed.write_text("not an obj\n")
    with pytest.raises(ValueError):
        load_raycast_scene(malformed)
