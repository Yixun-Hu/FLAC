"""Tests for ``data/RAF/render_depth.py`` — equirect depth rendering for RAF.

exp_19 (RAF finetune), contract section C, TDD cycle 8.

The oracle is NON-CIRCULAR by construction (plan Rev 2 section 4): the expected
distances below are hand-derived from an asymmetric six-wall box and written as
literals. Nothing in the oracle calls ``equirect_directions``, ``render_depth`` or
the raycaster; the ray directions themselves are re-derived by hand from the
equirectangular convention.

Hand derivation of the ray grid at h=2, w=4:

    phi_i   = (i + 0.5) * pi/2 - pi/2   -> i=0: -pi/4 (up), i=1: +pi/4 (down)
    theta_j = (j + 0.5) * pi/2 - pi     -> -3pi/4, -pi/4, +pi/4, +3pi/4
    dir     = (cos(phi) cos(theta), cos(phi) sin(theta), -sin(phi))

    cos(+-pi/4) = sin(+-pi/4) = +-0.70710678, so every direction is
    (+-0.5, +-0.5, +-0.70710678):

        row 0 (up):   (-.5,-.5,+c) (+.5,-.5,+c) (+.5,+.5,+c) (-.5,+.5,+c)
        row 1 (down): (-.5,-.5,-c) (+.5,-.5,-c) (+.5,+.5,-c) (-.5,+.5,-c)

    with c = 0.70710678, 1/c = 1.41421356.
"""
import json
import os
import sys

import numpy as np
import pytest
import open3d as o3d

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import raf_common  # noqa: E402
import render_depth as raf_render  # noqa: E402

assert os.path.dirname(os.path.abspath(raf_render.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# hand-built fixtures (RAF world coordinates: X front, Y up, Z left)
# --------------------------------------------------------------------------- #
def _raf_to_pipeline_by_hand(points):
    """(X, Y, Z)_RAF -> (X, Z, Y)_pipeline, written out rather than imported."""
    pts = np.asarray(points, dtype=np.float64)
    return np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], axis=1)


def _box_mesh_raf(x0, x1, y0, y1, z0, z1, drop_ceiling=False, to_pipeline=True):
    """Axis-aligned box with 8 hand-written vertices and 12 (or 10) triangles."""
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],   # 0..3  Z = z0
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],   # 4..7  Z = z1
    ], dtype=np.float64)
    faces = [
        (0, 1, 2), (0, 2, 3),      # Z = z0
        (4, 6, 5), (4, 7, 6),      # Z = z1
        (0, 4, 5), (0, 5, 1),      # Y = y0 (floor, RAF up-axis minimum)
        (0, 3, 7), (0, 7, 4),      # X = x0
        (1, 5, 6), (1, 6, 2),      # X = x1
    ]
    ceiling = [(3, 2, 6), (3, 6, 7)]   # Y = y1
    if not drop_ceiling:
        faces = faces + ceiling
    verts = _raf_to_pipeline_by_hand(v) if to_pipeline else v
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    return mesh


def test_hand_written_permutation_agrees_with_the_registered_constant():
    """Pins the gauge constant against the permutation this file's oracle uses."""
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    np.testing.assert_array_equal(_raf_to_pipeline_by_hand(pts),
                                  pts @ raf_common.RAF_TO_PIPELINE.T)


# --------------------------------------------------------------------------- #
# analytic six-wall oracle
# --------------------------------------------------------------------------- #
# Box (RAF): X in [-0.2, 5.0], Y in [0.0, 4.0] (up), Z in [-2.0, 6.0] (left).
_BOX = dict(x0=-0.2, x1=5.0, y0=0.0, y1=4.0, z0=-2.0, z1=6.0)


def test_render_depth_matches_the_hand_derived_box_distances_camera_one():
    """Camera RAF (0.3, 1.5, 5.0) -> pipeline (0.3, 5.0, 1.5). Wall distances:

        x- : 0.3 - (-0.2) = 0.5   -> t = 0.5/0.5 = 1.0     (rays with x = -0.5)
        x+ : 5.0 - 0.3    = 4.7   -> t = 4.7/0.5 = 9.4
        y- : 5.0 - (-2.0) = 7.0   -> t = 7.0/0.5 = 14.0
        y+ : 6.0 - 5.0    = 1.0   -> t = 1.0/0.5 = 2.0
        z+ : 4.0 - 1.5    = 2.5   -> t = 2.5/0.70710678 = 3.53553391
        z- : 1.5 - 0.0    = 1.5   -> t = 1.5/0.70710678 = 2.12132034

    Per-ray minimum:
        row 0 (up):   min(1.0,14.0,3.536)=1.0   min(9.4,14.0,3.536)=3.53553391
                      min(9.4,2.0,3.536)=2.0    min(1.0,2.0,3.536)=1.0
        row 1 (down): min(1.0,14.0,2.121)=1.0   min(9.4,14.0,2.121)=2.12132034
                      min(9.4,2.0,2.121)=2.0    min(1.0,2.0,2.121)=1.0
    """
    mesh = _box_mesh_raf(**_BOX)
    depth = raf_render.render_depth(mesh, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    expected = np.array([
        [1.0, 3.53553391, 2.0, 1.0],
        [1.0, 2.12132034, 2.0, 1.0],
    ])
    np.testing.assert_allclose(depth, expected, atol=1e-4)


def test_render_depth_matches_the_hand_derived_box_distances_camera_two():
    """Camera RAF (4.5, 2.5, -1.4) -> pipeline (4.5, -1.4, 2.5). Wall distances:

        x- : 4.5 + 0.2 = 4.7  -> t = 9.4
        x+ : 5.0 - 4.5 = 0.5  -> t = 1.0
        y- : -1.4 + 2.0 = 0.6 -> t = 1.2
        y+ : 6.0 + 1.4 = 7.4  -> t = 14.8
        z+ : 4.0 - 2.5 = 1.5  -> t = 2.12132034
        z- : 2.5 - 0.0 = 2.5  -> t = 3.53553391

    Together with camera one this exercises all six walls.
        row 0 (up):   min(9.4,1.2,2.121)=1.2   min(1.0,1.2,2.121)=1.0
                      min(1.0,14.8,2.121)=1.0  min(9.4,14.8,2.121)=2.12132034
        row 1 (down): min(9.4,1.2,3.536)=1.2   min(1.0,1.2,3.536)=1.0
                      min(1.0,14.8,3.536)=1.0  min(9.4,14.8,3.536)=3.53553391
    """
    mesh = _box_mesh_raf(**_BOX)
    depth = raf_render.render_depth(mesh, np.array([4.5, -1.4, 2.5]), h=2, w=4)
    expected = np.array([
        [1.2, 1.0, 1.0, 2.12132034],
        [1.2, 1.0, 1.0, 3.53553391],
    ])
    np.testing.assert_allclose(depth, expected, atol=1e-4)


def test_render_depth_puts_the_raf_plus_y_wall_at_the_pipeline_plus_z_pole():
    """Axis mapping: RAF +Y (up) must land on row 0 of the map, RAF -Y on row H-1.

    Box RAF X, Z in [-10, 10], Y in [0, 3]; camera RAF (0, 1, 0). The ceiling is
    2.0 m above the camera and the floor 1.0 m below, so the two poles are
    unambiguous (the side walls are 10 m away). Row 0 is 0.35 degrees off vertical,
    hence 2.0/cos(pi/512) = 2.00004 rather than exactly 2.
    """
    mesh = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0)
    depth = raf_render.render_depth(mesh, np.array([0.0, 0.0, 1.0]))
    assert depth.shape == (256, 512)
    np.testing.assert_allclose(np.median(depth[0]), 2.0, atol=1e-3)
    np.testing.assert_allclose(np.median(depth[-1]), 1.0, atol=1e-3)


def test_render_depth_output_dtype_and_shape():
    mesh = _box_mesh_raf(**_BOX)
    depth = raf_render.render_depth(mesh, np.array([0.3, 5.0, 1.5]), h=8, w=16)
    assert depth.shape == (8, 16)
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all() and (depth > 0).all()


def test_render_depth_aborts_on_a_ray_miss():
    """Registered miss policy: any inf aborts with a report. No silent fill."""
    mesh = _box_mesh_raf(drop_ceiling=True, **_BOX)
    with pytest.raises(RuntimeError) as exc:
        raf_render.render_depth(mesh, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    assert "miss" in str(exc.value).lower()


def test_render_depth_accepts_a_tensor_mesh():
    mesh = _box_mesh_raf(**_BOX)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    depth = raf_render.render_depth(tmesh, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    np.testing.assert_allclose(depth[0, 1], 3.53553391, atol=1e-4)


@pytest.mark.parametrize("position", [
    np.array([0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, np.nan, 1.0]),
])
def test_render_depth_rejects_a_bad_camera_position(position):
    mesh = _box_mesh_raf(**_BOX)
    with pytest.raises(ValueError):
        raf_render.render_depth(mesh, position, h=2, w=4)


# --------------------------------------------------------------------------- #
# per-map QA
# --------------------------------------------------------------------------- #
def test_depth_qa_on_a_good_map():
    mesh = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0)
    position = np.array([0.0, 0.0, 1.0])
    depth = raf_render.render_depth(mesh, position)
    qa = raf_render.depth_qa(depth, position)
    assert qa["passed"] is True
    assert qa["finite"] is True and qa["positive"] is True
    assert qa["hit_rate"] == 1.0
    assert qa["shape"] == [256, 512] and qa["dtype"] == "float32"
    assert qa["camera_height"] == 1.0
    np.testing.assert_allclose(qa["nadir_distance"], 1.0, atol=1e-3)
    assert abs(qa["floor_delta"]) < 0.05 and qa["floor_ok"] is True
    assert qa["min"] > 0 and qa["max"] >= qa["min"]
    assert qa["warnings"] == []


def test_depth_qa_flags_a_floor_mismatch_without_failing_the_map():
    """An occluder under the source is physically possible in FurnishedRoom, so a
    floor mismatch is a recorded warning; only structural defects fail a map."""
    mesh = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0)
    depth = raf_render.render_depth(mesh, np.array([0.0, 0.0, 1.0]))
    qa = raf_render.depth_qa(depth, np.array([0.0, 0.0, 2.0]))  # wrong height on purpose
    assert qa["floor_ok"] is False
    assert qa["passed"] is True
    assert any("floor" in w for w in qa["warnings"])


def test_depth_qa_fails_on_non_finite_and_non_positive_maps():
    good = np.full((4, 8), 2.0, dtype=np.float32)
    bad = good.copy()
    bad[0, 0] = np.inf
    qa = raf_render.depth_qa(bad, np.array([0.0, 0.0, 1.0]))
    assert qa["finite"] is False and qa["passed"] is False
    assert qa["hit_rate"] < 1.0

    bad2 = good.copy()
    bad2[1, 1] = -1.0
    qa2 = raf_render.depth_qa(bad2, np.array([0.0, 0.0, 1.0]))
    assert qa2["positive"] is False and qa2["passed"] is False


def test_depth_qa_fails_on_a_wrong_dtype():
    qa = raf_render.depth_qa(np.full((4, 8), 2.0, dtype=np.float64),
                             np.array([0.0, 0.0, 1.0]))
    assert qa["dtype"] == "float64"
    assert qa["passed"] is False


# --------------------------------------------------------------------------- #
# mesh loading + CLI
# --------------------------------------------------------------------------- #
def _write_fixture(tmp_path, room="EmptyRoom", keys=("aaaa000000000001", "bbbb000000000002")):
    raf_root = tmp_path / "raf"
    out = tmp_path / "runtime" / "RAF"
    mesh_dir = raf_root / "3d_models" / room
    mesh_dir.mkdir(parents=True)
    # written in RAF world coordinates: the loader is what applies the gauge
    mesh_raf = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0,
                             to_pipeline=False)
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.obj"), mesh_raf)
    meta_dir = out / room / "metadata"
    meta_dir.mkdir(parents=True)
    groups = {
        keys[0]: {"tx_xyz_p": [0.0, 0.0, 1.0], "depth_file": f"{keys[0]}_depth_image.npy",
                  "train_ids": ["000000"], "role": "train_test"},
        keys[1]: {"tx_xyz_p": [1.0, 2.0, 1.5], "depth_file": f"{keys[1]}_depth_image.npy",
                  "train_ids": ["000036"], "role": "val"},
    }
    with open(meta_dir / "groups_metadata.json", "w") as f:
        json.dump(groups, f)
    return raf_root, out, groups


def test_load_mesh_pipeline_applies_the_gauge(tmp_path):
    raf_root, _, _ = _write_fixture(tmp_path)
    mesh = raf_render.load_mesh_pipeline(str(raf_root / "3d_models" / "EmptyRoom" / "mesh.obj"))
    verts = np.asarray(mesh.vertices)
    # RAF Y in [0, 3] (up) must become pipeline z in [0, 3]
    assert verts[:, 2].min() == pytest.approx(0.0)
    assert verts[:, 2].max() == pytest.approx(3.0)
    assert verts[:, 1].min() == pytest.approx(-10.0)


def test_cli_renders_every_group_and_writes_qa(tmp_path):
    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom"])
    depth_dir = out / "EmptyRoom" / "depth_images"
    for key, entry in groups.items():
        path = depth_dir / entry["depth_file"]
        assert path.exists()
        arr = np.load(path)
        assert arr.shape == (256, 512) and arr.dtype == np.float32
        assert np.isfinite(arr).all()
    qa_path = depth_dir / "raf_depth_qa.json"
    assert qa_path.exists()
    with open(qa_path) as f:
        qa = json.load(f)
    assert set(qa["maps"]) == set(groups)
    assert all(m["passed"] for m in qa["maps"].values())
    # group 1 sits 1.0 m above the floor, group 2 1.5 m
    np.testing.assert_allclose(qa["maps"]["aaaa000000000001"]["nadir_distance"], 1.0,
                               atol=1e-3)
    np.testing.assert_allclose(qa["maps"]["bbbb000000000002"]["nadir_distance"], 1.5,
                               atol=1e-3)


def test_cli_render_is_consistent_with_the_pipeline_pixel_to_ray_map(tmp_path):
    """The emitted map, times the pipeline's own directions, is a point cloud whose
    zenith column points up: the renderer emits rows in the convention
    ``convert_equirect_to_camera_coord`` assumes, with no flipud in between."""
    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom"])
    arr = np.load(out / "EmptyRoom" / "depth_images" / groups["aaaa000000000001"]["depth_file"])
    cloud = arr[..., None] * raf_common.equirect_directions()
    assert cloud[0, :, 2].min() > 1.9      # ceiling 2.0 m above the camera
    assert cloud[-1, :, 2].max() < -0.9    # floor 1.0 m below it


def test_cli_aborts_when_the_mesh_is_missing(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    os.remove(raf_root / "3d_models" / "EmptyRoom" / "mesh.obj")
    with pytest.raises((FileNotFoundError, ValueError)):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom"])
