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
def _readback(tmp_path):
    """The R4 publish gate's input; canonical renders require a pinned record."""
    from test_raf_prepare_data import write_passing_readback_record

    return write_passing_readback_record(str(tmp_path / "readback.json"))


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
    # tx_height_raf_m is the RAW RAF Y, exactly as prepare_data publishes it: the
    # pipeline z of these cameras happens to equal it under the PINNED gauge, which
    # is the point -- under a wrong gauge the two diverge (r5 finding 5).
    groups = {
        keys[0]: {"tx_xyz_p": [0.0, 0.0, 1.0], "tx_height_raf_m": 1.0,
                  "depth_file": f"{keys[0]}_depth_image.npy",
                  "train_ids": ["000000"], "role": "train_test"},
        keys[1]: {"tx_xyz_p": [1.0, 2.0, 1.5], "tx_height_raf_m": 1.5,
                  "depth_file": f"{keys[1]}_depth_image.npy",
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
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
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
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    arr = np.load(out / "EmptyRoom" / "depth_images" / groups["aaaa000000000001"]["depth_file"])
    cloud = arr[..., None] * raf_common.equirect_directions()
    assert cloud[0, :, 2].min() > 1.9      # ceiling 2.0 m above the camera
    assert cloud[-1, :, 2].max() < -0.9    # floor 1.0 m below it


def test_cli_aborts_when_the_mesh_is_missing(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    os.remove(raf_root / "3d_models" / "EmptyRoom" / "mesh.obj")
    with pytest.raises((FileNotFoundError, ValueError)):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])


# --------------------------------------------------------------------------- #
# r2 R5: canonical dimensions are enforced, non-canonical ones taint the record
# --------------------------------------------------------------------------- #
def test_depth_qa_requires_the_canonical_256x512_float32():
    qa = raf_render.depth_qa(np.full((128, 256), 2.0, dtype=np.float32),
                             np.array([0.0, 0.0, 1.0]))
    assert qa["canonical_shape"] is False
    assert qa["passed"] is False
    assert any("256" in w for w in qa["warnings"])


def test_depth_qa_accepts_non_canonical_dims_only_when_declared():
    depth = np.full((4, 8), 2.0, dtype=np.float32)
    qa = raf_render.depth_qa(depth, np.array([0.0, 0.0, 1.0]), img_h=4, img_w=8,
                             canonical=False)
    assert qa["canonical_shape"] is True     # matches the declared grid
    assert qa["canonical"] is False          # ... but the record is tainted
    assert qa["passed"] is True


def test_cli_refuses_non_canonical_dims_without_the_flag(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--img-h", "128", "--img-w", "256",
                         "--readback-record", _readback(tmp_path)])   # canonical
    assert "--non-canonical" in str(exc.value)


def test_cli_taints_the_qa_record_under_non_canonical(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--img-h", "64", "--img-w", "128",
                     "--non-canonical", "--readback-record", _readback(tmp_path)])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["canonical"] is False
    assert "non-canonical" in " ".join(qa["taint"])


# --------------------------------------------------------------------------- #
# r2 R6: real-mesh QA
# --------------------------------------------------------------------------- #
def test_real_mesh_qa_confirms_camera_containment_and_bounds():
    mesh = _box_mesh_raf(**_BOX)
    position = np.array([0.3, 5.0, 1.5])
    depth = raf_render.render_depth(mesh, position, h=32, w=64)
    qa = raf_render.real_mesh_qa(depth, position, mesh, img_h=32, img_w=64)
    assert qa["camera_inside"] is True
    assert qa["bounds_ok"] is True
    assert qa["passed"] is True
    assert qa["mesh_bounds"]["min"] == pytest.approx([-0.2, -2.0, 0.0])
    assert qa["mesh_bounds"]["max"] == pytest.approx([5.0, 6.0, 4.0])


def test_real_mesh_qa_detects_a_camera_outside_the_room():
    mesh = _box_mesh_raf(**_BOX)
    outside = np.array([50.0, 50.0, 2.0])
    qa = raf_render.real_mesh_qa(np.full((32, 64), 1.0, dtype=np.float32), outside,
                                 mesh, img_h=32, img_w=64)
    assert qa["camera_inside"] is False
    assert qa["passed"] is False


def test_real_mesh_qa_landmark_bearing_agrees_with_the_mesh():
    """Non-circular gauge check: the map's farthest ray must point at the mesh's
    farthest surface. A transposed axis would swing this by ~90 degrees."""
    mesh = _box_mesh_raf(**_BOX)
    position = np.array([0.3, 5.0, 1.5])
    depth = raf_render.render_depth(mesh, position, h=64, w=128)
    qa = raf_render.real_mesh_qa(depth, position, mesh, img_h=64, img_w=128)
    assert qa["bearing_delta_deg"] < 15.0
    assert qa["bearing_ok"] is True


def test_real_mesh_qa_records_the_depth_scale_against_the_reference_range():
    mesh = _box_mesh_raf(**_BOX)
    position = np.array([0.3, 5.0, 1.5])
    depth = raf_render.render_depth(mesh, position, h=32, w=64)
    qa = raf_render.real_mesh_qa(depth, position, mesh, img_h=32, img_w=64)
    scale = qa["depth_scale"]
    assert set(scale) >= {"min", "max", "mean", "p50", "p95"}
    assert qa["scale_plausible"] is True
    assert set(qa["scale_reference"]) == {"AR", "HAA"}
    assert qa["scale_checked"] is False        # no reference corpus was supplied


def test_cli_persists_real_mesh_qa_and_a_render_benchmark(tmp_path):
    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    for entry in qa["maps"].values():
        assert entry["real_mesh"]["camera_inside"] is True
        assert entry["real_mesh"]["bounds_ok"] is True
    bench = qa["render_benchmark"]
    assert bench["n_maps"] == len(groups)
    assert bench["scene_build_s"] >= 0.0
    assert bench["mean_render_s"] > 0.0
    assert qa["landmark_bearings"]["aaaa000000000001"] == pytest.approx(
        qa["maps"]["aaaa000000000001"]["real_mesh"]["landmark_bearing_deg"])


def test_cli_aborts_when_a_camera_is_outside_the_mesh(tmp_path):
    raf_root, out, groups = _write_fixture(tmp_path)
    meta = out / "EmptyRoom" / "metadata" / "groups_metadata.json"
    with open(meta) as f:
        payload = json.load(f)
    payload["aaaa000000000001"]["tx_xyz_p"] = [99.0, 99.0, 1.0]
    with open(meta, "w") as f:
        json.dump(payload, f)
    with pytest.raises((RuntimeError, ValueError)):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])


# --------------------------------------------------------------------------- #
# r2 R12: one scene per room, reused across cameras
# --------------------------------------------------------------------------- #
def test_build_scene_is_reusable_across_cameras():
    mesh = _box_mesh_raf(**_BOX)
    scene = raf_render.build_scene(mesh)
    a = raf_render.render_depth(scene, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    b = raf_render.render_depth(scene, np.array([4.5, -1.4, 2.5]), h=2, w=4)
    np.testing.assert_allclose(a, [[1.0, 3.53553391, 2.0, 1.0],
                                   [1.0, 2.12132034, 2.0, 1.0]], atol=1e-4)
    np.testing.assert_allclose(b, [[1.2, 1.0, 1.0, 2.12132034],
                                   [1.2, 1.0, 1.0, 3.53553391]], atol=1e-4)


def test_cli_builds_the_scene_once_per_room(tmp_path, monkeypatch):
    raf_root, out, groups = _write_fixture(tmp_path)
    constructions = []
    real_build = raf_render.build_scene

    def counting_build(mesh):
        # a call that is handed an existing scene is a no-op passthrough; only a
        # call handed a MESH pays for the conversion + acceleration structure
        if not isinstance(mesh, o3d.t.geometry.RaycastingScene):
            constructions.append(1)
        return real_build(mesh)

    monkeypatch.setattr(raf_render, "build_scene", counting_build)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    assert len(groups) == 2
    assert len(constructions) == 1          # not once per camera, nor once per QA call


# --------------------------------------------------------------------------- #
# r2 Amendment 4: ray-miss policy on real scanned meshes
# --------------------------------------------------------------------------- #
def test_fill_missing_is_a_no_op_when_every_ray_hit():
    hits = np.full((4, 8), 2.0, dtype=np.float32)
    filled, report = raf_render.fill_missing(hits)
    assert np.array_equal(filled, hits)
    assert report["miss_count"] == 0
    assert report["miss_rate"] == 0.0
    assert report["within_cap"] is True
    assert report["filled_pixels_sha256"] == raf_render.EMPTY_FILL_HASH


def test_fill_missing_uses_the_nearest_valid_neighbour():
    """Hand-checkable: the hole at (1, 3) has (0,3)=7.0 directly above it at
    distance 1, while every other valid neighbour is further away."""
    hits = np.full((4, 8), 2.0, dtype=np.float32)
    hits[0, 3] = 7.0
    hits[1, 3] = np.inf
    hits[1, 2] = np.inf
    hits[1, 4] = np.inf
    hits[2, 3] = np.inf
    filled, report = raf_render.fill_missing(hits, max_miss_rate=1.0)
    assert filled[1, 3] == pytest.approx(7.0)
    assert np.isfinite(filled).all()
    assert report["miss_count"] == 4
    assert report["miss_rate"] == pytest.approx(4 / 32)
    assert report["filled_pixels"][:2] == [[1, 2], [1, 3]]


def test_fill_missing_records_a_coordinate_hash_that_identifies_the_repair():
    a = np.full((4, 8), 2.0, dtype=np.float32)
    a[1, 3] = np.inf
    b = np.full((4, 8), 2.0, dtype=np.float32)
    b[2, 5] = np.inf
    _, ra = raf_render.fill_missing(a, max_miss_rate=1.0)
    _, rb = raf_render.fill_missing(b, max_miss_rate=1.0)
    _, ra2 = raf_render.fill_missing(a.copy(), max_miss_rate=1.0)
    assert ra["filled_pixels_sha256"] == ra2["filled_pixels_sha256"]
    assert ra["filled_pixels_sha256"] != rb["filled_pixels_sha256"]
    assert len(ra["filled_pixels_sha256"]) == 64


def test_fill_missing_aborts_above_the_cap():
    hits = np.full((10, 10), 2.0, dtype=np.float32)
    hits[0, :2] = np.inf              # 2% >> 0.1%
    with pytest.raises(RuntimeError) as exc:
        raf_render.fill_missing(hits)
    message = str(exc.value)
    assert "miss" in message.lower()
    assert "0.1" in message or str(raf_render.DEFAULT_MAX_MISS_RATE) in message


def test_fill_missing_tolerates_exactly_the_cap():
    hits = np.full((1000, 10), 2.0, dtype=np.float32)
    hits[0, :10] = np.inf             # 10 / 10000 = 0.1%
    filled, report = raf_render.fill_missing(hits)
    assert report["miss_rate"] == pytest.approx(raf_render.DEFAULT_MAX_MISS_RATE)
    assert report["within_cap"] is True
    assert np.isfinite(filled).all()


def test_fill_missing_refuses_an_all_missing_map():
    with pytest.raises(RuntimeError):
        raf_render.fill_missing(np.full((4, 8), np.inf, dtype=np.float32),
                                max_miss_rate=1.0)


def _box_with_pinhole(divisions=32, **bounds):
    """The six-wall box with the ceiling subdivided and its centre quad removed.

    Stands in for a real scanned mesh's hole: FurnishedRoom missed 62 of 131,072
    rays at a real tx position, which is the regime this fixture reproduces.
    """
    mesh = _box_mesh_raf(drop_ceiling=True, **bounds)
    verts = list(np.asarray(mesh.vertices))
    faces = list(np.asarray(mesh.triangles))
    x0, x1 = bounds["x0"], bounds["x1"]
    z0, z1 = bounds["z0"], bounds["z1"]
    y1 = bounds["y1"]
    xs = np.linspace(x0, x1, divisions + 1)
    zs = np.linspace(z0, z1, divisions + 1)
    base = len(verts)
    for zi in zs:                               # RAF (x, y1, z) -> pipeline (x, z, y1)
        for xi in xs:
            verts.append([xi, zi, y1])
    skip = (divisions // 2, divisions // 2)
    for r in range(divisions):
        for c in range(divisions):
            if (r, c) == skip:
                continue
            a = base + r * (divisions + 1) + c
            b, d, e = a + 1, a + divisions + 1, a + divisions + 2
            faces.append([a, b, e])
            faces.append([a, e, d])
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.array(verts, dtype=np.float64))
    out.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    return out


def test_render_depth_repairs_a_scan_hole_within_the_cap():
    mesh = _box_with_pinhole(**_BOX)
    # off the zenith axis on purpose: directly under the hole it would sit at the
    # pole, where the equirect rows converge and a tiny hole swallows many rays
    position = np.array([0.5, 0.5, 1.5])
    depth, report = raf_render.render_depth(mesh, position, return_report=True)
    assert 0 < report["miss_count"] <= int(raf_render.DEFAULT_MAX_MISS_RATE * 256 * 512)
    assert report["within_cap"] is True
    assert np.isfinite(depth).all() and (depth > 0).all()
    assert depth.dtype == np.float32
    # the repaired pixels carry a real neighbouring distance, not a sentinel
    rows, cols = zip(*report["filled_pixels"])
    assert depth[rows, cols].min() > 2.0


def test_render_depth_still_aborts_above_the_cap():
    """A whole missing wall is not a scan hole; the registered abort stands."""
    mesh = _box_mesh_raf(drop_ceiling=True, **_BOX)
    with pytest.raises(RuntimeError) as exc:
        raf_render.render_depth(mesh, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    message = str(exc.value)
    assert "miss" in message.lower()
    assert "cap" in message.lower() or "rate" in message.lower()


def test_render_depth_report_is_optional_and_backward_compatible():
    mesh = _box_mesh_raf(**_BOX)
    depth = raf_render.render_depth(mesh, np.array([0.3, 5.0, 1.5]), h=2, w=4)
    assert isinstance(depth, np.ndarray)


def test_cli_records_the_miss_report_per_map(tmp_path):
    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["max_miss_rate"] == raf_render.DEFAULT_MAX_MISS_RATE
    for entry in qa["maps"].values():
        assert entry["misses"]["miss_count"] == 0          # the fixture box is closed
        assert entry["misses"]["within_cap"] is True
        assert entry["misses"]["filled_pixels_sha256"] == raf_render.EMPTY_FILL_HASH


# --------------------------------------------------------------------------- #
# r2 Amendment 4: bearing tie rule + floor tolerance
# --------------------------------------------------------------------------- #
def test_bearing_is_inapplicable_when_a_near_equal_surface_lies_far_in_bearing():
    """The EmptyRoom sample-3 false alarm: a second surface within 2% of the
    farthest distance but ~96 degrees away in bearing is a tie, not a gauge error."""
    mesh = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0)
    position = np.array([0.0, 0.0, 1.0])        # centre: four equidistant corners
    depth = raf_render.render_depth(mesh, position, h=64, w=128)
    qa = raf_render.real_mesh_qa(depth, position, mesh, img_h=64, img_w=128)
    assert qa["bearing_applicable"] is False
    assert qa["bearing_ok"] is True             # inapplicable never fails the map
    assert qa["passed"] is True
    assert any("not applicable" in w for w in qa["warnings"])
    assert qa["bearing_tie_distance_frac"] == 0.02
    assert qa["bearing_tie_angle_deg"] == 20.0


def test_bearing_stays_applicable_when_near_equal_surfaces_share_a_bearing():
    """Two surfaces within 2% of each other but only degrees apart do NOT excuse
    the check: that is the case the gauge test is meant to catch."""
    mesh = _box_mesh_raf(**_BOX)
    position = np.array([0.3, 5.0, 1.5])
    depth = raf_render.render_depth(mesh, position, h=64, w=128)
    qa = raf_render.real_mesh_qa(depth, position, mesh, img_h=64, img_w=128)
    assert qa["bearing_applicable"] is True
    assert qa["bearing_ok"] is True
    assert qa["bearing_delta_deg"] < 15.0


def test_floor_tolerance_accepts_a_real_scan_deficit():
    """Real EmptyRoom nadir deficits reach 0.10 m (scan content above y=0), which
    must not read as a defect; 0.15 m is the recorded threshold."""
    assert raf_render.DEFAULT_FLOOR_TOL == 0.15
    mesh = _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0)
    position = np.array([0.0, 0.0, 1.0])
    depth = raf_render.render_depth(mesh, position)
    qa = raf_render.depth_qa(depth, position + np.array([0.0, 0.0, 0.10]))
    assert qa["floor_ok"] is True
    assert qa["warnings"] == []
    far = raf_render.depth_qa(depth, position + np.array([0.0, 0.0, 0.30]))
    assert far["floor_ok"] is False
    assert far["passed"] is True                # still a warning, never an abort


# --------------------------------------------------------------------------- #
# r3 S2: the canonical miss cap is the constant, and QA enforces it itself
# --------------------------------------------------------------------------- #
def test_canonical_mode_refuses_a_looser_miss_cap():
    with pytest.raises(ValueError) as exc:
        raf_render.resolve_miss_cap(0.05, canonical=True)
    assert "--non-canonical" in str(exc.value)
    assert str(raf_render.DEFAULT_MAX_MISS_RATE) in str(exc.value)
    # lowering is always allowed; a looser cap taints non-canonical output
    assert raf_render.resolve_miss_cap(0.0001, canonical=True) == (0.0001, [])
    cap, taint = raf_render.resolve_miss_cap(0.05, canonical=False)
    assert cap == 0.05 and any("above the registered" in t for t in taint)


def test_cli_taints_a_non_canonical_run_with_a_looser_cap(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--max-miss-rate", "0.05",
                     "--readback-record", _readback(tmp_path), "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert any("above the registered" in t for t in qa["taint"])


def test_canonical_mode_allows_a_stricter_miss_cap(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--max-miss-rate", "0.0001",
                     "--readback-record", _readback(tmp_path), "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        assert json.load(f)["max_miss_rate"] == 0.0001


def test_direction_to_pixel_inverts_the_ray_grid():
    """Hand-checkable: the pixel a direction maps to must be the pixel whose own
    direction it is. Verified against equirect_directions itself, which is pinned
    independently in test_raf_common."""
    dirs = raf_common.equirect_directions(64, 128)
    for i, j in [(0, 0), (7, 3), (31, 64), (63, 127), (32, 0)]:
        assert raf_render.direction_to_pixel(dirs[i, j], 64, 128) == (i, j)


def test_direction_to_pixel_wraps_the_azimuth_seam():
    row, col = raf_render.direction_to_pixel(np.array([-1.0, -1e-9, 0.0]), 64, 128)
    assert 0 <= col < 128 and 0 <= row < 64


def test_rx_sightline_check_passes_when_receivers_are_visible():
    """The receivers are inside the same empty box, so every tx->rx ray must reach
    at least as far as the receiver before hitting a wall."""
    mesh = _box_mesh_raf(**_BOX)
    tx = np.array([0.5, 0.5, 1.5])
    rx = np.array([[4.5, 5.0, 1.2], [0.0, -1.5, 2.0], [3.0, 0.0, 0.5]])
    depth = raf_render.render_depth(mesh, tx)
    report = raf_render.rx_sightline_check(depth, tx, rx)
    assert report["n_receivers"] == 3
    assert report["n_blocked"] == 0
    assert report["passed"] is True
    assert report["worst_deficit_m"] <= report["tol_m"]
    assert len(report["per_receiver"]) == 3
    for entry in report["per_receiver"]:
        assert entry["depth_m"] >= entry["distance_m"] - report["tol_m"]


def test_rx_sightline_check_detects_an_occluded_receiver():
    """A receiver behind a partition: the ray stops at the partition, so the
    rendered depth is far short of the receiver distance."""
    mesh = _box_mesh_raf(**_BOX)
    wall = _box_mesh_raf(x0=2.0, x1=2.1, y0=0.0, y1=4.0, z0=-2.0, z1=6.0)
    combined = mesh + wall
    tx = np.array([0.5, 0.5, 1.5])
    rx = np.array([[4.5, 0.5, 1.5]])            # on the far side of the partition
    depth = raf_render.render_depth(combined, tx)
    report = raf_render.rx_sightline_check(depth, tx, rx)
    assert report["n_blocked"] == 1
    assert report["passed"] is False
    assert report["worst_deficit_m"] > 1.0


def test_rx_sightline_check_catches_a_transposed_gauge():
    """S5's point: this evidence is mesh-INDEPENDENT, so a consistently wrong
    horizontal gauge (rx transformed with y and z swapped) fails it, while the old
    landmark comparison against the same transformed mesh could not."""
    mesh = _box_mesh_raf(**_BOX)
    tx = np.array([0.5, 0.5, 1.5])
    rx = np.array([[4.5, 5.0, 1.2], [0.0, -1.5, 2.0], [3.0, 5.5, 0.5]])
    depth = raf_render.render_depth(mesh, tx)
    assert raf_render.rx_sightline_check(depth, tx, rx)["passed"] is True
    mis_gauged = rx[:, [0, 2, 1]]               # y <-> z transposition
    assert raf_render.rx_sightline_check(depth, tx, mis_gauged)["passed"] is False


def test_rx_sightline_check_uses_the_farthest_receivers():
    mesh = _box_mesh_raf(**_BOX)
    tx = np.array([0.5, 0.5, 1.5])
    rx = np.array([[0.6, 0.5, 1.5], [4.5, 5.0, 1.2], [0.0, -1.5, 2.0]])
    report = raf_render.rx_sightline_check(depth=raf_render.render_depth(mesh, tx),
                                           position_p=tx, rx_positions_p=rx,
                                           max_receivers=1)
    assert report["n_receivers"] == 1
    # the single probed receiver is the farthest one, not the first listed
    assert report["per_receiver"][0]["distance_m"] > 5.0


_SYNTHETIC_REFERENCE = os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                    "raf_depth_reference")
_REAL_HAA = os.path.join(_REPO_ROOT, "HAA")
# sha256 over "n_maps|min|max" of the four processed HAA base rooms, measured
# 2026-08-20. Pins the reference the canonical run will actually use.
_REAL_HAA_BAND_SHA256 = \
    "1d59babdbc1b0b6075b32216c864588acf5516454a92a4a6af946bd832656eb3"


def test_reference_depth_stats_read_the_committed_synthetic_corpus():
    """T9: unit tests must be checkout-reproducible, so they read a committed
    fixture rather than the unversioned HAA mount."""
    stats = raf_render.reference_depth_stats(_SYNTHETIC_REFERENCE)
    assert stats["available"] is True
    assert stats["n_maps"] == 2
    assert stats["min"] == pytest.approx(0.60, abs=1e-3)
    assert stats["max"] == pytest.approx(11.40, abs=1e-3)


@pytest.mark.skipif(not os.path.isdir(_REAL_HAA),
                    reason="processed HAA corpus is not present in this checkout")
def test_real_haa_reference_band_matches_its_pinned_hash():
    """Integration check on the actual reference the canonical run will use."""
    import hashlib

    stats = raf_render.reference_depth_stats(_REAL_HAA)
    assert stats["available"] is True and stats["n_maps"] == 4
    fingerprint = f"{stats['n_maps']}|{stats['min']:.6f}|{stats['max']:.6f}"
    assert hashlib.sha256(fingerprint.encode()).hexdigest() == _REAL_HAA_BAND_SHA256


def test_reference_depth_stats_record_an_unreadable_root(tmp_path):
    stats = raf_render.reference_depth_stats(str(tmp_path / "nope"))
    assert stats["available"] is False
    assert stats["n_maps"] == 0
    assert "not readable" in stats["reason"]


def test_scale_plausible_joins_passed_once_a_reference_is_present():
    mesh = _box_mesh_raf(**_BOX)
    tx = np.array([0.5, 0.5, 1.5])
    depth = raf_render.render_depth(mesh, tx, h=32, w=64)
    references = {"HAA": raf_render.reference_depth_stats(_SYNTHETIC_REFERENCE),
                  "AR": raf_render.reference_depth_stats("/nonexistent-ar-root")}
    qa = raf_render.real_mesh_qa(depth, tx, mesh, img_h=32, img_w=64,
                                 references=references)
    assert qa["scale_reference"]["HAA"]["available"] is True
    assert qa["scale_reference"]["AR"]["available"] is False
    assert qa["scale_plausible"] is True
    assert qa["scale_checked"] is True
    assert qa["passed"] is True

    huge = depth * 40.0                          # 40x the HAA band
    bad = raf_render.real_mesh_qa(huge, tx, mesh, img_h=32, img_w=64,
                                  references=references)
    assert bad["scale_plausible"] is False
    assert bad["passed"] is False


def test_landmark_bearing_is_recorded_but_no_longer_gates():
    """S5: the landmark compares the render against the same transformed mesh, so
    it is circular; it stays as recorded diagnostics only."""
    mesh = _box_mesh_raf(**_BOX)
    tx = np.array([0.3, 5.0, 1.5])
    depth = raf_render.render_depth(mesh, tx, h=32, w=64)
    qa = raf_render.real_mesh_qa(depth, tx, mesh, img_h=32, img_w=64)
    assert "landmark_bearing_deg" in qa
    assert qa["bearing_gates_publication"] is False


def test_real_mesh_qa_requires_sightlines_when_the_room_says_so():
    mesh = _box_mesh_raf(**_BOX)
    wall = _box_mesh_raf(x0=2.0, x1=2.1, y0=0.0, y1=4.0, z0=-2.0, z1=6.0)
    tx = np.array([0.5, 0.5, 1.5])
    rx = np.array([[4.5, 0.5, 1.5]])
    depth = raf_render.render_depth(mesh + wall, tx, h=256, w=512)
    required = raf_render.real_mesh_qa(depth, tx, mesh + wall, rx_positions_p=rx,
                                       rx_sightline_required=True)
    assert required["rx_sightline"]["passed"] is False
    assert required["passed"] is False
    recorded = raf_render.real_mesh_qa(depth, tx, mesh + wall, rx_positions_p=rx,
                                       rx_sightline_required=False)
    assert recorded["rx_sightline"]["passed"] is False
    assert recorded["passed"] is True            # occlusion-tolerant room
    assert any("recorded" in w for w in recorded["warnings"])


def test_cli_persists_the_s5_evidence_and_gates_on_it(tmp_path):
    raf_root, out, groups = _write_fixture(tmp_path)
    # give the runtime tree the poses the sightline probe reads (mesh-independent)
    meta = out / "EmptyRoom" / "metadata"
    poses = {f"{i:06d}": {"tx_xyz_p": [0.0, 0.0, 1.0], "quat_raw": [0, 0, 0, 1],
                          "rx_p": [float(x), float(y), 1.0], "group_key": "aaaa000000000001",
                          "split_role": "train"}
             for i, (x, y) in enumerate([(8.0, 8.0), (-8.0, -8.0), (8.0, -8.0)])}
    with open(meta / "poses_metadata.json", "w") as f:
        json.dump(poses, f)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--haa-depth-root", _SYNTHETIC_REFERENCE,
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["scale_reference"]["HAA"]["available"] is True
    for entry in qa["maps"].values():
        evidence = entry["real_mesh"]["rx_sightline"]
        assert evidence["n_receivers"] == 3
        assert evidence["passed"] is True
        assert evidence["required"] is True          # EmptyRoom is unconditional
        assert entry["real_mesh"]["scale_checked"] is True
    assert qa["rx_sightline_policy"]["EmptyRoom"] == "required"
    assert qa["rx_sightline_policy"]["FurnishedRoom"] == "recorded"


# --------------------------------------------------------------------------- #
# r4 T7: the miss audit reads the RAW hit mask, not the report's claims
# --------------------------------------------------------------------------- #
def test_miss_report_carries_the_raw_hit_mask_digest():
    mesh = _box_with_pinhole(**_BOX)
    position = np.array([0.5, 0.5, 1.5])
    depth, report = raf_render.render_depth(mesh, position, return_report=True)
    assert report["hit_mask_sha256"] == raf_render.fill_hash(report["filled_pixels"])
    assert report["n_rays"] == 256 * 512
    audit, warnings = raf_render.audit_miss_report(report, depth,
                                                   miss_mask=report["miss_mask"])
    assert audit["audit_ok"] is True
    assert audit["mask_verified"] is True
    # every number in the verdict is derived from the mask, not from the report
    assert audit["miss_count_from_mask"] == report["miss_count"]
    assert audit["filled_pixels_sha256_from_mask"] == report["filled_pixels_sha256"]


def _mask_with(shape, coords):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _report_for(coords, n_rays, **overrides):
    report = {"miss_count": len(coords), "miss_rate": len(coords) / n_rays,
              "within_cap": True, "max_miss_rate": raf_render.DEFAULT_MAX_MISS_RATE,
              "filled_pixels": [list(c) for c in coords],
              "filled_pixels_sha256": raf_render.fill_hash(coords), "n_rays": n_rays}
    report.update(overrides)
    return report


def test_audit_requires_the_raw_mask():
    """r5 finding 4: without the mask the report cannot be tied to this map, so
    mask_verified=None is no longer an acceptable verdict."""
    depth = np.full((256, 512), 2.0, dtype=np.float32)
    report = _report_for([], 256 * 512,
                         filled_pixels_sha256=raf_render.EMPTY_FILL_HASH)
    audit, warnings = raf_render.audit_miss_report(report, depth)
    assert audit["mask_verified"] is False
    assert audit["audit_ok"] is False
    assert any("raw" in w for w in warnings)


def test_audit_rejects_a_zero_miss_report_when_the_mask_says_otherwise():
    """The exact hostile probe: empty coordinates, the public empty-set hash, the
    right ray count -- and a map that actually missed rays."""
    depth = np.full((8, 16), 2.0, dtype=np.float32)
    mask = _mask_with((8, 16), [(3, 4)])
    forged = _report_for([], 128, filled_pixels_sha256=raf_render.EMPTY_FILL_HASH)
    audit, warnings = raf_render.audit_miss_report(forged, depth, miss_mask=mask,
                                                   canonical=False)
    assert audit["mask_verified"] is False
    assert audit["audit_ok"] is False
    assert audit["miss_count_from_mask"] == 1


def test_audit_accepts_a_report_that_agrees_with_the_mask():
    depth = np.full((8, 16), 2.0, dtype=np.float32)
    coords = [[3, 4]]
    audit, _ = raf_render.audit_miss_report(
        _report_for(coords, 128, max_miss_rate=0.05), depth,
        miss_mask=_mask_with((8, 16), coords), canonical=False)
    assert audit["mask_verified"] is True and audit["audit_ok"] is True


@pytest.mark.parametrize("reported", [
    [[3, 4], [3, 4]],          # duplicated coordinate
    [[3, 4], [999, 0]],        # out of bounds
    [[1, 1]],                  # a different pixel entirely
    [],                        # none at all
])
def test_audit_rejects_coordinates_that_are_not_the_masks(reported):
    depth = np.full((8, 16), 2.0, dtype=np.float32)
    mask = _mask_with((8, 16), [(3, 4)])
    report = _report_for(reported, 128, max_miss_rate=0.05)
    audit, warnings = raf_render.audit_miss_report(report, depth, miss_mask=mask,
                                                   canonical=False)
    assert audit["audit_ok"] is False
    assert any("mask" in w for w in warnings)


def test_audit_rejects_a_declared_ray_count_that_is_not_the_maps():
    depth = np.full((8, 16), 2.0, dtype=np.float32)
    coords = [[3, 4]]
    report = _report_for(coords, 999, max_miss_rate=0.05)
    audit, warnings = raf_render.audit_miss_report(report, depth,
                                                   miss_mask=_mask_with((8, 16), coords),
                                                   canonical=False)
    assert audit["audit_ok"] is False
    assert any("ray count" in w for w in warnings)


def test_audit_rejects_a_mask_of_the_wrong_shape():
    depth = np.full((8, 16), 2.0, dtype=np.float32)
    audit, warnings = raf_render.audit_miss_report(
        _report_for([], 128, filled_pixels_sha256=raf_render.EMPTY_FILL_HASH), depth,
        miss_mask=np.zeros((4, 4), dtype=bool))
    assert audit["audit_ok"] is False
    assert any("mask is" in w for w in warnings)


def test_qa_enforces_the_registered_cap_from_the_mask():
    """The cap is applied to the rate DERIVED FROM THE MASK, so a report claiming
    within_cap for a 5% miss rate cannot pass."""
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    coords = [[r, c] for r in range(5) for c in range(100)]   # 500 of 10,000
    report = _report_for(coords, 10000, max_miss_rate=0.05)
    audit, _ = raf_render.audit_miss_report(report, depth,
                                            miss_mask=_mask_with((100, 100), coords),
                                            canonical=True)
    assert audit["miss_rate_recomputed"] == pytest.approx(0.05)
    assert audit["cap_applied"] == raf_render.DEFAULT_MAX_MISS_RATE
    assert audit["within_cap_recomputed"] is False
    assert audit["audit_ok"] is False


def test_depth_qa_requires_a_mask_verified_report():
    mesh = _box_with_pinhole(**_BOX)
    position = np.array([0.5, 0.5, 1.5])
    depth, report = raf_render.render_depth(mesh, position, return_report=True)
    assert raf_render.depth_qa(depth, position, miss_report=report)["passed"] is True

    stripped = {k: v for k, v in report.items() if k != "miss_mask"}
    qa = raf_render.depth_qa(depth, position, miss_report=stripped)
    assert qa["misses"]["mask_verified"] is False
    assert qa["passed"] is False


def test_cli_audits_every_map_against_its_raw_mask(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    for entry in qa["maps"].values():
        assert entry["misses"]["mask_verified"] is True
        assert entry["misses"]["audit_ok"] is True
        assert "miss_mask" not in entry["misses"]      # the mask is not serialised


# --------------------------------------------------------------------------- #
# r4 T5: what the render CAN and CANNOT detect, and the vertical gate
# --------------------------------------------------------------------------- #
def _transform_all(points, matrix):
    """Apply a candidate gauge to EVERYTHING -- mesh vertices and poses alike."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(matrix).T


_GAUGE_XZY = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])       # the pinned one
_GAUGE_XYZ = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])       # identity: y up
_GAUGE_ZXY = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])       # horizontal swap


def _room_under_gauge(matrix):
    """A room + tx + rx, all mapped by the SAME candidate gauge (T5)."""
    raf_bounds = dict(x0=-0.2, x1=5.0, y0=0.0, y1=4.0, z0=-2.0, z1=6.0)
    mesh = _box_mesh_raf(to_pipeline=False, **raf_bounds)
    verts = _transform_all(np.asarray(mesh.vertices), matrix)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.triangles = mesh.triangles
    tx_raf = np.array([2.0, 1.5, 2.0])
    rx_raf = np.array([[4.5, 1.2, 5.0], [0.0, 2.0, -1.5], [3.0, 0.5, 0.0]])
    return out, _transform_all(tx_raf[None], matrix)[0], _transform_all(rx_raf, matrix)


def test_the_pinned_gauge_passes_every_render_check():
    mesh, tx, rx = _room_under_gauge(_GAUGE_XZY)
    depth = raf_render.render_depth(mesh, tx)
    qa = raf_render.real_mesh_qa(depth, tx, mesh, rx_positions_p=rx,
                                 tracked_height_m=1.5)
    assert qa["passed"] is True
    assert qa["vertical_axis"]["ok"] is True
    assert qa["rx_sightline"]["passed"] is True


def test_a_wrong_vertical_axis_is_detected_even_when_applied_consistently():
    """T5: the vertical axis IS gauge-discriminating, because the tracked height
    comes from the pose files and the nadir distance comes from the render."""
    mesh, tx, rx = _room_under_gauge(_GAUGE_XYZ)   # RAF Y stays in slot 1, not 2
    depth = raf_render.render_depth(mesh, tx)
    qa = raf_render.real_mesh_qa(depth, tx, mesh, rx_positions_p=rx,
                                 tracked_height_m=1.5)
    assert qa["vertical_axis"]["ok"] is False
    assert qa["passed"] is False
    assert any("vertical" in w for w in qa["warnings"])


def test_a_consistent_horizontal_permutation_is_render_undetectable():
    """T5, recorded honestly: swapping the two HORIZONTAL axes everywhere preserves
    distances, containment and visibility, so no render check can see it. The
    horizontal assignment is pinned by derivation, not by this evidence."""
    mesh, tx, rx = _room_under_gauge(_GAUGE_ZXY)
    depth = raf_render.render_depth(mesh, tx)
    qa = raf_render.real_mesh_qa(depth, tx, mesh, rx_positions_p=rx,
                                 tracked_height_m=1.5)
    assert qa["passed"] is True                     # it really does pass
    assert qa["detectability"]["horizontal_permutation"] == "undetectable by render"
    assert "derivation" in qa["detectability"]["horizontal_basis"]


def test_the_detectability_boundary_is_recorded_in_the_qa_record(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    boundary = qa["detectability"]
    assert boundary["vertical_axis"] == "gauge-discriminating (nadir vs tracked height)"
    assert boundary["horizontal_permutation"] == "undetectable by render"
    for entry in qa["maps"].values():
        assert entry["real_mesh"]["vertical_axis"]["ok"] is True


def test_vertical_gate_uses_the_tracked_height_not_the_camera_vector():
    """The height comes from the pose file; comparing the render against the same
    vector it was rendered from would be circular."""
    mesh, tx, rx = _room_under_gauge(_GAUGE_XZY)
    depth = raf_render.render_depth(mesh, tx)
    ok = raf_render.real_mesh_qa(depth, tx, mesh, tracked_height_m=1.5)
    assert ok["vertical_axis"]["ok"] is True
    wrong = raf_render.real_mesh_qa(depth, tx, mesh, tracked_height_m=3.0)
    assert wrong["vertical_axis"]["ok"] is False
    assert wrong["vertical_axis"]["tracked_height_m"] == 3.0


# --------------------------------------------------------------------------- #
# r5 finding 5: the vertical reference is the RAW RAF height, end to end
# --------------------------------------------------------------------------- #
def _write_gauge_fixture(tmp_path, matrix, room="EmptyRoom"):
    """A room + tx published under a CANDIDATE gauge, mesh and poses together.

    The mesh is written in RAF world coordinates and the runtime metadata carries
    the gauge-transformed tx (as prepare_data would under that gauge) plus the RAW
    RAF height, which no gauge touches.
    """
    raf_root = tmp_path / "raf"
    out = tmp_path / "runtime" / "RAF"
    mesh_dir = raf_root / "3d_models" / room
    mesh_dir.mkdir(parents=True)
    raf_bounds = dict(x0=-6.0, x1=6.0, y0=0.0, y1=3.0, z0=-4.0, z1=4.0)
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.obj"),
                               _box_mesh_raf(to_pipeline=False, **raf_bounds))
    tx_raf = np.array([1.0, 1.2, 2.0])           # RAF: height is 1.2 m
    tx_p = np.asarray(matrix) @ tx_raf
    key = "cccc000000000003"
    meta_dir = out / room / "metadata"
    meta_dir.mkdir(parents=True)
    with open(meta_dir / "groups_metadata.json", "w") as f:
        json.dump({key: {"tx_xyz_p": [float(v) for v in tx_p],
                         "tx_height_raf_m": float(tx_raf[1]),
                         "depth_file": f"{key}_depth_image.npy",
                         "train_ids": ["000000"], "role": "train_test"}}, f)
    with open(meta_dir / "poses_metadata.json", "w") as f:
        rx_raf = np.array([[5.0, 1.0, 3.0], [-5.0, 1.5, -3.0]])
        json.dump({f"{i:06d}": {"tx_xyz_p": [float(v) for v in tx_p],
                                "quat_raw": [0, 0, 0, 1],
                                "rx_p": [float(v) for v in (np.asarray(matrix) @ r)],
                                "group_key": key, "split_role": "train"}
                   for i, r in enumerate(rx_raf)}, f)
    return raf_root, out


def _render_under_gauge(tmp_path, matrix, readback):
    """Run the real CLI with the mesh transformed by the same candidate gauge."""
    raf_root, out = _write_gauge_fixture(tmp_path, matrix)
    import raf_common

    original = raf_common.RAF_TO_PIPELINE.copy()
    raf_render.RAF_TO_PIPELINE = np.asarray(matrix, dtype=np.float64)
    try:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record", readback,
                         "--non-canonical"])
    finally:
        raf_render.RAF_TO_PIPELINE = original
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        return json.load(f)


def test_cli_vertical_gate_passes_under_the_pinned_gauge(tmp_path):
    qa = _render_under_gauge(tmp_path, _GAUGE_XZY, _readback(tmp_path))
    entry = next(iter(qa["maps"].values()))
    vertical = entry["real_mesh"]["vertical_axis"]
    assert vertical["checked"] is True
    assert vertical["tracked_height_m"] == 1.2         # the RAW RAF Y
    assert vertical["ok"] is True
    assert entry["real_mesh"]["passed"] is True


def test_cli_vertical_gate_catches_a_candidate_gauge_through_production_wiring(tmp_path):
    """r5 finding 5: mesh AND poses transformed by the same wrong gauge, run
    through the actual CLI. The r4 test hand-fed a raw 1.5 and never exercised
    this path; production fed back the transformed height, which cannot disagree."""
    with pytest.raises(RuntimeError) as exc:
        _render_under_gauge(tmp_path, _GAUGE_XYZ, _readback(tmp_path))
    assert "failed QA" in str(exc.value)


def test_cli_refuses_metadata_without_the_raw_height(tmp_path):
    raf_root, out, _ = _write_fixture(tmp_path)
    meta = out / "EmptyRoom" / "metadata" / "groups_metadata.json"
    payload = json.loads(meta.read_text())
    for entry in payload.values():
        entry.pop("tx_height_raf_m")
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as exc:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                         "--non-canonical"])
    assert "tx_height_raf_m" in str(exc.value)


# --------------------------------------------------------------------------- #
# r5 finding 2: the canonical RENDER identity
# --------------------------------------------------------------------------- #
def _render_args(**overrides):
    import argparse

    values = {"rooms": ["EmptyRoom", "FurnishedRoom"], "img_h": 256, "img_w": 512,
              "floor_tol": raf_render.DEFAULT_FLOOR_TOL,
              "max_miss_rate": raf_render.DEFAULT_MAX_MISS_RATE}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_canonical_render_identity_is_the_registered_one():
    assert raf_render.CANONICAL_RENDER_PARAMS == {
        "rooms": ("EmptyRoom", "FurnishedRoom"), "img_h": 256, "img_w": 512,
        "floor_tol": 0.15, "max_miss_rate": 0.001,
    }
    assert raf_render.assert_canonical_render(_render_args()) == []


@pytest.mark.parametrize("overrides,needle", [
    ({"rooms": ["FurnishedRoom"]}, "rooms"),
    ({"rooms": ["EmptyRoom"]}, "rooms"),
    ({"floor_tol": 5.0}, "floor_tol"),
    ({"img_h": 128}, "img_h"),
    ({"img_w": 256}, "img_w"),
    ({"max_miss_rate": 0.05}, "max_miss_rate"),
])
def test_canonical_render_rejects_deviations(overrides, needle):
    """A Furnished-only render skipped EmptyRoom's unconditional sightline gate,
    and a loose --floor-tol disabled the vertical gate entirely."""
    with pytest.raises(ValueError) as exc:
        raf_render.assert_canonical_render(_render_args(**overrides))
    assert needle in str(exc.value)
    assert "--non-canonical" in str(exc.value)


def test_a_stricter_miss_cap_is_still_canonical():
    assert raf_render.canonical_render_deviations(
        _render_args(max_miss_rate=0.0001)) == []


def test_render_parameters_join_the_depth_marker(tmp_path):
    import publish as raf_publish

    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / raf_publish.marker_name("depth")) as f:
        marker = json.load(f)
    assert marker["parameters"]["rooms"] == ["EmptyRoom"]
    assert marker["parameters"]["floor_tol"] == raf_render.DEFAULT_FLOOR_TOL
    assert marker["canonical"] is False
    assert marker["canonical_parameters"] is False
    assert marker["readback_record"]["sha256"]
