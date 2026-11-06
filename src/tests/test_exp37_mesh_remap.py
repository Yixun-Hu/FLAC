"""exp_37 round B-a item 3: the ``--mesh-root`` remap (plan_loc_efficiency.md §3, §10 "Provenance").

The geometry audit binds the other host's absolute ``mesh_path`` (``<audit mesh_root>/<Scene>/<Room>.obj``). The
remap resolves every audited room to ``<--mesh-root>/<Scene>/<Room>.obj`` and refuses unless each local mesh hashes
to the audit's ``mesh_sha256``; the audit dict itself is never modified, so every hash binding (audit sha,
pilot binding, run identity) is untouched. The engine re-checks the bytes it actually loaded.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-a.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from src.localization.engine import reconstruct_room_base_candidates
from src.localization.geometry import build_lattice, classify_mesh_candidates, load_raycast_scene
from src.localization.mesh_remap import resolve_mesh_root

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/geometry_audit.json"
OTHER_HOST = "/home/someone/elsewhere/AcousticRooms/room_mesh_obj_format"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _box(path: Path, size: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_triangle_mesh(str(path), o3d.geometry.TriangleMesh.create_box(size, size, size),
                                      write_ascii=True)
    return path


@pytest.fixture
def two_rooms(tmp_path):
    """A local mesh root with two rooms and an audit binding the other host's absolute paths."""
    root = tmp_path / "meshes"
    a = _box(root / "Bathrooms" / "Bathrooms_idx_14.obj", 2.0)
    b = _box(root / "Cafe" / "Cafe_idx_1.obj", 3.0)
    audit = {"mesh_root": OTHER_HOST, "rooms": {
        "Bathrooms_idx_14": {"scene": "Bathrooms", "mesh_path": f"{OTHER_HOST}/Bathrooms/Bathrooms_idx_14.obj",
                             "mesh_sha256": _sha(a)},
        "Cafe_idx_1": {"scene": "Cafe", "mesh_path": f"{OTHER_HOST}/Cafe/Cafe_idx_1.obj", "mesh_sha256": _sha(b)}}}
    return root, audit


def test_every_room_is_resolved_under_the_new_root_with_its_digest(two_rooms):
    root, audit = two_rooms
    before = copy.deepcopy(audit)
    got = resolve_mesh_root(audit, root)
    assert audit == before                                              # the audit is never modified
    assert got["argument"] == str(root) and got["resolved"] == str(root.resolve())
    assert got["audit_mesh_root"] == OTHER_HOST
    assert got["rooms"] == {
        "Bathrooms_idx_14": {"mesh_path": str((root / "Bathrooms" / "Bathrooms_idx_14.obj").resolve()),
                             "mesh_sha256": audit["rooms"]["Bathrooms_idx_14"]["mesh_sha256"]},
        "Cafe_idx_1": {"mesh_path": str((root / "Cafe" / "Cafe_idx_1.obj").resolve()),
                       "mesh_sha256": audit["rooms"]["Cafe_idx_1"]["mesh_sha256"]}}


def test_a_mesh_whose_bytes_differ_from_the_audit_is_refused(two_rooms):
    root, audit = two_rooms
    _box(root / "Cafe" / "Cafe_idx_1.obj", 3.5)
    with pytest.raises(RuntimeError, match="Cafe_idx_1.*SHA-256"):
        resolve_mesh_root(audit, root)


def test_a_missing_mesh_or_root_is_refused(two_rooms, tmp_path):
    root, audit = two_rooms
    (root / "Cafe" / "Cafe_idx_1.obj").unlink()
    with pytest.raises(RuntimeError, match="Cafe_idx_1"):
        resolve_mesh_root(audit, root)
    with pytest.raises(RuntimeError, match="not a directory"):
        resolve_mesh_root(audit, tmp_path / "absent")


@pytest.mark.parametrize("bad_path", [
    "/home/someone/other_root/Bathrooms/Bathrooms_idx_14.obj",      # another prefix than the audit's mesh_root
    f"{OTHER_HOST}/Cafe/Bathrooms_idx_14.obj",                      # the wrong scene directory
    f"{OTHER_HOST}/Bathrooms/Bathrooms_idx_18.obj",                 # another room's file name
    f"{OTHER_HOST}/Bathrooms/sub/Bathrooms_idx_14.obj"])            # an extra level
def test_an_audit_path_outside_the_scene_room_layout_is_refused(two_rooms, bad_path):
    root, audit = two_rooms
    audit["rooms"]["Bathrooms_idx_14"]["mesh_path"] = bad_path
    with pytest.raises(RuntimeError, match="layout"):
        resolve_mesh_root(audit, root)


def test_a_relative_audit_mesh_root_is_refused(two_rooms):
    root, audit = two_rooms
    audit["mesh_root"] = "AcousticRooms/room_mesh_obj_format"
    with pytest.raises(RuntimeError, match="absolute"):
        resolve_mesh_root(audit, root)


def _room_audit(mesh: Path, recorded_path: str) -> dict:
    """An audit row for one box room whose base grid hash is computed exactly as the audit tool does."""
    scene = load_raycast_scene(mesh, compute_topology=False)
    raw = build_lattice(scene.aabb_min, scene.aabb_max, [0.5, 0.5, 0.5])
    mask, _ = classify_mesh_candidates(scene, raw, 0.2, eps=1e-4)
    return {"grid_spacing_m": [0.5, 0.5, 0.5], "surface_clearance_m": 0.2, "epsilon_m": 1e-4, "rooms": {"R": {
        "mesh_path": recorded_path, "mesh_sha256": _sha(mesh),
        "base_points_sha256": hashlib.sha256(raw[mask].astype("<f8").tobytes()).hexdigest()}}}


def test_the_engine_loads_the_remapped_file_and_rechecks_its_bytes(tmp_path):
    mesh = _box(tmp_path / "local" / "S" / "R.obj", 2.0)
    audit = _room_audit(mesh, f"{OTHER_HOST}/S/R.obj")
    with pytest.raises(FileNotFoundError):                              # the unmapped path is the other host's
        reconstruct_room_base_candidates("R", audit)
    base = reconstruct_room_base_candidates("R", audit, mesh_path=mesh)
    assert base.ndim == 2 and base.shape[1] == 3 and len(base) > 0
    audit["rooms"]["R"]["mesh_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="remapped mesh SHA-256"):
        reconstruct_room_base_candidates("R", audit, mesh_path=mesh)


def test_the_default_engine_path_is_unchanged(tmp_path):
    """Without an override the audit's own path is used and no mesh digest is demanded (the bundle's behaviour)."""
    mesh = _box(tmp_path / "S" / "R.obj", 2.0)
    audit = _room_audit(mesh, str(mesh))
    audit["rooms"]["R"]["mesh_sha256"] = "not checked on the default path"
    assert np.array_equal(reconstruct_room_base_candidates("R", audit),
                          reconstruct_room_base_candidates("R", {**audit, "rooms": {"R": {
                              **audit["rooms"]["R"], "mesh_sha256": _sha(mesh)}}}, mesh_path=mesh))


def test_the_real_audit_resolves_on_this_box_when_the_dataset_is_linked():
    """All 16 meshes of the committed exp_09 audit under AcousticRooms/room_mesh_obj_format (skipped without the
    dataset link); the audit file's bytes are untouched."""
    local = REPO / "AcousticRooms" / "room_mesh_obj_format"
    if not local.is_dir():
        pytest.skip("AcousticRooms is not linked in this checkout")
    before = AUDIT.read_bytes()
    got = resolve_mesh_root(json.loads(before), local)
    assert len(got["rooms"]) == 16 and AUDIT.read_bytes() == before
