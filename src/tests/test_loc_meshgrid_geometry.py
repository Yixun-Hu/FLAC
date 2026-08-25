"""exp_22 G1 -- mesh-valid candidate geometry (inherited plan §1.2/§1.3).

The validity rule is deliberately parity-over-31-frozen-directions plus a
separate surface-clearance prior, because the official meshes are neither
watertight nor manifold and a single ray (or Open3D's own occupancy) inherits
that fragility. These tests build synthetic rooms whose answers are known by
construction, and a real Cafe smoke runs the whole predicate over the metadata
anchors that must all be valid.
"""
import json
import os

import numpy as np
import pytest

from src.localization import meshgrid_geometry as mg
from src.localization import meshgrid_queries as mq

_CAFE_OBJ = ("/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/"
             "Cafe/Cafe_idx_1.obj")
_AR_METADATA = "AcousticRooms/metadata"


# --------------------------------------------------------------------------- #
# the lattice: room-global, exact, lexicographically stable
# --------------------------------------------------------------------------- #
def test_axis_snap_is_the_inclusive_multiples_of_the_spacing():
    assert mg.snap_axis_to_lattice(0.2, 1.7, 0.5).tolist() == [0.5, 1.0, 1.5]
    assert mg.snap_axis_to_lattice(0.0, 1.0, 0.5).tolist() == [0.0, 0.5, 1.0]
    # negative AABBs snap the same way (ceil below, floor above)
    assert mg.snap_axis_to_lattice(-1.2, -0.1, 0.5).tolist() == [-1.0, -0.5]
    assert mg.snap_axis_to_lattice(-0.6, 0.6, 0.5).tolist() == [-0.5, 0.0, 0.5]
    assert mg.snap_axis_to_lattice(0.6, 0.9, 0.5).tolist() == []      # no multiple inside


def test_lattice_is_room_global_and_lexicographically_ordered():
    lattice = mg.build_lattice([-0.6, 0.0, 0.2], [0.6, 1.0, 1.2], spacing=0.5)
    assert lattice.shape[1] == 3 and lattice.dtype == np.float64
    expected = [[x, y, z] for x in (-0.5, 0.0, 0.5) for y in (0.0, 0.5, 1.0)
                for z in (0.5, 1.0)]
    assert lattice.tolist() == expected
    assert mg.LATTICE_SPACING == 0.5

    # independent of any query: same AABB, same lattice, byte for byte
    again = mg.build_lattice([-0.6, 0.0, 0.2], [0.6, 1.0, 1.2], spacing=0.5)
    assert again.tobytes() == lattice.tobytes()


def test_lattice_refuses_a_degenerate_or_non_finite_aabb():
    for low, high in (([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
                      ([float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]),
                      ([0.0, 0.0, 0.0], [float("inf"), 1.0, 1.0])):
        with pytest.raises(ValueError):
            mg.build_lattice(low, high, spacing=0.5)
    with pytest.raises(ValueError, match="spacing"):
        mg.build_lattice([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], spacing=0.0)


# --------------------------------------------------------------------------- #
# the frozen direction set
# --------------------------------------------------------------------------- #
def test_directions_are_frozen_odd_and_never_axis_aligned():
    directions = mg.FROZEN_DIRECTIONS
    assert directions.shape == (31, 3) and directions.dtype == np.float64
    assert len(directions) % 2 == 1, "an even vote could tie"
    norms = np.linalg.norm(directions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)
    # no direction may lie in a coordinate plane: axis-aligned rays graze the
    # axis-aligned triangles these rooms are built from
    assert np.abs(directions).min() > 1e-3
    # deterministic, and keyed by the SELECTED seed
    assert mg.build_directions(31, seed=mg.FROZEN_DIRECTIONS_SEED).tobytes() == \
        directions.tobytes()
    assert mg.build_directions(31, seed=0).tobytes() != directions.tobytes()
    assert mg.build_directions(33, seed=mg.FROZEN_DIRECTIONS_SEED).tobytes() != \
        directions.tobytes()


# --------------------------------------------------------------------------- #
# a synthetic room: shell with an obstacle inside
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def shell_room(tmp_path_factory):
    """A 4x4x3 m hollow shell containing a 1 m solid cube at (2,2,0.5)."""
    path = tmp_path_factory.mktemp("mesh") / "shell.obj"
    mg.write_box_obj(str(path), boxes=[((0.0, 0.0, 0.0), (4.0, 4.0, 3.0)),
                                       ((1.5, 1.5, 0.0), (2.5, 2.5, 1.0))])
    return str(path)


def test_scene_loads_with_identity_metadata(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    assert scene.identity["path"] == shell_room
    assert len(scene.identity["sha256"]) == 64
    assert scene.identity["n_triangles"] > 0 and scene.identity["n_vertices"] > 0
    assert len(scene.identity["aabb_min"]) == 3
    assert scene.identity["aabb_min"][0] == pytest.approx(0.0)
    assert scene.identity["aabb_max"][2] == pytest.approx(3.0)
    assert "watertight" in scene.identity and "backend" in scene.identity
    assert scene.identity["backend"].startswith("open3d")


def test_parity_distinguishes_air_solid_and_outside(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    points = np.array([
        [0.5, 0.5, 1.5],       # air inside the shell
        [3.5, 3.5, 2.5],       # air inside, other corner
        [2.0, 2.0, 0.5],       # INSIDE the solid obstacle
        [-1.0, 2.0, 1.5],      # outside the room
        [2.0, 2.0, 5.0],       # above the room
    ])
    inside = mg.classify_free_space(scene, points)
    assert inside.tolist() == [True, True, False, False, False]


def test_surface_clearance_prior_is_a_separate_020_boundary(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    # x-distance to the obstacle face at x = 1.5
    points = np.array([[1.5 - 0.30, 2.0, 0.5],      # 0.30 m clear -> keep
                       [1.5 - 0.20, 2.0, 0.5],      # exactly 0.20 -> keep (eps)
                       [1.5 - 0.19, 2.0, 0.5],      # 0.19 m -> reject
                       [1.5 - 0.05, 2.0, 0.5]])     # hugging the wall -> reject
    verdict = mg.classify_mesh_candidates(scene, points)
    assert verdict["parity_valid"].tolist() == [True, True, True, True]
    assert verdict["clearance_valid"].tolist() == [True, True, False, False]
    assert verdict["valid"].tolist() == [True, True, False, False]
    assert verdict["clearance"] == pytest.approx(0.20)
    assert verdict["eps"] == pytest.approx(1e-4)
    assert np.all(np.isfinite(verdict["distance"]))


def test_mesh_validity_is_deterministic_under_chunking(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    lattice = mg.build_lattice(*mg.scene_aabb(scene), spacing=0.5)
    whole = mg.classify_mesh_candidates(scene, lattice)
    chunked_a = mg.classify_mesh_candidates(scene, lattice, chunk=7)
    chunked_b = mg.classify_mesh_candidates(scene, lattice, chunk=1024)
    for key in ("valid", "parity_valid", "clearance_valid"):
        assert whole[key].tobytes() == chunked_a[key].tobytes() == chunked_b[key].tobytes()
    assert whole["distance"].tobytes() == chunked_a["distance"].tobytes()


def test_missing_or_malformed_mesh_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        mg.load_raycast_scene(str(tmp_path / "absent.obj"))
    broken = tmp_path / "broken.obj"
    broken.write_text("this is not an OBJ\n")
    with pytest.raises(ValueError, match="triangle|parse"):
        mg.load_raycast_scene(str(broken))
    empty = tmp_path / "empty.obj"
    empty.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n")      # vertices, no faces
    with pytest.raises(ValueError, match="triangle"):
        mg.load_raycast_scene(str(empty))


# --------------------------------------------------------------------------- #
# per-query filters
# --------------------------------------------------------------------------- #
def _grid():
    return mg.build_lattice([0.0, 0.0, 0.0], [2.0, 2.0, 1.5], spacing=0.5)


def test_receiver_guard_removes_only_the_close_candidates():
    grid = _grid()
    receiver = np.array([1.0, 1.0, 1.0])
    kept = mg.filter_query_candidates(grid, receiver=receiver, context_sources=[])
    distances = np.linalg.norm(kept["candidates"] - receiver, axis=1)
    assert distances.min() + 1e-4 >= mg.RECEIVER_MIN_DISTANCE
    assert kept["n_dropped_receiver"] == int((np.linalg.norm(grid - receiver, axis=1)
                                              + 1e-4 < mg.RECEIVER_MIN_DISTANCE).sum())
    assert kept["n_dropped_receiver"] > 0


def test_context_duplicate_guard_is_a_025_ball():
    grid = _grid()
    receiver = np.array([5.0, 5.0, 5.0])                  # far away, guard inert
    context = [np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])]
    kept = mg.filter_query_candidates(grid, receiver=receiver, context_sources=context)
    for source in context:
        assert np.linalg.norm(kept["candidates"] - source, axis=1).min() > 0.25
    assert kept["n_dropped_context"] == 2                 # the two coincident lattice nodes
    assert mg.CONTEXT_GUARD_RADIUS == 0.25

    near = [np.array([1.0 + 0.24, 1.0, 1.0])]             # inside the ball, not on a node
    assert mg.filter_query_candidates(grid, receiver=receiver,
                                      context_sources=near)["n_dropped_context"] == 1
    # exactly on the boundary between two nodes: BOTH are 0.25 away, both drop
    boundary = [np.array([1.25, 1.0, 1.0])]
    assert mg.filter_query_candidates(grid, receiver=receiver,
                                      context_sources=boundary)["n_dropped_context"] == 2
    # a cell centre is sqrt(3)/4 = 0.433 m from every node -- the only place on a
    # 0.5 m lattice where a context can sit with the guard dropping nothing
    centre = [np.array([1.25, 1.25, 1.25])]
    assert mg.filter_query_candidates(grid, receiver=receiver,
                                      context_sources=centre)["n_dropped_context"] == 0


def test_z_band_is_derived_from_contexts_only_and_intersects_the_lattice():
    grid = _grid()
    receiver = np.array([5.0, 5.0, 5.0])
    context = [np.array([0.0, 0.0, 0.5]), np.array([1.0, 1.0, 1.0])]
    band = mg.context_z_band(context)
    assert band == (pytest.approx(0.0), pytest.approx(1.5))
    kept = mg.filter_query_candidates(grid, receiver=receiver, context_sources=context,
                                      z_band=band)
    assert kept["candidates"][:, 2].min() >= 0.0 - 1e-9
    assert kept["candidates"][:, 2].max() <= 1.5 + 1e-9
    assert kept["z_band"] == [pytest.approx(0.0), pytest.approx(1.5)]

    narrow = mg.filter_query_candidates(grid, receiver=receiver, context_sources=context,
                                        z_band=(0.9, 1.1))
    assert set(np.round(narrow["candidates"][:, 2], 6)) == {1.0}
    full = mg.filter_query_candidates(grid, receiver=receiver, context_sources=context)
    assert full["z_band"] is None and len(full["candidates"]) >= len(narrow["candidates"])


def test_z_band_branch_rule_falls_back_to_full_height():
    """Use the band globally only if every query stays nonempty AND it adds no
    e_oracle > 0.5 m query; otherwise full height, globally."""
    full_height = {"q0": 0.10, "q1": 0.40}
    band_ok = {"q0": 0.10, "q1": 0.45}
    band_damaging = {"q0": 0.10, "q1": 0.60}
    assert mg.choose_z_branch(full_height, band_ok, band_nonempty=True)["branch"] == "z_band"
    damaged = mg.choose_z_branch(full_height, band_damaging, band_nonempty=True)
    assert damaged["branch"] == "full_height"
    assert damaged["n_new_over_threshold"] == 1
    empty = mg.choose_z_branch(full_height, band_ok, band_nonempty=False)
    assert empty["branch"] == "full_height" and "nonempty" in empty["reason"]


def test_filters_never_insert_the_ground_truth():
    grid = _grid()
    truth = np.array([0.13, 1.77, 0.42])                  # not on the lattice
    kept = mg.filter_query_candidates(grid, receiver=np.array([5.0, 5.0, 5.0]),
                                      context_sources=[])
    assert not np.any(np.all(np.isclose(kept["candidates"], truth, atol=1e-9), axis=1))
    assert mg.grid_oracle_error(kept["candidates"], truth) > 0.0


def test_oracle_error_is_the_nearest_candidate_distance():
    candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert mg.grid_oracle_error(candidates, np.array([0.9, 0.0, 0.0])) == pytest.approx(0.1)
    assert mg.grid_oracle_error(candidates, np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="nonempty"):
        mg.grid_oracle_error(np.zeros((0, 3)), np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        mg.grid_oracle_error(candidates, np.array([float("nan"), 0.0, 0.0]))


def test_query_candidates_refuse_an_empty_result():
    grid = _grid()
    with pytest.raises(ValueError, match="nonempty"):
        mg.filter_query_candidates(grid, receiver=np.array([1.0, 1.0, 1.0]),
                                   context_sources=[], z_band=(9.0, 10.0))


# --------------------------------------------------------------------------- #
# the real Cafe mesh: every metadata anchor must be valid
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isfile(_CAFE_OBJ), reason="Cafe OBJ not present")
def test_real_cafe_mesh_loads_and_accepts_every_metadata_anchor():
    scene = mg.load_raycast_scene(_CAFE_OBJ)
    assert scene.identity["n_triangles"] > 100
    anchors = mg.metadata_anchors(os.path.join(_AR_METADATA, "Cafe", "Cafe_idx_1"))
    assert len(anchors["sources"]) > 0 and len(anchors["receivers"]) > 0

    # §1.3 rule 2 holds for EVERY anchor: inside the free-space classification.
    for label in ("sources", "receivers"):
        points = np.asarray(anchors[label], dtype=np.float64)
        verdict = mg.classify_mesh_candidates(scene, points)
        assert bool(verdict["parity_valid"].all()), (
            f"{label}: {int((~verdict['parity_valid']).sum())} anchors classified outside "
            "the room by the 31-direction parity vote")

    # §1.3 rule 3 names the SOURCE anchors: they must also survive the candidate
    # predicate, i.e. the 0.20 m source-distribution prior. Receivers are not
    # drawn from that distribution and are not candidates -- their own constraint
    # is the >= 0.5 m candidate-distance guard -- and on the real Cafe mesh two
    # receiver anchors do sit 0.100 m from a surface, so applying the source
    # prior to them would block a room the plan admits (flagged in the report).
    sources = np.asarray(anchors["sources"], dtype=np.float64)
    verdict = mg.classify_mesh_candidates(scene, sources)
    assert bool(verdict["valid"].all()), (
        f"sources: {int((~verdict['valid']).sum())} anchors fail parity + the 0.20 m "
        f"clearance prior (min distance {float(verdict['distance'].min()):.3f} m)")
    assert mg.audit_room_anchors(scene, anchors)["accepted"] is True


# --------------------------------------------------------------------------- #
# r2 F3 -- the direction set is pinned as literals, not regenerated
# --------------------------------------------------------------------------- #
def test_direction_set_is_a_literal_constant_with_a_pinned_digest():
    """r1 review F3: comparing the generator with its own output is not a pin."""
    import hashlib

    assert isinstance(mg.FROZEN_DIRECTIONS_LITERAL, tuple)
    assert len(mg.FROZEN_DIRECTIONS_LITERAL) == 31
    assert all(len(row) == 3 for row in mg.FROZEN_DIRECTIONS_LITERAL)
    digest = hashlib.sha256(mg.FROZEN_DIRECTIONS.tobytes()).hexdigest()
    assert digest == mg.FROZEN_DIRECTIONS_SHA256
    # the SELECTED set (seed 1), not the first one generated
    assert digest == "79544f2dbc880a37a4826aa527d40e99a3e54ce849cfd0ec9f1c6e847c528a8d"
    assert mg.FROZEN_DIRECTIONS_SEED == 1
    # the generator is provenance only: it must still reproduce the literals, and
    # if it ever stops doing so the literals win and this test says so
    assert np.allclose(mg.build_directions(31, seed=mg.FROZEN_DIRECTIONS_SEED),
                       mg.FROZEN_DIRECTIONS, atol=0, rtol=0)
    assert not np.allclose(mg.build_directions(31, seed=0), mg.FROZEN_DIRECTIONS)


def test_scene_identity_records_the_pinned_digest(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    assert scene.identity["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
    assert scene.identity["directions_sha256_pinned"] == mg.FROZEN_DIRECTIONS_SHA256


def test_the_meeting_room_discrepancy_is_recorded_as_resolved():
    """Resolved BY THE SELECTION, with the old set's failure kept as provenance."""
    assert mg.KNOWN_PARITY_DISCREPANCIES == ()
    assert mg.known_discrepancy("MeetingRoom/MeetingRoom_idx_32", [2.26, 0.48, 1.2],
                                "receivers") is None
    entry = mg.RESOLVED_PARITY_DISCREPANCIES[0]
    assert entry["room_id"] == "MeetingRoom/MeetingRoom_idx_32"
    assert entry["odd_votes_under_previous_pin"] == 15
    assert entry["previous_seed"] == 0
    assert entry["previous_sha256"].startswith("9ab4339f")
    assert "resolved" in entry["status"]


_MEETING_OBJ = ("/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/"
                "MeetingRoom/MeetingRoom_idx_32.obj")


@pytest.mark.skipif(not os.path.isfile(_MEETING_OBJ), reason="MeetingRoom OBJ not present")
def test_meeting_room_anchor_passes_under_the_selected_set_and_failed_under_the_old_one():
    """Both halves of the resolution, on the real mesh: the anchor the previous
    set rejected at 15/31 is interior at 16/31 under the selected set, and the
    room is accepted."""
    scene = mg.load_raycast_scene(_MEETING_OBJ)
    point = np.array([[2.26, 0.48, 1.2]])

    previous = mg.build_directions(31, seed=0)
    assert int(mg.odd_parity_votes(scene, point, directions=previous)[0]) == 15

    votes = int(mg.odd_parity_votes(scene, point)[0])
    assert votes >= 16, f"the selected set must classify this anchor interior, got {votes}/31"
    assert bool(mg.classify_free_space(scene, point)[0]) is True
    assert float(mg.surface_distance(scene, point)[0]) == pytest.approx(0.25005, abs=5e-4)

    audit = mg.audit_room_anchors(
        scene, mg.metadata_anchors("AcousticRooms/metadata/MeetingRoom/MeetingRoom_idx_32"),
        room_id="MeetingRoom/MeetingRoom_idx_32")
    assert audit["accepted"] is True, audit["rules"]


# --------------------------------------------------------------------------- #
# r2 F4 -- choose_z_branch is fail-closed about its inputs
# --------------------------------------------------------------------------- #
def test_branch_rule_requires_the_same_query_set_on_both_sides():
    full = {"q0": 0.1, "q1": 0.2}
    with pytest.raises(ValueError, match="same queries|query set"):
        mg.choose_z_branch(full, {"q0": 0.1}, band_nonempty=True)
    with pytest.raises(ValueError, match="same queries|query set"):
        mg.choose_z_branch(full, {"q0": 0.1, "q2": 0.2}, band_nonempty=True)
    with pytest.raises(ValueError, match="empty"):
        mg.choose_z_branch({}, {}, band_nonempty=True)


def test_branch_rule_requires_finite_oracles_and_never_defaults():
    """A missing full-height entry used to default to 0.0, which made the band
    look like it created a regression it did not (r1 review F4)."""
    full = {"q0": 0.1, "q1": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        mg.choose_z_branch(full, {"q0": 0.1, "q1": 0.2}, band_nonempty=True)
    with pytest.raises(ValueError, match="finite"):
        mg.choose_z_branch({"q0": 0.1, "q1": float("inf")}, {"q0": 0.1, "q1": 0.2},
                           band_nonempty=True)
    with pytest.raises(ValueError, match="finite"):
        mg.choose_z_branch({"q0": 0.1, "q1": 0.2}, {"q0": 0.1, "q1": float("nan")},
                           band_nonempty=True)
    # +inf on the BAND side is meaningful: an empty z-band set disqualifies the
    # branch instead of being replaced by its full-height value (r3 F3(b))
    verdict = mg.choose_z_branch({"q0": 0.1, "q1": 0.2},
                                 {"q0": 0.1, "q1": float("inf")}, band_nonempty=True)
    assert verdict["branch"] == "full_height" and verdict["n_empty_band"] == 1


def test_branch_rule_decides_on_new_regressions_only():
    full = {"q0": 0.10, "q1": 0.60, "q2": 0.20}
    # q1 was already over threshold at full height: the band does not "create" it
    band = {"q0": 0.10, "q1": 0.70, "q2": 0.20}
    assert mg.choose_z_branch(full, band, band_nonempty=True)["branch"] == "z_band"
    worse = {"q0": 0.10, "q1": 0.70, "q2": 0.55}
    verdict = mg.choose_z_branch(full, worse, band_nonempty=True)
    assert verdict["branch"] == "full_height" and verdict["n_new_over_threshold"] == 1
    assert verdict["queries"] == ["q2"]


# --------------------------------------------------------------------------- #
# r2 F4 -- the 16-room audit driver
# --------------------------------------------------------------------------- #
def _audit_module():
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
            "exp_22_loc_meshgrid_claude" / "audit_meshgrid_geometry.py")
    spec = importlib.util.spec_from_file_location("audit_meshgrid_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_world(tmp_path, rooms=("RoomA/RoomA_idx_1", "RoomB/RoomB_idx_2")):
    """A miniature world: one OBJ, one metadata dir and one context manifest."""
    mesh_root = tmp_path / "meshes"
    metadata_root = tmp_path / "metadata"
    records = []
    for position, room in enumerate(rooms):
        scene, scene_id = room.split("/")
        (mesh_root / scene).mkdir(parents=True, exist_ok=True)
        mg.write_box_obj(str(mesh_root / scene / f"{scene_id}.obj"),
                         boxes=[((0.0, 0.0, 0.0), (4.0, 4.0, 3.0))])
        directory = metadata_root / scene / scene_id
        directory.mkdir(parents=True, exist_ok=True)
        receiver = [2.0, 2.0, 1.5]
        sources = [[1.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 1.5]]
        for index, source in enumerate(sources, start=1):
            (directory / f"S00{index}_R008.json").write_text(
                json.dumps({"src_loc": source, "rec_loc": receiver}))
        # one query per room: target = source 1, contexts = sources 2 and 3
        relative = [[s[i] - receiver[i] for i in range(3)] for s in sources[1:]]
        records.append({
            "position": position,
            "query_id": f"{position}|single_channel_ir_1/{room}/S001_R008_hybrid_IR.wav",
            "room_id": room,
            "relpath": f"single_channel_ir_1/{room}/S001_R008_hybrid_IR.wav",
            "context_fingerprints": [",".join(f"{v:.6f}" for v in row) for row in relative],
            "context_audio_sha256": ["0" * 64] * 2,
            "context_width": 2, "eligible": 2, "target_absent": True})
    manifest = {"records": records, "n_filtered": len(records),
                "filtered_stream_sha256": "f" * 64}
    return {"mesh_root": str(mesh_root), "metadata_root": str(metadata_root),
            "manifest": manifest}


def test_audit_resolves_exactly_one_mesh_per_room(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    resolved = audit.resolve_room_meshes(["RoomA/RoomA_idx_1"], world["mesh_root"])
    assert list(resolved) == ["RoomA/RoomA_idx_1"]
    assert resolved["RoomA/RoomA_idx_1"].endswith("RoomA_idx_1.obj")

    with pytest.raises(ValueError, match="no OBJ|not found"):
        audit.resolve_room_meshes(["RoomZ/RoomZ_idx_9"], world["mesh_root"])

    duplicate = os.path.join(world["mesh_root"], "RoomA", "RoomA_idx_1_copy.obj")
    mg.write_box_obj(duplicate, boxes=[((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    os.rename(duplicate, os.path.join(world["mesh_root"], "RoomA", "RoomA_idx_1.OBJ"))
    with pytest.raises(ValueError, match="exactly one|ambiguous"):
        audit.resolve_room_meshes(["RoomA/RoomA_idx_1"], world["mesh_root"])


def test_audit_recovers_global_context_coordinates(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    record = world["manifest"]["records"][0]
    anchors = mg.metadata_anchors(os.path.join(world["metadata_root"], "RoomA",
                                               "RoomA_idx_1"))
    globals_ = audit.resolve_context_globals(record, receiver=[2.0, 2.0, 1.5],
                                             anchors=anchors)
    assert len(globals_) == 2
    assert np.allclose(sorted(map(list, globals_)), [[1.0, 3.0, 1.5], [3.0, 3.0, 1.0]])

    with pytest.raises(ValueError, match="anchor"):
        audit.resolve_context_globals(record, receiver=[9.0, 9.0, 9.0], anchors=anchors)


def test_audit_runs_end_to_end_on_the_fixture_world(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    out = tmp_path / "out"
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"], out_dir=str(out),
                             expected_queries=2, required_rooms=rooms,
                             _allow_fixture_census=True)

    assert report["n_rooms"] == 2 and report["n_queries"] == 2
    assert report["branch"]["branch"] in ("z_band", "full_height")
    assert set(report["oracle"]) == {"full_height", "z_band"}
    for branch in ("full_height", "z_band"):
        block = report["oracle"][branch]
        assert block["n_queries"] == 2 and np.isfinite(block["median"])
        assert block["n_over_threshold"] >= 0
    # the gate counts are per branch (r3 F4c); the dedicated test checks them
    cost = report["cost"]["full_height"]
    for key in ("candidate_query_pairs", "unique_receiver_candidate_pairs",
                "conditioner_calls_estimate", "artifact_bytes"):
        assert key in cost and cost[key] > 0
    assert report["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
    assert report["context_manifest_sha256"] == "f" * 64

    for room in ("RoomA/RoomA_idx_1", "RoomB/RoomB_idx_2"):
        entry = report["rooms"][room]
        assert entry["mesh"]["sha256"] and entry["n_queries"] == 1
        assert entry["candidate_manifest_sha256"]
        path = os.path.join(str(out), entry["candidate_manifest"])
        assert os.path.isfile(path)
        payload = json.load(open(path))
        assert payload["room_id"] == room
        assert payload["spacing"] == mg.LATTICE_SPACING
        assert payload["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
        assert len(payload["queries"]) == 1
        query = payload["queries"][0]
        assert query["n_candidates"] > 0
        assert np.isfinite(query["oracle"]["full_height"])
        assert "exclusion" in payload and payload["exclusion"]["room_id"]


def test_audit_refuses_a_wrong_query_count(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    with pytest.raises(ValueError, match="5,337|5337|expected"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"],
                        out_dir=str(tmp_path / "o2"), expected_queries=5337,
                        required_rooms=None, _allow_fixture_census=True)


def test_audit_fails_closed_on_a_missing_mesh(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    os.remove(os.path.join(world["mesh_root"], "RoomB", "RoomB_idx_2.obj"))
    with pytest.raises(ValueError, match="RoomB"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"],
                        out_dir=str(tmp_path / "o3"), expected_queries=2,
                        required_rooms=None, _allow_fixture_census=True)


# --------------------------------------------------------------------------- #
# r3 F3/F4 -- the audit refuses BEFORE it writes anything
# --------------------------------------------------------------------------- #
def test_required_room_set_is_pinned_as_a_literal():
    audit = _audit_module()
    assert len(audit.REQUIRED_ROOMS) == 16
    assert isinstance(audit.REQUIRED_ROOMS, tuple)
    assert "Cafe/Cafe_idx_1" in audit.REQUIRED_ROOMS
    assert "MeetingRoom/MeetingRoom_idx_32" in audit.REQUIRED_ROOMS
    assert mq.EXCLUDED_ROOM not in audit.REQUIRED_ROOMS
    assert sorted(audit.REQUIRED_ROOMS) == list(audit.REQUIRED_ROOMS)


def test_audit_refuses_a_room_set_that_is_not_the_required_sixteen(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    with pytest.raises(ValueError, match="room set|required"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"],
                        out_dir=str(tmp_path / "o"), expected_queries=2,
                        required_rooms=("RoomA/RoomA_idx_1", "RoomB/RoomB_idx_2",
                                        "RoomC/RoomC_idx_3"), _allow_fixture_census=True)
    assert not os.path.isdir(str(tmp_path / "o")) or os.listdir(str(tmp_path / "o")) == []


def test_audit_validates_the_record_stream(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))

    duped = dict(world["manifest"])
    duped["records"] = [world["manifest"]["records"][0]] * 2
    with pytest.raises(ValueError, match="duplicate|unique"):
        audit.run_audit(duped, mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(tmp_path / "a"),
                        expected_queries=2, required_rooms=rooms,
                             _allow_fixture_census=True)

    shuffled = dict(world["manifest"])
    shuffled["records"] = [dict(record, position=9)
                           for record in world["manifest"]["records"]]
    with pytest.raises(ValueError, match="position"):
        audit.run_audit(shuffled, mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(tmp_path / "b"),
                        expected_queries=2, required_rooms=rooms,
                             _allow_fixture_census=True)

    with pytest.raises(ValueError, match="histogram|census"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(tmp_path / "c"),
                        expected_queries=2, required_rooms=rooms,
                        expected_histogram={8: 2}, _allow_fixture_census=True)


def test_audit_aborts_before_writing_when_a_room_is_blocked(tmp_path, monkeypatch):
    """A blocked anchor audit must leave NOTHING behind -- no manifests, no
    report (r2 re-review: both were written before main() returned 1)."""
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / "blocked"

    real_audit = mg.audit_room_anchors

    def block_second(scene, anchors, room_id=None, **kwargs):
        report = real_audit(scene, anchors, room_id=room_id, **kwargs)
        if room_id == rooms[1]:
            report["accepted"] = False
            report["rules"]["receivers"]["failure"] = "rule 2: synthetic block"
        return report

    monkeypatch.setattr(mg, "audit_room_anchors", block_second)
    with pytest.raises(ValueError, match="blocked|anchor"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(out),
                        expected_queries=2, required_rooms=rooms,
                        _allow_fixture_census=True)
    assert not os.path.isdir(str(out)) or os.listdir(str(out)) == []

    # ... unless diagnostics are asked for, and then the report says what it is
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"], out_dir=str(out),
                             expected_queries=2, required_rooms=rooms,
                             diagnostics_only=True)
    assert report["diagnostics_only"] is True
    assert report["status"].startswith("DIAGNOSTICS")
    written = sorted(os.listdir(str(out)))
    assert written == ["geometry_diagnostics_report.json"]
    assert "candidate" not in "".join(written)


def test_audit_keeps_empty_z_band_queries_as_infinite(tmp_path, monkeypatch):
    """An empty z-band set is exactly what disqualifies the branch; substituting
    the full-height oracle hid it (r2 re-review)."""
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    real_filter = mg.filter_query_candidates

    def empty_band(candidates, receiver, context_sources=(), z_band=None, **kwargs):
        if z_band is not None:
            raise ValueError("every candidate was filtered away (synthetic)")
        return real_filter(candidates, receiver, context_sources, z_band=None, **kwargs)

    monkeypatch.setattr(mg, "filter_query_candidates", empty_band)
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"],
                             out_dir=str(tmp_path / "inf"), expected_queries=2,
                             required_rooms=rooms, _allow_fixture_census=True)
    assert report["branch"]["branch"] == "full_height"
    assert report["oracle"]["z_band"]["n_infinite"] == 2
    assert report["oracle"]["z_band"]["median"] == float("inf")
    assert np.isfinite(report["oracle"]["full_height"]["median"])


def test_audit_reports_gate_counts_for_both_branches(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"],
                             out_dir=str(tmp_path / "cost"), expected_queries=2,
                             required_rooms=rooms, _allow_fixture_census=True)
    assert set(report["cost"]) == {"full_height", "z_band", "chosen_branch"}
    for branch in ("full_height", "z_band"):
        block = report["cost"][branch]
        for key in ("candidate_query_pairs", "unique_receiver_candidate_pairs",
                    "conditioner_calls_estimate", "artifact_bytes"):
            assert key in block, (branch, key)
        assert block["candidate_query_pairs"] > 0
    assert report["cost"]["chosen_branch"] == report["branch"]["branch"]


def test_unique_receiver_candidate_pairs_hash_the_actual_index_sets():
    audit = _audit_module()
    a = np.array([0, 1, 2]), np.array([0, 1, 2])
    b = np.array([0, 1, 3])
    key_a = audit.candidate_set_key("R1", a[0])
    assert key_a == audit.candidate_set_key("R1", a[1])          # same set, same key
    assert key_a != audit.candidate_set_key("R1", b)             # same COUNT, different set
    assert key_a != audit.candidate_set_key("R2", a[0])          # different receiver
    assert len(audit.candidate_set_key("R1", a[0])[1]) == 64


def test_room_manifest_carries_indices_coordinates_branch_and_snapped_origin(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / "full"
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"], out_dir=str(out),
                             expected_queries=2, required_rooms=rooms,
                             _allow_fixture_census=True)
    room = rooms[0]
    payload = json.load(open(os.path.join(str(out), report["rooms"][room]["candidate_manifest"])))
    assert payload["chosen_branch"] == report["branch"]["branch"]
    # the SNAPPED lattice origin, not the raw AABB minimum
    origin = payload["lattice_origin"]
    assert all(abs(v / mg.LATTICE_SPACING - round(v / mg.LATTICE_SPACING)) < 1e-9
               for v in origin)
    query = payload["queries"][0]
    assert len(query["candidate_indices"]) == query["n_candidates"]
    assert len(query["candidate_indices_z_band"]) == query["n_candidates_z_band"]
    assert len(query["candidate_coordinates_sha256"]) == 64
    sidecar = os.path.join(str(out), payload["coordinates_npz"])
    assert os.path.isfile(sidecar)
    with np.load(sidecar) as data:
        lattice = data["base_candidates"]
        assert lattice.shape[1] == 3
        picked = lattice[np.asarray(query["candidate_indices"])]
        assert picked.shape[0] == query["n_candidates"]
        assert audit.coordinates_digest(picked) == query["candidate_coordinates_sha256"]


# --------------------------------------------------------------------------- #
# r4 -- union counts, census closure, -inf, and the artifact verifier
# --------------------------------------------------------------------------- #
def test_gate_counts_are_the_per_receiver_union():
    """r3 re-review: summing distinct sets reported 6 calls for a 4-element
    union."""
    audit = _audit_module()
    counts = audit.GateCounter()
    counts.add("R1", np.array([0, 1, 2]))
    counts.add("R1", np.array([0, 1, 3]))
    summary = counts.summary()
    assert summary["conditioner_calls_estimate"] == 4          # |{0,1,2,3}|
    assert summary["unique_receiver_candidate_pairs"] == 4     # true pair count
    assert summary["distinct_candidate_sets"] == 2             # labelled diagnostic
    assert summary["candidate_query_pairs"] == 6               # 3 + 3, as scored

    counts.add("R2", np.array([0, 1, 2]))
    summary = counts.summary()
    assert summary["conditioner_calls_estimate"] == 7          # 4 + 3
    assert summary["unique_receiver_candidate_pairs"] == 7
    assert summary["distinct_candidate_sets"] == 3


def test_branch_rule_refuses_negative_infinity_everywhere():
    """+inf means an empty z-band; -inf is not a meaningful oracle anywhere."""
    with pytest.raises(ValueError, match="finite"):
        mg.choose_z_branch({"q0": 0.1}, {"q0": float("-inf")}, band_nonempty=True)
    with pytest.raises(ValueError, match="finite"):
        mg.choose_z_branch({"q0": float("-inf")}, {"q0": 0.1}, band_nonempty=True)
    ok = mg.choose_z_branch({"q0": 0.1}, {"q0": float("inf")}, band_nonempty=True)
    assert ok["branch"] == "full_height" and ok["n_empty_band"] == 1


def test_registered_mode_cannot_choose_another_query_count(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    with pytest.raises(ValueError, match="diagnostics|registered"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(tmp_path / "n"),
                        expected_queries=2, required_rooms=rooms)
    # the same call IS allowed as a diagnostics run, and writes only the report
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"],
                             out_dir=str(tmp_path / "d"), expected_queries=2,
                             required_rooms=rooms, diagnostics_only=True)
    assert report["diagnostics_only"] is True
    assert sorted(os.listdir(str(tmp_path / "d"))) == ["geometry_diagnostics_report.json"]


def test_publish_refuses_a_non_empty_output_directory(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "leftover.json").write_text("{}")
    with pytest.raises(ValueError, match="empty|existing"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(out),
                        expected_queries=2, required_rooms=rooms, diagnostics_only=True)
    assert sorted(os.listdir(str(out))) == ["leftover.json"]


def _published(tmp_path, name="pub"):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / name
    report = audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                             metadata_root=world["metadata_root"], out_dir=str(out),
                             expected_queries=None, required_rooms=rooms,
                             expected_histogram=None, _allow_fixture_census=True)
    return audit, str(out), report, rooms


def test_verifier_accepts_what_the_audit_published(tmp_path):
    audit, out, report, rooms = _published(tmp_path)
    for room_id in rooms:
        entry = report["rooms"][room_id]
        verdict = audit.verify_room_manifest(os.path.join(out, entry["candidate_manifest"]))
        assert verdict["ok"] is True and verdict["reasons"] == []
        assert verdict["n_queries"] == entry["n_queries"]
        assert set(verdict["branches_reconstructed"]) == {"full_height", "z_band"}
    chain = audit.verify_report_chain(os.path.join(out, "geometry_audit_report.json"))
    assert chain["ok"] is True and chain["n_rooms"] == len(rooms)


def test_verifier_detects_a_corrupted_npz(tmp_path):
    audit, out, report, rooms = _published(tmp_path, name="corrupt")
    room = rooms[0]
    path = os.path.join(out, report["rooms"][room]["candidate_manifest"])
    payload = json.load(open(path))
    npz = os.path.join(out, payload["coordinates_npz"])
    with np.load(npz) as data:
        base = data["base_candidates"].copy()
    base[0, 0] += 0.5                                   # a real coordinate moves
    np.savez(npz, base_candidates=base)
    verdict = audit.verify_room_manifest(path)
    assert verdict["ok"] is False
    assert any("base" in reason and "sha256" in reason for reason in verdict["reasons"])


def test_verifier_detects_an_out_of_range_index_and_a_tampered_digest(tmp_path):
    audit, out, report, rooms = _published(tmp_path, name="tamper")
    room = rooms[0]
    path = os.path.join(out, report["rooms"][room]["candidate_manifest"])
    payload = json.load(open(path))

    broken = json.loads(json.dumps(payload))
    broken["queries"][0]["candidate_indices"] = [10 ** 9]
    broken["queries"][0]["n_candidates"] = 1
    out_of_range = os.path.join(out, "out_of_range.json")
    with open(out_of_range, "w") as handle:
        json.dump(broken, handle, indent=2, sort_keys=True)
    verdict = audit.verify_room_manifest(out_of_range)
    assert verdict["ok"] is False
    assert any("range" in reason or "index" in reason for reason in verdict["reasons"])

    tampered = json.loads(json.dumps(payload))
    tampered["queries"][0]["candidate_coordinates_sha256"] = "0" * 64
    tampered_path = os.path.join(out, "tampered.json")
    with open(tampered_path, "w") as handle:
        json.dump(tampered, handle, indent=2, sort_keys=True)
    verdict = audit.verify_room_manifest(tampered_path)
    assert verdict["ok"] is False
    assert any("coordinate" in reason for reason in verdict["reasons"])


def test_report_chain_detects_a_manifest_edited_after_publication(tmp_path):
    audit, out, report, rooms = _published(tmp_path, name="chain")
    room = rooms[0]
    path = os.path.join(out, report["rooms"][room]["candidate_manifest"])
    payload = json.load(open(path))
    payload["n_base_valid"] = int(payload["n_base_valid"]) + 1
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    chain = audit.verify_report_chain(os.path.join(out, "geometry_audit_report.json"))
    assert chain["ok"] is False
    assert any(room in reason for reason in chain["reasons"])


# --------------------------------------------------------------------------- #
# r5 -- stage, verify the STAGED files, then publish atomically
# --------------------------------------------------------------------------- #
def test_publish_is_staged_verified_then_atomic(tmp_path):
    audit, out, report, rooms = _published(tmp_path, name="staged")
    published = sorted(os.listdir(out))
    assert any(name.endswith("_gallery.json") is False for name in published)
    assert "geometry_audit_report.json" in published
    for room_id in rooms:
        entry = report["rooms"][room_id]
        assert entry["candidate_manifest"] in published
        assert os.path.basename(json.load(open(os.path.join(out, entry["candidate_manifest"])))
                                ["coordinates_npz"]) in published
    # no staging directory survives a successful publish
    siblings = os.listdir(os.path.dirname(out))
    assert not [name for name in siblings if name.startswith(".staging")]

    on_disk = json.load(open(os.path.join(out, "geometry_audit_report.json")))
    assert on_disk["verification"]["chain_ok"] is True
    assert set(on_disk["verification"]["rooms"]) == set(rooms)
    assert all(on_disk["verification"]["rooms"].values())


def test_verifier_failure_leaves_the_final_directory_empty(tmp_path, monkeypatch):
    """A staged artifact that does not verify is never published, and the
    staging directory is removed."""
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / "final"

    real_verify = audit.verify_room_manifest

    def fail_second(manifest_path, out_dir=None):
        verdict = real_verify(manifest_path, out_dir=out_dir)
        if verdict["room_id"] == rooms[1]:
            return {"ok": False, "reasons": ["synthetic verifier failure"],
                    "manifest": manifest_path, "room_id": verdict["room_id"],
                    "n_queries": verdict["n_queries"], "branches_reconstructed": []}
        return verdict

    monkeypatch.setattr(audit, "verify_room_manifest", fail_second)
    with pytest.raises(ValueError, match="verify"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(out),
                        expected_queries=None, required_rooms=rooms,
                        _allow_fixture_census=True)
    assert not os.path.isdir(str(out)) or os.listdir(str(out)) == []
    siblings = os.listdir(str(tmp_path))
    assert not [name for name in siblings if name.startswith(".staging")]


def test_pre_existing_output_is_still_refused_before_any_staging(tmp_path):
    audit = _audit_module()
    world = _fixture_world(tmp_path)
    rooms = tuple(sorted(r["room_id"] for r in world["manifest"]["records"]))
    out = tmp_path / "occupied2"
    out.mkdir()
    (out / "leftover.json").write_text("{}")
    with pytest.raises(ValueError, match="empty|existing"):
        audit.run_audit(world["manifest"], mesh_root=world["mesh_root"],
                        metadata_root=world["metadata_root"], out_dir=str(out),
                        expected_queries=None, required_rooms=rooms,
                        _allow_fixture_census=True)
    assert sorted(os.listdir(str(out))) == ["leftover.json"]
    assert not [name for name in os.listdir(str(tmp_path)) if name.startswith(".staging")]


# --------------------------------------------------------------------------- #
# r6 -- anchor-driven direction-set selection (exp_22 is self-authoritative)
# --------------------------------------------------------------------------- #
def _stub_scenes(rooms=("A/A_idx_1", "B/B_idx_2"), n_sources=3, n_receivers=4):
    import types

    return {room: {"scene": types.SimpleNamespace(room=room),
                   "sources": np.arange(n_sources * 3, dtype=np.float64).reshape(-1, 3),
                   "receivers": np.arange(n_receivers * 3, dtype=np.float64).reshape(-1, 3)}
            for room in rooms}


def _scripted_votes(failing_seeds, failing_room="B/B_idx_2", n_receivers=4):
    """Votes that fail exactly the named seeds, on ONE room's receivers."""
    def votes_fn(seed, directions, scene, points):
        base = np.full(points.shape[0], 31, dtype=np.int64)
        if (seed in failing_seeds and getattr(scene, "room", None) == failing_room
                and points.shape[0] == n_receivers):
            base[0] = 15                                        # one below the majority
        return base
    return votes_fn


def test_selection_rule_returns_the_smallest_passing_seed():
    scenes = _stub_scenes()
    votes = _scripted_votes({0})
    selection = mg.select_direction_seed(scenes, max_seed=8, votes_fn=votes)
    assert selection["seed"] == 1
    assert selection["report"]["ok"] is True
    assert selection["rule"] == mg.DIRECTION_SELECTION_RULE
    assert [attempt["seed"] for attempt in selection["attempts"]] == [0, 1]
    assert selection["attempts"][0]["ok"] is False
    assert selection["attempts"][0]["n_failures"] == 1
    assert np.allclose(selection["directions"], mg.build_directions(31, seed=1))


def test_selection_skips_every_failing_seed_in_order():
    selection = mg.select_direction_seed(_stub_scenes(), max_seed=8,
                                         votes_fn=_scripted_votes({0, 1, 2}))
    assert selection["seed"] == 3
    assert [attempt["seed"] for attempt in selection["attempts"]] == [0, 1, 2, 3]
    assert all(not attempt["ok"] for attempt in selection["attempts"][:3])


def test_selection_is_deterministic():
    scenes, votes = _stub_scenes(), _scripted_votes({0, 1})
    first = mg.select_direction_seed(scenes, max_seed=8, votes_fn=votes)
    second = mg.select_direction_seed(scenes, max_seed=8, votes_fn=votes)
    assert first["seed"] == second["seed"]
    assert first["directions"].tobytes() == second["directions"].tobytes()
    assert json.dumps(first["attempts"], sort_keys=True) == \
        json.dumps(second["attempts"], sort_keys=True)


def test_selection_refuses_when_no_seed_passes():
    with pytest.raises(ValueError, match="no seed"):
        mg.select_direction_seed(_stub_scenes(), max_seed=3,
                                 votes_fn=_scripted_votes({0, 1, 2, 3}))


def test_seed_evaluation_reports_every_failing_anchor():
    report = mg.evaluate_direction_seed(0, _stub_scenes(), votes_fn=_scripted_votes({0}))
    assert report["ok"] is False and report["n_failures"] == 1
    failure = report["failures"][0]
    assert failure["room_id"] == "B/B_idx_2" and failure["kind"] == "receivers"
    assert failure["odd_votes"] == 15
    assert report["majority"] == 16                       # >= 16 of 31, per the directive
    assert report["rooms"]["A/A_idx_1"]["sources"]["n_failing"] == 0
    assert len(report["directions_sha256"]) == 64


def test_evaluation_uses_the_strict_majority_of_sixteen():
    """Yixun's rule verbatim: >= 16/31 odd parity is interior."""
    exactly_sixteen = lambda seed, directions, scene, points: np.full(  # noqa: E731
        points.shape[0], 16, dtype=np.int64)
    assert mg.evaluate_direction_seed(0, _stub_scenes(),
                                      votes_fn=exactly_sixteen)["ok"] is True
    fifteen = lambda seed, directions, scene, points: np.full(  # noqa: E731
        points.shape[0], 15, dtype=np.int64)
    assert mg.evaluate_direction_seed(0, _stub_scenes(), votes_fn=fifteen)["ok"] is False


@pytest.mark.skipif(not os.path.isfile(_CAFE_OBJ), reason="Cafe OBJ not present")
def test_real_cafe_anchors_pass_under_the_pinned_set():
    """The pinned set must classify every Cafe anchor as interior."""
    scenes = mg.anchor_scenes(
        ["Cafe/Cafe_idx_1"],
        mesh_root="/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format",
        metadata_root="AcousticRooms/metadata")
    report = mg.evaluate_direction_seed(mg.FROZEN_DIRECTIONS_SEED, scenes)
    assert report["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
    assert report["ok"] is True, report["failures"]
    assert report["min_votes"] >= 16
    assert report["rooms"]["Cafe/Cafe_idx_1"]["sources"]["n"] == 10
    assert report["rooms"]["Cafe/Cafe_idx_1"]["receivers"]["n"] == 100


def test_manifest_schema_records_the_selected_seed_and_rule(tmp_path):
    audit, out, report, rooms = _published(tmp_path, name="schema")
    assert report["directions_seed"] == mg.FROZEN_DIRECTIONS_SEED == 1
    assert report["direction_selection_rule"] == mg.DIRECTION_SELECTION_RULE
    assert report["resolved_parity_discrepancies"][0]["previous_seed"] == 0
    payload = json.load(open(os.path.join(out, report["rooms"][rooms[0]]
                                          ["candidate_manifest"])))
    assert payload["directions_seed"] == 1
    assert payload["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
    assert "smallest generator seed" in payload["direction_selection_rule"]
