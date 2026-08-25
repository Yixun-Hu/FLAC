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
    assert mg.build_directions(31).tobytes() == directions.tobytes()   # deterministic
    assert mg.build_directions(31).tobytes() != mg.build_directions(33).tobytes()


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
    assert digest == "9ab4339fa893c00dca817b901a149c292b080d0e6971c90f0b8b0b88e858c261"
    # the generator is provenance only: it must still reproduce the literals, and
    # if it ever stops doing so the literals win and this test says so
    assert np.allclose(mg.build_directions(31), mg.FROZEN_DIRECTIONS, atol=0, rtol=0)


def test_scene_identity_records_the_pinned_digest(shell_room):
    scene = mg.load_raycast_scene(shell_room)
    assert scene.identity["directions_sha256"] == mg.FROZEN_DIRECTIONS_SHA256
    assert scene.identity["directions_sha256_pinned"] == mg.FROZEN_DIRECTIONS_SHA256


def test_known_discrepancy_is_documented_and_still_fails_closed():
    entry = mg.known_discrepancy("MeetingRoom/MeetingRoom_idx_32", [2.26, 0.48, 1.2],
                                 "receivers")
    assert entry is not None
    assert entry["odd_votes"] == 15 and entry["n_directions"] == 31
    assert entry["status"] == "documented, unresolved"
    assert mg.known_discrepancy("Cafe/Cafe_idx_1", [2.26, 0.48, 1.2], "receivers") is None
    assert mg.known_discrepancy("MeetingRoom/MeetingRoom_idx_32", [9.9, 9.9, 9.9],
                                "receivers") is None


_MEETING_OBJ = ("/media/diskstation/yixunhu/FLAC/AcousticRooms/room_mesh_obj_format/"
                "MeetingRoom/MeetingRoom_idx_32.obj")


@pytest.mark.skipif(not os.path.isfile(_MEETING_OBJ), reason="MeetingRoom OBJ not present")
def test_meeting_room_discrepancy_still_reads_exactly_as_documented():
    """An xfail-style marker: the documented anchor must keep failing in exactly
    the documented way until the exp_09 cross-check rules on it. A change in
    either direction -- fixed, or worse -- fails this test loudly."""
    scene = mg.load_raycast_scene(_MEETING_OBJ)
    point = np.array([[2.26, 0.48, 1.2]])
    votes = mg.odd_parity_votes(scene, point)[0]
    assert int(votes) == 15, f"documented 15/31 odd votes, measured {int(votes)}"
    assert bool(mg.classify_free_space(scene, point)[0]) is False
    distance = float(mg.surface_distance(scene, point)[0])
    assert distance == pytest.approx(0.25005, abs=5e-4)

    audit = mg.audit_room_anchors(
        scene, mg.metadata_anchors("AcousticRooms/metadata/MeetingRoom/MeetingRoom_idx_32"),
        room_id="MeetingRoom/MeetingRoom_idx_32")
    assert audit["accepted"] is False, "the room must stay blocked pending the ruling"
    receivers = audit["rules"]["receivers"]
    assert receivers["all_failures_documented"] is True
    assert receivers["known_discrepancies"][0]["odd_votes"] == 15
