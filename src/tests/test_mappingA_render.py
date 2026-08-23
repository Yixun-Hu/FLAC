"""Listener-positioned depth rendering (exp_21, contract C, cycle 7).

AR renders the panorama at the RECEIVER, so Mapping A does too: one map per target
microphone instead of one per source. Two consequences carry the gauge evidence
across: the nadir gate compares against the item's RAW RAF receiver height (which
no candidate gauge has touched), and the recorded sightline diagnostic probes
TRANSMITTER endpoints, because a camera at the receiver should be able to see the
sources.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import render_depth as raf_render  # noqa: E402

from test_raf_render_depth import _box_mesh_raf, _readback  # noqa: E402


def _items(n_slots=3, shared_position=False):
    items = {}
    for slot in range(n_slots):
        rx = [0.0, 0.0, 1.0] if shared_position else [0.1 * slot, 0.0, 1.0]
        items[f"{slot:06d}"] = {
            "item_id": f"EmptyRoom/p000/slot{slot:02d}",
            "room": "EmptyRoom", "placement_id": "p000", "mic_slot": slot,
            "target_capture_id": f"{slot:06d}",
            "tx_p": [2.0, 1.0, 1.5],
            "rx_target_p": rx,
            "rx_target_height_raf_m": 1.0,
            "depth_file": f"EmptyRoom_p000_slot{slot:02d}_depth_image.npy",
            "context": [{"capture_id": f"{100 + j:06d}", "tx_p": [float(j), 1.0, 1.4]}
                        for j in range(4)],
        }
    return items


def _write_items(tmp_path, items, room="EmptyRoom"):
    meta = tmp_path / "runtime" / "mappingA" / room / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / "mappingA_metadata.json"
    with open(path, "w") as f:
        json.dump(items, f)
    return str(path)


# --------------------------------------------------------------------------- #
# the render plan
# --------------------------------------------------------------------------- #
def test_the_plan_renders_at_the_target_receiver(tmp_path):
    plan = raf_render.mappingA_render_plan(_write_items(tmp_path, _items()))
    assert len(plan) == 3
    assert [p["position_p"] for p in plan] == [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
                                               [0.2, 0.0, 1.0]]
    assert all(p["positioned_at"] == "listener" for p in plan)


def test_the_plan_carries_the_raw_raf_receiver_height(tmp_path):
    """Independent of the gauge, exactly as exp_19 r5 established for tx heights."""
    plan = raf_render.mappingA_render_plan(_write_items(tmp_path, _items()))
    assert all(p["tracked_height_m"] == 1.0 for p in plan)


def test_the_plan_probes_transmitter_endpoints(tmp_path):
    """A listener-positioned map should see the SOURCES, not other receivers."""
    plan = raf_render.mappingA_render_plan(_write_items(tmp_path, _items()))
    endpoints = plan[0]["sightline_endpoints"]
    assert len(endpoints) == 5                     # 4 contexts + the target source
    assert [2.0, 1.0, 1.5] in endpoints


def test_identical_positions_are_rendered_once_and_shared(tmp_path):
    plan = raf_render.mappingA_render_plan(
        _write_items(tmp_path, _items(shared_position=True)))
    assert len(plan) == 1
    assert len(plan[0]["item_ids"]) == 3
    assert len(plan[0]["shared_depth_files"]) == 2


def test_the_plan_is_deterministic(tmp_path):
    path = _write_items(tmp_path, _items())
    assert raf_render.mappingA_render_plan(path) == raf_render.mappingA_render_plan(path)


# --------------------------------------------------------------------------- #
# end to end through the CLI
# --------------------------------------------------------------------------- #
def _fixture(tmp_path, items=None):
    raf_root = tmp_path / "raf"
    out = tmp_path / "runtime" / "mappingA"
    mesh_dir = raf_root / "3d_models" / "EmptyRoom"
    mesh_dir.mkdir(parents=True)
    import open3d as o3d

    o3d.io.write_triangle_mesh(
        str(mesh_dir / "mesh.obj"),
        _box_mesh_raf(x0=-10.0, x1=10.0, y0=0.0, y1=3.0, z0=-10.0, z1=10.0,
                      to_pipeline=False))
    _write_items(tmp_path, items if items is not None else _items())
    return raf_root, out


def test_the_cli_renders_listener_maps(tmp_path):
    raf_root, out = _fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--readback-record", _readback(tmp_path), "--non-canonical"])
    depth_dir = out / "EmptyRoom" / "depth_images"
    for slot in range(3):
        arr = np.load(depth_dir / f"EmptyRoom_p000_slot{slot:02d}_depth_image.npy")
        assert arr.shape == (256, 512) and arr.dtype == np.float32
    with open(depth_dir / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["positions_from"] == "mappingA"
    assert len(qa["maps"]) == 3
    for entry in qa["maps"].values():
        assert entry["positioned_at"] == "listener"
        assert entry["passed"] is True
        assert entry["real_mesh"]["vertical_axis"]["tracked_height_m"] == 1.0
        assert entry["real_mesh"]["vertical_axis"]["ok"] is True
        assert entry["real_mesh"]["rx_sightline"]["checked"] is True


def test_the_cli_still_renders_source_maps_by_default(tmp_path):
    """Mapping H's renderer must be unchanged: same flag absent, same behaviour."""
    from test_raf_render_depth import _write_fixture

    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["positions_from"] == "groups"
    assert all(e["positioned_at"] == "source" for e in qa["maps"].values())
    assert len(qa["maps"]) == len(groups)


def test_a_listener_map_outside_the_room_fails_qa(tmp_path):
    items = _items(n_slots=1)
    items["000000"]["rx_target_p"] = [99.0, 99.0, 1.0]
    raf_root, out = _fixture(tmp_path, items)
    with pytest.raises((RuntimeError, ValueError)):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                         "--readback-record", _readback(tmp_path), "--non-canonical"])


def test_a_wrong_tracked_receiver_height_fails_the_vertical_gate(tmp_path):
    items = _items(n_slots=1)
    items["000000"]["rx_target_height_raf_m"] = 2.5      # the map says 1.0
    raf_root, out = _fixture(tmp_path, items)
    with pytest.raises(RuntimeError) as exc:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                         "--readback-record", _readback(tmp_path), "--non-canonical"])
    assert "failed QA" in str(exc.value)


# --------------------------------------------------------------------------- #
# r2 N1: listener mode publishes under the mappingA_depth kind
# --------------------------------------------------------------------------- #
def test_listener_mode_publishes_the_mappingA_depth_marker(tmp_path):
    import publish as raf_publish

    raf_root, out = _fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--readback-record", _readback(tmp_path), "--non-canonical"])
    assert (out / raf_publish.marker_name("mappingA_depth")).exists()
    assert not (out / raf_publish.marker_name("depth")).exists()
    with open(out / raf_publish.marker_name("mappingA_depth")) as f:
        marker = json.load(f)
    parameters = marker["parameters"]
    assert parameters["positions_from"] == "mappingA"
    assert parameters["n_maps"] == 3              # derived from what was rendered
    assert parameters["img_h"] == 256 and parameters["img_w"] == 512
    assert len(parameters["readback_record_sha256"]) == 64
    assert "haa_reference_sha256" not in parameters       # the H payload is gone


def test_source_mode_still_publishes_the_depth_marker(tmp_path):
    import publish as raf_publish
    from test_raf_render_depth import _write_fixture

    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", _readback(tmp_path),
                     "--non-canonical"])
    assert (out / raf_publish.marker_name("depth")).exists()
    assert not (out / raf_publish.marker_name("mappingA_depth")).exists()


def test_a_canonical_listener_render_enforces_the_registered_map_count(tmp_path):
    raf_root, out = _fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                         "--haa-depth-root",
                         os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                      "raf_depth_reference"),
                         "--readback-record", _readback(tmp_path)])
    message = str(exc.value)
    assert "sha256" in message.lower() or "n_maps" in message or "rooms" in message


def _canonical_readback(tmp_path, raf_root, room="EmptyRoom", n_captures=3):
    """A canonical-SHAPED readback record for the fixture corpus (r3).

    Every content rule assert_canonical_content enforces, measured over the
    synthetic corpus rather than asserted: the capture count is re-derived from the
    fixture's own capture directories by the corpus-binding check.
    """
    import readback_audit as raf_readback

    record = {
        "schema_version": raf_readback.RECORD_SCHEMA_VERSION,
        "created_utc": "2026-08-22T00:00:00Z",
        "params": {"raf_root": str(raf_root), "synthetic": True},
        "rooms": {room: {
            "n_captures": n_captures,
            "onset": {"passed": True, "reasons": [], "slope_s_per_m": 1 / 343.0,
                      "r2": 0.99, "n": n_captures},
            "t30_validity": {"n": n_captures, "valid_full": n_captures,
                             "valid_crop": n_captures},
            "amplitude": {"peak_stats": {"count": n_captures, "max": 0.2,
                                         "min": 0.1}},
            "quaternion": {"identity_readings": [{"capture_id": "000000",
                                                  "quat": [0.0, 0.0, 0.0, 1.0],
                                                  "order": "xyzw"}]},
            "crosscheck": {"checked": n_captures, "mismatches": 0},
        }},
        "decisions": {"t60_headline": {"resolution": "headline"},
                      "amplitude_scalar": {"derived_from": "train supports only",
                                           "applied_scalar": None}},
        "adjudication": {"gauge_pinned": raf_readback.CANONICAL_GAUGE,
                         "quat_order_pinned": raf_readback.CANONICAL_QUAT_ORDER},
        "verdict": {"passed": True, "reasons": []},
    }
    path = tmp_path / "canonical_readback.json"
    path.write_text(json.dumps(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for i in range(n_captures):                     # what the binding check counts
        (raf_root / "archived" / room / "data" / f"{i:06d}").mkdir(parents=True,
                                                                   exist_ok=True)
    return str(path), digest


def test_the_renderer_marker_satisfies_RAF_A_md_end_to_end(tmp_path, monkeypatch):
    """N1's acceptance test: the marker the RENDERER produces -- untouched, from a
    CANONICAL render -- must satisfy the canonical verification RAF_A_md's gate
    performs (verify_combined_publication with flavor="mappingA").

    Only the REGISTRY is redirected to this fixture's scale: the room set, the map
    count and the pinned readback digest. Nothing the renderer wrote is edited,
    which is what the r2 review asked for -- the previous version ran
    --non-canonical and then patched canonical=True and the digest into the marker
    by hand, so the canonical path was never exercised end to end.
    """
    import publish as raf_publish
    import readback_audit as raf_readback

    raf_root, out = _fixture(tmp_path)
    rooms = ["EmptyRoom"]
    readback, digest = _canonical_readback(tmp_path, raf_root)

    # the three pins that name the real world, redirected to the fixture
    monkeypatch.setattr(raf_readback, "CANONICAL_RECORD_SHA256", digest)
    monkeypatch.setattr(raf_readback, "CANONICAL_ROOMS", tuple(rooms))
    monkeypatch.setattr(raf_publish, "canonical_record_digest", lambda: digest)
    monkeypatch.setattr(raf_publish, "CANONICAL_ROOMS", tuple(rooms))
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "rooms", rooms)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "n_maps", 3)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS,
                        "readback_record_sha256", digest)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS, "rooms", rooms)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS,
                        "readback_record_sha256", digest)

    # a prepare-side publication for the same tree, so the COMBINED check has both
    split_dir = tmp_path / "data" / "RAF_mappingA"
    prepare_params = dict(raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS,
                          readback_record_sha256=digest)
    pointer = {"split_dir": str(split_dir.resolve()), "output_dir": str(out.resolve()),
               "rooms": rooms, "flavor": "mappingA", "canonical": True, "taint": [],
               "parameters": prepare_params,
               "readback_record": {"sha256": digest}}
    with raf_publish.PublishTransaction(str(split_dir), kind="mappingA_prepare") as txn:
        runtime = txn.stage(str(out))
        splits = txn.stage(str(split_dir))
        with open(runtime.path("raf_publication.json"), "w") as f:
            json.dump(pointer, f)
        with open(splits.path("mappingA_eval.json"), "w") as f:
            json.dump({"EmptyRoom": []}, f)
        txn.commit(extra={"canonical": True, "taint": [], "canonical_parameters": True,
                          "parameters": prepare_params,
                          "readback_record": {"sha256": digest}})

    # the RENDERER writes the depth attestation -- canonically, and it is not touched
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--haa-depth-root",
                     os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                  "raf_depth_reference"),
                     "--readback-record", readback])
    marker_path = out / raf_publish.marker_name("mappingA_depth")
    before = marker_path.read_bytes()
    depth_marker = json.loads(before)
    assert depth_marker["canonical"] is True and depth_marker["taint"] == []
    assert depth_marker["parameters"]["n_maps"] == 3
    assert depth_marker["parameters"]["readback_record_sha256"] == digest
    assert depth_marker["readback_record"]["sha256"] == digest

    report = raf_publish.verify_combined_publication(
        str(split_dir), str(out), rooms=rooms, canonical=True, flavor="mappingA")
    assert report["published"] is True, report["reason"]
    assert marker_path.read_bytes() == before          # nothing was edited into it


def test_a_canonical_listener_render_refuses_the_wrong_map_count(tmp_path,
                                                                 monkeypatch):
    """The same canonical path, one deviation: the registry expects a different
    number of maps than the plan produces."""
    import publish as raf_publish
    import readback_audit as raf_readback

    raf_root, out = _fixture(tmp_path)
    readback, digest = _canonical_readback(tmp_path, raf_root)
    monkeypatch.setattr(raf_readback, "CANONICAL_RECORD_SHA256", digest)
    monkeypatch.setattr(raf_readback, "CANONICAL_ROOMS", ("EmptyRoom",))
    monkeypatch.setattr(raf_publish, "canonical_record_digest", lambda: digest)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "rooms",
                        ["EmptyRoom"])
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "n_maps", 1152)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS,
                        "readback_record_sha256", digest)
    with pytest.raises(ValueError) as exc:
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                         "--haa-depth-root",
                         os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                      "raf_depth_reference"),
                         "--readback-record", readback])
    assert "n_maps" in str(exc.value)
    assert not (out / raf_publish.marker_name("mappingA_depth")).exists()


# --------------------------------------------------------------------------- #
# r5 Amendment 4.2: the near-field disclosure reaches the QA record and marker
# --------------------------------------------------------------------------- #
def test_the_listener_render_records_the_near_field_disclosure(tmp_path, monkeypatch):
    """A listener map that sees the capture rig is published with the flag ON the
    record -- per map, naming the ITEMS it belongs to -- instead of failing."""
    import publish as raf_publish

    raf_root, out = _fixture(tmp_path)
    flagged = {"EmptyRoom_p000_slot01_depth_image.npy"}
    real_qa = raf_render.real_mesh_qa

    def near_field_qa(depth, position, mesh, **kwargs):
        report = real_qa(depth, position, mesh, **kwargs)
        # the second map sees structure 3 cm away; everything else is untouched
        if near_field_qa.calls in (1,):
            report = dict(report)
            report["scale_min_ok"] = False
            report["scale_plausible"] = False
            report["near_field"] = dict(report["near_field"], flagged=True,
                                        min_m=0.03)
        near_field_qa.calls += 1
        return report

    near_field_qa.calls = 0
    monkeypatch.setattr(raf_render, "real_mesh_qa", near_field_qa)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--haa-depth-root",
                     os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                  "raf_depth_reference"),
                     "--readback-record", _readback(tmp_path), "--non-canonical"])

    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    near = qa["near_field"]
    assert near["n_maps"] == 1
    assert near["min_side_gates_publication"] is False
    assert near["item_ids"] == ["EmptyRoom/p000/slot01"]
    assert near["maps"][0]["depth_file"] in flagged
    assert near["maps"][0]["min_m"] == pytest.approx(0.03)
    assert "capture rig" in near["note"]
    assert qa["n_failed"] == 0                      # recorded, not fatal

    with open(out / raf_publish.marker_name("mappingA_depth")) as f:
        marker = json.load(f)
    assert marker["near_field"]["n_maps"] == 1
    assert marker["near_field"]["item_ids"] == ["EmptyRoom/p000/slot01"]
    assert marker["near_field"]["min_side_gates_publication"] is False


def test_a_clean_listener_render_declares_no_near_field_maps(tmp_path):
    raf_root, out = _fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--haa-depth-root",
                     os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                  "raf_depth_reference"),
                     "--readback-record", _readback(tmp_path), "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["near_field"]["n_maps"] == 0
    assert qa["near_field"]["item_ids"] == []
    assert qa["near_field"]["min_side_gates_publication"] is False


def test_a_canonical_listener_render_uses_the_listener_cap_by_default(tmp_path,
                                                                      monkeypatch):
    """Amendment 4.3 end to end: with no --max-miss-rate the listener render takes
    0.7%, and that is the value its identity carries."""
    import publish as raf_publish
    import readback_audit as raf_readback

    raf_root, out = _fixture(tmp_path)
    readback, digest = _canonical_readback(tmp_path, raf_root)
    monkeypatch.setattr(raf_readback, "CANONICAL_RECORD_SHA256", digest)
    monkeypatch.setattr(raf_readback, "CANONICAL_ROOMS", ("EmptyRoom",))
    monkeypatch.setattr(raf_publish, "canonical_record_digest", lambda: digest)
    monkeypatch.setattr(raf_publish, "CANONICAL_ROOMS", ("EmptyRoom",))
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "rooms",
                        ["EmptyRoom"])
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS, "n_maps", 3)
    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS,
                        "readback_record_sha256", digest)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--positions-from", "mappingA",
                     "--haa-depth-root",
                     os.path.join(_REPO_ROOT, "src", "tests", "fixtures",
                                  "raf_depth_reference"),
                     "--readback-record", readback])
    with open(out / raf_publish.marker_name("mappingA_depth")) as f:
        marker = json.load(f)
    assert marker["parameters"]["max_miss_rate"] == raf_render.LISTENER_MAX_MISS_RATE
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["max_miss_rate"] == 0.007
    for record in qa["maps"].values():
        assert record["misses"]["max_miss_rate"] == 0.007
        assert record["misses"]["miss_cap_mode"] == "listener"


def test_the_source_cap_is_refused_for_a_canonical_listener_render(tmp_path):
    """The source cap is not this mode's registered protocol, however conservative
    it looks."""
    raf_root, out = _fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        raf_render.resolve_miss_cap(0.0025, canonical=True, listener_mode=True)
    assert "registered listener-map cap is exactly 0.007" in str(exc.value)
    # ... and a non-canonical run may use it, tainted and named
    cap, taint = raf_render.resolve_miss_cap(0.0025, canonical=False,
                                             listener_mode=True)
    assert cap == 0.0025 and "listener-map 0.007" in taint[0]
    assert out.parent.exists()


def test_a_source_mode_render_still_uses_the_source_cap():
    """Mapping H's published marker is bound to 0.25%; this amendment must not
    move it."""
    import publish as raf_publish

    assert raf_render.resolve_miss_cap(None, canonical=True,
                                       listener_mode=False) == (0.0025, [])
    assert raf_publish.CANONICAL_RENDER_PARAMS["max_miss_rate"] == 0.0025
    assert raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS["max_miss_rate"] == 0.007
