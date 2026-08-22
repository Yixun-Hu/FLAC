"""Listener-positioned depth rendering (exp_21, contract C, cycle 7).

AR renders the panorama at the RECEIVER, so Mapping A does too: one map per target
microphone instead of one per source. Two consequences carry the gauge evidence
across: the nadir gate compares against the item's RAW RAF receiver height (which
no candidate gauge has touched), and the recorded sightline diagnostic probes
TRANSMITTER endpoints, because a camera at the receiver should be able to see the
sources.
"""
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
