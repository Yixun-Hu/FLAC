"""Atomic staged publishing for RAF artifacts (exp_19 r2, Codex R7).

r1 wrote WAVs, metadata, splits, audits, depth maps and QA files straight into
their destinations, one after another. A late failure -- a bad file, a NAS
interruption, a full disk -- left a plausible-looking mixture of old and new
artifacts, and a rerun could overwrite audio before replacing the split that
described it. Everything now lands in a staging directory ON THE SAME FILESYSTEM,
is validated there, and is swapped in per file with ``os.replace``; the hash
manifest is written last, so its presence is the evidence that a publish
committed.
"""
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import prepare_data as raf_prepare  # noqa: E402
import publish as raf_publish  # noqa: E402
import render_depth as raf_render  # noqa: E402

from test_raf_prepare_data import (  # noqa: E402
    N_MICS, write_passing_readback_record, write_room,  # noqa: F401
)

assert os.path.dirname(os.path.abspath(raf_publish.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# StagedPublish mechanics
# --------------------------------------------------------------------------- #
def test_staging_directory_is_a_sibling_of_the_destination(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        # same parent => same filesystem => os.replace is atomic
        assert os.path.dirname(staged.staging_dir) == os.path.dirname(str(dest))
        assert os.path.isdir(staged.staging_dir)
        assert os.path.basename(staged.staging_dir).startswith(".dest.staging-")


def test_nothing_is_visible_until_commit(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        with open(staged.path("a", "one.txt"), "w") as f:
            f.write("hello")
        assert not (dest / "a" / "one.txt").exists()
        staged.commit()
    assert (dest / "a" / "one.txt").read_text() == "hello"
    assert not os.path.exists(staged.staging_dir)


def test_commit_writes_the_hash_manifest_last(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        with open(staged.path("one.txt"), "w") as f:
            f.write("hello")
        manifest = staged.commit()
    published = manifest["files"]["one.txt"]
    assert published["sha256"] == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    assert published["bytes"] == 5
    on_disk = json.loads((dest / raf_publish.MANIFEST_NAME).read_text())
    assert on_disk["files"] == manifest["files"]
    assert on_disk["committed_utc"]


def test_an_exception_publishes_nothing_and_cleans_up(tmp_path):
    dest = tmp_path / "dest"
    (dest).mkdir()
    (dest / "old.txt").write_text("previous good publish")
    staging = {}
    with pytest.raises(RuntimeError):
        with raf_publish.StagedPublish(str(dest)) as staged:
            staging["dir"] = staged.staging_dir
            with open(staged.path("new.txt"), "w") as f:
                f.write("half a publish")
            raise RuntimeError("NAS went away")
    assert not (dest / "new.txt").exists()
    assert (dest / "old.txt").read_text() == "previous good publish"
    assert not (dest / raf_publish.MANIFEST_NAME).exists()
    assert not os.path.exists(staging["dir"])


def test_validation_failure_publishes_nothing(tmp_path):
    dest = tmp_path / "dest"
    with pytest.raises(ValueError):
        with raf_publish.StagedPublish(str(dest)) as staged:
            with open(staged.path("broken.json"), "w") as f:
                f.write("{not json")
            staged.commit(validate_json=True)
    assert not (dest / "broken.json").exists()
    assert not (dest / raf_publish.MANIFEST_NAME).exists()


def test_commit_validates_every_staged_json(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        with open(staged.path("good.json"), "w") as f:
            json.dump({"ok": True}, f)
        staged.commit(validate_json=True)
    assert json.loads((dest / "good.json").read_text()) == {"ok": True}


def test_commit_refuses_an_expected_file_that_was_never_staged(tmp_path):
    dest = tmp_path / "dest"
    with pytest.raises(ValueError) as exc:
        with raf_publish.StagedPublish(str(dest)) as staged:
            with open(staged.path("one.txt"), "w") as f:
                f.write("x")
            staged.commit(expected=["one.txt", "two.txt"])
    assert "two.txt" in str(exc.value)
    assert not (dest / "one.txt").exists()


def test_replaced_files_are_swapped_not_merged(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "one.txt").write_text("old")
    with raf_publish.StagedPublish(str(dest)) as staged:
        with open(staged.path("one.txt"), "w") as f:
            f.write("new")
        staged.commit()
    assert (dest / "one.txt").read_text() == "new"


def test_verify_manifest_detects_a_tampered_artifact(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        with open(staged.path("one.txt"), "w") as f:
            f.write("hello")
        staged.commit()
    assert raf_publish.verify_manifest(str(dest)) == {"missing": [], "mismatched": []}
    (dest / "one.txt").write_text("tampered")
    assert raf_publish.verify_manifest(str(dest))["mismatched"] == ["one.txt"]
    os.remove(dest / "one.txt")
    assert raf_publish.verify_manifest(str(dest))["missing"] == ["one.txt"]


# --------------------------------------------------------------------------- #
# prepare_data publishes atomically
# --------------------------------------------------------------------------- #
def _prepare(tmp_path, rooms=("EmptyRoom",)):
    raf_root = tmp_path / "raf"
    for room in rooms:
        write_room(str(raf_root), room)
    out = tmp_path / "runtime" / "RAF"
    split_dir = tmp_path / "splits"
    argv = ["--raf-root", str(raf_root), "--output-dir", str(out),
            "--split-dir", str(split_dir), "--rooms", *rooms,
            "--n-groups", "1", "--n-val-groups", "1", "--n-diagnostic-groups", "1",
            "--n-train", "12", "--full-crosscheck", "--readback-record",
            write_passing_readback_record(str(tmp_path / "readback.json"))]
    return argv, out, split_dir


def test_prepare_publishes_a_manifest_covering_every_artifact(tmp_path):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    for root in (out, split_dir):
        manifest = json.loads((root / raf_publish.MANIFEST_NAME).read_text())
        assert manifest["files"]
        assert raf_publish.verify_manifest(str(root)) == {"missing": [], "mismatched": []}
    split_manifest = json.loads((split_dir / raf_publish.MANIFEST_NAME).read_text())
    for name in ("train_base.json", "val_base.json", "test_base.json",
                 "diagnostic_base.json", "raf_splits_record.json",
                 "raf_amplitude_audit.json"):
        assert name in split_manifest["files"]
    out_manifest = json.loads((out / raf_publish.MANIFEST_NAME).read_text())
    assert sum(1 for k in out_manifest["files"] if k.endswith(".wav")) == 3 * N_MICS


def test_prepare_leaves_a_previous_publish_intact_when_it_fails(tmp_path, monkeypatch):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    before = json.loads((split_dir / "train_base.json").read_text())
    before_manifest = json.loads((split_dir / raf_publish.MANIFEST_NAME).read_text())

    def explode(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(raf_prepare, "build_splits_record", explode)
    with pytest.raises(RuntimeError):
        raf_prepare.main(argv)

    assert json.loads((split_dir / "train_base.json").read_text()) == before
    assert json.loads((split_dir / raf_publish.MANIFEST_NAME).read_text()) == before_manifest
    assert raf_publish.verify_manifest(str(out)) == {"missing": [], "mismatched": []}
    leftovers = [n for n in os.listdir(tmp_path / "runtime") if ".staging-" in n]
    assert leftovers == []


def test_prepare_publishes_no_audio_when_a_later_room_fails(tmp_path, monkeypatch):
    """The r1 order wrote room 1's audio before room 2 was even read."""
    argv, out, split_dir = _prepare(tmp_path, rooms=("EmptyRoom", "FurnishedRoom"))
    real_group = raf_prepare.group_captures
    calls = []

    def failing_group(index, **kw):
        calls.append(1)
        if len(calls) == 2:          # fail while handling the SECOND room
            raise RuntimeError("corpus anomaly")
        return real_group(index, **kw)

    monkeypatch.setattr(raf_prepare, "group_captures", failing_group)
    with pytest.raises(RuntimeError):
        raf_prepare.main(argv)
    assert not (out / "EmptyRoom").exists()
    assert not (split_dir / "train_base.json").exists()


# --------------------------------------------------------------------------- #
# render_depth publishes atomically
# --------------------------------------------------------------------------- #
def test_render_depth_publishes_atomically(tmp_path, monkeypatch):
    from test_raf_render_depth import _write_fixture

    raf_root, out, groups = _write_fixture(tmp_path)
    record = write_passing_readback_record(str(tmp_path / "readback.json"))
    argv = ["--raf-root", str(raf_root), "--output-dir", str(out), "--rooms",
            "EmptyRoom", "--readback-record", record]
    raf_render.main(argv)
    depth_dir = out / "EmptyRoom" / "depth_images"
    assert raf_publish.verify_manifest(str(depth_dir)) == {"missing": [], "mismatched": []}
    before = json.loads((depth_dir / raf_publish.MANIFEST_NAME).read_text())

    real_qa = raf_render.real_mesh_qa

    def explode(*a, **kw):
        raise RuntimeError("mesh went away")

    monkeypatch.setattr(raf_render, "real_mesh_qa", explode)
    with pytest.raises(RuntimeError):
        raf_render.main(argv)
    assert json.loads((depth_dir / raf_publish.MANIFEST_NAME).read_text()) == before
    assert raf_publish.verify_manifest(str(depth_dir)) == {"missing": [], "mismatched": []}
    assert real_qa is not None


def test_render_depth_publishes_nothing_when_a_map_fails_qa(tmp_path):
    from test_raf_render_depth import _write_fixture

    raf_root, out, groups = _write_fixture(tmp_path)
    meta = out / "EmptyRoom" / "metadata" / "groups_metadata.json"
    payload = json.loads(meta.read_text())
    payload["bbbb000000000002"]["tx_xyz_p"] = [99.0, 99.0, 1.0]   # outside the room
    meta.write_text(json.dumps(payload))
    with pytest.raises((RuntimeError, ValueError)):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record",
                         write_passing_readback_record(str(tmp_path / "readback.json"))])
    depth_dir = out / "EmptyRoom" / "depth_images"
    assert not (depth_dir / "aaaa000000000001_depth_image.npy").exists()
    assert not (depth_dir / raf_publish.MANIFEST_NAME).exists()
