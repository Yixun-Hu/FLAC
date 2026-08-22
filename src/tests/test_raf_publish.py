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
from unittest import mock

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
            "--n-train", "12", "--full-crosscheck", "--non-canonical", "--readback-record",
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
            "EmptyRoom", "--readback-record", record, "--non-canonical"]
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
                         "--rooms", "EmptyRoom", "--non-canonical", "--readback-record",
                         write_passing_readback_record(str(tmp_path / "readback.json"))])
    depth_dir = out / "EmptyRoom" / "depth_images"
    assert not (depth_dir / "aaaa000000000001_depth_image.npy").exists()
    assert not (depth_dir / raf_publish.MANIFEST_NAME).exists()


# --------------------------------------------------------------------------- #
# r3 S3: one generation-bound commit marker over ALL roots
# --------------------------------------------------------------------------- #
def test_transaction_publishes_every_root_under_one_generation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        ra, rb = txn.stage(str(a)), txn.stage(str(b))
        open(ra.path("one.txt"), "w").write("A")
        open(rb.path("two.txt"), "w").write("B")
        marker = txn.commit()
    assert (a / "one.txt").read_text() == "A"
    assert (b / "two.txt").read_text() == "B"
    on_disk = json.loads((a / raf_publish.marker_name("prepare")).read_text())
    assert on_disk["generation"] == marker["generation"]
    assert set(on_disk["roots"]) == {str(a.resolve()), str(b.resolve())}
    for root in (a, b):
        assert json.loads((root / raf_publish.MANIFEST_NAME).read_text())["generation"] \
            == marker["generation"]
    assert raf_publish.verify_publication(str(a), kind="prepare")["published"] is True


def test_a_tree_without_a_marker_reads_as_unpublished(tmp_path):
    dest = tmp_path / "dest"
    with raf_publish.StagedPublish(str(dest)) as staged:
        open(staged.path("one.txt"), "w").write("A")
        staged.commit()                       # manifest only, no transaction marker
    report = raf_publish.verify_publication(str(dest), kind="prepare")
    assert report["published"] is False
    assert "marker" in report["reason"]


def test_previous_attestation_is_invalidated_before_any_replacement(tmp_path):
    """S3: the old manifest may not stay visible while files are being swapped."""
    a = tmp_path / "a"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        root = txn.stage(str(a))
        open(root.path("one.txt"), "w").write("first")
        txn.commit()
    first = json.loads((a / raf_publish.marker_name("prepare")).read_text())["generation"]

    seen = {}
    real_replace = os.replace

    def watching_replace(src, dst):
        if str(dst).endswith("one.txt"):
            # at the moment a payload file is swapped, no attestation may claim the
            # tree is published
            seen["marker"] = (a / raf_publish.marker_name("prepare")).exists()
            seen["manifest"] = (a / raf_publish.MANIFEST_NAME).exists()
        return real_replace(src, dst)

    with mock.patch("os.replace", watching_replace):
        with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
            root = txn.stage(str(a))
            open(root.path("one.txt"), "w").write("second")
            txn.commit()
    assert seen == {"marker": False, "manifest": False}
    assert (a / "one.txt").read_text() == "second"
    assert json.loads((a / raf_publish.marker_name("prepare")).read_text())["generation"] \
        != first


def test_failure_during_os_replace_leaves_the_tree_unpublished(tmp_path):
    a = tmp_path / "a"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        root = txn.stage(str(a))
        open(root.path("one.txt"), "w").write("first")
        open(root.path("two.txt"), "w").write("first")
        txn.commit()

    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src, dst):
        if str(dst).endswith(".txt"):
            calls["n"] += 1
            if calls["n"] == 2:               # die on the SECOND payload file
                raise OSError("NAS went away mid-swap")
        return real_replace(src, dst)

    with mock.patch("os.replace", failing_replace):
        with pytest.raises(OSError):
            with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
                root = txn.stage(str(a))
                open(root.path("one.txt"), "w").write("second")
                open(root.path("two.txt"), "w").write("second")
                txn.commit()

    report = raf_publish.verify_publication(str(a), kind="prepare")
    assert report["published"] is False       # mixed generations, no valid marker
    assert not (a / raf_publish.marker_name("prepare")).exists()


def test_failure_between_roots_leaves_the_tree_unpublished(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        ra, rb = txn.stage(str(a)), txn.stage(str(b))
        open(ra.path("one.txt"), "w").write("first")
        open(rb.path("two.txt"), "w").write("first")
        txn.commit()

    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith("two.txt"):      # the SECOND root's payload
            raise OSError("disk full between roots")
        return real_replace(src, dst)

    with mock.patch("os.replace", failing_replace):
        with pytest.raises(OSError):
            with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
                ra, rb = txn.stage(str(a)), txn.stage(str(b))
                open(ra.path("one.txt"), "w").write("second")
                open(rb.path("two.txt"), "w").write("second")
                txn.commit()
    assert raf_publish.verify_publication(str(a), kind="prepare")["published"] is False
    assert not (a / raf_publish.marker_name("prepare")).exists()


def test_verify_publication_rejects_a_generation_mismatch(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        ra, rb = txn.stage(str(a)), txn.stage(str(b))
        open(ra.path("one.txt"), "w").write("A")
        open(rb.path("two.txt"), "w").write("B")
        txn.commit()
    manifest_path = b / raf_publish.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = "0" * 32
    manifest_path.write_text(json.dumps(manifest))
    report = raf_publish.verify_publication(str(a), kind="prepare")
    assert report["published"] is False
    assert "generation" in report["reason"]


def test_verify_publication_rejects_a_tampered_payload(tmp_path):
    a = tmp_path / "a"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        root = txn.stage(str(a))
        open(root.path("one.txt"), "w").write("A")
        txn.commit()
    (a / "one.txt").write_text("tampered")
    report = raf_publish.verify_publication(str(a), kind="prepare")
    assert report["published"] is False
    assert report["roots"][str(a.resolve())]["mismatched"] == ["one.txt"]


def test_prepare_publishes_one_marker_covering_both_roots(tmp_path):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    report = raf_publish.verify_publication(str(split_dir), kind="prepare")
    assert report["published"] is True
    assert set(report["roots"]) == {str(out.resolve()), str(split_dir.resolve())}
    assert not (out / raf_publish.marker_name("prepare")).exists()   # exactly one marker


def test_prepare_failure_leaves_the_previous_generation_unattested(tmp_path, monkeypatch):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    first = json.loads((split_dir / raf_publish.marker_name("prepare")).read_text())

    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith("train_base.json"):
            raise OSError("NAS went away")
        return real_replace(src, dst)

    with mock.patch("os.replace", failing_replace):
        with pytest.raises(OSError):
            raf_prepare.main(argv)
    report = raf_publish.verify_publication(str(split_dir), kind="prepare")
    assert report["published"] is False
    assert first["generation"]


def test_render_depth_publishes_one_marker_covering_both_rooms(tmp_path):
    from test_raf_render_depth import _write_fixture

    raf_root, out, _ = _write_fixture(tmp_path)
    record = write_passing_readback_record(str(tmp_path / "readback.json"))
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", record,
                     "--non-canonical"])
    report = raf_publish.verify_publication(str(out), kind="depth")
    assert report["published"] is True
    assert str((out / "EmptyRoom" / "depth_images").resolve()) in report["roots"]


# --------------------------------------------------------------------------- #
# r4 T4: namespaced markers, validated root sets, composition
# --------------------------------------------------------------------------- #
def test_markers_are_namespaced_per_transaction_kind(tmp_path):
    """Both markers live at the same marker root -- as they do in production, where
    the depth marker sits at the runtime root that prepare also publishes -- and
    neither disturbs the other. Their payload ROOTS are disjoint (prepare owns the
    runtime tree and the split dir, depth owns each room's depth_images), which is
    the invariant one manifest per directory rests on."""
    marker_root = tmp_path / "runtime"
    prepare_root = marker_root / "tree"
    depth_root = marker_root / "EmptyRoom" / "depth_images"
    with raf_publish.PublishTransaction(str(marker_root), kind="prepare") as txn:
        open(txn.stage(str(prepare_root)).path("one.txt"), "w").write("A")
        txn.commit()
    with raf_publish.PublishTransaction(str(marker_root), kind="depth") as txn:
        open(txn.stage(str(depth_root)).path("two.npy"), "w").write("B")
        txn.commit()
    assert (marker_root / raf_publish.marker_name("prepare")).exists()
    assert (marker_root / raf_publish.marker_name("depth")).exists()
    assert raf_publish.verify_publication(str(marker_root), kind="prepare")["published"]
    assert raf_publish.verify_publication(str(marker_root), kind="depth")["published"]


def test_two_kinds_sharing_one_payload_root_fail_safe(tmp_path):
    """One manifest per directory: if a second kind ever published into the same
    payload root, the first attestation must read as UNPUBLISHED rather than as
    still-valid."""
    marker_root = tmp_path / "runtime"
    shared = marker_root / "shared"
    with raf_publish.PublishTransaction(str(marker_root), kind="prepare") as txn:
        open(txn.stage(str(shared)).path("one.txt"), "w").write("A")
        txn.commit()
    with raf_publish.PublishTransaction(str(marker_root), kind="depth") as txn:
        open(txn.stage(str(shared)).path("two.npy"), "w").write("B")
        txn.commit()
    stale = raf_publish.verify_publication(str(marker_root), kind="prepare")
    assert stale["published"] is False
    assert "manifest" in stale["reason"]


def test_a_prepare_rerun_does_not_orphan_the_depth_attestation(tmp_path):
    """T4: prepare staged the runtime root and renamed the render marker aside,
    although it never regenerates depth -- orphaning valid depth evidence."""
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)

    from test_raf_render_depth import _write_fixture

    depth_root = out / "EmptyRoom" / "depth_images"
    depth_root.mkdir(parents=True, exist_ok=True)
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        open(txn.stage(str(depth_root)).path("map.npy"), "w").write("depth")
        depth_marker = txn.commit()

    raf_prepare.main(argv)          # prepare AGAIN
    after = raf_publish.verify_publication(str(out), kind="depth")
    assert after["published"] is True
    assert after["generation"] == depth_marker["generation"]


def test_verify_publication_rejects_an_empty_root_set(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    marker = {"generation": "0" * 32, "kind": "prepare", "committed_utc": "now",
              "marker_root": str(a), "roots": {}}
    (a / raf_publish.marker_name("prepare")).write_text(json.dumps(marker))
    report = raf_publish.verify_publication(str(a), kind="prepare")
    assert report["published"] is False
    assert "no roots" in report["reason"]


def test_verify_publication_requires_the_expected_root_set(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        open(txn.stage(str(a)).path("one.txt"), "w").write("A")
        txn.commit()
    ok = raf_publish.verify_publication(str(a), kind="prepare",
                                        expected_roots=[str(a)])
    assert ok["published"] is True
    missing = raf_publish.verify_publication(str(a), kind="prepare",
                                             expected_roots=[str(a), str(b)])
    assert missing["published"] is False
    assert "expected roots" in missing["reason"]


def test_render_depth_marker_requires_every_canonical_room(tmp_path):
    """A FurnishedRoom-only depth publish must not read as a complete one."""
    from test_raf_render_depth import _write_fixture

    raf_root, out, _ = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record",
                     write_passing_readback_record(str(tmp_path / "rb.json")),
                     "--non-canonical"])
    partial = raf_publish.verify_publication(
        str(out), kind="depth",
        expected_roots=[str(out / room / "depth_images")
                        for room in ("EmptyRoom", "FurnishedRoom")])
    assert partial["published"] is False
    assert "expected roots" in partial["reason"]


def test_verify_combined_publication_requires_both_kinds(tmp_path):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    report = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                     rooms=["EmptyRoom"],
                                                     canonical=False)
    assert report["published"] is False          # depth has not been rendered yet
    assert "depth" in report["reason"]

    depth_root = out / "EmptyRoom" / "depth_images"
    depth_root.mkdir(parents=True, exist_ok=True)
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        open(txn.stage(str(depth_root)).path("map.npy"), "w").write("depth")
        txn.commit()
    complete = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                       rooms=["EmptyRoom"],
                                                       canonical=False)
    assert complete["published"] is True
    assert set(complete["kinds"]) == {"prepare", "depth"}
    # ... but a one-room publication is NOT canonical completeness (r5 finding 3)
    with pytest.raises(ValueError) as exc:
        raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                rooms=["EmptyRoom"], canonical=True)
    assert "canonical publication covers exactly" in str(exc.value)


def test_duplicate_roots_in_one_transaction_are_rejected(tmp_path):
    a = tmp_path / "a"
    with pytest.raises(ValueError) as exc:
        with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
            txn.stage(str(a))
            txn.stage(str(a))
    assert "duplicate" in str(exc.value)


def test_a_transaction_with_no_roots_refuses_to_commit(tmp_path):
    with pytest.raises(ValueError) as exc:
        with raf_publish.PublishTransaction(str(tmp_path / "a"), kind="prepare") as txn:
            txn.commit()
    assert "no roots" in str(exc.value)


def test_prepare_render_prepare_composition_keeps_both_attestations(tmp_path):
    from test_raf_render_depth import _write_fixture

    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    raf_root, fixture_out, _ = _write_fixture(tmp_path / "fixture")
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(fixture_out),
                     "--rooms", "EmptyRoom", "--readback-record",
                     write_passing_readback_record(str(tmp_path / "rb2.json")),
                     "--non-canonical"])
    raf_prepare.main(argv)
    assert raf_publish.verify_publication(str(split_dir), kind="prepare")["published"]
    assert raf_publish.verify_publication(str(fixture_out), kind="depth")["published"]


def test_a_crash_in_the_second_kind_leaves_the_first_attestation_intact(tmp_path):
    argv, out, split_dir = _prepare(tmp_path)
    raf_prepare.main(argv)
    before = json.loads((split_dir / raf_publish.marker_name("prepare")).read_text())

    depth_root = out / "EmptyRoom" / "depth_images"
    depth_root.mkdir(parents=True, exist_ok=True)
    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).endswith("map.npy"):
            raise OSError("crash during the depth transaction")
        return real_replace(src, dst)

    with mock.patch("os.replace", failing_replace):
        with pytest.raises(OSError):
            with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
                open(txn.stage(str(depth_root)).path("map.npy"), "w").write("depth")
                txn.commit()

    after = json.loads((split_dir / raf_publish.marker_name("prepare")).read_text())
    assert after == before
    assert raf_publish.verify_publication(str(out), kind="depth")["published"] is False


# --------------------------------------------------------------------------- #
# r5 finding 3: completeness is internal, provenance is validated
# --------------------------------------------------------------------------- #
def test_canonical_combined_verification_ignores_a_narrower_caller_room_list(tmp_path):
    """Passing ['EmptyRoom'] used to make a one-room depth marker 'complete'."""
    with pytest.raises(ValueError) as exc:
        raf_publish.verify_combined_publication(str(tmp_path / "s"), str(tmp_path / "o"),
                                                rooms=["EmptyRoom"], canonical=True)
    assert "canonical publication covers exactly" in str(exc.value)
    assert raf_publish.resolve_rooms(None, canonical=True) == list(
        raf_publish.CANONICAL_ROOMS)


@pytest.mark.parametrize("rooms", [[], ["EmptyRoom", "EmptyRoom"]])
def test_non_canonical_verification_rejects_empty_or_duplicate_rooms(tmp_path, rooms):
    with pytest.raises(ValueError):
        raf_publish.verify_combined_publication(str(tmp_path / "s"), str(tmp_path / "o"),
                                                rooms=rooms, canonical=False)


def test_duplicate_expected_roots_are_reported_not_collapsed(tmp_path):
    a = tmp_path / "a"
    with raf_publish.PublishTransaction(str(a), kind="prepare") as txn:
        open(txn.stage(str(a)).path("one.txt"), "w").write("A")
        txn.commit()
    report = raf_publish.verify_publication(str(a), kind="prepare",
                                            expected_roots=[str(a), str(a)])
    assert report["published"] is False
    assert "duplicate expected roots" in report["reason"]


def _prepare_extra(**overrides):
    extra = {"canonical": True, "canonical_parameters": True, "taint": [],
             "parameters": dict(raf_publish.CANONICAL_PREPARE_PARAMS,
                                amplitude_derivation_sha256=raf_publish.CANONICAL_PREPARE_PARAMS[
                                    "amplitude_derivation_sha256"]),
             "readback_record": {"sha256": raf_publish.canonical_record_digest()}}
    extra.update(overrides)
    return extra


def _depth_extra(**overrides):
    extra = {"canonical": True, "canonical_parameters": True, "taint": [],
             "parameters": dict(raf_publish.CANONICAL_RENDER_PARAMS),
             "readback_record": {"sha256": raf_publish.canonical_record_digest()}}
    extra.update(overrides)
    return extra


def test_canonical_combined_verification_validates_marker_provenance(tmp_path):
    """A tainted or non-canonical marker cannot attest a canonical publication."""
    split_dir, out = tmp_path / "splits", tmp_path / "runtime"
    rooms = list(raf_publish.CANONICAL_ROOMS)
    with raf_publish.PublishTransaction(str(split_dir), kind="prepare") as txn:
        open(txn.stage(str(out)).path("x.txt"), "w").write("x")
        open(txn.stage(str(split_dir)).path("y.txt"), "w").write("y")
        txn.commit(extra=_prepare_extra())
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        for room in rooms:
            open(txn.stage(str(out / room / "depth_images")).path("m.npy"), "w").write("d")
        txn.commit(extra=dict(_depth_extra(), taint=["non-registered render parameters"]))
    report = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                     canonical=True)
    assert report["published"] is False
    assert "tainted" in report["reason"]


def _pointer(rooms, flavor="mappingH", **overrides):
    """The runtime pointer a canonical publication must also agree with (r2 N5)."""
    pointer = {"rooms": list(rooms), "canonical": True, "taint": [],
               "parameters": {"rooms": list(rooms)},
               "readback_record": {"sha256": raf_publish.canonical_record_digest()}}
    if flavor != "mappingH":
        pointer["flavor"] = flavor
    pointer.update(overrides)
    return pointer


def test_canonical_combined_verification_accepts_a_clean_pair(tmp_path):
    split_dir, out = tmp_path / "splits", tmp_path / "runtime"
    rooms = list(raf_publish.CANONICAL_ROOMS)
    with raf_publish.PublishTransaction(str(split_dir), kind="prepare") as txn:
        staged = txn.stage(str(out))
        with open(staged.path(raf_publish.PUBLICATION_POINTER), "w") as f:
            json.dump(_pointer(rooms), f)
        open(staged.path("x.txt"), "w").write("x")
        open(txn.stage(str(split_dir)).path("y.txt"), "w").write("y")
        txn.commit(extra=_prepare_extra())
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        for room in rooms:
            open(txn.stage(str(out / room / "depth_images")).path("m.npy"), "w").write("d")
        txn.commit(extra=_depth_extra())
    report = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                     canonical=True)
    assert report["published"] is True, report["reason"]
    assert report["rooms"] == rooms


def test_a_pointer_that_disagrees_with_the_markers_fails_verification(tmp_path):
    """r2 N5: a reader reaches the publication THROUGH the pointer, so a stale one
    beside a fresh generation would hand out the wrong identity."""
    split_dir, out = tmp_path / "splits", tmp_path / "runtime"
    rooms = list(raf_publish.CANONICAL_ROOMS)
    with raf_publish.PublishTransaction(str(split_dir), kind="prepare") as txn:
        staged = txn.stage(str(out))
        with open(staged.path(raf_publish.PUBLICATION_POINTER), "w") as f:
            json.dump(_pointer(rooms, parameters={"n_groups": 99},
                               readback_record={"sha256": "d" * 64}), f)
        open(staged.path("x.txt"), "w").write("x")
        open(txn.stage(str(split_dir)).path("y.txt"), "w").write("y")
        txn.commit(extra=_prepare_extra())
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        for room in rooms:
            open(txn.stage(str(out / room / "depth_images")).path("m.npy"), "w").write("d")
        txn.commit(extra=_depth_extra())
    report = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                     canonical=True)
    assert report["published"] is False
    assert "pointer parameter n_groups=99" in report["reason"]
    assert "readback digests disagree" in report["reason"]


def test_a_missing_pointer_fails_a_canonical_verification(tmp_path):
    split_dir, out = tmp_path / "splits", tmp_path / "runtime"
    rooms = list(raf_publish.CANONICAL_ROOMS)
    with raf_publish.PublishTransaction(str(split_dir), kind="prepare") as txn:
        open(txn.stage(str(out)).path("x.txt"), "w").write("x")
        open(txn.stage(str(split_dir)).path("y.txt"), "w").write("y")
        txn.commit(extra=_prepare_extra())
    with raf_publish.PublishTransaction(str(out), kind="depth") as txn:
        for room in rooms:
            open(txn.stage(str(out / room / "depth_images")).path("m.npy"), "w").write("d")
        txn.commit(extra=_depth_extra())
    report = raf_publish.verify_combined_publication(str(split_dir), str(out),
                                                     canonical=True)
    assert report["published"] is False
    assert "names no publication" in report["reason"]


# --------------------------------------------------------------------------- #
# r6 F3: the consumer verifies identities, never the producer's booleans
# --------------------------------------------------------------------------- #
def test_marker_identity_accepts_the_registered_payload():
    assert raf_publish.marker_identity_problems("prepare", _prepare_extra()) == []
    assert raf_publish.marker_identity_problems("depth", _depth_extra()) == []


def test_marker_identity_rejects_a_bare_canonical_claim():
    """The r5 oracle accepted canonical markers with no parameters and no digest."""
    problems = raf_publish.marker_identity_problems(
        "prepare", {"canonical": True, "canonical_parameters": True, "taint": []})
    assert any("no parameter payload" in p for p in problems)
    assert any("readback digest" in p for p in problems)


@pytest.mark.parametrize("kind,mutate,needle", [
    ("prepare", {"parameters": dict(raf_publish.CANONICAL_PREPARE_PARAMS, seed=999)},
     "seed"),
    ("prepare", {"parameters": dict(raf_publish.CANONICAL_PREPARE_PARAMS, n_train=6)},
     "n_train"),
    ("prepare", {"parameters": dict(raf_publish.CANONICAL_PREPARE_PARAMS,
                                    rooms=["EmptyRoom"])}, "rooms"),
    ("depth", {"parameters": dict(raf_publish.CANONICAL_RENDER_PARAMS, floor_tol=5.0)},
     "floor_tol"),
    ("depth", {"parameters": dict(raf_publish.CANONICAL_RENDER_PARAMS,
                                  max_miss_rate=0.0001)}, "max_miss_rate"),
])
def test_marker_identity_rejects_altered_parameters(kind, mutate, needle):
    extra = (_prepare_extra if kind == "prepare" else _depth_extra)(**mutate)
    problems = raf_publish.marker_identity_problems(kind, extra)
    assert any(needle in p for p in problems)


def test_marker_identity_rejects_missing_and_extra_parameters():
    missing = dict(raf_publish.CANONICAL_PREPARE_PARAMS)
    missing.pop("seed")
    problems = raf_publish.marker_identity_problems(
        "prepare", _prepare_extra(parameters=missing))
    assert any("missing ['seed']" in p for p in problems)

    extra = dict(raf_publish.CANONICAL_PREPARE_PARAMS, smuggled=1)
    problems = raf_publish.marker_identity_problems(
        "prepare", _prepare_extra(parameters=extra))
    assert any("unregistered ['smuggled']" in p for p in problems)


def test_marker_identity_rejects_a_wrong_or_missing_digest():
    problems = raf_publish.marker_identity_problems(
        "depth", _depth_extra(readback_record={"sha256": "0" * 64}))
    assert any("readback digest" in p for p in problems)
    problems = raf_publish.marker_identity_problems("depth", _depth_extra(readback_record=None))
    assert any("readback digest" in p for p in problems)


def test_the_registered_digest_is_the_pinned_one():
    import readback_audit as raf_readback

    assert raf_publish.canonical_record_digest() == raf_readback.CANONICAL_RECORD_SHA256


def test_producers_and_verifier_share_one_registered_identity():
    import prepare_data as raf_prepare_mod
    import render_depth as raf_render_mod

    assert dict(raf_prepare_mod.CANONICAL_PARAMS, rooms=list(
        raf_prepare_mod.CANONICAL_PARAMS["rooms"])) == raf_publish.CANONICAL_PREPARE_PARAMS
    assert dict(raf_render_mod.CANONICAL_RENDER_PARAMS, rooms=list(
        raf_render_mod.CANONICAL_RENDER_PARAMS["rooms"])) == raf_publish.CANONICAL_RENDER_PARAMS


def test_the_canonical_derivation_hash_is_pinned_not_shaped():
    """Amendment 9.2 F3, closed: the derivation hash of canonical generation
    46a43f4ce82b is registered as a literal, so a marker carrying a DIFFERENT
    trained-ID union is rejected on its value, not merely on its shape."""
    pinned = raf_publish.CANONICAL_PREPARE_PARAMS["amplitude_derivation_sha256"]
    assert pinned == (
        "8a740feef8f430dbc2e65d8f3d5eefa3d6b191c00c615ff758163c7428eef00d")
    assert raf_publish.marker_identity_problems("prepare", _prepare_extra()) == []
    other = _prepare_extra(parameters=dict(
        raf_publish.CANONICAL_PREPARE_PARAMS, amplitude_derivation_sha256="b" * 64))
    problems = raf_publish.marker_identity_problems("prepare", other)
    assert any("amplitude_derivation_sha256" in p for p in problems)
