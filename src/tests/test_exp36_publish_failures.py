"""exp_36, cycle 16 (Codex code review III HIGH 3; fixed after review III, not re-reviewed (cap)): an internal I/O
failure never strands a publication.

* ``_write_new`` removes the file it created -- only while the name is still that inode -- when its write, fsync or
  close fails; a pre-existing name fails the exclusive open before anything is written and is never touched.
* ``_Manifest.write`` removes its transaction's writer temp whenever the replacement did not consume it (the temp's own
  write / fsync, the identity check or ``os.replace`` failed), and takes the new manifest's identity as soon as it is
  replaced (a failed directory fsync afterwards does not make the next write see "changed under the publication").
* the publication: a failure in the INITIAL manifest write crashes with no manifest and nothing written (a plain retry
  publishes; the CLI says so); a failure in any later manifest update, in a destination temp's write or in a directory
  fsync stops it ``partial`` (the stop recorded or, when even that write fails, the last in_progress manifest left) with
  no writer temp and no destination temp behind, and ``--exp36-publish-resume`` then publishes all 48 files. A
  genuinely foreign writer temp is still refused (here and in test_exp36_publish_concurrency).

Failures are injected through a proxy of the collector's own ``os`` (pytest and every other module keep the real one):
EIO on the n-th call of ``write`` / ``fsync`` / ``replace`` whose target matches.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per Codex code review III HIGH 3.
"""
import errno
import json
import os
import types

import pytest

import test_depth_canon_collect as T
from test_exp36_publish import _cli, _listing, pub, publish  # noqa: F401
from test_exp36_replay_check import world  # noqa: F401

C = T.C


def _path_of(fd):
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return ""


def inject(monkeypatch, name, match, at=1, half=False, every=False):
    """The collector's ``os.<name>`` fails with EIO on the ``at``-th call (``every``: on each call from the ``at``-th)
    whose target -- the fd's file for write / fsync, the source path for replace -- satisfies ``match``; ``half``: a
    failing write first writes half its bytes (ignored for fsync / replace). Returns the list of matching targets seen."""
    proxy = C.os if isinstance(C.os, types.SimpleNamespace) else types.SimpleNamespace(**vars(os))
    real, hits = getattr(os, name), []

    def fn(*a, **k):
        target = _path_of(a[0]) if name in ("write", "fsync") else os.fspath(a[0])
        if match(target):
            hits.append(target)
            if len(hits) == at or (every and len(hits) >= at):
                if half and name == "write":
                    real(a[0], bytes(a[1])[: max(1, len(a[1]) // 2)])
                raise OSError(errno.EIO, f"injected {name} failure on {target}")
        return real(*a, **k)

    setattr(proxy, name, fn)
    monkeypatch.setattr(C, "os", proxy)
    return hits


def _beside(path):
    d, b = os.path.split(path)
    return sorted(n for n in os.listdir(d) if n.startswith(b + ".tmp-"))


def _dest_temps(p):
    return sorted(n for n in os.listdir(p["dest"]) if n.startswith(C.EXP36_TMP_PREFIX))


def _completes(p):
    status, doc = publish(p, resume=True)
    assert status == "published" and doc["status"] == "complete", doc.get("reasons")
    assert sum(e["state"] in ("published", C._PRESENT) for e in doc["entries"]) == 48
    assert _beside(p["out"]) == [] and _dest_temps(p) == []


# ---- _write_new ---- #
@pytest.mark.parametrize("name", ["write", "fsync"])
def test_write_new_removes_the_file_it_created_on_a_failure(tmp_path, monkeypatch, name):
    path = str(tmp_path / "new.json")
    inject(monkeypatch, name, lambda t: t == os.path.realpath(path), half=True)
    with pytest.raises(OSError, match=f"injected {name}"):
        C._write_new(path, b"x" * 4096)
    assert not os.path.lexists(path)
    C._write_new(path, b"y")                                              # the retry (its call is not failed)
    assert open(path, "rb").read() == b"y"


def test_write_new_relative_to_a_directory_fd(tmp_path, monkeypatch):
    dfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        inject(monkeypatch, "fsync", lambda t: t == os.path.realpath(tmp_path / "rel.tmp"))
        with pytest.raises(OSError, match="injected fsync"):
            C._write_new("rel.tmp", b"z", dir_fd=dfd)
        assert not os.path.lexists(tmp_path / "rel.tmp")
    finally:
        os.close(dfd)


def test_write_new_never_touches_a_pre_existing_name(tmp_path):
    taken = tmp_path / "taken.json"
    taken.write_bytes(b"theirs")
    with pytest.raises(FileExistsError):
        C._write_new(str(taken), b"mine")
    link = tmp_path / "link.json"
    os.symlink(tmp_path / "nowhere", link)
    with pytest.raises(FileExistsError):
        C._write_new(str(link), b"mine")
    assert taken.read_bytes() == b"theirs" and os.path.islink(link) and not os.path.lexists(tmp_path / "nowhere")


def test_write_new_never_removes_a_name_that_is_no_longer_its_file(tmp_path, monkeypatch):
    path, real = str(tmp_path / "swapped.json"), os.write

    def write(fd, data):
        if _path_of(fd) == os.path.realpath(path):
            os.rename(path, path + ".moved")                              # the new file moved away ...
            with open(path, "wb") as f:                                   # ... and another put at its name
                f.write(b"another")
            raise OSError(errno.EIO, "injected write failure")
        return real(fd, data)

    proxy = types.SimpleNamespace(**vars(os))
    proxy.write = write
    monkeypatch.setattr(C, "os", proxy)
    with pytest.raises(OSError, match="injected write"):
        C._write_new(path, b"mine")
    assert open(path, "rb").read() == b"another" and os.path.exists(path + ".moved")


# ---- _Manifest.write ---- #
def _manifest(tmp_path):
    out = str(tmp_path / "m.json")
    m = C._Manifest(out, "a" * 32)
    m.write({"n": 0})
    return out, m


@pytest.mark.parametrize("name", ["write", "fsync"])
def test_a_failed_initial_manifest_write_leaves_no_manifest(tmp_path, monkeypatch, name):
    out = str(tmp_path / "m.json")
    inject(monkeypatch, name, lambda t: t == os.path.realpath(out), half=True)
    m = C._Manifest(out, "a" * 32)
    with pytest.raises(OSError, match=f"injected {name}"):
        m.write({"n": 0})
    assert not os.path.lexists(out) and _beside(out) == []
    m.write({"n": 1})
    assert json.load(open(out)) == {"n": 1}


UPDATE = [("write", True), ("fsync", False), ("replace", False)]


@pytest.mark.parametrize("name, half", UPDATE, ids=[n for n, _ in UPDATE])
def test_a_failed_manifest_update_removes_its_temp_and_the_next_one_replaces(tmp_path, monkeypatch, name, half):
    out, m = _manifest(tmp_path)
    before = open(out, "rb").read()
    prefix = (out if name == "replace" else os.path.realpath(out)) + ".tmp-"
    inject(monkeypatch, name, lambda t: t.startswith(prefix), half=half)
    with pytest.raises(OSError, match=f"injected {name}"):
        m.write({"n": 1})
    assert _beside(out) == [] and open(out, "rb").read() == before
    m.write({"n": 2})
    assert json.load(open(out)) == {"n": 2} and _beside(out) == []


def test_a_failed_directory_fsync_after_a_replace_keeps_the_manifests_identity(tmp_path, monkeypatch):
    out, m = _manifest(tmp_path)
    inject(monkeypatch, "fsync", lambda t: t == os.path.dirname(os.path.realpath(out)))
    with pytest.raises(OSError, match="injected fsync"):
        m.write({"n": 1})
    assert json.load(open(out)) == {"n": 1}                               # replaced; only its directory fsync failed
    m.write({"n": 2})                                                     # not "changed under the publication"
    assert json.load(open(out)) == {"n": 2} and _beside(out) == []


# ---- the publication ---- #
@pytest.mark.parametrize("name", ["write", "fsync"])
def test_a_failed_initial_manifest_write_crashes_with_nothing_written_and_a_plain_retry_publishes(pub, monkeypatch,
                                                                                                  name):
    before = _listing(pub["dest"])
    inject(monkeypatch, name, lambda t: t == os.path.realpath(pub["out"]), half=True)
    with pytest.raises(OSError, match=f"injected {name}"):
        publish(pub)
    assert not os.path.lexists(pub["out"]) and _beside(pub["out"]) == [] and _listing(pub["dest"]) == before
    status, doc = publish(pub)                                            # the same command, no resume
    assert status == "published" and doc["status"] == "complete", doc.get("reasons")


@pytest.mark.parametrize("name, half", UPDATE, ids=[n for n, _ in UPDATE])
def test_a_failed_manifest_update_stops_partial_with_no_temp_and_resume_publishes(pub, monkeypatch, name, half):
    prefix = (pub["out"] if name == "replace" else os.path.realpath(pub["out"])) + ".tmp-"
    inject(monkeypatch, name, lambda t: t.startswith(prefix), at=4, half=half)     # the 4th manifest update fails
    status, doc = publish(pub)
    assert status == "partial" and any(f"injected {name}" in x for x in doc["reasons"]), doc
    assert json.load(open(pub["out"]))["status"] == "partial"           # the stop is recorded
    assert _beside(pub["out"]) == [] and _dest_temps(pub) == []
    _completes(pub)


def test_when_even_the_stop_cannot_be_recorded_the_in_progress_manifest_resumes(pub, monkeypatch):
    inject(monkeypatch, "replace", lambda t: t.startswith(pub["out"] + ".tmp-"), at=4, every=True)
    status, doc = publish(pub)
    assert status == "partial" and any("could not record the stop" in x for x in doc["reasons"]), doc
    assert json.load(open(pub["out"]))["status"] == "in_progress" and _beside(pub["out"]) == []
    monkeypatch.setattr(C, "os", os)                                      # the disk recovers
    _completes(pub)


@pytest.mark.parametrize("name, half", [("write", True), ("fsync", False)])
def test_a_failed_destination_temp_write_stops_partial_with_no_temp_and_resume_publishes(pub, monkeypatch, name, half):
    inject(monkeypatch, name, lambda t: os.path.basename(t).startswith(C.EXP36_TMP_PREFIX), at=5, half=half)
    status, doc = publish(pub)
    assert status == "partial" and any(f"injected {name}" in x for x in doc["reasons"]), doc
    assert json.load(open(pub["out"]))["status"] == "partial" and _dest_temps(pub) == [] and _beside(pub["out"]) == []
    assert [e["state"] for e in doc["entries"]].count("published") == 4
    _completes(pub)


def test_a_failed_destination_fsync_stops_partial_and_resume_completes(pub, monkeypatch):
    inject(monkeypatch, "fsync", lambda t: t == os.path.realpath(pub["dest"]))
    status, doc = publish(pub)
    assert status == "partial" and all(e["state"] == "published" for e in doc["entries"]), doc
    _completes(pub)


def test_a_failed_manifest_directory_fsync_is_recorded_as_a_stop_and_resumes(pub, monkeypatch):
    parent = os.path.dirname(os.path.realpath(pub["out"]))
    inject(monkeypatch, "fsync", lambda t: t == parent, at=3)            # the directory fsync of the 2nd update
    status, doc = publish(pub)
    assert status == "partial" and json.load(open(pub["out"]))["status"] == "partial", doc
    assert not any("could not record" in x for x in doc["reasons"]) and _beside(pub["out"]) == []
    _completes(pub)


def test_a_resume_whose_first_manifest_write_fails_crashes_and_a_second_resume_publishes(pub, monkeypatch):
    inject(monkeypatch, "replace", lambda t: t.startswith(pub["out"] + ".tmp-"), at=3)
    assert publish(pub)[0] == "partial"                                   # the 3rd update failed, the 4th recorded it
    prior = open(pub["out"], "rb").read()
    inject(monkeypatch, "replace", lambda t: t.startswith(pub["out"] + ".tmp-"), at=1)      # a fresh counter
    with pytest.raises(OSError, match="injected replace"):
        publish(pub, resume=True)                                         # its first manifest write fails: a crash
    assert open(pub["out"], "rb").read() == prior and _beside(pub["out"]) == []
    _completes(pub)


def test_a_foreign_writer_temp_is_still_refused_after_a_cleaned_failure(pub, monkeypatch):
    inject(monkeypatch, "replace", lambda t: t.startswith(pub["out"] + ".tmp-"), at=2)
    assert publish(pub)[0] == "partial" and _beside(pub["out"]) == []
    open(pub["out"] + ".tmp-" + "0" * 32 + "-1", "w").close()            # not this transaction's
    status, doc = publish(pub, resume=True)
    assert status == "refused" and any("writer temp" in x for x in doc["reasons"]), doc


def test_cli_after_a_crash_before_any_write_says_to_rerun_the_same_command(pub, tmp_path, monkeypatch, capsys):
    out = tmp_path / "publication_A.json"
    inject(monkeypatch, "fsync", lambda t: t == os.path.realpath(out))
    assert _cli(pub, tmp_path, monkeypatch, "--exp36-set", "A") == 3
    text = capsys.readouterr().out
    assert "CRASHED" in text and "nothing was published" in text and not os.path.lexists(out), text
    assert _cli(pub, tmp_path, monkeypatch, "--exp36-set", "A") == 0
