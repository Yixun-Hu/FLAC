"""exp_36, cycle 13 (Codex code review II HIGH 1, HIGH 2): a serialized publication and an authenticated manifest.

HIGH 1 -- one nonblocking global publication lock (flock on the destination worktree's root directory: every publisher,
any set, any checkout, contends on one inode that a swap of the destination path does not replace) held from before
the first check through the final manifest and directory fsync; the destination opened ONCE (O_DIRECTORY | O_NOFOLLOW,
its inode checked against the path) and every temp creation, link, hash and unlink done relative to that fd; a
transaction id per publication names its temps, and a resume removes only the temps its validated prior manifest names.
HIGH 2 -- the manifest is created exclusively (O_EXCL | O_NOFOLLOW, file + directory fsync), replaced only while it is
still the file this publication wrote (inode + bytes) through an exclusive no-follow temp, and a resume requires a
regular non-symlink manifest of exactly this publication (check, set, pin, digest, base / destination, exp-25 HEAD,
verdict path + hash, publisher, the complete entry set with its source hashes, status in_progress | partial) with no
writer temp beside it.

Concurrency is exercised in-process: a publication paused inside its third link (a thread, holding the lock), the
competitor started meanwhile; flock locks belong to open file descriptions, so the two contend exactly as two processes.

Written by the exp_36 Coder (Claude Opus 5.5, max effort) per plan_yaw_c4_spread.md r0 + A1 + Codex code review II.
"""
import fcntl
import hashlib
import json
import os
import threading

import pytest

import test_depth_canon_collect as T
from test_exp36_publish import CELLS24, YAWOFF, _listing, pub, publish  # noqa: F401
from test_exp36_publish_publisher import both  # noqa: F401
from test_exp36_replay_check import world  # noqa: F401

C = T.C


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _paused(p, monkeypatch, at=3, **kw):
    """``publish(p, **kw)`` in a thread, blocked right after its ``at``-th link (it holds the lock); returns
    ``(thread, box, go)`` -- set ``go`` to let it finish, its result lands in ``box["result"]``."""
    real, n, paused, go, box = os.link, [0], threading.Event(), threading.Event(), {}

    def link(*a, **k):
        n[0] += 1
        r = real(*a, **k)
        if n[0] == at:
            paused.set()
            assert go.wait(120)
        return r

    monkeypatch.setattr(C.os, "link", link)
    t = threading.Thread(target=lambda: box.update(result=publish(p, **kw)))
    t.start()
    assert paused.wait(120), "the publication never reached its link"
    return t, box, go


def _finish(t, go):
    go.set()
    t.join(120)
    assert not t.is_alive()


# ---- HIGH 1 ---- #
def test_a_held_lock_refuses_before_anything_and_the_lock_spans_every_write(pub, monkeypatch):
    fd = os.open(pub["ref"], os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        before = _listing(pub["dest"])
        status, doc = publish(pub)
        assert status == "refused" and any("lock" in x for x in doc["reasons"]), doc
        assert _listing(pub["dest"]) == before and not os.path.lexists(pub["out"])
    finally:
        os.close(fd)
    held, real = [], C._fsync_dir

    def fsync_dir(path):                    # every manifest / destination fsync happens while the lock is held
        probe = os.open(pub["ref"], os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(False)
        except BlockingIOError:
            held.append(True)
        finally:
            os.close(probe)
        return real(path)

    monkeypatch.setattr(C, "_fsync_dir", fsync_dir)
    status, doc = publish(pub)
    assert status == "published" and len(held) >= 2 and all(held), (status, held)


def test_two_concurrent_same_set_publications(pub, monkeypatch):
    t, box, go = _paused(pub, monkeypatch)
    before, manifest = _listing(pub["dest"]), _bytes(pub["out"])
    status, doc = publish(pub)                                            # the same set, the same manifest path
    assert status == "refused" and any("lock" in x for x in doc["reasons"]), doc
    assert _listing(pub["dest"]) == before and _bytes(pub["out"]) == manifest
    _finish(t, go)
    assert box["result"][0] == "published", box["result"][1].get("reasons")


def test_concurrent_a_and_off_publications(both, tmp_path, monkeypatch):
    t, box, go = _paused(both, monkeypatch, cell_set="A")
    before, off_out = _listing(both["dest"]), str(tmp_path / "publication_OFF.json")
    status, doc = publish(both, cell_set="OFF", out=off_out)
    assert status == "refused" and any("lock" in x for x in doc["reasons"]), doc
    assert _listing(both["dest"]) == before and not os.path.lexists(off_out)
    _finish(t, go)
    assert box["result"][0] == "published"
    assert publish(both, cell_set="OFF", out=off_out)[0] == "published"   # then, serialized, it runs


def test_a_destination_path_swap_cannot_redirect_the_owner(pub, tmp_path, monkeypatch):
    t, box, go = _paused(pub, monkeypatch)
    moved = pub["dest"] + ".moved"
    os.rename(pub["dest"], moved)
    os.mkdir(pub["dest"])                                                 # the path now names another directory
    status, doc = publish(pub, out=str(tmp_path / "second.json"))
    assert status == "refused" and any("lock" in x for x in doc["reasons"]) and os.listdir(pub["dest"]) == []
    _finish(t, go)
    status, doc = box["result"]
    assert status == "partial" and any("no longer names" in x for x in doc["reasons"]), doc
    assert os.listdir(pub["dest"]) == []                                  # nothing redirected into the new directory
    for e in doc["entries"]:                                              # everything went through the locked fd
        assert e["state"] == "published" and hashlib.sha256(
            _bytes(os.path.join(moved, os.path.basename(e["dest"])))).hexdigest() == e["source_sha256"]


def test_temps_are_named_by_the_transaction_and_a_resume_removes_only_those(pub, monkeypatch):
    real, n = os.link, [0]

    def link(*a, **k):
        n[0] += 1
        if n[0] == 4:
            raise OSError("simulated crash")
        return real(*a, **k)

    monkeypatch.setattr(C.os, "link", link)
    assert publish(pub)[0] == "partial"
    monkeypatch.setattr(C.os, "link", real)
    doc = json.loads(_bytes(pub["out"]))
    assert len(doc["txid"]) == 32 and all(e["tmp"] == f"{C.EXP36_TMP_PREFIX}{doc['txid']}-{os.path.basename(e['dest'])}"
                                          for e in doc["entries"])
    ours = os.path.join(pub["dest"], doc["entries"][10]["tmp"])          # a hard crash between write and link
    open(ours, "w").close()
    foreign = os.path.join(pub["dest"], f"{C.EXP36_TMP_PREFIX}{'0' * 32}-{os.path.basename(doc['entries'][11]['dest'])}")
    open(foreign, "w").close()                                            # another transaction's temp
    status, d2 = publish(pub, resume=True)
    assert status == "refused" and any("temp" in x for x in d2["reasons"]) and os.path.exists(ours), d2
    os.unlink(foreign)
    status, d2 = publish(pub, resume=True)
    assert status == "published" and not os.path.lexists(ours) and d2["txid"] == doc["txid"], d2.get("reasons")


# ---- HIGH 2 ---- #
@pytest.fixture()
def stopped(pub, monkeypatch):
    """A publication stopped at its 4th link: a partial manifest to resume."""
    real, n = os.link, [0]

    def link(*a, **k):
        n[0] += 1
        if n[0] == 4:
            raise OSError("simulated crash")
        return real(*a, **k)

    monkeypatch.setattr(C.os, "link", link)
    assert publish(pub)[0] == "partial"
    monkeypatch.setattr(C.os, "link", real)
    return pub


def _edit(path, fn):
    doc = json.loads(_bytes(path))
    fn(doc)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def _symlinked(p):
    os.rename(p["out"], p["out"] + ".real")
    os.symlink(p["out"] + ".real", p["out"])


RESUME = [
    (lambda p: os.unlink(p["out"]), "no regular"),
    (lambda p: open(p["out"], "w").write("{}"), "not this publication"),
    (lambda p: open(p["out"], "w").write("not a manifest"), "not a JSON"),
    (_symlinked, "no regular"),
    (lambda p: _edit(p["out"], lambda d: d.update(set="OFF")), "not this publication"),
    (lambda p: open(p["verdict"], "a").write("\n"), "not this publication"),   # the same parsed verdict, other bytes
    (lambda p: _edit(p["out"], lambda d: d["entries"].pop()), "entry set"),
    (lambda p: _edit(p["out"], lambda d: d["entries"][3].update(source_sha256="0" * 64)), "entry set"),
    (lambda p: _edit(p["out"], lambda d: d.update(status="complete")), "status"),
    (lambda p: _edit(p["out"], lambda d: d.update(txid="x")), "transaction"),
    (lambda p: open(p["out"] + ".tmp-0-1", "w").close(), "writer temp"),
]


@pytest.mark.parametrize("edit, needle", RESUME, ids=["none", "json", "file", "symlink", "other_set", "verdict",
                                                      "entries", "source", "complete", "txid", "writer_temp"])
def test_a_resume_authenticates_the_prior_manifest(stopped, edit, needle):
    edit(stopped)
    before = _listing(stopped["dest"])
    state = os.path.islink(stopped["out"]), _bytes(stopped["out"]) if os.path.lexists(stopped["out"]) else None
    status, doc = publish(stopped, resume=True)
    assert status == "refused" and any(needle in x for x in doc["reasons"]), doc
    assert _listing(stopped["dest"]) == before
    assert (os.path.islink(stopped["out"]), _bytes(stopped["out"]) if os.path.lexists(stopped["out"]) else None) == state


def test_the_initial_manifest_is_created_exclusively(pub, tmp_path):
    os.symlink(str(tmp_path / "elsewhere.json"), pub["out"])              # a dangling symlink at the manifest path
    status, doc = publish(pub)
    assert status == "refused" and not os.path.lexists(tmp_path / "elsewhere.json"), doc
    os.unlink(pub["out"])
    assert publish(pub)[0] == "published" and not os.path.islink(pub["out"])


def test_an_update_never_replaces_a_manifest_it_did_not_write(pub, monkeypatch):
    t, box, go = _paused(pub, monkeypatch)
    os.rename(pub["out"], pub["out"] + ".mine")
    with open(pub["out"], "w") as f:                                      # the manifest swapped under the publication
        f.write("foreign\n")
    _finish(t, go)
    status, doc = box["result"]
    assert status == "partial" and any("changed under the publication" in x for x in doc["reasons"]), doc
    assert _bytes(pub["out"]) == b"foreign\n"                             # never replaced
    assert not [n for n in os.listdir(os.path.dirname(pub["out"])) if ".tmp-" in n]   # its own temp removed
