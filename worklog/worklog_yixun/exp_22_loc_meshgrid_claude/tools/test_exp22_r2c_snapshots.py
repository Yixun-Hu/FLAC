"""exp_22 A7 (Codex A7-C1 BLOCKING 1; A7 r2 (iv), r3 (g)): each file a line is judged by -- its log, T's JUnit file,
its outputs and inputs -- is read ONCE through a no-follow descriptor (bytes + device / inode); THOSE snapshots are
validated, THEIR hashes go into the dispatch record and the state, and the paths are authenticated against them again
right before the record, before every later line and before the final release: a file changed in between never
yields an admitted record, and the reservation is never released."""
import json
import os

import pytest

from conftest import E, S, NAMES


def _sha(raw):
    return E.sha256_bytes(raw)


# --------------------------------------------------------------------------- #
# the snapshot reader and the exclusive writer's identity
# --------------------------------------------------------------------------- #
def test_a_snapshot_is_one_no_follow_read_of_a_regular_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_bytes(b'{"a": 1}\n')
    s = E.snapshot(str(p))
    st = os.stat(p)
    assert s.raw == b'{"a": 1}\n' and s.sha256 == _sha(s.raw) and (s.dev, s.inode) == (st.st_dev, st.st_ino)
    os.symlink(p, tmp_path / "link.json")
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "dir").mkdir()
    for bad in ("link.json", "fifo", "dir", "absent"):
        assert E.snapshot(str(tmp_path / bad)) is None, bad                     # a link, a FIFO (no block), ...
    p.chmod(0)
    try:
        assert E.snapshot(str(p)) is None
    finally:
        p.chmod(0o644)


def test_the_exclusive_writer_returns_the_written_bytes_identity(tmp_path):
    s = E.write_exclusive_snap(str(tmp_path / "r.json"), b"record\n")
    st = os.stat(tmp_path / "r.json")
    assert s.sha256 == _sha(b"record\n") and (s.dev, s.inode) == (st.st_dev, st.st_ino)
    assert E.write_exclusive(str(tmp_path / "q.json"), b"q") == _sha(b"q")


# --------------------------------------------------------------------------- #
# a change between the validation and the record: never an admitted record, never a release
# --------------------------------------------------------------------------- #
def _mutate_after_evidence(monkeypatch, box, name, which, how):
    """Right after line ``name``'s evidence was evaluated -- before its record -- change ``which`` (log / junit /
    output) by ``how`` (append / replace: identical bytes on a new inode); returns the validated bytes."""
    ln, kept, real = box.lines[NAMES.index(name)], {}, S.Supervisor.evidence

    def evidence(self, line, rc, snaps):
        out = real(self, line, rc, snaps)
        if line["name"] == name:
            rel = {"log": line["log"], "junit": line["junit"], "output": (line["outputs"] or [None])[0]}[which]
            path = os.path.join(box.w, rel)
            kept["raw"], kept["rel"] = open(path, "rb").read(), rel
            if how == "append":
                open(path, "ab").write(b"\nNO-FLASH LAUNCH REFUSED (appended after the validation)\n")
            else:
                open(path + ".new", "wb").write(kept["raw"])
                os.replace(path + ".new", path)
        return out
    monkeypatch.setattr(S.Supervisor, "evidence", evidence)
    return ln, kept


@pytest.mark.parametrize("name, which, how", [("GV", "log", "append"), ("GV", "log", "replace"),
                                              ("T", "junit", "append"), ("T", "log", "append"),
                                              ("GV", "output", "append"), ("G3", "output", "replace")])
def test_a_file_changed_after_its_validation_never_yields_an_admitted_record(box, monkeypatch, name, which, how):
    ln, kept = _mutate_after_evidence(monkeypatch, box, name, which, how)
    assert box.sup().run() == 3
    st, doc = box.state(), json.load(open(box.path(ln["record"])))
    assert st["phase"] == "stopped" and st["lines"][-1]["name"] == name and not doc["admitted"]
    assert any("changed after it was validated" in r for r in doc["refusals"]) and os.path.exists(box.res)
    recorded = {"log": doc["log"]["sha256"], "junit": (doc["junit_xml"] or {}).get("sha256"),
                "output": doc["output_paths"][0]["sha256"] if doc["output_paths"] else None}[which]
    assert recorded == _sha(kept["raw"])                                     # the bytes that were VALIDATED
    assert st["lines"][-1]["files"][kept["rel"]]["sha256"] == _sha(kept["raw"])


def test_the_dispatch_record_and_the_state_hash_the_validated_snapshot_not_a_later_read(box, monkeypatch):
    """No second read establishes an expected hash: the record and the state's baseline come from the snapshots."""
    rc = box.sup().run()
    st = box.state()
    assert rc == 0, st["stop"]
    for ln, done in zip(box.lines, st["lines"]):
        doc = json.load(open(box.path(ln["record"])))
        assert done["files"][ln["log"]]["sha256"] == doc["log"]["sha256"]
        assert done["files"][ln["record"]]["sha256"] == done["record_sha256"] == _sha(open(box.path(ln["record"]), "rb").read())
        assert all(done["files"][o["path"]]["sha256"] == o["sha256"] for o in doc["output_paths"])
        ident = os.stat(box.path(ln["log"]))
        assert [done["files"][ln["log"]]["dev"], done["files"][ln["log"]]["inode"]] == [ident.st_dev, ident.st_ino]


@pytest.mark.parametrize("change", ["append", "replace"])
def test_a_file_changed_after_its_record_stops_the_next_line_and_the_release(box, monkeypatch, change):
    """Before every later line and before the final release the paths are authenticated against the recorded
    snapshots (bytes AND device / inode): a change after the record stops the pre-flight; nothing is released."""
    real = S.Supervisor.finish
    gv = box.lines[1]

    def finish(self):
        path = box.path(gv["outputs"][0])
        raw = open(path, "rb").read()
        if change == "append":
            open(path, "ab").write(b" ")
        else:
            open(path + ".new", "wb").write(raw)
            os.replace(path + ".new", path)
        return real(self)
    monkeypatch.setattr(S.Supervisor, "finish", finish)
    assert box.sup().run() == 3
    st = box.state()
    assert st["phase"] == "stopped" and "before the release" in st["stop"]["reason"]
    assert gv["outputs"][0] in st["stop"]["reason"] and os.path.exists(box.res) and st["released"] is None
