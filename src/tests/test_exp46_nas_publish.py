"""exp_46 fix round 2 -- Codex code r2 finding 8: the staged NAS publication is atomic and no-clobber.

The staging directory <dest root>/<run>.incoming is created EXCLUSIVELY before rsync (an existing entry or a link
refuses), and the admitted staging directory reaches <dest root>/<run> through a no-replace rename: a final path
created meanwhile refuses -- never replaced, never nested (an ordinary ``mv`` nests). Temporary directories only:
the NAS is never touched.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import ctypes
import errno
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402
import exp46_nas_publish as nas  # noqa: E402

RUN = "exp46_C4Ff100_l40_2x32"


def _listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), str(d)) for a, ds, fs in os.walk(str(d)) for f in ds + fs)


def _staged(root):
    """stage, then what rsync + the admission leave behind: a sealed run's files in the staging directory."""
    incoming = nas.stage(str(root), RUN)
    for name in (reg.TRAINING_RECEIPT, reg.COMMIT_MARKER, "CHECKPOINTS.sha256"):
        open(os.path.join(incoming, name), "w").write(name)
    return incoming


def test_stage_then_publish_moves_the_admitted_run_into_place(tmp_path):
    root = tmp_path / "nas"
    root.mkdir()
    incoming = _staged(root)
    assert incoming == str(root / f"{RUN}.incoming") and os.path.isdir(incoming) and not os.path.islink(incoming)
    final, how = nas.publish(str(root), RUN)
    assert final == str(root / RUN) and how in ("renameat2", "rename")
    assert _listing(root) == sorted([RUN] + [f"{RUN}/{n}" for n in (reg.TRAINING_RECEIPT, reg.COMMIT_MARKER,
                                                                      "CHECKPOINTS.sha256")])


@pytest.mark.parametrize("entry", ["staging dir", "staging file", "staging link", "dangling staging link", "final dir",
                                   "final link"])
def test_stage_refuses_any_existing_staging_or_published_entry_before_any_copy(tmp_path, entry):
    root = tmp_path / "nas"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = root / (RUN if entry.startswith("final") else f"{RUN}.incoming")
    if entry.endswith("dir"):
        target.mkdir()
    elif entry == "staging file":
        target.write_text("x")
    else:
        target.symlink_to(elsewhere if entry != "dangling staging link" else tmp_path / "absent")
    before = _listing(root)
    with pytest.raises(reg.GateError, match="exists"):
        nas.stage(str(root), RUN)
    assert _listing(root) == before and _listing(elsewhere) == []
    assert nas.main(["stage", "--dest-root", str(root), "--run", RUN]) == 2


def test_stage_never_creates_or_follows_its_root(tmp_path, monkeypatch):
    with pytest.raises(reg.GateError, match="not a real directory"):
        nas.stage(str(tmp_path / "unmounted"), RUN)
    assert not os.path.lexists(tmp_path / "unmounted")
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    with pytest.raises(reg.GateError, match="not a real directory"):
        nas.stage(str(tmp_path / "linked"), RUN)
    prot = tmp_path / "protected_exp45_nas"
    prot.mkdir()
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", (str(prot),))
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        nas.stage(str(prot), RUN)
    assert _listing(tmp_path / "real") == [] and _listing(prot) == []


@pytest.mark.parametrize("noreplace", [True, False], ids=["renameat2", "raw syscall"])
@pytest.mark.parametrize("final", ["a published run", "an empty directory", "a file", "a link"])
def test_publish_refuses_a_final_path_created_meanwhile_never_replacing_or_nesting(tmp_path, monkeypatch, final,
                                                                                noreplace):
    """The final path appears AFTER every check, just before the move (another publisher): the no-replace rename
    refuses -- renameat2(RENAME_NOREPLACE), or the fallback's fresh check -- and nothing is replaced or nested; the
    same final path present before publish is refused up front."""
    root = tmp_path / "nas"
    root.mkdir()
    incoming = _staged(root)
    dst = root / RUN

    def make_final():
        if final == "a published run":
            dst.mkdir()
            (dst / "theirs").write_text("theirs")
        elif final == "an empty directory":
            dst.mkdir()
        elif final == "a file":
            dst.write_text("theirs")
        else:
            dst.symlink_to(tmp_path / "absent")
    if not noreplace:                                   # a libc without the renameat2 wrapper: the raw syscall
        real_libc = ctypes.CDLL(None, use_errno=True)
        monkeypatch.setattr(nas.ctypes, "CDLL", lambda *a, **k: _OnlySyscall(real_libc))
    real, moves = nas._rename_noreplace, []

    def meanwhile(src, dst_):
        make_final()                                    # created after every check
        moves.append(real(src, dst_))
    monkeypatch.setattr(nas, "_rename_noreplace", meanwhile)
    with pytest.raises(reg.GateError, match="never replaced or nested"):
        nas.publish(str(root), RUN)
    monkeypatch.setattr(nas, "_rename_noreplace", real)
    after = _listing(root)
    assert moves == [] and os.path.isdir(incoming) and set(os.listdir(incoming)) >= {reg.COMMIT_MARKER}
    assert not os.path.lexists(dst / f"{RUN}.incoming") if dst.is_dir() and not dst.is_symlink() else True
    assert (dst / "theirs").read_text() == "theirs" if final == "a published run" else True
    with pytest.raises(reg.GateError, match="never replaced or nested"):      # present before publish: refused
        nas.publish(str(root), RUN)
    assert _listing(root) == after


@pytest.mark.parametrize("case", ["a staging link", "no staging", "unsealed"])
def test_publish_refuses_anything_but_a_real_sealed_staging_directory(tmp_path, case):
    root = tmp_path / "nas"
    root.mkdir()
    if case == "a staging link":
        (tmp_path / "elsewhere").mkdir()
        (root / f"{RUN}.incoming").symlink_to(tmp_path / "elsewhere")
    elif case == "unsealed":
        nas.stage(str(root), RUN)
    before = _listing(root)
    with pytest.raises(reg.GateError, match="staging directory|sealed"):
        nas.publish(str(root), RUN)
    assert _listing(root) == before
    assert nas.main(["publish", "--dest-root", str(root), "--run", RUN]) == 2


# ------------------------------------------------------------------ fix round 3, Codex code r3 finding 4: no fallback
class _OnlySyscall:
    """A libc WITHOUT the renameat2 wrapper (glibc < 2.28): only the real syscall(), so the raw renameat2 runs."""

    def __init__(self, real):
        self.syscall = real.syscall


class _Unsupported:
    """A box / filesystem without RENAME_NOREPLACE: renameat2 fails with ``err``."""

    def __init__(self, err):
        self.err = err

    def renameat2(self, *args):
        ctypes.set_errno(self.err)
        return -1


def test_the_raw_syscall_publishes_when_libc_has_no_wrapper(tmp_path, monkeypatch):
    """No wrapper (glibc < 2.28): the RAW renameat2 syscall moves the admitted staging dir -- the real kernel call;
    the race test above runs the same path against a final created meanwhile (EEXIST, nothing replaced)."""
    real = ctypes.CDLL(None, use_errno=True)
    monkeypatch.setattr(nas.ctypes, "CDLL", lambda *a, **k: _OnlySyscall(real))
    root = tmp_path / "nas"
    root.mkdir()
    _staged(root)
    assert nas.publish(str(root), RUN) == (str(root / RUN), "renameat2")
    assert sorted(os.listdir(root / RUN)) == sorted([reg.TRAINING_RECEIPT, reg.COMMIT_MARKER, "CHECKPOINTS.sha256"])


@pytest.mark.parametrize("err", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def test_an_unsupported_no_replace_rename_refuses_and_a_competitor_is_never_replaced(tmp_path, monkeypatch, err):
    """Codex code r3 finding 4: the old fallback (a lexists check, then os.rename) let a competitor's empty directory,
    created between the check and the rename, be replaced. renameat2 now reports ``err`` (no RENAME_NOREPLACE here) and
    a competitor creates the final directory immediately before any rename could run: publish REFUSES, no rename
    happens at all, the competitor's directory is untouched (empty) and the staging dir is kept."""
    root = tmp_path / "nas"
    root.mkdir()
    incoming = _staged(root)
    dst = root / RUN
    renamed, real_rename = [], os.rename

    def competitor_then_rename(src, dst_):
        renamed.append((src, dst_))
        os.mkdir(dst_)                    # another publisher's empty dir, immediately before the fallback rename
        return real_rename(src, dst_)
    monkeypatch.setattr(nas.os, "rename", competitor_then_rename)
    monkeypatch.setattr(nas.ctypes, "CDLL", lambda *a, **k: _Unsupported(err))
    with pytest.raises(reg.GateError, match="RENAME_NOREPLACE unsupported"):
        nas.publish(str(root), RUN)
    assert renamed == [] and not os.path.lexists(dst)                 # no fallback rename ever runs
    assert sorted(os.listdir(incoming)) == sorted([reg.TRAINING_RECEIPT, reg.COMMIT_MARKER, "CHECKPOINTS.sha256"])
    monkeypatch.setattr(nas.ctypes, "CDLL", lambda *a, **k: object())        # no wrapper and no syscall(): ENOSYS
    with pytest.raises(reg.GateError, match="RENAME_NOREPLACE unsupported"):
        nas.publish(str(root), RUN)
    assert renamed == [] and not os.path.lexists(dst)


# ------------------------------------------------------------------ fix round 3, Codex code r3 finding 7: provisioning
@pytest.fixture
def nas_mount(tmp_path, monkeypatch):
    """A stand-in NAS (<tmp>/diskstation/FLAC with its checkpoints folder) reported as the registered cifs; the two
    exp_46 roots (the A6000 save root = L7's dest root, and the evaluator's dump root) placed below it."""
    import exp46_eval as ev
    flac = tmp_path / "diskstation" / "FLAC"
    (flac / "checkpoints").mkdir(parents=True)
    ck, dump = flac / "checkpoints" / "exp46_c4flip_data_curve", flac / "exp46_pred_waveforms"
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(ck))
    monkeypatch.setattr(ev, "DUMP_ROOT", str(dump))
    monkeypatch.setattr(reg, "mount_fstype", lambda path: ("cifs", str(tmp_path / "diskstation")))
    return flac, ck, dump


def test_provision_creates_exactly_the_two_exp46_namespaces_once(nas_mount):
    """Both NEW exp_46 namespaces are created (real directories, nothing else on the NAS), a second run creates nothing
    (idempotent), and stage then finds its dest root (the first campaign no longer stops at publication / dumps)."""
    flac, ck, dump = nas_mount
    out = nas.provision()
    assert {v["path"]: v["created"] for v in out.values()} == {str(ck): True, str(dump): True}
    assert os.path.isdir(ck) and os.path.isdir(dump) and not os.path.islink(ck) and not os.path.islink(dump)
    assert _listing(flac) == sorted(["checkpoints", "checkpoints/exp46_c4flip_data_curve", "exp46_pred_waveforms"])
    assert [v["created"] for v in nas.provision().values()] == [False, False]
    assert nas.main(["provision"]) == 0
    assert nas.stage(str(ck), RUN) == str(ck / f"{RUN}.incoming")


def test_provision_refuses_a_missing_or_foreign_mount_before_creating_anything(nas_mount, tmp_path, monkeypatch):
    """A missing NAS parent (not mounted) or a parent on another filesystem (a local disk) refuses -- and neither root
    is created, not even the one whose parent exists."""
    flac, ck, dump = nas_mount
    os.rmdir(flac / "checkpoints")
    with pytest.raises(reg.GateError, match="not mounted"):
        nas.provision()
    assert _listing(tmp_path / "diskstation") == ["FLAC"]
    assert nas.main(["provision"]) == 2
    (flac / "checkpoints").mkdir()
    monkeypatch.setattr(reg, "mount_fstype", lambda path: ("ext4", "/"))
    with pytest.raises(reg.GateError, match="not the registered NAS filesystem"):
        nas.provision()
    assert _listing(flac) == ["checkpoints"]


@pytest.mark.parametrize("entry", ["a link", "a dangling link", "a file", "inside a protected root"])
def test_provision_refuses_anything_but_a_real_directory_at_either_root(nas_mount, tmp_path, monkeypatch, entry):
    """A link (dangling or not) or a file at either root, or a root inside a protected one, refuses before EITHER root
    is created (the protected root and the link's target untouched)."""
    flac, ck, dump = nas_mount
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if entry == "a link":
        ck.symlink_to(elsewhere)
    elif entry == "a dangling link":
        dump.symlink_to(tmp_path / "absent")
    elif entry == "a file":
        dump.write_text("x")
    else:
        prot = flac / "checkpoints" / "exp38_data_efficiency"
        prot.mkdir()
        monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", (str(prot),))
        monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(prot / "exp46_c4flip_data_curve"))
    before = _listing(tmp_path)
    with pytest.raises(reg.GateError):
        nas.provision()
    assert _listing(tmp_path) == before
