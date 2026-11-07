"""exp_37 final fix round (Codex code review 3, B-01): publication is atomic no-replace ONLY, directory durability is
required, and a failure never publishes or replaces a final path.

- ``exp37_waveforms.rename_noreplace`` is ``renameat2(RENAME_NOREPLACE)`` and nothing else: no libc symbol, or a
  filesystem that refuses the flag (ENOSYS / EINVAL / ENOTSUP), raises ``DurabilityError`` -- never a check-then-rename
  fallback -- and the dump keeps its staging for quarantine / retry; a target that appears at any moment before the
  rename (an empty directory included, which a plain rename would replace) is never replaced.
- ``exp37_waveforms.fsync_dir``: a directory fsync that fails or is unsupported (EINVAL / ENOTSUP / EBADF / EIO) is
  fatal: a dump never starts or publishes (a failure after the rename unpublishes it back to its staging name), and
  ``exp37_publish.publish`` leaves no final output path and no staging.
- ``tools/exp37_quarantine.py`` moves with the same no-replace rename.
CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), final fix round: fixed after review 3, not
re-reviewed (cap).
"""
import ctypes
import errno
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_publish as pub
from src.localization import exp37_waveforms as wav

REPO = Path(__file__).resolve().parents[2]
IDENTITY = {"checkpoint_sha256": "c" * 64, "sample_seed": 43, "attention_backend": "sdpa"}


def _dump(root):
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, np.zeros((2, 1, 10240), dtype=np.float32), np.array([0, 1]))
    return dump


def _finish(dump):
    return dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1])


class _NoRenameat2:
    def __getattr__(self, name):
        raise AttributeError(name)


class _Refusing:
    def __init__(self, code):
        self.code = code

    def renameat2(self, *args):
        ctypes.set_errno(self.code)
        return -1


def _forbid_plain_rename(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("os.rename / os.replace must never publish (no check-then-rename fallback)")
    monkeypatch.setattr(wav.os, "rename", boom)
    monkeypatch.setattr(wav.os, "replace", boom)


def _failing_dir_fsync(monkeypatch, code, when=lambda path: True):
    real = os.fsync

    def fsync(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if stat.S_ISDIR(os.fstat(fd).st_mode) and when(path):
            raise OSError(code, os.strerror(code))
        return real(fd)
    monkeypatch.setattr(os, "fsync", fsync)


def test_a_libc_without_renameat2_fails_closed_and_keeps_the_staging(tmp_path, monkeypatch):
    root = tmp_path / "w"
    dump = _dump(root)
    monkeypatch.setattr(wav.ctypes, "CDLL", lambda *a, **k: _NoRenameat2())
    _forbid_plain_rename(monkeypatch)
    with pytest.raises(wav.DurabilityError, match="RENAME_NOREPLACE"):
        _finish(dump)
    assert dump.staging.is_dir() and not os.path.lexists(root / "r") and not dump.finished


@pytest.mark.parametrize("code", [errno.ENOSYS, errno.EINVAL, errno.ENOTSUP])
def test_a_filesystem_without_the_flag_fails_closed(tmp_path, monkeypatch, code):
    root = tmp_path / "w"
    dump = _dump(root)
    monkeypatch.setattr(wav.ctypes, "CDLL", lambda *a, **k: _Refusing(code))
    _forbid_plain_rename(monkeypatch)
    with pytest.raises(wav.DurabilityError):
        _finish(dump)
    assert dump.staging.is_dir() and not os.path.lexists(root / "r")


def test_a_target_injected_just_before_the_rename_is_never_replaced(tmp_path, monkeypatch):
    root = tmp_path / "w"
    dump = _dump(root)
    real = wav.rename_noreplace

    def racing(source, target):
        Path(target).mkdir()                       # another writer takes the name: an EMPTY directory, which a plain
        monkeypatch.setattr(wav.os.path, "lexists", lambda p: False)   # rename would replace; an existence check
        return real(source, target)                                     # would be fooled -- none is used
    monkeypatch.setattr(wav, "rename_noreplace", racing)
    _forbid_plain_rename(monkeypatch)
    with pytest.raises(RuntimeError, match="exists"):
        _finish(dump)
    monkeypatch.undo()
    assert (root / "r").is_dir() and not any((root / "r").iterdir()) and dump.staging.is_dir()


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.EIO])
def test_fsync_dir_never_reports_a_failure_as_a_value(tmp_path, monkeypatch, code):
    assert wav.fsync_dir(tmp_path) == "ok"
    _failing_dir_fsync(monkeypatch, code)
    with pytest.raises(wav.DurabilityError, match="fsync"):
        wav.fsync_dir(tmp_path)


def test_an_unsupported_directory_fsync_refuses_the_dump_before_anything_publishes(tmp_path, monkeypatch):
    root = tmp_path / "w"
    _failing_dir_fsync(monkeypatch, errno.EINVAL)
    with pytest.raises(wav.DurabilityError):
        wav.WaveformDump(root / "r", run_stem="r", root=root)
    assert not os.path.lexists(root / "r")


def test_a_directory_fsync_failure_after_the_rename_unpublishes_the_dump(tmp_path, monkeypatch):
    root = tmp_path / "w"
    dump = _dump(root)
    _failing_dir_fsync(monkeypatch, errno.EIO, when=lambda path: path == str(root) and (root / "r").exists())
    with pytest.raises(wav.DurabilityError, match="unpublished"):
        _finish(dump)
    monkeypatch.undo()
    assert not os.path.lexists(root / "r") and not dump.finished
    assert (dump.staging / "dump_manifest.json").is_file()              # kept whole for the quarantine tool


@pytest.mark.parametrize("where", ["staging", "final"])
def test_publish_leaves_nothing_when_a_directory_fsync_fails(tmp_path, monkeypatch, where):
    out = tmp_path / "out"
    out.mkdir()
    finals = [out / "a.md", out / "b.json"]

    def render(staging):
        for final in finals:
            (staging / final.name).write_text("x\n")
    when = (lambda path: ".publish-" in path) if where == "staging" else (lambda path: path == str(out))
    _failing_dir_fsync(monkeypatch, errno.EINVAL, when=when)
    with pytest.raises(wav.DurabilityError):
        pub.publish(finals, render)
    monkeypatch.undo()
    assert sorted(p.name for p in out.iterdir()) == []                   # no final output path, no staging


def test_the_quarantine_tool_moves_with_the_no_replace_rename():
    source = (REPO / "tools" / "exp37_quarantine.py").read_text()
    assert "rename_noreplace" in source and "os.rename(" not in source
