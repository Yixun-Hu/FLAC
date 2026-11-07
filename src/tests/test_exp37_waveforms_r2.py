"""exp_37 fix round 2 (Codex code review 2, BLOCKING 2): the crash-safe, write-once waveform store.

- Writes go to a uniquely owned staging directory ``<root>/.staging-<stem>-<pid>-<token>`` (an owner record names
  the pid and its /proc start time); every file is created exclusively and fsync'd, directories are fsync'd; the
  manifest is written exclusively LAST; ``finish`` publishes by one no-replace rename into ``<root>/<stem>`` and
  refuses a second call; the final name never exists before publication.
- Each query keeps its actual row -> full-grid index vector (``<u4``, strictly increasing, a hashed companion file)
  next to the waveforms; ``load_dump_manifest`` re-verifies both files.
- A crash leaves only the staging directory: a new dump of the same stem is refused until
  ``tools/exp37_quarantine.py`` has moved the partial run and the dead staging into quarantine (never deleted; a
  hashed record); a live owner is never quarantined.
CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_waveforms as wav

REPO = Path(__file__).resolve().parents[2]
IDENTITY = {"checkpoint_sha256": "c" * 64, "sample_seed": 43, "attention_backend": "sdpa"}


def _waves(m, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (m, 1, 10240)).astype(np.float32)


def test_writes_are_staged_and_published_by_one_rename(tmp_path):
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    assert dump.staging.parent == root and dump.staging.name.startswith(f".staging-r-{os.getpid()}-")
    owner = json.loads((dump.staging / ".owner.json").read_text())
    assert owner["pid"] == os.getpid() and owner["run_stem"] == "r" and isinstance(owner["pid_start_ticks"], int)
    entry = dump.write(3, _waves(4), np.array([0, 2, 5, 9]))
    assert not (root / "r").exists() and (dump.staging / "query_00003.npy").is_file()
    assert np.array_equal(np.load(dump.path(entry["indices_path"])), np.array([0, 2, 5, 9], dtype="<u4"))
    assert entry["indices_sha256"] == hashlib.sha256(dump.path(entry["indices_path"]).read_bytes()).hexdigest()
    assert entry["full_indices_sha256"] == hashlib.sha256(np.array([0, 2, 5, 9], dtype="<u4").tobytes()).hexdigest()
    path = dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[3])
    assert path == root / "r" / "dump_manifest.json" and not dump.staging.exists()
    assert sorted(p.name for p in (root / "r").iterdir()) == [".owner.json", "dump_manifest.json", "query_00003.indices.npy",
                                                               "query_00003.npy"]
    with pytest.raises(RuntimeError, match="already"):
        dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[3])
    with pytest.raises(RuntimeError, match="published"):
        dump.write(4, _waves(1), np.array([0]))
    loaded = wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={3: 4})
    assert wav.dump_indices(root / "r", loaded["coverage"]["files"]["3"]).tolist() == [0, 2, 5, 9]


def test_every_file_and_directory_is_fsynced(tmp_path, monkeypatch):
    seen, real = [], os.fsync

    def recording(fd):
        seen.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real(fd)
    monkeypatch.setattr(os, "fsync", recording)
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(2), np.array([0, 1]))
    dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1])
    names = {Path(p).name for p in seen}
    assert {"query_00001.npy", "query_00001.indices.npy", "dump_manifest.json", ".owner.json"} <= names
    assert str(root) in seen and any(Path(p).name.startswith(".staging-r-") for p in seen)     # directories too


@pytest.mark.parametrize("indices, needle", [(np.array([0, 2, 2]), "increasing"), (np.array([3, 1, 2]), "increasing"),
                                             (np.array([0, 1]), "one full-grid index"), (np.array([-1, 0, 1]), "increasing")])
def test_the_index_vector_is_the_actual_increasing_mapping(tmp_path, indices, needle):
    dump = wav.WaveformDump(tmp_path / "w" / "r", run_stem="r", root=tmp_path / "w")
    with pytest.raises(ValueError, match=needle):
        dump.write(1, _waves(3), indices)


def test_a_tampered_index_vector_is_refused_on_load(tmp_path):
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(3), np.array([1, 4, 7]))
    dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1])
    target = root / "r" / "query_00001.indices.npy"
    target.chmod(0o644)
    np.save(target, np.array([1, 4, 8], dtype="<u4"))
    with pytest.raises(RuntimeError, match="index"):
        wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={1: 3})


def test_a_crash_leaves_only_staging_and_blocks_the_stem_until_quarantine(tmp_path):
    root, runs = tmp_path / "w", tmp_path / "runs"
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(2), np.array([0, 1]))
    owner = json.loads((dump.staging / ".owner.json").read_text())
    (dump.staging / ".owner.json").chmod(0o644)
    (dump.staging / ".owner.json").write_text(json.dumps({**owner, "pid": int(dead.stdout)}))   # its writer died
    assert not (root / "r").exists()
    with pytest.raises(RuntimeError, match="quarantine"):
        wav.WaveformDump(root / "r", run_stem="r", root=root)
    (runs / "r").mkdir(parents=True)
    (runs / "r" / "run_manifest.json").write_text("{}")
    tool = [sys.executable, str(REPO / "tools" / "exp37_quarantine.py"), "--run-dir", str(runs / "r"), "--dump-root",
            str(root)]
    done = subprocess.run(tool, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert not (runs / "r").exists() and not dump.staging.exists()
    records = list((runs / ".quarantine").glob("r.*.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["moved"]
    wav.WaveformDump(root / "r", run_stem="r", root=root)                      # the stem is usable again


def test_a_live_owner_is_never_quarantined(tmp_path):
    root, runs = tmp_path / "w", tmp_path / "runs"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)                # owned by THIS (live) process
    (runs / "r").mkdir(parents=True)
    done = subprocess.run([sys.executable, str(REPO / "tools" / "exp37_quarantine.py"), "--run-dir", str(runs / "r"),
                           "--dump-root", str(root)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 2 and "alive" in done.stderr and dump.staging.exists() and (runs / "r").exists()


def test_a_name_taken_before_publication_is_never_overwritten(tmp_path):
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(2), np.array([0, 1]))
    (root / "r").mkdir()                                                      # someone else took the final name
    with pytest.raises(RuntimeError, match="exists"):
        dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1])
    assert dump.staging.exists() and not any((root / "r").iterdir())
