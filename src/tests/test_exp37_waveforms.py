"""exp_37 fix round (Codex BLOCKING 1, announcement 08): the write-once predicted-waveform artifact contract.

Per query one ``.npy`` ``[M, 1, 10240]`` float32 -- the exact ``prepare_generated_audio`` output that was scored --
with its candidate / full-grid index mapping; a hashed dump manifest bound to the run manifest, checkpoint, seed,
backend and the full query coverage; the dump directory is ``<root>/<run stem>`` and must not exist.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import hashlib
import json

import numpy as np
import pytest

from src.localization import exp37_waveforms as wav

IDENTITY = {"checkpoint_sha256": "c" * 64, "sample_seed": 43, "attention_backend": "sdpa",
            "conditioning_method": "fa_invariant", "frame_average_angles": [0.0, 180.0]}


def _waves(m, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (m, 1, 10240)).astype(np.float32)


def test_a_dump_is_write_once_under_its_run_stem(tmp_path):
    root = tmp_path / "exp37_pred_waveforms"
    dump = wav.WaveformDump(root / "b2_0p50_P1_s43", run_stem="b2_0p50_P1_s43", root=root)
    with pytest.raises(RuntimeError, match="exists"):
        wav.WaveformDump(root / "b2_0p50_P1_s43", run_stem="b2_0p50_P1_s43", root=root)
    with pytest.raises(ValueError, match="run stem"):
        wav.WaveformDump(root / "other", run_stem="b2_0p50_P1_s43", root=root)
    with pytest.raises(ValueError, match="root"):
        wav.WaveformDump(tmp_path / "elsewhere" / "x", run_stem="x", root=root)
    entry = dump.write(7, _waves(3), np.array([0, 4, 9]))
    assert entry["path"] == "query_00007.npy" and entry["shape"] == [3, 1, 10240] and entry["dtype"] == "float32"
    assert entry["sha256"] == hashlib.sha256((dump.dir / "query_00007.npy").read_bytes()).hexdigest()
    assert entry["full_indices_sha256"] == hashlib.sha256(np.array([0, 4, 9], dtype="<u4").tobytes()).hexdigest()
    assert np.array_equal(np.load(dump.dir / "query_00007.npy"), _waves(3))
    with pytest.raises(RuntimeError, match="written"):
        dump.write(7, _waves(3), np.arange(3))


@pytest.mark.parametrize("bad", [np.zeros((2, 1, 10240), dtype=np.float64), np.zeros((2, 10240), dtype=np.float32),
                                 np.zeros((2, 1, 999), dtype=np.float32)], ids=["float64", "2d", "length"])
def test_only_exact_scored_tensors_are_accepted(tmp_path, bad):
    dump = wav.WaveformDump(tmp_path / "w" / "r", run_stem="r", root=tmp_path / "w")
    with pytest.raises(ValueError, match="float32"):
        dump.write(1, bad, np.arange(2))


def test_the_manifest_binds_run_checkpoint_seed_backend_and_coverage(tmp_path):
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(2), np.arange(2))
    with pytest.raises(RuntimeError, match="coverage"):
        dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1, 2])
    dump.write(2, _waves(4, 1), np.arange(4))
    path = dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="inline", query_indices=[1, 2])
    manifest = json.loads(path.read_text())
    assert manifest["run_manifest_sha256"] == "m" * 64 and manifest["sample_seed"] == 43
    assert manifest["checkpoint_sha256"] == "c" * 64 and manifest["attention_backend"] == "sdpa"
    assert manifest["mode"] == "inline" and sorted(manifest["coverage"]["files"]) == ["1", "2"]
    loaded = wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={1: 2, 2: 4})
    assert loaded["sha256"] == manifest["sha256"]


def test_a_dump_that_is_not_the_run_s_or_was_changed_is_refused(tmp_path):
    root = tmp_path / "w"
    dump = wav.WaveformDump(root / "r", run_stem="r", root=root)
    dump.write(1, _waves(2), np.arange(2))
    dump.finish(run_manifest_sha256="m" * 64, identity=IDENTITY, mode="replay", query_indices=[1],
                replay_of={"run_manifest_sha256": "t" * 64, "run_dir": "/runs/t"})
    with pytest.raises(RuntimeError, match="run manifest"):
        wav.load_dump_manifest(root / "r", run_manifest_sha256="x" * 64, query_counts={1: 2})
    with pytest.raises(RuntimeError, match="coverage"):
        wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={1: 2, 2: 3})
    with pytest.raises(RuntimeError, match="shape"):
        wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={1: 3})
    np.save(root / "r" / "query_00001.npy", _waves(2, 9))
    with pytest.raises(RuntimeError, match="sha256"):
        wav.load_dump_manifest(root / "r", run_manifest_sha256="m" * 64, query_counts={1: 2})
