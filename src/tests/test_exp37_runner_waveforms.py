"""exp_37 fix round (Codex BLOCKING 1): the runner / CLI waveform contract.

Timing passes never dump; every accuracy-only exp_37 run dumps inline (``--dump-waveforms <root>/<run stem>``);
``--replay-of <timing run>`` is an untimed replay whose identity must equal the timing run's except the timing flags,
that uses the timing run's per-query arrays as the bitwise reference (and a c2f run's own batch plan) and dumps; dump
runs are never resumed; the identity records ``waveform_dump`` / ``replay_of``; the dump manifest covers every query.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.localization import exp37_artifacts, exp37_rotation, exp37_waveforms, runner
from test_exp37_runner_wiring import E9, REPO, SELECTION_128, _identity, _run, frozen, stubbed  # noqa: F401


@pytest.fixture
def dumps(tmp_path, monkeypatch, stubbed):
    root = tmp_path / "waves"
    monkeypatch.setattr(exp37_waveforms, "DUMP_ROOT", root)
    calls = []

    def artifact_stub(**kw):
        calls.append(kw)
        n = int(kw["selected"]["candidate_count"])
        entry = kw["dump"].write(int(kw["record"]["index"]), np.zeros((n, 1, 10240), np.float32), np.arange(n))
        result = {"schema_version": 1, "run_manifest_sha256": kw["run_sha256"], "query_index": kw["selected"]["index"],
                  "query_id": kw["selected"]["query_id"], "candidate_count": n, "elapsed_seconds": 0.0,
                  "waveforms": entry}
        return result, np.zeros((n, 3), np.float32), np.zeros((n, 1), np.float32)
    monkeypatch.setattr(exp37_artifacts, "execute_query_artifacts", artifact_stub)
    monkeypatch.setattr(exp37_rotation, "execute_query_rotated", artifact_stub)
    return root, calls


def _dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def test_a_timing_pass_never_dumps(frozen, tmp_path, dumps):
    root, _ = dumps
    with pytest.raises(ValueError, match="named after|never dump"):
        _run(frozen, tmp_path, query_index=4467, dump_waveforms=root / "elsewhere")      # the stem is the run's
    with pytest.raises(ValueError, match="never dump"):
        _run(frozen, tmp_path, query_index=4467, measure_core_forward=True, synchronize_timing=True,
             dump_waveforms=root / "run")
    assert not (tmp_path / "run").exists() and not (root / "run").exists()


def test_an_accuracy_only_exp37_run_must_dump(frozen, tmp_path, dumps):
    for flags in ({"require_attention_backend": "sdpa"}, {"context_count": 1}, {"mesh_root": Path("m")}):
        with pytest.raises(ValueError, match="must dump"):
            _run(frozen, tmp_path, query_index=4467, **flags)
    assert not (tmp_path / "run").exists()
    _run(frozen, tmp_path, query_index=4467)                    # the bundle's own default run is unchanged


def test_an_inline_dump_run_records_and_covers_its_waveforms(frozen, tmp_path, dumps):
    root, calls = dumps
    _run(frozen, tmp_path, query_index=4467, dump_waveforms=root / "run")
    identity = _identity(tmp_path)
    assert identity["waveform_dump"] == {"mode": "inline", "dir": str(root / "run")} and "replay_of" not in identity
    assert calls[0]["reference"] is None and calls[0]["plan"] is None
    manifest = json.loads((root / "run" / "dump_manifest.json").read_text())
    assert manifest["mode"] == "inline" and manifest["coverage"]["query_indices"] == [4467]
    with pytest.raises(RuntimeError, match="exists"):                            # write-once, never resumed
        _run(frozen, tmp_path, query_index=4467, dump_waveforms=root / "run")


def test_a_replay_reproduces_a_timing_run_and_binds_to_it(frozen, tmp_path, dumps, stubbed):
    root, calls = dumps
    timing = _dir(tmp_path, "timing")
    _run(frozen, timing, query_index=4467, measure_core_forward=True, synchronize_timing=True, warmup_query_count=1)
    original = json.loads((timing / "run" / "run_manifest.json").read_text())
    replay = _dir(tmp_path, "replay")
    _run(frozen, replay, query_index=4467, dump_waveforms=root / "run", replay_of=timing / "run")
    identity = json.loads((replay / "run" / "run_manifest.json").read_text())["identity"]
    assert identity["replay_of"] == {"run_manifest_sha256": original["sha256"], "run_dir": str(timing / "run")}
    assert identity["waveform_dump"]["mode"] == "replay" and identity["measure_core_forward"] is False
    reference = calls[-1]["reference"](4467)
    assert reference.shape == (27, 1) and reference.dtype == np.float32
    assert json.loads((root / "run" / "dump_manifest.json").read_text())["replay_of"]["run_manifest_sha256"] == original["sha256"]


def test_a_replay_of_another_configuration_or_a_timed_replay_is_refused(frozen, tmp_path, dumps):
    root, _ = dumps
    timing = _dir(tmp_path, "timing")
    _run(frozen, timing, query_index=4467, measure_core_forward=True, synchronize_timing=True)
    with pytest.raises(RuntimeError, match="replay"):
        _run(frozen, _dir(tmp_path, "r1"), query_index=4467, sample_seed=43, dump_waveforms=root / "run",
             replay_of=timing / "run")
    with pytest.raises(ValueError, match="untimed"):
        _run(frozen, _dir(tmp_path, "r2"), query_index=4467, measure_core_forward=True, dump_waveforms=root / "run",
             replay_of=timing / "run")
    with pytest.raises(ValueError, match="dump"):
        _run(frozen, _dir(tmp_path, "r3"), query_index=4467, replay_of=timing / "run")
    inline = _dir(tmp_path, "inline")
    _run(frozen, inline, query_index=4467, dump_waveforms=root / "run")
    with pytest.raises(RuntimeError, match="not a timing pass"):              # replays are of timing passes only
        _run(frozen, _dir(tmp_path, "r4"), query_index=4467, dump_waveforms=root / "other", replay_of=inline / "run")
    assert not any((tmp_path / name / "run").exists() for name in ("r1", "r2", "r3", "r4"))


def test_a_rotated_run_dumps_inline(frozen, tmp_path, dumps):
    root, calls = dumps
    with pytest.raises(ValueError, match="dump"):
        _run(frozen, tmp_path, query_index=4467, cond_method="fa_invariant", rotate_deg=90.0)
    _run(frozen, tmp_path, query_index=4467, cond_method="fa_invariant", rotate_deg=90.0, dump_waveforms=root / "run")
    assert calls[0]["rotate_deg"] == 90.0 and calls[0]["dump"].dir == root / "run"


def test_the_cli_passes_the_waveform_flags(monkeypatch):
    sys.path.insert(0, str(REPO))
    import localize_FLAC

    seen = {}
    monkeypatch.setattr(localize_FLAC, "run_localization", lambda **kw: seen.update(kw) or {"ok": True})
    base = ["localize_FLAC.py", "--model-config", "c.json", "--ckpt-path", "k.ckpt", "--agree-ckpt", "a.pt",
            "--context-manifest", str(E9 / "context_manifest_exp01_seed42.json"),
            "--geometry-audit", str(E9 / "geometry_audit.json"), "--pilot-manifest", str(SELECTION_128),
            "--dataset-root", "AcousticRooms", "--output-dir", str(REPO / "never_created_by_this_test"),
            "--cond-method", "vanilla"]
    monkeypatch.setattr(sys, "argv", base + ["--dump-waveforms", "/w/x", "--replay-of", "/r/x"])
    localize_FLAC.main()
    assert seen["dump_waveforms"] == Path("/w/x") and seen["replay_of"] == Path("/r/x")
