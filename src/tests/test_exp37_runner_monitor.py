"""exp_37 fix round (Codex BLOCKING 3): every exp_37 timing pass (SDPA required + timing flags) runs the co-tenancy
monitor bound to its run manifest and card -- started after the start host record, ``gpu_resident`` after the model
loads, ``timed_start`` before the warm-up, ``timed_end`` after the last query, then stopped -- and is never resumed;
untimed or bundle runs start none. The localize_FLAC.py tree's only Python child is that monitor (isolated, the
stdlib-only tools/exp37_cotenancy_monitor.py). CPU; the monitor is a recording fake here.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import ast
import json
import os

import pytest

from src.localization import exp37_cotenancy, exp37_waveforms, runner
from pathlib import Path

from test_exp37_no_flash_launch_integration import REPO as _REPO, _import_closure, _spawn_violations, \
    _spawner_references
from test_exp37_runner_wiring import _run, frozen, stubbed  # noqa: F401

REPO = Path(_REPO)
TIMED = {"measure_core_forward": True, "synchronize_timing": True}


@pytest.fixture
def events(monkeypatch, stubbed):
    log = []

    class FakeMonitor:
        def __init__(self, directory, *, binding_sha256, card_uuid, **kw):
            log.append(("monitor", str(directory), binding_sha256, card_uuid))

        def start(self):
            log.append(("start",))
            return self

        def marker(self, name):
            log.append(("marker", name))

        def stop(self):
            log.append(("stop",))
            return 0
    monkeypatch.setattr(exp37_cotenancy, "Monitor", FakeMonitor)
    monkeypatch.setattr(runner, "producer_identity", lambda device: {"pid": os.getpid(), "pid_start_ticks": 7,
                                                                      "hostname": "h", "card_uuid": "6ac2604a-1111"})
    monkeypatch.setattr(runner, "attention_identity", lambda root, require: {"attention_backend": "sdpa"})
    monkeypatch.setattr(runner, "no_flash_launch_handoff", lambda root: {"pid": os.getpid()})
    monkeypatch.setattr(runner, "weights_footprint", lambda device: {"memory_allocated_bytes": 1})
    load, agree, execute = runner.load_flac_module, runner.load_agree_retrieval, runner.execute_query
    monkeypatch.setattr(runner, "load_flac_module", lambda *a: log.append(("load_flac",)) or load(*a))
    monkeypatch.setattr(runner, "load_agree_retrieval", lambda *a: log.append(("load_agree",)) or agree(*a))
    monkeypatch.setattr(runner, "execute_query", lambda **kw: log.append(("query", kw["selected"]["index"])) or execute(**kw))
    return log


def test_an_exp37_timing_pass_is_monitored_over_its_whole_timed_interval(frozen, tmp_path, events):
    _run(frozen, tmp_path, query_limit=2, warmup_query_count=1, require_attention_backend="sdpa", **TIMED)
    sha = json.loads((tmp_path / "run" / "run_manifest.json").read_text())["sha256"]
    assert events[0] == ("monitor", str(tmp_path / "run" / "cotenancy"), sha, "6ac2604a-1111")
    producer = json.loads((tmp_path / "run" / "producer.json").read_text())
    assert (producer["run_manifest_sha256"], producer["pid"], producer["pid_start_ticks"], producer["card_uuid"]) == (
        sha, os.getpid(), 7, "6ac2604a-1111")                     # bound before the monitor starts
    kinds = [e[0] if e[0] != "marker" else e[1] for e in events]
    assert kinds == ["monitor", "start", "load_flac", "load_agree", "gpu_resident", "timed_start",
                     "query", "query", "query", "timed_end", "stop"]            # warm-up + 2 queries inside


@pytest.mark.parametrize("flags", [{}, TIMED])
def test_untimed_or_bundle_runs_start_no_monitor(frozen, tmp_path, events, monkeypatch, flags):
    if not flags:                                                # an untimed exp_37 run dumps instead
        monkeypatch.setattr(exp37_waveforms, "DUMP_ROOT", tmp_path / "waves")
        from src.localization import exp37_artifacts

        def stub(**kw):
            n = int(kw["selected"]["candidate_count"])
            import numpy as np
            kw["dump"].write(int(kw["record"]["index"]), np.zeros((n, 1, 10240), np.float32), np.arange(n))
            return {"schema_version": 1, "run_manifest_sha256": kw["run_sha256"], "query_index": kw["selected"]["index"],
                    "query_id": kw["selected"]["query_id"], "candidate_count": n, "elapsed_seconds": 0.0}, \
                np.zeros((n, 3), np.float32), np.zeros((n, 1), np.float32)
        monkeypatch.setattr(exp37_artifacts, "execute_query_artifacts", stub)
        _run(frozen, tmp_path, query_index=4467, require_attention_backend="sdpa", dump_waveforms=tmp_path / "waves" / "run")
    else:                                                        # the bundle's own timing path (no SDPA requirement)
        _run(frozen, tmp_path, query_index=4467, **flags)
    assert not any(e[0] in ("monitor", "start", "marker", "stop") for e in events)


def test_a_monitored_pass_is_never_resumed(frozen, tmp_path, events):
    _run(frozen, tmp_path, query_index=4467, require_attention_backend="sdpa", **TIMED)
    before = sorted(p.name for p in (tmp_path / "run").iterdir())
    with pytest.raises(RuntimeError, match="never resumed"):
        _run(frozen, tmp_path, query_index=4467, require_attention_backend="sdpa", **TIMED)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == before


def test_the_only_python_child_of_the_localize_flac_tree_is_the_isolated_monitor():
    closure = _import_closure([str(REPO / "localize_FLAC.py")])
    cotenancy = str(REPO / "src" / "localization" / "exp37_cotenancy.py")
    assert cotenancy in closure
    refs = {(os.path.relpath(p, REPO), ref) for p in closure for ref in _spawner_references(p)}
    assert refs == {("src/localization/run_provenance.py", "subprocess.run"),
                    ("src/localization/exp37_cotenancy.py", "subprocess.Popen")}
    violations = [v for p in sorted(closure) for v in _spawn_violations(p)]
    assert len(violations) == 1 and violations[0].startswith(f"{cotenancy}:")
    calls = [n for n in ast.walk(ast.parse(open(cotenancy).read())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"]
    assert len(calls) == 1
    head = calls[0].args[0].elts[:4]
    assert ast.unparse(head[0]) == "sys.executable" and [h.value for h in head[1:3]] == ["-I", "-B"]
    assert ast.unparse(head[3]) == "str(MONITOR_SCRIPT)"
    assert exp37_cotenancy.MONITOR_SCRIPT == REPO / "tools" / "exp37_cotenancy_monitor.py"
