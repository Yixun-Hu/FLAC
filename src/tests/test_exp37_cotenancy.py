"""exp_37 fix round (Codex BLOCKING 3): ``exp37_cotenancy`` -- the client that runs the monitor child next to a timed
pass (write-once directory, hash-chained markers ``gpu_resident`` / ``timed_start`` / ``timed_end``) and
``monitor_verdict``, which is valid only if both chains are intact and bound to the run, the samples cover the timed
interval with no gap above ``MAX_GAP_SECONDS``, every ``nvidia-smi`` query succeeded, the run is on its card and no
other process is on it for the whole GPU residency, exp_36 never appeared on the host, the foreign CPU over the timed
interval stays within the limit, and the monitor stopped on the run's stop file. CPU; fake ``nvidia-smi``.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

from src.localization import exp37_cotenancy as cot

REPO = Path(__file__).resolve().parents[2]
B = "b" * 64


def _script():
    spec = importlib.util.spec_from_file_location("exp37_cotenancy_monitor", REPO / "tools" / "exp37_cotenancy_monitor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def world(tmp_path, *, edit=None, marks=(2.0, 3.0, 18.0), stop="stop_file", binding=B, drop=()):
    """A monitor stream (samples every second over t = 0..20 s) and its markers; ``edit(t, sample)`` alters one."""
    folder = tmp_path / "cotenancy"
    folder.mkdir()
    out = _script().ChainWriter(folder / "monitor.jsonl", binding_sha256=binding)
    out.write({"kind": "start", "target_pid": 100, "card_uuid": "GPU-ABC", "interval_seconds": 1.0})
    for t in range(21):
        if t in drop:
            continue
        sample = {"kind": "sample", "t_monotonic": float(t), "target_alive": True, "exp36": [],
                  "gpu": {"returncode": 0, "lines": ["GPU-abc, 100, python, 900 MiB"]},
                  "cpu": {"interval_seconds": 1.0, "foreign_cores": 0.5}, "foreign_processes": [
                      {"pid": 7, "comm": "kworker", "uid": 0, "cores": 0.5}]}
        out.write(edit(t, sample) if edit else sample)
    out.write({"kind": "stop", "reason": stop})
    markers = cot.ChainWriter(folder / "markers.jsonl", binding_sha256=B)
    for name, t in zip(cot.MARKERS, marks):
        markers.write({"marker": name, "pid": 100, "t_monotonic": t})
    return folder


def test_a_clean_monitored_interval_is_timing_valid(tmp_path):
    verdict = cot.monitor_verdict(world(tmp_path), binding_sha256=B)
    assert verdict["timing_valid"] is True and verdict["reasons"] == []
    assert verdict["foreign_cpu_cores_mean"] == pytest.approx(0.5) and verdict["max_gap_seconds"] == pytest.approx(1.0)
    assert verdict["foreign_cpu_processes"][0]["comm"] == "kworker" and len(verdict["monitor_head_sha256"]) == 64


def _edit_at(when, change):
    def edit(t, sample):
        if t == when:
            change(sample)
        return sample
    return edit


@pytest.mark.parametrize("kwargs, needle", [
    ({"drop": range(8, 13)}, "gap"),
    ({"marks": (2.0, 3.0, 25.0)}, "cover"),
    ({"edit": _edit_at(10, lambda s: s["gpu"]["lines"].append("GPU-abc, 999, other, 10 MiB"))}, "another process"),
    ({"edit": _edit_at(10, lambda s: s["gpu"].update(lines=[]))}, "not on its card"),
    ({"edit": _edit_at(15, lambda s: s.update(gpu={"error": "TimeoutExpired"}))}, "nvidia-smi"),
    ({"edit": _edit_at(1, lambda s: s.update(exp36=[{"pid": 5, "cwd": "/x"}]))}, "exp_36"),
    ({"edit": _edit_at(9, lambda s: s["cpu"].update(foreign_cores=90.0))}, "foreign CPU"),
    ({"edit": _edit_at(12, lambda s: s.update(target_alive=False))}, "alive"),
    ({"marks": (2.0, 18.0, 3.0)}, "markers"),
    ({"stop": "target_gone"}, "target_gone"),
])
def test_every_failure_mode_withholds_the_timing(tmp_path, kwargs, needle):
    verdict = cot.monitor_verdict(world(tmp_path, **kwargs), binding_sha256=B)
    assert verdict["timing_valid"] is False and any(needle in r for r in verdict["reasons"]), verdict["reasons"]


def test_a_co_tenant_on_another_card_or_before_residency_is_not_ours(tmp_path):
    def edit(t, sample):
        if t == 10:
            sample["gpu"]["lines"].append("GPU-def, 999, other, 10 MiB")
        if t == 1:
            sample["gpu"]["lines"] = ["GPU-abc, 999, other, 10 MiB"]        # before gpu_resident (t = 2)
        return sample
    assert cot.monitor_verdict(world(tmp_path, edit=edit), binding_sha256=B)["timing_valid"] is True


def test_a_broken_or_foreign_chain_is_refused(tmp_path):
    folder = world(tmp_path)
    lines = (folder / "monitor.jsonl").read_text().splitlines()
    row = json.loads(lines[5])
    row["gpu"]["lines"] = []                                                  # an edit after the fact
    (folder / "monitor.jsonl").write_text("\n".join(lines[:5] + [json.dumps(row)] + lines[6:]) + "\n")
    assert "broken" in " ".join(cot.monitor_verdict(folder, binding_sha256=B)["reasons"])
    other = world(tmp_path / "x", binding="c" * 64)
    assert cot.monitor_verdict(other, binding_sha256=B)["timing_valid"] is False
    assert cot.monitor_verdict(tmp_path / "missing", binding_sha256=B)["timing_valid"] is False


def test_the_client_runs_the_real_monitor_child_end_to_end(tmp_path, monkeypatch):
    folder = tmp_path / "bin"
    folder.mkdir()
    (folder / "nvidia-smi").write_text(f"#!/bin/sh\necho 'GPU-abc, {os.getpid()}, python, 1 MiB'\n")
    (folder / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")
    monitor = cot.Monitor(tmp_path / "run" / "cotenancy", binding_sha256=B, card_uuid="abc", interval=0.2).start()
    monitor.marker("gpu_resident")
    monitor.marker("timed_start")
    time.sleep(1.0)
    monitor.marker("timed_end")
    assert monitor.stop() == 0
    verdict = cot.monitor_verdict(tmp_path / "run" / "cotenancy", binding_sha256=B)
    assert verdict["timing_valid"] is True, verdict["reasons"]
    with pytest.raises(RuntimeError, match="write-once"):
        cot.Monitor(tmp_path / "run" / "cotenancy", binding_sha256=B, card_uuid="abc").start()
