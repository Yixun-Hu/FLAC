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
PRODUCER = {"pid": 100, "pid_start_ticks": 777, "hostname": "h", "card_uuid": "abc"}


def _script():
    spec = importlib.util.spec_from_file_location("exp37_cotenancy_monitor", REPO / "tools" / "exp37_cotenancy_monitor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def world(tmp_path, *, edit=None, marks=(2.0, 3.0, 18.0), stop="stop_file", binding=B, drop=(), proc=None,
          start_edit=None):
    """A monitor stream (samples every second over t = 0..20 s) and its markers; ``edit(t, sample)`` alters one."""
    folder = tmp_path / "cotenancy"
    folder.mkdir(parents=True)
    out = _script().ChainWriter(folder / "monitor.jsonl", binding_sha256=binding)
    start = {"kind": "start", "target_pid": 100, "target_start": 777, "target_comm": "python", "card_uuid": "GPU-ABC",
             "interval_seconds": 1.0, "hostname": "h", "exp36_roots": list(cot.EXP36_ROOTS),
             "proc": "/proc" if proc is None else proc}
    out.write(start_edit(start) if start_edit else start)
    for t in range(21):
        if t in drop:
            continue
        sample = {"kind": "sample", "t_monotonic": float(t), "target_alive": True, "exp36": [],
                  "gpu": {"returncode": 0, "lines": ["GPU-abc, 100, python, 900 MiB"]},
                  "cpu": {"interval_start": t - 1.0, "interval_end": float(t), "interval_seconds": 1.0,
                          "foreign_cores": 0.5}, "foreign_processes": [
                      {"pid": 7, "comm": "kworker", "uid": 0, "cores": 0.5}]}
        out.write(edit(t, sample) if edit else sample)
    out.write({"kind": "stop", "reason": stop})
    markers = cot.ChainWriter(folder / "markers.jsonl", binding_sha256=B, producer=PRODUCER)
    for name, t in zip(cot.MARKERS, marks):
        markers.write({"marker": name, "pid": 100, "t_monotonic": t})
    return folder


def test_a_clean_monitored_interval_is_timing_valid(tmp_path):
    verdict = cot.monitor_verdict(world(tmp_path), binding_sha256=B, producer=PRODUCER)
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
    ({"proc": "/tmp/fake_proc"}, "/proc"),
    ({"start_edit": lambda s: {**s, "exp36_roots": s["exp36_roots"][1:]}}, "root set"),
    ({"start_edit": lambda s: {**s, "target_pid": 101}}, "producer"),
    ({"start_edit": lambda s: {**s, "target_start": 778}}, "producer"),
    ({"start_edit": lambda s: {**s, "card_uuid": "GPU-def"}}, "producer"),
    ({"start_edit": lambda s: {**s, "hostname": "other"}}, "producer"),
])
def test_every_failure_mode_withholds_the_timing(tmp_path, kwargs, needle):
    verdict = cot.monitor_verdict(world(tmp_path, **kwargs), binding_sha256=B, producer=PRODUCER)
    assert verdict["timing_valid"] is False and any(needle in r for r in verdict["reasons"]), verdict["reasons"]


def test_a_co_tenant_on_another_card_or_before_residency_is_not_ours(tmp_path):
    def edit(t, sample):
        if t == 10:
            sample["gpu"]["lines"].append("GPU-def, 999, other, 10 MiB")
        if t == 1:
            sample["gpu"]["lines"] = ["GPU-abc, 999, other, 10 MiB"]        # before gpu_resident (t = 2)
        return sample
    assert cot.monitor_verdict(world(tmp_path, edit=edit), binding_sha256=B, producer=PRODUCER)["timing_valid"] is True


def test_a_broken_or_foreign_chain_is_refused(tmp_path):
    folder = world(tmp_path)
    lines = (folder / "monitor.jsonl").read_text().splitlines()
    row = json.loads(lines[5])
    row["gpu"]["lines"] = []                                                  # an edit after the fact
    (folder / "monitor.jsonl").write_text("\n".join(lines[:5] + [json.dumps(row)] + lines[6:]) + "\n")
    assert "broken" in " ".join(cot.monitor_verdict(folder, binding_sha256=B, producer=PRODUCER)["reasons"])
    other = world(tmp_path / "x", binding="c" * 64)
    assert cot.monitor_verdict(other, binding_sha256=B, producer=PRODUCER)["timing_valid"] is False
    assert cot.monitor_verdict(tmp_path / "missing", binding_sha256=B, producer=PRODUCER)["timing_valid"] is False


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
    import socket
    with open(f"/proc/{os.getpid()}/stat") as handle:
        start = int(handle.read().rpartition(")")[2].split()[19])
    me = {"pid": os.getpid(), "pid_start_ticks": start, "hostname": socket.gethostname(), "card_uuid": "abc"}
    verdict = cot.monitor_verdict(tmp_path / "run" / "cotenancy", binding_sha256=B, producer=me)
    host_state = ("exp_36 was on the host", "the foreign CPU")       # this CPU test box is not a quiet timing host
    assert [r for r in verdict["reasons"] if not r.startswith(host_state)] == [], verdict["reasons"]
    assert verdict["sample_count"] >= 4 and verdict["max_gap_seconds"] < 1.0
    with pytest.raises(RuntimeError, match="write-once"):
        cot.Monitor(tmp_path / "run" / "cotenancy", binding_sha256=B, card_uuid="abc").start()


def test_the_foreign_cpu_mean_is_time_weighted_over_the_exact_timed_interval(tmp_path):
    """Review 2 HIGH 2: sum(foreign x overlap) / sum(overlap) with [timed_start, timed_end] -- an unweighted mean of
    the four overlapping samples would be 4.5 cores (above the limit); the time-weighted one is 2.75."""
    folder = tmp_path / "cotenancy"
    folder.mkdir()
    out = _script().ChainWriter(folder / "monitor.jsonl", binding_sha256=B)
    out.write({"kind": "start", "target_pid": 100, "target_start": 777, "target_comm": "python", "card_uuid": "abc",
               "interval_seconds": 1.0, "hostname": "h", "exp36_roots": list(cot.EXP36_ROOTS), "proc": "/proc"})
    for t, (a, b), foreign in ((2.0, (1.0, 2.0), 0.0), (3.5, (2.0, 3.5), 14.0), (4.0, (3.5, 4.0), 2.0),
                               (6.5, (4.0, 6.5), 1.0), (7.5, (6.5, 7.5), 1.0)):
        out.write({"kind": "sample", "t_monotonic": t, "target_alive": True, "exp36": [],
                   "gpu": {"returncode": 0, "lines": ["GPU-abc, 100, python, 9 MiB"]},
                   "cpu": {"interval_start": a, "interval_end": b, "interval_seconds": b - a, "foreign_cores": foreign},
                   "foreign_processes": [{"pid": 7, "comm": "make", "uid": 1, "cores": foreign}]})
    out.write({"kind": "stop", "reason": "stop_file"})
    markers = cot.ChainWriter(folder / "markers.jsonl", binding_sha256=B)
    for name, t in zip(cot.MARKERS, (1.5, 3.0, 7.0)):
        markers.write({"marker": name, "pid": 100, "t_monotonic": t})
    verdict = cot.monitor_verdict(folder, binding_sha256=B, producer=PRODUCER)
    assert verdict["timing_valid"] is True, verdict["reasons"]
    assert verdict["foreign_cpu_cores_mean"] == pytest.approx(2.75) and verdict["foreign_cpu_cores_max"] == 14.0
    assert [round(i["overlap_seconds"], 6) for i in verdict["foreign_cpu_intervals"]] == [0.5, 0.5, 2.5, 0.5]
    assert verdict["foreign_cpu_processes"] == [{"pid": 7, "comm": "make", "cpu_seconds": pytest.approx(11.0)}]
