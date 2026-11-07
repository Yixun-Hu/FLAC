"""exp_37 fix round 2 (Codex code review 2, BLOCKING 1, HIGH 1 / 2): the monitor's exp_36 gate and CPU intervals.

- The gate's root set is the single pinned set -- the exp_36 run worktree ``exp-36-run``, ``exp-36-yaw-c4-spread``,
  ``.exp36-baseline-H35`` and the B3 leaf worktree ``exp-37-b3-synthesis`` -- recorded in every gate record and
  monitor stream, and identical in the verdict module; a process is exp_36 when its RESOLVED cwd, its executable, a
  script on its argv (absolute, or relative to its cwd) or an embedded argv path lies under a root (whole paths only).
- Every CPU sample records its exact interval (start / end) for the time-weighted verdict; the start record carries
  the watched pid's /proc start time and comm.
CPU; fake /proc trees. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import json
import subprocess
import sys

import pytest

from test_exp37_monitor import SCRIPT, _env, _module, fake_proc

ROOTS = ("/home/yixunhu/codespace/exp-36-run", "/home/yixunhu/codespace/exp-36-yaw-c4-spread",
         "/home/yixunhu/codespace/.exp36-baseline-H35", "/home/yixunhu/codespace/exp-37-b3-synthesis")


def test_the_root_set_is_pinned_once_for_the_gate_and_the_verdict():
    from src.localization import exp37_cotenancy as cot
    assert tuple(_module().EXP36_ROOTS) == ROOTS == tuple(cot.EXP36_ROOTS)


@pytest.mark.parametrize("root", ROOTS)
def test_a_shell_and_a_worker_in_every_root_trip_the_gate(tmp_path, root):
    mon = _module()
    proc = tmp_path / "proc"
    fake_proc(proc, {40: ("bash", 1, root, ["bash"]),                                       # a shell in the root
                     41: ("python", 1, root + "/sub", ["python", "worker.py", "--n", "3"]),  # relative script
                     42: ("python", 1, "/tmp", ["python", root + "/tools/w.py"]),           # absolute script
                     43: ("python", 1, "/tmp", ["python", "-m", "x"], root + "/.venv/bin/python"),   # its executable
                     44: ("python", 1, "/tmp", ["python", "x.py", f"--out={root}/o"]),         # an embedded path
                     45: ("bash", 1, root + "-other", ["bash", "x.sh"]),                         # a sibling prefix
                     46: ("python", 1, "/tmp", ["python", "../exp-36-run-x/w.py"])},               # outside
              busy=1, total=2)
    sampler = mon.Sampler(proc=proc, target_pid=1, monitor_pid=2, roots=mon.EXP36_ROOTS, run=None, clock=lambda: 0.0)
    hits = {h["pid"]: h for h in sampler.exp36_hits()}
    assert sorted(hits) == [40, 41, 42, 43, 44]
    assert hits[40]["rule"] == "cwd" and hits[43]["rule"] == "exe" and hits[42]["rule"] == "script"
    assert all({"pid", "comm", "cwd", "exe", "rule", "argv"} <= set(h) for h in hits.values())


def test_a_relative_script_is_resolved_against_its_own_cwd(tmp_path):
    mon = _module()
    proc = tmp_path / "proc"
    fake_proc(proc, {50: ("python", 1, "/home/yixunhu/codespace", ["python", "exp-36-run/run.py"]),
                     51: ("python", 1, "/home/yixunhu/codespace/elsewhere", ["python", "exp-36-run/run.py"])},
              busy=1, total=2)
    sampler = mon.Sampler(proc=proc, target_pid=1, monitor_pid=2, roots=mon.EXP36_ROOTS, run=None, clock=lambda: 0.0)
    assert [h["pid"] for h in sampler.exp36_hits()] == [50]


def test_every_cpu_sample_carries_its_exact_interval(tmp_path):
    mon = _module()
    proc, clock = tmp_path / "proc", iter([10.0, 12.5])
    fake_proc(proc, {10: ("python", 5, None, ["python"])}, busy=100, total=1000)
    sampler = mon.Sampler(proc=proc, target_pid=10, monitor_pid=11, roots=(), run=None, clock=lambda: next(clock))
    assert sampler.cpu_sample() is None
    fake_proc(proc, {10: ("python", 9, None, ["python"])}, busy=180, total=1900)
    cpu, _foreign = sampler.cpu_sample()
    assert (cpu["interval_start"], cpu["interval_end"], cpu["interval_seconds"]) == (10.0, 12.5, 2.5)


def test_the_start_record_and_the_gate_record_carry_the_pinned_roots(tmp_path):
    out, stop = tmp_path / "monitor.jsonl", tmp_path / "stop"
    import os
    import time
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SCRIPT), "--output", str(out), "--binding-sha256", "c" * 64,
                              "--target-pid", str(os.getpid()), "--card-uuid", "GPU-abc", "--interval", "0.2",
                              "--stop-file", str(stop)], env=_env(tmp_path, f"GPU-abc, {os.getpid()}, python, 1 MiB"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (not out.exists() or len(out.read_text().splitlines()) < 3):
        time.sleep(0.1)
    stop.touch()
    assert child.wait(timeout=30) == 0
    start = json.loads(out.read_text().splitlines()[0])
    with open(f"/proc/{os.getpid()}/stat") as handle:
        mine = int(handle.read().rpartition(")")[2].split()[19])
    assert start["exp36_roots"] == list(ROOTS) and start["target_start"] == mine and start["target_comm"]
    proc = tmp_path / "proc"
    fake_proc(proc, {30: ("python", 1, "/tmp", ["python", "x.py"])}, busy=1, total=2)
    subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--gate-only", "--proc", str(proc), "--output",
                    str(tmp_path / "gate.json")], env=_env(tmp_path, ""), check=True, capture_output=True)
    assert json.loads((tmp_path / "gate.json").read_text())["exp36_roots"] == list(ROOTS)
