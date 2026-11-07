"""exp_37 fix round (Codex BLOCKING 3): the run-bound co-tenancy monitor ``tools/exp37_cotenancy_monitor.py``.

A stdlib-only child process that samples, every ``--interval`` seconds until the run's stop file (or the run's death),
the card's compute apps (``nvidia-smi``; a failure is recorded), host CPU from ``/proc/stat`` minus the run's and its
own CPU (the foreign CPU) with EVERY foreign process active in the interval, and the exp_36 host gate (any process
whose cwd or argv lies under an exp_36 root) -- each record hash-chained and bound to the run's binding sha256.
``--gate-only`` writes one hashed gate record and exits 0 (clean) / 3 (exp_36 present). CPU; fake /proc trees and a
fake ``nvidia-smi``. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.localization.pilot import canonical_sha256

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "exp37_cotenancy_monitor.py"
ROOT36 = "/home/yixunhu/codespace/exp-36-yaw-c4-spread"


def _module():
    spec = importlib.util.spec_from_file_location("exp37_cotenancy_monitor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_proc(root, processes, busy, total):
    """A /proc tree: ``processes`` = {pid: (comm, ticks, cwd or None, argv)}; host ``busy`` / ``total`` ticks."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text(f"cpu  {busy} 0 0 {total - busy} 0 0 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0 0 0\n")
    for pid, (comm, ticks, cwd, argv) in processes.items():
        folder = root / str(pid)
        folder.mkdir(exist_ok=True)
        fields = ["S", "1"] + ["0"] * 9 + [str(ticks), "0", "0", "0"] + ["0"] * 4 + ["123"] + ["0"] * 20
        (folder / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
        if cwd and not (folder / "cwd").is_symlink():
            os.symlink(cwd, folder / "cwd")
        (folder / "cmdline").write_bytes(b"".join(a.encode() + b"\0" for a in argv))


def test_the_monitor_imports_only_the_standard_library():
    allowed = {"argparse", "datetime", "hashlib", "json", "os", "socket", "subprocess", "sys", "time"}
    names = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= allowed | {"__future__"}, names


def test_the_chain_hashes_like_the_repo_and_links_every_record(tmp_path):
    mon = _module()
    writer = mon.ChainWriter(tmp_path / "chain.jsonl", binding_sha256="b" * 64)
    writer.write({"kind": "start", "x": 1})
    writer.write({"kind": "sample", "y": [1.5, None]})
    rows = [json.loads(line) for line in (tmp_path / "chain.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in rows] == [0, 1] and rows[0]["prev_sha256"] is None
    assert rows[1]["prev_sha256"] == rows[0]["sha256"] and all(r["binding_sha256"] == "b" * 64 for r in rows)
    assert all(r["sha256"] == canonical_sha256({k: v for k, v in r.items() if k != "sha256"}) for r in rows)
    assert mon.canonical_sha256({"a": [1, {"b": 2}]}) == canonical_sha256({"a": [1, {"b": 2}]})


def test_foreign_cpu_is_host_busy_minus_the_run_and_the_monitor(tmp_path):
    mon = _module()
    proc, clock = tmp_path / "proc", iter([100.0, 102.0])
    fake_proc(proc, {10: ("python", 500, None, ["python"]), 11: ("monitor", 50, None, ["m"]),
                     12: ("make", 300, "/tmp", ["make"]), 13: ("idle", 7, None, ["idle"])}, busy=1000, total=10000)
    sampler = mon.Sampler(proc=proc, target_pid=10, monitor_pid=11, roots=(ROOT36,), run=None, clock=lambda: next(clock))
    first = sampler.cpu_sample()
    assert first is None                                                       # the baseline
    fake_proc(proc, {10: ("python", 700, None, ["python"]), 11: ("monitor", 60, None, ["m"]),
                     12: ("make", 500, "/tmp", ["make"]), 13: ("idle", 7, None, ["idle"]),
                     14: ("new", 40, None, ["new"])}, busy=1700, total=19600)
    cpu, foreign = sampler.cpu_sample()
    tick = 1.0 / os.sysconf("SC_CLK_TCK")
    assert cpu["interval_seconds"] == pytest.approx(2.0)
    assert cpu["host_busy_cores"] == pytest.approx(700 * tick / 2.0)          # cores = ticks / CLK_TCK / seconds
    assert cpu["target_cores"] == pytest.approx(200 * tick / 2.0)
    assert cpu["monitor_cores"] == pytest.approx(10 * tick / 2.0)
    assert cpu["foreign_cores"] == pytest.approx((700 - 200 - 10) * tick / 2.0)
    assert {(f["pid"], f["comm"]) for f in foreign} == {(12, "make"), (14, "new")}       # idle 13 did not run
    assert {f["pid"]: f["cores"] for f in foreign}[12] == pytest.approx(200 * tick / 2.0)


def test_the_exp36_gate_sees_cwd_and_argv_on_path_boundaries(tmp_path):
    mon = _module()
    proc = tmp_path / "proc"
    fake_proc(proc, {20: ("bash", 1, ROOT36 + "/runs", ["bash"]),
                     21: ("python", 1, "/tmp", ["python", "/home/yixunhu/codespace/.exp36-baseline-H35/eval.py"]),
                     22: ("python", 1, ROOT36 + "-other", ["python", ROOT36 + "-other/x.py"]),
                     23: ("python", 1, "/tmp", ["python", "x.py"])}, busy=1, total=2)
    sampler = mon.Sampler(proc=proc, target_pid=1, monitor_pid=2, roots=mon.EXP36_ROOTS, run=None, clock=time.monotonic)
    assert sorted(hit["pid"] for hit in sampler.exp36_hits()) == [20, 21]


def test_a_failed_gpu_query_is_recorded_never_hidden():
    mon = _module()

    def broken(argv, **kw):
        raise FileNotFoundError("nvidia-smi")
    sampler = mon.Sampler(proc=Path("/proc"), target_pid=1, monitor_pid=2, roots=(), run=broken, clock=time.monotonic)
    assert "FileNotFoundError" in sampler.gpu_sample()["error"]

    def nonzero(argv, **kw):
        return subprocess.CompletedProcess(argv, 9, stdout="", stderr="NVML: unknown error")
    sampler.run = nonzero
    assert sampler.gpu_sample()["returncode"] == 9


def _fake_smi(folder, line):
    folder.mkdir(parents=True, exist_ok=True)
    smi = folder / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho '{line}'\n")
    smi.chmod(0o755)
    return folder


def _env(tmp_path, line):
    return {"PATH": f"{_fake_smi(tmp_path / 'bin', line)}:/usr/bin:/bin", "HOME": str(tmp_path)}


def _records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_the_monitor_process_samples_until_its_stop_file(tmp_path):
    out, stop = tmp_path / "monitor.jsonl", tmp_path / "stop"
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SCRIPT), "--output", str(out), "--binding-sha256", "c" * 64,
                              "--target-pid", str(os.getpid()), "--card-uuid", "GPU-abc", "--interval", "0.2",
                              "--stop-file", str(stop)], env=_env(tmp_path, f"GPU-abc, {os.getpid()}, python, 10 MiB"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (not out.exists() or len(out.read_text().splitlines()) < 4):
        time.sleep(0.1)
    stop.touch()
    assert child.wait(timeout=30) == 0
    rows = _records(out)
    assert rows[0]["kind"] == "start" and rows[0]["target_pid"] == os.getpid() and rows[-1]["kind"] == "stop"
    assert rows[-1]["reason"] == "stop_file" and sum(r["kind"] == "sample" for r in rows) >= 3
    sample = next(r for r in rows if r["kind"] == "sample")
    assert sample["gpu"]["returncode"] == 0 and sample["gpu"]["lines"] == [f"GPU-abc, {os.getpid()}, python, 10 MiB"]
    assert sample["target_alive"] is True and isinstance(sample["exp36"], list)


def test_the_monitor_ends_when_its_run_dies(tmp_path):
    out = tmp_path / "monitor.jsonl"
    target = subprocess.Popen(["sleep", "1"])
    child = subprocess.Popen([sys.executable, "-I", "-B", str(SCRIPT), "--output", str(out), "--binding-sha256", "c" * 64,
                              "--target-pid", str(target.pid), "--card-uuid", "GPU-abc", "--interval", "0.2",
                              "--stop-file", str(tmp_path / "stop")], env=_env(tmp_path, "GPU-abc, 1, x, 1 MiB"))
    target.wait(timeout=30)
    assert child.wait(timeout=30) == 3
    assert _records(out)[-1]["reason"] == "target_gone"


def test_gate_only_writes_one_hashed_record(tmp_path):
    proc = tmp_path / "proc"
    fake_proc(proc, {30: ("python", 1, "/tmp", ["python", "x.py"])}, busy=1, total=2)
    clean = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--gate-only", "--proc", str(proc), "--output",
                            str(tmp_path / "gate.json")], env=_env(tmp_path, ""), capture_output=True, text=True)
    record = json.loads((tmp_path / "gate.json").read_text())
    assert clean.returncode == 0 and record["exp36_processes"] == [] and record["gate"] == "clean"
    assert record["sha256"] == canonical_sha256({k: v for k, v in record.items() if k != "sha256"})
    fake_proc(proc, {31: ("bash", 1, ROOT36, ["bash"])}, busy=1, total=2)
    dirty = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--gate-only", "--proc", str(proc), "--output",
                            str(tmp_path / "gate2.json")], env=_env(tmp_path, ""), capture_output=True, text=True)
    assert dirty.returncode == 3 and json.loads((tmp_path / "gate2.json").read_text())["gate"] == "exp36_present"
    again = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--gate-only", "--proc", str(proc), "--output",
                            str(tmp_path / "gate2.json")], env=_env(tmp_path, ""), capture_output=True, text=True)
    assert again.returncode == 2                                                 # write-once


def test_gate_only_creates_its_record_directory(tmp_path):
    """The runbook's gate steps write $D/gates/<name>.json into a directory that does not exist yet."""
    proc = tmp_path / "proc"
    fake_proc(proc, {30: ("python", 1, "/tmp", ["python", "x.py"])}, busy=1, total=2)
    out = tmp_path / "gates" / "p0.json"
    done = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--gate-only", "--proc", str(proc), "--output",
                           str(out)], env=_env(tmp_path, ""), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["gate"] == "clean"
