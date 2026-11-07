#!/usr/bin/env python3
"""exp_37 run-bound co-tenancy monitor (plan_loc_efficiency.md §3, §10; Codex code review 1, BLOCKING 3).

Started by the timed process it watches (the runner or the selector tool) as ``python -I -B <this file> ...`` --
isolated, standard library only, no repo import -- it appends one hash-chained JSON line per record to ``--output``,
each bound to ``--binding-sha256`` (the run manifest's sha256, or the selector run's binding):

- ``start``: the run's pid, this pid, the card UUID, the interval, the exp_36 roots, CLK_TCK, the CPU count;
- ``sample`` (every ``--interval`` s): ``nvidia-smi --query-compute-apps`` (return code and lines, or the error --
  never dropped); host CPU from ``/proc/stat`` as busy cores, the run's cores (utime + stime + cutime + cstime), this
  monitor's cores and the remainder, the FOREIGN cores, with EVERY foreign process that used CPU in the interval (pid,
  comm, uid, cores); the exp_36 host gate (every process whose cwd or argv lies under an exp_36 root); the run's
  liveness;
- ``stop``: ``stop_file`` (exit 0) once the run creates ``--stop-file`` (after one last sample), or ``target_gone``
  (exit 3) when the run died first.

``--gate-only --output <json>`` writes one hashed exp_36 host-gate record (write-once; exit 0 clean, 3 exp_36 present).
The verdict over the records is ``src/localization/exp37_cotenancy.py::monitor_verdict``.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

EXP36_ROOTS = ("/home/yixunhu/codespace/exp-36-yaw-c4-spread", "/home/yixunhu/codespace/.exp36-baseline-H35")
GPU_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"]
TARGET_GONE, EXISTS, EXP36_PRESENT = 3, 2, 3


def canonical_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


class ChainWriter:
    """Append-only JSON lines; every record carries its seq, the previous record's sha256 and the binding."""

    def __init__(self, path, *, binding_sha256: str):
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists (monitor streams are write-once)")
        self.path, self.binding, self.seq, self.prev = str(path), binding_sha256, 0, None

    def write(self, record: dict) -> dict:
        body = {**record, "seq": self.seq, "prev_sha256": self.prev, "binding_sha256": self.binding}
        line = {**body, "sha256": canonical_sha256(body)}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.seq, self.prev = self.seq + 1, line["sha256"]
        return line


def host_busy_ticks(proc) -> int:
    with open(os.path.join(proc, "stat")) as handle:
        values = [int(x) for x in handle.readline().split()[1:9]]     # user nice system idle iowait irq softirq steal
    return sum(values) - values[3] - values[4]


def read_process(proc, pid):
    """``/proc/<pid>/stat`` as {state, comm, start, own (utime+stime), all (+cutime+cstime)}; None if gone."""
    try:
        with open(os.path.join(proc, str(pid), "stat")) as handle:
            text = handle.read()
    except OSError:
        return None
    head, _, tail = text.rpartition(")")
    rest = tail.split()
    own = int(rest[11]) + int(rest[12])
    return {"state": rest[0], "comm": head.partition("(")[2], "start": rest[19], "own": own,
            "all": own + int(rest[13]) + int(rest[14])}


def _mentions(text: str, root: str) -> bool:
    """``root`` occurs in ``text`` as a whole path (followed by '/' or the end)."""
    start = text.find(root)
    while start >= 0:
        end = start + len(root)
        if end == len(text) or text[end] == "/":
            return True
        start = text.find(root, start + 1)
    return False


class Sampler:
    def __init__(self, *, proc, target_pid: int, monitor_pid: int, roots, run, clock):
        self.proc, self.target, self.me, self.roots = str(proc), int(target_pid), int(monitor_pid), tuple(roots)
        self.run, self.clock, self.tick, self.previous = run, clock, 1.0 / os.sysconf("SC_CLK_TCK"), None

    def pids(self) -> list[int]:
        return sorted(int(name) for name in os.listdir(self.proc) if name.isdigit())

    def _uid(self, pid):
        try:
            return os.stat(os.path.join(self.proc, str(pid))).st_uid
        except OSError:
            return None

    def cpu_sample(self):
        """None the first time (the baseline); then (cpu, foreign processes) over the interval since the last call."""
        now, busy = self.clock(), host_busy_ticks(self.proc)
        table = {pid: row for pid in self.pids() for row in (read_process(self.proc, pid),) if row is not None}
        previous, self.previous = self.previous, (now, busy, table)
        if previous is None:
            return None
        then, busy_before, before = previous
        seconds = now - then
        if seconds <= 0:
            return {"interval_seconds": seconds, "error": "non-positive interval"}, []

        def delta(pid, key):
            old, new = before.get(pid), table.get(pid)
            if new is None:
                return 0
            return new[key] if old is None or old["start"] != new["start"] else new[key] - old[key]

        def cores(ticks):
            return ticks * self.tick / seconds
        target, mine, host = delta(self.target, "all"), delta(self.me, "all"), busy - busy_before
        foreign = [{"pid": pid, "comm": row["comm"], "uid": self._uid(pid), "cores": cores(delta(pid, "own"))}
                   for pid, row in sorted(table.items()) if pid not in (self.target, self.me) and delta(pid, "own") > 0]
        return ({"interval_seconds": seconds, "host_busy_cores": cores(host), "target_cores": cores(target),
                 "monitor_cores": cores(mine), "foreign_cores": max(0.0, cores(host - target - mine))}, foreign)

    def exp36_hits(self) -> list[dict]:
        hits = []
        for pid in self.pids():
            base = os.path.join(self.proc, str(pid))
            try:
                cwd = os.readlink(os.path.join(base, "cwd"))
            except OSError:
                cwd = None
            try:
                with open(os.path.join(base, "cmdline"), "rb") as handle:
                    argv = [arg.decode(errors="replace") for arg in handle.read().split(b"\0") if arg]
            except OSError:
                argv = []
            if any(_mentions(text, root) for root in self.roots for text in ([cwd] if cwd else []) + argv):
                hits.append({"pid": pid, "cwd": cwd, "argv": argv[:8]})
        return hits

    def gpu_sample(self) -> dict:
        started = self.clock()
        try:
            done = self.run(GPU_QUERY, capture_output=True, text=True, timeout=10, check=False)
        except Exception as error:                     # noqa: BLE001 -- recorded, never hidden
            return {"error": f"{type(error).__name__}: {error}", "seconds": self.clock() - started}
        return {"returncode": done.returncode, "lines": [line.strip() for line in done.stdout.splitlines() if line.strip()],
                "stderr": (done.stderr or "").strip()[:500], "seconds": self.clock() - started}

    def target_alive(self, start) -> bool:
        row = read_process(self.proc, self.target)
        return row is not None and row["start"] == start and row["state"] not in ("Z", "X")


def run_monitor(args) -> int:
    try:
        writer = ChainWriter(args.output, binding_sha256=args.binding_sha256)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return EXISTS
    sampler = Sampler(proc=args.proc, target_pid=args.target_pid, monitor_pid=os.getpid(), roots=EXP36_ROOTS,
                      run=subprocess.run, clock=time.monotonic)
    target = read_process(args.proc, args.target_pid)
    writer.write({"kind": "start", "target_pid": args.target_pid, "target_start": target and target["start"],
                  "monitor_pid": os.getpid(), "card_uuid": args.card_uuid, "interval_seconds": args.interval,
                  "exp36_roots": list(EXP36_ROOTS), "proc": args.proc, "clk_tck": os.sysconf("SC_CLK_TCK"),
                  "cpu_count": os.cpu_count(),
                  "hostname": socket.gethostname(), "t_monotonic": time.monotonic(), "t_utc": utc_now()})
    sampler.cpu_sample()                                                   # the CPU baseline
    samples = 0
    while True:
        began = time.monotonic()
        stop = os.path.exists(args.stop_file)
        alive = target is not None and sampler.target_alive(target["start"])
        gpu = sampler.gpu_sample()
        cpu, foreign = sampler.cpu_sample()
        writer.write({"kind": "sample", "t_monotonic": began, "t_end_monotonic": time.monotonic(), "t_utc": utc_now(),
                      "gpu": gpu, "cpu": cpu, "foreign_processes": foreign, "exp36": sampler.exp36_hits(),
                      "target_alive": alive})
        samples += 1
        if stop or not alive:
            reason = "stop_file" if stop else "target_gone"
            writer.write({"kind": "stop", "reason": reason, "samples": samples, "t_monotonic": time.monotonic(),
                          "t_utc": utc_now()})
            return 0 if stop else TARGET_GONE
        time.sleep(max(0.0, args.interval - (time.monotonic() - began)))


def run_gate(args) -> int:
    if os.path.exists(args.output):
        print(f"{args.output} exists (gate records are write-once)", file=sys.stderr)
        return EXISTS
    sampler = Sampler(proc=args.proc, target_pid=-1, monitor_pid=os.getpid(), roots=EXP36_ROOTS, run=None,
                      clock=time.monotonic)
    hits = sampler.exp36_hits()
    body = {"schema_version": 1, "record_type": "exp37_exp36_host_gate", "checked_at_utc": utc_now(),
            "hostname": socket.gethostname(), "exp36_roots": list(EXP36_ROOTS), "exp36_processes": hits,
            "gate": "exp36_present" if hits else "clean"}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = f"{args.output}.tmp"
    with open(temporary, "w") as handle:
        handle.write(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"gate": body["gate"], "exp36_processes": len(hits), "output": args.output}))
    return EXP36_PRESENT if hits else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--proc", default="/proc")
    parser.add_argument("--binding-sha256")
    parser.add_argument("--target-pid", type=int)
    parser.add_argument("--card-uuid")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stop-file")
    args = parser.parse_args(argv)
    if args.gate_only:
        return run_gate(args)
    if not (args.binding_sha256 and args.target_pid and args.card_uuid and args.stop_file) or args.interval <= 0:
        parser.error("the monitor needs --binding-sha256, --target-pid, --card-uuid, --stop-file and a positive --interval")
    return run_monitor(args)


if __name__ == "__main__":
    sys.exit(main())
