"""exp_37 run-bound co-tenancy attestation (plan_loc_efficiency.md §3, §10; Codex code review 1, BLOCKING 3).

``Monitor`` runs ``tools/exp37_cotenancy_monitor.py`` as a child of the timed process (``python -I -B``: isolated,
standard library only, a minimal environment) in a write-once directory, waits for its first sample, and writes the
hash-chained markers ``gpu_resident`` (the models are on the card), ``timed_start`` (before the warm-up) and
``timed_end`` (after the last timed query); ``stop`` creates the stop file and requires exit 0.

``monitor_verdict`` is ``timing_valid`` only if: both chains are intact (sha256, seq, links) and bound to the run's
binding sha256; the stream is start / samples / stop with the stop on the stop file; the three markers come from the
monitored pid in order; the samples cover [timed_start, timed_end] with no gap above ``MAX_GAP_SECONDS``; every
``nvidia-smi`` query succeeded and the run was alive; from ``gpu_resident`` to the first sample after ``timed_end``
the run is on its card and no other process is; no exp_36 process was ever seen; and the mean foreign CPU over the
timed interval is at most ``FOREIGN_CPU_MEAN_LIMIT_CORES``. It also returns every foreign process's CPU seconds over
the interval and both chain heads (the host attestation the summaries publish).
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.localization.pilot import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR_SCRIPT = REPO_ROOT / "tools" / "exp37_cotenancy_monitor.py"
ENVIRONMENT_KEPT = ("PATH", "HOME", "LANG", "LC_ALL")
MARKERS = ("gpu_resident", "timed_start", "timed_end")
# the single pinned set of exp_36 execution roots; identical in tools/exp37_cotenancy_monitor.py (pinned by a test)
EXP36_ROOTS = ("/home/yixunhu/codespace/exp-36-run", "/home/yixunhu/codespace/exp-36-yaw-c4-spread",
               "/home/yixunhu/codespace/.exp36-baseline-H35", "/home/yixunhu/codespace/exp-37-b3-synthesis")
MAX_GAP_SECONDS = 3.0
FOREIGN_CPU_MEAN_LIMIT_CORES = 4.0      # mean foreign cores over the timed interval (48-core host; a Planner decision)


class ChainWriter:
    """The marker chain: the same line format as the monitor's own stream."""

    def __init__(self, path, *, binding_sha256: str):
        if Path(path).exists():
            raise RuntimeError(f"{path} exists (marker chains are write-once)")
        self.path, self.binding, self.seq, self.prev = Path(path), binding_sha256, 0, None

    def write(self, record: dict) -> dict:
        body = {**record, "seq": self.seq, "prev_sha256": self.prev, "binding_sha256": self.binding}
        line = {**body, "sha256": canonical_sha256(body)}
        with self.path.open("a") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.seq, self.prev = self.seq + 1, line["sha256"]
        return line


class Monitor:
    def __init__(self, directory, *, binding_sha256: str, card_uuid: str, interval: float = 1.0, timeout: float = 60.0):
        self.dir, self.binding, self.card = Path(directory), binding_sha256, str(card_uuid)
        self.interval, self.timeout, self.process = float(interval), float(timeout), None

    def start(self) -> "Monitor":
        if self.dir.exists():
            raise RuntimeError(f"{self.dir} exists (a co-tenancy directory is write-once)")
        self.dir.mkdir(parents=True)
        stream, self.markers = self.dir / "monitor.jsonl", ChainWriter(self.dir / "markers.jsonl", binding_sha256=self.binding)
        self.log = (self.dir / "monitor.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(MONITOR_SCRIPT), "--output", str(stream), "--binding-sha256", self.binding,
             "--target-pid", str(os.getpid()), "--card-uuid", self.card, "--interval", repr(self.interval),
             "--stop-file", str(self.dir / "stop")],
            env={k: os.environ[k] for k in ENVIRONMENT_KEPT if k in os.environ}, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, close_fds=True)
        deadline = time.monotonic() + self.timeout
        while not stream.exists() or len(stream.read_text().splitlines()) < 2:
            if self.process.poll() is not None or time.monotonic() > deadline:
                self._end()
                raise RuntimeError(f"the co-tenancy monitor wrote no sample (exit {self.process.returncode}); see "
                                   f"{self.dir / 'monitor.log'}")
            time.sleep(0.05)
        return self

    def marker(self, name: str) -> None:
        if name not in MARKERS:
            raise ValueError(f"unknown marker {name!r}")
        self.markers.write({"marker": name, "pid": os.getpid(), "t_monotonic": time.monotonic(),
                            "t_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")})

    def _end(self) -> int:
        (self.dir / "stop").touch()
        try:
            code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()                         # our own child, past its stop deadline
            code = self.process.wait(timeout=self.timeout)
        self.log.close()
        return code

    def stop(self) -> int:
        code = self._end()
        if code != 0:
            raise RuntimeError(f"the co-tenancy monitor ended with exit {code} (the timing is not attested)")
        return code


def read_chain(path, binding_sha256: str) -> list[dict]:
    rows, prev = [], None
    for seq, line in enumerate(Path(path).read_text().splitlines()):
        row = json.loads(line)
        if row.get("sha256") != canonical_sha256({k: v for k, v in row.items() if k != "sha256"}) or \
                row.get("seq") != seq or row.get("prev_sha256") != prev or row.get("binding_sha256") != binding_sha256:
            raise RuntimeError(f"{path} is broken or bound elsewhere at record {seq}")
        rows.append(row)
        prev = row["sha256"]
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def _uuid(text) -> str:
    text = str(text or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def _apps(lines) -> list[tuple[str, int]] | None:
    try:
        return [(_uuid(fields[0]), int(fields[1])) for fields in (line.split(",") for line in lines)]
    except (IndexError, ValueError):
        return None


def monitor_verdict(directory, *, binding_sha256: str) -> dict:
    directory, problems = Path(directory), {}

    def flag(reason, seq=None):
        count, first = problems.get(reason, (0, seq))
        problems[reason] = (count + 1, first)
    try:
        records = read_chain(directory / "monitor.jsonl", binding_sha256)
        markers = read_chain(directory / "markers.jsonl", binding_sha256)
    except (OSError, ValueError, RuntimeError) as error:
        return {"timing_valid": False, "reasons": [f"the monitor / marker chain is missing or broken: {error}"]}
    start, stop, samples = records[0], records[-1], [r for r in records if r.get("kind") == "sample"]
    if start.get("kind") != "start" or stop.get("kind") != "stop" or len(samples) != len(records) - 2:
        flag("the monitor stream is not start, samples, stop")
    if start.get("proc") != "/proc":
        flag(f"the monitor watched {start.get('proc')!r}, not the host's /proc")
    if stop.get("reason") != "stop_file":
        flag(f"the monitor stopped on {stop.get('reason')}, not on the run's stop file")
    times = [m.get("t_monotonic") for m in markers]
    if [m.get("marker") for m in markers] != list(MARKERS) or any(m.get("pid") != start.get("target_pid") for m in markers) \
            or not all(isinstance(t, (int, float)) for t in times) or times != sorted(times):
        return {"timing_valid": False, "reasons": [f"the markers are not {list(MARKERS)} in order from the run"]}
    resident, t0, t1 = times
    stamps = [s["t_monotonic"] for s in samples]
    before, after = [t for t in stamps if t <= t0], [t for t in stamps if t >= t1]
    gap = None
    if not before or not after:
        flag("the samples do not cover the timed interval")
    else:
        window = [t for t in stamps if before[-1] <= t <= after[0]]
        gap = max((b - a for a, b in zip(window, window[1:])), default=0.0)
        if gap > MAX_GAP_SECONDS:
            flag(f"an observation gap of {gap:.2f} s inside the timed interval (limit {MAX_GAP_SECONDS} s)")
    card, pid, last = _uuid(start.get("card_uuid")), start.get("target_pid"), after[0] if after else t1
    foreign, seconds = [], {}
    for s in samples:
        gpu, t = s.get("gpu") or {}, s["t_monotonic"]
        apps = None if gpu.get("error") or gpu.get("returncode") != 0 else _apps(gpu.get("lines", []))
        if apps is None:
            flag("an nvidia-smi query failed or was unreadable", s["seq"])
        elif resident <= t <= last:
            if any(u == card and p != pid for u, p in apps):
                flag("another process on the run's card during its residency", s["seq"])
            if (card, pid) not in apps:
                flag("the run is not on its card during its residency", s["seq"])
        if s.get("exp36"):
            flag("exp_36 was on the host", s["seq"])
        if not s.get("target_alive"):
            flag("the run was not alive", s["seq"])
        if t0 < t <= last and isinstance(s.get("cpu"), dict) and "foreign_cores" in s["cpu"]:
            foreign.append(float(s["cpu"]["foreign_cores"]))
            for row in s.get("foreign_processes", []):
                key = (row["pid"], row["comm"])
                seconds[key] = seconds.get(key, 0.0) + float(row["cores"]) * float(s["cpu"]["interval_seconds"])
    mean = sum(foreign) / len(foreign) if foreign else None
    if mean is None or mean > FOREIGN_CPU_MEAN_LIMIT_CORES:
        flag(f"the foreign CPU over the timed interval ({mean} cores) is unmeasured or above "
             f"{FOREIGN_CPU_MEAN_LIMIT_CORES} cores")
    reasons = [f"{reason} ({count} sample(s), first seq {first})" for reason, (count, first) in problems.items()]
    return {"timing_valid": not reasons, "reasons": reasons, "sample_count": len(samples),
            "timed_seconds": t1 - t0, "max_gap_seconds": gap, "foreign_cpu_cores_mean": mean,
            "foreign_cpu_cores_max": max(foreign, default=None),
            "foreign_cpu_processes": [{"pid": p, "comm": c, "cpu_seconds": v} for (p, c), v in
                                      sorted(seconds.items(), key=lambda item: -item[1])],
            "monitor_head_sha256": records[-1]["sha256"], "markers_head_sha256": markers[-1]["sha256"]}
