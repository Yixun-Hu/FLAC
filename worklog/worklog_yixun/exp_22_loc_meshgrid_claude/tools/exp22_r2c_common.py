#!/usr/bin/env python3
"""exp_22 R2-C pre-flight: the shared READ-ONLY card observation (plan_loc_meshgrid_r2_bf_yaw.md §9 A7 r2 (v),
r3 (b) (d) (h)). ONE implementation of the card rule for the detector, the launcher's sealed capture polls, its
launch re-checks and the supervisor's re-checks before and during every line. A card is FREE at an observation iff

  (a) nvidia-smi lists no compute process on its UUID;
  (b) no file <reservations>/<uuid>.json exists -- any holder, an exp_35 phase-(ii) reservation (its intent)
      included; a malformed file is present (busy);
  (c) GPU 0: no process holds the exp_35 supervisor lock (O_RDONLY + LOCK_SH | LOCK_NB, as the reviewed exp_35 dry
      run probes it; absent or unheld = may be free) -- held, GPU 0 is the exp_35 hand-over target;
  (d) GPU 1: Yixun's queue log carries an ``EVAL_DONE ACEattn s43 `` line and no ``exp30_attn_eval.sh`` runs.

An unreadable GPU map / compute table / reservation file or directory / lock / queue log / /proc entry raises
``Unreadable``: busy for the detector, a refusal for the launcher and the supervisor. Nothing in this module writes,
reserves, spawns a workload or signals a process: nvidia-smi queries and reads only.

Written by the exp_22 A7 Coder (Claude Opus 5.5, max effort).
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

# --- BEGIN REGISTERED CONFIG (the sandbox tests replace this block, and only this block) ---
RESERVATIONS = "/home/yixunhu/codespace/.gpu_reservations"
EXP35_LOCK = ("/home/yixunhu/codespace/exp-35-orbitrir-yaw/worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude/"
              "exp35_supervisor_state.json.lock")
QUEUE_LOG = ("/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/"
             "queue_20261101T233607_741947.log")
PROC = "/proc"
NVIDIA_SMI = "nvidia-smi"
POLL_S, WINDOW_POLLS, POLL_SLACK_S, FRESH_S = 60, 31, 15, 600     # A7 r2 (v)(1) / r3 (b): 31 polls = 30 min; 10 min
PY = "/home/yixunhu/miniconda3/envs/flac/bin/python"                 # $PY of the command list
PYTEST_MAIN = "/home/yixunhu/miniconda3/envs/flac/lib/python3.10/site-packages/pytest/__main__.py"
CHILD_POLL_S = 30                                                 # the supervisor's poll + heartbeat during a line (<= 60 s)
# --- END REGISTERED CONFIG ---

CARDS = {"GPU0": "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU1": "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"}
EVAL_DONE = "EVAL_DONE ACEattn s43 "                          # A7 r2 (v)(1)(d), the trailing space included
TABLE_JOB = "exp30_attn_eval.sh"
Config = collections.namedtuple("Config", "reservations exp35_lock queue_log proc nvidia_smi")


class Unreadable(ValueError):
    """An input of the card rule could not be read: busy (detector) or a refusal (launcher, supervisor)."""


def registered_config():
    return Config(RESERVATIONS, EXP35_LOCK, QUEUE_LOG, PROC, NVIDIA_SMI)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def iso(t):
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


ROW = re.compile(r"\s*(\d+),\s*(GPU-[0-9A-Za-z-]+)\s*")


def gpu_map(text):
    """``{index: uuid}`` from ``nvidia-smi --query-gpu=index,uuid --format=csv,noheader``; ValueError when malformed."""
    out = {}
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = ROW.fullmatch(line)
        if not m or int(m[1]) in out or m[2] in out.values():
            raise ValueError(f"malformed or duplicate nvidia-smi index,uuid row {line!r}")
        out[int(m[1])] = m[2]
    if not out:
        raise ValueError("nvidia-smi lists no GPU")
    return out


def compute_apps(text):
    """``[(pid, uuid)]`` from ``nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader``."""
    out = []
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = ROW.fullmatch(line)
        if not m:
            raise ValueError(f"malformed nvidia-smi compute-app row {line!r}")
        out.append((int(m[1]), m[2]))
    return out


class World:
    """The real world: wall time, sleep, nvidia-smi (every failure is ``Unreadable``)."""
    QUERIES = {"map": "--query-gpu=index,uuid", "apps": "--query-compute-apps=pid,gpu_uuid"}

    def __init__(self, nvidia_smi=NVIDIA_SMI):
        self.nvidia_smi = nvidia_smi

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(max(0.0, s))

    def nvsmi(self, query):
        if query not in self.QUERIES:
            raise ValueError(f"unknown nvidia-smi query {query!r}")
        argv = [self.nvidia_smi, self.QUERIES[query], "--format=csv,noheader"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as err:
            raise Unreadable(f"nvidia-smi {argv[1]}: {err}") from err
        if r.returncode != 0:
            raise Unreadable(f"nvidia-smi {argv[1]}: rc {r.returncode}: {r.stderr.strip()[:200]}")
        return r.stdout


def reservation_path(dirpath, uuid):
    return os.path.join(dirpath, f"{uuid}.json")


def reservation_status(dirpath, uuid):
    """'absent', 'present (holder X)' or 'present (...)' for any other file -- a dangling link, a malformed file;
    ``Unreadable`` when the directory is missing or unreadable or the file cannot be read."""
    try:
        os.listdir(dirpath)
    except OSError as err:
        raise Unreadable(f"the reservation directory {dirpath} is missing or unreadable ({err})") from err
    path = reservation_path(dirpath, uuid)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return "absent"
    except OSError as err:
        raise Unreadable(f"the reservation file {path} is unreadable ({err})") from err
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return "present (a dangling link, or removed while it was read)"
    except OSError as err:
        raise Unreadable(f"the reservation file {path} is unreadable ({err})") from err
    try:
        doc = json.loads(raw)
    except ValueError:
        return "present (malformed)"
    if not isinstance(doc, dict) or not isinstance(doc.get("holder"), str):
        return "present (malformed)"
    return f"present (holder {doc['holder']})"


def read_record(path):
    """(the bytes, (device, inode)) of the file at ``path`` through ONE descriptor, never through a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st, chunks = os.fstat(fd), []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return b"".join(chunks), (st.st_dev, st.st_ino)
            chunks.append(chunk)
    finally:
        os.close(fd)


def held_reasons(held):
    """[] iff the file at ``held['path']`` IS the record we wrote: its bytes' sha256 and its (device, inode)."""
    try:
        raw, ident = read_record(held["path"])
    except OSError as err:
        return [f"our reservation {held['path']} is gone or unreadable ({err})"]
    out = [] if sha256_bytes(raw) == held["sha256"] else [f"{held['path']} is not our record (other bytes)"]
    return out + ([] if list(ident) == [held["dev"], held["inode"]] else [f"{held['path']} is not our record (another inode)"])


def lock_status(path):
    """'absent' | 'unheld' | 'held' -- the reviewed exp_35 probe: open O_RDONLY (never created), LOCK_SH | LOCK_NB,
    closed at once so no lock outlives the probe; ``Unreadable`` on any other error."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return "absent"
    except OSError as err:
        raise Unreadable(f"the exp_35 lock {path} is unreadable ({err})") from err
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return "unheld"
    except BlockingIOError:
        return "held"
    except OSError as err:
        raise Unreadable(f"the exp_35 lock {path} cannot be probed ({err})") from err
    finally:
        os.close(fd)


def procs_named(name, proc=PROC):
    """PIDs (sorted; not this process) with an argv element whose basename is exactly ``name``. A vanished entry is
    skipped; an unreadable one is ``Unreadable`` (never silently skipped)."""
    try:
        entries = os.listdir(proc)
    except OSError as err:
        raise Unreadable(f"{proc} is unreadable ({err})") from err
    out = []
    for n in entries:
        if not n.isdigit() or int(n) == os.getpid():
            continue
        try:
            with open(os.path.join(proc, n, "cmdline"), "rb") as f:
                raw = f.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as err:
            raise Unreadable(f"{proc}/{n}/cmdline is unreadable ({err})") from err
        if any(os.path.basename(a.decode(errors="replace")) == name for a in raw.split(b"\0") if a):
            out.append(int(n))
    return sorted(out)


def _stat_identity(text):
    """``(pid, start time)``: field 1 and field 22 of ``/proc/<pid>/stat`` (after the LAST ')' of the comm)."""
    return int(text.split(" ", 1)[0]), int(text[text.rindex(")") + 2:].split()[19])


def proc_identity(pid, proc=PROC):
    """The reviewed exp_35 identity of a LIVE process -- pid, start time, cwd, cmdline sha256, fd 1 / 2 -- read through
    ONE directory fd between two stat reads that must agree; ``ValueError`` when gone, unreadable or reused."""
    pid = int(pid)
    try:
        fd = os.open(os.path.join(proc, str(pid)), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as err:
        raise ValueError(f"process {pid}: its /proc identity is unreadable ({type(err).__name__}: {err})") from err
    opener = lambda path, flags: os.open(path, flags, dir_fd=fd)  # noqa: E731
    try:
        with open("stat", opener=opener) as f:
            first = _stat_identity(f.read())
        with open("cmdline", "rb", opener=opener) as f:
            cmdline = f.read()
        links = [os.readlink(p, dir_fd=fd) for p in ("cwd", "fd/1", "fd/2")]
        with open("stat", opener=opener) as f:
            second = _stat_identity(f.read())
    except (OSError, ValueError, IndexError) as err:
        raise ValueError(f"process {pid}: its /proc identity is unreadable ({type(err).__name__}: {err})") from err
    finally:
        os.close(fd)
    if not first == second == (pid, first[1]):
        raise ValueError(f"process {pid}: its /proc entry changed during the sample ({first} -> {second}): PID reuse")
    return {"pid": pid, "start_time": first[1], "cwd": links[0], "cmdline_sha256": sha256_bytes(cmdline),
            "fd1": links[1], "fd2": links[2]}


def descendants(pid, proc=PROC):
    """``pid`` and every process below it by the ppid of ``/proc/<p>/stat`` (empty when ``pid`` is not listed)."""
    parent = {}
    for name in os.listdir(proc):
        if name.isdigit():
            try:
                with open(os.path.join(proc, name, "stat")) as f:
                    text = f.read()
                parent[int(name)] = int(text[text.rindex(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    out, todo = set(), [pid] if pid in parent else []
    while todo:
        p = todo.pop()
        out.add(p)
        todo += [c for c, pp in parent.items() if pp == p and c not in out]
    return out


def table_job_status(queue_log, proc=PROC):
    """(done, reasons): his s43 table job is terminally complete iff his queue log carries a line starting with
    ``EVAL_DONE ACEattn s43 `` and no ``exp30_attn_eval.sh`` process runs; ``Unreadable`` when the log is."""
    try:
        with open(queue_log, "rb") as f:
            lines = f.read().decode(errors="replace").splitlines()
    except OSError as err:
        raise Unreadable(f"his queue log {queue_log} is unreadable ({err})") from err
    out = [] if any(ln.startswith(EVAL_DONE) for ln in lines) else [
        f"his queue log carries no {EVAL_DONE!r} line: the s43 table job is not terminally complete"]
    running = procs_named(TABLE_JOB, proc)
    if running:
        out.append(f"{TABLE_JOB} runs (pids {running}): his table job")
    return not out, out


def observe(world, cfg, card, held=None, ours=()):
    """One observation of ``card`` ('GPU0' / 'GPU1') under the card rule; ``held`` -- the supervisor's own record
    (path / sha256 / dev / inode) -- turns 'no reservation file' into 'the file IS our record'; ``ours`` -- the
    running line's own process tree -- is not a foreign compute process."""
    if card not in CARDS:
        raise ValueError(f"unknown card {card!r} (GPU0 or GPU1)")
    uuid, t, map_raw = CARDS[card], world.now(), world.nvsmi("map")
    try:
        m = gpu_map(map_raw)
    except ValueError as err:
        raise Unreadable(f"the GPU map: {err}") from err
    if sorted(m.values()) != sorted(CARDS.values()):
        raise Unreadable(f"the GPU map {m} is not this box's two registered cards")
    apps_raw = world.nvsmi("apps")
    try:
        pids = sorted(p for p, u in compute_apps(apps_raw) if u == uuid and p not in ours)
    except ValueError as err:
        raise Unreadable(f"the compute table: {err}") from err
    reasons = []
    if held is None:
        res = reservation_status(cfg.reservations, uuid)
        if res != "absent":
            reasons.append(f"a reservation file exists for {card}: {res}"
                           + (" -- an exp_35 reservation (its lane or a phase-(ii) intent)" if "holder exp35" in res else ""))
    else:
        why = held_reasons(held)
        res, reasons = ("held" if not why else "NOT OUR RECORD"), reasons + why
    if card == "GPU0":
        lk = lock_status(cfg.exp35_lock)
        rule = f"exp35 lock {'HELD' if lk == 'held' else lk}"
        reasons += ["the exp_35 supervisor lock is held: GPU 0 is its hand-over target"] if lk == "held" else []
    else:
        done, why = table_job_status(cfg.queue_log, cfg.proc)
        rule, reasons = f"his s43 table job {'terminally complete' if done else 'NOT complete'}", reasons + why
    busy = [f"compute process(es) {pids} on {card} ({uuid})"] if pids else []
    return {"t": t, "iso": iso(t), "card": card, "uuid": uuid, "index": next(i for i, u in m.items() if u == uuid),
            "gpu_map_raw": map_raw, "apps_raw": apps_raw, "apps_sha256": sha256_bytes(apps_raw.encode()),
            "pids_on_card": pids, "reservation": res, "rule": rule, "free": not (busy or reasons),
            "reasons": busy + reasons, "rule_reasons": reasons}


def detect(world, cfg, max_seconds):
    """The READ-ONLY detector (A7 r2 (v)(1), r3 (d)): both cards observed every POLL_S s; ``(0, 'EVENT CANDIDATE ...')``
    once a card was free at WINDOW_POLLS consecutive polls (GPU 0 first), ``(3, 'EVENT TIMEOUT ...')`` when the next
    poll would lie past ``max_seconds``, ``(4, 'EVENT UNREADABLE ...')`` at the first unreadable input. A poll later
    than POLL_SLACK_S restarts every window at that poll (no catch-up burst). It keeps no memory across runs."""
    start = due = world.now()
    streak = {c: [] for c in CARDS}
    while True:
        world.sleep(due - world.now())
        if world.now() - due > POLL_SLACK_S:
            streak, due = {c: [] for c in CARDS}, world.now()
        try:
            seen = {c: observe(world, cfg, c) for c in CARDS}
        except Unreadable as err:
            return 4, f"EVENT UNREADABLE at {iso(world.now())}: {err} (busy: nothing admitted)"
        for c, o in seen.items():
            streak[c] = streak[c] + [o["t"]] if o["free"] else []
        ready = [c for c in CARDS if len(streak[c]) >= WINDOW_POLLS]
        if ready:
            c = "GPU0" if "GPU0" in ready else ready[0]
            first, last = streak[c][-WINDOW_POLLS], streak[c][-1]
            return 0, (f"EVENT CANDIDATE {c} {CARDS[c]} index {seen[c]['index']}: free at {WINDOW_POLLS} consecutive "
                       f"polls {POLL_S} s apart, {iso(first)} .. {iso(last)} (read-only detector: only the launcher's "
                       "sealed capture admits a card)")
        due += POLL_S
        if due - start > max_seconds:
            counts = ", ".join(f"{c} {len(streak[c])}/{WINDOW_POLLS}" for c in CARDS)
            return 3, f"EVENT TIMEOUT at {iso(world.now())} after {max_seconds} s: no candidate (consecutive free polls {counts}); re-arm"


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_22 R2-C read-only card observation / detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("observe").add_argument("--card", required=True, choices=sorted(CARDS))
    sub.add_parser("detect").add_argument("--max-seconds", type=int, required=True)
    args = ap.parse_args(argv)
    cfg, world = registered_config(), World(NVIDIA_SMI)
    if args.cmd == "observe":
        try:
            o = observe(world, cfg, args.card)
        except Unreadable as err:
            print(json.dumps({"card": args.card, "unreadable": str(err)}))
            return 4
        print(json.dumps(o))
        return 0 if o["free"] else 1
    if not 1 <= args.max_seconds <= 7200:
        print("EVENT ARGUMENT ERROR: --max-seconds must lie in 1..7200 (the Planner re-arms every <= 2 h)")
        return 2
    rc, line = detect(world, cfg, args.max_seconds)
    print(line, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
