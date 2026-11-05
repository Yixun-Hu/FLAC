#!/usr/bin/env python3
"""exp_35 co-tenant identity (plan_orbitrir_yaw_evidence.md r3 §12 B1.1-B1.2): the approved identity tuple
of the foreign compute processes a shared lane may hold, bound MECHANICALLY through /proc -- PID, start time
(``/proc/<pid>/stat`` field 22), cwd, the sha256 of ``/proc/<pid>/cmdline``, and ``readlink /proc/<pid>/fd/1``
and ``fd/2`` (the log the process writes) -- never an asserted string.

Used by ``depth_canon_eval_run.sh`` at its lane checks (``--check``; any extra / missing / changed process
refuses the cell) and by ``exp35_supervisor.py``. Read-only: it reads /proc, writes nothing, signals nothing.

  --snapshot --gpu-uuid U --pid P [--pid Q]   print the tuple of live processes (the launch entry freezes it)
  --validate FILE                             exit 0 iff FILE is a well-formed tuple
  --check --approved FILE --gpu-uuid U --pids P,Q   exit 0 iff the observed set EQUALS the tuple (1 mismatch,
                                              2 refused: unreadable / malformed)

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r3 §12.
"""
import argparse
import hashlib
import json
import os
import re
import sys

SCHEMA_VERSION = 1
FIELDS = ("pid", "start_time", "cwd", "cmdline_sha256", "fd1", "fd2")
TUPLE_KEYS = ("schema_version", "gpu_uuid", "processes")


def _stat_identity(text):
    """``(pid, start time)`` from a ``/proc/<pid>/stat`` line: field 1 and field 22 (fields 3.. follow the
    comm, which may hold spaces and parentheses -- hence the LAST ')')."""
    return int(text.split(" ", 1)[0]), int(text[text.rindex(")") + 2:].split()[19])


def proc_identity(pid, proc="/proc"):
    """The identity of a LIVE process from ``proc``; ``ValueError`` when it is gone or unreadable, or when
    its entry changed during the sample (Codex C2 HIGH 1): every field is read through ONE directory fd of
    ``/proc/<pid>`` (bound to that process, dead or alive), between two stat reads that must agree."""
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
        raise ValueError(f"process {pid}: its /proc entry changed during the sample ((pid, start time) {first} -> "
                         f"{second}): PID reuse, no identity")
    return {"pid": pid, "start_time": first[1], "cwd": links[0],
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(), "fd1": links[1], "fd2": links[2]}


def tuple_reasons(doc):
    """Named reasons ``doc`` is not a well-formed approved tuple ([] = valid)."""
    if not isinstance(doc, dict) or list(doc) != list(TUPLE_KEYS):
        return [f"the tuple's keys are {list(doc) if isinstance(doc, dict) else doc!r}, not {list(TUPLE_KEYS)}"]
    out = [] if doc["schema_version"] == SCHEMA_VERSION and type(doc["schema_version"]) is int else [
        f"schema_version {doc['schema_version']!r} != {SCHEMA_VERSION}"]
    if not (isinstance(doc["gpu_uuid"], str) and re.fullmatch(r"GPU-[0-9A-Za-z-]+", doc["gpu_uuid"])):
        out.append(f"gpu_uuid {doc['gpu_uuid']!r} is not a GPU uuid")
    procs = doc["processes"]
    if not (isinstance(procs, list) and procs):
        return out + ["processes must be a non-empty list"]
    for p in procs:
        if not (isinstance(p, dict) and list(p) == list(FIELDS) and type(p["pid"]) is int and p["pid"] > 0
                and type(p["start_time"]) is int and all(isinstance(p[k], str) and p[k] for k in FIELDS[2:])
                and re.fullmatch(r"[0-9a-f]{64}", p["cmdline_sha256"])):
            out.append(f"malformed process entry {p!r:.160}")
    pids = [p.get("pid") for p in procs if isinstance(p, dict)]
    return out + ([f"duplicate pids {pids}"] if len(set(pids)) != len(pids) else [])


def load_tuple(path):
    """The approved tuple at ``path``; ``ValueError`` when it is unreadable or malformed."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"the approved tuple {path} is unreadable ({err})") from err
    reasons = tuple_reasons(doc)
    if reasons:
        raise ValueError(f"the approved tuple {path} is malformed: " + "; ".join(reasons))
    return doc


def check_reasons(observed_pids, gpu_uuid, approved, proc="/proc"):
    """[] iff the foreign compute processes observed on ``gpu_uuid`` are EXACTLY the approved tuple's,
    each with its identity unchanged (a reused PID has another start time / cmdline / cwd / log)."""
    out = [] if gpu_uuid == approved["gpu_uuid"] else [
        f"the lane is {gpu_uuid}, the approved tuple names {approved['gpu_uuid']}"]
    want, got = sorted(p["pid"] for p in approved["processes"]), sorted(set(observed_pids))
    if got != want or len(got) != len(observed_pids):
        out.append(f"the foreign compute processes on {gpu_uuid} are {sorted(observed_pids)}, the approved tuple "
                   f"{want} (an extra, missing or replaced process)")
    for p in approved["processes"]:
        try:
            live = proc_identity(p["pid"], proc)
        except ValueError as err:
            out.append(str(err))
            continue
        changed = [k for k in FIELDS if live[k] != p[k]]
        if changed:
            out.append(f"process {p['pid']}: {changed} differ from the approved tuple (PID reuse or another process)")
    return out


def _pids(text):
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", action="store_true")
    mode.add_argument("--validate", metavar="FILE")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--approved")
    ap.add_argument("--gpu-uuid")
    ap.add_argument("--pid", type=int, action="append", default=[])
    ap.add_argument("--pids", default="")
    ap.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)       # tests: a fake /proc tree
    args = ap.parse_args(argv)
    try:
        if args.snapshot:
            doc = {"schema_version": SCHEMA_VERSION, "gpu_uuid": args.gpu_uuid,
                   "processes": [proc_identity(p, args.proc) for p in args.pid]}
            reasons = tuple_reasons(doc)
            if reasons:
                raise ValueError("; ".join(reasons))
            print(json.dumps(doc, indent=1))
            return 0
        if args.validate:
            load_tuple(args.validate)
            print(f"approved tuple OK: {args.validate}")
            return 0
        approved, pids = load_tuple(args.approved), _pids(args.pids)
    except ValueError as err:
        print(f"IDENTITY REFUSED: {err}")
        return 2
    reasons = check_reasons(pids, args.gpu_uuid, approved, args.proc)
    print(("IDENTITY OK " if not reasons else "IDENTITY MISMATCH ") + f"{args.gpu_uuid} pids {pids}"
          + "".join(f"\n  - {r}" for r in reasons))
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
