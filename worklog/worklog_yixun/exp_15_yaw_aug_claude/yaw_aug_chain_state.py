#!/usr/bin/env python
"""exp_15 — the chain's submission state machine (chain review, finding 4).

A self-chaining run has exactly one dangerous failure mode: submitting the same
boundary twice. Two jobs training 2500 -> 5000 into one run directory is not a
recoverable mess, and the previous implementation invited it — a manual replay
after a crash between `sbatch` and the status write would have done it.

So the successor is not "submitted"; a BOUNDARY is advanced through a state
machine, under its own flock, in a file the job owns:

    AUDITED    the leg's boundary checkpoint passed the completion audit
    INTENDED   the exact next-leg command is recorded, with a unique intent token
    SUBMITTED  the successor exists; its job id is recorded

`intend` is the idempotency point. Called on an already-SUBMITTED boundary it
reports the existing job id and exits 3, so the caller skips submission instead
of issuing a second one. Called on an INTENDED boundary it returns the SAME
token, so a replay re-uses the identity that may already be in the queue — which
is what makes `--recover` able to find it.

State lives beside the run (save-dir), never in the launch manifest: manifests
are sha-registered at launch and must stay immutable.
"""
import argparse
import datetime
import fcntl
import json
import os
import sys
import uuid


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


class State:
    def __init__(self, path, arm, cap):
        self.path = path
        self.lock_path = path + ".lock"
        self.arm = arm
        self.cap = cap
        self.lock_fd = None
        self.data = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self.lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)      # blocks: correctness over speed
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                self.data = json.loads(fh.read())
        else:
            self.data = {"_meta": {"experiment": "exp_15", "arm": self.arm, "cap": self.cap},
                         "boundaries": {}}
        return self

    def __exit__(self, *exc):
        os.close(self.lock_fd)
        return False

    def boundary(self, target):
        return self.data["boundaries"].get(str(target))

    def write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def cmd_record_audit(state, args):
    existing = state.boundary(args.target)
    if existing:
        if existing.get("ckpt_sha256") != args.ckpt_sha256:
            sys.exit(f"boundary {args.target} is already recorded with a different "
                     f"checkpoint ({existing.get('ckpt_sha256')}) — refusing to overwrite")
        print(f"boundary {args.target} already AUDITED (state {existing['state']})")
        return 0
    state.data["boundaries"][str(args.target)] = {
        "target_step": args.target, "state": "AUDITED",
        "ckpt_sha256": args.ckpt_sha256, "ckpt_path": args.ckpt_path,
        "parent_step": args.parent_step, "parent_ckpt_sha256": args.parent_ckpt_sha256,
        "job": args.job, "audited_utc": now(),
        "intent_token": None, "next_target": None, "next_leg_command": None,
        "next_leg_jid": None, "intended_utc": None, "submitted_utc": None,
    }
    state.write()
    print(f"boundary {args.target} AUDITED")
    return 0


def cmd_intend(state, args):
    entry = state.boundary(args.target)
    if not entry:
        sys.exit(f"boundary {args.target} has not been audited — refusing to intend a "
                 "successor for a boundary the chain never recorded")
    if entry["state"] == "SUBMITTED":
        print(f"ALREADY_SUBMITTED {entry['next_leg_jid']}")
        return 3                                   # caller must NOT submit again
    if entry["state"] == "INTENDED" and entry.get("intent_token"):
        print(f"INTENT {entry['intent_token']}")   # replay: same identity
        return 0
    entry["intent_token"] = f"{args.arm}-leg{args.next_target}-{uuid.uuid4().hex[:8]}"
    entry["next_target"] = args.next_target
    entry["next_leg_command"] = args.command
    entry["state"] = "INTENDED"
    entry["intended_utc"] = now()
    state.write()
    print(f"INTENT {entry['intent_token']}")
    return 0


def cmd_mark_submitted(state, args):
    entry = state.boundary(args.target)
    if not entry:
        sys.exit(f"boundary {args.target} is unknown")
    if entry["state"] == "SUBMITTED":
        if entry.get("next_leg_jid") != args.jid:
            sys.exit(f"boundary {args.target} is already SUBMITTED as job "
                     f"{entry.get('next_leg_jid')}, not {args.jid}")
        print(f"boundary {args.target} already SUBMITTED as {args.jid}")
        return 0
    if entry["state"] != "INTENDED":
        sys.exit(f"boundary {args.target} is {entry['state']}, not INTENDED")
    entry["state"] = "SUBMITTED"
    entry["next_leg_jid"] = args.jid
    entry["submitted_utc"] = now()
    state.write()
    print(f"boundary {args.target} SUBMITTED as job {args.jid}")
    return 0


def cmd_status(state, args):
    entry = state.boundary(args.target)
    if not entry:
        print("UNKNOWN")
        return 1
    print(f"{entry['state']} token={entry.get('intent_token')} jid={entry.get('next_leg_jid')} "
          f"next_target={entry.get('next_target')}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cap", type=int, default=40000)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("record-audit")
    a.add_argument("--target", type=int, required=True)
    a.add_argument("--ckpt-sha256", required=True)
    a.add_argument("--ckpt-path", required=True)
    a.add_argument("--parent-step", type=int, required=True)
    a.add_argument("--parent-ckpt-sha256", default=None)
    a.add_argument("--job", default=None)

    b = sub.add_parser("intend")
    b.add_argument("--target", type=int, required=True)
    b.add_argument("--next-target", type=int, required=True)
    b.add_argument("--command", required=True)

    c = sub.add_parser("mark-submitted")
    c.add_argument("--target", type=int, required=True)
    c.add_argument("--jid", required=True)

    d = sub.add_parser("status")
    d.add_argument("--target", type=int, required=True)

    args = ap.parse_args(argv)
    with State(args.state, args.arm, args.cap) as state:
        return {"record-audit": cmd_record_audit, "intend": cmd_intend,
                "mark-submitted": cmd_mark_submitted, "status": cmd_status}[args.cmd](state, args)


if __name__ == "__main__":
    sys.exit(main())
