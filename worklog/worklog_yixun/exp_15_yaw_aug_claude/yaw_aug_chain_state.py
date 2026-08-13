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

`transact-submit` is the ONLY sanctioned way to advance a boundary, and it is a
single transaction: the per-boundary lock is held across the state re-read, the
scheduler query, the sbatch, and the publication of SUBMITTED. The previous
split (`intend` … then an unlocked squeue → sbatch → mark-submitted) let two
replayers hold the same token, both see an empty queue, and both submit.

Three rules make recovery safe rather than merely automatic:

  * a scheduler query that FAILS is fatal. "squeue errored" is not "no job", and
    guessing there is guarantees the double submission this file exists to
    prevent;
  * a token-matching job is only ADOPTED if it can still run — it is already
    RUNNING/COMPLETED, or its `afterok` parent has not failed. A child whose
    parent died can never be released by Slurm, so adopting it would leave the
    chain silently dead;
  * such a child is reported STRANDED with its job id and the operation refuses.
    Nothing is cancelled from here: destroying a job is an operator's decision
    made with the queue in front of them, not a helper's side effect.

State lives beside the run (save-dir), never in the launch manifest: manifests
are sha-registered at launch and must stay immutable.
"""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import subprocess
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


SUCCESSFUL = {"COMPLETED"}
LIVE = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING",
        "SUSPENDED"}
# A child in ANY of these states is dead and is NEVER adopted, regardless of the
# parent's health (final verify, finding 1: the submitter itself cancels a job
# whose manifest publication fails, so a CANCELLED child with a COMPLETED parent
# is a reachable state — adopting it records a corpse as the chain's successor).
TERMINAL_FAILED = {"CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                   "BOOT_FAIL", "DEADLINE", "REVOKED", "PREEMPTED"}


def _run(cmd, what):
    """Run a scheduler query. A FAILED query is fatal — never 'no job'."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as error:
        sys.exit(f"SCHEDULER QUERY FAILED ({what}): {error}; refusing to submit")
    if proc.returncode != 0:
        sys.exit(f"SCHEDULER QUERY FAILED ({what}, rc {proc.returncode}): "
                 f"{proc.stderr.strip() or proc.stdout.strip()}; refusing to submit")
    return proc.stdout


def find_job_by_name(name, squeue_cmd, sacct_cmd):
    """-> (jid, state) for a job with this exact name, or (None, None)."""
    out = _run(squeue_cmd + ["-h", "-n", name, "-o", "%i %T"], "squeue")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            return parts[0], parts[1].upper()
    # Not queued: it may have already finished (or the crash happened after it
    # ran). sacct is consulted, and its failure is equally fatal.
    out = _run(sacct_cmd + ["-n", "-X", "--name", name, "--format=JobID,State"], "sacct")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            return parts[0], parts[1].upper().split("+")[0]
    return None, None


def job_state(jid, sacct_cmd):
    out = _run(sacct_cmd + ["-n", "-X", "-j", str(jid), "--format=State"], "sacct")
    for line in out.splitlines():
        if line.strip():
            return line.strip().upper().split("+")[0]
    return None


def cmd_transact_submit(state, args):
    """Re-read, query, submit, publish — all under ONE held lock."""
    entry = state.boundary(args.target)
    if not entry:
        sys.exit(f"boundary {args.target} has not been audited — refusing to submit a "
                 "successor for a boundary the chain never recorded")
    if entry["state"] == "SUBMITTED":
        print(f"ALREADY_SUBMITTED {entry['next_leg_jid']}")
        return 3

    squeue_cmd = shlex.split(os.environ.get("YAW_AUG_SQUEUE_CMD", "squeue"))
    sacct_cmd = shlex.split(os.environ.get("YAW_AUG_SACCT_CMD", "sacct"))

    if not entry.get("intent_token"):
        entry["intent_token"] = f"{args.arm}-leg{args.next_target}-{uuid.uuid4().hex[:8]}"
        entry["next_target"] = args.next_target
        entry["state"] = "INTENDED"
        entry["intended_utc"] = now()
    token = entry["intent_token"]
    job_name = f"exp15-{args.arm}-leg{args.next_target}-{token}"
    entry["next_leg_job_name"] = job_name
    # The advertised recovery command is this same state-aware operation, never a
    # raw sbatch: running the wrapper twice is safe, running sbatch twice is not.
    entry["next_leg_command"] = (
        f"python3 {os.path.abspath(__file__)} --state {state.path} --arm {args.arm} "
        f"--cap {state.cap} transact-submit --target {args.target} "
        f"--next-target {args.next_target} --submitter {args.submitter} "
        f"--resume {args.resume} --leg-steps {args.leg_steps}"
        + (f" --dependency {args.dependency}" if args.dependency else ""))
    state.write()

    existing_jid, existing_state = find_job_by_name(job_name, squeue_cmd, sacct_cmd)
    if existing_jid and existing_state in TERMINAL_FAILED:
        # Never adopted, never marked SUBMITTED. The intent token is ROTATED so
        # that a re-run of this same operation searches a fresh job name, finds
        # nothing, and submits exactly once — while the corpse stays on record.
        entry.setdefault("dead_children", []).append(
            {"jid": existing_jid, "state": existing_state, "token": token,
             "detected_utc": now()})
        entry["intent_token"] = f"{args.arm}-leg{args.next_target}-{uuid.uuid4().hex[:8]}"
        entry["next_leg_job_name"] = (
            f"exp15-{args.arm}-leg{args.next_target}-{entry['intent_token']}")
        entry["next_leg_jid"] = None
        state.write()
        print(f"CHILD_DEAD {existing_jid} state={existing_state}")
        print(f"  the token-matching successor ended {existing_state}; a dead child is "
              "NEVER adopted, whatever its parent's state.")
        print("  Not marked SUBMITTED. The intent token was rotated: re-run this same "
              "transact-submit operation to submit a fresh successor exactly once.")
        return 5
    if existing_jid:
        adopt = existing_state in SUCCESSFUL or existing_state == "RUNNING"
        if not adopt:
            parent = args.dependency.split(":")[-1] if args.dependency else None
            parent_state = job_state(parent, sacct_cmd) if parent else None
            if parent_state is None or parent_state in SUCCESSFUL or parent_state in LIVE:
                adopt = True
            else:
                print(f"STRANDED {existing_jid} parent={parent} parent_state={parent_state}")
                print(f"  job {existing_jid} ({existing_state}) can never be released: its "
                      f"afterok parent ended {parent_state}. Not adopting it, not submitting "
                      "a second leg, and NOT cancelling it — that is an operator decision.")
                return 4
        entry["state"] = "SUBMITTED"
        entry["next_leg_jid"] = existing_jid
        entry["submitted_utc"] = now()
        entry["adopted"] = True
        state.write()
        print(f"ADOPTED {existing_jid} (state {existing_state}) — not resubmitting")
        return 0

    env = dict(os.environ, CHAIN="1", LEG_STEPS=str(args.leg_steps),
               CHAIN_INTENT_TOKEN=token)
    if args.dependency:
        env["CHAIN_DEPENDENCY"] = args.dependency
    cmd = ["bash", args.submitter, args.arm, "--resume", args.resume,
           "--expected-step", str(args.target)]
    print(f"submitting: {' '.join(cmd)} (token {token})")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"SUBMIT FAILED rc {proc.returncode}; boundary stays INTENDED (replaying this "
              "operation submits exactly once)")
        return 1
    jid = None
    for line in proc.stdout.splitlines():
        if "submitted" in line and "-> job" in line:
            jid = line.split()[-1]
    entry["state"] = "SUBMITTED"
    entry["next_leg_jid"] = jid or "unknown"
    entry["submitted_utc"] = now()
    state.write()
    print(f"SUBMITTED {entry['next_leg_jid']}")
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

    e = sub.add_parser("transact-submit")
    e.add_argument("--target", type=int, required=True)
    e.add_argument("--next-target", type=int, required=True)
    e.add_argument("--submitter", required=True)
    e.add_argument("--resume", required=True)
    e.add_argument("--leg-steps", type=int, default=2500)
    e.add_argument("--dependency", default=None)

    args = ap.parse_args(argv)
    with State(args.state, args.arm, args.cap) as state:
        return {"record-audit": cmd_record_audit, "intend": cmd_intend,
                "mark-submitted": cmd_mark_submitted, "status": cmd_status,
                "transact-submit": cmd_transact_submit}[args.cmd](state, args)


if __name__ == "__main__":
    sys.exit(main())
