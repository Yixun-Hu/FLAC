#!/usr/bin/env python3
"""exp_37 runbook step receipts (Codex code review 2, HIGH 8). Standard library only.

    write --path P --step S --phase PH --status ok|skipped --head H --command C --log L --started T --finished T
    check --path P --step S [--status ok|skipped|any] [--head H] [--max-age-hours X]

``write`` creates the hashed receipt exclusively (write-once; exit 2 if it exists). ``check`` exits 0 only for an
intact receipt of THIS step with the required status, HEAD and age (from its ``finished`` time); otherwise 2.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys


def canonical_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(args) -> int:
    body = {"schema_version": 1, "record_type": "exp37_runbook_receipt", "step": args.step, "phase": args.phase,
            "status": args.status, "head": args.head, "command": args.command, "log": args.log,
            "started_utc": args.started, "finished_utc": args.finished}
    os.makedirs(os.path.dirname(os.path.abspath(args.path)), exist_ok=True)
    try:
        fd = os.open(args.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        print(f"REFUSED: {args.path} exists (receipts are write-once)", file=sys.stderr)
        return 2
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def check(args) -> int:
    try:
        with open(args.path) as handle:
            payload = json.load(handle)
        body = {k: v for k, v in payload.items() if k != "sha256"}
        finished = datetime.datetime.fromisoformat(payload["finished_utc"])
    except (OSError, ValueError, KeyError) as error:
        print(f"REFUSED: no readable receipt at {args.path} ({error})", file=sys.stderr)
        return 2
    reasons = []
    if payload.get("sha256") != canonical_sha256(body) or payload.get("record_type") != "exp37_runbook_receipt":
        reasons.append("the receipt fails its own sha256 / record type")
    if payload.get("step") != args.step:
        reasons.append(f"the receipt is step {payload.get('step')}'s, not {args.step}'s")
    if args.status != "any" and payload.get("status") != args.status:
        reasons.append(f"status {payload.get('status')} is not {args.status}")
    if args.head and payload.get("head") != args.head:
        reasons.append(f"HEAD {payload.get('head')} is not {args.head}")
    if args.max_age_hours is not None:
        age = datetime.datetime.now(datetime.timezone.utc) - finished
        if age > datetime.timedelta(hours=args.max_age_hours):
            reasons.append(f"the receipt is {age} old (limit {args.max_age_hours} h)")
    if reasons:
        print(f"REFUSED: {args.path}: {'; '.join(reasons)}", file=sys.stderr)
        return 2
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("write")
    for name in ("path", "step", "phase", "head", "command", "log", "started", "finished"):
        w.add_argument(f"--{name}", required=True)
    w.add_argument("--status", choices=("ok", "skipped"), required=True)
    c = sub.add_parser("check")
    c.add_argument("--path", required=True)
    c.add_argument("--step", required=True)
    c.add_argument("--status", choices=("ok", "skipped", "any"), default="any")
    c.add_argument("--head")
    c.add_argument("--max-age-hours", type=float)
    args = parser.parse_args(argv)
    return write(args) if args.mode == "write" else check(args)


if __name__ == "__main__":
    sys.exit(main())
