#!/usr/bin/env python3
"""exp_37: quarantine the remains of a FAILED model step so its registered stem can be re-run from scratch (Codex code
review 2, BLOCKING 2). Never deletes: it MOVES the partial run directory, every unpublished staging directory of the
stem whose owner process is dead, and (only with --include-published) a published dump of the stem, into
``<runs>/.quarantine/<stem>.<UTC>/`` and ``<dump root>/.quarantine/<stem>.<UTC>/``, and writes a hashed record
``<runs>/.quarantine/<stem>.<UTC>.json``. A staging directory whose owner (pid + /proc start time) is still alive is
never touched (exit 2). Every move is one ``renameat2(RENAME_NOREPLACE)`` (review 3, B-01: no check-then-rename;
a filesystem without it refuses, exit 2). Standard library only. Written by the exp_37 Coder (Claude Opus 5.5, max
effort), fix round 2; the final fix round (fixed after review 3, not re-reviewed (cap)).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def canonical_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rename_noreplace(source, target) -> None:
    """renameat2(RENAME_NOREPLACE) only: an existing target is FileExistsError; a libc / filesystem without the atomic
    no-replace rename raises OSError -- never a check-then-rename fallback."""
    import ctypes
    import errno
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (OSError, AttributeError) as error:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable") from error
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:      # AT_FDCWD, RENAME_NOREPLACE
        return
    code = ctypes.get_errno()
    if code in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(target))
    raise OSError(code, os.strerror(code), str(target))


def _start_ticks(pid):
    try:
        with open(f"/proc/{int(pid)}/stat") as handle:
            return int(handle.read().rpartition(")")[2].split()[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def owner_alive(staging: Path) -> bool:
    try:
        owner = json.loads((staging / ".owner.json").read_text())
    except (OSError, ValueError):
        return False                                   # no owner record: never started writing; nobody owns it
    return _start_ticks(owner.get("pid")) is not None and _start_ticks(owner.get("pid")) == owner.get("pid_start_ticks")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--include-published", action="store_true", help="also move a published dump of the stem")
    args = parser.parse_args(argv)
    stem, runs = args.run_dir.name, args.run_dir.parent
    stagings = sorted(args.dump_root.glob(f".staging-{stem}-*")) if args.dump_root.is_dir() else []
    alive = [s for s in stagings if owner_alive(s)]
    if alive:
        print(f"REFUSED: {alive[0]} belongs to a process that is still alive", file=sys.stderr)
        return 2
    published = args.dump_root / stem
    if published.exists() and not args.include_published:
        print(f"REFUSED: {published} is a published dump; pass --include-published to quarantine it", file=sys.stderr)
        return 2
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    moves = [(args.run_dir, runs / ".quarantine" / f"{stem}.{stamp}" / "run")] if args.run_dir.exists() else []
    moves += [(s, args.dump_root / ".quarantine" / f"{stem}.{stamp}" / s.name) for s in stagings]
    if published.exists():
        moves.append((published, args.dump_root / ".quarantine" / f"{stem}.{stamp}" / "published"))
    if not moves:
        print(f"nothing to quarantine for {stem}")
        return 0
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            rename_noreplace(source, target)                  # atomic no-replace (review 3, B-01)
        except FileExistsError:
            print(f"REFUSED: {target} exists", file=sys.stderr)
            return 2
        except OSError as error:
            print(f"REFUSED: cannot move {source} -> {target} atomically ({error})", file=sys.stderr)
            return 2
    body = {"schema_version": 1, "record_type": "exp37_quarantine", "stem": stem, "at_utc": stamp,
            "moved": [{"from": str(s), "to": str(t)} for s, t in moves]}
    record = runs / ".quarantine" / f"{stem}.{stamp}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"quarantined": stem, "moved": len(moves), "record": str(record)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
