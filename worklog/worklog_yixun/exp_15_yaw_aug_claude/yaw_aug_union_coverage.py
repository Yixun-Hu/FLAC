#!/usr/bin/env python
"""exp_15 — machine-check that the two guardtest environments really do cover
every case between them (chain review, finding 8).

The previous round claimed "union covers everything" and was wrong: one case was
skipped in BOTH transcripts, so it was green in neither. A claim about coverage
should be computed, not asserted, so the suite now emits a per-case ledger and
this reconciles them:

  * every case seen in any ledger must have at least one PASS somewhere;
  * a case that FAILs anywhere is a failure, even if it passes elsewhere;
  * a case skipped in every ledger is reported as UNCOVERED and fails the check.

Usage:  yaw_aug_union_coverage.py <ledger> [<ledger> ...]
"""
import collections
import sys


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        sys.exit("usage: yaw_aug_union_coverage.py <ledger> [<ledger> ...]")

    status = collections.defaultdict(set)
    for path in paths:
        seen = 0
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                status[parts[1]].add(parts[0])
                seen += 1
        print(f"{path}: {seen} case records")

    failed = sorted(name for name, states in status.items() if "FAIL" in states)
    uncovered = sorted(name for name, states in status.items() if "PASS" not in states)
    covered = sorted(name for name, states in status.items() if "PASS" in states)

    print(f"\ncases seen        : {len(status)}")
    print(f"covered (>=1 PASS): {len(covered)}")
    print(f"failed anywhere   : {len(failed)}")
    print(f"never passed      : {len(uncovered)}")
    for name in failed:
        print(f"  FAILED    {name}")
    for name in uncovered:
        if name not in failed:
            print(f"  UNCOVERED {name}  (skipped in every environment)")

    if failed or uncovered:
        print("\nUNION COVERAGE: NOT SATISFIED")
        return 1
    print("\nUNION COVERAGE: every case passed in at least one environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
