"""Post-run completion audit for an exp_17 FULL 40k training run.

Read-only by construction: the audit is a pure function over the log text and
the checkpoint file NAMES, so it can be applied to a run this session did not
launch without touching it. The CLI below only reads.

It answers one question the launching side does not: *did the run actually
finish?* "rc=0 plus a treatment banner" does not answer it — Lightning catches
``KeyboardInterrupt`` without re-raising (so an interrupted run exits 0) and the
banner is printed from ``on_fit_start`` (so it precedes step 0). See
``src/tests/test_exp17_full_completion_audit.py`` for the reasoning and the
literal-text regression pins.

Usage:
    python -m src.tools.exp17_full_audit --log <FULL.log> --ckpt-dir <save-dir> [--rc N]

Exit status: 0 if the run is complete and valid, 1 otherwise (problems printed).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The exact whole line src/training/diffusion.py prints when the treatment is on.
BANNER = "yaw_aug ENABLED img_w=512 seed=42"

ENDPOINT_STEPS = 40_000
CADENCE = 2_500
EXPECTED_CKPTS = ENDPOINT_STEPS // CADENCE          # 16
EXPECTED_RANKS = 2                                  # BN=64 needs both

# Lightning writes the assignment inside backticks; matching without them can
# never succeed. Pinned by test_the_endpoint_marker_is_matched_with_its_backticks.
TERMINATION_MARKER = f"stopped: `max_steps={ENDPOINT_STEPS}` reached"

_NONFINITE = re.compile(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", re.IGNORECASE)


def _lines(log: str) -> list[str]:
    """Split on either terminator: tqdm emits \\r, ordinary prints emit \\n."""
    return log.replace("\r", "\n").split("\n")


def audit_full_run(*, log: str, ckpt_names: list[str], rc: int) -> list[str]:
    """Return a list of problems; empty means the run is complete and valid.

    ``ckpt_names`` are basenames (or paths); only the ``step=<N>`` part is read.
    """
    problems: list[str] = []
    lines = _lines(log)

    if rc != 0:
        problems.append(f"training exited rc={rc}")

    # 1. Did it finish? This is the check the launching side lacks.
    if TERMINATION_MARKER not in log.replace("\r", "\n"):
        problems.append(
            f"the run did not reach its registered endpoint: Lightning's "
            f"'`Trainer.fit` stopped: `max_steps={ENDPOINT_STEPS}` reached.' marker "
            f"is absent (an interrupted run still exits 0)"
        )

    # 2. The deliverable itself.
    steps = {int(m.group(1))
             for n in ckpt_names
             if (m := re.search(r"step=(\d+)", str(n)))}
    if ENDPOINT_STEPS not in steps:
        problems.append(
            f"the endpoint checkpoint (*step={ENDPOINT_STEPS}.ckpt) is missing; "
            f"found steps {sorted(steps) or 'none'}"
        )
    if len(steps) != EXPECTED_CKPTS:
        problems.append(
            f"expected {EXPECTED_CKPTS} checkpoints at cadence {CADENCE}, found "
            f"{len(steps)}; the trajectory has gaps"
        )

    # 3. The treatment was live. Whole-line match: a preflight paraphrase that
    #    merely contains these words must not satisfy it.
    if BANNER not in [ln.strip() for ln in lines]:
        problems.append(
            f"the treatment banner '{BANNER}' is absent as a whole line: this run "
            f"may have trained WITHOUT yaw augmentation"
        )

    # 4. Topology: one rank silently halves the BatchNorm batch.
    if not any(f"Starting with {EXPECTED_RANKS} processes" in ln for ln in lines):
        problems.append(
            f"no evidence of {EXPECTED_RANKS} distributed processes; BN=64 requires "
            f"both ranks and accumulation cannot substitute"
        )

    # 5. Fit health. Restricted to loss-bearing lines so that words like
    #    "infer" or a path containing "nan" cannot trip it.
    for ln in lines:
        if "loss" in ln.lower() and _NONFINITE.search(ln):
            problems.append(f"non-finite loss in the log: {ln.strip()[:120]}")
            break

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--rc", type=int, default=0,
                    help="exit status train.py returned, if known")
    args = ap.parse_args(argv)

    if not args.log.is_file():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2
    if not args.ckpt_dir.is_dir():
        print(f"checkpoint dir not found: {args.ckpt_dir}", file=sys.stderr)
        return 2

    names = [p.name for p in args.ckpt_dir.rglob("*.ckpt")]
    problems = audit_full_run(
        log=args.log.read_text(errors="ignore"), ckpt_names=names, rc=args.rc
    )

    print(f"exp_17 FULL completion audit — {args.log}")
    print(f"  checkpoints found: {len(names)}")
    if problems:
        print("VERDICT: INCOMPLETE / INVALID")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VERDICT: COMPLETE — {ENDPOINT_STEPS} steps reached, "
          f"{EXPECTED_CKPTS} checkpoints, treatment banner present, "
          f"{EXPECTED_RANKS} ranks, finite loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
