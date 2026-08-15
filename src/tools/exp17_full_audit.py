"""Post-run completion audit for an exp_17 FULL 40k training run.

Read-only by construction: the audit is a pure function over the log text and
the checkpoint file NAMES, so it can be applied to a run this session did not
launch without touching it.

It answers one question the launching side does not: *did the run actually
finish?* "rc=0 plus a treatment banner" does not answer it — Lightning catches
``KeyboardInterrupt`` without re-raising (so an interrupted run exits 0) and the
banner is printed from ``on_fit_start`` (so it precedes step 0).

**Rev 2**, after a Codex review broke rev 1 with a single crafted input holding a
stale banner, ``Not Starting with 2 processes``, a *diagnostic quoting* the
termination marker, ``train/loss=Infinity``, and sixteen checkpoints including
``step=42500`` — which rev 1 audited clean. Rev 1 also failed OPEN on fit
health: its own happy-path fixture contained no loss line at all, yet it
reported "finite loss". The rules now:

* every marker is matched as a WHOLE LINE, never as a substring;
* the checkpoint set must be exactly the registered cadence;
* loss is PARSED and REQUIRED — absence of evidence is not evidence of health;
* the log is BOUND to the checkpoint directory it is used to bless;
* ``rc`` must be supplied — an unknown exit status is not a zero one.

Usage:
    python -m src.tools.exp17_full_audit --log <FULL.log> --ckpt-dir <dir> --rc <N>

Exit status: 0 if complete and valid, 1 if not, 2 on a usage/IO error.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

BANNER = "yaw_aug ENABLED img_w=512 seed=42"
SAVE_DIR_FLAG = "--save-dir"

ENDPOINT_STEPS = 40_000
CADENCE = 2_500
EXPECTED_STEPS = frozenset(range(CADENCE, ENDPOINT_STEPS + 1, CADENCE))
EXPECTED_RANKS = 2

# Whole lines, exactly as Lightning writes them. The backticks around the
# assignment are load-bearing: a pattern without them can never match.
TERMINATION_LINE = f"`Trainer.fit` stopped: `max_steps={ENDPOINT_STEPS}` reached."
TOPOLOGY_LINE = (
    f"All distributed processes registered. Starting with {EXPECTED_RANKS} processes"
)

# A FULL run logs thousands of these; requiring a floor is what stops "no loss
# line at all" from being read as health.
MIN_LOSS_OBSERVATIONS = 50

_LOSS = re.compile(r"train/loss=([^\s,\]}]+)")
_STEP = re.compile(r"(?:^|[^0-9a-zA-Z])step=(\d+)(?![0-9])")
_SAVE_DIR = re.compile(re.escape(SAVE_DIR_FLAG) + r"\s+(\S+)")


def _lines(log: str) -> list[str]:
    """Split on either terminator: tqdm emits \\r, ordinary prints emit \\n."""
    return [ln.strip() for ln in log.replace("\r", "\n").split("\n")]


def _finite(tok: str) -> bool:
    try:
        return math.isfinite(float(tok))
    except ValueError:
        return False


def audit_full_run(*, log: str, ckpt_names: list[str], rc: int | None,
                   save_dir: str) -> list[str]:
    """Return a list of problems; empty means the run is complete and valid."""
    problems: list[str] = []
    lines = _lines(log)
    lineset = set(lines)

    # --- exit status ------------------------------------------------------- #
    if rc is None:
        problems.append(
            "the training exit status is UNKNOWN; an unknown status is not a "
            "zero one, so this run cannot be blessed"
        )
    elif rc != 0:
        problems.append(f"training exited rc={rc}")

    # --- the log must belong to the directory it is blessing ---------------- #
    logged = {m.group(1).rstrip("/") for ln in lines for m in [_SAVE_DIR.search(ln)] if m}
    if not logged:
        problems.append(
            f"the log never records a {SAVE_DIR_FLAG}, so it cannot be bound to a "
            f"checkpoint directory; refusing to let an unidentified log bless one"
        )
    elif not any(save_dir.rstrip("/").endswith(d) or d.endswith(save_dir.rstrip("/"))
                 for d in logged):
        problems.append(
            f"this log does not belong to the audited directory: it records "
            f"{SAVE_DIR_FLAG} {sorted(logged)} but is being used to bless {save_dir!r}"
        )

    # --- did it finish? the check the launching side lacks ------------------ #
    if TERMINATION_LINE not in lineset:
        problems.append(
            f"the run did not reach its registered endpoint of {ENDPOINT_STEPS} "
            f"steps: Lightning's termination line is absent as a whole line "
            f"(an interrupted run still exits 0)"
        )

    # --- the deliverable: exactly the registered cadence -------------------- #
    steps = {int(m.group(1)) for n in ckpt_names if (m := _STEP.search(str(n)))}
    if missing := sorted(EXPECTED_STEPS - steps):
        problems.append(
            f"missing checkpoints at steps {missing} (expected every {CADENCE} "
            f"up to {ENDPOINT_STEPS}); the trajectory has gaps"
        )
    if extra := sorted(steps - EXPECTED_STEPS):
        problems.append(
            f"unexpected off-cadence checkpoints at steps {extra}: this directory "
            f"does not hold exactly the registered series"
        )

    # --- the treatment was live (whole line, not a paraphrase) -------------- #
    if BANNER not in lineset:
        problems.append(
            f"the treatment banner is absent as a whole line: this run may have "
            f"trained WITHOUT yaw augmentation"
        )

    # --- topology: one rank silently halves the BatchNorm batch ------------- #
    if TOPOLOGY_LINE not in lineset:
        problems.append(
            f"no whole-line evidence of {EXPECTED_RANKS} distributed processes; "
            f"BN=64 requires both ranks and accumulation cannot substitute"
        )

    # --- fit health: parsed, and REQUIRED ----------------------------------- #
    values = [m.group(1) for ln in lines for m in _LOSS.finditer(ln)]
    if len(values) < MIN_LOSS_OBSERVATIONS:
        problems.append(
            f"only {len(values)} loss observations in the log (need at least "
            f"{MIN_LOSS_OBSERVATIONS}); absence of loss evidence is not evidence "
            f"of a healthy fit"
        )
    if bad := [v for v in values if not _finite(v)]:
        problems.append(
            f"non-finite training loss observed {len(bad)} time(s), first {bad[0]!r}"
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp_17 FULL completion audit")
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--rc", type=int, required=True,
                    help="exit status train.py returned; required — an unknown "
                         "status is not a zero one")
    args = ap.parse_args(argv)

    if not args.log.is_file():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2
    if not args.ckpt_dir.is_dir():
        print(f"checkpoint dir not found: {args.ckpt_dir}", file=sys.stderr)
        return 2

    # Only regular, non-empty files count: a directory, a dangling symlink, or a
    # zero-byte file left by a crashed save is not a checkpoint.
    names = [p.name for p in args.ckpt_dir.rglob("*.ckpt")
             if p.is_file() and p.stat().st_size > 0]
    problems = audit_full_run(
        log=args.log.read_text(errors="ignore"), ckpt_names=names, rc=args.rc,
        save_dir=str(args.ckpt_dir),
    )

    print(f"exp_17 FULL completion audit — {args.log}")
    print(f"  usable checkpoints found: {len(names)}")
    if problems:
        print("VERDICT: INCOMPLETE / INVALID")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VERDICT: COMPLETE — endpoint reached, {len(EXPECTED_STEPS)} cadence "
          f"checkpoints, treatment banner present, {EXPECTED_RANKS} ranks, "
          f"all logged losses finite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
