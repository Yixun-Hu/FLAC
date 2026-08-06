#!/usr/bin/env python3
"""exp_11 arm-launch outcome classification (round-3 review B5).

The launcher's exit taxonomy lives here, as a pure function of (torchrun status,
tee status, the two log copies), so every class is unit-testable from fake logs
instead of only from a multi-day run:

    6  WORLD-SIZE      Lightning never reported the expected rank count, or
                       reported a different one  (the run is not the recipe)
    3  OOM             nonzero exit whose log carries a CUDA OOM
    4  NO-MARKER       exit 0 without Lightning's exact completion literal
                       (an early stop is not a finished budget)
    7  LOG-PROVENANCE  tee failed, a log copy is missing, or the two copies are
                       not byte-identical (the durable record is not durable)
    rc otherwise       the raw torchrun status, preserved

Precedence is world-size > OOM > no-marker > log-provenance > raw, so the most
specific statement about WHY the run is unusable wins. Usage:

    fa_orbit_classify.py --rc 0 --tee-rc 0 --ngpu 4 --maxsteps 40000 \\
        --log <path> --log-copy <path>
"""
import argparse
import filecmp
import os
import re
import sys

# Literals verified in the installed PL 2.1.0:
#   lightning_fabric/utilities/distributed.py:296
#   pytorch_lightning/loops/fit_loop.py:167
WORLD_RE = re.compile(r"All distributed processes registered\. Starting with (\d+) processes")
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError")

EXIT_OOM, EXIT_NO_MARKER, EXIT_WORLD_SIZE, EXIT_LOG = 3, 4, 6, 7


def _read(path):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def classify(rc, tee_rc, ngpu, maxsteps, log_path, log_copy_path):
    """Return ``(exit_code, [messages])``; never raises on missing files."""
    msgs = []
    text = _read(log_path)
    if text is None:
        return EXIT_LOG, [f"LOG-PROVENANCE: training log {log_path} is unreadable"]

    world = [int(m) for m in WORLD_RE.findall(text)]
    done = f"`Trainer.fit` stopped: `max_steps={maxsteps}` reached." in text
    oom = bool(OOM_RE.search(text))

    log_bad = []
    if tee_rc:
        log_bad.append(f"tee exited {tee_rc}")
    for p in (log_path, log_copy_path):
        if not os.path.isfile(p):
            log_bad.append(f"missing log copy {p}")
    if not log_bad and not filecmp.cmp(log_path, log_copy_path, shallow=False):
        log_bad.append("the two log copies are not byte-identical")

    if not world:
        msgs.append(f"WORLD-SIZE: Lightning never reported rank registration "
                    f"(expected {ngpu} processes) — this run did not train the recipe")
        return EXIT_WORLD_SIZE, msgs
    if any(w != ngpu for w in world):
        msgs.append(f"WORLD-SIZE: reported {world}, expected {ngpu} processes")
        return EXIT_WORLD_SIZE, msgs
    msgs.append(f"world size OK: {ngpu} processes registered")

    if rc != 0 and oom:
        msgs.append(f"OOM: torchrun exited {rc} with a CUDA out-of-memory in the log")
        return EXIT_OOM, msgs
    if rc == 0 and not done:
        msgs.append(f"NO-MARKER: exit 0 without `max_steps={maxsteps}` reached — "
                    "an early stop, not a finished budget")
        return EXIT_NO_MARKER, msgs
    if log_bad:
        msgs.append("LOG-PROVENANCE: " + "; ".join(log_bad))
        return EXIT_LOG, msgs
    if rc != 0:
        msgs.append(f"RUNTIME: torchrun exited {rc} (no OOM signature)")
        return rc, msgs
    msgs.append(f"COMPLETE: {maxsteps} steps reached, dual logs verified identical")
    return 0, msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description="classify an exp_11 arm run")
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--tee-rc", type=int, default=0)
    ap.add_argument("--ngpu", type=int, required=True)
    ap.add_argument("--maxsteps", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--log-copy", required=True)
    args = ap.parse_args(argv)
    code, msgs = classify(args.rc, args.tee_rc, args.ngpu, args.maxsteps,
                          args.log, args.log_copy)
    for m in msgs:
        print(m)
    return code


if __name__ == "__main__":
    sys.exit(main())
