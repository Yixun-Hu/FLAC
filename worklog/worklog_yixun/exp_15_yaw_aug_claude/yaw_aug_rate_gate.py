#!/usr/bin/env python
"""exp_15 — the waived smoke's POST-HOC rate gate (chain review, finding 5).

Yixun waived the smoke's `rate_at_least_0.9x_VANL` check on 2026-08-12 because
PL's displayed it/s is a running average from process start, so no 30-step smoke
can reach a whole-run floor — VANL would have failed its own gate. The waiver was
granted **with a condition**: windowed rates measured on the production run must
clear 0.9x VANL's own windows,

    steps 100 -> 300   >= 0.849 it/s   (VANL: 200 steps / 212 s = 0.943)
    steps 300 -> 1000  >= 0.843 it/s   (VANL: 700 steps / 747 s = 0.937)

and a breach stops the run. Every chain leg cited that waiver; nothing evaluated
it. This does.

It reads the leg's own training log, reconstructs (step, elapsed) from PL's
progress bar, and computes each window's rate. It is INITIAL-only by design: the
windows live inside the first leg (with LEG_STEPS=2500 both endpoints are), and
after a resume the bar's in-epoch counter no longer equals global_step.

Exit codes: 0 PASS, 1 BREACH, 2 INSUFFICIENT DATA. Insufficient data is NOT a
pass — an unmeasurable run is exactly the case the waiver's condition exists for.
"""
import argparse
import json
import os
import re
import sys

# PL's TQDM bar: "Epoch 0:   7%|▋   | 300/4550 [05:12<1:13:44,  1.04s/it]".
# Elapsed is the field before '<' and may be MM:SS or H:MM:SS.
BAR = re.compile(r"(\d+)/(\d+)\s*\[(\d+):(\d+)(?::(\d+))?<")


def parse_progress(text):
    """-> {step: elapsed_seconds}, keeping the FIRST sighting of each step."""
    seen = {}
    for match in BAR.finditer(text):
        step = int(match.group(1))
        a, b, c = match.group(3), match.group(4), match.group(5)
        elapsed = (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))
        seen.setdefault(step, elapsed)
    return seen


def window_rate(samples, lo, hi):
    """Rate over [lo, hi] using the nearest sighted steps at or beyond each end."""
    if not samples:
        return None, None
    lo_candidates = [s for s in samples if s >= lo]
    hi_candidates = [s for s in samples if s >= hi]
    if not lo_candidates or not hi_candidates:
        return None, None
    s_lo, s_hi = min(lo_candidates), min(hi_candidates)
    if s_hi <= s_lo:
        return None, None
    dt = samples[s_hi] - samples[s_lo]
    if dt <= 0:
        return None, (s_lo, s_hi)
    return (s_hi - s_lo) / dt, (s_lo, s_hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leg-target", type=int, required=True)
    ap.add_argument("--job", default=None)
    ap.add_argument("--window1", nargs=2, type=int, default=[100, 300])
    ap.add_argument("--floor1", type=float, default=0.849)
    ap.add_argument("--window2", nargs=2, type=int, default=[300, 1000])
    ap.add_argument("--floor2", type=float, default=0.843)
    args = ap.parse_args(argv)

    text = open(args.log, errors="replace").read() if os.path.exists(args.log) else ""
    samples = parse_progress(text)

    windows = []
    for (lo, hi), floor in ((args.window1, args.floor1), (args.window2, args.floor2)):
        applicable = hi <= args.leg_target
        rate, used = window_rate(samples, lo, hi) if applicable else (None, None)
        windows.append({
            "window": [lo, hi], "floor": floor, "applicable_to_this_leg": applicable,
            "steps_used": list(used) if used else None, "rate_steps_per_second": rate,
            "verdict": ("SKIPPED (window ends past this leg)" if not applicable
                        else "INSUFFICIENT DATA" if rate is None
                        else "PASS" if rate >= floor else "BREACH"),
        })

    measured = [w for w in windows if w["applicable_to_this_leg"]]
    if not measured:
        verdict, code = "INSUFFICIENT DATA", 2
    elif any(w["verdict"] == "INSUFFICIENT DATA" for w in measured):
        verdict, code = "INSUFFICIENT DATA", 2
    elif any(w["verdict"] == "BREACH" for w in measured):
        verdict, code = "BREACH", 1
    else:
        verdict, code = "PASS", 0

    record = {
        "_meta": {"experiment": "exp_15", "kind": "post-hoc windowed rate gate",
                  "authority": "worklog 2026-08-12T16:05:00-04:00 (Yixun waiver condition)",
                  "job": args.job, "leg_target": args.leg_target,
                  "progress_samples": len(samples)},
        "windows": windows,
        "verdict": verdict,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)

    print(f"rate gate: {verdict} -> {args.out}")
    for w in windows:
        rate = "n/a" if w["rate_steps_per_second"] is None else f"{w['rate_steps_per_second']:.3f}"
        print(f"  {w['window'][0]}->{w['window'][1]}: {rate} it/s (floor {w['floor']}) "
              f"[{w['verdict']}]")
    if code == 1:
        print("  BREACH: the waiver's condition is not met — the chain must stop")
    elif code == 2:
        print("  INSUFFICIENT DATA: the windows could not be measured; an unmeasurable")
        print("  run cannot satisfy a condition, so this refuses rather than passes")
    return code


if __name__ == "__main__":
    sys.exit(main())
