#!/usr/bin/env python3
"""exp_35 training guard (plan_orbitrir_yaw_evidence.md r2 §7 "Guard", r3 §12 B2, r4 §13 NIT 1): a READ-ONLY parser
of Yixun's s43 training log. It reads that one file; it writes nothing and signals nothing.

The log is PyTorch Lightning's tqdm bar, ``Epoch E:  P%|bar| n/T [elapsed<remaining, rate, postfix]``, refreshed
with ``\\r`` (``\\n`` works too). For each step n of an epoch the FIRST line that shows it gives the time step n
completed (a postfix refresh repeats the line; the epoch-end validation re-prints the last step with a later elapsed);
the elapsed field restarts with each epoch, so the validation between epochs lies in no window (PL's bar reset after
it, "Epoch E: 0/T [00:00<" under the old label once step T was seen, is skipped). Windows align to the
epoch's step counter -- [0, 100], [100, 200], ... and [0, 500], ... -- and a window is complete once a step at or past
its end was seen (an epoch's tail beyond its last full window is in none). Only the bytes up to the last line
terminator are parsed (a line being written is not).

  STOP (exit 1), judged over the completed windows that END after the baseline step (our first spawn; before a
  baseline nothing is judged), thresholds exact and strict: any 500-step window > 9.712 s/step; any 100-step window
  > 9.950 s/step; the mean s/step since the baseline (the within-epoch elapsed of the steps after it) > 8.521.
  HARD STOP (exit 2): against the persisted anchor the log's inode changed or its size shrank, or the 4 KiB before
  the anchor offset changed (a copy-truncate-and-regrow); a step went backwards within an epoch, an epoch number
  went backwards, or the bar total changed (a restart or another run in the same file); no training-bar line.
  REFUSED (exit 3): the log or the anchor file is unreadable, or the arguments are bad.   OK (exit 0): none of these.

  --log PATH [--baseline-step N] [--anchor FILE]   prints the observation (JSON, with the NEW anchor) and exits 0-3

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import argparse
import hashlib
import json
import os
import re
import sys

W500, W100, MEAN = 9.712, 9.950, 8.521        # §7: s/step, strict >
ANCHOR_BYTES = 4096                             # §13 NIT 1: the sha256 of the last 4 KiB parsed
EXIT_OK, EXIT_STOP, EXIT_HARD, EXIT_REFUSED = 0, 1, 2, 3
MAX_STEPS = 40000                               # the s43 training's max_steps
LINE = re.compile(r"Epoch (\d+):\s+\d+%\|[^|\r\n]*\|\s*(\d+)/(\d+) \[(\d+(?::\d\d){1,2})<")


def _seconds(text):
    parts = [int(x) for x in text.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse(text):
    """``({epoch: {"total": T, "steps": {n: elapsed s}}}, errors)`` -- the first sighting of each step; a repeat is
    accepted only for the latest step (a refresh, the post-validation re-print)."""
    epochs, errors, last = {}, [], None
    for m in LINE.finditer(text):
        e, n, total, el = int(m[1]), int(m[2]), int(m[3]), _seconds(m[4])
        if last is not None and e < last:
            return epochs, [f"epoch {e} after epoch {last}: a restart in the same log"]
        first = epochs[min(epochs)]["total"] if epochs else total
        if total != first:
            return epochs, [f"epoch {e}: the bar total changed ({first} -> {total}): another run in the same log"]
        ep = epochs.setdefault(e, {"total": total, "steps": {}})
        top = max(ep["steps"]) if ep["steps"] else None
        if (n, el, top) == (0, 0, total):
            continue        # PL resets the bar under the OLD label after the validation (the real s43 log)
        if top is not None and (n < top or (n > top and el < ep["steps"][top])):
            return epochs, [f"epoch {e}: step {n} at {el} s after step {top} at {ep['steps'][top]} s: the bar went backwards"]
        ep["steps"].setdefault(n, el)
        last = e
    return epochs, errors


def windows(epochs, size):
    """Completed ``size``-step windows ``(epoch, start, global end, s/step)``: from the last sample at or before the
    start to the first sample at or after the end, within one epoch."""
    out = []
    for e, ep in sorted(epochs.items()):
        ns, T = sorted(ep["steps"]), ep["total"]
        for a in range(0, T - size + 1, size):
            lo, hi = [n for n in ns if n <= a], [n for n in ns if n >= a + size]
            if lo and hi:
                out.append((e, a, e * T + a + size, (ep["steps"][hi[0]] - ep["steps"][lo[-1]]) / (hi[0] - lo[-1])))
    return out


def mean_since(epochs, baseline):
    """``(mean s/step, steps)`` of the steps completed after global step ``baseline`` (within-epoch elapsed only)."""
    t = steps = 0
    for e, ep in epochs.items():
        ns = [n for n in sorted(ep["steps"]) if e * ep["total"] + n >= baseline]
        if len(ns) > 1:
            t, steps = t + ep["steps"][ns[-1]] - ep["steps"][ns[0]], steps + ns[-1] - ns[0]
    return (t / steps if steps else None), steps


def judge(epochs, baseline):
    """(STOP reasons, the window summary): thresholds over the windows ending after ``baseline`` and the mean."""
    reasons, summary = [], {}
    for size, limit in ((500, W500), (100, W100)):
        got = [w for w in windows(epochs, size) if baseline is not None and w[2] > baseline]
        reasons += [f"the {size}-step window epoch {e} steps {a}-{a + size} ran {r:.3f} s/step > {limit}"
                    for e, a, _, r in got if r > limit]
        worst = max(got, key=lambda w: w[3], default=None)
        summary[str(size)] = {"judged": len(got), "limit": limit, "worst": None if worst is None else {
            "epoch": worst[0], "start": worst[1], "s_per_step": worst[3]}}
    mean, n = mean_since(epochs, baseline) if baseline is not None else (None, 0)
    reasons += [f"the mean since the baseline step {baseline} ran {mean:.3f} s/step over {n} steps > {MEAN}"] if (
        mean is not None and mean > MEAN) else []
    return reasons, summary, {"s_per_step": mean, "steps": n, "limit": MEAN}


def read_log(path):
    """``(stat, bytes, end)``: ONE fd, ``st_size`` bytes from offset 0, ``end`` = just past the last terminator."""
    fd = os.open(path, os.O_RDONLY)
    try:
        st, chunks, off = os.fstat(fd), [], 0
        while off < st.st_size:
            b = os.pread(fd, st.st_size - off, off)
            if not b:
                break
            chunks.append(b)
            off += len(b)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return st, raw, max(raw.rfind(b"\r"), raw.rfind(b"\n")) + 1


def anchor_of(st, raw, end):
    return {"inode": st.st_ino, "device": st.st_dev, "size": len(raw), "offset": end,
            "tail_sha256": hashlib.sha256(raw[max(0, end - ANCHOR_BYTES):end]).hexdigest()}


def anchor_reasons(st, raw, anchor):
    """HARD STOP reasons of this read against the persisted ``anchor`` (None: the first observation)."""
    if anchor is None:
        return []
    if (st.st_ino, st.st_dev) != (anchor["inode"], anchor["device"]):
        return [f"the log's inode changed ({anchor['inode']} -> {st.st_ino}): rotated or replaced"]
    if len(raw) < anchor["size"]:
        return [f"the log shrank ({anchor['size']} -> {len(raw)} bytes)"]
    off = anchor["offset"]
    if hashlib.sha256(raw[max(0, off - ANCHOR_BYTES):off]).hexdigest() != anchor["tail_sha256"]:
        return [f"the {ANCHOR_BYTES} bytes before the anchor offset {off} changed: a copy-truncate-and-regrow"]
    return []


def observe(path, *, baseline=None, anchor=None):
    """One read-only observation of the log: status, exit code, reasons, the step, the windows, the new anchor."""
    out = {"status": "REFUSED", "exit": EXIT_REFUSED, "reasons": [], "baseline": baseline,
           "thresholds": {"w500": W500, "w100": W100, "mean": MEAN}}
    try:
        st, raw, end = read_log(path)
    except OSError as err:
        return dict(out, reasons=[f"the log {path} is unreadable ({err})"])
    epochs, errors = parse(raw[:end].decode("utf-8", "replace"))
    hard = anchor_reasons(st, raw, anchor) + errors + ([] if epochs else ["no training-bar line in the log"])
    reasons, summary, mean = judge(epochs, baseline) if not hard else ([], {}, None)
    e = max(epochs) if epochs else None
    n = max(epochs[e]["steps"]) if e is not None else None
    step = None if e is None else {"epoch": e, "n": n, "total": epochs[e]["total"], "global": e * epochs[e]["total"] + n}
    status = "HARD_STOP" if hard else "STOP" if reasons else "OK"
    return dict(out, status=status, exit={"HARD_STOP": EXIT_HARD, "STOP": EXIT_STOP, "OK": EXIT_OK}[status],
                reasons=hard or reasons, step=step, finished=bool(step and step["global"] >= MAX_STEPS),
                windows=summary, mean_since_baseline=mean, anchor=anchor_of(st, raw, end),
                log={"path": path, "inode": st.st_ino, "size": len(raw), "mtime": st.st_mtime})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--baseline-step", type=int, default=None)
    ap.add_argument("--anchor", default=None, help="a JSON file holding the persisted anchor (read only)")
    args = ap.parse_args(argv)
    anchor = None
    if args.anchor:
        try:
            with open(args.anchor) as f:
                anchor = json.load(f)
        except (OSError, ValueError) as err:
            print(json.dumps({"status": "REFUSED", "exit": EXIT_REFUSED, "reasons": [f"the anchor is unreadable ({err})"]}))
            return EXIT_REFUSED
    report = observe(args.log, baseline=args.baseline_step, anchor=anchor)
    print(json.dumps(report, indent=1))
    return report["exit"]


if __name__ == "__main__":
    sys.exit(main())
