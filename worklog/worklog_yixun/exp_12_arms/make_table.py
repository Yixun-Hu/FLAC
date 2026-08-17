"""exp_12 -- build the deliverable: K=1 and K=8 tables of T60, C50, EDT, R@1, R@5, R@10.

    python make_table.py exp12A_c3c4 [exp12C_ray12 ...]

Reports mean +- sd over eval seeds 42-46. T60/C50/EDT are the eval's own units (T60 and
EDT relative % error, C50 absolute dB); R@k are percentages, higher is better.
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys

REPO = "/home/yixunhu/codespace/exp-12-arms"
ENDPOINT_STEP = 67500          # the registered 67.5k endpoint; override with --step
KEYS = [
    ("T60", "T60"),
    ("C50", "C50"),
    ("EDT", "EDT"),
    ("R@1", "RIR_to_GT_RIR_R@1"),
    ("R@5", "RIR_to_GT_RIR_R@5"),
    ("R@10", "RIR_to_GT_RIR_R@10"),
]


def cells(run: str, k: int, step: int = ENDPOINT_STEP) -> dict[str, list[float]]:
    """Collect one arm's eval cells at ONE checkpoint step.

    The step MUST be pinned. The eval-name pattern `<run>_D1_K<k>_s<seed>` does not
    contain the checkpoint step -- the step lives in the filename prefix -- so an
    unpinned `*_metrics_...` glob silently merges every checkpoint that was ever
    evaluated under that name into a single mean. That is not hypothetical: a parallel
    session evaluated arm A's step-40000 checkpoint on 2026-08-13 reusing this exact
    eval-name convention, and the unpinned glob then averaged step 40000 together with
    the step-67500 endpoint, shifting K=8 T60 from 9.4038 to 9.0824 and inflating the
    reported sd 36x (0.0093 -> 0.3389). A sd far above the seed-noise scale is the
    tell; pinning the step is the fix.
    """
    pat = os.path.join(
        REPO, "outputs_FLAC", run, "*", "*", "checkpoints",
        f"*step={step}_metrics_1_1.0_{run}_D1_K{k}_s*_fa_invariant_a1.json",
    )
    out: dict[str, list[float]] = {label: [] for label, _ in KEYS}
    files = sorted(glob.glob(pat))
    seeds = {re.search(r"_s(\d+)_fa_invariant", f).group(1) for f in files}
    if len(seeds) != len(files):
        raise SystemExit(
            f"REFUSE: {run} K={k} step={step}: {len(files)} files but only {len(seeds)} "
            "distinct seeds -- duplicate cells would be double-counted."
        )
    for f in files:
        m = json.load(open(f))["metrics"]
        for label, key in KEYS:
            if key in m:
                out[label].append(float(m[key]))
    out["_n"] = len(files)  # type: ignore[assignment]
    return out


def fmt(vals: list[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def main(runs: list[str], step: int = ENDPOINT_STEP) -> None:
    for k in (1, 8):
        print(f"\n### K = {k}   (6,337 unseen entries, step {step}, seeds 42-46, "
              f"fa_invariant [0], bf16)\n")
        print("| arm | " + " | ".join(label for label, _ in KEYS) + " | seeds |")
        print("|---" * (len(KEYS) + 2) + "|")
        for run in runs:
            c = cells(run, k, step)
            n = c.pop("_n")
            print(f"| {run} | " + " | ".join(fmt(c[label]) for label, _ in KEYS) + f" | {n}/5 |")


if __name__ == "__main__":
    argv = sys.argv[1:]
    step = ENDPOINT_STEP
    if "--step" in argv:
        i = argv.index("--step")
        step = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    main(argv or ["exp12A_c3c4", "exp12C_ray12"], step)
