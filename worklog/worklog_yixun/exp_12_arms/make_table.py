"""exp_12 -- build the deliverable: K=1 and K=8 tables of T60, C50, EDT, R@1, R@5, R@10.

    python make_table.py exp12A_c3c4 [exp12C_ray12 ...]

Reports mean +- sd over eval seeds 42-46. T60/C50/EDT are the eval's own units (T60 and
EDT relative % error, C50 absolute dB); R@k are percentages, higher is better.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

REPO = "/home/yixunhu/codespace/exp-12-arms"
KEYS = [
    ("T60", "T60"),
    ("C50", "C50"),
    ("EDT", "EDT"),
    ("R@1", "RIR_to_GT_RIR_R@1"),
    ("R@5", "RIR_to_GT_RIR_R@5"),
    ("R@10", "RIR_to_GT_RIR_R@10"),
]


def cells(run: str, k: int) -> dict[str, list[float]]:
    pat = os.path.join(
        REPO, "outputs_FLAC", run, "*", "*", "checkpoints",
        f"*_metrics_1_1.0_{run}_D1_K{k}_s*_fa_invariant_a1.json",
    )
    out: dict[str, list[float]] = {label: [] for label, _ in KEYS}
    files = sorted(glob.glob(pat))
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


def main(runs: list[str]) -> None:
    for k in (1, 8):
        print(f"\n### K = {k}   (6,337 unseen entries, seeds 42-46, fa_invariant [0], bf16)\n")
        print("| arm | " + " | ".join(label for label, _ in KEYS) + " | seeds |")
        print("|---" * (len(KEYS) + 2) + "|")
        for run in runs:
            c = cells(run, k)
            n = c.pop("_n")
            print(f"| {run} | " + " | ".join(fmt(c[label]) for label, _ in KEYS) + f" | {n}/5 |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["exp12A_c3c4", "exp12C_ray12"])
