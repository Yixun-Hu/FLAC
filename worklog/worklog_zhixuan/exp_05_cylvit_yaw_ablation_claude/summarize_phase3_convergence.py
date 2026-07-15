#!/usr/bin/env python3
"""Collect Phase 3 eval JSON files into a compact convergence report."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


METRICS = (
    "T60",
    "C50",
    "EDT",
    "FD",
    "RIR_to_GT_RIR_R@1",
    "RIR_to_GT_RIR_R@5",
    "RIR_to_GT_RIR_R@10",
)
PATTERN = re.compile(r"convergence_total(?P<total>\d+)k_yaw(?P<yaw>\d+)")


def collect(search_root: Path, model: str):
    rows = {}
    marker = f"exp05_{model}_convergence_"
    for path in search_root.rglob("*metrics*.json"):
        if marker not in path.name:
            continue
        match = PATTERN.search(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rows[(int(match.group("total")), int(match.group("yaw")))] = {
            "metrics": data["metrics"],
            "path": path,
        }
    return rows


def value(metrics, key):
    item = metrics.get(key)
    return "-" if item is None else f"{float(item):.4f}"


def delta(current, previous, key):
    if previous is None or key not in current or key not in previous:
        return "-"
    return f"{float(current[key]) - float(previous[key]):+.4f}"


def render(model: str, rows):
    lines = [
        f"# Phase 3 {model} Convergence",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Lower is better for T60/C50/EDT/FD; higher is better for "
        "RIR-to-GT R@1/R@5/R@10. "
        "Deltas are relative to the preceding available milestone.",
        "",
        "## Yaw 0 Milestones",
        "",
        "| Total step | T60 | dT60 | C50 | dC50 | EDT | dEDT | FD | dFD | GT R@1 | dR@1 | GT R@5 | dR@5 | GT R@10 | dR@10 |",
        "|" + "---:|" * (1 + 2 * len(METRICS)),
    ]
    previous = None
    for total in (5, 10, 15, 20, 25, 30):
        row = rows.get((total, 0))
        if row is None:
            cells = [f"{total}k"] + ["-"] * (2 * len(METRICS))
            lines.append("| " + " | ".join(cells) + " |")
            continue
        metrics = row["metrics"]
        cells = [f"{total}k"]
        for key in METRICS:
            cells.extend((value(metrics, key), delta(metrics, previous, key)))
        lines.append("| " + " | ".join(cells) + " |")
        previous = metrics

    lines.extend([
        "",
        "## Yaw Sweeps",
        "",
        "| Total step | Yaw | T60 | C50 | EDT | FD | GT R@1 | GT R@5 | GT R@10 |",
        "|" + "---:|" * (2 + len(METRICS)),
    ])
    sweep_totals = sorted({total for total, yaw in rows if yaw in (0, 90, 180, 270)})
    for total in sweep_totals:
        for yaw in (0, 90, 180, 270):
            row = rows.get((total, yaw))
            if row is None:
                continue
            metrics = row["metrics"]
            cells = [f"{total}k", str(yaw)] + [value(metrics, key) for key in METRICS]
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Sources", ""])
    for key in sorted(rows):
        lines.append(f"- total {key[0]}k, yaw {key[1]}: `{rows[key]['path']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("simplevit", "cylvit"))
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.search_root, args.model)
    args.output.write_text(render(args.model, rows))
    print(f"Wrote {args.output} from {len(rows)} eval records")


if __name__ == "__main__":
    main()
