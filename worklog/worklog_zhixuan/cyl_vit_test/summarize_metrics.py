#!/usr/bin/env python3
"""Aggregate cyl-vit-test metric JSONs by K and yaw."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


EVAL_RE = re.compile(
    r"cylvit_(?P<split>seen|unseen)_K(?P<k>[18])_s(?P<seed>\d+)_yaw(?P<yaw>-?\d+(?:\.\d+)?)"
)
PREFERRED_METRICS = (
    "T60",
    "C50",
    "EDT",
    "FD",
    "RIR_to_GT_RIR_R@1",
    "RIR_to_GT_RIR_R@5",
    "RIR_to_GT_RIR_R@10",
)


def load_records(checkpoint: Path) -> list[dict]:
    stem = checkpoint.stem
    records = []
    for path in sorted(checkpoint.parent.glob(f"{stem}_metrics_*_cylvit_*_K*.json")):
        match = EVAL_RE.search(path.name)
        if not match:
            continue
        with path.open() as handle:
            payload = json.load(handle)
        records.append(
            {
                "path": path,
                "split": match.group("split"),
                "k": int(match.group("k")),
                "seed": int(match.group("seed")),
                "yaw": float(match.group("yaw")),
                "metrics": payload["metrics"],
            }
        )
    return records


def format_stat(values: list[float]) -> str:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.4f}"
    std = statistics.stdev(values)
    return f"{mean:.4f} +/- {std:.4f}"


def build_summary(checkpoint: Path, records: list[dict]) -> str:
    lines = [
        "# CylindricalViT evaluation summary",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
    ]
    if not records:
        lines.append("No matching metric JSON files were found.")
        return "\n".join(lines) + "\n"

    grouped: dict[tuple[str, int, float], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["k"], record["yaw"])].append(record)

    available = set.intersection(*(set(record["metrics"]) for record in records))
    metrics = [name for name in PREFERRED_METRICS if name in available]
    for name in sorted(available):
        if name not in metrics and all(
            isinstance(record["metrics"].get(name), (int, float))
            and math.isfinite(float(record["metrics"][name]))
            for record in records
        ):
            metrics.append(name)

    header = ["split", "K", "yaw", "seeds", *metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for (split, k_value, yaw), group in sorted(grouped.items()):
        seeds = ",".join(str(record["seed"]) for record in sorted(group, key=lambda item: item["seed"]))
        row = [split, str(k_value), f"{yaw:g}", seeds]
        for metric in metrics:
            values = [float(record["metrics"][metric]) for record in group]
            row.append(format_stat(values))
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", f"Matched metric files: {len(records)}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output or checkpoint.parent / "cyl_vit_test_metrics_summary.md"
    records = load_records(checkpoint)
    output.write_text(build_summary(checkpoint, records))
    print(output)


if __name__ == "__main__":
    main()
