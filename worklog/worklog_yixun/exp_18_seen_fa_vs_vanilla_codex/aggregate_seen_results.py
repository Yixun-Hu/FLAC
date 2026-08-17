#!/usr/bin/env python3
"""Validate and aggregate the 20 exp18 FLAC seen evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

from run_seen_eval import ARMS, SEEDS, metrics_path, validate_cell


DISPLAY_METRICS = (
    ("T60", "T60"),
    ("C50", "C50"),
    ("EDT", "EDT"),
    ("RIR_to_GT_RIR_R@1", "R@1"),
    ("RIR_to_GT_RIR_R@5", "R@5"),
    ("RIR_to_GT_RIR_R@10", "R@10"),
    ("FD", "FD"),
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> int:
    rows = {
        (arm, k, seed): validate_cell(arm, k, seed)
        for arm in ARMS
        for k in (1, 8)
        for seed in SEEDS
    }

    summary: dict[str, object] = {
        "protocol": {
            "split": "FLAC seen",
            "n_samples": 6217,
            "seeds": list(SEEDS),
            "std": "sample (ddof=1)",
            "weights": "EMA",
            "cfg_scale": 1.0,
            "steps": 1,
            "cond_autocast": "bf16",
        },
        "arms": {},
        "paired_delta_fa_minus_vanilla": {},
        "source_files": [
            str(metrics_path(arm, k, seed).relative_to(Path(__file__).resolve().parents[3]))
            for arm in ARMS for k in (1, 8) for seed in SEEDS
        ],
    }

    for arm in ARMS:
        arm_result = {}
        for k in (1, 8):
            arm_result[f"K{k}"] = {
                key: stats([rows[(arm, k, seed)]["metrics"][key] for seed in SEEDS])
                for key, _ in DISPLAY_METRICS
            }
        summary["arms"][arm] = arm_result

    for k in (1, 8):
        summary["paired_delta_fa_minus_vanilla"][f"K{k}"] = {
            key: stats([
                rows[("BF", k, seed)]["metrics"][key]
                - rows[("P1", k, seed)]["metrics"][key]
                for seed in SEEDS
            ])
            for key, _ in DISPLAY_METRICS
        }

    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "seen_comparison_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("| Model | K | " + " | ".join(label for _, label in DISPLAY_METRICS) + " |")
    print("|---|---:|" + "---:|" * len(DISPLAY_METRICS))
    for arm, label in (("BF", "FA (exp07_BF @40k)"), ("P1", "Vanilla (exp07_P1 @40k)")):
        for k in (1, 8):
            cells = []
            for key, _ in DISPLAY_METRICS:
                item = summary["arms"][arm][f"K{k}"][key]
                cells.append(f"{item['mean']:.4f} ± {item['sample_std']:.4f}")
            print(f"| {label} | {k} | " + " | ".join(cells) + " |")
    print(f"\nValidated summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
