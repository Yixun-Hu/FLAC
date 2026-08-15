#!/usr/bin/env python3
"""Fail-closed validator for one exp_10 A6 C4 yaw evaluation JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_SOURCE = "069b72a5f0ccd43e52551ad4bde1355e8caab92d"
EXPECTED_METRICS = (
    "T60",
    "Invalid T60",
    "C50",
    "EDT",
    "FD",
    "RIR_to_GT_RIR_R@1",
    "RIR_to_GT_RIR_R@5",
    "RIR_to_GT_RIR_R@10",
    "RIR_to_geom_R@1",
    "RIR_to_geom_R@5",
    "RIR_to_geom_R@10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--arm", choices=("FA40", "VAN40"), required=True)
    parser.add_argument("--k", type=int, choices=(1, 8), required=True)
    parser.add_argument("--angle", type=int, choices=(180, 270), required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    args = parse_args()
    require(args.json_path.is_file(), f"missing metrics JSON: {args.json_path}")
    with args.json_path.open() as handle:
        record = json.load(handle)

    expected_dataset = (
        "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
        if args.k == 8
        else "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json"
    )
    expected_method = "fa_invariant" if args.arm == "FA40" else "vanilla"
    expected_name = (
        f"a6c4_{args.arm}_rot{args.angle}_"
        f"{'unseeneval' if args.k == 8 else 'unseeneval_1'}_s{args.seed}"
    )

    require(record.get("rotate_deg") == float(args.angle), "wrong rotation angle")
    require(record.get("cond_method") == expected_method, "wrong conditioning method")
    require(record.get("seed") == args.seed, "wrong evaluation seed")
    require(record.get("dataset_config") == expected_dataset, "wrong dataset config")
    require(record.get("eval_name") == expected_name, "wrong evaluation name")
    require(record.get("source_sha") == EXPECTED_SOURCE, "wrong evaluator source SHA")
    require(record.get("batch_size") == 64, "wrong evaluation batch size")
    require(record.get("n_samples") == 6337, "incomplete AR unseen split")
    require(record.get("cfg_scale") == 1.0, "wrong CFG scale")
    require(record.get("steps") == 1, "wrong diffusion step count")
    require(record.get("weights_source") == "ema", "evaluation did not use EMA")
    require(record.get("cond_autocast") == "bf16", "wrong conditioning precision")

    if args.arm == "FA40":
        require(
            record.get("frame_avg_angles") == [0.0, 90.0, 180.0, 270.0],
            "wrong FA orbit",
        )
        require(record.get("orbit_execution") == "batched", "wrong orbit execution")
        require(record.get("frame_avg_fwd_cap") == 64, "wrong FA forward cap")
    else:
        require(record.get("frame_avg_angles") is None, "vanilla unexpectedly used FA")
        require(record.get("orbit_execution") == "n/a", "wrong vanilla provenance")
        require(record.get("frame_avg_fwd_cap") is None, "vanilla has an FA cap")

    metrics = record.get("metrics")
    require(isinstance(metrics, dict), "missing metrics object")
    require(tuple(metrics) == EXPECTED_METRICS, "metric schema/order mismatch")
    require(
        all(isinstance(metrics[key], (int, float)) and math.isfinite(metrics[key]) for key in EXPECTED_METRICS),
        "non-finite metric",
    )
    print(f"VALID {args.arm} K={args.k} angle={args.angle} seed={args.seed}: {args.json_path}")


if __name__ == "__main__":
    main()
