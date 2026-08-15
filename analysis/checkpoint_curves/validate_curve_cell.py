#!/usr/bin/env python3
"""Validate one newly generated K=1, seed-42 checkpoint-curve cell."""

from __future__ import annotations

import argparse
import json
import math
import re
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


def validate(path: Path, arm: str, step: int) -> dict:
    if not path.is_file():
        raise ValueError(f"missing metrics JSON: {path}")
    with path.open() as handle:
        record = json.load(handle)

    expected_method = "fa_invariant" if arm == "FA" else "vanilla"
    expected_name = f"curve0_{arm}_S{step}_K1_s42"
    checks = {
        "checkpoint step": re.search(rf"step={step}(?:\D|$)", record.get("ckpt_path", "")) is not None,
        "yaw": record.get("rotate_deg") == 0.0,
        "conditioning": record.get("cond_method") == expected_method,
        "seed": record.get("seed") == 42,
        "dataset": record.get("dataset_config")
        == "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
        "eval name": record.get("eval_name") == expected_name,
        "source": record.get("source_sha") == EXPECTED_SOURCE,
        "batch size": record.get("batch_size") == 64,
        "split size": record.get("n_samples") == 6337,
        "cfg scale": record.get("cfg_scale") == 1.0,
        "sampling steps": record.get("steps") == 1,
        "weights": record.get("weights_source") == "ema",
        "precision": record.get("cond_autocast") == "bf16",
    }
    if arm == "FA":
        checks.update(
            {
                "FA orbit": record.get("frame_avg_angles")
                == [0.0, 90.0, 180.0, 270.0],
                "FA execution": record.get("orbit_execution") == "batched",
                "FA cap": record.get("frame_avg_fwd_cap") == 64,
            }
        )
    else:
        checks.update(
            {
                "no FA orbit": record.get("frame_avg_angles") is None,
                "vanilla execution": record.get("orbit_execution") == "n/a",
                "no FA cap": record.get("frame_avg_fwd_cap") is None,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid {arm} S{step} cell ({', '.join(failed)}): {path}")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or tuple(metrics) != EXPECTED_METRICS:
        raise ValueError(f"metric schema mismatch: {path}")
    if not all(
        isinstance(metrics[key], (int, float)) and math.isfinite(metrics[key])
        for key in EXPECTED_METRICS
    ):
        raise ValueError(f"non-finite metric: {path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--arm", choices=("FA", "VAN"), required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    validate(args.json_path, args.arm, args.step)
    print(f"VALID {args.arm} K=1 yaw=0 seed=42 step={args.step}: {args.json_path}")


if __name__ == "__main__":
    main()

