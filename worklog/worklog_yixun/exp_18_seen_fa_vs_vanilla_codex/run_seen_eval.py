#!/usr/bin/env python3
"""Run the exp07 BF/P1 40k comparison on the full FLAC seen split.

The two predefined shards keep paired FA/vanilla cells on the same GPU.  Runs
are restartable: an existing cell is skipped only after its metrics record
passes the protocol checks below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/home/yixunhu/miniconda3/envs/flac/bin/python")
EVAL = ROOT / "eval_FLAC.py"
EXPECTED_COUNT = 6217
SEEDS = (42, 43, 44, 45, 46)
METRIC_KEYS = {
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
}

ARMS = {
    "BF": {
        "config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
        "ckpt": "outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt",
        "cond": "fa_invariant",
        "name": "BF40",
    },
    "P1": {
        "config": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
        "ckpt": "outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=8-step=40000.ckpt",
        "cond": "vanilla",
        "name": "P140",
    },
}

DATASETS = {
    1: "src/configs/dataset_configs/AR/eval/acousticroom_seeneval_1.json",
    8: "src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json",
}

# Each (K, seed) pair stays on one GPU so paired BF-P1 deltas cannot inherit a
# between-device numerical difference.  Each shard has five BF and five P1 cells.
SHARDS = {
    0: [
        ("P1", 8, 42), ("BF", 8, 42),
        ("P1", 1, 42), ("BF", 1, 42),
        ("P1", 8, 44), ("BF", 8, 44),
        ("P1", 1, 44), ("BF", 1, 44),
        ("P1", 8, 46), ("BF", 8, 46),
    ],
    1: [
        ("P1", 8, 43), ("BF", 8, 43),
        ("P1", 1, 43), ("BF", 1, 43),
        ("P1", 8, 45), ("BF", 8, 45),
        ("P1", 1, 45), ("BF", 1, 45),
        ("P1", 1, 46), ("BF", 1, 46),
    ],
}


def eval_name(arm: str, k: int, seed: int) -> str:
    return f"exp18_seen_{ARMS[arm]['name']}_K{k}_s{seed}"


def metrics_path(arm: str, k: int, seed: int) -> Path:
    spec = ARMS[arm]
    ckpt = ROOT / spec["ckpt"]
    suffix = "_fa_invariant_a4" if spec["cond"] == "fa_invariant" else ""
    return ckpt.with_name(
        f"{ckpt.stem}_metrics_1_1.0_{eval_name(arm, k, seed)}{suffix}.json"
    )


def validate_cell(arm: str, k: int, seed: int) -> dict:
    spec = ARMS[arm]
    path = metrics_path(arm, k, seed)
    with path.open() as handle:
        row = json.load(handle)

    expected = {
        "ckpt_path": spec["ckpt"],
        "rotate_deg": 0.0,
        "cond_method": spec["cond"],
        "cond_autocast": "bf16",
        "batch_size": 64,
        "n_samples": EXPECTED_COUNT,
        "dataset_config": DATASETS[k],
        "seed": seed,
        "cfg_scale": 1.0,
        "steps": 1,
        "eval_name": eval_name(arm, k, seed),
        "weights_source": "ema",
        "device": "cuda",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"{path}: {key}={row.get(key)!r}, expected {value!r}")

    if row.get("source_sha") == "unknown":
        raise ValueError(f"{path}: missing source SHA")
    if row.get("orbit_execution") != (
        "batched" if spec["cond"] == "fa_invariant" else "n/a"
    ):
        raise ValueError(f"{path}: unexpected orbit_execution")
    expected_angles = [0.0, 90.0, 180.0, 270.0] if arm == "BF" else None
    if row.get("frame_avg_angles") != expected_angles:
        raise ValueError(f"{path}: unexpected frame_avg_angles")

    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS:
        raise ValueError(f"{path}: unexpected metric key set")
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path}: non-finite/non-numeric metric {key}={value!r}")

    return row


def command(arm: str, k: int, seed: int) -> list[str]:
    spec = ARMS[arm]
    cmd = [
        str(PYTHON), str(EVAL),
        "--model-config", spec["config"],
        "--dataset-config", DATASETS[k],
        "--ckpt-path", spec["ckpt"],
        "--cfg-scale", "1.0",
        "--steps", "1",
        "--batch-size", "64",
        "--num-workers", "4",
        "--device", "cuda",
        "--eval-name", eval_name(arm, k, seed),
        "--seed", str(seed),
        "--rotate-deg", "0",
        "--cond-method", spec["cond"],
        "--frame-avg-max-fwd-samples", "64",
        "--cond-autocast", "bf16",
    ]
    if arm == "BF":
        cmd.extend(["--frame-avg-angles", "0,90,180,270"])
    return cmd


def tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"<could not read log: {exc}>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    if not PYTHON.is_file() or not EVAL.is_file():
        raise SystemExit("FLAC evaluator or conda interpreter is missing")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=str(ROOT),
    )

    for index, (arm, k, seed) in enumerate(SHARDS[args.shard], 1):
        name = eval_name(arm, k, seed)
        try:
            row = validate_cell(arm, k, seed)
        except (OSError, ValueError, json.JSONDecodeError):
            row = None
        if row is not None:
            print(f"[{index}/10] SKIP valid {name}", flush=True)
            continue

        log = log_dir / f"{name}.log"
        print(f"[{index}/10] START gpu={args.gpu} {name}", flush=True)
        start = time.monotonic()
        with log.open("w") as handle:
            completed = subprocess.run(
                command(arm, k, seed),
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            print(tail(log), file=sys.stderr)
            raise SystemExit(f"FAILED rc={completed.returncode}: {name} (log: {log})")
        row = validate_cell(arm, k, seed)
        m = row["metrics"]
        elapsed = (time.monotonic() - start) / 60
        print(
            f"[{index}/10] DONE {name} {elapsed:.1f}m "
            f"T60={m['T60']:.4f} C50={m['C50']:.4f} EDT={m['EDT']:.4f} "
            f"R1={m['RIR_to_GT_RIR_R@1']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
