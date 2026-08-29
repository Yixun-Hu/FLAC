#!/usr/bin/env python3
"""Run a frozen Depth-AABB pilot with bounded CPU concurrency and resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)


def completed_result(path: Path, query_index: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("method") == "fem_sabine_depth_aabb"
        and int(payload.get("query_index", -1)) == query_index
        and payload.get("coverage_protocol", {}).get("strict_gate_passed") is True
        and "metrics" in payload
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--solver-threads", type=int, default=12)
    parser.add_argument("--mkl-runtime", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.solver_threads <= 0:
        raise ValueError("workers and solver threads must be positive")
    if not args.mkl_runtime.is_file():
        raise FileNotFoundError(args.mkl_runtime)

    selection = json.loads(args.selection.read_text())
    records = selection["records"]
    if len(records) != int(selection["query_count"]):
        raise ValueError("selection query count is inconsistent")
    indices = [int(record["index"]) for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("selection contains duplicate query indices")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["MKL_RT"] = str(args.mkl_runtime.resolve())
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-depth-aabb-pilot")

    def run_one(query_index: int) -> dict:
        result_path = output_dir / f"query_{query_index:05d}_depth_aabb_result.json"
        if completed_result(result_path, query_index):
            return {"query_index": query_index, "status": "resume", "seconds": 0.0}
        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_depth_aabb_fem.py"),
            "--query-index",
            str(query_index),
            "--context-manifest",
            str(args.context_manifest.resolve()),
            "--geometry-audit",
            str(args.geometry_audit.resolve()),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--output-dir",
            str(output_dir),
            "--solver-backend",
            "mkl_pardiso",
            "--solver-threads",
            str(args.solver_threads),
        ]
        started = time.perf_counter()
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        (log_dir / f"query_{query_index:05d}.log").write_text(process.stdout)
        if process.returncode != 0 or not completed_result(result_path, query_index):
            raise RuntimeError(
                f"query {query_index} failed with exit {process.returncode}; "
                f"see {log_dir / f'query_{query_index:05d}.log'}"
            )
        return {"query_index": query_index, "status": "completed", "seconds": elapsed}

    started = time.perf_counter()
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {executor.submit(run_one, index): index for index in indices}
        for future in as_completed(future_to_index):
            outcome = future.result()
            outcomes.append(outcome)
            print(json.dumps(outcome, sort_keys=True), flush=True)
    outcomes.sort(key=lambda item: indices.index(item["query_index"]))
    summary = {
        "schema_version": 1,
        "method": "fem_sabine_depth_aabb",
        "selection": str(args.selection.resolve()),
        "workers": args.workers,
        "solver_threads_per_worker": args.solver_threads,
        "query_count": len(indices),
        "completed_count": sum(item["status"] == "completed" for item in outcomes),
        "resumed_count": sum(item["status"] == "resume" for item in outcomes),
        "wall_seconds": time.perf_counter() - started,
        "outcomes": outcomes,
    }
    temporary = output_dir / "run_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_dir / "run_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
