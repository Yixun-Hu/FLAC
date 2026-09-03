#!/usr/bin/env python3
"""Measure strict-coverage detection plus random fallback on the 16 FEM failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.depth_aabb import depth_panorama_aabb, points_in_aabb
from src.localization.engine import (
    filter_frozen_query_candidates,
    reconstruct_room_base_candidates,
)
from src.localization.pilot import canonical_sha256
from src.localization.scoring import deterministic_random_candidate


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt SHA-256 in {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timed_failure_path(
    record: dict,
    geometry_audit: dict,
    room_base: np.ndarray,
    dataset_root: Path,
    *,
    seed: int,
) -> dict:
    total_started = time.perf_counter_ns()

    candidate_started = time.perf_counter_ns()
    candidates = filter_frozen_query_candidates(record, geometry_audit, room_base)
    receiver = np.asarray(record["receiver_global"], dtype=np.float64)
    source = np.asarray(record["source_global"], dtype=np.float64)
    contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
    candidate_seconds = (time.perf_counter_ns() - candidate_started) / 1e9

    gate_started = time.perf_counter_ns()
    receiver_id = int(record["filename"].split("_")[1][1:])
    depth_path = (
        dataset_root
        / "depth_map"
        / record["scene"]
        / record["room"]
        / f"{receiver_id}.npy"
    )
    depth = np.load(depth_path)
    lower, upper, _audit = depth_panorama_aabb(depth, padding_m=0.05)
    candidate_inside = points_in_aabb(candidates - receiver, lower, upper)
    source_inside = points_in_aabb((source - receiver)[None, :], lower, upper)
    context_inside = points_in_aabb(contexts - receiver, lower, upper)
    strict_gate_passed = bool(
        candidate_inside.all() and source_inside[0] and context_inside.all()
    )
    gate_seconds = (time.perf_counter_ns() - gate_started) / 1e9
    if strict_gate_passed:
        raise RuntimeError(f"query {record['index']} unexpectedly passes the strict gate")

    random_started = time.perf_counter_ns()
    prediction_index = deterministic_random_candidate(
        int(record["index"]), len(candidates), seed=seed
    )
    random_seconds = (time.perf_counter_ns() - random_started) / 1e9
    total_seconds = (time.perf_counter_ns() - total_started) / 1e9
    return {
        "candidate_preparation_seconds": candidate_seconds,
        "depth_aabb_strict_gate_seconds": gate_seconds,
        "random_candidate_selection_seconds": random_seconds,
        "fallback_total_seconds": total_seconds,
        "prediction_index": prediction_index,
        "candidate_inside_count": int(candidate_inside.sum()),
        "candidate_count": int(len(candidates)),
        "source_inside": bool(source_inside[0]),
        "context_inside_count": int(context_inside.sum()),
        "context_count": int(len(context_inside)),
        "depth_file": str(depth_path.resolve()),
        "depth_file_sha256": file_sha256(depth_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-selection", type=Path, required=True)
    parser.add_argument("--strict-selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_count <= 0:
        raise ValueError("repeat-count must be positive")

    full = load_hashed_json(args.full_selection.resolve())
    strict = load_hashed_json(args.strict_selection.resolve())
    full_records = {int(row["index"]): row for row in full["records"]}
    strict_indices = {int(row["index"]) for row in strict["records"]}
    failure_indices = sorted(set(full_records) - strict_indices)
    if len(full_records) != 128 or len(strict_indices) != 112 or len(failure_indices) != 16:
        raise RuntimeError("expected the frozen 128-query scope with 16 strict failures")

    context = load_hashed_json(args.context_manifest.resolve())
    context_records = {int(row["index"]): row for row in context["records"]}
    geometry = load_hashed_json(args.geometry_audit.resolve())
    if (
        full["context_manifest_sha256"] != context["sha256"]
        or full["geometry_audit_sha256"] != geometry["sha256"]
    ):
        raise RuntimeError("selection, context manifest, and geometry audit do not match")

    per_query = []
    for index in failure_indices:
        selected = full_records[index]
        record = context_records[index]
        if selected["query_id"] != record["query_id"]:
            raise RuntimeError(f"query identity mismatch at {index}")
        room_base = reconstruct_room_base_candidates(record["room"], geometry)
        repeats = [
            timed_failure_path(
                record,
                geometry,
                room_base,
                args.dataset_root.resolve(),
                seed=args.random_seed,
            )
            for _ in range(args.repeat_count)
        ]
        prediction_indices = {row["prediction_index"] for row in repeats}
        if len(prediction_indices) != 1:
            raise RuntimeError(f"random fallback changed across repeats at {index}")
        timing_fields = (
            "candidate_preparation_seconds",
            "depth_aabb_strict_gate_seconds",
            "random_candidate_selection_seconds",
            "fallback_total_seconds",
        )
        per_query.append(
            {
                "query_index": index,
                "query_id": selected["query_id"],
                "room": selected["room"],
                "candidate_count": int(selected["candidate_count"]),
                "candidate_indices_sha256": selected["candidate_indices_sha256"],
                "prediction_index": repeats[0]["prediction_index"],
                "strict_gate_audit": {
                    key: repeats[0][key]
                    for key in (
                        "candidate_inside_count",
                        "candidate_count",
                        "source_inside",
                        "context_inside_count",
                        "context_count",
                    )
                },
                "depth_file": repeats[0]["depth_file"],
                "depth_file_sha256": repeats[0]["depth_file_sha256"],
                "repeat_seconds": {
                    field: [float(row[field]) for row in repeats]
                    for field in timing_fields
                },
                "median_seconds": {
                    field: float(statistics.median(row[field] for row in repeats))
                    for field in timing_fields
                },
            }
        )

    body = {
        "schema_version": 1,
        "record_type": "fem_strict_failure_random_candidate_latency",
        "timing_boundary": (
            "frozen candidate preparation plus depth-AABB strict-coverage detection "
            "plus deterministic random candidate selection"
        ),
        "excluded": [
            "manifest and geometry-audit loading",
            "room base-grid reconstruction",
            "localization metric computation",
            "result serialization",
        ],
        "full_selection_sha256": full["sha256"],
        "strict_selection_sha256": strict["sha256"],
        "context_manifest_sha256": context["sha256"],
        "geometry_audit_sha256": geometry["sha256"],
        "repeat_count": int(args.repeat_count),
        "random_seed": int(args.random_seed),
        "query_count": len(per_query),
        "queries": per_query,
    }
    payload = {**body, "sha256": canonical_sha256(body)}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "query_count": len(per_query),
                "sha256": payload["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
