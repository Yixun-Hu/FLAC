#!/usr/bin/env python3
"""Audit frozen-candidate coverage of train-calibrated depth completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.depth_aabb import depth_panorama_aabb, points_in_aabb
from src.baselines.depth_polar_layout import (
    complete_polar_layout_toward_depth_aabb,
    depth_panorama_polar_layout,
    points_in_polar_layout,
)
from src.localization.engine import (
    filter_frozen_query_candidates,
    reconstruct_room_base_candidates,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    context = json.loads(args.context_manifest.read_text())
    context_by_index = {int(record["index"]): record for record in context["records"]}
    geometry = json.loads(args.geometry_audit.read_text())
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("method") != "depth_bounded_radial_completion_train_calibration":
        raise ValueError("wrong calibration contract")
    completion_distance = float(calibration["calibrated_completion_distance_m"])
    query_indices = []
    for path in args.pilot_manifest:
        pilot = json.loads(path.read_text())
        query_indices.extend(int(record["index"]) for record in pilot["records"])
    if len(query_indices) != len(set(query_indices)):
        raise ValueError("pilot manifests contain duplicate query indices")

    query_rows = []
    room_counts = defaultdict(lambda: {"query_count": 0, "completion_strict": 0, "aabb_strict": 0})
    for query_index in query_indices:
        record = context_by_index[query_index]
        receiver = np.asarray(record["receiver_global"], dtype=np.float64)
        source = np.asarray(record["source_global"], dtype=np.float64)
        contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
        room_base = reconstruct_room_base_candidates(record["room"], geometry)
        candidates = filter_frozen_query_candidates(record, geometry, room_base)
        receiver_id = int(record["filename"].split("_")[1][1:])
        depth_path = (
            args.dataset_root
            / "depth_map"
            / record["scene"]
            / record["room"]
            / f"{receiver_id}.npy"
        )
        depth = np.load(depth_path)
        layout, _ = depth_panorama_polar_layout(
            depth, padding_m=float(calibration["padding_m"])
        )
        lower, upper, _ = depth_panorama_aabb(
            depth, padding_m=float(calibration["padding_m"])
        )
        completed, _ = complete_polar_layout_toward_depth_aabb(
            layout, lower, upper, completion_distance_m=completion_distance
        )
        candidate_inside = points_in_polar_layout(
            candidates - receiver, completed, tolerance_m=1e-7
        )
        source_inside = bool(
            points_in_polar_layout(
                (source - receiver)[None, :], completed, tolerance_m=1e-7
            )[0]
        )
        context_inside = points_in_polar_layout(
            contexts - receiver, completed, tolerance_m=1e-7
        )
        completion_strict = bool(
            candidate_inside.all() and source_inside and context_inside.all()
        )
        aabb_points = np.concatenate((candidates, source[None, :], contexts), axis=0)
        aabb_strict = bool(
            points_in_aabb(
                aabb_points - receiver, lower, upper, tolerance_m=1e-7
            ).all()
        )
        row = {
            "query_index": query_index,
            "query_id": record["query_id"],
            "room": record["room"],
            "candidate_inside_count": int(candidate_inside.sum()),
            "candidate_count": int(len(candidates)),
            "source_inside": source_inside,
            "context_inside_count": int(context_inside.sum()),
            "context_count": int(len(contexts)),
            "completion_strict_gate": completion_strict,
            "depth_aabb_strict_gate": aabb_strict,
            "depth_sha256": file_sha256(depth_path),
        }
        query_rows.append(row)
        counts = room_counts[record["room"]]
        counts["query_count"] += 1
        counts["completion_strict"] += int(completion_strict)
        counts["aabb_strict"] += int(aabb_strict)

    payload = {
        "schema_version": 1,
        "audit_scope": "continuous reconstructed domains; tetrahedral gates audited separately",
        "candidate_policy": "frozen and identical; no candidate filtering or score substitution",
        "completion_distance_m": completion_distance,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": file_sha256(args.calibration),
        "context_manifest_sha256": file_sha256(args.context_manifest),
        "geometry_audit_sha256": file_sha256(args.geometry_audit),
        "pilot_manifests": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in args.pilot_manifest
        ],
        "query_count": len(query_rows),
        "candidate_full_coverage_query_count": sum(
            row["candidate_inside_count"] == row["candidate_count"] for row in query_rows
        ),
        "completion_strict_gate_query_count": sum(
            row["completion_strict_gate"] for row in query_rows
        ),
        "depth_aabb_strict_gate_query_count": sum(
            row["depth_aabb_strict_gate"] for row in query_rows
        ),
        "rooms": dict(sorted(room_counts.items())),
        "queries": query_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "queries"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
