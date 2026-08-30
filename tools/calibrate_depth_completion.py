#!/usr/bin/env python3
"""Calibrate a candidate-independent depth completion distance on train rooms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    required_horizontal_completion_distance,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path(root: Path, scene: str, room: str, filename: str) -> Path:
    suffix = "_hybrid_IR.wav"
    if not filename.endswith(suffix):
        raise ValueError(f"unexpected AcousticRooms filename: {filename}")
    source_id = int(filename.split("_")[0][1:])
    receiver_id = int(filename.split("_")[1][1:])
    # Reproduce AcousticRooms/FLAC's released metadata naming convention.
    metadata_name = f"S00{source_id}_R00{receiver_id}.json"
    return root / "metadata" / scene / room / metadata_name


def receiver_number(filename: str) -> int:
    return int(filename.split("_")[1][1:])


def select_receivers(groups: dict[int, list[str]], count: int) -> list[int]:
    values = sorted(groups)
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count + 2)[1:-1]
    return sorted({values[int(round(position))] for position in positions})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receivers-per-room", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--padding-m", type=float, default=0.05)
    args = parser.parse_args()
    if args.receivers_per_room <= 0:
        raise ValueError("receivers-per-room must be positive")
    if not 0.0 < args.quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")

    started = time.perf_counter()
    split = json.loads(args.train_split.read_text())
    required_values: list[np.ndarray] = []
    room_rows = []
    depth_hashes = []
    total_anchors = 0
    total_aabb_inside = 0

    for scene, rooms in split.items():
        for room, filenames in rooms.items():
            groups: dict[int, list[str]] = {}
            for filename in filenames:
                groups.setdefault(receiver_number(filename), []).append(filename)
            selected = select_receivers(groups, args.receivers_per_room)
            room_required = []
            room_anchor_count = 0
            room_aabb_inside = 0
            for receiver_id in selected:
                source_files: dict[int, str] = {}
                for filename in groups[receiver_id]:
                    source_id = int(filename.split("_")[0][1:])
                    source_files.setdefault(source_id, filename)
                metadata = [
                    json.loads(metadata_path(args.dataset_root, scene, room, filename).read_text())
                    for filename in source_files.values()
                ]
                receiver = np.asarray(metadata[0]["rec_loc"], dtype=np.float64)
                sources = np.asarray([item["src_loc"] for item in metadata], dtype=np.float64)
                local = sources - receiver
                depth_path = (
                    args.dataset_root / "depth_map" / scene / room / f"{receiver_id}.npy"
                )
                depth = np.load(depth_path)
                depth_hashes.append(file_sha256(depth_path))
                layout, _ = depth_panorama_polar_layout(depth, padding_m=args.padding_m)
                lower, upper, _ = depth_panorama_aabb(depth, padding_m=args.padding_m)
                inside_aabb = points_in_aabb(local, lower, upper, tolerance_m=1e-7)
                needed = required_horizontal_completion_distance(local[inside_aabb], layout)
                required_values.append(needed)
                room_required.append(needed)
                count = len(local)
                inside_count = int(inside_aabb.sum())
                room_anchor_count += count
                room_aabb_inside += inside_count
                total_anchors += count
                total_aabb_inside += inside_count
            concatenated = np.concatenate(room_required) if room_required else np.empty(0)
            room_rows.append(
                {
                    "scene": scene,
                    "room": room,
                    "selected_receiver_ids": selected,
                    "source_anchor_count": room_anchor_count,
                    "source_anchor_inside_same_view_aabb_count": room_aabb_inside,
                    "required_completion_p95_m": (
                        float(np.quantile(concatenated, 0.95)) if len(concatenated) else None
                    ),
                    "required_completion_max_m": (
                        float(concatenated.max()) if len(concatenated) else None
                    ),
                }
            )

    all_required = np.concatenate(required_values)
    calibrated = float(np.quantile(all_required, args.quantile))

    covered_after = 0
    # Re-evaluate from the compact per-room loop inputs to independently check
    # the frozen scalar.  This second pass is intentionally train-only.
    for scene, rooms in split.items():
        for room, filenames in rooms.items():
            groups: dict[int, list[str]] = {}
            for filename in filenames:
                groups.setdefault(receiver_number(filename), []).append(filename)
            for receiver_id in select_receivers(groups, args.receivers_per_room):
                source_files: dict[int, str] = {}
                for filename in groups[receiver_id]:
                    source_files.setdefault(int(filename.split("_")[0][1:]), filename)
                metadata = [
                    json.loads(metadata_path(args.dataset_root, scene, room, filename).read_text())
                    for filename in source_files.values()
                ]
                receiver = np.asarray(metadata[0]["rec_loc"], dtype=np.float64)
                local = np.asarray([item["src_loc"] for item in metadata]) - receiver
                depth = np.load(
                    args.dataset_root / "depth_map" / scene / room / f"{receiver_id}.npy"
                )
                layout, _ = depth_panorama_polar_layout(depth, padding_m=args.padding_m)
                lower, upper, _ = depth_panorama_aabb(depth, padding_m=args.padding_m)
                completed, _ = complete_polar_layout_toward_depth_aabb(
                    layout, lower, upper, completion_distance_m=calibrated
                )
                covered_after += int(
                    points_in_polar_layout(local, completed, tolerance_m=1e-7).sum()
                )

    payload = {
        "schema_version": 1,
        "method": "depth_bounded_radial_completion_train_calibration",
        "inference_inputs": "one receiver-centered metric depth panorama only",
        "training_supervision": (
            "source anchors from train rooms; no unseen-eval room mesh, depth, candidates, "
            "source anchors, or context anchors"
        ),
        "train_split": str(args.train_split.resolve()),
        "train_split_sha256": file_sha256(args.train_split),
        "dataset_root": str(args.dataset_root.resolve()),
        "room_count": len(room_rows),
        "receivers_per_room": args.receivers_per_room,
        "selected_receiver_view_count": len(depth_hashes),
        "selected_depth_sha256_set_sha256": hashlib.sha256(
            "\n".join(sorted(depth_hashes)).encode()
        ).hexdigest(),
        "source_anchor_count": total_anchors,
        "source_anchor_inside_same_view_aabb_count": total_aabb_inside,
        "source_anchor_inside_same_view_aabb_fraction": total_aabb_inside / total_anchors,
        "calibration_quantile": args.quantile,
        "calibrated_completion_distance_m": calibrated,
        "required_completion_statistics_m": {
            "median": float(np.median(all_required)),
            "p95": float(np.quantile(all_required, 0.95)),
            "p99": float(np.quantile(all_required, 0.99)),
            "p995": float(np.quantile(all_required, 0.995)),
            "maximum": float(all_required.max()),
        },
        "train_anchor_coverage_after_completion_count": covered_after,
        "train_anchor_coverage_after_completion_fraction": covered_after / total_anchors,
        "padding_m": args.padding_m,
        "runtime_seconds": time.perf_counter() - started,
        "rooms": room_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "rooms"}, indent=2))


if __name__ == "__main__":
    main()
