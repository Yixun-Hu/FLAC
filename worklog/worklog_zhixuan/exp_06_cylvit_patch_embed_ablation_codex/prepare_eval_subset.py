#!/usr/bin/env python3
"""Build the deterministic 32-per-room subset used by the exp06 C16 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length < count:
        raise ValueError(f"Cannot select {count} unique entries from a list of length {length}")
    if count == 1:
        return [length // 2]
    indices = [int(index * (length - 1) / (count - 1) + 0.5) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError(f"Even-spacing produced duplicate indices: {indices}")
    return indices


def build_subset(source: dict, per_room: int) -> tuple[dict, list[dict]]:
    subset: dict[str, object] = {}
    audit: list[dict] = []
    for scene, value in source.items():
        if isinstance(value, dict):
            subset[scene] = {}
            for room, filenames in value.items():
                ordered = sorted(filenames)
                indices = evenly_spaced_indices(len(ordered), per_room)
                subset[scene][room] = [ordered[index] for index in indices]
                audit.append(
                    {
                        "scene": scene,
                        "room": room,
                        "source_count": len(ordered),
                        "selected_count": per_room,
                        "selected_sorted_indices": indices,
                    }
                )
        else:
            ordered = sorted(value)
            indices = evenly_spaced_indices(len(ordered), per_room)
            subset[scene] = [ordered[index] for index in indices]
            audit.append(
                {
                    "scene": scene,
                    "room": scene,
                    "source_count": len(ordered),
                    "selected_count": per_room,
                    "selected_sorted_indices": indices,
                }
            )
    return subset, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/AR/unseen_eval.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_subset544",
    )
    parser.add_argument("--per-room", type=int, default=32)
    parser.add_argument("--expected-rooms", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with source_path.open() as handle:
        source = json.load(handle)
    subset, rooms = build_subset(source, args.per_room)
    if len(rooms) != args.expected_rooms:
        raise AssertionError(f"Expected {args.expected_rooms} rooms, found {len(rooms)}")

    expected_total = args.per_room * args.expected_rooms
    actual_total = sum(room["selected_count"] for room in rooms)
    if actual_total != expected_total:
        raise AssertionError(f"Expected {expected_total} samples, selected {actual_total}")

    subset_path = output_dir / "unseen_eval_subset32_per_room.json"
    subset_path.write_text(json.dumps(subset, indent=2) + "\n")
    manifest = {
        "format_version": 1,
        "selection": "uniformly spaced indices over each room's sorted filename list",
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "subset": str(subset_path),
        "subset_sha256": sha256(subset_path),
        "per_room": args.per_room,
        "room_count": len(rooms),
        "sample_count": actual_total,
        "rooms": rooms,
    }
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {subset_path}")
    print(f"Wrote {manifest_path}")
    print(f"rooms={len(rooms)} samples={actual_total} subset_sha256={manifest['subset_sha256']}")


if __name__ == "__main__":
    main()
