"""Build frozen nearest-endpoint contexts for FewshotRiR localization."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.data.fewshot_rir import (
    load_ar_inventory,
    load_ar_positions,
    select_near_coincident_contexts,
)
from src.localization.ar_queries import _canonical_sha
from src.localization.fewshot_rir import NEAR_CONTEXT_PROTOCOL


def build_fewshot_rir_context_manifest(
    base_manifest: dict,
    *,
    context_inventory_path: Path | str,
    dataset_root: Path | str,
    max_context: int = 8,
    seed: int = 42,
) -> dict:
    """Replace legacy contexts with deterministic nearest-endpoint AR RIRs."""

    if max_context <= 0:
        raise ValueError("max_context must be positive")
    if not isinstance(base_manifest.get("records"), list) or not base_manifest["records"]:
        raise ValueError("base context manifest contains no query records")
    source_hash = base_manifest.get("sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("base context manifest must have a verified SHA-256")
    inventory = load_ar_inventory(context_inventory_path)
    dataset_root = Path(dataset_root)
    room_pools: dict[tuple[str, str], tuple[str, ...]] = {}

    records = []
    for original in base_manifest["records"]:
        record = copy.deepcopy(original)
        key = (str(record["scene"]), str(record["room"]))
        if key not in inventory:
            raise ValueError(f"no context inventory for {key[0]}/{key[1]}")
        if key not in room_pools:
            room_pools[key] = select_near_coincident_contexts(
                dataset_root,
                key[0],
                key[1],
                inventory[key],
            )
        query_id = str(record["query_id"])
        candidates = [
            name
            for name in room_pools[key]
            if str(Path("single_channel_ir_1") / key[0] / key[1] / name) != query_id
        ]
        if not candidates:
            raise ValueError(f"no non-target near contexts for {key[0]}/{key[1]}")
        index = int(record["index"])
        rng = np.random.default_rng(int(seed) + index)
        if max_context <= len(candidates):
            selected_indices = rng.choice(
                len(candidates), size=int(max_context), replace=False
            )
        else:
            first = rng.choice(
                len(candidates), size=len(candidates), replace=False
            )
            repeated = rng.choice(
                len(candidates), size=max_context - len(candidates), replace=True
            )
            selected_indices = np.concatenate((first, repeated))
        filenames = [candidates[int(position)] for position in selected_indices]
        relpaths = [
            str(Path("single_channel_ir_1") / key[0] / key[1] / filename)
            for filename in filenames
        ]
        locations = []
        distances = []
        for filename, relpath in zip(filenames, relpaths):
            audio_path = dataset_root / relpath
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            source, receiver = load_ar_positions(dataset_root, key[0], key[1], filename)
            locations.append((source, receiver))
            distances.append(float(np.linalg.norm(source - receiver)))
        record["eligible_context_count"] = len(candidates)
        record["contexts"] = relpaths
        record["context_sources_global"] = [source.tolist() for source, _ in locations]
        record["context_receivers_global"] = [receiver.tolist() for _, receiver in locations]
        record["context_endpoint_distances_m"] = distances
        record["context_protocol"] = NEAR_CONTEXT_PROTOCOL
        records.append(record)

    payload = {
        "schema_version": 2,
        "protocol": {
            "name": NEAR_CONTEXT_PROTOCOL,
            "seed": int(seed),
            "max_context": int(max_context),
            "sampling": "all_available_without_replacement_then_repeat_shortage",
            "selection": "minimum_endpoint_distance_available_receiver_per_source",
            "anchor": "first_ordered_context_receiver",
            "pose": "receiver_xyz_then_source_xyz_relative_to_anchor",
            "unavoidable_adaptation": "AcousticRooms_contains_no_colocated_emit_receive_RIRs",
        },
        "source_manifest_sha256": source_hash,
        "context_inventory_path": str(context_inventory_path),
        "query_count": len(records),
        "records": records,
    }
    payload["sha256"] = _canonical_sha(payload)
    return payload
