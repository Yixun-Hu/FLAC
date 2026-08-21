"""Deterministic room-stratified pilot manifests for exp_09."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def build_pilot_manifest(
    context_manifest: dict,
    geometry_audit: dict,
    *,
    queries_per_room: int = 4,
    seed: int = 42,
) -> dict:
    """Select a fixed number of target queries from every audited room."""

    if queries_per_room <= 0:
        raise ValueError("queries_per_room must be positive")
    if geometry_audit.get("geometry_gate") != "PASS":
        raise ValueError("pilot selection requires a passed geometry audit")
    if geometry_audit.get("context_manifest_sha256") != context_manifest.get("sha256"):
        raise ValueError("context manifest and geometry audit do not match")

    context_by_index = {int(item["index"]): item for item in context_manifest["records"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for query in geometry_audit["queries"]:
        index = int(query["index"])
        record = context_by_index.get(index)
        if record is None or record["query_id"] != query["query_id"]:
            raise ValueError("geometry query is absent from the frozen context manifest")
        grouped[query["room"]].append(query)

    expected_rooms = int(geometry_audit["included_rooms"])
    if len(grouped) != expected_rooms:
        raise ValueError(f"expected {expected_rooms} audited rooms, got {len(grouped)}")

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for room in sorted(grouped):
        pool = sorted(grouped[room], key=lambda item: int(item["index"]))
        if len(pool) < queries_per_room:
            raise ValueError(f"room {room} has fewer than {queries_per_room} queries")
        positions = np.sort(rng.choice(len(pool), size=queries_per_room, replace=False))
        for position in positions:
            query = pool[int(position)]
            selected.append(
                {
                    "index": int(query["index"]),
                    "query_id": query["query_id"],
                    "scene": query["scene"],
                    "room": query["room"],
                    "receiver_id": query["receiver_id"],
                    "candidate_count": int(query["chosen_count"]),
                    "candidate_indices_sha256": query[
                        "z_indices_sha256"
                        if geometry_audit["z_branch"] == "z_band"
                        else "full_indices_sha256"
                    ],
                    "oracle_m": float(
                        query["z_oracle_m"]
                        if geometry_audit["z_branch"] == "z_band"
                        else query["full_oracle_m"]
                    ),
                }
            )

    payload = {
        "schema_version": 1,
        "selection": {
            "method": "room_stratified_without_replacement",
            "seed": int(seed),
            "queries_per_room": int(queries_per_room),
            "rng": "numpy.default_rng.PCG64",
        },
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "geometry_branch": geometry_audit["z_branch"],
        "room_count": len(grouped),
        "query_count": len(selected),
        "candidate_query_pairs": int(sum(item["candidate_count"] for item in selected)),
        "records": selected,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def save_pilot_manifest(manifest: dict, path: Path | str) -> None:
    path = Path(path)
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_sha256(content):
        raise ValueError("pilot manifest hash is stale or invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_pilot_manifest(path: Path | str) -> dict:
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256", None)
    actual = canonical_sha256(manifest)
    if expected != actual:
        raise ValueError("pilot manifest SHA-256 mismatch")
    manifest["sha256"] = expected
    return manifest


def resolve_pilot_records(
    pilot_manifest: dict,
    context_manifest: dict,
    geometry_audit: dict,
) -> list[tuple[dict, dict, dict]]:
    """Join pilot, context, and geometry rows while checking frozen identities."""

    if pilot_manifest["context_manifest_sha256"] != context_manifest["sha256"]:
        raise ValueError("pilot/context manifest mismatch")
    if pilot_manifest["geometry_audit_sha256"] != geometry_audit["sha256"]:
        raise ValueError("pilot/geometry audit mismatch")
    context = {int(item["index"]): item for item in context_manifest["records"]}
    geometry = {int(item["index"]): item for item in geometry_audit["queries"]}
    joined = []
    for selected in pilot_manifest["records"]:
        index = int(selected["index"])
        context_row = context.get(index)
        geometry_row = geometry.get(index)
        if context_row is None or geometry_row is None:
            raise ValueError(f"pilot query index {index} is missing")
        if not (
            selected["query_id"]
            == context_row["query_id"]
            == geometry_row["query_id"]
        ):
            raise ValueError(f"pilot query identity mismatch at index {index}")
        if int(selected["candidate_count"]) != int(geometry_row["chosen_count"]):
            raise ValueError(f"pilot candidate count mismatch at index {index}")
        joined.append((selected, context_row, geometry_row))
    if len(joined) != int(pilot_manifest["query_count"]):
        raise ValueError("pilot query count is inconsistent")
    return joined
