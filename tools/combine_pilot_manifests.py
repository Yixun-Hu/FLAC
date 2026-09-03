#!/usr/bin/env python3
"""Combine disjoint frozen pilot manifests into one hash-verified selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_pilot_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise ValueError(f"pilot manifest SHA-256 mismatch: {path}")
    return payload


def save_pilot_manifest(payload: dict, path: Path) -> None:
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise ValueError("combined pilot manifest hash is stale")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def combine(manifests: list[dict]) -> dict:
    if len(manifests) < 2:
        raise ValueError("at least two pilot manifests are required")
    reference = manifests[0]
    identity_fields = (
        "context_manifest_sha256",
        "geometry_audit_sha256",
        "geometry_branch",
        "room_count",
    )
    for manifest in manifests[1:]:
        for field in identity_fields:
            if manifest.get(field) != reference.get(field):
                raise ValueError(f"pilot manifest field {field} does not match")

    records = []
    source_hashes = []
    seen = set()
    for manifest in manifests:
        source_hashes.append(manifest["sha256"])
        for record in manifest["records"]:
            index = int(record["index"])
            if index in seen:
                raise ValueError(f"pilot manifests overlap at query {index}")
            seen.add(index)
            records.append(record)
    records.sort(key=lambda row: (row["room"], int(row["index"])))
    room_counts = Counter(row["room"] for row in records)
    if len(room_counts) != int(reference["room_count"]):
        raise ValueError("combined selection does not cover the expected rooms")
    if len(set(room_counts.values())) != 1:
        raise ValueError(f"combined selection is not room-balanced: {dict(room_counts)}")

    body = {
        "schema_version": 1,
        "selection": {
            "method": "verified_union_of_disjoint_frozen_pilot_manifests",
            "source_manifest_sha256": source_hashes,
            "queries_per_room": next(iter(room_counts.values())),
        },
        "context_manifest_sha256": reference["context_manifest_sha256"],
        "geometry_audit_sha256": reference["geometry_audit_sha256"],
        "geometry_branch": reference["geometry_branch"],
        "room_count": len(room_counts),
        "query_count": len(records),
        "candidate_query_pairs": int(sum(int(row["candidate_count"]) for row in records)),
        "records": records,
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = combine([load_pilot_manifest(path.resolve()) for path in args.manifest])
    combined["sha256"] = canonical_sha256(combined)
    save_pilot_manifest(combined, args.output.resolve())


if __name__ == "__main__":
    main()
