#!/usr/bin/env python3
"""Build a paired-query manifest from a frozen Depth-AABB coverage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import (
    canonical_sha256,
    load_pilot_manifest,
    save_pilot_manifest,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--source-pilot", type=Path, action="append", required=True)
    parser.add_argument("--exclude-room", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilots = [load_pilot_manifest(path) for path in args.source_pilot]
    if not pilots:
        raise ValueError("at least one source pilot is required")
    identity_fields = ("context_manifest_sha256", "geometry_audit_sha256", "geometry_branch")
    for field in identity_fields:
        if len({pilot[field] for pilot in pilots}) != 1:
            raise ValueError(f"source pilot mismatch: {field}")

    source_records = []
    seen_indices = set()
    for pilot in pilots:
        for record in pilot["records"]:
            index = int(record["index"])
            if index in seen_indices:
                raise ValueError(f"duplicate source query index: {index}")
            seen_indices.add(index)
            source_records.append(record)

    audit = json.loads(args.coverage_audit.read_text())
    if audit.get("candidate_policy") != (
        "frozen and identical; no candidate filtering or score substitution"
    ):
        raise ValueError("coverage audit does not preserve the candidate contract")
    audit_rows = {int(row["query_index"]): row for row in audit["queries"]}
    if set(audit_rows) != seen_indices:
        raise ValueError("coverage audit and source pilot query sets differ")

    excluded_rooms = set(args.exclude_room)
    eligible_source = [record for record in source_records if record["room"] not in excluded_rooms]
    selected = []
    rejected = []
    for record in eligible_source:
        row = audit_rows[int(record["index"])]
        if row["query_id"] != record["query_id"]:
            raise ValueError(f"query identity mismatch at index {record['index']}")
        if int(row["candidate_count"]) != int(record["candidate_count"]):
            raise ValueError(f"candidate count mismatch at index {record['index']}")
        if row["depth_aabb_strict_gate"]:
            selected.append(record)
        else:
            rejected.append(
                {
                    "index": int(record["index"]),
                    "query_id": record["query_id"],
                    "room": record["room"],
                    "candidate_count": int(record["candidate_count"]),
                }
            )

    payload = {
        "schema_version": 1,
        "selection": {
            "method": "depth_aabb_strict_complete_query_intersection",
            "candidate_policy": audit["candidate_policy"],
            "required_points": "receiver, target source, every context source, every candidate",
            "coverage_audit_path": str(args.coverage_audit.resolve()),
            "coverage_audit_file_sha256": file_sha256(args.coverage_audit),
            "source_pilot_manifest_sha256": [pilot["sha256"] for pilot in pilots],
            "excluded_rooms": sorted(excluded_rooms),
            "source_query_count_after_room_exclusion": len(eligible_source),
            "strict_gate_rejected_query_count": len(rejected),
            "strict_gate_rejected_queries": rejected,
            "strict_query_coverage_fraction": len(selected) / len(eligible_source),
        },
        "context_manifest_sha256": pilots[0]["context_manifest_sha256"],
        "geometry_audit_sha256": pilots[0]["geometry_audit_sha256"],
        "geometry_branch": pilots[0]["geometry_branch"],
        "room_count": len({record["room"] for record in selected}),
        "query_count": len(selected),
        "candidate_query_pairs": int(sum(record["candidate_count"] for record in selected)),
        "records": selected,
    }
    payload["sha256"] = canonical_sha256(payload)
    save_pilot_manifest(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": payload["sha256"],
                "room_count": payload["room_count"],
                "query_count": payload["query_count"],
                "candidate_query_pairs": payload["candidate_query_pairs"],
                "source_query_count": len(eligible_source),
                "rejected_query_count": len(rejected),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
