#!/usr/bin/env python3
"""Bind FewshotRiR model contexts to an existing frozen candidate protocol."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.engine import (
    filter_frozen_query_candidates,
    reconstruct_room_base_candidates,
)
from src.localization.pilot import canonical_sha256, load_pilot_manifest


IDENTITY_FIELDS = (
    "index",
    "query_id",
    "scene",
    "room",
    "filename",
    "source_global",
    "receiver_global",
)


def _load_hashed_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(content):
        raise ValueError(f"{label} SHA-256 mismatch")
    return payload


def _hashed(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-context-manifest", type=Path, required=True)
    parser.add_argument("--reference-context-manifest", type=Path, required=True)
    parser.add_argument("--reference-geometry-audit", type=Path, required=True)
    parser.add_argument("--reference-pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_context = load_context_manifest(args.model_context_manifest)
    reference_context = load_context_manifest(args.reference_context_manifest)
    reference_geometry = _load_hashed_json(
        args.reference_geometry_audit, "reference geometry audit"
    )
    reference_pilot = load_pilot_manifest(args.reference_pilot_manifest)
    if reference_geometry["context_manifest_sha256"] != reference_context["sha256"]:
        raise ValueError("reference geometry and context manifests differ")
    if (
        reference_pilot["context_manifest_sha256"] != reference_context["sha256"]
        or reference_pilot["geometry_audit_sha256"] != reference_geometry["sha256"]
    ):
        raise ValueError("reference pilot is not bound to the reference geometry")

    reference_by_index = {
        int(record["index"]): record for record in reference_context["records"]
    }
    rebound_records = []
    for model_record in model_context["records"]:
        index = int(model_record["index"])
        reference_record = reference_by_index.get(index)
        if reference_record is None:
            raise ValueError(f"reference context is missing query index {index}")
        if any(model_record[field] != reference_record[field] for field in IDENTITY_FIELDS):
            raise ValueError(f"query identity differs at index {index}")
        rebound = copy.deepcopy(model_record)
        rebound["candidate_filter_context_sources_global"] = copy.deepcopy(
            reference_record["context_sources_global"]
        )
        rebound_records.append(rebound)

    provenance = {
        "semantics": (
            "model inputs use FewshotRiR near-coincident contexts; candidate filtering "
            "uses the frozen reference contexts only to reproduce the shared grid"
        ),
        "model_context_manifest_sha256": model_context["sha256"],
        "reference_context_manifest_sha256": reference_context["sha256"],
        "reference_geometry_audit_sha256": reference_geometry["sha256"],
        "reference_pilot_manifest_sha256": reference_pilot["sha256"],
    }
    rebound_context = copy.deepcopy(model_context)
    rebound_context["records"] = rebound_records
    rebound_context["candidate_protocol_rebinding"] = provenance
    rebound_context = _hashed(rebound_context)

    output_dir = args.output_dir.resolve()
    context_path = output_dir / "context_manifest_fewshot_shared_candidates.json"
    geometry_path = output_dir / "geometry_audit_shared_candidates.json"
    pilot_path = output_dir / "pilot_shared_candidates.json"

    rebound_geometry = copy.deepcopy(reference_geometry)
    rebound_geometry["context_manifest"] = str(context_path)
    rebound_geometry["context_manifest_sha256"] = rebound_context["sha256"]
    rebound_geometry["candidate_protocol_rebinding"] = provenance
    rebound_geometry = _hashed(rebound_geometry)

    rebound_pilot = copy.deepcopy(reference_pilot)
    rebound_pilot["context_manifest_sha256"] = rebound_context["sha256"]
    rebound_pilot["geometry_audit_sha256"] = rebound_geometry["sha256"]
    rebound_pilot["candidate_protocol_rebinding"] = provenance
    rebound_pilot = _hashed(rebound_pilot)

    model_by_index = {int(record["index"]): record for record in rebound_records}
    reference_room_bases: dict[str, np.ndarray] = {}
    rebound_room_bases: dict[str, np.ndarray] = {}
    verified_pairs = 0
    for selected in reference_pilot["records"]:
        index = int(selected["index"])
        reference_record = reference_by_index[index]
        rebound_record = model_by_index[index]
        room = selected["room"]
        if room not in reference_room_bases:
            reference_room_bases[room] = reconstruct_room_base_candidates(
                room, reference_geometry
            )
            rebound_room_bases[room] = reconstruct_room_base_candidates(
                room, rebound_geometry
            )
        reference_candidates = filter_frozen_query_candidates(
            reference_record, reference_geometry, reference_room_bases[room]
        )
        rebound_candidates = filter_frozen_query_candidates(
            rebound_record, rebound_geometry, rebound_room_bases[room]
        )
        if not np.array_equal(reference_candidates, rebound_candidates):
            raise RuntimeError(f"candidate coordinates differ at query index {index}")
        if len(rebound_candidates) != int(selected["candidate_count"]):
            raise RuntimeError(f"candidate count differs at query index {index}")
        verified_pairs += len(rebound_candidates)
    if verified_pairs != int(reference_pilot["candidate_query_pairs"]):
        raise RuntimeError("verified candidate-pair total differs from reference pilot")

    _atomic_json(context_path, rebound_context)
    _atomic_json(geometry_path, rebound_geometry)
    _atomic_json(pilot_path, rebound_pilot)
    verification = _hashed(
        {
            "schema_version": 1,
            "query_count": int(reference_pilot["query_count"]),
            "room_count": int(reference_pilot["room_count"]),
            "candidate_query_pairs": verified_pairs,
            "coordinate_equality": "bit_exact_float64",
            "rebound_context_manifest_sha256": rebound_context["sha256"],
            "rebound_geometry_audit_sha256": rebound_geometry["sha256"],
            "rebound_pilot_manifest_sha256": rebound_pilot["sha256"],
            **provenance,
        }
    )
    _atomic_json(output_dir / "candidate_alignment_verification.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
