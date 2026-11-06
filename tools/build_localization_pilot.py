#!/usr/bin/env python3
"""Freeze the room-stratified 64-query localization pilot manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import (
    build_pilot_manifest,
    load_pilot_manifest,
    save_pilot_manifest,
)


def same_queries_reasons(manifest: dict, reference: dict) -> list[str]:
    """exp_37: a rebuilt pilot (e.g. on the 0.75 m audit) must select exactly the reference's queries, in order."""
    ours = [(int(r["index"]), r["query_id"]) for r in manifest["records"]]
    theirs = [(int(r["index"]), r["query_id"]) for r in reference["records"]]
    if ours == theirs:
        return []
    missing = sorted(set(theirs) - set(ours))[:5]
    return [f"the rebuilt pilot selects other queries than the reference ({len(set(theirs) - set(ours))} of "
            f"{len(theirs)} reference queries missing, e.g. {missing}; order equal: {sorted(ours) == sorted(theirs)})"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-room", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-pilot-manifest", type=Path)
    parser.add_argument("--require-same-queries-as", type=Path)                # exp_37: refuse other queries
    args = parser.parse_args()
    context = load_context_manifest(args.context_manifest)
    audit = json.loads(args.geometry_audit.read_text())
    excluded = (
        load_pilot_manifest(args.exclude_pilot_manifest)
        if args.exclude_pilot_manifest is not None
        else None
    )
    manifest = build_pilot_manifest(
        context,
        audit,
        queries_per_room=args.queries_per_room,
        seed=args.seed,
        excluded_pilot_manifest=excluded,
    )
    if args.require_same_queries_as is not None:
        reasons = same_queries_reasons(manifest, load_pilot_manifest(args.require_same_queries_as))
        if reasons:
            print("PILOT REBUILD REFUSED (nothing written): " + "; ".join(reasons), file=sys.stderr)
            raise SystemExit(3)
    save_pilot_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "sha256": manifest["sha256"],
                "rooms": manifest["room_count"],
                "queries": manifest["query_count"],
                "candidate_query_pairs": manifest["candidate_query_pairs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
