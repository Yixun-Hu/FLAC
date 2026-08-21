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
from src.localization.pilot import build_pilot_manifest, save_pilot_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-room", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    context = load_context_manifest(args.context_manifest)
    audit = json.loads(args.geometry_audit.read_text())
    manifest = build_pilot_manifest(
        context,
        audit,
        queries_per_room=args.queries_per_room,
        seed=args.seed,
    )
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
