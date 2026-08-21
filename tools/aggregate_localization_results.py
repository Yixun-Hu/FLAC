#!/usr/bin/env python3
"""Validate and aggregate the two completed 64-query pilot arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import load_pilot_manifest
from src.localization.reporting import aggregate_pilot, save_aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--fa-bf-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json.resolve(), args.output_md.resolve()):
        try:
            output.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError("aggregate outputs must stay inside NeuriPs_Workshop") from error
    pilot = load_pilot_manifest(args.pilot_manifest)
    aggregate = aggregate_pilot(
        pilot,
        {"vanilla": args.vanilla_dir, "fa_bf": args.fa_bf_dir},
    )
    save_aggregate(aggregate, args.output_json, args.output_md)
    print(json.dumps({"sha256": aggregate["sha256"], "queries": aggregate["query_count"]}, indent=2))


if __name__ == "__main__":
    main()
