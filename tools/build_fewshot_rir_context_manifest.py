#!/usr/bin/env python3
"""Freeze the nearest-endpoint context protocol used by FewshotRiR localization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest, save_context_manifest
from src.localization.fewshot_rir_contexts import build_fewshot_rir_context_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-context-manifest", type=Path, required=True)
    parser.add_argument("--context-inventory", type=Path, default=Path("data/AR/all_data.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_fewshot_rir_context_manifest(
        load_context_manifest(args.base_context_manifest),
        context_inventory_path=args.context_inventory,
        dataset_root=args.dataset_root,
        max_context=args.max_context,
        seed=args.seed,
    )
    save_context_manifest(manifest, args.output)
    print(f"wrote {len(manifest['records'])} FewshotRiR records to {args.output}")


if __name__ == "__main__":
    main()
