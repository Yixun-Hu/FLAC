#!/usr/bin/env python3
"""Write the exp_09 seed-42 original-loader context manifest."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import save_context_manifest
from src.localization.context_materializer import materialize_original_contexts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = materialize_original_contexts(args.dataset_config, args.dataset_root)
    save_context_manifest(manifest, args.output)
    print(f"saved {manifest['full_query_count']} contexts: {args.output}")
    print(f"sha256 {manifest['sha256']}")


if __name__ == "__main__":
    main()
