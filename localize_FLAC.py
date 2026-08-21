#!/usr/bin/env python3
"""Run the frozen 64-query exp_09 localization pilot safely and resumably."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import load_pilot_manifest
from src.localization.runner import run_localization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--agree-ckpt", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cond-method", choices=("vanilla", "fa_invariant"), required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--query-limit", type=int)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    try:
        output.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("localization output must stay inside the NeuriPs_Workshop worktree") from error
    context = load_context_manifest(args.context_manifest)
    audit = json.loads(args.geometry_audit.read_text())
    pilot = load_pilot_manifest(args.pilot_manifest)
    result = run_localization(
        model_config_path=args.model_config,
        checkpoint_path=args.ckpt_path,
        agree_checkpoint_path=args.agree_ckpt,
        context_manifest=context,
        geometry_audit=audit,
        pilot_manifest=pilot,
        dataset_root=args.dataset_root,
        output_dir=output,
        device=args.device,
        cond_method=args.cond_method,
        candidate_batch_size=args.candidate_batch_size,
        sample_seed=args.sample_seed,
        tau=args.tau,
        query_limit=args.query_limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
