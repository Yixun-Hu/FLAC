#!/usr/bin/env python3
"""Run the material-blind Few-ShotRIR-Waveform or FEM-Sabine baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.baseline_experiment import run_baseline_localization
from src.localization.pilot import load_pilot_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("few_shot_rir_waveform", "fem_sabine"),
        required=True,
    )
    parser.add_argument("--agree-ckpt", type=Path)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--tetra-mesh-manifest", type=Path)
    parser.add_argument(
        "--fem-solver-backend",
        choices=("auto", "superlu", "mkl_pardiso"),
        default="auto",
    )
    parser.add_argument(
        "--fem-superlu-ordering",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default="MMD_AT_PLUS_A",
    )
    parser.add_argument("--fem-solver-threads", type=int, default=1)
    parser.add_argument("--mkl-runtime", type=Path)
    args = parser.parse_args()

    if args.mkl_runtime is not None:
        runtime = args.mkl_runtime.resolve()
        if not runtime.is_file():
            raise FileNotFoundError(runtime)
        os.environ["MKL_RT"] = str(runtime)

    output = args.output_dir.resolve()
    try:
        output.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "baseline output must stay inside the NeuriPs_Workshop worktree"
        ) from error
    result = run_baseline_localization(
        method=args.method,
        agree_checkpoint_path=args.agree_ckpt,
        context_manifest=load_context_manifest(args.context_manifest),
        geometry_audit=json.loads(args.geometry_audit.read_text()),
        pilot_manifest=load_pilot_manifest(args.pilot_manifest),
        dataset_root=args.dataset_root,
        output_dir=output,
        device=args.device,
        candidate_batch_size=args.candidate_batch_size,
        random_seed=args.random_seed,
        query_limit=args.query_limit,
        model_config_path=args.model_config,
        checkpoint_path=args.ckpt_path,
        tetra_manifest_path=args.tetra_mesh_manifest,
        fem_solver_backend=args.fem_solver_backend,
        fem_superlu_ordering=args.fem_superlu_ordering,
        fem_solver_threads=args.fem_solver_threads,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
