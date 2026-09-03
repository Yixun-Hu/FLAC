#!/usr/bin/env python3
"""Evaluate FewshotRiR with its frozen direct coordinate readout (K=8, Q=1)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.baseline_experiment import load_fewshot_rir_checkpoint
from src.localization.fewshot_rir import load_fewshot_rir_query
from src.localization.fewshot_rir_readout import (
    READOUT_CONTEXT_COUNT,
    READOUT_GENERATION_COUNT,
    PAPER_REFERENCE_CONTEXT_COUNT,
    READOUT_PROTOCOL_NAME,
    build_readout_query_result,
    infer_fewshot_rir_readout_query,
    summarize_readout_results,
)
from src.localization.pilot import load_pilot_manifest, resolve_pilot_records
from src.localization.rir_localizer import load_rir_localizer_checkpoint
from src.localization.runner import (
    _atomic_json,
    _hashed_payload,
    file_sha256,
    initialize_run,
    verify_hashed_payload,
)


def _assert_matching_spectrogram_contracts(
    generator_config: dict,
    localizer_config: dict,
) -> None:
    generator_model = dict(generator_config.get("model", {}))
    localizer_preprocessing = dict(localizer_config.get("preprocessing", {}))
    pairs = {
        "sample_size": (
            generator_config.get("sample_size"),
            localizer_config.get("sample_size"),
        ),
        "sample_rate": (
            generator_config.get("sample_rate"),
            localizer_config.get("sample_rate"),
        ),
        "n_fft": (
            generator_model.get("n_fft", 511),
            localizer_preprocessing.get("n_fft", 511),
        ),
        "hop_length": (
            generator_model.get("hop_length", 40),
            localizer_preprocessing.get("hop_length", 40),
        ),
        "win_length": (
            generator_model.get("win_length", 248),
            localizer_preprocessing.get("win_length", 248),
        ),
        "log_epsilon": (
            generator_model.get("log_epsilon", 1e-8),
            localizer_preprocessing.get("log_epsilon", 1e-8),
        ),
    }
    mismatched = {
        key: values for key, values in pairs.items() if values[0] != values[1]
    }
    if mismatched:
        raise ValueError(
            f"FewshotRiR/localizer spectrogram contracts differ: {mismatched}"
        )


def _completed_query(
    output_dir: Path,
    *,
    query_index: int,
    query_id: str,
    run_sha256: str,
) -> dict | None:
    path = output_dir / "queries" / f"query_{query_index:05d}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    verify_hashed_payload(result, f"readout query {query_index}")
    if (
        result.get("schema_version") != 1
        or result.get("query_index") != query_index
        or result.get("query_id") != query_id
        or result.get("run_manifest_sha256") != run_sha256
        or result.get("protocol", {}).get("context_count") != READOUT_CONTEXT_COUNT
        or result.get("protocol", {}).get("generation_count")
        != READOUT_GENERATION_COUNT
    ):
        raise RuntimeError(f"readout query {query_index} resume identity mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model-config", type=Path, required=True)
    parser.add_argument("--generator-ckpt", type=Path, required=True)
    parser.add_argument("--localizer-ckpt", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--synchronize-latency", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("readout output must stay inside this worktree") from error
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    required_files = (
        args.generator_model_config,
        args.generator_ckpt,
        args.localizer_ckpt,
        args.context_manifest,
        args.geometry_audit,
        args.pilot_manifest,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.query_limit is not None and args.query_limit <= 0:
        raise ValueError("query_limit must be positive")

    context_manifest = load_context_manifest(args.context_manifest)
    geometry_audit = json.loads(args.geometry_audit.read_text())
    verify_hashed_payload(geometry_audit, "geometry audit")
    pilot_manifest = load_pilot_manifest(args.pilot_manifest)
    joined = resolve_pilot_records(
        pilot_manifest, context_manifest, geometry_audit
    )
    if args.query_limit is not None:
        joined = joined[: args.query_limit]
    if not joined:
        raise ValueError("the selected pilot contains no queries")

    generator, generator_config = load_fewshot_rir_checkpoint(
        args.generator_model_config, args.generator_ckpt, device
    )
    localizer, gt_transform, localizer_bundle = load_rir_localizer_checkpoint(
        args.localizer_ckpt, device
    )
    localizer_config = localizer_bundle["model_config"]
    _assert_matching_spectrogram_contracts(generator_config, localizer_config)

    identity = {
        "task": "fewshotrir_direct_coordinate_readout",
        "protocol": READOUT_PROTOCOL_NAME,
        "context_count": READOUT_CONTEXT_COUNT,
        "generation_count": READOUT_GENERATION_COUNT,
        "paper_reference_context_count": PAPER_REFERENCE_CONTEXT_COUNT,
        "directly_comparable_to_paper_n20_sle": False,
        "query_source_input": "ground_truth_continuous_coordinate",
        "candidate_search": False,
        "agree_scoring": False,
        "griffin_lim": False,
        "dataset_root": str(dataset_root),
        "generator_model_config": str(args.generator_model_config.resolve()),
        "generator_model_config_sha256": file_sha256(args.generator_model_config),
        "generator_checkpoint": str(args.generator_ckpt.resolve()),
        "generator_checkpoint_sha256": file_sha256(args.generator_ckpt),
        "localizer_checkpoint": str(args.localizer_ckpt.resolve()),
        "localizer_checkpoint_sha256": file_sha256(args.localizer_ckpt),
        "localizer_training_run_sha256": localizer_bundle[
            "run_manifest_sha256"
        ],
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_limit": args.query_limit,
        "synchronize_latency": bool(args.synchronize_latency),
    }
    run_manifest = initialize_run(output_dir, identity)
    print(
        "NOTICE: this is the frozen AR K=8 protocol; the Few-ShotRIR paper "
        "SLE table uses N=20, so the numbers are not directly comparable.",
        file=sys.stderr,
        flush=True,
    )
    run_sha256 = str(run_manifest["sha256"])
    generator_model = dict(generator_config["model"])
    adaptation = dict(generator_config.get("adaptation", {}))
    results = []

    for selected, record, _geometry in joined:
        query_index = int(selected["index"])
        completed = _completed_query(
            output_dir,
            query_index=query_index,
            query_id=selected["query_id"],
            run_sha256=run_sha256,
        )
        if completed is not None:
            results.append(completed)
            continue
        started = time.perf_counter()
        observed, context_metadata = load_fewshot_rir_query(
            record,
            dataset_root,
            sample_rate=int(generator_config["sample_rate"]),
            sample_size=int(generator_config["sample_size"]),
            n_fft=int(generator_model.get("n_fft", 511)),
            hop_length=int(generator_model.get("hop_length", 40)),
            win_length=int(generator_model.get("win_length", 248)),
            depth_size=tuple(adaptation.get("depth_size", (128, 256))),
            depth_max_m=float(adaptation.get("depth_max_m", 67.16327)),
        )
        inference = infer_fewshot_rir_readout_query(
            generator,
            localizer,
            gt_transform,
            observed_waveform=observed,
            context_metadata=context_metadata,
            source_global=record["source_global"],
            receiver_global=record["receiver_global"],
            device=device,
            synchronize_timing=args.synchronize_latency,
        )
        elapsed = time.perf_counter() - started
        result = build_readout_query_result(
            query_index=query_index,
            query_id=selected["query_id"],
            scene=selected["scene"],
            room=selected["room"],
            receiver_id=selected["receiver_id"],
            source_global=record["source_global"],
            receiver_global=record["receiver_global"],
            inference=inference,
            run_manifest_sha256=run_sha256,
            elapsed_seconds=elapsed,
        )
        result = _hashed_payload(result)
        _atomic_json(
            output_dir / "queries" / f"query_{query_index:05d}.json", result
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "query_index": query_index,
                    "query_id": selected["query_id"],
                    "generated_euclidean_error_m": result[
                        "generated_rir_readout"
                    ]["metrics"]["euclidean_error_m"],
                    "gt_euclidean_error_m": result["ground_truth_rir_readout"][
                        "metrics"
                    ]["euclidean_error_m"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = summarize_readout_results(
        results, run_manifest_sha256=run_sha256
    )
    summary["generator_checkpoint_sha256"] = identity[
        "generator_checkpoint_sha256"
    ]
    summary["localizer_checkpoint_sha256"] = identity[
        "localizer_checkpoint_sha256"
    ]
    summary = _hashed_payload(summary)
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
