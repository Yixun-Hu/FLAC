#!/usr/bin/env python3
"""Benchmark AGREE scoring and FEM-OMP without rerunning acoustic generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.baseline_runner import score_fem_room_helps_candidates
from src.localization.engine import encode_audio_features, load_agree_retrieval, load_frozen_query
from src.localization.pilot import canonical_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt SHA-256 in {path}")
    return payload


def median_fields(repeats: list[dict], fields: tuple[str, ...]) -> dict:
    return {
        field: float(statistics.median(float(row[field]) for row in repeats))
        for field in fields
    }


@torch.inference_mode()
def benchmark_agree_query(
    retrieval,
    observed: torch.Tensor,
    *,
    candidate_count: int,
    batch_size: int,
    device: str,
    repeat_count: int,
) -> dict:
    device_type = torch.device(device).type
    repeats = []
    for _repeat in range(repeat_count):
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        observed_started = time.perf_counter()
        observed_gpu = observed.unsqueeze(0).to(device=device, dtype=torch.float32)
        observed_features = encode_audio_features(retrieval, observed_gpu)
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        observed_seconds = time.perf_counter() - observed_started

        generated_encode_seconds = 0.0
        similarity_seconds = 0.0
        score_chunks = []
        for start in range(0, candidate_count, batch_size):
            count = min(batch_size, candidate_count - start)
            # A generated waveform is already resident on the accelerator. AGREE
            # compute is shape-dependent, so zeros avoid rerunning any generator.
            generated = torch.zeros((count, 1, 10240), dtype=torch.float32, device=device)
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            encode_started = time.perf_counter()
            features = encode_audio_features(retrieval, generated)
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            generated_encode_seconds += time.perf_counter() - encode_started

            similarity_started = time.perf_counter()
            scores = features @ observed_features.T
            score_chunks.append(torch.atleast_1d(scores).reshape(-1))
            similarity_seconds += time.perf_counter() - similarity_started

        selection_started = time.perf_counter()
        all_scores = torch.cat(score_chunks)
        if len(all_scores) != candidate_count:
            raise RuntimeError("AGREE score count mismatch")
        _winner = int(torch.argmax(all_scores).item())
        selection_seconds = time.perf_counter() - selection_started
        repeats.append(
            {
                "observed_encode_seconds": observed_seconds,
                "generated_encode_seconds": generated_encode_seconds,
                "similarity_seconds": similarity_seconds,
                "argmax_seconds": selection_seconds,
                "scoring_total_seconds": (
                    observed_seconds
                    + generated_encode_seconds
                    + similarity_seconds
                    + selection_seconds
                ),
            }
        )
    fields = (
        "observed_encode_seconds",
        "generated_encode_seconds",
        "similarity_seconds",
        "argmax_seconds",
        "scoring_total_seconds",
    )
    return {
        "repeat_seconds": {field: [row[field] for row in repeats] for field in fields},
        "median_seconds": median_fields(repeats, fields),
    }


def benchmark_omp_query(
    response: np.ndarray,
    bin_indices: np.ndarray,
    observed: torch.Tensor,
    *,
    repeat_count: int,
) -> dict:
    repeats = []
    winner = None
    for _repeat in range(repeat_count):
        started = time.perf_counter()
        _scores, recovery = score_fem_room_helps_candidates(
            response,
            bin_indices,
            observed,
            sample_count=int(observed.shape[-1]),
        )
        seconds = time.perf_counter() - started
        repeats.append(seconds)
        current = int(recovery.support[0])
        if winner is not None and current != winner:
            raise RuntimeError("OMP winner changed across repeats")
        winner = current
    return {
        "repeat_seconds": repeats,
        "median_seconds": float(statistics.median(repeats)),
        "prediction_index": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--strict-selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--agree-checkpoint", type=Path, required=True)
    parser.add_argument("--fem-response-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_batch_size <= 0 or args.repeat_count <= 0:
        raise ValueError("batch size and repeat count must be positive")

    selection = load_hashed_json(args.selection.resolve())
    strict = load_hashed_json(args.strict_selection.resolve())
    context = load_hashed_json(args.context_manifest.resolve())
    records = {int(row["index"]): row for row in selection["records"]}
    strict_indices = {int(row["index"]) for row in strict["records"]}
    context_records = {int(row["index"]): row for row in context["records"]}
    if len(records) != 128 or len(strict_indices) != 112:
        raise RuntimeError("expected frozen 128-query scope and 112 FEM successes")
    if selection["context_manifest_sha256"] != context["sha256"]:
        raise RuntimeError("selection and context manifest do not match")

    response_dir = args.fem_response_dir.resolve()
    cached_indices = {
        int(path.stem.split("_")[1])
        for path in response_dir.glob("query_*_response.npz")
    }
    actual_indices = strict_indices & cached_indices
    if len(actual_indices) != 97:
        raise RuntimeError(f"expected 97 actual FEM response caches, found {len(actual_indices)}")
    synthetic_indices = strict_indices - actual_indices
    failure_indices = set(records) - strict_indices
    if len(synthetic_indices) != 15 or len(failure_indices) != 16:
        raise RuntimeError("unexpected OMP source accounting")

    reference_path = response_dir / f"query_{min(actual_indices):05d}_response.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_bins = archive["bin_indices"].astype(np.int64)
    frequency_count = len(reference_bins)
    rng = np.random.default_rng(42)

    retrieval = load_agree_retrieval(args.agree_checkpoint.resolve(), args.device)
    # One unreported warm-up for both batch-one and full-batch AGREE paths.
    warmup = torch.zeros((args.candidate_batch_size, 1, 10240), device=args.device)
    _ = encode_audio_features(retrieval, warmup)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)

    rows = []
    for ordinal, index in enumerate(sorted(records), start=1):
        selected = records[index]
        record = context_records[index]
        if selected["query_id"] != record["query_id"]:
            raise RuntimeError(f"query identity mismatch at {index}")
        observed, _metadata = load_frozen_query(record, args.dataset_root.resolve())
        agree = benchmark_agree_query(
            retrieval,
            observed,
            candidate_count=int(selected["candidate_count"]),
            batch_size=args.candidate_batch_size,
            device=args.device,
            repeat_count=args.repeat_count,
        )

        if index in failure_indices:
            omp_source = "not_run_random_fallback"
            omp = {"repeat_seconds": [0.0] * args.repeat_count, "median_seconds": 0.0}
        elif index in actual_indices:
            response_path = response_dir / f"query_{index:05d}_response.npz"
            audit_path = response_dir / f"query_{index:05d}.json"
            audit = load_hashed_json(audit_path)
            if audit["response_file_sha256"] != file_sha256(response_path):
                raise RuntimeError(f"FEM response cache hash mismatch at {index}")
            with np.load(response_path, allow_pickle=False) as archive:
                response = archive["response"]
                bin_indices = archive["bin_indices"]
            omp_source = "actual_fem_response"
            omp = benchmark_omp_query(
                response, bin_indices, observed, repeat_count=args.repeat_count
            )
        else:
            count = int(selected["candidate_count"])
            response = (
                rng.standard_normal((count, frequency_count))
                + 1j * rng.standard_normal((count, frequency_count))
            ).astype(np.complex64)
            omp_source = "shape_matched_synthetic_response"
            omp = benchmark_omp_query(
                response, reference_bins, observed, repeat_count=args.repeat_count
            )

        rows.append(
            {
                "query_index": index,
                "query_id": selected["query_id"],
                "room": selected["room"],
                "candidate_count": int(selected["candidate_count"]),
                "candidate_indices_sha256": selected["candidate_indices_sha256"],
                "agree": agree,
                "fem_omp": {"source": omp_source, **omp},
            }
        )
        print(f"[{ordinal}/128] selector latency complete query {index}", flush=True)

    body = {
        "schema_version": 1,
        "record_type": "localization_selector_latency_kctx8_kgen1",
        "selection_sha256": selection["sha256"],
        "context_manifest_sha256": context["sha256"],
        "agree_checkpoint": str(args.agree_checkpoint.resolve()),
        "agree_checkpoint_file_sha256": file_sha256(args.agree_checkpoint.resolve()),
        "device": args.device,
        "candidate_batch_size": int(args.candidate_batch_size),
        "repeat_count": int(args.repeat_count),
        "query_count": len(rows),
        "fem_omp_source_counts": {
            "actual_fem_response": len(actual_indices),
            "shape_matched_synthetic_response": len(synthetic_indices),
            "not_run_random_fallback": len(failure_indices),
        },
        "queries": rows,
    }
    payload = {**body, "sha256": canonical_sha256(body)}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sha256": payload["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
