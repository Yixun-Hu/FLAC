#!/usr/bin/env python3
"""Real cached Vanilla FLAC throughput probe; never writes localization scores."""

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

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS,
    FA_CONTEXT_CONDITIONING_IDS,
    FA_DYNAMIC_CONDITIONING_IDS,
    SOURCE_CONDITIONING_IDS,
    cache_conditioning_branch,
    cache_invariant_conditioning_branch,
    candidate_metadata,
    candidate_seed,
    encode_audio_features,
    generate_and_score_batch,
    load_agree_retrieval,
    load_flac_module,
    load_frozen_query,
    merge_cached_conditioning,
    project_runtime_seconds,
    reconstruct_query_candidates,
)
from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES, invariant_conditioning


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: str, function):
    _sync(device)
    start = time.perf_counter()
    value = function()
    _sync(device)
    return value, time.perf_counter() - start


def _cat_branches(chunks: list[dict]) -> dict:
    output = {}
    for key in chunks[0]:
        tensors = torch.cat([chunk[key][0] for chunk in chunks], dim=0)
        masks = chunks[0][key][1]
        if masks is not None:
            masks = torch.cat([chunk[key][1] for chunk in chunks], dim=0)
        output[key] = [tensors, masks]
    return output


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _cache_branch(module, metadata, ids, device, cond_method):
    if cond_method == "fa_invariant":
        return cache_invariant_conditioning_branch(
            module.diffusion.conditioner,
            metadata,
            ids,
            device,
            DEFAULT_FRAME_ANGLES,
        )
    return cache_conditioning_branch(
        module.diffusion.conditioner,
        metadata,
        ids,
        device,
    )


def _full_conditioning(module, metadata, device, cond_method):
    if cond_method == "fa_invariant":
        return invariant_conditioning(
            module.diffusion.conditioner,
            metadata,
            device,
            DEFAULT_FRAME_ANGLES,
        )
    return module.diffusion.conditioner(metadata, device)


def _context_ids(cond_method):
    return (
        FA_CONTEXT_CONDITIONING_IDS
        if cond_method == "fa_invariant"
        else CONTEXT_CONDITIONING_IDS
    )


def _generate_score_call(
    module,
    retrieval,
    source_branch,
    context_branch,
    observation_features,
    seeds,
    candidate_items,
    device,
    cond_method,
):
    dynamic = None
    if cond_method == "fa_invariant":
        with torch.amp.autocast(torch.device(device).type):
            dynamic = _cache_branch(
                module,
                candidate_items,
                FA_DYNAMIC_CONDITIONING_IDS,
                device,
                cond_method,
            )
    return generate_and_score_batch(
        module,
        retrieval,
        source_branch,
        context_branch,
        observation_features,
        seeds,
        dynamic_branch=dynamic,
    )


def _conditioning_parity(
    module,
    candidate_batch,
    context_cache,
    device,
    cond_method,
):
    with torch.amp.autocast(torch.device(device).type):
        vectorized_uncached = _full_conditioning(
            module, candidate_batch, device, cond_method
        )
        candidatewise_uncached = _cat_branches(
            [
                _full_conditioning(module, [item], device, cond_method)
                for item in candidate_batch
            ]
        )
        source = _cache_branch(
            module,
            candidate_batch,
            SOURCE_CONDITIONING_IDS,
            device,
            cond_method,
        )
        dynamic = (
            _cache_branch(
                module,
                candidate_batch,
                FA_DYNAMIC_CONDITIONING_IDS,
                device,
                cond_method,
            )
            if cond_method == "fa_invariant"
            else None
        )
    cached = merge_cached_conditioning(
        source,
        context_cache,
        len(candidate_batch),
        dynamic_branch=dynamic,
    )
    details = {}
    for key in vectorized_uncached:
        # Match the execution shape of the branch being cached: source branches
        # are vectorized over candidates; query-context branches are computed
        # once per query.  A full vectorized call is retained as a transparent
        # mixed-precision batch-shape diagnostic, but it is not the cache
        # identity reference.
        reference = vectorized_uncached[key] if (
            key in SOURCE_CONDITIONING_IDS
            or (
                cond_method == "fa_invariant"
                and key in FA_DYNAMIC_CONDITIONING_IDS
            )
        ) else candidatewise_uncached[key]
        token_equal = torch.equal(reference[0], cached[key][0])
        mask_equal = torch.equal(reference[1], cached[key][1])
        max_abs = float((reference[0].float() - cached[key][0].float()).abs().max())
        vectorized_max_abs = float(
            (vectorized_uncached[key][0].float() - cached[key][0].float()).abs().max()
        )
        details[key] = {
            "shape_matched_tokens_bit_equal": token_equal,
            "shape_matched_masks_bit_equal": mask_equal,
            "shape_matched_max_abs_difference": max_abs,
            "full_vectorized_tokens_bit_equal": torch.equal(
                vectorized_uncached[key][0], cached[key][0]
            ),
            "full_vectorized_masks_bit_equal": torch.equal(
                vectorized_uncached[key][1], cached[key][1]
            ),
            "full_vectorized_max_abs_difference": vectorized_max_abs,
        }
    if not all(
        item["shape_matched_tokens_bit_equal"]
        and item["shape_matched_masks_bit_equal"]
        for item in details.values()
    ):
        raise RuntimeError(f"cached conditioning fails shape-matched bit identity: {details}")
    return details


def _project_all(totals, measurements, score_samples):
    return project_runtime_seconds(
        query_count=totals["context_branch_queries"],
        receiver_candidate_count=totals["chosen_receiver_candidate_pairs"],
        query_candidate_count=totals["chosen_candidate_query_pairs"],
        query_io_seconds_per_query=measurements["query_io_seconds_per_query"],
        context_seconds_per_query=measurements["context_seconds_per_query"],
        observation_seconds_per_query=measurements["observation_seconds_per_query"],
        source_candidates_per_second=measurements["source_candidates_per_second"],
        generated_scores_per_second=measurements["generated_scores_per_second"],
        score_samples=score_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--agree-ckpt", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cond-method", choices=("vanilla", "fa_invariant"), default="vanilla"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-queries", type=int, default=8)
    parser.add_argument("--source-candidates", type=int, default=512)
    parser.add_argument("--source-batch-size", type=int, default=64)
    parser.add_argument("--generation-batch-sizes", default="32,64,128")
    parser.add_argument("--generation-batches", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("probe output must stay inside the NeuriPs_Workshop worktree") from error
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("real throughput probe requires CUDA")

    manifest = load_context_manifest(args.context_manifest)
    audit = json.loads(args.geometry_audit.read_text())
    if audit["geometry_gate"] != "PASS" or audit["z_branch"] != "z_band":
        raise RuntimeError("throughput probe requires the passed frozen geometry audit")
    records = {record["index"]: record for record in manifest["records"]}
    audit_queries = sorted(audit["queries"], key=lambda item: item["chosen_count"], reverse=True)
    benchmark_audit = audit_queries[0]
    benchmark_record = records[benchmark_audit["index"]]

    distinct = []
    seen_rooms = set()
    for query in audit_queries:
        if query["room"] not in seen_rooms:
            distinct.append(records[query["index"]])
            seen_rooms.add(query["room"])
        if len(distinct) == args.context_queries:
            break

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(args.device)
    load_start = time.perf_counter()
    module, model_config = load_flac_module(args.model_config, args.ckpt_path, args.device)
    retrieval = load_agree_retrieval(args.agree_ckpt, args.device)
    _sync(args.device)
    model_load_seconds = time.perf_counter() - load_start

    loaded = []
    data_load_seconds = []
    for record in distinct:
        start = time.perf_counter()
        observed, metadata = load_frozen_query(record, args.dataset_root)
        data_load_seconds.append(time.perf_counter() - start)
        loaded.append((record, observed, metadata))

    context_times = []
    observation_times = []
    context_caches = []
    observation_features = []
    for _record, observed, metadata in loaded:
        def context_call():
            with torch.amp.autocast("cuda"):
                return _cache_branch(
                    module,
                    [metadata],
                    _context_ids(args.cond_method),
                    args.device,
                    args.cond_method,
                )

        cache, elapsed = _timed(args.device, context_call)
        context_caches.append(cache)
        context_times.append(elapsed)
        observed_gpu = observed.unsqueeze(0).to(args.device, dtype=torch.float32)
        features, elapsed = _timed(
            args.device, lambda: encode_audio_features(retrieval, observed_gpu)
        )
        observation_features.append(features)
        observation_times.append(elapsed)

    benchmark_position = next(
        i for i, (record, _audio, _metadata) in enumerate(loaded)
        if record["index"] == benchmark_record["index"]
    )
    benchmark_metadata = loaded[benchmark_position][2]
    benchmark_context = context_caches[benchmark_position]
    benchmark_observation = observation_features[benchmark_position]
    candidates = reconstruct_query_candidates(benchmark_record, audit)
    if len(candidates) != benchmark_audit["chosen_count"]:
        raise RuntimeError("candidate reconstruction count differs from audit")
    candidate_limit = min(args.source_candidates, len(candidates))
    candidates = candidates[:candidate_limit]
    candidate_items = candidate_metadata(
        benchmark_metadata, candidates, benchmark_record["receiver_global"]
    )

    parity = _conditioning_parity(
        module,
        candidate_items[:2],
        benchmark_context,
        args.device,
        args.cond_method,
    )

    source_chunks = []
    source_batch_times = []
    for start in range(0, candidate_limit, args.source_batch_size):
        stop = min(start + args.source_batch_size, candidate_limit)
        batch = candidate_items[start:stop]

        def source_call(batch=batch):
            with torch.amp.autocast("cuda"):
                return _cache_branch(
                    module,
                    batch,
                    SOURCE_CONDITIONING_IDS,
                    args.device,
                    args.cond_method,
                )

        chunk, elapsed = _timed(args.device, source_call)
        source_chunks.append(chunk)
        source_batch_times.append({"count": len(batch), "seconds": elapsed})
    source_cache = _cat_branches(source_chunks)
    source_total_seconds = sum(item["seconds"] for item in source_batch_times)
    source_rate = candidate_limit / source_total_seconds

    batch_sizes = [int(value) for value in args.generation_batch_sizes.split(",")]
    generation_results = []
    for batch_size in batch_sizes:
        if batch_size > candidate_limit:
            continue
        try:
            warm_seeds = [
                candidate_seed(args.seed, benchmark_record["index"], i, 0)
                for i in range(batch_size)
            ]
            _generate_score_call(
                module,
                retrieval,
                {key: [value[0][:batch_size], value[1][:batch_size]] for key, value in source_cache.items()},
                benchmark_context,
                benchmark_observation,
                warm_seeds,
                candidate_items[:batch_size],
                args.device,
                args.cond_method,
            )
            _sync(args.device)
            timings = []
            for batch_index in range(args.generation_batches):
                indices = [
                    (batch_index * batch_size + offset) % candidate_limit
                    for offset in range(batch_size)
                ]
                branch = {
                    key: [
                        value[0][indices],
                        value[1][indices] if value[1] is not None else None,
                    ]
                    for key, value in source_cache.items()
                }
                seeds = [
                    candidate_seed(args.seed, benchmark_record["index"], index, batch_index + 1)
                    for index in indices
                ]
                _scores, elapsed = _timed(
                    args.device,
                    lambda branch=branch, seeds=seeds, indices=indices: _generate_score_call(
                        module,
                        retrieval,
                        branch,
                        benchmark_context,
                        benchmark_observation,
                        seeds,
                        [candidate_items[index] for index in indices],
                        args.device,
                        args.cond_method,
                    ),
                )
                timings.append(elapsed)
            count = batch_size * len(timings)
            generation_results.append(
                {
                    "batch_size": batch_size,
                    "status": "PASS",
                    "batch_seconds": timings,
                    "generated_scores": count,
                    "aggregate_scores_per_second": count / sum(timings),
                    "median_batch_scores_per_second": batch_size / statistics.median(timings),
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            generation_results.append(
                {"batch_size": batch_size, "status": "OOM", "error": str(error)}
            )
            torch.cuda.empty_cache()

    passed = [result for result in generation_results if result["status"] == "PASS"]
    if not passed:
        raise RuntimeError("all generation batch sizes failed")
    winner = max(passed, key=lambda item: item["aggregate_scores_per_second"])
    measurements = {
        "query_io_seconds_per_query": statistics.mean(data_load_seconds),
        "context_seconds_per_query": statistics.mean(context_times),
        "observation_seconds_per_query": statistics.mean(observation_times),
        "source_candidates_per_second": source_rate,
        "generated_scores_per_second": winner["aggregate_scores_per_second"],
    }
    projections = {}
    for score_samples in (1, 2, 4):
        projection = _project_all(audit["totals"], measurements, score_samples)
        projection["total_gpu_hours"] = projection["total_seconds"] / 3600.0
        projection["total_gpu_days"] = projection["total_seconds"] / 86400.0
        projections[str(score_samples)] = projection

    budget_hours = 168.0
    selected_k = next(
        (k for k in (4, 2, 1) if projections[str(k)]["total_gpu_hours"] <= budget_hours),
        None,
    )
    result = {
        "schema_version": 1,
        "no_quality_values_saved": True,
        "conditioning_method": args.cond_method,
        "frame_average_angles": (
            list(DEFAULT_FRAME_ANGLES) if args.cond_method == "fa_invariant" else None
        ),
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "model_config": str(args.model_config.resolve()),
        "model_config_sha256": _sha256(args.model_config),
        "checkpoint": str(args.ckpt_path.resolve()),
        "checkpoint_sha256": _sha256(args.ckpt_path),
        "agree_checkpoint": str(args.agree_ckpt.resolve()),
        "agree_checkpoint_sha256": _sha256(args.agree_ckpt.resolve()),
        "context_manifest_sha256": manifest["sha256"],
        "geometry_audit_sha256": audit["sha256"],
        "geometry_totals": audit["totals"],
        "benchmark_query": {
            "index": benchmark_record["index"],
            "query_id": benchmark_record["query_id"],
            "room": benchmark_record["room"],
            "candidate_count": benchmark_audit["chosen_count"],
            "source_probe_candidates": candidate_limit,
        },
        "model_load_seconds": model_load_seconds,
        "data_load_seconds": data_load_seconds,
        "context_cache_seconds": context_times,
        "observation_encode_seconds": observation_times,
        "source_cache_batches": source_batch_times,
        "generation_batch_results": generation_results,
        "winning_generation_batch_size": winner["batch_size"],
        "measurements": measurements,
        "conditioning_parity": parity,
        "projections": projections,
        "budget_gpu_hours": budget_hours,
        "selected_score_samples": selected_k,
        "summary_statistics": {
            "context_seconds_median": statistics.median(context_times),
            "context_seconds_p10": _percentile(context_times, 10),
            "context_seconds_p90": _percentile(context_times, 90),
            "observation_seconds_median": statistics.median(observation_times),
            "source_cache_rate": source_rate,
            "generation_rate": winner["aggregate_scores_per_second"],
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, output)
    print(json.dumps({
        "sha256": result["sha256"],
        "winning_generation_batch_size": result["winning_generation_batch_size"],
        "measurements": result["measurements"],
        "projections": result["projections"],
        "selected_score_samples": result["selected_score_samples"],
    }, indent=2))


if __name__ == "__main__":
    main()
