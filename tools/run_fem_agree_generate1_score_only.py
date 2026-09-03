#!/usr/bin/env python3
"""Measure real-waveform FEM--AGREE generate=1 scoring from cached FEM responses."""

from __future__ import annotations

import argparse
import json
import os
import platform
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

from src.baselines.fem_agree import exact_bandlimit_waveforms, peak_normalize_waveforms
from src.baselines.fem_sabine import bandlimited_response_to_waveform
from src.localization.engine import encode_audio_features, load_agree_retrieval, load_frozen_query
from src.localization.pilot import canonical_sha256
from src.localization.real_rir_oracle import deterministic_agree_seed
from src.localization.runner import file_sha256, verify_hashed_payload
from src.localization.scoring import localization_metrics, stable_argmax


FREQUENCY_BAND_HZ = (80.0, 300.0)
TIMING_FIELDS = (
    "waveform_construction_seconds",
    "waveform_preprocessing_seconds",
    "observed_agree_encode_seconds",
    "candidate_agree_encode_seconds",
    "cosine_similarity_seconds",
    "argmax_seconds",
    "agree_selector_total_seconds",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def hashed(payload: dict) -> dict:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def load_hashed_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text())
    verify_hashed_payload(payload, label)
    return payload


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: str, function, *args, **kwargs):
    synchronize(device)
    started = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - started


def resolve_records(selection: dict, context: dict, query_index: int | None) -> list[tuple[dict, dict]]:
    context_by_index = {int(row["index"]): row for row in context["records"]}
    records = []
    for selected in selection["records"]:
        index = int(selected["index"])
        if query_index is not None and index != query_index:
            continue
        if index not in context_by_index:
            raise RuntimeError(f"query {index} is absent from the context manifest")
        record = context_by_index[index]
        for field in ("query_id", "scene", "room"):
            if selected[field] != record[field]:
                raise RuntimeError(f"query {index} has mismatched {field}")
        records.append((selected, record))
    if query_index is not None and len(records) != 1:
        raise RuntimeError(f"query {query_index} is not unique in the selection")
    if not records:
        raise RuntimeError("selection resolved no queries")
    return records


def load_cached_query(
    response_dir: Path,
    old_result_dir: Path,
    selected: dict,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    index = int(selected["index"])
    response_path = response_dir / f"query_{index:05d}_response.npz"
    audit_path = response_dir / f"query_{index:05d}.json"
    old_result_path = old_result_dir / f"query_{index:05d}.json"
    response_audit = load_hashed_json(audit_path, f"FEM response query {index}")
    if response_audit.get("response_file_sha256") != file_sha256(response_path):
        raise RuntimeError(f"query {index} response cache hash mismatch")
    if response_audit.get("query_id") != selected["query_id"]:
        raise RuntimeError(f"query {index} response identity mismatch")
    if response_audit.get("candidate_indices_sha256") != selected["candidate_indices_sha256"]:
        raise RuntimeError(f"query {index} response candidate hash mismatch")
    if int(response_audit.get("context_count", -1)) != 8:
        raise RuntimeError(f"query {index} response does not use K_ctx=8")
    with np.load(response_path, allow_pickle=False) as archive:
        response = archive["response"].copy()
        candidates = archive["candidates"].copy()
        bin_indices = archive["bin_indices"].copy()
        frequencies = archive["frequencies_hz"].copy()
    if response.shape != (int(selected["candidate_count"]), len(bin_indices)):
        raise RuntimeError(f"query {index} response shape mismatch")
    if candidates.shape != (int(selected["candidate_count"]), 3):
        raise RuntimeError(f"query {index} candidate shape mismatch")
    old_result = load_hashed_json(old_result_path, f"legacy FEM--AGREE query {index}")
    if old_result.get("response_cache_sha256") != response_audit["sha256"]:
        raise RuntimeError(f"query {index} legacy result response hash mismatch")
    return response_audit, response, candidates, bin_indices, frequencies, old_result


@torch.inference_mode()
def score_generate1(
    retrieval,
    response: np.ndarray,
    bin_indices: np.ndarray,
    candidates: np.ndarray,
    observed: torch.Tensor,
    truth: np.ndarray,
    *,
    query_index: int,
    score_seed: int,
    device: str,
    candidate_batch_size: int,
    target_peak: float,
) -> tuple[dict, np.ndarray, dict, dict]:
    synchronize(device)
    selector_started = time.perf_counter()

    candidate_waveforms, waveform_construction_seconds = timed_call(
        device,
        bandlimited_response_to_waveform,
        torch.from_numpy(response),
        bin_indices,
        sample_count=int(observed.shape[-1]),
    )

    def preprocess():
        observed_waveform = exact_bandlimit_waveforms(observed.unsqueeze(0), bin_indices)
        normalized_candidates, candidate_peaks = peak_normalize_waveforms(
            candidate_waveforms, target_peak=target_peak
        )
        normalized_observed, observed_peak = peak_normalize_waveforms(
            observed_waveform, target_peak=target_peak
        )
        return normalized_candidates, normalized_observed, candidate_peaks, observed_peak

    (
        normalized_candidates,
        normalized_observed,
        candidate_peaks,
        observed_peak,
    ), waveform_preprocessing_seconds = timed_call(device, preprocess)

    observation_seed = deterministic_agree_seed(
        score_seed, query_index, "fem_agree_bandlimited_observation"
    )
    candidate_seed = deterministic_agree_seed(
        score_seed, query_index, "fem_agree_bandlimited_candidates"
    )
    torch.manual_seed(observation_seed)
    observation_feature, observed_agree_encode_seconds = timed_call(
        device,
        encode_audio_features,
        retrieval,
        normalized_observed.to(device=device, dtype=torch.float32),
    )
    observation_feature = torch.atleast_2d(observation_feature).float()
    if observation_feature.shape[0] != 1:
        raise RuntimeError("AGREE must return exactly one observation feature")

    torch.manual_seed(candidate_seed)
    candidate_agree_encode_seconds = 0.0
    cosine_similarity_seconds = 0.0
    similarity_chunks = []
    for start in range(0, len(normalized_candidates), candidate_batch_size):
        chunk = normalized_candidates[start : start + candidate_batch_size]
        features, seconds = timed_call(
            device,
            encode_audio_features,
            retrieval,
            chunk.to(device=device, dtype=torch.float32),
        )
        candidate_agree_encode_seconds += seconds
        features = torch.atleast_2d(features).float()
        similarities, seconds = timed_call(
            device,
            lambda values: values @ observation_feature.T,
            features,
        )
        cosine_similarity_seconds += seconds
        if similarities.shape != (len(chunk), 1):
            raise RuntimeError("generate=1 AGREE similarity shape mismatch")
        similarity_chunks.append(similarities.detach().cpu())

    similarities = torch.cat(similarity_chunks, dim=0)
    prediction_index, argmax_seconds = timed_call(
        device, stable_argmax, similarities[:, 0]
    )
    synchronize(device)
    agree_selector_total_seconds = time.perf_counter() - selector_started
    metrics = localization_metrics(candidates, truth, prediction_index)
    scores = similarities[:, 0]
    metrics.update(
        {
            "prediction_global": candidates[prediction_index].astype(float).tolist(),
            "winning_score": float(scores[prediction_index]),
            "mean_candidate_score": float(scores.mean()),
        }
    )
    timing = {
        "waveform_construction_seconds": waveform_construction_seconds,
        "waveform_preprocessing_seconds": waveform_preprocessing_seconds,
        "observed_agree_encode_seconds": observed_agree_encode_seconds,
        "candidate_agree_encode_seconds": candidate_agree_encode_seconds,
        "cosine_similarity_seconds": cosine_similarity_seconds,
        "argmax_seconds": argmax_seconds,
        "agree_selector_total_seconds": agree_selector_total_seconds,
    }
    timing["component_sum_seconds"] = float(
        sum(timing[field] for field in TIMING_FIELDS if field != "agree_selector_total_seconds")
    )
    timing["unattributed_overhead_seconds"] = float(
        agree_selector_total_seconds - timing["component_sum_seconds"]
    )
    waveform_audit = {
        "target_peak": target_peak,
        "observed_pre_normalization_peak": float(observed_peak[0]),
        "candidate_pre_normalization_peak_min": float(candidate_peaks.min()),
        "candidate_pre_normalization_peak_median": float(candidate_peaks.median()),
        "candidate_pre_normalization_peak_max": float(candidate_peaks.max()),
    }
    seeds = {"observation": observation_seed, "candidates": candidate_seed}
    return metrics, similarities.numpy(), timing, {"waveform": waveform_audit, "seeds": seeds}


def summarize_vector(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary vector must be finite and nonempty")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sum": float(array.sum()),
    }


def render_markdown(summary: dict) -> str:
    latency = summary["latency_seconds"]["agree_selector_total_seconds"]
    metrics = summary["localization_metrics"]
    per_candidate = summary["per_candidate_latency_seconds"]
    return "\n".join(
        [
            "# FEM--AGREE generate=1 score-only latency on RTX A6000",
            "",
            (
                f"Scope: {summary['query_count']} queries / {summary['room_count']} rooms; "
                "K_ctx=8; exactly one AGREE encoding per FEM candidate; direct cosine + "
                "argmax; no log-mean-exp. Checkpoint loading and input-file loading are excluded."
            ),
            "",
            "| Queries | Mean [s/query] | Median | P90 | Total | Macro mean [ms/candidate] | Pooled [ms/candidate] |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {summary['query_count']} | {latency['mean']:.6f} | "
                f"{latency['median']:.6f} | {latency['p90']:.6f} | "
                f"{latency['sum']:.6f} | {1000 * per_candidate['macro_mean']:.6f} | "
                f"{1000 * per_candidate['pooled']:.6f} |"
            ),
            "",
            "| Mean error [m] | Median error [m] | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {metrics['mean_localization_error_m']:.6f} | "
                f"{metrics['median_localization_error_m']:.6f} | "
                f"{100 * metrics['success_0_5m']:.3f}% | "
                f"{100 * metrics['success_1_0m']:.3f}% | "
                f"{100 * metrics['oracle_normalized_success_0_5m']:.3f}% |"
            ),
            "",
            (
                "Legacy K=1 prediction agreement: "
                f"{summary['legacy_joint_k1_prediction_match_count']}/"
                f"{summary['query_count']}. The legacy run encoded K=1/4/8 jointly; "
                "its timing is not reused here."
            ),
            "",
        ]
    )


def write_aggregate(output_dir: Path, rows: list[dict], run_manifest: dict) -> dict:
    rows = sorted(rows, key=lambda row: int(row["query_index"]))
    errors = np.asarray(
        [float(row["localization_metrics"]["localization_error_m"]) for row in rows]
    )
    candidate_counts = np.asarray([int(row["candidate_count"]) for row in rows])
    selector_totals = np.asarray(
        [float(row["timing_seconds"]["agree_selector_total_seconds"]) for row in rows]
    )
    ratios = selector_totals / candidate_counts
    timing_summary = {
        field: summarize_vector([float(row["timing_seconds"][field]) for row in rows])
        for field in (*TIMING_FIELDS, "component_sum_seconds", "unattributed_overhead_seconds")
    }
    summary = hashed(
        {
            "schema_version": 1,
            "method": "fem_agree_generate1_score_only",
            "hardware": run_manifest["hardware"],
            "query_count": len(rows),
            "room_count": len({row["room"] for row in rows}),
            "candidate_evaluations": int(candidate_counts.sum()),
            "protocol": run_manifest["protocol"],
            "run_manifest_sha256": run_manifest["sha256"],
            "latency_seconds": timing_summary,
            "per_candidate_latency_seconds": {
                "macro_mean": float(ratios.mean()),
                "macro_median": float(np.median(ratios)),
                "macro_p90": float(np.quantile(ratios, 0.9)),
                "pooled": float(selector_totals.sum() / candidate_counts.sum()),
            },
            "localization_metrics": {
                "mean_localization_error_m": float(errors.mean()),
                "median_localization_error_m": float(np.median(errors)),
                "success_0_5m": float(
                    np.mean([row["localization_metrics"]["success_0_5m"] for row in rows])
                ),
                "success_1_0m": float(
                    np.mean([row["localization_metrics"]["success_1_0m"] for row in rows])
                ),
                "oracle_normalized_success_0_5m": float(
                    np.mean(
                        [
                            row["localization_metrics"]["oracle_normalized_success_0_5m"]
                            for row in rows
                        ]
                    )
                ),
            },
            "legacy_joint_k1_prediction_match_count": int(
                sum(
                    row["prediction_index"] == row["legacy_joint_k1_prediction_index"]
                    for row in rows
                )
            ),
        }
    )
    atomic_json(output_dir / "summary.json", summary)
    markdown_path = output_dir / "summary.md"
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(render_markdown(summary))
    os.replace(temporary, markdown_path)
    jsonl_path = output_dir / "per_query_latency.jsonl"
    temporary = jsonl_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary, jsonl_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--old-result-dir", type=Path, required=True)
    parser.add_argument("--agree-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--score-seed", type=int, default=42)
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--query-index", type=int)
    parser.add_argument("--require-gpu-name", default="NVIDIA RTX A6000")
    args = parser.parse_args()
    if args.candidate_batch_size <= 0:
        raise ValueError("candidate batch size must be positive")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(args.device)
    if args.require_gpu_name and gpu_name != args.require_gpu_name:
        raise RuntimeError(f"expected {args.require_gpu_name}, found {gpu_name}")

    selection = load_hashed_json(args.selection.resolve(), "97-query selection")
    context = load_hashed_json(args.context_manifest.resolve(), "context manifest")
    if selection.get("context_manifest_sha256") != context["sha256"]:
        raise RuntimeError("selection and context manifest do not match")
    records = resolve_records(selection, context, args.query_index)
    if args.query_index is None and len(records) != 97:
        raise RuntimeError(f"expected 97 queries, found {len(records)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.agree_checkpoint.resolve()
    runner_sha256 = file_sha256(Path(__file__).resolve())
    checkpoint_sha256 = file_sha256(checkpoint)
    hardware = {
        "hostname": platform.node(),
        "gpu_model": gpu_name,
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(args.device).total_memory
        ),
        "cpu_model": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "device": args.device,
    }
    protocol = {
        "context_count": 8,
        "generated_rirs_per_candidate": 1,
        "candidate_batch_size": args.candidate_batch_size,
        "score_seed": args.score_seed,
        "target_peak": args.target_peak,
        "similarity": "direct_cosine",
        "selection": "stable_argmax",
        "agree_similarity_shape": "[candidate_count, 1]",
        "log_mean_exp_used": False,
        "checkpoint_loading_included": False,
        "input_file_loading_included": False,
        "repeat_count": 1,
    }
    manifest = hashed(
        {
            "schema_version": 1,
            "record_type": "fem_agree_generate1_score_only_run",
            "selection": str(args.selection.resolve()),
            "selection_sha256": selection["sha256"],
            "context_manifest": str(args.context_manifest.resolve()),
            "context_manifest_sha256": context["sha256"],
            "response_dir": str(args.response_dir.resolve()),
            "old_result_dir": str(args.old_result_dir.resolve()),
            "agree_checkpoint": str(checkpoint),
            "agree_checkpoint_sha256": checkpoint_sha256,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": runner_sha256,
            "hardware": hardware,
            "protocol": protocol,
            "query_indices": [int(selected["index"]) for selected, _record in records],
        }
    )
    atomic_json(output_dir / "run_manifest.json", manifest)

    retrieval = load_agree_retrieval(checkpoint, args.device)
    # Unreported warm-ups for both shapes used below. Every query is reseeded afterwards.
    warmup_one = torch.zeros((1, 1, 10240), dtype=torch.float32, device=args.device)
    _ = encode_audio_features(retrieval, warmup_one)
    warmup_batch = torch.zeros(
        (args.candidate_batch_size, 1, 10240), dtype=torch.float32, device=args.device
    )
    _ = encode_audio_features(retrieval, warmup_batch)
    synchronize(args.device)

    completed_rows = []
    for ordinal, (selected, record) in enumerate(records, start=1):
        index = int(selected["index"])
        result_path = output_dir / "queries" / f"query_{index:05d}.json"
        if result_path.is_file():
            existing = load_hashed_json(result_path, f"generate=1 query {index}")
            expected = {
                "method": "fem_agree_generate1_score_only",
                "run_manifest_sha256": manifest["sha256"],
                "runner_sha256": runner_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
            if all(existing.get(field) == value for field, value in expected.items()):
                completed_rows.append(existing)
                print(f"[{ordinal}/{len(records)}] resume query {index}", flush=True)
                continue

        response_audit, response, candidates, bin_indices, frequencies, old_result = (
            load_cached_query(
                args.response_dir.resolve(), args.old_result_dir.resolve(), selected
            )
        )
        observed, _metadata = load_frozen_query(record, args.dataset_root.resolve())
        metrics, similarities, timing, audit = score_generate1(
            retrieval,
            response,
            bin_indices,
            candidates,
            observed,
            np.asarray(record["source_global"], dtype=np.float64),
            query_index=index,
            score_seed=args.score_seed,
            device=args.device,
            candidate_batch_size=args.candidate_batch_size,
            target_peak=args.target_peak,
        )
        arrays_path = output_dir / "queries" / f"query_{index:05d}_scores.npz"
        atomic_npz(
            arrays_path,
            candidates=candidates.astype(np.float32),
            similarities=similarities.astype(np.float32),
            scores=similarities[:, 0].astype(np.float32),
            bin_indices=bin_indices.astype(np.int64),
            frequencies_hz=frequencies.astype(np.float64),
        )
        legacy_metric = old_result["metrics_by_k_agree"]["1"]
        row = hashed(
            {
                "schema_version": 1,
                "method": "fem_agree_generate1_score_only",
                "execution_status": "completed",
                "result_source": "native_cached_fem_response",
                "query_index": index,
                "query_id": selected["query_id"],
                "scene": record["scene"],
                "room": record["room"],
                "candidate_count": len(candidates),
                "candidate_indices_sha256": selected["candidate_indices_sha256"],
                "context_count": 8,
                "generated_rirs_per_candidate": 1,
                "agree_similarity_shape": list(similarities.shape),
                "prediction_index": int(metrics["prediction_index"]),
                "prediction_global": metrics["prediction_global"],
                "localization_metrics": metrics,
                "legacy_joint_k1_prediction_index": int(legacy_metric["prediction_index"]),
                "legacy_joint_k1_localization_error_m": float(
                    legacy_metric["localization_error_m"]
                ),
                "timing_seconds": timing,
                "waveform_audit": audit["waveform"],
                "agree_seeds": audit["seeds"],
                "candidate_batch_size": args.candidate_batch_size,
                "score_seed": args.score_seed,
                "frequency_band_hz": list(FREQUENCY_BAND_HZ),
                "frequency_count": len(frequencies),
                "response_cache_sha256": response_audit["sha256"],
                "response_file_sha256": response_audit["response_file_sha256"],
                "legacy_joint_result_sha256": old_result["sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "runner_sha256": runner_sha256,
                "run_manifest_sha256": manifest["sha256"],
                "hardware": hardware,
                "arrays_file": arrays_path.name,
                "arrays_sha256": file_sha256(arrays_path),
            }
        )
        atomic_json(result_path, row)
        completed_rows.append(row)
        print(
            f"[{ordinal}/{len(records)}] query {index}: "
            f"prediction={row['prediction_index']} "
            f"selector={timing['agree_selector_total_seconds']:.6f}s",
            flush=True,
        )

    if args.query_index is None:
        summary = write_aggregate(output_dir, completed_rows, manifest)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
