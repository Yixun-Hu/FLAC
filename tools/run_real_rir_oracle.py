#!/usr/bin/env python3
"""Run the sparse real-RIR AGREE ground-truth-candidate upper bound for exp_09."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.engine import (
    SCORE_SAMPLE_COUNTS,
    _pad_crop_audio,
    encode_audio_features,
    load_agree_retrieval,
)
from src.localization.pilot import canonical_sha256, load_pilot_manifest
from src.localization.real_rir_oracle import (
    aggregate_oracle_rows,
    deterministic_agree_seed,
    discover_real_rir_bank,
    render_oracle_markdown,
    resolve_oracle_records,
    select_representative_cases,
    summarize_oracle_scores,
)
from src.localization.runner import file_sha256
from src.localization.scoring import log_mean_exp_scores


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", action="append", type=Path, required=True)
    parser.add_argument("--pilot-label", action="append", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--agree-ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--score-seed", type=int, default=42)
    parser.add_argument("--expected-query-count", type=int, default=128)
    parser.add_argument("--expected-room-count", type=int, default=16)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if len(args.pilot_manifest) != len(args.pilot_label):
        raise ValueError("every pilot manifest requires one pilot label")
    if (
        args.batch_size <= 0
        or args.expected_query_count <= 0
        or args.expected_room_count <= 0
        or (args.query_limit is not None and args.query_limit <= 0)
    ):
        raise ValueError("batch size and expected counts must be positive")
    for output in (args.output_json.resolve(), args.output_md.resolve()):
        try:
            output.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError("oracle outputs must stay inside NeuriPs_Workshop") from error

    context = load_context_manifest(args.context_manifest)
    pilots = [load_pilot_manifest(path) for path in args.pilot_manifest]
    records = resolve_oracle_records(list(zip(args.pilot_label, pilots)), context)
    if args.query_limit is not None:
        records = records[: args.query_limit]
    if len(records) != args.expected_query_count:
        raise RuntimeError(
            f"expected {args.expected_query_count} oracle queries, got {len(records)}"
        )
    room_count = len({(item["scene"], item["room"]) for item in records})
    if room_count != args.expected_room_count:
        raise RuntimeError(
            f"expected {args.expected_room_count} oracle rooms, got {room_count}"
        )
    started = time.perf_counter()
    torch.manual_seed(args.score_seed)
    retrieval = load_agree_retrieval(args.agree_ckpt, args.device)
    waveform_cache: dict[tuple[str, str, int], tuple[dict, torch.Tensor]] = {}
    results = []
    for ordinal, record in enumerate(records, start=1):
        bank = discover_real_rir_bank(record, args.dataset_root)
        cache_key = (record["scene"], record["room"], int(bank["receiver_id"]))
        cached = waveform_cache.get(cache_key)
        if cached is None:
            paths = [args.dataset_root / path for path in bank["rir_paths"]]
            candidate_waveforms = torch.stack(
                [_pad_crop_audio(path, 10240, clamp=True) for path in paths]
            )
            waveform_cache[cache_key] = (bank, candidate_waveforms)
        else:
            cached_bank, candidate_waveforms = cached
            if cached_bank["rir_paths"] != bank["rir_paths"]:
                raise RuntimeError("same receiver resolved to a different real-RIR bank")
        observed = _pad_crop_audio(
            args.dataset_root / record["query_id"], 10240, clamp=True
        )
        target_index = int(bank["target_index"])
        if not torch.equal(observed, candidate_waveforms[target_index]):
            raise RuntimeError("target candidate and observed waveform differ")
        observation_seed = deterministic_agree_seed(
            args.score_seed, int(record["index"]), "observation"
        )
        candidate_seed = deterministic_agree_seed(
            args.score_seed, int(record["index"]), "candidate_real_rirs"
        )
        torch.manual_seed(observation_seed)
        observation_feature = encode_audio_features(
            retrieval,
            observed.unsqueeze(0).to(device=args.device, dtype=torch.float32),
        ).float()
        maximum_samples = max(SCORE_SAMPLE_COUNTS)
        repeated = candidate_waveforms.repeat_interleave(maximum_samples, dim=0)
        if len(repeated) > args.batch_size:
            raise RuntimeError(
                f"encoder batch size {args.batch_size} is below required {len(repeated)}"
            )
        torch.manual_seed(candidate_seed)
        candidate_features = encode_audio_features(
            retrieval, repeated.to(device=args.device, dtype=torch.float32)
        ).float()
        similarities = (candidate_features @ observation_feature.T).reshape(
            len(candidate_waveforms), maximum_samples
        )
        score_tensors = log_mean_exp_scores(
            similarities, SCORE_SAMPLE_COUNTS, tau=args.tau
        )
        metrics_by_k = {
            str(count): summarize_oracle_scores(
                bank["positions_global"],
                values.numpy(),
                target_index=target_index,
                temperature=args.temperature,
            )
            for count, values in score_tensors.items()
        }
        primary_count = max(SCORE_SAMPLE_COUNTS)
        primary_scores = score_tensors[primary_count].numpy().astype(np.float64)
        primary_metrics = metrics_by_k[str(primary_count)]
        row = {
            "batch": record["batch"],
            "pilot_manifest_sha256": record["pilot_manifest_sha256"],
            "query_index": int(record["index"]),
            "query_id": record["query_id"],
            "scene": record["scene"],
            "room": record["room"],
            "source_global": list(map(float, record["source_global"])),
            "receiver_global": list(map(float, record["receiver_global"])),
            **bank,
            "agree_rng_seeds": {
                "observation": observation_seed,
                "candidate_real_rirs": candidate_seed,
            },
            "similarities": similarities.numpy().astype(float).tolist(),
            "scores_by_k": {
                str(count): values.numpy().astype(float).tolist()
                for count, values in score_tensors.items()
            },
            "metrics_by_k": metrics_by_k,
            "scores": primary_scores.astype(float).tolist(),
            **primary_metrics,
        }
        results.append(row)
        print(
            f"[{ordinal}/{len(records)}] {record['query_id']} "
            f"bank={primary_metrics['candidate_count']} "
            f"K8-rank={primary_metrics['target_rank']} "
            f"K8-margin={primary_metrics['target_margin']:.4f}",
            flush=True,
        )

    summary_by_k = {}
    for count in SCORE_SAMPLE_COUNTS:
        metric_rows = [
            {
                "query_id": item["query_id"],
                "scene": item["scene"],
                "room": item["room"],
                **item["metrics_by_k"][str(count)],
            }
            for item in results
        ]
        summary_by_k[str(count)] = aggregate_oracle_rows(metric_rows)
    primary_count = max(SCORE_SAMPLE_COUNTS)
    summary = summary_by_k[str(primary_count)]
    representative = select_representative_cases(results)
    case_fields = (
        "category",
        "batch",
        "query_index",
        "query_id",
        "scene",
        "room",
        "target_margin",
        "target_probability",
        "normalized_entropy",
        "hardest_negative_distance_m",
    )
    payload = {
        "schema_version": 2,
        "diagnostic": "sparse_metadata_bank_ground_truth_rir_upper_bound",
        "interpretation": (
            "Released real candidate RIRs replace FLAC outputs. The observation and every "
            "candidate copy are independently encoded because the frozen AGREE VAE audio "
            "tower samples even in eval mode. Fixed per-query seeds make the nested K=1/4/8 "
            "log-mean-exp control reproducible. T-scaled mass is not calibrated."
        ),
        "query_count": len(results),
        "query_limit": args.query_limit,
        "room_count": room_count,
        "unique_receiver_bank_count": len(waveform_cache),
        "score_seed": int(args.score_seed),
        "score_sample_counts": list(SCORE_SAMPLE_COUNTS),
        "primary_score_sample_count": primary_count,
        "tau": float(args.tau),
        "temperature": float(args.temperature),
        "pilot_manifest_sha256": [pilot["sha256"] for pilot in pilots],
        "context_manifest_sha256": context["sha256"],
        "agree_checkpoint_sha256": file_sha256(args.agree_ckpt),
        "audio_protocol": {
            "sample_rate_hz": 22050,
            "samples": 10240,
            "channels": 1,
            "clamp": [-1.0, 1.0],
            "encoder_normalized": True,
            "agree_vae_sampling": "independent_fixed_seed",
            "observation_encodes_per_query": 1,
            "candidate_encodes_per_rir": max(SCORE_SAMPLE_COUNTS),
        },
        "summary": summary,
        "summary_by_k": summary_by_k,
        "representative_cases": [
            {field: item[field] for field in case_fields} for item in representative
        ],
        "results": results,
    }
    payload["sha256"] = canonical_sha256(payload)
    _atomic_text(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.output_md, render_oracle_markdown(payload))
    print(
        json.dumps(
            {
                "sha256": payload["sha256"],
                "queries": len(results),
                "elapsed_seconds": time.perf_counter() - started,
                "summary_by_k": {
                    key: {
                        "target_recall_at_1": value["target_recall_at_1"],
                        "median_localization_error_m": value[
                            "median_localization_error_m"
                        ],
                        "median_target_margin": value["median_target_margin"],
                    }
                    for key, value in summary_by_k.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
