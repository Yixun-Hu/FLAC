#!/usr/bin/env python3
"""Aggregate recorded five-method query latency on the strict 97-query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
    "fem_agree",
)
LABELS = {
    "vanilla_flac": "Vanilla FLAC",
    "fa_bf_flac": "FA-BF FLAC",
    "yawaug_flac": "Yaw-Augmented FLAC",
    "few_shot_rir": "Few-ShotRIR",
    "fem_agree": "FEM--AGREE (Depth-AABB)",
}


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt SHA-256 in {path}")
    return payload


def load_query_results(directories: list[Path], expected_indices: set[int]) -> dict[int, dict]:
    results = {}
    for directory in directories:
        for path in sorted((directory / "queries").glob("query_*.json")):
            payload = load_hashed_json(path)
            index = int(payload["query_index"])
            if index not in expected_indices:
                continue
            if index in results:
                raise RuntimeError(f"duplicate query {index} across {directories}")
            results[index] = payload
    if set(results) != expected_indices:
        missing = sorted(expected_indices - set(results))
        raise RuntimeError(f"query coverage mismatch; missing {missing}")
    return results


def load_flat_results(directory: Path, pattern: str, expected_indices: set[int]) -> dict[int, dict]:
    results = {}
    for path in sorted(directory.glob(pattern)):
        payload = load_hashed_json(path)
        index = int(payload["query_index"])
        if index not in expected_indices:
            continue
        if index in results:
            raise RuntimeError(f"duplicate query {index} in {directory}")
        results[index] = payload
    if set(results) != expected_indices:
        missing = sorted(expected_indices - set(results))
        raise RuntimeError(f"query coverage mismatch in {directory}; missing {missing}")
    return results


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("latency values must be a nonempty finite vector")
    return {
        "query_count": int(len(array)),
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p90_seconds": float(np.quantile(array, 0.9)),
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "summed_seconds": float(array.sum()),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Five-method recorded inference latency",
        "",
        "Scope: the same strict matched 14-room/97-query subset used by the primary "
        "localization table. Values exclude one-time checkpoint loading and report "
        "recorded per-query method execution.",
        "",
        "## Overall",
        "",
        "| Method | Mean [s] | Median [s] | P90 [s] | Min--max [s] | Recorded execution |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        row = payload["overall"][method]
        lines.append(
            f"| {LABELS[method]} | {row['mean_seconds']:.2f} | "
            f"{row['median_seconds']:.2f} | {row['p90_seconds']:.2f} | "
            f"{row['minimum_seconds']:.2f}--{row['maximum_seconds']:.2f} | "
            f"{payload['configurations'][method]['recorded_execution']} |"
        )

    lines.extend(
        [
            "",
            "## Amortized latency per candidate",
            "",
            "| Method | Total time / all candidates [ms] | Median query ratio [ms] | P90 query ratio [ms] |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = payload["per_candidate"][method]
        lines.append(
            f"| {LABELS[method]} | {1000.0 * row['amortized_seconds']:.2f} | "
            f"{1000.0 * row['median_seconds']:.2f} | "
            f"{1000.0 * row['p90_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## By AcousticRooms scene type: mean / median seconds/query",
            "",
            "| Scene type | n | Vanilla | FA-BF | Yaw-Aug. | Few-ShotRIR | FEM--AGREE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene, block in payload["by_scene"].items():
        lines.append(
            f"| {scene} | {block['query_count']} | "
            + " | ".join(
                f"{block['methods'][method]['mean_seconds']:.2f} / "
                f"{block['methods'][method]['median_seconds']:.2f}"
                for method in METHODS
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## By exact room: median seconds/query",
            "",
            "| Room | n | Vanilla | FA-BF | Yaw-Aug. | Few-ShotRIR | FEM--AGREE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for room, block in payload["by_room"].items():
        lines.append(
            f"| {room} | {block['query_count']} | "
            + " | ".join(f"{block['methods'][method]['median_seconds']:.2f}" for method in METHODS)
            + " |"
        )

    fem = payload["fem_components"]
    lines.extend(
        [
            "",
            "## FEM--AGREE component means",
            "",
            f"- Depth-AABB mesh + operators + 102-bin FEM solve: {fem['fem_forward_mean_seconds']:.2f} s/query.",
            f"- Frozen AGREE scoring: {fem['agree_score_mean_seconds']:.2f} s/query.",
            f"- Combined: {fem['combined_mean_seconds']:.2f} s/query.",
            "",
            "## Interpretation boundary",
            "",
            "The three FLAC rows are measured joint K_gen={1,4,8} passes, so they "
            "generate eight samples per candidate and derive the K=1/4 prefixes from "
            "the same score matrix. Few-ShotRIR is a measured joint K_ctx={1,8} pass. "
            "FEM--AGREE combines CPU FEM forward time with GPU AGREE scoring. These are "
            "auditable observed pipeline latencies, but hardware and execution shape are "
            "not normalized; a publication-grade isolated-primary-setting benchmark "
            "should rerun all methods under one fixed latency harness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--vanilla-dir", type=Path, action="append", required=True)
    parser.add_argument("--fa-bf-dir", type=Path, action="append", required=True)
    parser.add_argument("--yawaug-dir", type=Path, action="append", required=True)
    parser.add_argument("--few-shot-dir", type=Path, action="append", required=True)
    parser.add_argument("--fem-omp-dir", type=Path, required=True)
    parser.add_argument("--fem-agree-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    selection = load_hashed_json(args.selection.resolve())
    records = {int(record["index"]): record for record in selection["records"]}
    if len(records) != 97:
        raise RuntimeError(f"expected 97 strict queries, got {len(records)}")
    expected_indices = set(records)

    learned = {
        "vanilla_flac": load_query_results(args.vanilla_dir, expected_indices),
        "fa_bf_flac": load_query_results(args.fa_bf_dir, expected_indices),
        "yawaug_flac": load_query_results(args.yawaug_dir, expected_indices),
        "few_shot_rir": load_query_results(args.few_shot_dir, expected_indices),
    }
    fem_omp = load_flat_results(
        args.fem_omp_dir.resolve(), "query_*_depth_aabb_result.json", expected_indices
    )
    fem_agree = load_flat_results(
        args.fem_agree_dir.resolve(), "query_*.json", expected_indices
    )

    values: dict[str, dict[int, float]] = {}
    for method, results in learned.items():
        values[method] = {index: float(result["elapsed_seconds"]) for index, result in results.items()}
    values["fem_agree"] = {}
    for index in sorted(expected_indices):
        reference = learned["vanilla_flac"][index]
        reference_candidate_count = int(reference["candidate_count"])
        for method, results in learned.items():
            result = results[index]
            if result["query_id"] != reference["query_id"] or result["room"] != reference["room"]:
                raise RuntimeError(f"learned-method identity mismatch at query {index}: {method}")
            if int(result["candidate_count"]) != reference_candidate_count:
                raise RuntimeError(f"learned-method candidate-count mismatch at query {index}: {method}")
        omp = fem_omp[index]
        agree = fem_agree[index]
        if omp["query_id"] != reference["query_id"] or agree["query_id"] != reference["query_id"]:
            raise RuntimeError(f"FEM identity mismatch at query {index}")
        if (
            int(omp["candidate_count"]) != reference_candidate_count
            or int(agree["candidate_count"]) != reference_candidate_count
        ):
            raise RuntimeError(f"FEM candidate-count mismatch at query {index}")
        if agree["source_fem_result_sha256"] != omp["sha256"]:
            raise RuntimeError(f"FEM source hash mismatch at query {index}")
        values["fem_agree"][index] = float(omp["runtime_seconds"]["total"]) + float(
            agree["runtime_seconds"]["total"]
        )

    scenes = {index: learned["vanilla_flac"][index]["scene"] for index in expected_indices}
    rooms = {index: learned["vanilla_flac"][index]["room"] for index in expected_indices}
    overall = {
        method: summarize([values[method][index] for index in sorted(expected_indices)])
        for method in METHODS
    }
    candidate_counts = {
        index: int(learned["vanilla_flac"][index]["candidate_count"])
        for index in expected_indices
    }
    total_candidate_evaluations = sum(candidate_counts.values())
    per_candidate = {}
    for method in METHODS:
        ratios = [
            values[method][index] / candidate_counts[index]
            for index in sorted(expected_indices)
        ]
        per_candidate[method] = summarize(ratios)
        per_candidate[method]["amortized_seconds"] = float(
            sum(values[method].values()) / total_candidate_evaluations
        )
        per_candidate[method]["candidate_evaluations"] = total_candidate_evaluations

    def grouped(labels: dict[int, str]) -> dict:
        groups = defaultdict(list)
        for index, label in labels.items():
            groups[label].append(index)
        output = {}
        for label in sorted(groups):
            indices = sorted(groups[label])
            output[label] = {
                "query_count": len(indices),
                "methods": {
                    method: summarize([values[method][index] for index in indices])
                    for method in METHODS
                },
            }
        return output

    fem_forward = [float(fem_omp[index]["runtime_seconds"]["total"]) for index in expected_indices]
    agree_score = [float(fem_agree[index]["runtime_seconds"]["total"]) for index in expected_indices]
    payload = {
        "schema_version": 1,
        "scope": {
            "room_count": 14,
            "query_count": 97,
            "selection_sha256": selection["sha256"],
            "aggregation": "query micro",
            "checkpoint_startup_included": False,
        },
        "configurations": {
            "vanilla_flac": {
                "recorded_execution": "joint K_gen=1/4/8 GPU pass",
                "hardware": "RTX A6000",
            },
            "fa_bf_flac": {
                "recorded_execution": "joint K_gen=1/4/8 GPU pass",
                "hardware": "RTX A6000",
            },
            "yawaug_flac": {
                "recorded_execution": "joint K_gen=1/4/8 GPU pass",
                "hardware": "RTX A6000",
            },
            "few_shot_rir": {
                "recorded_execution": "joint K_ctx=1/8 GPU pass",
                "hardware": "RTX A6000",
            },
            "fem_agree": {
                "recorded_execution": "Depth-AABB + 102-bin FEM + AGREE K=1/4/8 scoring",
                "hardware": "12-thread CPU FEM + RTX A6000 scoring",
            },
        },
        "overall": overall,
        "per_candidate": per_candidate,
        "by_scene": grouped(scenes),
        "by_room": grouped(rooms),
        "fem_components": {
            "fem_forward_mean_seconds": float(np.mean(fem_forward)),
            "agree_score_mean_seconds": float(np.mean(agree_score)),
            "combined_mean_seconds": float(np.mean(np.asarray(fem_forward) + np.asarray(agree_score))),
        },
    }
    payload["sha256"] = canonical_sha256(payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    json_temp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    md_temp = args.output_md.with_suffix(args.output_md.suffix + ".tmp")
    json_temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_temp.write_text(render_markdown(payload))
    os.replace(json_temp, args.output_json)
    os.replace(md_temp, args.output_md)
    print(json.dumps({"sha256": payload["sha256"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
