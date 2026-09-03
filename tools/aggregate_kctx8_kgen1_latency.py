#!/usr/bin/env python3
"""Aggregate repeat summaries for the unified K_ctx=8, K_gen=1 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


METHODS = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
    "fem_omp",
)
LABELS = {
    "vanilla_flac": "Vanilla FLAC",
    "fa_bf_flac": "OrbitRIR / FA-BF FLAC",
    "yawaug_flac": "Yaw-Augmented FLAC",
    "few_shot_rir": "Few-ShotRIR",
    "fem_omp": "FEM--OMP (Depth-AABB)",
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


def validate_repeat(reference: dict, repeat: dict, path: Path) -> None:
    if repeat.get("schema_version") != 3:
        raise RuntimeError(f"unsupported repeat summary schema in {path}")
    if repeat.get("latency_protocol") != reference.get("latency_protocol"):
        raise RuntimeError(f"latency protocol mismatch in {path}")
    for field in (
        "room_count",
        "query_count",
        "candidate_evaluations",
        "selection_sha256",
        "aggregation",
    ):
        if repeat.get("scope", {}).get(field) != reference.get("scope", {}).get(field):
            raise RuntimeError(f"scope field {field} mismatch in {path}")
    if set(repeat.get("overall", {})) != set(METHODS):
        raise RuntimeError(f"method coverage mismatch in {path}")
    if set(repeat.get("queries", {})) != set(reference.get("queries", {})):
        raise RuntimeError(f"query coverage mismatch in {path}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# Localization inference latency: final repeat aggregate",
        "",
        (
            f"Scope: {payload['scope']['room_count']} rooms / "
            f"{payload['scope']['query_count']} frozen queries / "
            f"{payload['repeat_count']} learned-model timing repeats. Every method uses "
            "K_ctx=8 and K_gen=1; FEM has 112 observed successful core timings and "
            "16 measured strict-failure/random-candidate fallback timings."
        ),
        (
            "For each method and query, the reported latency is the median across "
            "repeats; table statistics are then computed across frozen queries."
        ),
        (
            "AGREE/OMP scoring and candidate selection are included. Input loading, "
            "candidate filtering, evaluation metrics, and serialization are excluded."
        ),
        "",
        "| Method | Mean [s/query] | Median | P90 | Min--max |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = payload["overall"][method]
        lines.append(
            f"| {LABELS[method]} | {row['mean_seconds']:.3f} | "
            f"{row['median_seconds']:.3f} | {row['p90_seconds']:.3f} | "
            f"{row['minimum_seconds']:.3f}--{row['maximum_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Method | Dataset-wide mean [ms/candidate] | Query median | Query P90 |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = payload["per_candidate"][method]
        lines.append(
            f"| {LABELS[method]} | {1000.0 * row['amortized_seconds']:.3f} | "
            f"{1000.0 * row['median_seconds']:.3f} | "
            f"{1000.0 * row['p90_seconds']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = [path.resolve() for path in args.summary]
    if len(paths) != len(set(paths)):
        raise ValueError("repeat summary paths must be unique")
    repeats = [load_hashed_json(path) for path in paths]
    reference = repeats[0]
    for path, repeat in zip(paths, repeats):
        validate_repeat(reference, repeat, path)

    query_indices = sorted(reference["queries"], key=int)
    candidate_counts = {
        index: int(reference["queries"][index]["candidate_count"])
        for index in query_indices
    }
    if not query_indices or any(count <= 0 for count in candidate_counts.values()):
        raise RuntimeError("repeat summaries must contain queries with positive candidate counts")
    median_latency: dict[str, dict[str, float]] = {
        method: {} for method in METHODS
    }
    queries = {}
    for index in query_indices:
        reference_query = reference["queries"][index]
        for repeat, path in zip(repeats, paths):
            query = repeat["queries"][index]
            for field in ("query_id", "room", "candidate_count"):
                if query.get(field) != reference_query.get(field):
                    raise RuntimeError(f"query {index} field {field} mismatch in {path}")
            if set(query.get("latency_seconds", {})) != set(METHODS):
                raise RuntimeError(f"method coverage mismatch for query {index} in {path}")

        method_values = {}
        for method in METHODS:
            all_repeat_samples = [
                float(repeat["queries"][index]["latency_seconds"][method])
                for repeat in repeats
            ]
            if method == "fem_omp" and len(set(all_repeat_samples)) != 1:
                raise RuntimeError(f"reused FEM latency changed across repeats at query {index}")
            samples = all_repeat_samples[:1] if method == "fem_omp" else all_repeat_samples
            if not np.isfinite(samples).all() or any(value < 0.0 for value in samples):
                raise RuntimeError(f"invalid latency for {method} query {index}")
            median = float(np.median(samples))
            median_latency[method][index] = median
            method_values[method] = {
                "repeat_seconds": samples,
                "median_seconds": median,
            }
        queries[index] = {
            "query_id": reference_query["query_id"],
            "room": reference_query["room"],
            "candidate_count": candidate_counts[index],
            "methods": method_values,
            "fem_source": reference_query["fem_source"],
        }

    overall = {
        method: summarize([median_latency[method][index] for index in query_indices])
        for method in METHODS
    }
    total_candidates = sum(candidate_counts.values())
    per_candidate = {}
    for method in METHODS:
        ratios = [
            median_latency[method][index] / candidate_counts[index]
            for index in query_indices
        ]
        per_candidate[method] = summarize(ratios)
        per_candidate[method]["amortized_seconds"] = float(
            sum(median_latency[method].values()) / total_candidates
        )
        per_candidate[method]["candidate_evaluations"] = total_candidates

    payload = {
        "schema_version": 1,
        "aggregation_protocol": "per-query median over repeats, then query micro",
        "repeat_count": len(repeats),
        "measurement_repeat_count_by_method": {
            method: 1 if method == "fem_omp" else len(repeats) for method in METHODS
        },
        "repeat_summaries": [
            {"path": str(path), "sha256": repeat["sha256"]}
            for path, repeat in zip(paths, repeats)
        ],
        "latency_protocol": reference["latency_protocol"],
        "scope": reference["scope"],
        "overall": overall,
        "per_candidate": per_candidate,
        "queries": queries,
        "fem_provenance": reference["fem_provenance"],
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
    print(
        json.dumps(
            {"sha256": payload["sha256"], "output_json": str(args.output_json)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
