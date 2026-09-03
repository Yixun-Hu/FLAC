#!/usr/bin/env python3
"""Summarize the unified K_ctx=8, one-RIR-per-candidate latency benchmark."""

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


def load_fem_results(directory: Path, expected_indices: set[int]) -> dict[int, dict]:
    results = {}
    for path in sorted(directory.glob("query_*_depth_aabb_result.json")):
        payload = load_hashed_json(path)
        index = int(payload["query_index"])
        if index not in expected_indices:
            continue
        if index in results:
            raise RuntimeError(f"duplicate FEM query {index} in {directory}")
        results[index] = payload
    if set(results) != expected_indices:
        missing = sorted(expected_indices - set(results))
        raise RuntimeError(f"FEM query coverage mismatch; missing {missing}")
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
        "# Unified localization inference latency: K_ctx=8, K_gen=1",
        "",
        (
            f"Scope: {payload['scope']['room_count']} rooms / "
            f"{payload['scope']['query_count']} frozen queries. Every method uses eight "
            "context RIRs and evaluates exactly one generated or simulated RIR per candidate."
        ),
        "One-time model/checkpoint loading and result serialization are excluded.",
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
    lines.extend(
        [
            "",
            "FEM--OMP latency is one contiguous Depth-AABB path: candidate-mask "
            "application, depth/AABB mesh construction, query input loading, operator "
            "construction, the 102-bin FEM solve, and Room-Helps OMP scoring.",
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    selection = load_hashed_json(args.selection.resolve())
    records = {int(record["index"]): record for record in selection["records"]}
    expected_indices = set(records)
    if len(records) != int(selection["query_count"]):
        raise RuntimeError("selection query count is inconsistent")

    learned = {
        "vanilla_flac": load_query_results(args.vanilla_dir, expected_indices),
        "fa_bf_flac": load_query_results(args.fa_bf_dir, expected_indices),
        "yawaug_flac": load_query_results(args.yawaug_dir, expected_indices),
        "few_shot_rir": load_query_results(args.few_shot_dir, expected_indices),
    }
    fem = load_fem_results(args.fem_omp_dir.resolve(), expected_indices)

    values: dict[str, dict[int, float]] = {method: {} for method in METHODS}
    candidate_counts = {}
    for index in sorted(expected_indices):
        reference = learned["vanilla_flac"][index]
        candidate_count = int(reference["candidate_count"])
        if candidate_count <= 0:
            raise RuntimeError(f"query {index} has no candidates")
        candidate_counts[index] = candidate_count
        for method in ("vanilla_flac", "fa_bf_flac", "yawaug_flac"):
            result = learned[method][index]
            if result.get("score_sample_counts") != [1] or int(result.get("n_context", -1)) != 8:
                raise RuntimeError(f"{method} query {index} is not K_ctx=8, K_gen=1")
            if (
                result["query_id"] != reference["query_id"]
                or result["room"] != reference["room"]
                or int(result["candidate_count"]) != candidate_count
            ):
                raise RuntimeError(f"learned-method identity mismatch at query {index}: {method}")
            latency = float(result["elapsed_seconds"])
            if not np.isfinite(latency) or latency < 0.0:
                raise RuntimeError(f"{method} query {index} has invalid latency")
            values[method][index] = latency

        few = learned["few_shot_rir"][index]
        if few.get("context_counts") != [8]:
            raise RuntimeError(f"Few-ShotRIR query {index} is not K_ctx=8")
        if (
            few["query_id"] != reference["query_id"]
            or few["room"] != reference["room"]
            or int(few["candidate_count"]) != candidate_count
        ):
            raise RuntimeError(f"Few-ShotRIR identity mismatch at query {index}")
        few_latency = float(few["elapsed_seconds"])
        if not np.isfinite(few_latency) or few_latency < 0.0:
            raise RuntimeError(f"Few-ShotRIR query {index} has invalid latency")
        values["few_shot_rir"][index] = few_latency

        physical = fem[index]
        protocol = physical.get("latency_protocol", {})
        if (
            protocol.get("name") != "kctx8_kgen1"
            or protocol.get("context_count") != 8
            or protocol.get("generated_rirs_per_candidate") != 1
            or protocol.get("selector") != "room_helps_pulse_stacked_omp"
        ):
            raise RuntimeError(f"FEM query {index} has the wrong latency protocol")
        if (
            physical["query_id"] != reference["query_id"]
            or physical["room"] != reference["room"]
            or int(physical["candidate_count"]) != candidate_count
        ):
            raise RuntimeError(f"FEM identity mismatch at query {index}")
        latency_fields = physical.get("latency_seconds", {})
        component_names = (
            "candidate_preparation",
            "depth_aabb_and_mesh",
            "query_input_loading",
            "operator_construction",
            "fullband_solve",
            "omp_scoring",
        )
        fem_latency = float(latency_fields.get("localization_total", float("nan")))
        component_values = [
            float(latency_fields.get(stage, float("nan"))) for stage in component_names
        ]
        if (
            not np.isfinite([fem_latency, *component_values]).all()
            or fem_latency < 0.0
            or any(value < 0.0 for value in component_values)
            or sum(component_values) > fem_latency + 1e-6
        ):
            raise RuntimeError(f"FEM query {index} has invalid latency components")
        values["fem_omp"][index] = fem_latency

    ordered_indices = sorted(expected_indices)
    overall = {
        method: summarize([values[method][index] for index in ordered_indices])
        for method in METHODS
    }
    total_candidates = sum(candidate_counts.values())
    per_candidate = {}
    for method in METHODS:
        ratios = [
            values[method][index] / candidate_counts[index] for index in ordered_indices
        ]
        per_candidate[method] = summarize(ratios)
        per_candidate[method]["amortized_seconds"] = float(
            sum(values[method].values()) / total_candidates
        )
        per_candidate[method]["candidate_evaluations"] = total_candidates

    fem_stage_names = (
        "candidate_preparation",
        "depth_aabb_and_mesh",
        "query_input_loading",
        "operator_construction",
        "fullband_solve",
        "omp_scoring",
    )
    payload = {
        "schema_version": 2,
        "latency_protocol": {
            "context_count": 8,
            "generated_rirs_per_candidate": 1,
            "flac_score_sample_counts": [1],
            "few_shot_context_counts": [8],
            "fem_selector": "room_helps_pulse_stacked_omp",
            "checkpoint_startup_included": False,
            "result_serialization_included": False,
        },
        "scope": {
            "room_count": len({record["room"] for record in records.values()}),
            "query_count": len(records),
            "candidate_evaluations": total_candidates,
            "selection_sha256": selection["sha256"],
            "aggregation": "query micro",
        },
        "overall": overall,
        "per_candidate": per_candidate,
        "queries": {
            str(index): {
                "query_id": learned["vanilla_flac"][index]["query_id"],
                "room": learned["vanilla_flac"][index]["room"],
                "candidate_count": candidate_counts[index],
                "latency_seconds": {
                    method: values[method][index] for method in METHODS
                },
                "fem_omp_components_seconds": {
                    stage: float(fem[index]["latency_seconds"][stage])
                    for stage in fem_stage_names
                },
            }
            for index in ordered_indices
        },
        "fem_omp_component_means_seconds": {
            stage: float(
                np.mean(
                    [fem[index]["latency_seconds"][stage] for index in ordered_indices]
                )
            )
            for stage in fem_stage_names
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
    print(json.dumps({"sha256": payload["sha256"], "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
