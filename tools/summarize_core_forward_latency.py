#!/usr/bin/env python3
"""Summarize K_ctx=8, K_gen=1 localization inference latency on 128 queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
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


def load_query_results(directories: list[Path], expected: set[int]) -> dict[int, dict]:
    results = {}
    for directory in directories:
        for path in sorted((directory / "queries").glob("query_*.json")):
            payload = load_hashed_json(path)
            index = int(payload["query_index"])
            if index not in expected:
                continue
            if index in results:
                raise RuntimeError(f"duplicate learned query {index}")
            results[index] = payload
    if set(results) != expected:
        raise RuntimeError(f"learned query coverage mismatch; missing {sorted(expected-set(results))}")
    return results


def load_fem_results(
    primary_dir: Path,
    oversized_dir: Path,
    external_runtime_path: Path,
    records: dict[int, dict],
) -> dict[int, dict]:
    expected = set(records)
    results = {}
    for source, directory in (("local_primary_97", primary_dir), ("local_oversized", oversized_dir)):
        for path in sorted(directory.glob("query_*_depth_aabb_result.json")):
            payload = load_hashed_json(path)
            index = int(payload["query_index"])
            if index not in expected:
                continue
            if index in results:
                raise RuntimeError(f"duplicate FEM query {index}")
            runtime = payload.get("runtime_seconds", {})
            components = [
                float(runtime.get(name, float("nan")))
                for name in ("mesh_construction", "operator_construction", "fullband_solve")
            ]
            total = float(runtime.get("total", float("nan")))
            if not np.isfinite([*components, total]).all() or not np.isclose(
                sum(components), total, rtol=0.0, atol=1e-6
            ):
                raise RuntimeError(f"invalid FEM core timing in {path}")
            results[index] = {
                "query_id": payload["query_id"],
                "room": payload["room"],
                "candidate_count": int(payload["candidate_count"]),
                "candidate_indices_sha256": records[index]["candidate_indices_sha256"],
                "core_forward_seconds": total,
                "source": source,
                "source_file": str(path.resolve()),
                "source_sha256": payload["sha256"],
            }

    external = load_hashed_json(external_runtime_path)
    if external.get("record_type") != "external_server_per_query_runtime_recovery":
        raise RuntimeError("unexpected external FEM runtime record")
    for row in external["queries"]:
        index = int(row["query_index"])
        if index not in expected or index in results:
            raise RuntimeError(f"invalid or duplicate external FEM query {index}")
        results[index] = {
            "query_id": row["query_id"],
            "room": row["room"],
            "candidate_count": int(row["candidate_count"]),
            "candidate_indices_sha256": row["candidate_indices_sha256"],
            "core_forward_seconds": float(row["fem_internal_total_seconds"]),
            "source": "external_runtime_recovered",
            "source_file": str(external_runtime_path.resolve()),
            "source_sha256": external["sha256"],
            "wall_clock_elapsed_seconds_audit_only": float(
                row["wall_clock_elapsed_seconds"]
            ),
        }
    if len(results) != 112 or not set(results).issubset(expected):
        raise RuntimeError(
            "FEM observed source must be the exact 112-query subset; "
            f"found {len(results)}, missing {sorted(expected-set(results))}"
        )
    return results


def load_fem_fallback_results(
    path: Path, expected: set[int], records: dict[int, dict], selection_sha256: str
) -> dict[int, dict]:
    payload = load_hashed_json(path)
    if (
        payload.get("record_type") != "fem_strict_failure_random_candidate_latency"
        or payload.get("full_selection_sha256") != selection_sha256
        or int(payload.get("query_count", -1)) != 16
    ):
        raise RuntimeError("unexpected FEM random-fallback latency record")
    results = {}
    for row in payload["queries"]:
        index = int(row["query_index"])
        reference = records.get(index)
        if reference is None or index in results:
            raise RuntimeError(f"invalid or duplicate FEM fallback query {index}")
        if any(row[key] != reference[key] for key in ("query_id", "room", "candidate_count")):
            raise RuntimeError(f"FEM fallback identity mismatch at query {index}")
        results[index] = {
            "query_id": row["query_id"],
            "room": row["room"],
            "candidate_count": int(row["candidate_count"]),
            "candidate_indices_sha256": row["candidate_indices_sha256"],
            "core_forward_seconds": float(
                row["median_seconds"]["fallback_total_seconds"]
            ),
            "source": "strict_failure_random_candidate_measured",
            "fallback_measurement": {
                "prediction_index": int(row["prediction_index"]),
                "repeat_seconds": row["repeat_seconds"]["fallback_total_seconds"],
                "median_components_seconds": row["median_seconds"],
                "source_file": str(path.resolve()),
                "source_sha256": payload["sha256"],
            },
        }
    if set(results) != expected:
        raise RuntimeError(
            f"FEM fallback coverage mismatch; missing {sorted(expected-set(results))}"
        )
    return results


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("latency values must be a nonempty finite vector")
    return {
        "query_count": len(array),
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p90_seconds": float(np.quantile(array, 0.9)),
        "minimum_seconds": float(array.min()),
        "maximum_seconds": float(array.max()),
        "summed_seconds": float(array.sum()),
    }


def load_selector_latency(path: Path, records: dict[int, dict], selection_sha256: str) -> dict:
    payload = load_hashed_json(path)
    if (
        payload.get("record_type") != "localization_selector_latency_kctx8_kgen1"
        or payload.get("selection_sha256") != selection_sha256
        or int(payload.get("query_count", -1)) != 128
        or int(payload.get("repeat_count", -1)) != 3
    ):
        raise RuntimeError("unexpected localization selector-latency record")
    rows = {int(row["query_index"]): row for row in payload["queries"]}
    if set(rows) != set(records):
        raise RuntimeError("selector-latency query coverage mismatch")
    for index, row in rows.items():
        reference = records[index]
        if any(row[key] != reference[key] for key in ("query_id", "room", "candidate_count")):
            raise RuntimeError(f"selector-latency identity mismatch at query {index}")
    return payload | {"by_index": rows}


def render_markdown(payload: dict) -> str:
    lines = [
        "# Localization inference latency: K_ctx=8, K_gen=1",
        "",
        (
            f"Scope: {payload['scope']['room_count']} rooms / "
            f"{payload['scope']['query_count']} frozen queries."
        ),
        (
            "Latency includes model-specific conditioning, one generated RIR/response per "
            "candidate, and localization scoring/selection: AGREE for generated methods "
            "and OMP for successful FEM queries. Input loading, candidate filtering, "
            "evaluation metrics, and serialization are excluded."
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
            f"| {LABELS[method]} | {1000*row['amortized_seconds']:.3f} | "
            f"{1000*row['median_seconds']:.3f} | {1000*row['p90_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "FEM uses 112 observed successful core timings (97 local primary, 9 local "
            "oversized, and 6 recovered external per-query runtimes). The 16 "
            "strict-coverage failures use measured failure-detection plus deterministic "
            "random-candidate selection latency; no FEM solve occurs for those rows.",
            "FEM CPU hardware is mixed and is not hardware-normalized.",
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
    parser.add_argument("--fem-primary-dir", type=Path, required=True)
    parser.add_argument("--fem-oversized-dir", type=Path, required=True)
    parser.add_argument("--fem-external-runtime", type=Path, required=True)
    parser.add_argument("--fem-fallback-runtime", type=Path, required=True)
    parser.add_argument("--selector-latency", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    selection = load_hashed_json(args.selection.resolve())
    records = {int(record["index"]): record for record in selection["records"]}
    room_counts = Counter(record["room"] for record in records.values())
    if len(records) != 128 or len(room_counts) != 16 or set(room_counts.values()) != {8}:
        raise RuntimeError("core latency selection must be the balanced 16-room/128-query set")
    expected = set(records)
    learned = {
        "vanilla_flac": load_query_results(args.vanilla_dir, expected),
        "fa_bf_flac": load_query_results(args.fa_bf_dir, expected),
        "yawaug_flac": load_query_results(args.yawaug_dir, expected),
        "few_shot_rir": load_query_results(args.few_shot_dir, expected),
    }
    fem_observed = load_fem_results(
        args.fem_primary_dir.resolve(),
        args.fem_oversized_dir.resolve(),
        args.fem_external_runtime.resolve(),
        records,
    )
    fem_fallback = load_fem_fallback_results(
        args.fem_fallback_runtime.resolve(),
        expected - set(fem_observed),
        records,
        selection["sha256"],
    )
    fem = {**fem_observed, **fem_fallback}
    selector = load_selector_latency(
        args.selector_latency.resolve(), records, selection["sha256"]
    )

    values = {method: {} for method in METHODS}
    candidate_counts = {}
    queries = {}
    for index in sorted(expected):
        reference = records[index]
        candidate_count = int(reference["candidate_count"])
        candidate_counts[index] = candidate_count
        for method in ("vanilla_flac", "fa_bf_flac", "yawaug_flac"):
            result = learned[method][index]
            protocol = result.get("latency_protocol", {})
            if (
                result.get("score_sample_counts") != [1]
                or int(result.get("n_context", -1)) != 8
                or protocol.get("name") != "fem_core_aligned_kctx8_kgen1"
                or protocol.get("candidate_scoring_included") is not False
            ):
                raise RuntimeError(f"{method} query {index} has the wrong timing protocol")
            values[method][index] = float(
                result["latency_seconds"]["core_forward_total"]
                + selector["by_index"][index]["agree"]["median_seconds"][
                    "scoring_total_seconds"
                ]
            )
        few = learned["few_shot_rir"][index]
        if (
            few.get("context_counts") != [8]
            or few.get("latency_protocol", {}).get("name")
            != "fem_core_aligned_kctx8_kgen1"
        ):
            raise RuntimeError(f"Few-ShotRIR query {index} has the wrong timing protocol")
        values["few_shot_rir"][index] = float(
            few["latency_seconds"]["core_forward_total"]
            + selector["by_index"][index]["agree"]["median_seconds"][
                "scoring_total_seconds"
            ]
        )
        physical = fem[index]
        identity_results = {
            method: by_index[index] for method, by_index in learned.items()
        }
        identity_results["fem_omp"] = physical
        for method, result in identity_results.items():
            if (
                result["query_id"] != reference["query_id"]
                or result["room"] != reference["room"]
                or int(result["candidate_count"]) != candidate_count
            ):
                raise RuntimeError(f"identity mismatch for {method} query {index}")
        omp_row = selector["by_index"][index]["fem_omp"]
        if (
            physical["source"] == "strict_failure_random_candidate_measured"
        ) != (omp_row["source"] == "not_run_random_fallback"):
            raise RuntimeError(f"FEM OMP/fallback status mismatch at query {index}")
        values["fem_omp"][index] = float(
            physical["core_forward_seconds"] + omp_row["median_seconds"]
        )
        if not np.isfinite([values[m][index] for m in METHODS]).all():
            raise RuntimeError(f"non-finite latency at query {index}")
        queries[str(index)] = {
            "query_id": reference["query_id"],
            "room": reference["room"],
            "candidate_count": candidate_count,
            "latency_seconds": {method: values[method][index] for method in METHODS},
            "fem_source": physical["source"],
            "selector_latency_seconds": {
                "agree": selector["by_index"][index]["agree"]["median_seconds"],
                "fem_omp": float(omp_row["median_seconds"]),
                "fem_omp_source": omp_row["source"],
            },
        }
        if "fallback_measurement" in physical:
            queries[str(index)]["fem_fallback_measurement"] = physical[
                "fallback_measurement"
            ]

    ordered = sorted(expected)
    overall = {method: summarize([values[method][i] for i in ordered]) for method in METHODS}
    total_candidates = sum(candidate_counts.values())
    per_candidate = {}
    for method in METHODS:
        ratios = [values[method][i] / candidate_counts[i] for i in ordered]
        per_candidate[method] = summarize(ratios)
        per_candidate[method]["amortized_seconds"] = float(
            sum(values[method].values()) / total_candidates
        )
        per_candidate[method]["candidate_evaluations"] = total_candidates

    payload = {
        "schema_version": 3,
        "latency_protocol": {
            "name": "fem_core_aligned_kctx8_kgen1",
            "context_count": 8,
            "generated_rirs_per_candidate": 1,
            "input_loading_included": False,
            "candidate_filtering_included": False,
            "candidate_scoring_included": True,
            "candidate_selection_included": True,
            "evaluation_metrics_included": False,
            "result_serialization_included": False,
        },
        "scope": {
            "room_count": 16,
            "query_count": 128,
            "candidate_evaluations": total_candidates,
            "selection_sha256": selection["sha256"],
            "aggregation": "query micro",
        },
        "fem_provenance": {
            "hardware_normalized": False,
            "observed_query_count": len(fem_observed),
            "measured_random_fallback_query_count": len(fem_fallback),
            "failure_policy": {
                "method": "strict-coverage detection then deterministic random candidate",
                "fem_solve_on_failure": False,
            },
            "omp_source_counts": selector["fem_omp_source_counts"],
            "selector_latency_file": str(args.selector_latency.resolve()),
            "selector_latency_sha256": selector["sha256"],
            "source_counts": {
                source: sum(row["source"] == source for row in fem.values())
                for source in sorted({row["source"] for row in fem.values()})
            },
        },
        "overall": overall,
        "per_candidate": per_candidate,
        "queries": queries,
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


if __name__ == "__main__":
    main()
