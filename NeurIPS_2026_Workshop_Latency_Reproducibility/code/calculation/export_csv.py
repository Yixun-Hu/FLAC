#!/usr/bin/env python3
"""Export convenient CSV views from the canonical nested JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
    "fem_omp",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, help="default: <bundle>/expected"
    )
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parents[2]
    data = bundle / "data"
    output = (args.output_dir or bundle / "expected").resolve()
    final = load(bundle / "expected" / "summary_final.json")
    selector = load(data / "selector" / "selector_latency_128.json")
    selector_by_index = {str(row["query_index"]): row for row in selector["queries"]}

    query_fields = ["query_index", "query_id", "room", "candidate_count", "fem_source"]
    for method in METHODS:
        repeat_count = int(final["measurement_repeat_count_by_method"][method])
        query_fields.extend(
            [f"{method}_repeat_{ordinal}_seconds" for ordinal in range(1, repeat_count + 1)]
        )
        query_fields.append(f"{method}_median_seconds")
    query_rows = []
    for index in sorted(final["queries"], key=int):
        query = final["queries"][index]
        row = {
            "query_index": index,
            "query_id": query["query_id"],
            "room": query["room"],
            "candidate_count": query["candidate_count"],
            "fem_source": query["fem_source"],
        }
        for method in METHODS:
            result = query["methods"][method]
            for ordinal, seconds in enumerate(result["repeat_seconds"], start=1):
                row[f"{method}_repeat_{ordinal}_seconds"] = seconds
            row[f"{method}_median_seconds"] = result["median_seconds"]
        query_rows.append(row)
    write_csv(output / "latency_per_query.csv", query_fields, query_rows)

    summary_fields = [
        "method",
        "query_count",
        "mean_seconds_per_query",
        "median_seconds_per_query",
        "p90_seconds_per_query",
        "minimum_seconds_per_query",
        "maximum_seconds_per_query",
        "summed_seconds",
        "dataset_amortized_ms_per_candidate",
        "query_median_ms_per_candidate",
        "query_p90_ms_per_candidate",
        "candidate_evaluations",
    ]
    summary_rows = []
    for method in METHODS:
        overall = final["overall"][method]
        per_candidate = final["per_candidate"][method]
        summary_rows.append(
            {
                "method": method,
                "query_count": overall["query_count"],
                "mean_seconds_per_query": overall["mean_seconds"],
                "median_seconds_per_query": overall["median_seconds"],
                "p90_seconds_per_query": overall["p90_seconds"],
                "minimum_seconds_per_query": overall["minimum_seconds"],
                "maximum_seconds_per_query": overall["maximum_seconds"],
                "summed_seconds": overall["summed_seconds"],
                "dataset_amortized_ms_per_candidate": 1000.0
                * per_candidate["amortized_seconds"],
                "query_median_ms_per_candidate": 1000.0
                * per_candidate["median_seconds"],
                "query_p90_ms_per_candidate": 1000.0
                * per_candidate["p90_seconds"],
                "candidate_evaluations": per_candidate["candidate_evaluations"],
            }
        )
    write_csv(output / "latency_summary.csv", summary_fields, summary_rows)

    selector_fields = [
        "query_index",
        "query_id",
        "room",
        "candidate_count",
        "agree_observed_encode_seconds",
        "agree_generated_encode_seconds",
        "agree_similarity_seconds",
        "agree_argmax_seconds",
        "agree_scoring_total_seconds",
        "fem_omp_source",
        "fem_omp_median_seconds",
    ]
    selector_rows = []
    for index in sorted(selector_by_index, key=int):
        query = selector_by_index[index]
        agree = query["agree"]["median_seconds"]
        selector_rows.append(
            {
                "query_index": index,
                "query_id": query["query_id"],
                "room": query["room"],
                "candidate_count": query["candidate_count"],
                "agree_observed_encode_seconds": agree["observed_encode_seconds"],
                "agree_generated_encode_seconds": agree["generated_encode_seconds"],
                "agree_similarity_seconds": agree["similarity_seconds"],
                "agree_argmax_seconds": agree["argmax_seconds"],
                "agree_scoring_total_seconds": agree["scoring_total_seconds"],
                "fem_omp_source": query["fem_omp"]["source"],
                "fem_omp_median_seconds": query["fem_omp"]["median_seconds"],
            }
        )
    write_csv(output / "selector_latency_per_query.csv", selector_fields, selector_rows)

    fem_core: dict[str, dict] = {}
    for source, directory in (
        ("local_primary_97", data / "fem" / "primary_97"),
        ("local_oversized", data / "fem" / "oversized_9"),
    ):
        for path in directory.glob("query_*_depth_aabb_result.json"):
            payload = load(path)
            runtime = payload["runtime_seconds"]
            fem_core[str(payload["query_index"])] = {
                "source": source,
                "mesh_construction_seconds": runtime["mesh_construction"],
                "operator_construction_seconds": runtime["operator_construction"],
                "fullband_solve_seconds": runtime["fullband_solve"],
                "fem_core_total_seconds": runtime["total"],
                "external_wall_clock_elapsed_seconds_audit_only": "",
            }
    external = load(data / "fem" / "external_server_6query_runtime_recovery.json")
    for payload in external["queries"]:
        fem_core[str(payload["query_index"])] = {
            "source": "external_runtime_recovered",
            "mesh_construction_seconds": "",
            "operator_construction_seconds": "",
            "fullband_solve_seconds": "",
            "fem_core_total_seconds": payload["fem_internal_total_seconds"],
            "external_wall_clock_elapsed_seconds_audit_only": payload[
                "wall_clock_elapsed_seconds"
            ],
        }
    fallback = load(data / "fem" / "fem_random_fallback_latency_16.json")
    for payload in fallback["queries"]:
        median = payload["median_seconds"]
        fem_core[str(payload["query_index"])] = {
            "source": "strict_failure_random_candidate_measured",
            "mesh_construction_seconds": "",
            "operator_construction_seconds": "",
            "fullband_solve_seconds": "",
            "fem_core_total_seconds": median["fallback_total_seconds"],
            "external_wall_clock_elapsed_seconds_audit_only": "",
            "candidate_preparation_seconds": median["candidate_preparation_seconds"],
            "strict_gate_seconds": median["depth_aabb_strict_gate_seconds"],
            "random_selection_seconds": median["random_candidate_selection_seconds"],
        }

    fem_fields = [
        "query_index",
        "query_id",
        "room",
        "candidate_count",
        "source",
        "mesh_construction_seconds",
        "operator_construction_seconds",
        "fullband_solve_seconds",
        "candidate_preparation_seconds",
        "strict_gate_seconds",
        "random_selection_seconds",
        "fem_core_or_fallback_total_seconds",
        "omp_seconds",
        "final_fem_omp_seconds",
        "external_wall_clock_elapsed_seconds_audit_only",
    ]
    fem_rows = []
    for index in sorted(final["queries"], key=int):
        query = final["queries"][index]
        source = fem_core[index]
        fem_rows.append(
            {
                "query_index": index,
                "query_id": query["query_id"],
                "room": query["room"],
                "candidate_count": query["candidate_count"],
                "source": source["source"],
                "mesh_construction_seconds": source.get("mesh_construction_seconds", ""),
                "operator_construction_seconds": source.get(
                    "operator_construction_seconds", ""
                ),
                "fullband_solve_seconds": source.get("fullband_solve_seconds", ""),
                "candidate_preparation_seconds": source.get(
                    "candidate_preparation_seconds", ""
                ),
                "strict_gate_seconds": source.get("strict_gate_seconds", ""),
                "random_selection_seconds": source.get("random_selection_seconds", ""),
                "fem_core_or_fallback_total_seconds": source["fem_core_total_seconds"],
                "omp_seconds": selector_by_index[index]["fem_omp"]["median_seconds"],
                "final_fem_omp_seconds": query["methods"]["fem_omp"]["median_seconds"],
                "external_wall_clock_elapsed_seconds_audit_only": source[
                    "external_wall_clock_elapsed_seconds_audit_only"
                ],
            }
        )
    write_csv(output / "fem_latency_per_query.csv", fem_fields, fem_rows)
    print(f"Wrote four CSV views to {output}")


if __name__ == "__main__":
    main()
