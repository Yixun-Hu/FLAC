#!/usr/bin/env python3
"""Aggregate the complete mixed-hardware FEM--AGREE/OMP generate=1 ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256
from src.localization.runner import file_sha256, verify_hashed_payload
from src.localization.scoring import localization_metrics


# Values transcribed from the user-supplied seven-query L40 report. The script
# independently resolves each coordinate to a unique frozen candidate and
# recomputes the localization metrics below.
EXTERNAL_SEVEN = {
    30: {
        "agree_prediction_global": [5.0, 7.0, 1.0],
        "omp_prediction_global": [5.5, 6.5, 1.0],
        "agree_selector_seconds": 18.705201,
        "agree_total_seconds": 1561.112814,
        "omp_total_seconds": 1542.408,
    },
    42: {
        "agree_prediction_global": [3.5, 11.0, 2.0],
        "omp_prediction_global": [5.5, 6.0, 1.5],
        "agree_selector_seconds": 19.500204,
        "agree_total_seconds": 1547.749989,
        "omp_total_seconds": 1528.250,
    },
    535: {
        "agree_prediction_global": [13.0, 18.5, 1.0],
        "omp_prediction_global": [8.5, 15.0, 2.0],
        "agree_selector_seconds": 19.192533,
        "agree_total_seconds": 1610.463367,
        "omp_total_seconds": 1591.271,
    },
    715: {
        "agree_prediction_global": [3.5, 14.0, 2.0],
        "omp_prediction_global": [3.0, 16.5, 2.0],
        "agree_selector_seconds": 19.545311,
        "agree_total_seconds": 1587.237606,
        "omp_total_seconds": 1567.692,
    },
    917: {
        "agree_prediction_global": [14.5, 15.0, 2.5],
        "omp_prediction_global": [8.5, 16.5, 2.0],
        "agree_selector_seconds": 19.943322,
        "agree_total_seconds": 1577.919841,
        "omp_total_seconds": 1557.977,
    },
    3800: {
        "agree_prediction_global": [11.5, 21.5, 2.0],
        "omp_prediction_global": [11.5, 22.5, 1.5],
        "agree_selector_seconds": 14.229694,
        "agree_total_seconds": 4502.954337,
        "omp_total_seconds": 4488.725,
    },
    3841: {
        "agree_prediction_global": [13.0, 2.5, 2.0],
        "omp_prediction_global": [8.0, 2.5, 2.0],
        "agree_selector_seconds": 13.850458,
        "agree_total_seconds": 4540.200947,
        "omp_total_seconds": 4526.350,
    },
}
EXTERNAL_EIGHT = {335, 699, 3550, 3685, 3695, 3934, 4137, 4206}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def hashed(payload: dict) -> dict:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def load_hashed_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text())
    verify_hashed_payload(payload, label)
    return payload


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and nonempty")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sum": float(array.sum()),
    }


def resolve_prediction(candidates: np.ndarray, coordinate: list[float]) -> int:
    matches = np.flatnonzero(
        np.all(np.isclose(candidates, np.asarray(coordinate), atol=1e-6, rtol=0.0), axis=1)
    )
    if len(matches) != 1:
        raise RuntimeError(f"prediction coordinate has {len(matches)} candidate matches: {coordinate}")
    return int(matches[0])


def recompute_metrics(
    candidates: np.ndarray, source_global: list[float], prediction_index: int
) -> dict:
    metrics = localization_metrics(candidates, source_global, prediction_index)
    metrics["prediction_global"] = candidates[prediction_index].astype(float).tolist()
    return metrics


def load_external_eight(path: Path) -> dict[int, dict[str, dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    output: dict[int, dict[str, dict]] = {}
    for row in rows:
        expected = row.pop("sha256")
        if canonical_sha256(row) != expected:
            raise RuntimeError(f"invalid external-eight row hash for query {row.get('query_index')}")
        row["sha256"] = expected
        index = int(row["query_index"])
        method = str(row["method"])
        if index not in EXTERNAL_EIGHT or method not in {"fem_agree", "fem_omp"}:
            raise RuntimeError("unexpected row in external-eight JSONL")
        if method in output.setdefault(index, {}):
            raise RuntimeError(f"duplicate {method} row for query {index}")
        output[index][method] = row
    if set(output) != EXTERNAL_EIGHT or any(set(value) != {"fem_agree", "fem_omp"} for value in output.values()):
        raise RuntimeError("external-eight method/query coverage mismatch")
    return output


def load_candidate_query(query_dir: Path, index: int) -> tuple[dict, np.ndarray]:
    result_path = query_dir / f"query_{index:05d}.json"
    result = load_hashed_json(result_path, f"frozen query {index}")
    arrays_path = query_dir / f"query_{index:05d}.npz"
    if file_sha256(arrays_path) != result["arrays_sha256"]:
        raise RuntimeError(f"query {index} frozen candidate artifact hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        candidates = archive["candidates"].copy()
    if candidates.shape != (int(result["candidate_count"]), 3):
        raise RuntimeError(f"query {index} candidate shape mismatch")
    return result, candidates


def native_metric_from_source(source: dict, index: int) -> dict:
    result_path = Path(source["result_file"])
    result = load_hashed_json(result_path, f"native OMP query {index}")
    if int(result["query_index"]) != index:
        raise RuntimeError(f"native OMP source identity mismatch for query {index}")
    return result["metrics"]


def method_summary(rows: list[dict], method: str) -> dict:
    totals = [float(row[method]["total_seconds"]) for row in rows]
    ratios = [float(row[method]["ms_per_candidate"]) / 1000.0 for row in rows]
    candidates = sum(int(row["candidate_count"]) for row in rows)
    errors = [float(row[method]["metrics"]["localization_error_m"]) for row in rows]
    native = [row for row in rows if row[method]["status"] == "native"]
    fallback = [row for row in rows if row[method]["status"] == "coverage_fallback"]
    selector_rows = [row for row in native if row[method]["selector_seconds"] is not None]
    result = {
        "query_count": len(rows),
        "native_query_count": len(native),
        "coverage_fallback_query_count": len(fallback),
        "candidate_evaluations": candidates,
        "per_query_seconds": summarize(totals),
        "per_candidate_seconds": {
            "macro": summarize(ratios),
            "pooled": float(sum(totals) / candidates),
        },
        "accuracy": {
            "mean_localization_error_m": float(np.mean(errors)),
            "median_localization_error_m": float(np.median(errors)),
            "p90_localization_error_m": float(np.quantile(errors, 0.9)),
            "success_0_5m": float(np.mean([row[method]["metrics"]["success_0_5m"] for row in rows])),
            "success_1_0m": float(np.mean([row[method]["metrics"]["success_1_0m"] for row in rows])),
            "oracle_normalized_success_0_5m": float(
                np.mean(
                    [row[method]["metrics"]["oracle_normalized_success_0_5m"] for row in rows]
                )
            ),
        },
    }
    if selector_rows:
        selector_seconds = [float(row[method]["selector_seconds"]) for row in selector_rows]
        selector_ratios = [
            float(row[method]["selector_seconds"]) / int(row["candidate_count"])
            for row in selector_rows
        ]
        result["native_selector_latency"] = {
            "query_count": len(selector_rows),
            "candidate_evaluations": sum(int(row["candidate_count"]) for row in selector_rows),
            "per_query_seconds": summarize(selector_seconds),
            "per_candidate_seconds": {
                "macro": summarize(selector_ratios),
                "pooled": float(
                    sum(selector_seconds)
                    / sum(int(row["candidate_count"]) for row in selector_rows)
                ),
            },
        }
    return result


def render_markdown(summary: dict) -> str:
    lines = [
        "# FEM--AGREE vs FEM--OMP: complete 16-room generate=1 aggregation",
        "",
        (
            f"Scope: {summary['scope']['room_count']} rooms / {summary['scope']['query_count']} "
            f"queries / {summary['scope']['candidate_evaluations']:,} candidate-query pairs. "
            "Both methods use K_ctx=8 and one simulated RIR per candidate. The 112 strict-coverage "
            "queries are native and the 16 coverage failures use the same deterministic random choice."
        ),
        "",
        "Latency is component-combined and hardware-mixed: 97 AGREE selectors were measured on RTX A6000; "
        "the 15 Cafe/Auditorium native queries use the supplied L40 records. No hardware normalization is applied.",
        "",
        "## Per-query and per-candidate latency",
        "",
        "| Method | Native | Fallback | Mean [s/query] | Median | P90 | Total [s] | Macro mean [ms/candidate] | Pooled [ms/candidate] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("fem_agree", "FEM--AGREE"), ("fem_omp", "FEM--OMP")):
        item = summary["methods"][method]
        query = item["per_query_seconds"]
        candidate = item["per_candidate_seconds"]
        lines.append(
            f"| {label} | {item['native_query_count']} | {item['coverage_fallback_query_count']} | "
            f"{query['mean']:.6f} | {query['median']:.6f} | {query['p90']:.6f} | "
            f"{query['sum']:.6f} | {1000 * candidate['macro']['mean']:.6f} | "
            f"{1000 * candidate['pooled']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Macro mean first computes latency/candidate within each query and then averages the 128 ratios. "
            "Pooled divides total latency by all 92,608 candidate-query pairs.",
            "",
            "## Accuracy",
            "",
            "| Method | Mean error [m] | Median [m] | P90 [m] | SR@0.5m | SR@1m | Resolution-aware SR@0.5m |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in (("fem_agree", "FEM--AGREE"), ("fem_omp", "FEM--OMP")):
        metric = summary["methods"][method]["accuracy"]
        lines.append(
            f"| {label} | {metric['mean_localization_error_m']:.6f} | "
            f"{metric['median_localization_error_m']:.6f} | "
            f"{metric['p90_localization_error_m']:.6f} | "
            f"{100 * metric['success_0_5m']:.3f}% | {100 * metric['success_1_0m']:.3f}% | "
            f"{100 * metric['oracle_normalized_success_0_5m']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Detailed records are in `per_query.csv`, `per_query.jsonl`, and `per_candidate.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csvs(output_dir: Path, rows: list[dict]) -> None:
    query_fields = [
        "query_index",
        "query_id",
        "room",
        "candidate_count",
        "agree_status",
        "agree_total_seconds",
        "agree_ms_per_candidate",
        "agree_selector_seconds",
        "agree_prediction_index",
        "agree_error_m",
        "agree_source",
        "agree_hardware",
        "omp_status",
        "omp_total_seconds",
        "omp_ms_per_candidate",
        "omp_selector_seconds",
        "omp_prediction_index",
        "omp_error_m",
        "omp_source",
        "omp_hardware",
    ]
    query_path = output_dir / "per_query.csv"
    temporary = query_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=query_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_index": row["query_index"],
                    "query_id": row["query_id"],
                    "room": row["room"],
                    "candidate_count": row["candidate_count"],
                    "agree_status": row["fem_agree"]["status"],
                    "agree_total_seconds": row["fem_agree"]["total_seconds"],
                    "agree_ms_per_candidate": row["fem_agree"]["ms_per_candidate"],
                    "agree_selector_seconds": row["fem_agree"]["selector_seconds"],
                    "agree_prediction_index": row["fem_agree"]["metrics"]["prediction_index"],
                    "agree_error_m": row["fem_agree"]["metrics"]["localization_error_m"],
                    "agree_source": row["fem_agree"]["source"],
                    "agree_hardware": row["fem_agree"]["hardware"],
                    "omp_status": row["fem_omp"]["status"],
                    "omp_total_seconds": row["fem_omp"]["total_seconds"],
                    "omp_ms_per_candidate": row["fem_omp"]["ms_per_candidate"],
                    "omp_selector_seconds": row["fem_omp"]["selector_seconds"],
                    "omp_prediction_index": row["fem_omp"]["metrics"]["prediction_index"],
                    "omp_error_m": row["fem_omp"]["metrics"]["localization_error_m"],
                    "omp_source": row["fem_omp"]["source"],
                    "omp_hardware": row["fem_omp"]["hardware"],
                }
            )
    os.replace(temporary, query_path)

    candidate_path = output_dir / "per_candidate.csv"
    temporary = candidate_path.with_suffix(".csv.tmp")
    fields = [
        "method",
        "query_index",
        "room",
        "candidate_count",
        "total_seconds",
        "seconds_per_candidate",
        "ms_per_candidate",
        "selector_seconds",
        "selector_ms_per_candidate",
        "status",
        "source",
        "hardware",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in ("fem_agree", "fem_omp"):
                value = row[method]
                selector_ms = (
                    None
                    if value["selector_seconds"] is None
                    else 1000.0 * value["selector_seconds"] / row["candidate_count"]
                )
                writer.writerow(
                    {
                        "method": method,
                        "query_index": row["query_index"],
                        "room": row["room"],
                        "candidate_count": row["candidate_count"],
                        "total_seconds": value["total_seconds"],
                        "seconds_per_candidate": value["total_seconds"] / row["candidate_count"],
                        "ms_per_candidate": value["ms_per_candidate"],
                        "selector_seconds": value["selector_seconds"],
                        "selector_ms_per_candidate": selector_ms,
                        "status": value["status"],
                        "source": value["source"],
                        "hardware": value["hardware"],
                    }
                )
    os.replace(temporary, candidate_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-selection", type=Path, required=True)
    parser.add_argument("--strict-selection", type=Path, required=True)
    parser.add_argument("--fem-summary", type=Path, required=True)
    parser.add_argument("--selector-latency", type=Path, required=True)
    parser.add_argument("--fallback-latency", type=Path, required=True)
    parser.add_argument("--omp-merged", type=Path, required=True)
    parser.add_argument("--frozen-query-dir", type=Path, required=True)
    parser.add_argument("--agree-97-dir", type=Path, required=True)
    parser.add_argument("--external-eight-jsonl", type=Path, required=True)
    parser.add_argument("--external-seven-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    full = load_hashed_json(args.full_selection.resolve(), "full 128 selection")
    strict = load_hashed_json(args.strict_selection.resolve(), "strict 112 selection")
    fem_summary = load_hashed_json(args.fem_summary.resolve(), "FEM latency summary")
    selector = load_hashed_json(args.selector_latency.resolve(), "selector latency")
    agree_97_summary = load_hashed_json(
        (args.agree_97_dir.resolve() / "summary.json"), "AGREE 97 summary"
    )
    fallback_payload = json.loads(args.fallback_latency.resolve().read_text())
    omp_merged = json.loads(args.omp_merged.resolve().read_text())
    external_eight = load_external_eight(args.external_eight_jsonl.resolve())

    records = {int(record["index"]): record for record in full["records"]}
    strict_indices = {int(record["index"]) for record in strict["records"]}
    failure_indices = set(records) - strict_indices
    agree_97_paths = {
        int(path.stem.split("_")[1]): path
        for path in (args.agree_97_dir.resolve() / "queries").glob("query_*.json")
    }
    primary_indices = set(agree_97_paths)
    if len(records) != 128 or len(strict_indices) != 112 or len(failure_indices) != 16:
        raise RuntimeError("unexpected 128/112/16 coverage accounting")
    if len(primary_indices) != 97 or primary_indices | set(EXTERNAL_SEVEN) | EXTERNAL_EIGHT != strict_indices:
        raise RuntimeError("native AGREE source partition mismatch")
    if agree_97_summary["query_count"] != 97:
        raise RuntimeError("AGREE 97 summary scope mismatch")

    fem_queries = {int(index): value for index, value in fem_summary["queries"].items()}
    selector_queries = {int(row["query_index"]): row for row in selector["queries"]}
    fallback_queries = {int(row["query_index"]): row for row in fallback_payload["queries"]}
    omp_queries = {int(row["query_index"]): row for row in omp_merged["per_query"]}
    if set(fem_queries) != set(records) or set(selector_queries) != set(records):
        raise RuntimeError("FEM/selector 128-query coverage mismatch")
    if set(fallback_queries) != failure_indices or set(omp_queries) != strict_indices:
        raise RuntimeError("fallback/OMP coverage mismatch")

    report_text = args.external_seven_report.resolve().read_text()
    for index, source in EXTERNAL_SEVEN.items():
        if f"| {index} |" not in report_text:
            raise RuntimeError(f"external-seven report does not contain query {index}")
        for value in (source["agree_total_seconds"], source["omp_total_seconds"]):
            if f"{value:,.3f}" not in report_text:
                raise RuntimeError(f"external-seven latency {value} absent for query {index}")

    rows = []
    for index in sorted(records):
        selected = records[index]
        frozen, candidates = load_candidate_query(args.frozen_query_dir.resolve(), index)
        if selected["query_id"] != frozen["query_id"] or int(selected["candidate_count"]) != len(candidates):
            raise RuntimeError(f"frozen identity mismatch for query {index}")
        candidate_count = len(candidates)
        source_global = frozen["source_global"]

        if index in failure_indices:
            fallback = fallback_queries[index]
            prediction_index = int(fallback["prediction_index"])
            random_metric = frozen["random_candidate_metrics"]
            if prediction_index != int(random_metric["prediction_index"]):
                raise RuntimeError(f"random fallback prediction mismatch for query {index}")
            metrics = recompute_metrics(candidates, source_global, prediction_index)
            total_seconds = float(fallback["median_seconds"]["fallback_total_seconds"])
            selector_seconds = float(
                fallback["median_seconds"]["random_candidate_selection_seconds"]
            )
            agree = {
                "status": "coverage_fallback",
                "source": "strict_coverage_failure_deterministic_random",
                "hardware": "CPU fallback median of 3 repeats",
                "total_seconds": total_seconds,
                "selector_seconds": selector_seconds,
                "metrics": metrics,
            }
            omp = dict(agree)
        elif index in primary_indices:
            agree_result = load_hashed_json(agree_97_paths[index], f"A6000 AGREE query {index}")
            omp_metric = native_metric_from_source(omp_queries[index]["source"], index)
            omp_total = float(fem_queries[index]["methods"]["fem_omp"]["median_seconds"])
            omp_selector = float(selector_queries[index]["fem_omp"]["median_seconds"])
            agree_selector = float(
                agree_result["timing_seconds"]["agree_selector_total_seconds"]
            )
            base_fem = omp_total - omp_selector
            if base_fem <= 0.0:
                raise RuntimeError(f"invalid FEM base latency for query {index}")
            agree = {
                "status": "native",
                "source": "local_primary_FEM_plus_A6000_generate1_score_only",
                "hardware": "12-thread CPU FEM + NVIDIA RTX A6000 selector",
                "total_seconds": base_fem + agree_selector,
                "selector_seconds": agree_selector,
                "metrics": agree_result["localization_metrics"],
            }
            omp = {
                "status": "native",
                "source": "local_primary_FEM_plus_actual_OMP_selector",
                "hardware": "12-thread CPU FEM/OMP",
                "total_seconds": omp_total,
                "selector_seconds": omp_selector,
                "metrics": omp_metric,
            }
        elif index in EXTERNAL_EIGHT:
            agree_row = external_eight[index]["fem_agree"]
            omp_row = external_eight[index]["fem_omp"]
            for row in (agree_row, omp_row):
                if int(row["candidate_count"]) != candidate_count or row["query_id"] != frozen["query_id"]:
                    raise RuntimeError(f"external-eight identity mismatch for query {index}")
            agree_index = int(agree_row["prediction_index"])
            omp_index = int(omp_row["prediction_index"])
            agree = {
                "status": "native",
                "source": "external_L40_eight_query_native_rerun",
                "hardware": f"{agree_row['cpu_model']} + {agree_row['gpu_model']}",
                "total_seconds": float(agree_row["inference_total_seconds"]),
                "observed_wall_seconds": float(agree_row["wall_clock_total_seconds"]),
                "selector_seconds": float(agree_row["agree_selector_total_seconds"]),
                "metrics": recompute_metrics(candidates, source_global, agree_index),
                "source_record_sha256": agree_row["sha256"],
            }
            omp = {
                "status": "native",
                "source": "external_L40_eight_query_native_rerun",
                "hardware": f"{omp_row['cpu_model']} + {omp_row['gpu_model']}",
                "total_seconds": float(omp_row["inference_total_seconds"]),
                "observed_wall_seconds": float(omp_row["wall_clock_total_seconds"]),
                "selector_seconds": float(omp_row["omp_selector_total_seconds"]),
                "metrics": recompute_metrics(candidates, source_global, omp_index),
                "source_record_sha256": omp_row["sha256"],
            }
        else:
            source = EXTERNAL_SEVEN[index]
            agree_index = resolve_prediction(candidates, source["agree_prediction_global"])
            omp_index = resolve_prediction(candidates, source["omp_prediction_global"])
            agree = {
                "status": "native",
                "source": "external_L40_seven_query_report_component_combined",
                "hardware": "external 13-core CPU FEM + NVIDIA L40 selector",
                "total_seconds": float(source["agree_total_seconds"]),
                "selector_seconds": float(source["agree_selector_seconds"]),
                "metrics": recompute_metrics(candidates, source_global, agree_index),
            }
            omp = {
                "status": "native",
                "source": "external_seven_query_FEM_OMP_report",
                "hardware": "external 13-core CPU FEM/OMP",
                "total_seconds": float(source["omp_total_seconds"]),
                "selector_seconds": None,
                "metrics": recompute_metrics(candidates, source_global, omp_index),
            }

        for method in (agree, omp):
            if method["total_seconds"] < 0.0 or not np.isfinite(method["total_seconds"]):
                raise RuntimeError(f"invalid total latency for query {index}")
            method["ms_per_candidate"] = float(
                1000.0 * method["total_seconds"] / candidate_count
            )
        rows.append(
            hashed(
                {
                    "schema_version": 1,
                    "query_index": index,
                    "query_id": frozen["query_id"],
                    "scene": frozen["scene"],
                    "room": frozen["room"],
                    "candidate_count": candidate_count,
                    "candidate_indices_sha256": frozen["candidate_indices_sha256"],
                    "fem_agree": agree,
                    "fem_omp": omp,
                }
            )
        )

    if sum(row["candidate_count"] for row in rows) != 92608:
        raise RuntimeError("candidate-query total mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "full_selection": str(args.full_selection.resolve()),
        "full_selection_sha256": full["sha256"],
        "strict_selection": str(args.strict_selection.resolve()),
        "strict_selection_sha256": strict["sha256"],
        "fem_summary": str(args.fem_summary.resolve()),
        "fem_summary_sha256": fem_summary["sha256"],
        "selector_latency": str(args.selector_latency.resolve()),
        "selector_latency_sha256": selector["sha256"],
        "fallback_latency": str(args.fallback_latency.resolve()),
        "fallback_latency_file_sha256": file_sha256(args.fallback_latency.resolve()),
        "omp_merged": str(args.omp_merged.resolve()),
        "omp_merged_file_sha256": file_sha256(args.omp_merged.resolve()),
        "agree_97_summary": str((args.agree_97_dir.resolve() / "summary.json")),
        "agree_97_summary_sha256": agree_97_summary["sha256"],
        "external_eight_jsonl": str(args.external_eight_jsonl.resolve()),
        "external_eight_jsonl_file_sha256": file_sha256(args.external_eight_jsonl.resolve()),
        "external_seven_report": str(args.external_seven_report.resolve()),
        "external_seven_report_file_sha256": file_sha256(args.external_seven_report.resolve()),
        "aggregator": str(Path(__file__).resolve()),
        "aggregator_file_sha256": file_sha256(Path(__file__).resolve()),
    }
    summary = hashed(
        {
            "schema_version": 1,
            "record_type": "fem_agree_omp_generate1_complete_mixed_hardware",
            "scope": {
                "room_count": 16,
                "query_count": 128,
                "native_query_count": 112,
                "coverage_fallback_query_count": 16,
                "candidate_evaluations": 92608,
                "context_count": 8,
                "generated_rirs_per_candidate": 1,
            },
            "latency_caveat": (
                "component-combined, hardware-mixed observed latency; 97 AGREE selectors are "
                "A6000, 15 oversized-room AGREE selectors are L40; no normalization"
            ),
            "methods": {
                "fem_agree": method_summary(rows, "fem_agree"),
                "fem_omp": method_summary(rows, "fem_omp"),
            },
            "provenance": provenance,
        }
    )
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(output_dir / "summary.md", render_markdown(summary))
    atomic_text(
        output_dir / "per_query.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    write_csvs(output_dir, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
