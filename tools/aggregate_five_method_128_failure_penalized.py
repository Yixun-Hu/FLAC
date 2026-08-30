#!/usr/bin/env python3
"""Aggregate five methods with explicit FEM strict-coverage failure handling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256
from src.localization.scoring import deterministic_random_candidate, localization_metrics


MODEL_ORDER = ("vanilla", "fa_bf", "yawaug", "few_shot", "fem_omp")
MODEL_LABELS = {
    "vanilla": "Vanilla FLAC",
    "fa_bf": "OrbitRIR (FA-BF FLAC)",
    "yawaug": "Yaw-Augmented FLAC",
    "few_shot": "Few-ShotRIR",
    "fem_omp": "FEM-Sabine + Room-Helps OMP (Depth-AABB)",
}
PRIMARY_METRIC_KEY = {
    "vanilla": "1",
    "fa_bf": "1",
    "yawaug": "1",
    "few_shot": "8",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_sha256(content):
        raise RuntimeError(f"stale or corrupt hashed JSON: {path}")
    return payload


def load_query_results(directories: list[Path]) -> dict[int, tuple[dict, Path]]:
    results = {}
    for directory in directories:
        for path in sorted((directory / "queries").glob("query_*.json")):
            payload = load_hashed_json(path)
            index = int(payload["query_index"])
            if index in results:
                raise RuntimeError(f"duplicate query {index} across {directories}")
            results[index] = (payload, directory)
    return results


def compact_metric(metric: dict) -> dict:
    return {
        key: metric[key]
        for key in (
            "localization_error_m",
            "oracle_error_m",
            "excess_error_m",
            "success_0_5m",
            "success_1_0m",
            "oracle_normalized_success_0_5m",
            "oracle_normalized_success_1_0m",
        )
    }


def aggregate(metrics: list[dict]) -> dict:
    errors = np.asarray(
        [metric["localization_error_m"] for metric in metrics], dtype=np.float64
    )
    return {
        "query_count": len(metrics),
        "mean_localization_error_m": float(errors.mean()),
        "median_localization_error_m": float(np.median(errors)),
        "p90_localization_error_m": float(np.quantile(errors, 0.9)),
        "success_rate_at_0_5m": float(
            np.mean([metric["success_0_5m"] for metric in metrics])
        ),
        "success_rate_at_1_0m": float(
            np.mean([metric["success_1_0m"] for metric in metrics])
        ),
        "resolution_aware_success_rate_at_0_5m": float(
            np.mean(
                [metric["oracle_normalized_success_0_5m"] for metric in metrics]
            )
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ("vanilla", "fa-bf", "yawaug", "few-shot"):
        parser.add_argument(f"--{arm}-dir", type=Path, action="append", required=True)
    parser.add_argument("--fem-summary", type=Path, required=True)
    parser.add_argument(
        "--query-scope-dir",
        type=Path,
        help="Optional learned-result directory whose query IDs define a subset",
    )
    parser.add_argument(
        "--fem-failure-policy",
        choices=("worst_candidate", "random_candidate"),
        default="worst_candidate",
        help="Fallback used when Depth-AABB cannot evaluate every frozen candidate",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for the deterministic random-candidate FEM fallback",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    arms = {
        "vanilla": load_query_results(args.vanilla_dir),
        "fa_bf": load_query_results(args.fa_bf_dir),
        "yawaug": load_query_results(args.yawaug_dir),
        "few_shot": load_query_results(args.few_shot_dir),
    }
    all_indices = set(arms["vanilla"])
    if len(all_indices) != 128:
        raise RuntimeError(f"expected 128 Vanilla queries, got {len(all_indices)}")
    for model, rows in arms.items():
        if set(rows) != all_indices:
            raise RuntimeError(f"{model} query identities do not cover the same 128")
    if args.query_scope_dir is None:
        indices = all_indices
        query_scope_source = "all aligned learned queries"
    else:
        scoped = load_query_results([args.query_scope_dir])
        indices = set(scoped)
        if not indices or not indices.issubset(all_indices):
            raise RuntimeError("query scope is empty or is not a subset of the 128")
        for index, (payload, _directory) in scoped.items():
            if payload["query_id"] != arms["vanilla"][index][0]["query_id"]:
                raise RuntimeError(f"query scope identity mismatch for {index}")
        query_scope_source = str(args.query_scope_dir.resolve())

    fem = load_hashed_json(args.fem_summary)
    if fem.get("method") != "fem_sabine_depth_aabb":
        raise RuntimeError("unexpected FEM method")
    if fem.get("completion_status") != "complete":
        raise RuntimeError("FEM 112-query source is incomplete")
    fem_by_index = {int(row["query_index"]): row for row in fem["per_query"]}
    if len(fem_by_index) != 112 or not set(fem_by_index).issubset(all_indices):
        raise RuntimeError("FEM source is not the expected 112-query subset")

    per_query = []
    failure_rows = []
    rooms = Counter()
    for index in sorted(indices):
        reference, reference_dir = arms["vanilla"][index]
        rooms[reference["room"]] += 1
        shared = ("query_id", "room", "scene", "candidate_count")
        for model, results in arms.items():
            payload, _directory = results[index]
            for key in shared:
                if payload[key] != reference[key]:
                    raise RuntimeError(f"{model}/{index} mismatch: {key}")
            if payload["candidate_indices_sha256"] != reference[
                "candidate_indices_sha256"
            ]:
                raise RuntimeError(f"{model}/{index} candidate hash mismatch")

        model_metrics = {
            model: compact_metric(
                results[index][0]["metrics"][PRIMARY_METRIC_KEY[model]]
            )
            for model, results in arms.items()
        }
        if index in fem_by_index:
            fem_row = fem_by_index[index]
            if fem_row["query_id"] != reference["query_id"]:
                raise RuntimeError(f"FEM/{index} query identity mismatch")
            if fem_row["candidate_indices_sha256"] != reference[
                "candidate_indices_sha256"
            ]:
                raise RuntimeError(f"FEM/{index} candidate hash mismatch")
            fem_metric = compact_metric(fem_row["metrics"])
            fem_status = "evaluated"
            fem_source = fem_row["source"]
        else:
            arrays_path = reference_dir / reference["arrays_file"]
            if file_sha256(arrays_path) != reference["arrays_sha256"]:
                raise RuntimeError(f"Vanilla/{index} candidate arrays hash mismatch")
            with np.load(arrays_path, allow_pickle=False) as archive:
                candidates = np.asarray(archive["candidates"], dtype=np.float64)
            truth = np.asarray(reference["source_global"], dtype=np.float64)
            distances = np.linalg.norm(candidates - truth, axis=1)
            oracle = float(distances.min())
            reported_oracle = float(reference["metrics"]["1"]["oracle_error_m"])
            if not np.isclose(oracle, reported_oracle, atol=1e-7, rtol=0.0):
                raise RuntimeError(f"Vanilla/{index} oracle mismatch")
            if args.fem_failure_policy == "worst_candidate":
                fallback_error = float(distances.max())
                fem_metric = {
                    "localization_error_m": fallback_error,
                    "oracle_error_m": oracle,
                    "excess_error_m": fallback_error - oracle,
                    "success_0_5m": 0,
                    "success_1_0m": 0,
                    "oracle_normalized_success_0_5m": 0,
                    "oracle_normalized_success_1_0m": 0,
                }
                fem_status = "strict_coverage_failure_penalized"
                fem_source = {
                    "kind": "evaluation_penalty",
                    "penalty_rule": "maximum true-source distance over frozen candidates",
                    "candidate_arrays_file": str(arrays_path.resolve()),
                    "candidate_arrays_sha256": reference["arrays_sha256"],
                }
                failure_detail = {"penalty_error_m": fallback_error}
            else:
                prediction_index = deterministic_random_candidate(
                    index, len(candidates), seed=args.random_seed
                )
                calculated = localization_metrics(candidates, truth, prediction_index)
                registered = reference["random_candidate_metrics"]
                if prediction_index != int(registered["prediction_index"]):
                    raise RuntimeError(
                        f"Vanilla/{index} registered random-candidate index mismatch"
                    )
                for key, value in compact_metric(calculated).items():
                    if not np.isclose(value, registered[key], atol=1e-7, rtol=0.0):
                        raise RuntimeError(
                            f"Vanilla/{index} registered random-candidate metric mismatch: {key}"
                        )
                fem_metric = compact_metric(calculated)
                fallback_error = float(fem_metric["localization_error_m"])
                fem_status = "strict_coverage_failure_random_fallback"
                fem_source = {
                    "kind": "inference_fallback",
                    "fallback_rule": "uniform deterministic random frozen candidate",
                    "random_seed": int(args.random_seed),
                    "prediction_index": prediction_index,
                    "prediction_global": candidates[prediction_index].astype(float).tolist(),
                    "candidate_arrays_file": str(arrays_path.resolve()),
                    "candidate_arrays_sha256": reference["arrays_sha256"],
                }
                failure_detail = {
                    "random_seed": int(args.random_seed),
                    "prediction_index": prediction_index,
                    "fallback_error_m": fallback_error,
                    "success_0_5m": fem_metric["success_0_5m"],
                    "success_1_0m": fem_metric["success_1_0m"],
                    "resolution_aware_success_0_5m": fem_metric[
                        "oracle_normalized_success_0_5m"
                    ],
                }
            failure_rows.append(
                {
                    "query_index": index,
                    "query_id": reference["query_id"],
                    "room": reference["room"],
                    "candidate_count": int(reference["candidate_count"]),
                    "oracle_error_m": oracle,
                    **failure_detail,
                }
            )
        model_metrics["fem_omp"] = fem_metric
        per_query.append(
            {
                "query_index": index,
                "query_id": reference["query_id"],
                "scene": reference["scene"],
                "room": reference["room"],
                "candidate_count": int(reference["candidate_count"]),
                "candidate_indices_sha256": reference[
                    "candidate_indices_sha256"
                ],
                "metrics": model_metrics,
                "fem_evaluation_status": fem_status,
                "fem_source": fem_source,
            }
        )

    queries_per_room_values = set(rooms.values())
    if len(rooms) != 16 or len(queries_per_room_values) != 1:
        raise RuntimeError(f"expected a balanced 16-room scope, got {dict(rooms)}")
    queries_per_room = queries_per_room_values.pop()
    expected_failures = len(indices - set(fem_by_index))
    if len(failure_rows) != expected_failures:
        raise RuntimeError(
            f"expected {expected_failures} FEM coverage failures, got {len(failure_rows)}"
        )

    metrics = {
        model: aggregate([row["metrics"][model] for row in per_query])
        for model in MODEL_ORDER
    }
    metrics_by_room = defaultdict(lambda: defaultdict(list))
    for row in per_query:
        for model in MODEL_ORDER:
            metrics_by_room[row["room"]][model].append(row["metrics"][model])
    per_room = {
        room: {
            model: aggregate(model_metrics[model]) for model in MODEL_ORDER
        }
        for room, model_metrics in sorted(metrics_by_room.items())
    }
    room_macro = {
        model: {
            "room_count": len(rooms),
            "mean_localization_error_m": float(
                np.mean(
                    [per_room[room][model]["mean_localization_error_m"] for room in per_room]
                )
            ),
            "success_rate_at_0_5m": float(
                np.mean([per_room[room][model]["success_rate_at_0_5m"] for room in per_room])
            ),
            "success_rate_at_1_0m": float(
                np.mean([per_room[room][model]["success_rate_at_1_0m"] for room in per_room])
            ),
            "resolution_aware_success_rate_at_0_5m": float(
                np.mean(
                    [
                        per_room[room][model][
                            "resolution_aware_success_rate_at_0_5m"
                        ]
                        for room in per_room
                    ]
                )
            ),
        }
        for model in MODEL_ORDER
    }

    payload = {
        "schema_version": 1,
        "scope": f"16 rooms x {queries_per_room} frozen queries",
        "query_scope_source": query_scope_source,
        "query_count": len(indices),
        "room_count": 16,
        "queries_per_room": queries_per_room,
        "primary_settings": {
            "vanilla_flac_k_gen": 1,
            "orbitrir_fa_bf_flac_k_gen": 1,
            "yaw_augmented_flac_k_gen": 1,
            "few_shot_rir_k_ctx": 8,
            "fem_omp_k_ctx": 8,
        },
        "flac_k_gen": 1,
        "few_shot_k_ctx": 8,
        "fem_k_ctx": 8,
        "fem_coverage": {
            "evaluated_query_count": len(indices & set(fem_by_index)),
            "failed_query_count": len(failure_rows),
            "coverage_rate": len(indices & set(fem_by_index)) / len(indices),
            "failure_query_indices": [row["query_index"] for row in failure_rows],
        },
        "fem_failure_policy": (
            {
                "name": "worst_candidate",
                "success_metrics": "forced to zero",
                "localization_error": "maximum true-source distance over the query's frozen candidate set",
                "ground_truth_usage": "evaluation only; no inference-time information",
                "conditional_112_result_retained_separately": str(args.fem_summary.resolve()),
            }
            if args.fem_failure_policy == "worst_candidate"
            else {
                "name": "random_candidate",
                "selection": "uniform deterministic random choice over the frozen candidate set",
                "random_seed": int(args.random_seed),
                "random_key": "numpy.SeedSequence([seed, query_index, 0x52414E44])",
                "success_metrics": "calculated normally from the selected candidate",
                "ground_truth_usage": "metric calculation only; not used for fallback selection",
                "conditional_112_result_retained_separately": str(args.fem_summary.resolve()),
            }
        ),
        "metrics": metrics,
        "room_macro_metrics": room_macro,
        "per_room": per_room,
        "fem_failure_rows": failure_rows,
        "per_query": per_query,
    }
    payload["sha256"] = canonical_sha256(payload)
    output_json = args.output_dir / "summary.json"
    output_md = args.output_dir / "summary.md"
    atomic_json(output_json, payload)

    if args.fem_failure_policy == "worst_candidate":
        failure_title = "FEM coverage penalties"
        failure_description = (
            "their success indicators are zero and their localization error is "
            "the maximum true-source distance over the frozen candidate set."
        )
        interpretation = (
            "The failure penalty is deliberately pessimistic but finite and "
            "query-scale-aware. Ground truth is used only to calculate an evaluation "
            "error after the method has failed to produce a valid full-candidate "
            "prediction. Success flags are forced to zero regardless of the numerical "
            "penalty value."
        )
    else:
        failure_title = "FEM deterministic random fallback"
        failure_description = (
            "each uses the registered deterministic uniform random candidate "
            f"fallback (`seed={args.random_seed}`), and all success indicators are "
            "calculated normally from that selected candidate."
        )
        interpretation = (
            "The fallback selection uses no ground truth: it is a uniform random draw "
            "from the query's frozen candidate set, keyed independently by the fixed "
            f"seed `{args.random_seed}` and query index. Ground truth is used only after "
            "selection to calculate the standard localization metrics."
        )

    lines = [
        f"# Five-method {len(indices)}-query result with {failure_title}",
        "",
        f"This end-to-end table evaluates the same 16 rooms x {queries_per_room} "
        f"frozen queries for all five methods. The {len(failure_rows)} queries that "
        "fail the FEM strict-coverage gate are retained: " + failure_description,
        "",
        f"FEM coverage is **{len(indices & set(fem_by_index))}/{len(indices)} "
        f"({100 * len(indices & set(fem_by_index)) / len(indices):.1f}%)**. "
        "The previous 112-query conditional "
        "result remains a diagnostic and is not used as this full-scope table.",
        "",
        "## Matched localization metrics",
        "",
        f"`Localization Error [m]` is the median over all {len(indices)} queries.",
        "",
        "| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Native FEM coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        metric = metrics[model]
        coverage = (
            f"{100 * len(indices & set(fem_by_index)) / len(indices):.1f}%"
            if model == "fem_omp"
            else "100.0%"
        )
        lines.append(
            f"| {MODEL_LABELS[model]} | {metric['median_localization_error_m']:.3f} | "
            f"{100 * metric['success_rate_at_0_5m']:.1f}% | "
            f"{100 * metric['success_rate_at_1_0m']:.1f}% | "
            f"{100 * metric['resolution_aware_success_rate_at_0_5m']:.1f}% | "
            f"{coverage} |"
        )
    lines.extend(
        [
            "",
            "## Localization-error distribution",
            "",
            "| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        metric = metrics[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {metric['mean_localization_error_m']:.3f} | "
            f"{metric['median_localization_error_m']:.3f} | "
            f"{metric['p90_localization_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            interpretation,
            "",
            f"The four learned methods use their real predictions on all {len(indices)} "
            "queries. "
            "The three FLAC rows use the registered primary `K_gen=1`; Few-ShotRIR "
            "uses `K_ctx=8`; FEM uses "
            "eight acoustic context RIRs and Room-Helps one-support OMP.",
            "",
            "This is a FEM--OMP result, not a FEM--AGREE result.",
            "",
        ]
    )
    atomic_text(output_md, "\n".join(lines))
    print(json.dumps({"metrics": metrics, "fem_coverage": payload["fem_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
