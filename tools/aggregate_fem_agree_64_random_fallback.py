#!/usr/bin/env python3
"""Build the balanced seed-43 FEM--AGREE K=8 result with random fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIMARY_K = "8"
EXPECTED_CHECKPOINT_SHA256 = (
    "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
)


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def localization_metrics(
    candidates: np.ndarray, truth: np.ndarray, prediction_index: int
) -> dict:
    distances = np.linalg.norm(candidates - truth[None, :], axis=1)
    localization_error = float(distances[prediction_index])
    oracle_error = float(distances.min())
    excess_error = max(0.0, localization_error - oracle_error)
    return {
        "prediction_index": int(prediction_index),
        "localization_error_m": localization_error,
        "oracle_error_m": oracle_error,
        "excess_error_m": excess_error,
        "success_0_5m": int(localization_error <= 0.5),
        "success_1_0m": int(localization_error <= 1.0),
        "oracle_normalized_success_0_5m": int(excess_error <= 0.5),
        "oracle_normalized_success_1_0m": int(excess_error <= 1.0),
    }


def deterministic_random_candidate(
    query_index: int, candidate_count: int, seed: int = 42
) -> int:
    sequence = np.random.SeedSequence([int(seed), int(query_index), 0x52414E44])
    return int(np.random.default_rng(sequence).integers(candidate_count))


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


def compact_metric(metric: dict) -> dict:
    return {
        key: metric[key]
        for key in (
            "prediction_index",
            "prediction_global",
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


def validate_external_row(row: dict, reference: dict, candidates: np.ndarray) -> dict:
    for key, reference_key in (
        ("query_index", "query_index"),
        ("room", "room"),
        ("receiver_id", "receiver_id"),
        ("candidate_count", "candidate_count"),
    ):
        if row[key] != reference[reference_key]:
            raise RuntimeError(
                f"external query {row['query_index']} disagrees on {key}: "
                f"{row[key]!r} != {reference[reference_key]!r}"
            )
    prediction_index = int(row["prediction_index"])
    calculated = localization_metrics(
        candidates, np.asarray(reference["source_global"], dtype=np.float64), prediction_index
    )
    prediction = candidates[prediction_index].astype(float).tolist()
    if prediction != row["prediction_global"]:
        raise RuntimeError(
            f"external query {row['query_index']} prediction coordinate mismatch"
        )
    if not np.isclose(
        calculated["localization_error_m"],
        float(row["reported_agree_error_m"]),
        atol=5e-4,
        rtol=0.0,
    ):
        raise RuntimeError(f"external query {row['query_index']} rounded error mismatch")
    for key in ("success_0_5m", "success_1_0m"):
        if int(calculated[key]) != int(row[key]):
            raise RuntimeError(f"external query {row['query_index']} {key} mismatch")
    if int(calculated["oracle_normalized_success_0_5m"]) != int(
        row["resolution_aware_success_0_5m"]
    ):
        raise RuntimeError(
            f"external query {row['query_index']} resolution-aware success mismatch"
        )
    return {
        **calculated,
        "prediction_global": prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--existing-agree-dir", type=Path, required=True)
    parser.add_argument("--existing-agree-summary", type=Path, required=True)
    parser.add_argument("--external-report", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_hashed_json(args.pilot_manifest)
    if manifest.get("query_count") != 64 or manifest.get("room_count") != 16:
        raise RuntimeError("pilot manifest is not the expected 16-room/64-query scope")
    if manifest.get("selection", {}).get("seed") != 43:
        raise RuntimeError("pilot manifest is not the frozen seed-43 pilot")
    target = {int(row["index"]): row for row in manifest["records"]}
    if len(target) != 64:
        raise RuntimeError("pilot manifest contains duplicate query indices")

    existing_summary = load_hashed_json(args.existing_agree_summary)
    if (
        existing_summary.get("method") != "fem_sabine_depth_aabb_agree"
        or existing_summary.get("query_count") != 97
    ):
        raise RuntimeError("existing AGREE summary is not the frozen 97-query source")

    existing = {}
    for path in sorted(args.existing_agree_dir.glob("query_*.json")):
        payload = load_hashed_json(path)
        index = int(payload["query_index"])
        if index in target:
            if payload.get("agree_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise RuntimeError(f"query {index} uses an unexpected AGREE checkpoint")
            existing[index] = (payload, path)

    external_report = load_hashed_json(args.external_report)
    if external_report.get("agree_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("external report uses an unexpected AGREE checkpoint")
    external_all = {int(row["query_index"]): row for row in external_report["rows"]}
    external = {index: row for index, row in external_all.items() if index in target}
    excluded_external = sorted(set(external_all) - set(target))
    if set(existing) & set(external):
        raise RuntimeError("external AGREE rows overlap existing local AGREE rows")

    per_query = []
    status_counts = Counter()
    rooms = Counter()
    fallback_rows = []
    for index in sorted(target):
        manifest_row = target[index]
        reference_path = args.reference_dir / "queries" / f"query_{index:05d}.json"
        reference = load_hashed_json(reference_path)
        for key, manifest_key in (
            ("query_index", "index"),
            ("query_id", "query_id"),
            ("room", "room"),
            ("scene", "scene"),
            ("candidate_count", "candidate_count"),
            ("candidate_indices_sha256", "candidate_indices_sha256"),
        ):
            if reference[key] != manifest_row[manifest_key]:
                raise RuntimeError(f"reference query {index} disagrees on {key}")

        arrays_path = args.reference_dir / reference["arrays_file"]
        if file_sha256(arrays_path) != reference["arrays_sha256"]:
            raise RuntimeError(f"reference candidate archive hash mismatch for {index}")
        with np.load(arrays_path, allow_pickle=False) as archive:
            candidates = np.asarray(archive["candidates"], dtype=np.float64)
        if len(candidates) != int(reference["candidate_count"]):
            raise RuntimeError(f"reference candidate count mismatch for {index}")

        if index in existing:
            result, result_path = existing[index]
            for key in (
                "query_id",
                "room",
                "scene",
                "candidate_count",
                "candidate_indices_sha256",
            ):
                if result[key] != reference[key]:
                    raise RuntimeError(f"existing AGREE query {index} disagrees on {key}")
            metric = compact_metric(result["metrics_by_k_agree"][PRIMARY_K])
            status = "existing_agree_k8"
            source = {
                "kind": "existing_hashed_agree_result",
                "result_file": str(result_path.resolve()),
                "result_sha256": result["sha256"],
            }
        elif index in external:
            calculated = validate_external_row(external[index], reference, candidates)
            metric = compact_metric(calculated)
            status = "external_reported_agree_k8"
            source = {
                "kind": "external_hashed_report",
                "external_report_file": str(args.external_report.resolve()),
                "external_report_file_sha256": external_report["sha256"],
                "reported_result_summary_canonical_sha256": external_report[
                    "reported_result_summary_canonical_sha256"
                ],
                "reported_agree_error_m_rounded": external[index][
                    "reported_agree_error_m"
                ],
            }
        else:
            prediction_index = deterministic_random_candidate(
                index, len(candidates), seed=args.random_seed
            )
            calculated = localization_metrics(
                candidates,
                np.asarray(reference["source_global"], dtype=np.float64),
                prediction_index,
            )
            calculated["prediction_global"] = (
                candidates[prediction_index].astype(float).tolist()
            )
            metric = compact_metric(calculated)
            registered = reference["random_candidate_metrics"]
            for key, value in metric.items():
                registered_value = registered[key]
                if key == "prediction_global":
                    matches = np.array_equal(value, registered_value)
                else:
                    matches = np.isclose(value, registered_value, atol=1e-7, rtol=0.0)
                if not matches:
                    raise RuntimeError(
                        f"reference query {index} registered random metric mismatch: {key}"
                    )
            status = "random_choice_estimate"
            source = {
                "kind": "inference_fallback",
                "fallback_rule": "uniform deterministic random frozen candidate",
                "random_seed": int(args.random_seed),
                "random_key": "numpy.SeedSequence([seed, query_index, 0x52414E44])",
                "registered_random_candidate_metrics_verified": True,
                "candidate_arrays_file": str(arrays_path.resolve()),
                "candidate_arrays_sha256": reference["arrays_sha256"],
            }
            fallback_rows.append(
                {
                    "query_index": index,
                    "query_id": reference["query_id"],
                    "room": reference["room"],
                    "receiver_id": reference["receiver_id"],
                    "candidate_count": int(reference["candidate_count"]),
                    "metrics": metric,
                }
            )

        status_counts[status] += 1
        rooms[reference["room"]] += 1
        per_query.append(
            {
                "query_index": index,
                "query_id": reference["query_id"],
                "scene": reference["scene"],
                "room": reference["room"],
                "receiver_id": reference["receiver_id"],
                "candidate_count": int(reference["candidate_count"]),
                "candidate_indices_sha256": reference["candidate_indices_sha256"],
                "evaluation_status": status,
                "metrics": metric,
                "source": source,
            }
        )

    if len(rooms) != 16 or set(rooms.values()) != {4}:
        raise RuntimeError(f"expected balanced 16-room x 4-query scope, got {dict(rooms)}")
    expected_status_counts = {
        "existing_agree_k8": 47,
        "external_reported_agree_k8": 6,
        "random_choice_estimate": 11,
    }
    if dict(status_counts) != expected_status_counts:
        raise RuntimeError(
            f"unexpected source coverage: {dict(status_counts)} != {expected_status_counts}"
        )
    if excluded_external != [715]:
        raise RuntimeError(f"unexpected out-of-scope external rows: {excluded_external}")

    metrics = aggregate([row["metrics"] for row in per_query])
    by_room = defaultdict(list)
    for row in per_query:
        by_room[row["room"]].append(row)
    per_room = {}
    for room, rows in sorted(by_room.items()):
        per_room[room] = {
            "query_count": len(rows),
            "measured_query_count": sum(
                row["evaluation_status"] != "random_choice_estimate" for row in rows
            ),
            "estimated_query_count": sum(
                row["evaluation_status"] == "random_choice_estimate" for row in rows
            ),
            "metrics": aggregate([row["metrics"] for row in rows]),
        }

    measured_count = (
        status_counts["existing_agree_k8"]
        + status_counts["external_reported_agree_k8"]
    )
    fallback_rooms = sorted({row["room"] for row in fallback_rows})
    payload = {
        "schema_version": 1,
        "method": "fem_sabine_depth_aabb_agree_k8_with_random_fallback",
        "scope": "frozen seed-43 pilot: 16 rooms x 4 queries",
        "query_count": len(per_query),
        "room_count": len(rooms),
        "queries_per_room": 4,
        "primary_agree_k": 8,
        "agree_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "metrics": metrics,
        "coverage": {
            "measured_query_count": measured_count,
            "measured_coverage_rate": measured_count / len(per_query),
            "existing_local_agree_query_count": status_counts["existing_agree_k8"],
            "external_reported_agree_query_count": status_counts[
                "external_reported_agree_k8"
            ],
            "random_choice_estimated_query_count": status_counts[
                "random_choice_estimate"
            ],
            "rooms_with_random_choice_estimates": fallback_rooms,
            "room_count_with_random_choice_estimates": len(fallback_rooms),
        },
        "random_choice_policy": {
            "selection": "uniform deterministic random choice over the frozen candidate set",
            "random_seed": int(args.random_seed),
            "random_key": "numpy.SeedSequence([seed, query_index, 0x52414E44])",
            "ground_truth_usage": "metric calculation only; not used for fallback selection",
            "success_metrics": "calculated normally from the selected candidate",
        },
        "provenance": {
            "pilot_manifest": str(args.pilot_manifest.resolve()),
            "pilot_manifest_sha256": manifest["sha256"],
            "existing_agree_summary": str(args.existing_agree_summary.resolve()),
            "existing_agree_summary_sha256": existing_summary["sha256"],
            "external_report": str(args.external_report.resolve()),
            "external_report_sha256": external_report["sha256"],
            "reported_external_summary_canonical_sha256": external_report[
                "reported_result_summary_canonical_sha256"
            ],
            "reference_candidate_directory": str(args.reference_dir.resolve()),
            "external_query_indices_excluded_from_seed43_scope": excluded_external,
            "exclusion_reason": "query 715 belongs to the seed-42 pilot",
        },
        "fallback_rows": fallback_rows,
        "per_room": per_room,
        "per_query": per_query,
        "interpretation_boundary": (
            "This is a mixed measured/estimated 64-query result, not a completed native "
            "FEM--AGREE run. Random-choice rows must remain labeled as estimates."
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_json(args.output_dir / "summary.json", payload)

    lines = [
        "# FEM--AGREE K=8 balanced seed-43 16-room/64-query estimate",
        "",
        "This report combines 47 existing hashed FEM--AGREE K=8 results with 6 "
        "externally reported K=8 results from Cafe/Auditorium. The remaining 11 "
        f"queries across {len(fallback_rooms)} rooms use deterministic uniform "
        f"random choice (`seed={args.random_seed}`) over each frozen candidate set.",
        "",
        "**This is a mixed measured/estimated result, not a completed native "
        "16-room/64-query FEM--AGREE run.** Query `715` is excluded because it belongs "
        "to the seed-42 pilot; only the six seed-43 rows from the supplied 7-query "
        "report enter this aggregate.",
        "",
        "## Aggregate localization metrics",
        "",
        "`Localization Error [m]` is the query-micro mean over all 64 rows. P90 uses "
        "NumPy's default linear quantile.",
        "",
        "| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Measured coverage |",
        "|---|---:|---:|---:|---:|---:|",
        f"| FEM--AGREE K=8 + random fallback | {metrics['mean_localization_error_m']:.3f} | "
        f"{100 * metrics['success_rate_at_0_5m']:.1f}% | "
        f"{100 * metrics['success_rate_at_1_0m']:.1f}% | "
        f"{100 * metrics['resolution_aware_success_rate_at_0_5m']:.1f}% | "
        f"{100 * measured_count / len(per_query):.1f}% |",
        "",
        "## Error distribution",
        "",
        "| Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ |",
        "|---:|---:|---:|",
        f"| {metrics['mean_localization_error_m']:.3f} | "
        f"{metrics['median_localization_error_m']:.3f} | "
        f"{metrics['p90_localization_error_m']:.3f} |",
        "",
        "## Source coverage",
        "",
        "| Source | Queries | Share |",
        "|---|---:|---:|",
        f"| Existing local AGREE K=8 | {status_counts['existing_agree_k8']} | "
        f"{100 * status_counts['existing_agree_k8'] / len(per_query):.1f}% |",
        f"| Supplied external AGREE K=8 | {status_counts['external_reported_agree_k8']} | "
        f"{100 * status_counts['external_reported_agree_k8'] / len(per_query):.1f}% |",
        f"| Random-choice estimate | {status_counts['random_choice_estimate']} | "
        f"{100 * status_counts['random_choice_estimate'] / len(per_query):.1f}% |",
        f"| **Total** | **{len(per_query)}** | **100.0%** |",
        "",
        "## Per-room metrics",
        "",
        "| Room | Measured | Estimated | Mean [m] | Median [m] | P90 [m] | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for room, room_row in per_room.items():
        metric = room_row["metrics"]
        lines.append(
            f"| {room} | {room_row['measured_query_count']} | "
            f"{room_row['estimated_query_count']} | "
            f"{metric['mean_localization_error_m']:.3f} | "
            f"{metric['median_localization_error_m']:.3f} | "
            f"{metric['p90_localization_error_m']:.3f} | "
            f"{100 * metric['success_rate_at_0_5m']:.1f}% | "
            f"{100 * metric['success_rate_at_1_0m']:.1f}% | "
            f"{100 * metric['resolution_aware_success_rate_at_0_5m']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Random-choice estimates",
            "",
            "Selection is keyed independently by `seed=42` and query index. Ground "
            "truth is used only after selection to calculate metrics.",
            "",
            "| Query | Room | Receiver | Candidates | Prediction index | Prediction global [m] | Error [m] | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |",
            "|---:|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in fallback_rows:
        metric = row["metrics"]
        lines.append(
            f"| {row['query_index']} | {row['room']} | {row['receiver_id']} | "
            f"{row['candidate_count']:,} | {metric['prediction_index']:,} | "
            f"`{metric['prediction_global']}` | "
            f"{metric['localization_error_m']:.3f} | "
            f"{metric['success_0_5m']} | {metric['success_1_0m']} | "
            f"{metric['oracle_normalized_success_0_5m']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The aggregate is suitable as a failure-imputed estimate for the frozen "
            "seed-43 pilot. It must retain the `+ random fallback` label and the "
            "53/64 measured-coverage disclosure; it is not evidence that AGREE "
            "natively completed the 11 fallback queries.",
            "",
        ]
    )
    atomic_text(args.output_dir / "summary.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "metrics": metrics,
                "coverage": payload["coverage"],
                "sha256": payload["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
