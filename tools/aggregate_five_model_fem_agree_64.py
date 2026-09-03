#!/usr/bin/env python3
"""Aggregate five aligned models on the seed-43 16-room/64-query pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MODEL_ORDER = ("vanilla", "fa_bf", "yawaug", "few_shot", "fem_agree")
MODEL_LABELS = {
    "vanilla": "Vanilla FLAC",
    "fa_bf": "OrbitRIR (FA-BF FLAC)",
    "yawaug": "Yaw-Augmented FLAC",
    "few_shot": "Few-ShotRIR",
    "fem_agree": "FEM--AGREE K=8 + random fallback",
}
PRIMARY_METRIC_KEY = {
    "vanilla": "1",
    "fa_bf": "1",
    "yawaug": "1",
    "few_shot": "8",
}


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt hashed JSON: {path}")
    return payload


def load_query_results(directory: Path) -> dict[int, tuple[dict, Path]]:
    results = {}
    for path in sorted((directory / "queries").glob("query_*.json")):
        payload = load_hashed_json(path)
        index = int(payload["query_index"])
        if index in results:
            raise RuntimeError(f"duplicate query {index} in {directory}")
        results[index] = (payload, path)
    return results


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
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--fa-bf-dir", type=Path, required=True)
    parser.add_argument("--yawaug-dir", type=Path, required=True)
    parser.add_argument("--few-shot-dir", type=Path, required=True)
    parser.add_argument("--fem-agree-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_hashed_json(args.pilot_manifest)
    if (
        manifest.get("query_count") != 64
        or manifest.get("room_count") != 16
        or manifest.get("selection", {}).get("seed") != 43
    ):
        raise RuntimeError("manifest is not the frozen seed-43 16-room/64-query pilot")
    manifest_by_index = {int(row["index"]): row for row in manifest["records"]}
    target_indices = set(manifest_by_index)

    learned_dirs = {
        "vanilla": args.vanilla_dir,
        "fa_bf": args.fa_bf_dir,
        "yawaug": args.yawaug_dir,
        "few_shot": args.few_shot_dir,
    }
    learned = {
        model: load_query_results(directory)
        for model, directory in learned_dirs.items()
    }
    for model, results in learned.items():
        if set(results) != target_indices:
            missing = sorted(target_indices - set(results))
            extra = sorted(set(results) - target_indices)
            raise RuntimeError(f"{model} does not match pilot: missing={missing}, extra={extra}")

    fem_agree = load_hashed_json(args.fem_agree_summary)
    if (
        fem_agree.get("method")
        != "fem_sabine_depth_aabb_agree_k8_with_random_fallback"
        or fem_agree.get("query_count") != 64
        or fem_agree.get("primary_agree_k") != 8
    ):
        raise RuntimeError("unexpected FEM--AGREE summary")
    agree_by_index = {int(row["query_index"]): row for row in fem_agree["per_query"]}
    if set(agree_by_index) != target_indices:
        raise RuntimeError("FEM--AGREE summary does not match the seed-43 pilot")

    shared_keys = (
        ("query_id", "query_id"),
        ("room", "room"),
        ("scene", "scene"),
        ("candidate_count", "candidate_count"),
        ("candidate_indices_sha256", "candidate_indices_sha256"),
    )
    per_query = []
    rooms = Counter()
    for index in sorted(target_indices):
        manifest_row = manifest_by_index[index]
        reference = learned["vanilla"][index][0]
        if reference["query_index"] != index:
            raise RuntimeError(f"Vanilla query index mismatch for {index}")
        for key, manifest_key in shared_keys:
            if reference[key] != manifest_row[manifest_key]:
                raise RuntimeError(f"Vanilla/manifest mismatch for {index}: {key}")
        for model, results in learned.items():
            row = results[index][0]
            for key, _manifest_key in shared_keys:
                if row[key] != reference[key]:
                    raise RuntimeError(f"{model}/Vanilla mismatch for {index}: {key}")
        agree_row = agree_by_index[index]
        for key, _manifest_key in shared_keys:
            if agree_row[key] != reference[key]:
                raise RuntimeError(f"FEM--AGREE/Vanilla mismatch for {index}: {key}")

        metrics = {
            model: compact_metric(results[index][0]["metrics"][PRIMARY_METRIC_KEY[model]])
            for model, results in learned.items()
        }
        metrics["fem_agree"] = compact_metric(agree_row["metrics"])
        sources = {
            model: {
                "result_file": str(results[index][1].resolve()),
                "result_sha256": results[index][0]["sha256"],
            }
            for model, results in learned.items()
        }
        sources["fem_agree"] = {
            "evaluation_status": agree_row["evaluation_status"],
            "source": agree_row["source"],
        }
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
                "metrics": metrics,
                "sources": sources,
            }
        )

    if len(rooms) != 16 or set(rooms.values()) != {4}:
        raise RuntimeError(f"unbalanced room coverage: {dict(rooms)}")

    metrics = {
        model: aggregate([row["metrics"][model] for row in per_query])
        for model in MODEL_ORDER
    }
    by_room = defaultdict(lambda: defaultdict(list))
    for row in per_query:
        for model in MODEL_ORDER:
            by_room[row["room"]][model].append(row["metrics"][model])
    per_room = {
        room: {
            model: aggregate(model_metrics[model]) for model in MODEL_ORDER
        }
        for room, model_metrics in sorted(by_room.items())
    }
    room_macro_metrics = {
        model: {
            "room_count": len(per_room),
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
                        per_room[room][model]["resolution_aware_success_rate_at_0_5m"]
                        for room in per_room
                    ]
                )
            ),
        }
        for model in MODEL_ORDER
    }

    learned_result_set_hashes = {
        model: canonical_sha256(
            {
                str(index): results[index][0]["sha256"]
                for index in sorted(results)
            }
        )
        for model, results in learned.items()
    }
    payload = {
        "schema_version": 1,
        "scope": "frozen seed-43 pilot: 16 rooms x 4 queries",
        "query_count": len(per_query),
        "room_count": len(rooms),
        "queries_per_room": 4,
        "model_order": list(MODEL_ORDER),
        "model_labels": MODEL_LABELS,
        "primary_settings": {
            "vanilla_flac_k_gen": 1,
            "orbitrir_fa_bf_flac_k_gen": 1,
            "yaw_augmented_flac_k_gen": 1,
            "few_shot_rir_k_ctx": 8,
            "fem_agree_k": 8,
            "fem_agree_random_fallback_seed": 42,
        },
        "metrics": metrics,
        "room_macro_metrics": room_macro_metrics,
        "per_room": per_room,
        "per_query": per_query,
        "model_coverage": {
            "vanilla": {"native_query_count": 64, "native_coverage_rate": 1.0},
            "fa_bf": {"native_query_count": 64, "native_coverage_rate": 1.0},
            "yawaug": {"native_query_count": 64, "native_coverage_rate": 1.0},
            "few_shot": {"native_query_count": 64, "native_coverage_rate": 1.0},
            "fem_agree": {
                "measured_query_count": fem_agree["coverage"]["measured_query_count"],
                "measured_coverage_rate": fem_agree["coverage"]["measured_coverage_rate"],
                "random_choice_estimated_query_count": fem_agree["coverage"][
                    "random_choice_estimated_query_count"
                ],
            },
        },
        "provenance": {
            "pilot_manifest": str(args.pilot_manifest.resolve()),
            "pilot_manifest_sha256": manifest["sha256"],
            "learned_result_directories": {
                model: str(directory.resolve())
                for model, directory in learned_dirs.items()
            },
            "learned_result_set_sha256": learned_result_set_hashes,
            "fem_agree_summary": str(args.fem_agree_summary.resolve()),
            "fem_agree_summary_sha256": fem_agree["sha256"],
        },
        "interpretation_boundary": (
            "The four learned models have native predictions for all 64 queries. "
            "FEM--AGREE contains 53 measured K=8 predictions and 11 deterministic "
            "random-choice estimates and must retain the random-fallback label."
        ),
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_json(args.output_dir / "summary.json", payload)

    lines = [
        "# Five-model seed-43 16-room/64-query localization metrics",
        "",
        "All five rows use the same frozen seed-43 pilot: 16 rooms x 4 queries. "
        "Vanilla, OrbitRIR, Yaw-Augmented FLAC, and Few-ShotRIR have native "
        "predictions for all 64 queries. FEM--AGREE K=8 combines 53 measured "
        "predictions with 11 deterministic random-choice estimates (`seed=42`).",
        "",
        "## Aggregate localization metrics",
        "",
        "`Localization Error [m]` is the query-micro mean. Success rates are also "
        "query-micro averages over the same 64 queries.",
        "",
        "| Model | Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ | Native/measured coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        metric = metrics[model]
        coverage = (
            f"{100 * fem_agree['coverage']['measured_coverage_rate']:.1f}%"
            if model == "fem_agree"
            else "100.0%"
        )
        lines.append(
            f"| {MODEL_LABELS[model]} | {metric['mean_localization_error_m']:.3f} | "
            f"{100 * metric['success_rate_at_0_5m']:.1f}% | "
            f"{100 * metric['success_rate_at_1_0m']:.1f}% | "
            f"{100 * metric['resolution_aware_success_rate_at_0_5m']:.1f}% | "
            f"{coverage} |"
        )
    lines.extend(
        [
            "",
            "## Error distribution",
            "",
            "P90 uses NumPy's default linear quantile over the 64 per-query errors.",
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
            "## Per-room mean localization error [m]",
            "",
            "Each cell averages the four frozen queries in that room.",
            "",
            "| Room | Vanilla | OrbitRIR | Yaw-Aug FLAC | Few-ShotRIR | FEM--AGREE + random |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for room, room_metrics in per_room.items():
        lines.append(
            f"| {room} | "
            + " | ".join(
                f"{room_metrics[model]['mean_localization_error_m']:.3f}"
                for model in MODEL_ORDER
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a matched five-model comparison on the complete seed-43 query "
            "set. The FEM--AGREE row is failure-imputed rather than fully native: "
            "53/64 predictions are measured and 11/64 are random-choice estimates. "
            "It must be reported as `FEM--AGREE K=8 + random fallback`.",
            "",
        ]
    )
    atomic_text(args.output_dir / "summary.md", "\n".join(lines))
    print(json.dumps({"metrics": metrics, "sha256": payload["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
