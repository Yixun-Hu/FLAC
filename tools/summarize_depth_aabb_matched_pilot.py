#!/usr/bin/env python3
"""Validate and summarize a five-method Depth-AABB matched pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256


MODEL_LABELS = {
    "vanilla": "Vanilla FLAC",
    "fa_bf": "FA-BF FLAC",
    "yawaug": "YAWAUG FLAC",
    "few_shot": "Few-ShotRIR",
    "depth_aabb": "FEM-Sabine (Depth-AABB)",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt hashed JSON: {path}")
    return payload


def load_arm(directories: list[Path]) -> dict[int, tuple[dict, Path]]:
    results = {}
    for directory in directories:
        for path in sorted((directory / "queries").glob("query_*.json")):
            payload = load_hashed(path)
            index = int(payload["query_index"])
            if index in results:
                raise RuntimeError(f"duplicate query {index} across {directories}")
            results[index] = (payload, directory)
    return results


def aggregate(metrics: list[dict]) -> dict:
    error = np.asarray([row["localization_error_m"] for row in metrics], dtype=np.float64)
    return {
        "query_count": len(metrics),
        "mean_localization_error_m": float(error.mean()),
        "median_localization_error_m": float(np.median(error)),
        "success_rate_at_0_5m": float(np.mean([row["success_0_5m"] for row in metrics])),
        "success_rate_at_1_0m": float(np.mean([row["success_1_0m"] for row in metrics])),
        "resolution_aware_success_rate_at_0_5m": float(
            np.mean([row["oracle_normalized_success_0_5m"] for row in metrics])
        ),
    }


def aggregate_error_values(values) -> dict:
    error = np.asarray(values, dtype=np.float64)
    return {
        "query_count": len(error),
        "mean_localization_error_m": float(error.mean()),
        "median_localization_error_m": float(np.median(error)),
        "success_rate_at_1_0m": float(np.mean(error <= 1.0)),
    }


def aggregate_room_macro(metrics_by_room: dict[str, list[dict]]) -> dict:
    room_rows = []
    for room, metrics in sorted(metrics_by_room.items()):
        error = np.asarray(
            [row["localization_error_m"] for row in metrics], dtype=np.float64
        )
        room_rows.append(
            {
                "room": room,
                "query_count": len(metrics),
                "mean_localization_error_m": float(error.mean()),
                "median_localization_error_m": float(np.median(error)),
                "success_rate_at_0_5m": float(
                    np.mean([row["success_0_5m"] for row in metrics])
                ),
                "success_rate_at_1_0m": float(
                    np.mean([row["success_1_0m"] for row in metrics])
                ),
                "resolution_aware_success_rate_at_0_5m": float(
                    np.mean(
                        [row["oracle_normalized_success_0_5m"] for row in metrics]
                    )
                ),
            }
        )
    return {
        "room_count": len(room_rows),
        "room_macro_mean_localization_error_m": float(
            np.mean([row["mean_localization_error_m"] for row in room_rows])
        ),
        "mean_room_median_localization_error_m": float(
            np.mean([row["median_localization_error_m"] for row in room_rows])
        ),
        "room_macro_success_rate_at_0_5m": float(
            np.mean([row["success_rate_at_0_5m"] for row in room_rows])
        ),
        "room_macro_success_rate_at_1_0m": float(
            np.mean([row["success_rate_at_1_0m"] for row in room_rows])
        ),
        "room_macro_resolution_aware_success_rate_at_0_5m": float(
            np.mean(
                [
                    row["resolution_aware_success_rate_at_0_5m"]
                    for row in room_rows
                ]
            )
        ),
        "per_room": room_rows,
    }


def format_table(k_gen: int, rows: dict[str, dict]) -> list[str]:
    lines = [
        f"### K_gen = {k_gen}",
        "",
        "| Model | Median Localization Error [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR @ 0.5m ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("vanilla", "fa_bf", "yawaug", "few_shot", "depth_aabb"):
        row = rows[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {row['median_localization_error_m']:.3f} | "
            f"{100 * row['success_rate_at_0_5m']:.1f}% | "
            f"{100 * row['success_rate_at_1_0m']:.1f}% | "
            f"{100 * row['resolution_aware_success_rate_at_0_5m']:.1f}% |"
        )
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--vanilla-dir", type=Path, action="append", required=True)
    parser.add_argument("--fa-bf-dir", type=Path, action="append", required=True)
    parser.add_argument("--yawaug-dir", type=Path, action="append", required=True)
    parser.add_argument("--few-shot-dir", type=Path, action="append", required=True)
    parser.add_argument("--fem-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    records = selection["records"]
    indices = [int(record["index"]) for record in records]
    expected = set(indices)
    arms = {
        "vanilla": load_arm(args.vanilla_dir),
        "fa_bf": load_arm(args.fa_bf_dir),
        "yawaug": load_arm(args.yawaug_dir),
        "few_shot": load_arm(args.few_shot_dir),
    }
    fem = {}
    for path in sorted(args.fem_dir.glob("query_*_depth_aabb_result.json")):
        payload = load_hashed(path)
        fem[int(payload["query_index"])] = (payload, args.fem_dir)
    arms["depth_aabb"] = fem
    for model, results in arms.items():
        missing = expected - set(results)
        if missing:
            raise RuntimeError(f"{model} is missing selected queries: {sorted(missing)}")

    per_query = []
    maximum_solver_residual = 0.0
    for selected in records:
        index = int(selected["index"])
        loaded = {model: arms[model][index] for model in arms}
        reference = loaded["vanilla"][0]
        shared = ("query_id", "query_index", "room", "scene", "candidate_count")
        for model, (payload, _directory) in loaded.items():
            for key in shared:
                if payload[key] != reference[key]:
                    raise RuntimeError(f"{model}/{index} mismatch: {key}")
        hashes = {
            loaded[model][0]["candidate_indices_sha256"]
            for model in ("vanilla", "fa_bf", "yawaug", "few_shot")
        }
        if len(hashes) != 1:
            raise RuntimeError(f"candidate-index hash mismatch for query {index}")
        vanilla_payload, vanilla_dir = loaded["vanilla"]
        fem_payload, fem_dir = loaded["depth_aabb"]
        vanilla_arrays = np.load(vanilla_dir / vanilla_payload["arrays_file"])
        fem_arrays = np.load(fem_dir / fem_payload["arrays_file"])
        if not np.array_equal(vanilla_arrays["candidates"], fem_arrays["candidates"]):
            raise RuntimeError(f"FEM/FLAC candidate coordinates differ for query {index}")
        candidates = vanilla_arrays["candidates"].astype(np.float64)
        truth = np.asarray(reference["source_global"], dtype=np.float64)
        candidate_distances = np.linalg.norm(candidates - truth, axis=1)
        candidate_span = np.ptp(candidates, axis=0)
        if not fem_payload["coverage_protocol"]["strict_gate_passed"]:
            raise RuntimeError(f"Depth-AABB strict gate failed for query {index}")
        maximum_solver_residual = max(
            maximum_solver_residual,
            float(fem_payload["fem_audit"]["maximum_relative_solver_residual"]),
        )
        per_query.append(
            {
                "query_index": index,
                "query_id": reference["query_id"],
                "room": reference["room"],
                "candidate_count": int(reference["candidate_count"]),
                "candidate_indices_sha256": next(iter(hashes)),
                "candidate_geometry": {
                    "aabb_span_xyz_m": candidate_span.tolist(),
                    "aabb_diagonal_m": float(np.linalg.norm(candidate_span)),
                    "exact_random_success_probability_at_0_5m": float(
                        np.mean(candidate_distances <= 0.5)
                    ),
                    "exact_random_success_probability_at_1_0m": float(
                        np.mean(candidate_distances <= 1.0)
                    ),
                    "median_random_candidate_error_m": float(
                        np.median(candidate_distances)
                    ),
                },
                "localization_error_m": {
                    "vanilla": {
                        key: float(reference["metrics"][key]["localization_error_m"])
                        for key in ("1", "4", "8")
                    },
                    "fa_bf": {
                        key: float(loaded["fa_bf"][0]["metrics"][key]["localization_error_m"])
                        for key in ("1", "4", "8")
                    },
                    "yawaug": {
                        key: float(loaded["yawaug"][0]["metrics"][key]["localization_error_m"])
                        for key in ("1", "4", "8")
                    },
                    "few_shot_kctx8": float(
                        loaded["few_shot"][0]["metrics"]["8"]["localization_error_m"]
                    ),
                    "depth_aabb_kctx8": float(
                        fem_payload["metrics"]["localization_error_m"]
                    ),
                },
            }
        )

    slices = {}
    room_macro_slices = {}
    for k_gen in ("1", "4", "8"):
        metrics_by_model = {
            "vanilla": {
                index: arms["vanilla"][index][0]["metrics"][k_gen]
                for index in indices
            },
            "fa_bf": {
                index: arms["fa_bf"][index][0]["metrics"][k_gen]
                for index in indices
            },
            "yawaug": {
                index: arms["yawaug"][index][0]["metrics"][k_gen]
                for index in indices
            },
            "few_shot": {
                index: arms["few_shot"][index][0]["metrics"]["8"]
                for index in indices
            },
            "depth_aabb": {
                index: arms["depth_aabb"][index][0]["metrics"]
                for index in indices
            },
        }
        slices[k_gen] = {
            model: aggregate(list(model_metrics.values()))
            for model, model_metrics in metrics_by_model.items()
        }
        room_macro_slices[k_gen] = {}
        for model, model_metrics in metrics_by_model.items():
            metrics_by_room = defaultdict(list)
            for selected in records:
                index = int(selected["index"])
                metrics_by_room[selected["room"]].append(model_metrics[index])
            room_macro_slices[k_gen][model] = aggregate_room_macro(metrics_by_room)

    large_query_indices = [
        row["query_index"]
        for row in per_query
        if row["candidate_geometry"]["aabb_diagonal_m"] >= 5.0
    ]
    size_bias_audit = {
        "candidate_aabb_diagonal_median_m": float(
            np.median([row["candidate_geometry"]["aabb_diagonal_m"] for row in per_query])
        ),
        "exact_random_macro_success_probability_at_0_5m": float(
            np.mean(
                [
                    row["candidate_geometry"]["exact_random_success_probability_at_0_5m"]
                    for row in per_query
                ]
            )
        ),
        "exact_random_macro_success_probability_at_1_0m": float(
            np.mean(
                [
                    row["candidate_geometry"]["exact_random_success_probability_at_1_0m"]
                    for row in per_query
                ]
            )
        ),
        "large_candidate_domain_definition": "candidate AABB diagonal >= 5.0 m",
        "large_candidate_domain_query_indices": large_query_indices,
        "large_candidate_domain_exact_random_success_probability_at_1_0m": float(
            np.mean(
                [
                    row["candidate_geometry"]["exact_random_success_probability_at_1_0m"]
                    for row in per_query
                    if row["query_index"] in large_query_indices
                ]
            )
        ),
        "large_candidate_domain_k_gen_1": {
            "vanilla": aggregate_error_values(
                [arms["vanilla"][index][0]["metrics"]["1"]["localization_error_m"] for index in large_query_indices]
            ),
            "fa_bf": aggregate_error_values(
                [arms["fa_bf"][index][0]["metrics"]["1"]["localization_error_m"] for index in large_query_indices]
            ),
            "yawaug": aggregate_error_values(
                [arms["yawaug"][index][0]["metrics"]["1"]["localization_error_m"] for index in large_query_indices]
            ),
            "few_shot": aggregate_error_values(
                [arms["few_shot"][index][0]["metrics"]["8"]["localization_error_m"] for index in large_query_indices]
            ),
            "depth_aabb": aggregate_error_values(
                [arms["depth_aabb"][index][0]["metrics"]["localization_error_m"] for index in large_query_indices]
            ),
        },
    }

    run_summary = json.loads((args.fem_dir / "run_summary.json").read_text())
    nodes = [int(arms["depth_aabb"][index][0]["fem_audit"]["node_count"]) for index in indices]
    is_one_per_room_pilot = len(indices) == len({record["room"] for record in records})
    selected_query_count_by_room = defaultdict(int)
    for record in records:
        selected_query_count_by_room[record["room"]] += 1
    fem_error = np.asarray(
        [arms["depth_aabb"][index][0]["metrics"]["localization_error_m"] for index in indices]
    )
    vanilla_k1_error = np.asarray(
        [arms["vanilla"][index][0]["metrics"]["1"]["localization_error_m"] for index in indices]
    )
    paired_delta = fem_error - vanilla_k1_error
    scope = (
        "one fixed complete-coverage query from each of 14 non-oversized rooms"
        if is_one_per_room_pilot
        else "all strict-complete-coverage queries from 14 non-oversized rooms"
    )
    payload = {
        "schema_version": 1,
        "scope": scope,
        "selection": str(args.selection.resolve()),
        "selection_file_sha256": file_sha256(args.selection),
        "query_count": len(indices),
        "room_count": len({record["room"] for record in records}),
        "candidate_alignment_verified_query_count": len(indices),
        "k_gen_slices": [1, 4, 8],
        "fem_k_ctx": 8,
        "fem_has_k_gen": False,
        "slices": slices,
        "room_macro_slices": room_macro_slices,
        "selected_query_count_by_room": dict(sorted(selected_query_count_by_room.items())),
        "paired_depth_aabb_vs_vanilla_k_gen_1": {
            "depth_aabb_lower_error_count": int(np.sum(paired_delta < 0.0)),
            "equal_error_count": int(np.sum(paired_delta == 0.0)),
            "depth_aabb_higher_error_count": int(np.sum(paired_delta > 0.0)),
            "mean_paired_error_delta_m": float(paired_delta.mean()),
            "median_paired_error_delta_m": float(np.median(paired_delta)),
        },
        "size_bias_audit": size_bias_audit,
        "fem_runtime": {
            "workers": run_summary["workers"],
            "solver_threads_per_worker": run_summary["solver_threads_per_worker"],
            "completed_count": int(run_summary["completed_count"]),
            "resumed_count": int(run_summary["resumed_count"]),
            "wall_seconds": float(run_summary["wall_seconds"]),
            "summed_query_total_seconds": float(
                sum(arms["depth_aabb"][index][0]["runtime_seconds"]["total"] for index in indices)
            ),
            "minimum_node_count": min(nodes),
            "maximum_node_count": max(nodes),
            "maximum_relative_solver_residual": maximum_solver_residual,
        },
        "per_query": per_query,
    }
    payload["sha256"] = canonical_sha256(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "five_method_summary.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)

    heading = (
        "# Five-method Depth-AABB matched pilot"
        if is_one_per_room_pilot
        else "# Five-method Depth-AABB matched 97-query result"
    )
    scope_sentence = (
        "This is a deterministic one-query-per-room pilot over the 14 non-oversized rooms "
        f"(`n={len(indices)}` per K slice)."
        if is_one_per_room_pilot
        else "This is the complete frozen strict-coverage subset over the 14 non-oversized "
        f"rooms (`n={len(indices)}` per K slice; 97/112 source queries, 86.6% coverage)."
    )
    lines = [
        heading,
        "",
        scope_sentence,
        "",
        "Localization errors were not used for selection. This is nevertheless a "
        "coverage-conditioned sample and is not representative of the 15 source queries "
        "that Depth-AABB cannot contain.",
        "",
        "All FEM candidate coordinate arrays are byte-identical to Vanilla FLAC, all "
        "four learned arms share the same candidate-index hashes, and every Depth-AABB "
        "query passes the strict receiver/source/context/candidate coverage gate.",
        "",
    ]
    for k_gen in (1, 4, 8):
        lines.extend(format_table(k_gen, slices[str(k_gen)]))
    depth = slices["1"]["depth_aabb"]
    depth_room_macro = room_macro_slices["1"]["depth_aabb"]
    vanilla_room_macro = room_macro_slices["1"]["vanilla"]
    vanilla_large = size_bias_audit["large_candidate_domain_k_gen_1"]["vanilla"]
    depth_large = size_bias_audit["large_candidate_domain_k_gen_1"]["depth_aabb"]
    largest_failures = sorted(
        per_query,
        key=lambda row: row["localization_error_m"]["depth_aabb_kctx8"],
        reverse=True,
    )[:3]
    failure_text = ", ".join(
        f"{row['room']} q{row['query_index']} "
        f"({row['localization_error_m']['depth_aabb_kctx8']:.3f} m)"
        for row in largest_failures
    )
    lines.extend(
        [
            "## Interpretation",
            "",
            f"Depth-AABB FEM has median error {depth['median_localization_error_m']:.3f} m, "
            f"mean error {depth['mean_localization_error_m']:.3f} m, "
            f"SR@0.5m {100 * depth['success_rate_at_0_5m']:.1f}%, and "
            f"SR@1.0m {100 * depth['success_rate_at_1_0m']:.1f}% on this conditional scope.",
            "",
            "The strict subset is imbalanced across rooms, so the room-macro view is more "
            "conservative: Depth-AABB has room-macro mean error "
            f"{depth_room_macro['room_macro_mean_localization_error_m']:.3f} m and "
            f"room-macro SR@1.0m "
            f"{100 * depth_room_macro['room_macro_success_rate_at_1_0m']:.1f}%, versus "
            f"{vanilla_room_macro['room_macro_mean_localization_error_m']:.3f} m / "
            f"{100 * vanilla_room_macro['room_macro_success_rate_at_1_0m']:.1f}% for "
            "Vanilla FLAC at K_gen=1.",
            "",
            "Candidate-domain size remains a material confounder. An exactly uniform random "
            f"candidate has macro SR@1.0m "
            f"{100 * size_bias_audit['exact_random_macro_success_probability_at_1_0m']:.1f}%. "
            f"On the {len(large_query_indices)} queries whose candidate AABB diagonal is at "
            f"least 5 m, Depth-AABB FEM has median error "
            f"{depth_large['median_localization_error_m']:.3f} m and SR@1.0m "
            f"{100 * depth_large['success_rate_at_1_0m']:.1f}%, versus "
            f"{vanilla_large['median_localization_error_m']:.3f} m / "
            f"{100 * vanilla_large['success_rate_at_1_0m']:.1f}% for Vanilla FLAC at "
            "K_gen=1.",
            "",
            f"The three largest Depth-AABB errors are: {failure_text}.",
            "",
            "## FEM execution audit",
            "",
            f"- Wall time: {payload['fem_runtime']['wall_seconds'] / 60.0:.1f} min "
            f"with {run_summary['workers']} workers x {run_summary['solver_threads_per_worker']} MKL threads.",
            f"- Newly completed/resumed exact results: "
            f"{run_summary['completed_count']}/{run_summary['resumed_count']}.",
            f"- Mesh nodes: {min(nodes):,}--{max(nodes):,}.",
            f"- Maximum relative linear-solver residual: "
            f"`{maximum_solver_residual:.3e}`.",
            "- FEM is deterministic and is reused as the same reference across the three "
            "K_gen slices; it is not counted as three independent runs.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in payload.items() if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
