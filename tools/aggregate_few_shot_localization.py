#!/usr/bin/env python3
"""Validate and aggregate disjoint Few-ShotRIR localization pilot batches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.baseline_experiment import _completed_baseline_query
from src.localization.pilot import canonical_sha256, load_pilot_manifest
from src.localization.reporting import (
    _summarize_metric_rows,
    load_completed_arm,
    summarize_random_baseline,
)
from src.localization.runner import verify_hashed_payload


EXPECTED_CONTEXT_COUNTS = [1, 8]
SHARED_IDENTITY_KEYS = (
    "method",
    "agree_checkpoint_sha256",
    "candidate_batch_size",
    "checkpoint_sha256",
    "context_counts",
    "context_manifest_sha256",
    "geometry_audit_sha256",
    "model_config_sha256",
    "random_seed",
    "selection_rule",
)


def load_completed_baseline(output_dir: Path, pilot: dict) -> tuple[dict, list[dict]]:
    run_path = output_dir / "run_manifest.json"
    if not run_path.is_file():
        raise RuntimeError(f"missing run manifest: {run_path}")
    run = json.loads(run_path.read_text())
    verify_hashed_payload(run, "baseline run manifest")
    identity = run["identity"]
    expected_indices = [int(record["index"]) for record in pilot["records"]]
    if identity.get("method") != "few_shot_rir_waveform":
        raise RuntimeError("run is not Few-ShotRIR-Waveform")
    if identity.get("pilot_manifest_sha256") != pilot["sha256"]:
        raise RuntimeError("run does not use the requested pilot manifest")
    if identity.get("query_indices") != expected_indices:
        raise RuntimeError("run query order differs from the complete pilot")
    if identity.get("context_counts") != EXPECTED_CONTEXT_COUNTS:
        raise RuntimeError("Few-ShotRIR run does not use K_ctx={1,8}")

    results = []
    for record in pilot["records"]:
        result = _completed_baseline_query(
            output_dir,
            query_index=int(record["index"]),
            query_id=record["query_id"],
            candidate_count=int(record["candidate_count"]),
            run_sha256=run["sha256"],
        )
        if result is None:
            raise RuntimeError(f"incomplete query: {record['query_id']}")
        results.append(result)
    return run, results


def summarize_contexts(results: list[dict]) -> dict:
    output = {}
    for count in EXPECTED_CONTEXT_COUNTS:
        rows = []
        for result in results:
            row = dict(result["metrics"][str(count)])
            row["room"] = result["room"]
            rows.append(row)
        output[str(count)] = _summarize_metric_rows(rows)
    return output


def aggregate_batches(
    labels: list[str],
    pilots: list[dict],
    baseline_dirs: list[Path],
    reference_dirs: list[Path],
) -> dict:
    if not (len(labels) == len(pilots) == len(baseline_dirs) == len(reference_dirs)):
        raise ValueError("batch labels, pilots, baseline dirs, and references must align")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("at least two uniquely labeled batches are required")

    runs = []
    all_results = []
    seen_indices: set[int] = set()
    batch_payloads = []
    reference_runs = []
    for label, pilot, baseline_dir, reference_dir in zip(
        labels, pilots, baseline_dirs, reference_dirs
    ):
        run, results = load_completed_baseline(baseline_dir, pilot)
        reference_run, reference_results = load_completed_arm(reference_dir, pilot)
        if len(results) != len(reference_results):
            raise RuntimeError("baseline/reference query counts differ")
        for result, reference in zip(results, reference_results):
            if (
                result["query_id"] != reference["query_id"]
                or result["candidate_indices_sha256"]
                != reference["candidate_indices_sha256"]
                or result["random_candidate_metrics"]
                != reference["random_candidate_metrics"]
            ):
                raise RuntimeError(
                    f"Few-ShotRIR is not candidate-aligned with reference: {result['query_id']}"
                )
        indices = {int(result["query_index"]) for result in results}
        overlap = seen_indices & indices
        if overlap:
            raise RuntimeError(f"pilot batches overlap at query indices: {sorted(overlap)}")
        seen_indices.update(indices)
        runs.append(run)
        all_results.extend(results)
        reference_runs.append(reference_run["sha256"])
        batch_payloads.append(
            {
                "label": label,
                "pilot_manifest_sha256": pilot["sha256"],
                "run_manifest_sha256": run["sha256"],
                "reference_run_manifest_sha256": reference_run["sha256"],
                "query_count": len(results),
            }
        )

    first_identity = runs[0]["identity"]
    for run in runs[1:]:
        identity = run["identity"]
        if any(identity[key] != first_identity[key] for key in SHARED_IDENTITY_KEYS):
            raise RuntimeError("Few-ShotRIR batches do not share the frozen protocol")
    room_counts = Counter(result["room"] for result in all_results)
    if len(set(room_counts.values())) != 1:
        raise RuntimeError("combined query count is not balanced across rooms")

    payload = {
        "schema_version": 1,
        "method": "few_shot_rir_waveform",
        "query_count": len(all_results),
        "room_count": len(room_counts),
        "queries_per_room": next(iter(room_counts.values())),
        "context_counts": EXPECTED_CONTEXT_COUNTS,
        "checkpoint_sha256": first_identity["checkpoint_sha256"],
        "model_config_sha256": first_identity["model_config_sha256"],
        "agree_checkpoint_sha256": first_identity["agree_checkpoint_sha256"],
        "context_manifest_sha256": first_identity["context_manifest_sha256"],
        "geometry_audit_sha256": first_identity["geometry_audit_sha256"],
        "selection_rule": first_identity["selection_rule"],
        "batches": batch_payloads,
        "summary": summarize_contexts(all_results),
        "random_candidate_baseline": summarize_random_baseline(all_results),
        "summed_query_elapsed_seconds": float(
            sum(float(result["elapsed_seconds"]) for result in all_results)
        ),
        "reference_candidate_alignment": {
            "arm": "vanilla_flac",
            "verified_query_count": len(all_results),
            "run_manifest_sha256": reference_runs,
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def render_markdown(payload: dict) -> str:
    lines = [
        "# Few-ShotRIR-Waveform 128-query localization results",
        "",
        f"Scope: {payload['query_count']} queries / {payload['room_count']} rooms / "
        f"{payload['queries_per_room']} unique targets per room; "
        f"K_ctx={payload['context_counts']}.",
        "",
        "Every query ID, candidate-grid hash, and deterministic random-candidate result "
        "was validated against the corresponding Vanilla FLAC run.",
        "",
        "| Method | Context | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for count in ("1", "8"):
        item = payload["summary"][count]
        lines.append(
            f"| Few-ShotRIR-Waveform | K_ctx={count} | "
            f"{item['mean_localization_error_m']:.3f} | "
            f"{item['median_localization_error_m']:.3f} | "
            f"{item['success_0_5m']:.3f} | {item['success_1_0m']:.3f} | "
            f"{item['oracle_normalized_success_0_5m']:.3f} |"
        )
    random = payload["random_candidate_baseline"]
    lines.extend(
        [
            f"| Random candidate | — | {random['mean_localization_error_m']:.3f} | "
            f"{random['median_localization_error_m']:.3f} | "
            f"{random['success_0_5m']:.3f} | {random['success_1_0m']:.3f} | "
            f"{random['oracle_normalized_success_0_5m']:.3f} |",
            "",
            f"Summed measured query work: {payload['summed_query_elapsed_seconds']:.1f} seconds.",
            "",
            "This is the aligned two-batch room-stratified diagnostic scope, not the "
            "complete 5,337-query unseen-room evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-label", action="append", required=True)
    parser.add_argument("--pilot-manifest", action="append", type=Path, required=True)
    parser.add_argument("--baseline-dir", action="append", type=Path, required=True)
    parser.add_argument("--reference-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json.resolve(), args.output_md.resolve()):
        try:
            output.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError("aggregate outputs must stay inside NeuriPs_Workshop") from error
    pilots = [load_pilot_manifest(path) for path in args.pilot_manifest]
    payload = aggregate_batches(
        args.batch_label,
        pilots,
        args.baseline_dir,
        args.reference_dir,
    )
    atomic_text(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(args.output_md, render_markdown(payload))
    print(
        json.dumps(
            {
                "sha256": payload["sha256"],
                "queries": payload["query_count"],
                "rooms": payload["room_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
