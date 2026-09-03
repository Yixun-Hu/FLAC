#!/usr/bin/env python3
"""Validate and summarize one completed FewshotRiR localization run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256, load_pilot_manifest
from src.localization.reporting import _summarize_metric_rows, summarize_random_baseline
from src.localization.runner import verify_hashed_payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context-count", type=int, required=True)
    parser.add_argument("--alignment-verification", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    pilot = load_pilot_manifest(args.pilot_manifest)
    run = json.loads((args.run_dir / "run_manifest.json").read_text())
    verify_hashed_payload(run, "run manifest")
    identity = run["identity"]
    expected_indices = [int(record["index"]) for record in pilot["records"]]
    if identity.get("method") != "FewshotRiR":
        raise RuntimeError("run is not FewshotRiR")
    if identity.get("pilot_manifest_sha256") != pilot["sha256"]:
        raise RuntimeError("run and pilot hashes differ")
    if identity.get("query_indices") != expected_indices:
        raise RuntimeError("run query order differs from the frozen pilot")
    if identity.get("context_counts") != [args.context_count]:
        raise RuntimeError("run context count differs from the requested summary")

    results = []
    for selected in pilot["records"]:
        index = int(selected["index"])
        result = json.loads(
            (args.run_dir / "queries" / f"query_{index:05d}.json").read_text()
        )
        verify_hashed_payload(result, f"query {index}")
        if (
            result["run_manifest_sha256"] != run["sha256"]
            or int(result["query_index"]) != index
            or result["query_id"] != selected["query_id"]
            or int(result["candidate_count"]) != int(selected["candidate_count"])
            or result["candidate_indices_sha256"]
            != selected["candidate_indices_sha256"]
        ):
            raise RuntimeError(f"query identity or candidate grid mismatch at {index}")
        results.append(result)

    alignment = None
    if args.alignment_verification is not None:
        alignment = json.loads(args.alignment_verification.read_text())
        verify_hashed_payload(alignment, "candidate alignment verification")
        if (
            int(alignment["query_count"]) != len(results)
            or int(alignment["candidate_query_pairs"])
            != int(sum(result["candidate_count"] for result in results))
        ):
            raise RuntimeError("alignment verification does not match completed run")

    rows = []
    for result in results:
        row = dict(result["metrics"][str(args.context_count)])
        row["room"] = result["room"]
        rows.append(row)
    elapsed = np.asarray([float(result["elapsed_seconds"]) for result in results])
    candidate_counts = np.asarray(
        [int(result["candidate_count"]) for result in results]
    )
    metrics = _summarize_metric_rows(rows)
    metrics["p90_localization_error_m"] = float(
        np.quantile([row["localization_error_m"] for row in rows], 0.9)
    )
    payload = {
        "schema_version": 1,
        "method": "FewshotRiR",
        "context_count": args.context_count,
        "generated_rirs_per_candidate": 1,
        "selection_rule": "AGREE cosine",
        "query_count": len(results),
        "room_count": len({result["room"] for result in results}),
        "candidate_count": int(candidate_counts.sum()),
        "run_manifest_sha256": run["sha256"],
        "pilot_manifest_sha256": pilot["sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "metrics": metrics,
        "random_candidate_baseline": summarize_random_baseline(results),
        "latency": {
            "protocol": "synchronized full query pass; model loading excluded",
            "mean_seconds_per_query": float(elapsed.mean()),
            "median_seconds_per_query": float(np.median(elapsed)),
            "p90_seconds_per_query": float(np.quantile(elapsed, 0.9)),
            "summed_seconds": float(elapsed.sum()),
            "amortized_ms_per_candidate": float(
                1000.0 * elapsed.sum() / candidate_counts.sum()
            ),
        },
        "candidate_alignment_verification_sha256": (
            alignment["sha256"] if alignment is not None else None
        ),
    }
    payload["sha256"] = canonical_sha256(payload)

    metrics = payload["metrics"]
    latency = payload["latency"]
    markdown = "\n".join(
        [
            "# FewshotRiR shared-candidate localization",
            "",
            (
                f"Scope: {payload['query_count']} queries / {payload['room_count']} rooms / "
                f"{payload['candidate_count']} candidate-query pairs; "
                f"K_ctx={args.context_count}, K_gen=1, AGREE selection."
            ),
            "",
            "| Mean error (m) | Median error (m) | P90 error (m) | Success@0.5 | Success@1.0 | Resolution-aware@0.5 |",
            "|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics['mean_localization_error_m']:.3f} | "
                f"{metrics['median_localization_error_m']:.3f} | "
                f"{metrics['p90_localization_error_m']:.3f} | "
                f"{metrics['success_0_5m']:.4f} | {metrics['success_1_0m']:.4f} | "
                f"{metrics['oracle_normalized_success_0_5m']:.4f} |"
            ),
            "",
            (
                "All queries are native FewshotRiR predictions. Candidate coordinates were "
                "verified bit-exact against the reference seed-43 FEM/FLAC protocol."
            ),
            "",
            "| Mean s/query | Median s/query | P90 s/query | Amortized ms/candidate |",
            "|---:|---:|---:|---:|",
            (
                f"| {latency['mean_seconds_per_query']:.3f} | "
                f"{latency['median_seconds_per_query']:.3f} | "
                f"{latency['p90_seconds_per_query']:.3f} | "
                f"{latency['amortized_ms_per_candidate']:.3f} |"
            ),
            "",
        ]
    )
    _atomic_text(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.output_md, markdown)
    print(json.dumps({"sha256": payload["sha256"], "queries": len(results)}, indent=2))


if __name__ == "__main__":
    main()
