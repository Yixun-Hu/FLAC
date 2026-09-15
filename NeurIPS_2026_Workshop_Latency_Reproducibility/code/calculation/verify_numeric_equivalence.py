#!/usr/bin/env python3
"""Verify numeric equivalence while ignoring location-dependent provenance paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METHODS = (
    "vanilla_flac",
    "fa_bf_flac",
    "yawaug_flac",
    "few_shot_rir",
    "fem_omp",
)
STATISTICS = (
    "query_count",
    "mean_seconds",
    "median_seconds",
    "p90_seconds",
    "minimum_seconds",
    "maximum_seconds",
    "summed_seconds",
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("sha256") != canonical_sha256(payload):
        raise RuntimeError(f"invalid embedded SHA-256: {path}")
    return payload


def same_number(expected: float, actual: float, label: str) -> None:
    if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"numeric mismatch at {label}: {expected} != {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    expected = load(args.expected.resolve())
    actual = load(args.actual.resolve())

    for field in ("repeat_count", "measurement_repeat_count_by_method", "latency_protocol", "scope"):
        if expected[field] != actual[field]:
            raise RuntimeError(f"metadata mismatch at {field}")

    for method in METHODS:
        for statistic in STATISTICS:
            same_number(
                expected["overall"][method][statistic],
                actual["overall"][method][statistic],
                f"overall.{method}.{statistic}",
            )
            same_number(
                expected["per_candidate"][method][statistic],
                actual["per_candidate"][method][statistic],
                f"per_candidate.{method}.{statistic}",
            )
        for statistic in ("amortized_seconds", "candidate_evaluations"):
            same_number(
                expected["per_candidate"][method][statistic],
                actual["per_candidate"][method][statistic],
                f"per_candidate.{method}.{statistic}",
            )

    if set(expected["queries"]) != set(actual["queries"]):
        raise RuntimeError("query coverage mismatch")
    for index, expected_query in expected["queries"].items():
        actual_query = actual["queries"][index]
        for field in ("query_id", "room", "candidate_count", "fem_source"):
            if expected_query[field] != actual_query[field]:
                raise RuntimeError(f"query {index} identity mismatch at {field}")
        for method in METHODS:
            expected_method = expected_query["methods"][method]
            actual_method = actual_query["methods"][method]
            if len(expected_method["repeat_seconds"]) != len(actual_method["repeat_seconds"]):
                raise RuntimeError(f"query {index}/{method} repeat-count mismatch")
            for ordinal, (left, right) in enumerate(
                zip(expected_method["repeat_seconds"], actual_method["repeat_seconds"])
            ):
                same_number(left, right, f"queries.{index}.{method}.repeat_seconds[{ordinal}]")
            same_number(
                expected_method["median_seconds"],
                actual_method["median_seconds"],
                f"queries.{index}.{method}.median_seconds",
            )

    print(
        "PASS: all latency numbers, query identities, scope fields, and timing-protocol "
        "fields match expected/summary_final.json."
    )
    print(
        "Note: top-level SHA-256 values may differ after relocation because the original "
        "summaries retain resolved provenance paths; those paths are intentionally ignored."
    )


if __name__ == "__main__":
    main()
