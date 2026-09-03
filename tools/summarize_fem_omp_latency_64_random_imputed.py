#!/usr/bin/env python3
"""Build a balanced 16-room/64-query FEM--OMP latency estimate.

The frozen 64-query scope comes from the registered random-fallback accuracy
summary.  Native 12-thread timings are retained for standard rooms.  Missing
strict-coverage timings are imputed by deterministic same-room random choice.
Cafe/Auditorium use the externally reported 13-core seven-query distribution;
because its raw per-query timings were not archived, a seven-point proxy is
reconstructed from its reported mean/median/P90/min/max before sampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOG = REPO_ROOT / "worklog" / "worklog_yixun"


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_hashed_json(path: Path) -> dict:
    payload = load_json(path)
    expected = payload.get("sha256")
    if expected is None:
        raise RuntimeError(f"hashed JSON has no sha256: {path}")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_sha256(content) != expected:
        raise RuntimeError(f"stale or corrupt hashed JSON: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("latency values must be a nonempty finite vector")
    seconds = {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "sum": float(array.sum()),
    }
    return {
        "query_count": int(len(array)),
        "seconds": seconds,
        "minutes": {key: value / 60.0 for key, value in seconds.items()},
    }


def reconstruct_external_proxy(
    *, mean: float, median: float, p90: float, minimum: float, maximum: float
) -> list[float]:
    """Return seven sorted points matching the reported five statistics.

    NumPy's default linear P90 for n=7 is 0.6*x[5] + 0.4*x[6].  The two
    unregistered lower-middle points are placed at trisections between the
    minimum and median.  The remaining upper-middle point is fixed by the
    reported mean.  This is a transparent proxy, not recovered raw timing.
    """

    p90_lower = (p90 - 0.4 * maximum) / 0.6
    lower_second = minimum + (median - minimum) / 3.0
    lower_third = minimum + 2.0 * (median - minimum) / 3.0
    upper_middle = (
        7.0 * mean
        - minimum
        - lower_second
        - lower_third
        - median
        - p90_lower
        - maximum
    )
    values = [
        minimum,
        lower_second,
        lower_third,
        median,
        upper_middle,
        p90_lower,
        maximum,
    ]
    if any(left > right for left, right in zip(values, values[1:])):
        raise RuntimeError(f"reported external statistics are inconsistent: {values}")
    reconstructed = summarize(values)["seconds"]
    checks = {
        "mean": mean,
        "median": median,
        "p90": p90,
        "minimum": minimum,
        "maximum": maximum,
    }
    for key, expected in checks.items():
        if not np.isclose(reconstructed[key], expected, atol=1e-8, rtol=0.0):
            raise RuntimeError(
                f"external proxy does not preserve {key}: "
                f"{reconstructed[key]} != {expected}"
            )
    return values


def load_standard_timings(results_dir: Path) -> tuple[dict[int, dict], dict[str, list[dict]]]:
    by_index: dict[int, dict] = {}
    by_room: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("query_*_depth_aabb_result.json")):
        payload = load_hashed_json(path)
        if payload.get("method") != "fem_sabine_depth_aabb":
            raise RuntimeError(f"unexpected method in {path}")
        row = {
            "query_index": int(payload["query_index"]),
            "query_id": payload["query_id"],
            "room": payload["room"],
            "latency_seconds": float(payload["runtime_seconds"]["total"]),
            "result_file": str(path.resolve()),
            "result_sha256": payload["sha256"],
            "result_file_sha256": file_sha256(path),
        }
        if row["query_index"] in by_index:
            raise RuntimeError(f"duplicate standard query {row['query_index']}")
        by_index[row["query_index"]] = row
        by_room[row["room"]].append(row)
    if len(by_index) != 97:
        raise RuntimeError(f"expected 97 standard timings, got {len(by_index)}")
    return by_index, dict(by_room)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope-summary",
        type=Path,
        default=DEFAULT_WORKLOG / "exp_23_five_method_64_random_fallback/summary.json",
    )
    parser.add_argument(
        "--standard-results-dir",
        type=Path,
        default=DEFAULT_WORKLOG / "exp_16_depth_aabb_matched_97/results",
    )
    parser.add_argument(
        "--external-summary",
        type=Path,
        default=(
            DEFAULT_WORKLOG
            / "exp_18_depth_aabb_oversized_benchmark"
            / "external_server_7query_omp_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WORKLOG / "exp_24_fem_omp_latency_64_random_imputed",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--external-min-seconds",
        type=float,
        default=25.47 * 60.0,
        help="Minimum from the user-provided Cafe/Auditorium latency table",
    )
    parser.add_argument(
        "--external-max-seconds",
        type=float,
        default=75.44 * 60.0,
        help="Maximum from the user-provided Cafe/Auditorium latency table",
    )
    args = parser.parse_args()

    scope = load_hashed_json(args.scope_summary)
    if scope.get("query_count") != 64 or scope.get("room_count") != 16:
        raise RuntimeError("scope must be the balanced 16-room/64-query summary")
    room_counts = Counter(row["room"] for row in scope["per_query"])
    if len(room_counts) != 16 or set(room_counts.values()) != {4}:
        raise RuntimeError(f"scope is not 16 rooms x 4 queries: {dict(room_counts)}")

    standard_by_index, standard_by_room = load_standard_timings(
        args.standard_results_dir
    )
    external = load_json(args.external_summary)
    external_stats = external["observed_runtime_seconds_per_query"][
        "fem_sabine_depth_aabb_13_physical_cores"
    ]
    external_proxy = reconstruct_external_proxy(
        mean=float(external_stats["mean"]),
        median=float(external_stats["median"]),
        p90=float(external_stats["p90"]),
        minimum=float(args.external_min_seconds),
        maximum=float(args.external_max_seconds),
    )
    # The execution audit has five Cafe and two Auditorium queries.  Its two
    # largest runtimes correspond to the two-query Auditorium job; the five
    # lower proxy points therefore form the Cafe pool.
    external_by_room = {
        "Cafe_idx_1": external_proxy[:5],
        "Auditorium_idx_1": external_proxy[5:],
    }

    rng = np.random.default_rng(args.random_seed)
    rows = []
    for reference in sorted(scope["per_query"], key=lambda row: int(row["query_index"])):
        index = int(reference["query_index"])
        room = reference["room"]
        status = reference["fem_evaluation_status"]
        if index in standard_by_index and status == "evaluated":
            observed = standard_by_index[index]
            latency = observed["latency_seconds"]
            latency_status = "observed_standard_12_thread"
            donor = {
                "query_index": observed["query_index"],
                "result_file": observed["result_file"],
                "result_sha256": observed["result_sha256"],
                "result_file_sha256": observed["result_file_sha256"],
            }
            hardware = "12-thread CPU FEM"
        else:
            if room in external_by_room:
                pool = external_by_room[room]
                donor_index = int(rng.integers(0, len(pool)))
                latency = float(pool[donor_index])
                hardware = "13-physical-core external FEM estimate"
                donor = {
                    "proxy_pool_index": donor_index,
                    "proxy_pool_seconds": list(map(float, pool)),
                }
                latency_status = (
                    "random_choice_failure_estimate"
                    if status != "evaluated"
                    else "external_distribution_estimate"
                )
            else:
                pool = standard_by_room.get(room, [])
                if not pool:
                    raise RuntimeError(f"no same-room standard latency donors for {room}")
                donor_index = int(rng.integers(0, len(pool)))
                selected = pool[donor_index]
                latency = float(selected["latency_seconds"])
                hardware = "12-thread CPU FEM estimate"
                donor = {
                    "query_index": selected["query_index"],
                    "result_file": selected["result_file"],
                    "result_sha256": selected["result_sha256"],
                    "result_file_sha256": selected["result_file_sha256"],
                    "room_pool_size": len(pool),
                }
                latency_status = "random_choice_failure_estimate"
        rows.append(
            {
                "query_index": index,
                "query_id": reference["query_id"],
                "scene": reference["scene"],
                "room": room,
                "native_fem_status": status,
                "latency_status": latency_status,
                "latency_seconds": float(latency),
                "latency_minutes": float(latency / 60.0),
                "hardware": hardware,
                "donor": donor,
            }
        )

    if len(rows) != 64:
        raise RuntimeError(f"expected 64 output rows, got {len(rows)}")
    latency_status_counts = Counter(row["latency_status"] for row in rows)
    native_status_counts = Counter(row["native_fem_status"] for row in rows)
    if native_status_counts.get("strict_coverage_failure_random_fallback") != 7:
        raise RuntimeError(f"expected seven native FEM failures: {native_status_counts}")

    by_room = {
        room: summarize(
            [row["latency_seconds"] for row in rows if row["room"] == room]
        )
        for room in sorted(room_counts)
    }
    payload = {
        "schema_version": 1,
        "method": "fem_sabine_depth_aabb",
        "display_name": "FEM-Sabine + Room-Helps OMP (Depth-AABB)",
        "scope": "16 rooms x 4 frozen queries",
        "query_count": 64,
        "room_count": 16,
        "queries_per_room": 4,
        "random_seed": int(args.random_seed),
        "quantile_method": "numpy_default_linear",
        "hardware_boundary": (
            "hardware-mixed estimate: standard-room timings use 12-thread CPU FEM; "
            "Cafe/Auditorium estimates use the external 13-physical-core distribution"
        ),
        "latency_boundary": (
            "Depth-AABB mesh construction + operator construction + 102-bin FEM "
            "forward solve; Room-Helps OMP selection time was not separately recorded"
        ),
        "imputation_policy": {
            "name": "deterministic_same_room_random_choice",
            "seed": int(args.random_seed),
            "failed_standard_queries": (
                "sample with replacement from all observed 97-query timings in the "
                "same room"
            ),
            "cafe_auditorium_queries": (
                "sample with replacement from a room-specific seven-point proxy of "
                "the reported external 13-core distribution"
            ),
            "ground_truth_usage": "none",
        },
        "source_files": {
            "scope_summary": str(args.scope_summary.resolve()),
            "scope_summary_sha256": file_sha256(args.scope_summary),
            "standard_results_dir": str(args.standard_results_dir.resolve()),
            "external_summary": str(args.external_summary.resolve()),
            "external_summary_sha256": file_sha256(args.external_summary),
        },
        "external_proxy": {
            "raw_per_query_timings_available": False,
            "reported_statistics_seconds": {
                "mean": float(external_stats["mean"]),
                "median": float(external_stats["median"]),
                "p90": float(external_stats["p90"]),
                "minimum": float(args.external_min_seconds),
                "maximum": float(args.external_max_seconds),
            },
            "reconstructed_sorted_seconds": list(map(float, external_proxy)),
            "reconstructed_statistics": summarize(external_proxy),
            "room_split_assumption": (
                "five lower proxy points assigned to Cafe and two upper proxy points "
                "to Auditorium, consistent with the archived job-level runtime audit"
            ),
        },
        "counts": {
            "native_fem_status": dict(sorted(native_status_counts.items())),
            "latency_status": dict(sorted(latency_status_counts.items())),
        },
        "overall": summarize([row["latency_seconds"] for row in rows]),
        "per_room": by_room,
        "per_query": rows,
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_json(args.output_dir / "summary.json", payload)

    csv_path = args.output_dir / "per_query.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "query_index",
                "query_id",
                "scene",
                "room",
                "native_fem_status",
                "latency_status",
                "latency_seconds",
                "latency_minutes",
                "hardware",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    os.replace(csv_tmp, csv_path)

    overall = payload["overall"]
    seconds = overall["seconds"]
    minutes = overall["minutes"]
    lines = [
        "# FEM--OMP (Depth-AABB) balanced 16-room/64-query latency estimate",
        "",
        "This table uses the registered 16-room x 4-query scope. Native standard-room "
        "latencies are retained, while missing strict-coverage latencies are imputed "
        f"by deterministic same-room random choice (`seed={args.random_seed}`).",
        "",
        "| Scope | Mean | Median | P90 | Min--Max |",
        "|---|---:|---:|---:|---:|",
        (
            f"| 16 rooms / 64 queries | {seconds['mean']:.2f} s "
            f"({minutes['mean']:.2f} min) | {seconds['median']:.2f} s "
            f"({minutes['median']:.2f} min) | {seconds['p90']:.2f} s "
            f"({minutes['p90']:.2f} min) | {seconds['minimum']:.2f}--"
            f"{seconds['maximum']:.2f} s ({minutes['minimum']:.2f}--"
            f"{minutes['maximum']:.2f} min) |"
        ),
        "",
        "## Data accounting",
        "",
        f"- Exact standard 12-thread query timings: "
        f"{latency_status_counts.get('observed_standard_12_thread', 0)}.",
        f"- Strict-coverage failure timings imputed by same-room random choice: "
        f"{latency_status_counts.get('random_choice_failure_estimate', 0)}.",
        f"- Cafe/Auditorium evaluated-query timings estimated from the archived "
        f"13-core distribution: "
        f"{latency_status_counts.get('external_distribution_estimate', 0)}.",
        "",
        "## Interpretation boundary",
        "",
        "This is a hardware-mixed estimate, not a normalized benchmark. The external "
        "Cafe/Auditorium archive retained only aggregate latency statistics, so its "
        "seven-point per-query vector is reconstructed to match the reported mean, "
        "median, P90, minimum, and maximum before seeded sampling. The reconstruction "
        "and every donor choice are recorded in `summary.json`; query-level values are "
        "also exported in `per_query.csv`.",
        "",
        "The timing boundary is Depth-AABB mesh/operator construction plus the 102-bin "
        "FEM solve. OMP selection was not timed separately.",
        "",
    ]
    atomic_text(args.output_dir / "summary.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "counts": payload["counts"],
                "overall": overall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
