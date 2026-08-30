#!/usr/bin/env python3
"""Merge local and accepted external FEM--OMP results for the frozen 112 queries."""

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


METHOD = "fem_sabine_depth_aabb"
SELECTION_SHA256 = "67bdd25f3df704bc1c57558e7cb68cfaa5d9e60758f2c70e87a59eddc33bcfa9"


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


def load_local_results(directory: Path) -> dict[int, tuple[dict, Path]]:
    results = {}
    for path in sorted(directory.glob("query_*_depth_aabb_result.json")):
        payload = load_hashed_json(path)
        index = int(payload["query_index"])
        if payload.get("method") != METHOD:
            raise RuntimeError(f"query {index} has unexpected method in {path}")
        if index in results:
            raise RuntimeError(f"duplicate local query {index} in {directory}")
        arrays_path = directory / payload["arrays_file"]
        mesh_path = directory / payload["mesh_file"]
        if file_sha256(arrays_path) != payload["arrays_sha256"]:
            raise RuntimeError(f"query {index} scores SHA-256 mismatch")
        if file_sha256(mesh_path) != payload["mesh_sha256"]:
            raise RuntimeError(f"query {index} mesh SHA-256 mismatch")
        results[index] = (payload, path)
    return results


def metric_from_error(error: float, oracle: float) -> dict:
    excess = error - oracle
    return {
        "localization_error_m": error,
        "oracle_error_m": oracle,
        "excess_error_m": excess,
        "success_0_5m": int(error <= 0.5),
        "success_1_0m": int(error <= 1.0),
        "oracle_normalized_success_0_5m": int(excess <= 0.5),
        "oracle_normalized_success_1_0m": int(excess <= 1.0),
    }


def aggregate(rows: list[dict]) -> dict:
    errors = np.asarray(
        [row["metrics"]["localization_error_m"] for row in rows],
        dtype=np.float64,
    )
    return {
        "query_count": len(rows),
        "mean_localization_error_m": float(errors.mean()),
        "median_localization_error_m": float(np.median(errors)),
        "p90_localization_error_m": float(np.quantile(errors, 0.9)),
        "success_rate_at_0_5m": float(
            np.mean([row["metrics"]["success_0_5m"] for row in rows])
        ),
        "success_rate_at_1_0m": float(
            np.mean([row["metrics"]["success_1_0m"] for row in rows])
        ),
        "resolution_aware_success_rate_at_0_5m": float(
            np.mean(
                [
                    row["metrics"]["oracle_normalized_success_0_5m"]
                    for row in rows
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--oversized-dir", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    if selection.get("sha256") != SELECTION_SHA256:
        raise RuntimeError("unexpected frozen 112-query selection")
    selected = {int(row["index"]): row for row in selection["records"]}
    if len(selected) != 112:
        raise RuntimeError(f"expected 112 selected queries, got {len(selected)}")

    primary = load_local_results(args.primary_dir)
    oversized = load_local_results(args.oversized_dir)
    external = json.loads(args.external_summary.read_text())
    if not external.get("accepted_for_fem_omp_112_aggregation"):
        raise RuntimeError("external results have not passed the merge gate")
    if external.get("method_identity", {}).get("method_id") != METHOD:
        raise RuntimeError("external method mismatch")
    external_errors = {
        int(row["query_index"]): row
        for row in external["per_query_localization_error_m"]
    }

    rows = []
    replication_checks = []
    missing = []
    for index, selected_row in selected.items():
        local = primary.get(index) or oversized.get(index)
        external_row = external_errors.get(index)
        if local is not None:
            payload, path = local
            for key in ("query_id", "room", "candidate_count"):
                if payload[key] != selected_row[key]:
                    raise RuntimeError(f"query {index} identity mismatch: {key}")
            metrics = {
                key: payload["metrics"][key]
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
            source_kind = (
                "local_primary_97" if index in primary else "local_oversized"
            )
            source_record = {
                "kind": source_kind,
                "result_file": str(path.resolve()),
                "result_sha256": payload["sha256"],
            }
            if external_row is not None:
                reported = float(external_row[METHOD])
                local_error = float(metrics["localization_error_m"])
                if round(local_error, 3) != round(reported, 3):
                    raise RuntimeError(
                        f"query {index} local/external error mismatch: "
                        f"{local_error} versus {reported}"
                    )
                replication_checks.append(
                    {
                        "query_index": index,
                        "local_error_m": local_error,
                        "external_reported_error_m": reported,
                        "matches_to_reported_precision": True,
                    }
                )
        elif external_row is not None:
            if external_row["room"] != selected_row["room"]:
                raise RuntimeError(f"external query {index} room mismatch")
            metrics = metric_from_error(
                float(external_row[METHOD]), float(selected_row["oracle_m"])
            )
            source_record = {
                "kind": "external_verified",
                "external_summary": str(args.external_summary.resolve()),
                "verification_status": external["verification_status"],
            }
        else:
            missing.append(index)
            continue

        rows.append(
            {
                "query_index": index,
                "query_id": selected_row["query_id"],
                "room": selected_row["room"],
                "candidate_count": int(selected_row["candidate_count"]),
                "candidate_indices_sha256": selected_row[
                    "candidate_indices_sha256"
                ],
                "metrics": metrics,
                "source": source_record,
            }
        )

    metrics_by_room = defaultdict(list)
    for row in rows:
        metrics_by_room[row["room"]].append(row)
    per_room = {
        room: aggregate(room_rows)
        for room, room_rows in sorted(metrics_by_room.items())
    }
    room_macro = None
    if not missing:
        room_macro = {
            "room_count": len(per_room),
            "mean_localization_error_m": float(
                np.mean(
                    [value["mean_localization_error_m"] for value in per_room.values()]
                )
            ),
            "success_rate_at_0_5m": float(
                np.mean([value["success_rate_at_0_5m"] for value in per_room.values()])
            ),
            "success_rate_at_1_0m": float(
                np.mean([value["success_rate_at_1_0m"] for value in per_room.values()])
            ),
            "resolution_aware_success_rate_at_0_5m": float(
                np.mean(
                    [
                        value["resolution_aware_success_rate_at_0_5m"]
                        for value in per_room.values()
                    ]
                )
            ),
        }

    payload = {
        "schema_version": 1,
        "method": METHOD,
        "scope": "frozen strict-coverage 16-room 112-query secondary table",
        "completion_status": "complete" if not missing else "incomplete",
        "selection_internal_sha256": selection["sha256"],
        "selection_file_sha256": file_sha256(args.selection),
        "expected_query_count": 112,
        "included_query_count": len(rows),
        "missing_query_indices": missing,
        "source_counts": dict(
            sorted(Counter(row["source"]["kind"] for row in rows).items())
        ),
        "accepted_external_query_indices": sorted(external_errors),
        "external_only_query_indices": sorted(
            index for index in external_errors if index not in oversized
        ),
        "replication_checks": replication_checks,
        "metrics": aggregate(rows),
        "room_macro_metrics": room_macro,
        "per_room": per_room,
        "per_query": rows,
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_json(args.output_json, payload)

    metric = payload["metrics"]
    lines = [
        "# Merged 112-query FEM--OMP result",
        "",
        f"Status: **{payload['completion_status'].upper()}** "
        f"(`{len(rows)}/112` queries).",
        "",
        "The merge uses the frozen 112-query selection. External-server queries "
        "retain explicit provenance and are counted once; query `715` is a "
        "cross-server replication check whose local artifact is canonical.",
        "",
    ]
    if missing:
        lines.extend(
            [
                f"Pending queries: `{', '.join(map(str, missing))}`.",
                "",
                "The aggregate below is provisional until those queries complete.",
                "",
            ]
        )
    lines.extend(
        [
            "| Model | Mean [m] ↓ | Median [m] ↓ | P90 [m] ↓ | SR@0.5m ↑ | SR@1.0m ↑ | Resolution-Aware SR@0.5m ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| FEM-Sabine + Room-Helps OMP (Depth-AABB) | "
            f"{metric['mean_localization_error_m']:.3f} | "
            f"{metric['median_localization_error_m']:.3f} | "
            f"{metric['p90_localization_error_m']:.3f} | "
            f"{100 * metric['success_rate_at_0_5m']:.1f}% | "
            f"{100 * metric['success_rate_at_1_0m']:.1f}% | "
            f"{100 * metric['resolution_aware_success_rate_at_0_5m']:.1f}% |",
            "",
            "## Source accounting",
            "",
        ]
    )
    for source, count in payload["source_counts"].items():
        lines.append(f"- `{source}`: {count} queries")
    lines.extend(
        [
            "",
            "Accepted external query IDs: "
            + ", ".join(f"`{index}`" for index in sorted(external_errors))
            + ".",
            "",
            "This is a FEM--OMP result, not a FEM--AGREE result.",
            "",
        ]
    )
    atomic_text(args.output_md, "\n".join(lines))
    print(
        json.dumps(
            {
                "completion_status": payload["completion_status"],
                "included_query_count": len(rows),
                "missing_query_indices": missing,
                "metrics": metric,
                "source_counts": payload["source_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
