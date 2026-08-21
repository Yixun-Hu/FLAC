#!/usr/bin/env python3
"""Audit official meshes and count exp_09 candidate/query work exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.ar_queries import load_context_manifest
from src.localization.geometry import (
    EPSILON_METERS,
    build_lattice,
    choose_z_band_branch,
    classify_mesh_candidates,
    filter_query_candidates,
    grid_oracle_error,
    load_raycast_scene,
)


def _sha_indices(mask: np.ndarray) -> str:
    indices = np.flatnonzero(mask).astype("<u4", copy=False)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _summary(values, threshold=0.5):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "over_0_5m": int(np.count_nonzero(values > threshold)),
    }


def audit(
    context_manifest: Path,
    mesh_root: Path,
    *,
    continue_on_anchor_failure: bool = False,
    compute_topology: bool = True,
) -> dict:
    manifest = load_context_manifest(context_manifest)
    records = manifest["records"]
    missing = [record for record in records if record["room"] == "ListeningRoom_idx_2"]
    kept = [record for record in records if record["room"] != "ListeningRoom_idx_2"]
    if len(records) != 6337 or len(missing) != 1000 or len(kept) != 5337:
        raise RuntimeError("context manifest does not match the approved 6,337 -> 5,337 scope")
    missing_mesh = mesh_root / "ListeningRoom" / "ListeningRoom_idx_2.obj"
    if missing_mesh.exists():
        raise RuntimeError("previously missing ListeningRoom_idx_2 mesh now exists; re-review scope")

    grouped = defaultdict(list)
    for record in kept:
        grouped[(record["scene"], record["room"])].append(record)
    if len(grouped) != 16:
        raise RuntimeError(f"expected 16 included rooms, got {len(grouped)}")

    room_results = {}
    query_results = []
    total_raw_pairs = total_base_pairs = 0
    total_full_pairs = total_z_pairs = 0
    total_full_receiver_pairs = total_z_receiver_pairs = 0
    source_anchor_failures = []
    receiver_anchor_failures = []

    for (scene_name, room_name), room_records in sorted(grouped.items()):
        mesh_path = mesh_root / scene_name / f"{room_name}.obj"
        mesh = load_raycast_scene(mesh_path, compute_topology=compute_topology)
        raw_points = build_lattice(mesh.aabb_min, mesh.aabb_max, 0.5)
        base_mask, distances = classify_mesh_candidates(mesh, raw_points, 0.5)
        base_points = raw_points[base_mask]
        if len(base_points) == 0:
            raise RuntimeError(f"empty base grid: {room_name}")

        sources = np.unique(np.asarray([record["source_global"] for record in room_records]), axis=0)
        receivers = np.unique(np.asarray([record["receiver_global"] for record in room_records]), axis=0)
        source_mask, source_distance = classify_mesh_candidates(mesh, sources, 0.5)
        # Receivers must be finite and inside the free-space classification,
        # but they are not candidate source anchors and need not themselves be
        # 0.5 m from a surface. Candidate points receive a separate 0.5 m
        # receiver-clearance mask below.
        receiver_mask, receiver_distance = classify_mesh_candidates(mesh, receivers, 0.0)
        if not np.all(source_mask):
            failed_points = sources[~source_mask]
            source_anchor_failures.append(
                {
                    "room": room_name,
                    "count": int(len(failed_points)),
                    "coordinates": failed_points.tolist(),
                    "distances_m": source_distance[~source_mask].astype(float).tolist(),
                }
            )
            if not continue_on_anchor_failure:
                raise RuntimeError(
                    f"real source anchors fail mesh predicate in {room_name}: "
                    f"{failed_points[:10].tolist()}"
                )
        if not np.all(receiver_mask):
            failed_points = receivers[~receiver_mask]
            receiver_anchor_failures.append(
                {
                    "room": room_name,
                    "count": int(len(failed_points)),
                    "coordinates": failed_points.tolist(),
                    "distances_m": receiver_distance[~receiver_mask].astype(float).tolist(),
                }
            )
            if not continue_on_anchor_failure:
                raise RuntimeError(
                    f"real receiver anchors fail inside predicate in {room_name}: "
                    f"{failed_points[:10].tolist()}"
                )

        union_full: dict[str, np.ndarray] = {}
        union_z: dict[str, np.ndarray] = {}
        room_full_errors, room_z_errors = [], []
        room_full_counts, room_z_counts = [], []
        for record in room_records:
            receiver = np.asarray(record["receiver_global"], dtype=np.float64)
            contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
            truth = np.asarray(record["source_global"], dtype=np.float64)
            full_mask = filter_query_candidates(base_points, receiver, contexts)
            z_band = (float(contexts[:, 2].min() - 0.5), float(contexts[:, 2].max() + 0.5))
            z_mask = filter_query_candidates(base_points, receiver, contexts, z_band=z_band)
            full_count, z_count = int(full_mask.sum()), int(z_mask.sum())
            if full_count == 0:
                raise RuntimeError(f"empty full-height query grid: {record['query_id']}")
            full_error = grid_oracle_error(base_points[full_mask], truth)
            z_error = grid_oracle_error(base_points[z_mask], truth) if z_count else float("inf")
            receiver_id = record["filename"].split("_")[1]
            union_full.setdefault(receiver_id, np.zeros(len(base_points), dtype=bool))
            union_z.setdefault(receiver_id, np.zeros(len(base_points), dtype=bool))
            union_full[receiver_id] |= full_mask
            union_z[receiver_id] |= z_mask
            room_full_counts.append(full_count)
            room_z_counts.append(z_count)
            room_full_errors.append(full_error)
            room_z_errors.append(z_error)
            query_results.append(
                {
                    "index": record["index"],
                    "query_id": record["query_id"],
                    "scene": scene_name,
                    "room": room_name,
                    "receiver_id": receiver_id,
                    "full_count": full_count,
                    "z_count": z_count,
                    "full_oracle_m": full_error,
                    "z_oracle_m": z_error if np.isfinite(z_error) else None,
                    "z_band": list(z_band),
                    "full_indices_sha256": _sha_indices(full_mask),
                    "z_indices_sha256": _sha_indices(z_mask),
                }
            )

        full_receiver_pairs = sum(int(mask.sum()) for mask in union_full.values())
        z_receiver_pairs = sum(int(mask.sum()) for mask in union_z.values())
        total_raw_pairs += len(raw_points) * len(room_records)
        total_base_pairs += len(base_points) * len(room_records)
        total_full_pairs += sum(room_full_counts)
        total_z_pairs += sum(room_z_counts)
        total_full_receiver_pairs += full_receiver_pairs
        total_z_receiver_pairs += z_receiver_pairs
        room_results[room_name] = {
            "scene": scene_name,
            "query_count": len(room_records),
            "mesh_path": str(mesh.path),
            "mesh_sha256": mesh.sha256,
            "vertices": mesh.vertex_count,
            "triangles": mesh.triangle_count,
            "diagnostics": mesh.diagnostics,
            "aabb_min": mesh.aabb_min.tolist(),
            "aabb_max": mesh.aabb_max.tolist(),
            "raw_lattice_count": len(raw_points),
            "base_valid_count": len(base_points),
            "base_points_sha256": hashlib.sha256(base_points.astype("<f8").tobytes()).hexdigest(),
            "base_distance_min_m": float(distances[base_mask].min()),
            "source_anchor_min_distance_m": float(source_distance.min()),
            "receiver_anchor_min_distance_m": float(receiver_distance.min()),
            "source_anchor_survival": int(source_mask.sum()),
            "receiver_anchor_survival": int(receiver_mask.sum()),
            "full_candidate_counts": _summary(room_full_counts),
            "z_candidate_counts": _summary(room_z_counts),
            "full_oracle": _summary(room_full_errors),
            "z_oracle": _summary(room_z_errors),
            "full_receiver_candidate_pairs": full_receiver_pairs,
            "z_receiver_candidate_pairs": z_receiver_pairs,
        }

    query_results.sort(key=lambda value: value["index"])
    full_errors = [record["full_oracle_m"] for record in query_results]
    z_errors = [record["z_oracle_m"] if record["z_oracle_m"] is not None else np.inf for record in query_results]
    z_counts = [record["z_count"] for record in query_results]
    chosen = choose_z_band_branch(full_errors, z_errors, z_counts)
    chosen_pairs = total_z_pairs if chosen == "z_band" else total_full_pairs
    chosen_receiver_pairs = total_z_receiver_pairs if chosen == "z_band" else total_full_receiver_pairs
    chosen_errors = z_errors if chosen == "z_band" else full_errors
    for record in query_results:
        record["chosen_branch"] = chosen
        record["chosen_count"] = record["z_count"] if chosen == "z_band" else record["full_count"]

    result = {
        "schema_version": 1,
        "context_manifest": str(context_manifest),
        "context_manifest_sha256": manifest["sha256"],
        "mesh_root": str(mesh_root.resolve()),
        "excluded_room": "ListeningRoom_idx_2",
        "excluded_queries": 1000,
        "included_queries": 5337,
        "included_rooms": 16,
        "grid_spacing_m": [0.5, 0.5, 0.5],
        "surface_clearance_m": 0.5,
        "receiver_clearance_m": 0.5,
        "context_clearance_m": 0.25,
        "epsilon_m": EPSILON_METERS,
        "ground_truth_inserted": False,
        "topology_computed": compute_topology,
        "geometry_gate": "FAIL" if source_anchor_failures or receiver_anchor_failures else "PASS",
        "source_anchor_failures": source_anchor_failures,
        "receiver_anchor_failures": receiver_anchor_failures,
        "z_branch": chosen,
        "rooms": room_results,
        "queries": query_results,
        "totals": {
            "raw_candidate_query_pairs": total_raw_pairs,
            "base_candidate_query_pairs": total_base_pairs,
            "full_candidate_query_pairs": total_full_pairs,
            "z_candidate_query_pairs": total_z_pairs,
            "chosen_candidate_query_pairs": chosen_pairs,
            "full_receiver_candidate_pairs": total_full_receiver_pairs,
            "z_receiver_candidate_pairs": total_z_receiver_pairs,
            "chosen_receiver_candidate_pairs": chosen_receiver_pairs,
            "context_branch_queries": 5337,
            "context_vit_forwards": 5337 * 8,
            "chosen_oracle": _summary(chosen_errors),
        },
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def _markdown(result: dict) -> str:
    totals = result["totals"]
    lines = [
        "# Exp_09 geometry and cost preflight",
        "",
        f"Audit SHA-256: `{result['sha256']}`. Context manifest: `{result['context_manifest_sha256']}`.",
        "",
        f"Chosen geometry branch: **{result['z_branch']}**. Included: 5,337 queries / 16 rooms; excluded: 1,000 `ListeningRoom_idx_2` queries (missing official mesh).",
        f"Geometry gate: **{result['geometry_gate']}**; failing unique source anchors: **{sum(item['count'] for item in result['source_anchor_failures'])}** across **{len(result['source_anchor_failures'])}** rooms. A failed gate forbids generation; counts below are diagnostic.",
        f"Failing unique receiver inside anchors: **{sum(item['count'] for item in result['receiver_anchor_failures'])}** across **{len(result['receiver_anchor_failures'])}** rooms.",
        "",
        "| Room | Queries | Raw grid | Base valid | Full pairs | Z pairs | Full oracle >0.5 m | Z oracle >0.5 m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for room, value in sorted(result["rooms"].items()):
        lines.append(
            f"| {room} | {value['query_count']} | {value['raw_lattice_count']} | "
            f"{value['base_valid_count']} | {value['full_candidate_counts']['mean'] * value['query_count']:.0f} | "
            f"{value['z_candidate_counts']['mean'] * value['query_count']:.0f} | "
            f"{value['full_oracle']['over_0_5m']} | {value['z_oracle']['over_0_5m']} |"
        )
    lines += [
        "",
        "## Exact totals",
        "",
        f"- Raw AABB candidate-query pairs: `{totals['raw_candidate_query_pairs']:,}`.",
        f"- Mesh-valid base candidate-query pairs: `{totals['base_candidate_query_pairs']:,}`.",
        f"- Full-height query-valid pairs: `{totals['full_candidate_query_pairs']:,}`.",
        f"- Context-z query-valid pairs: `{totals['z_candidate_query_pairs']:,}`.",
        f"- Chosen query-valid pairs: `{totals['chosen_candidate_query_pairs']:,}`.",
        f"- Chosen unique receiver-candidate source branches: `{totals['chosen_receiver_candidate_pairs']:,}`.",
        f"- Context cache work: `{totals['context_branch_queries']:,}` queries / `{totals['context_vit_forwards']:,}` context ViT forwards.",
        f"- Chosen oracle >0.5 m: `{totals['chosen_oracle']['over_0_5m']}`; nonempty finite: `{totals['chosen_oracle']['finite_count']}/{totals['chosen_oracle']['count']}`.",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--diagnostic-continue-on-anchor-failure", action="store_true")
    parser.add_argument("--fast-cost-audit", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.context_manifest,
        args.mesh_root,
        continue_on_anchor_failure=args.diagnostic_continue_on_anchor_failure,
        compute_topology=not args.fast_cost_audit,
    )
    _atomic_write(args.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_write(args.output_md, _markdown(result))
    print(
        json.dumps(
            {
                "sha256": result["sha256"],
                "geometry_gate": result["geometry_gate"],
                "z_branch": result["z_branch"],
                **result["totals"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
