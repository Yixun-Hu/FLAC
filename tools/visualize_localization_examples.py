#!/usr/bin/env python3
"""Render deterministic 3-D examples from the two exp_09 pilot batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


K_VALUES = ("1", "4", "8")
CATEGORY_ORDER = (
    "both_good",
    "typical",
    "vanilla_advantage",
    "fa_bf_advantage",
)
CATEGORY_LABELS = {
    "both_good": "Both models accurate",
    "typical": "Typical combined error",
    "vanilla_advantage": "Largest Vanilla advantage",
    "fa_bf_advantage": "Largest FA-BF advantage",
}


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt SHA-256 in {path}")
    return payload


def load_arm_results(arm_dir: Path, expected_ids: set[str]) -> dict[str, dict]:
    paths = sorted((arm_dir / "queries").glob("query_*.json"))
    results = {}
    for path in paths:
        result = load_hashed_json(path)
        query_id = result["query_id"]
        if query_id in results:
            raise RuntimeError(f"duplicate query result in {arm_dir}: {query_id}")
        results[query_id] = result
    if set(results) != expected_ids:
        missing = sorted(expected_ids - set(results))
        extra = sorted(set(results) - expected_ids)
        raise RuntimeError(
            f"result coverage mismatch in {arm_dir}: missing={missing[:3]}, extra={extra[:3]}"
        )
    return results


def validate_pair(vanilla: dict, fa_bf: dict) -> None:
    shared_fields = (
        "query_id",
        "query_index",
        "room",
        "scene",
        "source_global",
        "receiver_global",
        "candidate_indices_sha256",
        "candidate_count",
    )
    for field in shared_fields:
        if vanilla[field] != fa_bf[field]:
            raise RuntimeError(f"arm mismatch for {vanilla['query_id']}: {field}")
    for k_gen in K_VALUES:
        if vanilla["metrics"][k_gen]["oracle_error_m"] != fa_bf["metrics"][k_gen][
            "oracle_error_m"
        ]:
            raise RuntimeError(f"oracle mismatch for {vanilla['query_id']} at K={k_gen}")


def load_combined_results(experiment_dir: Path) -> tuple[list[dict], list[str]]:
    batches = (
        (
            "batch1",
            experiment_dir / "pilot_manifest_seed42_4_per_room.json",
            experiment_dir / "pilot_results",
        ),
        (
            "batch2",
            experiment_dir / "pilot_manifest_seed43_batch2_4_per_room.json",
            experiment_dir / "pilot_results_batch2",
        ),
    )
    rows = []
    pilot_hashes = []
    observed_ids: set[str] = set()
    for batch_name, manifest_path, result_root in batches:
        manifest = load_hashed_json(manifest_path)
        pilot_hashes.append(manifest["sha256"])
        expected_ids = {item["query_id"] for item in manifest["records"]}
        overlap = observed_ids.intersection(expected_ids)
        if overlap:
            raise RuntimeError(f"pilot target overlap detected: {sorted(overlap)[:3]}")
        observed_ids.update(expected_ids)
        vanilla = load_arm_results(result_root / "vanilla", expected_ids)
        fa_bf = load_arm_results(result_root / "fa_bf", expected_ids)
        for query_id in sorted(expected_ids):
            validate_pair(vanilla[query_id], fa_bf[query_id])
            rows.append(
                {
                    "batch": batch_name,
                    "query_id": query_id,
                    "vanilla": vanilla[query_id],
                    "fa_bf": fa_bf[query_id],
                }
            )
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 non-overlapping targets, got {len(rows)}")
    return rows, pilot_hashes


def case_values(row: dict, k_gen: str) -> tuple[float, float, float]:
    vanilla_error = float(row["vanilla"]["metrics"][k_gen]["localization_error_m"])
    fa_error = float(row["fa_bf"]["metrics"][k_gen]["localization_error_m"])
    return vanilla_error, fa_error, 0.5 * (vanilla_error + fa_error)


def select_cases(rows: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """Select four informative cases per K while keeping all 12 targets distinct."""

    selected: dict[str, list[tuple[str, dict]]] = {}
    globally_used: set[str] = set()
    for k_gen in K_VALUES:
        available = [row for row in rows if row["query_id"] not in globally_used]
        combined_errors = np.asarray(
            [case_values(row, k_gen)[2] for row in available], dtype=np.float64
        )
        median_error = float(np.median(combined_errors))

        choices: dict[str, dict] = {}
        local_used: set[str] = set()

        def remaining() -> list[dict]:
            return [row for row in available if row["query_id"] not in local_used]

        vanilla_win = min(
            remaining(),
            key=lambda row: (
                -(case_values(row, k_gen)[1] - case_values(row, k_gen)[0]),
                row["query_id"],
            ),
        )
        choices["vanilla_advantage"] = vanilla_win
        local_used.add(vanilla_win["query_id"])

        fa_win = min(
            remaining(),
            key=lambda row: (
                -(case_values(row, k_gen)[0] - case_values(row, k_gen)[1]),
                row["query_id"],
            ),
        )
        choices["fa_bf_advantage"] = fa_win
        local_used.add(fa_win["query_id"])

        both_good = min(
            remaining(),
            key=lambda row: (
                max(case_values(row, k_gen)[:2]),
                case_values(row, k_gen)[2],
                row["query_id"],
            ),
        )
        choices["both_good"] = both_good
        local_used.add(both_good["query_id"])

        typical = min(
            remaining(),
            key=lambda row: (
                abs(case_values(row, k_gen)[2] - median_error),
                row["query_id"],
            ),
        )
        choices["typical"] = typical
        local_used.add(typical["query_id"])

        selected[k_gen] = [(category, choices[category]) for category in CATEGORY_ORDER]
        globally_used.update(local_used)
    if len(globally_used) != 12:
        raise RuntimeError("case selection did not produce 12 unique targets")
    return selected


def load_visual_mesh(mesh_path: Path, target_triangles: int):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"mesh has no drawable geometry: {mesh_path}")
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"mesh has non-finite vertices: {mesh_path}")
    return vertices, triangles


def set_equal_3d_axes(ax, vertices: np.ndarray) -> None:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = np.maximum(high - low, 0.1)
    padding = 0.06 * extent
    ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
    ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    ax.set_zlim(max(0.0, low[2] - padding[2]), high[2] + padding[2])
    ax.set_box_aspect(extent)
    ax.view_init(elev=27, azim=-58)
    ax.set_xlabel("x (m)", labelpad=2)
    ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("z (m)", labelpad=2)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_proj_type("ortho")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.62, 0.64, 0.66, 0.45))


def draw_case(ax, row: dict, k_gen: str, category: str, vertices, triangles) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    result = row["vanilla"]
    gt = np.asarray(result["source_global"], dtype=np.float64)
    receiver = np.asarray(result["receiver_global"], dtype=np.float64)
    vanilla_metric = result["metrics"][k_gen]
    fa_metric = row["fa_bf"]["metrics"][k_gen]
    vanilla_prediction = np.asarray(vanilla_metric["prediction_global"], dtype=np.float64)
    fa_prediction = np.asarray(fa_metric["prediction_global"], dtype=np.float64)

    surface = Poly3DCollection(
        vertices[triangles],
        facecolors=(0.66, 0.70, 0.74, 0.045),
        edgecolors=(0.25, 0.28, 0.31, 0.16),
        linewidths=0.13,
    )
    surface.set_rasterized(True)
    ax.add_collection3d(surface)

    ax.plot(*np.stack([gt, vanilla_prediction]).T, color="#e68613", lw=1.5, ls="--")
    ax.plot(*np.stack([gt, fa_prediction]).T, color="#1478b8", lw=1.5, ls=":")
    ax.scatter(
        *receiver,
        marker="^",
        s=42,
        c="#777777",
        edgecolors="white",
        linewidths=0.5,
        depthshade=False,
        label="Receiver",
    )
    ax.scatter(
        *vanilla_prediction,
        marker="o",
        s=105,
        facecolors="none",
        edgecolors="#e68613",
        linewidths=2.2,
        depthshade=False,
        label="Vanilla prediction",
    )
    ax.scatter(
        *fa_prediction,
        marker="x",
        s=100,
        c="#1478b8",
        linewidths=2.2,
        depthshade=False,
        label="FA-BF prediction",
    )
    ax.scatter(
        *gt,
        marker="*",
        s=180,
        c="#3bb44a",
        edgecolors="#17231a",
        linewidths=0.8,
        depthshade=False,
        label="Ground truth",
    )

    source_receiver = Path(result["query_id"]).stem.replace("_hybrid_IR", "")
    ax.set_title(
        f"{CATEGORY_LABELS[category]}\n"
        f"{result['room']} · {source_receiver}\n"
        f"Vanilla {vanilla_metric['localization_error_m']:.2f} m | "
        f"FA-BF {fa_metric['localization_error_m']:.2f} m",
        fontsize=9,
        pad=7,
    )
    set_equal_3d_axes(ax, vertices)


def build_case_record(
    row: dict, k_gen: str, category: str, mesh_entry: dict
) -> dict:
    vanilla = row["vanilla"]
    fa_bf = row["fa_bf"]
    vanilla_metric = vanilla["metrics"][k_gen]
    fa_metric = fa_bf["metrics"][k_gen]
    truth = np.asarray(vanilla["source_global"], dtype=np.float64)
    vanilla_prediction = np.asarray(
        vanilla_metric["prediction_global"], dtype=np.float64
    )
    fa_prediction = np.asarray(fa_metric["prediction_global"], dtype=np.float64)
    coordinates = np.stack([truth, vanilla_prediction, fa_prediction])
    if not np.isfinite(coordinates).all():
        raise RuntimeError(f"non-finite visualization coordinate: {row['query_id']}")
    expected_errors = (
        float(np.linalg.norm(vanilla_prediction - truth)),
        float(np.linalg.norm(fa_prediction - truth)),
    )
    recorded_errors = (
        float(vanilla_metric["localization_error_m"]),
        float(fa_metric["localization_error_m"]),
    )
    if not np.allclose(expected_errors, recorded_errors, atol=1e-9, rtol=0.0):
        raise RuntimeError(f"coordinate/error mismatch: {row['query_id']} at K={k_gen}")
    aabb_min = np.asarray(mesh_entry["aabb_min"], dtype=np.float64)
    aabb_max = np.asarray(mesh_entry["aabb_max"], dtype=np.float64)
    if np.any(coordinates < aabb_min - 1e-4) or np.any(coordinates > aabb_max + 1e-4):
        raise RuntimeError(f"plotted point falls outside audited mesh AABB: {row['query_id']}")
    return {
        "k_gen": int(k_gen),
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "batch": row["batch"],
        "query_index": int(vanilla["query_index"]),
        "query_id": row["query_id"],
        "scene": vanilla["scene"],
        "room": vanilla["room"],
        "ground_truth_global_m": vanilla["source_global"],
        "receiver_global_m": vanilla["receiver_global"],
        "vanilla_prediction_global_m": vanilla_metric["prediction_global"],
        "vanilla_error_m": float(vanilla_metric["localization_error_m"]),
        "fa_bf_prediction_global_m": fa_metric["prediction_global"],
        "fa_bf_error_m": float(fa_metric["localization_error_m"]),
        "oracle_error_m": float(vanilla_metric["oracle_error_m"]),
        "candidate_count": int(vanilla["candidate_count"]),
        "mesh_path": mesh_entry["mesh_path"],
        "mesh_sha256": mesh_entry["mesh_sha256"],
        "vanilla_result_sha256": vanilla["sha256"],
        "fa_bf_result_sha256": fa_bf["sha256"],
    }


def render_markdown(manifest: dict) -> str:
    lines = [
        "# Exp_09 localization prediction examples",
        "",
        "The 12 targets below were selected deterministically from the two completed, "
        "non-overlapping 64-query pilot batches. Selection uses no manual cherry-picking: "
        "for each `K_gen`, it shows a jointly accurate case, a typical combined-error "
        "case, the largest remaining Vanilla advantage, and the largest remaining FA-BF "
        "advantage. Targets are not reused across panels.",
        "",
        "Markers: ground truth = green star; Vanilla = orange open circle; FA-BF = blue "
        "cross; receiver = gray triangle. Dashed/dotted segments show localization error.",
        "",
    ]
    for k_gen in (1, 4, 8):
        lines.extend(
            [
                f"## K_gen = {k_gen}",
                "",
                f"![K_gen={k_gen} localization cases](localization_examples_k{k_gen}.png)",
                "",
                "| Case | Batch | Room / target | Vanilla error | FA-BF error |",
                "|---|---|---|---:|---:|",
            ]
        )
        for case in [item for item in manifest["cases"] if item["k_gen"] == k_gen]:
            target = Path(case["query_id"]).stem.replace("_hybrid_IR", "")
            lines.append(
                f"| {case['category_label']} | {case['batch']} | "
                f"{case['room']} / {target} | {case['vanilla_error_m']:.3f} m | "
                f"{case['fa_bf_error_m']:.3f} m |"
            )
        lines.append("")
    lines.extend(
        [
            "The translucent geometry is a display-only decimation of the audited official "
            "OBJ. All markers use the untouched AcousticRooms global coordinates, and errors "
            "come directly from the hash-validated result JSON files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-triangles", type=int, default=6000)
    parser.add_argument("--dpi", type=int, default=210)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(experiment_dir)
    except ValueError as error:
        raise ValueError("visualization outputs must remain inside the experiment directory") from error
    if args.mesh_triangles < 500:
        raise ValueError("mesh-triangles must be at least 500")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, pilot_hashes = load_combined_results(experiment_dir)
    selected = select_cases(rows)
    geometry_audit = load_hashed_json(experiment_dir / "geometry_audit.json")
    room_geometry = geometry_audit["rooms"]
    mesh_cache = {}
    case_records = []

    for k_gen in K_VALUES:
        figure = plt.figure(figsize=(16, 12.5), constrained_layout=True)
        axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
        for ax, (category, row) in zip(axes, selected[k_gen]):
            room = row["vanilla"]["room"]
            if room not in room_geometry:
                raise RuntimeError(f"missing audited geometry for {room}")
            mesh_entry = room_geometry[room]
            mesh_path = Path(mesh_entry["mesh_path"])
            if file_sha256(mesh_path) != mesh_entry["mesh_sha256"]:
                raise RuntimeError(f"official mesh hash changed: {mesh_path}")
            if room not in mesh_cache:
                mesh_cache[room] = load_visual_mesh(mesh_path, args.mesh_triangles)
            vertices, triangles = mesh_cache[room]
            draw_case(ax, row, k_gen, category, vertices, triangles)
            case_records.append(build_case_record(row, k_gen, category, mesh_entry))
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=4,
            frameon=False,
            fontsize=10,
        )
        figure.suptitle(
            f"Exp_09 localization predictions · K_gen={k_gen}",
            fontsize=17,
            fontweight="bold",
        )
        figure.savefig(
            output_dir / f"localization_examples_k{k_gen}.png",
            dpi=args.dpi,
            facecolor="white",
        )
        plt.close(figure)

    payload = {
        "schema_version": 1,
        "source_pilot_sha256": pilot_hashes,
        "combined_target_count": len(rows),
        "target_overlap_between_batches": 0,
        "selection_policy": {
            "categories": list(CATEGORY_ORDER),
            "unique_targets_across_all_panels": True,
            "manual_selection": False,
        },
        "mesh_visualization": {
            "source": "audited official AcousticRooms OBJ",
            "decimation_target_triangles": args.mesh_triangles,
            "decimation_affects_coordinates_or_metrics": False,
        },
        "cases": case_records,
    }
    payload["sha256"] = canonical_sha256(payload)
    (output_dir / "selected_cases.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "sha256": payload["sha256"],
                "cases": len(case_records),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
