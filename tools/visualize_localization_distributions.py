#!/usr/bin/env python3
"""Plot 16 paired FA-BF/Vanilla predictions per K across four rooms."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from visualize_localization_examples import (  # noqa: E402
    K_VALUES,
    canonical_sha256,
    file_sha256,
    load_combined_results,
    load_hashed_json,
    load_visual_mesh,
    set_equal_3d_axes,
)


def select_cross_scale_rooms(rows: list[dict]) -> list[dict]:
    """Select fixed batch-1 targets from four candidate-count ranks."""

    batch1 = [row for row in rows if row["batch"] == "batch1"]
    grouped: dict[str, list[dict]] = {}
    for row in batch1:
        grouped.setdefault(row["vanilla"]["room"], []).append(row)
    if len(grouped) != 16:
        raise RuntimeError(f"expected 16 batch-1 rooms, got {len(grouped)}")
    ranked = sorted(
        (
            statistics.median(
                int(row["vanilla"]["candidate_count"]) for row in room_rows
            ),
            room,
            sorted(room_rows, key=lambda row: int(row["vanilla"]["query_index"])),
        )
        for room, room_rows in grouped.items()
    )
    ranks = np.rint(np.linspace(0, len(ranked) - 1, 4)).astype(int).tolist()
    selected = []
    for rank in ranks:
        median_candidates, room, room_rows = ranked[rank]
        if len(room_rows) != 4:
            raise RuntimeError(f"expected four frozen batch-1 targets in {room}")
        selected.append(
            {
                "candidate_count_rank": int(rank),
                "median_candidate_count": float(median_candidates),
                "room": room,
                "rows": room_rows,
            }
        )
    return selected


def validate_prediction_coordinates(row: dict, k_gen: str, mesh_entry: dict) -> None:
    truth = np.asarray(row["vanilla"]["source_global"], dtype=np.float64)
    vanilla_metric = row["vanilla"]["metrics"][k_gen]
    fa_metric = row["fa_bf"]["metrics"][k_gen]
    vanilla_prediction = np.asarray(
        vanilla_metric["prediction_global"], dtype=np.float64
    )
    fa_prediction = np.asarray(fa_metric["prediction_global"], dtype=np.float64)
    coordinates = np.stack([truth, vanilla_prediction, fa_prediction])
    if not np.isfinite(coordinates).all():
        raise RuntimeError(f"non-finite coordinate for {row['query_id']}")
    computed = (
        float(np.linalg.norm(vanilla_prediction - truth)),
        float(np.linalg.norm(fa_prediction - truth)),
    )
    recorded = (
        float(vanilla_metric["localization_error_m"]),
        float(fa_metric["localization_error_m"]),
    )
    if not np.allclose(computed, recorded, atol=1e-9, rtol=0.0):
        raise RuntimeError(f"coordinate/error mismatch for {row['query_id']} at K={k_gen}")
    aabb_min = np.asarray(mesh_entry["aabb_min"], dtype=np.float64)
    aabb_max = np.asarray(mesh_entry["aabb_max"], dtype=np.float64)
    if np.any(coordinates < aabb_min - 1e-4) or np.any(coordinates > aabb_max + 1e-4):
        raise RuntimeError(f"coordinate outside mesh AABB for {row['query_id']}")


def draw_room_distribution(
    ax,
    room_selection: dict,
    k_gen: str,
    mesh_entry: dict,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    surface = Poly3DCollection(
        vertices[triangles],
        facecolors=(0.66, 0.70, 0.74, 0.04),
        edgecolors=(0.25, 0.28, 0.31, 0.14),
        linewidths=0.12,
    )
    surface.set_rasterized(True)
    ax.add_collection3d(surface)

    vanilla_errors = []
    fa_errors = []
    for target_number, row in enumerate(room_selection["rows"], start=1):
        validate_prediction_coordinates(row, k_gen, mesh_entry)
        vanilla = row["vanilla"]
        gt = np.asarray(vanilla["source_global"], dtype=np.float64)
        receiver = np.asarray(vanilla["receiver_global"], dtype=np.float64)
        vanilla_metric = vanilla["metrics"][k_gen]
        fa_metric = row["fa_bf"]["metrics"][k_gen]
        vanilla_prediction = np.asarray(
            vanilla_metric["prediction_global"], dtype=np.float64
        )
        fa_prediction = np.asarray(fa_metric["prediction_global"], dtype=np.float64)
        vanilla_errors.append(float(vanilla_metric["localization_error_m"]))
        fa_errors.append(float(fa_metric["localization_error_m"]))

        ax.plot(
            *np.stack([gt, vanilla_prediction]).T,
            color="#e68613",
            lw=1.25,
            ls="--",
            alpha=0.78,
        )
        ax.plot(
            *np.stack([gt, fa_prediction]).T,
            color="#1478b8",
            lw=1.25,
            ls=":",
            alpha=0.82,
        )
        ax.scatter(
            *receiver,
            marker="^",
            s=25,
            c="#777777",
            edgecolors="white",
            linewidths=0.4,
            depthshade=False,
            label="Receiver" if target_number == 1 else "_nolegend_",
        )
        ax.scatter(
            *vanilla_prediction,
            marker="o",
            s=72,
            facecolors="none",
            edgecolors="#e68613",
            linewidths=1.9,
            depthshade=False,
            label="Vanilla prediction" if target_number == 1 else "_nolegend_",
        )
        ax.scatter(
            *fa_prediction,
            marker="x",
            s=70,
            c="#1478b8",
            linewidths=1.9,
            depthshade=False,
            label="FA-BF prediction" if target_number == 1 else "_nolegend_",
        )
        ax.scatter(
            *gt,
            marker="*",
            s=125,
            c="#3bb44a",
            edgecolors="#17231a",
            linewidths=0.65,
            depthshade=False,
            label="Ground truth" if target_number == 1 else "_nolegend_",
        )
        ax.text(*gt, f" {target_number}", color="#126b25", fontsize=7, weight="bold")
        ax.text(
            *vanilla_prediction,
            f" V{target_number}",
            color="#a95b00",
            fontsize=6.5,
        )
        ax.text(
            *fa_prediction,
            f" F{target_number}",
            color="#075b91",
            fontsize=6.5,
        )

    ax.set_title(
        f"{room_selection['room']} · 4 targets\n"
        f"median candidates {room_selection['median_candidate_count']:.0f} | "
        f"mean error: Vanilla {np.mean(vanilla_errors):.2f} m, "
        f"FA-BF {np.mean(fa_errors):.2f} m",
        fontsize=9,
        pad=7,
    )
    set_equal_3d_axes(ax, vertices)


def build_manifest(
    selected_rooms: list[dict], geometry_audit: dict, pilot_sha256: str
) -> dict:
    rooms = []
    for selection in selected_rooms:
        mesh_entry = geometry_audit["rooms"][selection["room"]]
        targets = []
        for target_number, row in enumerate(selection["rows"], start=1):
            vanilla = row["vanilla"]
            target = {
                "target_number": target_number,
                "query_index": int(vanilla["query_index"]),
                "query_id": row["query_id"],
                "ground_truth_global_m": vanilla["source_global"],
                "receiver_global_m": vanilla["receiver_global"],
                "candidate_count": int(vanilla["candidate_count"]),
                "predictions": {},
            }
            for k_gen in K_VALUES:
                vanilla_metric = vanilla["metrics"][k_gen]
                fa_metric = row["fa_bf"]["metrics"][k_gen]
                target["predictions"][k_gen] = {
                    "vanilla_global_m": vanilla_metric["prediction_global"],
                    "vanilla_error_m": float(vanilla_metric["localization_error_m"]),
                    "fa_bf_global_m": fa_metric["prediction_global"],
                    "fa_bf_error_m": float(fa_metric["localization_error_m"]),
                }
            targets.append(target)
        rooms.append(
            {
                "room": selection["room"],
                "candidate_count_rank_zero_based": selection["candidate_count_rank"],
                "median_candidate_count": selection["median_candidate_count"],
                "mesh_path": mesh_entry["mesh_path"],
                "mesh_sha256": mesh_entry["mesh_sha256"],
                "targets": targets,
            }
        )
    payload = {
        "schema_version": 1,
        "source_pilot_sha256": pilot_sha256,
        "source_batch": "batch1",
        "room_count": 4,
        "targets_per_room": 4,
        "unique_target_count": 16,
        "predictions_per_model_per_k": 16,
        "k_gen": [1, 4, 8],
        "selection_policy": {
            "room_metric": "median candidate count",
            "room_ranks_from_16": [
                item["candidate_count_rank"] for item in selected_rooms
            ],
            "targets": "all four frozen seed-42 batch1 targets in each selected room",
            "uses_prediction_error": False,
            "same_rooms_and_targets_for_every_k": True,
        },
        "mesh_visualization": {
            "source": "audited official AcousticRooms OBJ",
            "decimation_affects_coordinates_or_metrics": False,
        },
        "rooms": rooms,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def render_markdown(manifest: dict) -> str:
    lines = [
        "# Exp_09 paired prediction distributions",
        "",
        "Each figure uses the same four rooms and the same four frozen batch-1 targets "
        "per room: 16 FA-BF predictions plus their 16 matched Vanilla predictions. Rooms "
        "are selected without using prediction error, at evenly spaced ranks after sorting "
        "the 16 rooms by median candidate count.",
        "",
        "Ground truth is a green star, Vanilla an orange open circle, FA-BF a blue cross, "
        "and the receiver a gray triangle. Labels `1`-`4`, `V1`-`V4`, and `F1`-`F4` identify "
        "matched targets and predictions within each room.",
        "",
    ]
    for k_gen in (1, 4, 8):
        lines.extend(
            [
                f"## K_gen = {k_gen}",
                "",
                f"![K_gen={k_gen} paired distributions](localization_distribution_k{k_gen}.png)",
                "",
                "| Room | Candidate-count rank | Mean Vanilla error | Mean FA-BF error |",
                "|---|---:|---:|---:|",
            ]
        )
        for room in manifest["rooms"]:
            predictions = [target["predictions"][str(k_gen)] for target in room["targets"]]
            vanilla_mean = statistics.mean(item["vanilla_error_m"] for item in predictions)
            fa_mean = statistics.mean(item["fa_bf_error_m"] for item in predictions)
            lines.append(
                f"| {room['room']} | {room['candidate_count_rank_zero_based']} / 15 | "
                f"{vanilla_mean:.3f} m | {fa_mean:.3f} m |"
            )
        lines.append("")
    lines.extend(
        [
            "The translucent room geometry is a display-only decimation of the hash-checked "
            "official OBJ. Marker coordinates and errors are unchanged, hash-validated model "
            "outputs in the global AcousticRooms coordinate system.",
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
    selected_rooms = select_cross_scale_rooms(rows)
    geometry_audit = load_hashed_json(experiment_dir / "geometry_audit.json")
    mesh_cache = {}

    for selection in selected_rooms:
        room = selection["room"]
        mesh_entry = geometry_audit["rooms"][room]
        mesh_path = Path(mesh_entry["mesh_path"])
        if file_sha256(mesh_path) != mesh_entry["mesh_sha256"]:
            raise RuntimeError(f"official mesh hash changed: {mesh_path}")
        mesh_cache[room] = load_visual_mesh(mesh_path, args.mesh_triangles)

    for k_gen in K_VALUES:
        figure = plt.figure(figsize=(16, 12.5), constrained_layout=True)
        axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
        for ax, selection in zip(axes, selected_rooms):
            room = selection["room"]
            vertices, triangles = mesh_cache[room]
            draw_room_distribution(
                ax,
                selection,
                k_gen,
                geometry_audit["rooms"][room],
                vertices,
                triangles,
            )
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
            f"Exp_09 paired prediction distributions · K_gen={k_gen}\n"
            "4 rooms × 4 targets = 16 paired predictions per model",
            fontsize=16,
            fontweight="bold",
        )
        figure.savefig(
            output_dir / f"localization_distribution_k{k_gen}.png",
            dpi=args.dpi,
            facecolor="white",
        )
        plt.close(figure)

    manifest = build_manifest(selected_rooms, geometry_audit, pilot_hashes[0])
    (output_dir / "distribution_cases.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(render_markdown(manifest))
    print(
        json.dumps(
            {
                "sha256": manifest["sha256"],
                "rooms": [item["room"] for item in selected_rooms],
                "targets": manifest["unique_target_count"],
                "predictions_per_model_per_k": manifest[
                    "predictions_per_model_per_k"
                ],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
