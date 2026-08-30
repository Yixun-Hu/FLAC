#!/usr/bin/env python3
"""Compare official room geometry with depth-only AABB FEM reconstructions.

The official OBJ is loaded strictly as a post-hoc visualization reference.  Every
FEM quantity (AABB, candidates, prediction, and scores) comes from an already
completed Depth-AABB result and is never recomputed from the official mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
EXP_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_15_depth_aabb_matched_pilot"
DEFAULT_OUTPUT_DIR = EXP_DIR / "visualizations" / "real_vs_depth_aabb"


@dataclass(frozen=True)
class Case:
    query: int
    label: str
    mesh_path: Path


DIAGNOSTIC_CASES = (
    Case(
        4467,
        "Bathroom14 (compact room)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/Bathrooms/Bathrooms_idx_14.obj"
        ),
    ),
    Case(
        5494,
        "LivingRoom30 (large-error case)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/LivingRoomsWithHallway/"
            "LivingRoomsWithHallway_idx_30.obj"
        ),
    ),
    Case(
        5271,
        "Apartments42 (large-error case)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/Apartments/Apartments_idx_42.obj"
        ),
    ),
)

LOW_ERROR_CASES = (
    Case(
        1204,
        "MeetingRoom20 (error < 0.5 m)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/MeetingRoom/MeetingRoom_idx_20.obj"
        ),
    ),
    Case(
        2188,
        "Bedroom33 (error < 0.5 m)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/Bedrooms/Bedrooms_idx_33.obj"
        ),
    ),
    Case(
        4728,
        "Bathroom18 (error < 0.5 m)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/Bathrooms/Bathrooms_idx_18.obj"
        ),
    ),
    Case(
        6204,
        "Restaurant24 (error < 0.5 m)",
        Path(
            "/home/zhixuanzhao/projects/rir2rir/third_party/AcousticRooms/"
            "room_mesh_obj_format/Restaurants/Restaurants_idx_24.obj"
        ),
    ),
)

COLORS = {
    "ink": "#182230",
    "muted": "#667085",
    "floor": "#D0D5DD",
    "surface": "#98A2B3",
    "wall": "#475467",
    "aabb": "#84CAFF",
    "aabb_edge": "#1570EF",
    "candidate": "#667085",
    "receiver": "#175CD3",
    "truth": "#D92D20",
    "prediction": "#7A5AF8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(case: Case) -> dict:
    result_path = EXP_DIR / "results" / f"query_{case.query:05d}_depth_aabb_result.json"
    result = json.loads(result_path.read_text())
    arrays_path = result_path.parent / result["arrays_file"]
    with np.load(arrays_path) as arrays:
        candidates = np.asarray(arrays["candidates"], dtype=np.float64)

    mesh = o3d.io.read_triangle_mesh(
        str(case.mesh_path), enable_post_processing=False
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(vertices) or not len(triangles):
        raise RuntimeError(f"failed to load official visualization mesh: {case.mesh_path}")

    receiver = np.asarray(result["receiver_global"], dtype=np.float64)
    truth = np.asarray(result["source_global"], dtype=np.float64)
    prediction = np.asarray(result["metrics"]["prediction_global"], dtype=np.float64)
    envelope = result["envelope_audit"]
    lower = receiver + np.asarray(envelope["lower_receiver_local_m"], dtype=np.float64)
    upper = receiver + np.asarray(envelope["upper_receiver_local_m"], dtype=np.float64)

    return {
        "case": case,
        "result": result,
        "result_path": result_path,
        "arrays_path": arrays_path,
        "vertices": vertices,
        "triangles": triangles,
        "candidates": candidates,
        "receiver": receiver,
        "truth": truth,
        "prediction": prediction,
        "lower": lower,
        "upper": upper,
    }


def mesh_projection(data: dict) -> dict:
    vertices = data["vertices"]
    faces = vertices[data["triangles"]]
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normal_z = np.divide(
        np.abs(cross[:, 2]), norm, out=np.zeros_like(norm), where=norm > 1e-12
    )
    center_z = faces[:, :, 2].mean(axis=1)
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    room_height = max(z_max - z_min, 1e-6)

    # Exclude the ceiling so the actual floor plan and internal objects remain visible.
    horizontal = (normal_z >= 0.82) & (center_z < z_max - 0.08 * room_height)
    floor = horizontal & (center_z <= z_min + 0.06 * room_height)
    surfaces = horizontal & ~floor
    vertical = (normal_z <= 0.25) & (norm > 1e-5)

    # The projected triangles of a vertical face are nearly line segments.  Keeping
    # all three edges preserves walls and furniture outlines without a 3-D renderer.
    vertical_paths = faces[vertical][:, [0, 1, 2, 0], :2]
    return {
        "floor": faces[floor][:, :, :2],
        "surfaces": faces[surfaces][:, :, :2],
        "walls": vertical_paths,
    }


def add_official_mesh(ax, projection: dict, *, strong: bool) -> None:
    alpha_scale = 1.0 if strong else 0.70
    if len(projection["floor"]):
        ax.add_collection(
            PolyCollection(
                projection["floor"],
                facecolor=COLORS["floor"],
                edgecolor="none",
                alpha=0.72 * alpha_scale,
                rasterized=True,
                zorder=1,
            )
        )
    if len(projection["surfaces"]):
        ax.add_collection(
            PolyCollection(
                projection["surfaces"],
                facecolor=COLORS["surface"],
                edgecolor="none",
                alpha=0.20 * alpha_scale,
                rasterized=True,
                zorder=2,
            )
        )
    if len(projection["walls"]):
        ax.add_collection(
            LineCollection(
                projection["walls"],
                colors=COLORS["wall"],
                linewidths=0.20 if strong else 0.16,
                alpha=0.30 * alpha_scale,
                rasterized=True,
                zorder=3,
            )
        )


def add_aabb(ax, lower: np.ndarray, upper: np.ndarray, *, filled: bool) -> None:
    width, height = upper[:2] - lower[:2]
    ax.add_patch(
        Rectangle(
            lower[:2],
            width,
            height,
            facecolor=COLORS["aabb"] if filled else "none",
            edgecolor=COLORS["aabb_edge"],
            linewidth=1.8,
            linestyle="-" if filled else (0, (5, 3)),
            alpha=0.28 if filled else 0.95,
            zorder=2 if filled else 5,
        )
    )
    if filled:
        # A light 0.5 m plan grid makes the rectangular FEM domain explicit while
        # avoiding an unreadable rendering of all tetrahedra.
        first_x = np.ceil(lower[0] / 0.5) * 0.5
        first_y = np.ceil(lower[1] / 0.5) * 0.5
        for x in np.arange(first_x, upper[0], 0.5):
            ax.plot(
                [x, x], [lower[1], upper[1]], color=COLORS["aabb_edge"],
                lw=0.35, alpha=0.20, zorder=3,
            )
        for y in np.arange(first_y, upper[1], 0.5):
            ax.plot(
                [lower[0], upper[0]], [y, y], color=COLORS["aabb_edge"],
                lw=0.35, alpha=0.20, zorder=3,
            )


def add_points(ax, data: dict) -> None:
    candidates = data["candidates"]
    ax.scatter(
        candidates[:, 0], candidates[:, 1], s=8, marker="o",
        c=COLORS["candidate"], alpha=0.48, edgecolors="none", rasterized=True,
        zorder=6,
    )
    ax.scatter(
        *data["receiver"][:2], s=72, marker="^", c=COLORS["receiver"],
        edgecolors="white", linewidths=0.8, zorder=9,
    )
    ax.scatter(
        *data["truth"][:2], s=92, marker="X", c=COLORS["truth"],
        edgecolors="white", linewidths=0.8, zorder=10,
    )
    ax.scatter(
        *data["prediction"][:2], s=110, marker="*", c=COLORS["prediction"],
        edgecolors="white", linewidths=0.8, zorder=11,
    )


def bounds_for(data: dict) -> tuple[float, float, float, float]:
    xy = np.vstack(
        [
            data["vertices"][:, :2],
            data["candidates"][:, :2],
            data["lower"][:2],
            data["upper"][:2],
            data["receiver"][:2],
            data["truth"][:2],
            data["prediction"][:2],
        ]
    )
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(high - low, 1.0)
    margin = 0.06 * max(span)
    return low[0] - margin, high[0] + margin, low[1] - margin, high[1] + margin


def style_axis(ax, limits, *, ylabel: bool) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", color=COLORS["muted"], fontsize=9)
    if ylabel:
        ax.set_ylabel("y (m)", color=COLORS["muted"], fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=8, colors=COLORS["muted"], length=2.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#D0D5DD")


def case_detail(data: dict) -> str:
    result = data["result"]
    dims = np.asarray(result["envelope_audit"]["dimensions_m"])
    error = float(result["metrics"]["localization_error_m"])
    nodes = int(result["fem_audit"]["node_count"])
    return (
        f"q{result['query_index']}  |  {len(data['candidates'])} candidates  |  "
        f"AABB {dims[0]:.2f}×{dims[1]:.2f}×{dims[2]:.2f} m  |  "
        f"{nodes:,} FEM nodes  |  error {error:.3f} m"
    )


def draw_case_row(axes, data: dict, *, show_column_titles: bool) -> None:
    projection = mesh_projection(data)
    limits = bounds_for(data)
    if show_column_titles:
        titles = (
            "Official room mesh (reference only)",
            "Depth-AABB FEM reconstruction",
            "Overlay: geometry mismatch",
        )
        for ax, title in zip(axes, titles):
            ax.set_title(
                title, loc="left", fontsize=12, fontweight="bold", pad=8, y=1.20
            )

    add_official_mesh(axes[0], projection, strong=True)
    add_points(axes[0], data)

    add_aabb(axes[1], data["lower"], data["upper"], filled=True)
    add_points(axes[1], data)

    add_official_mesh(axes[2], projection, strong=False)
    add_aabb(axes[2], data["lower"], data["upper"], filled=False)
    add_points(axes[2], data)

    for index, ax in enumerate(axes):
        style_axis(ax, limits, ylabel=index == 0)

    axes[0].text(
        0.0, 1.015, data["case"].label + "\n" + case_detail(data),
        transform=axes[0].transAxes, ha="left", va="bottom",
        fontsize=9.5, color=COLORS["ink"], fontweight="bold",
        clip_on=False,
    )


def legend_handles() -> list:
    return [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["floor"], edgecolor=COLORS["wall"], label="Official mesh"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["aabb"], alpha=0.35, edgecolor=COLORS["aabb_edge"], label="Depth-AABB"),
        Line2D([], [], marker="o", linestyle="", markersize=5, color=COLORS["candidate"], label="Frozen candidates"),
        Line2D([], [], marker="^", linestyle="", markersize=8, markerfacecolor=COLORS["receiver"], markeredgecolor="white", label="Receiver"),
        Line2D([], [], marker="X", linestyle="", markersize=8, markerfacecolor=COLORS["truth"], markeredgecolor="white", label="Ground-truth source"),
        Line2D([], [], marker="*", linestyle="", markersize=11, markerfacecolor=COLORS["prediction"], markeredgecolor="white", label="FEM prediction"),
    ]


def save_combined(
    all_data: list[dict], output_dir: Path, *, filename: str, heading: str
) -> Path:
    fig, axes = plt.subplots(len(all_data), 3, figsize=(15.8, 14.6), facecolor="white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.86, hspace=0.47, wspace=0.08)
    for row, data in enumerate(all_data):
        draw_case_row(axes[row], data, show_column_titles=row == 0)
    fig.suptitle(
        heading,
        x=0.055, y=0.972, ha="left", fontsize=17, fontweight="bold", color=COLORS["ink"],
    )
    fig.text(
        0.055, 0.942,
        "Top-down view. The official OBJ is post-hoc visualization only; it was not used by reconstruction or localization.",
        ha="left", fontsize=10.5, color=COLORS["muted"],
    )
    fig.legend(
        handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.52, 0.018),
        ncol=6, frameon=False, fontsize=9.5,
    )
    path = output_dir / filename
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_single(data: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.25), facecolor="white")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.16, top=0.78, wspace=0.08)
    draw_case_row(axes, data, show_column_titles=True)
    fig.legend(
        handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.52, 0.025),
        ncol=6, frameon=False, fontsize=9.2,
    )
    path = output_dir / f"real_vs_depth_aabb_q{data['case'].query:05d}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_manifest(all_data: list[dict], outputs: list[Path], output_dir: Path) -> Path:
    records = []
    for data in all_data:
        records.append(
            {
                "query_index": data["case"].query,
                "room": data["result"]["room"],
                "official_mesh_role": "post-hoc visualization reference only",
                "official_mesh_path": str(data["case"].mesh_path),
                "official_mesh_sha256": sha256(data["case"].mesh_path),
                "depth_path": data["result"]["depth_path"],
                "depth_sha256": data["result"]["depth_sha256"],
                "result_path": str(data["result_path"]),
                "result_sha256": sha256(data["result_path"]),
                "arrays_path": str(data["arrays_path"]),
                "arrays_sha256": sha256(data["arrays_path"]),
                "aabb_lower_global_m": data["lower"].tolist(),
                "aabb_upper_global_m": data["upper"].tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "description": "Post-hoc visualization of official geometry and frozen Depth-AABB FEM outputs.",
        "leakage_guard": "Official meshes are read only by this visualization script after localization results already exist.",
        "cases": records,
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
    }
    path = output_dir / "visualization_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preset",
        choices=("diagnostic", "low-error"),
        default="diagnostic",
        help="case group to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preset == "low-error":
        cases = LOW_ERROR_CASES
        filename = "real_vs_depth_aabb_low_error_rooms.png"
        heading = "Low-Error Cases: Official Geometry vs. Depth-AABB FEM Geometry"
    else:
        cases = DIAGNOSTIC_CASES
        filename = "real_vs_depth_aabb_three_rooms.png"
        heading = "Official Room Geometry vs. Single-Panorama Depth-AABB FEM Geometry"
    all_data = [load_case(case) for case in cases]
    outputs = [
        save_combined(
            all_data, args.output_dir, filename=filename, heading=heading
        )
    ]
    outputs.extend(save_single(data, args.output_dir) for data in all_data)
    manifest = write_manifest(all_data, outputs, args.output_dir)
    for path in [*outputs, manifest]:
        print(path)


if __name__ == "__main__":
    main()
