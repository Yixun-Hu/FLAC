#!/usr/bin/env python3
"""Render the final candidate grid used by one frozen localization query."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.geometry import (  # noqa: E402
    build_lattice,
    classify_mesh_candidates,
    filter_query_candidates,
    load_raycast_scene,
)

EXP09_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
DEFAULT_CONTEXT_MANIFEST = EXP09_DIR / "context_manifest_exp01_seed42.json"
DEFAULT_GEOMETRY_AUDIT = EXP09_DIR / "geometry_audit.json"
DEFAULT_OUTPUT = (
    EXP09_DIR
    / "candidate_grid_visualization"
    / "candidate_grid_case_meetingroom_idx_32_q922.png"
)
DEFAULT_QUERY_INDEX = 922

INK = "#182230"
MUTED = "#667085"


def _font(size: float, *, bold: bool = False) -> FontProperties:
    return FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal", size=size
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _index_sha256(mask: np.ndarray) -> str:
    indices = np.flatnonzero(mask).astype("<u4", copy=False)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _grid_segments(points: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Connect axis-adjacent candidates so the 3-D lattice reads as a grid."""
    decimals = 7
    lookup = {tuple(np.round(point, decimals)) for point in points}
    segments = []
    for point in points:
        for axis in range(3):
            neighbor = point.copy()
            neighbor[axis] += spacing[axis]
            if tuple(np.round(neighbor, decimals)) in lookup:
                segments.append([point, neighbor])
    return np.asarray(segments, dtype=np.float64)


def _mesh_cutaway(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split drawable faces and extract non-coplanar/boundary feature edges."""
    triangle_vertices = vertices[triangles]
    normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 1e-10
    normals[valid] /= normal_lengths[valid, None]
    centroids = triangle_vertices.mean(axis=1)
    horizontal = np.abs(normals[:, 2]) > 0.72
    ceiling = horizontal & (centroids[:, 2] > vertices[:, 2].max() - 0.16)
    floor_or_top = valid & horizontal & ~ceiling
    vertical = valid & ~horizontal
    drawable = floor_or_top | vertical

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in np.flatnonzero(drawable):
        triangle = triangles[face_index]
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = (int(min(first, second)), int(max(first, second)))
            edge_faces.setdefault(edge, []).append(int(face_index))
    sharp_threshold = np.cos(np.deg2rad(25.0))
    feature_edges = []
    for edge, faces in edge_faces.items():
        is_boundary = len(faces) == 1
        is_sharp = len(faces) >= 2 and any(
            abs(float(np.dot(normals[faces[0]], normals[other]))) < sharp_threshold
            for other in faces[1:]
        )
        if is_boundary or is_sharp:
            feature_edges.append(vertices[list(edge)])
    return (
        triangle_vertices[floor_or_top],
        triangle_vertices[vertical],
        np.asarray(feature_edges, dtype=np.float64),
        drawable,
    )


def render_case(
    *,
    context_manifest_path: Path,
    geometry_audit_path: Path,
    query_index: int,
    output_path: Path,
    dpi: int,
) -> dict:
    manifest = _load_json(context_manifest_path)
    audit = _load_json(geometry_audit_path)
    try:
        record = next(item for item in manifest["records"] if item["index"] == query_index)
        query_audit = next(item for item in audit["queries"] if item["index"] == query_index)
    except StopIteration as error:
        raise ValueError(f"query index {query_index} is absent from the frozen artifacts") from error

    room_audit = audit["rooms"][record["room"]]
    mesh = load_raycast_scene(room_audit["mesh_path"], compute_topology=False)
    spacing = np.asarray(audit["grid_spacing_m"], dtype=np.float64)
    raw_points = build_lattice(mesh.aabb_min, mesh.aabb_max, spacing)
    base_mask, _ = classify_mesh_candidates(
        mesh, raw_points, float(audit["surface_clearance_m"])
    )
    base_points = raw_points[base_mask]
    receiver = np.asarray(record["receiver_global"], dtype=np.float64)
    contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
    truth = np.asarray(record["source_global"], dtype=np.float64)
    final_mask = filter_query_candidates(
        base_points,
        receiver,
        contexts,
        receiver_clearance=float(audit["receiver_clearance_m"]),
        context_clearance=float(audit["context_clearance_m"]),
        z_band=tuple(query_audit["z_band"]),
        eps=float(audit["epsilon_m"]),
    )
    candidates = base_points[final_mask]

    checks = {
        "candidate_count": len(candidates) == int(query_audit["chosen_count"]),
        "candidate_indices_sha256": (
            _index_sha256(final_mask) == query_audit["z_indices_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"candidate reconstruction differs from the audit: {checks}")

    nearest_local_index = int(np.argmin(np.linalg.norm(candidates - truth, axis=1)))
    nearest = candidates[nearest_local_index]
    oracle_error = float(np.linalg.norm(nearest - truth))
    if not np.isclose(oracle_error, query_audit["z_oracle_m"], atol=1e-12):
        raise RuntimeError("nearest candidate differs from the audited grid oracle")

    legacy = o3d.io.read_triangle_mesh(str(mesh.path), enable_post_processing=False)
    vertices = np.asarray(legacy.vertices, dtype=np.float64)
    triangles = np.asarray(legacy.triangles, dtype=np.int64)
    segments = _grid_segments(candidates, spacing)
    horizontal_faces, vertical_faces, feature_edges, drawable_faces = _mesh_cutaway(
        vertices, triangles
    )

    fig = plt.figure(figsize=(12.8, 9.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.06, top=0.87)

    horizontal_surface = Poly3DCollection(
        horizontal_faces,
        facecolor="#B8C0CC",
        edgecolor="none",
        alpha=0.28,
        rasterized=True,
        label="Room mesh (cutaway)",
    )
    vertical_surface = Poly3DCollection(
        vertical_faces,
        facecolor="#98A2B3",
        edgecolor="none",
        alpha=0.17,
        rasterized=True,
    )
    ax.add_collection3d(horizontal_surface)
    ax.add_collection3d(vertical_surface)
    if len(feature_edges):
        ax.add_collection3d(
            Line3DCollection(
                feature_edges,
                colors="#667085",
                linewidths=0.72,
                alpha=0.62,
                rasterized=True,
            )
        )
    if len(segments):
        ax.add_collection3d(
            Line3DCollection(
                segments,
                colors="#84ADFF",
                linewidths=0.55,
                alpha=0.30,
                rasterized=True,
            )
        )

    layer_colors = {1.0: "#84ADFF", 1.5: "#1570EF", 2.0: "#1849A9"}
    for z in sorted(np.unique(candidates[:, 2])):
        layer = np.isclose(candidates[:, 2], z)
        ax.scatter(
            candidates[layer, 0],
            candidates[layer, 1],
            candidates[layer, 2],
            s=34,
            c=layer_colors.get(float(z), "#1570EF"),
            edgecolors="white",
            linewidths=0.45,
            alpha=0.96,
            depthshade=False,
            rasterized=True,
            label=f"Candidate grid, z={z:.1f} m ({int(layer.sum())})",
            zorder=6,
        )

    ax.scatter(
        *receiver,
        marker="^",
        s=125,
        c="#344054",
        edgecolors="white",
        linewidths=1.0,
        depthshade=False,
        label="Receiver",
        zorder=10,
    )
    ax.scatter(
        contexts[:, 0],
        contexts[:, 1],
        contexts[:, 2],
        marker="D",
        s=60,
        c="#F04438",
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        label="Context sources (8)",
        zorder=10,
    )
    ax.scatter(
        *truth,
        marker="*",
        s=230,
        c="#F79009",
        edgecolors="white",
        linewidths=1.0,
        depthshade=False,
        label="Target source (ground truth)",
        zorder=12,
    )
    ax.scatter(
        *nearest,
        marker="P",
        s=125,
        c="#12B76A",
        edgecolors="white",
        linewidths=1.0,
        depthshade=False,
        label="Nearest grid point",
        zorder=11,
    )
    ax.plot(
        [truth[0], nearest[0]],
        [truth[1], nearest[1]],
        [truth[2], nearest[2]],
        color="#DC6803",
        linewidth=2.0,
        linestyle=(0, (3, 2)),
        zorder=11,
    )
    midpoint = (truth + nearest) / 2
    ax.text(
        midpoint[0],
        midpoint[1],
        midpoint[2] + 0.09,
        f"{oracle_error:.3f} m",
        fontproperties=_font(9, bold=True),
        color="#DC6803",
        zorder=12,
    )

    margin = 0.12
    ax.set_xlim(mesh.aabb_min[0] - margin, mesh.aabb_max[0] + margin)
    ax.set_ylim(mesh.aabb_min[1] - margin, mesh.aabb_max[1] + margin)
    ax.set_zlim(mesh.aabb_min[2] - margin, mesh.aabb_max[2] + margin)
    ax.set_box_aspect(np.maximum(mesh.aabb_max - mesh.aabb_min, 0.1))
    ax.view_init(elev=25, azim=-57)
    ax.set_xlabel("x (m)", labelpad=5, color=MUTED)
    ax.set_ylabel("y (m)", labelpad=5, color=MUTED)
    ax.set_zlabel("z (m)", labelpad=5, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, pad=1)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.99, 0.45))
        axis.pane.set_edgecolor((0.82, 0.85, 0.89, 0.65))

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.00, 0.90),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.96,
        borderpad=0.7,
        labelspacing=0.75,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_fontproperties(_font(9))

    fig.suptitle(
        "Actual Candidate Grid for One Localization Query",
        x=0.045,
        y=0.955,
        ha="left",
        fontproperties=_font(21, bold=True),
        color=INK,
    )
    fig.text(
        0.047,
        0.908,
        f"{record['room']}  ·  query {query_index}  ·  {record['filename']}  ·  "
        f"0.5 m spacing  ·  {len(candidates)} candidates",
        ha="left",
        fontproperties=_font(11),
        color=MUTED,
    )
    fig.text(
        0.83,
        0.075,
        "Gray: real room mesh (ceiling removed)\n"
        "Blue lines: adjacent candidate points\n"
        "Ground truth is not inserted into the grid",
        ha="left",
        va="bottom",
        linespacing=1.55,
        fontproperties=_font(9),
        color=MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "output": str(output_path.resolve()),
        "room": record["room"],
        "query_index": query_index,
        "candidate_count": len(candidates),
        "candidate_z_layers_m": sorted(np.unique(candidates[:, 2]).astype(float).tolist()),
        "grid_edge_count": len(segments),
        "mesh_drawable_triangle_count": int(drawable_faces.sum()),
        "mesh_feature_edge_count": len(feature_edges),
        "grid_oracle_error_m": oracle_error,
        "reconstruction_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-manifest", type=Path, default=DEFAULT_CONTEXT_MANIFEST)
    parser.add_argument("--geometry-audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--query-index", type=int, default=DEFAULT_QUERY_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    summary = render_case(
        context_manifest_path=args.context_manifest,
        geometry_audit_path=args.geometry_audit,
        query_index=args.query_index,
        output_path=args.output,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
