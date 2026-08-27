#!/usr/bin/env python3
"""Visualize the exact exp_09 localization candidate-grid construction stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.geometry import (  # noqa: E402
    build_lattice,
    classify_mesh_candidates,
    filter_query_candidates,
    grid_oracle_error,
    load_raycast_scene,
)

EXP09_DIR = REPO_ROOT / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
DEFAULT_CONTEXT_MANIFEST = EXP09_DIR / "context_manifest_exp01_seed42.json"
DEFAULT_GEOMETRY_AUDIT = EXP09_DIR / "geometry_audit.json"
DEFAULT_OUTPUT = (
    EXP09_DIR
    / "candidate_grid_visualization"
    / "candidate_grid_generation_meetingroom_idx_32_q922.png"
)
DEFAULT_QUERY_INDEX = 922
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

COLORS = {
    "ink": "#182230",
    "muted": "#667085",
    "mesh": "#98A2B3",
    "raw": "#64748B",
    "invalid": "#D92D20",
    "base": "#079455",
    "z_reject": "#7A5AF8",
    "receiver_reject": "#F79009",
    "context_reject": "#DD2590",
    "candidate": "#1570EF",
    "receiver": "#344054",
    "context": "#F04438",
    "truth": "#DC6803",
    "oracle": "#12B76A",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _index_sha256(mask: np.ndarray) -> str:
    indices = np.flatnonzero(mask).astype("<u4", copy=False)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _font(size: float, *, bold: bool = False) -> FontProperties:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    return FontProperties(fname=str(path), size=size)


def _add_mesh_3d(ax, vertices: np.ndarray, triangles: np.ndarray) -> None:
    collection = Poly3DCollection(
        vertices[triangles],
        facecolor=COLORS["mesh"],
        edgecolor="none",
        alpha=0.055,
        rasterized=True,
    )
    ax.add_collection3d(collection)


def _add_mesh_xy(ax, vertices: np.ndarray, triangles: np.ndarray) -> None:
    # A faint top-down projection supplies room context without hiding grid points.
    collection = PolyCollection(
        vertices[triangles][:, :, :2],
        facecolor=COLORS["mesh"],
        edgecolor="none",
        alpha=0.018,
        rasterized=True,
    )
    ax.add_collection(collection)


def _draw_aabb(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    corners = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    for i, first in enumerate(corners):
        for second in corners[i + 1 :]:
            if np.count_nonzero(np.abs(first - second) > 1e-9) == 1:
                ax.plot(*zip(first, second), color=COLORS["muted"], lw=0.75, alpha=0.5)


def _style_3d(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    margin = 0.15
    ax.set_xlim(lower[0] - margin, upper[0] + margin)
    ax.set_ylim(lower[1] - margin, upper[1] + margin)
    ax.set_zlim(lower[2] - margin, upper[2] + margin)
    ax.set_xlabel("x (m)", labelpad=1, color=COLORS["muted"])
    ax.set_ylabel("y (m)", labelpad=1, color=COLORS["muted"])
    ax.set_zlabel("z (m)", labelpad=1, color=COLORS["muted"])
    ax.tick_params(labelsize=7, colors=COLORS["muted"], pad=0)
    ax.view_init(elev=23, azim=-57)
    ax.set_box_aspect(np.maximum(upper - lower, 0.1))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.99, 0.35))
        axis.pane.set_edgecolor((0.85, 0.87, 0.9, 0.4))


def _title(ax, heading: str, detail: str) -> None:
    ax.set_title(
        heading + "\n" + detail,
        fontproperties=_font(12, bold=True),
        color=COLORS["ink"],
        loc="left",
        pad=10,
    )


def _scatter_anchor_3d(ax, receiver, contexts, truth) -> None:
    ax.scatter(
        *receiver,
        marker="^",
        s=70,
        c=COLORS["receiver"],
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        label="接收器",
        zorder=8,
    )
    ax.scatter(
        contexts[:, 0],
        contexts[:, 1],
        contexts[:, 2],
        marker="D",
        s=32,
        c=COLORS["context"],
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
        label="上下文声源",
        zorder=8,
    )
    ax.scatter(
        *truth,
        marker="*",
        s=135,
        c=COLORS["truth"],
        edgecolors="white",
        linewidths=0.8,
        depthshade=False,
        label="目标真值",
        zorder=9,
    )


def _legend(ax, *, ncol: int = 1, location: str = "upper right") -> None:
    legend = ax.legend(
        loc=location,
        ncol=ncol,
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        borderpad=0.5,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_fontproperties(_font(8))


def render(
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
    raw_points = build_lattice(mesh.aabb_min, mesh.aabb_max, audit["grid_spacing_m"])
    base_mask, surface_distances = classify_mesh_candidates(
        mesh, raw_points, audit["surface_clearance_m"]
    )
    base_points = raw_points[base_mask]

    receiver = np.asarray(record["receiver_global"], dtype=np.float64)
    contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
    truth = np.asarray(record["source_global"], dtype=np.float64)
    z_low, z_high = map(float, query_audit["z_band"])
    eps = float(audit["epsilon_m"])
    receiver_ok = (
        np.linalg.norm(base_points - receiver, axis=1) + eps
        >= float(audit["receiver_clearance_m"])
    )
    nearest_context_distance = np.linalg.norm(
        base_points[:, None, :] - contexts[None, :, :], axis=2
    ).min(axis=1)
    context_ok = nearest_context_distance + eps >= float(audit["context_clearance_m"])
    z_ok = (base_points[:, 2] + eps >= z_low) & (base_points[:, 2] - eps <= z_high)
    final_mask = filter_query_candidates(
        base_points,
        receiver,
        contexts,
        receiver_clearance=float(audit["receiver_clearance_m"]),
        context_clearance=float(audit["context_clearance_m"]),
        z_band=(z_low, z_high),
        eps=eps,
    )

    base_sha = hashlib.sha256(base_points.astype("<f8").tobytes()).hexdigest()
    checks = {
        "raw_count": len(raw_points) == int(room_audit["raw_lattice_count"]),
        "base_count": len(base_points) == int(room_audit["base_valid_count"]),
        "base_sha256": base_sha == room_audit["base_points_sha256"],
        "final_count": int(final_mask.sum()) == int(query_audit["chosen_count"]),
        "final_sha256": _index_sha256(final_mask) == query_audit["z_indices_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"reconstructed candidate grid failed frozen-artifact checks: {checks}")

    legacy = o3d.io.read_triangle_mesh(str(mesh.path), enable_post_processing=False)
    vertices = np.asarray(legacy.vertices, dtype=np.float64)
    triangles = np.asarray(legacy.triangles, dtype=np.int64)
    final_points = base_points[final_mask]
    nearest_index = int(np.argmin(np.linalg.norm(final_points - truth, axis=1)))
    oracle = final_points[nearest_index]
    oracle_error = grid_oracle_error(final_points, truth)
    if not np.isclose(oracle_error, float(query_audit["z_oracle_m"]), atol=1e-12):
        raise RuntimeError("reconstructed oracle error differs from frozen geometry audit")

    # Disjoint categories for stage 3. The underlying predicate is the conjunction
    # of all three masks; the ordering here only keeps the visualization readable.
    removed_z = ~z_ok
    removed_receiver = z_ok & ~receiver_ok
    removed_context = z_ok & receiver_ok & ~context_ok
    if not np.array_equal(
        final_mask, ~(removed_z | removed_receiver | removed_context)
    ):
        raise RuntimeError("display categories do not partition the base candidates")

    fig = plt.figure(figsize=(15.5, 11.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.975,
        bottom=0.095,
        top=0.875,
        wspace=0.09,
        hspace=0.20,
    )
    ax_raw = fig.add_subplot(grid[0, 0], projection="3d")
    ax_base = fig.add_subplot(grid[0, 1], projection="3d")
    ax_filter = fig.add_subplot(grid[1, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[1, 1])
    for ax in (ax_raw, ax_base, ax_filter):
        _add_mesh_3d(ax, vertices, triangles)
        _style_3d(ax, mesh.aabb_min, mesh.aabb_max)

    _draw_aabb(ax_raw, mesh.aabb_min, mesh.aabb_max)
    ax_raw.scatter(
        raw_points[:, 0],
        raw_points[:, 1],
        raw_points[:, 2],
        s=10,
        c=COLORS["raw"],
        alpha=0.64,
        depthshade=False,
        rasterized=True,
        label=f"原始点阵：{len(raw_points)}",
    )
    _title(ax_raw, "① AABB 内生成全局点阵", "三轴均按 0.5 m 间距对齐到全局坐标")
    _legend(ax_raw)

    ax_base.scatter(
        raw_points[~base_mask, 0],
        raw_points[~base_mask, 1],
        raw_points[~base_mask, 2],
        s=13,
        marker="x",
        linewidths=0.65,
        c=COLORS["invalid"],
        alpha=0.38,
        depthshade=False,
        rasterized=True,
        label=f"剔除：{int((~base_mask).sum())}",
    )
    ax_base.scatter(
        base_points[:, 0],
        base_points[:, 1],
        base_points[:, 2],
        s=15,
        c=COLORS["base"],
        alpha=0.82,
        edgecolors="white",
        linewidths=0.2,
        depthshade=False,
        rasterized=True,
        label=f"房间级候选：{len(base_points)}",
    )
    _title(
        ax_base,
        "② Mesh 有效性与表面净空筛选",
        "31 方向射线奇偶多数票；距任意表面 ≥ 0.20 m",
    )
    _legend(ax_base)

    ax_filter.scatter(
        base_points[removed_z, 0],
        base_points[removed_z, 1],
        base_points[removed_z, 2],
        marker="x",
        s=17,
        linewidths=0.7,
        c=COLORS["z_reject"],
        alpha=0.42,
        depthshade=False,
        rasterized=True,
        label=f"z 带外：{int(removed_z.sum())}",
    )
    ax_filter.scatter(
        base_points[removed_receiver, 0],
        base_points[removed_receiver, 1],
        base_points[removed_receiver, 2],
        marker="X",
        s=44,
        c=COLORS["receiver_reject"],
        alpha=0.95,
        depthshade=False,
        label=f"接收器净空：{int(removed_receiver.sum())}",
    )
    ax_filter.scatter(
        base_points[removed_context, 0],
        base_points[removed_context, 1],
        base_points[removed_context, 2],
        marker="X",
        s=44,
        c=COLORS["context_reject"],
        alpha=0.95,
        depthshade=False,
        label=f"上下文净空：{int(removed_context.sum())}",
    )
    ax_filter.scatter(
        final_points[:, 0],
        final_points[:, 1],
        final_points[:, 2],
        s=18,
        c=COLORS["candidate"],
        alpha=0.84,
        edgecolors="white",
        linewidths=0.25,
        depthshade=False,
        rasterized=True,
        label=f"最终候选：{len(final_points)}",
    )
    _scatter_anchor_3d(ax_filter, receiver, contexts, truth)
    _title(
        ax_filter,
        "③ 查询级筛选",
        f"z ∈ [{z_low:.1f}, {z_high:.1f}] m；接收器 ≥ 0.50 m；上下文声源 ≥ 0.25 m",
    )
    _legend(ax_filter, ncol=2, location="upper right")

    _add_mesh_xy(ax_xy, vertices, triangles)
    ax_xy.add_patch(
        Rectangle(
            mesh.aabb_min[:2],
            *(mesh.aabb_max[:2] - mesh.aabb_min[:2]),
            fill=False,
            edgecolor=COLORS["muted"],
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            alpha=0.55,
        )
    )
    height_colors = {1.0: "#84ADFF", 1.5: "#1570EF", 2.0: "#1849A9"}
    for z in sorted(np.unique(final_points[:, 2])):
        layer = np.isclose(final_points[:, 2], z)
        ax_xy.scatter(
            final_points[layer, 0],
            final_points[layer, 1],
            s=31,
            c=height_colors.get(float(z), COLORS["candidate"]),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.90,
            label=f"候选层 z={z:.1f} m（{int(layer.sum())}）",
            zorder=4,
        )
    ax_xy.add_patch(
        Circle(
            receiver[:2],
            float(audit["receiver_clearance_m"]),
            fill=False,
            edgecolor=COLORS["receiver_reject"],
            linewidth=1.5,
            linestyle=(0, (5, 3)),
            alpha=0.9,
            label="接收器 0.50 m 净空球（XY 投影）",
            zorder=5,
        )
    )
    for index, context in enumerate(contexts):
        ax_xy.add_patch(
            Circle(
                context[:2],
                float(audit["context_clearance_m"]),
                fill=False,
                edgecolor=COLORS["context_reject"],
                linewidth=0.8,
                alpha=0.45,
                label="上下文 0.25 m 净空球（XY 投影）" if index == 0 else None,
                zorder=5,
            )
        )
    ax_xy.scatter(
        *receiver[:2],
        marker="^",
        s=90,
        c=COLORS["receiver"],
        edgecolors="white",
        linewidths=0.8,
        label="接收器",
        zorder=8,
    )
    ax_xy.scatter(
        contexts[:, 0],
        contexts[:, 1],
        marker="D",
        s=38,
        c=COLORS["context"],
        edgecolors="white",
        linewidths=0.6,
        label="上下文声源",
        zorder=8,
    )
    ax_xy.scatter(
        *truth[:2],
        marker="*",
        s=190,
        c=COLORS["truth"],
        edgecolors="white",
        linewidths=0.8,
        label="目标真值",
        zorder=10,
    )
    ax_xy.scatter(
        *oracle[:2],
        marker="P",
        s=95,
        c=COLORS["oracle"],
        edgecolors="white",
        linewidths=0.8,
        label="最近网格点",
        zorder=9,
    )
    ax_xy.plot(
        [truth[0], oracle[0]],
        [truth[1], oracle[1]],
        color=COLORS["truth"],
        linewidth=1.7,
        linestyle=(0, (3, 2)),
        zorder=9,
    )
    midpoint = (truth[:2] + oracle[:2]) / 2
    ax_xy.annotate(
        f"3D oracle = {oracle_error:.3f} m",
        midpoint,
        xytext=(8, 7),
        textcoords="offset points",
        fontproperties=_font(8, bold=True),
        color=COLORS["truth"],
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.9},
        zorder=11,
    )
    ax_xy.set_xlim(mesh.aabb_min[0] - 0.15, mesh.aabb_max[0] + 0.15)
    ax_xy.set_ylim(mesh.aabb_min[1] - 0.15, mesh.aabb_max[1] + 0.15)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x (m)", color=COLORS["muted"])
    ax_xy.set_ylabel("y (m)", color=COLORS["muted"])
    ax_xy.tick_params(labelsize=8, colors=COLORS["muted"])
    ax_xy.grid(color="#E4E7EC", linewidth=0.6, alpha=0.65)
    for spine in ax_xy.spines.values():
        spine.set_color("#D0D5DD")
    _title(
        ax_xy,
        "④ 最终候选点（俯视）",
        "三层候选进入评分；圆表示三维净空球的 XY 投影外包络",
    )
    _legend(ax_xy, ncol=2, location="upper right")

    fig.suptitle(
        "定位候选点网格是怎样生成的？",
        x=0.045,
        y=0.965,
        ha="left",
        fontproperties=_font(22, bold=True),
        color=COLORS["ink"],
    )
    fig.text(
        0.045,
        0.925,
        f"真实示例  {record['room']}  ·  query index {query_index}  ·  "
        f"{record['filename']}",
        ha="left",
        fontproperties=_font(11),
        color=COLORS["muted"],
    )
    flow_text = (
        f"AABB 点阵  {len(raw_points)}    →    Mesh + 0.20 m 净空  {len(base_points)}"
        f"    →    查询级 z/接收器/上下文筛选  {len(final_points)}"
        f"    →    网格几何下限  {oracle_error:.3f} m"
    )
    fig.text(
        0.5,
        0.041,
        flow_text,
        ha="center",
        va="center",
        fontproperties=_font(11, bold=True),
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#F2F4F7",
            "edgecolor": "#D0D5DD",
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.975,
        0.012,
        "候选网格不插入目标真值；oracle 仅用于量化离散网格本身的最佳可能误差。",
        ha="right",
        fontproperties=_font(8),
        color=COLORS["muted"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "output": str(output_path.resolve()),
        "query_index": query_index,
        "room": record["room"],
        "raw_lattice_count": len(raw_points),
        "mesh_valid_count": len(base_points),
        "z_rejected_count": int(removed_z.sum()),
        "receiver_rejected_after_z_count": int(removed_receiver.sum()),
        "context_rejected_after_z_receiver_count": int(removed_context.sum()),
        "final_candidate_count": len(final_points),
        "grid_oracle_error_m": oracle_error,
        "reconstruction_checks": checks,
        "minimum_surface_distance_m": float(surface_distances[base_mask].min()),
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
    summary = render(
        context_manifest_path=args.context_manifest,
        geometry_audit_path=args.geometry_audit,
        query_index=args.query_index,
        output_path=args.output,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
