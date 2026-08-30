#!/usr/bin/env python3
"""Render solid-room localization comparisons for four completed model arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOMS = (
    "Apartments_idx_42",
    "Bathrooms_idx_18",
    "Bedrooms_idx_33",
    "LivingRoomsWithHallway_idx_25",
    "MeetingRoom_idx_20",
    "Restaurants_idx_24",
)
OVERLAY_ROOMS = (
    "Apartments_idx_42",
    "Bathrooms_idx_18",
    "MeetingRoom_idx_20",
    "Restaurants_idx_24",
)
MODEL_SPECS = {
    "vanilla": {
        "label": "Vanilla FLAC 40k",
        "short_label": "Vanilla",
        "metric_key": "1",
        "setting": r"$K_{\mathrm{gen}}=1$",
        "color": "#E68613",
        "marker": "o",
    },
    "fa_bf": {
        "label": "FA-BF FLAC 40k",
        "short_label": "FA-BF",
        "metric_key": "1",
        "setting": r"$K_{\mathrm{gen}}=1$",
        "color": "#1478B8",
        "marker": "s",
    },
    "yawaug": {
        "label": "YAWAUG FLAC AR 40k",
        "short_label": "YAWAUG",
        "metric_key": "1",
        "setting": r"$K_{\mathrm{gen}}=1$",
        "color": "#7651B5",
        "marker": "D",
    },
    "few_shot": {
        "label": "Few-ShotRIR-Waveform 100k",
        "short_label": "Few-ShotRIR",
        "metric_key": "8",
        "setting": r"$K_{\mathrm{ctx}}=8$",
        "color": "#16856A",
        "marker": "h",
    },
}
CAMERA_VERTICAL_FOV_DEG = 36.0
MARKER_SCALE_FRACTION = 0.042
MARKER_RADIUS_RANGE_M = (0.16, 0.48)


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


def load_results(directory: Path, expected_ids: set[str]) -> dict[str, dict]:
    results = {}
    for path in sorted((directory / "queries").glob("query_*.json")):
        result = load_hashed_json(path)
        query_id = result["query_id"]
        if query_id in results:
            raise RuntimeError(f"duplicate query in {directory}: {query_id}")
        results[query_id] = result
    if set(results) != expected_ids:
        raise RuntimeError(f"query coverage mismatch in {directory}")
    return results


def validate_models(models: dict[str, dict], query_id: str) -> None:
    reference = models["vanilla"]
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
    for name, result in models.items():
        for field in shared_fields:
            if result[field] != reference[field]:
                raise RuntimeError(f"{name} differs at {query_id}: {field}")
        spec = MODEL_SPECS[name]
        metric = result["metrics"][spec["metric_key"]]
        truth = np.asarray(result["source_global"], dtype=np.float64)
        prediction = np.asarray(metric["prediction_global"], dtype=np.float64)
        measured = float(np.linalg.norm(prediction - truth))
        if not np.isclose(measured, metric["localization_error_m"], atol=1e-9):
            raise RuntimeError(f"{name} coordinate/error mismatch at {query_id}")


def load_four_model_rows(exp9: Path, exp10: Path) -> tuple[list[dict], list[str]]:
    batches = (
        (
            "batch1_seed42",
            exp9 / "pilot_manifest_seed42_4_per_room.json",
            exp9 / "pilot_results",
            exp10 / "few_shot_rir_localization_seed42_pilot64",
        ),
        (
            "batch2_seed43",
            exp9 / "pilot_manifest_seed43_batch2_4_per_room.json",
            exp9 / "pilot_results_batch2",
            exp10 / "few_shot_rir_localization_seed43_batch2_pilot64",
        ),
    )
    rows = []
    pilot_hashes = []
    observed_ids: set[str] = set()
    for label, pilot_path, flac_root, few_shot_dir in batches:
        pilot = load_hashed_json(pilot_path)
        pilot_hashes.append(pilot["sha256"])
        expected_ids = {record["query_id"] for record in pilot["records"]}
        if observed_ids & expected_ids:
            raise RuntimeError("pilot batches overlap")
        observed_ids.update(expected_ids)
        outputs = {
            "vanilla": load_results(flac_root / "vanilla", expected_ids),
            "fa_bf": load_results(flac_root / "fa_bf", expected_ids),
            "yawaug": load_results(flac_root / "yawaug_ar_40k_vanilla", expected_ids),
            "few_shot": load_results(few_shot_dir, expected_ids),
        }
        for query_id in sorted(expected_ids):
            models = {name: values[query_id] for name, values in outputs.items()}
            validate_models(models, query_id)
            rows.append({"batch": label, "query_id": query_id, "models": models})
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 aligned rows, got {len(rows)}")
    return rows, pilot_hashes


def combined_error(row: dict) -> float:
    return float(
        np.mean(
            [
                row["models"][name]["metrics"][spec["metric_key"]][
                    "localization_error_m"
                ]
                for name, spec in MODEL_SPECS.items()
            ]
        )
    )


def select_representative_rows(rows: list[dict]) -> dict[str, dict]:
    """Pick the lowest-error FA-BF query in every selected room."""

    selected = {}
    for room in ROOMS:
        available = [row for row in rows if row["models"]["vanilla"]["room"] == room]
        if len(available) != 8:
            raise RuntimeError(f"expected eight aligned targets in {room}, got {len(available)}")
        selected[room] = min(
            available,
            key=lambda row: (
                row["models"]["fa_bf"]["metrics"]["1"]["localization_error_m"],
                row["query_id"],
            ),
        )
    return selected


def load_visual_mesh(mesh_path: Path, target_triangles: int):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"mesh has no drawable geometry: {mesh_path}")
    if target_triangles > 0 and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"mesh has non-finite vertices: {mesh_path}")
    return vertices, triangles


def cutaway_faces(vertices: np.ndarray, triangles: np.ndarray):
    faces = vertices[triangles]
    centers = faces.mean(axis=1)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = np.maximum(high - low, 1e-6)

    horizontal = np.abs(normals[:, 2]) > 0.82
    ceiling = horizontal & (centers[:, 2] > low[2] + 0.76 * extent[2])
    vertical = np.abs(normals[:, 2]) < 0.42
    near_x = centers[:, 0] > high[0] - 0.065 * extent[0]
    near_y = centers[:, 1] < low[1] + 0.065 * extent[1]
    foreground_outer_wall = vertical & (near_x | near_y)
    keep = valid & ~ceiling & ~foreground_outer_wall

    kept_faces = faces[keep]
    kept_centers = centers[keep]
    kept_normals = normals[keep]
    kept_horizontal = np.abs(kept_normals[:, 2]) > 0.72
    floor = kept_horizontal & (kept_centers[:, 2] < low[2] + 0.08 * extent[2])
    upper_surface = kept_horizontal & ~floor
    wall = ~kept_horizontal

    colors = np.zeros((len(kept_faces), 4), dtype=np.float64)
    colors[floor] = (0.29, 0.42, 0.58, 0.88)
    colors[upper_surface] = (0.78, 0.74, 0.64, 0.92)
    colors[wall] = (0.56, 0.59, 0.62, 0.64)
    return kept_faces, colors


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float):
    import open3d as o3d

    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("cylinder endpoints must differ")
    unit = direction / length
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=20,
        split=2,
    )
    z_axis = np.asarray([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, unit), -1.0, 1.0))
    if dot < 1.0 - 1e-12:
        axis = np.cross(z_axis, unit)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.asarray([np.pi, 0.0, 0.0])
            )
        else:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis / axis_norm * np.arccos(dot)
            )
        cylinder.rotate(rotation, center=(0.0, 0.0, 0.0))
    cylinder.translate(0.5 * (np.asarray(start) + np.asarray(end)))
    cylinder.compute_vertex_normals()
    return cylinder


class SolidSceneRenderer:
    """Reusable Open3D offscreen renderer with real depth occlusion."""

    def __init__(self, width: int = 1100, height: int = 820):
        import open3d as o3d

        self.o3d = o3d
        self.width = width
        self.height = height
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.background = np.asarray([0.969, 0.973, 0.980, 1.0], dtype=np.float32)

    def material(self, color):
        material = self.o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [float(value) for value in color]
        material.base_roughness = 0.82
        material.base_metallic = 0.0
        return material

    def colored_room_mesh(self, vertices: np.ndarray, triangles: np.ndarray):
        faces, colors = cutaway_faces(vertices, triangles)
        flattened = faces.reshape(-1, 3)
        mesh = self.o3d.geometry.TriangleMesh()
        mesh.vertices = self.o3d.utility.Vector3dVector(flattened)
        mesh.triangles = self.o3d.utility.Vector3iVector(
            np.arange(len(flattened), dtype=np.int32).reshape(-1, 3)
        )
        mesh.vertex_colors = self.o3d.utility.Vector3dVector(
            np.repeat(colors[:, :3], 3, axis=0)
        )
        mesh.compute_vertex_normals()
        material = self.o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        material.base_roughness = 0.92
        material.base_metallic = 0.0
        return mesh, material

    def add_mesh(self, name: str, mesh, color) -> None:
        mesh.compute_vertex_normals()
        self.renderer.scene.add_geometry(name, mesh, self.material(color))
        self.renderer.scene.scene.geometry_shadows(name, False, False)

    def render(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        reference: dict,
        model_results: dict[str, dict],
    ) -> np.ndarray:
        scene = self.renderer.scene
        scene.clear_geometry()
        scene.set_background(self.background)
        scene.scene.set_sun_light(
            np.asarray([-0.45, 0.55, -0.75]),
            np.asarray([1.0, 0.98, 0.93]),
            72000.0,
        )
        scene.scene.enable_sun_light(True)
        scene.scene.set_indirect_light_intensity(26000.0)

        room_mesh, room_material = self.colored_room_mesh(vertices, triangles)
        scene.add_geometry("room", room_mesh, room_material)
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
        extent = np.maximum(high - low, 0.1)
        scale = float(max(extent[0], extent[1]))
        marker_radius = float(
            np.clip(
                MARKER_SCALE_FRACTION * scale,
                MARKER_RADIUS_RANGE_M[0],
                MARKER_RADIUS_RANGE_M[1],
            )
        )
        truth = np.asarray(reference["source_global"], dtype=np.float64)
        receiver = np.asarray(reference["receiver_global"], dtype=np.float64)

        cross_directions = (
            np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        )
        for index, direction in enumerate(cross_directions):
            cross = cylinder_between(
                truth - 0.95 * marker_radius * direction,
                truth + 0.95 * marker_radius * direction,
                0.13 * marker_radius,
            )
            self.add_mesh(f"truth_{index}", cross, [0.90, 0.07, 0.05, 1.0])

        receiver_marker = self.o3d.geometry.TriangleMesh.create_cone(
            radius=0.55 * marker_radius,
            height=1.25 * marker_radius,
            resolution=24,
        )
        receiver_marker.translate(receiver)
        self.add_mesh("receiver", receiver_marker, [0.08, 0.11, 0.13, 1.0])

        predictions = []
        for index, (model_name, result) in enumerate(model_results.items()):
            spec = MODEL_SPECS[model_name]
            prediction = np.asarray(
                result["metrics"][spec["metric_key"]]["prediction_global"],
                dtype=np.float64,
            )
            color = [
                int(spec["color"][offset : offset + 2], 16) / 255.0
                for offset in (1, 3, 5)
            ] + [1.0]
            predictions.append((index, prediction, color))
            if np.linalg.norm(prediction - truth) > 1.0e-8:
                error_line = cylinder_between(
                    truth,
                    prediction,
                    0.035 * marker_radius,
                )
                self.add_mesh(f"error_{index}", error_line, color)

        colocated_groups = []
        for item in predictions:
            for group in colocated_groups:
                if np.linalg.norm(item[1] - group[0][1]) <= 1.0e-8:
                    group.append(item)
                    break
            else:
                colocated_groups.append([item])
        for group in colocated_groups:
            if len(group) == 1:
                radii = [0.64]
                tube_radius = 0.13
            else:
                radii = np.linspace(0.46, 0.86, len(group))
                tube_radius = 0.09
            for (index, prediction, color), radius in zip(group, radii):
                ring = self.o3d.geometry.TriangleMesh.create_torus(
                    torus_radius=float(radius) * marker_radius,
                    tube_radius=tube_radius * marker_radius,
                    radial_resolution=30,
                    tubular_resolution=16,
                )
                ring.translate(prediction)
                self.add_mesh(f"prediction_{index}", ring, color)

        center = 0.5 * (low + high)
        center[2] = low[2] + 0.40 * extent[2]
        eye = center + np.asarray([0.92 * scale, -1.12 * scale, 1.48 * scale])
        self.renderer.setup_camera(
            CAMERA_VERTICAL_FOV_DEG,
            center.astype(np.float32),
            eye.astype(np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )
        return np.asarray(self.renderer.render_to_image())


def configure_axes(ax, vertices: np.ndarray) -> None:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    extent = np.maximum(high - low, 0.1)
    center = 0.5 * (low + high)
    radius_xy = 0.57 * max(extent[0], extent[1])
    ax.set_xlim(center[0] - radius_xy, center[0] + radius_xy)
    ax.set_ylim(center[1] - radius_xy, center[1] + radius_xy)
    ax.set_zlim(max(0.0, low[2] - 0.03 * extent[2]), high[2] + 0.08 * extent[2])
    ax.set_box_aspect((2 * radius_xy, 2 * radius_xy, max(extent[2], 0.35 * radius_xy)))
    ax.view_init(elev=38, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def draw_room(ax, vertices: np.ndarray, triangles: np.ndarray) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces, colors = cutaway_faces(vertices, triangles)
    surface = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        shade=False,
    )
    surface.set_rasterized(True)
    ax.add_collection3d(surface)
    configure_axes(ax, vertices)


def draw_reference_markers(ax, result: dict) -> None:
    truth = np.asarray(result["source_global"], dtype=np.float64)
    receiver = np.asarray(result["receiver_global"], dtype=np.float64)
    ax.scatter(
        *receiver,
        marker="^",
        s=80,
        c="#263238",
        edgecolors="white",
        linewidths=0.9,
        depthshade=False,
        zorder=40,
        label="Receiver",
    )
    ax.scatter(
        *truth,
        marker="x",
        s=165,
        c="#E52521",
        linewidths=3.0,
        depthshade=False,
        zorder=60,
        label="Ground truth speaker",
    )


def draw_prediction(ax, result: dict, model_name: str, overlay: bool) -> None:
    spec = MODEL_SPECS[model_name]
    metric = result["metrics"][spec["metric_key"]]
    truth = np.asarray(result["source_global"], dtype=np.float64)
    prediction = np.asarray(metric["prediction_global"], dtype=np.float64)
    ax.plot(
        *np.stack([truth, prediction]).T,
        color=spec["color"],
        lw=1.6,
        ls=(0, (2, 2)),
        alpha=0.9,
        zorder=45,
    )
    marker = spec["marker"] if overlay else "o"
    ax.scatter(
        *prediction,
        marker=marker,
        s=125 if overlay else 140,
        facecolors="none",
        edgecolors=spec["color"],
        linewidths=2.7,
        depthshade=False,
        zorder=55,
        label=f"{spec['short_label']} predicted speaker",
    )


def target_label(query_id: str) -> str:
    return Path(query_id).stem.replace("_hybrid_IR", "")


def render_same_method(
    output_dir: Path,
    model_name: str,
    selected: dict[str, dict],
    mesh_cache: dict,
    renderer: SolidSceneRenderer,
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spec = MODEL_SPECS[model_name]
    figure, axes_grid = plt.subplots(2, 3, figsize=(15.8, 9.6), facecolor="#F7F8FA")
    axes = list(axes_grid.ravel())
    for ax, room in zip(axes, ROOMS):
        row = selected[room]
        result = row["models"][model_name]
        vertices, triangles = mesh_cache[room]
        ax.set_facecolor("#F7F8FA")
        image = renderer.render(
            vertices,
            triangles,
            result,
            {model_name: result},
        )
        ax.imshow(image)
        ax.set_axis_off()
    handles = [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="#263238", markeredgecolor="white", markersize=11, label="Receiver"),
        Line2D([], [], marker="x", linestyle="none", color="#E52521", markeredgewidth=3.0, markersize=14, label="Ground truth speaker"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=spec["color"], markeredgewidth=2.7, markersize=13, label=f"{spec['short_label']} predicted speaker"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=13,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    figure.subplots_adjust(
        left=0.008,
        right=0.992,
        top=0.995,
        bottom=0.075,
        wspace=0.006,
        hspace=0.006,
    )
    path = output_dir / f"same_method_{model_name}_six_geometries.png"
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def render_overlay(
    output_dir: Path,
    room: str,
    row: dict,
    mesh_cache: dict,
    renderer: SolidSceneRenderer,
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, ax = plt.subplots(1, 1, figsize=(9.4, 7.4), facecolor="#F7F8FA")
    ax.set_facecolor("#F7F8FA")
    vertices, triangles = mesh_cache[room]
    image = renderer.render(
        vertices,
        triangles,
        row["models"]["vanilla"],
        row["models"],
    )
    ax.imshow(image)
    ax.set_axis_off()
    handles = [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="#263238", markeredgecolor="white", markersize=10, label="Receiver"),
        Line2D([], [], marker="x", linestyle="none", color="#E52521", markeredgewidth=2.8, markersize=13, label="Ground truth speaker"),
    ]
    handles.extend(
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=spec["color"], markeredgewidth=2.5, markersize=11.5, label=f"{spec['short_label']} predicted speaker")
        for spec in MODEL_SPECS.values()
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=11.5,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.115)
    path = output_dir / f"cross_method_{room.lower()}.png"
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def render_overlay_grid(
    output_dir: Path,
    selected: dict[str, dict],
    mesh_cache: dict,
    renderer: SolidSceneRenderer,
    dpi: int,
) -> Path:
    """Render the four cross-method room views as one 2x2 figure."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes_grid = plt.subplots(
        2, 2, figsize=(15.8, 13.5), facecolor="#F7F8FA"
    )
    for ax, room in zip(axes_grid.ravel(), OVERLAY_ROOMS):
        row = selected[room]
        vertices, triangles = mesh_cache[room]
        image = renderer.render(
            vertices,
            triangles,
            row["models"]["vanilla"],
            row["models"],
        )
        ax.imshow(image)
        ax.set_facecolor("#F7F8FA")
        ax.set_axis_off()

    handles = [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="#263238", markeredgecolor="white", markersize=10, label="Receiver"),
        Line2D([], [], marker="x", linestyle="none", color="#E52521", markeredgewidth=2.8, markersize=13, label="Ground truth speaker"),
    ]
    handles.extend(
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=spec["color"], markeredgewidth=2.5, markersize=11.5, label=f"{spec['short_label']} predicted speaker")
        for spec in MODEL_SPECS.values()
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=12.2,
        handletextpad=0.35,
        columnspacing=1.35,
    )
    figure.subplots_adjust(
        left=0.005,
        right=0.995,
        top=0.995,
        bottom=0.075,
        wspace=0.008,
        hspace=0.008,
    )
    path = output_dir / "cross_method_four_geometries.png"
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def case_record(room: str, row: dict, mesh_entry: dict) -> dict:
    reference = row["models"]["vanilla"]
    models = {}
    for name, spec in MODEL_SPECS.items():
        result = row["models"][name]
        metric = result["metrics"][spec["metric_key"]]
        models[name] = {
            "label": spec["label"],
            "setting": spec["setting"],
            "prediction_global_m": metric["prediction_global"],
            "localization_error_m": float(metric["localization_error_m"]),
            "result_sha256": result["sha256"],
        }
    return {
        "room": room,
        "batch": row["batch"],
        "query_index": int(reference["query_index"]),
        "query_id": row["query_id"],
        "selection_value_fa_bf_error_m": float(
            row["models"]["fa_bf"]["metrics"]["1"]["localization_error_m"]
        ),
        "ground_truth_global_m": reference["source_global"],
        "receiver_global_m": reference["receiver_global"],
        "candidate_indices_sha256": reference["candidate_indices_sha256"],
        "mesh_path": mesh_entry["mesh_path"],
        "mesh_sha256": mesh_entry["mesh_sha256"],
        "models": models,
    }


def render_readme(payload: dict) -> str:
    cases = {case["room"]: case for case in payload["cases"]}
    lines = [
        "# Four-model solid-geometry localization visualizations",
        "",
        "Primary settings are `K_gen=1` for the three FLAC checkpoints and `K_ctx=8` "
        "for deterministic Few-ShotRIR. All four methods use the same eight acoustic "
        "contexts at the model boundary.",
        "",
        "The six rooms are fixed before query selection. Within each room, the displayed "
        "target is selected deterministically as the lowest-localization-error FA-BF case "
        "among the eight aligned queries. The same target is reused in every method view.",
        "",
        "## One method across six shared geometries",
        "",
    ]
    for name, spec in MODEL_SPECS.items():
        panel_text = "; ".join(
            f"{room.replace('_', ' ')} "
            f"({target_label(cases[room]['query_id'])}, "
            f"{cases[room]['models'][name]['localization_error_m']:.2f} m)"
            for room in ROOMS
        )
        lines.extend(
            [
                f"### {spec['label']}",
                "",
                f"![{spec['label']}](same_method_{name}_six_geometries.png)",
                "",
                f"*Figure: {spec['label']} ({spec['setting']}) across six shared room "
                f"geometries. Panels in row-major order: {panel_text}.*",
                "",
            ]
        )
    lines.extend(["## Four methods on one shared geometry", ""])
    for room in OVERLAY_ROOMS:
        case = cases[room]
        error_text = "; ".join(
            f"{MODEL_SPECS[name]['short_label']} "
            f"{case['models'][name]['localization_error_m']:.2f} m"
            for name in MODEL_SPECS
        )
        lines.extend(
            [
                f"### {room}",
                "",
                f"![{room}](cross_method_{room.lower()}.png)",
                "",
                f"*Figure: Four-model localization comparison in "
                f"{room.replace('_', ' ')}, target {target_label(case['query_id'])}. "
                f"Localization errors: {error_text}.*",
                "",
            ]
        )
    combined_panel_text = "; ".join(
        f"{room.replace('_', ' ')} ({target_label(cases[room]['query_id'])}; "
        + "/".join(
            f"{cases[room]['models'][name]['localization_error_m']:.2f}"
            for name in MODEL_SPECS
        )
        + " m)"
        for room in OVERLAY_ROOMS
    )
    lines.extend(
        [
            "## Four-method 2x2 combined figure",
            "",
            "![Four methods across four geometries](cross_method_four_geometries.png)",
            "",
            "*Figure: Four methods across four shared room geometries. Panels in "
            f"row-major order: {combined_panel_text}. Errors are ordered "
            "Vanilla/FA-BF/YAWAUG/Few-ShotRIR.*",
            "",
        ]
    )
    lines.extend(
        [
            "The audited official OBJ is rendered as a solid cutaway: ceiling and the two "
            "camera-facing outer walls are omitted for visibility; no mesh edge lines are "
            "drawn. The camera uses a tight elevated isometric view. Predictions at the "
            "same candidate coordinate are shown as concentric rings, without marker "
            "shadows, so every method remains visible. These display transforms do not "
            "alter marker coordinates or metrics. Figure titles and panel descriptions "
            "are omitted from the PNG files; identifying text appears only in the external "
            "figure captions in this document. "
            "TeX Gyre Pagella is used as the available Palatino-compatible typeface.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp9-dir", type=Path, required=True)
    parser.add_argument("--exp10-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh-triangles",
        type=int,
        default=0,
        help="optional decimation target; 0 preserves the audited OBJ exactly",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    exp9 = args.exp9_dir.resolve()
    exp10 = args.exp10_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(exp10)
    except ValueError as error:
        raise ValueError("output must remain inside the exp_10 directory") from error
    if args.mesh_triangles != 0 and args.mesh_triangles < 1000:
        raise ValueError("mesh-triangles must be 0 or at least 1000")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella", "Palatino", "serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    rows, pilot_hashes = load_four_model_rows(exp9, exp10)
    selected = select_representative_rows(rows)
    geometry_audit = load_hashed_json(exp9 / "geometry_audit.json")
    mesh_cache = {}
    records = []
    for room in ROOMS:
        mesh_entry = geometry_audit["rooms"][room]
        mesh_path = Path(mesh_entry["mesh_path"])
        if file_sha256(mesh_path) != mesh_entry["mesh_sha256"]:
            raise RuntimeError(f"official mesh hash changed: {mesh_path}")
        mesh_cache[room] = load_visual_mesh(mesh_path, args.mesh_triangles)
        records.append(case_record(room, selected[room], mesh_entry))

    renderer = SolidSceneRenderer()
    method_images = [
        str(
            render_same_method(
                output_dir, name, selected, mesh_cache, renderer, args.dpi
            ).name
        )
        for name in MODEL_SPECS
    ]
    overlay_images = [
        str(
            render_overlay(
                output_dir, room, selected[room], mesh_cache, renderer, args.dpi
            ).name
        )
        for room in OVERLAY_ROOMS
    ]
    combined_overlay_image = render_overlay_grid(
        output_dir, selected, mesh_cache, renderer, args.dpi
    ).name
    payload = {
        "schema_version": 4,
        "source_pilot_sha256": pilot_hashes,
        "aligned_query_count": len(rows),
        "model_primary_settings": {
            name: spec["setting"] for name, spec in MODEL_SPECS.items()
        },
        "selection": {
            "rooms": list(ROOMS),
            "overlay_rooms": list(OVERLAY_ROOMS),
            "query_rule": "minimum_fa_bf_localization_error_per_room",
            "manual_query_selection": False,
        },
        "rendering": {
            "style": "solid_cutaway",
            "ceiling_removed": True,
            "camera_facing_outer_walls_removed": True,
            "camera_view": "elevated_tight_isometric",
            "camera_vertical_fov_deg": CAMERA_VERTICAL_FOV_DEG,
            "mesh_edges_drawn": False,
            "marker_shadows": False,
            "marker_scale_fraction": MARKER_SCALE_FRACTION,
            "marker_radius_range_m": list(MARKER_RADIUS_RANGE_M),
            "colocated_predictions": "concentric_rings_at_identical_coordinates",
            "figure_titles_drawn": False,
            "panel_descriptions_drawn": False,
            "description_location": "external_readme_figure_caption",
            "decimation_target_triangles": args.mesh_triangles or None,
            "font": "TeX Gyre Pagella (Palatino-compatible)",
        },
        "method_images": method_images,
        "overlay_images": overlay_images,
        "combined_overlay_image": combined_overlay_image,
        "cases": records,
    }
    payload["sha256"] = canonical_sha256(payload)
    atomic_text(output_dir / "visualization_manifest.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(output_dir / "README.md", render_readme(payload))
    print(
        json.dumps(
            {
                "sha256": payload["sha256"],
                "method_images": len(method_images),
                "overlay_images": len(overlay_images),
                "combined_overlay_image": combined_overlay_image,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
