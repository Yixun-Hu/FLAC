#!/usr/bin/env python3
"""Visualize exp_09 sparse ground-truth-RIR AGREE upper-bound diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256
from src.localization.runner import file_sha256


CASE_LABELS = {
    "sharp": "Sharp real-RIR match",
    "ambiguous": "Smallest target margin",
    "diffuse": "Most diffuse score field",
    "typical": "Typical target margin",
}


def resolve_visualization_cases(payload: dict) -> list[dict]:
    """Join registered case identities to the full score-field records."""

    by_query = {item["query_id"]: item for item in payload["results"]}
    if len(by_query) != len(payload["results"]):
        raise RuntimeError("oracle payload contains duplicate query ids")
    cases = []
    used = set()
    for registered in payload["representative_cases"]:
        query_id = registered["query_id"]
        if query_id in used or query_id not in by_query:
            raise RuntimeError("registered visualization case is duplicate or missing")
        cases.append({"category": registered["category"], **by_query[query_id]})
        used.add(query_id)
    return cases


def _load_hashed_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    expected = payload.get("sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if expected != canonical_sha256(body):
        raise RuntimeError(f"stale or corrupt SHA-256 in {path}")
    return payload


def _mesh_edges(mesh_path: Path, target_triangles: int = 5000) -> np.ndarray:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"mesh has no drawable geometry: {mesh_path}")
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[:, :2]
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"mesh has non-finite vertices: {mesh_path}")
    segments = np.concatenate(
        (
            vertices[triangles[:, [0, 1]]],
            vertices[triangles[:, [1, 2]]],
            vertices[triangles[:, [2, 0]]],
        ),
        axis=0,
    )
    return segments


def _render_summary(payload: dict, output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    results = payload["results"]
    margins = np.asarray([item["target_margin"] for item in results])
    probabilities = np.asarray([item["target_probability"] for item in results])
    entropy = np.asarray([item["normalized_entropy"] for item in results])
    negative_distance = np.asarray(
        [item["hardest_negative_distance_m"] for item in results]
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    axes[0, 0].hist(margins, bins=18, color="#1f77b4", alpha=0.86)
    axes[0, 0].axvline(np.median(margins), color="#202020", ls="--", lw=1.3)
    axes[0, 0].set(xlabel="target − hardest-negative score", ylabel="queries", title="Ground-truth-RIR margin")
    axes[0, 1].hist(probabilities, bins=18, color="#2ca02c", alpha=0.86)
    axes[0, 1].set(xlabel=f"target softmax mass (T={payload['temperature']})", ylabel="queries", title="Normalized target mass")
    axes[1, 0].hist(entropy, bins=18, color="#9467bd", alpha=0.86)
    axes[1, 0].set(xlabel="normalized entropy", ylabel="queries", title="Score-field diffuseness")
    scatter = axes[1, 1].scatter(
        negative_distance,
        margins,
        c=entropy,
        cmap="magma",
        s=30,
        alpha=0.85,
        edgecolors="none",
    )
    axes[1, 1].set(
        xlabel="hardest-negative distance (m)",
        ylabel="target margin",
        title="Spatial vs embedding ambiguity",
    )
    figure.colorbar(scatter, ax=axes[1, 1], label="normalized entropy")
    figure.suptitle(
        f"Real-RIR AGREE upper bound · nested K={payload['primary_score_sample_count']} · "
        f"{payload['query_count']} queries / "
        f"{payload['room_count']} rooms",
        fontsize=13,
    )
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _render_cases(
    cases: list[dict], geometry: dict, output: Path, dpi: int, primary_count: int
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots(
        len(cases), 2, figsize=(12.2, 3.45 * len(cases)), constrained_layout=True
    )
    if len(cases) == 1:
        axes = np.asarray([axes])
    mesh_cache = {}
    for row_index, case in enumerate(cases):
        positions = np.asarray(case["positions_global"], dtype=np.float64)
        scores = np.asarray(case["scores"], dtype=np.float64)
        probabilities = np.asarray(case["probabilities"], dtype=np.float64)
        target_index = int(case["target_index"])
        negative_index = int(case["hardest_negative_index"])
        target = positions[target_index]
        receiver = np.asarray(case["receiver_global"], dtype=np.float64)
        distances = np.linalg.norm(positions - target, axis=1)
        room = case["room"]
        if room not in mesh_cache:
            mesh_cache[room] = _mesh_edges(Path(geometry["rooms"][room]["mesh_path"]))
        ax_map, ax_score = axes[row_index]
        ax_map.add_collection(
            LineCollection(mesh_cache[room], colors="#8a8a8a", linewidths=0.18, alpha=0.18)
        )
        sizes = 30.0 + 170.0 * probabilities / max(probabilities.max(), 1e-12)
        points = ax_map.scatter(
            positions[:, 0],
            positions[:, 1],
            c=scores,
            s=sizes,
            cmap="viridis",
            vmin=float(scores.min()),
            vmax=1.0,
            edgecolors="#202020",
            linewidths=0.25,
            zorder=3,
        )
        ax_map.scatter(*receiver[:2], marker="^", s=75, c="#777777", edgecolors="white", linewidths=0.6, zorder=5)
        ax_map.scatter(*target[:2], marker="*", s=230, c="#42c451", edgecolors="#102814", linewidths=0.8, zorder=7)
        ax_map.scatter(*positions[negative_index, :2], marker="x", s=105, c="#e43d96", linewidths=2.0, zorder=6)
        ax_map.autoscale()
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set(xlabel="global x (m)", ylabel="global y (m)")
        figure.colorbar(
            points, ax=ax_map, label=f"real-RIR AGREE K={primary_count} score"
        )

        ax_score.scatter(distances, scores, s=sizes, c=scores, cmap="viridis", vmin=float(scores.min()), vmax=1.0, edgecolors="#202020", linewidths=0.25)
        ax_score.scatter(0.0, scores[target_index], marker="*", s=230, c="#42c451", edgecolors="#102814", linewidths=0.8, zorder=5)
        ax_score.scatter(distances[negative_index], scores[negative_index], marker="x", s=105, c="#e43d96", linewidths=2.0, zorder=6)
        ax_score.axvline(0.5, color="#999999", ls="--", lw=0.9)
        ax_score.axvline(1.0, color="#999999", ls=":", lw=0.9)
        ax_score.set(
            xlabel="distance from target (m)",
            ylabel=f"real-RIR AGREE K={primary_count} score",
            ylim=(min(scores.min() - 0.02, 0.98), 1.01),
        )

        label = CASE_LABELS.get(case["category"], case["category"])
        title = (
            f"{label} · {room} · {Path(case['query_id']).stem}\n"
            f"bank={case['candidate_count']}  margin={case['target_margin']:.4f}  "
            f"target mass={case['target_probability']:.3f}  entropy={case['normalized_entropy']:.3f}"
        )
        ax_map.set_title(title, loc="left", fontsize=9.5)
        ax_score.set_title(
            f"hardest negative: {case['hardest_negative_distance_m']:.2f} m, "
            f"score={case['hardest_negative_score']:.4f}",
            fontsize=9.5,
        )
    figure.suptitle(
        "Sparse metadata-bank score fields · green star = exact target · pink × = hardest negative",
        fontsize=13,
    )
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-json", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=210)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("visualization outputs must stay inside NeuriPs_Workshop") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    oracle = _load_hashed_json(args.oracle_json)
    geometry = _load_hashed_json(args.geometry_audit)
    cases = resolve_visualization_cases(oracle)
    summary_path = output_dir / "real_rir_oracle_summary.png"
    cases_path = output_dir / "real_rir_oracle_cases.png"
    _render_summary(oracle, summary_path, args.dpi)
    _render_cases(
        cases,
        geometry,
        cases_path,
        args.dpi,
        int(oracle["primary_score_sample_count"]),
    )
    manifest = {
        "schema_version": 1,
        "oracle_sha256": oracle["sha256"],
        "geometry_audit_sha256": geometry["sha256"],
        "case_query_ids": [item["query_id"] for item in cases],
        "summary_figure": summary_path.name,
        "summary_figure_sha256": file_sha256(summary_path),
        "case_figure": cases_path.name,
        "case_figure_sha256": file_sha256(cases_path),
    }
    manifest["sha256"] = canonical_sha256(manifest)
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
