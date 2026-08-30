#!/usr/bin/env python3
"""Render matched FEM--OMP and FEM--AGREE localization examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import visualize_four_model_localization as base


ROOMS = (
    "Apartments_idx_42",
    "Bathrooms_idx_18",
    "Bedrooms_idx_33",
    "LivingRoomsWithHallway_idx_25",
    "MeetingRoom_idx_20",
    "Restaurants_idx_24",
)

# Reuse the published six-room FLAC visualization cases whenever they pass the
# strict Depth-AABB gate. Apartments 42's published q5334 does not pass, and
# q5271 is the room's only strict paired query.
QUERY_INDICES = {
    "Apartments_idx_42": 5271,
    "Bathrooms_idx_18": 4728,
    "Bedrooms_idx_33": 2259,
    "LivingRoomsWithHallway_idx_25": 5782,
    "MeetingRoom_idx_20": 1368,
    "Restaurants_idx_24": 6236,
}

MODEL_SPECS = {
    "fem_omp": {
        "label": "FEM–OMP (Depth-AABB)",
        "short_label": "FEM–OMP",
        "metric_key": "main",
        "setting": "Room-Helps OMP",
        "color": "#D97706",
        "marker": "o",
    },
    "fem_agree": {
        "label": "FEM–AGREE (Depth-AABB)",
        "short_label": "FEM–AGREE",
        "metric_key": "main",
        "setting": "frozen AGREE, K=8",
        "color": "#7651B5",
        "marker": "o",
    },
}


def wrapped_result(reference: dict, metrics: dict) -> dict:
    return {
        "query_id": reference["query_id"],
        "query_index": int(reference["query_index"]),
        "room": reference["room"],
        "source_global": reference["source_global"],
        "receiver_global": reference["receiver_global"],
        "metrics": {"main": metrics},
    }


def load_cases(omp_dir: Path, agree_dir: Path) -> dict[str, dict]:
    selected = {}
    for room in ROOMS:
        query_index = QUERY_INDICES[room]
        omp_path = omp_dir / f"query_{query_index:05d}_depth_aabb_result.json"
        agree_path = agree_dir / f"query_{query_index:05d}.json"
        omp = base.load_hashed_json(omp_path)
        agree = base.load_hashed_json(agree_path)
        if omp["room"] != room or agree["room"] != room:
            raise RuntimeError(f"room mismatch at query {query_index}")
        if int(omp["query_index"]) != query_index or int(agree["query_index"]) != query_index:
            raise RuntimeError(f"query-index mismatch at query {query_index}")
        if omp["query_id"] != agree["query_id"]:
            raise RuntimeError(f"query identity mismatch at query {query_index}")
        with np.load(omp_path.parent / omp["arrays_file"]) as omp_arrays, np.load(
            agree_path.parent / agree["arrays_file"]
        ) as agree_arrays:
            if not np.array_equal(omp_arrays["candidates"], agree_arrays["candidates"]):
                raise RuntimeError(f"candidate coordinates mismatch at query {query_index}")
        if agree["source_fem_result_sha256"] != omp["sha256"]:
            raise RuntimeError(f"FEM source hash mismatch at query {query_index}")
        if agree["source_fem_omp_metrics"] != omp["metrics"]:
            raise RuntimeError(f"embedded OMP metrics mismatch at query {query_index}")

        omp_result = wrapped_result(omp, omp["metrics"])
        agree_result = wrapped_result(omp, agree["metrics_by_k_agree"]["8"])
        selected[room] = {
            "query_id": omp["query_id"],
            "query_index": query_index,
            "omp_sha256": omp["sha256"],
            "agree_sha256": agree["sha256"],
            "candidate_indices_sha256": agree["candidate_indices_sha256"],
            "models": {
                "fem_omp": omp_result,
                "fem_agree": agree_result,
            },
        }
    return selected


def render_overlay(
    output_dir: Path,
    selected: dict[str, dict],
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    renderer: base.SolidSceneRenderer,
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes_grid = plt.subplots(2, 3, figsize=(15.8, 9.6), facecolor="#F7F8FA")
    for ax, room in zip(axes_grid.ravel(), ROOMS):
        row = selected[room]
        vertices, triangles = mesh_cache[room]
        image = renderer.render(
            vertices,
            triangles,
            row["models"]["fem_omp"],
            row["models"],
        )
        ax.imshow(image)
        ax.set_facecolor("#F7F8FA")
        ax.set_axis_off()

    handles = [
        Line2D(
            [], [], marker="^", linestyle="none", markerfacecolor="#263238",
            markeredgecolor="white", markersize=11, label="Receiver"
        ),
        Line2D(
            [], [], marker="x", linestyle="none", color="#E52521",
            markeredgewidth=3.0, markersize=14, label="Ground truth speaker"
        ),
    ]
    handles.extend(
        Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=spec["color"], markeredgewidth=2.7,
            markersize=13, label=f"{spec['short_label']} predicted speaker"
        )
        for spec in MODEL_SPECS.values()
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=12.5,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    figure.subplots_adjust(
        left=0.008,
        right=0.992,
        top=0.995,
        bottom=0.075,
        wspace=0.006,
        hspace=0.006,
    )
    path = output_dir / "fem_omp_vs_agree_six_geometries.png"
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def render_readme(cases: list[dict]) -> str:
    lines = [
        "# FEM localization visualizations",
        "",
        "The figures use the same six-room solid-cutaway rendering contract as the "
        "published four-model visualization. FEM--OMP and FEM--AGREE use identical "
        "queries, Depth-AABB responses, frozen candidates, truth coordinates, receiver "
        "coordinates, official display meshes, and camera poses. FEM--AGREE reports K=8.",
        "",
        "Five cases reuse the earlier six-geometry figure exactly. Apartments 42's "
        "earlier q5334 fails the strict Depth-AABB coverage gate, so it is replaced by "
        "q5271, the room's only strict paired query.",
        "",
        "| Room | Query | Target | FEM--OMP error [m] | FEM--AGREE K=8 error [m] |",
        "|---|---:|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['room']} | {case['query_index']} | {base.target_label(case['query_id'])} "
            f"| {case['fem_omp_error_m']:.3f} | {case['fem_agree_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## FEM--OMP across six geometries",
            "",
            "![FEM--OMP](same_method_fem_omp_six_geometries.png)",
            "",
            "## FEM--AGREE across six geometries",
            "",
            "![FEM--AGREE](same_method_fem_agree_six_geometries.png)",
            "",
            "## Direct overlay",
            "",
            "![FEM--OMP versus FEM--AGREE](fem_omp_vs_agree_six_geometries.png)",
            "",
            "Black triangles are receivers, red crosses are ground-truth speakers, "
            "orange rings are FEM--OMP predictions, and purple rings are FEM--AGREE "
            "predictions. A line joins each prediction to the ground truth. Colocated "
            "predictions are rendered as concentric rings without altering coordinates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--omp-dir", type=Path, required=True)
    parser.add_argument("--agree-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
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

    selected = load_cases(args.omp_dir.resolve(), args.agree_dir.resolve())
    geometry = base.load_hashed_json(args.geometry_audit.resolve())
    mesh_cache = {}
    mesh_records = {}
    for room in ROOMS:
        mesh_entry = geometry["rooms"][room]
        mesh_path = Path(mesh_entry["mesh_path"])
        if base.file_sha256(mesh_path) != mesh_entry["mesh_sha256"]:
            raise RuntimeError(f"official mesh hash changed: {mesh_path}")
        mesh_cache[room] = base.load_visual_mesh(mesh_path, 0)
        mesh_records[room] = {
            "mesh_path": str(mesh_path),
            "mesh_sha256": mesh_entry["mesh_sha256"],
        }

    base.ROOMS = ROOMS
    base.MODEL_SPECS = MODEL_SPECS
    renderer = base.SolidSceneRenderer()
    omp_image = base.render_same_method(
        output_dir, "fem_omp", selected, mesh_cache, renderer, args.dpi
    )
    agree_image = base.render_same_method(
        output_dir, "fem_agree", selected, mesh_cache, renderer, args.dpi
    )
    overlay_image = render_overlay(output_dir, selected, mesh_cache, renderer, args.dpi)

    cases = []
    for room in ROOMS:
        row = selected[room]
        omp_metric = row["models"]["fem_omp"]["metrics"]["main"]
        agree_metric = row["models"]["fem_agree"]["metrics"]["main"]
        cases.append(
            {
                "room": room,
                "query_index": row["query_index"],
                "query_id": row["query_id"],
                "candidate_indices_sha256": row["candidate_indices_sha256"],
                "omp_result_sha256": row["omp_sha256"],
                "agree_result_sha256": row["agree_sha256"],
                "mesh": mesh_records[room],
                "ground_truth_global_m": row["models"]["fem_omp"]["source_global"],
                "receiver_global_m": row["models"]["fem_omp"]["receiver_global"],
                "fem_omp_prediction_global_m": omp_metric["prediction_global"],
                "fem_omp_error_m": float(omp_metric["localization_error_m"]),
                "fem_agree_prediction_global_m": agree_metric["prediction_global"],
                "fem_agree_error_m": float(agree_metric["localization_error_m"]),
            }
        )

    payload = {
        "schema_version": 1,
        "methods": {
            "fem_omp": "FEM-Sabine Depth-AABB with Room-Helps OMP",
            "fem_agree": "FEM-Sabine Depth-AABB with frozen AGREE K=8",
        },
        "selection": {
            "rooms": list(ROOMS),
            "reference": "five prior figure cases plus strict-only Apartments 42 q5271",
            "error_used_for_selection": False,
        },
        "rendering": {
            "style": "solid_cutaway",
            "official_mesh_display_only": True,
            "figure_titles_drawn": False,
            "panel_descriptions_drawn": False,
            "dpi": args.dpi,
        },
        "images": [omp_image.name, agree_image.name, overlay_image.name],
        "cases": cases,
    }
    payload["sha256"] = base.canonical_sha256(payload)
    base.atomic_text(
        output_dir / "visualization_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    base.atomic_text(output_dir / "README.md", render_readme(cases))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "images": payload["images"],
                "sha256": payload["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
