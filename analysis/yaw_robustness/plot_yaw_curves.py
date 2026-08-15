#!/usr/bin/env python3
"""Plot matched-budget yaw robustness curves from the exp_10 A6 JSONs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


LEGACY_ANGLES = (0, 45, 90)
C4_ANGLES = (0, 90, 180, 270)
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("Vanilla P1@40k", "FA B-F@40k")

FONT_DIR = Path(__file__).resolve().parent / "fonts"
PAGELLA_REGULAR = FONT_DIR / "texgyrepagella-regular.otf"
PAGELLA_BOLD = FONT_DIR / "texgyrepagella-bold.otf"
for font_path in (PAGELLA_REGULAR, PAGELLA_BOLD):
    if not font_path.is_file():
        raise FileNotFoundError(f"Missing bundled plotting font: {font_path}")
    font_manager.fontManager.addfont(str(font_path))

PLOT_FONT = font_manager.FontProperties(fname=str(PAGELLA_REGULAR)).get_name()
plt.rcParams.update(
    {
        "font.family": PLOT_FONT,
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ACOUSTIC_METRICS = {
    "T60": ("T60 error (%) ↓", 3),
    "C50": ("C50 error (dB) ↓", 3),
    "EDT": ("EDT error (ms) ↓", 3),
    "FD": ("FD ↓", 4),
}

RETRIEVAL_METRICS = {
    "RIR_to_GT_RIR_R@1": ("RIR→GT R@1 (%) ↑", 3),
    "RIR_to_GT_RIR_R@5": ("RIR→GT R@5 (%) ↑", 3),
    "RIR_to_GT_RIR_R@10": ("RIR→GT R@10 (%) ↑", 3),
    "RIR_to_geom_R@1": ("RIR→geometry R@1 (%) ↑", 3),
    "RIR_to_geom_R@5": ("RIR→geometry R@5 (%) ↑", 3),
    "RIR_to_geom_R@10": ("RIR→geometry R@10 (%) ↑", 3),
}

C4_FIGURE_METRICS = {
    "T60": ("T60 error (%) ↓", 3),
    "C50": ("C50 error (dB) ↓", 3),
    "EDT": ("EDT error (ms) ↓", 3),
    "RIR_to_GT_RIR_R@1": ("RIR→GT R@1 (%) ↑", 3),
    "RIR_to_GT_RIR_R@5": ("RIR→GT R@5 (%) ↑", 3),
    "RIR_to_GT_RIR_R@10": ("RIR→GT R@10 (%) ↑", 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flac-root",
        type=Path,
        default=Path("/home/yixunhu/codespace/FLAC"),
        help="Path to the FLAC checkout containing outputs_FLAC.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "c4", "all"),
        default="legacy",
        help="legacy plots 0/45/90; c4 plots 0/90/180/270 after those evals exist.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return json.load(handle)


def paths_for(flac_root: Path, method: str, k: int, angle: int) -> list[Path]:
    if method == "FA B-F@40k":
        root = (
            flac_root
            / "outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
        )
        if angle == 0:
            paths = [
                root
                / (
                    "epoch=8-step=40000_metrics_1_1.0_"
                    f"exp10_BF40_K{k}_s{seed}_fa_invariant_a4.json"
                )
                for seed in SEEDS
                if not (k == 8 and seed == 42)
            ]
            if k == 8:
                paths.insert(
                    0,
                    root
                    / (
                        "epoch=8-step=40000_metrics_1_1.0_"
                        "exp11_C4backfill_S40000_s42_K8_fa_invariant_a4.json"
                    ),
                )
            return paths
        dataset_token = "unseeneval" if k == 8 else "unseeneval_1"
        campaign = "a6" if angle in (45, 90) else "a6c4"
        return [
            root
            / (
                "epoch=8-step=40000_metrics_1_1.0_"
                f"{campaign}_FA40_rot{angle}_{dataset_token}_s{seed}_"
                f"fa_invariant_a4_rot{angle}.json"
            )
            for seed in SEEDS
        ]

    root = flac_root / "outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints"
    if angle == 0:
        paths = [
            root
            / (
                "epoch=8-step=40000_metrics_1_1.0_"
                f"exp07_P140_K{k}_s{seed}.json"
            )
            for seed in SEEDS
            if not (k == 8 and seed == 42)
        ]
        if k == 8:
            paths.insert(
                0,
                root
                / "epoch=8-step=40000_metrics_1_1.0_exp07_P1_screen_S40000_ema.json",
            )
        return paths
    dataset_token = "unseeneval" if k == 8 else "unseeneval_1"
    campaign = "a6" if angle in (45, 90) else "a6c4"
    return [
        root
        / (
            "epoch=8-step=40000_metrics_1_1.0_"
            f"{campaign}_VAN40_rot{angle}_{dataset_token}_s{seed}_rot{angle}.json"
        )
        for seed in SEEDS
    ]


def collect(flac_root: Path, angles: tuple[int, ...]) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    all_metrics = {**ACOUSTIC_METRICS, **RETRIEVAL_METRICS}

    for method in METHODS:
        for k in (1, 8):
            for angle in angles:
                paths = paths_for(flac_root, method, k, angle)
                if len(paths) != len(SEEDS):
                    raise RuntimeError(f"Expected five files for {method}, K={k}, {angle}°")
                docs = [load_json(path) for path in paths]
                observed_seeds = [int(doc.get("seed", seed)) for doc, seed in zip(docs, SEEDS)]
                if observed_seeds != list(SEEDS):
                    raise RuntimeError(
                        f"Seed mismatch for {method}, K={k}, {angle}°: {observed_seeds}"
                    )
                for seed, path, doc in zip(SEEDS, paths, docs):
                    observed_angle = float(doc.get("rotate_deg", 0.0))
                    if observed_angle != angle:
                        raise RuntimeError(f"Angle mismatch in {path}: {observed_angle}")
                    row = {
                        "method": method,
                        "K": k,
                        "angle_deg": angle,
                        "seed": seed,
                        "source_json": str(path),
                    }
                    row.update({metric: doc["metrics"][metric] for metric in all_metrics})
                    raw_rows.append(row)

                for metric in all_metrics:
                    values = np.asarray([doc["metrics"][metric] for doc in docs], dtype=float)
                    summary_rows.append(
                        {
                            "method": method,
                            "K": k,
                            "angle_deg": angle,
                            "metric": metric,
                            "n": values.size,
                            "mean": values.mean(),
                            "std": values.std(ddof=1),
                        }
                    )
    return raw_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup(
    summary_rows: list[dict],
    method: str,
    k: int,
    metric: str,
    angles: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in summary_rows
        if row["method"] == method and row["K"] == k and row["metric"] == metric
    ]
    rows.sort(key=lambda row: row["angle_deg"])
    if [row["angle_deg"] for row in rows] != list(angles):
        raise RuntimeError(f"Incomplete summary for {method}, K={k}, {metric}")
    return (
        np.asarray([row["mean"] for row in rows]),
        np.asarray([row["std"] for row in rows]),
    )


def plot_grid(
    summary_rows: list[dict],
    metrics: dict[str, tuple[str, int]],
    shape: tuple[int, int],
    output_stem: Path,
    angles: tuple[int, ...],
) -> None:
    colors = {"Vanilla P1@40k": "#2E63E7", "FA B-F@40k": "#F2762E"}
    markers = {"Vanilla P1@40k": "o", "FA B-F@40k": "s"}
    fig, axes = plt.subplots(
        *shape,
        figsize=(3.85 * shape[1], 2.5 * shape[0]),
        squeeze=False,
    )

    for ax, (metric, (ylabel, _)) in zip(axes.flat, metrics.items()):
        for method in METHODS:
            for k, linestyle, fillstyle in ((1, "--", "none"), (8, "-", "full")):
                means, stds = lookup(summary_rows, method, k, metric, angles)
                short_method = "Vanilla" if method.startswith("Vanilla") else "FA B-F"
                label = f"{short_method}, K={k}"
                ax.errorbar(
                    angles,
                    means,
                    yerr=stds,
                    color=colors[method],
                    marker=markers[method],
                    markerfacecolor=("white" if fillstyle == "none" else colors[method]),
                    markeredgecolor=colors[method],
                    markeredgewidth=1.35,
                    linestyle=linestyle,
                    linewidth=2.15,
                    markersize=6.6,
                    elinewidth=1.05,
                    capsize=2.8,
                    capthick=1.05,
                    label=label,
                )
        ax.set_title(ylabel, fontsize=13.5, pad=8)
        ax.set_xticks(angles)
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.0, length=4.5)
        ax.margins(x=0.06)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    fig.suptitle(
        "Yaw Robustness Generalization Test",
        y=0.99,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "(5 seeds, @40k training steps)",
        ha="center",
        fontsize=12,
    )
    fig.supxlabel("Yaw Rotation (degrees)", y=0.015, fontsize=13.5)
    # Pack the artists into a short outer canvas without reducing the axes
    # themselves; this avoids the large white bands above and below the grid.
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.105,
        top=0.735,
        wspace=0.18,
        hspace=0.72,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, include_c4: bool) -> None:
    c4_text = (
        " `yaw_c4_metrics` is the single 2×3 figure for the measured C4 angles "
        "0°/90°/180°/270°." if include_c4 else ""
    )
    path.write_text(
        """# Matched-budget yaw robustness curves\n\n"
        "Comparison: legacy per-angle FA B-F@40k versus Vanilla P1@40k on the "
        "full AR unseen split. Each point is the mean over evaluation seeds 42–46; "
        "error bars show sample standard deviation (ddof=1). Both K=1 and K=8 are "
        "shown. Only 0°, 45°, and 90° were evaluated; connecting lines do not imply "
        "measurements at intermediate angles.\n\n"
        "`yaw_acoustic_metrics` contains T60, C50, EDT, and FD. "
        "`yaw_retrieval_metrics` contains all six reported retrieval metrics. "
        "Invalid T60 is omitted because it is identically zero. The CSV files contain "
        "the plotted summaries and all source JSON paths." + c4_text + "\n"
        """,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("legacy", "all"):
        raw_rows, summary_rows = collect(args.flac_root.resolve(), LEGACY_ANGLES)
        write_csv(args.output_dir / "yaw_metrics_raw.csv", raw_rows)
        write_csv(args.output_dir / "yaw_metrics_summary.csv", summary_rows)
        plot_grid(
            summary_rows,
            ACOUSTIC_METRICS,
            (2, 2),
            args.output_dir / "yaw_acoustic_metrics",
            LEGACY_ANGLES,
        )
        plot_grid(
            summary_rows,
            RETRIEVAL_METRICS,
            (2, 3),
            args.output_dir / "yaw_retrieval_metrics",
            LEGACY_ANGLES,
        )
    if args.mode in ("c4", "all"):
        c4_raw, c4_summary = collect(args.flac_root.resolve(), C4_ANGLES)
        write_csv(args.output_dir / "yaw_c4_metrics_raw.csv", c4_raw)
        write_csv(args.output_dir / "yaw_c4_metrics_summary.csv", c4_summary)
        plot_grid(
            c4_summary,
            C4_FIGURE_METRICS,
            (2, 3),
            args.output_dir / "yaw_c4_metrics",
            C4_ANGLES,
        )
    write_readme(args.output_dir / "README.md", args.mode in ("c4", "all"))
    print(f"Wrote plots and CSVs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
