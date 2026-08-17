#!/usr/bin/env python3
"""Plot K=8 C4 spreads for all seven exp_17 performance metrics."""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
ARM_DIRS = {
    "P1 Vanilla": REPO_ROOT / "outputs_FLAC/exp17_P1CTRL_roteval",
    "Yaw-Aug P1": REPO_ROOT / "outputs_FLAC/exp17_YAWAUG_roteval",
}
COLORS = {"P1 Vanilla": "#D55E00", "Yaw-Aug P1": "#0072B2"}
METRICS = (
    ("T60", "ΔT60"),
    ("EDT", "ΔEDT"),
    ("C50", "ΔC50"),
    ("FD", "ΔFD"),
    ("RIR_to_GT_RIR_R@1", "ΔR@1"),
    ("RIR_to_GT_RIR_R@5", "ΔR@5"),
    ("RIR_to_GT_RIR_R@10", "ΔR@10"),
)
STEPS = list(range(2500, 40001, 2500))
ANGLES = (0, 90, 180, 270)
CELL_PATTERN = re.compile(r"_S(?P<step>\d+)_K8_rot(?P<angle>0|90|180|270)_seed42(?:_rot\d+)?\.json$")


def read_spreads(directory: Path) -> dict[str, list[float]]:
    """Calculate max-minus-min across C4 from the raw K=8, seed-42 cells."""
    cells: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for path in directory.glob("*.json"):
        match = CELL_PATTERN.search(path.name)
        if not match:
            continue
        step = int(match.group("step"))
        angle = int(match.group("angle"))
        if angle in cells[step]:
            raise ValueError(f"Duplicate step/angle cell: {path}")
        cells[step][angle] = json.loads(path.read_text(encoding="utf-8"))["metrics"]

    expected = {(step, angle) for step in STEPS for angle in ANGLES}
    observed = {(step, angle) for step, orbit in cells.items() for angle in orbit}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Invalid grid in {directory}: missing={missing}, extra={extra}")

    trajectories: dict[str, list[float]] = {}
    for raw_name, _ in METRICS:
        trajectories[raw_name] = []
        for step in STEPS:
            values = [float(cells[step][angle][raw_name]) for angle in ANGLES]
            trajectories[raw_name].append(max(values) - min(values))
    return trajectories


def main() -> None:
    arms = {label: read_spreads(directory) for label, directory in ARM_DIRS.items()}
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.3), sharex=True)

    for ax, (raw_name, display_name) in zip(axes.flat, METRICS):
        for label, trajectories in arms.items():
            values = trajectories[raw_name]
            ax.plot(
                STEPS,
                values,
                color=COLORS[label],
                marker="o",
                linewidth=2.15,
                markersize=4.2,
                label=label,
            )
            ax.annotate(
                f"{values[-1]:.3f}",
                xy=(STEPS[-1], values[-1]),
                xytext=(-6, 6),
                textcoords="offset points",
                ha="right",
                color=COLORS[label],
                fontsize=8.5,
                fontweight="bold",
            )

        metric_name = display_name.removeprefix("Δ")
        ax.set_title(display_name, fontsize=13, fontweight="semibold")
        ax.set_ylabel(f"{metric_name} spread (max − min)", fontsize=9.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(1500, 41000)
        ax.set_xticks([2500, 10000, 20000, 30000, 40000])
        ax.set_xticklabels(["2.5k", "10k", "20k", "30k", "40k"])
        ax.grid(color="#D8D8D8", linewidth=0.75, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :3]:
        ax.set_xlabel("Training step", fontsize=10.5)

    legend_ax = axes[-1, -1]
    legend_ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        fontsize=12,
        title="Training arm",
        title_fontsize=11,
    )
    legend_ax.text(
        0.5,
        0.24,
        "K=8 · eval seed 42\nC4 yaw spread: 0°, 90°, 180°, 270°\nLower spread = greater yaw stability",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
        linespacing=1.5,
        transform=legend_ax.transAxes,
    )

    fig.suptitle(
        "Yaw Sensitivity over Training: P1 Vanilla vs Yaw-Aug P1",
        fontsize=17,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.1, w_pad=1.5)

    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"p1_vs_yawaug_all_seven_spreads_k8.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
