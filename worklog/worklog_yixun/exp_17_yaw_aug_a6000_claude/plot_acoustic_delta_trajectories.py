#!/usr/bin/env python3
"""Plot K=8 C4 spread trajectories for all reported acoustic metrics."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ARM_TABLES = {
    "P1 Vanilla": ROOT / "results_yaw_aug_a6000_p1ctrl_grid_table.md",
    "Yaw-Aug P1": ROOT / "results_yaw_aug_a6000_c4_grid_table.md",
}
COLORS = {"P1 Vanilla": "#D55E00", "Yaw-Aug P1": "#0072B2"}
METRICS = (
    ("ΔT60", "T60 spread"),
    ("ΔEDT", "EDT spread"),
    ("ΔC50", "C50 spread"),
    ("ΔFD", "FD spread"),
)


def read_k8_rows(path: Path) -> list[dict[str, float]]:
    """Read K=8 rows from an exp_17 generated Markdown grid table."""
    header: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip("|").split("|")]
        if cells[:2] == ["step", "K"]:
            header = cells
            continue
        if header is None or not cells[0].isdigit() or cells[1] != "8":
            continue
        rows.append({name: float(value) for name, value in zip(header, cells)})
    if len(rows) != 16:
        raise ValueError(f"Expected 16 K=8 rows in {path}, found {len(rows)}")
    return rows


def main() -> None:
    arm_rows = {label: read_k8_rows(path) for label, path in ARM_TABLES.items()}
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True)

    for ax, (metric, title) in zip(axes.flat, METRICS):
        for label, rows in arm_rows.items():
            steps = [row["step"] for row in rows]
            values = [row[metric] for row in rows]
            ax.plot(
                steps,
                values,
                color=COLORS[label],
                marker="o",
                linewidth=2.25,
                markersize=4.5,
                label=label,
            )
            ax.annotate(
                f"{values[-1]:.3f}",
                xy=(steps[-1], values[-1]),
                xytext=(-7, 7),
                textcoords="offset points",
                ha="right",
                color=COLORS[label],
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=13, fontweight="semibold")
        ax.set_ylabel(f"{metric} (max − min)", fontsize=10.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(1500, 41000)
        ax.set_xticks([2500, 10000, 20000, 30000, 40000])
        ax.set_xticklabels(["2.5k", "10k", "20k", "30k", "40k"])
        ax.grid(color="#D8D8D8", linewidth=0.75, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Training step", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=11,
    )
    fig.suptitle(
        "Yaw Sensitivity of Acoustic Metrics over Training (K=8)",
        fontsize=16,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.0, w_pad=1.5)

    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"p1_vs_yawaug_acoustic_spreads_k8.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
