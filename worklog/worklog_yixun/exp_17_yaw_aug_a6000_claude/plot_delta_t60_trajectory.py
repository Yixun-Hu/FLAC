#!/usr/bin/env python3
"""Plot the K=8 C4 Delta-T60 trajectory for P1 and Yaw-Aug P1."""

from pathlib import Path

import matplotlib.pyplot as plt


STEPS = [
    2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000,
    22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000,
]
P1_DELTA_T60 = [
    0.150, 0.059, 0.086, 0.146, 0.133, 0.128, 0.258, 0.349,
    0.362, 0.447, 0.481, 0.544, 0.482, 0.627, 0.500, 0.903,
]
YAWAUG_DELTA_T60 = [
    0.036, 0.024, 0.061, 0.056, 0.055, 0.118, 0.177, 0.019,
    0.143, 0.057, 0.054, 0.030, 0.034, 0.160, 0.090, 0.075,
]


def main() -> None:
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.5))

    ax.plot(
        STEPS,
        P1_DELTA_T60,
        color="#D55E00",
        marker="o",
        linewidth=2.4,
        markersize=5,
        label="P1 Vanilla",
    )
    ax.plot(
        STEPS,
        YAWAUG_DELTA_T60,
        color="#0072B2",
        marker="o",
        linewidth=2.4,
        markersize=5,
        label="Yaw-Aug P1",
    )

    ax.annotate(
        "0.903",
        xy=(STEPS[-1], P1_DELTA_T60[-1]),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        color="#D55E00",
        fontweight="bold",
    )
    ax.annotate(
        "0.075",
        xy=(STEPS[-1], YAWAUG_DELTA_T60[-1]),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        color="#0072B2",
        fontweight="bold",
    )

    ax.set_title("Yaw Sensitivity over Training (K=8)", fontsize=15, pad=12)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel(r"$\Delta$T60 across yaw angles (max $-$ min)", fontsize=12)
    ax.set_xlim(1500, 41000)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([2500, 10000, 20000, 30000, 40000])
    ax.set_xticklabels(["2.5k", "10k", "20k", "30k", "40k"])
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    ax.grid(axis="both", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"p1_vs_yawaug_delta_t60_k8.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
