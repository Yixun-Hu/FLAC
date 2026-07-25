#!/usr/bin/env python3
"""Plot the exp06 subset544 C16 comparison from the canonical metric JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from summarize_eval import C16_ANGLES, Condition, scan_records


VARIANTS = ("linear", "cnn")
LABELS = {"linear": "CylViT-MLP/Linear", "cnn": "CylViT-CNN"}
COLORS = {"linear": "#4C78A8", "cnn": "#E45756"}
MARKERS = {"linear": "o", "cnn": "s"}

ACOUSTIC_METRICS = (
    ("T60", "T60 error (%) ↓"),
    ("C50", "C50 error (dB) ↓"),
    ("EDT", "EDT error (ms) ↓"),
)
RETRIEVAL_METRICS = (
    ("RIR_to_GT_RIR_R@1", "R@1 (%) ↑"),
    ("RIR_to_GT_RIR_R@5", "R@5 (%) ↑"),
    ("RIR_to_GT_RIR_R@10", "R@10 (%) ↑"),
)


def parse_args() -> argparse.Namespace:
    exp_dir = Path(__file__).resolve().parent
    flac_root = exp_dir.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-step", type=int, default=20000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--linear-dir",
        type=Path,
        default=flac_root / "outputs_FLAC/exp06_cylvit_pe_linear_trainS42/eval_subset544",
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=flac_root / "outputs_FLAC/exp06_cylvit_pe_cnn_patchlocal_trainS42/eval_subset544",
    )
    parser.add_argument("--output-dir", type=Path, default=exp_dir / "figures")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def collect_series(args: argparse.Namespace) -> dict[str, dict[str, list[float]]]:
    directories = {
        "linear": args.linear_dir.expanduser().resolve(),
        "cnn": args.cnn_dir.expanduser().resolve(),
    }
    records, notes = scan_records(directories, args.train_seed, args.train_step)
    if notes:
        raise RuntimeError("; ".join(notes))

    metric_keys = [key for key, _ in ACOUSTIC_METRICS + RETRIEVAL_METRICS]
    series: dict[str, dict[str, list[float]]] = {}
    missing: list[Condition] = []
    for variant in VARIANTS:
        series[variant] = {key: [] for key in metric_keys}
        for yaw in C16_ANGLES:
            condition = Condition(
                variant=variant,
                train_seed=args.train_seed,
                train_step=args.train_step,
                k=1,
                eval_seed=args.eval_seed,
                yaw=yaw,
            )
            record = records.get(condition)
            if record is None:
                missing.append(condition)
                continue
            for key in metric_keys:
                series[variant][key].append(record.metrics[key])
    if missing:
        details = ", ".join(
            f"{row.variant}/yaw{row.yaw:g}" for row in missing
        )
        raise RuntimeError(f"Missing C16 records: {details}")
    return series


def plot_group(
    series: dict[str, dict[str, list[float]]],
    metrics: tuple[tuple[str, str], ...],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), sharex=True)
    yaw_values = list(C16_ANGLES)

    for axis, (metric_key, ylabel) in zip(axes, metrics):
        for variant in VARIANTS:
            axis.plot(
                yaw_values,
                series[variant][metric_key],
                color=COLORS[variant],
                marker=MARKERS[variant],
                markersize=4.5,
                linewidth=1.8,
                label=LABELS[variant],
            )
        axis.set_xlabel("Yaw angle (degrees)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-4, 341.5)
        axis.set_xticks(yaw_values)
        axis.set_xticklabels(
            [str(int(yaw)) if float(yaw).is_integer() else f"{yaw:g}" for yaw in yaw_values],
            rotation=45,
            ha="right",
            fontsize=7.5,
        )
        axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, fontsize=13, y=1.07)
    fig.text(
        0.5,
        -0.035,
        "20k checkpoint · K=1 · generation seed 42 · deterministic subset544",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    series = collect_series(args)
    output_dir = args.output_dir.expanduser().resolve()
    plot_group(
        series,
        ACOUSTIC_METRICS,
        "C16 Yaw Sweep — Acoustic Metrics",
        output_dir / "c16_acoustic_metrics_linear_vs_cnn_subset544.png",
        args.dpi,
    )
    plot_group(
        series,
        RETRIEVAL_METRICS,
        "C16 Yaw Sweep — Retrieval Metrics",
        output_dir / "c16_retrieval_metrics_linear_vs_cnn_subset544.png",
        args.dpi,
    )


if __name__ == "__main__":
    main()
