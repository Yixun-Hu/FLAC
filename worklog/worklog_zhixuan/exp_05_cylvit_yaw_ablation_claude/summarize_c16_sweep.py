"""Summarize and visualize the C16 yaw-sweep evaluation (SimpleViT vs CylViT).

Reads the per-angle metrics JSONs written by ``run_c16_eval.sh``, then writes:

- ``c16_eval_results.md``  — absolute metrics, deltas from yaw 0, and a yaw
  robustness summary (mean |delta|, worst delta, std over angles) per model.
- ``c16_eval_all_metrics.md`` — optional raw all-metrics tables, one 16-angle
  table per model.
- ``figures/c16_absolute_metrics.{png,pdf}`` — metric vs yaw angle, both models.
- ``figures/c16_delta_from_yaw0.{png,pdf}`` — delta from each model's own yaw 0.

Missing angles are tolerated (reported as absent, skipped in plots), so the
script can run on a partially finished sweep. Depth-agnostic: locates the repo
root by walking up from this file until ``outputs_FLAC`` is found.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-c16-summary")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for parent in EXP_DIR.parents:
        if (parent / "outputs_FLAC").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repo root (no outputs_FLAC found upward)")


ANGLES = [i * 22.5 for i in range(16)]
MODELS = ("simplevit", "cylvit")
MODEL_LABELS = {"simplevit": "SimpleViT", "cylvit": "CylViT"}
# Reference categorical palette, fixed slot order (slots 1 and 2), with marker
# shape as the secondary (non-color) encoding.
MODEL_COLORS = {"simplevit": "#2a78d6", "cylvit": "#1baf7a"}
MODEL_MARKERS = {"simplevit": "o", "cylvit": "s"}

# (json key, display label, lower_is_better)
METRICS = [
    ("T60", "T60 ↓", True),
    ("C50", "C50 ↓", True),
    ("EDT", "EDT ↓", True),
    ("FD", "FD ↓", True),
    ("RIR_to_GT_RIR_R@1", "GT R@1 ↑", False),
    ("RIR_to_GT_RIR_R@5", "GT R@5 ↑", False),
    ("RIR_to_GT_RIR_R@10", "GT R@10 ↑", False),
]

# Full record exported separately from the compact summary above.
ALL_METRICS = [
    ("T60", "T60 (%) ↓"),
    ("Invalid T60", "Invalid T60 ↓"),
    ("C50", "C50 (dB) ↓"),
    ("EDT", "EDT (ms) ↓"),
    ("FD", "FD ↓"),
    ("RIR_to_GT_RIR_R@1", "GT R@1 (%) ↑"),
    ("RIR_to_GT_RIR_R@5", "GT R@5 (%) ↑"),
    ("RIR_to_GT_RIR_R@10", "GT R@10 (%) ↑"),
    ("RIR_to_geom_R@1", "Geom R@1 (%) ↑"),
    ("RIR_to_geom_R@5", "Geom R@5 (%) ↑"),
    ("RIR_to_geom_R@10", "Geom R@10 (%) ↑"),
]


def angle_str(angle: float) -> str:
    return str(int(angle)) if float(angle).is_integer() else str(angle)


def metrics_path(root: Path, model: str, ckpt_stem: str, total_label: str, angle: float) -> Path:
    save_dir = root / f"outputs_FLAC/exp05_{model}_phase3_total30k_s42"
    suffix = "" if angle == 0 else f"_rot{int(angle)}"
    name = f"{ckpt_stem}_metrics_1_1.0_exp05_{model}_c16_total{total_label}_yaw{angle_str(angle)}{suffix}.json"
    return save_dir / name


def load_sweep(root: Path, ckpt_stem: str, total_label: str) -> dict:
    data = {model: {} for model in MODELS}
    for model in MODELS:
        for angle in ANGLES:
            path = metrics_path(root, model, ckpt_stem, total_label, angle)
            if path.is_file():
                with open(path) as f:
                    data[model][angle] = json.load(f)["metrics"]
    return data


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "-"


def build_markdown(data: dict, ckpt_stem: str, total_label: str) -> str:
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# C16 Yaw-Sweep Results — total {total_label} milestone",
        "",
        f"Generated: {now}",
        "",
        f"Checkpoint: `outputs_FLAC/exp05_{{model}}_phase3_total30k_s42/{ckpt_stem}.ckpt` "
        "(matched milestone for both models). Eval: unseen rooms, K=1, steps=1, "
        "cfg_scale=1.0, seed 42. All 16 patch-grid yaw angles (n·22.5°) — the C16 "
        "group on which CylViT is equivariant by construction.",
        "",
        "Lower is better for T60/C50/EDT/FD; higher is better for GT R@1/R@5/R@10.",
        "",
        "## Absolute Metrics",
        "",
        "| Model | Yaw | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    for model in MODELS:
        for angle in ANGLES:
            row = data[model].get(angle)
            cells = [fmt(row.get(key)) if row else "-" for key, _, _ in METRICS]
            lines.append(f"| {MODEL_LABELS[model]} | {angle_str(angle)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Delta From Yaw 0",
        "",
        "For ↓ metrics positive deltas are worse; for ↑ metrics negative deltas are worse.",
        "",
        "| Model | Yaw | " + " | ".join("d" + label.rsplit(" ", 1)[0] for _, label, _ in METRICS) + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    for model in MODELS:
        base = data[model].get(0.0)
        for angle in ANGLES[1:]:
            row = data[model].get(angle)
            if base and row:
                cells = [f"{row[key] - base[key]:+.4f}" for key, _, _ in METRICS]
            else:
                cells = ["-"] * len(METRICS)
            lines.append(f"| {MODEL_LABELS[model]} | {angle_str(angle)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Yaw Robustness Summary (over available angles)",
        "",
        "| Model | Metric | yaw-0 value | mean abs delta | worst delta | std over angles |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        base = data[model].get(0.0)
        for key, label, lower_better in METRICS:
            rows = [data[model][a] for a in ANGLES if a in data[model]]
            if not base or len(rows) < 2:
                lines.append(f"| {MODEL_LABELS[model]} | {label} | {fmt(base.get(key)) if base else '-'} | - | - | - |")
                continue
            values = [r[key] for r in rows]
            deltas = [data[model][a][key] - base[key] for a in ANGLES[1:] if a in data[model]]
            mean_abs = sum(abs(d) for d in deltas) / len(deltas)
            worst = max(deltas) if lower_better else min(deltas)
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            lines.append(
                f"| {MODEL_LABELS[model]} | {label} | {fmt(base[key])} | "
                f"{mean_abs:.4f} | {worst:+.4f} | {std:.4f} |"
            )

    missing = [
        f"{MODEL_LABELS[m]} yaw {angle_str(a)}" for m in MODELS for a in ANGLES if a not in data[m]
    ]
    if missing:
        lines += ["", "## Missing (sweep incomplete)", ""] + [f"- {item}" for item in missing]

    lines += [
        "",
        "## Figures",
        "",
        f"- `figures/c16_{total_label}_absolute_metrics.png`",
        f"- `figures/c16_{total_label}_delta_from_yaw0.png`",
        "",
    ]
    return "\n".join(lines)


def build_all_metrics_markdown(data: dict, ckpt_stem: str, total_label: str) -> str:
    """Render the unaggregated metrics for all 16 angles, split by model."""
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# C16 All-Metrics Record — total {total_label} milestone",
        "",
        f"Generated: {now}",
        "",
        f"Checkpoint: `outputs_FLAC/exp05_{{model}}_phase3_total30k_s42/{ckpt_stem}.ckpt`. "
        "Eval: unseen rooms, K=1, steps=1, cfg_scale=1.0, seed 42.",
        "",
        "Raw per-angle metrics are recorded below without aggregation. Lower is better "
        "for T60/C50/EDT/FD; higher is better for retrieval metrics.",
    ]
    for model in MODELS:
        lines += [
            "",
            f"## {MODEL_LABELS[model]}",
            "",
            "| Yaw (°) | " + " | ".join(label for _, label in ALL_METRICS) + " |",
            "|---:|" + "---:|" * len(ALL_METRICS),
        ]
        for angle in ANGLES:
            row = data[model].get(angle)
            cells = [fmt(row.get(key)) if row else "-" for key, _ in ALL_METRICS]
            lines.append(f"| {angle_str(angle)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Source JSON patterns",
        "",
        f"- SimpleViT: `outputs_FLAC/exp05_simplevit_phase3_total30k_s42/{ckpt_stem}_metrics_1_1.0_exp05_simplevit_c16_total{total_label}_yaw*.json`",
        f"- CylViT: `outputs_FLAC/exp05_cylvit_phase3_total30k_s42/{ckpt_stem}_metrics_1_1.0_exp05_cylvit_c16_total{total_label}_yaw*.json`",
        "",
    ]
    return "\n".join(lines)


def style_axis(ax):
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999999")
    ax.tick_params(colors="#555555", labelsize=8)


def plot_sweep(data: dict, out_dir: Path, delta: bool, total_label: str) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 3.4))
    for ax, (key, label, _) in zip(axes, METRICS):
        for model in MODELS:
            base = data[model].get(0.0)
            angles = [a for a in ANGLES if a in data[model]]
            if delta:
                if not base:
                    continue
                ys = [data[model][a][key] - base[key] for a in angles]
            else:
                ys = [data[model][a][key] for a in angles]
            ax.plot(
                angles, ys,
                color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                linewidth=2, markersize=6, label=MODEL_LABELS[model],
            )
        if delta:
            ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set_title(("Δ " if delta else "") + label, fontsize=10, color="#333333")
        ax.set_xlabel("yaw (deg)", fontsize=8, color="#555555")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlim(-8, 345)
        style_axis(ax)
    axes[0].legend(fontsize=9, frameon=False, loc="best")
    fig.suptitle(
        f"C16 yaw sweep (total {total_label}) — "
        + ("delta from own yaw 0" if delta else "absolute metrics"),
        fontsize=12, color="#333333",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = f"c16_{total_label}_delta_from_yaw0" if delta else f"c16_{total_label}_absolute_metrics"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-stem", default="epoch=4-step=20000")
    parser.add_argument("--total-label", default="25k")
    parser.add_argument(
        "--write-all-metrics",
        action="store_true",
        help="Also write c16_eval_all_metrics_<total-label>.md with two full 16-angle tables",
    )
    args = parser.parse_args()

    root = find_repo_root()
    data = load_sweep(root, args.ckpt_stem, args.total_label)
    n_found = sum(len(v) for v in data.values())
    print(f"[c16-summary] found {n_found}/{len(MODELS) * len(ANGLES)} metrics JSONs")

    md_path = EXP_DIR / f"c16_eval_results_{args.total_label}.md"
    md_path.write_text(build_markdown(data, args.ckpt_stem, args.total_label))
    print(f"[c16-summary] wrote {md_path}")

    if args.write_all_metrics:
        all_metrics_path = EXP_DIR / f"c16_eval_all_metrics_{args.total_label}.md"
        all_metrics_path.write_text(build_all_metrics_markdown(data, args.ckpt_stem, args.total_label))
        print(f"[c16-summary] wrote {all_metrics_path}")

    plot_sweep(data, EXP_DIR / "figures", delta=False, total_label=args.total_label)
    plot_sweep(data, EXP_DIR / "figures", delta=True, total_label=args.total_label)
    print(f"[c16-summary] wrote figures to {EXP_DIR / 'figures'}")


if __name__ == "__main__":
    main()
