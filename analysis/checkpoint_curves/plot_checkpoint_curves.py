#!/usr/bin/env python3
"""Plot yaw-zero checkpoint performance for FA B-F and Vanilla P1."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path("/home/yixunhu/codespace/FLAC")
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
STEPS = tuple(range(2500, 40001, 2500))
METRICS = {
    "T60": "T60 error (%) ↓",
    "C50": "C50 error (dB) ↓",
    "EDT": "EDT error (ms) ↓",
    "RIR_to_GT_RIR_R@1": "RIR→GT R@1 (%) ↑",
    "RIR_to_GT_RIR_R@5": "RIR→GT R@5 (%) ↑",
    "RIR_to_GT_RIR_R@10": "RIR→GT R@10 (%) ↑",
}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def one(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one path for {pattern}, found {len(paths)}")
    return paths[0]


def path_for(arm: str, k: int, step: int) -> Path:
    if arm == "FA":
        base = "outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
        if k == 1:
            if step == 25000:
                suffix = f"*step={step}_metrics_1_1.0_exp10_BFmc_K1_s42_fa_invariant_a4.json"
            elif step == 40000:
                suffix = f"*step={step}_metrics_1_1.0_exp10_BF40_K1_s42_fa_invariant_a4.json"
            else:
                suffix = f"*step={step}_metrics_1_1.0_curve0_FA_S{step}_K1_s42_fa_invariant_a4.json"
        elif step <= 22500 or step == 27500:
            suffix = f"*step={step}_metrics_1_1.0_exp10_A4_FA_S{step}_fa_invariant_a4.json"
        elif step == 25000:
            suffix = f"*step={step}_metrics_1_1.0_exp10_BFmc_K8_s42_fa_invariant_a4.json"
        elif step < 40000:
            suffix = f"*step={step}_metrics_1_1.0_exp10_BFpre_S{step}_fa_invariant_a4.json"
        else:
            suffix = "*step=40000_metrics_1_1.0_exp11_C4backfill_S40000_s42_K8_fa_invariant_a4.json"
    else:
        base = "outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints"
        if k == 1:
            if step == 40000:
                suffix = "*step=40000_metrics_1_1.0_exp07_P140_K1_s42.json"
            else:
                suffix = f"*step={step}_metrics_1_1.0_curve0_VAN_S{step}_K1_s42.json"
        elif step in (10000, 20000, 30000, 40000):
            suffix = f"*step={step}_metrics_1_1.0_exp07_P1_screen_S{step}_ema.json"
        else:
            suffix = f"*step={step}_metrics_1_1.0_exp10_A4_VAN_S{step}.json"
    return one(f"{base}/{suffix}")


def collect() -> list[dict]:
    rows = []
    for arm in ("VAN", "FA"):
        for k in (1, 8):
            for step in STEPS:
                path = path_for(arm, k, step)
                doc = load(path)
                if float(doc.get("rotate_deg", 0.0)) != 0.0:
                    raise RuntimeError(f"nonzero yaw in {path}")
                expected = "fa_invariant" if arm == "FA" else "vanilla"
                if doc.get("cond_method") != expected:
                    raise RuntimeError(f"conditioning mismatch in {path}")
                row = {"method": "Per-angle FA" if arm == "FA" else "Vanilla FLAC", "K": k, "step": step, "source_json": str(path)}
                row.update({metric: float(doc["metrics"][metric]) for metric in METRICS})
                rows.append(row)
    return rows


def main() -> None:
    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "checkpoint_performance_seed42.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    font_dir = HERE.parent / "yaw_robustness/fonts"
    regular = font_dir / "texgyrepagella-regular.otf"
    bold = font_dir / "texgyrepagella-bold.otf"
    for path in (regular, bold):
        font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(regular)).get_name()
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 1.0})

    colors = {"Vanilla FLAC": "#2E64D6", "Per-angle FA": "#EF762F"}
    styles = {1: ("--", "o", "white"), 8: ("-", "s", None)}
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 5.5), sharex=True)
    for ax, (metric, ylabel) in zip(axes.flat, METRICS.items()):
        for method in ("Vanilla FLAC", "Per-angle FA"):
            for k in (1, 8):
                pts = [row for row in rows if row["method"] == method and row["K"] == k]
                line, marker, face = styles[k]
                ax.plot(
                    [row["step"] / 1000 for row in pts],
                    [row[metric] for row in pts],
                    color=colors[method], linestyle=line, marker=marker,
                    markerfacecolor=face or colors[method], markeredgecolor=colors[method],
                    linewidth=1.7, markersize=4.2, label=f"{method}, K={k}",
                )
        ax.set_title(ylabel, fontsize=11.5, pad=5)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Training steps (k)", fontsize=10.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Performance vs. Training Steps", fontsize=17, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "AcousticRooms unseen test · yaw = 0° · evaluation seed 42", ha="center", fontsize=10.5)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=4, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.79, wspace=0.25, hspace=0.38)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"checkpoint_performance_seed42.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {csv_path} and checkpoint_performance_seed42.{{png,pdf}}")


if __name__ == "__main__":
    main()

