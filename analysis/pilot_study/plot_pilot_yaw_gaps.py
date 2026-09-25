#!/usr/bin/env python3
"""Pilot-study figure: T60 / EDT differences under C4 conditioning rotation.

Data source (verbatim): exp-09 checkout,
worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/yaw_noninvariance_results.md
(released FLAC_EMA, unseen K=1 split, 6337 items / 17 rooms, seed 42).

Two panels (T60, EDT). For each rotation angle, two bars:
  - "accuracy degradation": Metric-2 error minus the alpha=0 baseline error (vs GT)
  - "prediction shift": Metric-1 gap of P_alpha vs P_0 (no GT involved)
The contrast IS the finding: predictions move ~5x more than net accuracy loses,
because rotation scatters predictions around GT in both directions.

Outputs analysis/pilot_study/generated_paper/pilot_yaw_gaps.{pdf,png} + a CSV
of the plotted values for the paper record.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANGLES = [90, 180, 270]

# Metric-2 (vs GT): degradation relative to alpha=0 baseline (T60 %, EDT ms).
DEG_T60 = [10.38 - 9.99, 10.72 - 9.99, 10.44 - 9.99]
DEG_EDT = [43.58 - 40.11, 46.39 - 40.11, 44.07 - 40.11]
# Metric-1 (vs P_0): invariance gap of the prediction itself.
GAP_T60 = [3.34, 3.33, 3.41]
GAP_EDT = [18.65, 20.11, 19.98]
# exp_01 single-eval noise floor (1 sigma).
SIGMA_T60 = 0.04
SIGMA_EDT = 0.37

C_DEG = "#0072B2"  # Okabe-Ito blue
C_GAP = "#E69F00"  # Okabe-Ito orange

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_paper")


def _panel(ax, deg, gap, sigma, title, ylabel):
    x = range(len(ANGLES))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], deg, w, color=C_DEG,
                label="accuracy degradation (vs GT)")
    b2 = ax.bar([i + w / 2 for i in x], gap, w, color=C_GAP,
                label=r"prediction shift (vs $P_0$)")
    ax.axhspan(0, 2 * sigma, color="0.85", zorder=0)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2,
                        r.get_height()), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{a}°" for a in ANGLES])
    ax.set_xlabel("conditioning yaw rotation")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    return b1, b2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.1))
    _panel(ax1, DEG_T60, GAP_T60, SIGMA_T60, "T60", r"$\Delta$T60 (%)")
    b1, b2 = _panel(ax2, DEG_EDT, GAP_EDT, SIGMA_EDT, "EDT", r"$\Delta$EDT (ms)")
    fig.legend(handles=[b1, b2], loc="upper center", ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"pilot_yaw_gaps.{ext}"), dpi=300,
                    bbox_inches="tight")
    with open(os.path.join(OUT_DIR, "pilot_yaw_gaps.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["angle_deg", "deg_T60_pct", "gap_T60_pct",
                       "deg_EDT_ms", "gap_EDT_ms"])
        for i, a in enumerate(ANGLES):
            wcsv.writerow([a, round(DEG_T60[i], 2), GAP_T60[i],
                           round(DEG_EDT[i], 2), GAP_EDT[i]])
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
