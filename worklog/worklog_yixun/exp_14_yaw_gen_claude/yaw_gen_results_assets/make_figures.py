#!/usr/bin/env python3
"""exp_14 — regenerate the results figures from ``results_bundle.json``.

Presentation layer only. Every number drawn here is READ from the collector's
own bundle: no value is computed, rounded-then-reused, or typed in by hand, so a
figure cannot disagree with `yaw_gen_results.md` or the collector's full report.

Two figures, each rendered twice — once for a light page surface and once for a
dark one (a PNG cannot follow `prefers-color-scheme`, and an auto-inverted chart
is not a dark-mode chart):

* ``yaw_gen_dose_response{,_dark}.png`` — the headline. Paired Δ (m_R − m_Z) at
  K=8 against group order, with 95% paired-t CIs (df=4). Three panels: ΔT60 over
  its full range, the same series zoomed onto the saturated arms, and ΔR@1.
  Three panels rather than two y-axes on one: ΔT60 is a percentage-error
  difference and ΔR@1 a recall difference, and a second scale on one frame is
  the standard way to imply a relationship that was never measured.
* ``yaw_gen_absolute_mr{,_dark}.png`` — m_Z → m_R per arm as a dumbbell (the
  form for "before → after per item"), one panel per metric, each with its own
  axis. Deliberately not grouped bars: these values sit in a narrow band, where
  a zero-based bar hides every difference and a cropped bar overstates it.

Colors: the validated categorical palette's slot 1 (blue), one series per panel,
so identity never rests on hue. The Δ figure's arms are distinguished by axis
position and direct labels; the m_R figure carries a two-entry legend (filled
= rotated, hollow = unrotated) plus the connector between them.

Determinism is a requirement, not a nicety: the PNG metadata is pinned to a
fixed string (matplotlib would otherwise stamp its own version), so regenerating
twice yields byte-identical files and a rebuild shows up in git only when a
number changed.

    /n/fs/gatrdp/envs/flac/bin/python make_figures.py [--bundle PATH] [--outdir DIR]
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import MultipleLocator                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# NOTHING about the campaign is written down here. The arms, the confirmatory K,
# the seed count, the degrees of freedom and the zoom window are all READ from
# the bundle, so a bundle that changes cannot leave a caption behind describing
# the campaign it used to be.
#
# The one judgement that stays in this file is a DISPLAY heuristic, not a claim:
# which arms the zoomed panel shows. It takes the arms whose |Δ| has collapsed to
# within ZOOM_FRACTION of the largest — the "saturated" end of the dose axis —
# and sizes the window from their own CIs.
ZOOM_FRACTION = 0.25
ZOOM_PAD = 1.35
DELTA_DP = 3            # ONE rounding policy for every Δ this file renders

# Validated palette (references/palette.md), slot 1, per surface. Checked with
# scripts/validate_palette.js in both modes: all six checks PASS.
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e",
              "line": "#d9dee5", "series": "#2a78d6", "zero": "#8a8a86"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7",
             "line": "#3a4048", "series": "#3987e5", "zero": "#8a8a86"},
}
PNG_METADATA = {"Software": "exp_14 yaw_gen make_figures.py"}      # no timestamp
DPI = 160


def load_bundle(path):
    with open(path) as fh:
        return json.load(fh)


def group_order(arm):
    """The order of the group this arm frame-averages over; vanilla is n=1.

    Vanilla is the identity "orbit" — the n=1 point of the same dose axis, which
    is why it belongs on this curve at all."""
    m = re.search(r"C(\d+)", arm)
    return int(m.group(1)) if m else 1


def arms_of(bundle):
    """[(arm, axis label)] in the campaign's own fixed order."""
    out = []
    for arm in bundle["campaign"]["arm_order"]:
        n = group_order(arm)
        out.append((arm, f"C{n}" if n > 1 else "vanilla\n(n=1)"))
    return out


def facts(bundle):
    """The caption's facts, read from the bundle rather than remembered."""
    k = str(bundle["hypotheses"]["K"])
    arms = arms_of(bundle)
    row = bundle["paired"][arms[0][0]][k]["metrics"]["T60"]
    seeds = bundle["paired"][arms[0][0]][k]["pairs"]
    alpha = bundle["campaign"]["alpha"]
    return {"K": k, "arms": arms, "seeds": seeds, "df": row["df"],
            "ci": f"{round((1 - alpha) * 100)}%"}


def paired(bundle, metric, arms, K):
    """(means, lo_err, hi_err) for the paired Δ of one metric, arm-ordered."""
    means, lo, hi = [], [], []
    for arm, _label in arms:
        row = bundle["paired"][arm][K]["metrics"][metric]
        means.append(row["mean"])
        lo.append(row["mean"] - row["lo"])
        hi.append(row["hi"] - row["mean"])
    return means, lo, hi


def absolute(bundle, block, metric, arms, K):
    """Mean of one metric per arm from the R or Z block."""
    return [bundle["blocks"][block][arm][K]["values"][metric][0]
            for arm, _label in arms]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["line"])
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=3)
    ax.grid(axis="y", color=t["line"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])
    ax.title.set_color(t["ink"])


def _delta_panel(ax, t, arms, means, lo, hi, *, ylabel, title, ylim=None,
                 label_idx=(), note=None, subset=None):
    """One Δ panel. ``subset`` restricts it to a slice of the arms — the zoomed
    panel shows only the saturated ones, because drawing the full series into the zoom window
    leaves a line entering from off-scale and empty tick labels that a reader has
    to decode."""
    lo_i, hi_i = (0, len(arms)) if subset is None else subset
    arms = arms[lo_i:hi_i]
    means, lo, hi = means[lo_i:hi_i], lo[lo_i:hi_i], hi[lo_i:hi_i]
    label_idx = [i - lo_i for i in label_idx if lo_i <= i < hi_i]
    x = range(len(arms))
    ax.axhline(0, color=t["zero"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.errorbar(list(x), means, yerr=[lo, hi], fmt="none", ecolor=t["series"],
                elinewidth=2, capsize=4, capthick=2, zorder=2)
    ax.plot(list(x), means, color=t["series"], linewidth=2, zorder=3)
    ax.plot(list(x), means, "o", markersize=8, color=t["series"],
            markeredgecolor=t["surface"], markeredgewidth=2, zorder=4)
    for i in label_idx:                       # selective direct labels, not all
        va, off = ("bottom", 9) if means[i] >= 0 else ("top", -9)
        ax.annotate(f"{means[i]:+.{DELTA_DP}f}", (i, means[i]),
                    textcoords="offset points",
                    xytext=(0, off), ha="center", va=va, fontsize=8.5,
                    color=t["ink"], fontweight="600")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _arm, lab in arms], fontsize=8.5)
    ax.set_xlabel("frame-averaging group order", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="600", loc="left", pad=8)
    if ylim:
        ax.set_ylim(*ylim)
    if note:
        ax.annotate(note, (0.5, 0.06), xycoords="axes fraction", ha="center",
                    fontsize=8, color=t["muted"], style="italic")
    _style(ax, t)


def zoom_view(means, lo, hi):
    """(subset, ylim) for the saturated end of the dose axis — derived, not typed.

    The subset is the trailing run of arms whose |Δ| has collapsed to within
    ZOOM_FRACTION of the largest; the window is their own worst CI edge, padded.
    """
    peak = max(abs(m) for m in means)
    keep = [i for i, m in enumerate(means) if abs(m) <= ZOOM_FRACTION * peak]
    if not keep:
        return None, None
    start = keep[0]
    edge = max(max(abs(means[i] - lo[i]), abs(means[i] + hi[i]))
               for i in range(start, len(means)))
    span = edge * ZOOM_PAD
    return (start, len(means)), (-span, span)


def dose_response(bundle, mode, outdir):
    t = THEME[mode]
    f = facts(bundle)
    arms, K = f["arms"], f["K"]
    t60, t60_lo, t60_hi = paired(bundle, "T60", arms, K)
    r1, r1_lo, r1_hi = paired(bundle, "RIR_to_GT_RIR_R@1", arms, K)
    subset, ylim = zoom_view(t60, t60_lo, t60_hi)
    zoom_names = ", ".join(lab.replace("\n", " ") for _a, lab in arms[subset[0]:])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), facecolor=t["surface"])
    flat = [lab.replace("\n", " ") for _a, lab in arms[subset[0] + 1:]]
    _delta_panel(axes[0], t, arms, t60, t60_lo, t60_hi,
                 ylabel="Δ T60 % (scene-mean), R − Z   (+ = worse)",
                 title="Degradation under random yaw",
                 label_idx=tuple(range(0, subset[0] + 1)))
    _delta_panel(axes[1], t, arms, t60, t60_lo, t60_hi,
                 ylabel="Δ T60 % (scene-mean), R − Z   (+ = worse)",
                 title=f"…same series, zoomed on {zoom_names}",
                 ylim=ylim, label_idx=tuple(range(subset[0], len(arms))),
                 subset=subset,
                 note=f"{' and '.join(flat)} are flat within their CI")
    _delta_panel(axes[2], t, arms, r1, r1_lo, r1_hi,
                 ylabel="Δ R@1 pp (split-level), R − Z   (+ = better)",
                 title="Retrieval, same contrast", label_idx=(0, 1))
    fig.suptitle(f"Paired degradation vs group order — K={K}, {f['seeds']} "
                 f"seed-paired diffs, {f['ci']} paired-t CI (df={f['df']})",
                 fontsize=10.5, fontweight="600", color=t["ink"], x=0.006,
                 ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(outdir, f"yaw_gen_dose_response{'_dark' if mode == 'dark' else ''}.png")
    fig.savefig(path, dpi=DPI, facecolor=t["surface"], metadata=PNG_METADATA)
    plt.close(fig)
    return path


def absolute_mr(bundle, mode, outdir):
    """m_Z → m_R per arm, as a DUMBBELL.

    Not a grouped bar chart: these values live in a narrow band (T60 ≈ 7.2–8.2%)
    where a zero-based bar makes every arm look identical, and a bar with a
    non-zero baseline overstates every difference by the amount it cropped. The
    comparison here is "before → after per item", whose form is the dumbbell —
    and the connector IS the degradation, which is the thing to see.
    """
    t = THEME[mode]
    f = facts(bundle)
    arms, K = f["arms"], f["K"]
    panels = [("T60", "T60 % (scene-mean)", "T60 — lower is better ↓"),
              ("C50", "C50 dB (scene-mean)", "C50 — lower is better ↓"),
              ("RIR_to_GT_RIR_R@1", "R@1 % (split-level)",
               "R@1 retrieval — higher is better ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2), facecolor=t["surface"])
    y = list(range(len(arms)))[::-1]            # first arm at the top
    for ax, (metric, xlabel, title) in zip(axes, panels):
        r = absolute(bundle, "R", metric, arms, K)
        z = absolute(bundle, "Z", metric, arms, K)
        for yi, zv, rv in zip(y, z, r):
            ax.plot([zv, rv], [yi, yi], color=t["series"], linewidth=2,
                    alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot(z, y, "o", markersize=8, markerfacecolor=t["surface"],
                markeredgecolor=t["series"], markeredgewidth=2, linestyle="none",
                zorder=3, label="unrotated  m_Z")
        ax.plot(r, y, "o", markersize=8, color=t["series"],
                markeredgecolor=t["surface"], markeredgewidth=1.5,
                linestyle="none", zorder=4, label="random yaw  m_R")
        span = max(max(r), max(z)) - min(min(r), min(z))
        ax.set_xlim(min(min(r), min(z)) - span * 0.22,
                    max(max(r), max(z)) + span * 0.22)
        ax.set_yticks(y)
        ax.set_yticklabels([lab.replace("\n", " ") for _arm, lab in arms],
                           fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="600", loc="left", pad=8)
        _style(ax, t)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["line"], linewidth=0.8, alpha=0.9)
        if metric == "C50":
            ax.xaxis.set_major_locator(MultipleLocator(0.03))
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", ncols=2, frameon=False,
                     fontsize=8.5, bbox_to_anchor=(0.006, 0.945))
    for text in leg.get_texts():
        text.set_color(t["muted"])
    fig.suptitle("Absolute performance under random yaw — m_Z → m_R per arm, "
                 f"K={K}, mean of {f['seeds']} eval seeds",
                 fontsize=10.5, fontweight="600", color=t["ink"], x=0.006,
                 ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = os.path.join(outdir, f"yaw_gen_absolute_mr{'_dark' if mode == 'dark' else ''}.png")
    fig.savefig(path, dpi=DPI, facecolor=t["surface"], metadata=PNG_METADATA)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default=os.path.join(HERE, "results_bundle.json"))
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args(argv)
    bundle = load_bundle(args.bundle)
    written = []
    for mode in ("light", "dark"):
        written.append(dose_response(bundle, mode, args.outdir))
        written.append(absolute_mr(bundle, mode, args.outdir))
    for path in written:
        print("wrote", os.path.relpath(path, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
