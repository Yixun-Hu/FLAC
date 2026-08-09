#!/usr/bin/env python3
"""PNG renders of the exp_11 trajectory figures (same data/palette as the HTML page).

Reuses series() from gen_trajectory_figures.py; emits a combined 2x3 grid PNG
plus one PNG per metric into fa_orbit_results_assets/. Light surface (PNG has
no theme switching); numbers come only from the metric JSONs.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_trajectory_figures import series, METRICS  # noqa: E402

ASSETS = os.path.join(HERE, "fa_orbit_results_assets")
os.makedirs(ASSETS, exist_ok=True)

DASH = {"": "-", "6 3": (0, (6, 3)), "2 3": (0, (2, 3)), "8 4": (0, (8, 4)), "3 2": (0, (3, 2))}

def draw(ax, data, mkey, mlabel, lower_better):
    for key, d in data.items():
        pxy = sorted((s, m[mkey]) for s, m in d["pts"].items() if mkey in m)
        if pxy:
            xs, ys = zip(*pxy)
            ax.plot(xs, ys, linestyle=DASH.get(d["dash"], "-"), color=d["cl"], lw=2,
                    marker="o", ms=4, label=d["label"], zorder=3)
        if d["conf"] and mkey in d["conf"]:
            m, sd = d["conf"][mkey]
            ax.errorbar([40400], [m], yerr=[sd], color=d["cl"], fmt="o", ms=7,
                        capsize=4, lw=2, zorder=4,
                        label=None if pxy else d["label"])
        # Q10 (>40k): 5-seed mean with a shaded +-1 sd band. A band rather than
        # error bars because at 2,500-step density the bars would collide; the
        # 40k conf point anchors it so the shading starts where the 5-seed
        # regime does, not mid-air.
        band = sorted((st, d["band"][st][mkey]) for st in d.get("band", {})
                      if mkey in d["band"][st])
        if band:
            anchor = [(40000, d["conf"][mkey])] if d.get("conf") and mkey in d["conf"] else []
            seq = anchor + band
            bx = [st for st, _ in seq]
            bm = [m for _, (m, _sd) in seq]
            blo = [m - sd for _, (m, sd) in seq]
            bhi = [m + sd for _, (m, sd) in seq]
            ax.fill_between(bx, blo, bhi, color=d["cl"], alpha=0.15, lw=0, zorder=2)
            ax.plot(bx, bm, linestyle=DASH.get(d["dash"], "-"), color=d["cl"], lw=2,
                    marker="o", ms=4, zorder=3,
                    label=None if (pxy or d.get("conf")) else d["label"])
    ax.set_title(f"{mlabel}  ({'lower' if lower_better else 'higher'} is better)",
                 fontsize=10, loc="left")
    ax.set_xlabel("training step", fontsize=9)
    ticks = [10000, 20000, 30000, 40000]
    if any(d.get("band") for d in data.values()):
        ticks += [50000, 60000, 70000, 80000, 90000, 100000]
    ax.set_xticks(ticks, [f"{t//1000}k" for t in ticks])
    ax.grid(True, axis="y", lw=0.5, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8.5)

def main():
  for k in (8, 1):
    data = series(k)
    # combined 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=160)
    for ax, (mk, ml, lb) in zip(axes.flat, METRICS):
        draw(ax, data, mk, ml, lb)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9)
    fig.suptitle(f"exp_11 — acoustic performance vs training step (K={k}, full unseen split; "
                 "endpoint dot = 5-seed mean ± sd)", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    combined = os.path.join(ASSETS, f"fa_orbit_trajectories_K{k}.png")
    fig.savefig(combined, facecolor="#fcfcfb")
    plt.close(fig)
    outs = [combined]
    # per-metric singles
    for mk, ml, lb in METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        draw(ax, data, mk, ml, lb)
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        p = os.path.join(ASSETS, f"fa_orbit_traj_K{k}_{mk.replace('@','').replace('RIR_to_GT_RIR_','')}.png")
        fig.savefig(p, facecolor="#fcfcfb")
        plt.close(fig)
        outs.append(p)
    for p in outs:
        print("wrote", os.path.relpath(p, HERE), f"({os.path.getsize(p)//1024} KB)")

if __name__ == "__main__":
    main()
