"""exp_14: the data-efficiency curve as a figure -- one per K, four pre-registered panels.

Reads ``data_curve.json`` (written by ``src.tools.data_curve.assemble``) and renders the
curve people will actually argue about. The design decisions are not taste:

* **Four panels, one axis each** -- T60, C50, EDT and R@1, never two measures sharing a
  y-scale. C50 is on the sheet precisely because CylDINO is *worse* there at 100 %; a
  figure that showed only the wins would be a different claim.
* **Identity by colour AND shape.** Stock DINOv3 is ``#1F77B4`` open circles, CylDINO
  ``#FF7F0E`` filled squares -- the project's exp_13 palette, whose blue/orange pair is the
  standard colour-vision-deficiency-safe pairing, with marker shape and fill as the
  secondary encoding that survives greyscale printing.
* **Nothing is drawn that the table refused.** A row the assembler marked incomplete has
  no mean here: the point is skipped and the line is *not* interpolated across it. When
  any such gap exists the figure says so in one footnote line; a complete figure carries
  none, which is the only text on it besides the axes and the legend.
* **The 100 % anchor is on the curve**, on the same axes as 25/50/75, because it is the
  same measurement -- with the caveat that it may be the marginal (exp_13 reference) form,
  which the assembler's disclosures state and the results text repeats.

Error bars are the seed sd the assembler computed (five eval seeds, one training seed per
cell): eval noise, not training-run uncertainty.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                       # rendering tool: never wants a display
import matplotlib.pyplot as plt             # noqa: E402  (after the backend is fixed)

#: The Palatino-family stack this project's figures already use, most-wanted first; the
#: box ships URW Palladio under its PostScript name P052. DejaVu Serif is the fallback
#: that always exists, so a missing font degrades to a serif rather than to sans.
SERIF_STACK = ["P052", "Palatino", "TeX Gyre Pagella", "DejaVu Serif"]
#: The four panels, in reading order. T60 and EDT are the verdict's primaries.
PANEL_METRICS = ("T60", "C50", "EDT", "R@1")
#: Which way is better, per panel -- printed in the axis label so no one has to remember.
BETTER = {"T60": "lower", "C50": "lower", "EDT": "lower",
          "R@1": "higher", "R@5": "higher", "R@10": "higher"}
SERIES_LABELS = {"van": "stock DINOv3 S", "cyl": "CylDINO core S"}
SERIES_STYLE = {
    "van": {"color": "#1F77B4", "marker": "o", "markerfacecolor": "none"},
    "cyl": {"color": "#FF7F0E", "marker": "s", "markerfacecolor": "#FF7F0E"},
}
#: Mark specs: thin lines, markers big enough to read the shape, recessive grid.
LINE_WIDTH, MARKER_SIZE, GRID_ALPHA = 1.6, 6.5, 0.25
EXIT_OK, EXIT_INPUT_ERROR = 0, 2

close = plt.close                            # re-exported so callers need no pyplot import


def series_points(doc, K, metric, arm):
    """``(xs, ys, errs, gaps)`` for one arm of one panel; incomplete cells become ``nan``.

    A cell the assembler did not mark ``complete`` keeps its x position with a ``nan`` y,
    which matplotlib draws as neither a marker nor a segment: the line BREAKS there.
    Dropping the x instead would have joined its two neighbours with a straight segment,
    drawing a value nobody measured.

    The gate is ``complete``, not ``mean is None`` (codex D3 finding 2): a four-seed
    aggregate has a perfectly numeric mean and is *not* the five-seed quantity the anchors
    were measured as, so plotting it would be the same lie with a number attached. An
    explicit ``complete: True`` is required -- a cell that has forgotten to say so is a gap.
    ``gaps`` names the fractions that broke, for the footnote.
    """
    rows = doc["curve"][f"K{K}"][metric]
    xs, ys, errs, gaps = [], [], [], []
    for pct in doc["fractions_pct"]:
        cell = rows[str(pct)].get(arm) or {}
        xs.append(float(pct))
        if cell.get("mean") is None or cell.get("complete") is not True:
            gaps.append(int(pct))
            ys.append(float("nan"))
            errs.append(0.0)
            continue
        ys.append(float(cell["mean"]))
        errs.append(float(cell["sd"] or 0.0))
    return xs, ys, errs, gaps


def build_figure(doc, K, panels=PANEL_METRICS):
    """``(figure, axes)`` for one K: a 2x2 sheet of panels, no titles, one shared legend.

    The rc settings are applied through a context manager, so importing this module never
    changes anyone else's matplotlib defaults.
    """
    with plt.rc_context({"font.family": "serif", "font.serif": SERIF_STACK,
                         "font.size": 9, "axes.linewidth": 0.7,
                         "svg.fonttype": "none"}):
        ncols = 2
        nrows = -(-len(panels) // ncols)
        figure, grid = plt.subplots(nrows, ncols, figsize=(7.0, 2.3 * nrows), sharex=True,
                                    squeeze=False)
        axes = list(grid.flat)
        for spare in axes[len(panels):]:          # an odd panel count leaves a blank cell
            spare.set_visible(False)
        axes = axes[:len(panels)]
        gaps = set()
        for axis, metric in zip(axes, panels):
            for arm in ("van", "cyl"):
                xs, ys, errs, missing = series_points(doc, K, metric, arm)
                gaps.update(missing)
                axis.errorbar(xs, ys, yerr=errs, label=SERIES_LABELS[arm],
                              linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                              markeredgewidth=1.3, capsize=2.5, elinewidth=0.9,
                              **SERIES_STYLE[arm])
            axis.set_ylabel(f"{metric} ({BETTER[metric]} better)")
            axis.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=0.6)
            axis.set_axisbelow(True)
            axis.margins(x=0.09)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        for axis in axes[-ncols:]:                # only the bottom row carries the x label
            axis.set_xlabel("training data fraction (%)")
        axes[0].set_xticks([float(pct) for pct in doc["fractions_pct"]])
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                      bbox_to_anchor=(0.5, 1.0))
        figure.tight_layout(rect=(0, 0.045 if gaps else 0, 1, 0.94))
        if gaps:
            figure.text(0.01, 0.01, "missing point(s) at "
                        + ", ".join(f"{pct} %" for pct in sorted(gaps))
                        + ": that cell is incomplete and is NOT interpolated",
                        fontsize=7, ha="left", va="bottom")
    return figure, axes


def arm_line(axis, arm):
    """The ``Line2D`` carrying one arm's points on ``axis``.

    ``errorbar`` labels the *container*, not the data line (the line is ``_nolegend_`` so
    the legend does not list it twice), so the arm has to be looked up by container -- which
    is also the only place the y-errors live.
    """
    for container in axis.containers:
        if container.get_label() == SERIES_LABELS[arm]:
            return container.lines[0]
    raise KeyError(f"no {arm!r} series on this axis")


def figure_paths(out_dir, K, stem="data_curve"):
    return {"png": os.path.join(out_dir, f"{stem}_K{K}.png"),
            "svg": os.path.join(out_dir, f"{stem}_K{K}.svg")}


def plot_curve(doc, out_dir, ks=None, panels=PANEL_METRICS, stem="data_curve", dpi=300):
    """Render one PNG + SVG per K into ``out_dir``; returns what was written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for K in (ks if ks is not None else sorted((key[1:] for key in doc["curve"]), key=int)):
        figure, _ = build_figure(doc, K, panels)
        try:
            paths = figure_paths(out_dir, K, stem)
            figure.savefig(paths["png"], dpi=dpi)
            figure.savefig(paths["svg"])
        finally:
            plt.close(figure)
        written.append(dict(paths, K=K))
    return written


def main(argv=None):
    """CLI: render the curve figures from an assembled ``data_curve.json``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.data_curve.plot",
        description="Render exp_14's data-efficiency curve (one PNG + SVG per K).")
    parser.add_argument("--curve-json", required=True,
                        help="the document written by src.tools.data_curve.assemble")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="data_curve")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        with open(args.curve_json) as fin:
            doc = json.load(fin)
        written = plot_curve(doc, args.out_dir, stem=args.stem, dpi=args.dpi)
    except (OSError, ValueError, KeyError) as err:
        print(f"plot: {type(err).__name__}: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    for entry in written:
        print(f"K={entry['K']}: {os.path.basename(entry['png'])} "
              f"+ {os.path.basename(entry['svg'])} -> {args.out_dir}")
    return EXIT_OK


if __name__ == "__main__":     # pragma: no cover - exercised through main() in tests
    sys.exit(main())
