"""Tests for exp_14's curve figure (plan §4 Round D, ``plot_data_curve``).

The figure is the form most people will actually read the result in, so the things it
must not do are the things a plot does silently:

* **Draw a point that is not a result.** A row the assembler marked incomplete (fewer than
  five eval seeds, or a run that has not finished) has no mean worth plotting; it must be
  absent from the line, not interpolated across.
* **Identify two arms by colour alone.** Stock DINOv3 is ``#1F77B4`` open circles and
  CylDINO ``#FF7F0E`` filled squares -- the project's exp_13 palette, plus a marker shape
  and fill that survive greyscale printing and colour-vision deficiency.
* **Drop the anchor.** The 100 % point is part of the curve, not a separate claim, so it
  is drawn on the same axes as 25/50/75.

Rendering is checked by rendering: the tests drive the real matplotlib Agg backend over a
checked-in ``data_curve.json`` fixture that was produced by ``assemble.build_curve``
itself, and one test proves the fixture still has the schema the assembler emits. CPU-only.
"""
import json
import math
import os

from src.tools.data_curve import assemble, plot

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "data_curve")
CURVE_JSON = os.path.join(FIXTURES, "data_curve_fixture.json")
ANCHORS_JSON = os.path.join(FIXTURES, "anchors_tier_S.json")
MANIFEST_JSON = os.path.join(FIXTURES, "split_manifest.json")


def load_fixture():
    with open(CURVE_JSON) as fin:
        return json.load(fin)


# ------------------------------------------------------------------------- the files
def test_one_png_and_one_svg_per_K(tmp_path):
    written = plot.plot_curve(load_fixture(), str(tmp_path))
    assert [entry["K"] for entry in written] == ["1", "8"]
    for entry in written:
        assert os.path.exists(entry["png"]) and os.path.exists(entry["svg"])
        assert entry["png"].endswith("data_curve_K{}.png".format(entry["K"]))


def test_the_files_really_are_a_png_and_an_svg(tmp_path):
    written = plot.plot_curve(load_fixture(), str(tmp_path))
    with open(written[0]["png"], "rb") as fin:
        assert fin.read(8) == b"\x89PNG\r\n\x1a\n"
    assert "<svg" in open(written[0]["svg"]).read(4096)


# ------------------------------------------------------------------------- the panels
def test_the_four_panels_are_the_pre_registered_endpoints():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        assert list(plot.PANEL_METRICS) == ["T60", "C50", "EDT", "R@1"]
        assert [ax.get_ylabel().split()[0] for ax in axes] == list(plot.PANEL_METRICS)
    finally:
        plot.close(figure)


def test_no_title_clutter():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        assert figure._suptitle is None
        assert all(ax.get_title() == "" for ax in axes)
    finally:
        plot.close(figure)


def test_the_axis_labels_are_set_in_the_serif_stack():
    assert plot.SERIF_STACK[0] == "P052"          # the Palatino clone shipped on this box
    assert "Palatino" in plot.SERIF_STACK
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        assert axes[0].yaxis.label.get_fontfamily() == ["serif"]
    finally:
        plot.close(figure)


# ------------------------------------------------------------------------- the series
def test_both_arms_use_the_project_palette_and_distinct_marker_shapes():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        stock = plot.arm_line(axes[0], "van")
        cyl = plot.arm_line(axes[0], "cyl")
        assert stock.get_color().upper() == "#1F77B4"
        assert cyl.get_color().upper() == "#FF7F0E"
        assert (stock.get_marker(), cyl.get_marker()) == ("o", "s")
        # open vs filled: identity survives greyscale and colour-vision deficiency.
        assert stock.get_markerfacecolor() in ("none", "None")
        assert cyl.get_markerfacecolor().upper() == "#FF7F0E"
    finally:
        plot.close(figure)


def test_every_measured_fraction_including_the_anchor_is_on_the_curve():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        assert sorted(plot.arm_line(axes[0], "cyl").get_xdata()) == \
            [25.0, 50.0, 75.0, 100.0]
    finally:
        plot.close(figure)


def test_the_seed_sd_is_drawn_as_an_error_bar():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        containers = [c for c in axes[0].containers if getattr(c, "has_yerr", False)]
        assert len(containers) == 2                 # one per arm
    finally:
        plot.close(figure)


def test_a_row_without_a_mean_breaks_the_line_instead_of_bridging_it():
    doc = load_fixture()
    doc["curve"]["K8"]["T60"]["50"]["cyl"] = {"mean": None, "sd": None, "n": 0,
                                              "seeds": [], "complete": False}
    doc["curve"]["K8"]["T60"]["50"]["complete"] = False
    figure, axes = plot.build_figure(doc, 8)
    try:
        line = plot.arm_line(axes[0], "cyl")
        xs, ys = list(line.get_xdata()), list(line.get_ydata())
        assert [x for x, y in zip(xs, ys) if y == y] == [25.0, 75.0, 100.0]
        # NaN, not a dropped x: a dropped x would have joined 25 % straight to 75 %.
        assert math.isnan(ys[xs.index(50.0)])
    finally:
        plot.close(figure)


def test_a_legend_names_both_arms():
    figure, axes = plot.build_figure(load_fixture(), 8)
    try:
        labels = [text.get_text() for text in figure.legends[0].get_texts()] \
            if figure.legends else [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert set(plot.SERIES_LABELS.values()) <= set(labels)
    finally:
        plot.close(figure)


# ------------------------------------------------------------------- fixture & CLI
def test_the_checked_in_fixture_still_has_the_schema_the_assembler_emits(tmp_path):
    # build_curve over an empty NAS yields the document SKELETON -- same keys, no numbers --
    # which is exactly what the fixture must keep matching as the assembler evolves.
    skeleton = assemble.build_curve(str(tmp_path), ANCHORS_JSON,
                                    split_manifest_path=MANIFEST_JSON)
    fixture = load_fixture()
    assert set(fixture) - {"_fixture"} == set(skeleton)
    assert set(fixture["curve"]) == set(skeleton["curve"])
    assert set(fixture["curve"]["K8"]) == set(skeleton["curve"]["K8"])
    assert set(fixture["curve"]["K8"]["T60"]) == set(skeleton["curve"]["K8"]["T60"])
    assert set(fixture["curve"]["K8"]["T60"]["25"]) == \
        set(skeleton["curve"]["K8"]["T60"]["25"])


def test_cli_renders_both_K_from_a_curve_json(tmp_path, capsys):
    rc = plot.main(["--curve-json", CURVE_JSON, "--out-dir", str(tmp_path)])
    assert rc == 0
    written = sorted(os.listdir(tmp_path))
    assert written == ["data_curve_K1.png", "data_curve_K1.svg",
                       "data_curve_K8.png", "data_curve_K8.svg"]
    assert "data_curve_K8.png" in capsys.readouterr().out


def test_cli_reports_a_missing_curve_json_instead_of_raising(tmp_path):
    assert plot.main(["--curve-json", str(tmp_path / "nope.json"),
                      "--out-dir", str(tmp_path)]) != 0
