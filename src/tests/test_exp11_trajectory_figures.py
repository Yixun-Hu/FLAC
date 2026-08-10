"""Tests for the exp_11 trajectory figure generators (re-pin review, fix 5).

Two claims the generators made but did not keep.

*The harvest.* ``gen_trajectory_figures`` drew a >40k band whenever five files
with the right name shape existed. It never proved the five were seeds 42-46
exactly once, never opened a sidecar, and never required one checkpoint / config
/ source SHA across them — so five duplicates of one seed, or a block whose rows
came from different pins or different checkpoints, became a published band. Every
step now goes through ``validate_cell(..., contract="traj")``, and a step with
any problem is refused and disclosed rather than narrowed.

*The scale.* The y-range came from the plotted points and the conf MEANS, so a
band (or a conf error bar) could be drawn outside the range it was scaled to.
Both generators now scale from ``value_extent``, which spans everything drawn.

The harvest tests run the REAL validator over synthetic on-disk cells, in the
same style as ``test_exp11_validate_rows``; the gate-plumbing tests inject a
fake validator so the refusal/aggregation logic is exercised on its own.
"""
import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_EXPDIR = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude")


def _load(name):
    if _EXPDIR not in sys.path:
        sys.path.insert(0, _EXPDIR)
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EXPDIR, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("gen_trajectory_figures")
V = _load("exp11_validate_rows")

ARM, K, STEP = "C8", 8, 42500
SEEDS = (42, 43, 44, 45, 46)
CKPT = (f"outputs_FLAC/exp11_{ARM}/FLAC_exp11_{ARM}/exp11_{ARM}/checkpoints/"
        f"epoch=9-step={STEP}.ckpt")
METRICS_FULL = {"T60": 12.3, "C50": 1.1, "EDT": 4.4, "FD": 2.2, "Invalid T60": 0.0,
                "RIR_to_GT_RIR_R@1": 0.5, "RIR_to_GT_RIR_R@5": 0.7,
                "RIR_to_GT_RIR_R@10": 0.9, "RIR_to_geom_R@1": 0.4,
                "RIR_to_geom_R@5": 0.6, "RIR_to_geom_R@10": 0.8}
ANGLES = [j * 360.0 / 8 for j in range(8)]


def _write_cell(tmp_path, seeds=SEEDS, step=STEP, ckpt=CKPT, metric_by_seed=None, **side_over):
    """One traj cell on disk: five metrics JSONs plus their screen sidecars."""
    paths = []
    for seed in seeds:
        name = (f"{os.path.basename(ckpt)[:-5]}_metrics_1_1.0_exp11_{ARM}_traj_S{step}"
                f"_s{seed}_K{K}_fa_invariant_a8.json")
        eval_name = f"exp11_{ARM}_traj_S{step}_s{seed}_K{K}"
        metrics = dict(METRICS_FULL)
        if metric_by_seed and seed in metric_by_seed:
            metrics.update(metric_by_seed[seed])
        rec = {"metrics": metrics, "ckpt_path": ckpt, "rotate_deg": 0.0,
               "cond_method": "fa_invariant", "frame_avg_angles": ANGLES,
               "cond_autocast": "bf16", "orbit_execution": "batched", "frame_avg_fwd_cap": 64,
               "source_sha": "d" * 40, "batch_size": 64, "n_samples": 6337,
               "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
               "seed": seed, "cfg_scale": 1.0, "steps": 1, "eval_name": eval_name,
               "weights_source": "ema", "device": "cuda"}
        side = {"arm": ARM, "step": step, "seed": seed, "K": K, "eval_name": eval_name,
                "cfg_scale": 1.0, "steps": 1,
                "model_config": f"worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_{ARM}.json",
                "model_config_sha256": "b" * 64,
                "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
                "ckpt_path": ckpt, "ckpt_sha256": "c" * 64, "use_ema": True,
                "frame_avg_angles": ANGLES, "cond_method": "fa_invariant",
                "cond_autocast": "bf16", "commit": "d" * 40}
        for key, per_seed in side_over.items():
            if seed in per_seed:
                side[key] = per_seed[seed]
        p = tmp_path / name
        p.write_text(json.dumps(rec))
        with open(V.sidecar_path_for(str(p)), "w") as fh:
            json.dump(side, fh)
        paths.append(str(p))
    return paths


def _band(tmp_path, refusals=None):
    return G.validated_band(str(tmp_path), ARM, K, "a8", refusals=refusals)


# --------------------------------------------------------------------------- #
# 1. the harvest goes through the validator (real validator, synthetic cells)
# --------------------------------------------------------------------------- #
def test_a_clean_five_seed_cell_becomes_a_band(tmp_path):
    _write_cell(tmp_path, metric_by_seed={s: {"T60": 10.0 + i} for i, s in enumerate(SEEDS)})
    band = _band(tmp_path)
    assert set(band) == {STEP}
    mean, sd = band[STEP]["T60"]
    assert mean == pytest.approx(12.0) and sd == pytest.approx(1.5811, rel=1e-3)


def test_a_partial_block_is_refused_not_narrowed(tmp_path):
    _write_cell(tmp_path, seeds=(42, 43, 44, 45))
    refusals = []
    assert _band(tmp_path, refusals) == {}
    assert refusals and refusals[0]["step"] == STEP and "missing" in refusals[0]["problems"][0]


def test_five_files_that_are_not_five_seeds_are_refused(tmp_path):
    """The exact defect the review named: five files became a band regardless of
    which seeds they were."""
    _write_cell(tmp_path, seeds=(42, 43, 44, 45, 45))   # a duplicate, not five seeds
    refusals = []
    assert _band(tmp_path, refusals) == {}
    assert any("46 is missing" in p or "more than once" in p for p in refusals[0]["problems"])


def test_a_mixed_provenance_block_is_refused(tmp_path):
    """Same five seeds, but one row was produced by a different checkpoint."""
    _write_cell(tmp_path, ckpt_sha256={44: "e" * 64})
    refusals = []
    assert _band(tmp_path, refusals) == {}
    assert any("ckpt_sha256" in p for p in refusals[0]["problems"])


def test_a_mixed_pin_block_is_refused(tmp_path):
    _write_cell(tmp_path, commit={45: "f" * 40})
    refusals = []
    assert _band(tmp_path, refusals) == {}
    assert any("commit" in p for p in refusals[0]["problems"])


@pytest.mark.parametrize("step,needle", [
    (40000, "strictly above"),        # the 40k endpoint is the conf dot, not a band point
    (42501, "checkpoint grid"),
    (102500, "budget"),
])
def test_the_grid_floor_and_ceiling_are_enforced_through_the_validator(tmp_path, step, needle):
    ckpt = CKPT.replace(f"step={STEP}", f"step={step}")
    _write_cell(tmp_path, step=step, ckpt=ckpt)
    refusals = []
    assert _band(tmp_path, refusals) == {}
    assert any(needle in p for p in refusals[0]["problems"])


# --------------------------------------------------------------------------- #
# 2. the gate plumbing itself
# --------------------------------------------------------------------------- #
def test_the_harvest_asks_for_the_traj_contract(tmp_path):
    seen = {}

    def fake(paths, arm, step, k, contract, **kw):
        seen.update(arm=arm, step=step, k=k, contract=contract, n=len(paths))
        return [], ["synthetic refusal"]

    _write_cell(tmp_path)
    refusals = []
    assert G.validated_band(str(tmp_path), ARM, K, "a8", validate=fake, refusals=refusals) == {}
    assert seen == {"arm": ARM, "step": STEP, "k": K, "contract": "traj", "n": 5}
    assert refusals[0]["problems"] == ["synthetic refusal"]


def test_the_default_validator_is_the_row_validator():
    """No private copy of the rules: the generator's default IS validate_cell."""
    assert G.V.validate_cell is V.validate_cell or G.V.validate_cell.__name__ == "validate_cell"


def test_rows_that_validate_but_carry_no_metric_are_not_invented(tmp_path):
    def fake(paths, arm, step, k, contract, **kw):
        return [{"metrics": {}} for _ in paths], []

    _write_cell(tmp_path)
    assert G.validated_band(str(tmp_path), ARM, K, "a8", validate=fake) == {}


# --------------------------------------------------------------------------- #
# 3. the y-range spans everything that is drawn
# --------------------------------------------------------------------------- #
def _data(pts=None, conf=None, band=None):
    return {"C8": {"label": "C8", "cl": "#000", "cd": "#fff", "dash": "",
                   "pts": pts or {}, "conf": conf, "band": band or {}}}


def test_value_extent_includes_the_band_envelope():
    """RED for the review's finding: the old range was min/max over points and
    conf MEANS, which puts a band of 10 ± 3 outside a 9-11 scale."""
    data = _data(pts={10000: {"T60": 9.0}, 40000: {"T60": 11.0}},
                 band={45000: {"T60": (10.0, 3.0)}})
    lo, hi = G.value_extent(data, "T60")
    assert lo <= 7.0 and hi >= 13.0


def test_value_extent_includes_the_conf_whiskers():
    data = _data(pts={10000: {"T60": 9.0}}, conf={"T60": (9.5, 2.0)})
    lo, hi = G.value_extent(data, "T60")
    assert lo <= 7.5 and hi >= 11.5


def test_value_extent_is_empty_safe():
    assert G.value_extent(_data(), "T60") == (0.0, 1.0)


def test_the_svg_panel_keeps_the_band_inside_the_plot_area():
    """End to end: a band far outside the point range must still render within
    the panel's drawing box, not above it."""
    data = _data(pts={10000: {"T60": 9.0}, 40000: {"T60": 11.0}},
                 conf={"T60": (11.0, 0.1)}, band={45000: {"T60": (10.0, 6.0)}})
    svg = G.svg_panel(data, "T60", "T60 (%)", True, W=560, H=340)
    ys = [float(tok.split(",")[1]) for tok in svg.replace("M", " ").replace("L", " ").split()
          if "," in tok and tok.replace(",", "").replace(".", "").isdigit()]
    assert ys, "no plotted coordinates found"
    assert min(ys) >= 0 and max(ys) <= 340


def test_the_png_generator_scales_from_the_same_extent():
    """Both generators must agree about what fits; the PNG used to rely on
    matplotlib autoscale while the SVG used its own (smaller) range."""
    src = open(os.path.join(_EXPDIR, "gen_trajectory_pngs.py")).read()
    assert "value_extent" in src and "set_ylim" in src


# --------------------------------------------------------------------------- #
# 4. the table view carries the band extrema
# --------------------------------------------------------------------------- #
def test_band_cell_renders_mean_sd_and_the_extrema():
    cell = G.band_cell((10.0, 2.0))
    assert "10.000 ± 2.000" in cell and "[8.000, 12.000]" in cell


def test_band_cell_handles_a_missing_metric():
    assert G.band_cell(None) == "<td>—</td>"
