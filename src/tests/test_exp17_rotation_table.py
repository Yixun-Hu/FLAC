"""exp_17 — aggregate the 128-cell C4 grid into the rotation-robustness table.

The scientific question this table answers is NOT "how good is Yaw-Aug" but
"how much does its quality MOVE when the scene is rotated". So the headline
quantity is a spread across the C4 orbit, and the pitfalls are about comparing
like with like:

* **Per-scene mean, not per-sample.** Paper headline numbers average per scene
  (CLAUDE.md); mixing the two conventions changes the numbers.
* **A cell is only comparable within its own (step, K).** A rotation spread
  computed across different K, or across checkpoints, is meaningless.
* **An incomplete orbit must not produce a spread.** With 3 of 4 angles present,
  a max-min is not the C4 spread — it is a smaller number that looks better.
* **Protocol agreement is a precondition, not a footnote.** A cell evaluated
  under a different cond_method/cond_autocast is not part of this grid even if
  its filename matches.

Written by the main session seat (Claude Opus 5, max effort).
"""
import json

import pytest

from src.tools.exp17_rotation_table import (
    load_cells, orbit_rows, ROTATIONS, METRIC_KEYS,
)


def _write(tmp_path, step, k, rot, t60=9.0, **over):
    rec = {
        "cond_method": "vanilla", "cond_autocast": "bf16",
        "rotate_deg": float(rot),
        # the keys as they exist in real records (NOT the stdout labels)
        "metrics": {"T60": t60, "C50": 1.0, "EDT": 40.0, "FD": 0.3,
                    "RIR_to_GT_RIR_R@1": 5.0},
    }
    rec.update(over)
    # REAL naming: eval_FLAC appends its own rotation suffix for non-zero
    # angles, so rotated files carry it twice: `..._rot90_seed42_rot90.json`.
    # The fixture must mirror the artifact, not an idealization -- an
    # end-anchored parser passed the idealized fixture while dropping every
    # rotated cell of the live grid.
    suffix = f"_rot{rot}" if rot != 0 else ""
    name = (f"epoch=0-step={step}_metrics_1_1.0_"
            f"exp17_YAWAUG_S{step}_K{k}_rot{rot}_seed42{suffix}.json")
    (tmp_path / name).write_text(json.dumps(rec))


def _full_orbit(tmp_path, step=40000, k=1, values=(9.0, 9.4, 9.1, 9.3)):
    for rot, v in zip(ROTATIONS, values):
        _write(tmp_path, step, k, rot, t60=v)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_a_complete_orbit_loads_as_four_cells(tmp_path):
    _full_orbit(tmp_path)
    cells = load_cells(tmp_path)
    assert len(cells) == 4
    assert {c.rotate_deg for c in cells} == set(ROTATIONS)


def test_step_and_K_are_read_from_the_filename(tmp_path):
    _write(tmp_path, 22500, 8, 180)
    (c,) = load_cells(tmp_path)
    assert (c.step, c.k, c.rotate_deg) == (22500, 8, 180)


def test_a_cell_from_a_different_protocol_is_refused(tmp_path):
    """Filename agreement is not protocol agreement."""
    _write(tmp_path, 40000, 1, 0, cond_autocast="default")
    with pytest.raises(ValueError, match="cond_autocast"):
        load_cells(tmp_path)


def test_a_contradictory_evaluator_suffix_is_refused(tmp_path):
    """`..._rot90_seed42_rot270.json` = produced under a different angle."""
    rec = {"cond_method": "vanilla", "cond_autocast": "bf16", "rotate_deg": 90.0,
           "metrics": {k: 1.0 for k in METRIC_KEYS}}
    bad = "epoch=0-step=2500_metrics_1_1.0_exp17_YAWAUG_S2500_K1_rot90_seed42_rot270.json"
    (tmp_path / bad).write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="suffix"):
        load_cells(tmp_path)


def test_a_stream_sidecar_is_ignored(tmp_path):
    _full_orbit(tmp_path)
    (tmp_path / "epoch=0-step=40000_metrics_1_1.0_exp17_YAWAUG_S40000_K1_rot0_seed42.stream.json").write_text("{}")
    assert len(load_cells(tmp_path)) == 4


def test_a_cell_whose_recorded_angle_contradicts_its_name_is_refused(tmp_path):
    """The one mislabel that would silently permute the orbit."""
    _write(tmp_path, 40000, 1, 90, rotate_deg=270.0)
    with pytest.raises(ValueError, match="rotate_deg"):
        load_cells(tmp_path)


# --------------------------------------------------------------------------- #
# the orbit spread — the actual quantity of interest
# --------------------------------------------------------------------------- #
def test_the_spread_is_max_minus_min_over_the_orbit(tmp_path):
    _full_orbit(tmp_path, values=(9.0, 9.4, 9.1, 9.3))
    (row,) = orbit_rows(load_cells(tmp_path))
    assert row.at_0["T60"] == pytest.approx(9.0)
    assert row.spread["T60"] == pytest.approx(0.4)


def test_a_rotation_invariant_arm_has_zero_spread(tmp_path):
    _full_orbit(tmp_path, values=(9.0, 9.0, 9.0, 9.0))
    (row,) = orbit_rows(load_cells(tmp_path))
    assert row.spread["T60"] == pytest.approx(0.0)


def test_an_incomplete_orbit_yields_no_row(tmp_path):
    """3 of 4 angles is not a C4 spread; it is a smaller, flattering number."""
    for rot in (0, 90, 180):
        _write(tmp_path, 40000, 1, rot)
    assert orbit_rows(load_cells(tmp_path)) == []


def test_orbits_are_grouped_by_step_AND_K(tmp_path):
    _full_orbit(tmp_path, step=40000, k=1)
    _full_orbit(tmp_path, step=40000, k=8)
    _full_orbit(tmp_path, step=20000, k=1)
    rows = orbit_rows(load_cells(tmp_path))
    assert {(r.step, r.k) for r in rows} == {(40000, 1), (40000, 8), (20000, 1)}


def test_a_spread_is_never_computed_across_K(tmp_path):
    """The regression guard for the mistake that would look like a real effect."""
    for rot, v in zip(ROTATIONS, (9.0, 9.0, 9.0, 9.0)):
        _write(tmp_path, 40000, 1, rot, t60=v)
    for rot, v in zip(ROTATIONS, (5.0, 5.0, 5.0, 5.0)):
        _write(tmp_path, 40000, 8, rot, t60=v)
    rows = orbit_rows(load_cells(tmp_path))
    assert all(r.spread["T60"] == pytest.approx(0.0) for r in rows), (
        "a K=1-vs-K=8 difference leaked into a rotation spread"
    )


def test_every_reported_metric_is_covered(tmp_path):
    _full_orbit(tmp_path)
    (row,) = orbit_rows(load_cells(tmp_path))
    for key in METRIC_KEYS:
        assert key in row.at_0 and key in row.spread


def test_rows_are_ordered_by_step_then_K(tmp_path):
    for step in (40000, 2500, 20000):
        for k in (8, 1):
            _full_orbit(tmp_path, step=step, k=k)
    rows = orbit_rows(load_cells(tmp_path))
    assert [(r.step, r.k) for r in rows] == sorted((r.step, r.k) for r in rows)
