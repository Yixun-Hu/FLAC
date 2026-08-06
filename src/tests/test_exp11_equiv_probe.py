"""Tests for the exp_11 batched-orbit qualification probe's decision logic.

The probe is an executable that decides whether the batched orbit execution is
acceptable, so its metric and verdict functions are regression assets in their
own right (SOP: every new executable gets tests). Everything here is pure —
tensors in, numbers out — so no GPU, no dataset and no DINOv3 are needed; the
probe keeps its heavy imports inside the functions that need them.

The properties under test are the ones the review demanded be fail-CLOSED:
both a normwise-relative AND a scale-aware max-absolute bound decide a gated
cell, near-zero references cannot manufacture a relative blow-up, NaN can never
be suppressed into a pass, and a missing cell or a missing ViT id is a FAIL
rather than a silent zero.
"""
import importlib.util
import math
import os

import pytest
import torch


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_PROBE_PY = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
    "fa_orbit_equiv_probe.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp11_equiv_probe", _PROBE_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_PROBE_PY}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_module()


def _cell(max_abs=0.0, rel_norm=0.0, rel_max=0.0, gated=True, finite=True,
          ids=("source_vit", "context_poses_vit")):
    return {
        "ids": {v: {"max_abs": max_abs, "rel_norm": rel_norm, "rel_max": rel_max} for v in ids},
        "finite": finite, "grads_finite": None, "gated": gated,
    }


# --------------------------------------------------------------------------- #
# 1. the cell plan
# --------------------------------------------------------------------------- #
def test_expected_cells_cover_the_review_matrix():
    cells = P.expected_cells()
    # B8 (the pinned per-rank training batch) for every orbit
    for n in (4, 8, 16, 32):
        assert ("eval", n, 8) in cells
    # the evaluation schedules: full batch and the 6,337-split tail
    assert ("eval", 4, 64) in cells and ("eval", 32, 1) in cells
    # train-mode qualification cells
    assert ("train", 4, 8) in cells and ("train", 32, 8) in cells
    assert len(cells) == len(set(cells)), "duplicate cells in the plan"


def test_orbit_angles():
    assert P.orbit(4) == (0.0, 90.0, 180.0, 270.0)
    assert P.orbit(32)[1] == pytest.approx(11.25)
    assert len(P.orbit(16)) == 16


# --------------------------------------------------------------------------- #
# 2. the deviation metric
# --------------------------------------------------------------------------- #
def test_deviation_is_zero_for_identical_tensors():
    a = torch.randn(4, 8)
    max_abs, rel_norm, rel_max = P.deviation(a, a.clone())
    assert (max_abs, rel_norm, rel_max) == (0.0, 0.0, 0.0)


def test_deviation_reports_absolute_and_normwise_error():
    b = torch.ones(2, 3)
    a = b + 0.5
    max_abs, rel_norm, rel_max = P.deviation(a, b)
    assert max_abs == pytest.approx(0.5)
    assert rel_norm == pytest.approx(0.5)          # ||0.5|| / ||1||
    assert rel_max == pytest.approx(0.5)


def test_deviation_floor_tames_a_near_zero_reference():
    """Elementwise relative error must not explode against a ~0 reference."""
    b = torch.tensor([[1e-12, 1.0]])
    a = torch.tensor([[1e-12 + 1e-9, 1.0]])
    _, rel_norm, rel_max = P.deviation(a, b)
    assert math.isfinite(rel_max) and rel_max <= 1e-9 / P.REL_ABS_FLOOR + 1
    assert rel_norm < 1e-8


def test_deviation_is_nan_proof():
    """A NaN anywhere yields inf — max(0.0, NaN) must never hide it."""
    b = torch.ones(2, 2)
    a = b.clone()
    a[0, 0] = float("nan")
    assert P.deviation(a, b) == (float("inf"),) * 3
    a[0, 0] = float("inf")
    assert P.deviation(a, b) == (float("inf"),) * 3


# --------------------------------------------------------------------------- #
# 3. the verdict: fail-closed in every direction
# --------------------------------------------------------------------------- #
def test_verdict_passes_a_complete_clean_run():
    plan = P.expected_cells()
    results = {c: _cell(gated=(c[0] == "eval")) for c in plan}
    ok, reasons = P.verdict(results, plan)
    assert ok and reasons == []


def test_verdict_fails_on_a_missing_cell():
    plan = P.expected_cells()
    results = {c: _cell() for c in plan[1:]}
    ok, reasons = P.verdict(results, plan)
    assert not ok and any("missing cells" in r for r in reasons)


def test_verdict_fails_on_an_empty_result_set():
    ok, reasons = P.verdict({}, P.expected_cells())
    assert not ok and any("no results" in r for r in reasons)


def test_verdict_fails_when_a_vit_id_is_absent():
    plan = (("eval", 4, 8),)
    results = {plan[0]: _cell(ids=("source_vit",))}
    ok, reasons = P.verdict(results, plan)
    assert not ok and any("ViT ids" in r for r in reasons)


def test_verdict_enforces_both_bounds_on_gated_cells():
    plan = (("eval", 4, 8),)
    over_rel = {plan[0]: _cell(rel_norm=P.TOL_REL_FP32 * 10, max_abs=0.0)}
    ok, reasons = P.verdict(over_rel, plan)
    assert not ok and any("rel_norm" in r for r in reasons)

    over_abs = {plan[0]: _cell(rel_norm=0.0, max_abs=P.TOL_ABS_FP32 * 10)}
    ok, reasons = P.verdict(over_abs, plan)
    assert not ok and any("max_abs" in r for r in reasons), "max_abs must decide, not just print"


def test_verdict_records_but_does_not_gate_train_mode_divergence():
    """Train-mode divergence is the DISCLOSED recipe change (chunk-shared RoPE
    draws), not an error: it is recorded, and only finiteness is required."""
    plan = (("train", 32, 8),)
    big = {plan[0]: _cell(rel_norm=0.5, max_abs=1.0, gated=False)}
    ok, reasons = P.verdict(big, plan)
    assert ok and reasons == []

    nonfinite = {plan[0]: _cell(rel_norm=float("nan"), gated=False)}
    ok, reasons = P.verdict(nonfinite, plan)
    assert not ok and any("non-finite" in r for r in reasons)


def test_verdict_fails_on_non_finite_forward_tensors():
    plan = (("eval", 4, 8),)
    ok, reasons = P.verdict({plan[0]: _cell(finite=False)}, plan)
    assert not ok and any("non-finite tensors" in r for r in reasons)


# --------------------------------------------------------------------------- #
# 4. the reported summary
# --------------------------------------------------------------------------- #
def test_summarize_separates_gated_from_recorded():
    results = {
        ("eval", 4, 8): _cell(rel_norm=1e-7, max_abs=1e-6, gated=True),
        ("train", 4, 8): _cell(rel_norm=0.4, max_abs=2.0, gated=False),
    }
    gated_rel, rec_rel = P.summarize(results, "rel_norm")
    gated_abs, rec_abs = P.summarize(results, "max_abs")
    assert gated_rel == pytest.approx(1e-7) and rec_rel == pytest.approx(0.4)
    assert gated_abs == pytest.approx(1e-6) and rec_abs == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 5. exact record identities (re-review NEW-3)
# --------------------------------------------------------------------------- #
def test_record_id_uses_the_relative_path_not_the_scene_label():
    """The scene label is not an identifier: eight records of one room all carry
    scene='Cafe', which is exactly what the old probe emitted eight times."""
    meta = {"scene": "Cafe", "idx": 3, "path": "/data/AR/Cafe/ir_0007.wav",
            "relpath": "Cafe/ir_0007.wav"}
    assert P.record_id(meta, 3) == "3:Cafe/ir_0007.wav"
    assert "Cafe/ir_0007.wav" in P.record_id(meta, 3)


def test_record_ids_are_distinct_across_records_of_one_scene():
    metas = [{"scene": "Cafe", "idx": i, "relpath": f"Cafe/ir_{i:04d}.wav"} for i in range(8)]
    ids = [P.record_id(m, i) for i, m in enumerate(metas)]
    assert len(set(ids)) == 8, ids
    assert all(":" in i for i in ids)


def test_record_id_falls_back_through_path_then_index():
    assert P.record_id({"idx": 5, "path": "/x/y/ir_5.wav"}, 5) == "5:ir_5.wav"
    assert P.record_id({"idx": 9}, 9) == "9:record9"
    assert P.record_id({}, 2) == "2:record2"          # no idx at all


def test_tolerances_are_the_preregistered_ones():
    assert P.TOL_REL_FP32 == 1e-6
    assert P.TOL_ABS_FP32 == 1e-5
    assert P.REL_ABS_FLOOR == 1e-8
    assert P.N_SAMPLES == 8
