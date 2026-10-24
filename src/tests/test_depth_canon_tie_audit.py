"""exp_25 depth_canon round 3b -- the pre-FULL tie audit
(``worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py``,
plan_depth_canon.md v8 §7 rung 3).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §7/§8.

CPU only, no dataset needed: synthetic scalar maps come from ``test_depth_canon``'s
seeded helper, split JSONs are written in the real ``{scene: {scene_id: [file, ...]}}``
schema of ``data/AR/*.json`` into a tmp tree.
"""
import importlib.util
import os

import numpy as np
import pytest
import torch

from src.data import yaw_rotation as yr
from test_depth_canon import _scalar_depth

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AUDIT_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_25_depth_canon_claude",
                         "depth_canon_tie_audit.py")
W = 512


def _load_audit():
    spec = importlib.util.spec_from_file_location("depth_canon_tie_audit", _AUDIT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()

# (name, peaks, period, expected raw tie class or None for a survivor)
FIXTURES = [
    ("unique", [(40, 300)], None, "none"),
    ("same_column", [(10, 77), (200, 77)], None, "same_column"),
    ("row_tie_sep1", [(90, 5), (90, 6)], None, "cross_column_image"),
    ("row_tie_wrap64", [(120, 480), (120, 32)], None, "cross_column_image"),
    ("row_tie_sep256", [(7, 100), (7, 356)], None, "cross_column_image"),
    ("cross_row_tie", [(30, 50), (220, 400)], None, "cross_column_image"),
    ("periodic_256", [(60, 11)], 256, None),
    ("periodic_128", [(60, 11)], 128, None),
]


def _fixture(name, peaks, period, seed=3):
    return _scalar_depth(256, W, seed=seed, period=period, peaks=peaks)


# --------------------------------------------------------------------------- #
# raw-side selector (pure numpy, independent of the proxy implementation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, peaks, period, want", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_raw_select_agrees_with_the_operator_selector(name, peaks, period, want):
    d = _fixture(name, peaks, period)
    j, tie_class, cands, survivors = audit.raw_select(d.numpy())
    P = d.max(dim=0).values
    assert cands == torch.nonzero(P == P.max()).flatten().tolist()
    if want is None:
        with pytest.raises(yr.StabilizedTieError) as err:
            yr.canonical_column(d)
        assert (j, tie_class) == (None, None) and tuple(survivors) == err.value.candidates
    else:
        assert (j, tie_class) == yr.canonical_column(d) and tie_class == want
        assert survivors == [j]


@pytest.mark.parametrize("roll", [1, 64, 255, 256, 511])
@pytest.mark.parametrize("name, peaks, period, want", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_raw_select_co_rotates(name, peaks, period, want, roll):
    raw = _fixture(name, peaks, period).numpy()
    j, tie_class, cands, survivors = audit.raw_select(raw)
    j2, tie_class2, cands2, survivors2 = audit.raw_select(np.roll(raw, roll, axis=1))
    assert tie_class2 == tie_class
    assert sorted(cands2) == sorted((c + roll) % W for c in cands)
    assert sorted(survivors2) == sorted((s + roll) % W for s in survivors)
    assert j2 == (None if j is None else (j + roll) % W)


def test_raw_select_uses_the_rooted_image_not_the_first_index():
    """A cross-column tie whose rooted-image winner is the LARGER column index."""
    for seed in range(20):
        raw = _fixture("t", [(9, 20), (9, 300)], None, seed=seed).numpy()
        j, tie_class, cands, _ = audit.raw_select(raw)
        if j == 300:
            assert cands == [20, 300] and tie_class == "cross_column_image"
            return
    pytest.fail("no seed gave a non-first-index winner")


@pytest.mark.parametrize("bad", [np.zeros((4,)), np.zeros((2, 3, 4)), np.zeros((0, 4)),
                                 np.zeros((4, 4), dtype=np.int64),
                                 np.array([[1.0, np.nan], [0.0, 1.0]]), [[1.0, 2.0]]])
def test_raw_select_refuses_malformed_maps(bad):
    with pytest.raises(ValueError):
        audit.raw_select(bad)


@pytest.mark.parametrize("cands, want", [([300], 0.0), ([0, 256], 180.0), ([480, 32], 45.0),
                                         ([1, 2, 511], 3 * 360.0 / W), ([5, 6], 360.0 / W)])
def test_max_circular_separation_in_degrees(cands, want):
    assert audit.max_circular_sep_deg(cands, W) == pytest.approx(want, abs=1e-12)
