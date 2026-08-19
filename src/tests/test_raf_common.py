"""Tests for ``data/RAF/raf_common.py`` — the single source of RAF gauge/equirect truth.

exp_19 (RAF finetune), contract section A. Developed test-first, one commit per
TDD cycle:

* cycle 1 — ``parse_tx_line`` / ``parse_rx_line`` / ``canonicalize_quat``
* cycle 2 — ``equirect_directions`` (+ round-trip against the pipeline's
  ``convert_equirect_to_camera_coord``) and ``stable_context_seed``
* cycle 3 — ``farthest_point_selection`` and the ``RAF_TO_PIPELINE`` constant

The RAF pose files are the only description of the capture geometry, so every
parser here is fail-closed: a malformed, short, long, or non-finite line must
raise rather than yield a plausible-looking pose that would silently mis-place a
capture in a group (and therefore in a split).
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import raf_common  # noqa: E402  (path prepended above)

# The RAF helpers are plain scripts under data/RAF/, not an installed package;
# pin the resolved file so a same-named module elsewhere on sys.path cannot
# silently satisfy the import.
assert os.path.dirname(os.path.abspath(raf_common.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# parse_tx_line
# --------------------------------------------------------------------------- #
_TX_LINE = "-0.030145,-0.998418,0.000917,-0.047452,-2.808042,1.640600,-0.695314"
_RX_LINE = "-3.292585,0.654455,0.353361"


def test_parse_tx_line_values_and_shapes():
    quat, xyz = raf_common.parse_tx_line(_TX_LINE)
    assert quat.shape == (4,)
    assert xyz.shape == (3,)
    assert quat.dtype == np.float64
    assert xyz.dtype == np.float64
    assert quat.tolist() == [-0.030145, -0.998418, 0.000917, -0.047452]
    assert xyz.tolist() == [-2.808042, 1.640600, -0.695314]


def test_parse_tx_line_tolerates_surrounding_whitespace():
    quat, xyz = raf_common.parse_tx_line("  " + _TX_LINE + " \n")
    assert xyz.tolist() == [-2.808042, 1.640600, -0.695314]
    assert quat[0] == -0.030145


@pytest.mark.parametrize("bad", [
    "",                                             # empty
    "   \n",                                        # blank
    "1,2,3,4,5,6",                                  # 6 fields
    "1,2,3,4,5,6,7,8",                              # 8 fields
    "1,2,3,4,5,6,seven",                            # non-numeric
    "1,2,3,4,5,6,",                                 # trailing separator
    "1 2 3 4 5 6 7",                                # space separated
    "1,2,3,4,5,6,nan",                              # non-finite
    "1,2,3,4,inf,6,7",                              # non-finite
])
def test_parse_tx_line_rejects_malformed(bad):
    with pytest.raises(ValueError):
        raf_common.parse_tx_line(bad)


# --------------------------------------------------------------------------- #
# parse_rx_line
# --------------------------------------------------------------------------- #
def test_parse_rx_line_values_and_shape():
    xyz = raf_common.parse_rx_line(_RX_LINE)
    assert xyz.shape == (3,)
    assert xyz.dtype == np.float64
    assert xyz.tolist() == [-3.292585, 0.654455, 0.353361]


@pytest.mark.parametrize("bad", [
    "",
    "1,2",
    "1,2,3,4",
    "1,2,three",
    "nan,nan,nan",       # the trailing sentinel line observed in all_rx_pos.txt
    "1,2,-inf",
])
def test_parse_rx_line_rejects_malformed(bad):
    with pytest.raises(ValueError):
        raf_common.parse_rx_line(bad)


# --------------------------------------------------------------------------- #
# canonicalize_quat
# --------------------------------------------------------------------------- #
def test_canonicalize_quat_flips_negative_leading_component():
    q = np.array([-0.5, 0.5, -0.5, 0.5])
    out = raf_common.canonicalize_quat(q)
    assert out.tolist() == [0.5, -0.5, 0.5, -0.5]


def test_canonicalize_quat_keeps_positive_leading_component():
    q = np.array([0.030145, -0.998418, 0.000917, -0.047452])
    out = raf_common.canonicalize_quat(q)
    assert out.tolist() == q.tolist()


def test_canonicalize_quat_skips_components_below_tolerance():
    # |q0| = 1e-13 < 1e-12 is treated as a zero component, so the sign is decided
    # by q1 (which is negative here and must therefore be flipped).
    q = np.array([1e-13, -0.6, 0.8, 0.0])
    out = raf_common.canonicalize_quat(q)
    assert out[1] == 0.6
    assert out[2] == -0.8
    assert out[0] == -1e-13


def test_canonicalize_quat_maps_q_and_minus_q_to_the_same_representative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        a = raf_common.canonicalize_quat(q)
        b = raf_common.canonicalize_quat(-q)
        assert np.array_equal(a, b)


def test_canonicalize_quat_does_not_mutate_input():
    q = np.array([-1.0, 2.0, 3.0, 4.0])
    before = q.copy()
    raf_common.canonicalize_quat(q)
    assert np.array_equal(q, before)


def test_canonicalize_quat_rejects_zero_quaternion():
    # A zero (or below-tolerance) quaternion is not a rotation; accepting it would
    # merge distinct source poses into one group with nothing downstream able to
    # notice.
    with pytest.raises(ValueError):
        raf_common.canonicalize_quat(np.zeros(4))


@pytest.mark.parametrize("bad", [
    np.zeros(3),
    np.zeros((2, 4)),
    np.array([1.0, 2.0, np.nan, 4.0]),
])
def test_canonicalize_quat_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        raf_common.canonicalize_quat(bad)
