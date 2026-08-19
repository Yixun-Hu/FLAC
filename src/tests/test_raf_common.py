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


# --------------------------------------------------------------------------- #
# equirect_directions (cycle 2)
# --------------------------------------------------------------------------- #
import hashlib      # noqa: E402
import importlib.util  # noqa: E402

import torch        # noqa: E402


def _load_haa_md():
    """Load HAA_md.py the same way the dataloader does (dynamic, by file path)."""
    path = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs",
                        "custom_metadata", "HAA_md.py")
    spec = importlib.util.spec_from_file_location("haa_md_for_raf_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equirect_directions_shape_dtype_and_unit_norm():
    dirs = raf_common.equirect_directions()
    assert dirs.shape == (256, 512, 3)
    assert dirs.dtype == np.float32
    norms = np.linalg.norm(dirs.astype(np.float64), axis=-1)
    assert np.abs(norms - 1.0).max() < 1e-6


def test_equirect_directions_exact_roundtrip_at_unit_depth():
    """``1.0 * dir`` must reproduce ``convert_equirect_to_camera_coord`` bit-exactly.

    Multiplication by 1.0 is exact in IEEE-754, so this is the strongest possible
    statement of "the directions ARE the inverse of the pipeline's pixel->ray
    equation": every trig value must agree to the last bit.
    """
    haa_md = _load_haa_md()
    depth = torch.ones(256, 512)
    expected = haa_md.convert_equirect_to_camera_coord(depth, 256, 512)
    got = depth.unsqueeze(-1) * torch.from_numpy(raf_common.equirect_directions())
    assert torch.equal(expected, got)


def test_equirect_directions_roundtrip_general_depth_within_one_ulp():
    """Same identity for a non-constant depth map, up to float32 non-associativity.

    ``convert_equirect_to_camera_coord`` evaluates ``(d * cos_phi) * cos_theta``
    while the factored form evaluates ``d * (cos_phi * cos_theta)``. Float
    multiplication is not associative, so bit-equality is unattainable here by
    construction; agreement to ~1 ULP is the correct contract.
    """
    haa_md = _load_haa_md()
    g = torch.Generator().manual_seed(19)
    depth = torch.rand(256, 512, generator=g) * 12.0 + 0.5
    expected = haa_md.convert_equirect_to_camera_coord(depth, 256, 512)
    got = depth.unsqueeze(-1) * torch.from_numpy(raf_common.equirect_directions())
    assert torch.allclose(expected, got, rtol=1e-6, atol=0.0)


def test_equirect_directions_row_zero_is_the_zenith_hand_computed():
    """Hand-computed orientation oracle for pixel (0, 0) at 256x512.

    phi_0   = (0 + 0.5) * pi/256 - pi/2 = pi/512 - pi/2
    theta_0 = (0 + 0.5) * 2pi/512 - pi  = pi/512 - pi
    cos(pi/512) = 0.9999811752, sin(pi/512) = 0.0061358846
      cos(phi_0)   =  sin(pi/512)  =  0.0061358846
      -sin(phi_0)  =  cos(pi/512)  =  0.9999811752
      cos(theta_0) = -cos(pi/512)  = -0.9999811752
      sin(theta_0) = -sin(pi/512)  = -0.0061358846
    => dir = (0.0061358846 * -0.9999811752,
              0.0061358846 * -0.0061358846,
              0.9999811752)
           = (-0.0061357691, -0.0000376491, 0.9999811752)
    Row 0 is therefore the ZENITH (+z, up) — there is no flipud anywhere in the
    RAF path, so this row order is what the renderer must emit.
    """
    dirs = raf_common.equirect_directions()
    np.testing.assert_allclose(
        dirs[0, 0], np.array([-0.0061357691, -0.0000376491, 0.9999811752]), atol=1e-6)
    assert dirs[0, :, 2].min() > 0.999      # top row looks up
    assert dirs[255, :, 2].max() < -0.999   # bottom row looks down


def test_equirect_directions_column_half_width_faces_plus_x():
    """theta at j = W/2 is +pi/512, i.e. essentially the +x (front) axis."""
    dirs = raf_common.equirect_directions()
    eq = dirs[128, 256]  # row 128 is just above the horizon, column 256 is +x
    assert eq[0] > 0.99
    assert 0.0 < eq[1] < 0.01


def test_equirect_directions_small_grid_matches_hand_computed_45_degree_rays():
    """h=2, w=4 gives exactly 45-degree rays — the fixture used by the depth tests.

    phi_i   = (i + 0.5) * pi/2 - pi/2  -> -pi/4 (i=0), +pi/4 (i=1)
    theta_j = (j + 0.5) * pi/2 - pi    -> -3pi/4, -pi/4, +pi/4, +3pi/4
    cos/sin of +-pi/4 = +-sqrt(2)/2 = +-0.70710678, so every component is
    +-0.5 (horizontal) or +-0.70710678 (vertical).
    """
    c = 0.70710678
    dirs = raf_common.equirect_directions(2, 4)
    assert dirs.shape == (2, 4, 3)
    expected = np.array([
        [[-0.5, -0.5, c], [0.5, -0.5, c], [0.5, 0.5, c], [-0.5, 0.5, c]],
        [[-0.5, -0.5, -c], [0.5, -0.5, -c], [0.5, 0.5, -c], [-0.5, 0.5, -c]],
    ])
    np.testing.assert_allclose(dirs, expected, atol=1e-6)


def test_equirect_directions_returns_a_fresh_array():
    a = raf_common.equirect_directions(2, 4)
    a[0, 0, 0] = 12345.0
    b = raf_common.equirect_directions(2, 4)
    assert b[0, 0, 0] != 12345.0


@pytest.mark.parametrize("h,w", [(0, 4), (2, 0), (-2, 4), (2, -4)])
def test_equirect_directions_rejects_bad_grid(h, w):
    with pytest.raises(ValueError):
        raf_common.equirect_directions(h, w)


# --------------------------------------------------------------------------- #
# stable_context_seed (cycle 2)
# --------------------------------------------------------------------------- #
def test_stable_context_seed_matches_independent_sha256_golden():
    """Golden value produced OUTSIDE this codebase (GNU coreutils sha256sum):

        $ printf 'RAF|EmptyRoom|000123' | sha256sum
        4587fa2e5363df5b7613293d1b43e038...
        int(0x4587fa2e5363df5b) & (2**63 - 1) = 5010248187347459931
    """
    assert raf_common.stable_context_seed("EmptyRoom", "000123") == 5010248187347459931


def test_stable_context_seed_reimplemented_formula():
    for room, cid in [("EmptyRoom", "000000"), ("FurnishedRoom", "039131")]:
        digest = hashlib.sha256(f"RAF|{room}|{cid}".encode("utf-8")).digest()
        expected = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        assert raf_common.stable_context_seed(room, cid) == expected


def test_stable_context_seed_is_deterministic_and_discriminating():
    a = raf_common.stable_context_seed("EmptyRoom", "000001")
    assert a == raf_common.stable_context_seed("EmptyRoom", "000001")
    assert a != raf_common.stable_context_seed("FurnishedRoom", "000001")
    assert a != raf_common.stable_context_seed("EmptyRoom", "000002")
    # the separator must not let (room, id) pairs collide by concatenation
    assert (raf_common.stable_context_seed("Empty", "Room|000001")
            != raf_common.stable_context_seed("EmptyRoom", "000001"))


def test_stable_context_seed_is_a_valid_torch_generator_seed():
    seed = raf_common.stable_context_seed("EmptyRoom", "047483")
    assert isinstance(seed, int)
    assert 0 <= seed < (1 << 63)
    torch.Generator().manual_seed(seed)  # must not raise
