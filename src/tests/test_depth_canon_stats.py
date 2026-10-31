"""exp_25 depth_canon -- tests for ``src/tools/depth_canon_stats.py``, the pure helpers
behind the Tier-2 waveform comparison and the numerical-composition control
(plan_depth_canon.md v8 §1 Tier 2, §6 test 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §2.2/§5/§6/§8.

Pinned denominators (plan §1): ``relL2 = ||a - b||_2 / max(||b||_2, 1e-6)`` and
``relLinf = max|a - b| / max(max|b|, 1e-6)``, per item over ``[N, T]``, with ``b``
the reference (theta = 0) waveform.
"""
import numpy as np
import pytest
import torch

import src.tools.depth_canon_stats as st

RNG = np.random.default_rng(0)
A, B = RNG.standard_normal((6, 40)), RNG.standard_normal((6, 40))


@pytest.mark.parametrize("fn", [st.rel_l2, st.rel_linf])
def test_identical_inputs_give_exact_zeros(fn):
    got = fn(A, A.copy())
    assert got.shape == (6,) and got.dtype == np.float64 and np.all(got == 0.0)


def test_known_values_and_the_reference_is_the_second_argument():
    a, b = np.array([[1.0, 1.0], [0.0, 2.0]]), np.array([[1.0, 0.0], [0.0, 4.0]])
    assert st.rel_l2(a, b).tolist() == [1.0, 0.5]
    assert st.rel_linf(a, b).tolist() == [1.0, 0.5]
    np.testing.assert_allclose(st.rel_l2(b, a), [2 ** -0.5, 1.0])  # not symmetric: b is the reference


def test_the_zero_vector_floor():
    a, b = np.full((1, 4), 1e-3), np.zeros((1, 4))
    assert st.REL_FLOOR == 1e-6
    assert st.rel_l2(a, b)[0] == pytest.approx(np.linalg.norm(a) / 1e-6)
    assert st.rel_linf(a, b)[0] == pytest.approx(1e-3 / 1e-6)
    tiny = np.full((1, 4), 1e-12)                          # ||b|| below the floor -> the floor
    assert st.rel_l2(a, tiny)[0] == pytest.approx(np.linalg.norm(a - tiny) / 1e-6)


@pytest.mark.parametrize("k", [3.7, 1e-2, 250.0])
def test_scale_invariance(k):
    np.testing.assert_allclose(st.rel_l2(k * A, k * B), st.rel_l2(A, B), rtol=1e-12)
    np.testing.assert_allclose(st.rel_linf(k * A, k * B), st.rel_linf(A, B), rtol=1e-12)


def test_rows_are_independent_items():
    full = st.rel_l2(A, B)
    assert [st.rel_l2(A[i:i + 1], B[i:i + 1])[0] for i in range(6)] == full.tolist()


def test_cpu_tensors_float32_and_float64_are_accepted():
    a32, b32 = torch.from_numpy(A).float(), torch.from_numpy(B).float()
    np.testing.assert_allclose(st.rel_l2(a32, b32), st.rel_l2(a32.numpy(), b32.numpy()))
    assert st.rel_linf(torch.from_numpy(A), torch.from_numpy(B)).tolist() == st.rel_linf(A, B).tolist()


@pytest.mark.parametrize("a, b", [
    (A, B[:, :-1]), (A[0], B[0]), (A[:0], B[:0]), (A[:, :0], B[:, :0]),
    (np.where(A > 1.5, np.nan, A), B), (A, np.where(B > 1.5, np.inf, B))])
def test_malformed_inputs_are_refused(a, b):
    for fn in (st.rel_l2, st.rel_linf):
        with pytest.raises(ValueError):
            fn(a, b)


def test_paired_summary_is_median_p99_max():
    v = RNG.random(1000)
    s = st.paired_summary(v)
    assert s == {"n": 1000, "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                 "max": float(v.max())}
    assert st.paired_summary(np.arange(101.0)) == {"n": 101, "median": 50.0, "p99": 99.0,
                                                    "max": 100.0}
    assert all(type(x) is float for k, x in s.items() if k != "n")


@pytest.mark.parametrize("bad", [np.array([]), np.zeros((2, 2)), np.array([1.0, np.nan]), [1.0, -1.0]])
def test_paired_summary_refuses_empty_non_1d_non_finite_and_negative_values(bad):
    with pytest.raises(ValueError):
        st.paired_summary(bad)
