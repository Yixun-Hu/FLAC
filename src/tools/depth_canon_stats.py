"""Paired waveform-difference statistics for exp_25 depth_canon (plan_depth_canon.md v8
§1 Tier 2 and the numerical-composition control, §6 test 21b).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md v8 §2.2/§5/§6/§8.

Pure helpers, no I/O. Per item (row) of two ``[N, T]`` arrays ``a`` (e.g. a rotated
cell's scored waveforms) and ``b`` (the REFERENCE, e.g. its theta = 0 pair), with the
denominators pinned in the plan:

* ``rel_l2(a, b)   = ||a - b||_2 / max(||b||_2, REL_FLOOR)``
* ``rel_linf(a, b) = max|a - b| / max(max|b|, REL_FLOOR)``

computed in float64. :func:`paired_summary` reduces such a per-item vector to the
registered statistics (median, ``np.percentile(..., 99)`` with numpy's default
linear interpolation, max).
"""
import numpy as np

REL_FLOOR = 1e-6


def _pair(a, b):
    """Both inputs as finite float64 ``[N, T]`` arrays of one shape, ``N, T >= 1``.
    Accepts numpy arrays and CPU torch tensors (anything ``np.asarray`` reads)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or 0 in a.shape:
        raise ValueError(f"need two [N, T] arrays of one shape with N, T >= 1, got "
                         f"{a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("inputs must be finite")
    return a, b


def rel_l2(a, b):
    """Per-item relative L2 error of ``a`` against the reference ``b`` -> ``[N]`` float64."""
    a, b = _pair(a, b)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), REL_FLOOR)


def rel_linf(a, b):
    """Per-item relative L-infinity error of ``a`` against ``b`` -> ``[N]`` float64."""
    a, b = _pair(a, b)
    return np.abs(a - b).max(axis=1) / np.maximum(np.abs(b).max(axis=1), REL_FLOOR)


def paired_summary(values):
    """``{"n", "median", "p99", "max"}`` of a non-empty 1-D vector of finite,
    non-negative per-item errors; ``p99 = np.percentile(values, 99)``."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"need a non-empty 1-D vector of finite non-negative values, got "
                         f"shape {v.shape}")
    return {"n": int(v.size), "median": float(np.median(v)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}
