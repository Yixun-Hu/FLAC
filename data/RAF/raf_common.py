"""Shared RAF helpers: coordinate gauge, pose parsing, equirect ray geometry.

exp_19 (RAF finetune), contract section A. This module is the SINGLE source of
truth for the RAF -> pipeline gauge and for the equirectangular ray convention:
``prepare_data.py`` and ``render_depth.py`` import it, and the loader hook
(``src/configs/dataset_configs/custom_metadata/RAF_md.py``) consumes only
already-transformed metadata — it never applies a transform of its own.

Every parser here is fail-closed. RAF's pose text files are the only description
of the capture geometry, and a silently-accepted malformed line would land a
capture in the wrong group, hence in the wrong split, with nothing downstream
able to detect it.
"""
import numpy as np


QUAT_ZERO_TOL = 1e-12


def parse_tx_line(s):
    """Parse one ``all_tx_pos.txt`` / ``tx_pos.txt`` line.

    Format: ``q0,q1,q2,q3,x,y,z`` (7 comma-separated finite floats).

    The quaternion COLUMN ORDER IS UNVERIFIED (wxyz vs xyzw is not stated by the
    RAF release and has not been pinned at the readback rung). It is therefore
    returned as an opaque 4-tuple, used only for grouping/identity — never to
    rotate anything — until readback verifies the order.

    Returns:
        (quat [4] float64, xyz [3] float64)
    """
    values = _parse_float_fields(s, 7, "tx")
    return values[:4].copy(), values[4:].copy()


def parse_rx_line(s):
    """Parse one ``all_rx_pos.txt`` / ``rx_pos.txt`` line: ``x,y,z``.

    Returns:
        xyz [3] float64
    """
    return _parse_float_fields(s, 3, "rx")


def _parse_float_fields(s, arity, kind):
    """Strict comma-separated float parser: exact arity, all fields finite."""
    if not isinstance(s, str):
        raise ValueError(f"{kind} line must be a str, got {type(s).__name__}")
    stripped = s.strip()
    if not stripped:
        raise ValueError(f"empty {kind} line")
    fields = stripped.split(",")
    if len(fields) != arity:
        raise ValueError(
            f"{kind} line must hold exactly {arity} comma-separated values, "
            f"got {len(fields)}: {stripped!r}")
    out = np.empty(arity, dtype=np.float64)
    for i, field in enumerate(fields):
        try:
            out[i] = float(field)
        except (TypeError, ValueError):
            raise ValueError(
                f"{kind} line field {i} is not a float: {field!r} (line {stripped!r})")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{kind} line holds a non-finite value: {stripped!r}")
    return out


def canonicalize_quat(q):
    """Return the canonical representative of ``q`` under the ``q == -q`` identity.

    The sign is chosen so that the FIRST component with ``|v| > QUAT_ZERO_TOL`` is
    positive. Components at or below the tolerance are treated as zero and skipped
    (their own sign is left untouched — it carries no information).

    Fail-closed on an all-zero quaternion: it is not a rotation, and accepting it
    would merge physically distinct source poses into a single group.
    """
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"quaternion holds a non-finite value: {arr.tolist()}")
    for v in arr:
        if abs(v) > QUAT_ZERO_TOL:
            return (arr.copy() if v > 0 else -arr)
    raise ValueError(
        f"quaternion is zero to within {QUAT_ZERO_TOL}: {arr.tolist()}; "
        "it does not describe a rotation and cannot be canonicalized")
