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
import hashlib

import numpy as np
import torch


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


def equirect_directions(img_h=256, img_w=512):
    """Unit ray directions for an equirectangular map, in the PIPELINE frame.

    These are exactly the inverse of ``convert_equirect_to_camera_coord``
    (``src/configs/dataset_configs/custom_metadata/HAA_md.py``), i.e. for a depth
    map ``d`` the pipeline's per-pixel point cloud is ``d[..., None] * dirs``:

        theta_j = (j + 0.5) * 2*pi / W - pi
        phi_i   = (i + 0.5) * pi / H - pi/2
        dir     = (cos(phi) cos(theta), cos(phi) sin(theta), -sin(phi))

    Row 0 is therefore the ZENITH (+z, up) and row H-1 the nadir. The RAF path
    applies NO flipud anywhere: the renderer emits rows in exactly this order and
    ``RAF_md.py`` consumes them unchanged.

    Implemented with torch (not numpy) on purpose: torch and numpy trig differ by
    a ULP on this grid, and only the torch expressions reproduce
    ``convert_equirect_to_camera_coord`` bit-for-bit.

    Returns:
        float32 [H, W, 3]; a fresh array on every call.
    """
    for name, value in (("img_h", img_h), ("img_w", img_w)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an int, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    dirs = torch.stack([cos_phi * cos_theta, cos_phi * sin_theta, -sin_phi], dim=-1)
    return dirs.contiguous().numpy()


def stable_context_seed(room, capture_id):
    """Platform-stable per-item seed for deterministic (eval) context draws.

    ``sha256("RAF|<room>|<capture id>")``, first 8 bytes big-endian, masked to 63
    bits so it is always a valid ``torch.Generator().manual_seed`` argument.
    Python's own ``hash()`` is salted per process and must never be used here: the
    eval context set has to be identical across processes, worker topologies,
    checkpoints and seeds.
    """
    if not isinstance(room, str) or not isinstance(capture_id, str):
        raise ValueError(
            "room and capture_id must be str (capture ids are zero-padded 6-digit "
            f"strings), got {type(room).__name__}/{type(capture_id).__name__}")
    digest = hashlib.sha256(f"RAF|{room}|{capture_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


# Registered candidate gauge, RAF world (X front, Y up, Z left) -> pipeline frame:
#     (x_p, y_p, z_p) = (X_RAF, Z_RAF, Y_RAF)
# so the pipeline's third axis is UP, matching AR/HAA (whose poses carry the
# camera/mic height in the third slot) and matching the vertical component of
# convert_equirect_to_camera_coord. det = -1: RAF's (front, up, left) triad is
# left-handed, the pipeline frame is right-handed.
#
# CANDIDATE, NOT FINAL: the mapping is pinned at the readback rung (plan Rev 2
# section 4). Everything downstream imports this one constant, so re-pinning is a
# one-line change here; nothing else in the RAF path transforms coordinates.
RAF_TO_PIPELINE = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def farthest_point_selection(points, k, start='centroid-nearest'):
    """Deterministic farthest-point selection over ``points`` [N, 3].

    Start = the point nearest the centroid (ties -> lowest index); each further
    step takes the point maximising the distance to the already-selected set
    (ties -> lowest index). No RNG is involved, so the result depends only on the
    input array — the split it produces is reproducible from the record alone.

    The returned sequence is a PREFIX sequence: ``fps(k)[:j] == fps(j)``, which is
    what lets ``select_splits`` take the first N_g entries as train/test groups and
    then *continue the same sequence* for the validation groups.

    Returns:
        list[int] of length k.
    """
    if start != 'centroid-nearest':
        raise ValueError(f"unsupported start policy {start!r}; only 'centroid-nearest'")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points hold non-finite values")
    n = pts.shape[0]
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k must be an int, got {k!r}")
    k = int(k)
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= N ({n}), got {k}")

    first = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    selected = [first]
    # Distance from every point to the selected set. Already-selected points are
    # parked at -1 so a degenerate all-duplicate cloud (every distance 0) can
    # never re-select one of them.
    min_dist = np.linalg.norm(pts - pts[first], axis=1)
    min_dist[first] = -1.0
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.linalg.norm(pts - pts[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -1.0
    return selected


# dBFS floor for an all-zero signal. -inf is not valid JSON (json.dump emits the
# non-standard -Infinity token), and every RAF audit artifact must be strictly
# parseable, so silence is reported at this floor instead.
DBFS_FLOOR = -200.0


def dbfs(peak):
    """Peak amplitude -> dBFS, floored at ``DBFS_FLOOR`` instead of -inf."""
    peak = float(peak)
    return float(20.0 * np.log10(peak)) if peak > 0.0 else DBFS_FLOOR


def distance_stats(values):
    """count/min/p25/median/p75/max/mean of a 1-D sample, JSON-safe."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None,
                "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
