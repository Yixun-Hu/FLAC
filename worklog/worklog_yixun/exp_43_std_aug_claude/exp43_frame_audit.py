#!/usr/bin/env python
"""exp_43 std_aug -- frame-alignment audit of the AcousticRooms depth panoramas (plan_std_aug.md 4.1).

Question: is the frame of the stored depth vectors the frame of the poses (src_loc / rec_loc), or does
``v_world = R(beta) S v_depth`` hold with some other heading beta or the opposite handedness S = M?

Subcommands
-----------
``freeze-inputs --ar-root AR --split data/AR/train.json --out INPUTS.json``
    The deterministic input selection (RULE_TEXT below, also written into INPUTS.json). Every selected
    file is recorded with its path relative to AR and its sha256. Files are read only to hash them and to
    validate their FORMAT; this step computes no score of any kind. Refuses to overwrite INPUTS.json.
``run --inputs INPUTS.json --ar-root AR --split data/AR/train.json --out RESULT.json --gate {registered,a2} [--threads N]``
    Bound to the registered inputs: refuses (nothing written) unless the bytes of INPUTS.json hash to
    REGISTERED_INPUTS_SHA256 and the canonical sha256 of its selection is REGISTERED_SELECTION_SHA256 (the
    committed ``exp43_frame_audit_inputs.json``); no flag changes or skips this. Re-hashes every input (any
    missing file or sha256 mismatch: refused, nothing written). ``--split`` is REQUIRED (missing: exit 2,
    nothing written): the selection is ALWAYS re-derived from the split and the AR tree and the run
    refuses unless it is identical (the result records "split_reverified": true). The registered
    categories are those of the frozen selection; each must hold at least one room, otherwise refused.
    Computes Score A and Score B for all 1,024 hypotheses; evaluates BOTH the registered gate and the
    proposed amendment A2; writes RESULT.json (refuses to overwrite) with both verdict blocks
    ("gate_registered", "gate_a2_proposed"), all curves, per-room arrays and provenance. ``--gate`` is
    REQUIRED (no default; missing or unknown: exit 2, nothing written): the named gate is the
    "gate_of_record" and alone drives the top-level "verdict" / "reasons" and the exit code; the other
    block is still computed and written.
Exit codes: 0 = PASS; 3 = FAIL (RESULT.json written); 2 = tool error / refusal (nothing written).
Python API: ``run_audit(..., expected_inputs_sha256=..., expected_selection_sha256=...)`` (keyword-only,
defaulting to the registered pair; the CLI never passes them) lets the synthetic tests bind their own
inputs. RESULT.json always carries "inputs_binding" with both bound hashes and the registered ones;
its "registered" is true only for the registered pair, and any other binding is labelled "NOT the
official exp_43 frame-audit record".

Conventions (AcousticRooms; ``src/configs/dataset_configs/custom_metadata/AR_md.py``)
--------------------------------------------------------------------------------------
Panorama H x W = 256 x 512. Pixel (i, j) (row i, column j) has the depth-frame unit vector
    e(i, j) = (cos phi_i cos theta_j, cos phi_i sin theta_j, -sin phi_i),
    theta_j = (j + 0.5) 2 pi / W - pi,     phi_i = (i + 0.5) pi / H - pi / 2,
and the vector ``v_depth(i, j) = D(i, j) e(i, j)`` (``convert_equirect_to_camera_coord``), D being the
scalar map ``<AR>/depth_map/<category>/<room>/<receiver>.npy`` (range along the ray). Row 0 looks up (+z);
theta = 0 (+x) is the seam between columns 255 and 256. World positions are ``src_loc`` / ``rec_loc`` of
``<AR>/metadata/<category>/<room>/S00{s}_R00{r}.json`` (the name AR_md builds); the model's pose vectors
are ``src_loc - rec_loc`` in world axes.

Hypotheses h = (S, k), S in {I, M = diag(1, -1, 1)}, k in 0..511, flat index h = 512 [S = M] + k:
    v_world = R(beta_k) S v_depth,   beta_k = 2 pi k / W,
    R(beta) = [[cos beta, -sin beta, 0], [sin beta, cos beta, 0], [0, 0, 1]]  (counter-clockwise
    about +z, +x towards +y).
Sign of k: under (I, k) depth column j looks along world azimuth theta_j + beta_k = theta_{j+k}, so a
world direction whose identity-frame column is c is stored in column (c - k) mod W, and the
world-aligned panorama is ``np.roll(D, k, axis=1)``. Under (M, k) column j looks along azimuth
beta_k - theta_j = theta_{W-1-j+k}; the world-aligned panorama is ``np.roll(D[:, ::-1], k, axis=1)``.

World direction -> panorama pixel under h (ONE implementation, ``_pixel_t``, used by both scores):
    d = S R(-beta_k) w                       (the inverse of the hypothesis; S^-1 = S),
    theta = atan2(d_y, d_x),  phi = atan2(-d_z, hypot(d_x, d_y)),
    col = floor((theta + pi) * (W / (2 pi))) mod W,   row = min(H - 1, floor((phi + pi / 2) * (H / pi))),
in float64. (row, col) is the cell that contains the direction, i.e. the pixel whose centre is nearest
in theta and in phi; on a cell boundary (measure zero) the upper cell is taken (theta = pi wraps to
column 0, phi = pi / 2 is clamped to row H - 1). In exact arithmetic this is the column-index arithmetic
    (I, k): col = (c(w) - k) mod W,      (M, k): col = (k - 1 - c(w)) mod W,      row unchanged,
with c(w) the column under (I, 0). The tests pin both forms, the sign of k and the round trip of every
pixel centre under every hypothesis family.

Invalid pixels: D == 0 (rendering holes; AcousticRooms maps carry a few). Such a pixel is never
"contained", never "agrees", is never a "violation", and is never used as a fixed point of panorama a.
A depth map that is not a (256, 512) floating array, or holds a negative or non-finite value, is refused.

Score A (pose-to-depth clearance). For every (receiver r, source s) pair with |u| > 1e-6 m,
u = src_loc - rec_loc (pairs with |u| <= 1e-6 m are excluded and counted):
    contained(h) = [D_r(pixel(u, h)) valid  and  D_r(pixel(u, h)) >= |u| - 0.10 m].
A_room(h) = contained pairs / pairs of the room;  A(h) = mean over rooms (equal room weights).

Score B (cross-receiver reprojection). For every ordered receiver pair (a, b), a != b, of a room
(pairs with |rec_a - rec_b| <= 1e-6 m are excluded and counted) and the 4,096 fixed pixels
(i, j) = (4 m + 2, 8 n + 4), m, n in 0..63 (row-major), that are valid in a:
    x = rec_a + R(beta_k) S v_a(i, j);   w = x - rec_b;   rho = |w|;   D_b = D_b(pixel(w, h));
    agree(h) = [valid and |D_b - rho| <= max(0.05 m, 0.02 rho)];
    violation(h) = [valid and D_b > rho + max(0.05 m, 0.02 rho)]   (a ray through a surface point).
It is evaluated in the algebraically identical form S R(-beta_k) w = v_a(i, j) + S R(-beta_k)(rec_a -
rec_b), rho = the norm of that vector (only float rounding differs). B_room(h) = agreeing points / valid
points summed over the room's ordered pairs; V_room(h) likewise for violations; B(h), V(h) = means over
the rooms with >= 2 receivers (a single-receiver room enters A only). Note: for ONE receiver pair, (M,
beta) and (I, 2 alpha - beta) give the same baseline image (alpha = the baseline azimuth), so B tells S
apart only through non-collinear receiver baselines; A does so through the sources' azimuth spread.

Gate (registered in plan 4.1; frozen; nothing here is tuned on real data). Accept set
``{(I, k) : k in {510, 511, 0, 1, 2}}``. The arg-max set of a curve is ``{h : curve[h] >= max - 1e-12}``
(exact ties AND float near-ties are kept); a curve passes iff its arg-max set is a SUBSET of the accept
set, so a tie with any hypothesis outside the accept set fails. PASS iff all of
  1. the full aggregate: A and B pass;
  2. every leave-one-category-out aggregate (all selected rooms except one category's, for every category
     of the FROZEN selection -- never derived from the rooms that happen to be scored; a registered
     category without a scored room, or a scored room of another category, refuses the run, AuditError,
     never a verdict): A and B pass; an undefined curve (no room, or no room with >= 2 receivers, left)
     fails;
  3. the room bootstrap: resample b (b = 0..999) is row b of
     ``numpy.random.default_rng(43).integers(0, n_rooms, size=(1000, n_rooms))`` over the selected rooms in
     inputs order; its curves are the multiplicity-weighted room means (B over its rooms with >= 2
     receivers; none -> the resample fails); it passes iff A and B pass; PASS needs >= 990 passing
     resamples (>= 99 %).
Any other outcome is FAIL (plan: STOP and report to Yixun with the curves).

Amendment A2 (PROPOSED 2026-11-13, awaiting Yixun; fixed before any real-data score). Reported next to
the registered gate, never instead of it unless chosen with ``--gate a2``. Coarse set
``COARSE = {(I, k) : circular column distance of k from 0 <= 32}`` (k in 480..511 or 0..32, S = I only;
32 columns = 22.5 deg = one C16 step); the accept set is unchanged. Same arg-max-set definition and tie
tolerance as registered. PASS iff all of
  (a) the full aggregate: the arg-max set of B is a subset of the accept set and the arg-max set of A is a
      subset of COARSE;
  (b) every leave-one-category-out aggregate: the same two conditions (an undefined curve fails);
  (c) the room bootstrap on the SAME 1,000 x n_rooms resample matrix (rng 43) as the registered gate: a
      resample passes iff B's arg-max set is a subset of the accept set and A's arg-max set is a subset of
      COARSE; PASS needs >= 990 passing resamples.
Report-only diagnostics (never gating; identical in both blocks and at the top level): A at B's
full-aggregate arg-max, max A and their difference; the circular column distance (and same-S flag)
between every A and every B full-aggregate arg-max; the bootstrap histograms of A's and B's arg-max
sets; the leave-one-category-out arg-max sets of A and B. The rule text is A2_RULE_TEXT.

Known property (synthetic evidence, recorded and NOT tuned): Score B is sharp to one column, Score A is
not -- a small wrong rotation hides about as many "just visible" sources behind occluder edges as it
reveals "just hidden" ones, so A's top is flat over a few columns (tests:
``test_random_layouts_make_score_A_flat_near_the_true_frame_while_B_stays_exact``).

Cost: Score B dominates, 1,024 hypotheses x 4,096 points per ordered receiver pair, vectorised in torch
float64 over the 512 rotations of one S at a time (512 x 4,096 elements, about 0.2 GB of temporaries);
pairs run sequentially (about 75 ms per ordered pair on 16 threads).
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Registered constants (plan 4.1)
# --------------------------------------------------------------------------- #
H, W = 256, 512
N_S = 2
N_HYP = N_S * W
S_LABELS = ("I", "M")
CONTAIN_TOL_M = 0.10
AGREE_ABS_M = 0.05
AGREE_REL = 0.02
MIN_DIST_M = 1e-6
TIE_TOL = 1e-12
ACCEPT_KS = (510, 511, 0, 1, 2)
ACCEPT = frozenset(0 * W + k for k in ACCEPT_KS)
BOOT_N = 1000
BOOT_SEED = 43
BOOT_MIN_PASS = 990
# Amendment A2 (PROPOSED 2026-11-13, awaiting Yixun; fixed before any real-data score)
COARSE_HALF_WIDTH = 32
COARSE = frozenset(0 * W + k for k in range(W) if min(k, W - k) <= COARSE_HALF_WIDTH)
GATE_CHOICES = ("registered", "a2")
_GATE_SETS = {"registered": {"A": "accept", "B": "accept"}, "a2": {"A": "coarse", "B": "accept"}}
_SETS = {"accept": ACCEPT, "coarse": COARSE}
_SET_TEXT = {"accept": "the accept set", "coarse": "the coarse set (S = I, circular |k| <= 32)"}
HASH_PREFIX = "exp43-frame-audit:"
ROOMS_PER_CATEGORY = 4
RECEIVERS_PER_ROOM = 6
POINT_ROWS = tuple(4 * m + 2 for m in range(64))
POINT_COLS = tuple(8 * n + 4 for n in range(64))
INPUTS_SCHEMA = "exp43_frame_audit_inputs/v1"
RESULT_SCHEMA = "exp43_frame_audit_result/v1"
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 3, 2
# The registered inputs (committed exp43_frame_audit_inputs.json, frozen before any real-data score):
# sha256 of the file's bytes and its canonical selection hash ("selection_sha256").
REGISTERED_INPUTS_SHA256 = "5124c23b504130378ab73385dc4b46023ec2ab2d09807092dd674e91e580af84"
REGISTERED_SELECTION_SHA256 = "035f620f6ce858ac7023f6e885c63a7a3f12ac15990eb818b94dfd61ecbb07e6"

RULE_TEXT = (
    "From the train split file (a JSON object {category: {room_id: [IR file names]}}), for every room "
    "category (its top-level keys, processed in sorted order): rank the category's rooms by the hexadecimal "
    "sha256 of the UTF-8 string 'exp43-frame-audit:' + room_id (ascending; room_id breaks ties) and take the "
    "first 4 rooms (all of them if the category has fewer). In each selected room the candidate receivers "
    "are the receiver indices r of the room's IR names 'S<digits>_R<digits>_hybrid_IR.wav', parsed as "
    "AR_md.py parses them (r = int of the digits after 'R', leading zeros allowed: the split holds both "
    "'S0010_R0010' and zero-padded 'S010_R010' styles; names of any other form are counted and unused); "
    "rank them by the sha256 of 'exp43-frame-audit:' + room_id + '/R' + str(r) (ascending; r breaks ties) "
    "and walk that order, skipping with a recorded reason every receiver without the regular file "
    "depth_map/<category>/<room_id>/<r>.npy and every receiver without any metadata JSON (position unknown), "
    "until 6 receivers are taken (fewer if the room has fewer). For every taken receiver every source s whose "
    "file metadata/<category>/<room_id>/S00{s}_R00{r}.json exists (exact canonical name, as AR_md.py builds "
    "it) is taken. Directory listings are sorted, so the result does not depend on listing order. A taken "
    "depth map must be a (256, 512) floating array, finite and >= 0 (zeros are rendering holes and are "
    "counted); every taken metadata JSON must hold 3 finite numbers in src_loc and in rec_loc, and all of a "
    "receiver's JSONs must give the identical rec_loc; any violation refuses the freeze. A selected room "
    "left without a receiver is recorded as skipped. This step reads files only to hash them and to "
    "validate their format; it computes no score."
)

REGISTERED_GATE_TEXT = (
    "Registered gate (plan_std_aug.md 4.1, frozen). Accept set {(I, k) : k in {510, 511, 0, 1, 2}}. The "
    "arg-max set of a curve is {h : curve[h] >= max - 1e-12}; a curve passes iff its arg-max set is a "
    "subset of the accept set. PASS iff all of: (1) the full aggregate: Score A and Score B pass; (2) every "
    "leave-one-category-out aggregate: A and B pass (an undefined curve fails); (3) the room bootstrap: "
    "resample b is row b of numpy.random.default_rng(43).integers(0, n_rooms, size=(1000, n_rooms)) over "
    "the selected rooms in inputs order, its curves are the multiplicity-weighted room means (B over its "
    "rooms with >= 2 receivers; none -> the resample fails), it passes iff A and B pass; PASS needs >= 990 "
    "passing resamples."
)

A2_RULE_TEXT = (
    "Amendment A2 (PROPOSED 2026-11-13, awaiting Yixun; fixed before any real-data score). Coarse set "
    "COARSE = {(I, k) : circular column distance of k from 0 <= 32}, i.e. k in 480..511 or 0..32 with S = I "
    "only (32 columns = 22.5 deg = one C16 step); the accept set is unchanged ({(I, k) : k in {510, 511, 0, "
    "1, 2}}); the arg-max-set definition and the tie tolerance (1e-12) are the registered ones. PASS iff "
    "all of: (a) the full aggregate: the arg-max set of Score B is a subset of the accept set and the "
    "arg-max set of Score A is a subset of COARSE; (b) every leave-one-category-out aggregate: the same two "
    "conditions (an undefined curve fails); (c) the room bootstrap on the same 1000 x n_rooms resample "
    "matrix as the registered gate (numpy.random.default_rng(43).integers(0, n_rooms, size=(1000, "
    "n_rooms))): a resample passes iff B's arg-max set is a subset of the accept set and A's arg-max set "
    "is a subset of COARSE; PASS needs >= 990 passing resamples. Report-only diagnostics (never gating): A "
    "at B's full-aggregate arg-max, max A and their difference; the circular column distance between A's "
    "and B's full-aggregate arg-maxes; the bootstrap arg-max histograms of A and B; the "
    "leave-one-category-out arg-max sets of A and B."
)

_IR_RE = re.compile(r"^S(\d+)_R(\d+)_hybrid_IR\.wav$")       # AR_md: int(name.split("_")[k][1:])
_JSON_RE = re.compile(r"^S00(\d+)_R00(\d+)\.json$")
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

_BETA = 2.0 * math.pi * np.arange(W) / W
_COS = np.cos(_BETA)
_SIN = np.sin(_BETA)
_COS_T = torch.from_numpy(_COS)
_SIN_T = torch.from_numpy(_SIN)
_COL_SCALE = W / (2.0 * math.pi)
_ROW_SCALE = H / math.pi


class AuditError(Exception):
    """A refusal: malformed or mismatching input, or a tool-level error."""


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def unit_grid():
    """[H, W, 3] float64 depth-frame unit vectors of the pixel centres (AR convention)."""
    j = np.arange(W, dtype=np.float64)
    i = np.arange(H, dtype=np.float64)
    theta = (j + 0.5) * 2.0 * np.pi / W - np.pi
    phi = (i + 0.5) * np.pi / H - np.pi / 2
    cp = np.cos(phi)[:, None] * np.ones((1, W))
    sp = np.sin(phi)[:, None] * np.ones((1, W))
    ct = np.ones((H, 1)) * np.cos(theta)[None, :]
    st = np.ones((H, 1)) * np.sin(theta)[None, :]
    return np.stack([cp * ct, cp * st, -sp], axis=-1)


_GRID = unit_grid()


def point_indices():
    """The 4,096 fixed pixels of Score B, row-major: rows 4m+2, columns 8n+4 (m, n in 0..63)."""
    rr, cc = np.meshgrid(np.asarray(POINT_ROWS), np.asarray(POINT_COLS), indexing="ij")
    return rr.ravel().astype(np.int64), cc.ravel().astype(np.int64)


def _pixel_t(dx, dy, dz):
    """Depth-frame direction(s) -> (row, col) int64 tensors (the definition in the module docstring)."""
    theta = torch.atan2(dy, dx)
    phi = torch.atan2(-dz, torch.hypot(dx, dy))
    col = torch.remainder(torch.floor((theta + math.pi) * _COL_SCALE).to(torch.int64), W)
    row = torch.clamp(torch.floor((phi + math.pi / 2) * _ROW_SCALE).to(torch.int64), 0, H - 1)
    return row, col


def _rot_inv_t(x, y, k, s):
    """(x, y) components of S R(-beta_k) (x, y, .) for one hypothesis (torch, float64)."""
    c, sn = float(_COS[k]), float(_SIN[k])
    xo = c * x + sn * y
    yo = -sn * x + c * y
    return xo, (-yo if s else yo)


def to_depth_frame(w, s, k):
    """d = S R(-beta_k) w for world vector(s) w [..., 3] (numpy float64)."""
    _check_hyp(s, k)
    w = np.asarray(w, dtype=np.float64)
    t = torch.from_numpy(np.ascontiguousarray(w.reshape(-1, 3)))
    xo, yo = _rot_inv_t(t[:, 0], t[:, 1], k % W, s)
    return torch.stack([xo, yo, t[:, 2]], dim=1).numpy().reshape(w.shape)


def pixel_of_direction(d):
    """Depth-frame direction(s) [..., 3] -> (rows, cols) numpy int64."""
    d = np.asarray(d, dtype=np.float64)
    t = torch.from_numpy(np.ascontiguousarray(d.reshape(-1, 3)))
    row, col = _pixel_t(t[:, 0], t[:, 1], t[:, 2])
    return row.numpy().reshape(d.shape[:-1]), col.numpy().reshape(d.shape[:-1])


def world_to_pixel(w, s, k):
    """World direction(s) [..., 3] -> (rows, cols) of the panorama pixel under hypothesis (s, k)."""
    return pixel_of_direction(to_depth_frame(w, s, k))


def _check_hyp(s, k):
    if s not in (0, 1) or isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise AuditError(f"bad hypothesis ({s!r}, {k!r})")


def hyp_label(h):
    h = int(h)
    return f"{S_LABELS[h // W]}:k={h % W}"


# --------------------------------------------------------------------------- #
# Input format
# --------------------------------------------------------------------------- #
def validate_depth(arr):
    """A depth map must be a (256, 512) floating ndarray, finite, >= 0. Returns float64 (C order)."""
    if not isinstance(arr, np.ndarray):
        raise AuditError(f"depth is not an ndarray ({type(arr).__name__})")
    if arr.dtype.kind != "f":
        raise AuditError(f"depth dtype {arr.dtype} is not floating")
    if arr.shape != (H, W):
        raise AuditError(f"depth shape {arr.shape} != {(H, W)}")
    if not np.isfinite(arr).all():
        raise AuditError("depth holds non-finite values")
    if (arr < 0).any():
        raise AuditError("depth holds negative values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _np_load(raw):
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # pickled object arrays, truncated files, npz, ...
        raise AuditError(f"depth file not loadable as a plain .npy array: {exc}") from exc


def load_depth_bytes(raw):
    return validate_depth(_np_load(raw))


def _vec3(value, what):
    if not isinstance(value, list) or len(value) != 3:
        raise AuditError(f"{what} is not a list of 3 numbers")
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise AuditError(f"{what} holds a non-finite or non-numeric entry")
    return [float(x) for x in value]


def parse_metadata_bytes(raw):
    """-> (src_loc, rec_loc) as lists of 3 floats."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuditError(f"metadata not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or "src_loc" not in doc or "rec_loc" not in doc:
        raise AuditError("metadata lacks src_loc / rec_loc")
    return _vec3(doc["src_loc"], "src_loc"), _vec3(doc["rec_loc"], "rec_loc")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _key_hash(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def canonical_sha256(obj):
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return _sha256(blob.encode("ascii"))


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _listdir(path):
    return sorted(os.listdir(path))


def _check_id(x, what):
    if not isinstance(x, str) or not _ID_RE.match(x):
        raise AuditError(f"unsafe or malformed {what} {x!r}")


def _depth_rel(cat, rid, r):
    return f"depth_map/{cat}/{rid}/{r}.npy"


def _meta_rel(cat, rid, s, r):
    return f"metadata/{cat}/{rid}/S00{s}_R00{r}.json"


def selection_constants():
    return {
        "rooms_per_category": ROOMS_PER_CATEGORY,
        "receivers_per_room": RECEIVERS_PER_ROOM,
        "room_key": HASH_PREFIX + "<room_id>",
        "receiver_key": HASH_PREFIX + "<room_id>/R<r>",
        "ir_name": "S<digits>_R<digits>_hybrid_IR.wav, s and r = int(digits) as in AR_md.py",
        "metadata_name": "S00{s}_R00{r}.json",
        "depth_name": "{r}.npy",
        "depth_shape": [H, W],
    }


# --------------------------------------------------------------------------- #
# Selection (freeze-inputs): no score is computed here
# --------------------------------------------------------------------------- #
def _parse_name(regex, name, fmt):
    m = regex.match(name)
    if not m:
        return None
    s, r = int(m.group(1)), int(m.group(2))
    return (s, r) if fmt.format(s=s, r=r) == name else None


def _select_room(ar_root, cat, rid, names, rank, skipped):
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise AuditError(f"split entry {cat}/{rid} is not a list of names")
    ir_pairs, unparsed, dup = set(), 0, 0
    for name in names:
        m = _IR_RE.match(name)
        if m is None:
            unparsed += 1
            continue
        p = (int(m.group(1)), int(m.group(2)))
        dup += p in ir_pairs
        ir_pairs.add(p)
    cands = sorted({r for _, r in ir_pairs})
    meta_dir = os.path.join(ar_root, "metadata", cat, rid)
    meta_by_rec, ignored = {}, 0
    if os.path.isdir(meta_dir):
        for name in _listdir(meta_dir):
            p = _parse_name(_JSON_RE, name, "S00{s}_R00{r}.json")
            if p is None or not os.path.isfile(os.path.join(meta_dir, name)):
                ignored += 1
                continue
            meta_by_rec.setdefault(p[1], []).append(p[0])
    ranked = sorted(cands, key=lambda r: (_key_hash(f"{HASH_PREFIX}{rid}/R{r}"), r))
    taken = []
    for r in ranked:
        if len(taken) == RECEIVERS_PER_ROOM:
            break
        drel = _depth_rel(cat, rid, r)
        dpath = os.path.join(ar_root, drel)
        if not os.path.isfile(dpath):
            skipped.append({"category": cat, "room_id": rid, "receiver": r, "reason": "no depth map " + drel})
            continue
        srcs = sorted(meta_by_rec.get(r, []))
        if not srcs:
            skipped.append({"category": cat, "room_id": rid, "receiver": r,
                            "reason": "no metadata JSON for this receiver (position unknown)"})
            continue
        raw = _read(dpath)
        try:
            stored = _np_load(raw)
            depth = validate_depth(stored)
        except AuditError as exc:
            raise AuditError(f"{drel}: {exc}") from exc
        meta, rec_loc = [], None
        for s in srcs:
            mrel = _meta_rel(cat, rid, s, r)
            mraw = _read(os.path.join(ar_root, mrel))
            try:
                _, rl = parse_metadata_bytes(mraw)
            except AuditError as exc:
                raise AuditError(f"{mrel}: {exc}") from exc
            if rec_loc is None:
                rec_loc = rl
            elif rl != rec_loc:
                raise AuditError(f"{mrel}: rec_loc {rl} differs from {rec_loc} of the same receiver")
            meta.append({"source": s, "path": mrel, "sha256": _sha256(mraw)})
        taken.append({
            "receiver": r,
            "receiver_hash": _key_hash(f"{HASH_PREFIX}{rid}/R{r}"),
            "depth": {"path": drel, "sha256": _sha256(raw), "shape": [H, W], "dtype": str(stored.dtype),
                      "n_zero_pixels": int(np.count_nonzero(depth == 0.0))},
            "metadata": meta,
            "n_metadata_not_in_split": sum((s, r) not in ir_pairs for s in srcs),
        })
    return {
        "category": cat,
        "room_id": rid,
        "room_hash": _key_hash(HASH_PREFIX + rid),
        "rank_in_category": rank,
        "n_candidate_receivers": len(cands),
        "n_split_names_unparsed": unparsed,
        "n_split_duplicate_pairs": dup,
        "n_metadata_names_ignored": ignored,
        "receivers": taken,
    }


def select_inputs(ar_root, split_path):
    """The registered selection (RULE_TEXT). Pure function of the AR tree and the split bytes."""
    raw = _read(split_path)
    try:
        split = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuditError(f"split not valid JSON: {exc}") from exc
    if not isinstance(split, dict) or not all(isinstance(v, dict) for v in split.values()):
        raise AuditError("split is not {category: {room_id: [names]}}")
    rooms, skipped, cats = [], [], []
    per_cat = {}
    for cat in sorted(split):
        _check_id(cat, "category")
        for rid in split[cat]:
            _check_id(rid, "room id")
        ranked = sorted(split[cat], key=lambda rid: (_key_hash(HASH_PREFIX + rid), rid))
        chosen = ranked[:ROOMS_PER_CATEGORY]
        cats.append({"category": cat, "n_rooms_in_split": len(ranked), "rooms_selected": chosen})
        c = {"rooms": 0, "receivers": 0, "pairs": 0, "ordered_receiver_pairs": 0}
        for rank, rid in enumerate(chosen):
            room = _select_room(ar_root, cat, rid, split[cat][rid], rank, skipped)
            if not room["receivers"]:
                skipped.append({"category": cat, "room_id": rid, "receiver": None, "reason": "no usable receiver"})
                continue
            n = len(room["receivers"])
            c["rooms"] += 1
            c["receivers"] += n
            c["pairs"] += sum(len(x["metadata"]) for x in room["receivers"])
            c["ordered_receiver_pairs"] += n * (n - 1)
            rooms.append(room)
        per_cat[cat] = c
    total = {k: sum(v[k] for v in per_cat.values()) for k in ("rooms", "receivers", "pairs", "ordered_receiver_pairs")}
    total["categories"] = sum(1 for v in per_cat.values() if v["rooms"])
    return {
        "rule": RULE_TEXT,
        "constants": selection_constants(),
        "split": {"sha256": _sha256(raw)},
        "categories": cats,
        "rooms": rooms,
        "skipped": skipped,
        "counts": {"per_category": per_cat, "total": total},
    }


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def _score_a_receiver(u, depth_t, valid_t):
    """Contained counts [2, W] (int64) of the pairs u [n, 3] (all |u| > MIN_DIST_M) of one receiver."""
    ut = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float64))
    dist = torch.sqrt((ut * ut).sum(dim=1))
    c, sn = _COS_T[:, None], _SIN_T[:, None]
    x = c * ut[:, 0] + sn * ut[:, 1]                       # [W, n]
    y = -sn * ut[:, 0] + c * ut[:, 1]
    z = ut[:, 2].expand_as(x)
    out = torch.zeros((N_S, W), dtype=torch.int64)
    for s, ys in ((0, y), (1, -y)):
        row, col = _pixel_t(x, ys, z)
        idx = row * W + col
        dd = depth_t[idx]
        ok = valid_t[idx] & (dd >= dist[None, :] - CONTAIN_TOL_M)
        out[s] = ok.sum(dim=1)
    return out.numpy()


def _score_b_pair(v, delta, depth_t, valid_t):
    """(agree, violation) counts [2, W] of a's valid fixed points v [P, 3] (depth frame) seen from b."""
    vt = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64))
    dx0, dy0, dz0 = (float(x) for x in delta)
    tx = _COS_T * dx0 + _SIN_T * dy0                       # R(-beta_k) delta, [W]
    ty = -_SIN_T * dx0 + _COS_T * dy0
    agree = torch.zeros((N_S, W), dtype=torch.int64)
    viol = torch.zeros((N_S, W), dtype=torch.int64)
    for s, tys in ((0, ty), (1, -ty)):
        dx = vt[None, :, 0] + tx[:, None]                  # [W, P]
        dy = vt[None, :, 1] + tys[:, None]
        dz = (vt[:, 2] + dz0)[None, :].expand_as(dx)
        rng = torch.hypot(torch.hypot(dx, dy), dz)
        row, col = _pixel_t(dx, dy, dz)
        idx = row * W + col
        db = depth_t[idx]
        ok = valid_t[idx]
        tol = torch.clamp(AGREE_REL * rng, min=AGREE_ABS_M)
        agree[s] = (ok & ((db - rng).abs() <= tol)).sum(dim=1)
        viol[s] = (ok & (db > rng + tol)).sum(dim=1)
    return agree.numpy(), viol.numpy()


def score_room(room):
    """Counts of one room: A_count [2, W], A_n; B_agree / B_viol [2, W], B_n (None if < 2 receivers)."""
    recs = room["receivers"]
    if not recs:
        raise AuditError(f"room {room.get('room_id')} has no receiver")
    prows, pcols = point_indices()
    prep = []
    for rec in recs:
        depth = validate_depth(rec["depth"])
        pos = np.asarray(rec["pos"], dtype=np.float64).reshape(3)
        src = np.asarray(rec["source_pos"], dtype=np.float64).reshape(-1, 3)
        if not (np.isfinite(pos).all() and np.isfinite(src).all()):
            raise AuditError("non-finite position")
        dflat = torch.from_numpy(depth.reshape(-1))
        prep.append((pos, src, depth, dflat, dflat > 0.0))
    a_count = np.zeros((N_S, W), dtype=np.int64)
    a_n = n_deg = 0
    for pos, src, _, dflat, valid in prep:
        u = src - pos
        keep = np.sqrt(np.sum(u * u, axis=1)) > MIN_DIST_M
        n_deg += int(np.count_nonzero(~keep))
        if keep.any():
            a_count += _score_a_receiver(u[keep], dflat, valid)
        a_n += int(np.count_nonzero(keep))
    if a_n == 0:
        raise AuditError(f"room {room.get('room_id')} has no usable (receiver, source) pair")
    out = {"A_count": a_count, "A_n": a_n, "B_agree": None, "B_viol": None, "B_n": None,
           "n_degenerate_pairs": n_deg, "n_excluded_receiver_pairs": 0, "n_ordered_pairs_scored": 0,
           "n_receivers": len(recs), "n_zero_pixels": [int(np.count_nonzero(p[2] == 0.0)) for p in prep]}
    if len(recs) < 2:
        return out
    b_agree = np.zeros((N_S, W), dtype=np.int64)
    b_viol = np.zeros((N_S, W), dtype=np.int64)
    b_n = n_excl = n_scored = 0
    for ia, (pa, _, da, _, _) in enumerate(prep):
        dpts = da[prows, pcols]
        ok = dpts > 0.0
        v = _GRID[prows[ok], pcols[ok]] * dpts[ok][:, None]
        for ib, (pb, _, _, dflat_b, valid_b) in enumerate(prep):
            if ia == ib:
                continue
            delta = pa - pb
            if math.sqrt(float(np.sum(delta * delta))) <= MIN_DIST_M:
                n_excl += 1
                continue
            n_scored += 1
            b_n += int(np.count_nonzero(ok))
            if not ok.any():
                continue
            ag, vi = _score_b_pair(v, delta, dflat_b, valid_b)
            b_agree += ag
            b_viol += vi
    out["n_excluded_receiver_pairs"] = n_excl
    out["n_ordered_pairs_scored"] = n_scored
    if b_n > 0:
        out.update(B_agree=b_agree, B_viol=b_viol, B_n=b_n)
    return out


def room_fractions(scores):
    """{'A': [2, W], 'B': [2, W] or None, 'V': [2, W] or None} as float64 fractions."""
    a = np.asarray(scores["A_count"], dtype=np.float64) / scores["A_n"]
    if scores.get("B_n"):
        b = np.asarray(scores["B_agree"], dtype=np.float64) / scores["B_n"]
        v = np.asarray(scores["B_viol"], dtype=np.float64) / scores["B_n"]
    else:
        b = v = None
    return {"A": a, "B": b, "V": v}


def aggregate(rooms):
    """Equal-weight room means. B / V over the rooms with a B score only (None if there is none)."""
    fr = [room_fractions(r["scores"]) for r in rooms]
    a = [f["A"] for f in fr]
    b = [f["B"] for f in fr if f["B"] is not None]
    v = [f["V"] for f in fr if f["V"] is not None]
    return {
        "A": np.mean(np.stack(a), axis=0) if a else None,
        "B": np.mean(np.stack(b), axis=0) if b else None,
        "V": np.mean(np.stack(v), axis=0) if v else None,
        "n_rooms_A": len(a),
        "n_rooms_B": len(b),
        "rooms": [r["room_id"] for r in rooms],
    }


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def argmax_set(curve):
    """Sorted flat hypothesis indices within TIE_TOL of the maximum (ties are never broken)."""
    c = np.asarray(curve, dtype=np.float64).ravel()
    if c.size != N_HYP or not np.isfinite(c).all():
        raise AuditError("a curve must hold 1,024 finite values")
    return [int(h) for h in np.flatnonzero(c >= c.max() - TIE_TOL)]


def _passes(curve, allowed=ACCEPT):
    return curve is not None and set(argmax_set(curve)) <= allowed


def _curve_report(curve, set_name="accept"):
    """Arg-max report of one curve; set_name None (the violation rate V): report only, no pass fields."""
    if curve is None:
        return None
    c = np.asarray(curve, dtype=np.float64).ravel()
    am = argmax_set(c)
    if set_name is None:
        return {"curve": c.reshape(N_S, W).tolist(), "argmax": [hyp_label(h) for h in am], "argmax_h": am,
                "max": float(c.max()), "min": float(c.min()),
                "argmin": [hyp_label(h) for h in np.flatnonzero(c <= c.min() + TIE_TOL)]}
    allowed = _SETS[set_name]
    inside = sorted(allowed)
    outside = np.ones(N_HYP, dtype=bool)
    outside[inside] = False
    bo = int(np.flatnonzero(outside)[np.argmax(c[outside])])
    bi = int(inside[int(np.argmax(c[inside]))])
    return {
        "curve": c.reshape(N_S, W).tolist(),
        "argmax": [hyp_label(h) for h in am],
        "argmax_h": am,
        "max": float(c.max()),
        "allowed_set": set_name,
        "passes": set(am) <= allowed,
        "best_inside_allowed": {"label": hyp_label(bi), "value": float(c[bi])},
        "best_outside_allowed": {"label": hyp_label(bo), "value": float(c[bo])},
        "margin_inside_minus_outside": float(c[bi] - c[bo]),
    }


def _agg_report(agg, sets):
    return {"A": _curve_report(agg["A"], sets["A"]), "B": _curve_report(agg["B"], sets["B"]),
            "V": _curve_report(agg["V"], None), "n_rooms_A": agg["n_rooms_A"], "n_rooms_B": agg["n_rooms_B"],
            "rooms": agg["rooms"]}


def _check_agg(agg, where, reasons, sets):
    for name in ("A", "B"):
        cur = agg[name]
        if cur is None:
            reasons.append(f"{where}: Score {name} undefined (no room with the needed receivers left)")
        elif not _passes(cur, _SETS[sets[name]]):
            labels = [hyp_label(h) for h in argmax_set(cur)]
            reasons.append(f"{where}: arg-max of Score {name} = {labels} is not inside {_SET_TEXT[sets[name]]}")


def bootstrap_indices(n_rooms, n=BOOT_N, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n_rooms, size=(n, n_rooms))


def _bootstrap_argmax(rooms):
    """The resample matrix and every resample's arg-max sets (shared by both gates)."""
    n_rooms = len(rooms)
    idx = bootstrap_indices(n_rooms)
    mult = np.zeros((BOOT_N, n_rooms), dtype=np.int64)
    for j in range(n_rooms):
        mult[:, j] = np.sum(idx == j, axis=1)
    fr = [room_fractions(r["scores"]) for r in rooms]
    acc_a = np.zeros((BOOT_N, N_HYP))
    acc_b = np.zeros((BOOT_N, N_HYP))
    wb = np.zeros(BOOT_N, dtype=np.int64)
    for j, f in enumerate(fr):                          # deterministic elementwise accumulation
        m = mult[:, j].astype(np.float64)[:, None]
        acc_a += m * f["A"].ravel()[None, :]
        if f["B"] is not None:
            acc_b += m * f["B"].ravel()[None, :]
            wb += mult[:, j]
    am_a, am_b, hist_a, hist_b = [], [], {}, {}
    for b in range(BOOT_N):
        a_set = argmax_set(acc_a[b] / n_rooms)
        b_set = argmax_set(acc_b[b] / wb[b]) if wb[b] > 0 else None
        am_a.append(a_set)
        am_b.append(b_set)
        key_a = "+".join(hyp_label(h) for h in a_set)
        key_b = "undefined" if b_set is None else "+".join(hyp_label(h) for h in b_set)
        hist_a[key_a] = hist_a.get(key_a, 0) + 1
        hist_b[key_b] = hist_b.get(key_b, 0) + 1
    return {
        "n_rooms": n_rooms,
        "indices_sha256": _sha256(np.ascontiguousarray(idx, dtype="<i8").tobytes()),
        "argmax_A": am_a,
        "argmax_B": am_b,
        "hist_A": dict(sorted(hist_a.items())),
        "hist_B": dict(sorted(hist_b.items())),
    }


def _bootstrap(shared, sets):
    allow_a, allow_b = _SETS[sets["A"]], _SETS[sets["B"]]
    n_pass, fails = 0, []
    for b in range(BOOT_N):
        a_set, b_set = shared["argmax_A"][b], shared["argmax_B"][b]
        if b_set is not None and set(a_set) <= allow_a and set(b_set) <= allow_b:
            n_pass += 1
        else:
            fails.append(b)
    return {
        "n_resamples": BOOT_N,
        "seed": BOOT_SEED,
        "rule": "numpy.random.default_rng(43).integers(0, n_rooms, size=(1000, n_rooms)); row b = resample b",
        "n_rooms": shared["n_rooms"],
        "indices_sha256": shared["indices_sha256"],
        "n_pass": n_pass,
        "min_pass": BOOT_MIN_PASS,
        "passes": n_pass >= BOOT_MIN_PASS,
        "failing_resamples": fails,
        "argmax_histogram_A": shared["hist_A"],
        "argmax_histogram_B": shared["hist_B"],
    }


def _labels(curve):
    return None if curve is None else [hyp_label(h) for h in argmax_set(curve)]


def _diagnostics(full, loco_aggs, shared):
    """Report-only (never gating): the same dictionary for both gates."""
    a = full["A"].ravel()
    am_a = argmax_set(a)
    am_b = None if full["B"] is None else argmax_set(full["B"])
    dist = []
    for ha in am_a:
        for hb in (am_b or []):
            dk = abs(ha % W - hb % W)
            dist.append({"A": hyp_label(ha), "B": hyp_label(hb), "same_S": ha // W == hb // W,
                         "circular_column_distance": int(min(dk, W - dk))})
    return {
        "A_argmax": [hyp_label(h) for h in am_a],
        "A_max": float(a.max()),
        "B_argmax": None if am_b is None else [hyp_label(h) for h in am_b],
        "A_at_B_argmax": None if am_b is None else [
            {"label": hyp_label(h), "A": float(a[h]), "A_max_minus_A_at_B_argmax": float(a.max() - a[h])}
            for h in am_b],
        "argmax_column_distance": dist,
        "bootstrap_argmax_histogram_A": shared["hist_A"],
        "bootstrap_argmax_histogram_B": shared["hist_B"],
        "loco_argmax": {c: ({"A": None, "B": None} if agg is None else {"A": _labels(agg["A"]), "B": _labels(agg["B"])})
                        for c, agg in loco_aggs.items()},
    }


def _registered_categories(categories):
    if not isinstance(categories, (list, tuple)) or not categories:
        raise AuditError("the registered categories (those of the frozen selection) must be a non-empty list")
    cats = list(categories)
    if not all(isinstance(c, str) for c in cats) or len(set(cats)) != len(cats):
        raise AuditError(f"malformed or duplicated registered categories {cats!r}")
    return cats


def _aggregates(rooms, categories):
    """The full aggregate and one leave-one-category-out aggregate for EVERY registered category (the
    frozen selection's, never derived from the scored rooms). A registered category without a scored
    room, or a scored room of an unregistered category, is refused (AuditError), never judged."""
    cats = _registered_categories(categories)
    present = {r["category"] for r in rooms}
    extra = sorted(present - set(cats))
    if extra:
        raise AuditError(f"scored room(s) of unregistered categories {extra} (registered: {cats})")
    missing = [c for c in cats if c not in present]
    if missing:
        raise AuditError(f"registered categories {missing} have no scored room: their leave-one-category-out "
                         "check cannot be made")
    full = aggregate(rooms)
    loco = {}
    for c in cats:
        sub = [r for r in rooms if r["category"] != c]
        loco[c] = aggregate(sub) if sub else None
    return full, loco


def _evaluate(gate, full, loco_aggs, shared):
    sets = _GATE_SETS[gate]
    reasons = []
    _check_agg(full, "full aggregate", reasons, sets)
    loco = {}
    for c, agg in loco_aggs.items():
        if agg is None:
            loco[c] = None
            reasons.append(f"leave-out {c}: no room left, Score A and Score B undefined")
            continue
        loco[c] = _agg_report(agg, sets)
        _check_agg(agg, f"leave-out {c}", reasons, sets)
    boot = _bootstrap(shared, sets)
    if not boot["passes"]:
        reasons.append(f"bootstrap: {boot['n_pass']} of {BOOT_N} resamples pass (need >= {BOOT_MIN_PASS})")
    out = {
        "gate": gate,
        "rule": REGISTERED_GATE_TEXT if gate == "registered" else A2_RULE_TEXT,
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "accept_set": [hyp_label(h) for h in sorted(ACCEPT)],
        "allowed_sets": dict(sets),
        "full": _agg_report(full, sets),
        "loco": loco,
        "bootstrap": boot,
        "diagnostics": _diagnostics(full, loco_aggs, shared),
    }
    if gate == "a2":
        out["coarse_half_width"] = COARSE_HALF_WIDTH
        out["coarse_set"] = [hyp_label(h) for h in sorted(COARSE)]
    return out


def evaluate_gates(rooms, *, categories):
    """Both gates over scored rooms [{'room_id', 'category', 'scores'}] (inputs order), one shared
    resample matrix: {'registered': ..., 'a2': ...}. ``categories`` (REQUIRED, no default) are the
    registered categories -- ``frozen_categories`` of the selection -- in inputs order; each must hold a
    scored room (see ``_aggregates``)."""
    if not rooms:
        raise AuditError("no room to evaluate")
    full, loco_aggs = _aggregates(rooms, categories)
    shared = _bootstrap_argmax(rooms)
    return {g: _evaluate(g, full, loco_aggs, shared) for g in GATE_CHOICES}


def evaluate_gate(rooms, *, categories):
    """The registered gate (plan 4.1)."""
    return evaluate_gates(rooms, categories=categories)["registered"]


def evaluate_gate_a2(rooms, *, categories):
    """Amendment A2 (PROPOSED 2026-11-13, awaiting Yixun)."""
    return evaluate_gates(rooms, categories=categories)["a2"]


def _reading(gate):
    """Signed k and beta (degrees) of the full-aggregate arg-maxes (for the plan 2.2 table)."""
    out = {}
    for name in ("A", "B"):
        rep = gate["full"][name]
        if rep is None:
            out[name] = None
            continue
        out[name] = [{"label": hyp_label(h), "S": S_LABELS[h // W],
                      "k_signed": (h % W) - W if (h % W) > W // 2 else h % W,
                      "beta_deg": round(360.0 * (((h % W) - W if (h % W) > W // 2 else h % W)) / W, 6)}
                     for h in rep["argmax_h"]]
    return out


# --------------------------------------------------------------------------- #
# Provenance, I/O
# --------------------------------------------------------------------------- #
def _tool_path():
    return os.path.abspath(__file__)


def _git(*args):
    try:
        r = subprocess.run(["git", "-C", os.path.dirname(_tool_path()), *args], capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _provenance(extra):
    tool = _tool_path()
    rel = os.path.basename(tool)
    tracked = _git("ls-files", "--error-unmatch", rel) is not None
    dirty = _git("status", "--porcelain", "--", rel)
    state = "untracked" if not tracked else ("unknown" if dirty is None else ("modified" if dirty else "clean"))
    prov = {
        "tool": "exp43_frame_audit.py",
        "tool_sha256": _sha256(_read(tool)),
        "tool_git_state": state,
        "git_head": _git("rev-parse", "HEAD"),
        "host": socket.gethostname(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }
    prov.update(extra)
    return prov


def _write_json_new(path, doc):
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise AuditError(f"refusing to overwrite {path}")
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".exp43_audit_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.chmod(tmp, 0o644)
        try:
            os.link(tmp, path)                       # atomic and never overwrites
        except FileExistsError as exc:
            raise AuditError(f"refusing to overwrite {path}") from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def freeze_inputs(ar_root, split_path, out_path):
    if os.path.exists(out_path):
        raise AuditError(f"refusing to overwrite {out_path}")
    sel = select_inputs(ar_root, split_path)
    doc = {
        "schema": INPUTS_SCHEMA,
        "selection": sel,
        "selection_sha256": canonical_sha256(sel),
        "provenance": _provenance({"ar_root": ar_root, "ar_root_realpath": os.path.realpath(ar_root),
                                   "split_path": split_path}),
    }
    _write_json_new(out_path, doc)
    return doc


def frozen_categories(sel):
    """The registered categories of a frozen selection (inputs order). Refuses a selection without a
    category, a category with zero selected rooms or without any room left in the selection (all its
    selected rooms skipped), a duplicated category or room, and a room its category did not select."""
    if not isinstance(sel, dict):
        raise AuditError("the frozen selection is not an object")
    cats, rooms = sel.get("categories"), sel.get("rooms")
    if not isinstance(cats, list) or not cats or not isinstance(rooms, list):
        raise AuditError("the frozen selection lists no category")
    names, chosen = [], {}
    for c in cats:
        if not isinstance(c, dict):
            raise AuditError("malformed category entry in the frozen selection")
        name, sel_rooms = c.get("category"), c.get("rooms_selected")
        _check_id(name, "category")
        if name in chosen:
            raise AuditError(f"category {name} is listed twice in the frozen selection")
        if not isinstance(sel_rooms, list) or not sel_rooms:
            raise AuditError(f"registered category {name} has zero selected rooms")
        for rid in sel_rooms:
            _check_id(rid, "room id")
        names.append(name)
        chosen[name] = set(sel_rooms)
    count, seen = dict.fromkeys(names, 0), set()
    for room in rooms:
        if not isinstance(room, dict):
            raise AuditError("malformed room entry in the frozen selection")
        cat, rid = room.get("category"), room.get("room_id")
        _check_id(cat, "category")
        _check_id(rid, "room id")
        if cat not in chosen or rid not in chosen[cat]:
            raise AuditError(f"room {cat}/{rid} is not among the rooms selected for a registered category")
        if (cat, rid) in seen:
            raise AuditError(f"room {cat}/{rid} is listed twice in the frozen selection")
        seen.add((cat, rid))
        count[cat] += 1
    empty = [c for c in names if count[c] == 0]
    if empty:
        raise AuditError(f"registered categories {empty} have no room in the frozen selection "
                         "(every selected room was skipped)")
    return names


def _binding_text(expected, registered):
    return "the registered exp43_frame_audit_inputs.json" if expected == registered else "a non-registered binding"


def inputs_binding(inputs_sha256, selection_sha256):
    """The result's "inputs_binding" block; "registered" is true only for the registered pair."""
    registered = inputs_sha256 == REGISTERED_INPUTS_SHA256 and selection_sha256 == REGISTERED_SELECTION_SHA256
    return {
        "registered": bool(registered),
        "inputs_sha256": inputs_sha256,
        "selection_sha256": selection_sha256,
        "registered_inputs_sha256": REGISTERED_INPUTS_SHA256,
        "registered_selection_sha256": REGISTERED_SELECTION_SHA256,
        "statement": ("the inputs are the registered exp43_frame_audit_inputs.json (file and selection "
                      "sha256 as pinned in the tool)") if registered else (
                      "NOT the official exp_43 frame-audit record: the inputs are bound to a non-registered "
                      "pair of hashes through the Python API (synthetic tests only)"),
    }


def load_inputs(inputs_path, ar_root, *, expected_inputs_sha256=REGISTERED_INPUTS_SHA256,
                expected_selection_sha256=REGISTERED_SELECTION_SHA256):
    """Verify the inputs file against its binding (default: the registered pair) and re-hash every
    listed file; -> (doc, inputs sha256, rooms)."""
    raw = _read(inputs_path)
    inputs_sha = _sha256(raw)
    if inputs_sha != expected_inputs_sha256:
        raise AuditError(f"inputs file sha256 {inputs_sha} is not the bound {expected_inputs_sha256} "
                         f"({_binding_text(expected_inputs_sha256, REGISTERED_INPUTS_SHA256)})")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise AuditError(f"inputs not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != INPUTS_SCHEMA:
        raise AuditError("inputs schema mismatch")
    sel = doc.get("selection")
    if not isinstance(sel, dict):
        raise AuditError("inputs hold no selection")
    sel_sha = canonical_sha256(sel)
    if sel_sha != doc.get("selection_sha256"):
        raise AuditError("inputs selection_sha256 does not match the selection (edited inputs file)")
    if sel_sha != expected_selection_sha256:
        raise AuditError(f"inputs selection sha256 {sel_sha} is not the bound {expected_selection_sha256} "
                         f"({_binding_text(expected_selection_sha256, REGISTERED_SELECTION_SHA256)})")
    if sel.get("constants") != selection_constants() or sel.get("rule") != RULE_TEXT:
        raise AuditError("inputs were frozen under a different rule or constants")
    frozen_categories(sel)
    problems, blobs = [], {}
    for room in sel["rooms"]:
        cat, rid = room["category"], room["room_id"]
        _check_id(cat, "category")
        _check_id(rid, "room id")
        for rec in room["receivers"]:
            r = rec["receiver"]
            if isinstance(r, bool) or not isinstance(r, int) or r < 0:
                raise AuditError(f"bad receiver index {r!r}")
            entries = [(rec["depth"]["path"], _depth_rel(cat, rid, r), rec["depth"]["sha256"])]
            for m in rec["metadata"]:
                s = m["source"]
                if isinstance(s, bool) or not isinstance(s, int) or s < 0:
                    raise AuditError(f"bad source index {s!r}")
                entries.append((m["path"], _meta_rel(cat, rid, s, r), m["sha256"]))
            for path, want, sha in entries:
                if path != want:
                    raise AuditError(f"inputs path {path!r} is not the canonical {want!r}")
                full = os.path.join(ar_root, path)
                if not os.path.isfile(full):
                    problems.append(f"missing {path}")
                    continue
                b = _read(full)
                if _sha256(b) != sha:
                    problems.append(f"sha256 mismatch {path}")
                    continue
                blobs[path] = b
    if problems:
        raise AuditError(f"{len(problems)} input file(s) missing or changed: " + "; ".join(problems[:20]))
    rooms = []
    for room in sel["rooms"]:
        recs = []
        for rec in room["receivers"]:
            try:
                depth = load_depth_bytes(blobs[rec["depth"]["path"]])
            except AuditError as exc:
                raise AuditError(f"{rec['depth']['path']}: {exc}") from exc
            ids, srcs, rec_loc = [], [], None
            for m in rec["metadata"]:
                sl, rl = parse_metadata_bytes(blobs[m["path"]])
                if rec_loc is None:
                    rec_loc = rl
                elif rl != rec_loc:
                    raise AuditError(f"{m['path']}: inconsistent rec_loc")
                ids.append(m["source"])
                srcs.append(sl)
            if rec_loc is None:
                raise AuditError(f"receiver {rec['receiver']} of {room['room_id']} has no metadata")
            recs.append({"receiver": rec["receiver"], "pos": np.asarray(rec_loc), "depth": depth,
                         "source_ids": ids, "source_pos": np.asarray(srcs, dtype=np.float64)})
        rooms.append({"room_id": room["room_id"], "category": room["category"], "receivers": recs})
    if not rooms:
        raise AuditError("inputs select no room")
    return doc, inputs_sha, rooms


def _room_record(room, sc):
    fr = room_fractions(sc)
    rec = {
        "room_id": room["room_id"],
        "category": room["category"],
        "receivers": [r["receiver"] for r in room["receivers"]],
        "n_receivers": sc["n_receivers"],
        "n_degenerate_pairs": sc["n_degenerate_pairs"],
        "n_zero_pixels": sc["n_zero_pixels"],
        "A": {"count": sc["A_count"].tolist(), "n": sc["A_n"], "frac": fr["A"].tolist(),
              "argmax": [hyp_label(h) for h in argmax_set(fr["A"])]},
        "B": None,
    }
    if fr["B"] is not None:
        rec["B"] = {"agree_count": sc["B_agree"].tolist(), "viol_count": sc["B_viol"].tolist(), "n": sc["B_n"],
                    "agree_frac": fr["B"].tolist(), "viol_frac": fr["V"].tolist(),
                    "n_ordered_pairs_scored": sc["n_ordered_pairs_scored"],
                    "n_excluded_receiver_pairs": sc["n_excluded_receiver_pairs"],
                    "argmax": [hyp_label(h) for h in argmax_set(fr["B"])]}
    return rec


def _strip_curves(rep):
    if rep is None:
        return None
    out = dict(rep)
    for k in ("A", "B", "V"):
        if out.get(k) is not None:
            out[k] = {kk: vv for kk, vv in out[k].items() if kk != "curve"}
    return out


def _curves(rep):
    if rep is None:
        return None
    return {k: (None if rep[k] is None else rep[k]["curve"]) for k in ("A", "B", "V")} | {
        "n_rooms_A": rep["n_rooms_A"], "n_rooms_B": rep["n_rooms_B"], "rooms": rep["rooms"]}


def _block(g):
    """A gate block without the curves (those live once, under "aggregates")."""
    return {k: v for k, v in g.items() if k not in ("full", "loco")} | {
        "full": _strip_curves(g["full"]), "loco": {c: _strip_curves(v) for c, v in g["loco"].items()}}


def run_audit(inputs_path, ar_root, out_path, *, gate, split_path, threads=16,
              expected_inputs_sha256=REGISTERED_INPUTS_SHA256,
              expected_selection_sha256=REGISTERED_SELECTION_SHA256):
    """The ``run`` subcommand. The inputs must match the bound hashes (default: the registered pair; the
    CLI never overrides it), the selection is always re-derived from ``split_path`` and the AR tree, and
    the registered categories are the frozen selection's. Any refusal raises AuditError before anything
    is written."""
    t0 = time.time()
    if gate not in GATE_CHOICES:
        raise AuditError(f"--gate must be one of {GATE_CHOICES}, got {gate!r}")
    if split_path is None:
        raise AuditError("--split is required: the selection is always re-derived from the split and the "
                         "AR tree")
    if os.path.exists(out_path):
        raise AuditError(f"refusing to overwrite {out_path}")
    torch.set_num_threads(int(threads))
    doc, inputs_sha, rooms = load_inputs(inputs_path, ar_root, expected_inputs_sha256=expected_inputs_sha256,
                                         expected_selection_sha256=expected_selection_sha256)
    categories = frozen_categories(doc["selection"])
    if select_inputs(ar_root, split_path) != doc["selection"]:
        raise AuditError("the split / AR tree no longer reproduce the frozen selection")
    scored = []
    for room in rooms:
        scored.append({"room_id": room["room_id"], "category": room["category"], "scores": score_room(room)})
    gates = evaluate_gates(scored, categories=categories)
    chosen, reg = gates[gate], gates["registered"]
    result = {
        "schema": RESULT_SCHEMA,
        "gate_of_record": gate,
        "verdict": chosen["verdict"],
        "reasons": chosen["reasons"],
        "inputs_path": inputs_path,
        "inputs_sha256": inputs_sha,
        "selection_sha256": doc["selection_sha256"],
        "inputs_binding": inputs_binding(inputs_sha, doc["selection_sha256"]),
        "split_reverified": True,
        "tool_sha256": _sha256(_read(_tool_path())),
        "provenance": _provenance({"ar_root": ar_root, "ar_root_realpath": os.path.realpath(ar_root),
                                   "split_path": split_path, "threads": int(threads),
                                   "inputs_provenance": doc.get("provenance")}),
        "constants": {
            "H": H, "W": W, "hypotheses": "h = 512 [S = M] + k, v_world = R(2 pi k / 512) S v_depth",
            "contain_tol_m": CONTAIN_TOL_M, "agree_abs_m": AGREE_ABS_M, "agree_rel": AGREE_REL,
            "min_dist_m": MIN_DIST_M, "tie_tol": TIE_TOL, "accept_set": [hyp_label(h) for h in sorted(ACCEPT)],
            "coarse_half_width": COARSE_HALF_WIDTH,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "min_pass": BOOT_MIN_PASS},
            "points": "rows 4m+2, cols 8n+4, m, n in 0..63, row-major (4,096)",
        },
        "registered_rule": REGISTERED_GATE_TEXT,
        "a2_rule": A2_RULE_TEXT,
        "reading": _reading(reg),
        "diagnostics": reg["diagnostics"],
        "gate_registered": _block(reg),
        "gate_a2_proposed": _block(gates["a2"]),
        "aggregates": {"all": _curves(reg["full"]), "loco": {c: _curves(v) for c, v in reg["loco"].items()}},
        "rooms": [_room_record(room, s["scores"]) for room, s in zip(rooms, scored)],
        "summary": {
            "n_rooms": len(rooms),
            "n_categories": len({r["category"] for r in rooms}),
            "n_receivers": sum(len(r["receivers"]) for r in rooms),
            "n_pairs_scored": sum(s["scores"]["A_n"] for s in scored),
            "n_degenerate_pairs": sum(s["scores"]["n_degenerate_pairs"] for s in scored),
            "n_ordered_receiver_pairs_scored": sum(s["scores"]["n_ordered_pairs_scored"] for s in scored),
        },
    }
    result["runtime_s"] = round(time.time() - t0, 3)
    _write_json_new(out_path, result)
    return EXIT_PASS if chosen["verdict"] == "PASS" else EXIT_FAIL


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parser():
    p = argparse.ArgumentParser(prog="exp43_frame_audit.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-inputs", help="freeze and hash the registered input list (no score)")
    f.add_argument("--ar-root", required=True)
    f.add_argument("--split", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("run", help="bound to the registered inputs: re-hash them, re-derive the selection from "
                                   "--split, score all 1,024 hypotheses, evaluate both gates")
    r.add_argument("--inputs", required=True)
    r.add_argument("--ar-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--split", required=True,
                   help="the split the inputs were frozen from (REQUIRED): the selection is re-derived from it "
                        "and the AR tree and must be identical")
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--gate", required=True, choices=GATE_CHOICES,
                   help="gate of record (REQUIRED): drives the top-level verdict and the exit code; both are reported")
    return p


def main(argv=None):
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_ERROR if exc.code else 0
    try:
        if args.cmd == "freeze-inputs":
            freeze_inputs(args.ar_root, args.split, args.out)
            return EXIT_PASS
        if args.threads < 1:
            raise AuditError("--threads must be >= 1")
        return run_audit(args.inputs, args.ar_root, args.out, gate=args.gate, split_path=args.split,
                         threads=args.threads)
    except AuditError as exc:
        print(f"exp43_frame_audit: REFUSED: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # any other failure is a tool error, never a FAIL verdict
        import traceback

        traceback.print_exc()
        print(f"exp43_frame_audit: TOOL ERROR: {exc!r}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
