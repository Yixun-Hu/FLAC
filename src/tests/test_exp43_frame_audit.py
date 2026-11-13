"""exp_43 std_aug -- the frame-alignment audit (exp43_frame_audit.py), plan section 4.1.

CPU only, synthetic data only. Analytic non-symmetric rooms (shoebox + interior boxes) are
ray-cast HERE, independently of the tool, in the AcousticRooms panorama convention
(``theta_j = (j + 0.5) 2 pi / W - pi``, ``phi_i = (i + 0.5) pi / H - pi / 2``, unit vectors
``(cos phi cos theta, cos phi sin theta, -sin phi)``) with an injected frame relation
``v_world = R(2 pi k / W) S v_depth``. The tool must recover the injection exactly as the
arg-max of both scores; only the identity may PASS the registered gate. No real
AcousticRooms file is read by this module.

Source layouts. Score B is exact for any layout (``test_random_layouts_...``). Score A is not: a
small wrong rotation hides as many "just visible" sources behind an occluder edge as it reveals
"just hidden" ones, so with uniformly random sources A's top is flat over a few columns and its
arg-max wanders (documented, not hidden, by ``test_random_layouts_make_score_A_flat_...``). For the
exact-recovery proof the 20 sources per room were therefore chosen once, at design time, from 1,500
random free-space candidates (>= 0.2 m from every surface, >= 0.5 m from every receiver): fully
visible sources whose small-shift losses cover every wrong hypothesis with a margin of >= 3 pairs,
plus up to 4 deeply occluded sources (no gain within +-8 columns). The literal coordinates are below;
the tests re-verify every property they rely on.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_EXP = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude"
TOOL_PATH = _EXP / "exp43_frame_audit.py"
INPUTS_PATH = _EXP / "exp43_frame_audit_inputs.json"
# the registered binding of the committed inputs file (Codex code review r1, finding 1)
REG_INPUTS = "5124c23b504130378ab73385dc4b46023ec2ab2d09807092dd674e91e580af84"
REG_SELECTION = "035f620f6ce858ac7023f6e885c63a7a3f12ac15990eb818b94dfd61ecbb07e6"
REG_CATEGORIES = ["Apartments", "Auditorium", "Bathrooms", "Bedrooms", "Cafe", "ListeningRoom",
                  "LivingRoomsWithHallway", "MeetingRoom", "Office", "Restaurants"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fa = _load("exp43_frame_audit", TOOL_PATH)
H, W = 256, 512
MIR = np.diag([1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- #
# Independent geometry (test side): AR grid, injected frame, analytic ray caster
# --------------------------------------------------------------------------- #
def ar_unit_grid():
    i = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    j = np.ones((H, 1)) * np.arange(W, dtype=np.float64)[None, :]
    theta = (j + 0.5) * 2.0 * np.pi / W - np.pi
    phi = (i + 0.5) * np.pi / H - np.pi / 2
    return np.stack([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)], axis=-1)


def rot_z(beta):
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hyp_matrix(s, k):
    """depth frame -> world frame under hypothesis (S, k)."""
    return rot_z(2.0 * np.pi * k / W) @ (MIR if s else np.eye(3))


def raycast(origin, dirs, size, boxes):
    """Distance along unit ``dirs`` from ``origin`` (inside the shoebox [0, size]) to the first
    surface: the shoebox's inner walls or the outside of any interior axis-aligned box."""
    origin = np.asarray(origin, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (0.0 - origin) * inv
        t2 = (np.asarray(size, dtype=np.float64) - origin) * inv
        best = np.min(np.maximum(t1, t2), axis=1)
        for lo, hi in boxes:
            t1 = (np.asarray(lo, dtype=np.float64) - origin) * inv
            t2 = (np.asarray(hi, dtype=np.float64) - origin) * inv
            tnear = np.max(np.minimum(t1, t2), axis=1)
            tfar = np.min(np.maximum(t1, t2), axis=1)
            hit = (tnear <= tfar) & (tnear > 0.0)
            best = np.where(hit & (tnear < best), tnear, best)
    assert np.isfinite(best).all() and (best > 0).all()
    return best


_GRID = ar_unit_grid().reshape(-1, 3)


def render(spec, rec, s, k):
    dirs = _GRID @ hyp_matrix(s, k).T
    return raycast(rec, dirs, spec["size"], spec["boxes"]).reshape(H, W)


def segment_blocked(a, b, boxes):
    """True if the open segment a -> b passes through an interior box (test-side visibility)."""
    a = np.asarray(a, float)
    d = np.asarray(b, float) - a
    n = np.linalg.norm(d)
    d = d / n
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (np.asarray(lo) - a) / d
            t2 = (np.asarray(hi) - a) / d
        tnear = np.max(np.minimum(t1, t2))
        tfar = np.min(np.maximum(t1, t2))
        if tnear <= tfar and 0.0 < tnear < n:
            return True
    return False


# --------------------------------------------------------------------------- #
# The synthetic rooms (non-symmetric shoeboxes, interior boxes, off-centre receivers)
# --------------------------------------------------------------------------- #
ROOMS = [
    dict(
        room_id="Alpha_idx_0", category="Alpha", size=(7.3, 5.1, 3.0),
        boxes=[((1.2, 3.1, 0.0), (2.0, 4.4, 1.1)), ((4.6, 0.9, 0.0), (4.9, 1.3, 2.4)),
               ((5.8, 3.6, 0.0), (6.6, 4.8, 0.8))],
        receivers=[(2.5, 1.7, 1.5), (5.9, 2.6, 1.2), (3.4, 4.2, 1.8), (0.9, 0.8, 1.3)],
    ),
    dict(
        room_id="Alpha_idx_7", category="Alpha", size=(9.2, 4.3, 2.7),
        boxes=[((2.9, 0.0, 0.0), (3.5, 1.6, 2.7)), ((6.1, 2.4, 0.0), (7.4, 3.1, 0.9))],
        receivers=[(1.1, 1.0, 1.4), (4.6, 2.9, 1.6), (8.3, 1.2, 1.1), (5.5, 0.6, 1.9)],
    ),
    dict(
        room_id="Beta_idx_3", category="Beta", size=(6.1, 6.7, 3.4),
        boxes=[((0.4, 4.9, 0.0), (1.9, 6.2, 1.5)), ((3.3, 2.2, 0.0), (3.8, 2.6, 3.4)),
               ((4.4, 0.3, 0.6), (5.6, 1.1, 1.0))],
        receivers=[(1.3, 1.4, 1.2), (4.9, 3.6, 1.7), (2.4, 5.6, 1.5), (5.1, 6.0, 1.0)],
    ),
    dict(
        room_id="Beta_idx_11", category="Beta", size=(11.5, 7.9, 4.2),
        boxes=[((3.0, 2.0, 0.0), (4.1, 2.7, 1.2)), ((7.6, 5.1, 0.0), (8.0, 5.5, 4.2)),
               ((9.4, 0.5, 0.0), (11.0, 1.9, 2.0)), ((1.0, 6.0, 0.0), (2.6, 7.4, 0.7))],
        receivers=[(1.6, 1.3, 1.5), (6.2, 3.9, 1.3), (10.4, 6.6, 1.8), (5.0, 6.9, 1.1)],
    ),
    dict(
        room_id="Gamma_idx_2", category="Gamma", size=(5.4, 3.6, 2.6),
        boxes=[((0.0, 2.5, 0.0), (1.1, 3.6, 2.0)), ((3.6, 1.2, 0.0), (3.9, 1.5, 2.6))],
        receivers=[(0.7, 0.6, 1.2), (2.4, 2.9, 1.5), (4.8, 0.9, 1.0), (2.9, 0.5, 1.8)],
    ),
    dict(
        room_id="Gamma_idx_5", category="Gamma", size=(8.4, 5.9, 3.1),
        boxes=[((2.1, 4.0, 0.0), (2.5, 4.4, 3.1)), ((5.0, 0.0, 0.0), (6.9, 1.2, 1.3)),
               ((6.6, 3.9, 0.0), (7.9, 5.4, 0.9))],
        receivers=[(1.0, 2.1, 1.6), (4.2, 4.9, 1.3), (7.7, 2.8, 1.5), (3.6, 1.7, 1.1)],
    ),
]


# Source layouts (design-time selection, see the module docstring): per room 20 free-space points,
# fully visible ones whose small-shift losses cover every wrong hypothesis, plus deeply occluded ones.
SOURCES = {
    'Alpha_idx_0': [
        (3.715, 2.633, 2.027), (5.526, 3.49, 2.745), (5.238, 1.666, 0.657), (5.908, 2.412, 2.256),
        (1.828, 2.706, 1.27), (1.559, 1.7, 1.766), (4.962, 3.272, 2.414), (3.628, 3.093, 0.453),
        (5.105, 3.008, 2.056), (2.713, 4.136, 0.453), (0.294, 3.881, 0.714), (0.835, 3.934, 0.778),
        (0.261, 4.722, 0.494), (5.289, 0.616, 1.564), (7.081, 4.878, 1.422), (0.882, 3.959, 0.376),
        (5.389, 0.573, 1.969), (7.034, 1.671, 2.32), (6.335, 1.549, 1.112), (6.257, 0.219, 0.212),
    ],
    'Alpha_idx_7': [
        (3.632, 2.657, 1.222), (2.03, 3.447, 1.964), (7.38, 3.396, 2.022), (4.747, 2.432, 0.904),
        (0.423, 3.754, 0.45), (0.307, 3.425, 1.166), (3.796, 3.704, 1.501), (0.691, 0.781, 1.843),
        (1.623, 0.848, 1.205), (0.747, 0.289, 1.709), (0.278, 0.362, 1.015), (8.919, 4.046, 1.622),
        (3.315, 1.804, 1.116), (8.878, 3.642, 2.131), (0.391, 4.004, 0.499), (4.041, 3.487, 0.442),
        (3.607, 1.841, 2.002), (2.892, 1.92, 1.983), (1.485, 2.633, 0.442), (2.301, 2.22, 1.114),
    ],
    'Beta_idx_3': [
        (4.528, 5.677, 3.198), (0.214, 5.871, 2.362), (1.452, 1.763, 3.131), (3.85, 3.4, 0.778),
        (1.619, 2.306, 2.142), (4.887, 4.347, 1.939), (5.061, 5.35, 1.619), (1.351, 3.752, 3.153),
        (0.352, 4.749, 3.007), (5.294, 0.205, 3.176), (3.099, 2.474, 0.886), (0.323, 0.956, 0.866),
        (0.667, 6.445, 0.825), (3.478, 1.091, 3.006), (3.506, 1.188, 0.403), (0.735, 6.402, 0.981),
        (0.995, 6.463, 1.089), (0.311, 6.48, 0.596), (4.881, 2.493, 2.755), (4.277, 1.752, 2.31),
    ],
    'Beta_idx_11': [
        (8.266, 6.672, 1.233), (4.025, 4.143, 0.941), (4.573, 4.603, 0.261), (0.226, 3.188, 2.368),
        (2.493, 4.996, 1.341), (10.698, 0.602, 3.662), (5.847, 5.857, 2.566), (4.061, 7.174, 2.629),
        (2.188, 1.423, 0.934), (0.5, 0.226, 3.074), (9.129, 5.792, 0.935), (0.319, 0.232, 0.236),
        (10.953, 2.109, 1.652), (5.528, 4.779, 0.761), (8.17, 4.686, 1.108), (8.591, 6.641, 3.773),
        (1.923, 0.696, 0.559), (11.242, 2.334, 0.259), (10.832, 2.796, 2.819), (10.614, 0.223, 1.007),
    ],
    'Gamma_idx_2': [
        (3.996, 2.454, 1.35), (3.891, 2.369, 1.973), (1.465, 0.33, 0.957), (0.247, 1.223, 1.552),
        (3.472, 0.845, 1.487), (0.989, 1.18, 2.207), (4.128, 2.811, 1.493), (3.056, 3.374, 1.427),
        (3.193, 2.286, 2.115), (0.378, 3.37, 2.346), (5.189, 0.4, 2.343), (4.306, 1.147, 2.372),
        (1.378, 3.221, 0.315), (0.412, 3.21, 2.223), (4.72, 3.338, 2.098), (3.853, 0.792, 1.482),
        (3.512, 1.771, 0.464), (3.136, 1.307, 1.51), (4.767, 3.191, 1.21), (4.306, 1.221, 1.057),
    ],
    'Gamma_idx_5': [
        (3.735, 0.462, 1.348), (0.481, 4.136, 0.425), (7.491, 1.791, 1.4), (5.216, 3.416, 1.166),
        (6.436, 0.253, 2.732), (7.33, 4.869, 2.625), (0.898, 1.27, 2.596), (2.911, 2.475, 0.774),
        (0.408, 5.507, 0.206), (0.974, 5.7, 2.191), (7.921, 0.236, 0.502), (8.162, 0.541, 1.191),
        (7.657, 5.634, 1.149), (7.781, 0.245, 1.082), (8.139, 4.905, 0.268), (2.202, 5.118, 2.317),
        (2.055, 5.531, 2.398), (2.578, 5.483, 0.347), (0.553, 4.067, 1.588), (0.719, 3.012, 1.248),
    ],
}


for _spec in ROOMS:
    _spec["sources"] = SOURCES[_spec["room_id"]]


def random_sources(spec, n, seed):
    """Uniform free-space sources (>= 0.2 m from walls and boxes) -- for the A-flatness record."""
    rng = np.random.default_rng(seed)
    out = []
    size = np.asarray(spec["size"])
    while len(out) < n:
        p = 0.2 + rng.uniform(0.0, 1.0, size=3) * (size - 0.4)
        if any(np.all(p > np.asarray(lo) - 0.2) and np.all(p < np.asarray(hi) + 0.2) for lo, hi in spec["boxes"]):
            continue
        out.append(tuple(np.round(p, 3)))
    return out


def room_struct(spec, s, k, receivers=None):
    """The tool's in-memory room structure, rendered under the injected (s, k)."""
    recs = spec["receivers"] if receivers is None else receivers
    src = np.asarray(spec["sources"], dtype=np.float64)
    return {
        "room_id": spec["room_id"],
        "category": spec["category"],
        "receivers": [
            {"receiver": r, "pos": np.asarray(p, dtype=np.float64), "depth": render(spec, p, s, k),
             "source_ids": list(range(len(src))), "source_pos": src.copy()}
            for r, p in enumerate(recs)
        ],
    }


_CACHE = {}


def scored(s, k):
    """Per-room scores of all synthetic rooms under the injected (s, k) (cached per module)."""
    if (s, k) not in _CACHE:
        rooms = []
        for spec in ROOMS:
            rs = room_struct(spec, s, k)
            rooms.append({"room_id": rs["room_id"], "category": rs["category"], "scores": fa.score_room(rs)})
        _CACHE[(s, k)] = rooms
    return _CACHE[(s, k)]


def h_of(s, k):
    return s * W + (k % W)


SYN_CATS = ["Alpha", "Beta", "Gamma"]          # the registered categories of the synthetic ROOMS


def _cats(rooms):
    """The categories of hand-made room lists, in first-appearance order (these fixtures register
    exactly the categories they hold; the registered-category refusals are tested separately)."""
    return list(dict.fromkeys(r["category"] for r in rooms))


# --------------------------------------------------------------------------- #
# Conventions: AR grid, pixel lookup, sign of k
# --------------------------------------------------------------------------- #
def test_unit_grid_is_the_AR_md_convention():
    ar_md = _load("AR_md_for_exp43_audit", _REPO / "src/configs/dataset_configs/custom_metadata/AR_md.py")
    import torch

    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 20.0, size=(H, W))
    ref = ar_md.convert_equirect_to_camera_coord(torch.from_numpy(depth), H, W).numpy()
    mine = fa.unit_grid() * depth[..., None]
    assert mine.shape == (H, W, 3) and mine.dtype == np.float64
    assert np.max(np.abs(mine - ref)) < 2e-5           # AR_md's trig is float32
    assert np.max(np.abs(fa.unit_grid() - ar_unit_grid())) < 1e-15
    # row 0 looks up (+z), the centre seam theta = 0 lies between columns 255 and 256 (+x)
    g = fa.unit_grid()
    assert g[0, 0, 2] > 0.99 and g[H - 1, 0, 2] < -0.99
    assert g[H // 2, W // 2, 0] > 0.99 and g[H // 2, W // 2 - 1, 0] > 0.99


def test_pixel_centres_map_back_to_themselves_under_every_hypothesis_family():
    g = ar_unit_grid().reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for s, k in [(0, 0), (0, 1), (0, 3), (0, 37), (0, 128), (0, 256), (0, 511), (1, 0), (1, 37), (1, 300)]:
        world = g @ hyp_matrix(s, k).T                      # where depth pixel (i, j) looks in the world
        rows, cols = fa.world_to_pixel(world, s, k)
        assert np.array_equal(rows, ii.ravel()) and np.array_equal(cols, jj.ravel()), (s, k)


def test_sign_of_k_and_column_index_arithmetic():
    # world direction at the centre of identity-frame column c=300, row 100
    th = (300 + 0.5) * 2 * np.pi / W - np.pi
    ph = (100 + 0.5) * np.pi / H - np.pi / 2
    w = np.array([[np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), -np.sin(ph)]])
    assert [int(x[0]) for x in fa.world_to_pixel(w, 0, 0)] == [100, 300]
    assert [int(x[0]) for x in fa.world_to_pixel(w, 0, 5)] == [100, 295]      # (c - k) mod W
    assert [int(x[0]) for x in fa.world_to_pixel(w, 0, 511)] == [100, 301]
    assert [int(x[0]) for x in fa.world_to_pixel(w, 1, 0)] == [100, 211]      # W - 1 - c
    assert [int(x[0]) for x in fa.world_to_pixel(w, 1, 5)] == [100, 216]      # (k - 1 - c) mod W
    # random directions away from cell boundaries: rotation formula == column-index arithmetic
    rng = np.random.default_rng(7)
    v = rng.normal(size=(20000, 3))
    c0 = fa.world_to_pixel(v, 0, 0)[1]
    frac = ((np.arctan2(v[:, 1], v[:, 0]) + np.pi) * W / (2 * np.pi)) % 1.0
    ok = (frac > 1e-6) & (frac < 1 - 1e-6)
    for k in [1, 2, 3, 37, 128, 255, 256, 400, 510, 511]:
        assert np.array_equal(fa.world_to_pixel(v, 0, k)[1][ok], ((c0 - k) % W)[ok])
        assert np.array_equal(fa.world_to_pixel(v, 1, k)[1][ok], ((k - 1 - c0) % W)[ok])
        assert np.array_equal(fa.world_to_pixel(v, 1, k)[0], fa.world_to_pixel(v, 0, 0)[0])


def test_poles_and_seam_are_clamped_and_wrapped():
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [-1.0, -0.0, 0.0], [1.0, 0.0, 0.0]])
    rows, cols = fa.pixel_of_direction(d)
    assert rows.tolist()[:2] == [0, H - 1]
    assert cols.tolist()[2] in (0, W - 1) and cols.tolist()[3] in (0, W - 1)
    assert cols.tolist()[4] == W // 2
    assert ((cols >= 0) & (cols < W)).all() and ((rows >= 0) & (rows < H)).all()


def test_injected_render_is_the_documented_roll_and_reversal():
    spec = ROOMS[0]
    rec = spec["receivers"][0]
    ident = render(spec, rec, 0, 0)
    assert np.allclose(render(spec, rec, 0, 3), np.roll(ident, -3, axis=1), atol=1e-9)
    assert np.allclose(render(spec, rec, 0, 37), np.roll(ident, -37, axis=1), atol=1e-9)
    assert np.allclose(render(spec, rec, 1, 0), ident[:, ::-1], atol=1e-9)
    assert np.allclose(render(spec, rec, 1, 37), np.roll(ident[:, ::-1], 37, axis=1), atol=1e-9)


def test_fixed_points_are_4096_regular_grid_indices():
    rows, cols = fa.point_indices()
    assert len(rows) == len(cols) == 4096
    assert sorted(set(rows.tolist())) == [4 * m + 2 for m in range(64)]
    assert sorted(set(cols.tolist())) == [8 * n + 4 for n in range(64)]
    assert rows[:65].tolist() == [2] * 64 + [6] and cols[:2].tolist() == [4, 12]


# --------------------------------------------------------------------------- #
# Synthetic proof: every injection is recovered exactly; only the identity PASSes
# --------------------------------------------------------------------------- #
def test_synthetic_rooms_are_non_symmetric_and_have_occlusion():
    n_blocked = 0
    for spec in ROOMS:
        recs = np.asarray(spec["receivers"])
        assert np.linalg.matrix_rank(recs[1:, :2] - recs[0, :2]) == 2        # receivers not collinear
        assert spec["size"][0] != spec["size"][1]
        for r in spec["receivers"]:
            for s in spec["sources"]:
                n_blocked += segment_blocked(r, s, spec["boxes"])
        assert len(spec["sources"]) >= 8 and len(spec["receivers"]) >= 3
    assert n_blocked >= 10


INJECTIONS = [(0, 0), (0, 3), (0, 128), (0, 256), (1, 0), (1, 37)]
BOUNDARY = [(0, 2), (0, 510), (0, 509)]


@pytest.mark.parametrize("s,k", INJECTIONS + BOUNDARY)
def test_injection_is_recovered_exactly_by_both_scores(s, k):
    rooms = scored(s, k)
    agg = fa.aggregate(rooms)
    assert fa.argmax_set(agg["A"].ravel()) == [h_of(s, k)]
    assert fa.argmax_set(agg["B"].ravel()) == [h_of(s, k)]
    for room in rooms:                     # each room on its own as well (so every resample agrees)
        fr = fa.room_fractions(room["scores"])
        assert fa.argmax_set(fr["A"].ravel()) == [h_of(s, k)], room["room_id"]
        assert fa.argmax_set(fr["B"].ravel()) == [h_of(s, k)], room["room_id"]


@pytest.mark.parametrize("s,k", INJECTIONS + BOUNDARY)
def test_gate_passes_only_injections_inside_the_accept_set(s, k):
    gate = fa.evaluate_gate(scored(s, k), categories=SYN_CATS)
    expect = "PASS" if (s == 0 and k in (510, 511, 0, 1, 2)) else "FAIL"
    assert gate["verdict"] == expect, gate["reasons"]
    if expect == "PASS":
        assert gate["reasons"] == []
        assert gate["bootstrap"]["n_pass"] == 1000
    else:
        assert any("full aggregate" in r for r in gate["reasons"])
        assert gate["bootstrap"]["n_pass"] == 0
    # amendment A2 (proposed): B within the accept set, A within the coarse set
    a2 = fa.evaluate_gate_a2(scored(s, k), categories=SYN_CATS)
    assert a2["verdict"] == expect, a2["reasons"]
    assert a2["bootstrap"]["n_pass"] == (1000 if expect == "PASS" else 0)
    both = fa.evaluate_gates(scored(s, k), categories=SYN_CATS)
    assert both["registered"] == gate and both["a2"] == a2


_RANDOM = {}


def random_scored():
    """The six rooms with 24 uniformly random sources each (seeds 4300 + n), TRUE frame (cached)."""
    if not _RANDOM:
        for n, spec in enumerate(ROOMS):
            sp = dict(spec, sources=random_sources(spec, 24, 4300 + n))
            _RANDOM[spec["room_id"]] = (sp, {"room_id": spec["room_id"], "category": spec["category"],
                                             "scores": fa.score_room(room_struct(sp, 0, 0))})
    return [v[1] for v in _RANDOM.values()]


def test_random_layouts_make_score_A_flat_near_the_true_frame_while_B_stays_exact():
    """Characterisation, not a gate input: 24 uniformly random sources per room, TRUE frame. B's
    arg-max is exactly (I, 0) in every room; A never confuses S but its top is flat over a few
    columns, so in several rooms its arg-max set is not {(I, 0)}. This is the main risk to the
    registered gate on real data (A must peak within +-2 columns); it is recorded, not tuned."""
    flat = []
    for room in random_scored():
        fr = fa.room_fractions(room["scores"])
        assert fa.argmax_set(fr["B"]) == [h_of(0, 0)], room["room_id"]
        am = fa.argmax_set(fr["A"])
        assert all(h // W == 0 for h in am), room["room_id"]
        if am != [h_of(0, 0)]:
            flat.append((room["room_id"], [fa.hyp_label(h) for h in am]))
    assert len(flat) >= 3, flat


def test_random_layouts_pass_a2_and_record_the_registered_verdict():
    """Same untuned random-layout fixture (flat A, exact B). Amendment A2 PASSes; the registered
    gate FAILs on it (recorded as observed: leave-out Alpha on A, bootstrap 659 / 1000)."""
    rooms = random_scored()
    a2 = fa.evaluate_gate_a2(rooms, categories=SYN_CATS)
    assert a2["verdict"] == "PASS", a2["reasons"]
    assert a2["bootstrap"]["n_pass"] == 1000
    reg = fa.evaluate_gate(rooms, categories=SYN_CATS)
    assert reg["verdict"] == "FAIL"
    assert reg["bootstrap"]["n_pass"] == 659
    assert [r.split(":")[0] for r in reg["reasons"]] == ["leave-out Alpha", "bootstrap"]
    d = a2["diagnostics"]
    assert d == reg["diagnostics"]
    assert d["B_argmax"] == ["I:k=0"] and d["A_argmax"] == reg["full"]["A"]["argmax"]
    assert d["A_at_B_argmax"][0]["label"] == "I:k=0"
    full_a = np.asarray(reg["full"]["A"]["curve"]).ravel()
    assert d["A_at_B_argmax"][0]["A"] == full_a[0] and d["A_max"] == full_a.max()
    assert d["A_at_B_argmax"][0]["A_max_minus_A_at_B_argmax"] == full_a.max() - full_a[0]
    assert all(x["same_S"] and x["circular_column_distance"] <= 10 for x in d["argmax_column_distance"])
    assert set(d["loco_argmax"]) == {"Alpha", "Beta", "Gamma"}
    assert d["loco_argmax"]["Alpha"] == {"A": reg["loco"]["Alpha"]["A"]["argmax"], "B": ["I:k=0"]}
    assert d["bootstrap_argmax_histogram_B"] == {"I:k=0": 1000}
    assert sum(d["bootstrap_argmax_histogram_A"].values()) == 1000


def test_true_frame_has_no_free_space_violation_ceiling_issue():
    """In the true frame a reprojected point never sees through a surface (beyond the
    nearest-pixel quantisation); a wrong frame does so often."""
    agg = fa.aggregate(scored(0, 0))
    v = agg["V"].ravel()
    assert v[h_of(0, 0)] < 0.03 and int(np.argmin(v)) == h_of(0, 0)
    assert v[h_of(0, 256)] > 3 * v[h_of(0, 0)] and v[h_of(1, 0)] > 3 * v[h_of(0, 0)]
    assert v[h_of(0, 1)] > v[h_of(0, 0)] and v[h_of(0, 511)] > v[h_of(0, 0)]


def test_single_receiver_room_contributes_to_A_but_not_B():
    spec = ROOMS[0]
    single = room_struct(spec, 0, 0, receivers=[spec["receivers"][0]])
    sc = fa.score_room(single)
    assert sc["B_n"] is None and sc["B_agree"] is None and sc["B_viol"] is None
    assert sc["A_n"] == len(spec["sources"])
    multi = scored(0, 0)[2]
    rooms = [{"room_id": "single", "category": "X", "scores": sc}, multi]
    agg = fa.aggregate(rooms)
    fs, fm = fa.room_fractions(sc), fa.room_fractions(multi["scores"])
    assert np.array_equal(agg["A"], (fs["A"] + fm["A"]) / 2.0)
    assert np.array_equal(agg["B"], fm["B"]) and np.array_equal(agg["V"], fm["V"])
    assert agg["n_rooms_A"] == 2 and agg["n_rooms_B"] == 1
    gate = fa.evaluate_gate(rooms, categories=_cats(rooms))
    assert gate["full"]["n_rooms_B"] == 1


def test_counts_match_a_direct_brute_force_of_score_A():
    """Score A of one receiver recomputed here with explicit matrices, pixel by pixel."""
    spec = ROOMS[3]
    rs = room_struct(spec, 0, 0, receivers=spec["receivers"][:1])
    rec = rs["receivers"][0]
    sc = fa.score_room(rs)
    u = rec["source_pos"] - rec["pos"]
    for s, k in [(0, 0), (0, 1), (0, 200), (1, 0), (1, 77)]:
        d = u @ np.linalg.inv(hyp_matrix(s, k)).T
        th = np.arctan2(d[:, 1], d[:, 0])
        ph = np.arctan2(-d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        col = np.floor((th + np.pi) * (W / (2 * np.pi))).astype(int) % W
        row = np.clip(np.floor((ph + np.pi / 2) * (H / np.pi)).astype(int), 0, H - 1)
        dd = rec["depth"][row, col]
        want = int(np.sum(dd >= np.linalg.norm(u, axis=1) - 0.10))
        assert sc["A_count"][s, k] == want


def test_score_B_matches_the_literal_world_placement_formula():
    """B computed by the tool equals the plan's literal recipe (place a's points in the world,
    direction and range from b, b's lookup under the same hypothesis), up to float rounding."""
    spec = ROOMS[4]
    rs = room_struct(spec, 0, 0, receivers=spec["receivers"][:2])
    a, b = rs["receivers"]
    sc = fa.score_room(rs)
    rows, cols = fa.point_indices()
    va = ar_unit_grid()[rows, cols] * a["depth"][rows, cols][:, None]
    for s, k in [(0, 0), (0, 2), (0, 129), (1, 0), (1, 400)]:
        agree = viol = 0
        for (src, dst) in [(a, b), (b, a)]:
            vs = ar_unit_grid()[rows, cols] * src["depth"][rows, cols][:, None]
            x = src["pos"] + vs @ hyp_matrix(s, k).T
            wv = x - dst["pos"]
            rng_ = np.linalg.norm(wv, axis=1)
            rr, cc = fa.world_to_pixel(wv, s, k)
            db = dst["depth"][rr, cc]
            tol = np.maximum(0.05, 0.02 * rng_)
            agree += int(np.sum(np.abs(db - rng_) <= tol))
            viol += int(np.sum(db > rng_ + tol))
        assert abs(int(sc["B_agree"][s, k]) - agree) <= 2        # boundary rounding only
        assert abs(int(sc["B_viol"][s, k]) - viol) <= 2
    assert va.shape == (4096, 3) and sc["B_n"] == 2 * 4096


def test_degenerate_pairs_and_invalid_pixels():
    spec = ROOMS[0]
    rs = room_struct(spec, 0, 0, receivers=spec["receivers"][:2])
    base = fa.score_room(rs)
    # a source exactly at the receiver is excluded (and counted)
    r0 = rs["receivers"][0]
    r0["source_pos"] = np.vstack([r0["source_pos"], r0["pos"][None]])
    r0["source_ids"] = r0["source_ids"] + [999]
    sc = fa.score_room(rs)
    assert sc["A_n"] == base["A_n"] and sc["n_degenerate_pairs"] == 1
    assert np.array_equal(sc["A_count"], base["A_count"])
    # zero-depth pixels (AR rendering holes): never contained / never agree; excluded as points
    rs2 = room_struct(spec, 0, 0, receivers=spec["receivers"][:2])
    rows, cols = fa.point_indices()
    rs2["receivers"][0]["depth"][rows[:10], cols[:10]] = 0.0
    rs2["receivers"][1]["depth"][:, :] = 0.0
    sc2 = fa.score_room(rs2)
    assert sc2["B_n"] == (4096 - 10) + 0                   # receiver 1 has no valid fixed point
    assert sc2["B_agree"].max() == 0 and sc2["B_viol"].max() == 0      # every lookup in 1 is a hole
    assert (sc2["A_count"] <= base["A_count"]).all() and sc2["A_n"] == base["A_n"]
    # holes in a only remove points: the 10 holed points of 0 -> 1 are simply absent
    rs3 = room_struct(spec, 0, 0, receivers=spec["receivers"][:2])
    rs3["receivers"][0]["depth"][rows[:10], cols[:10]] = 0.0
    sc3 = fa.score_room(rs3)
    assert sc3["B_n"] == (4096 - 10) + 4096
    assert (sc3["B_agree"] <= base["B_agree"]).all() and (sc3["B_viol"] <= base["B_viol"]).all()
    only_b_hole = room_struct(spec, 0, 0, receivers=spec["receivers"][:1])
    only_b_hole["receivers"][0]["depth"][:, :] = 0.0
    assert fa.score_room(only_b_hole)["A_count"].max() == 0


# --------------------------------------------------------------------------- #
# Gate mechanics on hand-made curves: ties, leave-one-category-out, bootstrap
# --------------------------------------------------------------------------- #
def _curve(base, peaks):
    c = np.full(2 * W, base, dtype=np.float64)
    for (s, k), v in peaks.items():
        c[h_of(s, k)] = v
    return c


def _room(rid, cat, a_curve, b_curve, n=1000):
    a = np.rint(np.asarray(a_curve) * n).astype(np.int64).reshape(2, W)
    b = None if b_curve is None else np.rint(np.asarray(b_curve) * n).astype(np.int64).reshape(2, W)
    sc = {"A_count": a, "A_n": n, "B_agree": b, "B_viol": None if b is None else np.zeros((2, W), np.int64),
          "B_n": None if b is None else n, "n_degenerate_pairs": 0, "n_excluded_receiver_pairs": 0}
    return {"room_id": rid, "category": cat, "scores": sc}


def test_argmax_set_tie_tolerance():
    c = _curve(0.1, {(0, 0): 0.5, (0, 3): 0.5 - 1e-13})
    assert fa.argmax_set(c) == [h_of(0, 0), h_of(0, 3)]
    c = _curve(0.1, {(0, 0): 0.5, (0, 3): 0.5 - 1e-11})
    assert fa.argmax_set(c) == [h_of(0, 0)]
    assert fa.TIE_TOL == 1e-12


def test_tie_outside_the_accept_set_fails_and_inside_passes():
    good_b = _curve(0.2, {(0, 0): 0.9})
    tie_out = [_room(f"R{i}", c, _curve(0.3, {(0, 0): 0.8, (0, 3): 0.8}), good_b) for i, c in enumerate("XXYY")]
    g = fa.evaluate_gate(tie_out, categories=_cats(tie_out))
    assert g["verdict"] == "FAIL" and any("A" in r and "full aggregate" in r for r in g["reasons"])
    assert g["full"]["A"]["argmax"] == ["I:k=0", "I:k=3"]
    tie_in = [_room(f"R{i}", c, _curve(0.3, {(0, 0): 0.8, (0, 511): 0.8}), good_b) for i, c in enumerate("XXYY")]
    assert fa.evaluate_gate(tie_in, categories=_cats(tie_in))["verdict"] == "PASS"
    tie_mirror = [_room(f"R{i}", c, _curve(0.3, {(0, 0): 0.8, (1, 0): 0.8}), good_b) for i, c in enumerate("XXYY")]
    assert fa.evaluate_gate(tie_mirror, categories=_cats(tie_mirror))["verdict"] == "FAIL"


def test_leave_one_category_out():
    good_b = _curve(0.2, {(0, 0): 0.9})
    rooms = [
        _room("X1", "X", _curve(0.4, {(0, 0): 0.9, (0, 40): 0.5}), good_b),
        _room("X2", "X", _curve(0.4, {(0, 0): 0.9, (0, 40): 0.5}), good_b),
        _room("Y1", "Y", _curve(0.4, {(0, 0): 0.55, (0, 40): 0.6}), good_b),
        _room("Y2", "Y", _curve(0.4, {(0, 0): 0.55, (0, 40): 0.6}), good_b),
        _room("Z1", "Z", _curve(0.4, {(0, 0): 0.55, (0, 40): 0.6}), good_b),
    ]
    g = fa.evaluate_gate(rooms, categories=_cats(rooms))
    assert g["full"]["A"]["argmax"] == ["I:k=0"]
    assert g["loco"]["X"]["A"]["argmax"] == ["I:k=40"]
    assert g["loco"]["Y"]["A"]["argmax"] == ["I:k=0"] and g["loco"]["Z"]["A"]["argmax"] == ["I:k=0"]
    assert g["verdict"] == "FAIL"
    assert any("leave-out X" in r and "A" in r for r in g["reasons"])
    assert not any("leave-out Y" in r or "leave-out Z" in r for r in g["reasons"])
    # the recorded leave-out curves are the plain means over the remaining rooms
    want = np.mean([fa.room_fractions(r["scores"])["A"] for r in rooms[2:]], axis=0)
    assert np.allclose(np.asarray(g["loco"]["X"]["A"]["curve"]), want, atol=1e-15)
    # B undefined after leaving a category out (only single-receiver rooms remain) -> FAIL
    rooms2 = [
        _room("X1", "X", _curve(0.4, {(0, 0): 0.9}), good_b),
        _room("Y1", "Y", _curve(0.4, {(0, 0): 0.9}), None),
    ]
    g2 = fa.evaluate_gate(rooms2, categories=_cats(rooms2))
    assert g2["verdict"] == "FAIL" and any("leave-out X" in r and "undefined" in r for r in g2["reasons"])


def test_bootstrap_is_seeded_43_and_matches_an_independent_count():
    good_b = _curve(0.2, {(0, 0): 0.9})
    rooms = [_room(f"G{i}", f"C{i}", _curve(0.4, {(0, 0): 0.6, (0, 100): 0.5}), good_b) for i in range(9)]
    rooms.append(_room("BAD", "C9", _curve(0.4, {(0, 0): 0.5, (0, 100): 1.0}), good_b))
    g1 = fa.evaluate_gate(rooms, categories=_cats(rooms))
    g2 = fa.evaluate_gate(rooms, categories=_cats(rooms))
    assert g1["bootstrap"] == g2["bootstrap"]
    idx = np.random.default_rng(43).integers(0, 10, size=(1000, 10))
    assert np.array_equal(fa.bootstrap_indices(10), idx)
    assert g1["bootstrap"]["indices_sha256"] == hashlib.sha256(np.ascontiguousarray(idx, dtype="<i8").tobytes()).hexdigest()
    m = (idx == 9).sum(axis=1)
    n_pass = int(np.sum(0.6 * (10 - m) + 0.5 * m > 0.5 * (10 - m) + 1.0 * m + 1e-9))
    assert g1["bootstrap"]["n_pass"] == n_pass and 0 < n_pass < 990
    assert g1["full"]["A"]["argmax"] == ["I:k=0"] and all(v["A"]["argmax"] == ["I:k=0"] for v in g1["loco"].values())
    assert g1["verdict"] == "FAIL" and len(g1["reasons"]) == 1 and "bootstrap" in g1["reasons"][0]
    assert fa.BOOT_N == 1000 and fa.BOOT_SEED == 43 and fa.BOOT_MIN_PASS == 990


def _four(a_curve, b_curve):
    return [_room(f"R{i}", c, a_curve, b_curve) for i, c in enumerate("XXYY")]


B_GOOD = _curve(0.2, {(0, 0): 0.9})


def test_a2_constants_and_rule_text():
    assert fa.COARSE_HALF_WIDTH == 32
    want = sorted(h_of(0, k) for k in list(range(0, 33)) + list(range(480, 512)))
    assert sorted(fa.COARSE) == want and len(fa.COARSE) == 65
    assert fa.ACCEPT <= fa.COARSE and sorted(fa.ACCEPT) == sorted(h_of(0, k) for k in (510, 511, 0, 1, 2))
    assert "PROPOSED 2026-11-13" in fa.A2_RULE_TEXT and ">= 990" in fa.A2_RULE_TEXT
    assert "Amendment A2 (PROPOSED 2026-11-13, awaiting Yixun; fixed before any real-data score)" in fa.__doc__


@pytest.mark.parametrize("ka", [5, 32, 480])
def test_a2_accepts_a_coarse_A_peak_that_the_registered_gate_rejects(ka):
    rooms = _four(_curve(0.3, {(0, ka): 0.8}), B_GOOD)
    reg = fa.evaluate_gate(rooms, categories=_cats(rooms))
    assert reg["verdict"] == "FAIL" and any("full aggregate" in r and "Score A" in r for r in reg["reasons"])
    a2 = fa.evaluate_gate_a2(rooms, categories=_cats(rooms))
    assert a2["verdict"] == "PASS" and a2["reasons"] == [] and a2["bootstrap"]["n_pass"] == 1000
    assert a2["coarse_half_width"] == 32 and a2["full"]["A"]["passes"] and a2["full"]["B"]["passes"]
    assert a2["diagnostics"]["argmax_column_distance"] == [
        {"A": f"I:k={ka}", "B": "I:k=0", "same_S": True, "circular_column_distance": min(ka, W - ka)}]


@pytest.mark.parametrize("sa,ka", [(0, 33), (0, 479), (0, 128), (1, 0), (1, 3)])
def test_a2_rejects_an_A_peak_outside_the_coarse_set(sa, ka):
    rooms = _four(_curve(0.3, {(sa, ka): 0.8}), B_GOOD)
    assert fa.evaluate_gate(rooms, categories=_cats(rooms))["verdict"] == "FAIL"
    a2 = fa.evaluate_gate_a2(rooms, categories=_cats(rooms))
    assert a2["verdict"] == "FAIL" and a2["bootstrap"]["n_pass"] == 0
    assert any(r.startswith("full aggregate") and "Score A" in r and "coarse" in r for r in a2["reasons"])
    assert not any("Score B" in r for r in a2["reasons"])


def test_a2_still_requires_B_in_the_accept_set():
    rooms = _four(_curve(0.3, {(0, 0): 0.8}), _curve(0.2, {(0, 3): 0.9}))
    assert fa.evaluate_gate(rooms, categories=_cats(rooms))["verdict"] == "FAIL"
    a2 = fa.evaluate_gate_a2(rooms, categories=_cats(rooms))
    assert a2["verdict"] == "FAIL"
    assert any(r.startswith("full aggregate") and "Score B" in r for r in a2["reasons"])
    assert not any("Score A" in r for r in a2["reasons"])


def test_a2_tie_between_I_and_M_fails():
    rooms = _four(_curve(0.3, {(0, 10): 0.8, (1, 10): 0.8}), B_GOOD)
    a2 = fa.evaluate_gate_a2(rooms, categories=_cats(rooms))
    assert a2["verdict"] == "FAIL" and a2["full"]["A"]["argmax"] == ["I:k=10", "M:k=10"]
    assert fa.evaluate_gate(rooms, categories=_cats(rooms))["verdict"] == "FAIL"
    dist = a2["diagnostics"]["argmax_column_distance"]
    assert {(x["A"], x["same_S"]) for x in dist} == {("I:k=10", True), ("M:k=10", False)}


def test_a2_leave_one_category_out():
    def rooms_with(kx):
        return [
            _room("X1", "X", _curve(0.4, {(0, 0): 0.9, (0, kx): 0.5}), B_GOOD),
            _room("X2", "X", _curve(0.4, {(0, 0): 0.9, (0, kx): 0.5}), B_GOOD),
            _room("Y1", "Y", _curve(0.4, {(0, 0): 0.55, (0, kx): 0.6}), B_GOOD),
            _room("Y2", "Y", _curve(0.4, {(0, 0): 0.55, (0, kx): 0.6}), B_GOOD),
            _room("Z1", "Z", _curve(0.4, {(0, 0): 0.55, (0, kx): 0.6}), B_GOOD),
        ]
    a2 = fa.evaluate_gate_a2(rooms_with(40), categories=["X", "Y", "Z"])   # leave-out X: A at (I, 40), outside COARSE
    assert a2["full"]["A"]["argmax"] == ["I:k=0"] and a2["loco"]["X"]["A"]["argmax"] == ["I:k=40"]
    assert a2["verdict"] == "FAIL"
    assert any(r.startswith("leave-out X") and "Score A" in r for r in a2["reasons"])
    assert not any(r.startswith("leave-out Y") or r.startswith("leave-out Z") for r in a2["reasons"])
    assert a2["diagnostics"]["loco_argmax"]["X"] == {"A": ["I:k=40"], "B": ["I:k=0"]}
    ok = fa.evaluate_gates(rooms_with(30), categories=["X", "Y", "Z"])     # (I, 30) is inside the coarse set
    assert not any(r.startswith("leave-out") for r in ok["a2"]["reasons"])
    assert any(r.startswith("leave-out X") for r in ok["registered"]["reasons"])
    # undefined B after leaving a category out fails A2 as registered
    g2 = fa.evaluate_gate_a2([_room("X1", "X", _curve(0.4, {(0, 0): 0.9}), B_GOOD),
                              _room("Y1", "Y", _curve(0.4, {(0, 0): 0.9}), None)], categories=["X", "Y"])
    assert g2["verdict"] == "FAIL" and any(r.startswith("leave-out X") and "undefined" in r for r in g2["reasons"])


@pytest.mark.parametrize("bad,n_fail", [(3, 10), (4, 11)])
def test_a2_bootstrap_threshold_990_passes_989_fails(bad, n_fail):
    """8 rooms, one category each. Good rooms: A(I,10) = 0.600 > A(I,100) = 0.500; the bad room:
    A(I,10) = 0.500 < A(I,100) = 0.629. A resample's A arg-max leaves the coarse set iff the bad room
    is drawn >= 4 times (4 x 129 > 4 x 100, 3 x 129 < 5 x 100). The shared rng-43 matrix then gives
    exactly 10 (bad = room 3) or 11 (bad = room 4) failing resamples."""
    rooms = [_room(f"G{i}", f"C{i}", _curve(0.3, {(0, 10): 0.6, (0, 100): 0.5}), B_GOOD) for i in range(8)]
    rooms[bad] = _room(f"BAD{bad}", f"C{bad}", _curve(0.3, {(0, 10): 0.5, (0, 100): 0.629}), B_GOOD)
    idx = np.random.default_rng(43).integers(0, 8, size=(1000, 8))
    assert int(np.sum((idx == bad).sum(axis=1) >= 4)) == n_fail          # independent count
    both = fa.evaluate_gates(rooms, categories=_cats(rooms))
    a2, reg = both["a2"], both["registered"]
    assert a2["bootstrap"]["n_pass"] == 1000 - n_fail
    assert a2["bootstrap"]["indices_sha256"] == reg["bootstrap"]["indices_sha256"]
    assert a2["full"]["A"]["argmax"] == ["I:k=10"] and all(v["A"]["passes"] for v in a2["loco"].values())
    if n_fail == 10:
        assert a2["verdict"] == "PASS" and a2["reasons"] == []
    else:
        assert a2["verdict"] == "FAIL" and a2["reasons"] == [
            f"bootstrap: 989 of 1000 resamples pass (need >= 990)"]
    assert reg["verdict"] == "FAIL" and reg["bootstrap"]["n_pass"] == 0        # (I, 10) is outside the accept set


def test_accept_set_is_the_registered_one():
    acc = sorted(fa.ACCEPT)
    assert acc == sorted(h_of(0, k) for k in (510, 511, 0, 1, 2))
    assert fa.hyp_label(h_of(1, 37)) == "M:k=37" and fa.hyp_label(h_of(0, 510)) == "I:k=510"


# --------------------------------------------------------------------------- #
# Synthetic AcousticRooms tree: freeze-inputs / run end to end
# --------------------------------------------------------------------------- #
def _ir(s, r):
    return f"S00{s}_R00{r}_hybrid_IR.wav"


def _js(s, r):
    return f"S00{s}_R00{r}.json"


def write_ar_tree(root, specs, s, k, rec_ids=None):
    """AcousticRooms layout: depth_map/<cat>/<room>/<r>.npy, metadata/<cat>/<room>/S00{s}_R00{r}.json."""
    split = {}
    for spec in specs:
        cat, rid = spec["category"], spec["room_id"]
        dd = root / "depth_map" / cat / rid
        md = root / "metadata" / cat / rid
        dd.mkdir(parents=True, exist_ok=True)
        md.mkdir(parents=True, exist_ok=True)
        ids = rec_ids or list(range(len(spec["receivers"])))
        names = []
        for r, p in zip(ids, spec["receivers"]):
            np.save(dd / f"{r}.npy", render(spec, p, s, k))
            for si, sp in enumerate(spec["sources"]):
                doc = {"sim_src_id": "x", "sim_rec_id": "y", "src_loc": [float(x) for x in sp],
                       "rec_loc": [float(x) for x in p], "IR_norm": 1.0}
                (md / _js(si, r)).write_text(json.dumps(doc))
                names.append(_ir(si, r))
        split.setdefault(cat, {})[rid] = names
    (root / "depth_map" / ".DS_Store").write_bytes(b"\0")
    split_path = root.parent / f"{root.name}_train.json"
    split_path.write_text(json.dumps(split, indent=1))
    return split_path


def _freeze(ar, split, out):
    return fa.main(["freeze-inputs", "--ar-root", str(ar), "--split", str(split), "--out", str(out)])


def _self_binding(inputs):
    """A synthetic inputs file bound to its OWN hashes (a non-registered binding; Python API only)."""
    raw = Path(inputs).read_bytes()
    return {"expected_inputs_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_selection_sha256": json.loads(raw)["selection_sha256"]}


def _run(inputs, ar, out, split, gate="registered", binding="self"):
    """``run_audit`` through the Python API: the CLI is bound to the registered inputs, which a
    synthetic tree never is. binding: "self" (the file's own hashes), None (the default, i.e. the
    registered pair) or a dict of the two ``expected_*`` keywords. AuditError -> EXIT_ERROR, as
    main() maps it; any other exception propagates."""
    if binding == "self":
        kw = _self_binding(inputs) if Path(inputs).exists() else {
            "expected_inputs_sha256": "0" * 64, "expected_selection_sha256": "0" * 64}
    else:
        kw = {} if binding is None else dict(binding)
    try:
        return fa.run_audit(str(inputs), str(ar), str(out), gate=gate,
                            split_path=None if split is None else str(split), threads=4, **kw)
    except fa.AuditError:
        return fa.EXIT_ERROR


def _cli(inputs, ar, out, split=None, gate="registered"):
    """The CLI ``run`` (always the registered binding)."""
    argv = ["run", "--inputs", str(inputs), "--ar-root", str(ar), "--out", str(out), "--threads", "4"]
    if split is not None:
        argv += ["--split", str(split)]
    if gate is not None:
        argv += ["--gate", gate]
    return fa.main(argv)


def test_end_to_end_identity_passes_and_records_everything(tmp_path):
    ar = tmp_path / "AR"
    split = write_ar_tree(ar, ROOMS, 0, 0)
    inp = tmp_path / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    doc = json.loads(inp.read_text())
    assert doc["selection"]["counts"]["total"]["rooms"] == 6
    assert doc["selection"]["counts"]["total"]["receivers"] == 24
    assert doc["selection"]["counts"]["total"]["pairs"] == sum(4 * len(r["sources"]) for r in ROOMS)
    assert doc["selection"]["counts"]["total"]["ordered_receiver_pairs"] == 6 * 12
    out = tmp_path / "audit.json"
    assert _run(inp, ar, out, split) == fa.EXIT_PASS
    res = json.loads(out.read_text())
    assert res["verdict"] == "PASS" and res["reasons"] == [] and res["gate_of_record"] == "registered"
    assert res["gate_registered"]["verdict"] == "PASS" and res["gate_a2_proposed"]["verdict"] == "PASS"
    assert res["a2_rule"] == fa.A2_RULE_TEXT and res["gate_a2_proposed"]["rule"] == fa.A2_RULE_TEXT
    assert res["gate_a2_proposed"]["coarse_half_width"] == 32
    assert res["inputs_sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
    assert res["tool_sha256"] == hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest()
    for key in ("git_head", "host", "timestamp"):
        assert key in res["provenance"]
    assert len(res["rooms"]) == 6
    per_room_A, per_room_B = [], []
    for room in res["rooms"]:
        a = np.asarray(room["A"]["count"])
        assert a.shape == (2, W)
        assert np.asarray(room["B"]["agree_count"]).shape == (2, W)
        assert np.asarray(room["B"]["viol_count"]).shape == (2, W)
        assert np.asarray(room["B"]["agree_frac"]).shape == (2, W)
        assert np.asarray(room["B"]["viol_frac"]).shape == (2, W)
        per_room_A.append(a / room["A"]["n"])
        per_room_B.append(np.asarray(room["B"]["agree_count"]) / room["B"]["n"])
    # every aggregate is recomputable from the per-room arrays
    assert np.array_equal(np.asarray(res["aggregates"]["all"]["A"]), np.mean(per_room_A, axis=0))
    assert np.allclose(np.asarray(res["aggregates"]["all"]["B"]), np.mean(per_room_B, axis=0), atol=1e-15)
    assert list(res["aggregates"]["loco"]) == SYN_CATS
    assert list(res["gate_registered"]["loco"]) == SYN_CATS == list(res["gate_a2_proposed"]["loco"])
    assert res["gate_registered"]["bootstrap"]["n_pass"] == 1000
    assert "gate" not in res
    # the split was re-derived (always), and the non-registered binding is labelled as such
    assert res["split_reverified"] is True
    assert res["selection_sha256"] == doc["selection_sha256"]
    b = res["inputs_binding"]
    assert b["registered"] is False and "NOT the official" in b["statement"]
    assert (b["inputs_sha256"], b["selection_sha256"]) == (res["inputs_sha256"], doc["selection_sha256"])
    assert (b["registered_inputs_sha256"], b["registered_selection_sha256"]) == (REG_INPUTS, REG_SELECTION)


def test_end_to_end_three_columns_fails_with_its_own_exit_code(tmp_path):
    ar = tmp_path / "AR"
    split = write_ar_tree(ar, ROOMS, 0, 3)
    inp = tmp_path / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    out = tmp_path / "audit.json"
    assert _run(inp, ar, out, split) == fa.EXIT_FAIL
    res = json.loads(out.read_text())
    assert res["verdict"] == "FAIL" and res["gate_registered"]["full"]["A"]["argmax"] == ["I:k=3"]
    assert res["gate_registered"]["full"]["B"]["argmax"] == ["I:k=3"]
    assert _run(inp, ar, tmp_path / "audit_a2.json", split, gate="a2") == fa.EXIT_FAIL
    assert json.loads((tmp_path / "audit_a2.json").read_text())["gate_a2_proposed"]["verdict"] == "FAIL"
    assert fa.EXIT_PASS == 0 and len({fa.EXIT_PASS, fa.EXIT_FAIL, fa.EXIT_ERROR, 1}) == 4


def test_run_refuses_any_input_hash_mismatch_or_missing_file(tmp_path):
    ar = tmp_path / "AR"
    split = write_ar_tree(ar, [ROOMS[0], ROOMS[2]], 0, 0)       # two categories (leave-one-out needs both)
    inp = tmp_path / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    doc = json.loads(inp.read_text())
    rel_json = doc["selection"]["rooms"][0]["receivers"][0]["metadata"][3]["path"]
    rel_depth = doc["selection"]["rooms"][1]["receivers"][2]["depth"]["path"]
    # (1) one changed byte in one metadata JSON
    p = ar / rel_json
    orig = p.read_bytes()
    p.write_bytes(orig.replace(b"1.0}", b"2.0}"))
    assert _run(inp, ar, tmp_path / "o1.json", split) == fa.EXIT_ERROR
    assert not (tmp_path / "o1.json").exists()
    p.write_bytes(orig)
    # (2) a missing depth map
    d = ar / rel_depth
    keep = d.read_bytes()
    d.unlink()
    assert _run(inp, ar, tmp_path / "o2.json", split) == fa.EXIT_ERROR
    d.write_bytes(keep)
    # (3) an edited inputs file (selection hash no longer matches)
    doc2 = json.loads(inp.read_text())
    doc2["selection"]["rooms"][0]["receivers"][0]["metadata"].pop()
    bad = tmp_path / "inputs_edited.json"
    bad.write_text(json.dumps(doc2))
    assert _run(bad, ar, tmp_path / "o3.json", split) == fa.EXIT_ERROR
    # (4) a consistent inputs file whose selection the split no longer reproduces
    doc3 = json.loads(inp.read_text())
    doc3["selection"]["rooms"][0]["receivers"][0]["metadata"].pop()
    doc3["selection_sha256"] = fa.canonical_sha256(doc3["selection"])
    forged = tmp_path / "inputs_forged.json"
    forged.write_text(json.dumps(doc3))
    assert _run(forged, ar, tmp_path / "o4.json", split) == fa.EXIT_ERROR
    # (5) the CLI refuses every one of them (bound to the registered pair), the untouched one included
    for i, x in enumerate((bad, forged, inp)):
        assert _cli(x, ar, tmp_path / f"c{i}.json", split) == fa.EXIT_ERROR
        assert not (tmp_path / f"c{i}.json").exists()
    assert not any((tmp_path / f"o{i}.json").exists() for i in range(1, 5))
    # sanity: the untouched inputs still run (Python API, bound to their own hashes)
    assert _run(inp, ar, tmp_path / "o5.json", split) == fa.EXIT_PASS


def test_run_requires_an_explicit_gate_and_both_gates_share_every_number(tmp_path):
    ar = tmp_path / "AR"
    specs = [dict(spec, sources=_RANDOM_SPECS()[spec["room_id"]]) for spec in ROOMS]
    split = write_ar_tree(ar, specs, 0, 0)                      # flat A, exact B (untuned random layouts)
    inp = tmp_path / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    missing = tmp_path / "no_gate.json"
    assert _cli(inp, ar, missing, split, gate=None) == fa.EXIT_ERROR and not missing.exists()
    assert _run(inp, ar, missing, split, gate=None) == fa.EXIT_ERROR and not missing.exists()
    bogus = tmp_path / "bogus_gate.json"
    assert _cli(inp, ar, bogus, split, gate="lenient") == fa.EXIT_ERROR and not bogus.exists()
    assert _run(inp, ar, bogus, split, gate="lenient") == fa.EXIT_ERROR and not bogus.exists()
    out_r, out_a = tmp_path / "reg.json", tmp_path / "a2.json"
    assert _run(inp, ar, out_r, split, gate="registered") == fa.EXIT_FAIL
    assert _run(inp, ar, out_a, split, gate="a2") == fa.EXIT_PASS
    r, a = json.loads(out_r.read_text()), json.loads(out_a.read_text())
    assert (r["gate_of_record"], r["verdict"]) == ("registered", "FAIL")
    assert (a["gate_of_record"], a["verdict"]) == ("a2", "PASS")
    assert r["reasons"] == r["gate_registered"]["reasons"] and a["reasons"] == a["gate_a2_proposed"]["reasons"] == []
    assert r["gate_registered"]["verdict"] == "FAIL" and r["gate_a2_proposed"]["verdict"] == "PASS"
    drop = {"gate_of_record", "verdict", "reasons", "provenance", "runtime_s"}
    assert {k: v for k, v in r.items() if k not in drop} == {k: v for k, v in a.items() if k not in drop}
    assert set(r) == set(a)


def _RANDOM_SPECS():
    return {spec["room_id"]: random_sources(spec, 24, 4300 + n) for n, spec in enumerate(ROOMS)}


def test_run_refuses_to_overwrite_and_reports_tool_errors(tmp_path):
    out = tmp_path / "exists.json"
    out.write_text("{}")
    assert _run(tmp_path / "missing_inputs.json", tmp_path, out, tmp_path / "split.json") == fa.EXIT_ERROR
    assert _cli(tmp_path / "missing_inputs.json", tmp_path, out, tmp_path / "split.json") == fa.EXIT_ERROR
    assert out.read_text() == "{}"


# --------------------------------------------------------------------------- #
# Selection: the frozen rule, determinism, listing order, skips
# --------------------------------------------------------------------------- #
def _fake_tree(root, layout):
    """layout: {cat: {room: dict(receivers=[...], no_depth=[...], no_meta=[...], sources=[...])}}"""
    split = {}
    for cat, rooms in layout.items():
        for rid, spec in rooms.items():
            dd = root / "depth_map" / cat / rid
            md = root / "metadata" / cat / rid
            dd.mkdir(parents=True, exist_ok=True)
            md.mkdir(parents=True, exist_ok=True)
            names = []
            for r in spec["receivers"]:
                if r not in spec.get("no_depth", ()):
                    np.save(dd / f"{r}.npy", np.full((H, W), 2.0 + 0.01 * r))
                for s in spec["sources"]:
                    names.append(f"S{s:03d}_R{r:03d}_hybrid_IR.wav" if spec.get("padded") else _ir(s, r))
                    if r in spec.get("no_meta", ()):
                        continue
                    doc = {"src_loc": [1.0 + s, 2.0, 1.5], "rec_loc": [0.5, 0.5 + 0.1 * r, 1.2]}
                    (md / _js(s, r)).write_text(json.dumps(doc))
            (md / ".DS_Store").write_bytes(b"\0")
            (md / "S0001_R001.json").write_text("{}")          # non-canonical name: never matched
            names.append("README_hybrid.txt")                  # never parsed
            split.setdefault(cat, {})[rid] = names
    sp = root.parent / "split.json"
    sp.write_text(json.dumps(split))
    return sp


LAYOUT = {
    "Alpha": {f"Alpha_idx_{i}": dict(receivers=list(range(10)), no_depth=[2, 5], no_meta=[7], sources=[0, 1, 3])
              for i in range(6)},
    "Beta": {f"Beta_idx_{i}": dict(receivers=[0, 1, 4], sources=[0, 2]) for i in range(2)},
    # zero-padded IR names ('S001_R010_hybrid_IR.wav'), as in many AR rooms; metadata stays 'S00{s}_R00{r}'
    "Gamma": {"Gamma_idx_0": dict(receivers=[3, 10, 12, 25, 31, 47, 99, 100], sources=[1, 10, 11], padded=True)},
}


def _expected_selection(layout):
    want = {}
    for cat in sorted(layout):
        ranked = sorted(layout[cat], key=lambda rid: hashlib.sha256(("exp43-frame-audit:" + rid).encode()).hexdigest())
        for rid in ranked[:4]:
            spec = layout[cat][rid]
            order = sorted(spec["receivers"],
                           key=lambda r: hashlib.sha256(f"exp43-frame-audit:{rid}/R{r}".encode()).hexdigest())
            usable = [r for r in order if r not in spec.get("no_depth", ()) and r not in spec.get("no_meta", ())]
            want[(cat, rid)] = usable[:6]
    return want


def test_selection_follows_the_registered_rule(tmp_path):
    ar = tmp_path / "AR"
    split = _fake_tree(ar, LAYOUT)
    sel = fa.select_inputs(str(ar), str(split))
    got = {(r["category"], r["room_id"]): [x["receiver"] for x in r["receivers"]] for r in sel["rooms"]}
    assert got == _expected_selection(LAYOUT)
    assert [r["room_id"] for r in sel["rooms"] if r["category"] == "Beta"] == sorted(
        LAYOUT["Beta"], key=lambda rid: hashlib.sha256(("exp43-frame-audit:" + rid).encode()).hexdigest())
    a0 = next(r for r in sel["rooms"] if r["category"] == "Alpha")
    for rec in a0["receivers"]:
        assert [m["source"] for m in rec["metadata"]] == [0, 1, 3]
        assert rec["depth"]["path"] == f"depth_map/Alpha/{a0['room_id']}/{rec['receiver']}.npy"
        assert rec["depth"]["sha256"] == hashlib.sha256((ar / rec["depth"]["path"]).read_bytes()).hexdigest()
        for m in rec["metadata"]:
            assert m["path"] == f"metadata/Alpha/{a0['room_id']}/S00{m['source']}_R00{rec['receiver']}.json"
            assert m["sha256"] == hashlib.sha256((ar / m["path"]).read_bytes()).hexdigest()
    reasons = {(s["room_id"], s.get("receiver")): s["reason"] for s in sel["skipped"]}
    skipped_r = {r for (rid, r) in reasons if rid == a0["room_id"]}
    assert skipped_r <= {2, 5, 7}
    for (rid, r), why in reasons.items():
        if r in (2, 5):
            assert "no depth map" in why
        if r == 7:
            assert "no metadata" in why
    c = sel["counts"]
    assert c["per_category"]["Alpha"]["rooms"] == 4 and c["per_category"]["Beta"]["rooms"] == 2
    assert c["per_category"]["Beta"]["receivers"] == 6 and c["per_category"]["Beta"]["pairs"] == 12
    assert c["per_category"]["Beta"]["ordered_receiver_pairs"] == 2 * 6
    assert c["per_category"]["Alpha"]["receivers"] == 24 and c["per_category"]["Alpha"]["pairs"] == 72
    assert sel["split"]["sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert "exp43-frame-audit:" in sel["rule"] and "first 4 rooms" in sel["rule"]
    g = next(r for r in sel["rooms"] if r["category"] == "Gamma")        # AR_md-style int() parse
    assert g["n_candidate_receivers"] == 8 and len(g["receivers"]) == 6
    assert g["n_split_names_unparsed"] == 1 and g["n_metadata_names_ignored"] == 2
    assert all(x["n_metadata_not_in_split"] == 0 for x in g["receivers"])
    assert {x["receiver"] for x in g["receivers"]} <= {3, 10, 12, 25, 31, 47, 99, 100}
    assert [m["path"].rsplit("/", 1)[1] for m in g["receivers"][0]["metadata"]] == [
        f"S00{s}_R00{g['receivers'][0]['receiver']}.json" for s in (1, 10, 11)]


def test_selection_is_deterministic_and_independent_of_listing_order(tmp_path, monkeypatch):
    ar = tmp_path / "AR"
    split = _fake_tree(ar, LAYOUT)
    ref = fa.select_inputs(str(ar), str(split))
    real_listdir = os.listdir
    for seed in (1, 2, 3):
        rnd = random.Random(seed)

        def shuffled(path=".", _rnd=rnd):
            out = list(real_listdir(path))
            _rnd.shuffle(out)
            return out

        monkeypatch.setattr(os, "listdir", shuffled)
        assert fa.select_inputs(str(ar), str(split)) == ref
        monkeypatch.setattr(os, "listdir", lambda p=".": sorted(real_listdir(p), reverse=True))
        assert fa.select_inputs(str(ar), str(split)) == ref
    monkeypatch.setattr(os, "listdir", real_listdir)
    # the split's key order does not matter either (only its bytes' hash changes)
    d = json.loads(split.read_text())
    rev = {c: {r: d[c][r][::-1] for r in reversed(list(d[c]))} for c in reversed(list(d))}
    split2 = tmp_path / "split_rev.json"
    split2.write_text(json.dumps(rev))
    sel2 = fa.select_inputs(str(ar), str(split2))
    assert sel2["rooms"] == ref["rooms"] and sel2["counts"] == ref["counts"]


def test_freeze_computes_no_score(tmp_path, monkeypatch):
    ar = tmp_path / "AR"
    split = _fake_tree(ar, LAYOUT)

    def boom(*a, **k):
        raise AssertionError("freeze-inputs must not score")

    for name in ("score_room", "_score_a_receiver", "_score_b_pair", "evaluate_gate", "aggregate"):
        monkeypatch.setattr(fa, name, boom)
    out = tmp_path / "inputs.json"
    assert _freeze(ar, split, out) == 0
    doc = json.loads(out.read_text())
    assert doc["selection_sha256"] == fa.canonical_sha256(doc["selection"])
    assert doc["schema"] == fa.INPUTS_SCHEMA
    assert "score" not in json.dumps(doc["selection"]).lower().replace("no score", "")
    assert _freeze(ar, split, out) == fa.EXIT_ERROR                  # never overwrites


@pytest.mark.parametrize("bad", [
    np.zeros((255, W)), np.zeros((H, 511)), np.zeros((3, H, W)), np.zeros((H, W), dtype=np.int64),
    np.full((H, W), np.nan), np.full((H, W), np.inf), np.full((H, W), -1.0), np.zeros((H, W), dtype=bool),
])
def test_malformed_depth_is_refused(bad):
    with pytest.raises(fa.AuditError):
        fa.validate_depth(bad)


def test_valid_depth_with_holes_is_accepted():
    d = np.full((H, W), 3.0)
    d[10, 20] = 0.0
    out = fa.validate_depth(d)
    assert out.dtype == np.float64 and out.shape == (H, W)
    assert fa.validate_depth(d.astype(np.float32)).dtype == np.float64


def test_freeze_and_run_refuse_malformed_depth(tmp_path):
    ar = tmp_path / "AR"
    split = _fake_tree(ar, {"Alpha": {"Alpha_idx_0": dict(receivers=[0, 1], sources=[0])}})
    np.save(ar / "depth_map/Alpha/Alpha_idx_0/1.npy", np.zeros((H, W - 1)))
    assert _freeze(ar, split, tmp_path / "i.json") == fa.EXIT_ERROR
    # a pickled object array is refused at load time (allow_pickle=False)
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(fa.AuditError):
        fa.load_depth_bytes(buf.getvalue())
    # run: a hand-made inputs file that points at a malformed map with its correct hash
    np.save(ar / "depth_map/Alpha/Alpha_idx_0/1.npy", np.full((H, W), 2.0))
    assert _freeze(ar, split, tmp_path / "i.json") == 0
    doc = json.loads((tmp_path / "i.json").read_text())
    rec = doc["selection"]["rooms"][0]["receivers"][0]
    np.save(ar / rec["depth"]["path"], np.zeros((H - 1, W)))
    rec["depth"]["sha256"] = hashlib.sha256((ar / rec["depth"]["path"]).read_bytes()).hexdigest()
    doc["selection_sha256"] = fa.canonical_sha256(doc["selection"])
    (tmp_path / "i2.json").write_text(json.dumps(doc))
    assert _run(tmp_path / "i2.json", ar, tmp_path / "o.json", split) == fa.EXIT_ERROR
    assert not (tmp_path / "o.json").exists()


def test_freeze_refuses_inconsistent_receiver_position(tmp_path):
    ar = tmp_path / "AR"
    split = _fake_tree(ar, {"Alpha": {"Alpha_idx_0": dict(receivers=[0, 1], sources=[0, 1])}})
    p = ar / "metadata/Alpha/Alpha_idx_0" / _js(1, 0)
    doc = json.loads(p.read_text())
    doc["rec_loc"][0] += 0.25
    p.write_text(json.dumps(doc))
    assert _freeze(ar, split, tmp_path / "i.json") == fa.EXIT_ERROR


# --------------------------------------------------------------------------- #
# The committed real-data input list (no real data read here: internal consistency only)
# --------------------------------------------------------------------------- #
def test_committed_inputs_file_is_internally_consistent():
    doc = json.loads(INPUTS_PATH.read_text())
    assert doc["schema"] == fa.INPUTS_SCHEMA
    assert doc["selection_sha256"] == fa.canonical_sha256(doc["selection"])
    sel = doc["selection"]
    assert sel["constants"] == fa.selection_constants()
    tot = {"rooms": 0, "receivers": 0, "pairs": 0, "ordered_receiver_pairs": 0}
    for room in sel["rooms"]:
        n = len(room["receivers"])
        assert 1 <= n <= 6
        tot["rooms"] += 1
        tot["receivers"] += n
        tot["pairs"] += sum(len(r["metadata"]) for r in room["receivers"])
        tot["ordered_receiver_pairs"] += n * (n - 1)
        for rec in room["receivers"]:
            assert rec["depth"]["path"] == f"depth_map/{room['category']}/{room['room_id']}/{rec['receiver']}.npy"
    assert {k: sel["counts"]["total"][k] for k in tot} == tot
    per_cat = {}
    for room in sel["rooms"]:
        per_cat[room["category"]] = per_cat.get(room["category"], 0) + 1
    assert all(v <= 4 for v in per_cat.values())
    assert set(per_cat) == {c["category"] for c in sel["categories"] if c["rooms_selected"]}
    assert all(c["rooms_selected"] for c in sel["categories"])          # no empty registered category
    assert list(per_cat) == [c["category"] for c in sel["categories"]] == REG_CATEGORIES


def test_committed_inputs_file_is_the_registered_one():
    """The registered constants pin the committed inputs file byte for byte (JSON only; no AR file)."""
    assert fa.REGISTERED_INPUTS_SHA256 == REG_INPUTS
    assert fa.REGISTERED_SELECTION_SHA256 == REG_SELECTION
    raw = INPUTS_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == fa.REGISTERED_INPUTS_SHA256
    doc = json.loads(raw)
    assert doc["selection_sha256"] == fa.REGISTERED_SELECTION_SHA256
    assert fa.canonical_sha256(doc["selection"]) == fa.REGISTERED_SELECTION_SHA256
    assert fa.frozen_categories(doc["selection"]) == REG_CATEGORIES
    assert doc["selection"]["rule"] == fa.RULE_TEXT


# --------------------------------------------------------------------------- #
# Binding of the frozen inputs (Codex code review r1, finding 1)
# --------------------------------------------------------------------------- #
def _two_room_inputs(base):
    """Synthetic AR tree with Alpha_idx_0 and Beta_idx_3 (two categories), frozen."""
    base.mkdir(parents=True, exist_ok=True)
    ar = base / "AR"
    split = write_ar_tree(ar, [ROOMS[0], ROOMS[2]], 0, 0)
    inp = base / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    return ar, split, inp


def _forge(src, dst, edit):
    """A self-consistent forgery: edit the selection, recompute its selection_sha256."""
    doc = json.loads(Path(src).read_text())
    edit(doc["selection"])
    doc["selection_sha256"] = fa.canonical_sha256(doc["selection"])
    Path(dst).write_text(json.dumps(doc))
    return dst


def _no_scoring(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a refused run must not compute a score")

    for name in ("score_room", "_score_a_receiver", "_score_b_pair", "evaluate_gates"):
        monkeypatch.setattr(fa, name, boom)


def _api(inputs, ar, out, split, **kw):
    kw.setdefault("gate", "registered")
    return fa.run_audit(str(inputs), str(ar), str(out), split_path=None if split is None else str(split),
                        threads=4, **kw)


def test_the_binding_defaults_to_the_registered_pair_and_no_cli_flag_changes_it(monkeypatch):
    for fn in (fa.run_audit, fa.load_inputs):
        params = inspect.signature(fn).parameters
        for name, want in (("expected_inputs_sha256", REG_INPUTS), ("expected_selection_sha256", REG_SELECTION)):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY and params[name].default == want, fn
    sp = inspect.signature(fa.run_audit).parameters["split_path"]
    assert sp.kind is inspect.Parameter.KEYWORD_ONLY and sp.default is inspect.Parameter.empty
    sub = next(a for a in fa._parser()._actions if isinstance(a, argparse._SubParsersAction)).choices["run"]
    assert {o for a in sub._actions for o in a.option_strings} == {
        "-h", "--help", "--inputs", "--ar-root", "--out", "--split", "--threads", "--gate"}
    assert {a.dest for a in sub._actions if a.required} == {"inputs", "ar_root", "out", "split", "gate"}
    # main passes no binding keyword, so run_audit's registered defaults always apply to the CLI
    seen = []

    def spy(inputs, ar_root, out_path, **kw):
        seen.append((inputs, ar_root, out_path, kw))
        return spy.rc

    monkeypatch.setattr(fa, "run_audit", spy)
    for rc in (fa.EXIT_PASS, fa.EXIT_FAIL):
        spy.rc = rc
        argv = ["run", "--inputs", "i.json", "--ar-root", "AR", "--out", "o.json", "--split", "s.json", "--gate", "a2"]
        assert fa.main(argv) == rc
    assert seen == [("i.json", "AR", "o.json", {"gate": "a2", "split_path": "s.json", "threads": 16})] * 2


def test_inputs_binding_is_registered_only_for_the_registered_pair():
    b = fa.inputs_binding(REG_INPUTS, REG_SELECTION)
    assert b["registered"] is True and "NOT" not in b["statement"]
    assert (b["inputs_sha256"], b["selection_sha256"]) == (REG_INPUTS, REG_SELECTION)
    other = "0" * 64
    for pair in ((REG_INPUTS, other), (other, REG_SELECTION), (other, other), (REG_SELECTION, REG_INPUTS)):
        b = fa.inputs_binding(*pair)
        assert b["registered"] is False and "NOT the official" in b["statement"], pair
        assert (b["inputs_sha256"], b["selection_sha256"]) == pair
        assert (b["registered_inputs_sha256"], b["registered_selection_sha256"]) == (REG_INPUTS, REG_SELECTION)


def test_a_self_consistent_forged_subset_is_refused_before_any_score(tmp_path, monkeypatch, capsys):
    ar, split, inp = _two_room_inputs(tmp_path / "t")
    forged = _forge(inp, tmp_path / "forged.json",                      # one (receiver, source) pair fewer
                    lambda sel: sel["rooms"][0]["receivers"][0]["metadata"].pop())
    dropped = _forge(inp, tmp_path / "dropped.json",                    # a whole room and its category fewer
                     lambda sel: (sel["rooms"].pop(), sel["categories"].pop()))
    room_only = _forge(inp, tmp_path / "room_only.json",                # the room fewer, its category kept
                       lambda sel: sel["rooms"].pop())
    _no_scoring(monkeypatch)
    # (1) the CLI, without and with --split: refused, nothing written
    for i, x in enumerate((forged, dropped, room_only, inp)):
        for j, sp in enumerate((None, split)):
            out = tmp_path / f"cli_{i}_{j}.json"
            assert _cli(x, ar, out, sp) == fa.EXIT_ERROR and not out.exists()
    err = capsys.readouterr().err
    assert "--split" in err and "REFUSED" in err and fa.REGISTERED_INPUTS_SHA256 in err
    # (2) the Python API under its default binding (the registered pair): refused
    out = tmp_path / "api.json"
    for x in (forged, dropped, room_only, inp):
        with pytest.raises(fa.AuditError, match="registered exp43_frame_audit_inputs.json"):
            _api(x, ar, out, split)
    # (3) bound to the original file (as the registered pair binds the committed one): the forged bytes
    with pytest.raises(fa.AuditError, match="inputs file sha256"):
        _api(forged, ar, out, split, **_self_binding(inp))
    # (4) bound to the forgery's own hashes: the split + tree re-derivation refuses it
    for x in (forged, dropped):
        with pytest.raises(fa.AuditError, match="no longer reproduce"):
            _api(x, ar, out, split, **_self_binding(x))
    with pytest.raises(fa.AuditError, match=r"registered categories \['Beta'\] have no room"):
        _api(room_only, ar, out, split, **_self_binding(room_only))     # refused at load already
    # (5) both hashes are enforced: right bytes + wrong selection hash; same selection + other bytes
    with pytest.raises(fa.AuditError, match="selection sha256"):
        _api(inp, ar, out, split, expected_inputs_sha256=_self_binding(inp)["expected_inputs_sha256"],
             expected_selection_sha256=_self_binding(forged)["expected_selection_sha256"])
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(json.loads(inp.read_text()), indent=3))
    assert json.loads(reformatted.read_text())["selection"] == json.loads(inp.read_text())["selection"]
    with pytest.raises(fa.AuditError, match="inputs file sha256"):
        _api(reformatted, ar, out, split, **_self_binding(inp))
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dropped.json", "forged.json", "reformatted.json", "room_only.json", "t"]


def test_run_requires_the_split(tmp_path, monkeypatch, capsys):
    ar, split, inp = _two_room_inputs(tmp_path / "t")
    _no_scoring(monkeypatch)
    out = tmp_path / "o.json"
    assert _cli(inp, ar, out, None) == fa.EXIT_ERROR and not out.exists()
    assert "the following arguments are required: --split" in capsys.readouterr().err
    with pytest.raises(fa.AuditError, match="split"):
        _api(inp, ar, out, None, **_self_binding(inp))
    with pytest.raises(TypeError):                                       # no default either
        fa.run_audit(str(inp), str(ar), str(out), gate="registered", **_self_binding(inp))
    assert not out.exists()


def test_a_changed_split_or_tree_is_refused(tmp_path, monkeypatch):
    ar, split, inp = _two_room_inputs(tmp_path / "t")
    bind = _self_binding(inp)
    orig = split.read_bytes()
    out = tmp_path / "o.json"

    def refused():
        with pytest.raises(fa.AuditError, match="no longer reproduce"):
            _api(inp, ar, out, split, **bind)
        assert not out.exists()

    with monkeypatch.context() as m:
        _no_scoring(m)
        # split (1): the same content in other bytes (the frozen selection records the split's sha256)
        split.write_text(json.dumps(json.loads(orig), indent=4))
        refused()
        # split (2): one selected receiver's IR names removed
        d = json.loads(orig)
        d["Alpha"]["Alpha_idx_0"] = [n for n in d["Alpha"]["Alpha_idx_0"] if "_R001_" not in n]
        split.write_text(json.dumps(d))
        refused()
        # split (3): one more category (a room of it would be selected)
        d = json.loads(orig)
        d["Gamma"] = {"Gamma_idx_2": [_ir(0, 0)]}
        split.write_text(json.dumps(d))
        refused()
        # split (4): a selected room removed
        d = json.loads(orig)
        del d["Beta"]
        split.write_text(json.dumps(d))
        refused()
        split.write_bytes(orig)
        # tree: a new canonical metadata JSON for a taken receiver (every frozen file is untouched)
        md = ar / "metadata" / "Alpha" / "Alpha_idx_0"
        src = json.loads((md / _js(0, 0)).read_text())
        src["src_loc"] = [3.0, 2.0, 1.0]
        extra = md / _js(99, 0)
        extra.write_text(json.dumps(src))
        refused()
        extra.unlink()
    # control: split and tree restored -> the run goes through
    assert _api(inp, ar, out, split, **bind) == fa.EXIT_PASS
    assert json.loads(out.read_text())["split_reverified"] is True


def test_a_newly_present_depth_map_of_a_skipped_receiver_is_refused(tmp_path, monkeypatch):
    rid = "Alpha_idx_0"
    first = min(range(8), key=lambda r: hashlib.sha256(f"exp43-frame-audit:{rid}/R{r}".encode()).hexdigest())
    base = tmp_path / "t"
    ar = base / "AR"
    split = _fake_tree(ar, {"Alpha": {rid: dict(receivers=list(range(8)), no_depth=[first], sources=[0, 1])},
                            "Beta": {"Beta_idx_0": dict(receivers=[0, 1], sources=[0])}})
    inp = base / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    sel = json.loads(inp.read_text())["selection"]
    assert {"category": "Alpha", "room_id": rid, "receiver": first, "reason": f"no depth map depth_map/Alpha/{rid}/{first}.npy"} in sel["skipped"]
    _no_scoring(monkeypatch)
    np.save(ar / "depth_map" / "Alpha" / rid / f"{first}.npy", np.full((H, W), 2.5))
    out = tmp_path / "o.json"
    with pytest.raises(fa.AuditError, match="no longer reproduce"):
        _api(inp, ar, out, split, **_self_binding(inp))
    assert not out.exists()


def test_frozen_categories_refuses_an_empty_or_inconsistent_category():
    sel = {"categories": [{"category": "A", "n_rooms_in_split": 3, "rooms_selected": ["A_1", "A_2"]},
                          {"category": "B", "n_rooms_in_split": 1, "rooms_selected": ["B_1"]}],
           "rooms": [{"category": "A", "room_id": "A_1"}, {"category": "A", "room_id": "A_2"},
                     {"category": "B", "room_id": "B_1"}]}
    assert fa.frozen_categories(sel) == ["A", "B"]

    def edited(fn):
        s = copy.deepcopy(sel)
        fn(s)
        return s

    bad = {
        "no category": edited(lambda s: s.update(categories=[], rooms=[])),
        "categories missing": edited(lambda s: s.pop("categories")),
        "zero selected rooms": edited(lambda s: s["categories"].append(
            {"category": "C", "n_rooms_in_split": 0, "rooms_selected": []})),
        "every selected room skipped": edited(lambda s: s["rooms"].pop()),
        "room of an unlisted category": edited(lambda s: s["rooms"].append({"category": "C", "room_id": "C_1"})),
        "room not selected in its category": edited(lambda s: s["rooms"].append({"category": "B", "room_id": "B_9"})),
        "duplicate category": edited(lambda s: s["categories"].append(dict(s["categories"][1]))),
        "duplicate room": edited(lambda s: s["rooms"].append(dict(s["rooms"][0]))),
        "unsafe name": edited(lambda s: s["categories"][0].update(category="../A")),
    }
    for why, s in bad.items():
        with pytest.raises(fa.AuditError):
            fa.frozen_categories(s)
            pytest.fail(why)


def test_run_refuses_inputs_with_an_empty_category(tmp_path, monkeypatch):
    # (1) a category with zero rooms in the split -> zero selected rooms (freeze records it as is)
    base = tmp_path / "t1"
    base.mkdir()
    ar = base / "AR"
    split = write_ar_tree(ar, [ROOMS[0], ROOMS[2]], 0, 0)
    d = json.loads(split.read_text())
    d["Delta"] = {}
    split.write_text(json.dumps(d))
    inp = base / "inputs.json"
    assert _freeze(ar, split, inp) == 0
    assert {"category": "Delta", "n_rooms_in_split": 0, "rooms_selected": []} in json.loads(inp.read_text())["selection"]["categories"]
    # (2) a category whose only selected room has no usable receiver (skipped at freeze)
    base2 = tmp_path / "t2"
    base2.mkdir()
    ar2 = base2 / "AR"
    split2 = write_ar_tree(ar2, [ROOMS[0], ROOMS[2], ROOMS[4]], 0, 0)
    for f in (ar2 / "depth_map" / "Gamma" / "Gamma_idx_2").iterdir():
        f.unlink()
    inp2 = base2 / "inputs.json"
    assert _freeze(ar2, split2, inp2) == 0
    sel2 = json.loads(inp2.read_text())["selection"]
    assert [r["category"] for r in sel2["rooms"]] == ["Alpha", "Beta"] and len(sel2["categories"]) == 3
    _no_scoring(monkeypatch)
    for (a, sp, x, why) in ((ar, split, inp, "registered category Delta has zero selected rooms"),
                            (ar2, split2, inp2, r"registered categories \['Gamma'\] have no room")):
        out = x.parent / "o.json"
        with pytest.raises(fa.AuditError, match=why):
            _api(x, a, out, sp, **_self_binding(x))
        with pytest.raises(fa.AuditError, match=why):
            fa.load_inputs(str(x), str(a), **_self_binding(x))
        assert _cli(x, a, out, sp) == fa.EXIT_ERROR
        assert not out.exists()


def test_every_registered_category_needs_a_scored_room_and_its_loco_check():
    rooms = scored(0, 0)
    assert fa.evaluate_gates(rooms, categories=SYN_CATS)["registered"]["verdict"] == "PASS"
    no_beta = [r for r in rooms if r["category"] != "Beta"]
    for fn in (fa.evaluate_gates, fa.evaluate_gate, fa.evaluate_gate_a2):
        with pytest.raises(fa.AuditError, match="Beta"):
            fn(no_beta, categories=SYN_CATS)                         # a registered category lost its rooms
        with pytest.raises(TypeError):
            fn(rooms)                                                # never derived from the rooms
    with pytest.raises(fa.AuditError, match="Gamma"):
        fa.evaluate_gates(rooms, categories=["Alpha", "Beta"])      # a room of an unregistered category
    for cats in ([], None, ["Alpha", "Beta", "Gamma", "Beta"], "Alpha"):
        with pytest.raises(fa.AuditError):
            fa.evaluate_gates(rooms, categories=cats)
    # every registered category receives its leave-out block, in registration order
    g = fa.evaluate_gates(rooms, categories=["Gamma", "Alpha", "Beta"])
    for gate in ("registered", "a2"):
        assert list(g[gate]["loco"]) == ["Gamma", "Alpha", "Beta"]
        assert list(g[gate]["diagnostics"]["loco_argmax"]) == ["Gamma", "Alpha", "Beta"]
        assert g[gate]["verdict"] == "PASS"


def test_run_refuses_when_a_registered_category_loses_its_rooms(tmp_path, monkeypatch):
    """A room drop after loading (simulated) can never silently remove a leave-out check."""
    ar, split, inp = _two_room_inputs(tmp_path / "t")
    real = fa.load_inputs

    def dropping(*a, **k):
        doc, sha, rooms = real(*a, **k)
        return doc, sha, [r for r in rooms if r["category"] != "Beta"]

    monkeypatch.setattr(fa, "load_inputs", dropping)
    out = tmp_path / "o.json"
    with pytest.raises(fa.AuditError, match="Beta"):
        _api(inp, ar, out, split, **_self_binding(inp))
    assert not out.exists()
