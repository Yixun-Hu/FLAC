"""Config-integrity tests for the exp_11 fa_orbit arm manifests.

The four arm configs ``FLAC_AR_BF_C{4L,8,16,32}.json`` are derived from the
exp_07 from-scratch fa manifest ``FLAC_AR_BF.json`` and must differ from it in
EXACTLY two leaf groups (plan Rev 3 §10): ``gradient_checkpointing``
``true -> false`` in BOTH ``ViTCoordinates`` conditioners (the Rev 3 fast
recipe drops ViT grad-ckpt), and ``training.frame_avg_angles`` for C8/C16/C32
ONLY -- C4L is the bridge control, angle-identical to the source manifest. Any
other differing leaf is a silent recipe change that would confound the sweep,
so the deep-diff test fails on it and names the offending path. The orbit is
additionally checked geometrically: uniform angles from 0.0 whose panorama
column shift ``a * W / 360`` (W = 512) is an exact integer AND a multiple of
the 16-px ViT patch. Pure CPU/JSON: no torch, no GPU, no network.
"""
import json
import os
import re

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_BF_CONFIG = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_07_fa_scratch_claude",
    "FLAC_AR_BF.json",
)
_EXP11_DIR = os.path.join(
    _REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude"
)

ARMS = ("C4L", "C8", "C16", "C32")
IMG_W = 512      # panorama width in columns
PATCH = 16       # DINOv3 patch size (px)

_MISSING = "<missing>"


def _load(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _arm_path(arm: str) -> str:
    return os.path.join(_EXP11_DIR, f"FLAC_AR_BF_{arm}.json")


def _n_from_name(arm: str) -> int:
    """Orbit size encoded in the arm/file name ('C4L' -> 4, 'C16' -> 16)."""
    m = re.fullmatch(r"C(\d+)L?", arm)
    assert m is not None, f"cannot read an orbit size from arm name {arm!r}"
    return int(m.group(1))


def _orbit(n: int) -> list:
    """The uniform Cn orbit in degrees, as floats starting at 0.0."""
    return [k * 360.0 / n for k in range(n)]


def _deep_diff(a, b, path: str = "") -> dict:
    """Recursive leaf-level diff: ``{dotted_path: (a_value, b_value)}``.

    Dicts recurse per key (a key present on one side only is itself a leaf
    difference); equal-length lists recurse per index, unequal-length lists are
    reported as one leaf at the list's own path; everything else compares by
    type and value (so ``1`` vs ``1.0`` counts as a difference)."""
    out = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out[sub] = (a.get(key, _MISSING), b.get(key, _MISSING))
            else:
                out.update(_deep_diff(a[key], b[key], sub))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[path] = (a, b)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.update(_deep_diff(x, y, f"{path}[{i}]"))
    elif type(a) is not type(b) or a != b:
        out[path] = (a, b)
    return out


def _vit_gc_paths(cfg: dict) -> list:
    """Dotted paths of the ``gradient_checkpointing`` leaf of every
    ``ViTCoordinates`` conditioner in ``cfg`` (exp_07 BF has exactly two)."""
    entries = cfg["model"]["conditioning"]["configs"]
    paths = [
        f"model.conditioning.configs[{i}].config.gradient_checkpointing"
        for i, e in enumerate(entries)
        if e.get("type") == "ViTCoordinates"
    ]
    assert len(paths) == 2, f"expected 2 ViTCoordinates conditioners, got {len(paths)}"
    return paths


# --------------------------------------------------------------------------- #
# 1. allowed-diff set vs the exp_07 source manifest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ARMS)
def test_allowed_diff_leaves(arm):
    bf = _load(_BF_CONFIG)
    cfg = _load(_arm_path(arm))

    expected = {p: (True, False) for p in _vit_gc_paths(bf)}
    if arm != "C4L":
        expected["training.frame_avg_angles"] = (
            bf["training"]["frame_avg_angles"], _orbit(_n_from_name(arm))
        )

    diff = _deep_diff(bf, cfg)

    unexpected = sorted(set(diff) - set(expected))
    assert not unexpected, (
        f"{arm}: unexpected differing leaf(s) vs exp_07 FLAC_AR_BF.json: "
        + "; ".join(f"{p}: {diff[p][0]!r} -> {diff[p][1]!r}" for p in unexpected)
    )
    absent = sorted(set(expected) - set(diff))
    assert not absent, f"{arm}: expected change(s) missing: {absent}"
    for path, values in expected.items():
        assert diff[path] == values, (
            f"{arm}: {path} changed {diff[path][0]!r} -> {diff[path][1]!r}, "
            f"expected {values[0]!r} -> {values[1]!r}"
        )


# --------------------------------------------------------------------------- #
# 2. orbit geometry: uniform from 0.0, exact + patch-aligned column shifts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ARMS)
def test_angle_lists(arm):
    angles = _load(_arm_path(arm))["training"]["frame_avg_angles"]
    n = len(angles)
    assert n >= 1, f"{arm}: empty orbit"
    for a in angles:
        assert isinstance(a, float), f"{arm}: angle {a!r} is {type(a).__name__}, not float"
    assert angles[0] == 0.0, f"{arm}: orbit must start at exactly 0.0, got {angles[0]!r}"
    assert all(angles[i] < angles[i + 1] for i in range(n - 1)), (
        f"{arm}: angles not strictly increasing: {angles}"
    )
    step = 360.0 / n
    for k, a in enumerate(angles):
        assert abs(a - k * step) < 1e-9, f"{arm}: angle[{k}] = {a!r} != {k * step}"
        shift = a * IMG_W / 360.0
        assert float(shift).is_integer(), f"{arm}: {a} deg -> fractional shift {shift}"
        assert int(shift) % PATCH == 0, (
            f"{arm}: {a} deg -> shift {int(shift)} px, not a multiple of {PATCH}"
        )


# --------------------------------------------------------------------------- #
# 3. orbit size matches the file name
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ARMS)
def test_n_matches_filename(arm):
    angles = _load(_arm_path(arm))["training"]["frame_avg_angles"]
    n = _n_from_name(arm)
    assert len(angles) == n, (
        f"{os.path.basename(_arm_path(arm))} declares C{n} but carries "
        f"{len(angles)} angles: {angles}"
    )


# --------------------------------------------------------------------------- #
# 4. C4L bridge control: training block identical to exp_07 BF
# --------------------------------------------------------------------------- #
def test_c4l_bridge_identity():
    bf_train = _load(_BF_CONFIG)["training"]
    c4l_train = _load(_arm_path("C4L"))["training"]
    assert set(bf_train) == set(c4l_train), (
        "C4L training keys differ from BF: "
        f"{sorted(set(bf_train) ^ set(c4l_train))}"
    )
    diff = _deep_diff(bf_train, c4l_train)
    assert not diff, f"C4L training block differs from BF at: {sorted(diff)}"
