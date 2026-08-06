"""Config-integrity tests for the exp_11 fa_orbit arm manifests.

The arm configs ``FLAC_AR_BF_C{4L,8,16,32}.json`` plus the P0 attribution
control ``FLAC_AR_BF_FA1.json`` are derived from the exp_07 from-scratch fa
manifest ``FLAC_AR_BF.json`` and must differ from it in EXACTLY ONE leaf group:
``training.frame_avg_angles`` (FA1/C8/C16/C32 only) -- and C4L must differ in
NOTHING at all, because it is the bridge control and now re-runs the exp_07 B-F
recipe verbatim.

ViT gradient checkpointing stays TRUE in every arm (Yixun, post-P0): P0 measured
that the no-ckpt recipe is INFEASIBLE for C8 and richer orbits -- OOM even at
micro-batch 8 (45,457 MiB) -- while the checkpointed recipe peaks at ~9.4 GB, so
the sweep runs uniformly checkpointed and the ONLY delta between arms is the
averaging orbit. FA1's single-angle orbit ``[0.0]`` keeps the fa dispatch and the
cylindrical pose path but runs exactly ONE ViT pass, so it is the profiling
baseline the per-orbit-pass cost is fitted against (round-2 re-review B6); its
spacing check is vacuous at n = 1. Any
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
_CANON_CONFIG = os.path.join(
    _REPO_ROOT, "src", "configs", "model_configs", "FLAC", "AR", "FLAC_AR.json"
)
_VANCKPT_CONFIG = os.path.join(_EXP11_DIR, "FLAC_AR_VANCKPT.json")

ARMS = ("FA1", "C4L", "C8", "C16", "C32")
IMG_W = 512      # panorama width in columns
PATCH = 16       # DINOv3 patch size (px)

_MISSING = "<missing>"


def _no_duplicate_keys(pairs: list) -> dict:
    """``object_pairs_hook`` rejecting duplicate keys (plain ``json.load`` is
    silently last-wins, which would let a shadowed leaf hide a recipe change)."""
    keys = [k for k, _ in pairs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"duplicate JSON key(s): {dups}"
    return dict(pairs)


def _reject_constant(name: str):
    """``parse_constant`` hook: NaN/Infinity are not valid JSON in a manifest."""
    raise AssertionError(f"non-standard JSON constant {name!r} in config")


def _load(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(
            fh, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant
        )


def _arm_path(arm: str) -> str:
    return os.path.join(_EXP11_DIR, f"FLAC_AR_BF_{arm}.json")


def _n_from_name(arm: str) -> int:
    """Orbit size encoded in the arm/file name ('C4L' -> 4, 'C16' -> 16, 'FA1' -> 1).

    FA1 is the P0 attribution control (re-review B6): fa_invariant with a
    single-angle orbit, i.e. the cylindrical pose path plus exactly ONE ViT pass
    (``yaw_rotation.invariant_conditioning`` returns the base pass when
    ``len(angles) == 1``), so FA1 -> C4L -> C8 isolates the per-orbit-pass cost."""
    m = re.fullmatch(r"(?:C(\d+)L?|FA(\d+))", arm)
    assert m is not None, f"cannot read an orbit size from arm name {arm!r}"
    return int(m.group(1) or m.group(2))


def _orbit(n: int) -> list:
    """The uniform Cn orbit in degrees, as floats starting at 0.0 (n = 1 -> [0.0])."""
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


def _vit_gc_leaves(cfg: dict) -> list:
    """``(dotted_path, value)`` of the ``gradient_checkpointing`` leaf of every
    ``ViTCoordinates`` conditioner in ``cfg`` (exp_07 BF has exactly two)."""
    entries = cfg["model"]["conditioning"]["configs"]
    leaves = [
        (f"model.conditioning.configs[{i}].config.gradient_checkpointing",
         e["config"]["gradient_checkpointing"])
        for i, e in enumerate(entries)
        if e.get("type") == "ViTCoordinates"
    ]
    assert len(leaves) == 2, f"expected 2 ViTCoordinates conditioners, got {len(leaves)}"
    return leaves


def _assert_allowed_diff(arm: str, bf: dict, cfg: dict) -> None:
    """Fail unless ``cfg`` differs from the exp_07 manifest ``bf`` in exactly the
    allowed leaves: ``training.frame_avg_angles`` for every arm except C4L, which
    must be leaf-for-leaf identical to the source manifest. Leaf VALUES are
    checked strictly: both checkpointing flags by identity (``is True``, so a
    truthy ``1`` is rejected) and the orbit through :func:`_deep_diff` (so an int
    ``45`` is rejected)."""
    gc_base, gc_arm = _vit_gc_leaves(bf), _vit_gc_leaves(cfg)
    expected_paths = set()
    if arm != "C4L":
        expected_paths.add("training.frame_avg_angles")

    diff = _deep_diff(bf, cfg)

    unexpected = sorted(set(diff) - expected_paths)
    assert not unexpected, (
        f"{arm}: unexpected differing leaf(s) vs exp_07 FLAC_AR_BF.json: "
        + "; ".join(f"{p}: {diff[p][0]!r} -> {diff[p][1]!r}" for p in unexpected)
    )
    absent = sorted(expected_paths - set(diff))
    assert not absent, f"{arm}: expected change(s) missing: {absent}"

    for (path, v_base), (path_arm, v_arm) in zip(gc_base, gc_arm):
        assert path == path_arm, f"{arm}: ViTCoordinates entries moved ({path_arm})"
        assert v_base is True, f"BF {path} is {v_base!r}, expected literal true"
        assert v_arm is True, (
            f"{arm}: {path} is {v_arm!r}, expected literal true — every arm runs the "
            "checkpointed recipe (no-ckpt is OOM-infeasible for C8+, P0 2026-08-05)"
        )

    if arm != "C4L":
        want = _orbit(_n_from_name(arm))
        got = diff["training.frame_avg_angles"][1]
        angle_diff = _deep_diff(want, got, "training.frame_avg_angles")
        assert not angle_diff, (
            f"{arm}: orbit mismatch at {sorted(angle_diff)}; got {got}, want {want}"
        )


# --------------------------------------------------------------------------- #
# 1. allowed-diff set vs the exp_07 source manifest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ARMS)
def test_allowed_diff_leaves(arm):
    _assert_allowed_diff(arm, _load(_BF_CONFIG), _load(_arm_path(arm)))


def test_non_boolean_gc_leaf_is_rejected():
    """Regression for the loose-comparison bug, re-pointed after the grad-ckpt
    pivot: ``1`` is truthy but is not ``True``, and a flipped ``False`` is a
    silent recipe change; neither may pass. Mutations live in memory only -- no
    temp config is written to the exp folder."""
    bf = _load(_BF_CONFIG)
    _assert_allowed_diff("C8", bf, _load(_arm_path("C8")))  # unmutated => passes
    for bad in (1, 1.0, False, 0):
        cfg = _load(_arm_path("C8"))
        entry = next(
            e for e in cfg["model"]["conditioning"]["configs"]
            if e.get("type") == "ViTCoordinates"
        )
        entry["config"]["gradient_checkpointing"] = bad
        with pytest.raises(AssertionError):
            _assert_allowed_diff("C8", bf, cfg)


def test_vanckpt_adds_only_grad_checkpointing():
    """The P0 vanilla cell must be the canonical recipe PLUS checkpointing.

    Post-pivot every P0 cell is checkpointed, so VAN can no longer run the
    canonical manifest (whose ViT blocks carry no ``gradient_checkpointing``
    leaf at all — the launcher's gate demands the key exists and is literally
    true). ``FLAC_AR_VANCKPT.json`` is that manifest with EXACTLY the two leaves
    added: any other differing leaf would make the vanilla baseline a different
    model and silently bias the FA1-vs-VAN contrast."""
    canon = _load(_CANON_CONFIG)
    vanckpt = _load(_VANCKPT_CONFIG)
    diff = _deep_diff(canon, vanckpt)
    expected = {p for p, _ in _vit_gc_leaves(vanckpt)}
    assert set(diff) == expected, (
        "VANCKPT must differ from the canonical config in exactly the two ViT "
        f"gradient_checkpointing leaves; got {sorted(diff)}"
    )
    for path in expected:
        was, now = diff[path]
        assert was is _MISSING or was == "<missing>", (
            f"{path}: canonical already carried {was!r}")
        assert now is True, f"{path}: VANCKPT has {now!r}, expected literal true"
    # the vanilla baseline must stay vanilla: no frame-averaging keys sneak in
    training = vanckpt["training"]
    assert "cond_method" not in training and "frame_avg_angles" not in training, (
        "VANCKPT carries frame-averaging keys — it would no longer be the vanilla baseline"
    )


def test_c4l_is_byte_identical_to_exp07_bf():
    """The bridge arm re-runs the exp_07 B-F recipe verbatim, so its manifest is
    the SAME BYTES — not merely the same parsed object."""
    with open(_BF_CONFIG, "rb") as a, open(_arm_path("C4L"), "rb") as b:
        assert a.read() == b.read(), "C4L must be byte-identical to exp_07 FLAC_AR_BF.json"


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
