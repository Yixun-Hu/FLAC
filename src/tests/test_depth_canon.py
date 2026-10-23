"""exp_25 depth_canon -- tests for the deepest-pixel yaw canonicalization operator
added to ``src.data.yaw_rotation`` (plan_depth_canon.md §2.1, §6 tests 1-9).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md §2.1/§6/§8.

Fixtures are synthetic scalar depth maps (seeded ``torch.Generator``) pushed through
the REAL ``AR_md.convert_equirect_to_camera_coord``, so no load-bearing test needs
the dataset. Planted maxima are ``PEAK = 8.0``, a power of two: ``-8*sin(phi)`` and
its division by ``-sin(phi)`` are exact, so a planted raw tie is a bitwise proxy tie
on every row. The background lies in ``[0.5, 7.5)``, strictly below the peak.
"""
import functools
import hashlib
import importlib.util
import math
import os

import numpy as np
import pytest
import torch

from src.data import yaw_rotation as yr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AR_MD_PATH = os.path.join(
    _REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata", "AR_md.py")
PEAK = 8.0


@functools.lru_cache(maxsize=1)
def _ar_md():
    """The release metadata module, loaded by path exactly as ``src/data/dataset.py`` does."""
    spec = importlib.util.spec_from_file_location("ar_md_for_depth_canon_tests", _AR_MD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scalar_depth(H, W, seed, period=None, peaks=()):
    """[H, W] float64 depth in [0.5, 7.5) plus planted PEAKs at ``peaks`` (row, col).

    With ``period`` the image is exactly W-periodic (each planted peak is replicated
    every ``period`` columns), i.e. it has a nontrivial cyclic stabilizer."""
    g = torch.Generator().manual_seed(seed)
    p = W if period is None else period
    d = (0.5 + 7.0 * torch.rand(H, p, generator=g, dtype=torch.float64)).repeat(1, W // p)
    for h, w in peaks:
        for k in range(W // p):
            d[h, (w + k * p) % W] = PEAK
    return d


def _cloud(d):
    """The metadata ``depth`` field exactly as AR_md builds it: a [3, H, W] permuted view."""
    H, W = d.shape
    return _ar_md().convert_equirect_to_camera_coord(d, H, W).permute(2, 0, 1)


def _roll(md, d, W):
    """roll_d: the reviewed yaw operator by ``d`` columns (depth AND poses rotate)."""
    return yr.rotate_scene_metadata(md, yr.offsets_to_radians([d], W)[0], W)


# --------------------------------------------------------------------------- #
# 1. recover_depth_proxy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,W", [(256, 512), (16, 32)])
def test_proxy_recovers_scalar_depth_through_real_converter(H, W):
    d = _scalar_depth(H, W, seed=1)
    D = yr.recover_depth_proxy(_cloud(d))
    assert D.shape == (H, W) and D.dtype == torch.float64
    assert float((D - d).abs().max()) <= 1e-14


def test_proxy_denominator_is_ar_md_sin_phi_bitwise():
    """convert(ones) stores z = -1 * sin(phi_row) exactly, so z / (-s') == 1 bitwise
    on every pixel iff the proxy's per-row sine s' IS AR_md's sine."""
    ones = torch.ones(256, 512, dtype=torch.float64)
    assert torch.equal(yr.recover_depth_proxy(_cloud(ones)), ones)


@pytest.mark.parametrize("dj", [1, 64, 128, 300, 511])
def test_proxy_is_bitwise_yaw_stable_under_rotate_scene_metadata(dj):
    W = 512
    md = {"depth": _cloud(_scalar_depth(256, W, seed=2)), "source": torch.tensor([1.0, 2.0, 0.5])}
    D0 = yr.recover_depth_proxy(md["depth"])
    Dr = yr.recover_depth_proxy(_roll(md, dj, W)["depth"])
    assert torch.equal(torch.roll(Dr, shifts=-dj, dims=1), D0)


def _bad_depth(name):
    ok = _cloud(_scalar_depth(4, 8, seed=3))
    bad = ok.clone()
    if name == "nan":
        bad[0, 1, 2] = float("nan")
    elif name == "inf":
        bad[2, 0, 0] = float("inf")
    return {
        "2d": ok[2], "4ch": torch.cat([ok, ok[:1]]), "4d": ok[None],
        "odd_h": _cloud(_scalar_depth(5, 8, seed=3)), "zero_h": ok[:, :0], "zero_w": ok[:, :, :0],
        "nan": bad, "inf": bad, "int": ok.to(torch.int64), "bool": ok > 1.0, "list": ok.tolist(),
    }[name]


@pytest.mark.parametrize("name", ["2d", "4ch", "4d", "odd_h", "zero_h", "zero_w",
                                  "nan", "inf", "int", "bool", "list"])
def test_proxy_rejects_contract_violations(name):
    with pytest.raises(ValueError):
        yr.recover_depth_proxy(_bad_depth(name))


# --------------------------------------------------------------------------- #
# 2. column_profile
# --------------------------------------------------------------------------- #
def test_column_profile_is_the_per_column_max():
    D = yr.recover_depth_proxy(_cloud(_scalar_depth(256, 512, seed=4, peaks=[(10, 20)])))
    P = yr.column_profile(D)
    assert P.shape == (512,)
    assert torch.equal(P, D.max(dim=0).values)
    assert int(P.argmax()) == 20 and float(P.max()) == PEAK


@pytest.mark.parametrize("bad", [torch.zeros(3, 4, 5), torch.zeros(4), torch.zeros(0, 4),
                                 torch.zeros(2, 4, dtype=torch.int64)])
def test_column_profile_rejects_non_proxy_input(bad):
    with pytest.raises(ValueError):
        yr.column_profile(bad)


# --------------------------------------------------------------------------- #
# fixtures for the selector: name -> (H, W, _scalar_depth kwargs, source, context, class)
# --------------------------------------------------------------------------- #
def _pol(r, az, z):
    return (r * math.cos(az), r * math.sin(az), z)


_SRC = (1.3, -0.7, 0.2)                          # generic target source
_CTX = ((0.9, 1.1, -0.3), (-1.2, 0.4, 0.1))      # generic context rows
_DEG_SRC = (0.0, 0.0, 0.4)                       # on the rotation (z) axis: azimuth undefined
_DEG_CTX = ((0.0, 0.0, 0.5), (1e-8, 0.0, -0.2))  # every context row degenerate (r < 1e-6)
_EPS = 1e-3
FIXTURES = {
    "unique": (256, 512, dict(seed=10, peaks=[(37, 100)]), _SRC, _CTX, "none"),
    "same_column": (256, 512, dict(seed=11, peaks=[(20, 300), (200, 300)]), _SRC, _CTX,
                    "same_column"),
    "row_tie_sep1": (256, 512, dict(seed=12, peaks=[(50, 7), (50, 8)]), _SRC, _CTX,
                     "cross_column_image"),
    "row_tie_sep64_wrap": (256, 512, dict(seed=13, peaks=[(50, 480), (50, 32)]), _SRC, _CTX,
                           "cross_column_image"),
    "row_tie_sep256": (256, 512, dict(seed=14, peaks=[(90, 0), (90, 256)]), _SRC, _CTX,
                       "cross_column_image"),
    "cross_row_tie": (256, 512, dict(seed=15, peaks=[(30, 10), (220, 400)]), _SRC, _CTX,
                      "cross_column_image"),
    "small_unique": (16, 32, dict(seed=20, peaks=[(3, 9)]), _SRC, _CTX, "none"),
    "small_same_column": (16, 32, dict(seed=21, peaks=[(2, 5), (13, 5)]), _SRC, _CTX,
                          "same_column"),
    "small_row_tie_sep1_wrap": (16, 32, dict(seed=22, peaks=[(4, 31), (4, 0)]), _SRC, _CTX,
                                "cross_column_image"),
    "small_row_tie_sep8": (16, 32, dict(seed=23, peaks=[(9, 3), (9, 11)]), _SRC, _CTX,
                           "cross_column_image"),
    "small_row_tie_sep16": (16, 32, dict(seed=24, peaks=[(6, 20), (6, 4)]), _SRC, _CTX,
                            "cross_column_image"),
}


def _names(*classes):
    return [n for n, f in FIXTURES.items() if f[5] in classes]


def _md(d, source=_SRC, context=_CTX):
    """A metadata dict shaped like AR_md's (context_poses_vit IS context_poses there)."""
    md = {"depth": _cloud(d), "scene": "synthetic_room", "padding_mask": torch.ones(8, dtype=torch.bool)}
    if source is not None:
        s = torch.tensor(source, dtype=torch.float32)
        md["source"], md["source_vit"] = s, s[None]
    if context is not None:
        c = torch.tensor(context, dtype=torch.float32)
        md["context_poses"] = md["context_poses_vit"] = c
        md["context_audio"] = torch.arange(c.shape[0] * 8, dtype=torch.float32).reshape(-1, 1, 8)
    return md


def _fixture(name):
    H, W, kw, source, context, cls = FIXTURES[name]
    d = _scalar_depth(H, W, **kw)
    return d, _md(d, source, context), cls


def _select(md, D=None):
    return yr.canonical_column(yr.recover_depth_proxy(md["depth"]) if D is None else D)


def _outcome(md, D=None):
    """(j*, class), or ('stabilized', candidates) when the selector fails closed."""
    try:
        return _select(md, D)
    except yr.StabilizedTieError as err:
        return "stabilized", err.candidates


def _max_columns(X):
    P = yr.column_profile(X)
    return torch.nonzero(P == P.max()).flatten().tolist()


def _oracle_image_winners(D, cands):
    """Independent oracle: Python list comparison of the column-major rooted images."""
    keys = {c: torch.roll(D, shifts=-c, dims=1).t().reshape(-1).tolist() for c in cands}
    best = max(keys.values())
    return [c for c in cands if keys[c] == best]


# --------------------------------------------------------------------------- #
# 3. canonical_column -- image-level classes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [n for n in FIXTURES if FIXTURES[n][5] in
                                  ("none", "same_column", "cross_column_image")])
def test_canonical_column_image_level_classes(name):
    _, md, cls = _fixture(name)
    D = yr.recover_depth_proxy(md["depth"])
    j, tie_class = _select(md)
    cands = _max_columns(D)
    assert tie_class == cls and isinstance(j, int)
    assert cands == sorted({w for _, w in FIXTURES[name][2]["peaks"]})
    assert [j] == (_oracle_image_winners(D, cands) if cls == "cross_column_image" else cands)


def test_periodic_image_without_any_pose_key_fails_closed():
    W = 512
    md = _md(_scalar_depth(256, W, seed=30, period=256, peaks=[(40, 70)]), None, None)
    with pytest.raises(yr.StabilizedTieError) as info:
        _select(md)
    assert isinstance(info.value, RuntimeError) and info.value.candidates == (70, 326)
    import pickle
    assert pickle.loads(pickle.dumps(info.value)).candidates == (70, 326)


@pytest.mark.parametrize("args", [
    (torch.rand(1, 4, 8, dtype=torch.float64),), (torch.ones(4, 8, dtype=torch.int64),),
    (torch.tensor([[1.0, float("nan")], [0.5, 0.5]]),),
], ids=["3d", "int", "nan"])
def test_canonical_column_rejects_contract_violations(args):
    with pytest.raises(ValueError):
        yr.canonical_column(*args)


# --------------------------------------------------------------------------- #
# 8. Option-A negative control (literal first max column; rejected by Yixun, D1)
# --------------------------------------------------------------------------- #
def _option_a(md):
    P = yr.column_profile(yr.recover_depth_proxy(md["depth"]))
    return int(torch.nonzero(P == P.max())[0, 0])


@pytest.mark.xfail(strict=True, reason="Option A (first max column) is not equivariant "
                                       "on a cross-column tie -- the D1 negative control")
def test_option_a_first_index_rule_breaks_equivariance():
    _, md, _ = _fixture("row_tie_sep64_wrap")
    j0 = _option_a(md)
    for d in range(512):
        assert _option_a(_roll(md, d, 512)) == (j0 + d) % 512


# --------------------------------------------------------------------------- #
# 9. raw scalar depth vs proxy: same candidate set, same selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(FIXTURES))
def test_raw_and_proxy_agree_on_candidates_and_selection(name):
    d, md, _ = _fixture(name)
    D = yr.recover_depth_proxy(md["depth"])
    assert _max_columns(d) == _max_columns(D)
    assert _outcome(md, d) == _outcome(md, D)


# --------------------------------------------------------------------------- #
# 5. canonical_shift
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W", [32, 512])
def test_canonical_shift_moves_j_star_to_column_zero(W):
    for j in range(W):
        dj = yr.canonical_shift(j, W)
        assert dj == (W - j) % W and 0 <= dj < W and (j + dj) % W == 0
        assert (dj == 0) == (j == 0)


@pytest.mark.parametrize("args", [(-1, 8), (8, 8), (True, 8), (1.0, 8), (1, 0), (1, True), (0, -4)])
def test_canonical_shift_rejects_out_of_contract_arguments(args):
    with pytest.raises(ValueError):
        yr.canonical_shift(*args)


# --------------------------------------------------------------------------- #
# 6. canonicalize_scene_metadata
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["unique", "same_column", "row_tie_sep64_wrap", "cross_row_tie",
                                  "small_row_tie_sep1_wrap"])
def test_canonicalize_scene_metadata_contract(name):
    _, md, cls = _fixture(name)
    W = md["depth"].shape[-1]
    ids = {k: id(v) for k, v in md.items()}
    snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}
    out, dj, tie_class = yr.canonicalize_scene_metadata(md, W)
    j0, c0 = _select(md)
    assert isinstance(dj, int) and (dj, tie_class) == (yr.canonical_shift(j0, W), c0) == (dj, cls)
    ref = yr.rotate_scene_metadata(md, yr.offsets_to_radians([dj], W)[0], W)
    assert out.keys() == md.keys() == ref.keys()
    for k, v in md.items():
        if torch.is_tensor(v):
            assert torch.equal(out[k], ref[k]) and (out[k].dtype, out[k].device) == (v.dtype, v.device)
    for k in ("context_audio", "scene", "padding_mask"):
        assert out[k] is md[k]
    assert _select(out) == (0, tie_class)                                  # post-condition
    out2, dj2, tc2 = yr.canonicalize_scene_metadata(out, W)                # idempotent
    assert (dj2, tc2) == (0, tie_class)
    assert all(torch.equal(out2[k], v) for k, v in out.items() if torch.is_tensor(v))
    assert {k: id(v) for k, v in md.items()} == ids                        # input untouched
    assert all(torch.equal(v, snapshot[k]) if torch.is_tensor(v) else v == snapshot[k]
               for k, v in md.items())


@pytest.mark.parametrize("field,value", [
    ("source", torch.zeros(1, 3)), ("source", torch.zeros(4)),
    ("source", torch.zeros(3, dtype=torch.int64)), ("source", torch.tensor([float("nan"), 0.0, 0.0])),
    ("source_vit", torch.zeros(3)), ("source_vit", torch.zeros(2, 3)),
    ("context_poses", torch.zeros(2)), ("context_poses", torch.zeros(2, 2)),
    ("context_poses", torch.zeros(0, 3)), ("context_poses", torch.zeros(2, 3, 1)),
    ("context_poses", torch.tensor([[0.0, float("inf"), 0.0], [1.0, 1.0, 1.0]])),
    ("context_poses_vit", torch.zeros(3, 3)), ("context_poses_vit", torch.zeros(2, 3, dtype=torch.int32)),
    ("depth", None), ("depth", _cloud(_scalar_depth(16, 16, seed=1))),
    ("depth", torch.zeros(3, 16, 32, dtype=torch.int64)),
])
def test_canonicalize_rejects_contract_violations(field, value):
    md = dict(_fixture("small_unique")[1])
    if value is None:
        del md[field]
    else:
        md[field] = value
    with pytest.raises(ValueError):
        yr.canonicalize_scene_metadata(md, 32)


@pytest.mark.parametrize("img_w", [0, -32, True, 32.0, "32"])
def test_canonicalize_rejects_bad_img_w(img_w):
    with pytest.raises(ValueError):
        yr.canonicalize_scene_metadata(_fixture("small_unique")[1], img_w)
    with pytest.raises(ValueError):
        yr.canonicalize_scene_metadata([_fixture("small_unique")[1]], 32)


@pytest.mark.parametrize("lie", [lambda j, c: (j + 1, c), lambda j, c: (j, "same_column")])
def test_canonicalize_post_condition_fails_closed(monkeypatch, lie):
    _, md, _ = _fixture("small_unique")
    real, calls = yr.canonical_column, []

    def fake(D, source=None, context=None):
        calls.append(1)
        j, c = real(D, source, context)
        return (j, c) if len(calls) == 1 else lie(j, c)   # the re-check on the output lies
    monkeypatch.setattr(yr, "canonical_column", fake)
    with pytest.raises(RuntimeError, match="post-condition"):
        yr.canonicalize_scene_metadata(md, 32)


# --------------------------------------------------------------------------- #
# 7. invariance of the canonical output + canonical_depth_sha256 (Tier-1 hash)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,variant", [
    ("row_tie_sep1", "generic"), ("row_tie_sep64_wrap", "centre"), ("row_tie_sep256", "minus_pi"),
    ("cross_row_tie", "plus_pi"), ("row_tie_sep64_wrap", "ctx_r_below_eps"),
    ("row_tie_sep256", "pi_minus_eps"), ("cross_row_tie", "degenerate")])
def test_canonical_output_is_invariant_under_rotation(name, variant):
    md, cls = _fixture_with(name, variant)
    ref, _, ref_class = yr.canonicalize_scene_metadata(md, 512)
    ref_D, ref_hash = yr.recover_depth_proxy(ref["depth"]), yr.canonical_depth_sha256(ref["depth"])
    assert ref_class == cls
    for d in (1, 64, 128, 255, 256, 384, 511):
        out, _, tie_class = yr.canonicalize_scene_metadata(_roll(md, d, 512), 512)
        assert tie_class == cls
        assert torch.equal(yr.recover_depth_proxy(out["depth"]), ref_D)
        assert yr.canonical_depth_sha256(out["depth"]) == ref_hash
        for k in ("depth",) + yr.POSE_KEYS:
            assert torch.allclose(out[k], ref[k], rtol=1e-5, atol=1e-6), (d, k)


def test_canonical_depth_sha256_hashes_the_proxy_bytes():
    md = _fixture("small_unique")[1]
    D = yr.recover_depth_proxy(md["depth"])
    assert yr.canonical_depth_sha256(md["depth"]) == \
        hashlib.sha256(D.contiguous().numpy().tobytes()).hexdigest()
    assert yr.canonical_depth_sha256(md["depth"]) != \
        yr.canonical_depth_sha256(_fixture("small_same_column")[1]["depth"])


@pytest.mark.parametrize("bad", [torch.zeros(3, 4, 8), torch.zeros(3, 4, 8, dtype=torch.float16),
                                 torch.zeros(3, 4, 8, dtype=torch.int64), "depth"])
def test_canonical_depth_sha256_requires_float64(bad):
    with pytest.raises(ValueError):
        yr.canonical_depth_sha256(bad)


def test_canonical_depth_sha256_ignores_the_sign_of_zero_depth():
    """Equal proxies must hash equal. AR_md stores z = -0.0 at zero depth on one
    hemisphere and a yaw rotation (z' = 0*x + 0*y + 1*z) writes +0.0 there: the
    proxies compare equal but their BYTES differ, so the hash normalizes signed zeros."""
    d = _scalar_depth(16, 32, seed=70, peaks=[(3, 9)])
    d[12, 7] = d[2, 20] = d[9, 0] = 0.0          # zero depth on both hemispheres
    md = _md(d)
    raw = md["depth"]
    assert bool(((raw[2] == 0) & torch.signbit(raw[2])).any())
    same = yr.rotate_scene_metadata(md, 0.0, 32)["depth"]          # identity yaw
    assert torch.equal(yr.recover_depth_proxy(same), yr.recover_depth_proxy(raw))
    assert yr.canonical_depth_sha256(same) == yr.canonical_depth_sha256(raw)
    ref = yr.canonical_depth_sha256(yr.canonicalize_scene_metadata(md, 32)[0]["depth"])
    for dd in range(32):
        out = yr.canonicalize_scene_metadata(_roll(md, dd, 32), 32)[0]
        assert yr.canonical_depth_sha256(out["depth"]) == ref, dd


# --------------------------------------------------------------------------- #
# optional real-map check (NOT load-bearing; duplicates the synthetic tests above)
# --------------------------------------------------------------------------- #
_AR_DEPTH_ROOT = os.path.join(_REPO_ROOT, "AcousticRooms", "depth_map")


@pytest.mark.skipif(not os.path.isdir(_AR_DEPTH_ROOT),
                    reason="AcousticRooms/depth_map absent (optional duplicate of synthetic tests)")
def test_real_ar_depth_map_proxy_selection_and_invariance():
    path = _AR_DEPTH_ROOT                               # <scene>/<id>/<receiver>.npy, first of each
    for _ in range(2):
        path = os.path.join(path, sorted(e for e in os.listdir(path)
                                         if os.path.isdir(os.path.join(path, e)))[0])
    path = os.path.join(path, sorted(e for e in os.listdir(path) if e.endswith(".npy"))[0])
    d = torch.from_numpy(np.load(path))
    md = _md(d)
    D = yr.recover_depth_proxy(md["depth"])
    assert d.shape == (256, 512) and float((D - d).abs().max()) <= 1e-14
    assert _max_columns(d) == _max_columns(D) and _outcome(md, d) == _outcome(md, D)
    ref = yr.canonicalize_scene_metadata(md, 512)[0]
    for k in (1, 128, 300):
        out = yr.canonicalize_scene_metadata(_roll(md, k, 512), 512)[0]
        assert yr.canonical_depth_sha256(out["depth"]) == yr.canonical_depth_sha256(ref["depth"])
        assert all(torch.allclose(out[q], ref[q], rtol=1e-5, atol=1e-6) for q in yr.POSE_KEYS)


# --------------------------------------------------------------------------- #
# code-review r1 fix (BLOCKING): poses never select a column; any tie the rooted-
# image key leaves (an exactly periodic proxy) fails closed, whatever the poses
# --------------------------------------------------------------------------- #
_R_BELOW = float(torch.nextafter(torch.tensor(1e-6), torch.tensor(0.0)))  # float32 1e-6 - ulp
_R_ABOVE = float(torch.nextafter(torch.tensor(1e-6), torch.tensor(1.0)))  # float32 1e-6 + ulp


def _centre(c, W, r=1.5, z=0.3):
    """A pose (to float32) exactly at the centre azimuth of column ``c``."""
    return _pol(r, (c + 0.5) * 2.0 * math.pi / W - math.pi, z)


def _pose_variant(name, W, col):
    """(source, context) on the boundaries the removed pose keys broke at."""
    return {
        "generic": (_SRC, _CTX),
        "centre": (_centre(col, W), (_centre(col, W, 1.0, 0.2), _centre((col + 1) % W, W, 0.8, -0.1))),
        "plus_pi": ((-1.5, 0.0, 0.3), ((-1.0, 0.0, 0.2), (1.0, 0.0, -0.1))),
        "minus_pi": ((-1.5, -0.0, 0.3), ((-1.0, -0.0, 0.2), (1.0, -0.0, -0.1))),
        "zero": ((1.5, 0.0, 0.3), ((1.0, 0.0, 0.2), (0.0, 0.0, -0.1))),
        "pi_minus_eps": (_pol(1.5, math.pi - _EPS, 0.3),
                         (_pol(1.0, -math.pi + _EPS, 0.2), _pol(1.0, math.pi - _EPS, -0.1))),
        "degenerate": (_DEG_SRC, _DEG_CTX),
        "ctx_r_below_eps": (_DEG_SRC, ((_R_BELOW, 0.0, 0.5), _centre(col, W, 1.0, 0.2))),
        "ctx_r_above_eps": (_DEG_SRC, ((_R_ABOVE, 0.0, 0.5), _centre(col, W, 1.0, 0.2))),
    }[name]


POSE_VARIANTS = ("generic", "centre", "plus_pi", "minus_pi", "zero", "pi_minus_eps", "degenerate",
                 "ctx_r_below_eps", "ctx_r_above_eps")
PERIODIC = {  # exactly periodic proxies (period W/2 and W/4): C-1 leaves several candidates
    "p256": (256, 512, dict(seed=60, period=256, peaks=[(80, 200)])),
    "p128": (256, 512, dict(seed=62, period=128, peaks=[(81, 5)])),
    "small_p16": (16, 32, dict(seed=61, period=16, peaks=[(2, 6)])),
    "small_p8": (16, 32, dict(seed=63, period=8, peaks=[(3, 1)])),
}
IMAGE_LEVEL = [n for n in FIXTURES if FIXTURES[n][5] in ("none", "same_column", "cross_column_image")]


def _fixture_with(name, variant):
    """A non-periodic fixture with a boundary pose variant anchored at its j*."""
    H, W, kw, _, _, cls = FIXTURES[name]
    d = _scalar_depth(H, W, **kw)
    j0 = yr.canonical_column(yr.recover_depth_proxy(_cloud(d)))[0]
    return _md(d, *_pose_variant(variant, W, j0)), cls


def test_tie_classes_are_image_level_only():
    assert yr.TIE_CLASSES == ("none", "same_column", "cross_column_image")


def test_canonical_column_takes_the_proxy_only():
    with pytest.raises(TypeError):
        yr.canonical_column(torch.rand(4, 8, dtype=torch.float64), torch.zeros(3))


def test_stabilized_tie_wording_claims_no_stabilizer_or_impossibility():
    for text in (str(yr.StabilizedTieError([3, 19])), yr.StabilizedTieError.__doc__):
        low = " ".join(text.lower().split())
        assert "exactly periodic" in low and "fail" in low
        assert "stabilizer" not in low and "no equivariant" not in low and "joint input" not in low


@pytest.mark.parametrize("variant", ["generic", "centre", "degenerate", "ctx_r_below_eps",
                                     "ctx_r_above_eps"])
@pytest.mark.parametrize("name", list(PERIODIC))
def test_periodic_image_fails_closed_for_every_roll_whatever_the_poses(name, variant):
    H, W, kw = PERIODIC[name]
    d = _scalar_depth(H, W, **kw)
    md = _md(d, *_pose_variant(variant, W, kw["peaks"][0][1]))
    cands0 = _max_columns(yr.recover_depth_proxy(md["depth"]))
    assert len(cands0) == W // kw["period"] and _oracle_image_winners(d, cands0) == cands0
    with pytest.raises(yr.StabilizedTieError) as raw:
        yr.canonical_column(d)                                   # raw scalar depth agrees
    assert raw.value.candidates == tuple(cands0)
    for k in range(W):
        rolled = _roll(md, k, W)
        with pytest.raises(yr.StabilizedTieError) as sel:
            yr.canonical_column(yr.recover_depth_proxy(rolled["depth"]))
        with pytest.raises(yr.StabilizedTieError) as canon:
            yr.canonicalize_scene_metadata(rolled, W)
        expected = tuple(sorted((c + k) % W for c in cands0))  # the candidates co-rotate
        assert sel.value.candidates == canon.value.candidates == expected, k


@pytest.mark.parametrize("name", IMAGE_LEVEL)
def test_selection_is_equivariant_for_every_roll_with_poses_on_the_column_centre(name):
    H, W, kw, _, _, cls = FIXTURES[name]
    d = _scalar_depth(H, W, **kw)
    j0, c0 = yr.canonical_column(yr.recover_depth_proxy(_cloud(d)))
    md = _md(d, *_pose_variant("centre", W, j0))               # poses exactly on j*'s centre
    assert c0 == cls
    for k in range(W):
        rolled = _roll(md, k, W)
        assert yr.canonical_column(yr.recover_depth_proxy(rolled["depth"])) == ((j0 + k) % W, c0)
        assert yr.canonicalize_scene_metadata(rolled, W)[1:] == (yr.canonical_shift((j0 + k) % W, W), c0)


@pytest.mark.parametrize("name", IMAGE_LEVEL)
def test_poses_never_influence_the_selection(name):
    H, W, kw, *_ = FIXTURES[name]
    d = _scalar_depth(H, W, **kw)
    j0, c0 = yr.canonical_column(yr.recover_depth_proxy(_cloud(d)))
    for variant in POSE_VARIANTS:
        md = _md(d, *_pose_variant(variant, W, j0))
        for k in (range(W) if W <= 32 else (0, 1, 255, 256, 511)):
            got = yr.canonicalize_scene_metadata(_roll(md, k, W), W)[1:]
            assert got == (yr.canonical_shift((j0 + k) % W, W), c0), (variant, k)
