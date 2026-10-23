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
import importlib.util
import math
import os

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
_PHI = 0.7                                       # half-turn-symmetric context pairs: PHI, PHI + pi
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
    # exactly periodic images (period W/2 and W/4): the source key decides
    "p256_src_generic": (256, 512, dict(seed=40, period=256, peaks=[(60, 30)]), _SRC, _CTX,
                         "cross_column_source"),
    "p128_src_plus_pi": (256, 512, dict(seed=41, period=128, peaks=[(61, 5)]),
                         (-1.5, 0.0, 0.3), _CTX, "cross_column_source"),
    "p128_src_minus_pi": (256, 512, dict(seed=42, period=128, peaks=[(62, 127)]),
                          (-1.5, -0.0, 0.3), _CTX, "cross_column_source"),
    "p256_src_pi_minus_eps": (256, 512, dict(seed=43, period=256, peaks=[(63, 255)]),
                              _pol(1.5, math.pi - _EPS, 0.3), _CTX, "cross_column_source"),
    "p256_src_minus_pi_plus_eps": (256, 512, dict(seed=44, period=256, peaks=[(64, 128)]),
                                   _pol(1.5, -math.pi + _EPS, 0.3), _CTX, "cross_column_source"),
    "p128_src_zero": (256, 512, dict(seed=45, period=128, peaks=[(65, 64)]), (1.5, 0.0, 0.3),
                      _CTX, "cross_column_source"),
    "small_p16_src_plus_pi": (16, 32, dict(seed=46, period=16, peaks=[(5, 2)]),
                              (-1.5, 0.0, 0.3), _CTX, "cross_column_source"),
    "small_p8_src_minus_pi": (16, 32, dict(seed=47, period=8, peaks=[(5, 7)]),
                              (-1.5, -0.0, 0.3), _CTX, "cross_column_source"),
    "small_p16_src_pi_minus_eps": (16, 32, dict(seed=48, period=16, peaks=[(6, 0)]),
                                   _pol(1.5, math.pi - _EPS, 0.3), _CTX, "cross_column_source"),
    "small_p8_src_zero": (16, 32, dict(seed=49, period=8, peaks=[(7, 3)]), (1.5, 0.0, 0.3),
                          _CTX, "cross_column_source"),
    # periodic image + degenerate source: the ORDERED context key decides
    "p256_ctx_r_only": (256, 512, dict(seed=50, period=256, peaks=[(70, 100)]), _DEG_SRC,
                        (_pol(1.0, _PHI, 0.3), _pol(2.0, _PHI + math.pi, 0.3)),
                        "cross_column_context"),
    "p256_ctx_z_only": (256, 512, dict(seed=51, period=256, peaks=[(71, 101)]), _DEG_SRC,
                        (_pol(1.0, _PHI, 0.3), _pol(1.0, _PHI + math.pi, -0.4)),
                        "cross_column_context"),
    "p256_ctx_az_only": (256, 512, dict(seed=52, period=256, peaks=[(72, 102)]), _DEG_SRC,
                         (_pol(1.0, _PHI, 0.3), _pol(1.0, _PHI + math.pi, 0.3)),
                         "cross_column_context"),
    "p128_ctx_first_row_degenerate": (256, 512, dict(seed=53, period=128, peaks=[(73, 64)]),
                                      _DEG_SRC, ((0.0, 0.0, 0.5), _pol(1.2, -2.0, -0.1)),
                                      "cross_column_context"),
    "small_p8_ctx_generic": (16, 32, dict(seed=54, period=8, peaks=[(3, 1)]), _DEG_SRC, _CTX,
                             "cross_column_context"),
    "small_p16_ctx_az_only": (16, 32, dict(seed=55, period=16, peaks=[(4, 9)]), _DEG_SRC,
                              (_pol(1.0, _PHI, 0.3), _pol(1.0, _PHI + math.pi, 0.3)),
                              "cross_column_context"),
    # periodic image + degenerate source + identical ordered contexts: fail closed (C-4)
    "p256_stabilized": (256, 512, dict(seed=60, period=256, peaks=[(80, 200)]), _DEG_SRC,
                        _DEG_CTX, "stabilized"),
    "small_p8_stabilized": (16, 32, dict(seed=61, period=8, peaks=[(2, 6)]), _DEG_SRC,
                            _DEG_CTX, "stabilized"),
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
    D = yr.recover_depth_proxy(md["depth"]) if D is None else D
    return yr.canonical_column(D, md.get("source"), md.get("context_poses"))


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
def test_tie_classes_constant():
    assert yr.TIE_CLASSES == ("none", "same_column", "cross_column_image",
                              "cross_column_source", "cross_column_context")


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
    (torch.tensor([[1.0, float("nan")], [0.5, 0.5]]),), (torch.rand(4, 8), torch.zeros(1, 3)),
    (torch.rand(4, 8), torch.zeros(3, dtype=torch.int64)), (torch.rand(4, 8), torch.tensor([1.0, float("inf"), 0.0])),
    (torch.rand(4, 8), None, torch.zeros(3)), (torch.rand(4, 8), None, torch.zeros(2, 2)),
], ids=["3d", "int", "nan", "src_1x3", "src_int", "src_inf", "ctx_1d", "ctx_nx2"])
def test_canonical_column_rejects_contract_violations(args):
    with pytest.raises(ValueError):
        yr.canonical_column(*args)


# --------------------------------------------------------------------------- #
# 4. exhaustive selector equivariance (non-stabilized fixtures, ALL rolls)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [n for n in FIXTURES if FIXTURES[n][5] != "stabilized"])
def test_selector_is_equivariant_for_every_roll(name):
    _, md, cls = _fixture(name)
    W = md["depth"].shape[-1]
    j0, c0 = _select(md)
    assert c0 == cls
    bad = [d for d in range(W) if _select(_roll(md, d, W)) != ((j0 + d) % W, c0)]
    assert bad == []


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
# 3./4. pose keys on exactly periodic images (C-2 source, C-3 ordered context, C-4)
# --------------------------------------------------------------------------- #
def _rel(p, c, W):
    """Oracle relative azimuth (Python math); +inf for an azimuth-degenerate pose."""
    x, y = float(p[0]), float(p[1])
    if math.hypot(x, y) < 1e-6:
        return math.inf
    return (math.atan2(y, x) - ((c + 0.5) * 2.0 * math.pi / W - math.pi)) % (2.0 * math.pi)


def _oracle_pose_winners(md, cands, W):
    """C-2 then C-3 as one lexicographic tuple (a degenerate source is +inf everywhere)."""
    def key(c):
        ctx = tuple((math.hypot(float(p[0]), float(p[1])), float(p[2]), _rel(p, c, W))
                    for p in md["context_poses"])
        return _rel(md["source"], c, W), ctx
    keys = {c: key(c) for c in cands}
    return [c for c in cands if keys[c] == min(keys.values())]


@pytest.mark.parametrize("name", _names("cross_column_source", "cross_column_context"))
def test_canonical_column_pose_key_classes(name):
    H, W, kw, _, _, cls = FIXTURES[name]
    _, md, _ = _fixture(name)
    D = yr.recover_depth_proxy(md["depth"])
    cands = _max_columns(D)
    assert len(cands) == W // kw["period"] and _oracle_image_winners(D, cands) == cands
    # fixture validity: no deciding azimuth sits within round-off of a candidate centre
    rels = [_rel(p, c, W) for p in [md["source"], *md["context_poses"]] for c in cands]
    assert all(r == math.inf or 1e-4 < r < 2.0 * math.pi - 1e-4 for r in rels)
    j, tie_class = _select(md)
    assert tie_class == cls and [j] == _oracle_pose_winners(md, cands, W)


@pytest.mark.parametrize("name", ["p256_ctx_r_only", "p256_ctx_z_only", "p256_ctx_az_only"])
def test_ordered_context_key_separates_what_an_azimuth_multiset_cannot(name):
    """Plan-review r2 B1: these context sets are half-turn symmetric in azimuth, so the
    v2 sorted multiset of relative azimuths ties at both candidates; the ordered,
    pair-preserving (r, z, azimuth) row key does not."""
    _, md, _ = _fixture(name)
    c0, c1 = _max_columns(yr.recover_depth_proxy(md["depth"]))
    multisets = [sorted(_rel(p, c, 512) for p in md["context_poses"]) for c in (c0, c1)]
    assert multisets[0] == pytest.approx(multisets[1], abs=1e-6)
    assert _select(md)[1] == "cross_column_context"


@pytest.mark.parametrize("name", _names("stabilized"))
def test_stabilized_input_fails_closed_for_every_roll(name):
    _, md, _ = _fixture(name)
    W = md["depth"].shape[-1]
    cands0 = _max_columns(yr.recover_depth_proxy(md["depth"]))
    assert len(cands0) > 1
    for d in range(W):
        with pytest.raises(yr.StabilizedTieError) as info:
            _select(_roll(md, d, W))
        assert info.value.candidates == tuple(sorted((c + d) % W for c in cands0))


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
@pytest.mark.parametrize("name", ["unique", "same_column", "row_tie_sep64_wrap", "p128_src_minus_pi",
                                  "p256_ctx_az_only", "small_row_tie_sep1_wrap"])
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


@pytest.mark.parametrize("name", _names("stabilized"))
def test_canonicalize_propagates_stabilized_tie_for_every_roll(name):
    _, md, _ = _fixture(name)
    W = md["depth"].shape[-1]
    for d in range(W):
        with pytest.raises(yr.StabilizedTieError):
            yr.canonicalize_scene_metadata(_roll(md, d, W), W)
