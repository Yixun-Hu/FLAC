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
_SRC = (1.3, -0.7, 0.2)                          # generic target source
_CTX = ((0.9, 1.1, -0.3), (-1.2, 0.4, 0.1))      # generic context rows
FIXTURES = {
    "unique": (256, 512, dict(seed=10, peaks=[(37, 100)]), _SRC, _CTX, "none"),
    "same_column": (256, 512, dict(seed=11, peaks=[(20, 300), (200, 300)]), _SRC, _CTX,
                    "same_column"),
    "small_unique": (16, 32, dict(seed=20, peaks=[(3, 9)]), _SRC, _CTX, "none"),
    "small_same_column": (16, 32, dict(seed=21, peaks=[(2, 5), (13, 5)]), _SRC, _CTX,
                          "same_column"),
}


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
    assert [j] == cands


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
@pytest.mark.parametrize("name", list(FIXTURES))
def test_selector_is_equivariant_for_every_roll(name):
    _, md, cls = _fixture(name)
    W = md["depth"].shape[-1]
    j0, c0 = _select(md)
    assert c0 == cls
    bad = [d for d in range(W) if _select(_roll(md, d, W)) != ((j0 + d) % W, c0)]
    assert bad == []


# --------------------------------------------------------------------------- #
# 9. raw scalar depth vs proxy: same candidate set, same selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(FIXTURES))
def test_raw_and_proxy_agree_on_candidates_and_selection(name):
    d, md, _ = _fixture(name)
    D = yr.recover_depth_proxy(md["depth"])
    assert _max_columns(d) == _max_columns(D)
    assert _outcome(md, d) == _outcome(md, D)
