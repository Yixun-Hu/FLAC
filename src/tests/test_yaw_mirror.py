"""exp_43 std_aug -- the horizontal mirror flip of FLAC's conditioning (plan §2.1, §3.1).

``src/data/yaw_mirror.py`` reflects ONE sample's conditioning across the plane ``y = 0``
(``M = diag(1, -1, 1)``): the ``[3, H, W]`` depth vector panorama has its columns reversed
and its channel 1 negated, and the four pose fields have their component 1 negated. Target
RIR, ``context_audio``, ``padding_mask``, ``scene`` and every other field pass through.

What is proven here, each against an oracle that does NOT call the code under test:

* the definition itself, by index arithmetic (``out[:, :, j] = in[:, :, W-1-j]``);
* exactness: bitwise involution (signed zeros and subnormals included), no mutation (also
  when ``source`` / ``source_vit`` alias one storage and ``context_poses_vit`` IS
  ``context_poses``, as AR_md builds them), dtype / device / shape kept, the x and z
  channels and every non-pose field bitwise unchanged, norms bitwise;
* geometry: the dihedral relation ``mirror(rotate(md, d)) = rotate(mirror(md), -d)`` with
  the reviewed yaw operator; **render equivalence** with the release
  ``AR_md.convert_equirect_to_camera_coord`` of the column-reversed scalar map (numerical,
  not bitwise -- plan F6: the float32 grid is not bit-symmetric); an **analytic room** -- a
  non-symmetric shoebox ray-cast from an off-centre listener, reflected in ``y`` -- whose
  re-rendered panorama and source positions the operator reproduces;
* every malformed input refused;
* the flip bits: exactly ``torch.randint(0, 2)`` of the CALLER's generator on a fixed-seed
  exact fixture, ``None`` refused, ``n = 0`` consumes nothing, a deterministic balance count
  on a pinned seed. No probabilistic assertion anywhere.
"""
import functools
import importlib.util
import math
import os
import random

import numpy as np
import pytest
import torch

from src.data import yaw_mirror as ym
from src.data import yaw_rotation as yr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_AR_MD_PATH = os.path.join(
    _REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata", "AR_md.py")
POSE_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit")
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@functools.lru_cache(maxsize=1)
def _ar_md():
    """The release metadata module, loaded by path exactly as ``src/data/dataset.py`` does."""
    spec = importlib.util.spec_from_file_location("ar_md_for_yaw_mirror_tests", _AR_MD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# helpers (independent of the code under test)
# --------------------------------------------------------------------------- #
def _bits(t):
    """Exact identity of a tensor's VALUES: dtype, shape and raw bytes (signed zeros count)."""
    t = t.detach().cpu().contiguous()
    raw = t.view(torch.int16) if t.dtype in (torch.float16, torch.bfloat16) else t
    return (t.dtype, tuple(t.shape), raw.numpy().tobytes())


def _make_md(seed, img_w=16, height=4, dtype=torch.float32, pose_dtype=torch.float32):
    """A metadata dict shaped like AR_md's: ``source_vit`` is a VIEW of ``source``'s storage
    (``unsqueeze(0)``) and ``context_poses_vit`` IS ``context_poses`` (one object)."""
    g = torch.Generator().manual_seed(seed)
    source = torch.randn(3, generator=g, dtype=torch.float64).to(pose_dtype)
    context = torch.randn(2, 3, generator=g, dtype=torch.float64).to(pose_dtype)
    return {
        "source": source,
        "source_vit": source.unsqueeze(0),
        "context_poses": context,
        "context_poses_vit": context,
        "context_audio": torch.randn(2, 8, generator=g),
        "depth": torch.randn(3, height, img_w, generator=g, dtype=torch.float64).to(dtype),
        "padding_mask": torch.ones(64, dtype=torch.bool),
        "scene": f"scene_{seed}",
        "idx": seed,
    }


def _unaliased(md):
    """The same values with every tensor in its own storage."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}


def _oracle_mirror(md):
    """The reflection by INDEX ARITHMETIC: column j <- column W-1-j, component 1 negated."""
    out = dict(md)
    depth = md["depth"]
    w = depth.shape[2]
    rev = torch.tensor([w - 1 - j for j in range(w)], dtype=torch.long)
    cols = depth.index_select(2, rev)
    out["depth"] = torch.stack([cols[0], torch.neg(cols[1]), cols[2]])
    for key in POSE_KEYS:
        p = md[key]
        out[key] = torch.stack([p[..., 0], torch.neg(p[..., 1]), p[..., 2]], dim=-1)
    return out


def _assert_bitwise(a, b, what):
    assert _bits(a) == _bits(b), f"{what} is not bitwise equal"


# --------------------------------------------------------------------------- #
# the definition and its exactness
# --------------------------------------------------------------------------- #
def test_mirror_axis_is_y():
    assert ym.MIRROR_AXIS == 1


@pytest.mark.parametrize("img_w", [16, 512])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mirror_equals_the_index_arithmetic_definition(img_w, dtype):
    md = _make_md(3, img_w=img_w, dtype=dtype, pose_dtype=dtype)
    got = ym.mirror_scene_metadata(md, img_w)
    want = _oracle_mirror(md)
    for key in ("depth",) + POSE_KEYS:
        _assert_bitwise(got[key], want[key], key)
        assert got[key].dtype == dtype and got[key].device == md[key].device
        assert got[key].shape == md[key].shape


def _with_special_values(md):
    """Signed zeros, a subnormal, and the extremes of the dtype planted in depth and poses."""
    md = _unaliased(md)
    fin = torch.finfo(md["depth"].dtype)
    specials = torch.tensor([0.0, -0.0, fin.tiny / 4, -fin.tiny / 4, fin.max, -fin.max,
                             fin.eps, -1.0], dtype=md["depth"].dtype)
    flat = md["depth"].view(-1)
    flat[: specials.numel()] = specials
    flat[-specials.numel():] = specials.flip(0)
    for key in POSE_KEYS:
        md[key].view(-1)[:3] = torch.tensor([-0.0, 0.0, -0.0], dtype=md[key].dtype)
    return md


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mirror_is_a_bitwise_involution(dtype):
    md = _with_special_values(_make_md(5, img_w=16, dtype=dtype, pose_dtype=dtype))
    twice = ym.mirror_scene_metadata(ym.mirror_scene_metadata(md, 16), 16)
    for key in ("depth",) + POSE_KEYS:
        _assert_bitwise(twice[key], md[key], f"mirror(mirror({key}))")
    once = ym.mirror_scene_metadata(md, 16)
    # the sign of zero really is flipped once (bitwise, not merely ==)
    assert _bits(once["source"]) != _bits(md["source"])
    assert torch.equal(once["source"][1:2], -md["source"][1:2])


def _snapshot(md):
    return {k: (_bits(v), v.data_ptr()) if torch.is_tensor(v) else v for k, v in md.items()}


def test_mirror_never_mutates_its_input_and_returns_fresh_storage():
    md = _make_md(7, img_w=16)
    before = _snapshot(md)
    keys_before = list(md)
    out = ym.mirror_scene_metadata(md, 16)
    assert _snapshot(md) == before and list(md) == keys_before
    assert out is not md
    for key in ("depth",) + POSE_KEYS:
        storage = out[key].untyped_storage().data_ptr()
        assert all(storage != md[k].untyped_storage().data_ptr() for k in ("depth",) + POSE_KEYS), key
    # writing into the output can never reach the caller's tensors
    for key in ("depth",) + POSE_KEYS:
        out[key].zero_()
    assert _snapshot(md) == before


def test_aliased_source_and_source_vit_are_each_mirrored_exactly_once():
    """AR_md: ``source_vit = source.unsqueeze(0)`` (one storage) and ``context_poses_vit`` IS
    ``context_poses``. An in-place negation through one alias would either corrupt the
    caller's batch or negate the second alias twice (i.e. not at all)."""
    md = _make_md(11, img_w=16)
    assert md["source_vit"].untyped_storage().data_ptr() == md["source"].untyped_storage().data_ptr()
    assert md["context_poses_vit"] is md["context_poses"]
    pristine = _unaliased(md)
    out = ym.mirror_scene_metadata(md, 16)
    want = _oracle_mirror(pristine)
    for key in POSE_KEYS:
        _assert_bitwise(out[key], want[key], key)
        _assert_bitwise(md[key], pristine[key], f"input {key}")
    assert out["source_vit"].untyped_storage().data_ptr() != out["source"].untyped_storage().data_ptr()
    assert out["context_poses_vit"] is not out["context_poses"]
    out["source"].zero_()
    _assert_bitwise(out["source_vit"], want["source_vit"], "source_vit after writing source")


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_dtype_device_and_shape_are_kept_also_for_a_permuted_view(dtype):
    """AR_md's depth is a permuted (non-contiguous) view of an [H, W, 3] tensor."""
    hw3 = torch.randn(4, 16, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64).to(dtype)
    md = _make_md(2, img_w=16, dtype=dtype, pose_dtype=dtype)
    md["depth"] = hw3.permute(2, 0, 1)
    assert not md["depth"].is_contiguous()
    out = ym.mirror_scene_metadata(md, 16)
    for key in ("depth",) + POSE_KEYS:
        assert out[key].dtype == md[key].dtype
        assert out[key].device == md[key].device
        assert out[key].shape == md[key].shape
    _assert_bitwise(out["depth"], _oracle_mirror(md)["depth"], "depth of a permuted view")


def test_x_z_channels_and_every_non_pose_field_are_bitwise_unchanged():
    md = _make_md(13, img_w=16)
    md.update({"path": "/x/y.wav", "relpath": "a/b/y.wav", "extra": torch.arange(5)})
    out = ym.mirror_scene_metadata(md, 16)
    assert list(out) == list(md)
    rev = torch.tensor(list(range(15, -1, -1)))
    for c in (0, 2):
        _assert_bitwise(out["depth"][c], md["depth"][c].index_select(1, rev), f"depth channel {c}")
    for key in POSE_KEYS:
        for c in (0, 2):
            _assert_bitwise(out[key][..., c], md[key][..., c], f"{key}[..., {c}]")
    for key in md:
        if key not in ("depth",) + POSE_KEYS:
            assert out[key] is md[key], f"{key} is not passed through as the same object"


def test_norms_are_bitwise_preserved_and_distances_to_float_tolerance():
    md = _make_md(17, img_w=16, height=6)
    out = ym.mirror_scene_metadata(md, 16)
    rev = torch.tensor(list(range(15, -1, -1)))
    _assert_bitwise(torch.linalg.vector_norm(out["depth"], dim=0),
                    torch.linalg.vector_norm(md["depth"], dim=0).index_select(1, rev), "pixel norms")
    for key in POSE_KEYS:
        _assert_bitwise(torch.linalg.vector_norm(out[key], dim=-1),
                        torch.linalg.vector_norm(md[key], dim=-1), f"{key} norms")
    pts_in = torch.cat([md["source"][None], md["context_poses"],
                        md["depth"].reshape(3, -1).t().float()])
    pts_out = torch.cat([out["source"][None], out["context_poses"],
                         out["depth"].index_select(2, rev).reshape(3, -1).t().float()])
    assert torch.allclose(torch.cdist(pts_out, pts_out), torch.cdist(pts_in, pts_in),
                          rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# the dihedral relation with the reviewed yaw operator
# --------------------------------------------------------------------------- #
_SWEEP = sorted(set(range(0, 512, 37)) | {1, 127, 128, 255, 256, 257, 383, 384, 511})
_QUARTER_TURNS = (0, 128, 256, 384)


def _rot(md, d, img_w=512):
    return yr.rotate_scene_metadata(md, yr.offsets_to_radians([d], img_w)[0], img_w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flip_after_roll_d_is_roll_minus_d_after_flip(dtype):
    """``flip . roll_d = roll_{-d} . flip``: the column permutations agree EXACTLY (the
    z channel, which neither operator changes in value, carries them bit for bit); the
    rotated vectors agree to <= 1e-6 (roll_{-d} is built from the angle (512-d)*2pi/512,
    whose float cos / sin are not bitwise those of d*2pi/512)."""
    md = _make_md(19, img_w=512, height=4, dtype=dtype)
    for d in _SWEEP:
        left = ym.mirror_scene_metadata(_rot(md, d), 512)
        right = _rot(ym.mirror_scene_metadata(md, 512), (-d) % 512)
        _assert_bitwise(left["depth"][2], right["depth"][2], f"z channel at d={d}")
        for key in ("depth",) + POSE_KEYS:
            assert left[key].dtype == right[key].dtype == md[key].dtype
            assert torch.allclose(left[key], right[key], rtol=0.0, atol=1e-6), (key, d)
        for key in md:
            if key not in ("depth",) + POSE_KEYS:
                assert left[key] is md[key] and right[key] is md[key]


@pytest.mark.parametrize("d", _QUARTER_TURNS)
def test_dihedral_relation_is_exact_at_quarter_turns_in_float32(d):
    """At a quarter turn the float32 rotation matrix is a signed permutation up to cos / sin
    residues of <= 1.9e-16, far below half an ulp of these unit-scale float32 values."""
    md = _make_md(23, img_w=512, height=4)
    left = ym.mirror_scene_metadata(_rot(md, d), 512)
    right = _rot(ym.mirror_scene_metadata(md, 512), (-d) % 512)
    for key in ("depth",) + POSE_KEYS:
        assert torch.equal(left[key], right[key]), (key, d)


# --------------------------------------------------------------------------- #
# render equivalence with the release AR_md grid (numerical, F6)
# --------------------------------------------------------------------------- #
def _cloud(D):
    """The metadata ``depth`` field exactly as AR_md builds it: a [3, H, W] permuted view."""
    H, W = D.shape
    return _ar_md().convert_equirect_to_camera_coord(D, H, W).permute(2, 0, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_render_equivalence_with_the_column_reversed_scalar_map(dtype):
    H, W = 256, 512
    ones = torch.ones(H, W, dtype=dtype)
    got = ym.mirror_scene_metadata({"depth": _cloud(ones)}, W)["depth"]
    want = _cloud(torch.flip(ones, dims=[1]))
    # F6: the float32 grid is NOT bit-symmetric, so equality is numerical (<= 1e-6 at unit range)
    assert not torch.equal(got, want)
    assert float((got - want).abs().max()) <= 1e-6
    _assert_bitwise(got[2], want[2], "z channel (independent of the column angle)")

    g = torch.Generator().manual_seed(29)
    D = (0.5 + 9.5 * torch.rand(H, W, generator=g, dtype=torch.float64)).to(dtype)
    got = ym.mirror_scene_metadata({"depth": _cloud(D)}, W)["depth"]
    D_rev = torch.flip(D, dims=[1])
    want = _cloud(D_rev)
    assert bool(((got - want).abs() <= 1e-6 * D_rev[None]).all()), "beyond 1e-6 of the range"
    _assert_bitwise(got[2], want[2], "z channel")


# --------------------------------------------------------------------------- #
# the analytic room: a non-symmetric shoebox, ray-cast, reflected in y
# --------------------------------------------------------------------------- #
def _ar_directions(H, W):
    """Unit view rays of AR's grid, in float64: (cos phi cos theta, cos phi sin theta, -sin phi)."""
    h = torch.arange(H, dtype=torch.float64)
    w = torch.arange(W, dtype=torch.float64)
    phi = ((h + 0.5) * math.pi / H - math.pi / 2)[:, None].expand(H, W)
    theta = ((w + 0.5) * 2.0 * math.pi / W - math.pi)[None, :].expand(H, W)
    return torch.stack([torch.cos(phi) * torch.cos(theta), torch.cos(phi) * torch.sin(theta),
                        -torch.sin(phi)])


def _ray_cast_box(lo, hi, eye, H, W):
    """[H, W] range from ``eye`` to the inside of the axis-aligned box [lo, hi] (float64)."""
    u = _ar_directions(H, W)
    ts = []
    for a in range(3):
        ua = u[a]
        t = torch.full_like(ua, math.inf)
        pos, neg = ua > 0, ua < 0
        t[pos] = (hi[a] - eye[a]) / ua[pos]
        t[neg] = (lo[a] - eye[a]) / ua[neg]
        ts.append(t)
    return torch.stack(ts).min(dim=0).values


def _room_metadata(lo, hi, listener, source, contexts, H, W):
    m = _ar_md()
    D = _ray_cast_box(lo, hi, listener, H, W)
    src = torch.Tensor(m.get_3d_point_camera_coord(listener, np.asarray(source))).float()
    ctx = torch.vstack([torch.Tensor(m.get_3d_point_camera_coord(listener, np.asarray(c))).float()
                        for c in contexts])
    md = {"depth": _cloud(D), "source": src, "source_vit": src.unsqueeze(0),
          "context_poses": ctx, "context_poses_vit": ctx, "scene": "shoebox"}
    return md, D


def _reflect_y(p):
    return [p[0], -p[1], p[2]]


def test_analytic_room_reflected_in_y_is_reproduced_by_the_mirror():
    H, W = 256, 512
    lo, hi = [0.0, 0.0, 0.0], [7.0, 5.0, 3.0]
    listener, source = [2.1, 1.3, 1.2], [5.6, 3.7, 1.9]
    contexts = [[1.0, 4.2, 0.8], [6.3, 0.7, 2.4], [3.3, 2.2, 0.4]]
    md_a, D_a = _room_metadata(lo, hi, listener, source, contexts, H, W)
    # the same room reflected through the world plane y = 0
    lo_b, hi_b = [lo[0], -hi[1], lo[2]], [hi[0], -lo[1], hi[2]]
    md_b, D_b = _room_metadata(lo_b, hi_b, _reflect_y(listener), _reflect_y(source),
                               [_reflect_y(c) for c in contexts], H, W)

    # the room is genuinely non-symmetric about the listener: the test is not vacuous
    assert float((D_a - torch.flip(D_a, dims=[1])).abs().max()) > 0.5
    assert float((D_a - torch.flip(D_a, dims=[0])).abs().max()) > 0.5
    # harness consistency: the reflected room's ranges are the column-reversed ranges
    assert torch.allclose(D_b, torch.flip(D_a, dims=[1]), rtol=1e-12, atol=0.0)

    got = ym.mirror_scene_metadata(md_a, W)
    assert bool(((got["depth"] - md_b["depth"]).abs() <= 1e-6 * D_b[None]).all())
    for key in POSE_KEYS:
        _assert_bitwise(got[key], md_b[key], f"{key} of the reflected room")
    assert got["scene"] is md_a["scene"]

    # negative controls: the two halves of the reflection are each necessary
    cols_only = torch.flip(md_a["depth"], dims=[2])
    negate_only = md_a["depth"].clone()
    negate_only[1] = -negate_only[1]
    for wrong in (cols_only, negate_only, md_a["depth"]):
        assert float((wrong - md_b["depth"]).abs().max()) > 0.5
    assert not torch.allclose(md_a["source"], md_b["source"], atol=0.5)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("img_w", [True, False, 16.0, "16", None, 0, -16])
def test_refuses_a_bad_img_w(img_w):
    with pytest.raises(ValueError, match="img_w"):
        ym.mirror_scene_metadata(_make_md(0), img_w)


@pytest.mark.parametrize("md", [None, [("depth", torch.zeros(3, 4, 16))], "md"])
def test_refuses_metadata_that_is_not_a_dict(md):
    with pytest.raises(ValueError, match="dict"):
        ym.mirror_scene_metadata(md, 16)


_BAD_DEPTHS = {
    "list": [[0.0] * 16],
    "numpy": np.zeros((3, 4, 16)),
    "int64": torch.zeros(3, 4, 16, dtype=torch.long),
    "bool": torch.zeros(3, 4, 16, dtype=torch.bool),
    "complex": torch.zeros(3, 4, 16, dtype=torch.complex64),
    "ndim1": torch.zeros(16),
    "ndim2": torch.zeros(3, 16),
    "ndim4": torch.zeros(3, 4, 16, 1),
    "channels1": torch.zeros(1, 4, 16),
    "channels4": torch.zeros(4, 4, 16),
    "width8": torch.zeros(3, 4, 8),
    "width17": torch.zeros(3, 4, 17),
}


@pytest.mark.parametrize("name", sorted(_BAD_DEPTHS))
def test_refuses_a_malformed_depth(name):
    md = _make_md(0)
    md["depth"] = _BAD_DEPTHS[name]
    with pytest.raises(ValueError, match="depth"):
        ym.mirror_scene_metadata(md, 16)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_refuses_a_non_finite_depth(value, dtype):
    md = _make_md(0, dtype=dtype)
    md["depth"] = md["depth"].clone()
    md["depth"][1, 2, 3] = value
    with pytest.raises(ValueError, match="depth.*finite"):
        ym.mirror_scene_metadata(md, 16)


_BAD_POSES = {
    "list": [0.0, 0.0, 0.0],
    "numpy": np.zeros(3),
    "int64": torch.zeros(3, dtype=torch.long),
    "scalar": torch.tensor(1.0),
    "trailing2": torch.zeros(2),
    "trailing4": torch.zeros(1, 4),
    "trailing0": torch.zeros(2, 0),
}


@pytest.mark.parametrize("key", POSE_KEYS)
@pytest.mark.parametrize("name", sorted(_BAD_POSES))
def test_refuses_a_malformed_pose(key, name):
    md = _unaliased(_make_md(0))
    md[key] = _BAD_POSES[name]
    with pytest.raises(ValueError, match=key):
        ym.mirror_scene_metadata(md, 16)


@pytest.mark.parametrize("pose_keys", ["source", None, ("source", 1), [("source",)]])
def test_refuses_malformed_pose_keys(pose_keys):
    with pytest.raises(ValueError, match="pose_keys"):
        ym.mirror_scene_metadata(_make_md(0), 16, pose_keys=pose_keys)


def test_absent_fields_are_skipped_like_rotate_scene_metadata():
    """Same absent-key semantics as the reviewed yaw operator (the training path requires
    all five fields itself, before any draw)."""
    md = {"source": torch.tensor([1.0, 2.0, 3.0]), "scene": "s"}
    out = ym.mirror_scene_metadata(md, 16)
    assert set(out) == {"source", "scene"}
    assert out["source"].tolist() == [1.0, -2.0, 3.0]
    assert ym.mirror_scene_metadata({}, 16) == {}


def test_pose_keys_restricts_the_reflected_poses():
    md = _make_md(31)
    out = ym.mirror_scene_metadata(md, 16, pose_keys=("source_vit",))
    assert out["source"] is md["source"] and out["context_poses"] is md["context_poses"]
    _assert_bitwise(out["source_vit"], _oracle_mirror(md)["source_vit"], "source_vit")
    _assert_bitwise(out["depth"], _oracle_mirror(md)["depth"], "depth")


def test_check_mirror_metadata_is_the_operators_contract():
    """The validator the training path runs on EVERY sample before the bit draw (so a refusal
    never depends on which bits came up) is exactly the operator's own contract."""
    md = _make_md(0)
    assert ym.check_mirror_metadata(md, 16) is None
    for name in ("int64", "width8"):
        bad = dict(md, depth=_BAD_DEPTHS[name])
        with pytest.raises(ValueError, match="depth"):
            ym.check_mirror_metadata(bad, 16)
    with pytest.raises(ValueError, match="context_poses"):
        ym.check_mirror_metadata(dict(md, context_poses=torch.zeros(2, 2)), 16)
    with pytest.raises(ValueError, match="img_w"):
        ym.check_mirror_metadata(md, True)


# --------------------------------------------------------------------------- #
# the flip bits
# --------------------------------------------------------------------------- #
# torch 2.7.0 CPU MT19937, seed 4242, n = 40: an EXACT fixture (captured once from
# ``torch.randint(0, 2, (40,), generator=torch.Generator().manual_seed(4242))`` in the pinned
# ``flac`` env, never from the code under test), not a statistic.
FIXTURE_SEED, FIXTURE_N = 4242, 40
EXPECTED_FIXTURE_BITS = [
    0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
]
# int(torch.randint(0, 2, (4096,), generator=torch.Generator().manual_seed(43)).sum()), same env
EXPECTED_ONES_SEED43_N4096 = 2077


def _state(g):
    return g.get_state().numpy().tobytes()


def test_flips_are_randint_0_2_of_the_callers_generator_on_an_exact_fixture():
    g = torch.Generator().manual_seed(FIXTURE_SEED)
    bits = ym.draw_mirror_flips(FIXTURE_N, g)
    assert bits.dtype == torch.long and tuple(bits.shape) == (FIXTURE_N,)
    assert bits.tolist() == EXPECTED_FIXTURE_BITS
    ref = torch.Generator().manual_seed(FIXTURE_SEED)
    assert torch.equal(bits, torch.randint(0, 2, (FIXTURE_N,), generator=ref, dtype=torch.long))
    # the same amount of the stream was consumed: the next draws coincide
    assert _state(g) == _state(ref)
    assert torch.equal(torch.randint(0, 512, (8,), generator=g),
                       torch.randint(0, 512, (8,), generator=ref))


def test_flips_after_offsets_leave_the_offsets_unchanged():
    """Plan §2.3: offsets first (the unchanged draw), then the bits from the same generator."""
    g = torch.Generator().manual_seed(77)
    offsets = yr.draw_yaw_offsets(32, 512, g)
    bits = ym.draw_mirror_flips(32, g)
    ref = torch.Generator().manual_seed(77)
    assert torch.equal(offsets, yr.draw_yaw_offsets(32, 512, ref))
    assert torch.equal(bits, torch.randint(0, 2, (32,), generator=ref, dtype=torch.long))


def test_balance_on_a_pinned_seed_is_an_exact_count():
    """Deterministic: the count on this pinned seed is a fixture, not a tolerance band."""
    bits = ym.draw_mirror_flips(4096, torch.Generator().manual_seed(43))
    assert set(bits.unique().tolist()) == {0, 1}
    assert int(bits.sum()) == EXPECTED_ONES_SEED43_N4096


def test_zero_flips_consume_nothing():
    g = torch.Generator().manual_seed(5)
    before = _state(g)
    bits = ym.draw_mirror_flips(0, g)
    assert bits.dtype == torch.long and tuple(bits.shape) == (0,)
    assert _state(g) == before


@pytest.mark.parametrize("generator", [None, 42, np.random.RandomState(0), "g"])
def test_flips_refuse_anything_but_a_dedicated_generator(generator):
    with pytest.raises(ValueError, match="Generator"):
        ym.draw_mirror_flips(4, generator)


@pytest.mark.parametrize("n", [True, False, 4.0, "4", None, -1])
def test_flips_refuse_a_bad_n(n):
    with pytest.raises(ValueError, match="n must be"):
        ym.draw_mirror_flips(n, torch.Generator().manual_seed(0))


def test_flips_leave_the_global_rng_streams_untouched():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    before = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())
    ym.draw_mirror_flips(64, torch.Generator().manual_seed(3))
    after = (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())
    assert before[0] == after[0] and before[1] == after[1] and torch.equal(before[2], after[2])
