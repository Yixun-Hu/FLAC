"""exp_14 round 1 — random per-sample yaw evaluation (plan §§3.3, 5.1–5.3).

The campaign draws an independent yaw offset ``d_i ~ Uniform{0..W-1}`` for every
evaluation item and rotates that item's conditioning by ``d_i * 2*pi/W`` — an
exact panorama-column roll, no interpolation. What makes the resulting numbers
*evidence* rather than plausible noise is that the assignment is reproducible and
auditable: the same rotation seed must yield the same offsets, in the same order,
attached to the same items, independently of how the loader chunks them.

RED first (cycle 1): ``draw_yaw_offsets`` / ``offsets_to_radians`` do not exist
(AttributeError). Later cycles add the eval-side plan resolution, naming,
canonical hashing and the batch application helper.
"""
import math

import pytest
import torch

from src.data import yaw_rotation as yr


# The pre-registered golden assignment (plan §4 gate G3): the first 16 offsets a
# rotation seed of 42 must produce on a 512-column panorama. Derived from the
# SPECIFIED algorithm (torch.randint over [0, 512) on a dedicated CPU generator
# seeded with 42), not from the implementation -- it is the algorithm that is
# pinned, so a silent change of draw mechanism fails here.
GOLDEN_SEED42_W512 = [102, 435, 348, 270, 106, 71, 188, 20,
                      102, 121, 466, 214, 330, 458, 87, 372]


def _gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _draw_in_chunks(seed, img_w, chunks):
    g = _gen(seed)
    return torch.cat([yr.draw_yaw_offsets(n, img_w, g) for n in chunks])


# --------------------------------------------------------------------------- #
# draw_yaw_offsets — shape / dtype / support
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_shape_dtype_and_support():
    out = yr.draw_yaw_offsets(1000, 512, _gen(0))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1000,)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0 and int(out.max()) < 512
    # a 1000-draw sample of Uniform{0..511} must not be degenerate
    assert len(set(out.tolist())) > 100


def test_draw_yaw_offsets_zero_length():
    """A tail batch of size 0 is not an error and must not consume randomness."""
    g = _gen(7)
    empty = yr.draw_yaw_offsets(0, 512, g)
    assert empty.shape == (0,)
    assert torch.equal(yr.draw_yaw_offsets(4, 512, g), yr.draw_yaw_offsets(4, 512, _gen(7)))


# --------------------------------------------------------------------------- #
# (1) determinism
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_deterministic_for_same_seed():
    a = yr.draw_yaw_offsets(256, 512, _gen(42))
    b = yr.draw_yaw_offsets(256, 512, _gen(42))
    assert torch.equal(a, b)


def test_draw_yaw_offsets_differs_across_seeds():
    a = yr.draw_yaw_offsets(256, 512, _gen(42))
    b = yr.draw_yaw_offsets(256, 512, _gen(43))
    assert not torch.equal(a, b)
    # ...and not merely by a shift: the streams are independent.
    assert (a != b).float().mean() > 0.9


# --------------------------------------------------------------------------- #
# (2) chunk independence — the batch size must not change the assignment
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_chunk_independence_8_vs_64():
    """Batch size is pinned for the campaign, but the *stream* must not depend on
    it: a batch-size drift would otherwise silently re-assign every offset."""
    flat = yr.draw_yaw_offsets(192, 512, _gen(42))
    by_8 = _draw_in_chunks(42, 512, [8] * 24)
    by_64 = _draw_in_chunks(42, 512, [64] * 3)
    assert torch.equal(flat, by_8)
    assert torch.equal(flat, by_64)


def test_draw_yaw_offsets_chunk_independence_with_tail_batch():
    """The real split (6337 items) never divides evenly by the batch size."""
    flat = yr.draw_yaw_offsets(137, 512, _gen(42))
    chunked = _draw_in_chunks(42, 512, [64, 64, 9])
    assert torch.equal(flat, chunked)


# --------------------------------------------------------------------------- #
# (3) global RNG isolation (Codex plan review B1)
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_does_not_touch_global_rng():
    """The dedicated generator must not perturb the evaluation's own sampling
    noise — otherwise a rotated cell and its paired unrotated cell would draw
    different diffusion noise and the pairing would be void."""
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    _ = yr.draw_yaw_offsets(4096, 512, _gen(42))
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    # and the global stream itself is unchanged downstream
    torch.manual_seed(1234)
    ref = torch.randn(8)
    torch.manual_seed(1234)
    _ = yr.draw_yaw_offsets(4096, 512, _gen(42))
    assert torch.equal(ref, torch.randn(8))


# --------------------------------------------------------------------------- #
# (10a) golden assignment
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_golden_seed42_prefix():
    out = yr.draw_yaw_offsets(len(GOLDEN_SEED42_W512), 512, _gen(42))
    assert out.tolist() == GOLDEN_SEED42_W512


# --------------------------------------------------------------------------- #
# offsets_to_radians
# --------------------------------------------------------------------------- #
def test_offsets_to_radians_exact_values():
    rad = yr.offsets_to_radians(torch.tensor([0, 1, 128, 256, 511]), 512)
    assert isinstance(rad, list)
    assert all(isinstance(a, float) for a in rad)
    assert rad[0] == 0.0
    assert rad[1] == 1 * 2.0 * math.pi / 512
    assert rad[2] == 128 * 2.0 * math.pi / 512   # == pi/2 exactly on this grid
    assert rad[3] == 256 * 2.0 * math.pi / 512
    assert rad[4] == 511 * 2.0 * math.pi / 512


def test_offsets_to_radians_accepts_plain_sequences():
    assert yr.offsets_to_radians([0, 5], 512) == yr.offsets_to_radians(
        torch.tensor([0, 5]), 512)


# --------------------------------------------------------------------------- #
# (4) exactness: every drawn angle round-trips through rotate_scene_metadata
# --------------------------------------------------------------------------- #
def test_every_offset_round_trips_through_rotate_scene_metadata_quantisation():
    """``rotate_scene_metadata`` re-quantises its angle to ``dj`` columns. For the
    drawn angles that re-quantisation must be the IDENTITY (dj == d) for every
    offset in the support — otherwise the recorded offset is not the applied one."""
    img_w = 512
    offsets = list(range(img_w))
    for d, alpha in zip(offsets, yr.offsets_to_radians(offsets, img_w)):
        dj = int(round(alpha * img_w / (2.0 * math.pi))) % img_w
        assert dj == d, f"offset {d} re-quantised to {dj}"


def test_drawn_offsets_round_trip_at_512_and_other_widths():
    for img_w in (512, 256, 360):
        drawn = yr.draw_yaw_offsets(64, img_w, _gen(42))
        for d, alpha in zip(drawn.tolist(), yr.offsets_to_radians(drawn, img_w)):
            dj = int(round(alpha * img_w / (2.0 * math.pi))) % img_w
            assert dj == d


# --------------------------------------------------------------------------- #
# (5) per-sample application == per-item scalar-path application
# --------------------------------------------------------------------------- #
def _consistent_depth(H=4, W=512):
    """A geometrically consistent equirectangular depth point cloud (same
    construction as test_yaw_symmetry)."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    d = 3.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    return torch.stack([d * torch.cos(el_g) * torch.cos(theta_g),
                        d * torch.cos(el_g) * torch.sin(theta_g),
                        d * torch.sin(el_g)], dim=0).contiguous()


def _make_md(seed=0, n_ctx=4, img_w=512):
    g = torch.Generator().manual_seed(seed)
    return {
        "idx": seed,
        "relpath": f"scene{seed}/id0/S001_R00{seed}_hybrid_IR.wav",
        "scene": f"scene{seed}",
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(n_ctx, 3, generator=g),
        "context_poses_vit": torch.randn(n_ctx, 3, generator=g),
        "context_audio": torch.randn(n_ctx, 16, generator=g),
        "depth": _consistent_depth(4, img_w),
    }


def test_per_item_angles_equal_scalar_path_per_item():
    """Applying a per-item angle list must equal calling the scalar rotation once
    per item -- the random path may not quietly share one angle across a batch."""
    img_w = 512
    batch = [_make_md(seed=s, img_w=img_w) for s in range(5)]
    offsets = yr.draw_yaw_offsets(len(batch), img_w, _gen(42))
    angles = yr.offsets_to_radians(offsets, img_w)
    assert len(set(offsets.tolist())) > 1, "degenerate fixture: all offsets equal"

    per_item = [yr.rotate_scene_metadata(md, a, img_w) for md, a in zip(batch, angles)]
    for md, a, got in zip(batch, angles, per_item):
        want = yr.rotate_scene_metadata(md, a, img_w)
        for key in ("depth",) + yr.POSE_KEYS:
            assert torch.equal(got[key], want[key]), key
        # untouched fields pass through as the SAME objects
        assert got["context_audio"] is md["context_audio"]
        assert got["scene"] == md["scene"]


def test_per_item_angles_actually_differ_between_items():
    """Sanity that the fixture proves something: two items drawing different
    offsets must end up with different rotated depth maps."""
    img_w = 512
    md_a, md_b = _make_md(seed=1, img_w=img_w), _make_md(seed=1, img_w=img_w)
    rot_a = yr.rotate_scene_metadata(md_a, yr.offsets_to_radians([10], img_w)[0], img_w)
    rot_b = yr.rotate_scene_metadata(md_b, yr.offsets_to_radians([11], img_w)[0], img_w)
    assert not torch.equal(rot_a["depth"], rot_b["depth"])


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #
def test_draw_yaw_offsets_requires_a_generator():
    """A missing generator would silently fall back to the GLOBAL RNG, which is
    exactly the failure mode isolation is meant to prevent."""
    with pytest.raises((TypeError, ValueError)):
        yr.draw_yaw_offsets(4, 512, None)


@pytest.mark.parametrize("img_w", [0, -1])
def test_draw_yaw_offsets_rejects_nonpositive_width(img_w):
    with pytest.raises(ValueError):
        yr.draw_yaw_offsets(4, img_w, _gen(0))
