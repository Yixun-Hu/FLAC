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
import json
import math
from pathlib import Path

import pytest
import torch

import eval_FLAC
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


# =========================================================================== #
# CYCLE 2 — rotation-plan resolution + artifact naming (plan §5.2, review B4/B5)
# =========================================================================== #
CKPT = "weights/FLAC/FLAC_EMA.ckpt"


# --------------------------------------------------------------------------- #
# (7) guards — a misconfigured rotation must never run, and never be ignored
# --------------------------------------------------------------------------- #
def test_fixed_mode_plan_is_the_legacy_behaviour():
    plan = eval_FLAC.resolve_rotation_plan("fixed", 45.0, None, 42)
    assert plan.mode == "fixed"
    assert plan.rotate_deg == 45.0
    assert plan.rotate_seed is None
    assert plan.is_random is False


def test_random_mode_defaults_the_rotation_seed_to_the_eval_seed():
    """Yixun 2026-08-10: rotation seed = eval seed. Then a cell's rotation
    assignment is a function of the seed it already reports."""
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, None, 44)
    assert plan.mode == "random"
    assert plan.rotate_seed == 44
    assert plan.rotate_deg is None      # no fixed angle applies -> recorded as null
    assert plan.is_random is True


def test_random_mode_accepts_an_explicit_rotation_seed():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 7, 42)
    assert plan.rotate_seed == 7        # explicit wins; eval seed is not used


def test_random_mode_resolves_a_string_eval_seed():
    """evaluate_model documents str seeds (it int()s them); the plan is resolved
    before that coercion, so it must do its own."""
    assert eval_FLAC.resolve_rotation_plan("random", 0.0, None, "43").rotate_seed == 43


def test_random_mode_with_nonzero_rotate_deg_raises():
    """A fixed angle and a per-sample draw are mutually exclusive protocols; a
    silent winner here is exactly the announcement-05 class of error."""
    with pytest.raises(ValueError, match="rotate_deg"):
        eval_FLAC.resolve_rotation_plan("random", 45.0, 42, 42)


def test_fixed_mode_with_explicit_rotate_seed_raises():
    """Review B4: an ignored --rotate-seed would make a manifest claim a rotation
    seed that never influenced anything."""
    with pytest.raises(ValueError, match="rotate_seed"):
        eval_FLAC.resolve_rotation_plan("fixed", 0.0, 42, 42)


def test_unknown_rotate_mode_raises():
    with pytest.raises(ValueError, match="rotate_mode"):
        eval_FLAC.resolve_rotation_plan("randon", 0.0, None, 42)


def test_random_mode_without_any_seed_raises():
    """No rotation seed and no eval seed -> unreproducible assignment; refuse."""
    with pytest.raises(ValueError, match="seed"):
        eval_FLAC.resolve_rotation_plan("random", 0.0, None, None)


# --------------------------------------------------------------------------- #
# (8) naming — injective across rotation seeds AND against fixed angles
# --------------------------------------------------------------------------- #
def test_rotation_token_and_suffix_fixed_are_the_legacy_renderings():
    for deg in (0.0, 45.0, 90.0, 5.625):
        assert eval_FLAC.rotation_token("fixed", deg, None) == eval_FLAC.rot_token(deg)
        assert eval_FLAC.rotation_suffix("fixed", deg, None) == eval_FLAC.rot_suffix(deg)
    assert eval_FLAC.rotation_suffix("fixed", 0.0, None) == ""       # byte-identical legacy
    assert eval_FLAC.rotation_suffix() == ""                          # all-default call


def test_rotation_token_and_suffix_random():
    assert eval_FLAC.rotation_token("random", None, 42) == "rand42"
    assert eval_FLAC.rotation_suffix("random", None, 42) == "_rotrand42"
    # suffix is always '_rot' + token (the eval-NAME analogue the kit interpolates)
    assert eval_FLAC.rotation_suffix("random", None, 43) == (
        "_rot" + eval_FLAC.rotation_token("random", None, 43))


def test_rotation_suffix_random_requires_a_resolved_seed():
    with pytest.raises(ValueError, match="seed"):
        eval_FLAC.rotation_suffix("random", None, None)


def test_random_paths_are_injective_across_rotation_seeds():
    """Review B5: reusing one eval name across rotation seeds must not overwrite —
    the exact failure class build_output_paths exists to prevent."""
    p42 = eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "exp14_C32_rgen_S40000_K8", cond_method="fa_invariant",
        n_angles=32, rotate_mode="random", rotate_seed=42)
    p43 = eval_FLAC.build_output_paths(
        CKPT, 1, 1.0, "exp14_C32_rgen_S40000_K8", cond_method="fa_invariant",
        n_angles=32, rotate_mode="random", rotate_seed=43)
    assert p42["metrics"] != p43["metrics"]
    assert p42["predictions"] != p43["predictions"]
    assert p42["metrics"].endswith("_fa_invariant_a32_rotrand42.json")
    assert p43["metrics"].endswith("_fa_invariant_a32_rotrand43.json")


def test_random_paths_are_injective_against_every_fixed_token():
    """'_rotrand<seed>' can never collide with a fixed '_rot<tok>': fixed tokens
    start with a digit or '-', so no fixed angle renders as 'rand...'."""
    names = set()
    for deg in (0.0, 42.0, 43.0, 45.0, 90.0, 5.625, 22.5, 180.0):
        names.add(eval_FLAC.build_output_paths(
            CKPT, 1, 1.0, "cell", rotate_deg=deg)["metrics"])
    n_fixed = len(names)
    for seed in (42, 43, 44, 45, 46):
        names.add(eval_FLAC.build_output_paths(
            CKPT, 1, 1.0, "cell", rotate_mode="random", rotate_seed=seed)["metrics"])
    assert len(names) == n_fixed + 5, "a random name collided with a fixed one"


def test_explicit_fixed_mode_paths_are_byte_identical_to_the_default_call():
    """Passing rotate_mode='fixed' explicitly may not change a single byte."""
    for cond_method, deg, n_angles in (("vanilla", 0.0, 4), ("vanilla", 45.0, 4),
                                       ("fa_invariant", 0.0, 4), ("fa_invariant", 45.0, 4)):
        default = eval_FLAC.build_output_paths(
            CKPT, 1, 1.0, "cell", cond_method=cond_method, rotate_deg=deg,
            n_angles=n_angles)
        explicit = eval_FLAC.build_output_paths(
            CKPT, 1, 1.0, "cell", cond_method=cond_method, rotate_deg=deg,
            n_angles=n_angles, rotate_mode="fixed", rotate_seed=None)
        assert default == explicit


# =========================================================================== #
# CYCLE 3 — the §3.3 assignment-integrity stream and its canonical hashes
# =========================================================================== #
# Golden canonical hashes, computed once from the SPECIFICATION's serialization
# (one JSON array per tuple, sort_keys=True, separators=(",", ":"), LF-joined,
# UTF-8, sha256 -- plan §3.3 / review B5), not from the implementation. They are
# the cross-machine contract: two cells agree iff these strings agree.
GOLDEN_INPUT_TUPLES = [
    [0, "0|sceneA/0000/S001_R000_hybrid_IR.wav", ["1.000000,2.000000,3.000000"], 512],
    [1, "1|sceneA/0000/S002_R000_hybrid_IR.wav",
        ["-1.500000,0.250000,0.000000", "4.000000,-2.000000,1.000000"], 512],
]
GOLDEN_ASSIGNMENT_TUPLES = [
    [0, "0|sceneA/0000/S001_R000_hybrid_IR.wav", 102],
    [1, "1|sceneA/0000/S002_R000_hybrid_IR.wav", 435],
]
GOLDEN_INPUT_HASH = "ca051d20f50751828872e1f7d70771db3c2f8d6ccc34cfc1953033975d5ffd8f"
GOLDEN_ASSIGNMENT_HASH = "903ced765558596534f74eb2c88509d608367e0c7dcbb85f4b8c3da95b9b48dd"
GOLDEN_EMPTY_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --------------------------------------------------------------------------- #
# (9) canonical hashing
# --------------------------------------------------------------------------- #
def test_canonical_stream_hash_matches_the_golden_values():
    assert eval_FLAC.canonical_stream_hash(GOLDEN_INPUT_TUPLES) == GOLDEN_INPUT_HASH
    assert eval_FLAC.canonical_stream_hash(GOLDEN_ASSIGNMENT_TUPLES) == GOLDEN_ASSIGNMENT_HASH
    assert eval_FLAC.canonical_stream_hash([]) == GOLDEN_EMPTY_HASH  # sha256 of b""


def test_canonical_stream_hash_is_order_sensitive():
    """Position IS the identity here: the same items in a different order are a
    different assignment, and the hash must say so."""
    assert eval_FLAC.canonical_stream_hash(
        list(reversed(GOLDEN_INPUT_TUPLES))) != GOLDEN_INPUT_HASH


def test_canonical_stream_hash_accepts_tuples_and_lists_alike():
    as_tuples = [tuple(t) for t in GOLDEN_ASSIGNMENT_TUPLES]
    assert eval_FLAC.canonical_stream_hash(as_tuples) == GOLDEN_ASSIGNMENT_HASH


def test_canonical_stream_hash_separates_neighbouring_fields():
    """The LF join and compact separators must not let two different streams
    serialize to the same bytes."""
    a = eval_FLAC.canonical_stream_hash([[0, "a"], [1, "b"]])
    b = eval_FLAC.canonical_stream_hash([[0, "a", 1, "b"]])
    assert a != b


# --------------------------------------------------------------------------- #
# per-sample identity extraction
# --------------------------------------------------------------------------- #
def test_sample_target_id_is_index_plus_relpath():
    md = _make_md(seed=3)
    md["idx"] = 17
    md["relpath"] = "sceneX/0002/S003_R004_hybrid_IR.wav"
    assert eval_FLAC.sample_target_id(md) == "17|sceneX/0002/S003_R004_hybrid_IR.wav"


def test_sample_target_id_falls_back_to_the_absolute_path():
    md = {"idx": 5, "path": "/data/AR/sceneX/0002/S003_R004_hybrid_IR.wav"}
    assert eval_FLAC.sample_target_id(md) == "5|/data/AR/sceneX/0002/S003_R004_hybrid_IR.wav"


def test_sample_target_id_without_any_path_raises():
    """An unidentifiable item makes the whole audit worthless -- fail, don't guess."""
    with pytest.raises(ValueError, match="identity"):
        eval_FLAC.sample_target_id({"idx": 5})


def test_sample_context_ids_are_ordered_and_position_derived():
    md = {"context_poses": torch.tensor([[1.0, 2.0, 3.0], [-1.5, 0.25, 0.0]])}
    assert eval_FLAC.sample_context_ids(md) == [
        "1.000000,2.000000,3.000000", "-1.500000,0.250000,0.000000"]
    # order is part of the identity
    flipped = {"context_poses": md["context_poses"].flip(0)}
    assert eval_FLAC.sample_context_ids(flipped) == list(
        reversed(eval_FLAC.sample_context_ids(md)))


def test_sample_context_ids_distinguish_different_context_draws():
    """AR_md draws the reference sources at random per sample; two different draws
    must not fingerprint identically, or cross-arm matching proves nothing."""
    a = {"context_poses": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    b = {"context_poses": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]])}
    assert eval_FLAC.sample_context_ids(a) != eval_FLAC.sample_context_ids(b)


def test_sample_context_ids_empty_when_absent():
    assert eval_FLAC.sample_context_ids({"idx": 0}) == []


def test_sample_context_ids_are_taken_before_rotation():
    """Rotation rewrites context_poses; a fingerprint taken afterwards would
    encode the rotation instead of the item and could never match a Z cell."""
    md = _make_md(seed=9)
    before = eval_FLAC.sample_context_ids(md)
    rotated = yr.rotate_scene_metadata(md, math.pi / 2, 512)
    assert eval_FLAC.sample_context_ids(rotated) != before


# --------------------------------------------------------------------------- #
# RotationStream accumulator
# --------------------------------------------------------------------------- #
def _stream_of(n=3, img_w=512, offsets=(102, 435, 348)):
    stream = eval_FLAC.RotationStream()
    for i in range(n):
        md = _make_md(seed=i, img_w=img_w)
        md["idx"] = i
        stream.record(md, offsets[i], img_w)
    return stream


def test_rotation_stream_records_positions_in_order():
    stream = _stream_of()
    assert len(stream) == 3
    assert [r.position for r in stream.rows] == [0, 1, 2]
    assert [r.offset for r in stream.rows] == [102, 435, 348]
    assert stream.img_w == 512
    assert stream.target_ids == [r.target_id for r in stream.rows]


def test_rotation_stream_hashes_use_the_canonical_serialization():
    stream = _stream_of()
    assert stream.input_hash() == eval_FLAC.canonical_stream_hash(stream.input_tuples())
    assert stream.assignment_hash() == eval_FLAC.canonical_stream_hash(
        stream.assignment_tuples())
    # input tuples carry (i, target, context ids, img_w); assignment (i, target, d)
    assert stream.input_tuples()[0] == [
        0, stream.rows[0].target_id, stream.rows[0].context_ids, 512]
    assert stream.assignment_tuples()[0] == [0, stream.rows[0].target_id, 102]


def test_rotation_stream_input_hash_is_offset_independent():
    """Z<->R pairing rests on this: the input hash must describe WHICH items and
    contexts were evaluated, and nothing about the rotation applied to them."""
    a = _stream_of(offsets=(102, 435, 348))
    b = _stream_of(offsets=(7, 8, 9))
    assert a.input_hash() == b.input_hash()
    assert a.assignment_hash() != b.assignment_hash()


def test_rotation_stream_detects_a_substituted_item():
    """SampleDataset silently substitutes a random item on a load/silence failure
    (dataset.py: `return self[random.randrange(len(self))]`). An offset-only hash
    would be blind to it; the identity-bearing input hash is not."""
    clean = _stream_of()
    swapped = eval_FLAC.RotationStream()
    for i, src in enumerate([0, 1, 99]):
        md = _make_md(seed=src)
        md["idx"] = src
        swapped.record(md, (102, 435, 348)[i], 512)
    assert swapped.input_hash() != clean.input_hash()
    assert swapped.assignment_hash() != clean.assignment_hash()


def test_rotation_stream_rejects_a_changing_panorama_width():
    stream = _stream_of()
    with pytest.raises(ValueError, match="img_w"):
        stream.record(_make_md(seed=4, img_w=256), 5, 256)


# --------------------------------------------------------------------------- #
# substitution guard: count enforcement
# --------------------------------------------------------------------------- #
def test_verify_stream_count_passes_on_the_expected_count():
    eval_FLAC.verify_stream_count(_stream_of(), 3)


@pytest.mark.parametrize("expected", [2, 4, 6337])
def test_verify_stream_count_raises_on_mismatch(expected):
    """FAILED, not a silent pass (plan §3.3): a short or long stream means the
    dataset did not deliver the split the cell claims to have evaluated."""
    with pytest.raises(RuntimeError, match="stream"):
        eval_FLAC.verify_stream_count(_stream_of(), expected)


def test_verify_stream_count_message_names_both_counts():
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.verify_stream_count(_stream_of(), 6337)
    assert "3" in str(exc.value) and "6337" in str(exc.value)


# =========================================================================== #
# CYCLE 4 — batch application, random-mode provenance, evaluate_model wiring
# =========================================================================== #
IMG_W = 512


def _batch(indices, img_w=IMG_W):
    out = []
    for i in indices:
        md = _make_md(seed=i, img_w=img_w)
        md["idx"] = i
        out.append(md)
    return out


# --------------------------------------------------------------------------- #
# apply_rotation_plan — fixed mode is the legacy code path, unchanged
# --------------------------------------------------------------------------- #
def test_apply_rotation_plan_fixed_zero_is_a_no_op_on_the_same_object():
    """rotate_deg 0 must not even touch the metadata (and must not require a
    'depth' key -- the legacy branch never looked at one)."""
    plan = eval_FLAC.resolve_rotation_plan("fixed", 0.0, None, 42)
    md_list = [{"scene": "a"}, {"scene": "b"}]
    assert eval_FLAC.apply_rotation_plan(md_list, plan) is md_list


def test_apply_rotation_plan_fixed_nonzero_matches_the_legacy_comprehension():
    plan = eval_FLAC.resolve_rotation_plan("fixed", 45.0, None, 42)
    batch = _batch([0, 1, 2])
    got = eval_FLAC.apply_rotation_plan(batch, plan)
    want = [yr.rotate_scene_metadata(md, math.radians(45.0), IMG_W) for md in batch]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        for key in ("depth",) + yr.POSE_KEYS:
            assert torch.equal(g[key], w[key]), key


def test_apply_rotation_plan_fixed_never_consumes_randomness():
    """A fixed rotation draws nothing, so the generator must come back untouched.

    (Fixed mode DOES fill a supplied stream -- with the run's constant column
    shift -- since the F2 round-1 fix; that contract is pinned separately below.)
    """
    plan = eval_FLAC.resolve_rotation_plan("fixed", 45.0, None, 42)
    g = _gen(42)
    before = g.get_state()
    eval_FLAC.apply_rotation_plan(_batch([0, 1]), plan, g, eval_FLAC.RotationStream())
    assert torch.equal(before, g.get_state()), "fixed mode consumed randomness"


# --------------------------------------------------------------------------- #
# apply_rotation_plan — random mode
# --------------------------------------------------------------------------- #
def test_apply_rotation_plan_random_applies_the_drawn_per_sample_angles():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    batch = _batch([0, 1, 2, 3])
    stream = eval_FLAC.RotationStream()
    got = eval_FLAC.apply_rotation_plan(batch, plan, _gen(42), stream)

    expected_offsets = GOLDEN_SEED42_W512[:4]
    assert stream.offsets == expected_offsets
    for md, d, out in zip(batch, expected_offsets, got):
        want = yr.rotate_scene_metadata(md, yr.offsets_to_radians([d], IMG_W)[0], IMG_W)
        for key in ("depth",) + yr.POSE_KEYS:
            assert torch.equal(out[key], want[key]), key


def test_apply_rotation_plan_random_does_not_mutate_the_input():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    batch = _batch([0, 1])
    ref = [md["depth"].clone() for md in batch]
    eval_FLAC.apply_rotation_plan(batch, plan, _gen(42), eval_FLAC.RotationStream())
    for md, r in zip(batch, ref):
        assert torch.equal(md["depth"], r)


def test_apply_rotation_plan_random_records_pre_rotation_identity():
    """The recorded context fingerprint must describe the item, not the rotation:
    otherwise an R cell could never match its unrotated Z pair."""
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    batch = _batch([0, 1, 2])
    stream = eval_FLAC.RotationStream()
    eval_FLAC.apply_rotation_plan(batch, plan, _gen(42), stream)
    for md, row in zip(batch, stream.rows):
        assert row.target_id == eval_FLAC.sample_target_id(md)
        assert row.context_ids == eval_FLAC.sample_context_ids(md)
        assert row.img_w == IMG_W


def test_apply_rotation_plan_random_stream_is_batch_partition_independent():
    """(2) again, but through the real application helper: 4+4 == 8 == 3+3+2."""
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)

    def run(sizes):
        stream = eval_FLAC.RotationStream()
        g = _gen(42)
        start = 0
        for n in sizes:
            eval_FLAC.apply_rotation_plan(_batch(range(start, start + n)), plan, g, stream)
            start += n
        return stream

    flat = run([8])
    assert flat.offsets == GOLDEN_SEED42_W512[:8]
    for sizes in ([4, 4], [3, 3, 2], [1] * 8):
        s = run(sizes)
        assert s.offsets == flat.offsets
        assert s.assignment_hash() == flat.assignment_hash()
        assert s.input_hash() == flat.input_hash()


def test_apply_rotation_plan_random_does_not_touch_global_rng():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    torch.manual_seed(99)
    before = torch.random.get_rng_state()
    eval_FLAC.apply_rotation_plan(_batch([0, 1, 2]), plan, _gen(42),
                                  eval_FLAC.RotationStream())
    assert torch.equal(before, torch.random.get_rng_state())


def test_apply_rotation_plan_random_requires_generator_and_stream():
    """Fail closed: a missing generator would fall back to the global RNG and a
    missing stream would run an unauditable assignment."""
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    with pytest.raises(ValueError):
        eval_FLAC.apply_rotation_plan(_batch([0]), plan, None, eval_FLAC.RotationStream())
    with pytest.raises(ValueError):
        eval_FLAC.apply_rotation_plan(_batch([0]), plan, _gen(42), None)


def test_apply_rotation_plan_random_requires_a_depth_panorama():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    with pytest.raises(ValueError, match="depth"):
        eval_FLAC.apply_rotation_plan([{"idx": 0, "relpath": "a.wav"}], plan, _gen(42),
                                      eval_FLAC.RotationStream())


# --------------------------------------------------------------------------- #
# (10b) integration spy: the drawn offsets are the ones that reach the rotation
# --------------------------------------------------------------------------- #
def test_drawn_offsets_reach_rotate_scene_metadata_over_a_two_batch_stream(monkeypatch):
    """Gate G3's pytest half. A correct draw that never reaches the rotation would
    produce a perfectly self-consistent -- and completely unrotated -- cell."""
    seen = []
    real = eval_FLAC.rotate_scene_metadata

    def spy(md, alpha_rad, img_w, **kwargs):
        seen.append((alpha_rad, img_w))
        return real(md, alpha_rad, img_w, **kwargs)

    monkeypatch.setattr(eval_FLAC, "rotate_scene_metadata", spy)

    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    stream = eval_FLAC.RotationStream()
    g = _gen(42)
    eval_FLAC.apply_rotation_plan(_batch(range(0, 8)), plan, g, stream)
    eval_FLAC.apply_rotation_plan(_batch(range(8, 16)), plan, g, stream)

    expected_angles = yr.offsets_to_radians(GOLDEN_SEED42_W512, IMG_W)
    assert [a for a, _ in seen] == expected_angles
    assert {w for _, w in seen} == {IMG_W}
    assert stream.offsets == GOLDEN_SEED42_W512
    assert len(stream) == 16


# --------------------------------------------------------------------------- #
# random-mode provenance in the metrics record / predictions meta
# --------------------------------------------------------------------------- #
def _random_record(**over):
    kwargs = dict(
        cond_autocast="bf16", batch_size=64, n_samples=6337,
        dataset_config="ds.json", seed=42, cfg_scale=1.0, steps=1,
        eval_name="exp14_C32_rgen_S40000_s42_K8", weights_source="ema", device="cuda",
        rotate_mode="random", rotate_seed=42, input_hash="a" * 64,
        assignment_hash="b" * 64, stream_count=6337, img_w=512,
    )
    kwargs.update(over)
    return eval_FLAC.build_metrics_record(
        {"T60": 1.0}, CKPT, 0.0, "fa_invariant", [0.0, 90.0], **kwargs)


def test_random_record_nulls_rotate_deg_and_carries_the_provenance():
    """rotate_deg must be null, never 0.0: a randomly-rotated cell and an
    unrotated one must not be readable as the same protocol."""
    rec = _random_record()
    assert rec["rotate_deg"] is None
    assert rec["rotate_mode"] == "random"
    assert rec["rotate_seed"] == 42
    assert rec["input_hash"] == "a" * 64
    assert rec["assignment_hash"] == "b" * 64
    assert rec["stream_count"] == 6337
    assert rec["img_w"] == 512
    assert json.loads(json.dumps(rec))["rotate_deg"] is None


def test_random_record_keeps_the_legacy_keys_and_their_order():
    """The random keys are APPENDED; every legacy key keeps its position, so a
    collector can read both row generations with one schema."""
    rec = _random_record()
    legacy = [k for k in rec if k not in ("rotate_mode", "rotate_seed", "input_hash",
                                          "assignment_hash", "stream_count", "img_w")]
    reference = eval_FLAC.build_metrics_record({"T60": 1.0}, CKPT, 0.0, "vanilla", None)
    assert legacy == list(reference.keys())
    assert list(rec)[-6:] == ["rotate_mode", "rotate_seed", "input_hash",
                              "assignment_hash", "stream_count", "img_w"]


def test_random_predictions_meta_carries_the_same_provenance():
    """Only written under --store_predictions (not this campaign), but a stored
    prediction set must still name the assignment that produced it (review B5)."""
    meta = eval_FLAC.build_predictions_meta(
        "ds.json", 42, 6337, "vanilla", None, 0.0, 64, "bf16",
        rotate_mode="random", rotate_seed=43, input_hash="c" * 64,
        assignment_hash="d" * 64, stream_count=6337, img_w=512,
    )
    assert meta["rotate_deg"] is None
    assert meta["rotate_mode"] == "random" and meta["rotate_seed"] == 43
    assert meta["input_hash"] == "c" * 64 and meta["assignment_hash"] == "d" * 64
    assert meta["stream_count"] == 6337 and meta["img_w"] == 512
    json.loads(json.dumps(meta))


# --------------------------------------------------------------------------- #
# evaluate_model wiring
# --------------------------------------------------------------------------- #
def test_evaluate_model_rotation_guard_fires_before_any_work(tmp_path, monkeypatch):
    """The guard must trip before file/model/dataloader work: every path below is
    nonexistent, so late validation would surface FileNotFoundError instead."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"),
    )
    with pytest.raises(ValueError, match="rotate_deg"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", rotate_mode="random", rotate_deg=45.0)
    with pytest.raises(ValueError, match="rotate_seed"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", rotate_mode="fixed", rotate_seed=42)
    with pytest.raises(ValueError, match="rotate_mode"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", rotate_mode="randon")


class _EmptyLoader:
    """An empty dataloader that still reports its (empty) dataset, so the eval
    loop is skipped without any model, GPU or data while the stream guard still
    has an expected count to check against."""
    dataset = []

    def __iter__(self):
        return iter([])


def _stub_eval_stack(monkeypatch, tmp_path):
    import types
    model_cfg = tmp_path / "model.json"
    model_cfg.write_text(json.dumps({
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1, "training": {"use_ema": False},
    }))
    dataset_cfg = tmp_path / "dataset.json"
    dataset_cfg.write_text(json.dumps({"datasets": [{"id": "toy"}]}))
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": {}}, str(ckpt))

    class _FakeModule:
        def __init__(self):
            self.diffusion = types.SimpleNamespace(
                model=object(), pretransform=None, conditioner=None)
            self.device = "cpu"

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda cfg: types.SimpleNamespace(load_state_dict=lambda sd, strict=False: ([], [])))
    monkeypatch.setattr(
        eval_FLAC, "create_training_wrapper_from_config", lambda cfg, model: _FakeModule())
    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config",
                        lambda *a, **k: _EmptyLoader())
    monkeypatch.setattr(
        eval_FLAC, "create_metric_callback_from_config",
        lambda *a, **k: types.SimpleNamespace(
            update_metrics=lambda *a, **k: None,
            compute_metrics=lambda split: {"T60": 1.0}))
    return model_cfg, dataset_cfg, ckpt


def test_evaluate_model_random_mode_writes_the_rotrand_artifact_with_provenance(
        tmp_path, monkeypatch):
    """End-to-end wiring on an empty split: the random plan must reach BOTH the
    filename and the record, or a cell's provenance would describe a protocol it
    did not run."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_wiring_K8", seed=44, rotate_mode="random")

    out = tmp_path / "toy_metrics_1_1.0_exp14_wiring_K8_rotrand44.json"
    assert out.exists(), "random-mode artifact not written at the _rotrand<seed> path"
    saved = json.loads(out.read_text())
    assert saved["rotate_deg"] is None
    assert saved["rotate_mode"] == "random"
    assert saved["rotate_seed"] == 44          # defaulted to the eval seed
    assert saved["stream_count"] == 0          # empty split, verified against it
    assert len(saved["input_hash"]) == 64 and len(saved["assignment_hash"]) == 64


def test_evaluate_model_fixed_mode_artifact_has_no_random_keys(tmp_path, monkeypatch):
    """The default invocation is untouched: legacy filename, legacy record."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_wiring_K8", seed=44)

    out = tmp_path / "toy_metrics_1_1.0_exp14_wiring_K8.json"
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["rotate_deg"] == 0.0
    for key in ("rotate_mode", "rotate_seed", "input_hash", "assignment_hash",
                "stream_count", "img_w"):
        assert key not in saved


def test_evaluate_model_random_mode_fails_on_a_short_stream(tmp_path, monkeypatch):
    """The substitution guard must fail the RUN, not warn: a stream shorter than
    the split means the reported metrics are not the split's metrics."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)

    class _ClaimsMoreItems(_EmptyLoader):
        dataset = [None] * 6337

    monkeypatch.setattr(eval_FLAC, "create_dataloader_from_config",
                        lambda *a, **k: _ClaimsMoreItems())
    with pytest.raises(RuntimeError, match="stream"):
        eval_FLAC.evaluate_model(
            str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
            device="cpu", eval_name="exp14_short", seed=42, rotate_mode="random")
    assert not (tmp_path / "toy_metrics_1_1.0_exp14_short_rotrand42.json").exists()


def test_cli_exposes_the_rotate_mode_and_seed_flags():
    """argparse enforces --rotate-mode's choices; --rotate-seed takes an int."""
    import subprocess
    import sys
    root = str(Path(__file__).resolve().parents[2])
    bad = subprocess.run([sys.executable, "eval_FLAC.py", "--rotate-mode", "randon"],
                         capture_output=True, cwd=root)
    assert bad.returncode == 2 and b"invalid choice" in bad.stderr
    good = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--rotate-mode", "random", "--rotate-seed", "42"],
        capture_output=True, cwd=root)
    assert good.returncode == 2                      # still missing required args...
    assert b"invalid choice" not in good.stderr      # ...but the flags parsed


# =========================================================================== #
# ROUND-1 FIX BATCH — F4: the context fingerprint is dtype-fragile, so pin it
# =========================================================================== #
# Codex code review N4 measured it on the real 6,415 unseen source/receiver pairs:
# rendering the SAME positions as float64 changed two six-decimal strings and
# float16 changed 5,032. The fingerprint is only stable because the loader happens
# to hand us float32, so that assumption is now asserted rather than assumed.
def test_context_fingerprint_schema_is_versioned():
    assert eval_FLAC.CONTEXT_FINGERPRINT_SCHEMA == 1


def test_sample_context_ids_rejects_non_float32_dtype():
    for dtype in (torch.float64, torch.float16):
        md = {"context_poses": torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)}
        with pytest.raises(ValueError, match="float32"):
            eval_FLAC.sample_context_ids(md)


def test_sample_context_ids_rejects_non_finite_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        md = {"context_poses": torch.tensor([[1.0, 2.0, bad]], dtype=torch.float32)}
        with pytest.raises(ValueError, match="finite"):
            eval_FLAC.sample_context_ids(md)


@pytest.mark.parametrize("shape", [(3,), (2,), (1, 4), (2, 2), (0, 3), (1, 2, 3)])
def test_sample_context_ids_rejects_wrong_shapes(shape):
    md = {"context_poses": torch.zeros(shape, dtype=torch.float32)}
    with pytest.raises(ValueError, match="shape"):
        eval_FLAC.sample_context_ids(md)


def test_sample_context_ids_still_accepts_the_real_loader_shape():
    md = {"context_poses": torch.tensor([[1.0, 2.0, 3.0], [-1.5, 0.25, 0.0]],
                                        dtype=torch.float32)}
    assert eval_FLAC.sample_context_ids(md) == [
        "1.000000,2.000000,3.000000", "-1.500000,0.250000,0.000000"]


# =========================================================================== #
# ROUND-1 FIX BATCH — F3: position i must carry dataset item i (accepted ruling 3)
# =========================================================================== #
# With shuffle=False and a sequential sampler, stream position i IS dataset index
# i. The only way that breaks is SampleDataset's recursive substitution on a
# load/silence failure -- which is precisely the event the audit exists to catch,
# and which a hash alone can only reveal by comparison against another cell.
def test_stream_rows_carry_the_dataset_index():
    stream = _stream_of()
    assert [r.dataset_idx for r in stream.rows] == [0, 1, 2]


def test_verify_stream_positions_passes_on_a_sequential_stream():
    eval_FLAC.verify_stream_positions(_stream_of())


def test_verify_stream_positions_detects_a_substituted_item():
    stream = eval_FLAC.RotationStream()
    for position, idx in enumerate([0, 1, 4813, 3]):
        md = _make_md(seed=idx)
        md["idx"] = idx
        stream.record(md, 0, IMG_W)
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.verify_stream_positions(stream)
    msg = str(exc.value)
    assert "2" in msg and "4813" in msg          # first offender, named
    assert "3" not in msg.split("4813")[0]       # reported before any later row


def test_verify_stream_positions_requires_an_index():
    """An item with no 'idx' cannot be checked, so it is not silently accepted."""
    stream = eval_FLAC.RotationStream()
    md = _make_md(seed=0)
    md.pop("idx", None)
    stream.record(md, 0, IMG_W)
    with pytest.raises(RuntimeError, match="idx"):
        eval_FLAC.verify_stream_positions(stream)


def test_evaluate_model_random_mode_runs_the_position_check(tmp_path, monkeypatch):
    """Wiring proof: the check must run on the real path, not just be available."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(eval_FLAC, "verify_stream_positions", lambda s: calls.append(s))
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_poscheck", seed=42, rotate_mode="random")
    assert len(calls) == 1

    calls.clear()
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_poscheck_fixed", seed=42)
    assert calls == [], "fixed mode without --record-stream accumulates no stream"


# =========================================================================== #
# ROUND-1 FIX BATCH — F1: the count check must be pre-registered, not tautological
# =========================================================================== #
# Codex code review B1: comparing the stream against len(dataset) compares the run
# with itself. A zero-item dataset produced a perfectly "valid" random artifact.
# The campaign's 6,337 is a PRE-REGISTERED number and has to be asserted as one.
def test_verify_stream_count_accepts_a_matching_expected_count():
    eval_FLAC.verify_stream_count(_stream_of(), 3, 3)


def test_verify_stream_count_rejects_an_empty_dataset_against_the_expectation():
    """The exact hole the review found: 0 == 0 used to pass."""
    empty = eval_FLAC.RotationStream()
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.verify_stream_count(empty, 0, 6337)
    assert "0" in str(exc.value) and "6337" in str(exc.value)


def test_verify_stream_count_rejects_a_self_consistent_wrong_size_split():
    """Stream and dataset agree with each other and BOTH disagree with the
    pre-registered split -- e.g. a subsampled eval config (announcement 01)."""
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.verify_stream_count(_stream_of(), 3, 6337)
    assert "3" in str(exc.value) and "6337" in str(exc.value)


def test_verify_stream_count_rejects_a_nonpositive_expectation():
    with pytest.raises(RuntimeError, match="expected"):
        eval_FLAC.verify_stream_count(_stream_of(), 3, 0)


def test_verify_stream_count_without_an_expectation_keeps_the_old_check():
    eval_FLAC.verify_stream_count(_stream_of(), 3)
    with pytest.raises(RuntimeError, match="stream"):
        eval_FLAC.verify_stream_count(_stream_of(), 4)


def test_evaluate_model_enforces_the_expected_stream_count(tmp_path, monkeypatch):
    """End of the tautology on the real path: the empty stub split that used to
    produce a valid artifact is now rejected, and nothing is written."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.evaluate_model(
            str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
            device="cpu", eval_name="exp14_expcount", seed=42, rotate_mode="random",
            expected_stream_count=6337)
    assert "6337" in str(exc.value)
    assert not (tmp_path / "toy_metrics_1_1.0_exp14_expcount_rotrand42.json").exists()


def test_expected_stream_count_without_a_stream_is_an_error(tmp_path, monkeypatch):
    """Never silently ignored (the review-B4 class): in a plain fixed-mode run no
    stream is accumulated, so the expectation could not be checked at all."""
    monkeypatch.setattr(
        eval_FLAC, "create_model_from_config",
        lambda *a, **k: pytest.fail("evaluate_model reached model construction"))
    with pytest.raises(ValueError, match="expected_stream_count"):
        eval_FLAC.evaluate_model(
            str(tmp_path / "m.json"), str(tmp_path / "d.json"), str(tmp_path / "c.ckpt"),
            steps=1, cfg_scale=1.0, device="cpu", expected_stream_count=6337)


def test_cli_exposes_the_expected_stream_count_flag():
    import subprocess
    import sys
    root = str(Path(__file__).resolve().parents[2])
    bad = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--expected-stream-count", "many"],
        capture_output=True, cwd=root)
    assert bad.returncode == 2 and b"invalid int value" in bad.stderr
    good = subprocess.run(
        [sys.executable, "eval_FLAC.py", "--expected-stream-count", "6337"],
        capture_output=True, cwd=root)
    assert good.returncode == 2 and b"invalid int value" not in good.stderr


# =========================================================================== #
# ROUND-1 FIX BATCH — F2: the opt-in .stream.json sidecar (review B2, ruling 1)
# =========================================================================== #
# A hash without its preimage is unfalsifiable: two cells can only be declared
# rotation-matched, never diagnosed when they are not. The sidecar carries the
# full canonical tuples and the offsets, so a mismatch can be localised to a
# position. It is a SEPARATE file precisely so the metrics record and its path
# stay byte-identical in both modes (the STEP-0 snapshots are untouched by F2).
def test_yaw_column_shift_is_the_quantisation_rotate_scene_metadata_applies():
    """The recorded fixed-mode offset must be the column shift actually applied,
    so both come from one function rather than two copies of a rule."""
    assert yr.yaw_column_shift(0.0, 512) == 0
    assert yr.yaw_column_shift(math.radians(90.0), 512) == 128
    assert yr.yaw_column_shift(math.radians(45.0), 512) == 64
    assert yr.yaw_column_shift(math.radians(5.625), 512) == 8
    assert yr.yaw_column_shift(math.radians(360.0), 512) == 0     # wraps
    assert yr.yaw_column_shift(math.radians(-90.0), 512) == 384   # wraps

    # ...and it is EXACTLY what rotate_scene_metadata rolls by: reconstructing the
    # rotation from yaw_column_shift alone must reproduce the real output bit for
    # bit, at angles on and off the column grid.
    depth = _make_md(seed=2)["depth"]
    for deg in (0.0, 45.0, 90.0, 5.625, 200.0, 37.3):
        alpha = math.radians(deg)
        dj = yr.yaw_column_shift(alpha, IMG_W)
        rot = yr.azimuth_rotation_matrix(dj * 2.0 * math.pi / IMG_W)
        expected = torch.einsum("ij,jhw->ihw", rot.to(depth.dtype),
                                torch.roll(depth, shifts=dj, dims=2))
        got = yr.rotate_scene_metadata({"depth": depth}, alpha, IMG_W)["depth"]
        assert torch.equal(got, expected), f"{deg} deg"


def test_stream_sidecar_path_sits_next_to_the_metrics_json():
    metrics = "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp14_C32_rgen_rotrand42.json"
    assert eval_FLAC.stream_sidecar_path(metrics) == (
        "weights/FLAC/FLAC_EMA_metrics_1_1.0_exp14_C32_rgen_rotrand42.stream.json")


def test_build_stream_record_payload_is_complete_and_self_verifying():
    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    stream = _stream_of()
    rec = eval_FLAC.build_stream_record(plan, stream)

    assert rec["schema_version"] == 1
    assert rec["fingerprint_schema"] == eval_FLAC.CONTEXT_FINGERPRINT_SCHEMA
    assert rec["rotate_mode"] == "random"
    assert rec["rotate_seed"] == 42
    assert rec["rotate_deg"] is None
    assert rec["img_w"] == 512
    assert rec["stream_count"] == 3
    assert rec["offsets"] == [102, 435, 348]
    assert rec["input_tuples"] == stream.input_tuples()
    assert rec["assignment_tuples"] == stream.assignment_tuples()
    # the digests must be recomputable from the stored preimages -- that is the
    # entire point of storing them (review B2).
    assert eval_FLAC.canonical_stream_hash(rec["input_tuples"]) == rec["input_hash"]
    assert eval_FLAC.canonical_stream_hash(
        rec["assignment_tuples"]) == rec["assignment_hash"]
    json.loads(json.dumps(rec))


def test_build_stream_record_fixed_mode_nulls_the_seed_and_keeps_the_angle():
    plan = eval_FLAC.resolve_rotation_plan("fixed", 90.0, None, 42)
    rec = eval_FLAC.build_stream_record(plan, _stream_of(offsets=(128, 128, 128)))
    assert rec["rotate_mode"] == "fixed"
    assert rec["rotate_seed"] is None
    assert rec["rotate_deg"] == 90.0
    assert rec["offsets"] == [128, 128, 128]


# --------------------------------------------------------------------------- #
# fixed-mode stream accumulation (only under --record-stream)
# --------------------------------------------------------------------------- #
def test_apply_rotation_plan_fixed_records_the_constant_column_shift():
    plan = eval_FLAC.resolve_rotation_plan("fixed", 90.0, None, 42)
    batch = _batch([0, 1, 2])
    stream = eval_FLAC.RotationStream()
    got = eval_FLAC.apply_rotation_plan(batch, plan, None, stream)
    assert stream.offsets == [128, 128, 128]      # 90 deg == 128 columns of 512
    assert [r.dataset_idx for r in stream.rows] == [0, 1, 2]
    # ...and the rotation itself is unchanged by the recording
    want = [yr.rotate_scene_metadata(md, math.radians(90.0), IMG_W) for md in batch]
    for g, w in zip(got, want):
        assert torch.equal(g["depth"], w["depth"])


def test_apply_rotation_plan_fixed_zero_records_zero_offsets_without_rotating():
    plan = eval_FLAC.resolve_rotation_plan("fixed", 0.0, None, 42)
    batch = _batch([0, 1])
    stream = eval_FLAC.RotationStream()
    got = eval_FLAC.apply_rotation_plan(batch, plan, None, stream)
    assert stream.offsets == [0, 0]
    assert got is batch, "an unrotated run must still hand back the same objects"


def test_apply_rotation_plan_fixed_zero_without_a_stream_is_still_a_pure_no_op():
    """The legacy path must not acquire a 'depth' requirement (F2 regression)."""
    plan = eval_FLAC.resolve_rotation_plan("fixed", 0.0, None, 42)
    md_list = [{"scene": "a"}]
    assert eval_FLAC.apply_rotation_plan(md_list, plan) is md_list


# --------------------------------------------------------------------------- #
# evaluate_model wiring
# --------------------------------------------------------------------------- #
def test_evaluate_model_writes_the_sidecar_only_when_asked(tmp_path, monkeypatch):
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_nosidecar", seed=42, rotate_mode="random")
    assert not (tmp_path / "toy_metrics_1_1.0_exp14_nosidecar_rotrand42.stream.json").exists()

    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_sidecar", seed=42, rotate_mode="random",
        record_stream=True)
    side = tmp_path / "toy_metrics_1_1.0_exp14_sidecar_rotrand42.stream.json"
    assert side.exists()
    payload = json.loads(side.read_text())
    assert payload["schema_version"] == 1
    assert payload["rotate_mode"] == "random" and payload["rotate_seed"] == 42
    assert payload["rotate_deg"] is None
    assert eval_FLAC.canonical_stream_hash(
        payload["input_tuples"]) == payload["input_hash"]


def test_evaluate_model_fixed_mode_sidecar_leaves_the_metrics_bytes_alone(
        tmp_path, monkeypatch):
    """--record-stream may not perturb a single byte of the fixed-mode record."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(eval_FLAC, "source_sha", lambda: "STUB")

    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_zref", seed=42)
    plain = (tmp_path / "toy_metrics_1_1.0_exp14_zref.json").read_text()

    eval_FLAC.evaluate_model(
        str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
        device="cpu", eval_name="exp14_zref", seed=42, record_stream=True)
    with_flag = (tmp_path / "toy_metrics_1_1.0_exp14_zref.json").read_text()
    assert with_flag == plain

    side = tmp_path / "toy_metrics_1_1.0_exp14_zref.stream.json"
    assert side.exists()
    payload = json.loads(side.read_text())
    assert payload["rotate_mode"] == "fixed"
    assert payload["rotate_seed"] is None and payload["rotate_deg"] == 0.0


def test_evaluate_model_no_sidecar_when_validation_fails(tmp_path, monkeypatch):
    """Written only AFTER validation: a rejected cell leaves no artifact at all."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        eval_FLAC.evaluate_model(
            str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
            device="cpu", eval_name="exp14_badcount", seed=42, rotate_mode="random",
            record_stream=True, expected_stream_count=6337)
    assert not (tmp_path / "toy_metrics_1_1.0_exp14_badcount_rotrand42.stream.json").exists()
    assert not (tmp_path / "toy_metrics_1_1.0_exp14_badcount_rotrand42.json").exists()


def test_expected_stream_count_is_allowed_in_fixed_mode_under_record_stream(
        tmp_path, monkeypatch):
    """--record-stream makes a fixed-mode (Z) cell countable, so the expectation
    becomes checkable there too -- and is enforced."""
    model_cfg, dataset_cfg, ckpt = _stub_eval_stack(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as exc:
        eval_FLAC.evaluate_model(
            str(model_cfg), str(dataset_cfg), str(ckpt), steps=1, cfg_scale=1.0,
            device="cpu", eval_name="exp14_zcount", seed=42, record_stream=True,
            expected_stream_count=6337)
    assert "6337" in str(exc.value)


def test_cli_exposes_the_record_stream_flag():
    import subprocess
    import sys
    root = str(Path(__file__).resolve().parents[2])
    good = subprocess.run([sys.executable, "eval_FLAC.py", "--record-stream"],
                          capture_output=True, cwd=root)
    assert good.returncode == 2
    assert b"unrecognized arguments" not in good.stderr


# =========================================================================== #
# ROUND-1 FIX BATCH — F5: prove the ordering through a REAL multi-worker loader
# =========================================================================== #
# Review N5: every other test drives the helper directly or through an empty stub
# loader, so worker-process ordering was argued (PyTorch 2.7 defaults to
# in_order=True; the repo pins shuffle=False / drop_last=false) rather than
# observed. This observes it: two worker processes, a ragged batch split, and the
# assertion that the golden seed-42 offsets land on items 0..7 in sampler order.
class _TinyEvalDataset(torch.utils.data.Dataset):
    """8 in-memory items shaped like the AR eval loader's output.

    Module-level (not a closure) so it survives pickling to worker processes, and
    the tensors are built per __getitem__ so the workers copy a description rather
    than ~200 kB of panoramas.
    """

    def __init__(self, n=8, img_w=IMG_W, height=4):
        self.n, self.img_w, self.height = n, img_w, height

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(1000 + idx)
        info = {
            "idx": idx,
            "relpath": f"sceneA/000{idx}/S00{idx}_R000_hybrid_IR.wav",
            "scene": "sceneA",
            "context_poses": torch.randn(2, 3, generator=g),
            "depth": torch.randn(3, self.height, self.img_w, generator=g),
        }
        return torch.zeros(1, 16), info


def test_random_offsets_attach_in_sampler_order_through_a_multiworker_dataloader():
    from src.data.dataset import collation_fn   # the eval loader's own collate

    dataset = _TinyEvalDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, shuffle=False, drop_last=False, num_workers=2,
        collate_fn=collation_fn,
    )

    plan = eval_FLAC.resolve_rotation_plan("random", 0.0, 42, 42)
    stream = eval_FLAC.RotationStream()
    g = _gen(42)
    batch_sizes = []
    for reals, metadata in loader:
        batch_sizes.append(len(metadata))
        eval_FLAC.apply_rotation_plan(metadata, plan, g, stream)

    assert batch_sizes == [3, 3, 2], "ragged tail batch expected (drop_last=False)"
    # the pre-registered assignment, item by item, across a worker boundary
    assert stream.offsets == GOLDEN_SEED42_W512[:8]
    assert [r.dataset_idx for r in stream.rows] == list(range(8))
    assert [r.target_id for r in stream.rows] == [
        f"{i}|sceneA/000{i}/S00{i}_R000_hybrid_IR.wav" for i in range(8)]
    # and both guards accept the stream this really produced
    eval_FLAC.verify_stream_positions(stream)
    eval_FLAC.verify_stream_count(stream, len(dataset), 8)


# --------------------------------------------------------------------------- #
# round-3 fix R3F1 — opt-in per-scene recording (the plan §4 estimand)
#
# The campaign's per-seed observation is the PER-SCENE mean (plan §4, and the
# repo convention: paper headline numbers average per-scene results). The metric
# callback can already produce it -- AcousticMetricsCallback(eval_per_scene=True)
# accumulates a second, per-scene set of metric objects and returns them under
# `by_scene` -- but eval_FLAC never asked for it, so no committed metrics JSON
# carries one. These tests pin the flag that asks, and the record shape it adds.
# --------------------------------------------------------------------------- #
def test_per_scene_block_is_absent_without_the_flag():
    """Legacy records are FROZEN: no new key appears unless the flag is passed."""
    record = eval_FLAC.build_metrics_record(
        {"T60": 9.0}, "/ckpt/epoch=8-step=40000.ckpt", 0.0, "vanilla", None)
    for key in ("by_scene", "per_scene_schema", "scene_count"):
        assert key not in record, f"{key} appeared in a record that never asked for it"


def test_per_scene_block_is_added_only_when_recorded():
    by_scene = {"Cafe/Cafe_idx_1": {"T60": 8.0, "RIR_to_GT_RIR_R@1": 5.0},
                "Office/Office_idx_10": {"T60": 10.0, "RIR_to_GT_RIR_R@1": 7.0}}
    record = eval_FLAC.build_metrics_record(
        {"T60": 9.0}, "/ckpt/epoch=8-step=40000.ckpt", 0.0, "vanilla", None,
        by_scene=by_scene)
    assert record["by_scene"] == by_scene
    assert record["scene_count"] == 2
    assert record["per_scene_schema"] == eval_FLAC.PER_SCENE_SCHEMA
    # the flat metrics block keeps its legacy shape: every consumer that reads
    # record["metrics"][key] as a number must keep working
    assert all(isinstance(v, (int, float)) for v in record["metrics"].values())


def test_per_scene_recording_refuses_an_empty_scene_map():
    """A run that asked for per-scene results and produced none did not measure
    the estimand; recording {} would publish that silence as a fact."""
    with pytest.raises(ValueError):
        eval_FLAC.build_metrics_record(
            {"T60": 9.0}, "/ckpt/epoch=8-step=40000.ckpt", 0.0, "vanilla", None,
            by_scene={})


def test_split_per_scene_block_separates_the_nested_map_from_flat_metrics():
    """compute_metrics returns the per-scene map INSIDE the metrics dict; the
    record keeps `metrics` flat and numeric, so the nested map is lifted out."""
    computed = {"T60": 9.0, "C50": 1.0,
                "by_scene": {"Cafe/Cafe_idx_1": {"T60": 8.0}}}
    flat, by_scene = eval_FLAC.split_per_scene_metrics(computed)
    assert flat == {"T60": 9.0, "C50": 1.0}
    assert by_scene == {"Cafe/Cafe_idx_1": {"T60": 8.0}}
    assert "by_scene" not in flat
    # and without a per-scene block the metrics pass through untouched
    flat2, none2 = eval_FLAC.split_per_scene_metrics({"T60": 9.0})
    assert flat2 == {"T60": 9.0} and none2 is None


def test_record_per_scene_is_a_cli_flag_that_reaches_the_callback():
    """--record-per-scene is what turns eval_per_scene on; the factory already
    takes per_scene, so the flag must be threaded through to it."""
    import inspect
    src = inspect.getsource(eval_FLAC)
    assert "--record-per-scene" in src
    assert "record_per_scene" in inspect.signature(eval_FLAC.evaluate_model).parameters
    # exp_21: the module now has a SECOND factory call (the per-item callback, whose
    # per_scene is always False), so this looks for the headline call among them all
    # rather than assuming the first one is it.
    calls = [chunk.split(")")[0] for chunk in
             src.split("create_metric_callback_from_config(")[1:]]
    assert calls, "no metric-callback construction at all"
    assert any("per_scene=record_per_scene" in call for call in calls), calls
