"""exp_46 c4flip_data_curve -- plan r3 §3.1 tests 2-6: the wrapper's C4 + flip (C4F) path.

``DiffusionCondTrainingWrapper(..., yaw_aug_offset_step=128, *, yaw_aug_flip=True)`` at ``img_w`` 512: per
training sample, in UniFuse's order, (1) a column offset ``d ~ Uniform{0..511}`` snapped DOWN onto the
128-column grid -- an exactly uniform roll over {0, 90, 180, 270} degrees (each grid point has 128 preimages,
plan G3) -- and (2) a mirror across ``y = 0`` iff its bit ``b ~ Uniform{0, 1}`` is 1. The bits are drawn AFTER
the ``len(metadata)`` offsets from the same private ``(seed, global_step, global_rank)`` generator (plan G2).

2. the constructor accepts the C4F arguments and still refuses a flip / a grid on a disabled arm;
3. the stream: offsets = an independent oracle (``randint(0, 512)`` snapped down to multiples of 128), all in
   {0, 128, 256, 384}; bits = the RF arm's bits for the same tuple AND the same micro-batch length (32, and 8:
   a second length shows the dependence); the global python / NumPy / torch RNGs are untouched;
4. coverage: all 8 (offset, bit) combinations occur on a pinned-seed fixture, with the oracle's exact counts;
5. a C4F training step equals a vanilla step on oracle-transformed inputs: the column permutation is checked
   EXACTLY on the rotation- and mirror-invariant z channel of the panorama; the rotated vectors (depth and
   poses) to exp_43's float tolerance (plan G3: floating-point sine / cosine); fed the production transforms
   the two steps agree bit for bit; the target RIR is untouched; validation / test steps never augment;
6. the banner: exactly ``yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5``, once, flushed, rank 0.

The tiny model and helpers are exp_40's golden harness (``exp40_yaw_aug_golden_capture.py``), as in exp_43's
``test_yaw_aug_flip.py``; every oracle here is written from first principles.
"""
import collections
import importlib.util
import math
import random
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr

_HERE = Path(__file__).resolve().parent
POSE_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit")
W, GRID, SEED = 512, 128, 42
C4F = {"offset_step": GRID, "flip": True}
RF = {"flip": True}
C4F_BANNER = "yaw_aug ENABLED img_w=512 seed=42 offset_step=128 flip=0.5"
GRID_POINTS = (0, 128, 256, 384)
STREAM_CASES = ((0, 0), (1, 1), (2499, 0), (2500, 1), (39999, 1))   # (global_step, global_rank)


def _load_exp40_capture():
    spec = importlib.util.spec_from_file_location(
        "exp40_yaw_aug_golden_capture_for_exp46", _HERE / "exp40_yaw_aug_golden_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_exp40_capture()


@pytest.fixture(autouse=True)
def _no_flash(monkeypatch):
    """CPU tests: force the DiT's SDPA branch (the installed flash kernel is CUDA-only)."""
    import src.models.transformer as transformer
    monkeypatch.setattr(transformer, "flash_attn_func", None)


# --------------------------------------------------------------------------- #
# independent references (no call into the code under test except the seed counter)
# --------------------------------------------------------------------------- #
def _oracle_stream(seed, step, rank, n, img_w=W, grid=GRID):
    """``n`` raw offsets ``randint(0, img_w)`` FIRST, then ``n`` bits ``randint(0, 2)``, from one generator per
    ``(seed, step, rank)``; the offsets snapped down to the grid by integer arithmetic."""
    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(seed, step, rank))
    raw = torch.randint(0, img_w, (n,), generator=gen, dtype=torch.long).tolist()
    bits = torch.randint(0, 2, (n,), generator=gen, dtype=torch.long).tolist()
    return [grid * (d // grid) for d in raw], bits


def _oracle_roll(md, d, img_w):
    """Depth roll + z-rotation of every 3-vector from first principles (float64 matrix)."""
    alpha = d * 2.0 * math.pi / img_w
    c, s = math.cos(alpha), math.sin(alpha)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    out = dict(md)
    depth = torch.roll(md["depth"], shifts=d, dims=2).to(torch.float64)
    out["depth"] = torch.einsum("ij,jhw->ihw", rot, depth).to(md["depth"].dtype)
    for key in POSE_KEYS:
        out[key] = torch.einsum("ij,...j->...i", rot, md[key].to(torch.float64)).to(md[key].dtype)
    return out


def _oracle_mirror(md):
    """Column j <- column W-1-j, component 1 negated: INDEX ARITHMETIC, no torch.flip."""
    out = dict(md)
    w = md["depth"].shape[2]
    rev = torch.tensor([w - 1 - j for j in range(w)], dtype=torch.long)
    cols = md["depth"].index_select(2, rev)
    out["depth"] = torch.stack([cols[0], torch.neg(cols[1]), cols[2]])
    for key in POSE_KEYS:
        p = md[key]
        out[key] = torch.stack([p[..., 0], torch.neg(p[..., 1]), p[..., 2]], dim=-1)
    return out


def _oracle(md, d, bit, img_w):
    rolled = _oracle_roll(md, d, img_w)
    return _oracle_mirror(rolled) if bit == 1 else rolled


def _oracle_columns(d, bit, img_w):
    """The bare column permutation: output column j holds input column ``perm[j]`` (roll by d, then reverse)."""
    perm = [(j - d) % img_w for j in range(img_w)]
    return [perm[img_w - 1 - j] for j in range(img_w)] if bit == 1 else perm


def _production_transform(md, offsets, bits, img_w):
    """The PRODUCTION operators applied by hand: rotate by d, then mirror where the bit is 1."""
    from src.data import yaw_mirror as ym
    out = []
    for x, d, b in zip(md, offsets, bits):
        r = yr.rotate_scene_metadata(x, yr.offsets_to_radians([d], img_w)[0], img_w)
        out.append(ym.mirror_scene_metadata(r, img_w) if b == 1 else r)
    return out


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _construct(**kw):
    """Direct construction. Every yaw_aug guard runs before the model is touched; past them the constructor
    reads only ``model.diffusion_objective`` (use_ema=False)."""
    model = types.SimpleNamespace(diffusion_objective="rectified_flow")
    return tdiff.DiffusionCondTrainingWrapper(model, lr=5e-6, use_ema=False, **kw)


_ENABLED_KW = {"yaw_aug_enabled": True, "yaw_aug_img_w": W, "yaw_aug_seed": SEED}


class _Spies:
    """Wrap rotate / draw / mirror in ``tdiff``: the column shift and output of every rotation in call (=
    sample) order, the drawn bits, and the (input, output) of every mirror call."""

    def __init__(self, monkeypatch):
        self.shifts, self.rotated, self.draws, self.mirrored = [], [], [], []
        rot, draw, mir = tdiff.rotate_scene_metadata, tdiff.draw_mirror_flips, tdiff.mirror_scene_metadata

        def rot_spy(md, alpha, img_w):
            out = rot(md, alpha, img_w)
            self.shifts.append(yr.yaw_column_shift(alpha, img_w))
            self.rotated.append(out)
            return out

        def draw_spy(n, generator):
            bits = draw(n, generator)
            self.draws.append(bits.tolist())
            return bits

        def mir_spy(md, img_w):
            out = mir(md, img_w)
            self.mirrored.append((md, out))
            return out

        monkeypatch.setattr(tdiff, "rotate_scene_metadata", rot_spy)
        monkeypatch.setattr(tdiff, "draw_mirror_flips", draw_spy)
        monkeypatch.setattr(tdiff, "mirror_scene_metadata", mir_spy)

    def reset(self):
        for x in (self.shifts, self.rotated, self.draws, self.mirrored):
            x.clear()

    def applied_indices(self):
        idx = []
        for md_in, _ in self.mirrored:
            hits = [i for i, r in enumerate(self.rotated) if r is md_in]
            assert len(hits) == 1, "a mirror input that is not exactly one rotated sample"
            idx.append(hits[0])
        return idx


def _production_stream(w, spies, md, step, rank):
    """(offsets, bits, applied mirror indices, output) of one ``_apply_yaw_aug`` call at (step, rank)."""
    cap.attach_stub_trainer(w, step, rank)
    spies.reset()
    out = w._apply_yaw_aug(md)
    draws = list(spies.draws)
    assert len(draws) <= 1
    return list(spies.shifts), (draws[0] if draws else None), spies.applied_indices(), out


# --------------------------------------------------------------------------- #
# 2. constructor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("grid", [128, 32, 4, 1])
def test_the_constructor_accepts_flip_with_a_valid_offset_step(grid):
    w = _construct(**_ENABLED_KW, yaw_aug_offset_step=grid, yaw_aug_flip=True)
    assert (w.yaw_aug_enabled, w.yaw_aug_img_w, w.yaw_aug_seed, w.yaw_aug_offset_step, w.yaw_aug_flip) == \
        (True, W, SEED, grid, True)


def test_the_constructor_still_refuses_flip_on_a_disabled_arm():
    with pytest.raises(ValueError, match="yaw_aug_flip=True requires yaw_aug_enabled=True"):
        _construct(yaw_aug_flip=True)
    with pytest.raises(ValueError, match="yaw_aug_offset_step requires yaw_aug_enabled=True"):
        _construct(yaw_aug_offset_step=128, yaw_aug_flip=True)


@pytest.mark.parametrize("grid", [30, 0, 512, 100, True, 128.0, "128"])
def test_the_constructor_still_refuses_an_invalid_offset_step_with_flip(grid):
    with pytest.raises(ValueError, match="yaw_aug_offset_step: "):
        _construct(**_ENABLED_KW, yaw_aug_offset_step=grid, yaw_aug_flip=True)


@pytest.mark.parametrize("flip", [1, 0, "true", None, np.bool_(True)])
def test_the_constructor_still_refuses_a_non_literal_bool_flip_with_the_grid(flip):
    with pytest.raises(ValueError, match="yaw_aug_flip must be a bool"):
        _construct(**_ENABLED_KW, yaw_aug_offset_step=128, yaw_aug_flip=flip)


def test_the_existing_guards_still_refuse_a_c4f_arm():
    with pytest.raises(ValueError, match="cond_method='fa_invariant'"):
        _construct(**_ENABLED_KW, cond_method="fa_invariant", yaw_aug_offset_step=128, yaw_aug_flip=True)
    with pytest.raises(ValueError, match="canonicalize_enabled=True with yaw_aug_enabled=True"):
        _construct(**_ENABLED_KW, canonicalize_enabled=True, yaw_aug_offset_step=128, yaw_aug_flip=True)


def test_the_factory_builds_a_c4f_wrapper():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    assert (w.yaw_aug_offset_step, w.yaw_aug_flip) == (GRID, True)


# --------------------------------------------------------------------------- #
# 3. the stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("micro", [32, 8])
def test_the_c4f_offsets_are_the_snapped_oracle_and_its_bits_are_the_rf_bits(monkeypatch, micro):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    c4f = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    rf = cap.build_wrapper(m, cap.tiny_config(W, RF))
    md = [cap.make_md(m, 300 + i, W, height=2) for i in range(micro)]
    for step, rank in STREAM_CASES:
        offsets, bits, applied, out = _production_stream(c4f, spies, md, step, rank)
        want_offsets, want_bits = _oracle_stream(SEED, step, rank, micro)
        assert offsets == want_offsets, (micro, step, rank)
        assert set(offsets) <= set(GRID_POINTS), (micro, step, rank)
        assert bits == want_bits, (micro, step, rank)
        assert applied == [i for i, b in enumerate(bits) if b == 1], (micro, step, rank)
        mirrored_out = {id(o) for _, o in spies.mirrored}
        for i, b in enumerate(bits):
            assert (id(out[i]) in mirrored_out) is (b == 1)
        rf_offsets, rf_bits, _, _ = _production_stream(rf, spies, md, step, rank)
        assert rf_bits == bits, (micro, step, rank)                       # G2: the same bits as RF
        assert [GRID * (d // GRID) for d in rf_offsets] == offsets        # = RF's continuous draw snapped down


def test_the_bits_depend_on_the_micro_batch_length():
    """G2: the bits follow ``len(metadata)`` offset draws, so equal bits need equal (seed, step, rank) AND equal
    micro-batch length (registered: 32). For the pinned tuples the 8-long bit stream is not the 32-long one's
    prefix."""
    differs = []
    for step, rank in STREAM_CASES:
        _, b8 = _oracle_stream(SEED, step, rank, 8)
        _, b32 = _oracle_stream(SEED, step, rank, 32)
        differs.append(b8 != b32[:8])
    assert any(differs)


def test_the_c4f_draws_leave_the_global_rng_untouched_and_the_batch_unmutated():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    md = [cap.make_md(m, i, W, height=2) for i in range(32)]
    md[0]["source_vit"] = md[0]["source"].unsqueeze(0)          # AR_md's aliasing
    md[0]["context_poses_vit"] = md[0]["context_poses"]
    pristine = cap.metadata_digests(md)
    for step, rank in STREAM_CASES:
        cap.attach_stub_trainer(w, step, rank)
        _seed_all(3)
        before = cap.rng_digest(m)
        out = w._apply_yaw_aug(md)
        assert cap.rng_digest(m) == before, (step, rank)
        assert cap.metadata_digests(md) == pristine
        assert len(out) == len(md) and all(o is not x for o, x in zip(out, md))


def test_the_c4f_draws_are_resume_exact(monkeypatch):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    md = [cap.make_md(m, 200 + i, W, height=2) for i in range(32)]
    sequential = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    run = [_production_stream(sequential, spies, md, s, 1)[:3] for s in range(6)]
    fresh = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    assert _production_stream(fresh, spies, md, 5, 1)[:3] == run[5]
    assert _production_stream(fresh, spies, md, 2, 1)[:3] == run[2]
    for s, (offsets, bits, applied) in enumerate(run):
        assert (offsets, bits) == tuple(_oracle_stream(SEED, s, 1, 32))
        assert applied == [i for i, b in enumerate(bits) if b == 1]


# --------------------------------------------------------------------------- #
# 4. coverage
# --------------------------------------------------------------------------- #
def test_all_eight_offset_bit_combinations_occur_with_the_oracles_counts(monkeypatch):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    md = [cap.make_md(m, 400 + i, W, height=2) for i in range(32)]
    got, want = collections.Counter(), collections.Counter()
    for step in range(16):
        for rank in (0, 1):
            offsets, bits, _, _ = _production_stream(w, spies, md, step, rank)
            got.update(zip(offsets, bits))
            o, b = _oracle_stream(SEED, step, rank, 32)
            want.update(zip(o, b))
    assert set(got) == {(d, b) for d in GRID_POINTS for b in (0, 1)}
    assert got == want and sum(got.values()) == 16 * 2 * 32
    assert min(got.values()) > 0


# --------------------------------------------------------------------------- #
# 5. a C4F step equals a vanilla step on oracle-transformed inputs
# --------------------------------------------------------------------------- #
def _run_step(wrapper, batch, seed):
    seen = {}
    original = wrapper._compute_conditioning

    def spy(metadata):
        seen["metadata"] = metadata
        out = original(metadata)
        seen["conditioning"] = {k: v[0].detach().clone() for k, v in out.items()}
        return out

    wrapper._compute_conditioning = spy
    _seed_all(seed)
    loss = wrapper.training_step(batch, 0)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in wrapper.named_parameters() if p.grad is not None}
    return seen, loss.detach(), grads


_STEP_N = 8


@pytest.mark.parametrize("img_w,grid", [(512, 128), (16, 4)])
@pytest.mark.parametrize("step,rank", [(0, 0), (7, 1), (2500, 3)])
def test_a_c4f_step_equals_a_vanilla_step_on_oracle_transformed_inputs(img_w, grid, step, rank):
    m = cap._mods()
    _seed_all(cap.GOLDEN_SEED)
    c4f = cap.build_wrapper(m, cap.tiny_config(img_w, {"offset_step": grid, "flip": True}))
    cfg_plain = cap.tiny_config(img_w)
    del cfg_plain["training"]["yaw_aug"]
    _seed_all(cap.GOLDEN_SEED)
    plain = cap.build_wrapper(m, cfg_plain)
    sd_a, sd_b = c4f.state_dict(), plain.state_dict()
    assert sd_a.keys() == sd_b.keys() and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    cap.attach_stub_trainer(c4f, step, rank)
    cap.attach_stub_trainer(plain, step, rank)

    offsets, bits = _oracle_stream(SEED, step, rank, _STEP_N, img_w, grid)
    assert set(offsets) <= set(range(0, img_w, grid)) and len(set(offsets)) > 1
    assert 0 < sum(bits) < _STEP_N, "the case must exercise both bit values"
    g = torch.Generator().manual_seed(9000)
    reals = torch.randn(_STEP_N, 4, 64, generator=g)
    md = [cap.make_md(m, i, img_w) for i in range(_STEP_N)]
    pristine = cap.metadata_digests(md)
    oracle = [_oracle(x, d, b, img_w) for x, d, b in zip(md, offsets, bits)]

    seen_a, loss_a, grads_a = _run_step(c4f, (reals.clone(), md), seed=123)
    seen_b, loss_b, grads_b = _run_step(plain, (reals.clone(), oracle), seed=123)
    assert cap.metadata_digests(md) == pristine, "the C4F step mutated its input batch"
    for i, (x, y) in enumerate(zip(seen_a["metadata"], seen_b["metadata"])):
        # EXACT: the z channel is invariant under a rotation about z and under the mirror (which negates
        # channel 1 only), so it carries the bare column permutation (plan §3.1 test 5, G3)
        cols = torch.tensor(_oracle_columns(offsets[i], bits[i], img_w), dtype=torch.long)
        assert torch.equal(x["depth"][2], md[i]["depth"][2].index_select(1, cols)), i
        for key in ("depth",) + POSE_KEYS:
            assert x[key].dtype == y[key].dtype
            assert torch.allclose(x[key], y[key], rtol=0.0, atol=1e-5), (i, key)
        for key in ("context_audio", "padding_mask", "scene"):
            assert x[key] is md[i][key], (i, key)
    assert seen_a["conditioning"].keys() == seen_b["conditioning"].keys()
    for k in seen_a["conditioning"]:
        assert torch.allclose(seen_a["conditioning"][k], seen_b["conditioning"][k], atol=1e-4), k
    assert torch.allclose(loss_a, loss_b, rtol=1e-5, atol=1e-6)
    assert grads_a.keys() == grads_b.keys() and grads_a
    for k in grads_a:
        assert torch.allclose(grads_a[k], grads_b[k], rtol=1e-4, atol=1e-6), k

    # fed the PRODUCTION transforms instead, the two paths agree bit for bit
    production = _production_transform(md, offsets, bits, img_w)
    _seed_all(cap.GOLDEN_SEED)
    plain2 = cap.build_wrapper(m, cfg_plain)
    cap.attach_stub_trainer(plain2, step, rank)
    _, loss_c, grads_c = _run_step(plain2, (reals.clone(), production), seed=123)
    assert torch.equal(loss_a, loss_c) and all(torch.equal(grads_a[k], grads_c[k]) for k in grads_a)


def test_a_c4f_step_leaves_the_target_rir_untouched():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    cap.attach_stub_trainer(w, 7, 1)
    reals = torch.randn(4, 4, 64, generator=torch.Generator().manual_seed(1))
    reference = reals.clone()
    w.training_step((reals, [cap.make_md(m, i, W, height=2) for i in range(4)]), 0)
    assert torch.equal(reals, reference)


def _forbid_all_augmentation(monkeypatch, w):
    w._apply_yaw_aug = lambda md: (_ for _ in ()).throw(AssertionError("augmented"))
    for name in ("rotate_scene_metadata", "draw_yaw_offsets", "quantize_yaw_offsets", "draw_mirror_flips",
                 "mirror_scene_metadata", "check_mirror_metadata"):
        monkeypatch.setattr(tdiff, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(AssertionError(_n)))


def test_the_validation_step_never_augments_a_c4f_arm(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    cap.attach_stub_trainer(w, 5, 0)
    seen = []
    original = w._compute_conditioning

    def spy(md):
        seen.append(md)
        return original(md)

    w._compute_conditioning = spy
    _forbid_all_augmentation(monkeypatch, w)
    md = [cap.make_md(m, i, W, height=2) for i in range(2)]
    before = cap.metadata_digests(md)
    g = torch.Generator().manual_seed(1)
    w.validation_step((torch.randn(2, 4, 64, generator=g), md), 0)
    assert len(seen) == 1 and cap.metadata_digests(seen[0]) == before


def test_the_test_step_never_augments_a_c4f_arm(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    cap.attach_stub_trainer(w, 5, 0)
    assert (w.yaw_aug_offset_step, w.yaw_aug_flip) == (GRID, True)
    _forbid_all_augmentation(monkeypatch, w)
    seen = []
    w.metric_callback = types.SimpleNamespace(
        update_metrics=lambda split, fakes, reals, scenes, **k: seen.append((split, list(scenes))))
    w.samples, w.cfg_scale, w.steps, w.store_predictions, w.preds = 64, 1.0, 1, False, []
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        w.test_step((torch.randn(2, 1, w.samples, generator=g),
                     [cap.make_md(m, i, W, height=2) for i in range(2)]), 0)
    assert seen == [("test", ["scene_0", "scene_1"])]


# --------------------------------------------------------------------------- #
# 6. the banner
# --------------------------------------------------------------------------- #
def test_the_c4f_banner_is_one_flushed_line_from_rank_zero_only(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, C4F))
    calls = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(tdiff.rank_zero_only, "rank", 0)
    cap.attach_stub_trainer(w, 0, 0)
    w.on_fit_start()
    assert calls == [((C4F_BANNER,), {"flush": True})]
    calls.clear()
    monkeypatch.setattr(tdiff.rank_zero_only, "rank", 1)
    cap.attach_stub_trainer(w, 0, 1)
    w.on_fit_start()
    assert calls == []


@pytest.mark.parametrize("extra,line", [
    ({}, "yaw_aug ENABLED img_w=512 seed=42"),
    ({"offset_step": 32}, "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"),
    ({"offset_step": 128}, "yaw_aug ENABLED img_w=512 seed=42 offset_step=128"),
    (RF, "yaw_aug ENABLED img_w=512 seed=42 flip=0.5"),
    (C4F, C4F_BANNER),
])
def test_the_c4f_banner_is_a_different_whole_line_from_every_other_arm(capsys, monkeypatch, extra, line):
    # rank 0 explicitly (hermetic): lightning's global rank_zero_only.rank is process state that another test may have
    # left at 1 (test_exp45_sampler.py's LightningEnvironment.set_global_rank), which would silence the banner
    monkeypatch.setattr(tdiff.rank_zero_only, "rank", 0)
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(W, extra))
    cap.attach_stub_trainer(w, 0, 0)
    w.on_fit_start()
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("yaw_aug ")]
    assert out == [line]
    assert (line == C4F_BANNER) is (extra == C4F)
