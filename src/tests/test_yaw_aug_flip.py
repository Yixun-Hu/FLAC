"""exp_43 std_aug -- the wrapper's mirror-flip path (plan §2.1, §2.3, §3.3).

``DiffusionCondTrainingWrapper(..., yaw_aug_offset_step=None, *, yaw_aug_flip=False)``: with
``yaw_aug_flip=True`` every training sample is, in UniFuse's order, (1) rolled by the unchanged
exp_15 draw ``d ~ Uniform{0..W-1}`` and (2) mirrored across ``y = 0`` iff its bit
``b ~ Uniform{0, 1}`` is 1. The bits are drawn AFTER the offsets from the same private
``(seed, global_step, global_rank)`` generator. What is proven here:

* the keyword-only constructor contract, its refusals and the banner lines;
* flip ABSENT (or ``False``) = the committed exp_40 golden record, with the flip machinery
  proven untouched (raising stubs), i.e. no extra draw;
* flip ON: the offsets are the reference draws; a training step equals a vanilla step on
  inputs pre-transformed by an INDEPENDENT oracle written here (roll by first principles,
  then mirror by index arithmetic); the sample indices actually passed to
  ``mirror_scene_metadata`` are the drawn bits; resume-exactness; global RNG isolation;
  malformed samples refused whatever their bit; ``validation_step`` / ``test_step`` never flip.

The flip-path regression golden is in ``test_exp43_yaw_aug_flip_golden.py``.
"""
import hashlib
import importlib.util
import inspect
import json
import math
import random
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import src.training.diffusion as tdiff
from src.data import yaw_mirror as ym
from src.data import yaw_rotation as yr

_HERE = Path(__file__).resolve().parent
EXP40_FIXTURE = _HERE / "fixtures" / "exp40_yaw_aug_enabled_golden.json"
EXP40_RECORD_SHA256 = "76c1f63419f1038d4ceb22cc788941ef5843ac5699c44603f17caa6181427909"
POSE_KEYS = ("source", "source_vit", "context_poses", "context_poses_vit")
FLIP = {"flip": True}


def _load_exp40_capture():
    spec = importlib.util.spec_from_file_location(
        "exp40_yaw_aug_golden_capture_for_exp43", _HERE / "exp40_yaw_aug_golden_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_exp40_capture()


@pytest.fixture(autouse=True)
def _no_flash(monkeypatch):
    """CPU tests: force the DiT's SDPA branch (the installed flash kernel is CUDA-only)."""
    import src.models.transformer as transformer
    monkeypatch.setattr(transformer, "flash_attn_func", None)


@pytest.fixture(scope="module")
def golden():
    fixture = json.loads(EXP40_FIXTURE.read_text())
    assert cap.canonical_sha256(fixture["record"]) == EXP40_RECORD_SHA256
    return fixture["record"]


# --------------------------------------------------------------------------- #
# independent references (no call into the code under test)
# --------------------------------------------------------------------------- #
def _reference_draws(seed, step, rank, n, img_w):
    """Offsets then bits, from the counter-seeded generator, by bare ``torch.randint``."""
    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(seed, step, rank))
    offsets = torch.randint(0, img_w, (n,), generator=gen, dtype=torch.long)
    bits = torch.randint(0, 2, (n,), generator=gen, dtype=torch.long)
    return offsets.tolist(), bits.tolist()


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


def _production(md, offsets, bits, img_w):
    """The PRODUCTION operators applied by hand: rotate by d, then mirror where the bit is 1."""
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
    """Direct construction. Every yaw_aug guard runs before the model is touched; past them
    the constructor reads only ``model.diffusion_objective`` (use_ema=False)."""
    model = types.SimpleNamespace(diffusion_objective="rectified_flow")
    return tdiff.DiffusionCondTrainingWrapper(model, lr=5e-6, use_ema=False, **kw)


_ENABLED_KW = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}


# --------------------------------------------------------------------------- #
# constructor: keyword-only flag and its refusals
# --------------------------------------------------------------------------- #
def test_yaw_aug_flip_is_keyword_only_with_default_false():
    params = inspect.signature(tdiff.DiffusionCondTrainingWrapper.__init__).parameters
    p = params["yaw_aug_flip"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    names = list(params)
    assert names[-2:] == ["yaw_aug_offset_step", "yaw_aug_flip"]
    assert params["yaw_aug_offset_step"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert not any(q.kind is inspect.Parameter.VAR_POSITIONAL for q in params.values())


def test_a_positional_slip_cannot_reach_yaw_aug_flip():
    params = list(inspect.signature(tdiff.DiffusionCondTrainingWrapper.__init__).parameters.values())
    positional = [q for q in params[1:] if q.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    args = [types.SimpleNamespace()] + [q.default for q in positional[1:]]
    with pytest.raises(TypeError):
        tdiff.DiffusionCondTrainingWrapper(*args, True)


@pytest.mark.parametrize("flip", [1, 0, 1.0, "true", "false", None, np.bool_(True), [True]])
def test_constructor_refuses_a_non_literal_bool_flip(flip):
    with pytest.raises(ValueError, match="yaw_aug_flip must be a bool"):
        _construct(**_ENABLED_KW, yaw_aug_flip=flip)


def test_constructor_refuses_flip_on_a_disabled_arm():
    with pytest.raises(ValueError, match="yaw_aug_flip=True requires yaw_aug_enabled=True"):
        _construct(yaw_aug_flip=True)


def test_constructor_refuses_flip_with_offset_step():
    with pytest.raises(ValueError, match="yaw_aug_flip=True with yaw_aug_offset_step"):
        _construct(**_ENABLED_KW, yaw_aug_offset_step=32, yaw_aug_flip=True)


def test_existing_guards_still_refuse_a_flip_arm():
    with pytest.raises(ValueError, match="cond_method='fa_invariant'"):
        _construct(**_ENABLED_KW, cond_method="fa_invariant", yaw_aug_flip=True)
    with pytest.raises(ValueError, match="canonicalize_enabled=True with yaw_aug_enabled=True"):
        _construct(**_ENABLED_KW, canonicalize_enabled=True, yaw_aug_flip=True)


def test_false_flip_is_accepted_everywhere_and_stored_as_false():
    assert _construct().yaw_aug_flip is False
    assert _construct(yaw_aug_flip=False).yaw_aug_flip is False
    assert _construct(**_ENABLED_KW, yaw_aug_offset_step=32, yaw_aug_flip=False).yaw_aug_flip is False
    assert _construct(**_ENABLED_KW, yaw_aug_flip=True).yaw_aug_flip is True


def test_the_factory_builds_a_flip_wrapper():
    m = cap._mods()
    assert cap.build_wrapper(m, cap.tiny_config(512, FLIP)).yaw_aug_flip is True
    assert cap.build_wrapper(m, cap.tiny_config(512)).yaw_aug_flip is False
    assert cap.build_wrapper(m, cap.tiny_config(512, {"flip": False})).yaw_aug_flip is False


# --------------------------------------------------------------------------- #
# banner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra,line", [
    ({}, "yaw_aug ENABLED img_w=512 seed=42"),
    ({"flip": False}, "yaw_aug ENABLED img_w=512 seed=42"),
    ({"offset_step": 32}, "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"),
    ({"offset_step": 32, "flip": False}, "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"),
    (FLIP, "yaw_aug ENABLED img_w=512 seed=42 flip=0.5"),
])
def test_banner_lines(capsys, extra, line):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, extra))
    cap.attach_stub_trainer(w, 0, 0)
    w.on_fit_start()
    out = capsys.readouterr().out.splitlines()
    assert [x for x in out if x.startswith("yaw_aug ")] == [line]


def test_flip_banner_is_one_flushed_print(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, FLIP))
    cap.attach_stub_trainer(w, 0, 0)
    calls = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append((a, kw)))
    w.on_fit_start()
    assert calls == [(("yaw_aug ENABLED img_w=512 seed=42 flip=0.5",), {"flush": True})]


# --------------------------------------------------------------------------- #
# flip absent / false: the committed exp_40 golden, with no flip machinery touched
# --------------------------------------------------------------------------- #
def _forbid_flip_machinery(monkeypatch):
    def boom(name):
        return lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} called"))
    for name in ("draw_mirror_flips", "mirror_scene_metadata", "check_mirror_metadata"):
        monkeypatch.setattr(tdiff, name, boom(name))


def test_flip_absent_reproduces_the_exp40_golden_and_never_touches_the_flip(monkeypatch, golden):
    _forbid_flip_machinery(monkeypatch)
    got = cap.capture_all()
    assert got == golden
    assert cap.canonical_sha256(got) == EXP40_RECORD_SHA256


def test_flip_false_is_the_reference_path(monkeypatch, golden):
    _forbid_flip_machinery(monkeypatch)
    got = cap.capture_all(tiny_yaw_aug={"flip": False}, real_yaw_aug={"flip": False})
    assert got == golden


# --------------------------------------------------------------------------- #
# flip on: offsets, bits, the applied indices
# --------------------------------------------------------------------------- #
class _Spies:
    """Wrap rotate / draw / mirror in ``tdiff``: the rotated objects in call (= sample) order,
    the drawn bits, and the (input object, output object) of every mirror call."""

    def __init__(self, monkeypatch):
        self.rotated, self.draws, self.mirrored = [], [], []
        rot, draw, mir = tdiff.rotate_scene_metadata, tdiff.draw_mirror_flips, tdiff.mirror_scene_metadata

        def rot_spy(md, alpha, img_w):
            out = rot(md, alpha, img_w)
            self.rotated.append((yr.yaw_column_shift(alpha, img_w), out))
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
        self.rotated.clear()
        self.draws.clear()
        self.mirrored.clear()

    def applied_indices(self):
        idx = []
        for md_in, _ in self.mirrored:
            hits = [i for i, (_, r) in enumerate(self.rotated) if r is md_in]
            assert len(hits) == 1, "a mirror input that is not exactly one rotated sample"
            idx.append(hits[0])
        return idx


def test_flip_offsets_are_the_reference_draws_and_the_global_rng_is_untouched(golden):
    got = cap.capture_all(tiny_yaw_aug=FLIP, real_yaw_aug=FLIP)
    assert got["real"]["offsets"] == golden["real"]["offsets"]
    assert got["real"]["global_rng_untouched"] is True
    for case, ref in golden["tiny"].items():
        assert got["tiny"][case]["offsets"] == ref["offsets"]
        assert got["tiny"][case]["alpha_hex"] == ref["alpha_hex"]
        # the flip consumes no GLOBAL randomness: same RNG digests either side of the step
        assert got["tiny"][case]["rng_before"] == ref["rng_before"]
        assert got["tiny"][case]["rng_after"] == ref["rng_after"]


def test_tiny_flip_step_feeds_the_oracle_transformed_metadata(golden):
    m = cap._mods()
    got = cap.capture_all(tiny_yaw_aug=FLIP)["tiny"]
    n_flipped = 0
    for (step, rank) in cap.TINY_CASES:
        case = f"step{step}_rank{rank}"
        offsets, bits = _reference_draws(42, step, rank, cap.TINY_N, cap.TINY_IMG_W)
        assert got[case]["offsets"] == offsets == golden["tiny"][case]["offsets"]
        md = [cap.make_md(m, i, cap.TINY_IMG_W) for i in range(cap.TINY_N)]
        production = _production(md, offsets, bits, cap.TINY_IMG_W)
        assert got[case]["metadata_into_conditioner"] == cap.metadata_digests(production)
        for i, b in enumerate(bits):
            same = got[case]["metadata_into_conditioner"][i] == golden["tiny"][case]["metadata_into_conditioner"][i]
            assert same is (b == 0), (case, i, b)
            oracle = _oracle(md[i], offsets[i], b, cap.TINY_IMG_W)
            for key in ("depth",) + POSE_KEYS:
                assert torch.allclose(production[i][key], oracle[key], rtol=0.0, atol=1e-6), (case, i, key)
        n_flipped += sum(bits)
    assert 0 < n_flipped < cap.TINY_N * len(cap.TINY_CASES)


def test_applied_mirror_indices_equal_the_drawn_bits_at_production_width(monkeypatch):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    w = cap.build_wrapper(m, cap.tiny_config(512, FLIP))
    applied_total = drawn_total = 0
    for rung, (world, micro) in cap.RUNGS.items():
        md = [cap.make_md(m, 100 + s, 512, height=2) for s in range(micro)]
        for step in cap.REAL_STEPS:
            for rank in range(world):
                cap.attach_stub_trainer(w, step, rank)
                spies.reset()
                out = w._apply_yaw_aug(md)
                offsets, bits = _reference_draws(42, step, rank, micro, 512)
                assert [d for d, _ in spies.rotated] == offsets, (rung, step, rank)
                assert spies.draws == [bits], (rung, step, rank)
                want = [i for i, b in enumerate(bits) if b == 1]
                assert spies.applied_indices() == want, (rung, step, rank)
                mirrored_out = {id(o) for _, o in spies.mirrored}
                for i, b in enumerate(bits):
                    if b == 1:
                        assert id(out[i]) in mirrored_out
                    else:
                        assert out[i] is spies.rotated[i][1]
                applied_total += len(want)
                drawn_total += micro
    assert 0 < applied_total < drawn_total


def test_flip_bits_are_resume_exact(monkeypatch):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    md = [cap.make_md(m, 200 + i, 512, height=2) for i in range(32)]

    def applied_at(w, step):
        cap.attach_stub_trainer(w, step, 1)
        spies.reset()
        w._apply_yaw_aug(md)
        return [d for d, _ in spies.rotated], spies.draws[0], spies.applied_indices()

    sequential = cap.build_wrapper(m, cap.tiny_config(512, FLIP))
    run = [applied_at(sequential, s) for s in range(6)]
    fresh = cap.build_wrapper(m, cap.tiny_config(512, FLIP))
    assert applied_at(fresh, 5) == run[5]
    assert applied_at(fresh, 2) == run[2]
    for s, (offsets, bits, applied) in enumerate(run):
        assert (offsets, bits) == _reference_draws(42, s, 1, 32, 512)
        assert applied == [i for i, b in enumerate(bits) if b == 1]


def test_apply_with_flip_leaves_the_global_rng_untouched():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, FLIP))
    cap.attach_stub_trainer(w, 11, 0)
    md = [cap.make_md(m, i, 512, height=2) for i in range(8)]
    _seed_all(3)
    before = cap.rng_digest(m)
    w._apply_yaw_aug(md)
    assert cap.rng_digest(m) == before


def test_apply_with_flip_never_mutates_the_batch():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(16, FLIP))
    cap.attach_stub_trainer(w, 7, 1)
    md = [cap.make_md(m, i, 16) for i in range(6)]
    md[0]["source_vit"] = md[0]["source"].unsqueeze(0)          # AR_md's aliasing
    md[0]["context_poses_vit"] = md[0]["context_poses"]
    before = cap.metadata_digests(md)
    out = w._apply_yaw_aug(md)
    assert cap.metadata_digests(md) == before
    assert len(out) == len(md) and all(o is not x for o, x in zip(out, md))


# --------------------------------------------------------------------------- #
# a flip step equals a vanilla step on oracle-transformed inputs
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


_STEP_N = 6


@pytest.mark.parametrize("step,rank", [(0, 0), (7, 1), (2500, 3)])
def test_flip_step_equals_a_vanilla_step_on_oracle_transformed_inputs(step, rank):
    m = cap._mods()
    img_w = cap.TINY_IMG_W
    _seed_all(cap.GOLDEN_SEED)
    flip = cap.build_wrapper(m, cap.tiny_config(img_w, FLIP))
    cfg_plain = cap.tiny_config(img_w)
    del cfg_plain["training"]["yaw_aug"]
    _seed_all(cap.GOLDEN_SEED)
    plain = cap.build_wrapper(m, cfg_plain)
    sd_a, sd_b = flip.state_dict(), plain.state_dict()
    assert sd_a.keys() == sd_b.keys() and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    cap.attach_stub_trainer(flip, step, rank)
    cap.attach_stub_trainer(plain, step, rank)

    offsets, bits = _reference_draws(42, step, rank, _STEP_N, img_w)
    assert 0 < sum(bits) < _STEP_N, "the case must exercise both bit values"
    g = torch.Generator().manual_seed(9000)
    reals = torch.randn(_STEP_N, 4, 64, generator=g)
    md = [cap.make_md(m, i, img_w) for i in range(_STEP_N)]
    pristine = cap.metadata_digests(md)
    oracle = [_oracle(x, d, b, img_w) for x, d, b in zip(md, offsets, bits)]

    seen_a, loss_a, grads_a = _run_step(flip, (reals.clone(), md), seed=123)
    seen_b, loss_b, grads_b = _run_step(plain, (reals.clone(), oracle), seed=123)
    assert cap.metadata_digests(md) == pristine, "the flip step mutated its input batch"
    for i, (x, y) in enumerate(zip(seen_a["metadata"], seen_b["metadata"])):
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
    production = _production(md, offsets, bits, img_w)
    _seed_all(cap.GOLDEN_SEED)
    plain2 = cap.build_wrapper(m, cfg_plain)
    cap.attach_stub_trainer(plain2, step, rank)
    _, loss_c, grads_c = _run_step(plain2, (reals.clone(), production), seed=123)
    assert torch.equal(loss_a, loss_c) and all(torch.equal(grads_a[k], grads_c[k]) for k in grads_a)


def test_a_flip_step_leaves_the_target_rir_untouched():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(16, FLIP))
    cap.attach_stub_trainer(w, 7, 1)
    reals = torch.randn(4, 4, 64, generator=torch.Generator().manual_seed(1))
    reference = reals.clone()
    w.training_step((reals, [cap.make_md(m, i, 16) for i in range(4)]), 0)
    assert torch.equal(reals, reference)


# --------------------------------------------------------------------------- #
# malformed samples are refused whatever their bit
# --------------------------------------------------------------------------- #
def _bad_sample_case(bit_wanted, n=8, step=7, rank=1, img_w=16):
    _, bits = _reference_draws(42, step, rank, n, img_w)
    return bits.index(bit_wanted)


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("defect", ["nan", "inf", "int_depth", "int_pose"])
def test_a_malformed_sample_is_refused_before_the_bit_draw(monkeypatch, bit, defect):
    m = cap._mods()
    spies = _Spies(monkeypatch)
    w = cap.build_wrapper(m, cap.tiny_config(16, FLIP))
    cap.attach_stub_trainer(w, 7, 1)
    md = [cap.make_md(m, i, 16) for i in range(8)]
    i = _bad_sample_case(bit)
    if defect == "nan":
        md[i]["depth"] = md[i]["depth"].clone()
        md[i]["depth"][0, 1, 2] = math.nan
    elif defect == "inf":
        md[i]["depth"] = md[i]["depth"].clone()
        md[i]["depth"][2, 0, 5] = -math.inf
    elif defect == "int_depth":
        md[i]["depth"] = torch.ones(3, 4, 16, dtype=torch.long)
    else:
        md[i]["context_poses"] = torch.ones(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match=rf"yaw_aug: metadata\[{i}\]"):
        w._apply_yaw_aug(md)
    assert spies.draws == [] and spies.mirrored == []


def test_the_flip_absent_path_does_not_add_the_mirror_contract(monkeypatch):
    """Unchanged exp_15 semantics: a non-finite depth is not refused on the roll-only arm."""
    m = cap._mods()
    _forbid_flip_machinery(monkeypatch)
    w = cap.build_wrapper(m, cap.tiny_config(16))
    cap.attach_stub_trainer(w, 7, 1)
    md = [cap.make_md(m, i, 16) for i in range(2)]
    md[0]["depth"] = md[0]["depth"].clone()
    md[0]["depth"][0, 0, 0] = math.nan
    assert len(w._apply_yaw_aug(md)) == 2


# --------------------------------------------------------------------------- #
# evaluation never flips
# --------------------------------------------------------------------------- #
def _forbid_all_augmentation(monkeypatch, w):
    w._apply_yaw_aug = lambda md: (_ for _ in ()).throw(AssertionError("augmented"))
    _forbid_flip_machinery(monkeypatch)
    for name in ("rotate_scene_metadata", "draw_yaw_offsets"):
        monkeypatch.setattr(tdiff, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError(name)))


def test_validation_step_never_flips(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(cap.TINY_IMG_W, FLIP))
    cap.attach_stub_trainer(w, 5, 0)
    seen = []
    original = w._compute_conditioning

    def spy(md):
        seen.append(md)
        return original(md)

    w._compute_conditioning = spy
    _forbid_all_augmentation(monkeypatch, w)
    md = [cap.make_md(m, i, cap.TINY_IMG_W) for i in range(2)]
    before = cap.metadata_digests(md)
    g = torch.Generator().manual_seed(1)
    w.validation_step((torch.randn(2, 4, 64, generator=g), md), 0)
    assert len(seen) == 1 and cap.metadata_digests(seen[0]) == before


def test_test_step_never_flips(monkeypatch):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(cap.TINY_IMG_W, FLIP))
    cap.attach_stub_trainer(w, 5, 0)
    assert w.yaw_aug_flip is True
    _forbid_all_augmentation(monkeypatch, w)
    seen = []
    w.metric_callback = types.SimpleNamespace(
        update_metrics=lambda split, fakes, reals, scenes, **k: seen.append((split, list(scenes))))
    w.samples, w.cfg_scale, w.steps, w.store_predictions, w.preds = 64, 1.0, 1, False, []
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        w.test_step((torch.randn(2, 1, w.samples, generator=g),
                     [cap.make_md(m, i, cap.TINY_IMG_W) for i in range(2)]), 0)
    assert seen == [("test", ["scene_0", "scene_1"])]


def test_yaw_rotation_module_is_still_the_exp15_bytes():
    import subprocess
    repo = _HERE.parents[1]
    base = subprocess.run(["git", "show", "2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932:src/data/yaw_rotation.py"],
                          capture_output=True, check=True, cwd=repo).stdout
    assert hashlib.sha256((repo / "src/data/yaw_rotation.py").read_bytes()).digest() == \
        hashlib.sha256(base).digest()
