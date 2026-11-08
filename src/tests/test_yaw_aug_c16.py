"""exp_40 yaw_aug_c16 -- the optional ``training.yaw_aug.offset_step`` (plan §§2-3, 6).

The C16 arm must differ from the reference Yaw-Aug arms (exp_15 neuronic 8x8, exp_17
A6000 2x32) in exactly the set of column offsets. Two things are therefore proven
here, against a fixture captured at the reference arms' own source pins (E1):

* **absent key => the reference behaviour, exactly**: the whole enabled-path record
  (four tiny ``training_step`` cases + the real-width draws of both rungs) captured at
  ``fb4fc749`` / ``88897e02`` (exp_15's first and last training pins), ``41aa31dc``
  (exp_17's audited base) and ``2b42468c`` (this branch's base) is reproduced
  byte-for-byte by the current tree;
* **key present => the same draws snapped down onto the grid**: ``d -> s * (d // s)``
  elementwise, nothing else changed (same global RNG consumption, same rotation).

The fixture is only READ here; it is regenerated with
``src/tests/exp40_yaw_aug_golden_capture.py capture|assemble`` (see its docstring).
"""
import collections
import copy
import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr
from src.training.factory import _parse_yaw_aug_config, create_training_wrapper_from_config

_HERE = Path(__file__).resolve().parent
FIXTURE = _HERE / "fixtures" / "exp40_yaw_aug_enabled_golden.json"
REFERENCE_PINS = {
    "fb4fc7497c3e024bfa35b7c02744a32d9a1a0d48",   # exp_15 first training pin
    "88897e02e75710feef37044b6c0937135fa49a22",   # exp_15 last leg (37.5k -> 40k)
    "41aa31dc0f5787019f26912654e4c5a14be7feeb",   # exp_17 audited base
    "2b42468c8f429b1d93da93b3e4ddb7bd6f8f1932",   # exp_40 base (localization-exp)
}
GOLDEN_RECORD_SHA256 = "76c1f63419f1038d4ceb22cc788941ef5843ac5699c44603f17caa6181427909"
C16 = 32


def _load_capture():
    spec = importlib.util.spec_from_file_location(
        "exp40_yaw_aug_golden_capture", _HERE / "exp40_yaw_aug_golden_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_capture()


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def golden(fixture):
    return fixture["record"]


def _snap(values, s):
    return [s * (int(v) // s) for v in values]


# --------------------------------------------------------------------------- #
# E1 / E2 -- the fixture and the absent-key regression
# --------------------------------------------------------------------------- #
def test_fixture_binds_the_four_reference_pins(fixture):
    meta = fixture["_meta"]
    assert set(meta["pins"]) == REFERENCE_PINS
    assert set(meta["pins"].values()) == {GOLDEN_RECORD_SHA256}
    assert meta["record_sha256"] == GOLDEN_RECORD_SHA256
    assert cap.canonical_sha256(fixture["record"]) == GOLDEN_RECORD_SHA256
    assert meta["yaw_seed"] == 42 and meta["rungs"] == {"8x8": [8, 8], "2x32": [2, 32]}


def test_absent_offset_step_reproduces_the_reference_record_exactly(golden):
    got = cap.capture_all()
    assert got["real"] == golden["real"], "real-width draws changed with offset_step absent"
    for case, ref in golden["tiny"].items():
        assert got["tiny"][case] == ref, f"tiny training_step {case} changed"
    assert cap.canonical_sha256(got) == GOLDEN_RECORD_SHA256


def test_offset_step_one_is_the_reference_path(golden):
    got = cap.capture_all(tiny_yaw_aug={"offset_step": 1}, real_yaw_aug={"offset_step": 1})
    assert got == golden


# --------------------------------------------------------------------------- #
# the quantised path, against the same golden
# --------------------------------------------------------------------------- #
def test_real_c16_offsets_are_the_reference_draws_snapped_down(golden):
    got = cap.capture_all(real_yaw_aug={"offset_step": C16})["real"]
    assert got["global_rng_untouched"] is True
    for rung, per_step in golden["real"]["offsets"].items():
        assert set(got["offsets"][rung]) == set(per_step)
        for step, ranks in per_step.items():
            assert got["offsets"][rung][step] == [_snap(r, C16) for r in ranks], (rung, step)
    support = {d for per_step in got["offsets"].values() for ranks in per_step.values()
               for r in ranks for d in r}
    assert support <= set(range(0, 512, C16))
    assert len(support) == 16, "the fixture's 1024 draws should hit every C16 offset"


def test_tiny_c4_training_step_rotates_by_the_snapped_offsets(golden):
    m = cap._mods()
    got = cap.capture_all(tiny_yaw_aug={"offset_step": 4})["tiny"]
    for case, ref in golden["tiny"].items():
        want = _snap(ref["offsets"], 4)
        assert got[case]["offsets"] == want
        assert got[case]["alpha_hex"] == [float(a).hex() for a in yr.offsets_to_radians(want, 16)]
        expected = [yr.rotate_scene_metadata(cap.make_md(m, i, cap.TINY_IMG_W),
                                             yr.offsets_to_radians([d], 16)[0], 16)
                    for i, d in enumerate(want)]
        assert got[case]["metadata_into_conditioner"] == cap.metadata_digests(expected)
        # samples whose reference offset is already on the grid are untouched by the snap
        for i, (d_ref, d_new) in enumerate(zip(ref["offsets"], want)):
            if d_ref == d_new:
                assert (got[case]["metadata_into_conditioner"][i]
                        == ref["metadata_into_conditioner"][i])
        # the snap consumes no randomness: the step leaves the global RNG where it did
        assert got[case]["rng_before"] == ref["rng_before"]
        assert got[case]["rng_after"] == ref["rng_after"]


def test_c16_draws_are_resume_exact():
    m = cap._mods()
    wrapper = cap.build_wrapper(m, cap.tiny_config(512, {"offset_step": C16}))
    md = [cap.make_md(m, 100 + i, 512, height=2) for i in range(32)]

    def offsets_at(w, step):
        cap.attach_stub_trainer(w, step, 1)
        with cap.RotationSpy(m) as spy:
            w._apply_yaw_aug(md)
        return [c[0] for c in spy.calls]

    sequential = [offsets_at(wrapper, s) for s in range(6)]
    fresh = cap.build_wrapper(m, cap.tiny_config(512, {"offset_step": C16}))
    assert offsets_at(fresh, 5) == sequential[5]
    assert all(d % C16 == 0 for row in sequential for d in row)


# --------------------------------------------------------------------------- #
# quantize_yaw_offsets / check_yaw_offset_step
# --------------------------------------------------------------------------- #
def test_quantize_is_exactly_uniform_over_the_grid():
    snapped = yr.quantize_yaw_offsets(torch.arange(512, dtype=torch.long), 512, C16)
    counts = collections.Counter(snapped.tolist())
    assert sorted(counts) == list(range(0, 512, C16))
    assert set(counts.values()) == {C16}
    assert snapped.tolist() == [C16 * (u // C16) for u in range(512)]


def test_quantize_keeps_dtype_and_device_and_does_not_mutate():
    x = torch.tensor([0, 31, 32, 33, 479, 480, 511], dtype=torch.long)
    before = x.clone()
    y = yr.quantize_yaw_offsets(x, 512, C16)
    assert torch.equal(x, before)
    assert y.dtype == torch.long and y.device == x.device
    assert y.tolist() == [0, 0, 32, 32, 448, 480, 480]
    assert yr.quantize_yaw_offsets(torch.zeros(0, dtype=torch.long), 512, C16).numel() == 0


@pytest.mark.parametrize("step", [0, -32, 30, 512, 600, True, 32.0, "32", None])
def test_check_offset_step_refuses(step):
    with pytest.raises(ValueError, match="offset_step"):
        yr.check_yaw_offset_step(step, 512)


@pytest.mark.parametrize("step", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_check_offset_step_accepts_divisors(step):
    assert yr.check_yaw_offset_step(step, 512) == step


@pytest.mark.parametrize("bad", [
    torch.tensor([0, 512], dtype=torch.long),
    torch.tensor([-1, 0], dtype=torch.long),
    torch.tensor([0, 32], dtype=torch.int32),
    torch.tensor([[0, 32]], dtype=torch.long),
    [0, 32],
])
def test_quantize_refuses_malformed_offsets(bad):
    with pytest.raises(ValueError, match="offsets"):
        yr.quantize_yaw_offsets(bad, 512, C16)


# --------------------------------------------------------------------------- #
# factory and wrapper
# --------------------------------------------------------------------------- #
_ENABLED = {"enabled": True, "img_w": 512, "seed": 42}


def _training(block):
    return {"yaw_aug": block}


def test_factory_absent_key_is_the_exp15_kwargs():
    assert _parse_yaw_aug_config(_training(dict(_ENABLED))) == {
        "yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}


def test_factory_passes_offset_step():
    got = _parse_yaw_aug_config(_training({**_ENABLED, "offset_step": C16}))
    assert got == {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
                   "yaw_aug_offset_step": C16}
    assert type(got["yaw_aug_offset_step"]) is int


@pytest.mark.parametrize("step", [0, 30, 512, True, 32.0, "32", None])
def test_factory_refuses_bad_offset_step(step):
    with pytest.raises(ValueError, match="yaw_aug.offset_step"):
        _parse_yaw_aug_config(_training({**_ENABLED, "offset_step": step}))


def test_factory_disabled_block_with_offset_step_passes_nothing():
    assert _parse_yaw_aug_config(_training({"enabled": False, "offset_step": C16})) == {}


def test_factory_still_refuses_unknown_keys():
    with pytest.raises(ValueError, match="offset_steps"):
        _parse_yaw_aug_config(_training({**_ENABLED, "offset_steps": C16}))


def test_wrapper_builds_through_the_factory_with_offset_step():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, {"offset_step": C16}))
    assert w.yaw_aug_enabled is True and w.yaw_aug_offset_step == C16
    assert cap.build_wrapper(m, cap.tiny_config(512)).yaw_aug_offset_step is None


def _construct(**kw):
    return tdiff.DiffusionCondTrainingWrapper(types.SimpleNamespace(), lr=5e-6, use_ema=False, **kw)


def test_wrapper_refuses_offset_step_on_a_disabled_arm():
    with pytest.raises(ValueError, match="yaw_aug_offset_step requires yaw_aug_enabled"):
        _construct(yaw_aug_offset_step=C16)


@pytest.mark.parametrize("step", [0, 30, 512, True, 32.0, "32"])
def test_wrapper_refuses_bad_offset_step(step):
    with pytest.raises(ValueError, match="yaw_aug_offset_step"):
        _construct(yaw_aug_enabled=True, yaw_aug_img_w=512, yaw_aug_seed=42,
                   yaw_aug_offset_step=step)


def test_offset_step_is_the_last_constructor_parameter():
    import inspect
    params = list(inspect.signature(tdiff.DiffusionCondTrainingWrapper.__init__).parameters)
    assert params[-1] == "yaw_aug_offset_step"


@pytest.mark.parametrize("extra,line", [
    ({}, "yaw_aug ENABLED img_w=512 seed=42"),
    ({"offset_step": C16}, "yaw_aug ENABLED img_w=512 seed=42 offset_step=32"),
])
def test_banner_lines(capsys, extra, line):
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(512, extra))
    cap.attach_stub_trainer(w, 0, 0)
    w.on_fit_start()
    out = capsys.readouterr().out.splitlines()
    assert out.count(line) == 1
    assert [x for x in out if x.startswith("yaw_aug ")] == [line]


# --------------------------------------------------------------------------- #
# plan r2 HIGH 7 / r1 HIGH 2: the fixture is bound to the committed capture tool, and the
# quantised step is checked against an INDEPENDENT pre-rotated reference
# --------------------------------------------------------------------------- #
def test_capture_tool_bytes_match_the_fixture(fixture):
    tool = (_HERE / "exp40_yaw_aug_golden_capture.py").read_bytes()
    assert hashlib.sha256(tool).hexdigest() == fixture["_meta"]["capture_tool_sha256"], (
        "the capture tool changed after the fixture was captured: recapture all four pins")


def _seed_all(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _first_principles_rotate(md, d, img_w):
    """Depth roll + z-rotation of every 3-vector, computed WITHOUT rotate_scene_metadata."""
    import math
    alpha = d * 2.0 * math.pi / img_w
    c, s = math.cos(alpha), math.sin(alpha)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    out = dict(md)
    depth = torch.roll(md["depth"], shifts=d, dims=2).to(torch.float64)
    out["depth"] = torch.einsum("ij,jhw->ihw", rot, depth).to(md["depth"].dtype)
    for key in ("source", "source_vit", "context_poses", "context_poses_vit"):
        out[key] = torch.einsum("ij,...j->...i", rot, md[key].to(torch.float64)).to(md[key].dtype)
    return out


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


@pytest.mark.parametrize("step,rank", [(7, 0), (2500, 3)])
def test_c16_step_equals_a_vanilla_step_on_prerotated_inputs(step, rank):
    m = cap._mods()
    img_w = cap.TINY_IMG_W
    _seed_all(cap.GOLDEN_SEED)
    c16 = cap.build_wrapper(m, cap.tiny_config(img_w, {"offset_step": 4}))
    cfg_plain = cap.tiny_config(img_w)
    del cfg_plain["training"]["yaw_aug"]
    _seed_all(cap.GOLDEN_SEED)
    plain = cap.build_wrapper(m, cfg_plain)
    sd_a, sd_b = c16.state_dict(), plain.state_dict()
    assert sd_a.keys() == sd_b.keys() and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    cap.attach_stub_trainer(c16, step, rank)
    cap.attach_stub_trainer(plain, step, rank)

    gen = torch.Generator()
    gen.manual_seed(tdiff._yaw_aug_step_seed(42, step, rank))
    snapped = [4 * (int(u) // 4) for u in yr.draw_yaw_offsets(cap.TINY_N, img_w, gen)]
    g = torch.Generator().manual_seed(9000)
    reals = torch.randn(cap.TINY_N, 4, 64, generator=g)
    md = [cap.make_md(m, i, img_w) for i in range(cap.TINY_N)]
    pre = [yr.rotate_scene_metadata(x, yr.offsets_to_radians([d], img_w)[0], img_w)
           for x, d in zip(md, snapped)]
    for got, x, d in zip(pre, md, snapped):            # rotate_scene_metadata vs geometry
        ref = _first_principles_rotate(x, d, img_w)
        for key in ("depth", "source", "source_vit", "context_poses", "context_poses_vit"):
            assert torch.allclose(got[key], ref[key], atol=1e-5), (key, d)

    seen_a, loss_a, grads_a = _run_step(c16, (reals.clone(), md), seed=123)
    seen_b, loss_b, grads_b = _run_step(plain, (reals.clone(), pre), seed=123)
    for x, y in zip(seen_a["metadata"], seen_b["metadata"]):
        for key in ("depth", "source", "source_vit", "context_poses", "context_poses_vit"):
            assert torch.equal(x[key], y[key])
    assert seen_a["conditioning"].keys() == seen_b["conditioning"].keys()
    for k in seen_a["conditioning"]:
        assert torch.equal(seen_a["conditioning"][k], seen_b["conditioning"][k]), k
    assert torch.equal(loss_a, loss_b)
    assert grads_a.keys() == grads_b.keys() and grads_a
    for k in grads_a:
        assert torch.equal(grads_a[k], grads_b[k]), k


def test_validation_step_never_augments_with_offset_step():
    m = cap._mods()
    w = cap.build_wrapper(m, cap.tiny_config(cap.TINY_IMG_W, {"offset_step": 4}))
    cap.attach_stub_trainer(w, 5, 0)
    w._apply_yaw_aug = lambda md: (_ for _ in ()).throw(AssertionError("validation augmented"))
    g = torch.Generator().manual_seed(1)
    w.validation_step((torch.randn(2, 4, 64, generator=g),
                       [cap.make_md(m, i, cap.TINY_IMG_W) for i in range(2)]), 0)


def test_canonicalize_still_refuses_a_c16_arm():
    from src.training.factory import _parse_canonicalize_config
    t = {"yaw_aug": {**_ENABLED, "offset_step": C16},
         "canonicalize": {"enabled": True, "img_w": 512}}
    with pytest.raises(ValueError, match="yaw_aug"):
        _parse_canonicalize_config(t)
