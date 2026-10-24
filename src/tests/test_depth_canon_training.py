"""exp_25 depth_canon round 2 -- training-side canonicalization: the factory parser,
the wrapper's constructor contract, banner, tie counters, the three step hooks and
the batch-transfer override (plan_depth_canon.md v7 §2.2, §6 tests 10-14, 12b, 12c).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md
v7 §2.2/§3/§6/§8.

Tiny CPU-only ``diffusion_cond`` through the REAL factories (``test_cond_dispatch``'s
config); metadata = synthetic depth through the REAL AR_md converter
(``test_depth_canon``'s helpers), i.e. the loader's float64 point cloud. Test 15 (the
golden disabled path) is ``test_yaw_aug_training.py::test_disabled_path_matches_golden``.
"""
import dataclasses
import random
import re
import types

import numpy as np
import pytest
import pytorch_lightning as pl
import pytorch_lightning.core.hooks as pl_hooks
import torch
from torch.distributed.utils import _to_kwargs

import src.models.transformer as _transformer
import src.training.diffusion as tdiff
from src.data.dataset import collation_fn
from src.data import yaw_rotation as yr
from src.data.yaw_rotation import TIE_CLASSES
from src.models.factory import create_model_from_config
from src.training.factory import (
    CANON_KEYS, _parse_canonicalize_config, create_training_wrapper_from_config)
from test_cond_dispatch import _base_config, _neutralize_for_steps  # (pytest prepends src/tests)
from test_depth_canon import _cloud, _scalar_depth  # synthetic depth via the real AR_md
from test_yaw_aug_training import PRE_CHANGE_WRAPPER_KWARGS, _rng_digest
from test_yaw_aug_training import _base_config as _forward_config  # real DiT forward

CANON_ON = {"enabled": True, "img_w": 512}
YAW_ON = {"enabled": True, "img_w": 512, "seed": 42}
CANON_KWARGS = {"canonicalize_enabled", "canonicalize_img_w"}


# --------------------------------------------------------------------------- #
# 10. _parse_canonicalize_config (+ the factory splice)
# --------------------------------------------------------------------------- #
def _expect_parse_error(training, *needles):
    with pytest.raises(ValueError) as excinfo:
        _parse_canonicalize_config(training)
    for needle in ("canonicalize",) + needles:
        assert needle in str(excinfo.value), f"{excinfo.value!s} does not name {needle!r}"


def test_canon_keys_are_exactly_enabled_and_img_w():
    assert CANON_KEYS == ("enabled", "img_w")


@pytest.mark.parametrize("training", [
    {},
    {"canonicalize": {"enabled": False}},
    {"canonicalize": {"enabled": False, "img_w": 512}},
    {"canonicalize": {"enabled": False, "img_w": "not-validated-when-off"}},
    {"canonicalize": {"enabled": False}, "yaw_aug": YAW_ON},
    {"canonicalize": {"enabled": False}, "cond_method": "fa_invariant"},
])
def test_absent_or_disabled_block_yields_no_kwargs(training):
    assert _parse_canonicalize_config(training) == {}


@pytest.mark.parametrize("neighbours", [
    {}, {"cond_method": "vanilla"}, {"yaw_aug": {"enabled": False}}])
def test_valid_block_yields_exact_kwargs(neighbours):
    got = _parse_canonicalize_config({"canonicalize": dict(CANON_ON), **neighbours})
    assert got == {"canonicalize_enabled": True, "canonicalize_img_w": 512}
    assert got["canonicalize_enabled"] is True and type(got["canonicalize_img_w"]) is int


_MISSING = object()


@pytest.mark.parametrize("enabled", ["true", "false", 1, 0, 1.0, None, _MISSING])
def test_enabled_must_be_a_literal_bool(enabled):
    block = {"img_w": 512} if enabled is _MISSING else {"enabled": enabled, "img_w": 512}
    _expect_parse_error({"canonicalize": block}, "enabled")


@pytest.mark.parametrize("block", [True, "enabled", ["enabled"], 512, None])
def test_non_dict_block_is_rejected(block):
    _expect_parse_error({"canonicalize": block})


@pytest.mark.parametrize("block", [
    {"enabled": True, "img_w": 512, "tiebreak": "image"},
    {"enabled": False, "seed": 42},
])
def test_unknown_keys_are_rejected_even_when_disabled(block):
    unknown = (set(block) - set(CANON_KEYS)).pop()
    _expect_parse_error({"canonicalize": block}, unknown)


@pytest.mark.parametrize("img_w", [True, False, 0, -512, 512.0, "512", None, _MISSING])
def test_bad_img_w_is_rejected(img_w):
    block = {"enabled": True} if img_w is _MISSING else {"enabled": True, "img_w": img_w}
    _expect_parse_error({"canonicalize": block}, "img_w")


EXCLUSIONS = [({"yaw_aug": dict(YAW_ON)}, "yaw_aug"),
              ({"cond_method": "fa_invariant"}, "fa_invariant")]


@pytest.mark.parametrize("extra, needle", EXCLUSIONS)
def test_enabled_with_an_excluded_treatment_is_rejected(extra, needle):
    _expect_parse_error({"canonicalize": dict(CANON_ON), **extra}, needle)


def _captured_factory_kwargs(monkeypatch, training_overrides):
    """The real factory with the wrapper class stubbed (exp_15 pattern)."""
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    create_training_wrapper_from_config(cfg, object())
    return captured


@pytest.mark.parametrize("training", [{}, {"canonicalize": {"enabled": False}}])
def test_factory_disabled_path_is_the_pre_change_call(monkeypatch, training):
    assert set(_captured_factory_kwargs(monkeypatch, training)) == PRE_CHANGE_WRAPPER_KWARGS


def test_factory_enabled_passes_exactly_the_canon_kwargs(monkeypatch):
    kwargs = _captured_factory_kwargs(monkeypatch, {"canonicalize": dict(CANON_ON)})
    assert set(kwargs) == PRE_CHANGE_WRAPPER_KWARGS | CANON_KWARGS
    assert kwargs["canonicalize_enabled"] is True and kwargs["canonicalize_img_w"] == 512
    assert kwargs["cond_method"] == "vanilla"


@pytest.mark.parametrize("extra, needle", EXCLUSIONS)
def test_factory_refuses_the_exclusions(extra, needle):
    cfg = _base_config()
    cfg["training"].update({"canonicalize": dict(CANON_ON), **extra})
    with pytest.raises(ValueError, match=needle):
        create_training_wrapper_from_config(cfg, object())


# --------------------------------------------------------------------------- #
# 11. wrapper raw-argument validation and exclusions at construction
# --------------------------------------------------------------------------- #
def _build(**training_overrides):
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    return create_training_wrapper_from_config(cfg, create_model_from_config(cfg))


def _construct_directly(**kwargs):
    """Bypass the factory: the constructor must fail closed on its own (the checks run
    before the model is touched, so a bare namespace stands in for it)."""
    return tdiff.DiffusionCondTrainingWrapper(types.SimpleNamespace(), lr=5e-6, use_ema=False,
                                              **kwargs)


@pytest.mark.parametrize("enabled", ["true", "false", 1, 0, 1.0, None])
def test_constructor_rejects_non_bool_enabled(enabled):
    with pytest.raises(ValueError, match="canonicalize_enabled"):
        _construct_directly(canonicalize_enabled=enabled)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("img_w", ["512", 512.0, True, False, None, 0, -1])
def test_constructor_rejects_bad_img_w(enabled, img_w):
    with pytest.raises(ValueError, match="canonicalize_img_w"):
        _construct_directly(canonicalize_enabled=enabled, canonicalize_img_w=img_w)


@pytest.mark.parametrize("extra, needle", [
    ({"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}, "yaw_aug"),
    ({"cond_method": "fa_invariant"}, "fa_invariant")])
def test_constructor_refuses_the_exclusions(extra, needle):
    with pytest.raises(ValueError, match=needle):
        _construct_directly(canonicalize_enabled=True, canonicalize_img_w=512, **extra)


def test_wrapper_defaults_have_canonicalization_off():
    wrapper = _build()
    assert wrapper.canonicalize_enabled is False and wrapper.canonicalize_img_w == 512
    assert wrapper.canon_tie_counts == {c: 0 for c in TIE_CLASSES}
    assert list(wrapper.canon_tie_counts) == list(TIE_CLASSES)


def test_factory_flags_reach_the_wrapper_uncoerced():
    wrapper = _build(canonicalize={"enabled": True, "img_w": 16})
    assert wrapper.canonicalize_enabled is True
    assert type(wrapper.canonicalize_img_w) is int and wrapper.canonicalize_img_w == 16
    assert wrapper.canon_tie_counts == {c: 0 for c in TIE_CLASSES}


def test_yaw_aug_arm_still_constructs_with_canonicalization_absent():
    wrapper = _build(yaw_aug={"enabled": True, "img_w": 16, "seed": 42})
    assert wrapper.yaw_aug_enabled is True and wrapper.canonicalize_enabled is False


# --------------------------------------------------------------------------- #
# 14. banner (exact, rank 0, flushed, only when enabled); 13. tie totals at fit end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("img_w", [512, 16])
def test_banner_is_the_exact_whole_line_once(capsys, img_w):
    _build(canonicalize={"enabled": True, "img_w": img_w}).on_fit_start()
    lines = capsys.readouterr().out.splitlines()
    assert lines.count(f"canonicalize ENABLED img_w={img_w} tiebreak=image") == 1
    assert sum("canonicalize" in line for line in lines) == 1


def test_banner_and_tie_totals_are_flushed(monkeypatch):
    calls = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append(kw))
    wrapper = _build(canonicalize=dict(CANON_ON))
    wrapper.on_fit_start()
    wrapper.on_fit_end()
    assert len(calls) == 3 and all(kw.get("flush") is True for kw in calls)  # + the latency line


@pytest.mark.parametrize("training", [{}, {"canonicalize": {"enabled": False}},
                                      {"yaw_aug": {"enabled": True, "img_w": 16, "seed": 42}}])
def test_nothing_canonicalize_is_printed_when_disabled(capsys, training):
    wrapper = _build(**training)
    wrapper.on_fit_start()
    wrapper.on_fit_end()
    out = capsys.readouterr().out
    assert "canonicalize" not in out
    assert ("yaw_aug ENABLED" in out) == ("yaw_aug" in training)


def test_tie_totals_are_printed_per_class_at_fit_end(capsys):
    wrapper = _build(canonicalize=dict(CANON_ON))
    wrapper.canon_tie_counts.update({"none": 7, "same_column": 2, "cross_column_image": 3})
    wrapper.on_fit_end()
    assert capsys.readouterr().out.splitlines() == [
        "canonicalize tie_counts rank0 none=7 same_column=2 cross_column_image=3",
        # r3b-F1 fix 2: no batch was canonicalized here, so the mean is undefined (nan)
        "canonicalize latency rank0 batches=0 samples=0 total_s=0.000000 mean_ms_per_batch=nan"]


LATENCY_LINE = re.compile(r"canonicalize latency rank0 batches=(\d+) samples=(\d+) "
                          r"total_s=(\d+\.\d{6}) mean_ms_per_batch=(\S+)")


def test_latency_is_measured_per_batch_and_logged_on_the_training_path_only():
    """r3b-F1 fix 2 (Codex r3b BLOCKING 2): perf_counter around the per-batch loop; totals
    on the instance; ``canon/latency_ms`` logged per training step (same guard as the ties)."""
    wrapper, logged = _canon_wrapper(training=True)
    wrapper._canonicalize_batch([_canon_md(0, "cross_column_image"), _canon_md(1, "same_column")])
    wrapper._canonicalize_batch([_canon_md(2, "none")])
    lat = wrapper.canon_latency
    assert (lat["batches"], lat["samples"]) == (2, 3) and lat["total_s"] > 0
    ms = [v for name, v, _ in logged if name == "canon/latency_ms"]
    assert len(ms) == 2 and all(type(v) is float and v > 0 for v in ms)
    assert abs(sum(ms) / 1e3 - lat["total_s"]) < 1e-9
    wrapper.trainer.training = False
    wrapper._canonicalize_batch([_canon_md(3, "none")])
    assert len([1 for name, _, _ in logged if name == "canon/latency_ms"]) == 2
    assert wrapper.canon_latency["batches"] == 3       # validation batches still measured


def test_latency_covers_the_carrier_unwrap_and_the_schema_check(monkeypatch):
    """rB-F1 A5 (Codex rA NIT): the timer starts BEFORE the carrier unwrap and the validation,
    so the latency is the whole treatment overhead of _canonicalize_batch."""
    import src.training.diffusion as D
    clock = [100.0]
    monkeypatch.setattr(D, "perf_counter", lambda: clock[0])
    wrapper, _ = _canon_wrapper(training=False)
    real = wrapper._check_canon_metadata

    def slow_check(metadata):
        clock[0] += 5.0                                  # the validation "takes" 5 s
        return real(metadata)
    monkeypatch.setattr(wrapper, "_check_canon_metadata", slow_check)
    wrapper._canonicalize_batch(D._CpuMetadataCarrier([_canon_md(0, "none")]))
    assert wrapper.canon_latency["total_s"] == 5.0 and wrapper.canon_latency["batches"] == 1


# --------------------------------------------------------------------------- #
# 13 / 12c. _canonicalize_batch: schema + CPU-float64 contract, operator, counters
# --------------------------------------------------------------------------- #
W, H = 16, 8
TIES = {"none": {}, "same_column": {"peaks": [(1, 5), (6, 5)]},
        "cross_column_image": {"peaks": [(2, 3), (2, 11)]}}


def _canon_md(seed, tie="none", period=None, width=W):
    g = torch.Generator().manual_seed(1000 + seed)
    src, ctx = torch.randn(3, generator=g), torch.randn(2, 3, generator=g)
    return {"depth": _cloud(_scalar_depth(H, width, seed, period=period, **TIES[tie])),
            "source": src, "source_vit": src[None].clone(), "context_poses": ctx,
            "context_poses_vit": ctx.clone(), "context_audio": torch.randn(2, 1, 32, generator=g),
            "padding_mask": torch.ones(64), "scene": f"scene_{seed}", "idx": seed,
            "path": f"/fake/{seed}.wav"}


def _canon_wrapper(training=None):
    wrapper = _build(canonicalize={"enabled": True, "img_w": W})
    logged = []
    wrapper.log = lambda name, value, **kw: logged.append((name, value, kw))
    if training is not None:
        wrapper.trainer = types.SimpleNamespace(training=training)
    return wrapper, logged


def _expect_schema_error(metadata, *needles):
    with pytest.raises(ValueError) as excinfo:
        _canon_wrapper()[0]._canonicalize_batch(metadata)
    for needle in ("canonicalize:",) + needles:
        assert needle in str(excinfo.value), f"{excinfo.value!s} does not name {needle!r}"


def _with(field, value, i=1):
    batch = [_canon_md(0), _canon_md(1)]
    if value is _MISSING:
        del batch[i][field]
    else:
        batch[i][field] = value
    return batch


@pytest.mark.parametrize("metadata", [None, {}, "metadata", []])
def test_schema_rejects_a_non_list_or_empty_batch(metadata):
    _expect_schema_error(metadata)


@pytest.mark.parametrize("field, value, needle", [
    ("depth", _MISSING, "depth"), ("depth", [[0.0] * W], "depth"),
    ("depth", torch.zeros(3, W, dtype=torch.float64), "depth"),
    ("depth", torch.zeros(1, H, W, dtype=torch.float64), "depth"),
    ("depth", torch.zeros(3, H, W, 1, dtype=torch.float64), "depth"),
    ("depth", _cloud(_scalar_depth(H, 8, 0)), "8 columns"),
    ("depth", _cloud(_scalar_depth(H, W, 0)).to("meta"), "metadata[1]"),
    ("depth", _cloud(_scalar_depth(H, W, 0)).float(), "metadata[1]"),
    ("depth", _cloud(_scalar_depth(H, W, 0)).half(), "float64")]
    + [(k, v, k) for k in yr.POSE_KEYS for v in (_MISSING, [0.0, 0.0, 0.0], torch.tensor(1.0),
                                                 torch.zeros(1, 2))])
def test_schema_rejects_a_malformed_sample(field, value, needle):
    _expect_schema_error(_with(field, value), needle)


def test_schema_rejects_a_sample_that_is_not_a_dict():
    _expect_schema_error([_canon_md(0), "not-a-dict"], "metadata[1]")


@pytest.mark.parametrize("bad", ["cpu_device", "float64"])
def test_device_and_dtype_errors_name_the_contract(bad):
    depth = _cloud(_scalar_depth(H, W, 0))
    depth = depth.to("meta") if bad == "cpu_device" else depth.float()
    needle = "cpu" if bad == "cpu_device" else "float64"
    _expect_schema_error(_with("depth", depth, i=0), "metadata[0]", needle)


def _nan_depth():
    depth = _cloud(_scalar_depth(H, W, 0)).clone()
    depth[0, 1, 2] = float("nan")
    return depth


@pytest.mark.parametrize("field, value, needle", [  # passes the schema, fails the operator
    ("source_vit", torch.zeros(3), "source_vit"),
    ("context_poses_vit", torch.zeros(3, 3), "same N"),
    ("source", torch.tensor([0.0, float("inf"), 1.0]), "non-finite"),
    ("source", torch.zeros(3, dtype=torch.int64), "floating"),
    ("depth", _cloud(_scalar_depth(7, W, 0)), "even H"),
    ("depth", _nan_depth(), "non-finite")])
def test_operator_contract_errors_name_the_prefix_and_the_sample(field, value, needle):
    with pytest.raises(ValueError) as excinfo:
        _canon_wrapper()[0]._canonicalize_batch(_with(field, value))
    assert str(excinfo.value).startswith("canonicalize: metadata[1]: ") and needle in str(
        excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_batch_is_canonicalized_per_sample_without_mutating_the_input():
    wrapper, _ = _canon_wrapper()
    batch = tuple(_canon_md(i, tie) for i, tie in enumerate(TIE_CLASSES))
    pristine = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in md.items()}
                for md in batch]
    out = wrapper._canonicalize_batch(batch)
    assert isinstance(out, list) and len(out) == len(batch)
    for md_in, md_out, ref in zip(batch, out, pristine):
        expected, _, _ = yr.canonicalize_scene_metadata(ref, W)
        assert md_out is not md_in
        for key in ("depth",) + yr.POSE_KEYS:
            assert torch.equal(md_out[key], expected[key]), key
        assert yr.canonical_column(yr.recover_depth_proxy(md_out["depth"]))[0] == 0
        for key in ("context_audio", "padding_mask", "scene", "idx", "path"):
            assert md_out[key] is md_in[key], key
        for key, value in ref.items():
            if torch.is_tensor(value):
                assert torch.equal(md_in[key], value), f"input {key} mutated"


def test_tie_counts_accumulate_across_batches_and_touch_no_rng():
    wrapper, logged = _canon_wrapper()
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    before = _rng_digest()
    wrapper._canonicalize_batch([_canon_md(i, t) for i, t in enumerate(TIE_CLASSES + ("none",))])
    wrapper._canonicalize_batch([_canon_md(9, "cross_column_image")])
    assert _rng_digest() == before
    assert wrapper.canon_tie_counts == {"none": 2, "same_column": 1, "cross_column_image": 2}
    assert logged == []  # no Trainer attached -> nothing logged


def test_cumulative_counts_are_logged_on_the_training_path_only():
    wrapper, logged = _canon_wrapper(training=True)
    wrapper._canonicalize_batch([_canon_md(0, "cross_column_image"), _canon_md(1, "same_column")])
    wrapper._canonicalize_batch([_canon_md(2, "cross_column_image")])
    kw = {"rank_zero_only": True, "on_step": True, "on_epoch": False}
    ties = [entry for entry in logged if entry[0] != "canon/latency_ms"]   # r3b-F1 fix 2 adds it
    assert ties == [("canon/cross_column_ties", 1, kw), ("canon/same_column_ties", 1, kw),
                    ("canon/cross_column_ties", 2, kw), ("canon/same_column_ties", 1, kw)]
    assert all(type(value) is float for _, value, _ in logged)
    wrapper.trainer.training = False  # validation / test inside a Trainer
    wrapper._canonicalize_batch([_canon_md(3, "same_column")])
    assert len(logged) == 6 and wrapper.canon_tie_counts["same_column"] == 2


def test_a_periodic_panorama_fails_closed_and_counts_nothing():
    wrapper, _ = _canon_wrapper()
    with pytest.raises(yr.StabilizedTieError) as excinfo:
        wrapper._canonicalize_batch([_canon_md(0, "cross_column_image"),
                                     _canon_md(1, period=W // 2)])
    assert excinfo.value.sample == {"idx": 1, "path": "/fake/1.wav"}
    assert wrapper.canon_tie_counts == {c: 0 for c in TIE_CLASSES}
    # never re-wrapped by the ValueError prefixing (it is a RuntimeError, not a ValueError)
    assert type(excinfo.value) is yr.StabilizedTieError and excinfo.value.__cause__ is not None
    assert not issubclass(yr.StabilizedTieError, ValueError)
    assert not str(excinfo.value).startswith("canonicalize: metadata[")


# --------------------------------------------------------------------------- #
# 12. dispatch spy: canonicalization runs once per step, FIRST, at all three sites
# --------------------------------------------------------------------------- #
STEPS = ["training_step", "validation_step", "test_step"]


def _step_wrapper(monkeypatch, enabled=True):
    """A wrapper whose DiT / sampler / metric sink are neutralized (test_cond_dispatch)
    and whose _canonicalize_batch and _compute_conditioning record (and delegate)."""
    wrapper = _build(canonicalize={"enabled": True, "img_w": W}) if enabled else _build()
    _neutralize_for_steps(wrapper, monkeypatch)
    wrapper.trainer.training = True
    wrapper.log = lambda *a, **k: None
    events, canon, cond = [], wrapper._canonicalize_batch, wrapper._compute_conditioning

    def _canon(metadata):
        events.append(("canon", metadata))
        events.append(("canon_out", canon(metadata)))
        return events[-1][1]

    wrapper._canonicalize_batch = _canon
    wrapper._compute_conditioning = lambda md: events.append(("cond", md)) or cond(md)
    return wrapper, events


def _step_batch():
    return torch.randn(2, 1, 64), [_canon_md(0, "cross_column_image"), _canon_md(1)]


@pytest.mark.parametrize("step", STEPS)
def test_canonicalization_runs_once_first_and_feeds_the_conditioner(monkeypatch, step):
    wrapper, events = _step_wrapper(monkeypatch)
    reals, metadata = _step_batch()
    getattr(wrapper, step)((reals, metadata), 0)
    assert [name for name, _ in events] == ["canon", "canon_out", "cond"]
    assert events[0][1] is metadata and events[2][1] is events[1][1]
    for md in events[2][1]:
        assert yr.canonical_column(yr.recover_depth_proxy(md["depth"]))[0] == 0


@pytest.mark.parametrize("step", STEPS)
def test_disabled_steps_never_canonicalize(monkeypatch, step):
    wrapper, events = _step_wrapper(monkeypatch, enabled=False)
    reals, metadata = _step_batch()
    getattr(wrapper, step)((reals, metadata), 0)
    assert [name for name, _ in events] == ["cond"] and events[0][1] is metadata


def test_canonicalization_precedes_the_yaw_aug_hook(monkeypatch):
    """The two treatments are mutually exclusive by construction, so the flag is flipped
    on the built object purely to pin the hook ORDER inside training_step."""
    wrapper, events = _step_wrapper(monkeypatch)
    wrapper.yaw_aug_enabled = True
    wrapper._apply_yaw_aug = lambda md: events.append(("yaw", md)) or md
    wrapper.training_step(_step_batch(), 0)
    assert [name for name, _ in events] == ["canon", "canon_out", "yaw", "cond"]
    assert events[2][1] is events[1][1]


# --------------------------------------------------------------------------- #
# 12b. transfer_batch_to_device: enabled keeps metadata on the CPU; disabled = parent
# (the "meta" device stands in for an accelerator on CPU: whatever moves, shows it)
# --------------------------------------------------------------------------- #
META = torch.device("meta")


def _spy(monkeypatch, owner, name):
    calls, real = [], getattr(owner, name)

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(owner, name, spy)
    return calls


@pytest.mark.parametrize("container", [list, tuple])
def test_enabled_transfer_moves_reals_only(monkeypatch, container):
    calls = _spy(monkeypatch, pl_hooks, "move_data_to_device")
    reals, metadata = _step_batch()
    depths = [md["depth"] for md in metadata]
    out = _build(canonicalize=dict(CANON_ON)).transfer_batch_to_device(
        container((reals, metadata)), META, 0)
    assert len(calls) == 1 and calls[0][0] is reals and calls[0][1] == META
    assert type(out) is container and out[0].device == META
    assert getattr(out[1], "metadata", None) is metadata
    assert type(out[1]) is tdiff._CpuMetadataCarrier
    for md, depth in zip(out[1].metadata, depths):
        assert md["depth"] is depth and depth.device.type == "cpu" and depth.dtype == torch.float64


OTHER_SHAPES = [lambda r, md: {"reals": r, "md": md}, lambda r, md: [r, md, r],
                lambda r, md: [md, r], lambda r, md: (r, r)]


@pytest.mark.parametrize("enabled, make", [(True, m) for m in OTHER_SHAPES]
                         + [(False, m) for m in [lambda r, md: [r, md]] + OTHER_SHAPES])
def test_other_shapes_and_the_disabled_path_delegate_to_the_parent(monkeypatch, enabled, make):
    calls = _spy(monkeypatch, pl_hooks.DataHooks, "transfer_batch_to_device")
    batch = make(*_step_batch())
    wrapper = _build(canonicalize=dict(CANON_ON)) if enabled else _build()
    out = wrapper.transfer_batch_to_device(batch, META, 3)
    assert len(calls) == 1 and calls[0][1:] == (batch, META, 3)
    _assert_same_transfer(out, pl_hooks.move_data_to_device(batch, META))


def _assert_same_transfer(a, b):
    assert type(a) is type(b)
    if torch.is_tensor(a):
        assert (a.device, a.dtype, a.shape) == (b.device, b.dtype, b.shape)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b) and all(_assert_same_transfer(x, y) is None for x, y in zip(a, b))
    elif isinstance(a, dict):
        assert a.keys() == b.keys() and all(_assert_same_transfer(a[k], b[k]) is None for k in a)
    else:
        assert a == b


def _ddp_inputs(batch, device, side_stream):
    """What ``DistributedDataParallel._pre_forward`` does to ``training_step``'s positional
    inputs ``(batch, batch_idx)`` when ``device_ids`` is set (torch 2.7
    ``nn/parallel/distributed.py:1536``), through the REAL ``_to_kwargs``."""
    moved_inputs, moved_kwargs = _to_kwargs((batch, 0), {}, device, side_stream)
    assert moved_kwargs == ({},) and moved_inputs[0][1] == 0
    return moved_inputs[0]


@pytest.mark.parametrize("side_stream", [True, False])  # True is DDP's default
def test_enabled_metadata_survives_the_ddp_input_mover(monkeypatch, side_stream):
    wrapper, events = _step_wrapper(monkeypatch)
    reals, metadata = _step_batch()
    depths = [md["depth"] for md in metadata]
    hooked = wrapper.transfer_batch_to_device([reals, metadata], META, 0)
    batch, _ = _ddp_inputs(hooked, META, side_stream)
    carried = getattr(batch[1], "metadata", batch[1])
    assert type(batch) is list and batch[0].device == META
    assert all(md["depth"].device.type == "cpu" for md in carried), "DDP moved the metadata"
    assert batch[1] is hooked[1] and type(batch[1]) is tdiff._CpuMetadataCarrier
    assert carried is metadata and all(md["depth"] is d and d.dtype == torch.float64
                                       for md, d in zip(carried, depths))
    wrapper.training_step([reals, batch[1]], 0)  # the DDP-delivered carrier, CPU reals
    assert [name for name, _ in events] == ["canon", "canon_out", "cond"]
    assert events[0][1] is batch[1] and type(events[2][1]) is list
    assert events[2][1] is events[1][1] and len(events[2][1]) == len(metadata)
    for md in events[2][1]:
        assert md["depth"].device.type == "cpu"
        assert yr.canonical_column(yr.recover_depth_proxy(md["depth"]))[0] == 0


def test_the_carrier_is_opaque_to_both_recursive_movers():
    """Not a Tensor / list / tuple / dict / namedtuple / dataclass, no ``.to`` and no
    ``__dict__``: Lightning's ``move_data_to_device`` and DDP's ``_to_kwargs`` both return
    it, and the CPU float64 tensors inside it, unchanged."""
    _, metadata = _step_batch()
    carrier = tdiff._CpuMetadataCarrier(metadata)
    assert not isinstance(carrier, (torch.Tensor, list, tuple, dict))
    assert not dataclasses.is_dataclass(carrier) and not hasattr(carrier, "to")
    assert not hasattr(carrier, "__dict__") and carrier.metadata is metadata
    for moved in (pl_hooks.move_data_to_device([carrier], META)[0],
                  _ddp_inputs([carrier], META, True)[0][0]):
        assert moved is carrier and all(md["depth"].device.type == "cpu" for md in metadata)


@pytest.mark.parametrize("source", ["disabled_wrapper", "bare_pair"])
def test_without_the_carrier_the_ddp_mover_moves_every_metadata_tensor(source):
    """Baseline: the disabled path's batch, and a bare [reals, metadata] list (the round-2
    behaviour), lose their CPU metadata to DDP's mover."""
    reals, metadata = _step_batch()
    batch = (_build().transfer_batch_to_device([reals, metadata], META, 0)
             if source == "disabled_wrapper" else [reals.to(META), metadata])
    moved, _ = _ddp_inputs(batch, META, True)
    assert moved[0].device == META
    assert all(md[k].device == META for md in moved[1] for k in ("depth",) + yr.POSE_KEYS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_transfer_on_a_real_cuda_device_with_tiny_tensors():
    dev = torch.device("cuda", torch.cuda.current_device())
    reals, metadata = _step_batch()  # a few KB in total
    wrapper = _build(canonicalize={"enabled": True, "img_w": W})
    on = _ddp_inputs(wrapper.transfer_batch_to_device([reals, metadata], dev, 0), dev, True)[0]
    off = _ddp_inputs(_build().transfer_batch_to_device([reals, metadata], dev, 0), dev, True)[0]
    try:
        assert on[0].device.type == "cuda" and torch.equal(on[0].cpu(), reals)
        carried = getattr(on[1], "metadata", on[1])
        assert all(md["depth"].device.type == "cpu" for md in carried) and carried is metadata
        assert type(wrapper._canonicalize_batch(on[1])) is list
        assert all(md["depth"].device.type == "cuda" for md in off[1])
        with pytest.raises(ValueError, match=r"metadata\[1\]"):
            wrapper._canonicalize_batch([carried[0], off[1][1]])
    finally:
        del on, off
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 12c. order in a REAL Lightning fit: transfer -> canonicalize -> conditioning
# --------------------------------------------------------------------------- #
def test_lightning_fit_transfers_then_canonicalizes_then_conditions(monkeypatch, capsys,
                                                                    tmp_path):
    """One optimizer step of the toy model on CPU through pl.Trainer.fit, fed by the
    loader's own collation_fn, with the three hooks spied in place."""
    monkeypatch.setattr(_transformer, "flash_attn_func", None)  # CPU attention fallback
    cfg = _forward_config()
    cfg["training"]["canonicalize"] = {"enabled": True, "img_w": W}
    wrapper = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
    events = []
    transfer, canon, cond = (wrapper.transfer_batch_to_device, wrapper._canonicalize_batch,
                             wrapper._compute_conditioning)
    wrapper.transfer_batch_to_device = lambda *a: events.append(("transfer", transfer(*a))) \
        or events[-1][1]
    wrapper._canonicalize_batch = lambda md: events.append(("canon", md)) or canon(md)
    wrapper._compute_conditioning = lambda md: events.append(("cond", md)) or cond(md)
    samples = [(torch.randn(4, 64), _canon_md(i, tie))
               for i, tie in enumerate(("cross_column_image", "same_column"))]
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0,
                         log_every_n_steps=1, default_root_dir=str(tmp_path))
    trainer.fit(wrapper, torch.utils.data.DataLoader(samples, batch_size=2,
                                                     collate_fn=collation_fn))
    assert [name for name, _ in events] == ["transfer", "canon", "cond"]
    assert events[1][1] is events[0][1][1]
    assert type(events[1][1]) is tdiff._CpuMetadataCarrier
    assert all(md["depth"].device.type == "cpu" and md["depth"].dtype == torch.float64
               for md in events[1][1].metadata)
    lines = capsys.readouterr().out.splitlines()
    assert lines.count(f"canonicalize ENABLED img_w={W} tiebreak=image") == 1
    assert lines.count("canonicalize tie_counts rank0 none=0 same_column=1 "
                       "cross_column_image=1") == 1
    assert float(trainer.callback_metrics["canon/cross_column_ties"]) == 1.0
    assert float(trainer.callback_metrics["canon/same_column_ties"]) == 1.0
    # r3b-F1 fix 2: exactly one latency line, positive and finite, after a real 1-step fit
    lat = [LATENCY_LINE.fullmatch(ln) for ln in lines if ln.startswith("canonicalize latency")]
    assert len(lat) == 1 and lat[0] is not None, lines
    batches, samples, total_s, mean_ms = lat[0].groups()
    assert int(batches) >= 1 and int(samples) == 2 * int(batches)
    assert 0 < float(total_s) and 0 < float(mean_ms) < float("inf")
    assert float(trainer.callback_metrics["canon/latency_ms"]) > 0
