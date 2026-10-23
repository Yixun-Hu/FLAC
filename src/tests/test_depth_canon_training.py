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
import types

import pytest
import torch

import src.training.diffusion as tdiff
from src.data import yaw_rotation as yr
from src.data.yaw_rotation import TIE_CLASSES
from src.models.factory import create_model_from_config
from src.training.factory import (
    CANON_KEYS, _parse_canonicalize_config, create_training_wrapper_from_config)
from test_cond_dispatch import _base_config  # tiny CPU diffusion_cond (pytest prepends src/tests)
from test_depth_canon import _cloud, _scalar_depth  # synthetic depth via the real AR_md
from test_yaw_aug_training import PRE_CHANGE_WRAPPER_KWARGS

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
    assert len(calls) == 2 and all(kw.get("flush") is True for kw in calls)


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
        "canonicalize tie_counts rank0 none=7 same_column=2 cross_column_image=3"]


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
        _canon_wrapper()[0]._check_canon_metadata(metadata)
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
