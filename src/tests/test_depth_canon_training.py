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
import pytest

import src.training.diffusion as tdiff
from src.training.factory import (
    CANON_KEYS, _parse_canonicalize_config, create_training_wrapper_from_config)
from test_cond_dispatch import _base_config  # tiny CPU diffusion_cond (pytest prepends src/tests)
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
