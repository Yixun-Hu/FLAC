"""exp_43 std_aug -- the optional ``training.yaw_aug.flip`` key of the factory (plan §3.2).

Rules, mirroring exp_40's ``offset_step`` exactly:

* unknown keys are ALWAYS refused (enabled or not);
* ``enabled: false`` -> every optional value is ignored, the parser returns ``{}``;
* enabled -> ``flip`` must be a literal JSON boolean (nothing is coerced);
* ``flip: true`` adds ``yaw_aug_flip=True``; ``flip`` absent or ``false`` adds NO kwarg, so the
  wrapper call of every existing config (and of an explicit ``false``) is literally unchanged;
* ``flip: true`` together with ``offset_step`` is refused (the RF arm is the continuous roll).

Every rule is checked on ``_parse_yaw_aug_config`` and, for the kwargs, through the real
``create_training_wrapper_from_config`` with the wrapper class stubbed (the model is never
touched on the ``diffusion_cond`` branch, hence a sentinel).
"""
import numpy as np
import pytest

import src.training.diffusion as tdiff
from src.training import factory
from src.training.factory import _parse_yaw_aug_config, create_training_wrapper_from_config

_ENABLED = {"enabled": True, "img_w": 512, "seed": 42}
_EXP15_KWARGS = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}
_SENTINEL_MODEL = object()


def _training(block, **extra):
    return {"yaw_aug": block, **extra}


def _wrapper_kwargs(monkeypatch, training):
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    create_training_wrapper_from_config(
        {"model_type": "diffusion_cond", "training": dict(training)}, _SENTINEL_MODEL)
    assert captured["model"] is _SENTINEL_MODEL
    return captured["kwargs"]


def test_flip_is_appended_last_to_the_allowed_keys():
    assert factory.YAW_AUG_KEYS == ("enabled", "img_w", "seed", "offset_step", "flip")


# --- absent / false: no kwarg, the literal pre-change call ------------------ #
def test_absent_flip_is_the_exp15_kwargs():
    got = _parse_yaw_aug_config(_training(dict(_ENABLED)))
    assert got == _EXP15_KWARGS and "yaw_aug_flip" not in got


def test_false_flip_adds_no_kwarg():
    got = _parse_yaw_aug_config(_training({**_ENABLED, "flip": False}))
    assert got == _EXP15_KWARGS and "yaw_aug_flip" not in got


def test_false_flip_makes_literally_the_same_wrapper_call_as_absent(monkeypatch):
    absent = _wrapper_kwargs(monkeypatch, _training(dict(_ENABLED)))
    false = _wrapper_kwargs(monkeypatch, _training({**_ENABLED, "flip": False}))
    assert list(false) == list(absent) and false == absent
    assert "yaw_aug_flip" not in false


def test_false_flip_with_offset_step_is_the_exp40_kwargs():
    got = _parse_yaw_aug_config(_training({**_ENABLED, "offset_step": 32, "flip": False}))
    assert got == {**_EXP15_KWARGS, "yaw_aug_offset_step": 32}


# --- true: exactly one added kwarg ------------------------------------------ #
def test_true_flip_adds_exactly_yaw_aug_flip_true():
    got = _parse_yaw_aug_config(_training({**_ENABLED, "flip": True}))
    assert got == {**_EXP15_KWARGS, "yaw_aug_flip": True}
    assert got["yaw_aug_flip"] is True


def test_true_flip_reaches_the_wrapper_call_and_changes_nothing_else(monkeypatch):
    absent = _wrapper_kwargs(monkeypatch, _training(dict(_ENABLED)))
    flip = _wrapper_kwargs(monkeypatch, _training({**_ENABLED, "flip": True}))
    assert set(flip) == set(absent) | {"yaw_aug_flip"}
    assert flip["yaw_aug_flip"] is True
    assert {k: v for k, v in flip.items() if k != "yaw_aug_flip"} == absent


def test_key_order_in_the_block_does_not_matter():
    block = {"flip": True, "seed": 42, "img_w": 512, "enabled": True}
    assert _parse_yaw_aug_config(_training(block)) == {**_EXP15_KWARGS, "yaw_aug_flip": True}


# --- refusals ---------------------------------------------------------------- #
@pytest.mark.parametrize("flip", [1, 0, 1.0, 0.0, "true", "false", "True", None, [], {},
                                  [True], np.bool_(True)])
def test_enabled_block_refuses_a_non_literal_bool_flip(flip):
    with pytest.raises(ValueError, match=r"training\.yaw_aug\.flip must be a literal boolean"):
        _parse_yaw_aug_config(_training({**_ENABLED, "flip": flip}))


@pytest.mark.parametrize("step", [32, 1, 4])
def test_true_flip_with_offset_step_is_refused(step):
    with pytest.raises(ValueError,
                       match=r"training\.yaw_aug\.flip=true with training\.yaw_aug\.offset_step"):
        _parse_yaw_aug_config(_training({**_ENABLED, "offset_step": step, "flip": True}))


def test_true_flip_with_an_invalid_offset_step_is_still_refused():
    with pytest.raises(ValueError, match=r"training\.yaw_aug\.offset_step: "):
        _parse_yaw_aug_config(_training({**_ENABLED, "offset_step": 30, "flip": True}))


@pytest.mark.parametrize("key", ["flips", "mirror", "Flip", "flip_p", "hflip"])
@pytest.mark.parametrize("enabled", [True, False])
def test_unknown_keys_are_always_refused(key, enabled):
    block = {**_ENABLED, "enabled": enabled, key: True}
    with pytest.raises(ValueError, match=key):
        _parse_yaw_aug_config(_training(block))


def test_true_flip_with_fa_invariant_is_refused():
    with pytest.raises(ValueError, match="yaw_aug.enabled=true with cond_method='fa_invariant'"):
        _parse_yaw_aug_config(_training({**_ENABLED, "flip": True}, cond_method="fa_invariant"))


def test_canonicalize_refuses_a_flip_arm():
    t = _training({**_ENABLED, "flip": True}, canonicalize={"enabled": True, "img_w": 512})
    with pytest.raises(ValueError, match="canonicalize.enabled=true with training.yaw_aug.enabled=true"):
        factory._parse_canonicalize_config(t)


# --- disabled: optional values ignored --------------------------------------- #
@pytest.mark.parametrize("flip", [True, False, "garbage", 1, None])
def test_disabled_block_ignores_flip(monkeypatch, flip):
    block = {"enabled": False, "flip": flip}
    assert _parse_yaw_aug_config(_training(block)) == {}
    assert _parse_yaw_aug_config(_training({**block, "offset_step": 32})) == {}
    kwargs = _wrapper_kwargs(monkeypatch, _training(block))
    assert not any(k.startswith("yaw_aug") for k in kwargs)
    assert kwargs == _wrapper_kwargs(monkeypatch, {})
