"""exp_46 c4flip_data_curve -- plan r3 §3.1 test 1 (the factory) and the operator byte-identity.

C4F = C4 + flip: ``training.yaw_aug = {"enabled": true, "img_w": 512, "seed": 42, "offset_step": 128,
"flip": true}`` (plan §2, decision C1). Until exp_46 the factory refused ``flip: true`` together with
``offset_step`` (an exp_43 scope guard, plan G1). exp_46 removes that refusal and nothing else:

* the C4F block parses to exactly the five kwargs, in ``YAW_AUG_KEYS`` order, and reaches the real
  ``create_training_wrapper_from_config`` call unchanged (wrapper class stubbed, as exp_43's factory test);
* an invalid ``offset_step`` is still refused by the step check, flip or not;
* the RF, C16, YAW and disabled blocks -- inline and as the committed arm configs -- parse to exactly the
  kwargs of the factory at ``X45f`` (exp_45's frozen tip: its ``src/training/factory.py`` is executed from git
  as an independent oracle), which also still refuses the C4F block;
* ``src/data/yaw_rotation.py``, ``yaw_mirror.py`` and ``yaw_offset_step.py`` are byte-identical to ``X45f``.
"""
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

import src.training.diffusion as tdiff
from src.training import factory
from src.training.factory import _parse_yaw_aug_config, create_training_wrapper_from_config

_REPO = Path(__file__).resolve().parents[2]
X45F = "1e37ead2a352bd78d50cdda9ed02e95738a4ad04"        # exp_45's frozen tip, the base of exp_46 (plan §3.0)

C4F_BLOCK = {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 128, "flip": True}
C4F_KWARGS = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
              "yaw_aug_offset_step": 128, "yaw_aug_flip": True}
_EXP15_KWARGS = {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}
_SENTINEL_MODEL = object()

# the committed arm configs whose parse must not move (plan §3.1 test 1: regression)
ARM_CONFIGS = {
    "rf": "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json",
    "c16": "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/FLAC_AR_YAWAUG_C16.json",
    "yaw": "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
    "disabled": "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
}
ARM_KWARGS = {
    "rf": {**_EXP15_KWARGS, "yaw_aug_flip": True},
    "c16": {**_EXP15_KWARGS, "yaw_aug_offset_step": 32},
    "yaw": dict(_EXP15_KWARGS),
    "disabled": {},
}


def _git_show(rev, rel):
    return subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=str(_REPO), capture_output=True,
                          check=True).stdout


def _factory_at_x45f():
    """``src/training/factory.py`` as committed at ``X45f``, executed as a sibling module of the live package
    (its relative imports resolve to the live ``src.data`` / ``src.models``; the operators it imports are
    byte-identical to ``X45f``, tested below)."""
    mod = types.ModuleType("src.training._exp46_factory_at_x45f")
    mod.__package__ = "src.training"
    mod.__file__ = f"<{X45F[:8]}:src/training/factory.py>"
    exec(compile(_git_show(X45F, "src/training/factory.py"), mod.__file__, "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def x45f():
    return _factory_at_x45f()


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


# --------------------------------------------------------------------------- #
# the C4F block
# --------------------------------------------------------------------------- #
def test_the_c4f_block_parses_to_exactly_the_five_kwargs():
    got = _parse_yaw_aug_config(_training(dict(C4F_BLOCK)))
    assert got == C4F_KWARGS
    assert list(got) == ["yaw_aug_enabled", "yaw_aug_img_w", "yaw_aug_seed", "yaw_aug_offset_step",
                         "yaw_aug_flip"]
    assert got["yaw_aug_flip"] is True and type(got["yaw_aug_offset_step"]) is int
    assert list(C4F_BLOCK) == list(factory.YAW_AUG_KEYS)          # the registered key order (plan §2)


def test_the_c4f_block_in_any_key_order_parses_the_same():
    block = {"flip": True, "offset_step": 128, "seed": 42, "img_w": 512, "enabled": True}
    assert _parse_yaw_aug_config(_training(block)) == C4F_KWARGS


def test_the_c4f_block_reaches_the_real_wrapper_call_with_exactly_the_two_added_kwargs(monkeypatch):
    plain = _wrapper_kwargs(monkeypatch, {})
    yaw = _wrapper_kwargs(monkeypatch, _training({"enabled": True, "img_w": 512, "seed": 42}))
    c4f = _wrapper_kwargs(monkeypatch, _training(dict(C4F_BLOCK)))
    assert {k: v for k, v in c4f.items() if k.startswith("yaw_aug")} == C4F_KWARGS
    assert {k: v for k, v in c4f.items() if k not in ("yaw_aug_offset_step", "yaw_aug_flip")} == yaw
    assert {k: v for k, v in c4f.items() if not k.startswith("yaw_aug")} == plain


@pytest.mark.parametrize("step", [30, 0, -128, 512, 1024, 100, True, 128.0, "128", None])
@pytest.mark.parametrize("flip", [True, False])
def test_an_invalid_offset_step_is_still_refused_by_the_step_check(step, flip):
    with pytest.raises(ValueError, match=r"training\.yaw_aug\.offset_step: "):
        _parse_yaw_aug_config(_training({**C4F_BLOCK, "offset_step": step, "flip": flip}))


@pytest.mark.parametrize("flip", [1, 0, 1.0, "true", None, np.bool_(True), [True]])
def test_a_non_literal_bool_flip_with_the_grid_is_still_refused(flip):
    with pytest.raises(ValueError, match=r"training\.yaw_aug\.flip must be a literal boolean"):
        _parse_yaw_aug_config(_training({**C4F_BLOCK, "flip": flip}))


def test_the_existing_exclusions_still_refuse_the_c4f_block():
    with pytest.raises(ValueError, match="yaw_aug.enabled=true with cond_method='fa_invariant'"):
        _parse_yaw_aug_config(_training(dict(C4F_BLOCK), cond_method="fa_invariant"))
    t = _training(dict(C4F_BLOCK), canonicalize={"enabled": True, "img_w": 512})
    with pytest.raises(ValueError, match="canonicalize.enabled=true with training.yaw_aug.enabled=true"):
        factory._parse_canonicalize_config(t)


def test_the_factory_at_x45f_refused_the_c4f_block(x45f):
    """The before-state: exactly the exp_43 scope refusal (plan G1), which exp_46 removes."""
    with pytest.raises(ValueError, match=r"training\.yaw_aug\.flip=true with training\.yaw_aug\.offset_step "
                                         r"is out of scope for exp_43"):
        x45f._parse_yaw_aug_config(_training(dict(C4F_BLOCK)))


# --------------------------------------------------------------------------- #
# regression: every other block parses exactly as at X45f
# --------------------------------------------------------------------------- #
_BLOCKS = {
    "rf": {"enabled": True, "img_w": 512, "seed": 42, "flip": True},
    "rf_any_order": {"flip": True, "seed": 42, "img_w": 512, "enabled": True},
    "c16": {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 32},
    "c16_flip_false": {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 32, "flip": False},
    "c4_no_flip": {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 128},
    "c4_flip_false": {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 128, "flip": False},
    "yaw": {"enabled": True, "img_w": 512, "seed": 42},
    "yaw_flip_false": {"enabled": True, "img_w": 512, "seed": 42, "flip": False},
    "disabled": {"enabled": False},
    "disabled_with_c4f_values": {"enabled": False, "offset_step": 128, "flip": True},
    "disabled_with_junk": {"enabled": False, "offset_step": "x", "flip": "garbage"},
}
_BLOCK_KWARGS = {
    "rf": {**_EXP15_KWARGS, "yaw_aug_flip": True},
    "rf_any_order": {**_EXP15_KWARGS, "yaw_aug_flip": True},
    "c16": {**_EXP15_KWARGS, "yaw_aug_offset_step": 32},
    "c16_flip_false": {**_EXP15_KWARGS, "yaw_aug_offset_step": 32},
    "c4_no_flip": {**_EXP15_KWARGS, "yaw_aug_offset_step": 128},
    "c4_flip_false": {**_EXP15_KWARGS, "yaw_aug_offset_step": 128},
    "yaw": dict(_EXP15_KWARGS),
    "yaw_flip_false": dict(_EXP15_KWARGS),
    "disabled": {},
    "disabled_with_c4f_values": {},
    "disabled_with_junk": {},
}


@pytest.mark.parametrize("name", sorted(_BLOCKS))
def test_every_other_block_parses_exactly_as_at_x45f(x45f, name):
    block = _BLOCKS[name]
    got = _parse_yaw_aug_config(_training(dict(block)))
    want = x45f._parse_yaw_aug_config(_training(dict(block)))
    assert got == want == _BLOCK_KWARGS[name]
    assert list(got) == list(want)


def test_an_absent_block_parses_to_nothing_as_at_x45f(x45f):
    assert _parse_yaw_aug_config({}) == x45f._parse_yaw_aug_config({}) == {}


@pytest.mark.parametrize("arm", sorted(ARM_CONFIGS))
def test_the_committed_arm_configs_parse_exactly_as_at_x45f(x45f, arm):
    cfg = json.loads((_REPO / ARM_CONFIGS[arm]).read_text())
    assert (_REPO / ARM_CONFIGS[arm]).read_bytes() == _git_show(X45F, ARM_CONFIGS[arm])
    got = _parse_yaw_aug_config(cfg["training"])
    want = x45f._parse_yaw_aug_config(cfg["training"])
    assert got == want == ARM_KWARGS[arm]
    assert list(got) == list(want)


@pytest.mark.parametrize("name", ["c16", "c4_no_flip", "rf", "yaw"])
def test_every_other_block_reaches_the_wrapper_call_as_at_x45f(monkeypatch, x45f, name):
    """The live factory's full wrapper call for each non-C4F block equals the X45f factory's."""
    captured = []

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    for mod in (factory, x45f):
        mod.create_training_wrapper_from_config(
            {"model_type": "diffusion_cond", "training": _training(dict(_BLOCKS[name]))}, _SENTINEL_MODEL)
    assert len(captured) == 2 and captured[0] == captured[1] and list(captured[0]) == list(captured[1])


# --------------------------------------------------------------------------- #
# the operators stay byte-identical (plan §3.1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", ["src/data/yaw_rotation.py", "src/data/yaw_mirror.py",
                                 "src/data/yaw_offset_step.py"])
def test_the_rotation_mirror_and_grid_modules_are_byte_identical_to_x45f(rel):
    assert hashlib.sha256((_REPO / rel).read_bytes()).hexdigest() == \
        hashlib.sha256(_git_show(X45F, rel)).hexdigest()
