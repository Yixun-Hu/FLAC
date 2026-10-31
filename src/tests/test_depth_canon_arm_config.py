"""exp_25 depth_canon -- the CANON arm config is P1's config plus exactly one block
(plan_depth_canon.md v7 §3, §6 test 22).

Written by the exp_25 Coder seat (Claude Opus, max effort) per plan_depth_canon.md
v7 §2.2/§3/§6/§8.

``FLAC_AR_CANON.json`` must be exp_07's ``FLAC_AR_BVp1.json`` (the config P1 was
trained with, sha256 pinned below) plus ONE insertion, ``training.canonicalize``, as
the last key of ``training``. The proof is built FORWARDS on bytes (exp_15/exp_17
pattern): the expected arm file is constructed from the control's own bytes plus
the literal insertion, then compared wholesale -- ``read_text`` would apply
universal-newline decoding, under which a CRLF rewrite compares equal. The control
file is read here and never written.
"""
import hashlib
import json
from pathlib import Path

import pytest

import src.training.diffusion as tdiff
from src.training.factory import (
    _parse_are_config, _parse_canonicalize_config, _parse_frame_avg_cap_config,
    _parse_yaw_aug_config, create_training_wrapper_from_config)
from test_yaw_aug_training import PRE_CHANGE_WRAPPER_KWARGS  # (pytest prepends src/tests)

_REPO = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARM_CONFIG = _REPO / "worklog/worklog_yixun/exp_25_depth_canon_claude/FLAC_AR_CANON.json"
CONTROL_SHA256 = "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"

INSERTED_BYTES = (b',\n'
                  b'        "canonicalize": {\n'
                  b'            "enabled": true,\n'
                  b'            "img_w": 512\n'
                  b'        }')
TRAILER_BYTES = b'\n    }\n}'  # close of "training" and of the root object
EXPECTED_BLOCK = {"enabled": True, "img_w": 512}


@pytest.fixture(scope="module")
def control_bytes():
    return CONTROL_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def arm_bytes():
    assert ARM_CONFIG.is_file(), f"arm config not found: {ARM_CONFIG}"
    return ARM_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def arm(arm_bytes):
    return json.loads(arm_bytes.decode())


def test_the_control_config_is_the_pinned_one(control_bytes):
    assert hashlib.sha256(control_bytes).hexdigest() == CONTROL_SHA256


def test_arm_is_the_control_plus_exactly_one_insertion(arm_bytes, control_bytes):
    assert control_bytes.count(TRAILER_BYTES) == 1 and control_bytes.endswith(TRAILER_BYTES)
    expected = control_bytes[: -len(TRAILER_BYTES)] + INSERTED_BYTES + TRAILER_BYTES
    assert arm_bytes == expected


def test_byte_comparison_would_catch_newline_drift(control_bytes):
    genuine = control_bytes[: -len(TRAILER_BYTES)] + INSERTED_BYTES + TRAILER_BYTES
    drifted = genuine.replace(b"\n", b"\r\n")
    assert drifted != genuine and drifted.decode().splitlines() == genuine.decode().splitlines()


def _strict_diff(a, b, path="root"):
    """First type-then-value difference of two parsed JSON objects, or None
    (``1 == True`` and ``0 == 0.0`` in Python, so plain ``==`` is not enough)."""
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys {list(a)} != {list(b)}"
        return next((d for k in a if (d := _strict_diff(a[k], b[k], f"{path}.{k}"))), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        return next((d for i, (x, y) in enumerate(zip(a, b))
                     if (d := _strict_diff(x, y, f"{path}[{i}]"))), None)
    return None if a == b else f"{path}: {a!r} != {b!r}"


def test_parsed_arm_is_the_parsed_control_plus_the_block(arm, control_bytes):
    expected = json.loads(control_bytes.decode())
    expected["training"]["canonicalize"] = dict(EXPECTED_BLOCK)
    assert _strict_diff(arm, expected) is None  # values, types AND key order
    assert list(arm["training"])[-1] == "canonicalize"


def test_the_strict_comparator_is_stricter_than_equality(control_bytes):
    control = json.loads(control_bytes.decode())
    drifted = json.loads(control_bytes.decode())
    drifted["training"]["mask_padding_dropout"] = 0  # int 0 == float 0.0
    assert drifted == control and "mask_padding_dropout" in _strict_diff(drifted, control)


def _vits(cfg):
    return [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]


def test_arm_pins_vanilla_ema_and_gradient_checkpointing(arm):
    assert "cond_method" not in arm["training"] and "yaw_aug" not in arm["training"]
    assert arm["training"]["use_ema"] is True
    assert [c["id"] for c in _vits(arm)] == ["source_vit", "context_poses_vit"]
    assert all(c["config"]["gradient_checkpointing"] is True for c in _vits(arm))


def check_width_pin(cfg):
    """ONE checker for the pin and its mutation guard: the canonical shift is an integer
    number of panorama columns, so a width mismatch would rotate by the wrong angle."""
    widths = {c["config"]["ViT"]["img_w"] for c in _vits(cfg)}
    if widths != {cfg["training"]["canonicalize"]["img_w"]} or widths != {512}:
        raise AssertionError(f"canonicalize.img_w / ViT img_w mismatch: {widths}")


def test_canonicalize_width_matches_both_vit_widths(arm):
    check_width_pin(arm)
    for mutate in (lambda c: c["training"]["canonicalize"].__setitem__("img_w", 256),
                   lambda c: _vits(c)[1]["config"]["ViT"].__setitem__("img_w", 256)):
        mutated = json.loads(json.dumps(arm))
        mutate(mutated)
        with pytest.raises(AssertionError, match="mismatch"):
            check_width_pin(mutated)


def test_parsers_see_exactly_one_treatment(arm, control_bytes):
    control = json.loads(control_bytes.decode())
    assert _parse_canonicalize_config(arm["training"]) == {
        "canonicalize_enabled": True, "canonicalize_img_w": 512}
    assert _parse_canonicalize_config(control["training"]) == {}
    assert _parse_yaw_aug_config(arm["training"]) == {}
    assert _parse_frame_avg_cap_config(arm["training"]) == {}
    assert _parse_are_config(arm, arm["training"]) == {}


def test_the_real_factory_accepts_the_arm(monkeypatch, arm):
    captured = {}

    class _Stub:  # the wrapper is stubbed so the test does not build DINOv3
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    create_training_wrapper_from_config(json.loads(json.dumps(arm)), object())
    assert set(captured) == PRE_CHANGE_WRAPPER_KWARGS | {"canonicalize_enabled",
                                                         "canonicalize_img_w"}
    assert captured["canonicalize_enabled"] is True and captured["canonicalize_img_w"] == 512
    assert captured["cond_method"] == "vanilla" and captured["use_ema"] is True
