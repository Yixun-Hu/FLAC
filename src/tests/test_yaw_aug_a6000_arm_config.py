"""exp_17 round 1 — the A6000 Yaw-Aug arm config is P1's config plus one block.

Plan Rev 2 §2.1: ``FLAC_AR_YAWAUG_A6000.json`` must be a byte-copy of exp_07's
``FLAC_AR_BVp1.json`` — the config the P1 control arm was trained with — plus
**exactly one** addition, ``training.yaw_aug``. The whole experiment is a
single-delta contrast against P1, so any second difference between the two
files, however innocuous, silently turns it into a two-factor comparison that no
downstream statistic can untangle.

Hence a byte-level assertion built FORWARDS (construct the expected arm file
from the control's own bytes plus the pinned insertion, then compare wholesale)
rather than by subtraction, and ``read_bytes`` rather than ``read_text``: text
decoding applies universal-newline translation, under which a wholesale CRLF
rewrite would compare equal to the LF original.

The four pins below are not decoration. ``img_w`` in particular must equal the
ViT's ``img_w``: the augmentation rolls the depth panorama by an integer number
of columns, so a mismatch rotates the conditioning by the wrong angle while
training proceeds without error — a silent corruption of the treatment.

exp_07's file is read here and never written. Written by the main session seat
(Claude Opus 5, max effort) per the model-attribution rule.
"""
import json
from pathlib import Path

import pytest

from src.training.factory import _parse_yaw_aug_config


_REPO = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
ARM_CONFIG = _REPO / "worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json"

# The literal insertion, comma included: appended as the last key of the
# "training" block so every other byte of the control config is untouched.
INSERTED_BYTES = (
    b',\n'
    b'        "yaw_aug": {\n'
    b'            "enabled": true,\n'
    b'            "img_w": 512,\n'
    b'            "seed": 42\n'
    b'        }'
)

# The end of the control file: the close of "training" and of the root object.
TRAILER_BYTES = b'\n    }\n}'

EXPECTED_BLOCK = {"enabled": True, "img_w": 512, "seed": 42}


@pytest.fixture(scope="module")
def arm_bytes():
    assert ARM_CONFIG.is_file(), f"arm config not found: {ARM_CONFIG}"
    return ARM_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def control_bytes():
    assert CONTROL_CONFIG.is_file(), f"control config not found: {CONTROL_CONFIG}"
    return CONTROL_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def arm(arm_bytes):
    return json.loads(arm_bytes.decode())


@pytest.fixture(scope="module")
def control(control_bytes):
    return json.loads(control_bytes.decode())


# --------------------------------------------------------------------------- #
# 1. the single-delta proof
# --------------------------------------------------------------------------- #
def test_arm_config_is_the_control_plus_exactly_one_insertion(arm_bytes, control_bytes):
    assert control_bytes.count(TRAILER_BYTES) == 1, (
        "the end-of-training boundary is not unique in the control config; the "
        "insertion point can no longer be located unambiguously"
    )
    assert control_bytes.endswith(TRAILER_BYTES)

    prefix = control_bytes[: -len(TRAILER_BYTES)]
    expected_arm_bytes = prefix + INSERTED_BYTES + TRAILER_BYTES

    assert arm_bytes == expected_arm_bytes, (
        "FLAC_AR_YAWAUG_A6000.json is not byte-for-byte P1's config plus the "
        "yaw_aug block — the arm would no longer be a single-delta treatment"
    )


def test_byte_comparison_would_catch_newline_drift(control_bytes):
    """Non-vacuity guard for the test above: prove the comparison is strict.

    Done entirely in memory; no file is touched.
    """
    prefix = control_bytes[: -len(TRAILER_BYTES)]
    genuine = prefix + INSERTED_BYTES + TRAILER_BYTES
    crlf_drifted = genuine.replace(b"\n", b"\r\n")

    assert crlf_drifted != genuine, "the drift fixture is a no-op"
    assert crlf_drifted.decode().splitlines() == genuine.decode().splitlines(), (
        "text-level comparison cannot tell these apart — which is exactly why "
        "the assertion above is on bytes"
    )


def strict_diff(a, b, path="root"):
    """First type-then-value difference between two parsed JSON objects, or None.

    Module level so the non-vacuity guard below can call the very same function
    the contract test relies on.
    """
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: key sets differ ({set(a) ^ set(b)})"
        for k in a:
            found = strict_diff(a[k], b[k], f"{path}.{k}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = strict_diff(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def test_removing_yaw_aug_leaves_a_type_strict_copy_of_the_control(arm, control):
    """Semantic mirror of the byte test: types must match, not just values.

    ``0 == 0.0`` and ``1 == True`` in Python, so a numeric-type drift in any
    field would pass a plain ``==`` comparison of the parsed objects.
    """
    stripped = json.loads(json.dumps(arm))
    stripped["training"].pop("yaw_aug")

    assert strict_diff(stripped, control) is None


def test_the_strict_comparator_rejects_a_type_drift_that_plain_equality_misses(control):
    """Non-vacuity guard: prove ``strict_diff`` is stricter than ``==``.

    ``mask_padding_dropout`` is ``0.0`` in the control; rewriting it as the int
    ``0`` leaves the two objects ``==``-equal while changing the type the
    trainer would see.
    """
    drifted = json.loads(json.dumps(control))
    assert isinstance(drifted["training"]["mask_padding_dropout"], float)
    drifted["training"]["mask_padding_dropout"] = 0  # int 0 == float 0.0

    assert drifted == control, "plain equality cannot see the type change"

    found = strict_diff(drifted, control)
    assert found is not None and "mask_padding_dropout" in found, (
        f"strict_diff missed an int-for-float substitution: {found!r}"
    )


# --------------------------------------------------------------------------- #
# 2. the treatment block itself
# --------------------------------------------------------------------------- #
def test_the_yaw_aug_block_is_exactly_the_registered_treatment(arm):
    block = arm["training"]["yaw_aug"]
    assert block == EXPECTED_BLOCK
    assert isinstance(block["enabled"], bool), "enabled must be a literal boolean"
    assert isinstance(block["img_w"], int) and not isinstance(block["img_w"], bool)
    assert isinstance(block["seed"], int) and not isinstance(block["seed"], bool)


def test_the_factory_accepts_the_arm_block_and_enables_the_treatment(arm):
    """The config must survive the real parser, not just look right."""
    kwargs = _parse_yaw_aug_config(arm["training"])
    assert kwargs.get("yaw_aug_enabled") is True, kwargs
    assert kwargs.get("yaw_aug_img_w") == 512
    assert kwargs.get("yaw_aug_seed") == 42


def test_the_control_config_enables_nothing(control):
    """P1's own config must yield no augmentation kwargs at all."""
    assert _parse_yaw_aug_config(control["training"]) == {}


# --------------------------------------------------------------------------- #
# 3. the four pins (plan Rev 2 §2.1)
# --------------------------------------------------------------------------- #
def test_pin_vanilla_conditioning(arm):
    assert "cond_method" not in arm["training"], (
        "this arm must be vanilla-conditioned; a cond_method here would make the "
        "contrast against P1 two-factor"
    )


def test_pin_ema_is_on(arm):
    assert arm["training"]["use_ema"] is True


def test_pin_gradient_checkpointing_on_both_vit_conditioners(arm):
    vits = [
        c for c in arm["model"]["conditioning"]["configs"]
        if c["type"] == "ViTCoordinates"
    ]
    assert len(vits) == 2, f"expected two ViT conditioners, found {len(vits)}"
    for c in vits:
        assert c["config"]["gradient_checkpointing"] is True, c["id"]


def test_pin_augmentation_width_matches_the_vit_panorama_width(arm):
    """The pin that silently corrupts the treatment if it drifts.

    The augmentation rolls the depth panorama by an integer number of columns of
    width ``yaw_aug.img_w``; the ViT consumes a panorama of width
    ``ViT.img_w``. If the two disagree, every training sample is rotated by the
    wrong angle and nothing raises.
    """
    aug_w = arm["training"]["yaw_aug"]["img_w"]
    vit_ws = {
        c["config"]["ViT"]["img_w"]
        for c in arm["model"]["conditioning"]["configs"]
        if c["type"] == "ViTCoordinates"
    }
    assert vit_ws == {aug_w}, f"yaw_aug.img_w={aug_w} but ViT widths are {vit_ws}"


# --------------------------------------------------------------------------- #
# 4. mutation guards — the pins must actually fail when violated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda t: t["yaw_aug"].__setitem__("enabled", "true"), "string instead of bool"),
        (lambda t: t["yaw_aug"].__setitem__("enabled", 1), "int instead of bool"),
        (lambda t: t["yaw_aug"].__setitem__("img_w", 512.0), "float instead of int"),
        (lambda t: t["yaw_aug"].__setitem__("rotate", True), "unknown key"),
        (lambda t: t["yaw_aug"].pop("enabled"), "missing enabled"),
    ],
)
def test_the_factory_rejects_a_malformed_treatment_block(arm, mutate, why):
    training = json.loads(json.dumps(arm))["training"]
    mutate(training)
    with pytest.raises(ValueError):
        _parse_yaw_aug_config(training)


def test_a_width_mismatch_is_caught_by_the_pin(arm):
    """Mutation guard for the img_w pin: 256 must fail the check above."""
    mutated = json.loads(json.dumps(arm))
    mutated["training"]["yaw_aug"]["img_w"] = 256

    aug_w = mutated["training"]["yaw_aug"]["img_w"]
    vit_ws = {
        c["config"]["ViT"]["img_w"]
        for c in mutated["model"]["conditioning"]["configs"]
        if c["type"] == "ViTCoordinates"
    }
    assert vit_ws != {aug_w}, (
        "the width pin would not notice a 256/512 mismatch — it is vacuous"
    )


def test_fa_invariant_plus_yaw_aug_is_refused(arm):
    """The parser's own guard: this combination is untested and must fail closed."""
    training = json.loads(json.dumps(arm))["training"]
    training["cond_method"] = "fa_invariant"
    with pytest.raises(ValueError, match="fa_invariant"):
        _parse_yaw_aug_config(training)
