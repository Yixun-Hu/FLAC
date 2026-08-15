"""exp_17 — the A6000 Yaw-Aug arm config vs P1's, delta by delta.

Plan Rev 2 §2.1 required ``FLAC_AR_YAWAUG_A6000.json`` to be exp_07's
``FLAC_AR_BVp1.json`` — the config P1 was trained with — plus **exactly one**
addition, ``training.yaw_aug``.

**Amendment (Yixun, 2026-08-15): ViT gradient checkpointing is turned OFF** to
buy wall-clock (~43 h → ~30 h projected) now that both A6000s are dedicated to
this arm. That makes the file differ from P1 in THREE places, so the contract
below asserts exactly those three and nothing else:

  1. ``training.yaw_aug`` added                        — the treatment
  2. ``source_vit.gradient_checkpointing``     true→false
  3. ``context_poses_vit.gradient_checkpointing`` true→false

Deltas 2–3 change *how* activations are obtained in the backward pass —
recomputed rather than stored. Gradient checkpointing is **intended** to
preserve the mathematical computation. What that intent is actually backed by,
at its true strength (corrected twice: r2 caught the docstring claiming bitwise
identity, r3 caught the surviving claim that it "cannot move the trajectory"):

  - ``src/tests/test_vit_gradient_checkpointing.py::test_gradient_identity_on_vs_off``
    pins ON-vs-OFF parameter gradients as ``torch.allclose(atol=1e-6,
    rtol=1e-5)`` over ≥100 tensors — **not** ``torch.equal``, and it only
    *prints* the observed max difference rather than asserting it is 0.0.
  - That probe runs in **float32 on CPU**; this arm trains **bf16-mixed on
    CUDA**, where bitwise reproducibility is not guaranteed in either
    configuration.
  - The "210 tensors, max abs diff 0.0" figure is a one-time exp_07 worklog
    *observation*, not a pinned invariant. It is genuine, and it is consistent
    with the maths, but nothing in CI re-checks it.

So the honest statement is: checkpointing is *intended* to preserve the
computation, CPU gradients were *observed* allclose at one step — and
**bf16-mixed CUDA trajectory equivalence over 40,000 steps is NOT established.**
A single-step CPU allclose probe cannot establish it, and sub-tolerance
differences compound across a long optimisation.

**Therefore deltas 2–3 are a disclosed numerical confound, not a free knob.** An
arm built from this file cannot attribute a Yaw-Aug-vs-P1 difference to the
augmentation alone; it can only report the augmentation *and* the checkpointing
change together. ⚠️ This is why, on 2026-08-15, Yixun chose to rely on the
concurrently-running arm that keeps checkpointing ON and matches P1 exactly —
that arm carries no such confound. This file is retained as the record of the
OFF variant; see yaw_aug_a6000_codex_code_r3_review.md before reusing it.

The comparison is built FORWARDS (construct the expected file from the control's
own bytes plus the pinned edits, then compare wholesale) rather than by
subtraction, and on ``read_bytes`` rather than ``read_text``: text decoding
applies universal-newline translation, under which a wholesale CRLF rewrite
would compare equal to the LF original.

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

# The control's content, pinned. Without this, the live control file and the arm
# file could drift together and every "arm == control + deltas" test below would
# still pass while both had moved away from what P1 was actually trained with
# (review finding). Recorded 2026-08-15 from the file P1's launcher consumed.
CONTROL_SHA256 = "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"

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

# The gradient-equivalent memory/time edit (see module docstring): both ViT
# conditioners drop gradient checkpointing.
GRADCKPT_ON = b'"gradient_checkpointing": true'
GRADCKPT_OFF = b'"gradient_checkpointing": false'

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
def test_the_control_config_is_the_pinned_one(control_bytes):
    """Guard against coordinated drift: the baseline itself must not move."""
    import hashlib
    actual = hashlib.sha256(control_bytes).hexdigest()
    assert actual == CONTROL_SHA256, (
        f"the P1 control config has changed (sha256 {actual}); every "
        "'arm == control + deltas' assertion below is meaningless until this "
        "pin is re-derived from the file P1 was actually trained with"
    )


def test_arm_config_is_the_control_plus_exactly_the_registered_deltas(arm_bytes, control_bytes):
    """Forward construction: control bytes + the three registered edits.

    Anything else — a stray key, a reordered block, a newline-encoding change —
    makes this fail, which is the point.
    """
    assert control_bytes.count(TRAILER_BYTES) == 1, (
        "the end-of-training boundary is not unique in the control config; the "
        "insertion point can no longer be located unambiguously"
    )
    assert control_bytes.endswith(TRAILER_BYTES)
    assert control_bytes.count(GRADCKPT_ON) == 2, (
        "expected exactly two gradient_checkpointing:true in the control"
    )

    prefix = control_bytes[: -len(TRAILER_BYTES)]
    expected = prefix + INSERTED_BYTES + TRAILER_BYTES          # delta 1
    expected = expected.replace(GRADCKPT_ON, GRADCKPT_OFF)      # deltas 2 and 3

    assert arm_bytes == expected, (
        "FLAC_AR_YAWAUG_A6000.json is not P1's config plus exactly the three "
        "registered deltas (yaw_aug block; both gradient_checkpointing flags "
        "flipped off) — the arm is no longer the registered treatment"
    )


def test_gradient_checkpointing_is_off_on_both_vit_conditioners(arm):
    """The memory/time knob is deliberate, so assert it explicitly.

    It is gradient-equivalent (see the module docstring for the evidence's
    true strength), which is why it may differ from P1 at all.
    """
    vits = [c for c in arm["model"]["conditioning"]["configs"]
            if c["type"] == "ViTCoordinates"]
    assert len(vits) == 2
    for c in vits:
        assert c["config"]["gradient_checkpointing"] is False, c["id"]


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
    # Undo the other two registered deltas too, so what remains must be the
    # control exactly. Only these three keys may be reverted here: any further
    # drift has nowhere to hide.
    for c in stripped["model"]["conditioning"]["configs"]:
        if c["type"] == "ViTCoordinates":
            c["config"]["gradient_checkpointing"] = True

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


def check_width_pin(cfg):
    """Raise unless the augmentation width matches every ViT's panorama width.

    ONE checker, called by both the pin test and its mutation guard — the guard
    must exercise the real check, not re-implement a comparison beside it
    (review finding: the earlier guard merely asserted 256 != 512 and would have
    passed even if the pin itself were deleted).
    """
    aug_w = cfg["training"]["yaw_aug"]["img_w"]
    vit_ws = {
        c["config"]["ViT"]["img_w"]
        for c in cfg["model"]["conditioning"]["configs"]
        if c["type"] == "ViTCoordinates"
    }
    if vit_ws != {aug_w}:
        raise AssertionError(
            f"yaw_aug.img_w={aug_w} but ViT widths are {vit_ws}: the augmentation "
            "would roll the panorama by the wrong number of columns and training "
            "would proceed without error"
        )


def test_pin_augmentation_width_matches_the_vit_panorama_width(arm):
    check_width_pin(arm)


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


@pytest.mark.parametrize(
    "mutate, why",
    [
        (lambda c: c["training"]["yaw_aug"].__setitem__("img_w", 256), "aug side"),
        (lambda c: c["model"]["conditioning"]["configs"][1]["config"]["ViT"]
                    .__setitem__("img_w", 256), "ViT side"),
    ],
)
def test_a_width_mismatch_is_caught_by_the_pin(arm, mutate, why):
    """Mutation guard that calls the REAL checker, from either side."""
    mutated = json.loads(json.dumps(arm))
    mutate(mutated)
    with pytest.raises(AssertionError, match="wrong number of columns"):
        check_width_pin(mutated)


def test_the_width_checker_passes_the_unmutated_arm(arm):
    """Non-vacuity: the guard above must not pass for a trivial reason."""
    check_width_pin(arm)  # must not raise


def test_fa_invariant_plus_yaw_aug_is_refused(arm):
    """The parser's own guard: this combination is untested and must fail closed."""
    training = json.loads(json.dumps(arm))["training"]
    training["cond_method"] = "fa_invariant"
    with pytest.raises(ValueError, match="fa_invariant"):
        _parse_yaw_aug_config(training)
