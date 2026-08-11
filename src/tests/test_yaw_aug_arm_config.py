"""exp_15 round 2 — the YAWAUG arm config is the control's config plus one block.

Plan §3.3: ``FLAC_AR_YAWAUG.json`` must be a byte-copy of exp_11's
``FLAC_AR_VANCKPT.json`` (sha ``733ca52b…``, the registry-pinned config VANL was
trained with) plus **exactly one** addition — ``training.yaw_aug``. exp_15's
whole claim is a single-delta contrast against a historical control, so any
second difference between the two configs, however innocuous it looks, silently
turns the experiment into a two-factor comparison that no downstream statistic
can untangle. Hence a byte-level assertion rather than a semantic one: the arm
file with the inserted text removed must equal the control file exactly.

exp_11's file is read here and never written.
"""
import json
from pathlib import Path

import pytest

import src.training.diffusion as tdiff
from src.training.factory import _parse_yaw_aug_config, create_training_wrapper_from_config


_REPO = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = _REPO / "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json"
ARM_CONFIG = _REPO / "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"

# The literal insertion, comma included: appended as the last key of the
# "training" block, so every other byte of the control config is untouched.
# BYTES, not text: read_text() applies universal-newline decoding, under which a
# wholesale CRLF drift would compare equal to the LF original.
INSERTED_BYTES = (
    b',\n'
    b'        "yaw_aug": {\n'
    b'            "enabled": true,\n'
    b'            "img_w": 512,\n'
    b'            "seed": 42\n'
    b'        }'
)

# The end of the control file: the close of "training" and of the root object.
# The insertion point is immediately before it.
TRAILER_BYTES = b'\n    }\n}'

EXPECTED_BLOCK = {"enabled": True, "img_w": 512, "seed": 42}


@pytest.fixture(scope="module")
def arm_bytes():
    assert ARM_CONFIG.is_file(), f"arm config not found: {ARM_CONFIG}"
    return ARM_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def control_bytes():
    return CONTROL_CONFIG.read_bytes()


@pytest.fixture(scope="module")
def arm_text(arm_bytes):
    return arm_bytes.decode()


@pytest.fixture(scope="module")
def control_text(control_bytes):
    return control_bytes.decode()


def test_arm_config_is_the_control_plus_exactly_one_insertion(arm_bytes, control_bytes):
    """Byte-level single-delta proof, built forwards rather than by subtraction.

    The expected arm file is CONSTRUCTED from the control's own bytes plus the
    pinned insertion at a uniquely located boundary, then compared for exact
    equality — so nothing (not even a newline-encoding change) can differ
    anywhere else.
    """
    assert control_bytes.count(TRAILER_BYTES) == 1, (
        "the end-of-training boundary is not unique in the control config; "
        "the insertion point can no longer be located unambiguously"
    )
    assert control_bytes.endswith(TRAILER_BYTES)

    prefix = control_bytes[: -len(TRAILER_BYTES)]
    expected_arm_bytes = prefix + INSERTED_BYTES + TRAILER_BYTES

    assert arm_bytes == expected_arm_bytes, (
        "FLAC_AR_YAWAUG.json is not byte-for-byte the control config plus the "
        "yaw_aug block — the arm would no longer be a single-delta treatment"
    )


def test_byte_comparison_would_catch_newline_drift(control_bytes):
    """Negative control for the test above (review finding 4).

    The previous version decoded both files with universal newlines, so a
    wholesale CRLF rewrite compared EQUAL to the LF original. Proven here in
    memory — no file is touched — that the byte comparison rejects it.
    """
    prefix = control_bytes[: -len(TRAILER_BYTES)]
    genuine = prefix + INSERTED_BYTES + TRAILER_BYTES
    crlf_drifted = genuine.replace(b"\n", b"\r\n")

    assert crlf_drifted != genuine
    assert crlf_drifted.decode().replace(INSERTED_BYTES.decode(), "", 1) != control_bytes.decode()
    # ...whereas the OLD text-level comparison could not tell them apart:
    assert crlf_drifted.decode().splitlines() == genuine.decode().splitlines()


def test_arm_config_semantic_diff_is_only_training_yaw_aug(arm_text, control_text):
    arm, control = json.loads(arm_text), json.loads(control_text)
    assert arm["training"]["yaw_aug"] == EXPECTED_BLOCK

    stripped = json.loads(arm_text)
    stripped["training"].pop("yaw_aug")
    assert stripped == control

    # key ORDER too: same top-level order, and training gains yaw_aug at the end
    assert list(arm) == list(control)
    assert list(arm["training"]) == list(control["training"]) + ["yaw_aug"]


def test_arm_config_stays_vanilla():
    """Augmentation instead of architecture: the arm must NOT frame-average."""
    arm = json.loads(ARM_CONFIG.read_text())
    assert "cond_method" not in arm["training"], (
        "the YAWAUG arm must be vanilla-conditioned (announcement 05: the eval "
        "flag must match how the checkpoint was trained)"
    )
    assert arm["training"]["use_ema"] is True
    for conditioner in arm["model"]["conditioning"]["configs"]:
        if conditioner["type"] == "ViTCoordinates":
            assert conditioner["config"]["gradient_checkpointing"] is True


def test_arm_config_parses_into_the_wrapper_flags():
    arm = json.loads(ARM_CONFIG.read_text())
    assert _parse_yaw_aug_config(arm["training"]) == {
        "yaw_aug_enabled": True,
        "yaw_aug_img_w": 512,
        "yaw_aug_seed": 42,
    }


def test_arm_config_trips_no_factory_guard(monkeypatch):
    """The real factory must accept this config — with the wrapper stubbed, so
    the test does not build DINOv3."""
    captured = {}

    class _Stub:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tdiff, "DiffusionCondTrainingWrapper", _Stub)
    create_training_wrapper_from_config(json.loads(ARM_CONFIG.read_text()), object())

    assert captured["yaw_aug_enabled"] is True
    assert captured["yaw_aug_img_w"] == 512
    assert captured["yaw_aug_seed"] == 42
    assert captured["cond_method"] == "vanilla"


def test_control_config_is_unmodified():
    """exp_11's file is read-only for exp_15 (registry-pinned sha)."""
    import hashlib

    assert hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest() == (
        "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"
    )
