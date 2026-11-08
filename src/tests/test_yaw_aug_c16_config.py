"""exp_40 -- ``FLAC_AR_YAWAUG_C16.json`` is the reference Yaw-Aug config plus ONE key.

The reference is exp_15's ``FLAC_AR_YAWAUG.json`` (sha ``b14caae2...``): the config BOTH
reference arms trained with (exp_15 neuronic; exp_17 A6000, whose checkpoint ``ac1f2603``
embeds it). The arm file is constructed FORWARDS from the reference bytes with exactly
one insertion -- ``"offset_step": 32`` as the last key of ``training.yaw_aug`` -- and
compared byte for byte (plan §2.3), then checked semantically (type-strict single
delta) and through the real factory. The reference file is read, never written.
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.training.factory import _parse_yaw_aug_config

_REPO = Path(__file__).resolve().parents[2]
BASE = _REPO / "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"
ARM = _REPO / "worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/FLAC_AR_YAWAUG_C16.json"
BASE_SHA256 = "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004"
ARM_SHA256 = "b497dfe937debde029057d7442863a9fb6ef87dd333d868f90f5bc311326f5c9"

ANCHOR = b'            "seed": 42\n        }\n    }\n}'
REPLACEMENT = b'            "seed": 42,\n            "offset_step": 32\n        }\n    }\n}'


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _strict_equal(a, b):
    """``==`` with the JSON types enforced (``1 != True``, ``0 != 0.0``)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def base_bytes():
    return BASE.read_bytes()


@pytest.fixture(scope="module")
def arm_bytes():
    assert ARM.is_file(), f"arm config missing: {ARM}"
    return ARM.read_bytes()


def test_reference_config_is_the_pinned_file(base_bytes):
    assert _sha(base_bytes) == BASE_SHA256


def test_arm_is_the_reference_plus_exactly_one_insertion(base_bytes, arm_bytes):
    assert base_bytes.count(ANCHOR) == 1 and base_bytes.endswith(ANCHOR)
    assert arm_bytes == base_bytes[: -len(ANCHOR)] + REPLACEMENT
    assert _sha(arm_bytes) == ARM_SHA256


def test_parsed_single_delta_is_type_strict(base_bytes, arm_bytes):
    base, arm = json.loads(base_bytes), json.loads(arm_bytes)
    assert arm["training"]["yaw_aug"]["offset_step"] == 32
    assert type(arm["training"]["yaw_aug"]["offset_step"]) is int
    del arm["training"]["yaw_aug"]["offset_step"]
    assert _strict_equal(arm, base)


def test_recipe_pins(arm_bytes):
    cfg = json.loads(arm_bytes)
    t = cfg["training"]
    assert t["yaw_aug"] == {"enabled": True, "img_w": 512, "seed": 42, "offset_step": 32}
    assert t["use_ema"] is True
    for key in ("cond_method", "canonicalize", "are_lambda", "frame_avg_max_fwd_samples"):
        assert key not in t
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    assert sorted(c["id"] for c in vits) == ["context_poses_vit", "source_vit"]
    for c in vits:
        assert c["config"]["gradient_checkpointing"] is True
        assert c["config"]["ViT"]["img_w"] == t["yaw_aug"]["img_w"] == 512
        assert c["config"]["ViT"]["img_h"] == 256


def test_factory_parses_the_arm_file(arm_bytes):
    assert _parse_yaw_aug_config(json.loads(arm_bytes)["training"]) == {
        "yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
        "yaw_aug_offset_step": 32}
