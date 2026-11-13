"""exp_43 std_aug -- ``FLAC_AR_YAWAUG_FLIP.json`` is the reference Yaw-Aug config plus ONE key.

Plan §3.5: the arm config of ``RF`` (full-circle roll + horizontal mirror flip) is exp_15's
``FLAC_AR_YAWAUG.json`` (sha ``b14caae2...``; the config both reference ``YAW`` arms trained
with) with ``training.yaw_aug = {"enabled": true, "img_w": 512, "seed": 42, "flip": true}``.
Against the ``YAW`` configuration the only difference is ``flip: true`` (plan §1), so the
file is constructed FORWARDS from the reference bytes with exactly one insertion --
``"flip": true`` as the last key of ``training.yaw_aug`` -- and compared byte for byte, then
checked semantically (type-strict single delta) and through the real factory parser. The
reference file is read, never written.
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.training.factory import _parse_yaw_aug_config

_REPO = Path(__file__).resolve().parents[2]
BASE = _REPO / "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"
ARM = _REPO / "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json"
BASE_SHA256 = "b14caae27afec2c53c38bea6e2574a48aa94bc86848ab17afae94cadf694b004"
ARM_SHA256 = "d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d"

# BYTES, not text (read_text() would hide a CRLF drift): the end of the reference file is the
# last key of training.yaw_aug, then the closes of yaw_aug, training and the root object.
ANCHOR = b'            "seed": 42\n        }\n    }\n}'
REPLACEMENT = b'            "seed": 42,\n            "flip": true\n        }\n    }\n}'


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _strict_equal(a, b):
    """``==`` with the JSON types enforced (``1 != True``, ``0 != 0.0``)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def _leaf_paths(obj, prefix=()):
    """Every (path, value) leaf of a parsed JSON object, dict keys in file order."""
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.extend(_leaf_paths(v, prefix + (k,)))
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out.extend(_leaf_paths(v, prefix + (i,)))
        return out
    return [(prefix, obj)]


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


def test_parsed_objects_differ_in_exactly_the_flip_key(base_bytes, arm_bytes):
    base, arm = json.loads(base_bytes), json.loads(arm_bytes)
    base_leaves, arm_leaves = _leaf_paths(base), _leaf_paths(arm)
    added = [(p, v) for p, v in arm_leaves if p not in {q for q, _ in base_leaves}]
    assert added == [(("training", "yaw_aug", "flip"), True)]
    assert type(arm["training"]["yaw_aug"]["flip"]) is bool
    assert list(arm["training"]["yaw_aug"]) == ["enabled", "img_w", "seed", "flip"]
    del arm["training"]["yaw_aug"]["flip"]
    assert _strict_equal(arm, base), "a second difference besides training.yaw_aug.flip"


def test_recipe_pins(arm_bytes):
    cfg = json.loads(arm_bytes)
    t = cfg["training"]
    assert t["yaw_aug"] == {"enabled": True, "img_w": 512, "seed": 42, "flip": True}
    assert "offset_step" not in t["yaw_aug"]
    assert t["use_ema"] is True
    for key in ("cond_method", "canonicalize", "are_lambda", "frame_avg_max_fwd_samples"):
        assert key not in t
    vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
    assert sorted(c["id"] for c in vits) == ["context_poses_vit", "source_vit"]
    for c in vits:
        assert c["config"]["ViT"]["img_w"] == t["yaw_aug"]["img_w"] == 512
        assert c["config"]["ViT"]["img_h"] == 256


def test_factory_parses_the_arm_file(arm_bytes):
    got = _parse_yaw_aug_config(json.loads(arm_bytes)["training"])
    assert got == {"yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42,
                   "yaw_aug_flip": True}
    assert got["yaw_aug_flip"] is True


def test_factory_parses_the_reference_file_without_a_flip_kwarg(base_bytes):
    assert _parse_yaw_aug_config(json.loads(base_bytes)["training"]) == {
        "yaw_aug_enabled": True, "yaw_aug_img_w": 512, "yaw_aug_seed": 42}
