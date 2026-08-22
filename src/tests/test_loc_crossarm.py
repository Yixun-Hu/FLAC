"""exp_20 cross-arm machinery (`src/localization/crossarm.py`).

The admission tests build real (tiny) checkpoints rather than mocking torch:
the contract is about what a file on disk contains, so a fixture that cannot be
loaded by ``torch.load`` would prove nothing about the gate.
"""
import copy
import json
import os

import numpy as np
import pytest
import torch

from src.localization import crossarm as ca

# --------------------------------------------------------------------------- #
# checkpoint fixtures
# --------------------------------------------------------------------------- #
_CONFIG = {
    "model": {"conditioning": {"configs": [{"id": "a", "config": {}},
                                           {"id": "b", "config": {"gradient_checkpointing": True}}]}},
    "sample_size": 64,
    "training": {"use_ema": True, "cfg_dropout_prob": 0.1},
}


def _state_dict(n=3, partial_ema=False, wrong_shape=False, wrong_dtype=False):
    online = {f"diffusion.model.block{i}.weight": torch.zeros(2, 3) for i in range(n)}
    ema = {f"diffusion_ema.ema_model.block{i}.weight": torch.zeros(2, 3) for i in range(n)}
    if partial_ema:
        ema.pop(f"diffusion_ema.ema_model.block{n - 1}.weight")
    if wrong_shape:
        ema[f"diffusion_ema.ema_model.block0.weight"] = torch.zeros(4, 3)
    if wrong_dtype:
        ema[f"diffusion_ema.ema_model.block0.weight"] = torch.zeros(2, 3, dtype=torch.float64)
    other = {"diffusion.conditioner.x": torch.zeros(1),
             "diffusion_ema.initted": torch.tensor(True),
             "diffusion_ema.step": torch.tensor(40000)}
    return {**online, **ema, **other}


def _write_ckpt(path, config=None, step=40000, **state_kwargs):
    torch.save({"global_step": step, "epoch": 7,
                "state_dict": _state_dict(**state_kwargs),
                "model_config": copy.deepcopy(config if config is not None else _CONFIG)},
               str(path))
    return str(path)


def _write_config(path, config=None):
    payload = copy.deepcopy(config if config is not None else _CONFIG)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return str(path)


@pytest.fixture()
def arm_files(tmp_path):
    return (_write_ckpt(tmp_path / "arm.ckpt"), _write_config(tmp_path / "arm.json"))


# --------------------------------------------------------------------------- #
# B2 -- checkpoint admission
# --------------------------------------------------------------------------- #
def test_admission_accepts_a_clean_checkpoint(arm_files):
    ckpt, config = arm_files
    record = ca.admit_checkpoint(ckpt, config, arm="P1", expect_step=40000,
                                 check_load_integrity=False)
    assert record["admitted"] is True and record["reasons"] == []
    assert record["arm"] == "P1" and record["global_step"] == 40000
    assert len(record["sha256"]) == 64 and len(record["config_sha256"]) == 64
    assert record["ema_key_count"] == 3 and record["online_model_key_count"] == 3
    assert record["embedded_config_canonical_sha256"] == record["config_canonical_sha256"]
    assert record["load_integrity"]["checked"] is False
    assert record["cond_method"] == "vanilla"


def test_admission_refuses_a_partial_ema_family(arm_files, tmp_path):
    _ckpt, config = arm_files
    partial = _write_ckpt(tmp_path / "partial.ckpt", partial_ema=True)
    record = ca.admit_checkpoint(partial, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("mirror" in reason.lower() for reason in record["reasons"]), record["reasons"]
    assert record["ema_key_count"] is None


@pytest.mark.parametrize("kwargs,fragment", [({"wrong_shape": True}, "shape"),
                                             ({"wrong_dtype": True}, "dtype")])
def test_admission_refuses_a_drifted_ema_family(tmp_path, arm_files, kwargs, fragment):
    _ckpt, config = arm_files
    drifted = _write_ckpt(tmp_path / "drift.ckpt", **kwargs)
    record = ca.admit_checkpoint(drifted, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any(fragment in reason for reason in record["reasons"]), record["reasons"]


@pytest.mark.parametrize("step", [39999, 40001, 0])
def test_admission_refuses_a_step_mismatch(tmp_path, arm_files, step):
    _ckpt, config = arm_files
    wrong = _write_ckpt(tmp_path / f"step{step}.ckpt", step=step)
    record = ca.admit_checkpoint(wrong, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("global_step" in reason for reason in record["reasons"])


def test_admission_refuses_a_step_that_is_not_a_plain_int(tmp_path, arm_files):
    """40000.0 and True both equal 40000 under int(); neither IS the endpoint."""
    _ckpt, config = arm_files
    for value in (40000.0, True):
        path = _write_ckpt(tmp_path / f"step_{type(value).__name__}.ckpt", step=value)
        record = ca.admit_checkpoint(path, config, arm="P1", check_load_integrity=False)
        assert record["admitted"] is False
        assert any("plain int" in reason for reason in record["reasons"]), record["reasons"]


def test_admission_refuses_a_config_the_checkpoint_was_not_trained_with(tmp_path, arm_files):
    ckpt, _config = arm_files
    other = copy.deepcopy(_CONFIG)
    other["training"]["cfg_dropout_prob"] = 0.2
    record = ca.admit_checkpoint(ckpt, _write_config(tmp_path / "other.json", other),
                                 arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("canonical" in reason for reason in record["reasons"]), record["reasons"]


def test_admission_is_type_sensitive_about_the_config(tmp_path, arm_files):
    """True == 1 in Python; the canonical bytes are `true` and `1`."""
    ckpt, _config = arm_files
    coerced = copy.deepcopy(_CONFIG)
    coerced["training"]["use_ema"] = 1
    record = ca.admit_checkpoint(ckpt, _write_config(tmp_path / "coerced.json", coerced),
                                 arm="P1", check_load_integrity=False)
    assert record["admitted"] is False


# --------------------------------------------------------------------------- #
# B2 -- the arm IDENTITY embedded in the checkpoint
# --------------------------------------------------------------------------- #
def _fa_config():
    config = copy.deepcopy(_CONFIG)
    config["training"]["cond_method"] = "fa_invariant"
    config["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
    return config


def _yaw_config():
    config = copy.deepcopy(_CONFIG)
    config["training"]["yaw_aug"] = {"enabled": True, "img_w": 512, "seed": 42}
    return config


def test_admission_reads_the_conditioning_method_out_of_the_checkpoint(tmp_path):
    """The embedded training config names the arm's conditioning method, so the
    refusal does not have to rest on the manifest alone."""
    fa = _write_ckpt(tmp_path / "bf.ckpt", config=_fa_config())
    record = ca.admit_checkpoint(fa, _write_config(tmp_path / "bf.json", _fa_config()),
                                 arm="BF", check_load_integrity=False)
    assert record["admitted"] is True
    assert record["cond_method"] == "fa_invariant"
    assert record["frame_avg_angles"] == [0.0, 90.0, 180.0, 270.0]


def test_admission_refuses_an_arm_whose_embedded_identity_is_wrong(tmp_path):
    vanilla_ckpt = _write_ckpt(tmp_path / "p1.ckpt")
    vanilla_config = _write_config(tmp_path / "p1.json")
    # a vanilla checkpoint offered as the frame-averaged arm
    record = ca.admit_checkpoint(vanilla_ckpt, vanilla_config, arm="BF",
                                 check_load_integrity=False)
    assert record["admitted"] is False
    assert any("cond_method" in reason for reason in record["reasons"]), record["reasons"]

    # the frame-averaged checkpoint offered as the vanilla arm
    fa_ckpt = _write_ckpt(tmp_path / "bf.ckpt", config=_fa_config())
    fa_config = _write_config(tmp_path / "bf.json", _fa_config())
    record = ca.admit_checkpoint(fa_ckpt, fa_config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("cond_method" in reason for reason in record["reasons"])

    # the yaw arm must carry its augmentation block, and the others must not
    yaw_ckpt = _write_ckpt(tmp_path / "yaw.ckpt", config=_yaw_config())
    yaw_config = _write_config(tmp_path / "yaw.json", _yaw_config())
    assert ca.admit_checkpoint(yaw_ckpt, yaw_config, arm="YAW",
                               check_load_integrity=False)["admitted"] is True
    record = ca.admit_checkpoint(yaw_ckpt, yaw_config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("yaw_aug" in reason for reason in record["reasons"])
    record = ca.admit_checkpoint(vanilla_ckpt, vanilla_config, arm="YAW",
                                 check_load_integrity=False)
    assert record["admitted"] is False
    assert any("yaw_aug" in reason for reason in record["reasons"])


def test_admission_record_is_json_serialisable_and_names_its_inputs(arm_files):
    ckpt, config = arm_files
    record = ca.admit_checkpoint(ckpt, config, arm="P1", check_load_integrity=False)
    text = json.dumps(record, sort_keys=True)
    assert ckpt in text and config in text
    assert record["expect_step"] == 40000
    assert record["created_utc"].endswith("+00:00")


def test_admission_primitives_agree_with_the_exp15_kit(arm_files):
    """The ported semantics must BE the exp_15 semantics, not merely resemble
    them: both implementations are run over the same fixture."""
    import importlib.util
    import pathlib

    kit = (pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" /
           "exp_15_yaw_aug_claude" / "yaw_aug_record_control.py")
    if not kit.is_file():
        pytest.skip("exp_15 kit not present")
    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", kit)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    ckpt, _config = arm_files
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert ca.canonical_bytes(_CONFIG) == reference.canonical_bytes(_CONFIG)
    assert ca.canonical_sha256(_CONFIG) == reference.canonical_sha256(_CONFIG)
    assert ca.summarize_ema(payload["state_dict"]) == reference.summarize_ema(
        payload["state_dict"])
    for bad in ({1: "int key"}, {"x": float("nan")}, {"x": {1, 2}}):
        with pytest.raises(ValueError):
            ca.canonical_bytes(bad)


def test_admission_refuses_a_checkpoint_that_moved_while_it_was_read(tmp_path, monkeypatch):
    ckpt = _write_ckpt(tmp_path / "moving.ckpt")
    config = _write_config(tmp_path / "moving.json")
    original = ca.safe_load_checkpoint

    def replace_then_load(path):
        payload = original(path)
        _write_ckpt(tmp_path / "moving.ckpt", step=1)      # the file changes mid-read
        return payload

    monkeypatch.setattr(ca, "safe_load_checkpoint", replace_then_load)
    record = ca.admit_checkpoint(ckpt, config, arm="P1", check_load_integrity=False)
    assert record["admitted"] is False
    assert any("changed" in r or "replaced" in r for r in record["reasons"]), record["reasons"]
