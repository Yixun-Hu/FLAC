"""exp_15 round 2 — the control-admission recorder (plan §3.3-1, §6.4, §6.5-10).

exp_11's registry pins VANL's launch manifest, commit, config, VAE, rung and seed
— but **not** the 40,000-step checkpoint (its job ended FAILED after the save and
``final_ckpt_sha256`` was never backfilled). exp_15 compares against that
checkpoint, so it writes its own immutable admission record binding the file by
sha256 together with what is embedded inside it, and every VANL eval cell
re-validates that record before running (gate G4).

The recorder is therefore held to the standards of evidence, not convenience:
it never writes when a check fails, never overwrites an existing record, and is
strictly read-only with respect to the checkpoint and the config (exp_11 owns
both).

These tests run against a *tiny synthetic* checkpoint built in ``tmp_path``; the
real 724 MB checkpoint is touched exactly once, outside pytest.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


_REPO = Path(__file__).resolve().parents[2]
_EXP15 = _REPO / "worklog/worklog_yixun/exp_15_yaw_aug_claude"
RECORDER_PATH = _EXP15 / "yaw_aug_record_control.py"
CONTROL_CONFIG = _REPO / "worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json"

REGISTRY_CONFIG_SHA = "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"


def _load_recorder():
    assert RECORDER_PATH.is_file(), f"recorder not found: {RECORDER_PATH}"
    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", RECORDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = pytest.fixture(scope="module")(lambda: _load_recorder())


@pytest.fixture
def synthetic_ckpt(tmp_path):
    """A PL-shaped checkpoint small enough to write in a test."""
    def _make(global_step=40000, with_ema=True):
        state = {
            "diffusion.model.layer.weight": torch.zeros(2, 2),
            "diffusion.model.layer.bias": torch.zeros(2),
        }
        if with_ema:
            state["diffusion_ema.ema_model.layer.weight"] = torch.zeros(2, 2)
            state["diffusion_ema.ema_model.layer.bias"] = torch.zeros(2)
            state["diffusion_ema.initted"] = torch.tensor(True)
        path = tmp_path / f"epoch=8-step={global_step}.ckpt"
        torch.save(
            {
                "global_step": global_step,
                "epoch": 8,
                "state_dict": state,
                "optimizer_states": [{"state": {}}],
                "lr_schedulers": [{"last_epoch": global_step}],
            },
            path,
        )
        return path

    return _make


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def test_sha256_file_streams_and_matches_hashlib(rc, tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 4096)          # 1 MiB, > one chunk
    assert rc.sha256_file(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# record content
# --------------------------------------------------------------------------- #
def test_record_content(rc, synthetic_ckpt):
    ckpt = synthetic_ckpt()
    record = rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)

    ck = record["checkpoint"]
    assert ck["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert ck["bytes"] == ckpt.stat().st_size
    assert ck["global_step"] == 40000
    assert ck["epoch"] == 8
    assert ck["ema_prefix"] == "diffusion_ema.ema_model."
    assert ck["ema_key_count"] == 2
    assert ck["optimizer_states"] == 1
    assert ck["lr_schedulers"] == 1
    assert ck["state_dict_keys"] == 5

    assert record["config"]["sha256"] == REGISTRY_CONFIG_SHA

    xref = record["exp_11_cross_references"]
    assert xref["manifest_sha256"] == (
        "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1"
    )
    assert xref["commit"] == "81ddac372076ea92751ae09cbaf371df70f396e5"
    assert xref["training_seed"] == 42
    assert xref["rung"] == "8x8"
    assert xref["vae_sha256"] == (
        "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
    )
    assert xref["job"] == "3661520"

    assert record["checks"] == {
        "global_step_equals_expected": True,
        "config_sha256_matches_registry": True,
        "ema_state_present": True,
    }
    # the record must be JSON-serialisable as written
    json.dumps(record)


def test_record_is_readonly_wrt_its_inputs(rc, synthetic_ckpt, tmp_path):
    ckpt = synthetic_ckpt()
    before = (hashlib.sha256(ckpt.read_bytes()).hexdigest(),
              hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest())
    rc.write_record(
        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000), tmp_path / "rec.json"
    )
    after = (hashlib.sha256(ckpt.read_bytes()).hexdigest(),
             hashlib.sha256(CONTROL_CONFIG.read_bytes()).hexdigest())
    assert before == after


# --------------------------------------------------------------------------- #
# fail-closed behaviour
# --------------------------------------------------------------------------- #
def test_refuses_to_overwrite_an_existing_record(rc, synthetic_ckpt, tmp_path):
    out = tmp_path / "rec.json"
    record = rc.build_record(synthetic_ckpt(), CONTROL_CONFIG, expect_step=40000)
    rc.write_record(record, out)
    original = out.read_bytes()

    with pytest.raises(FileExistsError):
        rc.write_record(record, out)
    assert out.read_bytes() == original, "the existing record was modified"


def test_detects_step_mismatch(rc, synthetic_ckpt):
    ckpt = synthetic_ckpt(global_step=37500)
    with pytest.raises(ValueError, match="37500"):
        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)


def test_detects_config_sha_mismatch(rc, synthetic_ckpt, tmp_path):
    impostor = tmp_path / "not_the_control_config.json"
    impostor.write_text('{"model_type": "diffusion_cond"}')
    with pytest.raises(ValueError, match="733ca52b"):
        rc.build_record(synthetic_ckpt(), impostor, expect_step=40000)


def test_detects_missing_ema_state(rc, synthetic_ckpt):
    ckpt = synthetic_ckpt(with_ema=False)
    with pytest.raises(ValueError, match="EMA"):
        rc.build_record(ckpt, CONTROL_CONFIG, expect_step=40000)


def test_failed_validation_writes_nothing(rc, synthetic_ckpt, tmp_path):
    out = tmp_path / "rec.json"
    with pytest.raises(ValueError):
        rc.build_record(synthetic_ckpt(global_step=1), CONTROL_CONFIG, expect_step=40000)
    assert not out.exists()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_main_writes_the_record(rc, synthetic_ckpt, tmp_path, capsys):
    ckpt = synthetic_ckpt()
    out = tmp_path / "yaw_aug_control_admission.json"
    rc.main([
        "--ckpt", str(ckpt),
        "--config", str(CONTROL_CONFIG),
        "--out", str(out),
        "--expect-step", "40000",
    ])
    written = json.loads(out.read_text())
    assert written["checkpoint"]["global_step"] == 40000
    assert written["config"]["sha256"] == REGISTRY_CONFIG_SHA
    assert "sha256" in capsys.readouterr().out


def test_main_refuses_to_overwrite(rc, synthetic_ckpt, tmp_path):
    ckpt = synthetic_ckpt()
    out = tmp_path / "rec.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        rc.main([
            "--ckpt", str(ckpt),
            "--config", str(CONTROL_CONFIG),
            "--out", str(out),
            "--expect-step", "40000",
        ])
    assert out.read_text() == "{}"
