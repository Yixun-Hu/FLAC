from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("validate_curve_cell.py")
SPEC = importlib.util.spec_from_file_location("validate_curve_cell", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(arm: str = "FA", step: int = 2500) -> dict:
    method = "fa_invariant" if arm == "FA" else "vanilla"
    return {
        "metrics": {key: 1.0 for key in MODULE.EXPECTED_METRICS},
        "ckpt_path": f"/checkpoints/epoch=0-step={step}.ckpt",
        "rotate_deg": 0.0,
        "cond_method": method,
        "frame_avg_angles": [0.0, 90.0, 180.0, 270.0] if arm == "FA" else None,
        "cond_autocast": "bf16",
        "orbit_execution": "batched" if arm == "FA" else "n/a",
        "frame_avg_fwd_cap": 64 if arm == "FA" else None,
        "source_sha": MODULE.EXPECTED_SOURCE,
        "batch_size": 64,
        "n_samples": 6337,
        "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
        "seed": 42,
        "cfg_scale": 1.0,
        "steps": 1,
        "eval_name": f"curve0_{arm}_S{step}_K1_s42",
        "weights_source": "ema",
    }


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("arm", ["FA", "VAN"])
def test_accepts_exact_protocol(tmp_path: Path, arm: str) -> None:
    assert MODULE.validate(write(tmp_path, record(arm)), arm, 2500)["seed"] == 42


@pytest.mark.parametrize(
    ("field", "value"),
    [("rotate_deg", 90.0), ("seed", 43), ("weights_source", "online"), ("n_samples", 100)],
)
def test_rejects_protocol_drift(tmp_path: Path, field: str, value: object) -> None:
    data = record()
    data[field] = value
    with pytest.raises(ValueError):
        MODULE.validate(write(tmp_path, data), "FA", 2500)


def test_rejects_wrong_fa_orbit(tmp_path: Path) -> None:
    data = record()
    data["frame_avg_angles"] = [0.0, 180.0]
    with pytest.raises(ValueError):
        MODULE.validate(write(tmp_path, data), "FA", 2500)
