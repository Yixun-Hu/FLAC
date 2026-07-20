"""exp-09 C1 forward-counter probe + smoke-log verifier tests
(integrative-review finding 2, items 3 & 4).

CPU-only, offline, tiny-scene machinery. Covers:

* counter K-parametrization: K=3 => 4, K=8 => 9 (fabricated batch), and that fa_invariant
  matches vanilla (no extra frame-average pass);
* throughput parse floor: it/s AND s/it units, one-sided >= 0.0395 steps/s;
* finite-loss parse: a nan loss / a loss-less log both fail.
"""
import os
import sys
from pathlib import Path

# CPU-only + offline BEFORE torch initialises (mandate).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

_EXP09_DIR = Path(__file__).resolve().parents[1]
_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_EXP09_DIR), str(_WORKTREE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import forward_counter_probe as fcp  # noqa: E402
from cylindrical_dinov3 import CylindricalDINOv3ViTConfig, CylindricalDINOv3ViTModel  # noqa: E402
from src.models.conditioners import (  # noqa: E402
    create_multi_conditioner_from_conditioning_config,
)


# ------------------------------------------------------------------------------------- #
# tiny conditioner fixture (mirrors the Stage-B integration tiny-scene machinery)
# ------------------------------------------------------------------------------------- #
def _save_tiny_cyl(dirpath) -> str:
    cfg = CylindricalDINOv3ViTConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, num_register_tokens=4,
        attention_dropout=0.0, drop_path_rate=0.0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    CylindricalDINOv3ViTModel(cfg).save_pretrained(dirpath)
    return str(dirpath)


def _tiny_conditioning(vit_path):
    block = {"hf_model_name_or_path": vit_path, "ch_dim": 3, "freeze": False,
             "from_scratch": False, "img_h": 256, "img_w": 512,
             "implementation": "cylindrical_dinov3", "gauge": "cylindrical_xyz"}
    return {
        "configs": [
            {"id": "source", "type": "dist_embedder",
             "config": {"num_freqs": 20, "max_freq": 10, "ch_dim": 1, "include_in": True}},
            {"id": "source_vit", "type": "ViTCoordinates",
             "config": {"ViT": copy.deepcopy(block), "max_value": 1, "gradient_checkpointing": False}},
            {"id": "context_poses_vit", "type": "ViTCoordinates",
             "config": {"ViT": copy.deepcopy(block), "max_value": 1, "gradient_checkpointing": False}},
            {"id": "context_poses", "type": "dist_embedder",
             "config": {"num_freqs": 20, "max_freq": 10, "ch_dim": 1, "include_in": True}},
            {"id": "context_audio", "type": "rir",
             "config": {"in_channels": 1, "n_fft": 124, "win_length": 31,
                        "hop_length": 62, "project_out": True}},
        ],
        "cond_dim": 256,
    }


@pytest.fixture(scope="module")
def tiny_mc(tmp_path_factory):
    vit = _save_tiny_cyl(tmp_path_factory.mktemp("tiny_cyl_fcp"))
    return create_multi_conditioner_from_conditioning_config(_tiny_conditioning(vit))


# ------------------------------------------------------------------------------------- #
# 1. counter K-parametrization: K=3 => 4, K=8 => 9  (fabricated batch)
# ------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,expected", [(3, 4), (8, 9)])
def test_forward_count_k_parametrization(tiny_mc, k, expected):
    md = fcp.fabricate_scene_batch(2, H=32, W=128, k=k, seed=0)
    rec = fcp.probe_counts(tiny_mc, md, device="cpu", angles=(0.0,))
    assert rec["K"] == k
    assert rec["expected_backbone_calls_per_batch"] == expected
    assert rec["n_vanilla"] == expected, rec
    assert rec["n_fa_invariant"] == expected, rec
    assert rec["no_extra_frame_passes"] is True
    assert rec["pass"] is True


def test_k8_is_nine_the_real_data_criterion(tiny_mc):
    """The C1 records pin K=8 => NINE backbone calls per batch (1 source + 8 context)."""
    md = fcp.fabricate_scene_batch(2, k=8, seed=1)
    rec = fcp.probe_counts(tiny_mc, md, device="cpu")
    assert rec["n_fa_invariant"] == 9 == rec["n_vanilla"]


def test_extra_frame_angle_would_break_the_count(tiny_mc):
    """Control that the counter has teeth: a SECOND frame angle re-runs the ViT path, so the
    fa count exceeds the vanilla 1+K — probe_counts must then NOT pass."""
    md = fcp.fabricate_scene_batch(2, k=3, seed=0)
    rec = fcp.probe_counts(tiny_mc, md, device="cpu", angles=(0.0, 90.0))
    assert rec["n_fa_invariant"] > rec["n_vanilla"]
    assert rec["no_extra_frame_passes"] is False
    assert rec["pass"] is False


# ------------------------------------------------------------------------------------- #
# 2. throughput parse floor (it/s AND s/it; one-sided >= 0.0395)
# ------------------------------------------------------------------------------------- #
def test_throughput_parses_it_per_s():
    assert fcp.parse_throughput_steps_per_s("...,  1.92it/s, ...") == [1.92]


def test_throughput_parses_s_per_it_as_reciprocal():
    # a ~12.66 s/step run prints s/it: rate = 1/12.66 ~ 0.079 steps/s
    rates = fcp.parse_throughput_steps_per_s("... 55:53,  12.66s/it, ...")
    assert len(rates) == 1
    assert abs(rates[0] - 1.0 / 12.66) < 1e-9


def test_best_throughput_is_the_max():
    log = "3.00s/it ... 0.50it/s ... 10.0s/it"  # rates: 0.333, 0.5, 0.1 -> best 0.5
    assert abs(fcp.best_throughput_steps_per_s(log) - 0.5) < 1e-9


@pytest.mark.parametrize("log,ok", [
    ("12.66s/it train/loss=0.4", True),    # 0.079 >= 0.0395
    ("0.08it/s train/loss=0.4", True),     # 0.08  >= 0.0395
    ("30.0s/it train/loss=0.4", False),    # 0.033 <  0.0395
    ("0.02it/s train/loss=0.4", False),    # 0.02  <  0.0395
])
def test_throughput_floor_is_one_sided(log, ok):
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert rec["throughput_ok"] is ok


def test_absent_throughput_fails_closed():
    rec = fcp.verify_log("train/loss=0.4 (no progress bar here)", min_steps_per_s=0.0395)
    assert rec["best_steps_per_s"] is None
    assert rec["throughput_ok"] is False
    assert rec["pass"] is False


# ------------------------------------------------------------------------------------- #
# 3. finite-loss parse
# ------------------------------------------------------------------------------------- #
def test_finite_loss_accepts_a_real_loss_line():
    rec = fcp.parse_finite_losses("Epoch 0: 1.92it/s, v_num=0, train/loss=0.367, train/mse_loss=0.367")
    assert rec["n_loss_samples"] >= 1
    assert rec["all_finite"] is True


def test_nan_loss_is_rejected():
    rec = fcp.parse_finite_losses("train/loss=nan")
    assert rec["n_non_finite"] == 1
    assert rec["all_finite"] is False


def test_loss_less_log_is_not_ok():
    rec = fcp.parse_finite_losses("no loss logged at all, 1.92it/s")
    assert rec["n_loss_samples"] == 0
    assert rec["all_finite"] is False


def test_verify_log_all_pass():
    log = "Epoch 0:  1.00it/s, train/loss=0.42"   # 1.0 >= 0.0395 and finite loss
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert rec["pass"] is True


def test_verify_log_fails_on_nan_even_if_fast():
    log = "Epoch 0:  5.00it/s, train/loss=inf"
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert rec["throughput_ok"] is True
    assert rec["finite_loss_ok"] is False
    assert rec["pass"] is False
