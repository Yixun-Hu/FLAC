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
# 2. SUSTAINED throughput gate (integrative-review r2 blocker 2)
# ------------------------------------------------------------------------------------- #
def test_max_tick_parsers_still_work_but_are_descriptive_only():
    assert fcp.parse_throughput_steps_per_s("...,  1.92it/s, ...") == [1.92]
    rates = fcp.parse_throughput_steps_per_s("... 55:53,  12.66s/it, ...")  # s/it -> 1/x
    assert abs(rates[0] - 1.0 / 12.66) < 1e-9
    assert abs(fcp.best_throughput_steps_per_s("3.00s/it 0.50it/s 10.0s/it") - 0.5) < 1e-9


@pytest.mark.parametrize("hms,secs", [("2:46:40", 10000), ("55:53", 3353), ("00:50", 50), ("50", 50)])
def test_parse_hms(hms, secs):
    assert fcp._parse_hms(hms) == secs


def test_parse_sustained_from_final_progress_line():
    log = ("Epoch 0:   1%| 1/100 [00:25<41:15, 0.040it/s]\n"
           "Epoch 0: 100%|##| 100/100 [2:46:40<00:00, 100.00s/it]")
    steps, elapsed = fcp.parse_sustained_from_log(log)
    assert steps == 100 and elapsed == 10000       # last line: 100 steps in 2:46:40


def _reviewer_scenario_log():
    """The reviewer's EXACT scenario: one fast tick (0.040it/s) + many 100s/it ticks. The final
    cumulative is 100 steps over 2:46:40 (=10000 s) => sustained 0.01 steps/s (< 0.0395)."""
    lines = ["Epoch 0:   1%| 1/100 [00:25<41:15,  0.040it/s, train/loss=0.40]"]
    lines += [f"Epoch 0: {i:3d}%| {i}/100 [00:00<00:00, 100.00s/it, train/loss=0.40]"
              for i in range(2, 100)]
    lines += ["Epoch 0: 100%|##| 100/100 [2:46:40<00:00, 100.00s/it, train/loss=0.40]"]
    return "\n".join(lines)


def test_reviewer_scenario_one_fast_tick_many_slow_FAILS():
    """r2 blocker 2 core: a single fast tick must NOT pass a slow smoke. Sustained 0.01 < 0.0395."""
    rec = fcp.verify_log(_reviewer_scenario_log(), min_steps_per_s=0.0395)
    assert abs(rec["sustained_steps_per_s"] - 0.01) < 1e-9
    assert rec["throughput_ok"] is False
    assert rec["pass"] is False
    # the max tick (0.040) is retained DESCRIPTIVELY and would have false-passed the old max gate
    assert rec["max_observed_steps_per_s"] >= 0.0395
    assert rec["sustained_source"] == "log_final_cumulative"


def test_uniformly_fast_log_passes():
    log = "Epoch 0: 100%|##| 100/100 [00:50<00:00, 2.00it/s, train/loss=0.42]"  # 100/50 = 2.0/s
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert abs(rec["sustained_steps_per_s"] - 2.0) < 1e-9
    assert rec["pass"] is True


def _log_reaching(n: int) -> str:
    """A minimal smoke log whose OBSERVED furthest step is n (a final ``n/3000`` progress line)."""
    return f"Epoch 0:   3%| {n}/3000 [42:11<20:00:00, 100.00s/it, train/loss=0.40]"


@pytest.mark.parametrize("steps,wall,ok", [(100, 10000, False), (100, 1000, True), (100, 2532, False)])
def test_sustained_from_authoritative_wall_clock(steps, wall, ok):
    """c1_smoke.sh passes the achieved step count + a MONOTONIC delta -> the authoritative rate."""
    rec = fcp.verify_log(_log_reaching(steps), min_steps_per_s=0.0395,
                         sustained_steps=steps, sustained_wall_s=wall)
    assert rec["throughput_ok"] is ok
    assert rec["sustained_source"] == "wall_clock_monotonic_ceiled"


def test_wall_clock_params_take_precedence_over_log_parse():
    """Even a log whose final cumulative looks fast is overridden by the authoritative wall/steps."""
    fast_log = "Epoch 0: 100%|##| 100/100 [00:10<00:00, 10.0it/s, train/loss=0.4]"
    rec = fcp.verify_log(fast_log, min_steps_per_s=0.0395, sustained_steps=100, sustained_wall_s=10000)
    assert abs(rec["sustained_steps_per_s"] - 0.01) < 1e-9
    assert rec["throughput_ok"] is False


# --- r3 blocker: whole-second truncation false-pass; use a CEIL'd fractional monotonic delta ----
def test_boundary_fractional_2531_9_is_ceiled_and_FAILS():
    """The reviewer's boundary: actual 2531.9 s / 100 steps. The verifier CEILs 2531.9 -> 2532,
    so sustained = 100/2532 = 0.0394944 < 0.0395 -> FAIL. (The old integer-truncation shell path
    reported 2531 -> 100/2531 = 0.039510 -> false-pass.)"""
    rec = fcp.verify_log(_log_reaching(100), min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=2531.9)
    assert rec["sustained_wall_s_ceiled"] == 2532
    assert abs(rec["sustained_steps_per_s"] - 100.0 / 2532) < 1e-12
    assert rec["throughput_ok"] is False
    assert rec["pass"] is False


def test_ceil_is_conservative_upward_even_below_half():
    """2531.4 s would PASS on the raw fractional rate (100/2531.4 = 0.0395035 >= 0.0395) but the
    conservative CEIL to 2532 makes it FAIL — elapsed rounds UP / rate rounds DOWN (fail-closed)."""
    assert 100.0 / 2531.4 >= 0.0395           # raw fractional would pass
    rec = fcp.verify_log(_log_reaching(100), min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=2531.4)
    assert rec["sustained_wall_s_ceiled"] == 2532
    assert rec["throughput_ok"] is False


def test_comfortably_fast_fractional_passes():
    rec = fcp.verify_log(_log_reaching(100), min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=999.4)  # ceil 1000 -> 0.1
    assert rec["sustained_wall_s_ceiled"] == 1000
    assert rec["pass"] is True


# --- r3 LOW: declared-vs-OBSERVED step reconciliation --------------------------------------- #
def test_parse_observed_steps_max_over_progress_and_step_eq():
    # progress numerators require the real tqdm bracket [elapsed<...]; MAX over all is observed
    assert fcp.parse_observed_steps(
        "Epoch 0: 1/100 [00:01<00:02, 1it/s]\nEpoch 0: 87/3000 [30:00<10:00, 0.05it/s]") == 87
    assert fcp.parse_observed_steps("saved epoch=0-step=100.ckpt") == 100
    assert fcp.parse_observed_steps("no steps here") is None
    # a declared --max-steps flag echo must NOT be mistaken for an observed step
    assert fcp.parse_observed_steps("--max-steps 100 --max-steps=100") is None


def test_observed_equals_declared_passes():
    rec = fcp.verify_log(_log_reaching(100), min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=500.0)
    assert rec["observed_steps"] == 100 and rec["declared_steps"] == 100
    assert rec["steps_ok"] is True and rec["pass"] is True


def test_observed_short_of_declared_FAILS_with_mismatch_message():
    """r3 LOW / KeyboardInterrupt swallow: a run that reached step 87 with --max-steps 100 must
    FAIL, never divide by the declared 100 blind."""
    rec = fcp.verify_log(_log_reaching(87), min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=500.0)
    assert rec["observed_steps"] == 87
    assert rec["steps_ok"] is False
    assert rec["pass"] is False
    assert "observed 87" in rec["step_mismatch"] and "declared 100" in rec["step_mismatch"]
    assert rec["sustained_steps_per_s"] is None  # never computed on a mismatch


def test_no_observed_step_with_declared_fails_closed():
    rec = fcp.verify_log("train/loss=0.4 (no progress line at all)", min_steps_per_s=0.0395,
                         sustained_steps=100, sustained_wall_s=500.0)
    assert rec["observed_steps"] is None
    assert rec["steps_ok"] is False and rec["pass"] is False


def test_absent_sustained_fails_closed():
    # no declared steps and no progress bar -> no step check, but no sustained rate -> fail
    rec = fcp.verify_log("train/loss=0.4 (nothing to time)", min_steps_per_s=0.0395)
    assert rec["sustained_steps_per_s"] is None
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
    log = "Epoch 0: 100%|##| 100/100 [00:50<00:00, 2.00it/s, train/loss=0.42]"  # sustained 2.0
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert rec["pass"] is True


def test_verify_log_fails_on_nan_even_if_fast():
    log = "Epoch 0: 100%|##| 100/100 [00:20<00:00, 5.00it/s, train/loss=inf]"  # fast but nan loss
    rec = fcp.verify_log(log, min_steps_per_s=0.0395)
    assert rec["throughput_ok"] is True     # sustained 100/20 = 5.0 >= floor
    assert rec["finite_loss_ok"] is False
    assert rec["pass"] is False
