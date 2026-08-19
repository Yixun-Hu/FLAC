"""RAF policy in the metric stack (exp_19, contract section F, TDD cycle 12).

Covers ``AcousticMetricsCallback`` and ``RT60Error`` for ``dataset_name="RAF"``,
the fail-closed FD/retrieval guard (no AGREE model was ever trained on RAF), and
regression tests pinning that the AcousticRooms and HAA paths are untouched.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.metrics.metric_callback import SUPPORTED_DATASETS, AcousticMetricsCallback
from src.metrics.modules.RT60 import RT60Error

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_MODEL_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC",
                                 "RAF", "FLAC_RAF_finetune.json")

# eval_l1_distance_multires is deliberately absent: L1_STFT_MultiRes.update calls
# torch.hann_window(n_fft).cuda() (src/metrics/modules/l1_stft_multires.py:41), a
# pre-existing CUDA-only path this contract does not touch. Its registration for
# RAF is covered separately, without an update.
_METRIC_FLAGS = dict(eval_T60=True, eval_C50=True, eval_EDT=True, eval_env=True,
                     eval_l1_distance=True)


def _decaying_rir(seed, n=10240, sr=22050, tau=0.08):
    """A synthetic RIR with a real exponential decay, so T60/EDT are measurable."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32) / sr
    return (torch.randn(n, generator=g) * torch.exp(-t / tau)).view(1, 1, n) * 0.5


def _callback(dataset_name="RAF", **kwargs):
    flags = dict(_METRIC_FLAGS)
    flags.update(kwargs)
    return AcousticMetricsCallback(dataset_name=dataset_name, device="cpu", **flags)


# --------------------------------------------------------------------------- #
# RAF support
# --------------------------------------------------------------------------- #
def test_raf_is_a_supported_dataset():
    assert "RAF" in SUPPORTED_DATASETS


def test_raf_uses_the_haa_metric_window():
    cb = _callback("RAF")
    assert cb.max_len == 9600
    assert cb.max_len_magenv == 9600


def test_raf_accumulates_per_scene_without_an_explicit_flag():
    """Two rooms of unequal size: RAF headline numbers are per-scene means."""
    cb = _callback("RAF")
    assert cb.eval_by_scene is True


def test_raf_update_compute_and_by_scene():
    cb = _callback("RAF")
    pred = torch.cat([_decaying_rir(1), _decaying_rir(2)], dim=0)
    ref = torch.cat([_decaying_rir(3), _decaying_rir(4)], dim=0)
    cb.update_metrics("test", pred, ref, scene=["EmptyRoom", "FurnishedRoom"])
    metrics = cb.compute_metrics("test")
    for key in ("T60", "Invalid T60", "C50", "EDT", "Env", "L1_STFT"):
        assert key in metrics, key
        assert np.isfinite(float(metrics[key] if not isinstance(metrics[key], tuple)
                                 else metrics[key][0]))
    assert set(metrics["by_scene"]) == {"EmptyRoom", "FurnishedRoom"}
    for scene in metrics["by_scene"].values():
        for key in ("T60", "C50", "EDT", "Env", "L1_STFT"):
            assert key in scene, key
    assert "FD" not in metrics and "R@1" not in metrics


def test_raf_registers_the_multires_metric():
    """The RAF config asks for it; only its update path is CUDA-bound."""
    cb = _callback("RAF", eval_l1_distance_multires=True)
    assert "test" in cb.l1_stft_multires
    assert "EmptyRoom" not in cb.scene_metrics["test"]   # created lazily, per scene


def test_raf_compute_resets_the_accumulators():
    cb = _callback("RAF")
    pred, ref = _decaying_rir(1), _decaying_rir(2)
    cb.update_metrics("test", pred, ref, scene=["EmptyRoom"])
    first = cb.compute_metrics("test")
    cb.update_metrics("test", pred, pred, scene=["EmptyRoom"])
    second = cb.compute_metrics("test")
    assert second["C50"] != first["C50"]      # not contaminated by the first batch
    assert abs(float(second["C50"])) < 1e-6   # pred == ref


def test_raf_rejects_a_wrong_sample_rate_or_channel_count():
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", sample_rate=44100, device="cpu")
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", audio_channels=2, device="cpu")


# --------------------------------------------------------------------------- #
# FD / retrieval guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["eval_FD", "eval_retrieval"])
def test_raf_refuses_agree_based_metrics(flag):
    """No AGREE model exists for RAF: FD/Recall are unavailable, never zero."""
    with pytest.raises(ValueError) as exc:
        AcousticMetricsCallback(dataset_name="RAF", device="cpu",
                                **{flag: True, "AGREE_ckpt": "weights/AGREE_HAA.pt"})
    assert "RAF" in str(exc.value)


@pytest.mark.parametrize("flag", ["eval_FD", "eval_retrieval"])
def test_raf_guard_fires_even_without_a_checkpoint(flag):
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="RAF", device="cpu", **{flag: True})


def test_unsupported_dataset_still_raises():
    with pytest.raises(ValueError):
        AcousticMetricsCallback(dataset_name="Matterport", device="cpu")


# --------------------------------------------------------------------------- #
# RT60 policy
# --------------------------------------------------------------------------- #
def test_rt60_raf_mirrors_the_haa_decay_window():
    """RAF RIRs are real, truncated measurements like HAA's, so RAF adopts HAA's
    T30 window (decay_db=30) and its invalid-sample masking rather than AR's T20
    on clean simulated RIRs."""
    assert RT60Error(dataset_name="RAF").decay_db == 30
    assert RT60Error(dataset_name="HAA").decay_db == 30
    assert RT60Error(dataset_name="AcousticRooms").decay_db == 20


def test_rt60_unknown_dataset_still_raises():
    with pytest.raises(NotImplementedError):
        RT60Error(dataset_name="Matterport")


def test_rt60_raf_counts_invalid_samples_like_haa():
    silent = torch.zeros(1, 1, 9600)
    raf = RT60Error(dataset_name="RAF")
    raf.update(silent, silent)
    error, invalid = raf.compute()
    assert invalid == 1.0
    assert error == 100.0     # invalid samples are charged the maximum error

    ar = RT60Error(dataset_name="AcousticRooms")
    ar.update(silent, silent)
    assert ar.compute()[1] == 0.0   # AR path records no invalids (unchanged)


# --------------------------------------------------------------------------- #
# AR / HAA regressions
# --------------------------------------------------------------------------- #
def test_acousticrooms_behaviour_is_unchanged():
    cb = _callback("AcousticRooms")
    assert cb.max_len == 8000
    assert cb.max_len_magenv == 9600
    assert cb.eval_by_scene is False


def test_acousticrooms_per_scene_flag_still_works():
    assert _callback("AcousticRooms", eval_per_scene=True).eval_by_scene is True


def test_haa_behaviour_is_unchanged():
    cb = _callback("HAA")
    assert cb.max_len == 9600
    assert cb.eval_by_scene is True


def test_haa_still_allows_agree_metrics_when_a_checkpoint_is_given():
    """The RAF guard must not leak into HAA/AR: only the missing-checkpoint rule
    applies there."""
    with pytest.raises(ValueError) as exc:
        AcousticMetricsCallback(dataset_name="HAA", device="cpu", eval_FD=True)
    assert "AGREE_ckpt" in str(exc.value)


def test_haa_dampened_scene_is_still_excluded_from_t60():
    cb = _callback("HAA")
    pred, ref = _decaying_rir(1), _decaying_rir(2)
    cb.update_metrics("test", pred, ref, scene=["dampenedBase"])  # noqa: E501
    assert len(cb.RT60["test"].t60_error) == 0


# --------------------------------------------------------------------------- #
# config integration
# --------------------------------------------------------------------------- #
def test_callback_builds_from_the_raf_model_config():
    with open(_RAF_MODEL_CONFIG) as f:
        metrics_config = json.load(f)["training"]["metrics"]
    cb = AcousticMetricsCallback(
        dataset_name=metrics_config["dataset_name"],
        device="cpu",
        eval_T60=metrics_config.get("eval_T60", False),
        eval_C50=metrics_config.get("eval_C50", False),
        eval_EDT=metrics_config.get("eval_EDT", False),
        eval_l1_distance=metrics_config.get("eval_l1_distance", False),
        eval_l1_distance_multires=metrics_config.get("eval_l1_distance_multires", False),
        eval_FD=metrics_config.get("eval_FD", False),
        eval_retrieval=metrics_config.get("eval_retrieval", False),
        eval_env=metrics_config.get("eval_env", False),
        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
    )
    assert cb.dataset_name == "RAF"
    assert cb.max_len == 9600
    assert cb.eval_FD is False and cb.eval_retrieval is False
    assert cb.AGREE_model is None
