"""Tests for ``cond_method`` dispatch in ``DiffusionCondTrainingWrapper``
(exp_03 TDD cycle 4, plan §2c/§2d/§3).

RED first: the wrapper does not yet accept a ``cond_method`` and its step methods
call ``self.diffusion.conditioner`` directly, so these fail with ``AttributeError``
/ ``Failed: DID NOT RAISE`` until commit #9. They pin the contract: an unknown
``cond_method`` raises ``ValueError`` at CONSTRUCTION (fail fast; the
``_compute_conditioning`` raise is only the backstop); ``vanilla`` / ``fa_invariant``
both construct with ``frame_avg_angles`` defaulting to ``DEFAULT_FRAME_ANGLES`` but
honoring ``training.frame_avg_angles``; ``fa_invariant`` routes conditioning through
``src.training.diffusion.invariant_conditioning`` in training_step, validation_step
AND test_step (the all-three-sites requirement the repo's ``flow_source`` precedent
demands -- a pre-revert attempt missed two sites); ``vanilla`` never calls it.

The model is a tiny CPU-only ``diffusion_cond`` built through the *real* factories
with no pretransform. The dispatch tests neutralize the DiT forward, the sampler and
the metric callback -- this checks *which* conditioning function each step calls, not
transformer maths -- so the deliberately shrunken (not forward-consistent) DiT config
is used verbatim.
"""
import types

import pytest
import torch

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config
import src.training.diffusion as tdiff


# --------------------------------------------------------------------------- #
# tiny diffusion_cond config (no pretransform, CPU only)
# --------------------------------------------------------------------------- #
def _base_config():
    return {
        "model_type": "diffusion_cond",
        "sample_size": 64,
        "sample_rate": 22050,
        "audio_channels": 1,
        "model": {
            "conditioning": {
                "configs": [
                    {"id": "source", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                    {"id": "context_poses", "type": "dist_embedder",
                     "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                ],
                "cond_dim": 32,
            },
            "diffusion": {
                "cross_attention_cond_ids": ["context_poses"],
                "global_cond_ids": ["source"],
                "type": "dit",
                "diffusion_objective": "rectified_flow",
                "config": {
                    "io_channels": 4, "embed_dim": 32, "depth": 1, "num_heads": 2,
                    "cond_token_dim": 32, "global_cond_dim": 64,
                    "transformer_type": "continuous_transformer",
                    "global_cond_type": "adaLN",
                },
            },
            "io_channels": 4,
        },
        "training": {
            "timestep_sampler": "uniform",
            "cfg_dropout_prob": 0.0,
            "use_ema": False,
            "optimizer_configs": {
                "diffusion": {"optimizer": {"type": "AdamW",
                    "config": {"lr": 5e-6, "betas": [0.9, 0.999], "weight_decay": 1e-3}}}
            },
        },
    }


def _build_wrapper(**training_overrides):
    cfg = _base_config()
    cfg["training"].update(training_overrides)
    model = create_model_from_config(cfg)
    return create_training_wrapper_from_config(cfg, model)


def _make_md(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "context_poses": torch.randn(2, 3, generator=g),
        "padding_mask": torch.ones(64),          # training_step reads this
        "scene": f"s{seed}",                       # test_step reads this
        "depth": torch.randn(3, 8, 16, generator=g),  # test_step depth path
    }


def _batch(n=2):
    return torch.randn(n, 1, 64), [_make_md(s) for s in range(n)]


def _neutralize_for_steps(wrapper, monkeypatch):
    """Stub everything the step methods touch *after* conditioning, so a dispatch
    test never depends on the DiT maths, the sampler, the metric callback, a
    Lightning Trainer or the logger. Instance-level stubs live on the fresh
    per-test wrapper (no leakage); the module-global sampler goes through
    monkeypatch (auto-undone)."""
    # DiT forward (training_step + validation_step) -> zeros shaped like input.
    wrapper.diffusion.forward = lambda x, t, **kw: torch.zeros_like(x)
    # training_step: logging + optimizer-lr readout.
    wrapper.log_dict = lambda *a, **k: None
    wrapper.trainer = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 5e-6}])]
    )
    # test_step: attrs normally set by set_test_config (which builds a heavy
    # AcousticMetricsCallback) + a no-op sampler and metric sink.
    wrapper.samples = 64
    wrapper.steps = 1
    wrapper.cfg_scale = 1.0
    wrapper.store_predictions = False
    wrapper.preds = []
    wrapper.metric_callback = types.SimpleNamespace(update_metrics=lambda *a, **k: None)
    monkeypatch.setattr(tdiff, "sample_discrete_euler", lambda model, x, *a, **k: x)


class _Spy:
    """Stand-in for ``invariant_conditioning``: counts calls and delegates to the
    real conditioner so the surrounding step code still receives valid
    conditioning."""

    def __init__(self):
        self.calls = 0

    def __call__(self, conditioner, metadata, device, *args, **kwargs):
        self.calls += 1
        return conditioner(metadata, device)


# 1. unknown cond_method -> ValueError at construction (fail fast)
@pytest.mark.parametrize("bad", ["canon", "fa-invariant"])
def test_unknown_cond_method_raises(bad):
    with pytest.raises(ValueError):
        _build_wrapper(cond_method=bad)


# 2. known cond_methods construct; frame_avg_angles default + override
@pytest.mark.parametrize("method", ["vanilla", "fa_invariant"])
def test_known_cond_methods_construct(method):
    wrapper = _build_wrapper(cond_method=method)
    assert wrapper.cond_method == method
    assert tuple(wrapper.frame_avg_angles) == tuple(DEFAULT_FRAME_ANGLES)


def test_frame_avg_angles_override():
    wrapper = _build_wrapper(cond_method="fa_invariant", frame_avg_angles=[0.0, 180.0])
    assert tuple(wrapper.frame_avg_angles) == (0.0, 180.0)


def test_default_cond_method_is_vanilla():
    wrapper = _build_wrapper()
    assert wrapper.cond_method == "vanilla"


# 3. fa_invariant dispatches through invariant_conditioning in ALL three steps
def test_dispatch_all_three_sites(monkeypatch):
    wrapper = _build_wrapper(cond_method="fa_invariant")
    _neutralize_for_steps(wrapper, monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(tdiff, "invariant_conditioning", spy)

    reals, metadata = _batch(2)

    before = spy.calls
    wrapper.training_step((reals, metadata), 0)
    assert spy.calls == before + 1, "training_step did not dispatch through invariant_conditioning"

    before = spy.calls
    wrapper.validation_step((reals, metadata), 0)
    assert spy.calls == before + 1, "validation_step did not dispatch through invariant_conditioning"

    before = spy.calls
    wrapper.test_step((reals, metadata), 0)
    assert spy.calls == before + 1, "test_step did not dispatch through invariant_conditioning"

    assert spy.calls >= 3


# 4. vanilla does NOT symmetrize (invariant_conditioning untouched)
def test_vanilla_no_symmetrization(monkeypatch):
    wrapper = _build_wrapper(cond_method="vanilla")
    _neutralize_for_steps(wrapper, monkeypatch)
    spy = _Spy()
    monkeypatch.setattr(tdiff, "invariant_conditioning", spy)

    reals, metadata = _batch(2)
    wrapper.training_step((reals, metadata), 0)
    assert spy.calls == 0, "vanilla cond_method must not call invariant_conditioning"
