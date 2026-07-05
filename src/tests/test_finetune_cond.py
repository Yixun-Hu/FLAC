"""Tests for ``finetune_cond.py`` config injection and CLI parsing (exp_03 cycle 6).

These pin the contract of the non-destructive fine-tune driver *before* it exists
(TDD RED). Two public surfaces are under test:

* ``build_finetune_training_config`` — a pure function that deep-copies a FLAC model
  config and injects the fine-tune recipe (cond_method / frame_avg_angles, use_ema off,
  constant LR by dropping the InverseLR scheduler) without mutating the input or any key
  outside the ``training`` block.
* ``build_parser`` — the argparse construction, exercised on a full arg vector; importing
  the module must have no side effects.

Constant-LR mechanism under test: the injected config *removes* the ``scheduler`` key from
``optimizer_configs['diffusion']``. ``DiffusionCondTrainingWrapper.configure_optimizers``
(src/training/diffusion.py:195) only builds a scheduler when that key is present, so its
absence yields a bare optimizer whose LR never changes — a constant schedule, and the most
decisive neutralization of the InverseLR warm-up restart.
"""
import copy
import os

import pytest
import pytorch_lightning as pl
import torch

import finetune_cond
from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config
from src.training.utils import create_optimizer_from_config
from test_cond_dispatch import _base_config  # sibling tiny diffusion_cond config (pytest prepends src/tests)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FLAC_AR_CONFIG = os.path.join(
    _REPO_ROOT, "src", "configs", "model_configs", "FLAC", "AR", "FLAC_AR.json"
)


def _load_flac_ar_config():
    """Load the real FLAC_AR.json model config as a fresh dict."""
    import json

    with open(_FLAC_AR_CONFIG) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# 1. Injection: cond_method / frame_avg_angles / use_ema / lr / constant scheduler
# --------------------------------------------------------------------------------------

def test_injects_cond_method_and_angles():
    cfg = _load_flac_ar_config()
    angles = [0.0, 90.0, 180.0, 270.0]
    out = finetune_cond.build_finetune_training_config(cfg, "fa_invariant", 5e-6, angles)
    assert out["training"]["cond_method"] == "fa_invariant"
    assert out["training"]["frame_avg_angles"] == angles


def test_use_ema_forced_false():
    cfg = _load_flac_ar_config()
    assert cfg["training"]["use_ema"] is True  # baseline sanity: input has EMA on
    out = finetune_cond.build_finetune_training_config(cfg, "vanilla", 5e-6, [0.0])
    assert out["training"]["use_ema"] is False


def test_optimizer_lr_overridden():
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(cfg, "vanilla", 5e-6, [0.0])
    lr = out["training"]["optimizer_configs"]["diffusion"]["optimizer"]["config"]["lr"]
    assert lr == 5e-6


def test_scheduler_absent_yields_constant_lr():
    """The chosen constant-LR mechanism: scheduler key removed from the diffusion opt config.

    Mirrors configure_optimizers (diffusion.py:195): with no 'scheduler' key it returns a
    bare optimizer and never steps an LR schedule. We assert the key is gone AND construct
    the optimizer exactly as the wrapper would, confirming its LR equals the requested lr
    (constant for all steps because nothing schedules it).
    """
    cfg = _load_flac_ar_config()
    assert "scheduler" in cfg["training"]["optimizer_configs"]["diffusion"]  # input HAS one
    out = finetune_cond.build_finetune_training_config(cfg, "fa_invariant", 5e-6, [0.0, 90.0])

    diff_cfg = out["training"]["optimizer_configs"]["diffusion"]
    assert "scheduler" not in diff_cfg  # configure_optimizers -> no scheduler branch

    dummy = [torch.nn.Parameter(torch.zeros(1))]
    opt = create_optimizer_from_config(diff_cfg["optimizer"], dummy)
    assert opt.param_groups[0]["lr"] == 5e-6
    # No scheduler exists, so stepping never changes the LR: constant at 5e-6 for step 0/100/5000.


# --------------------------------------------------------------------------------------
# 2. Non-mutation of the caller's config
# --------------------------------------------------------------------------------------

def test_input_config_not_mutated():
    cfg = _load_flac_ar_config()
    before = copy.deepcopy(cfg)
    finetune_cond.build_finetune_training_config(cfg, "fa_invariant", 5e-6, [0.0, 90.0, 180.0])
    assert cfg == before  # deep compare: original untouched


def test_returns_new_object():
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(cfg, "vanilla", 5e-6, [0.0])
    assert out is not cfg
    assert out["training"] is not cfg["training"]


# --------------------------------------------------------------------------------------
# 3. Only the training block is touched (dataset-side / K keys structurally untouched)
# --------------------------------------------------------------------------------------

def test_only_training_block_changes():
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(cfg, "fa_invariant", 5e-6, [0.0])
    for key in cfg:
        if key == "training":
            continue
        assert out[key] == cfg[key], f"non-training key '{key}' changed"
    # The 'model' block (architecture, conditioning topology, K-independent) is bit-identical.
    assert out["model"] == cfg["model"]
    # Dataset K lives in the dataset config, which this function never receives -> cannot touch it.
    assert "modalities" not in out and "datasets" not in out


# --------------------------------------------------------------------------------------
# 4. Validation of cond_method
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["canon", "frame_avg", "typo", "", "Vanilla"])
def test_unknown_cond_method_raises(bad):
    cfg = _load_flac_ar_config()
    with pytest.raises(ValueError):
        finetune_cond.build_finetune_training_config(cfg, bad, 5e-6, [0.0])


@pytest.mark.parametrize("good", ["vanilla", "fa_invariant"])
def test_known_cond_method_accepted(good):
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(cfg, good, 5e-6, [0.0])
    assert out["training"]["cond_method"] == good


# --------------------------------------------------------------------------------------
# 5. argparse: no import side effects; full-vector parse yields expected namespace
# --------------------------------------------------------------------------------------

def test_import_has_no_side_effects():
    # Importing the module (done at top of file) must not execute the driver. main() is
    # guarded by __main__, so the public helpers exist but nothing ran.
    assert hasattr(finetune_cond, "build_parser")
    assert hasattr(finetune_cond, "build_finetune_training_config")
    assert callable(finetune_cond.build_parser)


def test_parse_full_arg_vector():
    argv = [
        "--model-config", "m.json",
        "--dataset-config", "d.json",
        "--ckpt-path", "c.ckpt",
        "--pretransform-ckpt-path", "v.safetensors",
        "--save-dir", "/tmp/out",
        "--name", "myrun",
        "--cond-method", "fa_invariant",
        "--frame-avg-angles", "0,90,180,270",
        "--lr", "5e-6",
        "--max-steps", "10000",
        "--checkpoint-every", "2000",
        "--batch-size", "8",
        "--num-workers", "4",
        "--precision", "bf16-mixed",
        "--gradient-clip-val", "1.0",
        "--seed", "123",
        "--smoke",
    ]
    ns = finetune_cond.build_parser().parse_args(argv)
    assert ns.model_config == "m.json"
    assert ns.dataset_config == "d.json"
    assert ns.ckpt_path == "c.ckpt"
    assert ns.pretransform_ckpt_path == "v.safetensors"
    assert ns.save_dir == "/tmp/out"
    assert ns.name == "myrun"
    assert ns.cond_method == "fa_invariant"
    assert ns.frame_avg_angles == "0,90,180,270"
    assert ns.lr == 5e-6
    assert ns.max_steps == 10000
    assert ns.checkpoint_every == 2000
    assert ns.batch_size == 8
    assert ns.num_workers == 4
    assert ns.precision == "bf16-mixed"
    assert ns.gradient_clip_val == 1.0
    assert ns.seed == 123
    assert ns.smoke is True


def test_parse_defaults():
    argv = [
        "--model-config", "m.json",
        "--dataset-config", "d.json",
        "--ckpt-path", "c.ckpt",
        "--save-dir", "/tmp/out",
    ]
    ns = finetune_cond.build_parser().parse_args(argv)
    assert ns.cond_method == "vanilla"          # safe control by default
    assert ns.frame_avg_angles == "0,90,180,270"
    assert ns.lr == 5e-6                          # constant fine-tune LR
    assert ns.precision == "bf16-mixed"
    assert ns.gradient_clip_val == 0.0            # upstream parity (defaults.ini)
    assert ns.seed == 42
    assert ns.smoke is False                      # opt-in
    assert ns.pretransform_ckpt_path is None      # optional; VAE also present in main ckpt


def test_parse_rejects_bad_cond_method():
    argv = [
        "--model-config", "m.json",
        "--dataset-config", "d.json",
        "--ckpt-path", "c.ckpt",
        "--save-dir", "/tmp/out",
        "--cond-method", "canon",
    ]
    with pytest.raises(SystemExit):
        finetune_cond.build_parser().parse_args(argv)


# --------------------------------------------------------------------------------------
# 6. Review fixes (finetune round): configure_optimizers pin, exact-recipe pin,
#    grad-clip parity default, smoke checkpointing guarantee
# --------------------------------------------------------------------------------------

def test_configure_optimizers_bare_after_injection():
    """The injected config must yield a scheduler-free configure_optimizers().

    Built through the REAL factories (tiny CPU diffusion_cond config from
    test_cond_dispatch, given the FLAC_AR InverseLR scheduler so injection has
    something to remove). Per diffusion.py:191-203 Lightning conventions: bare
    optimizer -> a plain list [opt]; scheduler present -> ([opt], [sched_config])
    tuple. The un-injected control proves the assertion discriminates.
    """
    tiny = _base_config()
    tiny["training"]["optimizer_configs"]["diffusion"]["scheduler"] = {
        "type": "InverseLR",
        "config": {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99},
    }

    injected = finetune_cond.build_finetune_training_config(tiny, "fa_invariant", 5e-6, [0.0, 90.0])
    wrapper = create_training_wrapper_from_config(injected, create_model_from_config(injected))
    result = wrapper.configure_optimizers()
    assert isinstance(result, list) and not isinstance(result, tuple)  # no scheduler leg
    assert len(result) == 1
    assert isinstance(result[0], torch.optim.Optimizer)
    assert result[0].param_groups[0]["lr"] == 5e-6

    # Control: the scheduler-bearing config takes the two-leg branch -> the pin is non-vacuous.
    control = create_training_wrapper_from_config(tiny, create_model_from_config(tiny))
    control_result = control.configure_optimizers()
    assert isinstance(control_result, tuple) and len(control_result) == 2
    assert "scheduler" in control_result[1][0]


def _flatten(d, prefix=""):
    """Flatten a nested dict to dotted-leaf-key -> value (lists are leaves)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def test_recipe_touches_exactly_the_pinned_keys():
    """Flat-diff pin: the injected training block differs from the original in EXACTLY
    the approved recipe keys. Any silent drop/change of timestep_sampler,
    cfg_dropout_prob, mask_padding, betas, weight_decay, metrics.* etc. lands in one
    of these sets and fails."""
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(
        cfg, "fa_invariant", 5e-6, [0.0, 90.0, 180.0, 270.0]
    )
    orig = _flatten(cfg["training"])
    new = _flatten(out["training"])
    added = set(new) - set(orig)
    removed = set(orig) - set(new)
    changed = {k for k in set(orig) & set(new) if orig[k] != new[k]}
    assert added == {"cond_method", "frame_avg_angles"}
    assert removed == {
        "optimizer_configs.diffusion.scheduler.type",
        "optimizer_configs.diffusion.scheduler.config.inv_gamma",
        "optimizer_configs.diffusion.scheduler.config.power",
        "optimizer_configs.diffusion.scheduler.config.warmup",
    }
    assert changed == {"use_ema", "optimizer_configs.diffusion.optimizer.config.lr"}


def test_parser_gradient_clip_default_matches_upstream():
    """Recipe parity: original FLAC training used defaults.ini gradient_clip_val=0.0
    (train.py passes it straight through); a nonzero default would silently deviate
    R1/R2 from the parity-control recipe."""
    argv = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--ckpt-path", "c.ckpt", "--save-dir", "/tmp/out"]
    ns = finetune_cond.build_parser().parse_args(argv)
    assert ns.gradient_clip_val == 0.0


def test_smoke_trainer_kwargs_disable_checkpointing():
    """--smoke must make checkpointing impossible: enable_checkpointing=False (so
    Lightning cannot inject its default ModelCheckpoint) AND no ModelCheckpoint in
    the callbacks list."""
    kw = finetune_cond.build_trainer_kwargs(
        precision="bf16-mixed", max_steps=10, gradient_clip_val=0.0,
        checkpoint_every=500, save_dir="/tmp/unused", smoke=True,
    )
    assert kw["enable_checkpointing"] is False
    assert not any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])


def test_nonsmoke_trainer_kwargs_keep_checkpointing():
    kw = finetune_cond.build_trainer_kwargs(
        precision="bf16-mixed", max_steps=2000, gradient_clip_val=0.0,
        checkpoint_every=500, save_dir="/tmp/unused", smoke=False,
    )
    assert kw["enable_checkpointing"] is True
    assert any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in kw["callbacks"])
