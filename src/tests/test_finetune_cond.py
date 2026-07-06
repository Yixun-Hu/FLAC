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
import types

import pytest
import pytorch_lightning as pl
import torch

import finetune_cond
from src.models.factory import create_model_from_config
from src.training.factory import create_training_wrapper_from_config
from src.training.utils import InverseLR, create_optimizer_from_config
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
        "--accumulate-grad-batches", "2",
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
    assert ns.accumulate_grad_batches == 2
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
    assert ns.accumulate_grad_batches == 1        # no accumulation unless requested
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


def test_trainer_kwargs_accumulate_grad_batches():
    """Shared-GPU launch adaptation: batch 4 x accumulate 2 = effective batch 8.
    Accumulation is a Trainer arg (upstream train.py passes --accum-batches the
    same way), NOT a training-config key -- so the recipe flat-diff pin
    (test_recipe_touches_exactly_the_pinned_keys) is untouched by it."""
    kw = finetune_cond.build_trainer_kwargs(
        precision="bf16-mixed", max_steps=2000, gradient_clip_val=0.0,
        checkpoint_every=500, save_dir="/tmp/unused", smoke=False,
        accumulate_grad_batches=2,
    )
    assert kw["accumulate_grad_batches"] == 2
    # Default when not requested: 1 (no accumulation), in smoke and non-smoke alike.
    kw_default = finetune_cond.build_trainer_kwargs(
        precision="bf16-mixed", max_steps=10, gradient_clip_val=0.0,
        checkpoint_every=500, save_dir="/tmp/unused", smoke=True,
    )
    assert kw_default["accumulate_grad_batches"] == 1


# --------------------------------------------------------------------------------------
# 7. exp_04 warmup: warmup_lr_factor + WarmupLR callback + --warmup-steps
#    (plan_warmup_unblock.md §1; accumulation semantics = plan-review finding 2)
# --------------------------------------------------------------------------------------

def _trainer_kwargs(**overrides):
    """build_trainer_kwargs with boilerplate args filled; overrides forwarded."""
    base = dict(precision="bf16-mixed", max_steps=625, gradient_clip_val=0.0,
                checkpoint_every=500, save_dir="/tmp/unused", smoke=False)
    base.update(overrides)
    return finetune_cond.build_trainer_kwargs(**base)


def test_warmup_lr_factor():
    """Pure contract: linear ramp (step+1)/warmup_steps capped at 1.0; <=0 disables."""
    assert finetune_cond.warmup_lr_factor(0, 200) == 1 / 200
    assert finetune_cond.warmup_lr_factor(99, 200) == 100 / 200
    assert finetune_cond.warmup_lr_factor(199, 200) == 1.0   # exactly at boundary
    assert finetune_cond.warmup_lr_factor(5000, 200) == 1.0  # constant after warmup
    for step in (0, 1, 100, 5000):
        assert finetune_cond.warmup_lr_factor(step, 0) == 1.0  # 0 = off, incl. step 0


def test_warmup_callback_sets_lr():
    """After on_train_batch_start, EVERY param group's lr == target * factor(global_step)."""
    target = 5e-6
    p1, p2 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p1]}, {"params": [p2]}], lr=123.0)  # lr overwritten
    cb = finetune_cond.WarmupLR(target_lr=target, warmup_steps=200)
    fake_trainer = types.SimpleNamespace(global_step=0, optimizers=[opt])
    for step in (0, 100, 199, 5000):
        fake_trainer.global_step = step
        cb.on_train_batch_start(fake_trainer, None, None, 0)
        expected = target * min(1.0, (step + 1) / 200)
        for group in opt.param_groups:
            assert group["lr"] == expected


def test_warmup_accumulation_semantics():
    """Plan-review finding 2: warmup is measured in OPTIMIZER steps (trainer.global_step),
    never micro-batches. 32 on_train_batch_start calls at global_step=0 (one accumulation
    group at accumulate_grad_batches=32) must leave lr at target * 1/200 after EVERY call;
    the next 32 at global_step=1 give target * 2/200. An implementation advancing on an
    internal counter or on batch_idx fails here (batch_idx varies across the calls)."""
    cb = finetune_cond.WarmupLR(target_lr=1.0, warmup_steps=200)  # target 1.0 -> lr == factor
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=999.0)
    fake_trainer = types.SimpleNamespace(global_step=0, optimizers=[opt])
    for micro_batch_idx in range(32):
        cb.on_train_batch_start(fake_trainer, None, None, micro_batch_idx)
        assert opt.param_groups[0]["lr"] == 1 / 200
    fake_trainer.global_step = 1
    for micro_batch_idx in range(32, 64):
        cb.on_train_batch_start(fake_trainer, None, None, micro_batch_idx)
        assert opt.param_groups[0]["lr"] == 2 / 200


def test_warmup_default_off():
    """Default 0 = byte-identical current behavior (R1b); >0 appends a configured WarmupLR."""
    argv = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--ckpt-path", "c.ckpt", "--save-dir", "/tmp/out"]
    ns = finetune_cond.build_parser().parse_args(argv)
    assert ns.warmup_steps == 0

    kw_off = _trainer_kwargs(target_lr=5e-6, warmup_steps=0)
    assert not any(isinstance(cb, finetune_cond.WarmupLR) for cb in kw_off["callbacks"])
    kw_legacy = _trainer_kwargs()  # no warmup args at all == pre-exp_04 call
    assert [type(cb) for cb in kw_off["callbacks"]] == [type(cb) for cb in kw_legacy["callbacks"]]

    kw_on = _trainer_kwargs(target_lr=5e-6, warmup_steps=200)
    warmups = [cb for cb in kw_on["callbacks"] if isinstance(cb, finetune_cond.WarmupLR)]
    assert len(warmups) == 1
    assert warmups[0].target_lr == 5e-6
    assert warmups[0].warmup_steps == 200

    with pytest.raises(ValueError):  # warmup without a target lr is a config error
        _trainer_kwargs(warmup_steps=200)


def test_warmup_recorded():
    """The recipe echo (printed at launch, quoted in worklogs) must record warmup_steps."""
    echo = finetune_cond.build_recipe_echo(
        cond_method="vanilla", frame_avg_angles=[0.0, 90.0, 180.0, 270.0],
        lr=5e-6, warmup_steps=200,
    )
    assert "warmup_steps=200" in echo
    assert "cond_method=vanilla" in echo  # still carries the rest of the recipe
    assert "lr=5e-06" in echo


# --------------------------------------------------------------------------------------
# 8. exp_05 freeze-bn: FreezeBN callback + --freeze-bn (amendment review §5, 7 tests)
#    Context: exp_04 W0 proved BN running-stat adaptation ALONE causes fine-tune damage;
#    V1' fine-tunes with BN frozen (eval mode, buffers pinned, affine still trainable).
# --------------------------------------------------------------------------------------

class _BNNet(torch.nn.Module):
    """Tiny synthetic net with two BatchNorm2d layers (one nested) for FreezeBN tests."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.block = torch.nn.Sequential(torch.nn.Conv2d(4, 4, kernel_size=1),
                                         torch.nn.BatchNorm2d(4))

    def forward(self, x):
        return self.block(self.bn1(self.conv(x)))


def _freeze_bn_on(net):
    """Attach a FreezeBN to a synthetic module the way Lightning would drive it:
    discovery at fit start, then Lightning's fit-loop reset() re-train()s the whole
    module, then the first batch hook re-asserts. Returns the callback."""
    cb = finetune_cond.FreezeBN()
    fake_trainer = types.SimpleNamespace()
    cb.on_fit_start(fake_trainer, net)
    net.train()  # _FitLoop.reset() calls trainer.model.train() AFTER on_fit_start
    cb.on_train_batch_start(fake_trainer, net, None, 0)
    return cb


def test_freeze_bn_parser_default_off():
    """1. --freeze-bn is store_true, default False."""
    base = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--ckpt-path", "c.ckpt", "--save-dir", "/tmp/out"]
    ns = finetune_cond.build_parser().parse_args(base)
    assert ns.freeze_bn is False
    ns_on = finetune_cond.build_parser().parse_args(base + ["--freeze-bn"])
    assert ns_on.freeze_bn is True


def test_freeze_bn_recorded():
    """2. Recipe echo records freeze_bn (True when set, False by default)."""
    echo_on = finetune_cond.build_recipe_echo(
        cond_method="vanilla", frame_avg_angles=[0.0, 90.0, 180.0, 270.0],
        lr=5e-6, warmup_steps=200, freeze_bn=True,
    )
    assert "freeze_bn=True" in echo_on
    assert "warmup_steps=200" in echo_on  # rest of the recipe still carried
    echo_default = finetune_cond.build_recipe_echo(
        cond_method="vanilla", frame_avg_angles=[0.0], lr=5e-6, warmup_steps=0,
    )
    assert "freeze_bn=False" in echo_default


def test_freeze_bn_trainer_kwargs():
    """3. build_trainer_kwargs: FreezeBN present iff freeze_bn=True; False/omitted
    leaves the callback list byte-identical to the legacy call."""
    kw_on = _trainer_kwargs(freeze_bn=True)
    freezes = [cb for cb in kw_on["callbacks"] if isinstance(cb, finetune_cond.FreezeBN)]
    assert len(freezes) == 1
    kw_off = _trainer_kwargs(freeze_bn=False)
    assert not any(isinstance(cb, finetune_cond.FreezeBN) for cb in kw_off["callbacks"])
    kw_legacy = _trainer_kwargs()  # pre-exp_05 call shape
    assert [type(cb) for cb in kw_off["callbacks"]] == [type(cb) for cb in kw_legacy["callbacks"]]


def test_freeze_bn_forces_eval_and_survives_retrain():
    """4. CORE: BN modules forced to eval while the parent stays in train mode, and the
    callback re-asserts eval after EVERY re-train() (Lightning re-train()s the module at
    fit-loop reset and after each validation loop via on_validation_model_train)."""
    net = _BNNet()
    cb = finetune_cond.FreezeBN()
    fake_trainer = types.SimpleNamespace()
    cb.on_fit_start(fake_trainer, net)
    net.train()  # Lightning fit-loop reset() undoes any fit-start eval
    assert net.bn1.training is True and net.block[1].training is True  # train() really won
    cb.on_train_batch_start(fake_trainer, net, None, 0)
    assert net.training is True          # parent (and conv layers) stay in train mode
    assert net.conv.training is True
    assert net.bn1.training is False
    assert net.block[1].training is False
    # Lightning re-train()s again (next epoch / post-validation): re-assert must win again.
    net.train()
    assert net.bn1.training is True
    cb.on_train_batch_start(fake_trainer, net, None, 1)
    assert net.bn1.training is False
    assert net.block[1].training is False
    assert net.training is True


def test_freeze_bn_buffers_unchanged():
    """5. BN buffers bit-unchanged through a train-mode forward WITH the callback;
    negative control: the same forward WITHOUT it mutates them."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 5, 5) + 1.0  # nonzero mean -> running_mean must move if unfrozen
    net = _BNNet()
    _freeze_bn_on(net)
    before = {n: b.clone() for n, b in net.named_buffers()}
    net(x)
    for n, b in net.named_buffers():
        assert torch.equal(before[n], b), f"buffer {n} mutated under FreezeBN"
    # Negative control (no callback): train-mode forward mutates BN buffers.
    net2 = _BNNet()
    net2.train()
    nbt_before = net2.bn1.num_batches_tracked.clone()
    rm_before = net2.bn1.running_mean.clone()
    net2(x)
    assert not torch.equal(net2.bn1.num_batches_tracked, nbt_before)
    assert not torch.equal(net2.bn1.running_mean, rm_before)


def test_freeze_bn_affine_still_trainable():
    """6. BN affine weight/bias keep requires_grad=True (freezing them would be a second
    intervention -- amendment review finding 3)."""
    net = _BNNet()
    _freeze_bn_on(net)
    for bn in (net.bn1, net.block[1]):
        assert bn.weight.requires_grad is True
        assert bn.bias.requires_grad is True


def test_freeze_bn_names_and_count():
    """7. The callback exposes the discovered BN qualified names + count (the real run
    logs these; the RIR encoder currently has 20 -- discovered, never hardcoded)."""
    net = _BNNet()
    cb = finetune_cond.FreezeBN()
    cb.on_fit_start(types.SimpleNamespace(), net)
    assert cb.frozen_bn_names == ["bn1", "block.1"]
    assert cb.frozen_bn_count == 2
    # Production scope: when the pl_module has a .diffusion attribute (the training
    # wrapper), discovery walks THAT subtree (the trainable path).
    wrapper = types.SimpleNamespace(diffusion=net)
    cb2 = finetune_cond.FreezeBN()
    cb2.on_fit_start(types.SimpleNamespace(), wrapper)
    assert cb2.frozen_bn_names == ["bn1", "block.1"]
    assert cb2.frozen_bn_count == 2


# --------------------------------------------------------------------------------------
# 9. exp_06 --lr-schedule: arm L5 needs the ORIGINAL InverseLR schedule restored
#    (plan_gradpath_bisect.md §S2 / plan-review finding 1). 'constant' stays the default
#    and byte-identical to previous behavior -- the untouched no-kwarg pins above
#    (test_recipe_touches_exactly_the_pinned_keys, configure_optimizers bare branch)
#    remain the negative controls.
# --------------------------------------------------------------------------------------

_SCHEDULER_LEAVES = {
    "optimizer_configs.diffusion.scheduler.type",
    "optimizer_configs.diffusion.scheduler.config.inv_gamma",
    "optimizer_configs.diffusion.scheduler.config.power",
    "optimizer_configs.diffusion.scheduler.config.warmup",
}


def test_lr_schedule_parser():
    """1. --lr-schedule choices {constant, inverse-restart}, default 'constant'."""
    base = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--ckpt-path", "c.ckpt", "--save-dir", "/tmp/out"]
    ns = finetune_cond.build_parser().parse_args(base)
    assert ns.lr_schedule == "constant"
    ns_l5 = finetune_cond.build_parser().parse_args(base + ["--lr-schedule", "inverse-restart"])
    assert ns_l5.lr_schedule == "inverse-restart"
    with pytest.raises(SystemExit):
        finetune_cond.build_parser().parse_args(base + ["--lr-schedule", "cosine"])


@pytest.mark.parametrize("lr_schedule,expected_removed", [
    ("constant", _SCHEDULER_LEAVES),   # explicit kwarg == default behavior (scheduler removed)
    ("inverse-restart", set()),         # L5: scheduler PRESERVED -> nothing removed
])
def test_recipe_flat_diff_by_lr_schedule(lr_schedule, expected_removed):
    """2. Flat-diff parametrized over lr_schedule: 'constant' removes exactly the four
    scheduler leaves (as ever); 'inverse-restart' removes NOTHING while lr is still
    overridden and cond_method/frame_avg_angles/use_ema behave identically."""
    cfg = _load_flac_ar_config()
    out = finetune_cond.build_finetune_training_config(
        cfg, "fa_invariant", 5e-6, [0.0, 90.0, 180.0, 270.0], lr_schedule=lr_schedule,
    )
    orig = _flatten(cfg["training"])
    new = _flatten(out["training"])
    added = set(new) - set(orig)
    removed = set(orig) - set(new)
    changed = {k for k in set(orig) & set(new) if orig[k] != new[k]}
    assert added == {"cond_method", "frame_avg_angles"}
    assert removed == expected_removed
    assert changed == {"use_ema", "optimizer_configs.diffusion.optimizer.config.lr"}
    if lr_schedule == "inverse-restart":
        src_sched = cfg["training"]["optimizer_configs"]["diffusion"]["scheduler"]
        out_sched = out["training"]["optimizer_configs"]["diffusion"]["scheduler"]
        assert out_sched == src_sched          # preserved exactly as in the source config
        assert out_sched is not src_sched      # deep copy, no aliasing back to the input
        assert out_sched["type"] == "InverseLR"
        assert out_sched["config"] == {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99}


def test_configure_optimizers_scheduler_branch_inverse_restart():
    """3. inverse-restart-injected config through the REAL factories: configure_optimizers
    returns the ([optimizers], [scheduler_configs]) branch with an InverseLR carrying the
    original (inv_gamma 1e6, power 0.5, warmup 0.99) and the overridden lr as its base.
    The existing bare-optimizer test pins the 'constant' branch as the negative control."""
    tiny = _base_config()
    tiny["training"]["optimizer_configs"]["diffusion"]["scheduler"] = {
        "type": "InverseLR",
        "config": {"inv_gamma": 1000000, "power": 0.5, "warmup": 0.99},
    }
    injected = finetune_cond.build_finetune_training_config(
        tiny, "fa_invariant", 5e-6, [0.0, 90.0], lr_schedule="inverse-restart",
    )
    wrapper = create_training_wrapper_from_config(injected, create_model_from_config(injected))
    result = wrapper.configure_optimizers()
    assert isinstance(result, tuple) and len(result) == 2
    opts, sched_configs = result
    assert isinstance(opts[0], torch.optim.Optimizer)
    sched = sched_configs[0]["scheduler"]
    assert isinstance(sched, InverseLR)
    assert sched.inv_gamma == 1000000
    assert sched.power == 0.5
    assert sched.warmup == 0.99
    assert sched.base_lrs == [5e-6]  # the --lr override feeds the schedule's base lr
    assert sched_configs[0]["interval"] == "step"


def test_lr_schedule_recorded():
    """4. Recipe echo records lr_schedule (and defaults to constant)."""
    echo_l5 = finetune_cond.build_recipe_echo(
        cond_method="vanilla", frame_avg_angles=[0.0, 90.0, 180.0, 270.0],
        lr=5e-5, warmup_steps=0, freeze_bn=True, lr_schedule="inverse-restart",
    )
    assert "lr_schedule=inverse-restart" in echo_l5
    assert "freeze_bn=True" in echo_l5  # rest of the recipe still carried
    echo_default = finetune_cond.build_recipe_echo(
        cond_method="vanilla", frame_avg_angles=[0.0], lr=5e-6, warmup_steps=0,
    )
    assert "lr_schedule=constant" in echo_default


def test_unknown_lr_schedule_raises():
    """5. Programmatic callers: unknown lr_schedule fails fast with ValueError."""
    cfg = _load_flac_ar_config()
    with pytest.raises(ValueError):
        finetune_cond.build_finetune_training_config(
            cfg, "vanilla", 5e-6, [0.0], lr_schedule="cosine",
        )
