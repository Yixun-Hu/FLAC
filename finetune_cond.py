"""Non-destructive fine-tune of a released FLAC checkpoint (exp_03 Route 1).

Adapted from ``worklog/archive_pre_revert_2026-07-04/finetune_frame_avg.py`` with the
deliberate differences that each trace to a documented failure of the pre-revert attempt:

* ``--cond-method {vanilla, fa_invariant}`` is injected into the model config's ``training``
  block (with ``--frame-avg-angles``) and consumed by the cycle-4 factory plumbing
  (``create_training_wrapper_from_config`` -> ``DiffusionCondTrainingWrapper``); the wrapper
  dispatches conditioning through the Route-1 symmetrized path at every train/val/test step.
* ``--lr`` sets a **constant** learning rate: the optimizer LR is overridden AND the
  ``InverseLR`` scheduler is removed, so ``configure_optimizers`` (diffusion.py:195) returns
  a bare optimizer with no warm-up restart -- the artifact that corrupted the pre-revert run.
* ``use_ema`` is forced ``False``: the released weights already ARE the EMA average, so a
  fresh EMA warm-up would corrupt the control.
* The VAE pretransform is frozen (only the DiT + conditioner are updated).
* No ``max_context`` override: the dataset config's K is used as-is.

The exported PyTorch-Lightning checkpoints share the released wrapper format, so they are
directly loadable by ``eval_FLAC.py`` with ``--cond-method fa_invariant``.
"""
import argparse
import copy
import json
import os

import pytorch_lightning as pl
import torch

from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.models import create_model_from_config
from src.models.utils import load_ckpt_state_dict
from src.training import create_training_wrapper_from_config

VALID_COND_METHODS = ("vanilla", "fa_invariant")


def build_finetune_training_config(model_config, cond_method, lr, frame_avg_angles):
    """Deep-copy a FLAC model config and inject the non-destructive fine-tune recipe.

    Only the ``training`` block is modified; the caller's dict (and everything outside
    ``training``, including the architecture and conditioning topology) is left untouched.
    The constant learning rate is realised by overriding the optimizer LR *and* deleting the
    ``scheduler`` key, which makes ``DiffusionCondTrainingWrapper.configure_optimizers`` skip
    scheduler construction (src/training/diffusion.py:195) -- the most decisive neutralization
    of the InverseLR warm-up restart.

    Parameters
    ----------
    model_config : dict
        A FLAC ``diffusion_cond`` model config (as loaded from JSON). Not mutated.
    cond_method : str
        Conditioning method, one of ``VALID_COND_METHODS``.
    lr : float
        Constant learning rate for the fine-tune.
    frame_avg_angles : list of float
        Yaw frame angles (degrees) for ``fa_invariant`` frame averaging.

    Returns
    -------
    dict
        A new config with the fine-tune recipe injected into its ``training`` block.

    Raises
    ------
    ValueError
        If ``cond_method`` is not in ``VALID_COND_METHODS``.
    """
    if cond_method not in VALID_COND_METHODS:
        raise ValueError(
            f"Unknown cond_method: {cond_method!r}. Valid options: {VALID_COND_METHODS}."
        )

    cfg = copy.deepcopy(model_config)
    training = cfg["training"]

    training["cond_method"] = cond_method
    training["frame_avg_angles"] = list(frame_avg_angles)

    # Init weights ARE the released EMA average; a fresh-EMA warm-up would corrupt the control.
    training["use_ema"] = False

    # Constant LR: override the base LR and drop the InverseLR scheduler entirely so that
    # configure_optimizers returns an unscheduled optimizer (no warm-up restart).
    diffusion_opt = training["optimizer_configs"]["diffusion"]
    diffusion_opt["optimizer"]["config"]["lr"] = lr
    diffusion_opt.pop("scheduler", None)

    return cfg


def load_flac_ema_weights(model, ckpt_path):
    """Load a released FLAC checkpoint into a freshly built model (EMA weights, non-strict).

    Mirrors ``eval_FLAC.py``/the archive: strip a single ``diffusion.`` prefix, then remap
    EMA DiT weights ``diffusion_ema.ema_model.*`` onto ``model.*``. For an already-exported
    ``FLAC_EMA.ckpt`` (bare ``model.*`` / ``conditioner.*`` / ``pretransform.*`` keys) both
    steps are no-ops and the state dict loads directly -- crucially this also supplies the VAE
    ``pretransform.model.*`` weights. Non-strict so leftover EMA bookkeeping keys are ignored.

    Returns ``model`` with weights loaded in place.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "", 1)] = state_dict.pop(key)

    if any(k.startswith("diffusion_ema.ema_model.") for k in state_dict.keys()):
        print("Initializing DiT from EMA weights")
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                new_key = "model." + key[len("diffusion_ema.ema_model."):]
                state_dict[new_key] = state_dict.pop(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


class _SmokeLossPrinter(pl.Callback):
    """Print a crisp ``step / loss`` line each optimizer step (smoke-run evidence only)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        print(f"[smoke] global_step={trainer.global_step} loss={float(loss):.6f}", flush=True)


def warmup_lr_factor(step, warmup_steps):
    """Linear warmup factor: ``(step + 1) / warmup_steps`` capped at 1.0.

    Ramps over the first ``warmup_steps`` optimizer steps, then stays 1.0.
    ``warmup_steps <= 0`` disables warmup (factor 1.0 at every step, incl. step 0).
    """
    return min(1.0, (step + 1) / max(1, warmup_steps))


class WarmupLR(pl.Callback):
    """Linear lr warmup to ``target_lr`` over the first ``warmup_steps`` optimizer steps.

    Writes ``target_lr * warmup_lr_factor(trainer.global_step, warmup_steps)`` into every
    param group of every optimizer at each ``on_train_batch_start``. Keyed ONLY off
    ``trainer.global_step`` -- which counts OPTIMIZER steps in Lightning 2.1 (verified
    against the installed source in the exp_04 plan review) -- so under gradient
    accumulation the same lr is redundantly re-written for each micro-batch of one
    optimizer step and warmup length is measured in optimizer steps, never micro-batches.
    After warmup it keeps writing the constant target: idempotent, and it coexists with
    the removed-scheduler constant-lr recipe (no scheduler object involved).
    """

    def __init__(self, target_lr, warmup_steps):
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        lr = self.target_lr * warmup_lr_factor(trainer.global_step, self.warmup_steps)
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr


class FreezeBN(pl.Callback):
    """Freeze BatchNorm running statistics during fine-tuning (exp_05 V1').

    exp_04's W0 null control (lr=0) proved BN running-stat adaptation ALONE damages the
    released checkpoint; this callback eliminates that channel. At fit start it discovers
    every ``nn.BatchNorm1d/2d/3d`` under the trainable path (``pl_module.diffusion`` when
    present, else the module itself) -- practically the RIR encoder's BatchNorm2d stack,
    but discovered dynamically, never hardcoded -- and logs the qualified names + count.
    Discovered modules are forced to eval mode: running stats are USED but never UPDATED,
    matching released-inference normalization. Affine ``weight``/``bias`` keep
    ``requires_grad`` (freezing them would be a second intervention -- amendment review
    finding 3).

    Lightning 2.1 re-``train()``s the whole module AFTER the ``on_fit_start`` callbacks
    (``_FitLoop.reset()``) and again after every validation loop
    (``on_validation_model_train``), so eval is re-asserted on EVERY
    ``on_train_batch_start`` -- a cheap idempotent loop that gets the last word before
    each training forward.
    """

    _BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

    def __init__(self):
        self.frozen_bn_names = []
        self._frozen_modules = []
        self._discovered = False

    @property
    def frozen_bn_count(self):
        return len(self.frozen_bn_names)

    def _discover(self, pl_module):
        root = getattr(pl_module, "diffusion", pl_module)
        named = [(n, m) for n, m in root.named_modules() if isinstance(m, self._BN_TYPES)]
        self.frozen_bn_names = [n for n, _ in named]
        self._frozen_modules = [m for _, m in named]
        self._discovered = True

    def _apply_eval(self):
        for module in self._frozen_modules:
            module.eval()

    def on_fit_start(self, trainer, pl_module):
        self._discover(pl_module)
        print(
            f"[finetune_cond] FreezeBN: {self.frozen_bn_count} BatchNorm modules frozen "
            f"(eval mode, buffers pinned, affine trainable): {self.frozen_bn_names}"
        )
        self._apply_eval()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._discovered:
            self._discover(pl_module)
        self._apply_eval()


def build_recipe_echo(cond_method, frame_avg_angles, lr, warmup_steps, freeze_bn=False):
    """Build the one-line recipe echo printed at launch (quoted in worklogs).

    Returns
    -------
    str
    """
    return (
        f"[finetune_cond] recipe: cond_method={cond_method} "
        f"frame_avg_angles={frame_avg_angles} lr={lr} warmup_steps={warmup_steps} "
        f"freeze_bn={freeze_bn} use_ema=False scheduler=constant(removed)"
    )


def build_trainer_kwargs(precision, max_steps, gradient_clip_val, checkpoint_every, save_dir,
                         smoke, accumulate_grad_batches=1, target_lr=None, warmup_steps=0,
                         freeze_bn=False):
    """Build the ``pl.Trainer`` keyword arguments (side-effect free; unit-testable).

    ``smoke`` guarantees no checkpointing two ways: the ``ModelCheckpoint`` callback is
    replaced by a loss printer AND ``enable_checkpointing=False`` stops Lightning from
    injecting its default ``ModelCheckpoint`` when none is supplied (review finding 2).
    ``accumulate_grad_batches`` mirrors upstream ``train.py --accum-batches`` (effective
    batch = batch size x accumulation), for shared-GPU capacity.
    ``warmup_steps > 0`` appends a ``WarmupLR(target_lr, warmup_steps)`` callback
    (exp_04); ``0`` leaves the callback list exactly as before.
    ``freeze_bn=True`` appends a ``FreezeBN`` callback (exp_05 V1'); ``False`` leaves
    the callback list exactly as before.

    Returns
    -------
    dict
        Keyword arguments for ``pl.Trainer``.
    """
    if smoke:
        callbacks = [_SmokeLossPrinter()]
    else:
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=checkpoint_every, dirpath=save_dir, save_top_k=-1
            )
        ]
    if warmup_steps > 0:
        if target_lr is None:
            raise ValueError("target_lr is required when warmup_steps > 0")
        callbacks.append(WarmupLR(target_lr=target_lr, warmup_steps=warmup_steps))
    if freeze_bn:
        callbacks.append(FreezeBN())
    return {
        "devices": 1,
        "accelerator": "gpu",
        "precision": precision,
        "max_steps": max_steps,
        "callbacks": callbacks,
        "logger": False,
        "log_every_n_steps": 1 if smoke else 50,
        "gradient_clip_val": gradient_clip_val,
        "accumulate_grad_batches": accumulate_grad_batches,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": not smoke,
    }


def finetune(
    model_config_path,
    dataset_config_path,
    ckpt_path,
    pretransform_ckpt_path,
    save_dir,
    name,
    cond_method,
    frame_avg_angles,
    lr,
    warmup_steps,
    freeze_bn,
    max_steps,
    checkpoint_every,
    batch_size,
    num_workers,
    accumulate_grad_batches,
    precision,
    gradient_clip_val,
    seed,
    smoke,
):
    """Run the fine-tune and (unless ``smoke``) save/export checkpoints.

    Builds the model, loads the released EMA weights (which also supply the VAE), optionally
    refreshes the VAE from ``pretransform_ckpt_path`` (mirroring ``train.py``; consistent since
    the main ckpt already carries ``pretransform.model.*``), freezes the VAE, then fits.

    Parameters
    ----------
    model_config_path, dataset_config_path, ckpt_path : str
        FLAC model config, training dataset config, released-checkpoint paths.
    pretransform_ckpt_path : str or None
        Optional VAE ``.safetensors`` refresh.
    save_dir, name : str
        Output directory and exported-model basename.
    cond_method : {"vanilla", "fa_invariant"}
        Control vs Route-1 symmetrized conditioning.
    frame_avg_angles : list of float
        Yaw frame angles (degrees) for frame averaging.
    lr : float
        Constant learning rate (the warmup target when ``warmup_steps > 0``).
    warmup_steps : int
        Linear lr warmup length in optimizer steps; 0 = off (pre-exp_04 behavior).
    freeze_bn : bool
        Freeze all BatchNorm modules in the trainable path (eval mode, running stats
        pinned, affine still trainable); False = off (pre-exp_05 behavior).
    max_steps, checkpoint_every, batch_size, num_workers : int
        Trainer / dataloader sizing (``max_steps`` / ``batch_size`` overridden by ``smoke``).
    accumulate_grad_batches : int
        Gradient-accumulation batches (effective batch = ``batch_size`` x this); mirrors
        upstream ``train.py --accum-batches``.
    precision : str
        PyTorch-Lightning precision string (e.g. ``"bf16-mixed"``).
    gradient_clip_val : float
        Gradient clipping value.
    seed : int
        Random seed.
    smoke : bool
        If ``True``: ``max_steps=10``, batch forced small, no checkpointing, no final export.
    """
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed, workers=True)
    os.makedirs(save_dir, exist_ok=True)

    if smoke:
        max_steps = 10
        batch_size = min(batch_size, 2)

    with open(model_config_path) as f:
        model_config = json.load(f)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    model_config = build_finetune_training_config(model_config, cond_method, lr, frame_avg_angles)
    print(build_recipe_echo(cond_method, frame_avg_angles, lr, warmup_steps, freeze_bn))

    model = create_model_from_config(model_config)
    model = load_flac_ema_weights(model, ckpt_path)

    # The main checkpoint already carries the VAE; an explicit --pretransform-ckpt-path is a
    # consistent refresh (and robustness if a future ckpt lacks it), mirroring train.py.
    if pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path))
        print(f"Loaded pretransform weights from {pretransform_ckpt_path}")

    module = create_training_wrapper_from_config(model_config, model)
    print(
        f"[finetune_cond] wrapper cond_method={module.cond_method} "
        f"frame_avg_angles={module.frame_avg_angles}"
    )

    # Freeze the VAE pretransform: only the DiT + conditioner are fine-tuned.
    if module.diffusion.pretransform is not None:
        module.diffusion.pretransform.enable_grad = False
        module.diffusion.pretransform.requires_grad_(False)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )

    trainer = pl.Trainer(**build_trainer_kwargs(
        precision=precision,
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        checkpoint_every=checkpoint_every,
        save_dir=save_dir,
        smoke=smoke,
        accumulate_grad_batches=accumulate_grad_batches,
        target_lr=lr,
        warmup_steps=warmup_steps,
        freeze_bn=freeze_bn,
    ))

    trainer.fit(module, train_dl)

    if smoke:
        print(f"[smoke] finished at global_step={trainer.global_step} (no export)")
        return

    export_path = os.path.join(save_dir, name + ".ckpt")
    module.export_model(export_path, use_safetensors=False)
    print(f"Exported fine-tuned model to {export_path}")


def build_parser():
    """Build the CLI argument parser (side-effect free; used by tests and ``main``).

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Non-destructive conditioning fine-tune of FLAC.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--pretransform-ckpt-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="FLAC_cond_ft")
    parser.add_argument("--cond-method", type=str, default="vanilla", choices=list(VALID_COND_METHODS))
    parser.add_argument("--frame-avg-angles", type=str,
                        default=",".join(str(int(a)) for a in DEFAULT_FRAME_ANGLES))
    parser.add_argument("--lr", type=float, default=5e-6, help="Constant learning rate.")
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Linear lr warmup to --lr over this many OPTIMIZER steps (keyed off "
             "trainer.global_step, so gradient accumulation does not stretch it); "
             "0 = off (previous behavior).",
    )
    parser.add_argument(
        "--freeze-bn", action="store_true",
        help="Freeze all BatchNorm modules in the trainable path: eval mode + running "
             "stats pinned (the exp_04 W0-proven damage channel), affine still "
             "trainable. Names/count logged at fit start.",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--accumulate-grad-batches", type=int, default=1,
        help="Gradient-accumulation batches (effective batch = batch-size x this); "
             "mirrors upstream train.py --accum-batches. For shared-GPU capacity.",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument(
        "--gradient-clip-val", type=float, default=0.0,
        help="Matches upstream training default (defaults.ini gradient_clip_val=0.0); "
             "nonzero values are a recipe deviation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Ladder rung e: max_steps=10, no checkpointing, small batch, no export.",
    )
    return parser


def main():
    """Parse CLI arguments and launch the fine-tune."""
    args = build_parser().parse_args()

    finetune(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        ckpt_path=args.ckpt_path,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        save_dir=args.save_dir,
        name=args.name,
        cond_method=args.cond_method,
        frame_avg_angles=[float(x) for x in args.frame_avg_angles.split(",") if x.strip() != ""],
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        freeze_bn=args.freeze_bn,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        seed=args.seed,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
