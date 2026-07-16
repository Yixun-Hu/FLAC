import torch
import json
import os
import pytorch_lightning as pl

from prefigure.prefigure import get_all_args, push_wandb_config
from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from src.training import create_training_wrapper_from_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def _as_bool(value):
    """Coerce a prefigure-parsed flag to a genuine ``bool``.

    prefigure parses the lowercase ini literal ``false``/``true`` as the *string*
    "false"/"true" (ast.literal_eval rejects lowercase, so the flag is registered
    ``type=str``); only the capitalized ``False``/``True`` yield a real bool. Accept
    str or bool so a forwarded Trainer kwarg is always a genuine boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"cannot interpret sync_batchnorm={value!r} as a boolean")
    raise TypeError(f"sync_batchnorm must be bool or str, got {type(value).__name__}")

def build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    """Assemble the pl.Trainer keyword arguments (side-effect free; unit-testable).

    Reproduces the kwargs that were previously inlined into pl.Trainer(...) exactly;
    the only behavioral change is that max_steps is now sourced from args.max_steps
    (default 1000000 via defaults.ini) instead of a hard-coded literal, so a training
    budget can be set without editing code. strategy / callbacks / logger /
    checkpoint_dir / val_args are the values main() derives and passes straight through.

    sync_batchnorm (default off -> key ABSENT, so the kwargs are byte-identical to the
    pre-change dict and PL's own default False applies) forwards to
    Trainer(sync_batchnorm=True) only when enabled. It is a multi-GPU-only feature, so
    enabling it with num_gpus < 2 is a fail-closed ValueError (Yixun mandate) rather than
    a silently-ignored no-op. The guard lives here so both construct_trainer/main() and
    any direct caller hit it; val_args may NOT smuggle the key past the guard.
    """
    if "sync_batchnorm" in val_args:
        raise ValueError("sync_batchnorm must come from args (guarded), not val_args")
    sync_batchnorm = _as_bool(getattr(args, "sync_batchnorm", False))
    if sync_batchnorm and args.num_gpus < 2:
        raise ValueError(
            "sync_batchnorm=True requires multi-GPU training (num_gpus >= 2); got "
            f"num_gpus={args.num_gpus}. SyncBatchNorm synchronises BatchNorm statistics "
            "across ranks and is a no-op / unsupported on a single device -- set "
            "--num-gpus >= 2 or drop --sync-batchnorm."
        )
    kwargs = {
        "devices": args.num_gpus,
        "accelerator": "gpu",
        "num_nodes": args.num_nodes,
        "strategy": strategy,
        "precision": args.precision,
        "accumulate_grad_batches": args.accum_batches,
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": 100,
        "max_steps": args.max_steps, # HAA finetune recipe: --max-steps 1000
        "default_root_dir": checkpoint_dir,
        "gradient_clip_val": args.gradient_clip_val,
        "reload_dataloaders_every_n_epochs": 0,
        "num_sanity_val_steps": 0, # If you need to debug validation, change this line
        **val_args,
    }
    if sync_batchnorm:
        kwargs["sync_batchnorm"] = True  # multi-GPU only; guarded above (fail-closed)
    return kwargs

def construct_trainer(args, strategy, callbacks, logger, checkpoint_dir, val_args):
    """Construct the pl.Trainer from the assembled kwargs (the tested Trainer boundary)."""
    return pl.Trainer(**build_trainer_kwargs(args, strategy, callbacks, logger, checkpoint_dir, val_args))

def main():
    torch.set_float32_matmul_precision('medium') 
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )

    val_dl = None
    val_dataset_config = None
    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 1),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        print('Loading pretrained model...')
        weights = load_ckpt_state_dict(args.pretrained_ckpt_path)
        weights = {k.replace('diffusion.', ''): v for k, v in weights.items()} # For diffusion
        weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()} # For VAE
        disc_weights = {k: v for k, v in weights.items() if 'discriminator' in k}
        disc_weights = {k.replace('discriminator.', ''): v for k, v in disc_weights.items()}
        weights = {k: v for k, v in weights.items() if 'discriminator' not in k}
        weights = {k: v for k, v in weights.items() if 'losses' not in k}
        model.load_state_dict(weights, strict=True)

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name, name=args.experiment_name)
        logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.name, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
        
    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    # Logger 
    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = construct_trainer(
        args,
        strategy=strategy,
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
        logger=logger,
        checkpoint_dir=checkpoint_dir,
        val_args=val_args,
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
