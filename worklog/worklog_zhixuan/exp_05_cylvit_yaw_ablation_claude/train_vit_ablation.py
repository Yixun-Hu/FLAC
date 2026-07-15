"""Phase 2 short-run training for the SimpleViT vs CylViT FLAC ablation.

This script intentionally does *not* use train.py because the upstream trainer
has a fixed 1,000,000 step budget. It reuses the normal FLAC dataloader, model
factory and Lightning training wrapper, but exposes a bounded ``--max-steps``.

It also avoids direct strict loading of ``FLAC_EMA.ckpt`` into the new ViT
architectures. The released checkpoint contains DINOv3 geometry-conditioner
weights, whose shapes do not match SimpleViT/CylViT. We therefore load only
shape-compatible weights (VAE/pretransform, DiT, RIR conditioner, dist embedders)
and leave the new geometry ViT branches randomly initialized.
"""
import argparse
import copy
import json
import math
import os
import sys
import types
from pathlib import Path

import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.models.utils import load_ckpt_state_dict
from src.training import create_training_wrapper_from_config


def build_training_config(
    model_config: dict,
    lr: float,
    use_scheduler: bool = False,
    use_ema: bool = False,
) -> dict:
    """Return a trainable copy with stable short-run defaults."""
    cfg = copy.deepcopy(model_config)
    training = cfg["training"]
    training["use_ema"] = use_ema
    training["cond_method"] = "vanilla"
    diffusion_opt = training["optimizer_configs"]["diffusion"]
    diffusion_opt["optimizer"]["config"]["lr"] = lr
    if not use_scheduler:
        diffusion_opt.pop("scheduler", None)
    return cfg


def remap_flac_state_dict(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Load a FLAC checkpoint and remap EMA weights like eval_FLAC.py."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = dict(ckpt["state_dict"])
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "", 1)] = state_dict.pop(key)

    if any(key.startswith("diffusion_ema.ema_model.") for key in state_dict):
        print("[exp05] initializing DiT from EMA weights")
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                new_key = "model." + key[len("diffusion_ema.ema_model."):]
                state_dict[new_key] = state_dict.pop(key)
    return state_dict


def load_shape_compatible_weights(model: torch.nn.Module, ckpt_path: str) -> tuple[list[str], list[str], list[str]]:
    """Load only checkpoint tensors whose keys and shapes match ``model``."""
    model_state = model.state_dict()
    ckpt_state = remap_flac_state_dict(ckpt_path)

    compatible = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in ckpt_state.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        "[exp05] loaded compatible weights: "
        f"{len(compatible)} tensors; missing_after_load={len(missing)}; "
        f"skipped_shape={len(skipped_shape)}; skipped_missing={len(skipped_missing)}; "
        f"unexpected={len(unexpected)}"
    )
    if skipped_shape:
        print("[exp05] first skipped_shape keys:")
        for key in skipped_shape[:12]:
            print(f"  - {key}: ckpt={tuple(ckpt_state[key].shape)} model={tuple(model_state[key].shape)}")
    return list(missing), skipped_shape, skipped_missing


def freeze_pretransform(module) -> None:
    """Freeze the VAE/pretransform so Phase 2 trains conditioner + DiT only."""
    if module.diffusion.pretransform is not None:
        module.diffusion.pretransform.enable_grad = False
        module.diffusion.pretransform.requires_grad_(False)


def phase_lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    schedule: str,
) -> float:
    """Return the shared warmup/decay multiplier for one Phase 3 step."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_parameter_groups(module, geometry_lr: float, dit_lr: float, other_cond_lr: float):
    """Partition trainable FLAC parameters into disjoint Phase 3 ownership groups."""
    conditioners = module.diffusion.conditioner.conditioners
    geometry_modules = [conditioners["source_vit"], conditioners["context_poses_vit"]]

    geometry_by_id = {
        id(param): param
        for geometry_module in geometry_modules
        for param in geometry_module.parameters()
        if param.requires_grad
    }
    dit_by_id = {id(param): param for param in module.diffusion.model.parameters() if param.requires_grad}
    other_cond_by_id = {
        id(param): param
        for param in module.diffusion.conditioner.parameters()
        if param.requires_grad and id(param) not in geometry_by_id
    }

    overlap = (geometry_by_id.keys() & dit_by_id.keys()) | (geometry_by_id.keys() & other_cond_by_id.keys())
    overlap |= dit_by_id.keys() & other_cond_by_id.keys()
    if overlap:
        raise RuntimeError(f"Phase 3 optimizer groups overlap on {len(overlap)} parameters")

    covered = geometry_by_id.keys() | dit_by_id.keys() | other_cond_by_id.keys()
    unowned = [
        name
        for name, param in module.diffusion.named_parameters()
        if param.requires_grad and id(param) not in covered
    ]
    if unowned:
        raise RuntimeError("Unowned trainable Phase 3 parameters: " + ", ".join(unowned[:12]))

    specs = (
        ("geometry", geometry_by_id, geometry_lr),
        ("dit", dit_by_id, dit_lr),
        ("other_conditioners", other_cond_by_id, other_cond_lr),
    )
    groups = []
    summary = {}
    for name, params_by_id, lr in specs:
        params = list(params_by_id.values())
        if not params:
            continue
        if lr <= 0:
            for param in params:
                param.requires_grad_(False)
            summary[name] = {"lr": lr, "parameters": sum(p.numel() for p in params), "frozen": True}
            continue
        groups.append({"name": name, "params": params, "lr": lr})
        summary[name] = {"lr": lr, "parameters": sum(p.numel() for p in params), "frozen": False}
    return groups, summary


def install_phase3_optimizer(module, args, model_config: dict):
    """Install grouped AdamW and a checkpointable per-step warmup/decay schedule."""
    geometry_lr = args.geometry_lr if args.geometry_lr is not None else args.lr
    dit_lr = args.dit_lr if args.dit_lr is not None else args.lr
    other_cond_lr = args.other_cond_lr if args.other_cond_lr is not None else args.lr
    groups, summary = build_parameter_groups(module, geometry_lr, dit_lr, other_cond_lr)
    optimizer_config = model_config["training"]["optimizer_configs"]["diffusion"]["optimizer"]["config"]
    betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(groups, betas=betas, weight_decay=weight_decay)
        lr_lambda = lambda step: phase_lr_multiplier(
            step,
            args.warmup_steps,
            10 if args.smoke else args.max_steps,
            args.min_lr_ratio,
            args.scheduler,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda] * len(groups),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    module.configure_optimizers = types.MethodType(configure_optimizers, module)
    return summary


class StepPrinter(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps == 0:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            print(f"[exp05] step={trainer.global_step} loss={float(loss):.6f}", flush=True)

    def __init__(self, every_n_steps: int):
        self.every_n_steps = every_n_steps


def build_trainer(args, has_validation: bool):
    callbacks = [StepPrinter(args.log_every_n_steps)]
    if not args.smoke:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                every_n_train_steps=args.checkpoint_every,
                dirpath=args.save_dir,
                save_last=True,
                save_top_k=-1,
            )
        )
        if has_validation:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    monitor="val/avg_loss",
                    mode="min",
                    filename="best-{step:06d}",
                    auto_insert_metric_name=False,
                    dirpath=args.save_dir,
                    save_top_k=1,
                )
            )
    validation_kwargs = {}
    if has_validation:
        validation_kwargs = {
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every * args.accumulate_grad_batches,
            "limit_val_batches": args.limit_val_batches,
        }
    return pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_steps=10 if args.smoke else args.max_steps,
        callbacks=callbacks,
        logger=False,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=not args.smoke,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        **validation_kwargs,
    )


def run(args) -> None:
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.model_config) as f:
        model_config = build_training_config(
            json.load(f), args.lr, args.use_scheduler, use_ema=args.use_ema
        )
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    model = create_model_from_config(model_config)
    missing, skipped_shape, skipped_missing = load_shape_compatible_weights(model, args.ckpt_path)
    if args.require_full_load and (missing or skipped_shape or skipped_missing):
        raise RuntimeError(
            "Phase 3 requires an exact model-weight load: "
            f"missing={len(missing)} skipped_shape={len(skipped_shape)} "
            f"skipped_missing={len(skipped_missing)}"
        )

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
        print(f"[exp05] loaded explicit pretransform weights from {args.pretransform_ckpt_path}")

    module = create_training_wrapper_from_config(model_config, model)
    freeze_pretransform(module)
    optimizer_groups = install_phase3_optimizer(module, args, model_config)

    source_vit = module.diffusion.conditioner.conditioners["source_vit"]
    context_vit = module.diffusion.conditioner.conditioners["context_poses_vit"]
    print(f"[exp05] model_config={args.model_config}")
    print(f"[exp05] source_vit={type(source_vit.vit).__name__} token_pool={source_vit.token_pool}")
    print(f"[exp05] context_poses_vit={type(context_vit.vit).__name__} token_pool={context_vit.token_pool}")
    print(f"[exp05] optimizer_groups={json.dumps(optimizer_groups, sort_keys=True)}")
    print(
        f"[exp05] scheduler={args.scheduler} warmup_steps={args.warmup_steps} "
        f"min_lr_ratio={args.min_lr_ratio} max_steps={args.max_steps} use_ema={args.use_ema} "
        f"smoke={args.smoke}"
    )

    if args.dry_run:
        print("[exp05] dry run complete; not training.")
        return

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=min(args.batch_size, 2) if args.smoke else args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )
    val_dl = None
    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)
        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=min(args.val_batch_size, 2) if args.smoke else args.val_batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 1),
            shuffle=False,
        )

    trainer = build_trainer(args, has_validation=val_dl is not None)
    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume_from if args.resume_from else None)

    if args.smoke:
        print(f"[exp05] smoke finished at global_step={trainer.global_step}; no export.")
        return

    export_path = os.path.join(args.save_dir, args.name + ".ckpt")
    module.export_model(export_path, use_safetensors=False)
    print(f"[exp05] exported model to {export_path}")

    metadata_path = os.path.join(args.save_dir, args.name + "_load_report.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "model_config": args.model_config,
                "dataset_config": args.dataset_config,
                "ckpt_path": args.ckpt_path,
                "missing_after_load": missing,
                "skipped_shape": skipped_shape,
                "skipped_missing": skipped_missing,
                "lr": args.lr,
                "optimizer_groups": optimizer_groups,
                "warmup_steps": args.warmup_steps,
                "scheduler": args.scheduler,
                "min_lr_ratio": args.min_lr_ratio,
                "use_ema": args.use_ema,
                "val_dataset_config": args.val_dataset_config,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"[exp05] wrote load report to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", default="src/configs/dataset_configs/AR/train/acousticroom_train.json")
    parser.add_argument("--ckpt-path", default="weights/FLAC/FLAC_EMA.ckpt")
    parser.add_argument("--pretransform-ckpt-path", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--geometry-lr", type=float, default=None)
    parser.add_argument("--dit-lr", type=float, default=None)
    parser.add_argument("--other-cond-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=625)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=32)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--val-dataset-config", default=None)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--limit-val-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-n-steps", type=int, default=25)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--require-full-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional PyTorch-Lightning training checkpoint, e.g. epoch=0-step=250.ckpt. "
             "Use this for optimizer-state resume, not the exported FLAC_exp05_*.ckpt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
