#!/usr/bin/env python3
"""Train one exp06 CylViT patch-embedding variant with the original FLAC recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import pytorch_lightning as pl
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
FLAC_ROOT = SCRIPT_DIR.parents[2]
if str(FLAC_ROOT) not in sys.path:
    sys.path.insert(0, str(FLAC_ROOT))

from src.data.dataset import create_dataloader_from_config  # noqa: E402
from src.models import create_model_from_config  # noqa: E402
from src.training import create_training_wrapper_from_config  # noqa: E402


DEFAULT_CONFIGS = {
    "linear": "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_Linear.json",
    "cnn": "src/configs/model_configs/FLAC/AR/FLAC_AR_CylViT_PE_CNN.json",
}
PATCH_EMBED_CONTRACTS = {
    "linear": "flattened_patch_v1",
    "cnn": "independent_3x16x32_patch_cnn_v2",
}
DEFAULT_DATASET_CONFIG = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
DEFAULT_VAL_DATASET_CONFIG = "src/configs/dataset_configs/AR/eval/acousticroom_seeneval.json"
DEFAULT_INIT_DIR = FLAC_ROOT / "outputs_FLAC" / "exp06_cylvit_pe_matched_initializations"
DEFAULT_MILESTONES = (5_000, 10_000, 20_000, 30_000)


def resolve_from_flac(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (FLAC_ROOT / candidate).resolve()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def parse_milestones(value: str) -> tuple[int, ...]:
    try:
        milestones = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated positive integers") from exc
    if not milestones or any(step <= 0 for step in milestones):
        raise argparse.ArgumentTypeError("milestones must contain positive integers")
    return milestones


def assert_value(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"Original-FLAC recipe violation: {label}={actual!r}, expected {expected!r}")


def validate_original_flac_recipe(config: Mapping[str, Any], variant: str) -> None:
    training = config["training"]
    diffusion = config["model"]["diffusion"]
    optimizer = training["optimizer_configs"]["diffusion"]["optimizer"]
    scheduler = training["optimizer_configs"]["diffusion"]["scheduler"]

    assert_value("diffusion_objective", diffusion["diffusion_objective"], "rectified_flow")
    assert_value("timestep_sampler", training["timestep_sampler"], "log_snr")
    assert_value("mask_padding", training["mask_padding"], True)
    assert_value("mask_padding_dropout", training["mask_padding_dropout"], 0.0)
    assert_value("cfg_dropout_prob", training["cfg_dropout_prob"], 0.1)
    assert_value("use_ema", training["use_ema"], True)
    assert_value("optimizer.type", optimizer["type"], "AdamW")
    assert_value("optimizer.lr", optimizer["config"]["lr"], 5e-5)
    assert_value("optimizer.betas", optimizer["config"]["betas"], [0.9, 0.999])
    assert_value("optimizer.weight_decay", optimizer["config"]["weight_decay"], 1e-3)
    assert_value("scheduler.type", scheduler["type"], "InverseLR")
    assert_value("scheduler.inv_gamma", scheduler["config"]["inv_gamma"], 1_000_000)
    assert_value("scheduler.power", scheduler["config"]["power"], 0.5)
    assert_value("scheduler.warmup", scheduler["config"]["warmup"], 0.99)

    vit_types = []
    for conditioner in config["model"]["conditioning"]["configs"]:
        if conditioner["id"] in ("source_vit", "context_poses_vit"):
            vit_types.append(conditioner["config"]["ViT"].get("patch_embed_type", "linear"))
    assert_value("geometry patch_embed_type entries", vit_types, [variant, variant])


def load_init_checkpoint(
    model: torch.nn.Module,
    init_path: Path,
    variant: str,
    seed: int,
    config_path: Path,
) -> dict[str, Any]:
    payload = torch.load(init_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "exp06_init" not in payload:
        raise RuntimeError(f"Not an audited exp06 initialization checkpoint: {init_path}")
    metadata = payload["exp06_init"]
    assert_value("init experiment", metadata.get("experiment"), "exp06_cylvit_patch_embed_ablation")
    assert_value("init variant", metadata.get("variant"), variant)
    assert_value("init training_seed", metadata.get("training_seed"), seed)
    assert_value("init model_config_sha256", metadata.get("model_config_sha256"), sha256_file(config_path))

    state = payload["state_dict"]
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatch = sorted(
        key for key in set(model_state) & set(state) if model_state[key].shape != state[key].shape
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Initialization does not exactly match the model: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )
    model.load_state_dict(state, strict=True)

    conditioners = model.conditioner.conditioners
    if conditioners["source_vit"].vit is not conditioners["context_poses_vit"].vit:
        raise AssertionError("source_vit and context_poses_vit must share one ViT")
    actual_type = getattr(conditioners["source_vit"].vit, "patch_embed_type", None)
    assert_value("constructed patch_embed_type", actual_type, variant)
    return dict(metadata)


def freeze_and_assert_vae(module: pl.LightningModule) -> None:
    pretransform = module.diffusion.pretransform
    if pretransform is None:
        raise RuntimeError("FLAC exp06 requires the pretrained VAE/pretransform")
    pretransform.enable_grad = False
    pretransform.eval().requires_grad_(False)
    trainable = [name for name, parameter in pretransform.named_parameters() if parameter.requires_grad]
    if trainable:
        raise AssertionError(f"VAE/pretransform still has trainable parameters: {trainable[:8]}")


def parameter_summary(module: pl.LightningModule) -> dict[str, int]:
    conditioner = module.diffusion.conditioner
    geometry_modules = (
        conditioner.conditioners["source_vit"],
        conditioner.conditioners["context_poses_vit"],
    )
    geometry_by_id = {
        id(parameter): parameter for geometry in geometry_modules for parameter in geometry.parameters()
    }
    geometry_ids = set(geometry_by_id)
    dit_ids = {id(parameter) for parameter in module.diffusion.model.parameters()}
    other_conditioner_ids = {
        id(parameter) for parameter in conditioner.parameters() if id(parameter) not in geometry_ids
    }
    if (geometry_ids & dit_ids) or (geometry_ids & other_conditioner_ids) or (dit_ids & other_conditioner_ids):
        raise AssertionError("Parameter ownership groups overlap")

    summary = {
        "geometry_trainable": sum(
            parameter.numel() for parameter in geometry_by_id.values() if parameter.requires_grad
        ),
        "dit_trainable": sum(parameter.numel() for parameter in module.diffusion.model.parameters() if parameter.requires_grad),
        "other_conditioners_trainable": sum(
            parameter.numel()
            for parameter in conditioner.parameters()
            if parameter.requires_grad and id(parameter) in other_conditioner_ids
        ),
        "vae_trainable": sum(
            parameter.numel() for parameter in module.diffusion.pretransform.parameters() if parameter.requires_grad
        ),
    }
    if not summary["geometry_trainable"] or not summary["dit_trainable"] or not summary["other_conditioners_trainable"]:
        raise AssertionError(f"Joint-training contract is not satisfied: {summary}")
    assert_value("vae_trainable", summary["vae_trainable"], 0)
    return summary


class Exp06Metadata(pl.Callback):
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__()
        self.metadata = dict(metadata)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["exp06_run"] = dict(self.metadata)
        checkpoint["exp06_run"]["saved_global_step"] = int(trainer.global_step)


class StepPrinter(pl.Callback):
    def __init__(self, every_n_steps: int) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.last_printed = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step <= 0 or step == self.last_printed or step % self.every_n_steps:
            return
        self.last_printed = step
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        value = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        print(f"[exp06:train] step={step} loss={value:.8f} lr={lr:.10g}", flush=True)


class SparseCheckpoint(pl.Callback):
    """Save immutable sparse milestones and one atomically replaced rolling last."""

    def __init__(self, directory: Path, milestones: tuple[int, ...], last_every: int) -> None:
        super().__init__()
        self.directory = directory
        self.milestones = set(milestones)
        self.last_every = last_every
        self.handled_steps: set[int] = set()

    @property
    def last_path(self) -> Path:
        return self.directory / "last.ckpt"

    def milestone_path(self, step: int) -> Path:
        return self.directory / f"step={step:09d}.ckpt"

    def _atomic_save(self, trainer: pl.Trainer, destination: Path) -> None:
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    def _point_last_at_milestone(self, milestone: Path) -> None:
        temporary = self.last_path.with_name(self.last_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(milestone, temporary)
        except OSError:
            shutil.copy2(milestone, temporary)
        os.replace(temporary, self.last_path)

    def _save_step(self, trainer: pl.Trainer, step: int, force_last: bool = False) -> None:
        if not trainer.is_global_zero or step <= 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)

        milestone = None
        if step in self.milestones:
            milestone = self.milestone_path(step)
            if not milestone.exists():
                print(f"[exp06:ckpt] saving milestone {milestone}", flush=True)
                self._atomic_save(trainer, milestone)

        if milestone is not None:
            self._point_last_at_milestone(milestone)
            print(f"[exp06:ckpt] rolling last -> step {step}", flush=True)
        elif force_last or step % self.last_every == 0:
            print(f"[exp06:ckpt] saving rolling last at step {step}", flush=True)
            self._atomic_save(trainer, self.last_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(trainer.global_step)
        if step in self.handled_steps:
            return
        if step in self.milestones or (step > 0 and step % self.last_every == 0):
            self._save_step(trainer, step)
            self.handled_steps.add(step)

    def on_train_end(self, trainer, pl_module) -> None:
        step = int(trainer.global_step)
        if step not in self.handled_steps:
            self._save_step(trainer, step, force_last=True)
            self.handled_steps.add(step)

    def on_exception(self, trainer, pl_module, exception) -> None:
        step = int(trainer.global_step)
        if not trainer.is_global_zero or step <= 0:
            return
        try:
            print(
                f"[exp06:ckpt] exception at step={step}; attempting best-effort rolling last",
                flush=True,
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            self._atomic_save(trainer, self.last_path)
        except Exception as checkpoint_error:
            print(
                f"[exp06:ckpt] WARNING: exception checkpoint failed: {checkpoint_error!r}",
                flush=True,
            )


def run_metadata(
    args: argparse.Namespace,
    model_config: Path,
    dataset_config: Path,
    val_dataset_config: Path | None,
    init_checkpoint: Path,
    init_metadata: Mapping[str, Any],
    milestones: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "exp06_cylvit_patch_embed_ablation",
        "variant": args.variant,
        "patch_embedding_contract": PATCH_EMBED_CONTRACTS[args.variant],
        "training_seed": args.seed,
        "model_config": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "dataset_config": str(dataset_config),
        "dataset_config_sha256": sha256_file(dataset_config),
        "val_dataset_config": str(val_dataset_config) if val_dataset_config else None,
        "val_dataset_config_sha256": sha256_file(val_dataset_config) if val_dataset_config else None,
        "val_every_optimizer_steps": args.val_every,
        "val_batch_size": args.val_batch_size,
        "limit_val_batches": args.limit_val_batches,
        "init_checkpoint": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "init_common_geometry_sha256": init_metadata["common_geometry_sha256"],
        "init_non_geometry_sha256": init_metadata["non_geometry_sha256"],
        "max_steps": args.max_steps,
        "micro_batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "effective_batch_size": args.batch_size * args.accumulate_grad_batches,
        "precision": args.precision,
        "gradient_clip_val": 0.0,
        "milestones": list(milestones),
        "rolling_last_every": args.last_every,
        "optimizer": {
            "type": "AdamW",
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-3,
        },
        "scheduler": {"type": "InverseLR", "inv_gamma": 1_000_000, "power": 0.5, "warmup": 0.99},
        "ema": {"scope": "DiT only", "beta": 0.9999, "power": 0.75, "from_step": 0},
    }


def validate_resume(path: Path, expected: Mapping[str, Any]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("exp06_run") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Resume checkpoint lacks exp06_run metadata: {path}")
    for key in (
        "experiment",
        "variant",
        "training_seed",
        "model_config_sha256",
        "dataset_config_sha256",
        "val_dataset_config_sha256",
        "val_every_optimizer_steps",
        "val_batch_size",
        "limit_val_batches",
        "init_checkpoint_sha256",
        "max_steps",
        "micro_batch_size",
        "accumulate_grad_batches",
        "precision",
        "gradient_clip_val",
        "milestones",
        "rolling_last_every",
        "optimizer",
        "scheduler",
        "ema",
    ):
        if metadata.get(key) != expected.get(key):
            raise RuntimeError(
                f"Resume contract mismatch for {key}: checkpoint={metadata.get(key)!r}, "
                f"current={expected.get(key)!r}"
            )
    # The linear path did not change, so its legacy checkpoints remain resumable.
    # Old CNN checkpoints have compatible tensor shapes but used whole-panorama
    # convolution, so accepting one would silently mix two different models.
    if expected.get("variant") == "cnn":
        key = "patch_embedding_contract"
        if metadata.get(key) != expected.get(key):
            raise RuntimeError(
                f"Resume contract mismatch for {key}: checkpoint={metadata.get(key)!r}, "
                f"current={expected.get(key)!r}; restart the corrected CNN from initialization"
            )
    print(
        f"[exp06:train] validated resume={path} saved_global_step={metadata.get('saved_global_step')}",
        flush=True,
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("linear", "cnn"), required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--val-dataset-config", default=DEFAULT_VAL_DATASET_CONFIG)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--milestones", type=parse_milestones, default=DEFAULT_MILESTONES)
    parser.add_argument("--last-every", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument(
        "--val-every",
        type=int,
        default=2_500,
        help="Validation interval in optimizer steps; 0 disables validation.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Lightning validation fraction; 1.0 evaluates the full seen K=8 split.",
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument(
        "--resume-from",
        default="auto",
        help="auto, none, or a full Lightning checkpoint path; auto uses SAVE_DIR/last.ckpt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and strictly validate the model/init/recipe, then exit before data or GPU setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0 or args.last_every <= 0:
        raise ValueError("max_steps and last_every must be positive")
    if args.batch_size <= 0 or args.val_batch_size <= 0 or args.accumulate_grad_batches <= 0:
        raise ValueError("batch sizes and accumulate-grad-batches must be positive")
    if args.val_every < 0 or not 0.0 < args.limit_val_batches <= 1.0:
        raise ValueError("val-every must be >= 0 and limit-val-batches must be in (0, 1]")
    if args.devices != 1:
        raise ValueError("exp06 launch contract is one independent process/device per variant, not DDP")

    model_config = resolve_from_flac(args.model_config or DEFAULT_CONFIGS[args.variant])
    dataset_config = resolve_from_flac(args.dataset_config)
    val_dataset_config = (
        resolve_from_flac(args.val_dataset_config) if args.val_every > 0 else None
    )
    init_checkpoint = resolve_from_flac(
        args.init_checkpoint
        or DEFAULT_INIT_DIR / f"cylvit_pe_{args.variant}_trainS{args.seed}_init.ckpt"
    )
    save_dir = resolve_from_flac(
        args.save_dir
        or (
            f"outputs_FLAC/exp06_cylvit_pe_cnn_patchlocal_trainS{args.seed}"
            if args.variant == "cnn"
            else f"outputs_FLAC/exp06_cylvit_pe_linear_trainS{args.seed}"
        )
    )
    required_paths = [model_config, dataset_config, init_checkpoint]
    if val_dataset_config is not None:
        required_paths.append(val_dataset_config)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    milestones = tuple(args.milestones)
    if any(step > args.max_steps for step in milestones):
        raise ValueError(f"Milestones beyond max_steps={args.max_steps}: {milestones}")

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)
    config = load_json(model_config)
    validate_original_flac_recipe(config, args.variant)

    model = create_model_from_config(config)
    init_metadata = load_init_checkpoint(model, init_checkpoint, args.variant, args.seed, model_config)

    # Model construction consumes a variant-dependent number of random draws.
    # Reset immediately before wrapper construction so its internal state starts
    # from the same stream, then reset once more so dataloader workers, diffusion
    # noise and CFG dropout begin identically in the paired processes.
    pl.seed_everything(args.seed, workers=True)
    module = create_training_wrapper_from_config(config, model)
    freeze_and_assert_vae(module)
    parameters = parameter_summary(module)
    pl.seed_everything(args.seed, workers=True)
    metadata = run_metadata(
        args,
        model_config,
        dataset_config,
        val_dataset_config,
        init_checkpoint,
        init_metadata,
        milestones,
    )

    print(f"[exp06:train] variant={args.variant} seed={args.seed}")
    print(f"[exp06:train] model_config={model_config}")
    print(f"[exp06:train] init_checkpoint={init_checkpoint}")
    print(f"[exp06:train] save_dir={save_dir}")
    print(
        f"[exp06:train] validation={val_dataset_config or 'disabled'} "
        f"every_optimizer_steps={args.val_every} limit={args.limit_val_batches}",
        flush=True,
    )
    print(f"[exp06:train] parameters={json.dumps(parameters, sort_keys=True)}")
    print(
        "[exp06:train] rng_alignment=seed reset before wrapper and after wrapper; "
        "paired dataloader/noise streams start from the same seed",
        flush=True,
    )
    print(
        "[exp06:train] recipe="
        f"AdamW(lr=5e-5,betas=(0.9,0.999),wd=1e-3) "
        f"InverseLR(inv_gamma=1000000,power=0.5,warmup=0.99) "
        f"micro_batch={args.batch_size} accum={args.accumulate_grad_batches} "
        f"effective_batch={args.batch_size * args.accumulate_grad_batches} "
        f"precision={args.precision} max_steps={args.max_steps} clip=0 EMA=DiT-only-from-step0",
        flush=True,
    )
    if args.dry_run:
        print("[exp06:train] dry run passed; no dataloader, GPU, checkpoint, or training was started.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    resume_path: Path | None
    if args.resume_from == "auto":
        candidate = save_dir / "last.ckpt"
        resume_path = candidate if candidate.is_file() else None
    elif args.resume_from.lower() in ("", "none"):
        resume_path = None
    else:
        resume_path = resolve_from_flac(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    if resume_path is not None:
        validate_resume(resume_path, metadata)

    manifest = dict(metadata)
    manifest["parameter_summary"] = parameters
    manifest["resume_from"] = str(resume_path) if resume_path else None
    atomic_write_json(save_dir / "run_manifest.json", manifest)
    print(f"[exp06:train] wrote manifest={save_dir / 'run_manifest.json'}", flush=True)

    dataset = load_json(dataset_config)
    train_dl = create_dataloader_from_config(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config["sample_rate"],
        sample_size=config["sample_size"],
        audio_channels=config.get("audio_channels", 1),
    )
    val_dl = None
    if val_dataset_config is not None:
        val_dataset = load_json(val_dataset_config)
        val_dl = create_dataloader_from_config(
            val_dataset,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            sample_rate=config["sample_rate"],
            sample_size=config["sample_size"],
            audio_channels=config.get("audio_channels", 1),
            shuffle=False,
        )

    callbacks: list[pl.Callback] = [
        # Suppress Lightning's implicit default checkpoint. All actual saves are
        # owned by SparseCheckpoint below.
        # Keep the callback state compatible with torch.load(weights_only=True):
        # passing a Path makes Lightning serialize pathlib.PosixPath here.
        pl.callbacks.ModelCheckpoint(dirpath=str(save_dir), save_top_k=0),
        Exp06Metadata(metadata),
        SparseCheckpoint(save_dir, milestones, args.last_every),
        StepPrinter(args.log_every_n_steps),
    ]
    validation_kwargs: dict[str, Any] = {}
    if val_dl is not None:
        # Lightning counts train dataloader batches here, so multiply by the
        # accumulation factor to express the CLI interval in optimizer steps.
        validation_kwargs = {
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every * args.accumulate_grad_batches,
            "limit_val_batches": args.limit_val_batches,
        }
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="auto",
        precision=args.precision,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=0.0,
        callbacks=callbacks,
        logger=False,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=0,
        **validation_kwargs,
    )
    trainer.fit(module, train_dl, val_dl, ckpt_path=str(resume_path) if resume_path else None)
    print(f"[exp06:train] complete variant={args.variant} global_step={trainer.global_step}")


if __name__ == "__main__":
    main()
