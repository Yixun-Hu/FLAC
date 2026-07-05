"""Short fine-tune of a trained FLAC checkpoint with frame-averaged conditioning.

This validates whether the yaw-symmetry prior helps when it is baked into training
(rather than applied only at inference, which we found does not help). It loads the
released FLAC checkpoint (reusing the eval_FLAC-style EMA loading), switches the
training-time conditioning to frame averaging over a yaw frame, and fine-tunes for a
small number of steps. The resulting PyTorch-Lightning checkpoints are directly
loadable by ``eval_FLAC.py`` (same wrapper format as the released checkpoint), so they
can be evaluated with ``--cond-method frame_avg``.

The VAE pretransform is frozen; only the DiT and the (DINOv3) conditioner are updated.
"""
import argparse
import json
import os

import pytorch_lightning as pl
import torch

from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config


def load_flac_ema_weights(model, ckpt_path):
    """
    Load a released FLAC checkpoint into a freshly built model (EMA weights, non-strict).

    Mirrors ``eval_FLAC.py``: strip the ``diffusion.`` prefix, then remap the EMA DiT
    weights ``diffusion_ema.ema_model.*`` onto ``model.*``. The conditioner and
    pretransform keep their online weights. Loading is non-strict so leftover EMA
    bookkeeping keys are ignored.

    Parameters
    ----------
    model : torch.nn.Module
        A freshly built ``ConditionedDiffusionModelWrapper``.
    ckpt_path : str
        Path to the released ``.ckpt`` (e.g. ``weights/FLAC/FLAC_EMA.ckpt``).

    Returns
    -------
    torch.nn.Module
        The same model with weights loaded in place.
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


def finetune(
    model_config_path,
    dataset_config_path,
    ckpt_path,
    save_dir,
    name,
    cond_method,
    frame_avg_angles,
    max_steps,
    checkpoint_every,
    batch_size,
    num_workers,
    max_context,
    precision,
    gradient_clip_val,
    seed,
):
    """
    Run the frame-averaging fine-tune and save checkpoints.

    Parameters
    ----------
    model_config_path : str
        Path to the FLAC model config JSON.
    dataset_config_path : str
        Path to the training dataset config JSON.
    ckpt_path : str
        Path to the released FLAC checkpoint used for initialization.
    save_dir : str
        Directory where PL checkpoints and the exported model are written.
    name : str
        Basename for the exported final model.
    cond_method : str
        Training conditioning method: "frame_avg" or "vanilla" (control).
    frame_avg_angles : list of float
        Yaw frame angles in degrees for frame averaging.
    max_steps : int
        Number of fine-tune steps.
    checkpoint_every : int
        Save a checkpoint every this many steps.
    batch_size : int
        Training batch size.
    num_workers : int
        Dataloader worker count.
    max_context : int
        Override for K (acoustic_context.max_context); 0 keeps the config value.
    precision : str
        PyTorch-Lightning precision string (e.g. "bf16-mixed").
    gradient_clip_val : float
        Gradient clipping value.
    seed : int
        Random seed.

    Returns
    -------
    None
    """
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed, workers=True)
    os.makedirs(save_dir, exist_ok=True)

    with open(model_config_path) as f:
        model_config = json.load(f)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    # Inject the frame-averaging training config
    model_config["training"]["cond_method"] = cond_method
    model_config["training"]["frame_avg_angles"] = frame_avg_angles

    if max_context > 0:
        dataset_config["modalities"]["acoustic_context"]["max_context"] = max_context

    model = create_model_from_config(model_config)
    model = load_flac_ema_weights(model, ckpt_path)

    module = create_training_wrapper_from_config(model_config, model)

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

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=checkpoint_every, dirpath=save_dir, save_top_k=-1
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=precision,
        max_steps=max_steps,
        callbacks=[ckpt_callback],
        logger=False,
        log_every_n_steps=50,
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_dl)

    export_path = os.path.join(save_dir, name + ".ckpt")
    module.export_model(export_path, use_safetensors=False)
    print(f"Exported fine-tuned model to {export_path}")


def main():
    """
    Parse CLI arguments and launch the fine-tune.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Frame-averaging fine-tune of FLAC.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--name", type=str, default="FLAC_frameavg_ft")
    parser.add_argument("--cond-method", type=str, default="frame_avg", choices=["frame_avg", "vanilla"])
    parser.add_argument("--frame-avg-angles", type=str, default="0,90,180,270")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=1, help="Override K; 0 keeps config value.")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        ckpt_path=args.ckpt_path,
        save_dir=args.save_dir,
        name=args.name,
        cond_method=args.cond_method,
        frame_avg_angles=[float(x) for x in args.frame_avg_angles.split(",") if x.strip() != ""],
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_context=args.max_context,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
