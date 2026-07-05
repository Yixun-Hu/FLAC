import os
import argparse
import json
import math
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import rotate_scene_metadata, invariant_conditioning, DEFAULT_FRAME_ANGLES
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config, create_metric_callback_from_config


def build_output_paths(
    ckpt_path,
    steps,
    cfg_scale,
    eval_name,
    cond_method='vanilla',
    rotate_deg=0.0,
    n_angles=4,
):
    """Construct the metrics-JSON and predictions-.pt output paths for one run.

    Pure (no filesystem access): both paths sit in ``ckpt_path``'s directory,
    named ``<ckpt>_<kind>_<steps>_<cfg>_<eval_name><method><rot>.<ext>``.

    The vanilla + ``rotate_deg == 0`` name is byte-identical to the legacy
    ``eval_FLAC`` output, so exp_01 / exp_02 artifacts reproduce exactly. A
    non-vanilla ``cond_method`` inserts ``_<cond_method>_a<n_angles>`` and a
    non-zero ``rotate_deg`` appends ``_rot<int(rotate_deg)>``. Both suffixes now
    also land on the predictions name -- fixing the exp_02 bug where two
    ``rotate_deg`` values sharing an ``eval_name`` overwrote one predictions file.

    Returns
    -------
    dict
        ``{'metrics': <...>.json, 'predictions': <...>.pt}``.
    """
    ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
    directory = os.path.dirname(ckpt_path)
    method_suffix = '' if cond_method == 'vanilla' else f'_{cond_method}_a{n_angles}'
    rot_suffix = '' if rotate_deg == 0.0 else f'_rot{int(rotate_deg)}'
    stem = f'{steps}_{cfg_scale}_{eval_name}{method_suffix}{rot_suffix}'
    return {
        'metrics': os.path.join(directory, f'{ckpt_name}_metrics_{stem}.json'),
        'predictions': os.path.join(directory, f'{ckpt_name}_predictions_{stem}.pt'),
    }


def build_metrics_record(metrics_dict, ckpt_path, rotate_deg, cond_method, frame_avg_angles):
    """Assemble the dict written to the metrics JSON.

    Extends the legacy ``{metrics, ckpt_path, rotate_deg}`` record with
    ``cond_method`` and ``frame_avg_angles`` (the C4 frame-average angles used
    for ``fa_invariant``; ``None`` for vanilla).
    """
    return {
        "metrics": metrics_dict,
        "ckpt_path": ckpt_path,
        "rotate_deg": rotate_deg,
        "cond_method": cond_method,
        "frame_avg_angles": frame_avg_angles,
    }


def evaluate_model(
    model_config_path,
    dataset_config_path,
    ckpt_path,
    steps,
    cfg_scale,
    batch_size=64,
    num_workers=6,
    eval_name='FLAC_eval', 
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    store_predictions=False,
    rotate_deg=0.0,
    cond_method='vanilla',
    frame_avg_angles=None,
):
    # Fail fast on an unknown cond_method (the CLI is guarded by argparse
    # choices, but programmatic callers would otherwise silently run vanilla
    # while filenames/meta record the unknown method).
    if cond_method not in ('vanilla', 'fa_invariant'):
        raise ValueError(
            f"Unknown cond_method: {cond_method!r}; valid options: 'vanilla', 'fa_invariant'."
        )

    torch.set_float32_matmul_precision('medium')

    # Resolve the C4 frame-average angles (only used when cond_method == 'fa_invariant').
    if frame_avg_angles is None:
        frame_avg_angles = DEFAULT_FRAME_ANGLES
    frame_avg_angles = tuple(float(a) for a in frame_avg_angles)

    # Load configurations
    with open(model_config_path) as f:
        model_config = json.load(f)

    training_config = model_config.get('training', None)
    
    # Load ckpt
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('diffusion.'):
            new_key = key.replace('diffusion.', '')
            state_dict[new_key] = state_dict.pop(key)
    
    # Use EMA weights if available
    if training_config.get("use_ema", False) and any(k.startswith('diffusion_ema.ema_model.') for k in state_dict.keys()):
        print('Using EMA model')
        for key in list(state_dict.keys()):
            if key.startswith('diffusion_ema.ema_model.'):
                new_key = key.replace('diffusion_ema.ema_model.', 'model.')
                state_dict[new_key] = state_dict.pop(key)
        training_config['use_ema'] = False

    # Build model
    model = create_model_from_config(model_config)
    model.load_state_dict(state_dict, strict=False)

    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    model_config['training'] = training_config
    module = create_training_wrapper_from_config(model_config, model)
    module.eval().requires_grad_(False)
    module.to(device)
    
    with torch.amp.autocast(device):
        model = module.diffusion.model

    if module.diffusion.pretransform is not None:
            samples = model_config["sample_size"] // module.diffusion.pretransform.downsampling_ratio
    else: 
        samples = model_config["sample_size"]
    
    # Fix seed
    if isinstance(seed, str):
        seed = int(seed)
    pl.seed_everything(seed, workers=True)

    # Dataloader Eval
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    eval_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=False
    )

    # Metrics 
    metric_callback = create_metric_callback_from_config(model_config, dataset_id=dataset_config['datasets'][0]['id'])

    # Eval
    c=0
    if store_predictions:
        decoded_samples = []

    with torch.no_grad():
        for batch in tqdm(eval_dl, desc="Evaluating"):
            reals, metadata = batch
            reals = reals.to(device)

            # Optional yaw-rotation diagnostic: physically rotate the conditioning
            # (depth panorama + source/context poses) while leaving context_audio
            # and the target RIR fixed. See src/data/yaw_rotation.py.
            if rotate_deg != 0.0:
                alpha_rad = math.radians(rotate_deg)
                img_w = int(metadata[0]["depth"].shape[-1])
                metadata = [rotate_scene_metadata(md, alpha_rad, img_w) for md in metadata]

            with torch.amp.autocast(device):
                if cond_method == 'fa_invariant':
                    # Route-1 symmetrized conditioning: cylindrical pose invariants
                    # + C4 frame average of the ViT depth path. Applied AFTER the
                    # optional --rotate-deg above (that composition is the sanity check).
                    conditioning = invariant_conditioning(
                        module.diffusion.conditioner, metadata, module.device, frame_avg_angles
                    )
                else:
                    conditioning = module.diffusion.conditioner(metadata, module.device)
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            noise = torch.randn([reals.shape[0], module.diffusion.io_channels, samples]).to(module.device)

            if hasattr(model, "diffusion_objective"):
                objective = model.diffusion_objective
            else:
                objective = getattr(model, "objective", "rectified_flow")

            if objective == "v":
                from src.inference.sampling import sample
                fakes = sample(model, noise, steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
            elif objective == "rectified_flow":
                from src.inference.sampling import sample_discrete_euler
                fakes = sample_discrete_euler(model, noise, steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            elif objective == "rf_denoiser":
                from src.inference.sampling import sample_flow_pingpong
                logsnr = torch.linspace(-6, 2, steps+1).to(module.device)
                sigmas = torch.sigmoid(-logsnr)
                sigmas[0] = 1.0
                sigmas[-1] = 0.0
                fakes = sample_flow_pingpong(model, noise, sigmas=sigmas, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
            else:
                raise ValueError(f"Unknown diffusion objective: {objective}")

            # Decode 
            if module.diffusion.pretransform is not None:
                fakes = module.diffusion.pretransform.decode(fakes)
            
            if store_predictions:
                decoded_samples.append(fakes.cpu())
            
            # Clamp and pad if necessary
            fakes = fakes.clamp(-1.0, 1.0) 
            if fakes.shape != reals.shape:
                if fakes.shape[-1] < reals.shape[-1]:
                    fakes = torch.nn.functional.pad(fakes, (0, reals.shape[-1] - fakes.shape[-1]))
                else:
                    reals = torch.nn.functional.pad(reals, (0, fakes.shape[-1] - reals.shape[-1]))
    
            # Compute metrics
            scene_list = [md["scene"] for md in metadata]
            depth_list = [md["depth"] if 'depth' in md else None for md in metadata]
            query_list = [md["source"] if 'source' in md else None for md in metadata]
            depthMinusSource_list = [(d[:3, :, :] - source_pose[:, None, None]).unsqueeze(0).float().to(device) for d, source_pose in zip(depth_list, query_list)]
            metric_callback.update_metrics("test", fakes, reals, scene_list, depth=depthMinusSource_list)
            c += reals.shape[0]
    

    # Compute and print metrics
    metrics_dict = metric_callback.compute_metrics("test")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == 'T60' or 'to' in metric_name:
            metric_name += ' (%)'
        elif metric_name == 'EDT':
            metric_name += ' (ms)'
        elif metric_name == 'C50':
            metric_name += ' (dB)'
        print('Test/' + metric_name, metric_value)
    
    # Save metrics in a file
    output_paths = build_output_paths(
        ckpt_path, steps, cfg_scale, eval_name,
        cond_method=cond_method, rotate_deg=rotate_deg, n_angles=len(frame_avg_angles),
    )
    frame_angles_record = list(frame_avg_angles) if cond_method == 'fa_invariant' else None
    metrics_to_save = build_metrics_record(
        metrics_dict, ckpt_path, rotate_deg, cond_method, frame_angles_record,
    )
    path2save = output_paths['metrics']
    with open(path2save, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Metrics saved to {path2save}")

    if store_predictions:
        decoded_samples_all = torch.cat(decoded_samples, dim=0)
        path2save_preds = output_paths['predictions']
        preds_bundle = {
            "predictions": decoded_samples_all,
            "meta": {
                "dataset_config": dataset_config_path,
                "seed": seed,
                "n_samples": int(decoded_samples_all.shape[0]),
                "cond_method": cond_method,
                "frame_avg_angles": frame_angles_record,
                "rotate_deg": rotate_deg,
                "batch_size": batch_size,
            },
        }
        torch.save(preds_bundle, path2save_preds)
        print(f"Decoded samples saved to {path2save_preds}")

    print("Evaluation complete!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=1, help="Number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on")
    parser.add_argument("--eval-name", type=str, default='', help="Name of the evaluation run (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--store_predictions", action='store_true', help="Whether to store predictions or not")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Yaw-rotate the conditioning (depth + poses) by this many degrees before eval; 0 disables (default).")
    parser.add_argument("--cond-method", type=str, default="vanilla", choices=["vanilla", "fa_invariant"], help="Conditioning method: 'vanilla' (single conditioner pass) or 'fa_invariant' (cylindrical pose invariants + C4 ViT frame average). Composes with --rotate-deg (rotation applied first).")
    parser.add_argument("--frame-avg-angles", type=str, default="0,90,180,270", help="Comma-separated yaw angles in degrees for fa_invariant frame averaging; the first must be 0. Ignored when --cond-method vanilla.")
    args = parser.parse_args()

    if args.store_predictions:
        print('Warning: Storing predictions can use a lot of memory.')

    frame_avg_angles = tuple(float(a) for a in args.frame_avg_angles.split(","))

    evaluate_model(
        args.model_config,
        args.dataset_config,
        args.ckpt_path,
        cfg_scale=args.cfg_scale,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        eval_name=args.eval_name,
        seed=args.seed,
        store_predictions=args.store_predictions,
        rotate_deg=args.rotate_deg,
        cond_method=args.cond_method,
        frame_avg_angles=frame_avg_angles,
    )
