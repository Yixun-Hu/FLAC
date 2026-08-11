import os
import argparse
import contextlib
import json
import math
import subprocess
from typing import NamedTuple, Optional
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import (rotate_scene_metadata, invariant_conditioning,
                                   DEFAULT_FRAME_ANGLES, ORBIT_EXECUTION,
                                   FRAME_AVG_MAX_FWD_SAMPLES)
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config, create_metric_callback_from_config


def rot_suffix(rotate_deg):
    """The injective filename suffix for a yaw offset: ``5.625 -> '_rot5p625'``.

    Integer rotations keep their historical byte-identical form (exp_02's
    ``_rot180`` artifacts must still resolve); fractional ones would collide
    under ``int()`` -- R3 evaluates 5.625 deg, which used to land on ``_rot5``
    alongside 5 deg -- so they get a decimal-safe form (round-4 review B3).

    Exposed as its own function because the R3 EVAL NAME needs the same
    rendering as the filename: five rotation rows of one cell otherwise share an
    eval name, and the identity that distinguishes them lives only in a field.
    """
    return '' if float(rotate_deg) == 0.0 else '_rot' + rot_token(rotate_deg)


def rot_token(rotate_deg):
    """The decimal-safe rendering of a yaw offset, with no empty case.

    ``rot_suffix`` renders 0 as the empty string so unrotated filenames stay
    byte-identical to the legacy ones. An eval NAME cannot do that -- `rot` with
    nothing after it is not a name -- so the R3 naming uses this instead:
    ``0 -> '0'``, ``5.625 -> '5p625'``.
    """
    d = float(rotate_deg)
    return str(int(d)) if d.is_integer() else repr(d).replace('.', 'p')


ROTATE_MODES = ('fixed', 'random')


class RotationPlan(NamedTuple):
    """The resolved yaw-rotation protocol for one evaluation run.

    ``mode='fixed'``  -- the legacy behaviour: every sample is rotated by the same
    ``rotate_deg`` (0.0 meaning "not rotated at all"); ``rotate_seed`` is None.

    ``mode='random'`` -- exp_14's estimand: each sample independently draws a
    panorama-column offset from a generator seeded with ``rotate_seed``, so no
    single angle describes the run and ``rotate_deg`` is None (recorded as JSON
    ``null``), not 0.0 -- an unrotated run and a randomly-rotated one must never
    be readable as the same protocol.
    """
    mode: str
    rotate_deg: Optional[float]
    rotate_seed: Optional[int]

    @property
    def is_random(self):
        return self.mode == 'random'


def resolve_rotation_plan(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None,
                          eval_seed=None):
    """Resolve (and validate) the rotation protocol before anything else runs.

    Every failure here is a HARD error, never a silent precedence rule --- this is
    the announcement-05 trap (eval-protocol flags are part of the experiment): a
    mismatched flag produces plausible-looking, catastrophically wrong numbers.

    - ``random`` with a non-zero ``rotate_deg``: a fixed angle and a per-sample
      draw are mutually exclusive protocols, so there is no correct winner.
    - ``fixed`` with an explicit ``rotate_seed``: silently ignoring it would let a
      manifest record a rotation seed that influenced nothing (review B4).
    - ``random`` with no seed anywhere: the assignment would be unreproducible.

    In ``random`` mode an omitted ``rotate_seed`` resolves to the EVAL seed
    (Yixun 2026-08-10), so a cell's rotation assignment is a function of the seed
    it already reports.
    """
    if rotate_mode not in ROTATE_MODES:
        raise ValueError(
            f"Unknown rotate_mode: {rotate_mode!r}; valid options: {list(ROTATE_MODES)}."
        )
    if rotate_mode == 'fixed':
        if rotate_seed is not None:
            raise ValueError(
                f"rotate_seed={rotate_seed!r} (--rotate-seed) is meaningless with "
                "rotate_mode='fixed' and must not be silently ignored: a fixed rotation "
                "draws nothing. Pass --rotate-mode random to use it."
            )
        return RotationPlan('fixed', float(rotate_deg), None)

    if float(rotate_deg) != 0.0:
        raise ValueError(
            f"rotate_mode='random' is incompatible with rotate_deg={rotate_deg!r} "
            "(--rotate-deg): a per-sample random yaw and a single fixed angle are "
            "different protocols. Drop --rotate-deg (or use --rotate-mode fixed)."
        )
    resolved = rotate_seed if rotate_seed is not None else eval_seed
    if resolved is None:
        raise ValueError(
            "--rotate-mode random needs a rotation seed: pass --rotate-seed, or an "
            "eval --seed for it to default to. An unseeded assignment is not reproducible."
        )
    return RotationPlan('random', None, int(resolved))


def rotation_token(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None):
    """The identity token for a run's rotation: ``'45'`` / ``'5p625'`` / ``'rand42'``.

    Fixed angles keep :func:`rot_token`'s rendering unchanged. A random-mode run
    is identified by its RESOLVED rotation seed, because that seed --- not any
    angle --- is what determines the assignment.
    """
    if rotate_mode == 'random':
        if rotate_seed is None:
            raise ValueError("random-mode naming requires a resolved rotate_seed")
        return f'rand{int(rotate_seed)}'
    return rot_token(rotate_deg)


def rotation_suffix(rotate_mode='fixed', rotate_deg=0.0, rotate_seed=None):
    """The filename suffix for a run's rotation; ``''`` for the unrotated case.

    Fixed mode is byte-identical to :func:`rot_suffix` (legacy artifacts must
    still resolve). Random mode appends ``_rotrand<seed>``, which is injective
    both across rotation seeds and against every fixed token: a fixed token
    always starts with a digit or ``'-'``, so it can never render as ``rand...``
    (review B5 --- reusing one eval name across rotation seeds must not overwrite).
    """
    if rotate_mode == 'random':
        return '_rot' + rotation_token(rotate_mode, rotate_deg, rotate_seed)
    return rot_suffix(rotate_deg)


def build_output_paths(
    ckpt_path,
    steps,
    cfg_scale,
    eval_name,
    cond_method='vanilla',
    rotate_deg=0.0,
    n_angles=4,
    rotate_mode='fixed',
    rotate_seed=None,
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

    ``rotate_mode='random'`` (exp_14) replaces the angle suffix with
    ``_rotrand<rotate_seed>``; the fixed-mode names are unchanged down to the byte.

    Returns
    -------
    dict
        ``{'metrics': <...>.json, 'predictions': <...>.pt}``.
    """
    ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
    directory = os.path.dirname(ckpt_path)
    method_suffix = '' if cond_method == 'vanilla' else f'_{cond_method}_a{n_angles}'
    rot = rotation_suffix(rotate_mode, rotate_deg, rotate_seed)
    stem = f'{steps}_{cfg_scale}_{eval_name}{method_suffix}{rot}'
    return {
        'metrics': os.path.join(directory, f'{ckpt_name}_metrics_{stem}.json'),
        'predictions': os.path.join(directory, f'{ckpt_name}_predictions_{stem}.pt'),
    }


def resolve_weights_source(training_config, state_dict_keys):
    """Which weights an evaluation will actually use: ``'ema'`` or ``'online'``.

    ``evaluate_model`` swaps in EMA weights only when the config asks for them AND
    the checkpoint carries ``diffusion_ema.ema_model.*`` entries; otherwise it
    silently evaluates the online weights. A screen that merely *asserts* EMA in a
    sidecar can therefore be wrong, so the record states what actually happened
    (round-4 review B1)."""
    wants_ema = bool((training_config or {}).get("use_ema", False))
    has_ema = any(str(k).startswith("diffusion_ema.ema_model.") for k in state_dict_keys)
    return "ema" if (wants_ema and has_ema) else "online"


def source_sha():
    """Short-circuited ``git rev-parse HEAD`` for provenance; ``'unknown'`` if git
    is unavailable. Provenance must never be able to break an evaluation."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, timeout=10,
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def orbit_provenance(cond_method):
    """``(orbit_execution, frame_avg_fwd_cap)`` for a conditioning method.

    A vanilla evaluation executes NO orbit, so labelling it ``batched`` would be
    false provenance and would make a vanilla row look protocol-compatible with a
    batched frame-averaged row."""
    if cond_method == "fa_invariant":
        return ORBIT_EXECUTION, FRAME_AVG_MAX_FWD_SAMPLES
    return "n/a", None


def build_metrics_record(metrics_dict, ckpt_path, rotate_deg, cond_method, frame_avg_angles,
                         cond_autocast='default', batch_size=None, n_samples=None,
                         dataset_config=None, seed=None, cfg_scale=None, steps=None,
                         eval_name=None, weights_source=None, device=None):
    """Assemble the dict written to the metrics JSON.

    Extends the legacy ``{metrics, ckpt_path, rotate_deg}`` record with
    ``cond_method``, ``frame_avg_angles`` (the frame-average angles used for
    ``fa_invariant``; ``None`` for vanilla), ``cond_autocast``, and the ORBIT
    EXECUTION provenance (exp_11): which implementation produced the
    conditioning (``batched`` vs the legacy per-angle ``loop``), its per-forward
    sample cap, and the source SHA. Rows produced by different orbit executions
    are not interchangeable — the batched path changes the train-mode
    augmentation schedule and regroups the evaluation tail batch — so this is
    what makes "legacy-loop" a checkable label rather than a footnote.
    """
    execution, cap = orbit_provenance(cond_method)
    return {
        "metrics": metrics_dict,
        "ckpt_path": ckpt_path,
        "rotate_deg": rotate_deg,
        "cond_method": cond_method,
        "frame_avg_angles": frame_avg_angles,
        "cond_autocast": cond_autocast,
        "orbit_execution": execution,
        "frame_avg_fwd_cap": cap,
        "source_sha": source_sha(),
        # the batched path regroups the split's TAIL batch, so the schedule that
        # produced a row must be reconstructible from the row itself
        "batch_size": batch_size,
        "n_samples": n_samples,
        # ...and so must the rest of the runtime protocol: which split ran, how
        # many items were actually evaluated, and which weights were used. Omitted
        # values stay explicitly None rather than a plausible-looking guess
        # (round-4 review B2).
        "dataset_config": dataset_config,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "eval_name": eval_name,
        "weights_source": weights_source,
        "device": device,
    }


def resolve_cond_autocast(mode):
    """Map a ``--cond-autocast`` mode to ``(enabled, dtype)`` for the conditioning call.

    ``'default'`` -> ``(True, None)``: ``torch.amp.autocast(device)`` with no explicit
    dtype, i.e. the torch per-device default (fp16 on cuda) -- byte-identical to the
    exp_01/exp_02 protocol. ``'bf16'`` -> ``(True, torch.bfloat16)``: matches
    ``finetune_cond``'s bf16-mixed training precision. ``'off'`` -> ``(False, None)``:
    no autocast (fp32), for exactness measurements.
    """
    modes = {"default": (True, None), "bf16": (True, torch.bfloat16), "off": (False, None)}
    if mode not in modes:
        raise ValueError(f"Unknown cond_autocast: {mode!r}; valid options: {sorted(modes)}.")
    return modes[mode]


def build_predictions_meta(dataset_config_path, seed, n_samples, cond_method,
                           frame_avg_angles, rotate_deg, batch_size, cond_autocast):
    """Sidecar meta saved by ``--store_predictions`` (read by the exp_02 comparator
    guard). Carries the same orbit-execution provenance as the metrics record:
    at the default evaluation batch the batched path degenerates to one angle per
    call for every full batch, but the split's tail batch is regrouped, so a
    prediction set must name the execution that produced it."""
    execution, cap = orbit_provenance(cond_method)
    return {
        "dataset_config": dataset_config_path,
        "seed": seed,
        "n_samples": n_samples,
        "cond_method": cond_method,
        "frame_avg_angles": frame_avg_angles,
        "rotate_deg": rotate_deg,
        "batch_size": batch_size,
        "cond_autocast": cond_autocast,
        "orbit_execution": execution,
        "frame_avg_fwd_cap": cap,
        "source_sha": source_sha(),
    }


# Unexpected-key prefixes a PL-wrapper checkpoint legitimately leaves after
# evaluate_model's 'diffusion.' strip: EMA copy/bookkeeping and loss-module buffers.
# Verified against outputs_FLAC/ft_vanilla/epoch=0-step=2000.ckpt (1279 keys):
# all 213 leftovers are diffusion_ema.* (212) + losses.* (1). Exported/bare
# checkpoints (FLAC_EMA.ckpt, *_ft.ckpt: 1066 keys) must load with zero of both.
LOAD_WHITELIST_PREFIXES = ("diffusion_ema.", "losses.")


def check_load_integrity(missing, unexpected, allow_partial_load=False,
                         whitelist_prefixes=LOAD_WHITELIST_PREFIXES):
    """Report ``load_state_dict(strict=False)`` results and fail on real mismatches.

    Missing keys are never acceptable (an un-initialized model weight silently
    corrupts metrics); unexpected keys are tolerated only under
    ``whitelist_prefixes``. Any other mismatch raises ``RuntimeError`` unless
    ``allow_partial_load`` is set, which downgrades it to a printed warning.
    """
    missing = list(missing)
    stray = [k for k in unexpected if not k.startswith(whitelist_prefixes)]
    n_benign = len(list(unexpected)) - len(stray)
    print(f"Checkpoint load: {len(missing)} missing, {len(stray)} stray unexpected, "
          f"{n_benign} whitelisted wrapper-leftover keys.")
    if not (missing or stray):
        return
    msg = (f"Checkpoint did not load cleanly: {len(missing)} missing keys "
           f"(first: {missing[:5]}), {len(stray)} stray unexpected keys "
           f"(first: {stray[:5]}). Wrong model-config/checkpoint pairing? "
           "Re-export the checkpoint, or pass --allow-partial-load to continue anyway.")
    if allow_partial_load:
        print("WARNING (--allow-partial-load): " + msg)
    else:
        raise RuntimeError(msg)


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
    cond_autocast='default',
    allow_partial_load=False,
):
    # Fail fast on an unknown cond_method (the CLI is guarded by argparse
    # choices, but programmatic callers would otherwise silently run vanilla
    # while filenames/meta record the unknown method).
    if cond_method not in ('vanilla', 'fa_invariant'):
        raise ValueError(
            f"Unknown cond_method: {cond_method!r}; valid options: 'vanilla', 'fa_invariant'."
        )

    # Fail fast on an unknown cond_autocast too (full-review condition C1).
    ac_enabled, ac_dtype = resolve_cond_autocast(cond_autocast)

    def cond_autocast_ctx():
        """Fresh autocast context for one conditioning call (same semantics per batch)."""
        if not ac_enabled:
            return contextlib.nullcontext()
        if ac_dtype is None:
            return torch.amp.autocast(device)  # per-device default: exp_01/02 protocol
        return torch.amp.autocast(device, dtype=ac_dtype)

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
    
    # Use EMA weights if available (the resolved source is recorded, not assumed)
    weights_source = resolve_weights_source(training_config, state_dict.keys())
    if training_config.get("use_ema", False) and any(k.startswith('diffusion_ema.ema_model.') for k in state_dict.keys()):
        print('Using EMA model')
        for key in list(state_dict.keys()):
            if key.startswith('diffusion_ema.ema_model.'):
                new_key = key.replace('diffusion_ema.ema_model.', 'model.')
                state_dict[new_key] = state_dict.pop(key)
        training_config['use_ema'] = False

    # Build model; assert the checkpoint actually loaded (full-review condition C2).
    model = create_model_from_config(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected, allow_partial_load)

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

            with cond_autocast_ctx():
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
        cond_autocast=cond_autocast, batch_size=batch_size, n_samples=c,
        dataset_config=dataset_config_path, seed=seed, cfg_scale=cfg_scale,
        steps=steps, eval_name=eval_name, weights_source=weights_source,
        device=str(device),
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
            "meta": build_predictions_meta(
                dataset_config_path, seed, int(decoded_samples_all.shape[0]),
                cond_method, frame_angles_record, rotate_deg, batch_size, cond_autocast,
            ),
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
    parser.add_argument("--frame-avg-angles", type=str, default=",".join(str(int(a)) for a in DEFAULT_FRAME_ANGLES), help="Comma-separated yaw angles in degrees for fa_invariant frame averaging; the first must be 0. Ignored when --cond-method vanilla.")
    parser.add_argument("--cond-autocast", type=str, default="default", choices=["default", "bf16", "off"], help="Autocast mode for the conditioning call: 'default' = torch per-device default dtype (fp16 on cuda; the exp_01/exp_02 protocol), 'bf16' = bfloat16 (matches finetune_cond's bf16-mixed training), 'off' = no autocast (fp32, for exactness measurements).")
    parser.add_argument("--allow-partial-load", action='store_true', help="Continue with a warning when the checkpoint does not load cleanly (missing or non-whitelisted unexpected keys) instead of raising.")
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
        cond_autocast=args.cond_autocast,
        allow_partial_load=args.allow_partial_load,
    )
