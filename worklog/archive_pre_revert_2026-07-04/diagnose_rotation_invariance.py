"""Phase-1 diagnostic: cylindrical (yaw) rotation invariance of a trained FLAC model.

This script probes whether an *already trained* FLAC checkpoint respects the
azimuthal (yaw) rotation symmetry of a mono room impulse response (RIR). For a
mono RIR, rigidly rotating the whole scene (room geometry + source + listener)
about the vertical axis leaves the ground-truth RIR unchanged. We therefore
apply a physically-consistent yaw rotation to the *conditioning only* --- the
equirectangular depth panorama (a horizontal circular roll plus a rotation of
its stored 3D vectors) and the source/context poses (a rotation about z) ---
while keeping ``context_audio``, the target RIR, and the diffusion noise fixed.

Two quantities are reported per rotation angle ``alpha``:

* Metric 1 (invariance gap): ``metric(P_alpha, P0)`` where ``P0`` is the
  prediction on the original conditioning and ``P_alpha`` on the rotated
  conditioning. Needs no ground truth. A large gap means the model is sensitive
  to absolute panorama yaw orientation.
* Metric 2 (accuracy degradation): ``metric(P0, GT)`` vs ``metric(P_alpha, GT)``.
  If the error grows under rotation, the orientation sensitivity actually hurts
  prediction accuracy.

Only depth-free acoustic metrics (T60, C50, EDT, L1-STFT, multires L1-STFT, Env)
are used. FD / retrieval are intentionally disabled because they run AGREE on
the (rotated) depth, and AGREE is itself not rotation invariant, which would
confound the measurement.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scipy.io.wavfile as wav
import torch

from src.data.dataset import create_dataloader_from_config
from src.data.yaw_rotation import (
    azimuth_rotation_matrix,
    rotate_scene_metadata,
    yaw_transform_consistency,
)
from src.inference.sampling import sample, sample_discrete_euler, sample_flow_pingpong
from src.metrics.metric_callback import AcousticMetricsCallback
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config

GAP_METRIC_KEYS: Tuple[str, ...] = ("T60", "C50", "EDT", "L1_STFT", "L1_STFT_MultiRes", "Env")


def build_module(
    model_config: Dict[str, object], ckpt_path: str, device: str
) -> Tuple[pl.LightningModule, int, str]:
    """
    Load a trained FLAC checkpoint into an eval-ready training wrapper.

    Mirrors the checkpoint-loading logic of ``eval_FLAC.py`` (prefix stripping
    and EMA-weight remapping) so the diagnostic uses the exact same weights and
    inference path as the paper evaluation.

    Parameters
    ----------
    model_config : Dict[str, object]
        Parsed FLAC model config JSON.
    ckpt_path : str
        Path to the PyTorch-Lightning ``.ckpt`` file.
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    Tuple[pytorch_lightning.LightningModule, int, str]
        The eval-mode training-wrapper module, the number of latent frames the
        diffusion model samples, and the diffusion objective string.
    """
    training_config = model_config["training"]
    assert isinstance(training_config, dict)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "")] = state_dict.pop(key)

    use_ema = bool(training_config.get("use_ema", False))
    has_ema = any(k.startswith("diffusion_ema.ema_model.") for k in state_dict.keys())
    if use_ema and has_ema:
        print("Using EMA model")
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(key)
        training_config["use_ema"] = False

    model = create_model_from_config(model_config)
    model.load_state_dict(state_dict, strict=False)

    module = create_training_wrapper_from_config(model_config, model)
    module.eval().requires_grad_(False)
    module.to(device)

    pretransform = module.diffusion.pretransform
    sample_size = int(model_config["sample_size"])
    if pretransform is not None:
        samples = sample_size // pretransform.downsampling_ratio
    else:
        samples = sample_size

    diffusion_config = model_config["model"]["diffusion"]
    assert isinstance(diffusion_config, dict)
    objective = str(diffusion_config["diffusion_objective"])

    return module, samples, objective


def make_metric_callback(
    dataset_name: str, sample_rate: int, sample_size: int, device: str
) -> AcousticMetricsCallback:
    """
    Create an acoustic-only metric accumulator (no AGREE-based metrics).

    Parameters
    ----------
    dataset_name : str
        Dataset id, one of ``"AcousticRooms"`` or ``"HAA"``.
    sample_rate : int
        Audio sample rate in Hz (22050 for the supported datasets).
    sample_size : int
        Waveform length used by the metric callback.
    device : str
        Torch device string.

    Returns
    -------
    AcousticMetricsCallback
        A callback with T60, C50, EDT, L1-STFT, multires L1-STFT and Env enabled
        and FD / retrieval disabled.
    """
    return AcousticMetricsCallback(
        dataset_name=dataset_name,
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=1,
        eval_per_scene=False,
        device=device,
        eval_T60=True,
        eval_C50=True,
        eval_EDT=True,
        eval_l1_distance=True,
        eval_l1_distance_multires=True,
        eval_FD=False,
        eval_retrieval=False,
        eval_env=True,
        AGREE_ckpt=None,
    )


def run_model(
    module: pl.LightningModule,
    metadata: List[Dict[str, object]],
    noise: torch.Tensor,
    steps: int,
    cfg_scale: float,
    objective: str,
    device: str,
) -> torch.Tensor:
    """
    Run one forward inference pass (condition -> sample -> decode).

    Uses the same objective dispatch and (autocast around conditioning only)
    convention as ``eval_FLAC.py``. The provided ``noise`` is cloned so it can be
    reused across rotation angles without in-place mutation.

    Parameters
    ----------
    module : pytorch_lightning.LightningModule
        The eval-mode FLAC training wrapper from :func:`build_module`.
    metadata : List[Dict[str, object]]
        Per-sample metadata dicts for the batch.
    noise : torch.Tensor
        Fixed starting noise of shape ``[B, io_channels, samples]``.
    steps : int
        Number of sampling steps.
    cfg_scale : float
        Classifier-free guidance scale.
    objective : str
        One of ``"rectified_flow"``, ``"v"`` or ``"rf_denoiser"``.
    device : str
        Torch device string.

    Returns
    -------
    torch.Tensor
        Decoded, clamped waveform predictions of shape ``[B, 1, T]``.

    Raises
    ------
    ValueError
        If ``objective`` is not a recognised diffusion objective.
    """
    diffusion = module.diffusion
    model = diffusion.model
    x = noise.clone()

    device_type = torch.device(device).type
    with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        conditioning = diffusion.conditioner(metadata, module.device)
    cond_inputs = diffusion.get_conditioning_inputs(conditioning)

    if objective == "rectified_flow":
        fakes = sample_discrete_euler(
            model, x, steps, **cond_inputs, cfg_scale=cfg_scale,
            dist_shift=diffusion.dist_shift, batch_cfg=True, disable_tqdm=True,
        )
    elif objective == "v":
        fakes = sample(
            model, x, steps, 0, **cond_inputs, cfg_scale=cfg_scale,
            dist_shift=diffusion.dist_shift, batch_cfg=True,
        )
    elif objective == "rf_denoiser":
        logsnr = torch.linspace(-6, 2, steps + 1).to(module.device)
        sigmas = torch.sigmoid(-logsnr)
        sigmas[0] = 1.0
        sigmas[-1] = 0.0
        fakes = sample_flow_pingpong(
            model, x, sigmas=sigmas, **cond_inputs, cfg_scale=cfg_scale,
            dist_shift=diffusion.dist_shift, batch_cfg=True, disable_tqdm=True,
        )
    else:
        raise ValueError(f"Unknown diffusion objective: {objective}")

    if diffusion.pretransform is not None:
        fakes = diffusion.pretransform.decode(fakes)

    return fakes.clamp(-1.0, 1.0)


def waveform_gap(pred: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-sample waveform-domain divergence between two predictions.

    Parameters
    ----------
    pred : torch.Tensor
        Waveform batch of shape ``[B, 1, T]`` (rotated-condition prediction).
    ref : torch.Tensor
        Waveform batch of shape ``[B, 1, T]`` (original-condition prediction).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Per-sample mean absolute difference ``[B]`` and relative L2 distance
        ``[B]`` (``||pred - ref|| / ||ref||``), both computed over the common
        (minimum) length.
    """
    length = min(pred.shape[-1], ref.shape[-1])
    a = pred[..., :length].float()
    b = ref[..., :length].float()
    l1 = (a - b).abs().mean(dim=(-1, -2))
    rel_l2 = (a - b).pow(2).sum(dim=(-1, -2)).sqrt() / (b.pow(2).sum(dim=(-1, -2)).sqrt() + 1e-8)
    return l1, rel_l2


def energy_decay_db(ir: np.ndarray) -> np.ndarray:
    """
    Compute a normalized Schroeder energy decay curve in decibels.

    Parameters
    ----------
    ir : numpy.ndarray
        A 1D impulse response.

    Returns
    -------
    numpy.ndarray
        The energy decay curve in dB, normalized so the first sample is 0 dB.
    """
    energy = np.flip(np.cumsum(np.flip(ir.astype(np.float64) ** 2)))
    energy = energy / (energy[0] + 1e-12)
    return 10.0 * np.log10(energy + 1e-12)


def plot_sample_comparison(
    gt: np.ndarray,
    p0: np.ndarray,
    pa: Dict[int, np.ndarray],
    sample_rate: int,
    out_png: str,
) -> None:
    """
    Save a qualitative figure comparing GT, P0 and rotated predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        Ground-truth RIR (1D).
    p0 : numpy.ndarray
        Prediction on original conditioning (1D).
    pa : Dict[int, numpy.ndarray]
        Mapping from rotation angle in degrees to the rotated-condition
        prediction (1D).
    sample_rate : int
        Audio sample rate in Hz.
    out_png : str
        Output path for the PNG figure.

    Returns
    -------
    None
    """
    length = min([gt.shape[0], p0.shape[0]] + [v.shape[0] for v in pa.values()])
    gt, p0 = gt[:length], p0[:length]
    pa = {k: v[:length] for k, v in pa.items()}
    view = min(length, int(0.2 * sample_rate))
    t = np.arange(view) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    axes[0].plot(t, gt[:view], color="black", lw=1.0, label="GT")
    axes[0].plot(t, p0[:view], color="tab:blue", lw=1.0, alpha=0.8, label="P0 (orig)")
    for deg, wav_a in sorted(pa.items()):
        axes[0].plot(t, wav_a[:view], lw=0.9, alpha=0.7, label=f"P{deg}deg")
    axes[0].set_title("Waveform (first 200 ms)")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(energy_decay_db(gt), color="black", lw=1.2, label="GT")
    axes[1].plot(energy_decay_db(p0), color="tab:blue", lw=1.2, alpha=0.8, label="P0 (orig)")
    for deg, wav_a in sorted(pa.items()):
        axes[1].plot(energy_decay_db(wav_a), lw=1.0, alpha=0.7, label=f"P{deg}deg")
    axes[1].set_title("Energy decay curve (dB)")
    axes[1].set_xlabel("sample")
    axes[1].set_ylim(-80, 2)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _fmt(value: object) -> str:
    """
    Format a scalar metric value for tabular printing.

    Parameters
    ----------
    value : object
        A float, int, or torch scalar tensor (or None).

    Returns
    -------
    str
        A fixed-width string, ``"   n/a"`` if the value is None.
    """
    if value is None:
        return "   n/a"
    if isinstance(value, torch.Tensor):
        value = value.item()
    assert isinstance(value, (int, float))
    return f"{value:8.4f}"


def diagnose(
    model_config_path: str,
    dataset_config_path: str,
    ckpt_path: str,
    steps: int,
    cfg_scale: float,
    alphas_deg: List[float],
    batch_size: int,
    num_workers: int,
    max_batches: int,
    max_context: int,
    out_dir: str,
    dump_sample: bool,
    shuffle: bool,
    seed: int,
    device: str,
) -> None:
    """
    Run the Phase-1 yaw-invariance diagnostic and write results.

    Parameters
    ----------
    model_config_path : str
        Path to the FLAC model config JSON.
    dataset_config_path : str
        Path to the eval dataset config JSON.
    ckpt_path : str
        Path to the trained ``.ckpt`` file.
    steps : int
        Number of sampling steps.
    cfg_scale : float
        Classifier-free guidance scale.
    alphas_deg : List[float]
        Yaw rotation angles in degrees to evaluate (0 acts as a control).
    batch_size : int
        Eval batch size.
    num_workers : int
        Dataloader worker count.
    max_batches : int
        Maximum number of batches to process (0 means all).
    max_context : int
        Override for ``acoustic_context.max_context`` (0 keeps the config value).
    out_dir : str
        Directory to write the JSON / markdown / figure artifacts.
    dump_sample : bool
        Whether to save a qualitative figure for the first sample of batch 0.
    shuffle : bool
        Whether to shuffle the dataset before subsetting. Use True when running only
        a subset (``max_batches`` > 0), otherwise the scene-clustered eval JSON yields
        a non-representative single-scene subset.
    seed : int
        Random seed for reproducibility.
    device : str
        Torch device string.

    Returns
    -------
    None
    """
    torch.set_float32_matmul_precision("medium")
    os.makedirs(out_dir, exist_ok=True)

    with open(model_config_path) as f:
        model_config = json.load(f)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    if max_context > 0:
        dataset_config["modalities"]["acoustic_context"]["max_context"] = max_context

    dataset_id = dataset_config["datasets"][0]["id"]
    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

    module, samples, objective = build_module(model_config, ckpt_path, device)

    pl.seed_everything(seed, workers=True)
    eval_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=1,
        shuffle=shuffle,
    )

    angles: List[Tuple[float, float]] = [(deg, math.radians(deg)) for deg in alphas_deg]
    base_cb = make_metric_callback(dataset_id, sample_rate, sample_size, device)
    acc_cb = {deg: make_metric_callback(dataset_id, sample_rate, sample_size, device) for deg, _ in angles}
    gap_cb = {deg: make_metric_callback(dataset_id, sample_rate, sample_size, device) for deg, _ in angles}
    gap_l1 = {deg: 0.0 for deg, _ in angles}
    gap_rel = {deg: 0.0 for deg, _ in angles}
    n_seen = 0

    io_channels = int(module.diffusion.io_channels)

    for batch_idx, batch in enumerate(eval_dl):
        if 0 < max_batches <= batch_idx:
            break
        reals, metadata = batch
        reals = reals.to(device)
        scene_list = [md["scene"] for md in metadata]

        depth0 = metadata[0]["depth"]
        assert isinstance(depth0, torch.Tensor)
        img_w = int(depth0.shape[-1])

        if batch_idx == 0:
            consistency = yaw_transform_consistency(depth0, img_w, (90.0, 180.0))
            r_check = azimuth_rotation_matrix(math.radians(90.0)) @ torch.tensor([1.0, 0.0, 0.0])
            print("\n----- yaw transform self-check (max azimuth deviation in rad; expect ~0) -----")
            for deg, dev in sorted(consistency.items()):
                print(f"  rotate {deg:5.1f} deg -> max azimuth dev = {dev:.2e}")
            print(f"  R(+90 deg) @ [1,0,0] = {[round(v, 4) for v in r_check.tolist()]} (expect ~[0, 1, 0])")
            ok = all(dev < 1e-2 for dev in consistency.values())
            print(f"  self-check: {'PASS' if ok else 'FAIL -- roll/rotation sign mismatch!'}\n")

        noise = torch.randn([reals.shape[0], io_channels, samples], device=module.device)

        with torch.no_grad():
            p0 = run_model(module, metadata, noise, steps, cfg_scale, objective, device)
            base_cb.update_metrics("test", p0, reals, scene=scene_list)

            pa_for_plot: Dict[int, np.ndarray] = {}
            for deg, rad in angles:
                md_rot = [rotate_scene_metadata(md, rad, img_w) for md in metadata]
                pa = run_model(module, md_rot, noise, steps, cfg_scale, objective, device)

                acc_cb[deg].update_metrics("test", pa, reals, scene=scene_list)
                gap_cb[deg].update_metrics("test", pa, p0, scene=scene_list)

                l1, rel = waveform_gap(pa, p0)
                gap_l1[deg] += float(l1.sum().item())
                gap_rel[deg] += float(rel.sum().item())

                if dump_sample and batch_idx == 0:
                    pa_for_plot[int(round(deg))] = pa[0, 0].detach().cpu().numpy()

        n_seen += reals.shape[0]
        print(f"batch {batch_idx}: {n_seen} samples processed")

        if dump_sample and batch_idx == 0:
            fig_path = os.path.join(out_dir, "sample0_rotation_comparison.png")
            plot_sample_comparison(
                reals[0, 0].detach().cpu().numpy(),
                p0[0, 0].detach().cpu().numpy(),
                pa_for_plot,
                sample_rate,
                fig_path,
            )
            wav.write(os.path.join(out_dir, "sample0_gt.wav"), sample_rate, reals[0, 0].cpu().numpy())
            wav.write(os.path.join(out_dir, "sample0_p0.wav"), sample_rate, p0[0, 0].cpu().numpy())
            print(f"Saved qualitative figure to {fig_path}")

    base_metrics = base_cb.compute_metrics("test")
    results: Dict[str, object] = {
        "config": {
            "ckpt_path": ckpt_path,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "max_context": max_context,
            "n_samples": n_seen,
            "alphas_deg": alphas_deg,
        },
        "baseline_vs_gt": base_metrics,
        "per_angle": {},
    }

    header = "alpha |" + "|".join(f" {k:>16}" for k in GAP_METRIC_KEYS)
    print("\n===== Metric 1: invariance gap  metric(P_alpha, P0) =====")
    print("       waveform: rel-L2 and mean-|.|  (per-sample mean)")
    print(header)
    for deg, _ in angles:
        gap_metrics = gap_cb[deg].compute_metrics("test")
        row = f"{deg:5.0f} |" + "|".join(f" {_fmt(gap_metrics.get(k)):>16}" for k in GAP_METRIC_KEYS)
        rel = gap_rel[deg] / max(n_seen, 1)
        l1 = gap_l1[deg] / max(n_seen, 1)
        print(row + f"   | relL2={rel:.4f} l1={l1:.5f}")
        results_per_angle = {
            "gap_vs_p0": gap_metrics,
            "gap_waveform_relL2": rel,
            "gap_waveform_l1": l1,
        }
        assert isinstance(results["per_angle"], dict)
        results["per_angle"][str(deg)] = results_per_angle

    print("\n===== Metric 2: accuracy  metric(P_alpha, GT)  (baseline row = alpha 0 target) =====")
    print("baseline (P0 vs GT):", {k: _fmt(base_metrics.get(k)) for k in GAP_METRIC_KEYS})
    print(header)
    for deg, _ in angles:
        acc_metrics = acc_cb[deg].compute_metrics("test")
        row = f"{deg:5.0f} |" + "|".join(f" {_fmt(acc_metrics.get(k)):>16}" for k in GAP_METRIC_KEYS)
        print(row)
        assert isinstance(results["per_angle"], dict)
        angle_entry = results["per_angle"][str(deg)]
        assert isinstance(angle_entry, dict)
        angle_entry["acc_vs_gt"] = acc_metrics

    out_json = os.path.join(out_dir, "rotation_invariance_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if isinstance(o, torch.Tensor) else str(o))
    print(f"\nSaved results to {out_json}")


def main() -> None:
    """
    Parse command-line arguments and launch the diagnostic.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="FLAC yaw-rotation invariance diagnostic (Phase 1).")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--alphas-deg", type=str, default="0,45,90,135,180,225,270,315",
                        help="Comma-separated yaw angles in degrees.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means process the whole set.")
    parser.add_argument("--max-context", type=int, default=0, help="Override K (0 keeps config value).")
    parser.add_argument("--out-dir", type=str, default="outputs_FLAC/rotation_diag")
    parser.add_argument("--dump-sample", action="store_true")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle before subsetting; use when --max-batches > 0 to get a representative sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    alphas_deg = [float(x) for x in args.alphas_deg.split(",") if x.strip() != ""]

    diagnose(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        ckpt_path=args.ckpt_path,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        alphas_deg=alphas_deg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        max_context=args.max_context,
        out_dir=args.out_dir,
        dump_sample=args.dump_sample,
        shuffle=args.shuffle,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
