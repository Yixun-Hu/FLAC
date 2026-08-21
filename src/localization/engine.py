"""Frozen FLAC/AGREE cache primitives and real localization inference."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio

from eval_FLAC import check_load_integrity
from src.configs.dataset_configs.custom_metadata.AR_md import convert_equirect_to_camera_coord
from src.data.yaw_rotation import (
    DEFAULT_FRAME_ANGLES,
    cylindrical_pose_features,
    rotate_scene_metadata,
)
from src.inference.sampling import sample_discrete_euler
from src.localization.geometry import (
    SURFACE_CLEARANCE_METERS,
    build_lattice,
    classify_mesh_candidates,
    filter_query_candidates,
    load_raycast_scene,
)
from src.metrics.modules.Retrieval import Retrieval
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config


SOURCE_CONDITIONING_IDS = ("source_vit", "source")
CONTEXT_CONDITIONING_IDS = (
    "context_poses_vit",
    "context_poses",
    "context_audio",
)
FA_CONTEXT_CONDITIONING_IDS = ("context_poses_vit", "context_audio")
FA_DYNAMIC_CONDITIONING_IDS = ("context_poses",)
ALL_CONDITIONING_IDS = (*SOURCE_CONDITIONING_IDS, *CONTEXT_CONDITIONING_IDS)
SCORE_SAMPLE_COUNTS = (1, 4, 8)


def _offline_dinov3_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "AGREE"
        / "AGREE"
        / "model_configs"
        / "dinov3_vits16_local"
    )


def _route_dinov3_to_strict_checkpoint_architecture(model_config: dict) -> None:
    """Instantiate DINOv3 offline; the strict FLAC load replaces every tensor."""
    local_path = _offline_dinov3_config_path()
    if not (local_path / "config.json").is_file():
        raise FileNotFoundError(local_path / "config.json")
    for item in model_config["model"]["conditioning"]["configs"]:
        if item["type"] == "ViTCoordinates" and "ViT" in item["config"]:
            item["config"]["ViT"]["hf_model_name_or_path"] = str(local_path)
            item["config"]["ViT"]["from_scratch"] = True


def cache_conditioning_branch(conditioner, metadata, ids, device):
    """Run exactly one named conditioner branch without mutating metadata."""
    return conditioner(metadata, device, only_ids=tuple(ids))


def cache_invariant_conditioning_branch(
    conditioner,
    metadata,
    ids,
    device,
    angles=DEFAULT_FRAME_ANGLES,
):
    """Cache one FA-BF branch with the released cylindrical+C4 semantics."""
    ids = tuple(ids)
    angles = tuple(float(angle) for angle in angles)
    if not angles or angles[0] != 0.0:
        raise ValueError("frame-average angles must start with the 0-degree identity")
    invariant_metadata = [cylindrical_pose_features(item) for item in metadata]
    base = conditioner(invariant_metadata, device, only_ids=ids)
    vit_ids = tuple(
        key for key in ("source_vit", "context_poses_vit") if key in ids and key in base
    )
    if not vit_ids or len(angles) == 1 or "depth" not in metadata[0]:
        return base
    image_width = int(metadata[0]["depth"].shape[-1])
    accum = {key: base[key][0].clone() for key in vit_ids}
    for angle in angles[1:]:
        variants = [
            rotate_scene_metadata(
                item,
                math.radians(angle),
                image_width,
                pose_keys=vit_ids,
            )
            for item in invariant_metadata
        ]
        part = conditioner(variants, device, only_ids=vit_ids)
        for key in vit_ids:
            accum[key] = accum[key] + part[key][0]
    for key in vit_ids:
        base[key][0] = accum[key] / float(len(angles))
    return base


def _expand_cached(value, batch_size: int):
    tensor, mask = value
    if tensor.shape[0] == batch_size:
        expanded_tensor = tensor
    elif tensor.shape[0] == 1:
        expanded_tensor = tensor.expand(batch_size, *tensor.shape[1:])
    else:
        raise ValueError("cached tensor batch does not match requested batch")
    if mask is None or mask.shape[0] == batch_size:
        expanded_mask = mask
    elif mask.shape[0] == 1:
        expanded_mask = mask.expand(batch_size, *mask.shape[1:])
    else:
        raise ValueError("cached mask batch does not match requested batch")
    return [expanded_tensor, expanded_mask]


def merge_cached_conditioning(
    source_branch,
    context_branch,
    batch_size: int,
    dynamic_branch=None,
):
    """Compose receiver-candidate source tokens with one query context cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if set(source_branch) != set(SOURCE_CONDITIONING_IDS):
        raise ValueError("source cache has unexpected conditioner ids")
    dynamic_branch = {} if dynamic_branch is None else dynamic_branch
    combined_keys = set(source_branch) | set(context_branch) | set(dynamic_branch)
    if combined_keys != set(ALL_CONDITIONING_IDS):
        raise ValueError("conditioning branches do not cover the frozen conditioner ids")
    if (set(source_branch) & set(context_branch)) or (
        (set(source_branch) | set(context_branch)) & set(dynamic_branch)
    ):
        raise ValueError("conditioning branches must be disjoint")
    output = {}
    for key in ALL_CONDITIONING_IDS:
        if key in source_branch:
            branch = source_branch
        elif key in context_branch:
            branch = context_branch
        else:
            branch = dynamic_branch
        output[key] = _expand_cached(branch[key], batch_size)
    return output


def encode_audio_features(retrieval: Retrieval, waveforms: torch.Tensor) -> torch.Tensor:
    """Delegate preprocessing/encoding to FLAC's reviewed Retrieval helper."""
    features = retrieval.compute_audio_features(waveforms)
    return torch.atleast_2d(features)


def prepare_generated_audio(waveforms: torch.Tensor, sample_size: int = 10240) -> torch.Tensor:
    if waveforms.dtype != torch.float32:
        raise ValueError("generated waveform must be float32")
    if waveforms.ndim != 3 or waveforms.shape[1] != 1:
        raise ValueError("generated waveform must have shape [B, 1, T]")
    if waveforms.shape[-1] != sample_size:
        raise ValueError(f"generated waveform must have exactly {sample_size} samples")
    return waveforms.clamp(-1.0, 1.0)


def candidate_seed(base_seed: int, query_index: int, candidate_index: int, sample_index: int) -> int:
    payload = f"{int(base_seed)}:{int(query_index)}:{int(candidate_index)}:{int(sample_index)}"
    value = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "little")
    return value % (2**63 - 1)


def deterministic_noise(
    seeds: Sequence[int],
    sample_shape: Sequence[int],
    *,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate each row from its own seed so batching cannot change samples."""
    rows = []
    generator_device = torch.device(device).type
    for seed in seeds:
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))
        rows.append(
            torch.randn((1, *sample_shape), generator=generator, device=device, dtype=dtype)
        )
    if not rows:
        return torch.empty((0, *sample_shape), device=device, dtype=dtype)
    return torch.cat(rows, dim=0)


def project_runtime_seconds(
    *,
    query_count: int,
    receiver_candidate_count: int,
    query_candidate_count: int,
    query_io_seconds_per_query: float,
    context_seconds_per_query: float,
    observation_seconds_per_query: float,
    source_candidates_per_second: float,
    generated_scores_per_second: float,
    score_samples: int,
) -> dict[str, float]:
    if min(query_count, receiver_candidate_count, query_candidate_count, score_samples) <= 0:
        raise ValueError("projection counts must be positive")
    if min(
        context_seconds_per_query,
        query_io_seconds_per_query,
        observation_seconds_per_query,
        source_candidates_per_second,
        generated_scores_per_second,
    ) <= 0:
        raise ValueError("projection timings/rates must be positive")
    parts = {
        "query_io_seconds": query_count * query_io_seconds_per_query,
        "context_cache_seconds": query_count * context_seconds_per_query,
        "observation_seconds": query_count * observation_seconds_per_query,
        "source_cache_seconds": receiver_candidate_count / source_candidates_per_second,
        "generation_score_seconds": (
            query_candidate_count * score_samples / generated_scores_per_second
        ),
    }
    parts["total_seconds"] = sum(parts.values())
    return parts


def load_flac_module(model_config_path: Path | str, ckpt_path: Path | str, device: str):
    """Load a clean frozen checkpoint with eval_FLAC's integrity rules."""
    model_config = json.loads(Path(model_config_path).read_text())
    model_config = copy.deepcopy(model_config)
    state_bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = dict(state_bundle["state_dict"])
    for key in list(state_dict):
        if key.startswith("diffusion."):
            state_dict[key.removeprefix("diffusion.")] = state_dict.pop(key)
    training = copy.deepcopy(model_config["training"])
    if training.get("use_ema", False) and any(
        key.startswith("diffusion_ema.ema_model.") for key in state_dict
    ):
        for key in list(state_dict):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(
                    key
                )
    # The chosen state is frozen for inference; constructing a second unused EMA
    # copy changes memory only and is deliberately disabled.
    training["use_ema"] = False
    model_config["training"] = training
    _route_dinov3_to_strict_checkpoint_architecture(model_config)
    model = create_model_from_config(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected, allow_partial_load=False)
    module = create_training_wrapper_from_config(model_config, model)
    module.eval().requires_grad_(False).to(device)
    return module, model_config


def load_agree_retrieval(ckpt_path: Path | str, device: str) -> Retrieval:
    from AGREE.AGREE.factory import get_model_config
    from AGREE.AGREE.model import AGREE

    config = get_model_config("dinoV3")
    vision = dict(config["vision_cfg"])
    vision["hf_model_name"] = str(_offline_dinov3_config_path())
    vision["from_scratch"] = True
    config["vision_cfg"] = vision
    # The full AGREE checkpoint contains the complete audio tower.  Avoid the
    # training-time relative VAE bootstrap path, then require the full load to
    # replace every randomly initialized tensor below.
    audio = dict(config["audio_cfg"])
    audio["pretrained"] = None
    config["audio_cfg"] = audio
    agree = AGREE(**config)
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    agree.load_state_dict(bundle["state_dict"], strict=True)
    agree.eval().requires_grad_(False).to(device)
    return Retrieval(AGREE=agree).eval().requires_grad_(False).to(device)


def _pad_crop_audio(path: Path, samples: int, *, clamp: bool) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    if sample_rate != 22050:
        raise ValueError(f"unexpected sample rate {sample_rate}: {path}")
    if audio.shape[0] != 1:
        audio = audio.mean(dim=0, keepdim=True)
    output = torch.zeros((1, samples), dtype=torch.float32)
    output[:, : min(samples, audio.shape[-1])] = audio[:, :samples].float()
    return output.clamp(-1.0, 1.0) if clamp else output


def load_frozen_query(record: dict, dataset_root: Path | str) -> tuple[torch.Tensor, dict]:
    """Load one manifest query without consuming any random context selection."""
    dataset_root = Path(dataset_root)
    receiver = np.asarray(record["receiver_global"], dtype=np.float32)
    source = np.asarray(record["source_global"], dtype=np.float32)
    context_sources = np.asarray(record["context_sources_global"], dtype=np.float32)
    target_audio = _pad_crop_audio(dataset_root / record["query_id"], 10240, clamp=True)
    context_audio = torch.stack(
        [_pad_crop_audio(dataset_root / relpath, 9600, clamp=False) for relpath in record["contexts"]]
    )
    receiver_id = int(record["filename"].split("_")[1][1:])
    depth_path = (
        dataset_root
        / "depth_map"
        / record["scene"]
        / record["room"]
        / f"{receiver_id}.npy"
    )
    depth_map = torch.from_numpy(np.load(depth_path))
    depth = convert_equirect_to_camera_coord(depth_map, 256, 512).permute(2, 0, 1).float()
    relative_source = torch.from_numpy(source - receiver)
    relative_context = torch.from_numpy(context_sources - receiver)
    metadata = {
        "scene": record["scene"],
        "source": relative_source,
        "source_vit": relative_source.unsqueeze(0),
        "context_poses": relative_context,
        "context_poses_vit": relative_context,
        "context_audio": context_audio,
        "depth": depth,
    }
    return target_audio, metadata


def candidate_metadata(metadata: dict, candidates_global, receiver_global) -> list[dict]:
    receiver = np.asarray(receiver_global, dtype=np.float32)
    output = []
    for candidate in np.asarray(candidates_global, dtype=np.float32):
        relative = torch.from_numpy(candidate - receiver)
        item = dict(metadata)
        item["source"] = relative
        item["source_vit"] = relative.unsqueeze(0)
        output.append(item)
    return output


def reconstruct_room_base_candidates(room_name: str, audit: dict) -> np.ndarray:
    """Rebuild and hash-check one room's mesh-valid global base lattice."""

    room = audit["rooms"][room_name]
    mesh = load_raycast_scene(room["mesh_path"], compute_topology=False)
    raw = build_lattice(mesh.aabb_min, mesh.aabb_max, audit["grid_spacing_m"])
    base_mask, _distance = classify_mesh_candidates(
        mesh,
        raw,
        audit.get("surface_clearance_m", SURFACE_CLEARANCE_METERS),
        eps=audit["epsilon_m"],
    )
    base = raw[base_mask]
    expected_hash = hashlib.sha256(base.astype("<f8").tobytes()).hexdigest()
    if expected_hash != room["base_points_sha256"]:
        raise RuntimeError("reconstructed base candidate hash does not match geometry audit")
    return base


def filter_frozen_query_candidates(record: dict, audit: dict, base: np.ndarray) -> np.ndarray:
    """Apply and verify the frozen query mask to a checked room base grid."""

    contexts = np.asarray(record["context_sources_global"], dtype=np.float64)
    z_band = (float(contexts[:, 2].min() - 0.5), float(contexts[:, 2].max() + 0.5))
    mask = filter_query_candidates(
        base,
        record["receiver_global"],
        contexts,
        receiver_clearance=audit["receiver_clearance_m"],
        context_clearance=audit["context_clearance_m"],
        z_band=z_band if audit["z_branch"] == "z_band" else None,
        eps=audit["epsilon_m"],
    )
    query_rows = [
        item
        for item in audit["queries"]
        if int(item["index"]) == int(record["index"])
    ]
    if len(query_rows) != 1 or query_rows[0]["query_id"] != record["query_id"]:
        raise RuntimeError("query identity does not match geometry audit")
    query = query_rows[0]
    expected_indices_hash = query[
        "z_indices_sha256" if audit["z_branch"] == "z_band" else "full_indices_sha256"
    ]
    actual_indices_hash = hashlib.sha256(
        np.flatnonzero(mask).astype("<u4", copy=False).tobytes()
    ).hexdigest()
    if actual_indices_hash != expected_indices_hash:
        raise RuntimeError("reconstructed query mask hash does not match geometry audit")
    candidates = base[mask]
    if len(candidates) != int(query["chosen_count"]):
        raise RuntimeError("reconstructed query count does not match geometry audit")
    return candidates


def reconstruct_query_candidates(record: dict, audit: dict) -> np.ndarray:
    base = reconstruct_room_base_candidates(record["room"], audit)
    return filter_frozen_query_candidates(record, audit, base)


@torch.inference_mode()
def generate_and_score_batch(
    module,
    retrieval: Retrieval,
    source_branch: dict,
    context_branch: dict,
    observation_features: torch.Tensor,
    seeds: Sequence[int],
    dynamic_branch: dict | None = None,
    *,
    steps: int = 1,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    diffusion = module.diffusion
    if diffusion.diffusion_objective != "rectified_flow":
        raise ValueError("exp_09 probe is pinned to the rectified-flow checkpoint")
    batch_size = len(seeds)
    conditioning = merge_cached_conditioning(
        source_branch,
        context_branch,
        batch_size,
        dynamic_branch=dynamic_branch,
    )
    inputs = diffusion.get_conditioning_inputs(conditioning)
    latent_samples = 10240 // diffusion.pretransform.downsampling_ratio
    dtype = next(diffusion.model.parameters()).dtype
    noise = deterministic_noise(
        seeds,
        (diffusion.io_channels, latent_samples),
        device=module.device,
        dtype=dtype,
    )
    fakes = sample_discrete_euler(
        diffusion.model,
        noise,
        steps,
        **inputs,
        cfg_scale=cfg_scale,
        dist_shift=diffusion.dist_shift,
        batch_cfg=True,
        disable_tqdm=True,
    )
    fakes = diffusion.pretransform.decode(fakes).float()
    fakes = prepare_generated_audio(fakes, sample_size=10240)
    generated_features = encode_audio_features(retrieval, fakes)
    return generated_features @ observation_features.T
