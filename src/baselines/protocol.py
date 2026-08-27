"""Shared input/output contract for deterministic localization baselines."""

from __future__ import annotations

import torch

from src.localization.engine import encode_audio_features, prepare_generated_audio


TARGET_SAMPLES = 10240
CONTEXT_SAMPLES = 9600
SAMPLE_RATE = 22050


def nested_context_prefix(
    context_audio: torch.Tensor,
    context_coordinates: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an ordered context prefix without copying or resampling it."""

    if context_audio.ndim != 4 or context_audio.shape[2] != 1:
        raise ValueError("context audio must have shape [B, K, 1, T]")
    if context_coordinates.ndim != 3 or context_coordinates.shape[-1] != 3:
        raise ValueError("context coordinates must have shape [B, K, 3]")
    if context_audio.shape[:2] != context_coordinates.shape[:2]:
        raise ValueError("audio and coordinate context axes must match")
    count = int(count)
    if count <= 0 or count > context_audio.shape[1]:
        raise ValueError("context count is outside the available ordered set")
    return context_audio[:, :count], context_coordinates[:, :count]


def validate_baseline_waveforms(waveforms: torch.Tensor) -> torch.Tensor:
    """Apply the reviewed FLAC waveform contract to one baseline output."""

    if not torch.is_tensor(waveforms) or not torch.isfinite(waveforms).all():
        raise ValueError("generated waveform must be a finite tensor")
    return prepare_generated_audio(waveforms, sample_size=TARGET_SAMPLES)


@torch.inference_mode()
def prepare_and_score_waveforms(
    retrieval,
    waveforms: torch.Tensor,
    observation_features: torch.Tensor,
) -> torch.Tensor:
    """Score deterministic candidate waveforms through the frozen FLAC path."""

    prepared = validate_baseline_waveforms(waveforms)
    observation = torch.as_tensor(observation_features, dtype=torch.float32)
    if observation.ndim != 2 or observation.shape[0] != 1:
        raise ValueError("exactly one observation embedding is required")
    if not torch.isfinite(observation).all():
        raise ValueError("observation embedding must be finite")
    norm = torch.linalg.vector_norm(observation, dim=-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5):
        raise ValueError("observation embedding must be normalized")
    generated = encode_audio_features(retrieval, prepared).float()
    if generated.shape[1] != observation.shape[1]:
        raise ValueError("generated and observed embedding widths differ")
    return (generated @ observation.to(generated.device).T).squeeze(1)
