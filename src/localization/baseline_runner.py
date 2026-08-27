"""Deterministic adapters for Few-ShotRIR AGREE and FEM Room Helps selection."""

from __future__ import annotations

import numpy as np
import torch

from src.baselines.protocol import nested_context_prefix, prepare_and_score_waveforms
from src.baselines.room_helps_sparse import (
    RoomHelpsSparseResult,
    extract_rir_frequency_response,
    room_helps_pulse_omp,
)


def build_few_shot_candidate_batch(
    metadata: dict,
    candidates_global: np.ndarray,
    receiver_global: np.ndarray,
    *,
    context_count: int,
) -> dict[str, torch.Tensor]:
    """Translate one frozen FLAC query into an AR-adapted Few-ShotRIR batch."""

    required = ("depth", "context_audio", "context_poses")
    if any(key not in metadata for key in required):
        raise ValueError(f"metadata must contain {required}")
    geometry = torch.as_tensor(metadata["depth"], dtype=torch.float32)
    context_audio = torch.as_tensor(metadata["context_audio"], dtype=torch.float32)
    context_coordinates = torch.as_tensor(metadata["context_poses"], dtype=torch.float32)
    if geometry.ndim != 3 or geometry.shape[0] != 3:
        raise ValueError("FLAC geometry must have shape [3, H, W]")
    if context_audio.ndim != 3 or context_coordinates.ndim != 2:
        raise ValueError("FLAC context tensors have unexpected ranks")
    context_audio, context_coordinates = nested_context_prefix(
        context_audio.unsqueeze(0), context_coordinates.unsqueeze(0), context_count
    )
    candidates = np.asarray(candidates_global, dtype=np.float32)
    receiver = np.asarray(receiver_global, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or receiver.shape != (3,):
        raise ValueError("candidate and receiver coordinates must be [M,3] and [3]")
    if len(candidates) == 0 or not np.isfinite(candidates).all() or not np.isfinite(receiver).all():
        raise ValueError("candidate batch must be nonempty and finite")
    batch = len(candidates)
    return {
        "geometry": geometry.unsqueeze(0).expand(batch, -1, -1, -1),
        "context_audio": context_audio.expand(batch, -1, -1, -1),
        "context_coordinates": context_coordinates.expand(batch, -1, -1),
        "query_source": torch.from_numpy(candidates - receiver),
        "query_receiver": torch.zeros(batch, 3, dtype=torch.float32),
        "context_mask": torch.ones(batch, context_count, dtype=torch.bool),
    }


@torch.inference_mode()
def score_few_shot_candidates(
    model,
    retrieval,
    metadata: dict,
    candidates_global: np.ndarray,
    *,
    receiver_global: np.ndarray,
    observation_features: torch.Tensor,
    context_count: int,
    candidate_batch_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Generate deterministic waveform candidates and rank them with AGREE."""

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    candidates = np.asarray(candidates_global, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) == 0:
        raise ValueError("candidates must have shape [M, 3]")
    scores = []
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, len(candidates), candidate_batch_size):
            batch = build_few_shot_candidate_batch(
                metadata,
                candidates[start : start + candidate_batch_size],
                receiver_global,
                context_count=context_count,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            waveforms = model(**batch).float()
            scores.append(
                prepare_and_score_waveforms(
                    retrieval, waveforms, observation_features
                ).detach().cpu()
            )
    finally:
        model.train(was_training)
    return torch.cat(scores)


def score_fem_room_helps_candidates(
    response: torch.Tensor | np.ndarray,
    bin_indices: np.ndarray | torch.Tensor,
    observed_rir: torch.Tensor,
    *,
    sample_count: int,
) -> tuple[torch.Tensor, RoomHelpsSparseResult]:
    """Rank FEM atoms with Room Helps' stacked pulse-source complex OMP."""

    observed_response = extract_rir_frequency_response(
        observed_rir, bin_indices, sample_count=sample_count
    )
    recovery = room_helps_pulse_omp(
        response,
        observed_response,
        source_count=1,
    )
    scores = torch.from_numpy(recovery.first_step_scores.copy()).float()
    return scores, recovery
