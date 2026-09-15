"""Matched-band waveform preprocessing and AGREE scoring for FEM diagnostics."""

from __future__ import annotations

import math

import numpy as np
import torch

from src.localization.engine import encode_audio_features
from src.localization.real_rir_oracle import deterministic_agree_seed
from src.localization.scoring import log_mean_exp_scores


def exact_bandlimit_waveforms(
    waveforms: torch.Tensor,
    bin_indices: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Keep only selected real-DFT bins without changing sample alignment."""

    values = torch.as_tensor(waveforms)
    if values.ndim != 3 or values.shape[1] != 1 or values.shape[-1] < 4:
        raise ValueError("waveforms must have shape [B, 1, T] with T >= 4")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("waveforms must be finite floating-point values")
    indices = torch.as_tensor(bin_indices, dtype=torch.long, device=values.device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("bin_indices must be a nonempty vector")
    if not torch.all(indices[1:] > indices[:-1]):
        raise ValueError("bin_indices must be strictly increasing")
    sample_count = values.shape[-1]
    if indices[0] <= 0 or indices[-1] >= sample_count // 2:
        raise ValueError("bin_indices must exclude DC and Nyquist")

    spectrum = torch.fft.rfft(values.double(), n=sample_count, dim=-1)
    selected = torch.zeros_like(spectrum)
    selected[..., indices] = spectrum[..., indices]
    return torch.fft.irfft(selected, n=sample_count, dim=-1).float()


def peak_normalize_waveforms(
    waveforms: torch.Tensor,
    *,
    target_peak: float = 0.95,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove arbitrary scalar gain using independent per-waveform peak scaling."""

    values = torch.as_tensor(waveforms)
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError("waveforms must have shape [B, 1, T]")
    if not torch.is_floating_point(values) or not torch.isfinite(values).all():
        raise ValueError("waveforms must be finite floating-point values")
    target_peak = float(target_peak)
    epsilon = float(epsilon)
    if not math.isfinite(target_peak) or not 0 < target_peak <= 1:
        raise ValueError("target_peak must lie in (0, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    peaks = values.abs().amax(dim=(-2, -1), keepdim=True)
    if torch.any(peaks <= epsilon):
        raise ValueError("every waveform must have nonzero peak amplitude")
    normalized = values * (target_peak / peaks)
    return normalized.float(), peaks[:, 0, 0].float()


@torch.inference_mode()
def score_fem_waveforms_with_agree(
    retrieval,
    candidate_waveforms: torch.Tensor,
    observed_waveform: torch.Tensor,
    *,
    query_index: int,
    score_seed: int,
    device: str | torch.device,
    candidate_batch_size: int = 32,
    sample_counts: tuple[int, ...] = (1, 4, 8),
    tau: float = 0.1,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, int]]:
    """Return nested fixed-seed AGREE cosine scores for deterministic FEM RIRs."""

    candidates = torch.as_tensor(candidate_waveforms, dtype=torch.float32)
    observed = torch.as_tensor(observed_waveform, dtype=torch.float32)
    if candidates.ndim != 3 or candidates.shape[1] != 1 or len(candidates) == 0:
        raise ValueError("candidate_waveforms must have shape [N, 1, T]")
    if observed.shape != (1, 1, candidates.shape[-1]):
        raise ValueError("observed_waveform must have shape [1, 1, T]")
    if not torch.isfinite(candidates).all() or not torch.isfinite(observed).all():
        raise ValueError("AGREE waveforms must be finite")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    counts = tuple(int(value) for value in sample_counts)
    if counts != tuple(sorted(set(counts))) or any(value <= 0 for value in counts):
        raise ValueError("sample_counts must be unique, increasing, and positive")
    maximum_samples = max(counts)

    observation_seed = deterministic_agree_seed(
        score_seed, int(query_index), "fem_agree_bandlimited_observation"
    )
    candidate_seed = deterministic_agree_seed(
        score_seed, int(query_index), "fem_agree_bandlimited_candidates"
    )
    torch.manual_seed(observation_seed)
    observation_feature = encode_audio_features(
        retrieval, observed.to(device=device, dtype=torch.float32)
    ).float()
    if observation_feature.shape[0] != 1:
        raise RuntimeError("AGREE must return exactly one observation feature")

    torch.manual_seed(candidate_seed)
    similarity_chunks = []
    for start in range(0, len(candidates), candidate_batch_size):
        chunk = candidates[start : start + candidate_batch_size]
        repeated = chunk.repeat_interleave(maximum_samples, dim=0)
        features = encode_audio_features(
            retrieval, repeated.to(device=device, dtype=torch.float32)
        ).float()
        similarities = (features @ observation_feature.T).reshape(
            len(chunk), maximum_samples
        )
        similarity_chunks.append(similarities.detach().cpu())
    all_similarities = torch.cat(similarity_chunks, dim=0)
    scores = {
        count: values.detach().cpu()
        for count, values in log_mean_exp_scores(
            all_similarities, counts, tau=tau
        ).items()
    }
    return scores, all_similarities, {
        "observation": observation_seed,
        "candidates": candidate_seed,
    }
