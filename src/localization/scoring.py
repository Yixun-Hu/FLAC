"""Stable candidate scoring and localization metrics for exp_09."""

from __future__ import annotations

import math

import numpy as np
import torch


def log_mean_exp_scores(
    similarities: torch.Tensor,
    sample_counts=(1, 4, 8),
    *,
    tau: float = 0.1,
) -> dict[int, torch.Tensor]:
    """Aggregate nested stochastic samples for every candidate."""

    if similarities.ndim != 2 or similarities.shape[0] == 0:
        raise ValueError("similarities must have shape [candidate, sample]")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not torch.isfinite(similarities).all():
        raise ValueError("similarities must be finite")
    counts = tuple(int(value) for value in sample_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("sample counts must be positive")
    if max(counts) > similarities.shape[1]:
        raise ValueError("sample count exceeds available stochastic samples")
    values = similarities.float()
    return {
        count: tau
        * (torch.logsumexp(values[:, :count] / tau, dim=1) - math.log(count))
        for count in counts
    }


def stable_argmax(scores: torch.Tensor | np.ndarray) -> int:
    values = torch.as_tensor(scores)
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("scores must be one-dimensional, nonempty, and finite")
    return int(torch.argmax(values).item())


def localization_metrics(candidates, truth, prediction_index: int) -> dict[str, float | int]:
    candidates = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) == 0:
        raise ValueError("candidates must have shape [M, 3]")
    if truth.shape != (3,) or not np.isfinite(candidates).all() or not np.isfinite(truth).all():
        raise ValueError("coordinates must be finite three-dimensional values")
    if prediction_index < 0 or prediction_index >= len(candidates):
        raise ValueError("prediction index is outside the candidate set")
    distances = np.linalg.norm(candidates - truth[None, :], axis=1)
    location_error = float(distances[prediction_index])
    oracle_error = float(distances.min())
    excess_error = max(0.0, location_error - oracle_error)
    return {
        "prediction_index": int(prediction_index),
        "localization_error_m": location_error,
        "oracle_error_m": oracle_error,
        "excess_error_m": excess_error,
        "success_0_5m": int(location_error <= 0.5),
        "success_1_0m": int(location_error <= 1.0),
        "oracle_normalized_success_0_5m": int(excess_error <= 0.5),
        "oracle_normalized_success_1_0m": int(excess_error <= 1.0),
    }


def deterministic_random_candidate(query_index: int, candidate_count: int, seed: int = 42) -> int:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    sequence = np.random.SeedSequence([int(seed), int(query_index), 0x52414E44])
    return int(np.random.default_rng(sequence).integers(candidate_count))
