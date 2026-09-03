"""Localization adapter for the official-architecture AcousticRooms FewshotRiR."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.protocol import prepare_and_score_waveforms
from src.data.fewshot_rir import (
    load_ar_depth,
    load_rir_waveform,
    parse_rir_filename,
    rir_magnitude_spectrogram,
)


NEAR_CONTEXT_PROTOCOL = "fewshot_rir_near_coincident_v1"


def load_fewshot_rir_query(
    record: dict,
    dataset_root: Path | str,
    *,
    sample_rate: int = 22050,
    sample_size: int = 10240,
    n_fft: int = 511,
    hop_length: int = 40,
    win_length: int = 248,
    depth_size: tuple[int, int] = (128, 256),
    depth_max_m: float = 67.16327,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | np.ndarray]]:
    """Load one frozen query with near-coincident AR contexts and 6-D poses."""

    if record.get("context_protocol") != NEAR_CONTEXT_PROTOCOL:
        raise ValueError(
            "FewshotRiR requires a near-coincident context manifest; rebuild the manifest, geometry audit, and pilot"
        )
    contexts = record.get("contexts")
    context_sources = np.asarray(record.get("context_sources_global"), dtype=np.float32)
    context_receivers = np.asarray(record.get("context_receivers_global"), dtype=np.float32)
    recorded_distances = np.asarray(
        record.get("context_endpoint_distances_m"), dtype=np.float32
    )
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("FewshotRiR context manifest contains no context paths")
    expected_shape = (len(contexts), 3)
    if context_sources.shape != expected_shape or context_receivers.shape != expected_shape:
        raise ValueError("FewshotRiR context coordinates have incompatible shapes")
    if not np.isfinite(context_sources).all() or not np.isfinite(context_receivers).all():
        raise ValueError("FewshotRiR context coordinates must be finite")
    if recorded_distances.shape != (len(contexts),) or not np.isfinite(
        recorded_distances
    ).all():
        raise ValueError("FewshotRiR context endpoint distances are invalid")
    actual_distances = np.linalg.norm(context_sources - context_receivers, axis=1)
    if not np.allclose(actual_distances, recorded_distances, atol=1e-5, rtol=1e-6):
        raise ValueError("FewshotRiR context endpoint-distance audit does not match")

    dataset_root = Path(dataset_root)
    observed = load_rir_waveform(
        dataset_root / record["query_id"],
        sample_rate=sample_rate,
        sample_size=sample_size,
    ).unsqueeze(0)
    context_waveforms = torch.stack(
        [
            load_rir_waveform(
                dataset_root / relpath,
                sample_rate=sample_rate,
                sample_size=sample_size,
            )
            for relpath in contexts
        ]
    )
    anchor = context_receivers[0]
    context_poses = torch.from_numpy(
        np.concatenate(
            (context_receivers - anchor, context_sources - anchor), axis=1
        ).astype(np.float32)
    )
    context_depth = torch.stack(
        [
            load_ar_depth(
                dataset_root,
                record["scene"],
                record["room"],
                parse_rir_filename(relpath)[1],
                depth_size=depth_size,
                depth_max_m=depth_max_m,
            )
            for relpath in contexts
        ]
    )
    metadata: dict[str, torch.Tensor | np.ndarray] = {
        "context_depth": context_depth,
        "context_magnitude": rir_magnitude_spectrogram(
            context_waveforms,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        ),
        "context_poses": context_poses,
        "anchor_global": anchor,
    }
    return observed, metadata


def build_fewshot_rir_candidate_batch(
    metadata: dict[str, torch.Tensor | np.ndarray],
    candidates_global: np.ndarray,
    receiver_global: np.ndarray,
    *,
    context_count: int,
) -> dict[str, torch.Tensor]:
    context_depth = torch.as_tensor(metadata["context_depth"], dtype=torch.float32)
    context_magnitude = torch.as_tensor(metadata["context_magnitude"], dtype=torch.float32)
    context_poses = torch.as_tensor(metadata["context_poses"], dtype=torch.float32)
    anchor = np.asarray(metadata["anchor_global"], dtype=np.float32)
    candidates = np.asarray(candidates_global, dtype=np.float32)
    receiver = np.asarray(receiver_global, dtype=np.float32)
    context_count = int(context_count)
    if context_count <= 0 or context_count > context_depth.shape[0]:
        raise ValueError("context_count is outside the frozen near-context prefix")
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) == 0:
        raise ValueError("candidates must have shape [M, 3]")
    if receiver.shape != (3,) or anchor.shape != (3,):
        raise ValueError("receiver and anchor coordinates must have shape [3]")
    if not np.isfinite(candidates).all() or not np.isfinite(receiver).all():
        raise ValueError("candidate coordinates must be finite")
    query_receivers = np.repeat((receiver - anchor)[None], len(candidates), axis=0)
    query_sources = candidates - anchor
    return {
        "context_depth": context_depth[None, :context_count],
        "context_spectrograms": context_magnitude[None, :context_count],
        "context_poses": context_poses[None, :context_count],
        "query_poses": torch.from_numpy(
            np.concatenate((query_receivers, query_sources), axis=1).astype(np.float32)
        ).unsqueeze(0),
        "context_mask": torch.ones(1, context_count, dtype=torch.bool),
        "query_mask": torch.ones(1, len(candidates), dtype=torch.bool),
    }


@torch.inference_mode()
def score_fewshot_rir_candidates(
    model,
    retrieval,
    metadata: dict[str, torch.Tensor | np.ndarray],
    candidates_global: np.ndarray,
    *,
    receiver_global: np.ndarray,
    observation_features: torch.Tensor,
    context_count: int,
    candidate_batch_size: int,
    device: str | torch.device,
    griffin_lim_iterations: int = 32,
    griffin_lim_momentum: float = 0.99,
    core_timing: dict[str, float] | None = None,
    synchronize_timing: bool = False,
) -> torch.Tensor:
    """Predict magnitudes, reconstruct deterministic waveforms, then rank with AGREE."""

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    candidates = np.asarray(candidates_global, dtype=np.float32)
    scores = []
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, len(candidates), candidate_batch_size):
            if core_timing is not None and synchronize_timing and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            generation_started = time.perf_counter()
            batch = build_fewshot_rir_candidate_batch(
                metadata,
                candidates[start : start + candidate_batch_size],
                receiver_global,
                context_count=context_count,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            raw = model(**batch)
            magnitude = model.output_to_magnitude(raw)
            waveforms = model.magnitude_to_waveform(
                magnitude,
                iterations=griffin_lim_iterations,
                momentum=griffin_lim_momentum,
            ).squeeze(0)
            if not torch.isfinite(waveforms).all():
                raise RuntimeError("FewshotRiR Griffin-Lim produced non-finite waveforms")
            if core_timing is not None:
                if synchronize_timing and torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                core_timing["candidate_conditioning_generation_and_griffin_lim"] = (
                    core_timing.get("candidate_conditioning_generation_and_griffin_lim", 0.0)
                    + time.perf_counter()
                    - generation_started
                )
            scores.append(
                prepare_and_score_waveforms(
                    retrieval,
                    waveforms.float(),
                    observation_features,
                ).detach().cpu()
            )
    finally:
        model.train(was_training)
    return torch.cat(scores)
