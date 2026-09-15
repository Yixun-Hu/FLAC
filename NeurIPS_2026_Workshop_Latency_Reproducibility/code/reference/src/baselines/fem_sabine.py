"""Sabine boundary estimation and FEM frequency-response waveform utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyroomacoustics
import torch

from .protocol import SAMPLE_RATE, TARGET_SAMPLES


AR_RT60_DECAY_DB = 20
SABINE_CONSTANT_SI = 0.161


@dataclass(frozen=True)
class ContextT60Estimate:
    t60_seconds: float
    valid_values: tuple[float, ...]
    valid_count: int
    invalid_count: int


@dataclass(frozen=True)
class SabineBoundary:
    volume_m3: float
    surface_area_m2: float
    t60_seconds: float
    raw_absorption: float
    absorption: float
    reflection_magnitude: float
    normalized_impedance: float
    was_clipped: bool


def estimate_context_t60(
    context_waveforms: torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    decay_db: int = AR_RT60_DECAY_DB,
) -> ContextT60Estimate:
    """Estimate room T60 as the median of valid FLAC-parity context estimates."""

    if context_waveforms.ndim != 3 or context_waveforms.shape[1] != 1:
        raise ValueError("context waveforms must have shape [K, 1, T]")
    if context_waveforms.shape[0] == 0 or not torch.isfinite(context_waveforms).all():
        raise ValueError("at least one finite context waveform is required")
    if sample_rate <= 0 or decay_db <= 0:
        raise ValueError("sample_rate and decay_db must be positive")
    valid: list[float] = []
    invalid_count = 0
    for waveform in context_waveforms[:, 0].detach().cpu().numpy():
        try:
            value = float(
                pyroomacoustics.experimental.measure_rt60(
                    waveform, fs=int(sample_rate), decay_db=int(decay_db)
                )
            )
        except (ValueError, FloatingPointError):
            invalid_count += 1
            continue
        if not math.isfinite(value) or value <= 0:
            invalid_count += 1
            continue
        valid.append(value)
    if not valid:
        raise ValueError("no valid context T60 estimate")
    return ContextT60Estimate(
        t60_seconds=float(np.median(valid)),
        valid_values=tuple(valid),
        valid_count=len(valid),
        invalid_count=invalid_count,
    )


def sabine_boundary(
    volume_m3: float,
    surface_area_m2: float,
    t60_seconds: float,
    *,
    absorption_limits: tuple[float, float] = (1e-4, 0.99),
) -> SabineBoundary:
    """Convert one room T60 into a uniform, zero-phase Sabine impedance."""

    values = tuple(map(float, (volume_m3, surface_area_m2, t60_seconds)))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("volume, surface area, and T60 must be finite and positive")
    lower, upper = map(float, absorption_limits)
    if not (0 < lower < upper < 1):
        raise ValueError("absorption limits must satisfy 0 < lower < upper < 1")
    raw = SABINE_CONSTANT_SI * values[0] / (values[1] * values[2])
    absorption = min(max(raw, lower), upper)
    reflection = math.sqrt(1.0 - absorption)
    impedance = (1.0 + reflection) / (1.0 - reflection)
    return SabineBoundary(
        volume_m3=values[0],
        surface_area_m2=values[1],
        t60_seconds=values[2],
        raw_absorption=raw,
        absorption=absorption,
        reflection_magnitude=reflection,
        normalized_impedance=impedance,
        was_clipped=not math.isclose(raw, absorption, rel_tol=0.0, abs_tol=1e-15),
    )


def dft_frequency_bins(
    sample_rate: int = SAMPLE_RATE,
    sample_count: int = TARGET_SAMPLES,
    minimum_hz: float = 80.0,
    maximum_hz: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact positive real-DFT bins contained in a closed frequency band."""

    if sample_rate <= 0 or sample_count <= 0:
        raise ValueError("sample_rate and sample_count must be positive")
    if not (math.isfinite(minimum_hz) and math.isfinite(maximum_hz)):
        raise ValueError("frequency limits must be finite")
    nyquist = sample_rate / 2.0
    if minimum_hz <= 0 or maximum_hz < minimum_hz or maximum_hz >= nyquist:
        raise ValueError("frequency band must lie strictly between DC and Nyquist")
    spacing = sample_rate / sample_count
    first = math.ceil(minimum_hz / spacing)
    last = math.floor(maximum_hz / spacing)
    if first > last:
        raise ValueError("frequency band contains no DFT bins")
    indices = np.arange(first, last + 1, dtype=np.int64)
    return indices, indices.astype(np.float64) * spacing


def bandlimited_response_to_waveform(
    response: torch.Tensor,
    bin_indices: np.ndarray | torch.Tensor,
    *,
    sample_count: int = TARGET_SAMPLES,
    unit_gain: float = 1.0,
) -> torch.Tensor:
    """Place complex FEM bins in a real spectrum and apply a fixed inverse DFT."""

    if response.ndim != 2 or not torch.is_complex(response):
        raise ValueError("response must be complex with shape [candidate, frequency]")
    if not torch.isfinite(response.real).all() or not torch.isfinite(response.imag).all():
        raise ValueError("frequency response must be finite")
    indices = torch.as_tensor(bin_indices, dtype=torch.long, device=response.device)
    if indices.ndim != 1 or indices.numel() != response.shape[1]:
        raise ValueError("one DFT bin index is required per response frequency")
    if indices.numel() == 0 or not torch.all(indices[1:] > indices[:-1]):
        raise ValueError("DFT bin indices must be nonempty and strictly increasing")
    if indices[0] <= 0 or indices[-1] >= sample_count // 2:
        raise ValueError("selected bins must exclude DC and Nyquist")
    unit_gain = float(unit_gain)
    if not math.isfinite(unit_gain) or unit_gain <= 0:
        raise ValueError("unit_gain must be finite and positive")
    spectrum = torch.zeros(
        response.shape[0],
        sample_count // 2 + 1,
        dtype=torch.complex64,
        device=response.device,
    )
    spectrum[:, indices] = response.to(torch.complex64) * unit_gain
    return torch.fft.irfft(spectrum, n=sample_count, dim=-1).unsqueeze(1).float()
