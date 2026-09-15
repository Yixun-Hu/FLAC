"""Room Helps pulse-source sparse recovery over a multifrequency FEM dictionary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RoomHelpsSparseResult:
    support: tuple[int, ...]
    coefficients: np.ndarray
    first_step_scores: np.ndarray
    relative_residual_norm: float


def _complex_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.complex128)


def extract_rir_frequency_response(
    waveform: torch.Tensor,
    bin_indices: np.ndarray | torch.Tensor,
    *,
    sample_count: int,
) -> np.ndarray:
    """Read exact complex DFT bins from one beginning-aligned observed RIR."""

    value = torch.as_tensor(waveform)
    while value.ndim > 1 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1 or value.numel() != sample_count:
        raise ValueError("observed RIR must contain exactly one waveform of sample_count")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError("observed RIR must be a finite real waveform")
    indices = torch.as_tensor(bin_indices, dtype=torch.long, device=value.device)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("frequency bin indices must be a nonempty vector")
    if not torch.all(indices[1:] > indices[:-1]):
        raise ValueError("frequency bin indices must be strictly increasing")
    if indices[0] <= 0 or indices[-1] >= sample_count // 2:
        raise ValueError("frequency bins must exclude DC and Nyquist")
    spectrum = torch.fft.rfft(value.double(), n=sample_count)
    return spectrum[indices].detach().cpu().numpy().astype(np.complex128, copy=False)


def room_helps_pulse_omp(
    candidate_frequency_responses,
    observed_frequency_response,
    *,
    source_count: int = 1,
    epsilon: float = 1e-12,
) -> RoomHelpsSparseResult:
    """Recover a spatially sparse pulse source from vertically stacked frequencies.

    Candidate input has shape ``[candidate, frequency]``.  Because an RIR is the
    response to a unit impulse, the spatial source coefficient is shared across
    frequencies as in Room Helps Eq. (14); complex OMP is applied to the stacked
    system.  The first-step score is the fraction of observed energy explained by
    the orthogonal projection onto each single candidate atom.
    """

    dictionary = _complex_numpy(candidate_frequency_responses)
    observed = _complex_numpy(observed_frequency_response)
    if dictionary.ndim != 2 or dictionary.shape[0] == 0 or dictionary.shape[1] == 0:
        raise ValueError("candidate frequency dictionary must have shape [N, F]")
    if observed.ndim != 1 or observed.shape[0] != dictionary.shape[1]:
        raise ValueError("observed frequency response must match dictionary frequencies")
    if not np.isfinite(dictionary).all() or not np.isfinite(observed).all():
        raise ValueError("sparse recovery inputs must be finite")
    if not isinstance(source_count, (int, np.integer)) or isinstance(source_count, bool):
        raise ValueError("source_count must be an integer")
    source_count = int(source_count)
    if source_count <= 0 or source_count > dictionary.shape[0]:
        raise ValueError("source_count must lie within the candidate count")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    design = dictionary.T
    atom_energies = np.sum(np.abs(design) ** 2, axis=0).real
    if np.any(atom_energies <= epsilon):
        raise ValueError("every candidate atom must have nonzero multifrequency energy")
    observed_energy = float(np.vdot(observed, observed).real)
    if observed_energy <= epsilon:
        raise ValueError("observed frequency response has zero energy")

    support: list[int] = []
    residual = observed.copy()
    coefficients = np.empty(0, dtype=np.complex128)
    first_step_scores: np.ndarray | None = None
    for _step in range(source_count):
        correlations = design.conj().T @ residual
        goodness = np.abs(correlations) ** 2 / atom_energies
        if first_step_scores is None:
            first_step_scores = (goodness / observed_energy).astype(np.float64)
        selectable = goodness.copy()
        selectable[support] = -np.inf
        selected = int(np.argmax(selectable))
        if not np.isfinite(selectable[selected]):
            raise RuntimeError("Room Helps OMP has no selectable candidate")
        support.append(selected)
        selected_design = design[:, support]
        coefficients = np.linalg.lstsq(selected_design, observed, rcond=None)[0]
        residual = observed - selected_design @ coefficients

    relative_residual = float(np.linalg.norm(residual) / np.sqrt(observed_energy))
    if first_step_scores is None or not np.isfinite(relative_residual):
        raise RuntimeError("Room Helps OMP produced an invalid solution")
    return RoomHelpsSparseResult(
        support=tuple(support),
        coefficients=coefficients,
        first_step_scores=first_step_scores,
        relative_residual_norm=relative_residual,
    )
