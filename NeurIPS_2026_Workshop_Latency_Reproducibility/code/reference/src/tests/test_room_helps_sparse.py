import numpy as np
import pytest
import torch

from src.baselines.room_helps_sparse import (
    extract_rir_frequency_response,
    room_helps_pulse_omp,
)


def test_pulse_stack_omp_recovers_one_source_with_unknown_complex_gain():
    dictionary = np.array(
        [
            [1 + 0j, 0 + 1j, 1 - 1j, 0.5 + 0.2j, -0.3 + 0.1j],
            [0.1 + 0.7j, 1 + 0j, -0.4 + 0.2j, 0.3 - 0.9j, 0.8 + 0.1j],
            [0.3 - 0.1j, -0.2 + 0.4j, 1 + 0j, -0.5 + 0.2j, 0.4 + 0.9j],
        ],
        dtype=np.complex128,
    )
    observed = (2.0 - 0.75j) * dictionary[1]

    result = room_helps_pulse_omp(dictionary, observed, source_count=1)

    assert result.support == (1,)
    assert int(np.argmax(result.first_step_scores)) == 1
    assert result.coefficients[0] == pytest.approx(2.0 - 0.75j)
    assert result.relative_residual_norm < 1e-12


def test_pulse_stack_omp_uses_a_common_sparse_support_across_frequencies():
    frequencies = 8
    basis = np.fft.fft(np.eye(frequencies)) / np.sqrt(frequencies)
    dictionary = basis[:, :4].T
    observed = 0.4 * dictionary[0] + (1.2 + 0.3j) * dictionary[2]

    result = room_helps_pulse_omp(dictionary, observed, source_count=2)

    assert set(result.support) == {0, 2}
    assert result.relative_residual_norm < 1e-12


def test_pulse_stack_omp_uses_frozen_candidate_order_for_exact_ties():
    atom = np.array([1 + 0j, 0.5 + 0.5j, -0.2j])
    dictionary = np.stack((atom, atom, np.array([0.2, 0.3, 0.4])))

    result = room_helps_pulse_omp(dictionary, atom, source_count=1)

    assert result.support == (0,)
    assert result.first_step_scores[0] == pytest.approx(result.first_step_scores[1])


def test_extract_rir_frequency_response_reads_exact_complex_dft_bins():
    sample_count = 32
    spectrum = torch.zeros(sample_count // 2 + 1, dtype=torch.complex64)
    spectrum[2] = 1.0 + 2.0j
    spectrum[5] = -0.5 + 0.25j
    waveform = torch.fft.irfft(spectrum, n=sample_count).reshape(1, sample_count)

    response = extract_rir_frequency_response(
        waveform, np.array([2, 5]), sample_count=sample_count
    )

    assert response.shape == (2,)
    assert np.allclose(response, spectrum[[2, 5]].numpy(), atol=1e-6)


def test_pulse_stack_omp_rejects_zero_energy_or_inconsistent_inputs():
    with pytest.raises(ValueError, match="observed"):
        room_helps_pulse_omp(np.ones((2, 3), dtype=complex), np.zeros(3), source_count=1)
    with pytest.raises(ValueError, match="candidate"):
        room_helps_pulse_omp(np.zeros((2, 3), dtype=complex), np.ones(3), source_count=1)
    with pytest.raises(ValueError, match="source_count"):
        room_helps_pulse_omp(np.ones((2, 3), dtype=complex), np.ones(3), source_count=3)
