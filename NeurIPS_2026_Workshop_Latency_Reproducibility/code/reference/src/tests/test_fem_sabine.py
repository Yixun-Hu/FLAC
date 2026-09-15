import math

import numpy as np
import pytest
import torch

import src.baselines.fem_sabine as fem_sabine
from src.baselines.fem_sabine import (
    bandlimited_response_to_waveform,
    dft_frequency_bins,
    estimate_context_t60,
    sabine_boundary,
)


def test_context_t60_matches_flac_ar_decay_setting_and_uses_valid_median(monkeypatch):
    calls = []

    def fake_measure(waveform, fs, decay_db):
        calls.append((fs, decay_db, float(waveform[0])))
        if waveform[0] == 2:
            raise ValueError("invalid decay")
        return {1.0: 0.4, 3.0: 0.8}[float(waveform[0])]

    monkeypatch.setattr(
        fem_sabine.pyroomacoustics.experimental, "measure_rt60", fake_measure
    )
    contexts = torch.zeros(3, 1, 9600)
    contexts[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])

    estimate = estimate_context_t60(contexts)

    assert estimate.t60_seconds == pytest.approx(0.6)
    assert estimate.valid_count == 2
    assert estimate.invalid_count == 1
    assert all(fs == 22050 and decay_db == 20 for fs, decay_db, _ in calls)


def test_context_t60_fails_closed_when_every_context_is_invalid(monkeypatch):
    def invalid(*_args, **_kwargs):
        raise ValueError("no decay")

    monkeypatch.setattr(fem_sabine.pyroomacoustics.experimental, "measure_rt60", invalid)
    with pytest.raises(ValueError, match="no valid"):
        estimate_context_t60(torch.zeros(2, 1, 9600))


def test_sabine_boundary_matches_hand_computation_and_reports_clipping():
    boundary = sabine_boundary(volume_m3=100.0, surface_area_m2=150.0, t60_seconds=0.5)
    expected_alpha = 0.161 * 100.0 / (150.0 * 0.5)
    expected_reflection = math.sqrt(1.0 - expected_alpha)
    assert boundary.raw_absorption == pytest.approx(expected_alpha)
    assert boundary.absorption == pytest.approx(expected_alpha)
    assert boundary.reflection_magnitude == pytest.approx(expected_reflection)
    assert boundary.normalized_impedance == pytest.approx(
        (1.0 + expected_reflection) / (1.0 - expected_reflection)
    )
    assert boundary.was_clipped is False

    clipped = sabine_boundary(100.0, 1.0, 0.01, absorption_limits=(1e-4, 0.99))
    assert clipped.absorption == pytest.approx(0.99)
    assert clipped.was_clipped is True


def test_dft_frequency_bins_are_exact_and_inside_requested_band():
    indices, frequencies = dft_frequency_bins(22050, 10240, 80.0, 300.0)
    spacing = 22050 / 10240
    assert indices[0] == math.ceil(80.0 / spacing)
    assert indices[-1] == math.floor(300.0 / spacing)
    assert np.allclose(frequencies, indices * spacing)
    assert frequencies[0] >= 80.0 and frequencies[-1] <= 300.0


def test_bandlimited_response_ifft_preserves_selected_complex_bins():
    indices, _ = dft_frequency_bins(22050, 320, 80.0, 300.0)
    response = torch.complex(
        torch.arange(1, len(indices) + 1, dtype=torch.float32).repeat(2, 1),
        torch.ones(2, len(indices)),
    )
    waveforms = bandlimited_response_to_waveform(
        response, indices, sample_count=320, unit_gain=0.25
    )
    reconstructed = torch.fft.rfft(waveforms[:, 0], n=320)
    outside = torch.ones(reconstructed.shape[-1], dtype=torch.bool)
    outside[torch.as_tensor(indices)] = False

    assert waveforms.shape == (2, 1, 320)
    assert waveforms.dtype == torch.float32
    assert torch.allclose(reconstructed[:, indices], response * 0.25, atol=1e-5)
    assert torch.allclose(reconstructed[:, outside], torch.zeros_like(reconstructed[:, outside]), atol=1e-5)


def test_bandlimited_ifft_preserves_a_synthetic_direct_path_delay():
    sample_count = 10240
    indices, frequencies = dft_frequency_bins(22050, sample_count, 80.0, 300.0)
    delay_samples = 50
    delay_seconds = delay_samples / 22050.0
    response = torch.from_numpy(
        np.exp(-2j * np.pi * frequencies * delay_seconds)[None, :]
    ).to(torch.complex64)

    waveform = bandlimited_response_to_waveform(
        response, indices, sample_count=sample_count
    )[0, 0]

    assert int(torch.argmax(waveform.abs()).item()) == delay_samples
