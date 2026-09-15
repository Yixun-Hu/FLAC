import numpy as np
import pytest
import torch

from src.baselines.fem_agree import (
    exact_bandlimit_waveforms,
    peak_normalize_waveforms,
    score_fem_waveforms_with_agree,
)


def test_exact_bandlimit_preserves_only_requested_complex_bins():
    generator = torch.Generator().manual_seed(4)
    waveforms = torch.randn(2, 1, 64, generator=generator)
    indices = np.asarray([2, 4, 7], dtype=np.int64)

    filtered = exact_bandlimit_waveforms(waveforms, indices)
    original_spectrum = torch.fft.rfft(waveforms.double(), dim=-1)
    filtered_spectrum = torch.fft.rfft(filtered.double(), dim=-1)
    outside = torch.ones(filtered_spectrum.shape[-1], dtype=torch.bool)
    outside[indices] = False

    assert filtered.shape == waveforms.shape
    assert filtered.dtype == torch.float32
    assert torch.allclose(
        filtered_spectrum[..., indices],
        original_spectrum[..., indices],
        atol=2e-6,
    )
    assert torch.allclose(
        filtered_spectrum[..., outside],
        torch.zeros_like(filtered_spectrum[..., outside]),
        atol=2e-6,
    )


def test_peak_normalization_is_independent_and_rejects_silence():
    waveforms = torch.tensor(
        [[[0.0, -2.0, 1.0]], [[0.0, 0.25, -0.5]]], dtype=torch.float32
    )

    normalized, peaks = peak_normalize_waveforms(waveforms, target_peak=0.95)

    torch.testing.assert_close(peaks, torch.tensor([2.0, 0.5]))
    torch.testing.assert_close(
        normalized.abs().amax(dim=(-2, -1)), torch.tensor([0.95, 0.95])
    )
    with pytest.raises(ValueError, match="nonzero"):
        peak_normalize_waveforms(torch.zeros(1, 1, 8))


class _FakeRetrieval:
    def compute_audio_features(self, waveforms):
        values = waveforms[:, 0]
        features = torch.stack((values.sum(dim=-1), values[:, 0] + 2.0), dim=-1)
        return torch.nn.functional.normalize(features, dim=-1)


def test_agree_scoring_returns_nested_candidate_scores_and_fixed_seeds():
    candidates = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]
    )
    observed = candidates[:1].clone()

    first_scores, first_similarities, first_seeds = score_fem_waveforms_with_agree(
        _FakeRetrieval(),
        candidates,
        observed,
        query_index=17,
        score_seed=42,
        device="cpu",
        candidate_batch_size=1,
    )
    second_scores, second_similarities, second_seeds = score_fem_waveforms_with_agree(
        _FakeRetrieval(),
        candidates,
        observed,
        query_index=17,
        score_seed=42,
        device="cpu",
        candidate_batch_size=1,
    )

    assert set(first_scores) == {1, 4, 8}
    assert all(value.shape == (2,) for value in first_scores.values())
    assert first_similarities.shape == (2, 8)
    assert first_seeds == second_seeds
    torch.testing.assert_close(first_similarities, second_similarities)
    for count in first_scores:
        torch.testing.assert_close(first_scores[count], second_scores[count])
