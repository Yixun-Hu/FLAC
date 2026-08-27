import pytest
import torch

from src.baselines.protocol import (
    nested_context_prefix,
    prepare_and_score_waveforms,
    validate_baseline_waveforms,
)


class _FakeRetrieval:
    def compute_audio_features(self, waveforms):
        first = waveforms[..., :4].sum(dim=-1)
        second = waveforms[..., 4:8].sum(dim=-1)
        features = torch.cat((first, second), dim=1)
        return torch.nn.functional.normalize(features, dim=-1)


def test_nested_context_prefix_is_an_exact_ordered_prefix():
    audio = torch.arange(2 * 8 * 12, dtype=torch.float32).reshape(2, 8, 1, 12)
    coordinates = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)

    audio_1, coordinates_1 = nested_context_prefix(audio, coordinates, 1)
    audio_8, coordinates_8 = nested_context_prefix(audio, coordinates, 8)

    assert torch.equal(audio_1, audio_8[:, :1])
    assert torch.equal(coordinates_1, coordinates_8[:, :1])
    assert audio_1.data_ptr() == audio.data_ptr()
    assert coordinates_1.data_ptr() == coordinates.data_ptr()


@pytest.mark.parametrize("k", [0, 9])
def test_nested_context_prefix_rejects_invalid_counts(k):
    audio = torch.zeros(1, 8, 1, 9600)
    coordinates = torch.zeros(1, 8, 3)
    with pytest.raises(ValueError):
        nested_context_prefix(audio, coordinates, k)


def test_nested_context_prefix_rejects_misaligned_contexts():
    with pytest.raises(ValueError):
        nested_context_prefix(torch.zeros(1, 8, 1, 8), torch.zeros(1, 7, 3), 1)


def test_validate_baseline_waveforms_delegates_flac_shape_and_clamp_contract():
    waveforms = torch.tensor([[[2.0] * 10240]], dtype=torch.float32)
    prepared = validate_baseline_waveforms(waveforms)
    assert prepared.shape == (1, 1, 10240)
    assert prepared.dtype == torch.float32
    assert prepared.max().item() == 1.0

    with pytest.raises(ValueError):
        validate_baseline_waveforms(waveforms.double())
    with pytest.raises(ValueError):
        validate_baseline_waveforms(waveforms[..., :-1])
    with pytest.raises(ValueError):
        validate_baseline_waveforms(torch.full_like(waveforms, float("nan")))


def test_prepare_and_score_waveforms_uses_retrieval_audio_features_and_cosine_dot():
    candidates = torch.zeros(2, 1, 10240, dtype=torch.float32)
    candidates[0, 0, :4] = 1.0
    candidates[1, 0, 4:8] = 1.0
    observation_features = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    scores = prepare_and_score_waveforms(
        _FakeRetrieval(), candidates, observation_features
    )

    assert scores.shape == (2,)
    assert scores.tolist() == pytest.approx([1.0, 0.0])


def test_prepare_and_score_waveforms_requires_one_normalized_observation_embedding():
    candidates = torch.zeros(1, 1, 10240, dtype=torch.float32)
    with pytest.raises(ValueError):
        prepare_and_score_waveforms(_FakeRetrieval(), candidates, torch.zeros(2, 2))
    with pytest.raises(ValueError):
        prepare_and_score_waveforms(_FakeRetrieval(), candidates, torch.tensor([[2.0, 0.0]]))
