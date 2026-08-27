import numpy as np
import pytest
import torch
from torch import nn

from src.localization.baseline_runner import (
    build_few_shot_candidate_batch,
    score_fem_room_helps_candidates,
    score_few_shot_candidates,
)


class _FakeRetrieval:
    def compute_audio_features(self, waveforms):
        features = torch.cat(
            (waveforms[..., :4].sum(-1), waveforms[..., 4:8].sum(-1)), dim=1
        )
        return torch.nn.functional.normalize(features, dim=-1)


class _FakeWaveformModel(nn.Module):
    def forward(
        self,
        geometry,
        context_audio,
        context_coordinates,
        query_source,
        query_receiver,
        context_mask=None,
    ):
        output = torch.zeros(query_source.shape[0], 1, 10240, device=query_source.device)
        positive = query_source[:, 0] >= 0
        output[positive, 0, :4] = 1.0
        output[~positive, 0, 4:8] = 1.0
        return output


def _metadata():
    return {
        "depth": torch.randn(3, 12, 16),
        "context_audio": torch.arange(8 * 9600, dtype=torch.float32).reshape(8, 1, 9600),
        "context_poses": torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3),
    }


def test_build_few_shot_batch_uses_receiver_relative_candidates_and_nested_contexts():
    candidates = np.array([[2.0, 3.0, 4.0], [0.0, 1.0, 2.0]])
    receiver = np.array([1.0, 1.0, 1.0])
    k1 = build_few_shot_candidate_batch(_metadata(), candidates, receiver, context_count=1)
    k8 = build_few_shot_candidate_batch(_metadata(), candidates, receiver, context_count=8)

    assert k1["geometry"].shape == (2, 3, 12, 16)
    assert torch.equal(k1["context_audio"], k8["context_audio"][:, :1])
    assert torch.equal(k1["context_coordinates"], k8["context_coordinates"][:, :1])
    assert torch.allclose(k8["query_source"], torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
    assert torch.equal(k8["query_receiver"], torch.zeros(2, 3))
    assert k8["context_mask"].all()


def test_few_shot_candidate_scoring_is_batch_invariant_and_uses_common_scorer():
    candidates = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    observation = torch.tensor([[1.0, 0.0]])
    expected = [1.0, 0.0, 1.0]
    outputs = []
    for batch_size in (1, 2, 8):
        outputs.append(
            score_few_shot_candidates(
                _FakeWaveformModel(),
                _FakeRetrieval(),
                _metadata(),
                candidates,
                receiver_global=np.zeros(3),
                observation_features=observation,
                context_count=1,
                candidate_batch_size=batch_size,
                device="cpu",
            )
        )
    for scores in outputs:
        assert scores.tolist() == pytest.approx(expected)


def test_fem_candidates_use_room_helps_complex_pulse_sparse_recovery():
    sample_count = 32
    bin_indices = np.array([2, 3, 4])
    response = torch.tensor(
        [[1 + 0j, 0.2 + 0.1j, -0.5j], [0.1j, 1 - 0.2j, 0.3 + 0.4j]]
    )
    spectrum = torch.zeros(sample_count // 2 + 1, dtype=torch.complex64)
    spectrum[bin_indices] = (1.7 - 0.2j) * response[1]
    observed = torch.fft.irfft(spectrum, n=sample_count).reshape(1, sample_count)

    scores, recovery = score_fem_room_helps_candidates(
        response,
        bin_indices,
        observed,
        sample_count=sample_count,
    )

    assert scores.shape == (2,)
    assert int(torch.argmax(scores)) == 1
    assert recovery.support == (1,)
    assert recovery.relative_residual_norm < 1e-6
