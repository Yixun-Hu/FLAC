import inspect

import pytest
import torch

from src.baselines.few_shot_rir_waveform import (
    FLACCoordinateEmbedding,
    FewShotRIRWaveform,
    FewShotRIRWaveformLoss,
    energy_decay_loss,
    multi_resolution_stft_loss,
)


def _tiny_model(output_samples=320):
    return FewShotRIRWaveform(
        geometry_channels=3,
        embedding_dim=16,
        num_heads=4,
        context_layers=1,
        decoder_layers=1,
        audio_channels=8,
        geometry_channels_hidden=8,
        waveform_channels=4,
        output_samples=output_samples,
        coordinate_num_frequencies=3,
        coordinate_max_frequency=2.0,
    )


def test_flac_coordinate_embedding_matches_frozen_fourier_definition():
    encoder = FLACCoordinateEmbedding(num_frequencies=2, max_frequency=1.0, max_value=5.0)
    coordinates = torch.tensor([[[5.0, 0.0, -5.0]]])
    encoded = encoder.features(coordinates)
    normalized = coordinates / 5.0
    frequencies = 2.0 ** torch.linspace(0.0, 1.0, 2)
    expected = torch.cat(
        [normalized]
        + [torch.sin(normalized * frequency) for frequency in frequencies]
        + [torch.cos(normalized * frequency) for frequency in frequencies],
        dim=-1,
    )
    assert encoded.shape == (1, 1, 15)
    assert torch.allclose(encoded, expected)


def test_waveform_model_accepts_flac_ar_inputs_and_outputs_direct_waveform():
    torch.manual_seed(0)
    model = _tiny_model(output_samples=320)
    geometry = torch.randn(2, 3, 16, 24)
    context_audio = torch.randn(2, 8, 1, 96)
    context_coordinates = torch.randn(2, 8, 3)
    query_source = torch.randn(2, 3)
    query_receiver = torch.zeros(2, 3)

    output = model(
        geometry=geometry,
        context_audio=context_audio,
        context_coordinates=context_coordinates,
        query_source=query_source,
        query_receiver=query_receiver,
        context_mask=torch.ones(2, 8, dtype=torch.bool),
    )

    assert output.shape == (2, 1, 320)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert output.abs().max() <= 1.0
    assert "rgb" not in inspect.signature(model.forward).parameters


def test_one_model_supports_k1_and_k8_with_an_explicit_validity_mask():
    model = _tiny_model(output_samples=320).eval()
    geometry = torch.randn(1, 3, 16, 16)
    audio = torch.randn(1, 8, 1, 96)
    coordinates = torch.randn(1, 8, 3)
    source = torch.randn(1, 3)
    receiver = torch.zeros(1, 3)

    with torch.no_grad():
        k1 = model(
            geometry,
            audio[:, :1],
            coordinates[:, :1],
            source,
            receiver,
            context_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        k8 = model(
            geometry,
            audio,
            coordinates,
            source,
            receiver,
            context_mask=torch.ones(1, 8, dtype=torch.bool),
        )
    assert k1.shape == k8.shape == (1, 1, 320)

    with pytest.raises(ValueError):
        model(
            geometry,
            audio,
            coordinates,
            source,
            receiver,
            context_mask=torch.zeros(1, 8, dtype=torch.bool),
        )


def test_context_memory_is_permutation_invariant_and_ignores_masked_slots():
    torch.manual_seed(4)
    model = _tiny_model(output_samples=320).eval()
    geometry = torch.randn(1, 3, 16, 16)
    audio = torch.randn(1, 8, 1, 96)
    coordinates = torch.randn(1, 8, 3)
    source = torch.randn(1, 3)
    receiver = torch.zeros(1, 3)
    permutation = torch.tensor([4, 1, 6, 0, 3, 7, 2, 5])

    with torch.no_grad():
        original = model(geometry, audio, coordinates, source, receiver)
        permuted = model(
            geometry,
            audio[:, permutation],
            coordinates[:, permutation],
            source,
            receiver,
        )
        prefix_only = model(
            geometry,
            audio[:, :1],
            coordinates[:, :1],
            source,
            receiver,
        )
        prefix_masked = model(
            geometry,
            audio,
            coordinates,
            source,
            receiver,
            context_mask=torch.tensor([[True, False, False, False, False, False, False, False]]),
        )

    assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-5)
    assert torch.allclose(prefix_only, prefix_masked, atol=1e-6, rtol=1e-5)


def test_reconstruction_losses_are_zero_for_identity_and_have_finite_gradients():
    target = torch.randn(1, 1, 512)
    identical = target.clone()
    assert multi_resolution_stft_loss(identical, target, fft_sizes=(64, 128)) == pytest.approx(0.0)
    assert energy_decay_loss(identical, target) == pytest.approx(0.0)

    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    objective = FewShotRIRWaveformLoss(fft_sizes=(64, 128))
    losses = objective(prediction, target)
    assert set(losses) == {"loss", "waveform", "mrstft", "edc"}
    assert losses["loss"].item() > 0
    losses["loss"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_energy_decay_loss_ignores_samples_outside_padding_mask():
    target = torch.zeros(1, 1, 64)
    target[..., :32] = torch.linspace(1.0, 0.1, 32)
    prediction = target.clone()
    prediction[..., 32:] = 10.0
    padding_mask = torch.zeros(1, 64, dtype=torch.bool)
    padding_mask[:, :32] = True

    assert energy_decay_loss(prediction, target, padding_mask=padding_mask) == pytest.approx(0.0)


def test_waveform_loss_is_normalized_by_valid_samples_not_padded_length():
    objective = FewShotRIRWaveformLoss(
        waveform_weight=1.0,
        mrstft_weight=0.0,
        edc_weight=0.0,
        fft_sizes=(16,),
    )
    target = torch.zeros(1, 1, 64)
    prediction = target.clone()
    prediction[..., :8] = 1.0
    short_mask = torch.zeros(1, 64, dtype=torch.bool)
    short_mask[..., :8] = True
    long_mask = torch.zeros(1, 64, dtype=torch.bool)
    long_mask[..., :16] = True

    short = objective(prediction, target, padding_mask=short_mask)["waveform"]
    long = objective(prediction, target, padding_mask=long_mask)["waveform"]

    assert short == pytest.approx(1.0)
    assert long == pytest.approx(0.5)


def test_energy_decay_loss_includes_silent_gaps_before_the_last_reflection():
    target = torch.zeros(1, 1, 64)
    target[..., 0] = 1.0
    target[..., 31] = 0.5
    prediction = target.clone()
    prediction[..., 16] = 0.5
    prediction[..., 0] = 0.75**0.5  # preserve total energy and the final-tail ratio

    assert energy_decay_loss(prediction, target).item() > 0.1


def test_waveform_model_rejects_wrong_geometry_and_context_contracts():
    model = _tiny_model(output_samples=320)
    valid = dict(
        geometry=torch.randn(1, 3, 16, 16),
        context_audio=torch.randn(1, 2, 1, 96),
        context_coordinates=torch.randn(1, 2, 3),
        query_source=torch.randn(1, 3),
        query_receiver=torch.zeros(1, 3),
    )
    with pytest.raises(ValueError):
        model(**{**valid, "geometry": torch.randn(1, 4, 16, 16)})
    with pytest.raises(ValueError):
        model(**{**valid, "context_coordinates": torch.randn(1, 3, 3)})
