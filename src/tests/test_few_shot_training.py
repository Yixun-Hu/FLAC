import json
from pathlib import Path
from types import SimpleNamespace

import torch

import train
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config


def _config():
    return {
        "model_type": "few_shot_rir_waveform",
        "sample_rate": 22050,
        "sample_size": 320,
        "audio_channels": 1,
        "model": {
            "geometry_channels": 3,
            "embedding_dim": 16,
            "num_heads": 4,
            "context_layers": 1,
            "decoder_layers": 1,
            "audio_channels": 8,
            "geometry_channels_hidden": 8,
            "waveform_channels": 4,
            "coordinate_num_frequencies": 3,
            "coordinate_max_frequency": 2.0,
        },
        "training": {
            "context_counts": [1, 8],
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "loss": {
                "waveform_weight": 1.0,
                "mrstft_weight": 1.0,
                "edc_weight": 0.01,
                "fft_sizes": [64, 128],
            },
        },
    }


def _batch(batch_size=2):
    target = torch.randn(batch_size, 1, 320)
    metadata = []
    for _ in range(batch_size):
        metadata.append(
            {
                "depth": torch.randn(3, 16, 16),
                "context_audio": torch.randn(8, 1, 96),
                "context_poses": torch.randn(8, 3),
                "source": torch.randn(3),
                "padding_mask": torch.ones(320),
            }
        )
    return target, tuple(metadata)


def test_model_and_training_factories_construct_waveform_baseline_from_scratch():
    config = _config()
    model = create_model_from_config(config)
    wrapper = create_training_wrapper_from_config(config, model)
    assert wrapper.model is model
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not any("agree" in name.lower() or "localization" in name.lower() for name, _ in wrapper.named_parameters())


def test_training_batch_uses_one_nested_context_count_and_direct_target():
    config = _config()
    wrapper = create_training_wrapper_from_config(config, create_model_from_config(config))
    target, metadata = _batch()
    inputs, prepared_target, padding_mask = wrapper.prepare_batch(
        (target, metadata), context_count=1
    )
    assert inputs["context_audio"].shape == (2, 8, 1, 96)
    assert inputs["context_coordinates"].shape == (2, 8, 3)
    assert torch.equal(
        inputs["context_mask"],
        torch.tensor([[True, False, False, False, False, False, False, False]]).expand(2, -1),
    )
    assert inputs["query_receiver"].eq(0).all()
    assert torch.equal(prepared_target, target)
    assert padding_mask.shape == (2, 320)


def test_one_training_step_updates_randomly_initialized_baseline_weights(monkeypatch):
    torch.manual_seed(0)
    config = _config()
    wrapper = create_training_wrapper_from_config(config, create_model_from_config(config))
    monkeypatch.setattr(wrapper, "draw_context_count", lambda: 1)
    optimizer = wrapper.configure_optimizers()
    before = next(wrapper.model.parameters()).detach().clone()
    loss = wrapper.training_step(_batch(), 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = next(wrapper.model.parameters()).detach()
    assert torch.isfinite(loss)
    assert not torch.equal(before, after)


def test_validation_reports_both_primary_context_counts(monkeypatch):
    config = _config()
    wrapper = create_training_wrapper_from_config(config, create_model_from_config(config))
    calls = []

    def fake_shared_step(_batch, context_count):
        calls.append(context_count)
        value = torch.tensor(float(context_count))
        return {"loss": value, "waveform": value, "mrstft": value, "edc": value}

    monkeypatch.setattr(wrapper, "_shared_step", fake_shared_step)
    loss = wrapper.validation_step(_batch(), 0)

    assert calls == [1, 8]
    assert loss == torch.tensor(4.5)


def test_few_shot_checkpointing_selects_lowest_reconstruction_validation_loss():
    args = SimpleNamespace(checkpoint_every=10_000, val_every=2_500)
    callbacks = train.build_checkpoint_callbacks(
        args,
        "checkpoints",
        _config(),
        has_validation=True,
    )
    assert len(callbacks) == 2
    periodic, best = callbacks
    assert periodic._every_n_train_steps == 10_000
    assert best.monitor == "val/reconstruction_loss"
    assert best.mode == "min"
    assert best.save_top_k == 1
    assert best.save_last is True


def test_few_shot_checkpointing_fails_closed_without_validation():
    args = SimpleNamespace(checkpoint_every=10_000, val_every=-1)
    try:
        train.build_checkpoint_callbacks(
            args,
            "checkpoints",
            _config(),
            has_validation=False,
        )
    except ValueError as error:
        assert "validation dataset" in str(error)
    else:
        raise AssertionError("Few-ShotRIR must not train without validation selection")


def test_committed_ar_training_config_disables_time_shift_augmentation():
    path = Path("src/configs/dataset_configs/AR/train/acousticroom_train_few_shot_waveform.json")
    config = json.loads(path.read_text())
    assert config["random_crop"] is False
    assert config["augs"] is False
    assert config["modalities"]["acoustic_context"]["max_context"] == 8
    assert config["modalities"]["acoustic_context"]["max_len"] == 9600
