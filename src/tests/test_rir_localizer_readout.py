import json

import numpy as np
import torch
from torch import nn

from src.data.fewshot_rir import rir_magnitude_spectrogram
from src.localization.fewshot_rir_readout import (
    READOUT_CONTEXT_COUNT,
    READOUT_GENERATION_COUNT,
    build_readout_query_result,
    infer_fewshot_rir_readout_query,
    summarize_readout_results,
)
from src.localization.rir_localizer import (
    AcousticRoomsRIRLocalizationDataset,
    RIRCoordinateLocalizer,
    RIRLogMagnitude,
    load_rir_localizer_checkpoint,
    localizer_checkpoint_payload,
    split_localizer_rooms,
)
from train_rir_localizer import RoomBalancedBatchSampler, meter_l1_distance_loss


def _tiny_config():
    return {
        "model_type": "rir_coordinate_localizer",
        "sample_size": 128,
        "sample_rate": 22050,
        "audio_channels": 1,
        "model": {
            "architecture": "tiny_test",
            "feature_dimensions": 7,
            "output_dimensions": 3,
        },
        "preprocessing": {
            "n_fft": 31,
            "hop_length": 8,
            "win_length": 16,
            "log_epsilon": 1e-8,
            "input_normalization": "none",
        },
    }


def test_log_magnitude_transform_matches_fewshotrir_preprocessing():
    torch.manual_seed(3)
    waveforms = torch.randn(2, 128)
    transform = RIRLogMagnitude(
        sample_size=128,
        n_fft=31,
        hop_length=8,
        win_length=16,
        log_epsilon=1e-8,
    )
    actual = transform(waveforms)
    magnitude = rir_magnitude_spectrogram(
        waveforms,
        n_fft=31,
        hop_length=8,
        win_length=16,
    )
    expected = torch.log(magnitude + 1e-8).permute(0, 3, 1, 2)
    assert actual.shape == (2, 1, 16, 17)
    torch.testing.assert_close(actual, expected)


def test_dataset_uses_receiver_relative_coordinates(tmp_path, monkeypatch):
    split = {
        "SceneA": {
            "Room1": ["S001_R002_hybrid_IR.wav"],
            "Room2": ["S003_R004_hybrid_IR.wav"],
        }
    }
    split_path = tmp_path / "train.json"
    split_path.write_text(json.dumps(split))
    monkeypatch.setattr(
        "src.localization.rir_localizer.load_rir_waveform",
        lambda *_args, **_kwargs: torch.arange(16, dtype=torch.float32),
    )
    monkeypatch.setattr(
        "src.localization.rir_localizer.load_ar_positions",
        lambda *_args, **_kwargs: (
            np.asarray([4.0, 8.0, 2.0], dtype=np.float32),
            np.asarray([1.0, 3.0, 1.5], dtype=np.float32),
        ),
    )
    dataset = AcousticRoomsRIRLocalizationDataset(
        dataset_root=tmp_path,
        split_path=split_path,
        sample_size=16,
    )
    item = dataset[0]
    torch.testing.assert_close(
        item["relative_source"], torch.tensor([3.0, 5.0, 0.5])
    )
    assert item["waveform"].shape == (16,)

    training, validation = split_localizer_rooms(
        split_path, validation_fraction=0.5, seed=9
    )
    assert set(training).isdisjoint(validation)
    assert set(training).union(validation) == {
        ("SceneA", "Room1"),
        ("SceneA", "Room2"),
    }


def test_tiny_localizer_checkpoint_round_trip(tmp_path):
    model = RIRCoordinateLocalizer(
        architecture="tiny_test",
        feature_dimensions=7,
    )
    checkpoint = localizer_checkpoint_payload(
        model=model,
        model_config=_tiny_config(),
        step=11,
        best_validation_l1_m=1.25,
        run_manifest_sha256="a" * 64,
    )
    assert "normalization" not in checkpoint
    path = tmp_path / "localizer.pt"
    torch.save(checkpoint, path)
    loaded, transform, bundle = load_rir_localizer_checkpoint(path, "cpu")
    values = torch.randn(2, 1, 16, 17)
    torch.testing.assert_close(model(values), loaded(values))
    assert transform.sample_size == 128
    assert bundle["step"] == 11
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_localizer_uses_official_log_magnitude_without_z_score():
    class _RecordingEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, values):
            self.seen = values.detach().clone()
            return torch.ones(values.shape[0], 7)

    model = RIRCoordinateLocalizer(
        architecture="tiny_test",
        feature_dimensions=7,
    )
    recorder = _RecordingEncoder()
    model.encoder = recorder
    values = torch.linspace(-8.0, 2.0, 16 * 17).reshape(1, 1, 16, 17)
    output = model(values)
    torch.testing.assert_close(recorder.seen, values)
    assert output.shape == (1, 3)
    assert not hasattr(model, "log_spectrogram_mean")


def test_training_loss_is_equal_weight_meter_space_sle():
    prediction = torch.zeros(1, 3, requires_grad=True)
    target = torch.tensor([[1.0, 2.0, 3.0]])
    loss = meter_l1_distance_loss(prediction, target)
    assert float(loss) == 6.0
    loss.backward()
    torch.testing.assert_close(prediction.grad, -torch.ones_like(prediction))


class _RecordingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, **batch):
        self.calls += 1
        assert batch["context_depth"].shape[:2] == (1, READOUT_CONTEXT_COUNT)
        assert batch["query_poses"].shape[:2] == (1, READOUT_GENERATION_COUNT)
        return torch.full((1, 1, 4, 5, 1), 2.0)


class _RecordingLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, values):
        self.inputs.append(values.detach().clone())
        return values.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 3)


class _FixedTransform(nn.Module):
    def forward(self, waveforms):
        assert waveforms.shape == (1, 32)
        return torch.full((1, 1, 4, 5), 5.0, device=waveforms.device)


def test_readout_generates_one_rir_and_passes_raw_log_magnitude_directly():
    generator = _RecordingGenerator()
    localizer = _RecordingLocalizer()
    metadata = {
        "context_depth": torch.zeros(8, 1, 3, 4),
        "context_magnitude": torch.ones(8, 4, 5, 1),
        "context_poses": torch.zeros(8, 6),
        "anchor_global": np.zeros(3, dtype=np.float32),
    }
    inference = infer_fewshot_rir_readout_query(
        generator,
        localizer,
        _FixedTransform(),
        observed_waveform=torch.zeros(1, 32),
        context_metadata=metadata,
        source_global=np.asarray([4.0, 5.0, 6.0]),
        receiver_global=np.asarray([1.0, 1.0, 1.0]),
        device="cpu",
    )
    assert generator.calls == 1
    assert len(localizer.inputs) == 2
    assert localizer.inputs[0].shape == (1, 1, 4, 5)
    torch.testing.assert_close(localizer.inputs[0], torch.full((1, 1, 4, 5), 2.0))
    torch.testing.assert_close(localizer.inputs[1], torch.full((1, 1, 4, 5), 5.0))
    np.testing.assert_allclose(inference["generated_prediction_global"], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(inference["gt_prediction_global"], [6.0, 6.0, 6.0])


def test_readout_result_and_summary_are_continuous_point_metrics():
    inference = {
        "generated_prediction_relative": np.asarray([2.0, 3.0, 4.0]),
        "generated_prediction_global": np.asarray([3.0, 4.0, 5.0]),
        "gt_prediction_relative": np.asarray([3.0, 4.0, 5.0]),
        "gt_prediction_global": np.asarray([4.0, 5.0, 6.0]),
        "timing_seconds": {
            "fewshotrir_generation": 0.2,
            "generated_rir_localizer": 0.1,
            "gt_rir_localizer": 0.1,
            "generated_pipeline": 0.3,
        },
        "spectrogram_diagnostics": {},
    }
    result = build_readout_query_result(
        query_index=7,
        query_id="query.wav",
        scene="Scene",
        room="Room",
        receiver_id=2,
        source_global=np.asarray([4.0, 5.0, 6.0]),
        receiver_global=np.asarray([1.0, 1.0, 1.0]),
        inference=inference,
        run_manifest_sha256="b" * 64,
        elapsed_seconds=0.5,
    )
    assert result["protocol"]["candidate_search"] is False
    assert result["protocol"]["output"] == "one_continuous_xyz_point"
    assert result["protocol"]["paper_reference_context_count"] == 20
    assert result["protocol"]["directly_comparable_to_paper_n20_sle"] is False
    assert result["generated_rir_readout"]["metrics"]["l1_distance_m"] == 3.0
    assert result["ground_truth_rir_readout"]["metrics"]["euclidean_error_m"] == 0.0
    summary = summarize_readout_results(
        [result, result], run_manifest_sha256="b" * 64
    )
    assert summary["query_count"] == 2
    assert summary["protocol"]["paper_reference_context_count"] == 20
    assert summary["protocol"]["directly_comparable_to_paper_n20_sle"] is False
    assert summary["metrics"]["generated_rir_readout"]["coordinate_mae_m"] == 1.0
    assert summary["metrics"]["ground_truth_rir_readout"]["success_rate_0_5m"] == 1.0


def test_room_balanced_sampler_is_resume_stable():
    class _Dataset:
        room_indices = {
            ("A", "one"): (0, 1),
            ("B", "two"): (2, 3, 4),
        }

    complete = list(
        RoomBalancedBatchSampler(
            _Dataset(), batch_size=6, seed=42, start_step=0, stop_step=8
        )
    )
    resumed = list(
        RoomBalancedBatchSampler(
            _Dataset(), batch_size=6, seed=42, start_step=3, stop_step=8
        )
    )
    assert complete[3:] == resumed
    assert all(len(batch) == 6 for batch in complete)
