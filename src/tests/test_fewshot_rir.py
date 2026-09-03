import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from src.baselines.fewshot_rir import (
    FewshotRiR,
    FewshotRiRLoss,
    MagnitudeSpectrogramDecoder,
)
from src.data.fewshot_rir import (
    AcousticRoomsFewshotRiRDataset,
    select_near_coincident_contexts,
)
from src.localization.ar_queries import _canonical_sha
from src.localization.fewshot_rir import (
    NEAR_CONTEXT_PROTOCOL,
    build_fewshot_rir_candidate_batch,
    load_fewshot_rir_query,
)
from src.localization.fewshot_rir_contexts import build_fewshot_rir_context_manifest
from src.localization.baseline_experiment import load_fewshot_rir_checkpoint
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config


def _tiny_model():
    return FewshotRiR(
        sample_size=64,
        n_fft=31,
        hop_length=4,
        win_length=16,
        embedding_dim=64,
        hidden_dim=128,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        context_encoder_architecture="tiny_test",
        context_encoder_features=16,
        decoder_input_channels=4,
        decoder_channels=(8, 4),
        decoder_final_output_padding=(0, 1),
    )


def _tiny_config():
    return {
        "model_type": "FewshotRiR",
        "sample_rate": 22050,
        "sample_size": 64,
        "audio_channels": 1,
        "model": {
            "n_fft": 31,
            "hop_length": 4,
            "win_length": 16,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "num_heads": 4,
            "context_encoder_architecture": "tiny_test",
            "context_encoder_features": 16,
            "decoder_input_channels": 4,
            "decoder_channels": [8, 4],
            "decoder_final_output_padding": [0, 1],
        },
        "training": {
            "context_count": 8,
            "loss": {"energy_decay_weight": 0.01},
        },
    }


def _write_room(root: Path):
    scene, room = "Office", "Office_idx_0"
    audio_dir = root / "single_channel_ir_1" / scene / room
    metadata_dir = root / "metadata" / scene / room
    depth_dir = root / "depth_map" / scene / room
    audio_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    self_names = []
    for index in range(8):
        filename = f"S00{index}_R00{index}_hybrid_IR.wav"
        self_names.append(filename)
        waveform = torch.zeros(1, 64)
        waveform[0, index + 1] = 0.5
        torchaudio.save(audio_dir / filename, waveform, 22050)
        location = [float(index), float(index + 1), 1.5]
        (metadata_dir / f"S00{index}_R00{index}.json").write_text(
            json.dumps({"src_loc": location, "rec_loc": location})
        )
        np.save(depth_dir / f"{index}.npy", np.full((8, 16), index + 1, np.float32))
    query_names = []
    for source, receiver in ((0, 1), (2, 3)):
        filename = f"S00{source}_R00{receiver}_hybrid_IR.wav"
        query_names.append(filename)
        waveform = torch.zeros(1, 64)
        waveform[0, source + receiver + 2] = 0.75
        torchaudio.save(audio_dir / filename, waveform, 22050)
        (metadata_dir / f"S00{source}_R00{receiver}.json").write_text(
            json.dumps(
                {
                    "src_loc": [float(source), float(source + 1), 1.5],
                    "rec_loc": [float(receiver), float(receiver + 1), 1.5],
                }
            )
        )
    return scene, room, self_names, query_names


def test_model_retains_independent_context_tokens_and_deterministic_griffin_lim():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    inputs = {
        "context_depth": torch.rand(1, 2, 1, 8, 16),
        "context_spectrograms": torch.rand(1, 2, 16, 17, 1),
        "context_poses": torch.rand(1, 2, 6),
        "query_poses": torch.rand(1, 3, 6),
    }
    raw = model(**inputs)
    permutation = torch.tensor([1, 0])
    permuted = model(
        context_depth=inputs["context_depth"][:, permutation],
        context_spectrograms=inputs["context_spectrograms"][:, permutation],
        context_poses=inputs["context_poses"][:, permutation],
        query_poses=inputs["query_poses"],
    )
    assert raw.shape == (1, 3, 16, 17, 1)
    assert torch.allclose(raw, permuted, atol=1e-5, rtol=1e-5)
    magnitude = model.output_to_magnitude(raw)
    first = model.magnitude_to_waveform(magnitude, iterations=2, momentum=0.0)
    second = model.magnitude_to_waveform(magnitude, iterations=2, momentum=0.0)
    assert first.shape == (1, 3, 1, 64)
    assert torch.equal(first, second)


def test_decoder_preserves_upstream_positional_kaiming_call(monkeypatch):
    calls = []
    original = torch.nn.init.kaiming_normal_

    def record(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
        calls.append((float(a), mode, nonlinearity))
        return original(
            tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
            generator=generator,
        )

    monkeypatch.setattr(torch.nn.init, "kaiming_normal_", record)
    MagnitudeSpectrogramDecoder(
        64,
        input_channels=4,
        channels=(8, 4),
        output_frequency_bins=16,
        output_frames=17,
        final_output_padding=(0, 1),
    )
    assert calls
    assert all(
        a == torch.nn.init.calculate_gain("relu")
        and mode == "fan_in"
        and nonlinearity == "leaky_relu"
        for a, mode, nonlinearity in calls
    )


def test_factory_training_wrapper_and_official_spectral_losses():
    model = create_model_from_config(_tiny_config())
    wrapper = create_training_wrapper_from_config(_tiny_config(), model)
    batch = {
        "context_depth": torch.rand(2, 8, 1, 8, 16),
        "context_magnitude": torch.rand(2, 8, 16, 17, 1),
        "context_poses": torch.rand(2, 8, 6),
        "target_magnitude": torch.rand(2, 2, 16, 17, 1),
        "query_poses": torch.rand(2, 2, 6),
        "context_mask": torch.ones(2, 8, dtype=torch.bool),
        "query_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    inputs, target, mask = wrapper.prepare_batch(batch, context_count=1)
    assert inputs["context_depth"].shape[:2] == (2, 1)
    raw = model(**inputs)
    values = FewshotRiRLoss()(raw, target, mask)
    assert set(values) == {"loss", "spectral_l1", "spectral_energy_decay"}
    assert all(torch.isfinite(value) for value in values.values())
    assert isinstance(wrapper.configure_optimizers(), torch.optim.Adam)

    calls = []

    def fixed_step(_batch, context_count):
        calls.append(context_count)
        value = torch.tensor(1.0, requires_grad=True)
        return {
            "loss": value,
            "spectral_l1": value,
            "spectral_energy_decay": value,
        }

    wrapper._shared_step = fixed_step
    wrapper.training_step(batch, 0)
    wrapper.validation_step(batch, 0)
    assert calls == [8, 8]


def test_upstream_query_mask_contract_is_all_valid_and_not_sent_to_transformer():
    model = _tiny_model().eval()
    inputs = {
        "context_depth": torch.rand(1, 1, 1, 8, 16),
        "context_spectrograms": torch.rand(1, 1, 16, 17, 1),
        "context_poses": torch.rand(1, 1, 6),
        "query_poses": torch.rand(1, 2, 6),
    }
    all_valid = model(
        **inputs, query_mask=torch.tensor([[True, True]])
    )
    one_marked_invalid = model(
        **inputs, query_mask=torch.tensor([[True, False]])
    )
    assert torch.equal(all_valid, one_marked_invalid)
    with torch.no_grad():
        raw = torch.zeros(1, 2, 16, 17, 1)
        target = torch.ones_like(raw)
    with pytest.raises(ValueError, match="every query slot"):
        FewshotRiRLoss()(raw, target, torch.tensor([[True, False]]))


def test_short_episode_sampling_uses_every_item_before_repetition():
    names = ("a", "b", "c")
    sampled = AcousticRoomsFewshotRiRDataset._sample_names(
        np.random.default_rng(5), names, 5
    )
    assert set(sampled[:3]) == set(names)
    assert len(sampled) == 5


def test_fewshot_rir_checkpoint_loader_uses_exact_public_name(tmp_path):
    config = _tiny_config()
    model = create_model_from_config(config)
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "model.ckpt"
    config_path.write_text(json.dumps(config))
    torch.save(
        {
            "model_config": config,
            "state_dict": {
                f"model.{key}": value for key, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    loaded, loaded_config = load_fewshot_rir_checkpoint(
        config_path, checkpoint_path, "cpu"
    )
    assert loaded_config["model_type"] == "FewshotRiR"
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_room_episode_and_localization_manifest_use_near_contexts(tmp_path):
    root = tmp_path / "AcousticRooms"
    scene, room, self_names, query_names = _write_room(root)
    query_path = tmp_path / "queries.json"
    inventory_path = tmp_path / "all.json"
    query_path.write_text(json.dumps({scene: {room: query_names}}))
    inventory_path.write_text(json.dumps({scene: {room: [*self_names, *query_names]}}))
    dataset = AcousticRoomsFewshotRiRDataset(
        {
            "dataset": {
                "path": str(root),
                "query_json_file_path": str(query_path),
                "context_json_file_path": str(inventory_path),
            },
            "episodes_per_room": 1,
            "max_context": 8,
            "max_queries": 2,
            "depth_size": [8, 16],
            "n_fft": 31,
            "hop_length": 4,
            "win_length": 16,
            "seed": 4,
        },
        sample_rate=22050,
        sample_size=64,
    )
    episode = dataset[0]
    assert episode["context_depth"].shape == (8, 1, 8, 16)
    assert episode["context_magnitude"].shape == (8, 16, 17, 1)
    assert episode["target_magnitude"].shape == (2, 16, 17, 1)
    assert torch.allclose(episode["context_poses"][:, :3], episode["context_poses"][:, 3:])

    query_relpath = str(Path("single_channel_ir_1") / scene / room / query_names[0])
    base = {
        "schema_version": 1,
        "records": [
            {
                "index": 0,
                "query_id": query_relpath,
                "filename": query_names[0],
                "scene": scene,
                "room": room,
                "source_global": [0.0, 1.0, 1.5],
                "receiver_global": [1.0, 2.0, 1.5],
                "contexts": [],
                "context_sources_global": [],
            }
        ],
    }
    base["sha256"] = _canonical_sha(base)
    manifest = build_fewshot_rir_context_manifest(
        base,
        context_inventory_path=inventory_path,
        dataset_root=root,
        seed=8,
    )
    record = manifest["records"][0]
    assert record["context_protocol"] == NEAR_CONTEXT_PROTOCOL
    assert len(record["contexts"]) == 8
    assert record["context_sources_global"] == record["context_receivers_global"]
    assert record["context_endpoint_distances_m"] == [0.0] * 8

    observed, metadata = load_fewshot_rir_query(
        record,
        root,
        sample_size=64,
        n_fft=31,
        hop_length=4,
        win_length=16,
        depth_size=(8, 16),
    )
    candidate_batch = build_fewshot_rir_candidate_batch(
        metadata,
        np.array([[4.0, 5.0, 1.5], [6.0, 7.0, 1.5]], dtype=np.float32),
        np.array(record["receiver_global"], dtype=np.float32),
        context_count=1,
    )
    assert observed.shape == (1, 64)
    assert candidate_batch["context_depth"].shape == (1, 1, 1, 8, 16)
    assert candidate_batch["query_poses"].shape == (1, 2, 6)


def test_near_context_selection_uses_geometry_not_matching_numeric_ids(tmp_path):
    root = tmp_path / "AcousticRooms"
    metadata = root / "metadata" / "Office" / "Office_idx_0"
    metadata.mkdir(parents=True)
    entries = {
        "S001_R001_hybrid_IR.wav": ([0.0, 0.0, 0.0], [9.0, 0.0, 0.0]),
        "S001_R002_hybrid_IR.wav": ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
        "S002_R001_hybrid_IR.wav": ([10.0, 0.0, 0.0], [9.0, 0.0, 0.0]),
        "S002_R002_hybrid_IR.wav": ([10.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
    }
    for filename, (source, receiver) in entries.items():
        source_index, receiver_index = filename.split("_")[:2]
        (metadata / f"{source_index}_{receiver_index}.json").write_text(
            json.dumps({"src_loc": source, "rec_loc": receiver})
        )
    assert select_near_coincident_contexts(
        root,
        "Office",
        "Office_idx_0",
        list(entries),
    ) == ("S001_R002_hybrid_IR.wav", "S002_R001_hybrid_IR.wav")
