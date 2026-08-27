import json

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import src.localization.baseline_experiment as baseline_experiment
from src.baselines.fem_pipeline import save_tetrahedral_mesh_npz
from src.baselines.fem_solver import TetrahedralMesh
from src.localization.baseline_experiment import (
    build_baseline_query_result,
    load_few_shot_waveform_checkpoint,
    load_room_tetrahedral_mesh,
    execute_baseline_query,
)
from src.localization.runner import file_sha256
from src.models import create_model_from_config


def _model_config():
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
            "loss": {"fft_sizes": [64]},
        },
    }


def test_few_shot_checkpoint_loader_accepts_lightning_prefix_and_is_strict(tmp_path):
    config = _model_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    expected = create_model_from_config(config)
    checkpoint = {
        "state_dict": {f"model.{key}": value for key, value in expected.state_dict().items()},
        "model_config": config,
    }
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, checkpoint_path)

    loaded, loaded_config = load_few_shot_waveform_checkpoint(
        config_path, checkpoint_path, "cpu"
    )

    assert loaded_config == config
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for key, value in expected.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)

    checkpoint["state_dict"].pop(next(iter(checkpoint["state_dict"])))
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError):
        load_few_shot_waveform_checkpoint(config_path, checkpoint_path, "cpu")

    checkpoint["state_dict"] = {
        f"model.{key}": value for key, value in expected.state_dict().items()
    }
    checkpoint["model_config"] = {**config, "sample_rate": 16000}
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="embedded model config"):
        load_few_shot_waveform_checkpoint(config_path, checkpoint_path, "cpu")


def test_room_tetra_mesh_must_match_official_surface_mesh_hash(tmp_path):
    mesh = TetrahedralMesh(
        nodes=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        elements=np.array([[0, 1, 2, 3]]),
    )
    mesh_path = tmp_path / "room.npz"
    save_tetrahedral_mesh_npz(mesh, mesh_path, source_mesh_sha256="official")
    manifest = {
        "schema_version": 1,
        "rooms": {
            "Room_1": {"path": str(mesh_path), "npz_sha256": file_sha256(mesh_path)}
        },
    }
    geometry_audit = {"rooms": {"Room_1": {"mesh_sha256": "official"}}}

    loaded, metadata = load_room_tetrahedral_mesh(
        "Room_1", manifest, geometry_audit
    )
    assert np.array_equal(loaded.nodes, mesh.nodes)
    assert metadata["source_mesh_sha256"] == "official"

    geometry_audit["rooms"]["Room_1"]["mesh_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="surface mesh hash"):
        load_room_tetrahedral_mesh("Room_1", manifest, geometry_audit)

    geometry_audit["rooms"]["Room_1"]["mesh_sha256"] = "official"
    mesh_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="NPZ hash"):
        load_room_tetrahedral_mesh("Room_1", manifest, geometry_audit)


def test_baseline_query_result_reports_k1_k8_and_stable_candidate_tie_break():
    candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    candidate_scores = torch.tensor([[0.5, 0.7], [0.5, 0.6]])
    result = build_baseline_query_result(
        query_index=7,
        query_id="Room/query.wav",
        scene="scene",
        room="Room_1",
        receiver_id="R006",
        candidates=candidates,
        source_global=np.zeros(3),
        receiver_global=np.ones(3),
        candidate_scores=candidate_scores,
        context_counts=(1, 8),
        random_seed=42,
        elapsed_seconds=1.5,
        diagnostics={"fem": {"valid": True}},
    )

    assert result["context_counts"] == [1, 8]
    assert result["receiver_id"] == "R006"
    assert result["metrics"]["1"]["prediction_index"] == 0
    assert result["metrics"]["8"]["prediction_index"] == 0
    assert result["metrics"]["1"]["localization_error_m"] == 0.0
    assert result["diagnostics"]["fem"]["valid"] is True


def test_fem_query_selects_with_room_helps_without_loading_agree(monkeypatch):
    sample_count = 10240
    bins = np.array([40, 41, 42])
    response = torch.tensor(
        [[1 + 0j, 0.2 + 0.1j, -0.5j], [0.1j, 1 - 0.2j, 0.3 + 0.4j]]
    )
    spectrum = torch.zeros(sample_count // 2 + 1, dtype=torch.complex64)
    spectrum[bins] = (1.3 + 0.4j) * response[1]
    observed = torch.fft.irfft(spectrum, n=sample_count).reshape(1, sample_count)
    metadata = {"context_audio": torch.zeros(8, 1, 9600)}
    candidates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    monkeypatch.setattr(
        baseline_experiment,
        "load_frozen_query",
        lambda _record, _root: (observed, metadata),
    )
    monkeypatch.setattr(
        baseline_experiment,
        "filter_frozen_query_candidates",
        lambda _record, _audit, _base: candidates,
    )
    monkeypatch.setattr(
        baseline_experiment,
        "run_fem_sabine_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            response=response,
            bin_indices=bins,
            audit={"maximum_relative_solver_residual": 1e-12},
        ),
    )
    selected = {
        "index": 7,
        "query_id": "Room/query.wav",
        "scene": "scene",
        "room": "Room_1",
        "receiver_id": 0,
        "candidate_count": 2,
        "candidate_indices_sha256": "candidate-hash",
    }
    record = {
        "contexts": [f"context-{index}" for index in range(8)],
        "source_global": [1.0, 0.0, 0.0],
        "receiver_global": [0.0, 0.0, 0.0],
    }

    result, returned_candidates, candidate_scores = execute_baseline_query(
        method="fem_sabine",
        predictor=None,
        retrieval=None,
        selected=selected,
        record=record,
        geometry_audit={},
        room_base=candidates,
        tetrahedral_mesh=TetrahedralMesh(
            nodes=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            elements=np.array([[0, 1, 2, 3]]),
        ),
        dataset_root="unused",
        device="cpu",
        candidate_batch_size=2,
        random_seed=42,
        run_sha256="run",
    )

    assert np.array_equal(returned_candidates, candidates)
    assert candidate_scores.shape == (2, 2)
    assert result["metrics"]["1"]["prediction_index"] == 1
    assert result["metrics"]["8"]["prediction_index"] == 1
    assert result["diagnostics"]["1"]["sparse_recovery"]["support"] == [1]
    assert result["diagnostics"]["1"]["selection_rule"] == "room_helps_pulse_stacked_omp"
    assert result["candidate_score_name"] == "room_helps_projection_fraction"
