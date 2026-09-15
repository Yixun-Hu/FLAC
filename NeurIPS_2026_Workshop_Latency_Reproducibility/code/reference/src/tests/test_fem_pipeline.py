from pathlib import Path

import numpy as np
import pytest
import torch

import src.baselines.fem_pipeline as fem_pipeline
from src.baselines.fem_sabine import ContextT60Estimate
from src.baselines.fem_solver import TetrahedralMesh
from src.baselines.fem_pipeline import (
    load_tetrahedral_mesh_npz,
    run_fem_sabine_forward,
    save_tetrahedral_mesh_npz,
)


def _mesh():
    return TetrahedralMesh(
        nodes=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        elements=np.array([[0, 1, 2, 3]]),
    )


def test_fem_maximum_edge_gate_is_uniformly_022():
    assert fem_pipeline.FEM_MAXIMUM_EDGE_M == pytest.approx(0.22)


def test_tetrahedral_npz_round_trip_is_schema_checked(tmp_path):
    destination = tmp_path / "room_tetra.npz"
    save_tetrahedral_mesh_npz(_mesh(), destination, source_mesh_sha256="abc123")
    mesh, metadata = load_tetrahedral_mesh_npz(destination)
    assert np.array_equal(mesh.nodes, _mesh().nodes)
    assert np.array_equal(mesh.elements, _mesh().elements)
    assert metadata["source_mesh_sha256"] == "abc123"
    assert metadata["schema_version"] == 1

    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, nodes=np.zeros((4, 3)), elements=np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="schema"):
        load_tetrahedral_mesh_npz(invalid)


def test_end_to_end_fem_sabine_forward_returns_waveforms_and_audit(monkeypatch):
    monkeypatch.setattr(
        fem_pipeline,
        "estimate_context_t60",
        lambda _waveforms, **_kwargs: ContextT60Estimate(0.5, (0.5,), 1, 0),
    )
    result = run_fem_sabine_forward(
        _mesh(),
        receiver_point=np.array([0.1, 0.1, 0.1]),
        candidate_points=np.array([[0.2, 0.1, 0.1], [0.1, 0.2, 0.1]]),
        context_waveforms=torch.zeros(1, 1, 9600),
        sample_rate=22050,
        sample_count=320,
        minimum_hz=80.0,
        maximum_hz=300.0,
        unit_gain=0.1,
        maximum_allowed_edge_m=None,
    )
    assert result.waveforms.shape == (2, 1, 320)
    assert result.response.shape == (2, len(result.frequencies_hz))
    assert torch.isfinite(result.waveforms).all()
    assert result.audit["candidate_count"] == 2
    assert result.audit["element_count"] == 1
    assert result.audit["t60_seconds"] == pytest.approx(0.5)
    assert result.audit["frequency_min_hz"] >= 80.0
    assert result.audit["frequency_max_hz"] <= 300.0
    assert result.audit["maximum_relative_solver_residual"] < 1e-10
    assert result.audit["maximum_element_edge_m"] == pytest.approx(2**0.5)
    assert 0 < result.audit["minimum_element_mean_ratio"] <= 1


def test_fem_forward_rejects_candidate_or_receiver_outside_tetrahedral_air_domain(monkeypatch):
    monkeypatch.setattr(
        fem_pipeline,
        "estimate_context_t60",
        lambda _waveforms, **_kwargs: ContextT60Estimate(0.5, (0.5,), 1, 0),
    )
    with pytest.raises(ValueError, match="outside"):
        run_fem_sabine_forward(
            _mesh(),
            receiver_point=np.array([0.1, 0.1, 0.1]),
            candidate_points=np.array([[2.0, 2.0, 2.0]]),
            context_waveforms=torch.zeros(1, 1, 9600),
            sample_count=320,
            maximum_allowed_edge_m=None,
        )


def test_fem_forward_fails_closed_when_mesh_is_too_coarse(monkeypatch):
    monkeypatch.setattr(
        fem_pipeline,
        "estimate_context_t60",
        lambda _waveforms, **_kwargs: ContextT60Estimate(0.5, (0.5,), 1, 0),
    )
    with pytest.raises(ValueError, match="maximum edge"):
        run_fem_sabine_forward(
            _mesh(),
            receiver_point=np.array([0.1, 0.1, 0.1]),
            candidate_points=np.array([[0.2, 0.1, 0.1]]),
            context_waveforms=torch.zeros(1, 1, 9600),
            sample_count=320,
        )


def test_fem_forward_can_skip_waveform_construction_for_sparse_recovery(monkeypatch):
    monkeypatch.setattr(
        fem_pipeline,
        "estimate_context_t60",
        lambda _waveforms, **_kwargs: ContextT60Estimate(0.5, (0.5,), 1, 0),
    )
    result = run_fem_sabine_forward(
        _mesh(),
        receiver_point=np.array([0.1, 0.1, 0.1]),
        candidate_points=np.array([[0.2, 0.1, 0.1]]),
        context_waveforms=torch.zeros(1, 1, 9600),
        sample_count=320,
        maximum_allowed_edge_m=None,
        construct_waveforms=False,
    )

    assert result.waveforms is None
    assert result.response.shape[0] == 1
    assert result.audit["waveform_constructed"] is False
