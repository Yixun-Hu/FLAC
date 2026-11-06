"""exp_37 round B-c: the B3 rotation contract (plan_loc_efficiency.md §10, Codex HIGH 3).

An immutable ``base_metadata``; ``rotate_scene_metadata(base, θ, 512)`` ONCE for the query / context cache; every
candidate item built from ``base`` and then rotated ONCE; candidate arrays, filtering, truth and metrics in the global
frame. Gates: θ = 0 is the existing headline path (the rotated executor refuses it); θ = 90 is exactly one
128-column roll + one pose rotation for query and candidate metadata (never two). CPU fakes.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
"""
import copy
import math

import numpy as np
import pytest
import torch

from exp37_fakes import GRID, FakeModule, FakeRetrieval, _fake_query
from src.data.yaw_rotation import azimuth_rotation_matrix
from src.localization import exp37_rotation as rot
from src.localization import runner
from src.localization.engine import candidate_metadata


def _base():
    _audio, md = _fake_query({"contexts": ["c"] * 8}, ".")
    md["depth"] = torch.rand(3, 4, 512, generator=torch.Generator().manual_seed(3))     # asymmetric panorama
    return md


def _once(depth, columns):
    R = azimuth_rotation_matrix(2.0 * math.pi * columns / 512)
    return torch.einsum("ij,jhw->ihw", R, torch.roll(depth, shifts=columns, dims=2))


def test_theta_90_rotates_the_query_once_and_leaves_the_base_untouched():
    base = _base()
    frozen = copy.deepcopy(base)
    query = rot.rotated_query_metadata(base, 90.0)
    assert torch.equal(query["depth"], _once(base["depth"], 128)) and not torch.equal(query["depth"], _once(base["depth"], 256))
    R = azimuth_rotation_matrix(math.pi / 2)
    assert torch.equal(query["context_poses"], torch.einsum("ij,...j->...i", R, base["context_poses"]))
    assert torch.equal(query["context_audio"], base["context_audio"])
    assert all(torch.equal(base[k], frozen[k]) for k in frozen if isinstance(frozen[k], torch.Tensor))


def test_theta_90_candidate_items_are_built_from_the_base_and_rotated_once():
    base = _base()
    receiver = np.array([0.0, 0.5, 1.0])
    items = rot.rotated_candidate_items(base, GRID[:3], receiver, 90.0)
    plain = candidate_metadata(base, GRID[:3], receiver)
    R = azimuth_rotation_matrix(math.pi / 2)
    query = rot.rotated_query_metadata(base, 90.0)
    for item, ref in zip(items, plain):
        assert torch.equal(item["source"], torch.einsum("ij,...j->...i", R, ref["source"]))       # R (c - r), once
        assert torch.equal(item["source_vit"], torch.einsum("ij,...j->...i", R, ref["source_vit"]))
        assert torch.equal(item["depth"], query["depth"]) and torch.equal(item["context_poses"], query["context_poses"])


def test_theta_0_candidate_items_equal_the_existing_items():
    base = _base()
    items = rot.rotated_candidate_items(base, GRID[:3], [0.0, 0.5, 1.0], 0.0)
    for item, ref in zip(items, candidate_metadata(base, GRID[:3], [0.0, 0.5, 1.0])):
        assert all(torch.equal(item[k], ref[k]) for k in ref if isinstance(ref[k], torch.Tensor))


@pytest.fixture
def inputs(monkeypatch):
    for module in (runner, rot):
        monkeypatch.setattr(module, "load_frozen_query", lambda record, root: (_fake_query(record, root)[0], _base()))
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    selected = {"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": len(GRID), "candidate_indices_sha256": "h"}
    return dict(module=FakeModule(), retrieval=FakeRetrieval(), selected=selected, record=record, audit={},
                room_base=None, dataset_root=".", device="cpu", candidate_batch_size=16, sample_seed=42, tau=0.1,
                run_sha256="run", score_sample_counts=(1,))


class NormConditioner:
    """Exactly yaw-invariant fake: per pose key the horizontal norms and heights only."""

    def __call__(self, metadata, device, only_ids=None):
        out = {}
        for key in only_ids:
            values = []
            for item in metadata:
                v = item[key].float().reshape(-1, item[key].shape[-1])
                values.append(torch.cat([v[:, :2].norm(dim=-1)[:1], v[:, -1][:1], v.reshape(-1)[:1] * 0 + v.shape[0]]))
            out[key] = [torch.stack(values).unsqueeze(1).to(device), torch.ones(len(metadata), 1, device=device)]
        return out


def test_an_invariant_model_scores_theta_90_like_theta_0_and_metrics_stay_global(inputs):
    module = FakeModule()
    module.diffusion.conditioner = NormConditioner()
    head, head_c, head_s = runner.execute_query(cond_method="vanilla", **{**inputs, "module": module})
    turned, turned_c, turned_s = rot.execute_query_rotated(cond_method="vanilla", rotate_deg=90.0,
                                                           **{**inputs, "module": module})
    assert np.array_equal(turned_c, head_c) and np.allclose(turned_s, head_s, rtol=0.0, atol=1e-6)
    assert turned["source_global"] == head["source_global"] and turned["rotation"]["column_shift"] == 128
    variant = rot.execute_query_rotated(cond_method="vanilla", rotate_deg=90.0, **inputs)[2]
    assert not np.allclose(variant, runner.execute_query(cond_method="vanilla", **inputs)[2], atol=1e-6)


def test_theta_0_timing_and_off_column_angles_are_refused(inputs):
    with pytest.raises(ValueError, match="headline"):
        rot.execute_query_rotated(cond_method="vanilla", rotate_deg=0.0, **inputs)
    with pytest.raises(ValueError, match="column"):
        rot.execute_query_rotated(cond_method="vanilla", rotate_deg=10.0, **inputs)
    with pytest.raises(ValueError, match="accuracy"):
        rot.execute_query_rotated(cond_method="vanilla", rotate_deg=90.0, measure_core_forward=True, **inputs)
