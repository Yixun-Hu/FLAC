"""exp_37 round B-c: the headline ``execute_query``'s two documented plumbing parameters (plan §4, §5, §10).

- ``frame_angles`` (B3 orbit sizes {0}, {0,180}, C4; fa_invariant only; default = the bundle's C4);
- ``candidate_subset`` (the dedicated nested-1.0 m timing pass): a callable returning FULL 0.5 m indices, so the
  generation seeds keep the full-grid index (common random numbers with the 0.5 m run).
The bundle's timed code is otherwise unchanged (test_exp37_headline_timer_guard.py pins exactly these two deltas).
CPU fakes; per-row fake model, so a subset's rows must equal the full run's rows bit for bit.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-c.
"""
import hashlib

import numpy as np
import pytest

from exp37_fakes import GRID, FakeModule, FakeRetrieval, _fake_query
from src.localization import runner
from src.localization.exp37_grid import nested_subset

C4 = (0.0, 90.0, 180.0, 270.0)


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(runner, "load_frozen_query", _fake_query)
    monkeypatch.setattr(runner, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    selected = {"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": len(GRID), "candidate_indices_sha256": "h"}
    return dict(module=FakeModule(), retrieval=FakeRetrieval(), selected=selected, record=record, audit={},
                room_base=None, dataset_root=".", device="cpu", candidate_batch_size=16, sample_seed=42, tau=0.1,
                run_sha256="run", score_sample_counts=(1,), synchronize_timing=True, measure_core_forward=True)


def _run(inputs, **kw):
    return runner.execute_query(**{**inputs, **kw})


def test_the_default_frame_angles_are_the_bundle_c4(inputs):
    default, _, sims = _run(inputs, cond_method="fa_invariant")
    explicit, _, sims_c4 = _run(inputs, cond_method="fa_invariant", frame_angles=C4)
    assert np.array_equal(sims, sims_c4) and default["metrics"] == explicit["metrics"]


def test_smaller_orbits_change_only_the_fa_vit_average(inputs):
    sims = {angles: _run(inputs, cond_method="fa_invariant", frame_angles=angles)[2]
            for angles in (C4, (0.0, 180.0), (0.0,))}
    assert not np.array_equal(sims[C4], sims[(0.0,)]) and not np.array_equal(sims[(0.0,)], sims[(0.0, 180.0)])


def test_frame_angles_are_refused_for_vanilla_and_must_start_at_zero(inputs):
    with pytest.raises(ValueError, match="frame"):
        _run(inputs, cond_method="vanilla", frame_angles=(0.0,))
    with pytest.raises(ValueError, match="0-degree"):
        _run(inputs, cond_method="fa_invariant", frame_angles=(90.0,))


@pytest.mark.parametrize("cond_method", ["vanilla", "fa_invariant"])
def test_the_nested_subset_keeps_full_grid_common_random_numbers(inputs, cond_method, monkeypatch):
    """The seeds are keyed by the FULL 0.5 m index (exact); the similarities then agree up to batch-composition
    rounding (BLAS blocking differs with the batch -- ULP level here; the same effect as the bundle's 61/64 Vanilla
    rerun), which is why the 1.0 m ACCURACY of record is the offline readout of the 0.5 m run's scores."""
    seeds = []
    real_seed = runner.candidate_seed
    monkeypatch.setattr(runner, "candidate_seed", lambda *a: seeds.append(int(a[2])) or real_seed(*a))
    full, full_c, full_s = _run(inputs, cond_method=cond_method)
    full_seeds, seeds[:] = list(seeds), []
    sub, sub_c, sub_s = _run(inputs, cond_method=cond_method, candidate_subset=nested_subset)
    index = nested_subset(GRID)
    assert full_seeds == list(range(len(GRID))) and seeds == index.tolist()
    assert np.array_equal(sub_c, full_c[index]) and np.allclose(sub_s, full_s[index], rtol=0.0, atol=1e-6)
    assert sub["candidate_count"] == len(index) and full["candidate_count"] == len(GRID)
    assert sub["candidate_subset"] == {"name": "nested_subset", "full_candidate_count": len(GRID),
                                       "indices_sha256": hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()}
    assert "candidate_subset" not in full


def test_a_subset_that_is_not_full_grid_indices_is_refused(inputs):
    with pytest.raises(ValueError, match="candidate_subset"):
        _run(inputs, cond_method="vanilla", candidate_subset=lambda candidates: np.array([5, 2]))      # unsorted
    with pytest.raises(ValueError, match="candidate_subset"):
        _run(inputs, cond_method="vanilla", candidate_subset=lambda candidates: np.array([], dtype=int))


def test_a_subset_result_s_active_hash_is_the_subset_s_and_the_parent_is_kept(inputs):
    """HIGH 2 consistency: the active candidate hash describes the scored set; the full grid's hash is the parent."""
    sub, _, _ = _run(inputs, cond_method="vanilla", candidate_subset=nested_subset)
    index = nested_subset(GRID)
    assert sub["candidate_indices_sha256"] == hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()
    assert sub["parent_candidate_indices_sha256"] == "h"
    full, _, _ = _run(inputs, cond_method="vanilla")
    assert full["candidate_indices_sha256"] == "h" and "parent_candidate_indices_sha256" not in full
