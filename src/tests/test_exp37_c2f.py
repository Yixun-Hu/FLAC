"""exp_37 round B-c: the coarse-to-fine SEQUENTIAL timing pass (plan_loc_efficiency.md §4, §10 "B2 time axis").

The actual deployed path, timed end to end with the bundle's sync placement: the observed RIR encoded once ->
context -> coarse (nested 1.0 m: generation + AGREE encoding + similarity) -> top-k + Chebyshev neighbours ->
refined (only the NEW candidates: generation + encoding + similarity) -> union argmax; seeds keyed by the full
0.5 m index. Its union and winner must follow the registered rule applied to its own scores, and match the offline
readout of a full 0.5 m pass. CPU fakes. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import numpy as np
import pytest
import torch

from exp37_fakes import GRID, FakeModule, _fake_query
from src.localization import exp37_c2f, runner
from src.localization.exp37_grid import c2f_indices, query_scores, subset_readout


class SeparatedRetrieval:
    """Features that vary with the candidate (one sample per latent position of the fake decoder, whose output is
    piecewise constant over 1024 samples), so ranks are far from ULP ties."""

    def compute_audio_features(self, waveforms):
        return torch.nn.functional.normalize(waveforms[:, 0, ::1024] * torch.linspace(1.0, 3.0, 10), dim=1)


@pytest.fixture
def inputs(monkeypatch):
    for module in (runner, exp37_c2f):
        monkeypatch.setattr(module, "load_frozen_query", _fake_query)
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    selected = {"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": len(GRID), "candidate_indices_sha256": "h"}
    return dict(module=FakeModule(), retrieval=SeparatedRetrieval(), selected=selected, record=record, audit={},
                room_base=None, dataset_root=".", device="cpu", candidate_batch_size=8, sample_seed=42, tau=0.1,
                run_sha256="run", score_sample_counts=(1,), synchronize_timing=True, measure_core_forward=True)


@pytest.mark.parametrize("cond_method", ["vanilla", "fa_invariant"])
def test_the_sequential_pass_follows_the_registered_rule_and_matches_the_offline_readout(inputs, cond_method, monkeypatch):
    _full, _cands, full_sims = runner.execute_query(cond_method=cond_method, **inputs)
    scores = query_scores(full_sims, 0.1)
    offline = c2f_indices(GRID, scores, k=5)
    expected = subset_readout(GRID, scores, np.asarray([1.0, 1.0, 1.0]), offline["union"])
    seen = []
    real_seed = exp37_c2f.candidate_seed
    monkeypatch.setattr(exp37_c2f, "candidate_seed", lambda *a: seen.append(int(a[2])) or real_seed(*a))
    result, candidates, sims = exp37_c2f.execute_query_c2f(cond_method=cond_method, top_k_count=5, **inputs)
    c2f = result["c2f"]
    assert c2f["union_indices"] == offline["union"].tolist() and c2f["top_indices"] == offline["top"].tolist()
    assert sorted(seen) == sorted(set(offline["coarse"]) | set(offline["refined"]))         # each candidate once
    assert len(seen) == len(set(seen)) == c2f["generated_count"]
    assert np.array_equal(candidates, GRID[offline["union"]]) and sims.shape == (len(offline["union"]), 1)
    assert np.allclose(sims[:, 0], full_sims[offline["union"], 0], rtol=0.0, atol=1e-6)
    assert result["metrics"]["1"]["prediction_index"] == expected["prediction_index"]
    assert result["metrics"]["1"]["oracle_error_m"] == expected["oracle_error_m"]          # fresh oracle (union)
    assert result["candidate_count"] == len(offline["union"])


def test_the_timing_fields_cover_the_whole_sequential_path(inputs):
    result, _, _ = exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, **inputs)
    seconds = result["latency_seconds"]
    parts = ("observed_encoding", "context_conditioning", "coarse_generation", "coarse_scoring", "selection",
             "refined_generation", "refined_scoring", "union_argmax")
    assert all(seconds[p] >= 0.0 for p in parts)
    assert seconds["c2f_total"] == pytest.approx(sum(seconds[p] for p in parts))
    assert seconds["core_forward_total"] == pytest.approx(
        seconds["context_conditioning"] + seconds["coarse_generation"] + seconds["refined_generation"])
    assert result["latency_protocol"]["name"] == "exp37_c2f_sequential_kctx8_kgen1"
    assert result["latency_protocol"]["top_k"] == 5 and result["latency_protocol"]["chebyshev_radius_m"] == 0.5


def test_the_sequential_pass_needs_the_timers_and_defines_its_own_sets(inputs):
    with pytest.raises(ValueError, match="measure_core_forward"):
        exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, **{**inputs, "measure_core_forward": False})
    with pytest.raises(ValueError, match="candidate_subset"):
        exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, candidate_subset=lambda c: np.arange(3),
                                    **inputs)
    with pytest.raises(ValueError, match="top_k"):
        exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=0, **inputs)


def test_every_selector_component_is_the_median_of_three_on_the_exact_tensors(inputs):
    """Codex BLOCKING 2: one generator execution; observed encoding, coarse scoring, selection, refined scoring and
    union argmax each measured three times (bundle selector protocol), the medians summed into c2f_total."""
    import statistics
    result, _, _ = exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, **inputs)
    repeats, seconds = result["selector_repeat_seconds"], result["latency_seconds"]
    for part in ("observed_encoding", "coarse_scoring", "selection", "refined_scoring", "union_argmax"):
        assert len(repeats[part]) == 3 and seconds[part] == statistics.median(repeats[part])
    assert result["latency_protocol"]["selector_repeat_count"] == 3
    assert "median" in result["latency_protocol"]["selector_statistic"]


class FlakyRetrieval(SeparatedRetrieval):
    """Returns a perturbed embedding on the second call for each batch shape -> unstable scores."""

    def __init__(self):
        self.calls = {}

    def compute_audio_features(self, waveforms):
        key = int(waveforms.shape[0])
        self.calls[key] = self.calls.get(key, 0) + 1
        out = super().compute_audio_features(waveforms)
        return out + 1e-3 if self.calls[key] == 2 else out


def test_unstable_repeats_are_refused(inputs):
    with pytest.raises(RuntimeError, match="repeats"):
        exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, **{**inputs, "retrieval": FlakyRetrieval()})


def test_the_active_candidate_hash_describes_the_union_and_the_parent_is_kept(inputs):
    import hashlib
    result, _, _ = exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, **inputs)
    union = np.asarray(result["c2f"]["union_indices"])
    assert result["candidate_indices_sha256"] == hashlib.sha256(union.astype("<u4").tobytes()).hexdigest()
    assert result["candidate_indices_sha256"] == result["c2f"]["union_indices_sha256"]
    assert result["parent_candidate_indices_sha256"] == "h"                       # the selected full-grid hash
    assert result["c2f"]["chebyshev_radius_m"] == 0.5 and result["c2f"]["top_k"] == 5
    assert result["c2f"]["generated_count"] == result["c2f"]["coarse_count"] + result["c2f"]["new_count"]
