"""exp_37 round B-b: the SEPARATE instrumented stage-split pass (plan_loc_efficiency.md §10 BLOCKING 1).

Stage boundaries: candidate conditioning = batch start -> after ``merge_cached_conditioning`` +
``get_conditioning_inputs``; DiT = ``deterministic_noise`` + the one Euler step; VAE decode = after
``sample_discrete_euler`` -> the decoded, clamped batch; context encoding = the existing context timer. Memory:
absolute per-segment peaks plus a running whole-query maximum (never a sum). The split pass must compute exactly
what the headline ``execute_query`` computes (tested on CPU with a tiny fake model, vanilla and FA).

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""
import numpy as np
import pytest
import torch

from src.localization import runner, stage_split
from src.localization.engine import generate_rir_batch

SPLIT_ONLY = ("stage_seconds", "stage_peak_memory", "stage_sum_seconds")
TIMING = ("elapsed_seconds", "latency_seconds", "peak_memory_bytes", "latency_protocol")


from exp37_fakes import (  # noqa: E402  (shared CPU fakes)
    CANDIDATES, FakeModule, FakeRetrieval, _fake_query,
)




@pytest.fixture
def inputs(monkeypatch):
    for module in (runner, stage_split):
        monkeypatch.setattr(module, "load_frozen_query", _fake_query)
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: CANDIDATES)
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    selected = {"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": 70, "candidate_indices_sha256": "h"}
    return dict(module=FakeModule(), retrieval=FakeRetrieval(), selected=selected, record=record, audit={},
                room_base=None, dataset_root=".", device="cpu", candidate_batch_size=64, sample_seed=42, tau=0.1,
                run_sha256="run", score_sample_counts=(1,), synchronize_timing=True, measure_core_forward=True)


@pytest.mark.parametrize("cond_method", ["vanilla", "fa_invariant"])
def test_the_split_pass_computes_exactly_what_the_headline_computes(inputs, cond_method):
    head, head_c, head_s = runner.execute_query(cond_method=cond_method, **inputs)
    split, split_c, split_s = stage_split.execute_query_stage_split(cond_method=cond_method, **inputs)
    assert np.array_equal(head_c, split_c) and np.array_equal(head_s, split_s)
    assert {k: v for k, v in split.items() if k not in SPLIT_ONLY + TIMING} == \
        {k: v for k, v in head.items() if k not in TIMING}
    assert split["latency_protocol"] == {**head["latency_protocol"], "stage_split": True}
    assert set(split["latency_seconds"]) == set(head["latency_seconds"])
    assert set(split["stage_seconds"]) == {"context_conditioning", "candidate_conditioning", "dit", "vae_decode"}
    assert all(v >= 0 for v in split["stage_seconds"].values())
    assert split["stage_sum_seconds"] == pytest.approx(sum(split["stage_seconds"].values()))
    assert set(split["stage_peak_memory"]) == {"observed_encode", "context_conditioning", "candidate_conditioning",
                                               "dit", "vae_decode", "agree_scoring", "whole_query"}
    assert split["stage_peak_memory"]["whole_query"] is None and split["peak_memory_bytes"] == 0      # CPU


def test_the_staged_generator_equals_the_engine_generator(inputs):
    module = FakeModule()
    items = runner.candidate_metadata(_fake_query(inputs["record"], ".")[1], CANDIDATES[:5], [0.0, 0.5, 1.0])
    source = runner._cache_branch(module, items, ("source_vit", "source"), "cpu", "vanilla")
    context = runner._cache_branch(module, [_fake_query(inputs["record"], ".")[1]],
                                   ("context_poses_vit", "context_poses", "context_audio"), "cpu", "vanilla")
    seeds = [runner.candidate_seed(42, 4467, i, 0) for i in range(5)]
    clock, memory, marks = stage_split.HostClock(), stage_split.PeakMemory("cpu"), {}
    staged = stage_split.generate_rir_batch_staged(module, source, context, seeds, clock=clock, memory=memory,
                                                   marks=marks)
    assert torch.equal(staged, generate_rir_batch(module, source, context, seeds))
    assert list(marks) == ["conditioned", "dit", "decoded"]


def test_peak_memory_keeps_absolute_segment_peaks_and_a_running_maximum():
    counters = {"allocated": 0, "reserved": 0}
    resets = []
    memory = stage_split.PeakMemory("cuda:0", enabled=True, reset=lambda device: resets.append(device),
                                    allocated=lambda device: counters["allocated"],
                                    reserved=lambda device: counters["reserved"])
    for name, allocated in (("context_conditioning", 700), ("dit", 1200), ("vae_decode", 900)):
        memory.begin()
        counters.update(allocated=allocated, reserved=allocated + 100)
        memory.end(name)
    assert memory.segments["dit"] == {"allocated_bytes": 1200, "reserved_bytes": 1300}
    assert memory.query_peak == {"allocated_bytes": 1200, "reserved_bytes": 1300}     # a maximum, never a sum
    assert resets == ["cuda:0"] * 3
    cpu = stage_split.PeakMemory("cpu")
    cpu.begin()
    cpu.end("dit")
    assert cpu.segments["dit"] == {"allocated_bytes": None, "reserved_bytes": None} and cpu.query_peak is None


def test_the_split_pass_needs_the_core_forward_timers(inputs):
    with pytest.raises(ValueError, match="measure_core_forward"):
        stage_split.execute_query_stage_split(cond_method="vanilla", **{**inputs, "measure_core_forward": False})
