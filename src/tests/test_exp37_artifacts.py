"""exp_37 fix round (Codex BLOCKING 1): the untimed artifact executor and the artifact REPLAY.

The executor computes exactly what the timed headline computes (same caches, full-grid seeds, batches, generator and
scoring), never times, and dumps each query's scored waveforms; with a reference (a replay of a timing run) it
refuses unless the regenerated similarities equal the timing run's bitwise. A c2f pass is replayed with its own batch
plan (coarse batches, then the new refined ones; output in union order). Rotated (accuracy-only) runs dump inline.
CPU fakes. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import numpy as np
import pytest

from exp37_fakes import GRID, FakeModule, FakeRetrieval, _fake_query
from src.localization import exp37_artifacts as art
from src.localization import exp37_c2f, exp37_rotation, runner
from src.localization.engine import generate_rir_batch
from src.localization.exp37_grid import nested_subset
from src.localization.exp37_waveforms import WaveformDump
from test_exp37_c2f import SeparatedRetrieval


@pytest.fixture
def inputs(monkeypatch):
    for module in (runner, art, exp37_c2f, exp37_rotation):
        monkeypatch.setattr(module, "load_frozen_query", _fake_query)
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    selected = {"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": len(GRID), "candidate_indices_sha256": "h"}
    return dict(module=FakeModule(), retrieval=SeparatedRetrieval(), selected=selected, record=record, audit={},
                room_base=None, dataset_root=".", device="cpu", candidate_batch_size=8, sample_seed=42, tau=0.1,
                run_sha256="run", score_sample_counts=(1,))


def _dump(tmp_path, stem):
    return WaveformDump(tmp_path / "w" / stem, run_stem=stem, root=tmp_path / "w")


@pytest.mark.parametrize("cond_method", ["vanilla", "fa_invariant"])
def test_the_untimed_executor_computes_the_headline_and_dumps_the_scored_tensors(inputs, cond_method, tmp_path):
    _, cands, sims = runner.execute_query(cond_method=cond_method, measure_core_forward=True, synchronize_timing=True,
                                          **inputs)
    dump = _dump(tmp_path, f"a_{cond_method}")
    result, a_cands, a_sims = art.execute_query_artifacts(cond_method=cond_method, dump=dump, **inputs)
    assert np.array_equal(a_cands, cands) and np.array_equal(a_sims, sims)                  # bitwise, same batches
    waves = np.load(dump.dir / result["waveforms"]["path"])
    assert waves.shape == (len(GRID), 1, 10240) and waves.dtype == np.float32
    module = inputs["module"]
    first = runner._cache_branch(module, runner.candidate_metadata(_fake_query(inputs["record"], ".")[1], GRID[:8],
                                                                   [0.0, 0.5, 1.0]),
                                 ("source_vit", "source"), "cpu", cond_method)
    assert result["waveforms"]["sha256"] == dump.files[4467]["sha256"] and "latency_seconds" not in result
    if cond_method == "vanilla":                                             # the exact scored tensor, row for row
        context = runner._cache_branch(module, [_fake_query(inputs["record"], ".")[1]],
                                       ("context_poses_vit", "context_poses", "context_audio"), "cpu", "vanilla")
        seeds = [runner.candidate_seed(42, 4467, i, 0) for i in range(8)]
        assert np.array_equal(waves[:8], generate_rir_batch(module, first, context, seeds).numpy())


def test_a_replay_must_reproduce_the_timing_run_bitwise(inputs, tmp_path):
    _, _, sims = runner.execute_query(cond_method="vanilla", measure_core_forward=True, synchronize_timing=True, **inputs)
    result, _, _ = art.execute_query_artifacts(cond_method="vanilla", dump=_dump(tmp_path, "ok"),
                                               reference=lambda index: sims, **inputs)
    assert result["waveforms"]["path"] == "query_04467.npy"
    tampered = sims.copy()
    tampered[3, 0] = np.nextafter(tampered[3, 0], np.float32(2.0))
    with pytest.raises(RuntimeError, match="bitwise"):
        art.execute_query_artifacts(cond_method="vanilla", dump=_dump(tmp_path, "bad"), reference=lambda index: tampered,
                                    **inputs)


def test_a_nested_subset_replay_keeps_its_subset_order_and_hashes(inputs, tmp_path):
    _, cands, sims = runner.execute_query(cond_method="vanilla", measure_core_forward=True, synchronize_timing=True,
                                          candidate_subset=nested_subset, **inputs)
    dump = _dump(tmp_path, "nested")
    result, a_cands, _ = art.execute_query_artifacts(cond_method="vanilla", dump=dump, candidate_subset=nested_subset,
                                                     reference=lambda index: sims, **inputs)
    assert np.array_equal(a_cands, cands) and result["candidate_indices_sha256"] == result["candidate_subset"]["indices_sha256"]
    assert result["parent_candidate_indices_sha256"] == "h" and dump.files[4467]["candidate_count"] == len(cands)


def test_a_c2f_pass_is_replayed_with_its_own_batch_plan(inputs, tmp_path):
    c2f, _, c2f_sims = exp37_c2f.execute_query_c2f(cond_method="vanilla", top_k_count=5, measure_core_forward=True,
                                                   synchronize_timing=True, **inputs)
    plan = art.c2f_replay_plan(c2f["c2f"])
    result, a_cands, a_sims = art.execute_query_artifacts(cond_method="vanilla", dump=_dump(tmp_path, "c2f"), plan=plan,
                                                          reference=lambda index: c2f_sims, **inputs)
    assert np.array_equal(a_cands, GRID[c2f["c2f"]["union_indices"]]) and np.array_equal(a_sims, c2f_sims)
    assert result["candidate_indices_sha256"] == c2f["candidate_indices_sha256"]


def test_the_executor_is_never_timed(inputs, tmp_path):
    with pytest.raises(ValueError, match="untimed"):
        art.execute_query_artifacts(cond_method="vanilla", dump=_dump(tmp_path, "t"), measure_core_forward=True, **inputs)


def test_a_rotated_run_dumps_its_scored_waveforms_inline(inputs, tmp_path):
    dump = _dump(tmp_path, "rot")
    result, cands, _ = exp37_rotation.execute_query_rotated(cond_method="vanilla", rotate_deg=90.0, dump=dump, **inputs)
    waves = np.load(dump.dir / result["waveforms"]["path"])
    assert waves.shape == (len(cands), 1, 10240) and result["waveforms"]["sha256"] == dump.files[4467]["sha256"]
