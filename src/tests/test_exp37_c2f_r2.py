"""exp_37 fix round 2 (Codex code review 2, BLOCKING 3 / HIGH 5): the c2f point's accuracy comes from SEQUENTIAL c2f
runs for all three seeds, and its timing evidence has an exact schema.

- ``execute_query_artifacts(c2f_top_k=k)`` is the timed c2f pass without timers or repeats (same coarse batches, the
  same top-k / Chebyshev refinement from its own coarse scores, the same new-candidate batches, union order): its
  candidates, similarities and its ``c2f`` record equal the timed pass's bitwise; it dumps the union rows inline.
- The runner runs such an untimed c2f ACCURACY pass only with a dump; an untimed nested pass stays refused (the nested
  accuracy is the offline readout of the parent's fixed subset).
- ``load_c2f_run(timing=False)`` verifies an accuracy-only c2f run against its parent (the same arm / seed 0.5 m run);
  a timing record needs the EXACT repeat schema (the five selector parts, three finite nonnegative samples each, every
  median recomputed), finite nonnegative latency components, ``c2f_total`` = their sum and ``core_forward_total`` =
  context + both generations, and the exact protocol block.
CPU fakes and the real executors. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import math

import numpy as np
import pytest

from exp37_fakes import GRID, FakeModule, _fake_query, example_identity
from src.localization import exp37_artifacts as art
from src.localization import exp37_c2f, runner
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.exp37_waveforms import WaveformDump
from src.localization.runner import initialize_run, save_query_result
from test_exp37_c2f import SeparatedRetrieval
from test_exp37_c2f_loader import AUDIT, _load, _pilot, _tamper, runs  # noqa: F401


@pytest.fixture
def common(monkeypatch):
    for module in (runner, art, exp37_c2f):
        monkeypatch.setattr(module, "load_frozen_query", _fake_query)
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    pilot = _pilot()
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    return pilot, dict(module=FakeModule(), retrieval=SeparatedRetrieval(), selected=pilot["records"][0], record=record,
                       audit={}, room_base=None, dataset_root=".", device="cpu", candidate_batch_size=8, tau=0.1,
                       score_sample_counts=(1,), cond_method="vanilla")


def test_the_untimed_c2f_run_is_the_timed_pass_without_timers(common, tmp_path):
    _pilot_, kw = common
    timed, cands, sims = exp37_c2f.execute_query_c2f(top_k_count=5, measure_core_forward=True, synchronize_timing=True,
                                                     sample_seed=43, run_sha256="t", **kw)
    dump = WaveformDump(tmp_path / "w" / "acc", run_stem="acc", root=tmp_path / "w")
    acc, a_cands, a_sims = art.execute_query_artifacts(dump=dump, c2f_top_k=5, sample_seed=43, run_sha256="a", **kw)
    assert np.array_equal(a_cands, cands) and np.array_equal(a_sims, sims) and acc["c2f"] == timed["c2f"]
    assert (acc["candidate_indices_sha256"], acc["parent_candidate_indices_sha256"]) == (
        timed["candidate_indices_sha256"], timed["parent_candidate_indices_sha256"])
    assert "latency_seconds" not in acc and "selector_repeat_seconds" not in acc
    assert dump.files[4467]["candidate_count"] == len(timed["c2f"]["union_indices"])


def _acc_runs(tmp_path, common, seed=43):
    """An accuracy-only parent (full, inline dump) and an accuracy-only c2f run of the same seed, as the runner writes
    them (registered untimed identities)."""
    pilot, kw = common
    out = {}
    for name, extra, execute in (("parent", {}, lambda **k: art.execute_query_artifacts(**k)),
                                 ("c2f", {"candidate_subset": "c2f", "c2f_top_k": 5},
                                  lambda **k: art.execute_query_artifacts(c2f_top_k=5, **k))):
        want = ident.registered_identity("P1", selection=pilot, audit=AUDIT, seed=seed, timing=False,
                                         waveform_mode="inline", **extra)
        identity = example_identity(want, AUDIT)
        identity["waveform_dump"] = {"mode": "inline", "dir": str(tmp_path / "w" / name)}
        manifest = initialize_run(tmp_path / name, identity)
        dump = WaveformDump(tmp_path / "w" / name, run_stem=name, root=tmp_path / "w")
        result, cands, sims = execute(dump=dump, sample_seed=seed, run_sha256=manifest["sha256"], **kw)
        save_query_result(tmp_path / name, result=result, candidates=cands, similarities=sims)
        dump.finish(run_manifest_sha256=manifest["sha256"], identity=identity, mode="inline", query_indices=[4467])
        out[name] = tmp_path / name
    return pilot, out


def test_an_accuracy_only_c2f_run_is_verified_against_its_seed_s_parent(common, tmp_path):
    pilot, dirs = _acc_runs(tmp_path, common)
    rows = lat.load_c2f_run(dirs["c2f"], arm="P1", selection=pilot, audit=AUDIT, top_k=5, parent_run_dir=dirs["parent"],
                            seed=43, timing=False)
    assert rows[4467]["union_count"] > 0 and rows[4467]["c2f_total"] is None
    with pytest.raises(RuntimeError, match="sample_seed|timing|measure_core_forward"):
        lat.load_c2f_run(dirs["c2f"], arm="P1", selection=pilot, audit=AUDIT, top_k=5, parent_run_dir=dirs["parent"])


@pytest.mark.parametrize("change, needle", [
    (lambda p: p["selector_repeat_seconds"].pop("selection"), "repeat"),
    (lambda p: p["selector_repeat_seconds"].update(extra=[0.1, 0.1, 0.1]), "repeat"),
    (lambda p: p["selector_repeat_seconds"].update(selection=[0.1, 0.1]), "repeat"),
    (lambda p: p["selector_repeat_seconds"].update(selection=[0.1, -0.1, 0.1]), "repeat"),
    (lambda p: p["selector_repeat_seconds"].update(selection=[0.1, math.nan, 0.1]), "repeat"),
    (lambda p: p["latency_seconds"].update(selection=p["latency_seconds"]["selection"] + 1e-3), "median"),
    (lambda p: p["latency_seconds"].update(context_conditioning=-1.0), "latency"),
    (lambda p: p["latency_seconds"].update(core_forward_total=p["latency_seconds"]["core_forward_total"] + 1.0),
     "core_forward_total"),
    (lambda p: p["latency_protocol"].pop("selector_statistic"), "protocol"),
], ids=["missing-part", "extra-part", "two-samples", "negative", "nan", "median", "component", "core", "protocol"])
def test_the_c2f_timing_evidence_has_an_exact_schema(runs, change, needle):
    pilot, dirs = runs
    _tamper(dirs["c2f"] / "queries" / "query_04467.json", change)
    with pytest.raises(RuntimeError, match=needle):
        _load(pilot, dirs)


def test_the_runner_runs_an_untimed_c2f_pass_only_as_an_accuracy_dump(tmp_path, monkeypatch):
    import json
    from src.localization import exp37_waveforms
    from test_exp37_runner_wiring import E9, SELECTION_128, _run  # noqa: F401
    from src.localization.ar_queries import load_context_manifest
    from src.localization.pilot import load_pilot_manifest
    frozen = (load_context_manifest(E9 / "context_manifest_exp01_seed42.json"),
              json.loads((E9 / "geometry_audit.json").read_text()), load_pilot_manifest(SELECTION_128))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(runner, "load_flac_module", lambda *a: (object(), {}))
    monkeypatch.setattr(runner, "load_agree_retrieval", lambda *a: object())
    monkeypatch.setattr(runner, "reconstruct_room_base_candidates", lambda room, audit, mesh_path=None: np.zeros((0, 3)))
    monkeypatch.setattr(runner, "write_host_record", lambda *a, **k: None)
    monkeypatch.setattr(runner, "weights_footprint", lambda device: {"memory_allocated_bytes": 1})
    monkeypatch.setattr(exp37_waveforms, "DUMP_ROOT", tmp_path / "waves")
    calls = []

    def stub(**kw):
        calls.append(kw)
        n = int(kw["selected"]["candidate_count"])
        kw["dump"].write(int(kw["record"]["index"]), np.zeros((n, 1, 10240), np.float32), np.arange(n))
        return {"schema_version": 1, "run_manifest_sha256": kw["run_sha256"], "query_index": kw["selected"]["index"],
                "query_id": kw["selected"]["query_id"], "candidate_count": n, "elapsed_seconds": 0.0}, \
            np.zeros((n, 3), np.float32), np.zeros((n, 1), np.float32)
    monkeypatch.setattr(art, "execute_query_artifacts", stub)
    with pytest.raises(ValueError, match="measure-core-forward|timing"):
        _run(frozen, tmp_path, query_index=4467, candidate_subset="c2f")
    with pytest.raises(ValueError, match="measure-core-forward|timing"):
        _run(frozen, tmp_path, query_index=4467, candidate_subset="nested_1m", dump_waveforms=tmp_path / "waves" / "run")
    assert not (tmp_path / "run").exists()
    _run(frozen, tmp_path, query_index=4467, candidate_subset="c2f", c2f_top_k=5, dump_waveforms=tmp_path / "waves" / "run")
    assert calls[0]["c2f_top_k"] == 5 and calls[0]["plan"] is None and calls[0]["candidate_subset"] is None
    identity = json.loads((tmp_path / "run" / "run_manifest.json").read_text())["identity"]
    assert identity["candidate_subset"] == "c2f" and identity["waveform_dump"]["mode"] == "inline"
