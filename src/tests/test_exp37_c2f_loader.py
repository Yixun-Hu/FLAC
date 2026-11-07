"""exp_37 fix round (Codex HIGH 2): ``load_c2f_run`` re-verifies every c2f record against its parent 0.5 m run --
the union (sorted, unique, in range, its sha256 = the active candidate hash), the parent hash, the stored candidates
= the parent's ``full[union]``, the top-k recomputed from the stored coarse scores, the Chebyshev-0.5 m refinement,
the union, the generated / union counts, the winner and the full K=1 block, the radius and the median-of-three
latency fields. Both runs are produced by the real executors on CPU fakes.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import json

import numpy as np
import pytest

from exp37_fakes import GRID, FakeModule, _fake_query, example_identity
from src.localization import exp37_c2f, runner
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result
from test_exp37_c2f import SeparatedRetrieval

AUDIT = {"sha256": "a" * 64, "mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}


def _pilot():
    records = [{"index": 4467, "query_id": "q.wav", "scene": "S", "room": "R", "receiver_id": "R017",
                "candidate_count": len(GRID), "candidate_indices_sha256": "h"}]
    body = {"schema_version": 1, "query_count": 1, "records": records}
    return {**body, "sha256": canonical_sha256(body)}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    for module in (runner, exp37_c2f):
        monkeypatch.setattr(module, "load_frozen_query", _fake_query)
        monkeypatch.setattr(module, "filter_frozen_query_candidates", lambda record, audit, base: GRID)
    pilot = _pilot()
    record = {"index": 4467, "contexts": ["c"] * 8, "source_global": [1.0, 1.0, 1.0], "receiver_global": [0.0, 0.5, 1.0]}
    common = dict(module=FakeModule(), retrieval=SeparatedRetrieval(), selected=pilot["records"][0], record=record,
                  audit={}, room_base=None, dataset_root=".", device="cpu", candidate_batch_size=8, sample_seed=42,
                  tau=0.1, score_sample_counts=(1,), synchronize_timing=True, measure_core_forward=True,
                  cond_method="vanilla")
    out = {}
    for name, extra, execute in (("parent", {}, runner.execute_query),
                                 ("c2f", {"candidate_subset": "c2f", "c2f_top_k": 5},
                                  lambda **kw: exp37_c2f.execute_query_c2f(top_k_count=5, **kw))):
        expected = ident.registered_identity("P1", selection=pilot, audit=AUDIT, seed=42, timing=True, **extra)
        manifest = initialize_run(tmp_path / name, example_identity(expected, AUDIT))
        result, candidates, sims = execute(run_sha256=manifest["sha256"], **common)
        save_query_result(tmp_path / name, result=result, candidates=candidates, similarities=sims)
        out[name] = tmp_path / name
    return pilot, out


def _load(pilot, runs):
    return lat.load_c2f_run(runs["c2f"], arm="P1", selection=pilot, audit=AUDIT, top_k=5, parent_run_dir=runs["parent"])


def test_a_faithful_c2f_run_is_verified(runs):
    pilot, dirs = runs
    row = _load(pilot, dirs)[4467]
    assert row["c2f_total"] > 0 and row["generated_count"] >= row["union_count"] > 0


def _tamper(path, change, arrays=None):
    payload = json.loads(path.read_text())
    change(payload)
    if arrays is not None:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        from src.localization.runner import file_sha256
        payload["arrays_sha256"] = file_sha256(path.with_suffix(".npz"))
    payload["sha256"] = canonical_sha256({k: v for k, v in payload.items() if k != "sha256"})
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("change, needle", [
    (lambda p: p.update(candidate_indices_sha256="0" * 64), "hash"),
    (lambda p: p.update(parent_candidate_indices_sha256="0" * 64), "parent"),
    (lambda p: p["c2f"].update(top_indices=p["c2f"]["top_indices"][::-1]), "top"),
    (lambda p: p["c2f"].update(generated_count=p["c2f"]["generated_count"] + 1), "count"),
    (lambda p: p["c2f"].update(winner_full_index=-1), "winner"),
    (lambda p: p["latency_protocol"].update(chebyshev_radius_m=0.75), "radius|protocol"),   # the exact block (r2)
    (lambda p: p["c2f"].update(union_indices=p["c2f"]["union_indices"][::-1]), "union|subset"),  # its hash (r2)
    (lambda p: p["latency_seconds"].update(c2f_total=0.0), "c2f_total"),
], ids=["active-hash", "parent-hash", "top", "generated", "winner", "radius", "union-order", "total"])
def test_a_tampered_c2f_record_is_refused(runs, change, needle):
    pilot, dirs = runs
    _tamper(dirs["c2f"] / "queries" / "query_04467.json", change)
    with pytest.raises(RuntimeError, match=needle):
        _load(pilot, dirs)


def test_stored_candidates_that_are_not_the_parent_s_union_rows_are_refused(runs):
    pilot, dirs = runs
    path = dirs["c2f"] / "queries" / "query_04467.json"
    with np.load(path.with_suffix(".npz")) as z:
        arrays = {"candidates": z["candidates"] + np.float32(0.5), "similarities": z["similarities"]}
    _tamper(path, lambda p: None, arrays=arrays)
    with pytest.raises(RuntimeError, match="candidates"):
        _load(pilot, dirs)
