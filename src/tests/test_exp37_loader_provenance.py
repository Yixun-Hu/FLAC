"""exp_37 fix round (HIGH 1): the summary loaders enforce the full registered identity and the selector provenance.

``load_arm_run`` / ``load_c2f_run`` compare a run's manifest identity with ``exp37_identity.registered_identity``
(seed, AGREE, model config, sampler, launcher, mesh map, ...); ``load_selector`` checks the selector's AGREE and
context digests and accepts a different selection only when it is a REGISTERED superset whose records contain the
requested ones identically. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
"""
import json

import numpy as np
import pytest

from exp37_fakes import example_identity
from src.localization import exp37_identity as ident
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256
from src.localization.runner import initialize_run, save_query_result

AUDIT = {"sha256": "a" * 64, "mesh_root": "/m", "rooms": {"R": {"mesh_sha256": "b" * 64}}}


def _selection(counts, extra=None):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c, "candidate_indices_sha256": f"h{i}"}
               for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records, **(extra or {})}
    return {**body, "sha256": canonical_sha256(body)}


def _run(tmp_path, name, selection, *, arm="P1", seed=42, over=None):
    expected = ident.registered_identity(arm, selection=selection, audit=AUDIT, seed=seed, timing=True)
    run = tmp_path / name
    manifest = initialize_run(run, {**example_identity(expected, AUDIT), **(over or {})})
    for r in selection["records"]:
        n = r["candidate_count"]
        result = {"run_manifest_sha256": manifest["sha256"], "query_index": r["index"], "query_id": r["query_id"],
                  "candidate_count": n, "peak_memory_bytes": 1, "candidate_indices_sha256": r["candidate_indices_sha256"],
                  "latency_protocol": {"name": "fem_core_aligned_kctx8_kgen1", "context_count": 8},
                  "latency_seconds": {"context_conditioning": 0.1, "candidate_conditioning_and_generation": 0.2,
                                      "core_forward_total": 0.3}}
        save_query_result(run, result=result, candidates=np.zeros((n, 3)), similarities=np.zeros((n, 1)))
    return run


def test_a_registered_run_loads(tmp_path):
    selection = _selection({1: 3, 2: 4})
    rows = lat.load_arm_run(_run(tmp_path, "ok", selection), arm="P1", selection=selection, audit=AUDIT, seed=42)
    assert sorted(rows) == [1, 2]


@pytest.mark.parametrize("over, needle", [
    ({"sample_seed": 43}, "sample_seed"), ({"agree_checkpoint_sha256": "x"}, "agree_checkpoint_sha256"),
    ({"model_config_sha256": "x"}, "model_config_sha256"), ({"sampler": {"type": "x"}}, "sampler"),
    ({"no_flash_launch": {"overlay_sha256": ident.OVERLAY_SHA256, "launcher_sha256": "x", "sentinel": ident.SENTINEL}},
     "no_flash_launch"),
    ({"mesh_root_override": None}, "mesh_root_override"), ({"context_manifest_sha256": "x"}, "context_manifest_sha256")])
def test_any_unregistered_field_is_refused(tmp_path, over, needle):
    selection = _selection({1: 3})
    with pytest.raises(RuntimeError, match=needle):
        lat.load_arm_run(_run(tmp_path, "bad", selection, over=over), arm="P1", selection=selection, audit=AUDIT, seed=42)


def test_the_registered_seed_is_required(tmp_path):
    selection = _selection({1: 3})
    run = _run(tmp_path, "s43", selection, seed=43)
    with pytest.raises(RuntimeError, match="sample_seed"):
        lat.load_arm_run(run, arm="P1", selection=selection, audit=AUDIT, seed=42)
    assert lat.load_arm_run(run, arm="P1", selection=selection, audit=AUDIT, seed=43)


def _selector(tmp_path, selection, **over):
    rows = [{"query_index": r["index"], "query_id": r["query_id"], "room": r["room"], "candidate_count": r["candidate_count"],
             "candidate_indices_sha256": r["candidate_indices_sha256"],
             "agree": {"median_seconds": {"scoring_total_seconds": 0.5}}} for r in selection["records"]]
    body = {"record_type": "exp37_agree_selector_latency", "repeat_count": 3, "candidate_batch_size": 64,
            "selection_sha256": selection["sha256"], "context_manifest_sha256": ident.CONTEXT_MANIFEST_SHA256,
            "agree_checkpoint_file_sha256": ident.AGREE_SHA256, "queries": rows, **over}
    path = tmp_path / f"selector_{len(rows)}_{len(over)}.json"
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


def test_the_selector_must_be_the_registered_agree_and_context(tmp_path):
    selection = _selection({1: 3, 2: 4})
    assert sorted(lat.load_selector(_selector(tmp_path, selection), selection)) == [1, 2]
    for over, needle in (({"agree_checkpoint_file_sha256": "x"}, "AGREE"), ({"context_manifest_sha256": "x"}, "context")):
        with pytest.raises(RuntimeError, match=needle):
            lat.load_selector(_selector(tmp_path, selection, **over), selection)


def test_another_selection_is_accepted_only_as_a_registered_superset(tmp_path, monkeypatch):
    big = _selection({1: 3, 2: 4, 3: 5})
    small = _selection({1: 3, 3: 5})
    with pytest.raises(RuntimeError, match="superset"):
        lat.load_selector(_selector(tmp_path, big), small)
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], (big["sha256"],))
    assert sorted(lat.load_selector(_selector(tmp_path, big), small)) == [1, 3]
    reordered = _selection({1: 3, 3: 5, 2: 4})
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], (reordered["sha256"],))
    assert sorted(lat.load_selector(_selector(tmp_path, reordered), small)) == [1, 3]


def test_the_real_64_pilot_is_an_ordered_subsequence_of_the_registered_128_selection():
    from pathlib import Path

    from src.localization.pilot import load_pilot_manifest
    repo = Path(__file__).resolve().parents[2]
    pilot = load_pilot_manifest(repo / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/pilot_manifest_seed42_4_per_room.json")
    big = load_pilot_manifest(repo / "NeurIPS_2026_Workshop_Latency_Reproducibility/data/selection/frozen_16room_128.json")
    assert lat.REGISTERED_SUPERSETS[pilot["sha256"]] == (big["sha256"],)
    assert lat.ordered_subsequence(pilot["records"], big["records"])
