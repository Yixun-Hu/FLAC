"""exp_37 fix round 2 (Codex code review 2, HIGH 4): ONE query-versus-selection rule for every source.

``exp37_queries.query_reasons``: a FULL-grid record must be this run's, the selection's query, with the selection's
count AND candidate hash and no parent / subset; a SUBSET record (nested / c2f) must carry the expected active hash and
count and the selection's hash as its parent. Every loader uses it: full / nested / c2f timing passes, accuracy sources
(and their replays) and the offline readout. CPU; the synthetic registered world. Written by the exp_37 Coder (Claude
Opus 5.5, max effort), fix round 2.
"""
import json
from pathlib import Path

import pytest

from exp37_world import World
from src.localization import exp37_grid as grid
from src.localization import exp37_latency as lat
from src.localization import exp37_queries as queries
from src.localization import exp37_sources as sources
from src.localization.pilot import canonical_sha256

RECORD = {"index": 7, "query_id": "q7.wav", "candidate_count": 15, "candidate_indices_sha256": "h7"}


def _full(**over):
    return {"run_manifest_sha256": "m", "query_index": 7, "query_id": "q7.wav", "candidate_count": 15,
            "candidate_indices_sha256": "h7", **over}


@pytest.mark.parametrize("over, needle", [({"run_manifest_sha256": "x"}, "run"), ({"query_index": 8}, "query"),
                                          ({"query_id": "q8.wav"}, "query"), ({"candidate_count": 14}, "full"),
                                          ({"candidate_indices_sha256": "h8"}, "full"),
                                          ({"parent_candidate_indices_sha256": "h7"}, "parent")])
def test_a_full_record_is_the_selection_s_query_and_candidate_set(over, needle):
    assert queries.query_reasons(_full(), RECORD, run_sha256="m") == []
    assert any(needle in r for r in queries.query_reasons(_full(**over), RECORD, run_sha256="m"))


def test_a_subset_record_has_its_active_hash_and_the_selection_as_parent():
    sub = _full(candidate_count=6, candidate_indices_sha256="s7", parent_candidate_indices_sha256="h7")
    assert queries.query_reasons(sub, RECORD, run_sha256="m", active_sha256="s7", count=6) == []
    assert queries.query_reasons(sub, RECORD, run_sha256="m", active_sha256="s9", count=6)
    assert queries.query_reasons({**sub, "parent_candidate_indices_sha256": "h8"}, RECORD, run_sha256="m",
                                 active_sha256="s7", count=6)
    assert queries.query_reasons(sub, {**RECORD, "candidate_indices_sha256": None}, run_sha256="m",
                                 active_sha256="s7", count=6)                 # the selection must carry its hash


def _retarget(run_dir, index, **changes):
    path = Path(run_dir) / "queries" / f"query_{index:05d}.json"
    body = {k: v for k, v in json.loads(path.read_text()).items() if k != "sha256"}
    body.update(changes)
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path).build()


def test_every_loader_refuses_a_record_of_another_candidate_set(world):
    a50 = world.audits["audit_0p50"]
    _retarget(world.runs / "b2_0p50_P1_s42", 2, candidate_indices_sha256="another")
    with pytest.raises(RuntimeError, match="candidate"):
        lat.load_arm_run(world.runs / "b2_0p50_P1_s42", arm="P1", selection=world.pilot, audit=a50)
    with pytest.raises(RuntimeError, match="candidate"):
        grid.readout_run(world.runs / "b2_0p50_P1_s42", world.pilot)
    _retarget(world.runs / "b2_0p50_BF_s43", 1, candidate_indices_sha256="another")
    with pytest.raises(RuntimeError, match="candidate"):
        sources.accuracy_source(world.runs / "b2_0p50_BF_s43", arm="BF", selection=world.pilot, audit=a50, seed=43,
                                timing=False)


def test_subset_passes_must_name_the_selection_as_their_parent(world):
    a50 = world.audits["audit_0p50"]
    nested = json.loads((world.root / "nested.json").read_text())
    _retarget(world.runs / "b2t_nested_P1_s42", 1, parent_candidate_indices_sha256="another")
    with pytest.raises(RuntimeError, match="parent"):
        lat.load_arm_run(world.runs / "b2t_nested_P1_s42", arm="P1", selection=world.pilot, audit=a50,
                         subset_manifest=nested)
    _retarget(world.runs / "b2t_c2f_BF_s42", 2, parent_candidate_indices_sha256="another")
    with pytest.raises(RuntimeError, match="parent"):
        lat.load_c2f_run(world.runs / "b2t_c2f_BF_s42", arm="BF", selection=world.pilot, audit=a50,
                         parent_run_dir=world.runs / "b2_0p50_BF_s42")
