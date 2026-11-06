"""exp_37 round B-b: K_ctx = 1 (plan_loc_efficiency.md §2 item (3), §8 D2, §10 "Provenance").

K_ctx = 1 feeds ``contexts[0]`` of the frozen exp_09 context manifest to the model while the candidate filter keeps
the 8-context rule (the engine's ``candidate_filter_context_sources_global`` override), so the candidate set -- and
its audit hash -- is the K_ctx = 8 one. K_ctx = 8 is the record itself (the bundle path, unchanged).

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.localization import runner
from src.localization.engine import (
    filter_frozen_query_candidates,
    load_frozen_query,
    reconstruct_room_base_candidates,
)

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"
DATA = REPO / "AcousticRooms"


@pytest.fixture(scope="module")
def record():
    manifest = json.loads((E9 / "context_manifest_exp01_seed42.json").read_text())
    return next(r for r in manifest["records"] if r["index"] == 4467)


def test_k1_feeds_context_zero_and_keeps_the_eight_context_filter(record):
    before = copy.deepcopy(record)
    view = runner.context_count_view(record, 1)
    assert record == before                                                  # the frozen record is never mutated
    assert view["contexts"] == record["contexts"][:1]
    assert view["context_sources_global"] == record["context_sources_global"][:1]
    assert view["candidate_filter_context_sources_global"] == record["context_sources_global"]
    assert {k: v for k, v in view.items() if k not in ("contexts", "context_sources_global",
                                                       "candidate_filter_context_sources_global")} == \
        {k: v for k, v in record.items() if k not in ("contexts", "context_sources_global")}


def test_k8_is_the_record_itself(record):
    assert runner.context_count_view(record, 8) is record


@pytest.mark.parametrize("count", [0, 2, 4, 7, 9])
def test_other_context_counts_are_refused(record, count):
    with pytest.raises(ValueError, match="context count"):
        runner.context_count_view(record, count)


def test_a_record_without_eight_contexts_or_with_an_override_is_refused(record):
    with pytest.raises(ValueError, match="8 contexts"):
        runner.context_count_view({**record, "contexts": record["contexts"][:4]}, 1)
    with pytest.raises(ValueError, match="override"):
        runner.context_count_view({**record, "candidate_filter_context_sources_global": [[0.0, 0.0, 0.0]]}, 1)


def test_execute_query_refuses_a_record_of_another_context_count(monkeypatch, record):
    monkeypatch.setattr(runner, "load_frozen_query", lambda rec, root: (torch.zeros(1, 10240), {}))
    monkeypatch.setattr(runner, "filter_frozen_query_candidates", lambda rec, audit, base: np.zeros((27, 3)))
    common = dict(module=None, retrieval=None, selected={"candidate_count": 27}, audit={}, room_base=None,
                  dataset_root=".", device="cpu", cond_method="vanilla", candidate_batch_size=64, sample_seed=42,
                  tau=0.1, run_sha256="x")
    with pytest.raises(RuntimeError, match="N_ctx=1"):
        runner.execute_query(record=record, context_count=1, **common)
    with pytest.raises(RuntimeError, match="N_ctx=8"):
        runner.execute_query(record=runner.context_count_view(record, 1), **common)


@pytest.mark.skipif(not (DATA / "room_mesh_obj_format").is_dir(), reason="AcousticRooms is not linked")
def test_on_real_data_k1_keeps_the_frozen_candidates_and_feeds_one_context(record):
    audit = json.loads((E9 / "geometry_audit.json").read_text())
    mesh = DATA / "room_mesh_obj_format" / "Bathrooms" / "Bathrooms_idx_14.obj"
    base = reconstruct_room_base_candidates("Bathrooms_idx_14", audit, mesh_path=mesh)
    full = filter_frozen_query_candidates(record, audit, base)              # each call re-checks the audit hash
    one = filter_frozen_query_candidates(runner.context_count_view(record, 1), audit, base)
    assert full.shape == (27, 3) and np.array_equal(full, one)
    _audio, metadata = load_frozen_query(runner.context_count_view(record, 1), DATA)
    assert metadata["context_audio"].shape[0] == 1 and tuple(metadata["context_poses"].shape) == (1, 3)
    _audio8, metadata8 = load_frozen_query(record, DATA)
    assert torch.equal(metadata["context_audio"][0], metadata8["context_audio"][0])
    assert torch.equal(metadata["context_poses"][0], metadata8["context_poses"][0])
