"""exp_37 round B-c: the nested-1.0 m subset manifest (``tools/exp37_build_subset_manifest.py``): per pilot query,
the integer-coordinate subset of its frozen 0.5 m candidates (count + indices sha256), hashed and bound to its parent
pilot / context / audit -- the input of the 1.0 m AGREE selector vector (plan §10 "selector vectors re-measured for
the 0.75 m and the nested 1.0 m counts"). Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.localization.ar_queries import load_context_manifest
from src.localization.exp37_grid import nested_subset
from src.localization.pilot import canonical_sha256, load_pilot_manifest
from tools import exp37_build_subset_manifest as build

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"


@pytest.fixture(scope="module")
def frozen():
    pilot = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    arrays = {}
    for record in pilot["records"]:
        with np.load(E9 / "pilot_results" / "vanilla" / "queries" / f"query_{int(record['index']):05d}.npz") as z:
            arrays[int(record["index"])] = z["candidates"].astype(np.float64)
    return pilot, arrays


def test_the_manifest_carries_each_querys_nested_subset(frozen):
    pilot, arrays = frozen
    manifest = build.build_subset_manifest(pilot, context_sha256="c" * 64, audit_sha256=pilot["geometry_audit_sha256"],
                                           candidates_for=lambda record: arrays[int(record["index"])])
    assert manifest["subset"] == "nested_1m_integer" and manifest["parent_pilot_sha256"] == pilot["sha256"]
    assert manifest["query_count"] == 64 and manifest["candidate_query_pairs"] == 6610
    first = manifest["records"][0]
    index = nested_subset(arrays[int(first["index"])])
    assert first["candidate_count"] == len(index) and first["full_candidate_count"] == pilot["records"][0]["candidate_count"]
    assert first["candidate_indices_sha256"] == hashlib.sha256(index.astype("<u4").tobytes()).hexdigest()
    assert first["parent_candidate_indices_sha256"] == pilot["records"][0]["candidate_indices_sha256"]
    assert manifest["sha256"] == canonical_sha256({k: v for k, v in manifest.items() if k != "sha256"})


def test_a_candidate_array_of_the_wrong_size_is_refused(frozen):
    pilot, arrays = frozen
    with pytest.raises(RuntimeError, match="candidate count"):
        build.build_subset_manifest(pilot, context_sha256="c", audit_sha256=pilot["geometry_audit_sha256"],
                                    candidates_for=lambda record: arrays[int(record["index"])][:-1])


@pytest.mark.skipif(not (REPO / "AcousticRooms" / "room_mesh_obj_format").is_dir(), reason="AcousticRooms is not linked")
def test_the_frozen_candidates_are_reconstructed_from_the_audit_and_the_remapped_meshes(frozen):
    pilot, arrays = frozen
    context = load_context_manifest(E9 / "context_manifest_exp01_seed42.json")
    audit = json.loads((E9 / "geometry_audit.json").read_text())
    reconstruct = build.frozen_candidates(context, audit, REPO / "AcousticRooms" / "room_mesh_obj_format")
    record = next(r for r in pilot["records"] if r["room"] == "Bathrooms_idx_14")
    assert np.array_equal(reconstruct(record), arrays[int(record["index"])])
