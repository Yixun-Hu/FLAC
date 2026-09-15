import json

import numpy as np
import pytest
import torch

from src.localization.runner import (
    _build_query_result,
    _repeat_branch,
    completed_query_result,
    initialize_run,
    save_query_result,
)


def test_conditioning_branch_repeat_interleaves_candidates():
    branch = {"source": [torch.tensor([[1], [2]]), torch.tensor([[True], [False]])]}
    repeated = _repeat_branch(branch, 3)
    assert repeated["source"][0].flatten().tolist() == [1, 1, 1, 2, 2, 2]
    assert repeated["source"][1].flatten().tolist() == [True, True, True, False, False, False]


def test_atomic_query_artifact_roundtrip_and_corruption_guard(tmp_path):
    run = initialize_run(tmp_path / "run", {"test": "identity"})
    base = {
        "schema_version": 1,
        "run_manifest_sha256": run["sha256"],
        "query_index": 7,
        "query_id": "room/query.wav",
        "candidate_count": 2,
    }
    save_query_result(
        tmp_path / "run",
        result=base,
        candidates=np.zeros((2, 3), dtype=np.float32),
        similarities=np.ones((2, 8), dtype=np.float32),
    )
    loaded = completed_query_result(
        tmp_path / "run",
        query_index=7,
        query_id="room/query.wav",
        candidate_count=2,
        run_sha256=run["sha256"],
    )
    assert loaded is not None
    arrays_path = tmp_path / "run" / loaded["arrays_file"]
    arrays_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="missing or corrupt"):
        completed_query_result(
            tmp_path / "run",
            query_index=7,
            query_id="room/query.wav",
            candidate_count=2,
            run_sha256=run["sha256"],
        )


def test_single_generation_query_result_has_only_k1_metrics():
    result = _build_query_result(
        selected={
            "index": 7,
            "query_id": "room/query.wav",
            "scene": "scene",
            "room": "room",
            "receiver_id": "R001",
            "candidate_indices_sha256": "candidate-hash",
        },
        record={
            "contexts": [f"context-{index}" for index in range(8)],
            "source_global": [0.0, 0.0, 0.0],
            "receiver_global": [1.0, 1.0, 1.0],
        },
        candidates=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        similarities=torch.tensor([[0.9], [0.1]]),
        run_sha256="run",
        elapsed_seconds=1.0,
        peak_memory_bytes=0,
        tau=0.1,
        random_seed=42,
        score_sample_counts=(1,),
    )

    assert result["score_sample_counts"] == [1]
    assert list(result["metrics"]) == ["1"]


def test_run_manifest_refuses_parameter_drift_and_unowned_output(tmp_path):
    output = tmp_path / "run"
    initialize_run(output, {"batch": 64})
    with pytest.raises(RuntimeError, match="does not match"):
        initialize_run(output, {"batch": 32})
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "some_file").write_text("user data")
    with pytest.raises(RuntimeError, match="nonempty"):
        initialize_run(unowned, {"batch": 64})
