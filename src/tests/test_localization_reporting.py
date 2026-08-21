import numpy as np

from src.localization.pilot import canonical_sha256
from src.localization.reporting import aggregate_pilot, render_markdown
from src.localization.runner import initialize_run, save_query_result


def _identity(pilot_sha, method, checkpoint):
    return {
        "model_config_sha256": "model",
        "checkpoint_sha256": checkpoint,
        "agree_checkpoint_sha256": "agree",
        "context_manifest_sha256": "context",
        "geometry_audit_sha256": "geometry",
        "pilot_manifest_sha256": pilot_sha,
        "query_indices": [7],
        "conditioning_method": method,
        "frame_average_angles": None if method == "vanilla" else [0.0, 90.0, 180.0, 270.0],
        "n_context": 8,
        "score_sample_counts": [1, 4, 8],
        "tau": 0.1,
        "sample_seed": 42,
        "candidate_batch_size": 64,
        "sampler": {"type": "rectified_flow_discrete_euler", "steps": 1, "cfg_scale": 1.0},
    }


def _query_result(run_sha, shift):
    metric = {
        "prediction_index": 0,
        "localization_error_m": 0.5 + shift,
        "oracle_error_m": 0.2,
        "excess_error_m": 0.3 + shift,
        "success_0_5m": int(shift == 0),
        "success_1_0m": 1,
        "oracle_normalized_success_0_5m": 1,
        "oracle_normalized_success_1_0m": 1,
        "prediction_global": [0.0, 0.0, 0.0],
        "winning_score": 0.4,
        "mean_candidate_score": 0.2,
    }
    random = dict(metric)
    random["localization_error_m"] = 1.5
    random["excess_error_m"] = 1.3
    random["success_0_5m"] = 0
    random["success_1_0m"] = 0
    random["oracle_normalized_success_0_5m"] = 0
    random["oracle_normalized_success_1_0m"] = 0
    return {
        "schema_version": 1,
        "run_manifest_sha256": run_sha,
        "query_index": 7,
        "query_id": "room/query.wav",
        "scene": "scene",
        "room": "room",
        "receiver_id": "R001",
        "candidate_count": 2,
        "candidate_indices_sha256": "candidates",
        "source_global": [0.0, 0.0, 0.0],
        "receiver_global": [1.0, 0.0, 0.0],
        "n_context": 8,
        "score_sample_counts": [1, 4, 8],
        "tau": 0.1,
        "metrics": {str(count): dict(metric) for count in (1, 4, 8)},
        "random_candidate_metrics": random,
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 100,
    }


def test_two_arm_aggregate_validates_and_renders(tmp_path):
    pilot = {
        "schema_version": 1,
        "query_count": 1,
        "room_count": 1,
        "records": [
            {
                "index": 7,
                "query_id": "room/query.wav",
                "candidate_count": 2,
            }
        ],
    }
    pilot["sha256"] = canonical_sha256(pilot)
    directories = {}
    for arm, method, checkpoint, shift in (
        ("vanilla", "vanilla", "v", 0.0),
        ("fa_bf", "fa_invariant", "f", 0.1),
    ):
        directory = tmp_path / arm
        run = initialize_run(directory, _identity(pilot["sha256"], method, checkpoint))
        save_query_result(
            directory,
            result=_query_result(run["sha256"], shift),
            candidates=np.zeros((2, 3), dtype=np.float32),
            similarities=np.zeros((2, 8), dtype=np.float32),
        )
        directories[arm] = directory
    aggregate = aggregate_pilot(pilot, directories)
    assert aggregate["query_count"] == 1
    assert aggregate["arms"]["vanilla"]["summary"]["8"]["mean_localization_error_m"] == 0.5
    assert "64-query localization pilot" in render_markdown(aggregate)
