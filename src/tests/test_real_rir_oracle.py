import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.localization.real_rir_oracle import (
    aggregate_oracle_rows,
    deterministic_agree_seed,
    discover_real_rir_bank,
    render_oracle_markdown,
    resolve_oracle_records,
    select_representative_cases,
    summarize_oracle_scores,
)
from tools.visualize_real_rir_oracle import resolve_visualization_cases


def _write_metadata(path: Path, source, receiver) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"src_loc": source, "rec_loc": receiver}))


def test_discover_real_rir_bank_sorts_numeric_sources_and_finds_target(tmp_path):
    room_audio = tmp_path / "single_channel_ir_1" / "Scene" / "Room"
    room_metadata = tmp_path / "metadata" / "Scene" / "Room"
    room_audio.mkdir(parents=True)
    receiver = [3.0, 4.0, 1.5]
    for source_id, source in ((10, [1.0, 0.0, 1.5]), (2, [0.0, 1.0, 1.5])):
        (room_audio / f"S{source_id:03d}_R006_hybrid_IR.wav").touch()
        _write_metadata(
            room_metadata / f"S00{source_id}_R006.json",
            source,
            receiver,
        )
    record = {
        "query_id": "single_channel_ir_1/Scene/Room/S010_R006_hybrid_IR.wav",
        "scene": "Scene",
        "room": "Room",
        "source_global": [1.0, 0.0, 1.5],
        "receiver_global": receiver,
    }

    bank = discover_real_rir_bank(record, tmp_path)

    assert bank["source_ids"] == [2, 10]
    assert bank["target_index"] == 1
    assert bank["receiver_id"] == 6
    assert bank["rir_paths"][1].endswith("S010_R006_hybrid_IR.wav")
    np.testing.assert_allclose(bank["positions_global"], [[0, 1, 1.5], [1, 0, 1.5]])


def test_discover_real_rir_bank_rejects_metadata_coordinate_mismatch(tmp_path):
    room_audio = tmp_path / "single_channel_ir_1" / "Scene" / "Room"
    room_audio.mkdir(parents=True)
    (room_audio / "S001_R001_hybrid_IR.wav").touch()
    (room_audio / "S002_R001_hybrid_IR.wav").touch()
    _write_metadata(
        tmp_path / "metadata" / "Scene" / "Room" / "S001_R001.json",
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    _write_metadata(
        tmp_path / "metadata" / "Scene" / "Room" / "S002_R001.json",
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    )
    record = {
        "query_id": "single_channel_ir_1/Scene/Room/S001_R001_hybrid_IR.wav",
        "scene": "Scene",
        "room": "Room",
        "source_global": [9.0, 9.0, 9.0],
        "receiver_global": [1.0, 1.0, 1.0],
    }

    with pytest.raises(RuntimeError, match="target source coordinate"):
        discover_real_rir_bank(record, tmp_path)


def test_summarize_oracle_scores_reports_identity_and_ambiguity_diagnostics():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    scores = np.asarray([0.8, 1.0, 0.9])

    result = summarize_oracle_scores(positions, scores, target_index=1, temperature=0.1)

    expected_probabilities = np.exp((scores - scores.max()) / 0.1)
    expected_probabilities /= expected_probabilities.sum()
    assert result["prediction_index"] == 1
    assert result["target_rank"] == 1
    assert result["hardest_negative_index"] == 2
    assert result["localization_error_m"] == 0.0
    assert result["target_score"] == 1.0
    assert result["hardest_negative_score"] == 0.9
    assert result["target_margin"] == pytest.approx(0.1)
    assert result["hardest_negative_distance_m"] == pytest.approx(math.sqrt(5.0))
    assert result["target_probability"] == pytest.approx(expected_probabilities[1])
    assert result["probability_mass_0_5m"] == pytest.approx(expected_probabilities[1])
    assert result["probability_mass_1_0m"] == pytest.approx(
        expected_probabilities[0] + expected_probabilities[1]
    )
    np.testing.assert_allclose(result["probabilities"], expected_probabilities)


def test_agree_rng_seeds_are_stable_and_role_separated():
    first = deterministic_agree_seed(42, 17, "observation")

    assert first == deterministic_agree_seed(42, 17, "observation")
    assert first != deterministic_agree_seed(42, 17, "candidate_real_rirs")
    assert first != deterministic_agree_seed(42, 18, "observation")
    with pytest.raises(ValueError, match="role"):
        deterministic_agree_seed(42, 17, "")


def test_aggregate_and_case_selection_are_deterministic():
    rows = []
    for index, (room, margin, entropy, probability) in enumerate(
        [
            ("A", 0.40, 0.10, 0.90),
            ("A", 0.05, 0.80, 0.20),
            ("B", 0.20, 0.95, 0.40),
            ("B", 0.25, 0.40, 0.60),
            ("C", 0.30, 0.30, 0.70),
        ]
    ):
        rows.append(
            {
                "query_id": f"q{index}",
                "room": room,
                "candidate_count": 10,
                "localization_error_m": float(index == 1),
                "target_rank": 1 if index != 1 else 2,
                "target_score": 1.0,
                "hardest_negative_score": 1.0 - margin,
                "target_margin": margin,
                "target_probability": probability,
                "normalized_entropy": entropy,
                "hardest_negative_distance_m": float(index + 1),
                "probability_mass_0_5m": probability,
                "probability_mass_1_0m": min(1.0, probability + 0.1),
                "success_0_5m": int(index != 1),
                "success_1_0m": 1,
            }
        )

    summary = aggregate_oracle_rows(rows)
    cases = select_representative_cases(rows)

    assert summary["query_count"] == 5
    assert summary["room_count"] == 3
    assert summary["candidate_count_median"] == 10
    assert summary["target_recall_at_1"] == pytest.approx(0.8)
    assert summary["mean_localization_error_m"] == pytest.approx(0.2)
    assert summary["median_target_margin"] == pytest.approx(0.25)
    assert [item["category"] for item in cases] == [
        "sharp",
        "ambiguous",
        "diffuse",
        "typical",
    ]
    assert len({item["query_id"] for item in cases}) == 4
    assert cases[0]["query_id"] == "q0"
    assert cases[1]["query_id"] == "q1"


def test_case_selection_supports_one_query_smoke():
    rows = [{"query_id": "q0", "target_margin": 0.2, "normalized_entropy": 0.4}]

    assert select_representative_cases(rows) == [
        {
            "category": "sharp",
            "query_id": "q0",
            "target_margin": 0.2,
            "normalized_entropy": 0.4,
        }
    ]


def test_resolve_oracle_records_combines_nonoverlapping_pilots():
    context = {
        "sha256": "context",
        "records": [
            {"index": 1, "query_id": "q1", "scene": "S", "room": "A"},
            {"index": 2, "query_id": "q2", "scene": "S", "room": "B"},
        ],
    }
    first = {
        "sha256": "pilot1",
        "context_manifest_sha256": "context",
        "query_count": 1,
        "records": [{"index": 1, "query_id": "q1"}],
    }
    second = {
        "sha256": "pilot2",
        "context_manifest_sha256": "context",
        "query_count": 1,
        "records": [{"index": 2, "query_id": "q2"}],
    }

    rows = resolve_oracle_records([("batch1", first), ("batch2", second)], context)

    assert [(item["batch"], item["query_id"]) for item in rows] == [
        ("batch1", "q1"),
        ("batch2", "q2"),
    ]
    with pytest.raises(ValueError, match="overlap"):
        resolve_oracle_records([("batch1", first), ("again", first)], context)


def test_render_oracle_markdown_labels_ground_truth_rir_upper_bound():
    payload = {
        "query_count": 128,
        "room_count": 16,
        "score_sample_counts": [1],
        "primary_score_sample_count": 1,
        "tau": 0.1,
        "temperature": 0.1,
        "summary": {
            "target_recall_at_1": 1.0,
            "mean_localization_error_m": 0.0,
            "median_localization_error_m": 0.0,
            "success_0_5m": 1.0,
            "success_1_0m": 1.0,
            "candidate_count_min": 9,
            "candidate_count_median": 10.0,
            "candidate_count_max": 10,
            "mean_target_score": 1.0,
            "mean_hardest_negative_score": 0.8,
            "mean_target_margin": 0.2,
            "median_target_margin": 0.2,
            "target_margin_p10": 0.1,
            "target_margin_p90": 0.3,
            "mean_target_probability": 0.5,
            "median_target_probability": 0.5,
            "mean_normalized_entropy": 0.4,
            "mean_hardest_negative_distance_m": 2.0,
            "median_hardest_negative_distance_m": 1.8,
            "mean_probability_mass_0_5m": 0.6,
            "mean_probability_mass_1_0m": 0.7,
        },
        "representative_cases": [
            {
                "category": "sharp",
                "batch": "batch1",
                "room": "Room",
                "query_id": "q1",
                "target_margin": 0.2,
                "target_probability": 0.5,
                "normalized_entropy": 0.4,
                "hardest_negative_distance_m": 2.0,
            }
        ],
    }
    payload["summary_by_k"] = {"1": payload["summary"]}

    markdown = render_oracle_markdown(payload)

    assert "ground-truth-RIR upper bound" in markdown
    assert "independently passed through AGREE's stochastic" in markdown
    assert "128 queries / 16 rooms" in markdown


def test_resolve_visualization_cases_preserves_registered_order():
    payload = {
        "representative_cases": [
            {"category": "sharp", "query_id": "q2"},
            {"category": "ambiguous", "query_id": "q1"},
        ],
        "results": [
            {"query_id": "q1", "room": "A"},
            {"query_id": "q2", "room": "B"},
        ],
    }

    cases = resolve_visualization_cases(payload)

    assert [(item["category"], item["query_id"]) for item in cases] == [
        ("sharp", "q2"),
        ("ambiguous", "q1"),
    ]
