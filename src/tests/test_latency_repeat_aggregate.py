import json
import sys
from pathlib import Path

import pytest

from tools.aggregate_kctx8_kgen1_latency import METHODS, canonical_sha256, main


def _repeat_payload(vanilla_latencies: tuple[float, float]) -> dict:
    queries = {}
    for offset, (index, candidate_count) in enumerate(((1, 10), (2, 20))):
        latency = {method: float(2 + offset) for method in METHODS}
        latency["vanilla_flac"] = vanilla_latencies[offset]
        queries[str(index)] = {
            "query_id": f"query-{index}",
            "room": f"room-{index}",
            "candidate_count": candidate_count,
            "latency_seconds": latency,
            "fem_source": "reused",
        }
    payload = {
        "schema_version": 3,
        "latency_protocol": {
            "context_count": 8,
            "generated_rirs_per_candidate": 1,
        },
        "scope": {
            "room_count": 2,
            "query_count": 2,
            "candidate_evaluations": 30,
            "selection_sha256": "selection",
            "aggregation": "query micro",
        },
        "overall": {method: {} for method in METHODS},
        "fem_provenance": {"hardware_normalized": False},
        "queries": queries,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def test_repeat_aggregate_uses_per_query_medians(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summaries = []
    for number, latencies in enumerate(((1.0, 2.0), (7.0, 6.0), (3.0, 4.0)), start=1):
        path = tmp_path / f"repeat_{number}.json"
        path.write_text(json.dumps(_repeat_payload(latencies)))
        summaries.append(path)
    output_json = tmp_path / "final.json"
    output_md = tmp_path / "final.md"
    argv = ["aggregate_kctx8_kgen1_latency.py"]
    for path in summaries:
        argv.extend(("--summary", str(path)))
    argv.extend(("--output-json", str(output_json), "--output-md", str(output_md)))
    monkeypatch.setattr(sys, "argv", argv)

    main()

    result = json.loads(output_json.read_text())
    assert result["repeat_count"] == 3
    assert result["queries"]["1"]["methods"]["vanilla_flac"]["median_seconds"] == 3.0
    assert result["queries"]["2"]["methods"]["vanilla_flac"]["median_seconds"] == 4.0
    assert result["overall"]["vanilla_flac"]["mean_seconds"] == 3.5
    assert result["sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "sha256"}
    )
    assert output_md.is_file()
