import copy

import pytest

from src.localization.pilot import (
    build_pilot_manifest,
    load_pilot_manifest,
    resolve_pilot_records,
    save_pilot_manifest,
)


def _inputs():
    records = []
    queries = []
    for room_number, room in enumerate(("room_a", "room_b")):
        for offset in range(6):
            index = room_number * 6 + offset
            query_id = f"{room}/query_{index}.wav"
            records.append({"index": index, "query_id": query_id, "room": room})
            queries.append(
                {
                    "index": index,
                    "query_id": query_id,
                    "scene": "scene",
                    "room": room,
                    "receiver_id": f"R{offset:03d}",
                    "chosen_count": index + 10,
                    "z_indices_sha256": f"hash-{index}",
                    "full_indices_sha256": f"full-{index}",
                    "z_oracle_m": 0.2,
                    "full_oracle_m": 0.2,
                }
            )
    context = {"sha256": "context-hash", "records": records}
    audit = {
        "sha256": "audit-hash",
        "context_manifest_sha256": "context-hash",
        "geometry_gate": "PASS",
        "included_rooms": 2,
        "z_branch": "z_band",
        "queries": queries,
    }
    return context, audit


def test_room_stratified_pilot_is_deterministic_and_roundtrips(tmp_path):
    context, audit = _inputs()
    first = build_pilot_manifest(context, audit, queries_per_room=2, seed=42)
    second = build_pilot_manifest(context, audit, queries_per_room=2, seed=42)
    assert first == second
    assert first["query_count"] == 4
    assert {item["room"] for item in first["records"]} == {"room_a", "room_b"}
    assert sum(item["room"] == "room_a" for item in first["records"]) == 2
    path = tmp_path / "pilot.json"
    save_pilot_manifest(first, path)
    assert load_pilot_manifest(path) == first
    assert len(resolve_pilot_records(first, context, audit)) == 4


def test_pilot_hash_and_source_mismatches_fail_closed(tmp_path):
    context, audit = _inputs()
    manifest = build_pilot_manifest(context, audit, queries_per_room=2)
    path = tmp_path / "pilot.json"
    save_pilot_manifest(manifest, path)
    payload = path.read_text().replace("query_", "changed_", 1)
    path.write_text(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_pilot_manifest(path)
    changed = copy.deepcopy(audit)
    changed["sha256"] = "changed"
    with pytest.raises(ValueError, match="pilot/geometry"):
        resolve_pilot_records(manifest, context, changed)
