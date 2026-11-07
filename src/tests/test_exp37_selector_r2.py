"""exp_37 fix round 2 (Codex code review 2, HIGH 3): the strict selector loader.

``load_selector`` accepts only the registered exp_37 vector (record type, timed function, device cuda:0, 3 repeats,
batch 64, AGREE, context manifest) whose rows UNIQUELY and EXACTLY cover the selection it claims -- this one, or a
REGISTERED superset loaded from its registered path and hash-checked, of which this selection must be an ordered
subsequence -- each row naming its query, room, count and (non-optionally) candidate hash, with exactly the bundle's
five fields x three finite nonnegative repeats, every median recomputed and every repeat's total = its four parts;
peaks are nonnegative integers. The bundle's own vector is read only through ``bundle=True`` (its record type), with
the same row checks. CPU. Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import json
import math
from pathlib import Path

import pytest

from exp37_fakes import selector_fields
from src.localization import exp37_latency as lat
from src.localization.pilot import canonical_sha256, load_pilot_manifest

REPO = Path(__file__).resolve().parents[2]
LB = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility"


def _selection(counts):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": "r", "candidate_count": c, "candidate_indices_sha256": f"h{i}"}
               for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def _write(path, body):
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    return path


def test_a_complete_vector_loads(tmp_path):
    selection = _selection({1: 10, 2: 20})
    rows = lat.load_selector(_write(tmp_path / "s.json", selector_fields(selection, 0.5)), selection)
    assert rows[2]["scoring_total_seconds"] == pytest.approx(0.5) and rows[2]["peak_memory_allocated_bytes"] == 5


def _mutate(selection, change):
    body = selector_fields(selection, 0.5)
    change(body)
    return body


@pytest.mark.parametrize("change, needle", [
    (lambda b: b["queries"].append(dict(b["queries"][0])), "duplicate"),
    (lambda b: b.update(query_count=1), "query_count"),
    (lambda b: (b["queries"].pop(), b.update(query_count=1)), "coverage"),      # a consistent but partial vector
    (lambda b: b["queries"][0].pop("candidate_indices_sha256"), "identity"),
    (lambda b: b.update(device="cuda:1"), "device"),
    (lambda b: b.update(timed_function="mine"), "timed function"),
    (lambda b: b["queries"][0]["agree"]["repeat_seconds"].pop("argmax_seconds"), "fields"),
    (lambda b: b["queries"][0]["agree"]["repeat_seconds"]["argmax_seconds"].pop(), "three"),
    (lambda b: b["queries"][0]["agree"]["repeat_seconds"]["argmax_seconds"].__setitem__(0, -1.0), "three"),
    (lambda b: b["queries"][0]["agree"]["repeat_seconds"]["argmax_seconds"].__setitem__(0, math.nan), "three"),
    (lambda b: b["queries"][0]["agree"]["median_seconds"].update(scoring_total_seconds=0.1), "median"),
    (lambda b: b["queries"][0]["agree"]["repeat_seconds"]["scoring_total_seconds"].__setitem__(1, 0.9), "total|median"),
    (lambda b: b["queries"][0].update(peak_memory_allocated_bytes=-1), "peak"),
])
def test_every_incomplete_or_miscomputed_vector_is_refused(tmp_path, change, needle):
    selection = _selection({1: 10, 2: 20})
    with pytest.raises(RuntimeError, match=needle):
        lat.load_selector(_write(tmp_path / "s.json", _mutate(selection, change)), selection)


def test_a_superset_vector_must_cover_the_registered_superset_exactly(tmp_path, monkeypatch):
    big, small = _selection({1: 10, 2: 20, 3: 30}), _selection({1: 10, 3: 30})
    big_path = _write(tmp_path / "big_selection.json", {k: v for k, v in big.items() if k != "sha256"})
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], {"sha256": big["sha256"], "path": str(big_path)})
    rows = lat.load_selector(_write(tmp_path / "s.json", selector_fields(big, 0.5)), small)
    assert sorted(rows) == [1, 3]
    partial = selector_fields(big, 0.5)
    partial["queries"] = [q for q in partial["queries"] if q["query_index"] != 2]       # claims 128, holds the 64
    partial["query_count"] = 2
    with pytest.raises(RuntimeError, match="coverage"):
        lat.load_selector(_write(tmp_path / "p.json", partial), small)
    monkeypatch.setitem(lat.REGISTERED_SUPERSETS, small["sha256"], {"sha256": "0" * 64, "path": str(big_path)})
    with pytest.raises(RuntimeError, match="superset"):
        lat.load_selector(_write(tmp_path / "s2.json", selector_fields(big, 0.5)), small)


def test_the_registered_superset_is_the_bundle_selection_on_disk():
    pilot = load_pilot_manifest(REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude/"
                                       "pilot_manifest_seed42_4_per_room.json")
    entry = lat.REGISTERED_SUPERSETS[pilot["sha256"]]
    assert entry["sha256"] == load_pilot_manifest(REPO / entry["path"])["sha256"] == (
        "9bf75810a7fe6972049650748e015f9856b586f0c66d0554ef1b67c6bf8e041b")


def test_the_bundle_vector_is_read_only_explicitly_and_with_the_row_checks():
    selection = json.loads((LB / "data/selection/frozen_16room_128.json").read_text())
    with pytest.raises(RuntimeError, match="record type"):
        lat.load_selector(LB / "data/selector/selector_latency_128.json", selection)
    rows = lat.load_selector(LB / "data/selector/selector_latency_128.json", selection, bundle=True)
    assert len(rows) == 128
