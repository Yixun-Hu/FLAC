"""exp_37 round B-b: the AGREE selector latency over an ARBITRARY query / candidate-count manifest (plan §10 "B2
time axis": warm-up, batch 64, 3 repeats per query, median-of-three kept; the bundle's timed function itself,
``tools/benchmark_localization_selector_latency.py::benchmark_agree_query``, unchanged; FEM inputs not required).
CPU only with a fake retrieval. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-b.
"""
import json
from pathlib import Path

import pytest
import torch

from src.localization.pilot import canonical_sha256
from tools import benchmark_localization_selector_latency as bundle_selector
from tools import exp37_selector_latency as sel

REPO = Path(__file__).resolve().parents[2]
SELECTION_128 = REPO / "NeurIPS_2026_Workshop_Latency_Reproducibility/data/selection/frozen_16room_128.json"


class FakeRetrieval:
    def __init__(self):
        self.batches = []

    def compute_audio_features(self, waveforms):
        self.batches.append(int(waveforms.shape[0]))
        return torch.ones(waveforms.shape[0], 4)


def _selection(counts):
    records = [{"index": i, "query_id": f"q{i}.wav", "room": f"r{i % 2}", "candidate_count": c,
                "candidate_indices_sha256": f"h{i}"} for i, c in counts.items()]
    body = {"schema_version": 1, "query_count": len(records), "records": records}
    return {**body, "sha256": canonical_sha256(body)}


def _contexts(selection):
    return {r["index"]: {"index": r["index"], "query_id": r["query_id"]} for r in selection["records"]}


def test_the_timed_function_is_the_bundle_s():
    assert sel.benchmark_agree_query is bundle_selector.benchmark_agree_query


def test_every_query_gets_three_repeats_and_their_medians_in_index_order():
    selection = _selection({7: 130, 3: 27})
    retrieval = FakeRetrieval()
    rows = sel.measure_selector(selection, _contexts(selection), Path("."), retrieval, device="cpu",
                                load_query=lambda record, root: (torch.zeros(1, 10240), {}))
    assert [r["query_index"] for r in rows] == [3, 7]
    assert rows[1]["candidate_count"] == 130 and rows[1]["candidate_indices_sha256"] == "h7"
    for row in rows:
        assert all(len(v) == 3 for v in row["agree"]["repeat_seconds"].values())
        assert set(row["agree"]["median_seconds"]) == {"observed_encode_seconds", "generated_encode_seconds",
                                                       "similarity_seconds", "argmax_seconds", "scoring_total_seconds"}
        assert row["peak_memory_allocated_bytes"] is None and row["peak_memory_reserved_bytes"] is None   # CPU
    # one unreported warm-up batch of 64, then per repeat: observed (1) + ceil(count / 64) generated batches
    assert retrieval.batches[0] == 64
    assert retrieval.batches[1:] == ([1, 27] * 3) + ([1, 64, 64, 2] * 3)


def test_identity_mismatches_and_bad_manifests_are_refused(tmp_path):
    selection = _selection({3: 27})
    with pytest.raises(RuntimeError, match="identity"):
        sel.measure_selector(selection, {3: {"index": 3, "query_id": "other.wav"}}, Path("."), FakeRetrieval(),
                             device="cpu", load_query=lambda r, root: (torch.zeros(1, 10240), {}))
    path = tmp_path / "s.json"
    path.write_text(json.dumps({**selection, "query_count": 5}))
    with pytest.raises(RuntimeError, match="sha256"):
        sel.load_selection(path)
    body = {"schema_version": 1, "query_count": 2, "records": selection["records"] * 2}
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    with pytest.raises(RuntimeError, match="duplicate"):
        sel.load_selection(path)
    body = {"schema_version": 1, "query_count": 1, "records": [{**selection["records"][0], "candidate_count": 0}]}
    path.write_text(json.dumps({**body, "sha256": canonical_sha256(body)}))
    with pytest.raises(RuntimeError, match="candidate_count"):
        sel.load_selection(path)


def test_the_frozen_128_selection_loads():
    selection = sel.load_selection(SELECTION_128)
    assert len(selection["records"]) == 128
    assert sum(r["candidate_count"] for r in selection["records"]) == 92608


def test_the_payload_is_hashed_and_records_its_inputs():
    selection = _selection({3: 27})
    rows = sel.measure_selector(selection, _contexts(selection), Path("."), FakeRetrieval(), device="cpu",
                                load_query=lambda r, root: (torch.zeros(1, 10240), {}))
    payload = sel.build_payload(selection=selection, context_sha256="c" * 64, agree_path=Path("a.pt"),
                                agree_sha256="a" * 64, device="cpu", batch_size=64, repeat_count=3, rows=rows,
                                host={"pid": 1})
    assert payload["record_type"] == "exp37_agree_selector_latency" and payload["query_count"] == 1
    assert payload["selection_sha256"] == selection["sha256"] and payload["repeat_count"] == 3
    assert payload["timed_function"] == "tools/benchmark_localization_selector_latency.py::benchmark_agree_query"
    assert payload["sha256"] == canonical_sha256({k: v for k, v in payload.items() if k != "sha256"})
