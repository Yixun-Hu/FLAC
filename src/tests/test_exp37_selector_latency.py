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


def test_the_binding_names_this_vector_s_inputs_and_output():
    """BLOCKING 3: the selector vector is bound (before the measurement) to its selection / AGREE / context / batch /
    repeats / device and its output file; the payload carries the binding."""
    from src.localization.exp37_latency import SELECTOR_BOUND
    selection = _selection({3: 27})
    binding = sel.make_binding(selection=selection, context_sha256="c" * 64, agree_sha256="a" * 64, device="cuda:0",
                               batch_size=64, repeat_count=3, output=Path("/x/selector_128.json"))
    assert binding["record_type"] == "exp37_selector_binding" and binding["output"] == "/x/selector_128.json"
    assert binding["sha256"] == canonical_sha256({k: v for k, v in binding.items() if k != "sha256"})
    payload = sel.build_payload(selection=selection, context_sha256="c" * 64, agree_path=Path("a.pt"),
                                agree_sha256="a" * 64, device="cuda:0", batch_size=64, repeat_count=3, rows=[],
                                host={"pid": 1}, binding=binding)
    assert payload["binding"] == binding and all(payload[k] == binding[k] for k in SELECTOR_BOUND)


def test_a_cuda_selector_run_is_monitored_around_its_measurement(tmp_path, monkeypatch):
    import sys
    events = []

    class FakeMonitor:
        def __init__(self, directory, *, binding_sha256, card_uuid, **kw):
            events.append(("monitor", str(directory), binding_sha256, card_uuid))

        def start(self):
            return self

        def marker(self, name):
            events.append(("marker", name))

        def stop(self):
            events.append(("stop",))
            return 0
    monkeypatch.setattr(sel, "Monitor", FakeMonitor)
    monkeypatch.setattr(sel, "producer_identity", lambda device: {"pid": 9, "pid_start_ticks": 7, "hostname": "h",
                                                                   "card_uuid": "6ac2604a-1111"})
    monkeypatch.setattr(sel, "load_agree_retrieval", lambda path, device: events.append(("load",)) or FakeRetrieval())
    monkeypatch.setattr(sel, "measure_selector", lambda *a, **k: events.append(("measure",)) or [])
    monkeypatch.setattr(sel, "host_snapshot", lambda device: {"pid": 1})
    selection_path = tmp_path / "selection.json"
    selection = _selection({3: 27})
    selection_path.write_text(json.dumps(selection))
    context = {"records": [{"index": 3, "query_id": "q3.wav"}]}
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps({**context, "sha256": canonical_sha256(context)}))
    agree = tmp_path / "agree.pt"
    agree.write_bytes(b"agree")
    output = tmp_path / "selector.json"
    argv = ["exp37_selector_latency.py", "--selection", str(selection_path), "--context-manifest", str(context_path),
            "--dataset-root", str(tmp_path), "--agree-checkpoint", str(agree), "--device", "cuda:0", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    sel.main()
    payload = json.loads(output.read_text())
    assert events[0] == ("monitor", f"{output}.cotenancy", payload["binding"]["sha256"], "6ac2604a-1111")
    assert (payload["binding"]["pid"], payload["binding"]["pid_start_ticks"], payload["binding"]["card_uuid"]) == (
        9, 7, "6ac2604a-1111")
    assert [e[1] if e[0] == "marker" else e[0] for e in events[1:]] == ["load", "gpu_resident", "timed_start", "measure",
                                                                         "timed_end", "stop"]
    output.unlink()
    (tmp_path / "selector.json.cotenancy").mkdir()
    with pytest.raises(RuntimeError, match="cotenancy"):
        sel.main()
