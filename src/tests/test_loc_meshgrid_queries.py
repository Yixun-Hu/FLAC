"""exp_22 D1 -- the release-parity context materializer.

The protocol here is deliberately the RELEASED loader path, not exp_18's pinned
4/4 one: the drawn contexts depend on the exp_01 batch/worker layout and on the
released module's global NumPy RNG, so anything that changes those changes the
conditioning. These tests pin the properties that make the manifest reusable --
the census, the materialize-then-filter order, exclusion exactness, and reload
stability -- and the real-dataset pins are skipped when the dataset is absent.
"""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_queries as mq

_UNSEEN_CONFIG = "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"
_AR_ROOT = "AcousticRooms"


# --------------------------------------------------------------------------- #
# the released eligible pool: the S00{node} quirk is part of the protocol
# --------------------------------------------------------------------------- #
def _room(tmp_path, sources, receiver="R008", room="Cafe/Cafe_idx_1"):
    """A room directory holding one wav per source, in the released naming."""
    directory = tmp_path / room
    directory.mkdir(parents=True, exist_ok=True)
    for node in sources:
        (directory / f"S{node}_{receiver}_hybrid_IR.wav").write_bytes(b"stub")
    return directory


def test_eligible_pool_mirrors_the_released_enumeration(tmp_path):
    """`S00{node}` renders node 10 as S0010, which no file is called: source 10
    is therefore never eligible, and that quirk is protocol, not a bug to fix."""
    directory = _room(tmp_path, ["001", "002", "003", "010"])
    target = str(directory / "S001_R008_hybrid_IR.wav")
    pool = mq.eligible_context_pool(target)
    assert [os.path.basename(p) for p in pool] == ["S002_R008_hybrid_IR.wav",
                                                   "S003_R008_hybrid_IR.wav"]
    assert all("S0010" not in p for p in pool)
    assert mq.eligible_pool_size(target) == 2


def test_eligible_pool_excludes_the_target_source_only(tmp_path):
    directory = _room(tmp_path, [f"00{i}" for i in range(1, 10)])
    target = str(directory / "S005_R008_hybrid_IR.wav")
    pool = mq.eligible_context_pool(target)
    assert len(pool) == 8 and all("S005_" not in os.path.basename(p) for p in pool)


def test_eligible_pool_agrees_with_the_released_module(tmp_path, monkeypatch):
    """The pool must be the SAME list the released selector draws from."""
    directory = _room(tmp_path, ["001", "002", "003", "004", "010"])
    target = str(directory / "S001_R008_hybrid_IR.wav")
    seen = {}

    def spy(paths, size, replace=False):
        seen["paths"] = list(paths)
        raise RuntimeError("stop after the pool is built")

    ar_md = mq.released_metadata_module()
    monkeypatch.setattr(ar_md.np.random, "choice", spy)
    with pytest.raises(RuntimeError):
        ar_md.get_ir_and_location_for_other_sources(target, 8, metadata_path="unused")
    assert seen["paths"] == mq.eligible_context_pool(target)


# --------------------------------------------------------------------------- #
# per-record evidence
# --------------------------------------------------------------------------- #
def _md(position=0, room="Cafe/Cafe_idx_1", node="S001", receiver="R008", width=8,
        root="AcousticRooms"):
    relpath = f"single_channel_ir_1/{room}/{node}_{receiver}_hybrid_IR.wav"
    generator = torch.Generator().manual_seed(position + 1)
    return {
        "idx": position, "relpath": relpath, "path": f"{root}/{relpath}",
        "context_poses": torch.arange(width * 3, dtype=torch.float32).reshape(width, 3),
        "context_audio": torch.randn(width, 1, 10240, generator=generator),
        "source": torch.tensor([1.5, 2.5, 0.5]),
    }


def test_record_carries_identity_fingerprints_and_audio_digests():
    md = _md()
    record = mq.context_record(md, position=0, eligible=8)
    assert record["position"] == 0
    assert record["query_id"].startswith("0|")
    assert record["room_id"] == "Cafe/Cafe_idx_1"
    assert len(record["context_fingerprints"]) == 8
    assert record["context_fingerprints"][0].count(",") == 2      # 3 coordinates
    assert len(record["context_audio_sha256"]) == 8
    assert all(len(digest) == 64 for digest in record["context_audio_sha256"])
    assert record["eligible"] == 8 and record["context_width"] == 8


def test_audio_digests_are_content_addressed_and_order_sensitive():
    md = _md()
    first = mq.context_record(md, 0, 8)["context_audio_sha256"]
    again = mq.context_record(md, 0, 8)["context_audio_sha256"]
    assert first == again
    shuffled = dict(md, context_audio=md["context_audio"].flip(0))
    assert mq.context_record(shuffled, 0, 8)["context_audio_sha256"] == first[::-1]


def test_target_absence_guard_refuses_a_context_that_is_the_target():
    md = _md()
    md["source"] = md["context_poses"][3].clone()               # a context IS the target
    with pytest.raises(ValueError, match="target"):
        mq.context_record(md, 0, 8)
    ok = _md()
    assert mq.context_record(ok, 0, 8)["target_absent"] is True


def test_record_refuses_a_context_width_that_is_not_the_released_eight():
    md = _md(width=6)
    with pytest.raises(ValueError, match="width"):
        mq.context_record(md, 0, 6)


# --------------------------------------------------------------------------- #
# materialize -> census -> filter, in that order
# --------------------------------------------------------------------------- #
def _materialized(n=12, room_of=None, eligible_of=None):
    room_of = room_of or (lambda i: "Cafe/Cafe_idx_1" if i < 6 else
                          "ListeningRoom/ListeningRoom_idx_2")
    eligible_of = eligible_of or (lambda i: 8)
    records = []
    for position in range(n):
        md = _md(position=position, room=room_of(position))
        records.append(mq.context_record(md, position, eligible_of(position)))
    return {"records": records, "complete": True, "n_records": n,
            "protocol": dict(mq.EXP01_LOADER), "dataset_config": "fixture.json"}


def test_filtering_before_a_complete_pass_is_refused():
    """Filtering first would change worker assignment and RNG consumption for
    the retained queries (inherited plan §1.1)."""
    partial = _materialized()
    partial["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        mq.filter_excluded_room(partial, expected_excluded=6)


def test_filter_removes_exactly_the_excluded_room():
    full = _materialized()
    filtered = mq.filter_excluded_room(full, expected_excluded=6)
    assert filtered["n_records"] == 6
    assert {r["room_id"] for r in filtered["records"]} == {"Cafe/Cafe_idx_1"}
    assert filtered["excluded"]["room_id"] == mq.EXCLUDED_ROOM
    assert filtered["excluded"]["n_excluded"] == 6
    assert len(filtered["excluded"]["query_ids"]) == 6
    assert filtered["complete"] is True


def test_filter_refuses_any_extra_loss():
    full = _materialized()
    with pytest.raises(ValueError, match="exactly"):
        mq.filter_excluded_room(full, expected_excluded=5)
    # a record that vanished for another reason must not be silently absorbed
    damaged = dict(full, records=full["records"][:-1], n_records=len(full["records"]) - 1)
    with pytest.raises(ValueError, match="exactly"):
        mq.filter_excluded_room(damaged, expected_excluded=6)


def test_histogram_counts_eligible_pool_sizes():
    full = _materialized(n=10, eligible_of=lambda i: 6 if i < 3 else (7 if i < 5 else 8))
    assert mq.eligible_histogram(full["records"]) == {6: 3, 7: 2, 8: 5}


def test_short_context_queries_are_replacement_drawn_never_dropped():
    """The 520 short-pool queries keep width 8 through the released replace=True
    fallback; a dropped or narrowed context is a protocol violation."""
    records = _materialized(n=4, room_of=lambda i: "Cafe/Cafe_idx_1",
                            eligible_of=lambda i: 6 if i < 2 else 8)["records"]
    audit = mq.short_context_audit(records, width=8)
    assert audit["n_short"] == 2 and audit["rooms"] == ["Cafe/Cafe_idx_1"]
    assert audit["all_width_eight"] is True and audit["n_dropped"] == 0

    narrowed = [dict(records[0], context_width=6, context_fingerprints=[""] * 6)]
    with pytest.raises(ValueError, match="width"):
        mq.short_context_audit(narrowed, width=8)


# --------------------------------------------------------------------------- #
# the manifest: content-hashed, byte-stable, and never a redraw
# --------------------------------------------------------------------------- #
def test_manifest_hashes_both_streams_and_reloads_byte_stable(tmp_path):
    full = _materialized()
    filtered = mq.filter_excluded_room(full, expected_excluded=6)
    manifest = mq.build_manifest(full, filtered)
    assert len(manifest["full_stream_sha256"]) == 64
    assert len(manifest["filtered_stream_sha256"]) == 64
    assert manifest["full_stream_sha256"] != manifest["filtered_stream_sha256"]
    assert manifest["n_full"] == 12 and manifest["n_filtered"] == 6
    assert manifest["protocol"] == dict(mq.EXP01_LOADER)
    assert manifest["excluded"]["room_id"] == mq.EXCLUDED_ROOM

    path = mq.write_manifest(str(tmp_path / "ctx.json"), manifest)
    first = open(path, "rb").read()
    reloaded = mq.load_manifest(path)
    assert reloaded == manifest
    assert mq.write_manifest(str(tmp_path / "again.json"), reloaded)
    assert open(str(tmp_path / "again.json"), "rb").read() == first


def test_manifest_reload_does_not_touch_the_loader(tmp_path, monkeypatch):
    """Reuse means reuse: loading a frozen manifest may not redraw anything."""
    manifest = mq.build_manifest(_materialized(),
                                 mq.filter_excluded_room(_materialized(),
                                                         expected_excluded=6))
    path = mq.write_manifest(str(tmp_path / "ctx.json"), manifest)

    def refuse(*_args, **_kwargs):
        raise AssertionError("load_manifest built a dataloader")

    monkeypatch.setattr(mq, "build_release_stack", refuse)
    assert mq.load_manifest(path)["full_stream_sha256"] == manifest["full_stream_sha256"]


def test_manifest_detects_a_tampered_record(tmp_path):
    full = _materialized()
    filtered = mq.filter_excluded_room(full, expected_excluded=6)
    manifest = mq.build_manifest(full, filtered)
    path = mq.write_manifest(str(tmp_path / "ctx.json"), manifest)
    payload = json.loads(open(path).read())
    payload["records"][0]["context_fingerprints"][0] = "9.999999,9.999999,9.999999"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with pytest.raises(ValueError, match="sha256"):
        mq.load_manifest(path)


def test_manifest_verifies_the_census_when_it_is_the_real_split(tmp_path):
    full = _materialized(n=12)
    filtered = mq.filter_excluded_room(full, expected_excluded=6)
    manifest = mq.build_manifest(full, filtered)
    assert manifest["census_verified"] is False        # a fixture is not the split
    with pytest.raises(ValueError, match="6,337|6337"):
        mq.assert_registered_census(manifest)


# --------------------------------------------------------------------------- #
# the real split: pinned census (skipped when the dataset is absent)
# --------------------------------------------------------------------------- #
_REAL = os.path.isdir(_AR_ROOT) and os.path.isfile(_UNSEEN_CONFIG)


def test_registered_constants_match_the_inherited_plan():
    assert mq.FULL_COUNT == 6337 and mq.FILTERED_COUNT == 5337
    assert mq.EXCLUDED_COUNT == 1000
    assert mq.EXCLUDED_ROOM == "ListeningRoomsWithHallway/ListeningRoomsWithHallway_idx_2" \
        or mq.EXCLUDED_ROOM.endswith("ListeningRoom_idx_2")
    assert mq.FULL_ELIGIBLE_HISTOGRAM == {6: 91, 7: 429, 8: 5263, 9: 554}
    assert mq.FILTERED_ELIGIBLE_HISTOGRAM == {6: 91, 7: 429, 8: 4363, 9: 454}
    assert sum(mq.FULL_ELIGIBLE_HISTOGRAM.values()) == mq.FULL_COUNT
    assert sum(mq.FILTERED_ELIGIBLE_HISTOGRAM.values()) == mq.FILTERED_COUNT
    assert mq.EXP01_LOADER == {"seed": 42, "batch_size": 64, "num_workers": 4,
                               "shuffle": False}
    assert mq.CONTEXT_WIDTH == 8
    # the two histograms differ only inside the excluded room
    difference = {size: mq.FULL_ELIGIBLE_HISTOGRAM[size]
                  - mq.FILTERED_ELIGIBLE_HISTOGRAM[size]
                  for size in mq.FULL_ELIGIBLE_HISTOGRAM}
    assert difference == {6: 0, 7: 0, 8: 900, 9: 100}
    assert sum(difference.values()) == mq.EXCLUDED_COUNT


@pytest.mark.skipif(not _REAL, reason="AcousticRooms not present")
def test_real_split_eligible_histogram_matches_the_pin():
    """Pure file-tree census -- no loader, no RNG, so it is cheap and exact."""
    histogram = mq.census_from_split(_UNSEEN_CONFIG)
    assert histogram["full"] == mq.FULL_ELIGIBLE_HISTOGRAM
    assert histogram["filtered"] == mq.FILTERED_ELIGIBLE_HISTOGRAM
    assert histogram["n_full"] == mq.FULL_COUNT
    assert histogram["n_filtered"] == mq.FILTERED_COUNT
    assert histogram["short_context_rooms"] == ["Cafe/Cafe_idx_1"]


@pytest.mark.skipif(not _REAL, reason="AcousticRooms not present")
def test_real_bounded_slice_materializes_through_the_released_loader(tmp_path):
    """A bounded slice only: the full 6,337-record pass is a ladder step."""
    materialized = mq.materialize_contexts(_UNSEEN_CONFIG, limit=8)
    assert materialized["complete"] is False          # a slice is never complete
    assert materialized["n_records"] == 8
    assert materialized["protocol"] == dict(mq.EXP01_LOADER)
    for record in materialized["records"]:
        assert record["context_width"] == 8
        assert len(record["context_audio_sha256"]) == 8
        assert record["target_absent"] is True
    with pytest.raises(ValueError, match="complete"):
        mq.filter_excluded_room(materialized, expected_excluded=mq.EXCLUDED_COUNT)


# --------------------------------------------------------------------------- #
# r2 F1 -- the COMPLETE released initialization order
# --------------------------------------------------------------------------- #
def test_release_stack_builds_the_metric_callback_before_the_iterator():
    """PyTorch draws each worker's base seed when the ITERATOR is created, and
    the released evaluator constructs the AGREE metric stack in between (r1
    review F1). Seeding and iterating immediately gives different workers a
    different NumPy stream, hence different contexts."""
    order = mq.RELEASE_CALL_GRAPH
    assert order == ("seed_everything", "build_dataloader", "build_metric_stack",
                     "create_iterator")


@pytest.mark.skipif(not _REAL, reason="AcousticRooms not present")
def test_release_stack_resolves_and_records_the_agree_checkpoint():
    loader, facts = mq.build_release_stack(_UNSEEN_CONFIG)
    assert facts["call_graph"] == list(mq.RELEASE_CALL_GRAPH)
    assert facts["agree_ckpt"].endswith("AGREE_fullAR.pt")
    assert os.path.isfile(facts["agree_ckpt"])
    assert facts["agree_configured"] == "weights/AGREE/AGREE_fullAR.pt"
    assert facts["agree_resolution"] in ("configured", "basename_fallback")
    assert len(facts["agree_sha256"]) == 64
    assert facts["metric_stack_built"] is True
    assert len(facts["rng_digest_at_iter"]) == 64
    assert loader is not None


def test_release_stack_refuses_a_missing_agree_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(mq, "AGREE_SEARCH_ROOTS", (str(tmp_path),))
    with pytest.raises(ValueError, match="AGREE"):
        mq.resolve_agree_checkpoint("weights/AGREE/does_not_exist.pt")


@pytest.mark.skipif(not _REAL, reason="AcousticRooms not present")
def test_our_call_graph_matches_an_eval_flac_faithful_reference():
    """Both graphs, on a bounded real slice: same RNG state at iterator creation
    and the same drawn contexts."""
    import json as _json

    import pytorch_lightning as pl

    from src.data.dataset import create_dataloader_from_config
    from src.training import create_metric_callback_from_config

    n_records = 4

    def reference():
        """eval_FLAC.py's order, written out here rather than imported."""
        with open(_UNSEEN_CONFIG) as handle:
            dataset_config = _json.load(handle)
        with open(mq.DEFAULT_MODEL_CONFIG) as handle:
            model_config = _json.load(handle)
        model_config = mq.with_resolved_agree(model_config)
        pl.seed_everything(42, workers=True)
        loader = create_dataloader_from_config(
            dataset_config, batch_size=64, num_workers=4,
            sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 1), shuffle=False)
        create_metric_callback_from_config(
            model_config, dataset_id=dataset_config["datasets"][0]["id"], per_scene=False)
        digest = mq.rng_state_digest()
        drawn = []
        for _reals, metadata in loader:
            for md in metadata:
                drawn.append(mq.context_record(md, len(drawn), 8))
                if len(drawn) >= n_records:
                    return digest, drawn
        return digest, drawn

    reference_digest, reference_records = reference()
    ours = mq.materialize_contexts(_UNSEEN_CONFIG, limit=n_records)
    assert ours["protocol_facts"]["rng_digest_at_iter"] == reference_digest
    assert [r["query_id"] for r in ours["records"]] == \
        [r["query_id"] for r in reference_records]
    assert [r["context_fingerprints"] for r in ours["records"]] == \
        [r["context_fingerprints"] for r in reference_records]
    assert [r["context_audio_sha256"] for r in ours["records"]] == \
        [r["context_audio_sha256"] for r in reference_records]


@pytest.mark.skipif(not _REAL, reason="AcousticRooms not present")
def test_skipping_the_metric_stack_changes_the_draw():
    """The guard is only meaningful if the omission is detectable."""
    ours = mq.materialize_contexts(_UNSEEN_CONFIG, limit=4)
    without = mq.materialize_contexts(_UNSEEN_CONFIG, limit=4, _skip_metric_stack=True)
    assert without["protocol_facts"]["metric_stack_built"] is False
    assert (without["protocol_facts"]["rng_digest_at_iter"]
            != ours["protocol_facts"]["rng_digest_at_iter"])


# --------------------------------------------------------------------------- #
# r2 F2 -- the positional substitution guard
# --------------------------------------------------------------------------- #
def test_expected_enumeration_comes_from_the_split(tmp_path):
    expected = mq.expected_enumeration({"datasets": [{"path": "AcousticRooms",
                                                      "json_file_path": "data/AR/unseen_eval.json",
                                                      "folder_name": "single_channel_ir_1"}]})
    if not _REAL:
        pytest.skip("AcousticRooms not present")
    assert len(expected) == mq.FULL_COUNT
    assert all(name.endswith("_hybrid_IR.wav") for name in expected)
    assert len(set(expected)) == len(expected)


def test_position_guard_accepts_the_matching_item():
    md = _md(position=3, room="Cafe/Cafe_idx_1", node="S001", receiver="R008")
    mq.assert_stream_position(md, position=3, expected_relpath=md["relpath"])


def test_position_guard_rejects_a_substituted_item():
    """SampleDataset silently returns a random OTHER item when one fails to
    load; counts and even the histogram can survive that (r1 review F2)."""
    md = _md(position=3, room="Cafe/Cafe_idx_1", node="S001")
    substituted = dict(md, idx=97,
                       relpath="single_channel_ir_1/Cafe/Cafe_idx_1/S007_R019_hybrid_IR.wav")
    with pytest.raises(ValueError, match="idx"):
        mq.assert_stream_position(substituted, position=3,
                                  expected_relpath=md["relpath"])
    same_idx_other_file = dict(md, relpath="single_channel_ir_1/Cafe/Cafe_idx_1/"
                                           "S009_R002_hybrid_IR.wav")
    with pytest.raises(ValueError, match="relpath"):
        mq.assert_stream_position(same_idx_other_file, position=3,
                                  expected_relpath=md["relpath"])


def test_materializer_aborts_on_the_first_substitution(tmp_path, monkeypatch):
    """An adversarial substitution at one position must stop the pass, not be
    recorded as if it were that position's query."""
    records = []

    class _FakeLoader:
        def __iter__(self):
            for position in range(4):
                md = _md(position=position, room="Cafe/Cafe_idx_1",
                         node=f"S00{position + 1}")
                if position == 2:                       # the impostor
                    md = dict(md, idx=99,
                              relpath="single_channel_ir_1/Cafe/Cafe_idx_1/"
                                      "S008_R008_hybrid_IR.wav")
                yield None, [md]

    expected = [f"AcousticRooms/single_channel_ir_1/Cafe/Cafe_idx_1/"
                f"S00{i + 1}_R008_hybrid_IR.wav" for i in range(4)]
    monkeypatch.setattr(mq, "build_release_stack",
                        lambda *a, **k: (_FakeLoader(), {"metric_stack_built": True,
                                                         "rng_digest_at_iter": "d" * 64,
                                                         "call_graph": list(mq.RELEASE_CALL_GRAPH)}))
    monkeypatch.setattr(mq, "expected_enumeration", lambda *a, **k: expected)
    monkeypatch.setattr(mq, "eligible_pool_size", lambda path: 8)
    monkeypatch.setattr(mq, "assert_split_enumeration", lambda *a, **k: len(expected))
    with pytest.raises(ValueError, match="position 2"):
        mq.materialize_contexts("fixture.json")
    assert records == []


def test_materializer_refuses_a_short_or_reordered_stream(tmp_path, monkeypatch):
    class _ShortLoader:
        def __iter__(self):
            for position in range(2):
                yield None, [_md(position=position, node=f"S00{position + 1}")]

    expected = [f"AcousticRooms/single_channel_ir_1/Cafe/Cafe_idx_1/"
                f"S00{i + 1}_R008_hybrid_IR.wav" for i in range(4)]
    monkeypatch.setattr(mq, "build_release_stack",
                        lambda *a, **k: (_ShortLoader(), {"metric_stack_built": True,
                                                          "rng_digest_at_iter": "d" * 64,
                                                          "call_graph": list(mq.RELEASE_CALL_GRAPH)}))
    monkeypatch.setattr(mq, "expected_enumeration", lambda *a, **k: expected)
    monkeypatch.setattr(mq, "eligible_pool_size", lambda path: 8)
    monkeypatch.setattr(mq, "assert_split_enumeration", lambda *a, **k: len(expected))
    with pytest.raises(ValueError, match="2 records|declares 4"):
        mq.materialize_contexts("fixture.json")


def test_in_scope_loss_is_refused_even_at_the_right_count():
    """A record lost from an INCLUDED room and replaced by an extra excluded-room
    record keeps the total at 6,337; the split comparison is what catches it."""
    full = _materialized(n=12)
    full["records"][0] = dict(full["records"][0],
                              room_id="ListeningRoom/ListeningRoom_idx_2")
    with pytest.raises(ValueError, match="exactly"):
        mq.filter_excluded_room(full, expected_excluded=6)


# --------------------------------------------------------------------------- #
# r3 F2 -- exact canonical equality, hard enumeration, mandatory censuses
# --------------------------------------------------------------------------- #
def test_canonical_relpath_strips_the_dataset_root_once():
    roots = ("AcousticRooms",)
    tail = "single_channel_ir_1/Cafe/Cafe_idx_1/S006_R008_hybrid_IR.wav"
    assert mq.canonical_relpath(f"AcousticRooms/{tail}", roots) == tail
    assert mq.canonical_relpath(f"./AcousticRooms/{tail}", roots) == tail
    assert mq.canonical_relpath(tail, roots) == tail
    assert mq.canonical_relpath(f"/abs/AcousticRooms/{tail}", roots) == tail


def test_position_guard_is_exact_not_a_suffix_match():
    """r2 re-review: bidirectional endswith accepts a basename or a partial
    component, so a different room's file could impersonate this position."""
    expected = "AcousticRooms/single_channel_ir_1/Cafe/Cafe_idx_1/S001_R008_hybrid_IR.wav"
    md = _md(position=3)
    md["relpath"] = "single_channel_ir_1/Cafe/Cafe_idx_1/S001_R008_hybrid_IR.wav"
    mq.assert_stream_position(md, 3, expected)

    for spoof in ("S001_R008_hybrid_IR.wav",                       # basename only
                  "Cafe_idx_1/S001_R008_hybrid_IR.wav",            # partial components
                  "single_channel_ir_1/Cafe/Cafe_idx_11/S001_R008_hybrid_IR.wav",
                  "single_channel_ir_1/Office/Office_idx_11/S001_R008_hybrid_IR.wav"):
        with pytest.raises(ValueError, match="relpath"):
            mq.assert_stream_position(dict(md, relpath=spoof), 3, expected)


def test_enumeration_failure_is_a_refusal_not_an_empty_expectation(monkeypatch):
    """An enumeration that cannot be built disabled the path check entirely."""
    def explode(*_args, **_kwargs):
        raise OSError("split unreadable")

    monkeypatch.setattr(mq, "expected_enumeration", explode)
    monkeypatch.setattr(mq, "build_release_stack",
                        lambda *a, **k: (iter(()), {"metric_stack_built": True,
                                                    "rng_digest_at_iter": "d" * 64,
                                                    "call_graph": list(mq.RELEASE_CALL_GRAPH)}))
    with pytest.raises(ValueError, match="enumeration"):
        mq.materialize_contexts("fixture.json")


def test_split_identities_are_asserted_before_the_pass(monkeypatch, tmp_path):
    """6,337 unique ordered identities, checked before a single draw."""
    good = [f"AcousticRooms/single_channel_ir_1/Cafe/Cafe_idx_1/S{i:03d}_R008_hybrid_IR.wav"
            for i in range(1, 5)]
    assert mq.assert_split_enumeration(good, expected_count=4) == 4
    with pytest.raises(ValueError, match="4,337|expects|declares"):
        mq.assert_split_enumeration(good, expected_count=4337)
    with pytest.raises(ValueError, match="duplicate"):
        mq.assert_split_enumeration(good + [good[0]], expected_count=5)
    with pytest.raises(ValueError, match="empty|no entries"):
        mq.assert_split_enumeration([], expected_count=0)


def test_census_is_mandatory_after_a_complete_pass():
    """A complete pass whose histogram is not the registered one is refused."""
    full = _materialized(n=12)
    with pytest.raises(ValueError, match="census|histogram"):
        mq.assert_pass_census(full)
    assert mq.assert_pass_census(full, expected_count=12,
                                 expected_histogram={8: 12}) is True

    wrong = _materialized(n=12, eligible_of=lambda i: 7 if i == 0 else 8)
    with pytest.raises(ValueError, match="census|histogram"):
        mq.assert_pass_census(wrong, expected_count=12, expected_histogram={8: 12})


def test_full_pass_enforces_the_registered_census(monkeypatch):
    """The materializer applies the census itself; a slice never does."""
    records = [mq.context_record(_md(position=i), i, 8) for i in range(3)]

    class _Loader:
        def __iter__(self):
            for position in range(3):
                yield None, [_md(position=position, node=f"S00{position + 1}")]

    expected = [f"AcousticRooms/single_channel_ir_1/Cafe/Cafe_idx_1/"
                f"S00{i + 1}_R008_hybrid_IR.wav" for i in range(3)]
    monkeypatch.setattr(mq, "build_release_stack",
                        lambda *a, **k: (_Loader(), {"metric_stack_built": True,
                                                     "rng_digest_at_iter": "d" * 64,
                                                     "call_graph": list(mq.RELEASE_CALL_GRAPH)}))
    monkeypatch.setattr(mq, "expected_enumeration", lambda *a, **k: expected)
    monkeypatch.setattr(mq, "eligible_pool_size", lambda path: 8)
    monkeypatch.setattr(mq, "assert_split_enumeration", lambda *a, **k: len(expected))
    with pytest.raises(ValueError, match="census|histogram"):
        mq.materialize_contexts("fixture.json")          # 3 records != 6,337
    sliced = mq.materialize_contexts("fixture.json", limit=2)
    assert sliced["complete"] is False and sliced["n_records"] == 2
    assert records[0]["query_id"]
