"""exp_22 r9b -- the sparse/metadata-bank AGREE retrieval control (§2).

The control answers one question per query: where would pure AGREE
nearest-neighbour retrieval place the source if it were restricted to the REAL
dataset RIRs that actually exist at this query's receiver? Almost every test
here is therefore a BANK test -- what may enter the bank, what may never enter
it, and what happens when nothing can -- plus the label tests that keep its
numbers from being read as the dense-grid engine's.

The fixture extends the r9 report fixture (one definition of the run, the
binding, the G1 plan and the D1 records) with a dataset tree the bank can be
built from: pair metadata for every (source, receiver) of each room and IR files
for all but a deliberately missing one. No GPU and no AGREE checkpoint is
involved: the embedder is the report fixture's synthetic one, and the waveform
reader is injected.
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization import meshgrid_retrieval_control as rc

# the r9 fixture builders, loaded by path -- the same seam
# test_loc_meshgrid_offgrid_probe.py uses, so there is ONE fixture run.
_SPEC = importlib.util.spec_from_file_location(
    "_meshgrid_report_fixtures",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "test_loc_meshgrid_report.py"))
fx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fx)


# --------------------------------------------------------------------------- #
# the dataset tree the bank is built from
# --------------------------------------------------------------------------- #
#: every source node of each fixture room and its world position. The three
#: query sources carry the truths the r9 fixture registers, because ``src_loc``
#: is a property of the SOURCE and the pair files must agree about it.
FIXTURE_SOURCES = {
    "A/A_idx_1": {1: [1.1, 1.1, 0.6],     # the truth of query 0 (S001_R002)
                  2: [2.2, 0.2, 0.7],
                  3: [0.4, 2.4, 0.6],     # the truth of query 1 (S003_R004)
                  5: [1.6, 1.6, 1.4],     # the truth of query 2 (S005_R002)
                  7: [0.3, 0.3, 1.2],
                  10: [2.4, 2.4, 0.9]},
    "B/B_idx_2": {1: [2.4, 0.4, 1.1],     # the truth of query 3 (S001_R009)
                  2: [0.6, 2.2, 1.0],
                  4: [1.9, 1.9, 1.3]},
}

#: every receiver node of each room and its world position (the r9 fixture's).
FIXTURE_RECEIVERS = {
    "A/A_idx_1": {2: [0.0, 0.0, 0.5], 4: [2.5, 2.5, 1.5]},
    "B/B_idx_2": {9: [1.0, 1.0, 1.0]},
}

#: the one (source, receiver) whose IR file is deliberately absent, so "only
#: where an exact dataset RIR exists" is exercised rather than assumed.
MISSING_IR = ("A/A_idx_1", 7, 2)

FIXTURE_ROOT = "ir"                       # the r9 fixture's folder_name


def _ir_relpath(room_id, src_node, rec_node):
    scene, scene_id = room_id.split("/")
    return f"{FIXTURE_ROOT}/{scene}/{scene_id}/S{src_node:03d}_R{rec_node:03d}_hybrid_IR.wav"


def _write_dataset_tree(tmp_path, metadata_root):
    """Pair metadata for every (src, rec) plus the IR files that exist.

    The IR payload is a marker, not audio: the tests that score inject their own
    reader. One real wav is written by the reader test itself.
    """
    dataset_root = str(tmp_path / "AcousticRooms")
    for room_id, sources in FIXTURE_SOURCES.items():
        scene, scene_id = room_id.split("/")
        meta_dir = os.path.join(metadata_root, scene, scene_id)
        ir_dir = os.path.join(dataset_root, FIXTURE_ROOT, scene, scene_id)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(ir_dir, exist_ok=True)
        for src_node, src_loc in sources.items():
            for rec_node, rec_loc in FIXTURE_RECEIVERS[room_id].items():
                with open(os.path.join(meta_dir,
                                       f"S{src_node:03d}_R{rec_node:03d}.json"), "w") as handle:
                    json.dump({"src_loc": [float(v) for v in src_loc],
                               "rec_loc": [float(v) for v in rec_loc]}, handle)
                if (room_id, src_node, rec_node) == MISSING_IR:
                    continue
                with open(os.path.join(dataset_root,
                                       _ir_relpath(room_id, src_node, rec_node)), "wb") as handle:
                    handle.write(f"{room_id}|{src_node}|{rec_node}".encode("utf-8"))
    return dataset_root


class FakeReader:
    """A deterministic stand-in for the released waveform read.

    Every RIR is a different constant ramp keyed by its file name, so no two
    bank entries can embed to the same vector and a cosine can order them.
    """

    def __init__(self, length=16):
        self.length = int(length)
        self.calls = []

    def __call__(self, path):
        self.calls.append(str(path))
        src, rec = rc.parse_ir_filename(os.path.basename(str(path)))
        base = 0.01 * src + 0.001 * rec
        return (torch.arange(self.length, dtype=torch.float32).reshape(1, 1, -1) * 0.001
                + base)


def build_control_fixture(tmp_path, pre_register=True):
    """The r9 run fixture plus the dataset tree and the control's inputs.

    The sparse-bank digest is computed and PRE-REGISTERED by default (the r9e
    canonical model), so a fixture report is canonical except for whatever the
    test deliberately deviates -- and ``allow_non_canonical`` is on, because most
    tests run a 64-resample bootstrap rather than the registered 10,000.
    """
    fixture = fx.build_fixture_run(tmp_path)
    fixture["dataset_root"] = _write_dataset_tree(tmp_path, fixture["metadata_root"])
    fixture["embedder"] = fx.SyntheticEngine()._embed
    fixture["reader"] = FakeReader()
    fixture["out_dir"] = str(tmp_path / "retrieval")
    fixture["totals"] = rc.retrieval_totals(rooms=2, queries=4)
    document = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                              fixture["records"])
    fixture["bank_document"] = document
    fixture["bank_gate"] = rc.assert_bank_digest(
        document["sha256"], expected=document["sha256"] if pre_register else None,
        allow_non_canonical=not pre_register)
    fixture["agree_ckpt_sha256"] = rc.REGISTERED_AGREE_SHA256
    fixture["scorer_readout"] = me.SCORER_READOUT
    fixture["tau"] = me.TAU
    fixture["bank_rule"] = rc.REGISTERED_BANK_RULE
    fixture["allow_non_canonical"] = True
    return fixture


def run_control(fixture, **kwargs):
    kwargs.setdefault("reader", fixture["reader"])
    kwargs.setdefault("records", fixture["records"])
    records = kwargs.pop("records")
    return rc.run_retrieval(fixture["embedder"], list(fixture["items"]), records,
                            fixture["plan"], metadata_root=fixture["metadata_root"],
                            dataset_root=fixture["dataset_root"], **kwargs)


def _by_id(results):
    return {result["query_id"]: result for result in results}


def _query_id(room_id, index):
    return fx.fixture_query_id(room_id, fx.FIXTURE_QUERIES[room_id][index])


# --------------------------------------------------------------------------- #
# the labels: what this control is, and what it is NOT
# --------------------------------------------------------------------------- #
def test_the_control_label_says_it_is_not_the_dense_grid():
    for phrase in ("SPARSE / METADATA-BANK", "real dataset RIRs", "NOT the dense",
                   "oracle floor", "never the dense-grid"):
        assert phrase in rc.CONTROL_LABEL


def test_the_sparse_oracle_label_names_the_bank_it_is_taken_over():
    assert "SPARSE-BANK ORACLE" in rc.SPARSE_ORACLE_LABEL
    assert "dense" in rc.SPARSE_ORACLE_LABEL


def test_the_self_pair_rule_states_the_exclusion_out_loud():
    assert "own observation" in rc.SELF_PAIR_RULE
    assert "excluded" in rc.SELF_PAIR_RULE


def test_the_bank_rule_note_names_the_released_selector_quirk():
    assert rc.REGISTERED_BANK_RULE == "numeric_identity"
    assert set(rc.BANK_RULES) == {"numeric_identity", "released_eligible_pool"}
    assert "S010" in rc.BANK_RULE_NOTE
    assert "SUPERSET" in rc.BANK_RULE_NOTE


def test_the_overlap_with_the_models_conditioning_is_stated(tmp_path):
    for phrase in ("OVERLAPS", "NOT independent", "same real evidence"):
        assert phrase in rc.CONTEXT_OVERLAP_NOTE
    fixture = build_control_fixture(tmp_path)
    report = rc.build_report(run_control(fixture), fixture, n_boot=64)
    published = rc.write_report(fixture["out_dir"], report)
    assert json.load(open(published["paths"]["json"]))["labels"]["context_overlap"] == \
        rc.CONTEXT_OVERLAP_NOTE
    assert json.load(open(published["paths"]["handoff"]))["context_overlap"] == \
        rc.CONTEXT_OVERLAP_NOTE
    assert rc.CONTEXT_OVERLAP_NOTE in open(published["paths"]["markdown"]).read()


def test_the_control_carries_the_engines_own_leakage_caveat_verbatim():
    # copied from the engine, never paraphrased
    assert rc.AGREE_LEAKAGE_CAVEAT == me.AGREE_LEAKAGE_CAVEAT
    assert rc.SUBSET_LABEL == mr.SUBSET_LABEL


def test_the_control_key_is_the_one_the_r1_report_names(tmp_path):
    assert rc.CONTROL_KEY in mr.CONTROLS_ELSEWHERE
    where = mr.CONTROLS_ELSEWHERE[rc.CONTROL_KEY]
    assert "meshgrid_retrieval_control.py" in where
    assert "NOT IMPLEMENTED" not in where
    assert "built (r9b)" in where
    assert rc.HANDOFF_JSON in where


# --------------------------------------------------------------------------- #
# the bank: what may enter it
# --------------------------------------------------------------------------- #
def test_the_bank_is_the_same_receivers_other_sources(tmp_path):
    fixture = build_control_fixture(tmp_path)
    bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                               "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))
    assert [entry.src_node for entry in bank["entries"]] == [2, 3, 5, 10]
    assert {entry.rec_node for entry in bank["entries"]} == {2}
    assert bank["counts"]["n_bank"] == 4


def test_the_query_pair_is_never_in_its_own_bank(tmp_path):
    fixture = build_control_fixture(tmp_path)
    for room_id, index, src_node, rec_node in (("A/A_idx_1", 0, 1, 2),
                                               ("A/A_idx_1", 1, 3, 4),
                                               ("A/A_idx_1", 2, 5, 2),
                                               ("B/B_idx_2", 0, 1, 9)):
        relpath = _ir_relpath(room_id, src_node, rec_node)
        bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                                   room_id, relpath)
        assert (src_node, rec_node) not in [(e.src_node, e.rec_node) for e in bank["entries"]]
        assert bank["counts"]["n_self_excluded"] == 1
        assert os.path.basename(relpath) not in [os.path.basename(e.ir_path)
                                                 for e in bank["entries"]]


def test_a_query_whose_own_pair_metadata_is_absent_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    os.remove(os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json"))
    with pytest.raises(ValueError, match="no pair metadata of its own"):
        rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                            "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))


def test_another_receivers_sources_never_enter_the_bank(tmp_path):
    fixture = build_control_fixture(tmp_path)
    bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                               "A/A_idx_1", _ir_relpath("A/A_idx_1", 3, 4))
    assert {entry.rec_node for entry in bank["entries"]} == {4}
    assert [entry.src_node for entry in bank["entries"]] == [1, 2, 5, 7, 10]
    # R002's four entries exist, and none of them is here
    assert bank["counts"]["n_pairs_at_receiver"] == 6


def test_a_pair_without_an_ir_file_is_counted_and_dropped(tmp_path):
    fixture = build_control_fixture(tmp_path)
    bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                               "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))
    assert bank["counts"]["n_missing_ir"] == 1
    assert bank["missing_ir"] == [[MISSING_IR[1], MISSING_IR[2]]]
    assert 7 not in [entry.src_node for entry in bank["entries"]]


def test_an_ir_file_with_no_pair_metadata_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    stray = os.path.join(fixture["dataset_root"], _ir_relpath("A/A_idx_1", 42, 2))
    with open(stray, "wb") as handle:
        handle.write(b"stray")
    with pytest.raises(ValueError, match="no pair metadata"):
        rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                            "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))


def test_a_bank_entry_at_a_different_receiver_position_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S002_R002.json")
    payload = json.load(open(path))
    payload["rec_loc"] = [9.0, 9.0, 9.0]
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="disagree about the receiver"):
        rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                            "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))


def test_an_empty_bank_is_refused_with_the_counts_that_made_it_empty(tmp_path):
    fixture = build_control_fixture(tmp_path)
    scene_dir = os.path.join(fixture["dataset_root"], FIXTURE_ROOT, "B", "B_idx_2")
    for src_node in (2, 4):
        os.remove(os.path.join(scene_dir, f"S{src_node:03d}_R009_hybrid_IR.wav"))
    with pytest.raises(ValueError) as error:
        rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                            "B/B_idx_2", _ir_relpath("B/B_idx_2", 1, 9))
    message = str(error.value)
    assert "B/B_idx_2" in message and "R009" in message
    assert "n_pairs_at_receiver=3" in message
    assert "n_missing_ir=2" in message
    assert "n_self_excluded=1" in message


def test_a_bank_entry_sitting_on_the_truth_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S002_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = list(FIXTURE_SOURCES["A/A_idx_1"][1])      # the query's truth
    with open(path, "w") as handle:
        json.dump(payload, handle)
    bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                               "A/A_idx_1", _ir_relpath("A/A_idx_1", 1, 2))
    with pytest.raises(ValueError, match="sits on the held-out target"):
        rc.assert_bank_excludes_the_target(bank["entries"], FIXTURE_SOURCES["A/A_idx_1"][1],
                                           query_id="0|x.wav")


def test_the_released_pool_rule_drops_the_s010_the_selector_cannot_render(tmp_path):
    fixture = build_control_fixture(tmp_path)
    relpath = _ir_relpath("A/A_idx_1", 1, 2)
    numeric = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                                  "A/A_idx_1", relpath, rule="numeric_identity")
    released = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                                   "A/A_idx_1", relpath, rule="released_eligible_pool")
    assert [e.src_node for e in numeric["entries"]] == [2, 3, 5, 10]
    assert [e.src_node for e in released["entries"]] == [2, 3, 5]
    assert numeric["counts"]["n_released_eligible"] == 3
    assert released["rule"] == "released_eligible_pool"
    # the released pool is exactly what the released selector itself returns
    pool = mq.eligible_context_pool(os.path.join(fixture["dataset_root"], relpath))
    assert sorted(os.path.basename(p) for p in pool) == \
        sorted(os.path.basename(e.ir_path) for e in released["entries"])


def test_an_unknown_bank_rule_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown bank rule"):
        rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"], "A/A_idx_1",
                            _ir_relpath("A/A_idx_1", 1, 2), rule="whatever")


def test_the_bank_is_ordered_by_parsed_numeric_identity(tmp_path):
    fixture = build_control_fixture(tmp_path)
    bank = rc.build_query_bank(fixture["metadata_root"], fixture["dataset_root"],
                               "A/A_idx_1", _ir_relpath("A/A_idx_1", 3, 4))
    identities = [(entry.src_node, entry.rec_node) for entry in bank["entries"]]
    assert identities == sorted(identities)
    assert rc.assert_bank_order(bank["entries"]) is True
    with pytest.raises(ValueError, match="ascending numeric identity"):
        rc.assert_bank_order(list(reversed(bank["entries"])))


# --------------------------------------------------------------------------- #
# the sparse-bank digest: the bytes the numbers are made of (r9c BLOCKER)
# --------------------------------------------------------------------------- #
def test_the_digest_algorithm_is_documented():
    for phrase in ("canonical_sha256", "observation", "truth", "bank", "MEMBERSHIP"):
        assert phrase in rc.BANK_DIGEST_ALGORITHM


def test_the_digest_covers_every_byte_the_control_reads(tmp_path):
    fixture = build_control_fixture(tmp_path)
    base = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                          fixture["records"])
    assert base["n_queries"] == 4
    assert base["n_wav_files"] > 0 and base["n_pair_files"] > 0
    assert base["algorithm"] == rc.BANK_DIGEST_ALGORITHM
    assert base["rule"] == rc.REGISTERED_BANK_RULE

    def digest():
        return rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                              fixture["records"])["sha256"]

    assert digest() == base["sha256"]                     # reproducible

    # (a) a bank entry's WAVEFORM bytes
    wav = os.path.join(fixture["dataset_root"], _ir_relpath("A/A_idx_1", 2, 2))
    original = open(wav, "rb").read()
    with open(wav, "wb") as handle:
        handle.write(original + b"!")
    assert digest() != base["sha256"]
    with open(wav, "wb") as handle:
        handle.write(original)
    assert digest() == base["sha256"]

    # (b) a bank entry's POSITION file
    pair = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S002_R002.json")
    payload = json.load(open(pair))
    with open(pair, "w") as handle:
        json.dump(dict(payload, src_loc=[9.0, 9.0, 9.0]), handle)
    assert digest() != base["sha256"]
    with open(pair, "w") as handle:
        json.dump(payload, handle)
    assert digest() == base["sha256"]

    # (c) the TRUTH-carrying pair file of the query itself
    truth_pair = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json")
    payload = json.load(open(truth_pair))
    with open(truth_pair, "w") as handle:
        json.dump(dict(payload, src_loc=[1.2, 1.1, 0.6]), handle)
    assert digest() != base["sha256"]
    with open(truth_pair, "w") as handle:
        json.dump(payload, handle)

    # (d) the OBSERVED RIR's bytes
    observation = os.path.join(fixture["dataset_root"], _ir_relpath("A/A_idx_1", 1, 2))
    original = open(observation, "rb").read()
    with open(observation, "wb") as handle:
        handle.write(original + b"?")
    assert digest() != base["sha256"]
    with open(observation, "wb") as handle:
        handle.write(original)

    # (e) MEMBERSHIP: restoring an IR file the bank was missing changes it too
    restored = os.path.join(fixture["dataset_root"],
                            _ir_relpath(MISSING_IR[0], MISSING_IR[1], MISSING_IR[2]))
    with open(restored, "wb") as handle:
        handle.write(b"restored")
    assert digest() != base["sha256"]
    os.remove(restored)
    assert digest() == base["sha256"]

    # ... and a file no bank reads does not. The name is deliberately one the
    # RELEASED selector can still parse: a stem without an underscore breaks
    # AR_md's own int(fn.split("_")[0][1:]) long before any bank is built, and
    # that tree anomaly is the D1 materialization's to refuse, not this test's
    with open(os.path.join(fixture["dataset_root"], "ir", "A", "A_idx_1",
                           "S999_R002_notes.txt"), "w") as handle:
        handle.write("not an IR")
    assert digest() == base["sha256"]


def test_the_digest_does_not_depend_on_the_order_the_records_arrive_in(tmp_path):
    fixture = build_control_fixture(tmp_path)
    forward = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                             fixture["records"])
    backward = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                              list(reversed(fixture["records"])))
    assert forward["sha256"] == backward["sha256"]


def test_the_digest_changes_with_the_bank_rule(tmp_path):
    fixture = build_control_fixture(tmp_path)
    numeric = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                             fixture["records"], rule="numeric_identity")
    released = rc.bank_digest(fixture["metadata_root"], fixture["dataset_root"],
                              fixture["records"], rule="released_eligible_pool")
    assert numeric["sha256"] != released["sha256"]
    assert released["rule"] == "released_eligible_pool"


def test_a_canonical_run_requires_a_pre_registered_digest():
    found = "a" * 64
    gate = rc.assert_bank_digest(found, expected=found)
    assert gate["canonical"] is True and gate["mode"] == "pre_registered"
    assert gate["sparse_bank_sha256"] == found

    with pytest.raises(ValueError, match="requires the pre-registered sparse-bank digest"):
        rc.assert_bank_digest(found, expected=None)

    recorded = rc.assert_bank_digest(found, expected=None, allow_non_canonical=True)
    assert recorded["canonical"] is False
    assert recorded["mode"] == "recorded_not_pre_registered"
    assert rc.NON_CANONICAL_BANK_NOTE in recorded["note"]

    with pytest.raises(ValueError, match="not the pre-registered sparse bank"):
        rc.assert_bank_digest(found, expected="b" * 64)
    # ... and a mismatch is not excusable by --non-canonical: the operator DID
    # pre-register a value and the tree does not match it
    with pytest.raises(ValueError, match="not the pre-registered sparse bank"):
        rc.assert_bank_digest(found, expected="b" * 64, allow_non_canonical=True)


def test_the_walk_refuses_a_bank_that_moved_after_the_digest(tmp_path):
    fixture = build_control_fixture(tmp_path)
    document = fixture["bank_document"]
    assert run_control(fixture, bank_document=document)                 # the happy path

    os.remove(os.path.join(fixture["dataset_root"], _ir_relpath("A/A_idx_1", 2, 2)))
    with pytest.raises(ValueError, match="changed after the sparse-bank digest"):
        run_control(fixture, bank_document=document)


# --------------------------------------------------------------------------- #
# the score and the prediction
# --------------------------------------------------------------------------- #
def _entries(nodes, rec=2, xyz=None):
    return [rc.BankEntry(src_node=int(node), rec_node=int(rec),
                         ir_path=f"S{int(node):03d}_R{int(rec):03d}_hybrid_IR.wav",
                         pair_path=f"S{int(node):03d}_R{int(rec):03d}.json",
                         src_xyz=np.asarray((xyz or {}).get(node, [float(node), 0.0, 0.0]),
                                            dtype=np.float64),
                         rec_xyz=np.zeros(3))
            for node in nodes]


def test_a_tie_is_broken_by_the_smallest_numeric_identity():
    entries = _entries([2, 3, 5, 10])
    scores = torch.tensor([0.5, 0.9, 0.9, 0.9])
    assert rc.predict_row(scores, entries) == 1          # S003, not S005 or S010
    # ... and the order the tie is broken in is the identity's, not the name's:
    # "S0010" would sort before "S003" lexicographically as a STRING
    assert rc.predict_row(torch.tensor([0.1, 0.2, 0.3, 0.3]), entries) == 2


def test_the_prediction_is_the_bank_entrys_own_source_position():
    entries = _entries([2, 3], xyz={2: [1.0, 2.0, 3.0], 3: [4.0, 5.0, 6.0]})
    scores = torch.tensor([0.1, 0.8])
    row = rc.predict_row(scores, entries)
    assert entries[row].src_xyz.tolist() == [4.0, 5.0, 6.0]


def test_the_score_is_the_cosine_bit_for_bit():
    # r9c review, MAJOR: the K = 1 log-mean-exp equals the cosine ALGEBRAICALLY,
    # but tau * (logsumexp(s / tau) - log 1) is a float32 divide, exp/log and
    # multiply, which can move a score by an ulp and turn two adjacent scores
    # into a tie. The score IS the cosine, and this pins that bit for bit.
    sims = torch.tensor([0.1, -0.25, 0.99, 0.30000001, 0.3, 1.0, -1.0], dtype=torch.float32)
    scores = rc.bank_scores(sims)
    assert torch.equal(scores, sims)
    assert scores.dtype == torch.float32
    assert scores.data_ptr() != sims.data_ptr()          # a copy, never an alias

    # the algebraic equivalence is still true, and it is still only algebraic:
    # the engine's aggregator is allowed to differ in the last bits
    lme = me.nested_scores(sims.reshape(-1, 1), tau=me.TAU, prefixes=(1,))[1]["scores"]
    assert scores.tolist() == pytest.approx(lme.tolist(), abs=1e-6)


def test_the_score_never_reaches_for_tau():
    import inspect

    for name in ("bank_scores", "evaluate_bank_query", "run_retrieval"):
        signature = inspect.signature(getattr(rc, name))
        assert "tau" not in signature.parameters, f"{name} still takes tau"
    # the compiled body, not the docstring: the docstring is allowed to explain
    # the algebra it deliberately does not compute
    names = set(rc.bank_scores.__code__.co_names)
    for forbidden in ("nested_scores", "logsumexp", "log", "aggregate"):
        assert forbidden not in names


# --------------------------------------------------------------------------- #
# the per-query readouts
# --------------------------------------------------------------------------- #
def test_the_sparse_oracle_is_the_nearest_bank_source_to_the_truth():
    entries = _entries([2, 3, 5], xyz={2: [2.2, 0.2, 0.7], 3: [0.4, 2.4, 0.6],
                                       5: [1.6, 1.6, 1.4]})
    truth = np.asarray([1.1, 1.1, 0.6])
    sims = torch.tensor([0.9, 0.1, 0.2])                 # S002 wins
    result = rc.evaluate_bank_query(entries, sims, truth, query_id="0|q.wav",
                                    room_id="A/A_idx_1", position=0, receiver_id="r",
                                    receiver_xyz=[0.0, 0.0, 0.5])
    distances = np.linalg.norm(np.stack([e.src_xyz for e in entries]) - truth, axis=1)
    assert result["e_oracle_sparse"] == pytest.approx(float(distances.min()))
    assert result["oracle_src_node"] == 5                # the nearest is S005
    assert result["e_loc"] == pytest.approx(float(distances[0]))
    assert result["e_excess"] == pytest.approx(float(distances[0] - distances.min()))
    assert result["prediction_src_node"] == 2
    assert result["n_candidates"] == 3 and result["num_samples"] == 1


def test_the_excess_is_clamped_at_zero_when_the_oracle_is_predicted():
    entries = _entries([2, 3], xyz={2: [0.0, 0.0, 0.0], 3: [9.0, 0.0, 0.0]})
    result = rc.evaluate_bank_query(entries, torch.tensor([0.9, 0.1]),
                                    np.asarray([0.5, 0.0, 0.0]), query_id="q", room_id="A/A",
                                    position=0, receiver_id="r", receiver_xyz=[0.0, 0.0, 0.0])
    assert result["e_loc"] == pytest.approx(0.5)
    assert result["e_oracle_sparse"] == pytest.approx(0.5)
    assert result["e_excess"] == 0.0
    assert result["success_raw"]["0.5"] == 1.0
    assert result["success_oracle_normalized"]["0.5"] == 1.0
    assert result["success_raw"]["1.0"] == 1.0


def test_a_prediction_further_than_a_radius_is_not_a_success():
    entries = _entries([2, 3], xyz={2: [0.0, 0.0, 0.0], 3: [9.0, 0.0, 0.0]})
    result = rc.evaluate_bank_query(entries, torch.tensor([0.1, 0.9]),
                                    np.asarray([0.5, 0.0, 0.0]), query_id="q", room_id="A/A",
                                    position=0, receiver_id="r", receiver_xyz=[0.0, 0.0, 0.0])
    assert result["e_loc"] == pytest.approx(8.5)
    assert result["success_raw"]["0.5"] == 0.0 and result["success_raw"]["1.0"] == 0.0
    # the sparse-bank oracle is 0.5 m away, so the excess is what the retrieval
    # lost against its OWN bank
    assert result["e_excess"] == pytest.approx(8.0)
    assert result["success_oracle_normalized"]["1.0"] == 0.0


# --------------------------------------------------------------------------- #
# the whole control on the fixture run
# --------------------------------------------------------------------------- #
def test_the_control_scores_every_registered_query_once(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    assert [result["position"] for result in results] == [0, 1, 2, 3]
    by_id = _by_id(results)
    assert by_id[_query_id("A/A_idx_1", 0)]["n_candidates"] == 4
    assert by_id[_query_id("A/A_idx_1", 1)]["n_candidates"] == 5
    assert by_id[_query_id("B/B_idx_2", 0)]["n_candidates"] == 2
    for result in results:
        assert result["control_label"] == rc.CONTROL_LABEL
        assert result["bank_rule"] == rc.REGISTERED_BANK_RULE
        assert result["num_samples"] == 1
        assert result["e_oracle_sparse"] <= result["e_loc"] + 1e-12
        assert set(result["success_raw"]) == {"0.5", "1.0"}


def test_the_control_records_the_dense_grid_oracle_only_as_a_contrast(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    for result in results:
        query = next(q for room in sorted(fixture["plan"].rooms)
                     for q in me.load_room_plan(fixture["plan"], room).queries
                     if q.query_id == result["query_id"])
        assert result["e_oracle_grid"] == pytest.approx(float(query.oracle))
        assert result["n_grid_candidates"] == query.n_candidates
        # the sparse bank is nine real positions, the grid is a half-metre
        # lattice: the control's own oracle is the sparse one
        assert result["e_oracle_sparse"] != pytest.approx(result["e_oracle_grid"])


def test_the_control_is_deterministic(tmp_path):
    fixture = build_control_fixture(tmp_path)
    first = run_control(fixture)
    second = run_control(fixture)
    assert [r["prediction_src_node"] for r in first] == [r["prediction_src_node"] for r in second]
    assert [r["e_loc"] for r in first] == [r["e_loc"] for r in second]
    assert [r["sims"] for r in first] == [r["sims"] for r in second]


def test_a_bank_entry_is_embedded_once_and_reused_across_the_receivers_queries(tmp_path):
    fixture = build_control_fixture(tmp_path)
    reader = fixture["reader"]
    run_control(fixture)
    # queries 0 and 2 share receiver R002; every distinct IR file is read once
    assert len(reader.calls) == len(set(reader.calls))


def test_the_control_refuses_a_truncated_stream(tmp_path):
    fixture = build_control_fixture(tmp_path)
    with pytest.raises(ValueError, match="the stream ended before"):
        rc.run_retrieval(fixture["embedder"], list(fixture["items"])[:2], fixture["records"],
                         fixture["plan"], metadata_root=fixture["metadata_root"],
                         dataset_root=fixture["dataset_root"], reader=fixture["reader"])


def test_the_control_refuses_a_receiver_that_is_not_the_manifests(tmp_path):
    fixture = build_control_fixture(tmp_path)
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json")
    payload = json.load(open(path))
    payload["rec_loc"] = [0.25, 0.0, 0.5]
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="not resolving the same query"):
        run_control(fixture)


def test_the_control_refuses_a_truth_whose_grid_oracle_is_not_the_manifests(tmp_path):
    fixture = build_control_fixture(tmp_path)
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = [1.11, 1.11, 0.61]           # the truth moves off its cell
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="not looking at the same query"):
        run_control(fixture)


def test_a_g1_room_manifest_that_names_one_query_twice_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    room_plan = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    assert set(rc.query_index(room_plan)) == {q.query_id for q in room_plan.queries}
    room_plan.queries = room_plan.queries + [room_plan.queries[0]]
    with pytest.raises(ValueError, match="twice"):
        rc.query_index(room_plan)


def test_a_room_the_audit_does_not_publish_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    records = [dict(record) for record in fixture["records"]]
    records[3]["room_id"] = "C/C_idx_3"
    with pytest.raises(ValueError, match="no candidate manifest for room"):
        run_control(fixture, records=records)


def test_a_query_the_audit_does_not_publish_in_that_room_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    records = [dict(record) for record in fixture["records"]]
    records[3]["room_id"] = "A/A_idx_1"               # B's query, claimed for room A
    with pytest.raises(ValueError, match="not in the G1 audit's queries"):
        run_control(fixture, records=records)


def test_a_stream_whose_rooms_are_not_contiguous_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    records = [dict(record) for record in fixture["records"]]
    records[1]["room_id"] = "B/B_idx_2"               # A at 0 and 2, B at 1 and 3
    with pytest.raises(ValueError, match="not contiguous"):
        run_control(fixture, records=records)


def test_the_control_refuses_a_context_draw_that_is_not_the_registered_one(tmp_path):
    fixture = build_control_fixture(tmp_path)
    records = [dict(record) for record in fixture["records"]]
    records[0]["context_audio_sha256"] = ["0" * 64] * 8
    with pytest.raises(ValueError, match="context audio digest"):
        rc.run_retrieval(fixture["embedder"], list(fixture["items"]), records,
                         fixture["plan"], metadata_root=fixture["metadata_root"],
                         dataset_root=fixture["dataset_root"], reader=fixture["reader"])


def test_the_control_never_reads_the_held_out_target_from_the_loader(tmp_path):
    fixture = build_control_fixture(tmp_path)

    class Tripwire(dict):
        def __getitem__(self, key):
            if key in ("source", "source_vit"):
                raise AssertionError(f"the control read {key!r} from the loader item")
            return super().__getitem__(key)

    items = [(wav, Tripwire(md)) for wav, md in fixture["items"]]
    results = rc.run_retrieval(fixture["embedder"], items, fixture["records"],
                               fixture["plan"], metadata_root=fixture["metadata_root"],
                               dataset_root=fixture["dataset_root"], reader=fixture["reader"])
    assert len(results) == 4


def test_a_query_with_no_observation_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    items = [(None, md) for _wav, md in fixture["items"]]
    with pytest.raises(ValueError, match="no observed waveform"):
        rc.run_retrieval(fixture["embedder"], items, fixture["records"], fixture["plan"],
                         metadata_root=fixture["metadata_root"],
                         dataset_root=fixture["dataset_root"], reader=fixture["reader"])


# --------------------------------------------------------------------------- #
# the released waveform reader
# --------------------------------------------------------------------------- #
def test_the_reader_returns_a_single_channel_rir_at_the_released_rate(tmp_path):
    import torchaudio

    path = str(tmp_path / "S001_R002_hybrid_IR.wav")
    wave = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32).reshape(1, -1)
    torchaudio.save(path, wave, 22050)
    read = rc.read_rir(path)
    assert tuple(read.shape) == (1, 1, 64)
    assert read.dtype == torch.float32
    assert read[0, 0].tolist() == pytest.approx(wave[0].tolist(), abs=1e-4)


def test_the_reader_refuses_a_rate_the_release_asserts_against(tmp_path):
    import torchaudio

    path = str(tmp_path / "S001_R002_hybrid_IR.wav")
    torchaudio.save(path, torch.zeros(1, 64), 16000)
    with pytest.raises(ValueError, match="22050"):
        rc.read_rir(path)


def test_the_reader_refuses_a_multichannel_file(tmp_path):
    import torchaudio

    path = str(tmp_path / "S001_R002_hybrid_IR.wav")
    torchaudio.save(path, torch.zeros(2, 64), 22050)
    with pytest.raises(ValueError, match="single-channel"):
        rc.read_rir(path)


def test_the_scorer_sees_the_same_input_whether_the_rir_was_cropped_or_not():
    from src.localization.agree_embed import preprocess_for_scoring

    raw = torch.rand(1, 1, 30000) * 2.0 - 1.0
    cropped = raw[..., :10240]                            # the loader's PadCrop at 10240
    assert torch.equal(preprocess_for_scoring(raw), preprocess_for_scoring(cropped))
    short = torch.rand(1, 1, 500) * 2.0 - 1.0
    padded = torch.nn.functional.pad(short, (0, 10240 - short.shape[-1]))
    assert torch.equal(preprocess_for_scoring(short), preprocess_for_scoring(padded))


# --------------------------------------------------------------------------- #
# aggregation, the sparse oracle and the bootstrap
# --------------------------------------------------------------------------- #
def test_the_metric_family_is_the_r1_reports_own():
    assert rc.SUCCESS_RADII == mr.SUCCESS_RADII
    assert rc.BOOTSTRAP_SEED == mr.BOOTSTRAP_SEED == 20260825
    assert rc.BOOTSTRAP_N == mr.BOOTSTRAP_N == 10000
    assert rc.flat_stat_names() == mr.flat_stat_names()


def test_the_room_bootstrap_is_reproducible_from_seed_and_room_count(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    first = rc.build_report(results, fixture, n_boot=256)
    second = rc.build_report(results, fixture, n_boot=256)
    for name in rc.flat_stat_names():
        a, b = first["metrics"]["across_rooms"][name], second["metrics"]["across_rooms"][name]
        assert (a["point"], a["ci_lo"], a["ci_hi"]) == (b["point"], b["ci_lo"], b["ci_hi"])
    other = rc.build_report(results, fixture, n_boot=256, bootstrap_seed=7)
    assert other["protocol"]["bootstrap"]["seed"] == 7
    assert first["protocol"]["bootstrap"]["seed"] == rc.BOOTSTRAP_SEED
    assert first["metrics"]["across_rooms"]["_settings"]["aggregation"].startswith("room-first")


def test_the_report_publishes_the_bank_size_distribution(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    bank = report["bank"]
    assert bank["per_query_histogram"] == {"2": 1, "4": 2, "5": 1}
    assert bank["pooled"]["min"] == 2 and bank["pooled"]["max"] == 5
    assert bank["per_room"]["A/A_idx_1"]["n_queries"] == 3
    assert bank["per_room"]["A/A_idx_1"]["min"] == 4
    assert bank["per_room"]["A/A_idx_1"]["mean"] == pytest.approx(13.0 / 3.0)
    assert bank["per_room"]["B/B_idx_2"]["mean"] == pytest.approx(2.0)
    assert bank["n_missing_ir_total"] == 2               # queries 0 and 2 share S007_R002
    assert bank["rule"] == rc.REGISTERED_BANK_RULE
    assert rc.BANK_RULE_NOTE in bank["rule_note"]


def test_a_distribution_over_two_bank_rules_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    mixed = [dict(results[0], bank_rule="released_eligible_pool")] + results[1:]
    with pytest.raises(ValueError, match="more than one bank rule"):
        rc.bank_report(mixed)
    with pytest.raises(ValueError, match="at least one scored query"):
        rc.bank_report([])


def test_the_sparse_oracle_block_is_labelled_as_the_banks_own(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    oracle = report["sparse_oracle"]
    assert oracle["label"] == rc.SPARSE_ORACLE_LABEL
    assert oracle["pooled"]["median_e_oracle"] == pytest.approx(
        float(np.median([r["e_oracle_sparse"] for r in results])))
    assert sorted(oracle["per_room"]) == sorted(fixture["plan"].rooms)
    contrast = report["oracle_contrast"]
    assert contrast["n_queries"] == 4
    assert "not comparable" in contrast["note"]


def test_the_scalar_oracle_check_no_longer_claims_to_pin_the_truth(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    crosscheck = report["crosscheck"]["grid_oracle"]
    assert crosscheck["n_queries"] == 4
    assert crosscheck["max_abs_delta_m"] <= mr.ORACLE_TOLERANCE
    assert crosscheck["tolerance_m"] == mr.ORACLE_TOLERANCE
    assert "not injective" in crosscheck["note"]
    # what it establishes, and what it does not (r9c review, BLOCKER 2)
    assert crosscheck["establishes"] == rc.GRID_ORACLE_ESTABLISHES
    assert "does NOT establish" in rc.GRID_ORACLE_ESTABLISHES
    assert report["gates"]["grid_oracle_rederived_matches_the_g1_scalar"] is True
    assert report["gates"]["truth_bytes_pinned_by_pre_registered_bank_digest"] is True
    assert "pins_the_truth" not in json.dumps(report["gates"])

    # the truth-integrity claim reduces to the bank digest, and says so
    assert rc.TRUTH_INTEGRITY_NOTE == report["labels"]["truth_integrity"]
    for phrase in ("pair-metadata JSON", "sparse-bank digest", "REDUCES TO"):
        assert phrase in rc.TRUTH_INTEGRITY_NOTE


def test_an_unpinned_bank_leaves_the_truth_claim_false(tmp_path):
    fixture = build_control_fixture(tmp_path, pre_register=False)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    assert report["gates"]["truth_bytes_pinned_by_pre_registered_bank_digest"] is False
    assert report["gates"]["grid_oracle_rederived_matches_the_g1_scalar"] is True
    assert report["protocol"]["canonical"] is False
    assert any("sparse_bank" in deviation["input"]
               for deviation in report["protocol"]["deviations"])


def test_an_unregistered_setting_is_labelled_a_sensitivity_check(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    registered = rc.build_report(results, fixture, n_boot=rc.BOOTSTRAP_N)
    assert registered["protocol"]["bootstrap"]["is_registered"] is True
    assert registered["protocol"]["bank_rule_is_registered"] is True
    assert "SENSITIVITY CHECK" not in rc.render_markdown(registered)

    other = rc.build_report(results, fixture, n_boot=64, bootstrap_seed=7)
    assert other["protocol"]["bootstrap"]["is_registered"] is False
    assert other["protocol"]["bootstrap"]["registered"] == {"seed": rc.BOOTSTRAP_SEED,
                                                            "n_boot": rc.BOOTSTRAP_N}
    assert "SENSITIVITY CHECK, not the registered bootstrap" in rc.render_markdown(other)

    released = rc.build_report([dict(r, bank_rule="released_eligible_pool") for r in results],
                               fixture, n_boot=64)
    assert released["protocol"]["bank_rule_is_registered"] is False
    assert "SENSITIVITY CHECK, not the registered bank" in rc.render_markdown(released)


def test_the_census_refuses_a_partial_pass(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    with pytest.raises(ValueError, match="have no published row"):
        rc.assert_retrieval_census(results[:2], fixture["records"], totals=fixture["totals"])
    with pytest.raises(ValueError, match="registered census is"):
        rc.assert_retrieval_census(results, fixture["records"],
                                   totals=rc.retrieval_totals(rooms=2, queries=9))


# --------------------------------------------------------------------------- #
# the published artifacts
# --------------------------------------------------------------------------- #
def test_every_artifact_carries_the_control_label_and_the_caveats(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    published = rc.write_report(fixture["out_dir"], report)

    payload = json.load(open(published["paths"]["json"]))
    assert payload["control_label"] == rc.CONTROL_LABEL
    assert payload["labels"]["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    assert payload["labels"]["subset"] == mr.SUBSET_LABEL
    assert payload["labels"]["scorer_readout_deviation"] == me.SCORER_READOUT_DEVIATION
    assert payload["labels"]["sparse_oracle"] == rc.SPARSE_ORACLE_LABEL
    assert payload["labels"]["self_pair_rule"] == rc.SELF_PAIR_RULE
    assert payload["census"]["n_queries"] == 4

    markdown = open(published["paths"]["markdown"]).read()
    for phrase in (rc.CONTROL_LABEL, me.AGREE_LEAKAGE_CAVEAT, mr.SUBSET_LABEL,
                   rc.SPARSE_ORACLE_LABEL, rc.SELF_PAIR_RULE):
        assert phrase in markdown
    assert markdown.count(fixture["binding_sha256"][:16]) >= 3

    for name, path in published["paths"].items():
        assert me.file_sha256(path) == published["sha256"][name]


def test_a_report_that_cannot_name_its_run_binding_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    context = {key: value for key, value in fixture.items() if key != "binding_sha256"}
    with pytest.raises(ValueError, match="must name the run binding"):
        rc.build_report(results, context, n_boot=16)
    with pytest.raises(ValueError, match="at least one scored query"):
        rc.build_report([], fixture, n_boot=16)


def test_the_handoff_names_the_control_the_r1_report_is_waiting_for(tmp_path):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    report = rc.build_report(results, fixture, n_boot=64)
    published = rc.write_report(fixture["out_dir"], report)
    handoff = json.load(open(published["paths"]["handoff"]))

    assert handoff["control_key"] == rc.CONTROL_KEY
    assert handoff["control_key"] in mr.CONTROLS_ELSEWHERE
    assert handoff["control_label"] == rc.CONTROL_LABEL
    assert handoff["status"].startswith("run (")
    assert handoff["report_json"] == os.path.basename(published["paths"]["json"])
    assert handoff["report_sha256"] == published["sha256"]["json"]
    assert handoff["binding_sha256"] == fixture["binding_sha256"]
    assert handoff["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    for name in rc.flat_stat_names():
        assert name in handoff["headline"]
    assert handoff["bank"]["rule"] == rc.REGISTERED_BANK_RULE
    assert handoff["sparse_oracle"]["median_e_oracle"] == pytest.approx(
        report["sparse_oracle"]["across_rooms"]["median_e_oracle"]["point"])
    assert any("dense" in line for line in handoff["not_the_dense_grid"])


# --------------------------------------------------------------------------- #
# the driver: the binding gate and the CLI
# --------------------------------------------------------------------------- #
def test_the_binding_fields_partition_the_run_binding_three_ways():
    lists = (rc.RETRIEVAL_BINDING_FIELDS, rc.RETRIEVAL_BINDING_REGISTERED_ONLY,
             rc.RETRIEVAL_BINDING_NOT_CHECKED)
    assert sum(len(names) for names in lists) == len(me.RUN_BINDING_FIELDS)
    assert set().union(*(set(names) for names in lists)) == set(me.RUN_BINDING_FIELDS)
    # gated against the registered value rather than against the run's, because
    # it is inert here and a stamped tau deviation must stay expressible
    assert rc.RETRIEVAL_BINDING_REGISTERED_ONLY == ("tau",)
    for field in ("agree_ckpt_sha256", "scorer_readout", "d1_manifest_sha256",
                  "g1_report_sha256", "room_manifest_sha256", "branch",
                  "dataset_config_sha256",
                  # r9c review, BLOCKER 3: this config builds the observed-RIR
                  # loader, so its sample rate/size move obs_wav and every cosine
                  "model_config_sha256"):
        assert field in rc.RETRIEVAL_BINDING_FIELDS
    # nothing this control never uses is allowed to refuse it
    for field in ("ckpt_sha256", "steps", "cfg_scale", "noise_policy", "num_samples"):
        assert field in rc.RETRIEVAL_BINDING_NOT_CHECKED
    assert "generates nothing" in rc.BINDING_SCOPE_NOTE


def test_the_input_surface_covers_every_result_affecting_input():
    surface = rc.RESULT_AFFECTING_INPUTS
    # the run binding is a SUBSET of the surface, not the whole of it (r9c
    # review, MAJOR: a syntactically complete partition of RUN_BINDING_FIELDS
    # still omitted the inputs that live outside it)
    assert set(me.RUN_BINDING_FIELDS) < set(surface)
    for outside in ("sparse_bank_sha256", "metadata_root", "dataset_root", "tau",
                    "bank_rule", "bootstrap_seed", "n_boot", "success_radii_m", "device"):
        assert outside in surface, f"{outside} is a result-affecting input and is not named"

    for name, entry in surface.items():
        assert entry["class"] in rc.INPUT_CLASSES, name
        assert entry.get("why"), f"{name} has no reason"
        assert isinstance(entry.get("in_run_binding"), bool), name
        assert entry["in_run_binding"] == (name in me.RUN_BINDING_FIELDS), name
        if entry["class"] == "gated_against_registered":
            assert "registered" in entry, name

    # the classification and the binding lists cannot drift apart
    for field in rc.RETRIEVAL_BINDING_FIELDS:
        assert surface[field]["class"] in ("gated_against_run_binding",
                                           "gated_against_registered")
    for field in rc.RETRIEVAL_BINDING_REGISTERED_ONLY:
        assert surface[field]["class"] == "gated_against_registered"
    for field in rc.RETRIEVAL_BINDING_NOT_CHECKED:
        assert surface[field]["class"] == "stamped_not_checked"


def test_the_registered_values_are_the_ones_the_r1_report_enforces():
    # cross-pin (r9e): duplicated rather than imported into the module this round
    # so the control does not couple to r9d's file mid-flight; this assert is the
    # pin, and the two constants collapse into one in a later round
    assert rc.REGISTERED_AGREE_SHA256 == \
        "3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
    assert rc.REGISTERED_AGREE_SHA256 == mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"]
    assert rc.RESULT_AFFECTING_INPUTS["scorer_readout"]["registered"] == me.SCORER_READOUT
    assert rc.RESULT_AFFECTING_INPUTS["tau"]["registered"] == me.TAU == 0.1
    assert rc.RESULT_AFFECTING_INPUTS["bank_rule"]["registered"] == rc.REGISTERED_BANK_RULE
    assert rc.RESULT_AFFECTING_INPUTS["bootstrap_seed"]["registered"] == rc.BOOTSTRAP_SEED


def test_the_input_surface_report_carries_what_was_actually_used(tmp_path):
    fixture = build_control_fixture(tmp_path)
    report = rc.build_report(run_control(fixture), fixture, n_boot=64)
    surface = report["input_surface"]
    assert set(surface) == set(rc.RESULT_AFFECTING_INPUTS)
    assert surface["agree_ckpt_sha256"]["used"] == rc.REGISTERED_AGREE_SHA256
    assert surface["agree_ckpt_sha256"]["matches_registered"] is True
    assert surface["n_boot"]["used"] == 64
    assert surface["n_boot"]["matches_registered"] is False
    assert surface["sparse_bank_sha256"]["used"] == fixture["bank_document"]["sha256"]
    assert surface["metadata_root"]["class"] == "stamped_not_checked"
    assert surface["ckpt_sha256"]["used"] == fixture["binding"]["ckpt_sha256"]


@pytest.mark.parametrize("field, value", [
    ("agree_ckpt_sha256", "0" * 64),
    ("scorer_readout", "sample"),
    ("tau", 0.2),
    ("bank_rule", "released_eligible_pool"),
    ("bootstrap_seed", 7),
])
def test_a_deviation_from_a_registered_setting_is_refused_unless_it_is_stamped(
        tmp_path, field, value):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    if field == "bank_rule":
        results = [dict(result, bank_rule=value) for result in results]
    context = dict(fixture, allow_non_canonical=False)
    context[field] = value
    kwargs = {"n_boot": rc.BOOTSTRAP_N}
    if field == "bootstrap_seed":
        kwargs = {"n_boot": rc.BOOTSTRAP_N, "bootstrap_seed": value}
    with pytest.raises(ValueError, match="is not the registered"):
        rc.build_report(results, context, **kwargs)

    stamped = rc.build_report(results, dict(context, allow_non_canonical=True), **kwargs)
    assert stamped["protocol"]["canonical"] is False
    assert [d["input"] for d in stamped["protocol"]["deviations"]] == [field]
    assert rc.NON_CANONICAL_NOTE in rc.render_markdown(stamped)


@pytest.mark.parametrize("missing", ["bank_gate", "agree_ckpt_sha256", "scorer_readout", "tau"])
def test_a_report_that_was_not_told_what_it_used_cannot_state_canonicality(tmp_path, missing):
    fixture = build_control_fixture(tmp_path)
    results = run_control(fixture)
    context = {key: value for key, value in fixture.items() if key != missing}
    with pytest.raises(ValueError, match="cannot state canonicality"):
        rc.build_report(results, context, n_boot=64)


def test_a_fully_registered_run_is_stamped_canonical(tmp_path):
    fixture = build_control_fixture(tmp_path)
    report = rc.build_report(run_control(fixture),
                             dict(fixture, allow_non_canonical=False), n_boot=rc.BOOTSTRAP_N)
    assert report["protocol"]["canonical"] is True
    assert report["protocol"]["deviations"] == []
    assert report["protocol"]["sparse_bank_sha256"] == fixture["bank_document"]["sha256"]
    assert report["protocol"]["sparse_bank_pre_registered"] is True
    markdown = rc.render_markdown(report)
    assert rc.NON_CANONICAL_NOTE not in markdown
    assert "CANONICAL" in markdown

    published = rc.write_report(fixture["out_dir"], report)
    handoff = json.load(open(published["paths"]["handoff"]))
    assert handoff["canonical"] is True and handoff["deviations"] == []
    assert handoff["sparse_bank_sha256"] == fixture["bank_document"]["sha256"]
    assert handoff["status"] == "run (canonical)"


def test_a_non_canonical_run_says_so_in_every_artifact(tmp_path):
    fixture = build_control_fixture(tmp_path, pre_register=False)
    report = rc.build_report(run_control(fixture), fixture, n_boot=64)
    published = rc.write_report(fixture["out_dir"], report)
    payload = json.load(open(published["paths"]["json"]))
    handoff = json.load(open(published["paths"]["handoff"]))
    markdown = open(published["paths"]["markdown"]).read()

    assert payload["protocol"]["canonical"] is False
    assert payload["labels"]["non_canonical"] == rc.NON_CANONICAL_NOTE
    assert handoff["canonical"] is False
    assert handoff["status"] == "run (NON-CANONICAL)"
    assert sorted(d["input"] for d in handoff["deviations"]) == ["n_boot", "sparse_bank_sha256"]
    assert rc.NON_CANONICAL_NOTE in markdown
    assert markdown.splitlines()[2].startswith("> **NON-CANONICAL")


def test_the_control_accepts_the_runs_own_binding(tmp_path):
    fixture = build_control_fixture(tmp_path)
    gate = rc.assert_retrieval_binding(fixture["run_dir"], fixture["binding"])
    assert gate["binding_sha256"] == fixture["binding_sha256"]
    assert set(gate["checked"]) == set(rc.RETRIEVAL_BINDING_FIELDS)
    assert gate["recorded_not_checked"]["ckpt_sha256"] == fixture["binding"]["ckpt_sha256"]


@pytest.mark.parametrize("field, value", [
    ("agree_ckpt_sha256", "0" * 64),
    ("scorer_readout", "sample"),
    ("d1_manifest_sha256", "0" * 64),
    ("g1_report_sha256", "0" * 64),
    ("room_manifest_sha256", {"A/A_idx_1": "0" * 64}),
    ("branch", "full_height"),
    ("dataset_config_sha256", "0" * 64),
    # the config that builds the observed-RIR loader (r9c review, BLOCKER 3)
    ("model_config_sha256", "0" * 64),
])
def test_the_control_refuses_a_binding_that_moved(tmp_path, field, value):
    fixture = build_control_fixture(tmp_path)
    binding = dict(fixture["binding"])
    binding[field] = value
    with pytest.raises(ValueError, match=field):
        rc.assert_retrieval_binding(fixture["run_dir"], binding)


def test_a_model_config_that_is_not_the_runs_refuses_before_anything_is_scored(tmp_path):
    fixture = build_control_fixture(tmp_path)
    args = rc.parse_args(["--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                          "--context-manifest", fixture["context_manifest"],
                          "--audit-report", fixture["audit_report"],
                          "--dataset-config", fixture["context_manifest"],
                          # a real file that is NOT the config the run bound
                          "--model-config", fixture["audit_report"]])
    binding = rc.build_retrieval_binding(args, fixture["plan"],
                                         agree_sha256=fixture["binding"]["agree_ckpt_sha256"])
    assert binding["model_config_sha256"] == me.file_sha256(fixture["audit_report"])
    with pytest.raises(ValueError, match="model_config_sha256"):
        rc.assert_retrieval_binding(fixture["run_dir"], binding)


@pytest.mark.parametrize("field, value", [("ckpt_sha256", "0" * 64), ("steps", 4),
                                          ("cfg_scale", 3.0), ("noise_policy", "per_candidate")])
def test_a_generation_only_field_does_not_refuse_a_control_that_generates_nothing(
        tmp_path, field, value):
    fixture = build_control_fixture(tmp_path)
    binding = dict(fixture["binding"], **{field: value})
    assert rc.assert_retrieval_binding(fixture["run_dir"], binding)["binding_sha256"]


def test_the_control_refuses_an_edited_published_binding(tmp_path):
    fixture = build_control_fixture(tmp_path)
    path = os.path.join(fixture["run_dir"], me.BINDING_FILENAME)
    payload = json.load(open(path))
    payload["agree_ckpt_sha256"] = "0" * 64
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="does not match its own content"):
        rc.assert_retrieval_binding(fixture["run_dir"], fixture["binding"])


def test_a_binding_missing_a_checked_field_is_refused(tmp_path):
    fixture = build_control_fixture(tmp_path)
    binding = {key: value for key, value in fixture["binding"].items()
               if key != "scorer_readout"}
    with pytest.raises(ValueError, match="missing the registered field"):
        rc.assert_retrieval_binding(fixture["run_dir"], binding)


def test_the_binding_is_built_by_the_drivers_own_builder(tmp_path):
    import localize_meshgrid

    assert callable(localize_meshgrid.build_run_binding)
    args = rc.parse_args(["--run-dir", "run", "--out-dir", "out"])
    for field in ("context_manifest", "dataset_config", "k_prefixes", "num_samples", "tau",
                  "seed", "noise_policy", "steps", "cfg_scale", "cond_method", "cond_autocast",
                  "dump_cases_sha256"):
        assert hasattr(args, field), f"build_run_binding reads args.{field}"

    # ... and the control's binding is a SLICE of that builder's output, not a
    # second set of expressions for the same fields
    fixture = build_control_fixture(tmp_path)
    args = rc.parse_args(["--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                          "--context-manifest", fixture["context_manifest"],
                          "--audit-report", fixture["audit_report"],
                          "--dataset-config", fixture["context_manifest"]])
    built = rc.build_retrieval_binding(args, fixture["plan"], agree_sha256="c" * 64)
    full = localize_meshgrid.build_run_binding(
        args, fixture["plan"], ckpt_sha256=None, agree_sha256="c" * 64,
        model_config_sha256=me.file_sha256(args.model_config))
    assert set(built) == set(rc.RETRIEVAL_BINDING_FIELDS)
    assert built == {field: full[field] for field in rc.RETRIEVAL_BINDING_FIELDS}
    assert built["agree_ckpt_sha256"] == "c" * 64
    assert built["scorer_readout"] == me.SCORER_READOUT
    assert built["branch"] == fixture["plan"].branch
    assert built["model_config_sha256"] == me.file_sha256(args.model_config)


def test_the_cli_defaults_are_the_registered_settings():
    args = rc.parse_args(["--run-dir", "run", "--out-dir", "out",
                          "--expect-bank-sha256", "a" * 64])
    assert args.bank_rule == rc.REGISTERED_BANK_RULE
    assert args.bootstrap_seed == rc.BOOTSTRAP_SEED and args.n_boot == rc.BOOTSTRAP_N
    assert args.metadata_root == os.path.join("AcousticRooms", "metadata")
    assert args.dataset_root == "AcousticRooms"
    assert args.context_manifest == os.path.join("outputs_loc", "exp22",
                                                 "d1_context_manifest.json")
    assert args.non_canonical is False
    assert rc.validate_args(args) is True


def test_a_scoring_run_without_a_pre_registered_digest_is_refused_at_startup():
    args = rc.parse_args(["--run-dir", "run", "--out-dir", "out"])
    with pytest.raises(SystemExit, match="--expect-bank-sha256"):
        rc.validate_args(args)
    # ... and --non-canonical is the only way through, announced on stdout
    assert rc.validate_args(rc.parse_args(["--run-dir", "run", "--out-dir", "out",
                                           "--non-canonical"])) is True


def test_the_digest_mode_needs_no_run_and_prints_the_value_to_pre_register(
        tmp_path, capsys, monkeypatch):
    fixture = build_control_fixture(tmp_path)
    # the REAL manifest loader, with only its 6,337 -> 5,337 census relaxed for
    # the four-query fixture. The production path keeps the census (main() calls
    # mq.load_manifest with its default), so this relaxation exists nowhere but
    # here -- the same fixture seam mr.evaluate_run's require_manifest_census is
    real_loader = mq.load_manifest
    monkeypatch.setattr(mq, "load_manifest",
                        lambda path, require_census=True: real_loader(path,
                                                                     require_census=False))
    # the pre-registration happens BEFORE the P1 merge exists, so this mode may
    # not require a run directory (PLANNER RULING 2)
    assert rc.main(["--print-bank-sha256",
                    "--context-manifest", fixture["context_manifest"],
                    "--metadata-root", fixture["metadata_root"],
                    "--dataset-root", fixture["dataset_root"]]) == 0
    out = capsys.readouterr().out
    assert fixture["bank_document"]["sha256"] in out
    assert rc.BANK_DIGEST_ALGORITHM.split(":")[0] in out
    assert "--expect-bank-sha256" in out


@pytest.mark.parametrize("argv", [["--bank-rule", "nonsense"], ["--n-boot", "0"],
                                  ["--tau", "0"]])
def test_the_cli_refuses_an_unregistered_setting(argv):
    with pytest.raises(SystemExit):
        rc.validate_args(rc.parse_args(["--run-dir", "run", "--out-dir", "out"] + argv))


def test_a_control_may_not_write_into_the_run_it_reports_against(tmp_path):
    args = rc.parse_args(["--run-dir", str(tmp_path), "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="may not be the scored run directory"):
        rc.validate_args(args)


def test_the_unregistered_bank_rule_is_announced_rather_than_silently_taken(capsys):
    args = rc.parse_args(["--run-dir", "run", "--out-dir", "out", "--non-canonical",
                          "--bank-rule", "released_eligible_pool"])
    assert rc.validate_args(args) is True
    assert "not the registered" in capsys.readouterr().out


@pytest.mark.parametrize("argv, message", [
    (["--bank-rule", "released_eligible_pool"], "--bank-rule"),
    (["--tau", "0.2"], "--tau"),
    (["--bootstrap-seed", "7"], "bootstrap"),
    (["--n-boot", "128"], "bootstrap"),
])
def test_a_canonical_run_refuses_a_deviating_setting_at_startup(argv, message):
    base = ["--run-dir", "run", "--out-dir", "out", "--expect-bank-sha256", "a" * 64]
    with pytest.raises(SystemExit, match=message):
        rc.validate_args(rc.parse_args(base + argv))
    # ... and passes once the deviation is declared
    assert rc.validate_args(rc.parse_args(base + argv + ["--non-canonical"])) is True
