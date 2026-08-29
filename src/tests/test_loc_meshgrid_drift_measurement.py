"""exp_22 r9r -- the drift measurement that replaces r9p's derivation.

The measurement is the thing the bound rests on, so it is tested for the
properties a bound needs: the sample is deterministic and unsteerable, the
matched-batching replay really is the production path (it reproduces the
fixture's own scored sims exactly), the substitution arithmetic is the gate's
arithmetic, and ``derive_bound`` REFUSES rather than picking a number when the
two distributions overlap.
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_drift_measurement as dm
from src.localization import meshgrid_engine as me
from src.localization import meshgrid_offgrid_probe as op
from src.localization import meshgrid_report as mr

_spec = importlib.util.spec_from_file_location(
    "loc_meshgrid_report_fixtures",
    os.path.join(os.path.dirname(__file__), "test_loc_meshgrid_report.py"))
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

_probe_spec = importlib.util.spec_from_file_location(
    "loc_meshgrid_probe_fixtures",
    os.path.join(os.path.dirname(__file__), "test_loc_meshgrid_offgrid_probe.py"))
pfx = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(pfx)


def _fixture(tmp_path):
    fixture = pfx._probe_fixture(tmp_path)
    with open(os.path.join(fixture["run_dir"], "merge_report.json")) as handle:
        fixture["merge_report"] = json.load(handle)
    fixture["shard_by_room"] = dm.room_shard_map(fixture["merge_report"])
    # room A's shard is "produced on" the measuring device, room B's is not, so
    # both sides of the GPU axis exist in one fixture
    fixture["shard_devices"] = {"shard_A_A_idx_1": "cpu", "shard_B_B_idx_2": "cuda:9"}
    return fixture


def _measure(fixture, **kwargs):
    kwargs.setdefault("per_room", 4)
    kwargs.setdefault("matched_per_room", 1)
    return dm.run_measurement(
        fixture["engine"], list(fixture["items"]), fixture["records"], fixture["plan"],
        fixture["run_dir"], device="cpu", shard_by_room=fixture["shard_by_room"],
        shard_devices=fixture["shard_devices"], binding_sha256=fixture["binding_sha256"],
        binding=fixture["binding"], tau=fx.FIXTURE_TAU, num_samples=fx.FIXTURE_SAMPLES,
        batch_rows=8, source_chunk=2, **kwargs)


# --------------------------------------------------------------------------- #
# what the round is FOR: the module says why r9p was rejected
# --------------------------------------------------------------------------- #
def test_the_module_records_why_the_derived_bound_was_rejected():
    doc = dm.__doc__
    for phrase in ("sign-COHERENT", "SCORE_TOLERANCE", "CONDITIONER TOKENS",
                   "does not argue", "measures"):
        assert phrase in doc, phrase


def test_the_substitution_note_retires_the_dynamic_range_metric():
    assert "separation_vs_span" in dm.SUBSTITUTION_NOTE
    assert "dynamic range, not substitution evidence" in dm.SUBSTITUTION_NOTE


# --------------------------------------------------------------------------- #
# where a row was produced
# --------------------------------------------------------------------------- #
def test_the_room_to_shard_map_comes_from_the_runs_own_merge_report(tmp_path):
    fixture = _fixture(tmp_path)
    assert set(fixture["shard_by_room"]) == set(fx.FIXTURE_QUERIES)
    for room_id, shard in fixture["shard_by_room"].items():
        assert me.room_stem(room_id) in shard


def test_a_room_claimed_by_two_shards_is_refused():
    report = {"shards": [{"dir": "a", "rooms": ["R/R_idx_1"]},
                         {"dir": "b", "rooms": ["R/R_idx_1"]}]}
    with pytest.raises(ValueError, match="claimed by two shards"):
        dm.room_shard_map(report)


def test_a_merge_report_with_no_shard_rooms_is_refused():
    with pytest.raises(ValueError, match="no shard rooms"):
        dm.room_shard_map({"shards": []})


def test_the_shard_device_map_is_explicit_and_unambiguous():
    devices = dm.parse_shard_devices(["cafe=cuda:0", "rest15=cuda:1"])
    assert devices == {"cafe": "cuda:0", "rest15": "cuda:1"}
    assert dm.shard_device("outputs/i1_P1_cafe", devices) == "cuda:0"
    with pytest.raises(ValueError, match="--shard-device expects"):
        dm.parse_shard_devices(["cuda:0"])
    # a shard matching nothing, or two keys, is a refusal rather than a guess
    with pytest.raises(ValueError, match="only measurable when every shard"):
        dm.shard_device("outputs/i1_P1_other", devices)
    with pytest.raises(ValueError, match="only measurable when every shard"):
        dm.shard_device("outputs/cafe_rest15", devices)


# --------------------------------------------------------------------------- #
# the sample
# --------------------------------------------------------------------------- #
def test_the_sample_contains_every_registered_probe_query(tmp_path):
    fixture = _fixture(tmp_path)
    selected = dm.select_queries(fixture["plan"], per_room=4)
    probes = me.registered_probe_queries(fixture["plan"])
    chosen = {entry["query_id"] for entry in selected}
    assert set(probes.values()) <= chosen
    # ... and they are flagged, so the gate's own cases can be read out later
    flagged = {entry["query_id"] for entry in selected
               if entry["is_registered_probe_query"]}
    assert flagged == set(probes.values())


def test_the_sample_is_deterministic_in_the_seed(tmp_path):
    fixture = _fixture(tmp_path)
    first = [entry["query_id"] for entry in dm.select_queries(fixture["plan"], seed=7, per_room=2)]
    again = [entry["query_id"] for entry in dm.select_queries(fixture["plan"], seed=7, per_room=2)]
    assert first == again
    assert len(first) == len(set(first))


def test_a_room_with_fewer_queries_than_asked_for_gives_what_it_has(tmp_path):
    fixture = _fixture(tmp_path)
    selected = dm.select_queries(fixture["plan"], per_room=10)
    per_room = {}
    for entry in selected:
        per_room[entry["room_id"]] = per_room.get(entry["room_id"], 0) + 1
    assert per_room == {room: len(queries) for room, queries in fx.FIXTURE_QUERIES.items()}


def test_the_first_candidate_is_the_gates_own_headline_row(tmp_path):
    fixture = _fixture(tmp_path)
    query = dm.select_queries(fixture["plan"], per_room=1)[0]["query"]
    row = op.load_grid_row(fixture["run_dir"], query,
                           binding_sha256=fixture["binding_sha256"])
    rows = dm.select_candidate_rows(row, extra=1)
    headline, _index, _k = op.tie_candidate_row(row)
    assert rows[0] == headline
    assert len(rows) == len(set(rows))
    assert dm.select_candidate_rows(row, extra=1) == rows           # deterministic


def test_the_extra_candidate_is_never_the_headline_one(tmp_path):
    fixture = _fixture(tmp_path)
    for entry in dm.select_queries(fixture["plan"], per_room=4):
        row = op.load_grid_row(fixture["run_dir"], entry["query"],
                               binding_sha256=fixture["binding_sha256"])
        rows = dm.select_candidate_rows(row, extra=1)
        if int(row["n_candidates"]) > 1:
            assert len(rows) == 2 and rows[1] != rows[0]


# --------------------------------------------------------------------------- #
# one comparison
# --------------------------------------------------------------------------- #
def test_compare_slice_reports_sign_coherence_not_just_magnitude():
    stored = np.zeros(8, dtype=np.float64)
    coherent = dm.compare_slice(stored, stored + 1e-3, tau=0.1)
    assert coherent["sign_coherent"] is True
    assert coherent["n_positive"] == 8 and coherent["n_negative"] == 0
    mixed = dm.compare_slice(stored, stored + np.array([1e-3, -1e-3] * 4), tau=0.1)
    assert mixed["sign_coherent"] is False
    assert mixed["sign_coherence"] == pytest.approx(0.5)


def test_a_coherent_shift_moves_the_aggregate_by_the_whole_shift():
    """The property that killed sqrt(K): coherent error does not average down."""
    stored = np.full(8, 0.5)
    shifted = dm.compare_slice(stored, stored + 2e-3, stored_aggregate=0.5, tau=0.1)
    assert shifted["aggregate_delta"] == pytest.approx(2e-3, abs=1e-6)
    # ... while the same magnitude spread over alternating signs largely cancels
    alternating = dm.compare_slice(stored, stored + np.array([2e-3, -2e-3] * 4),
                                   stored_aggregate=0.5, tau=0.1)
    assert abs(alternating["aggregate_delta"]) < 2e-4


def test_compare_slice_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="disagree on K"):
        dm.compare_slice(np.zeros(8), np.zeros(4))


def test_the_row_aggregate_is_the_rows_own_float32_score(tmp_path):
    fixture = _fixture(tmp_path)
    query = dm.select_queries(fixture["plan"], per_room=1)[0]["query"]
    row = op.load_grid_row(fixture["run_dir"], query,
                           binding_sha256=fixture["binding_sha256"])
    from src.localization.reaggregate import decode_scores

    expected = np.asarray(decode_scores(row["by_k"]["8"]["scores_hex"]), dtype=np.float64)
    for candidate_row in (0, int(row["n_candidates"]) - 1):
        assert dm.row_aggregate(row, candidate_row) == pytest.approx(expected[candidate_row])


# --------------------------------------------------------------------------- #
# the two paths
# --------------------------------------------------------------------------- #
def test_the_matched_replay_reproduces_the_runs_own_sims_exactly(tmp_path):
    """The control: at the run's OWN batching the replay is the production path.

    The fixture's engine is deterministic on CPU, so 'the same path' means
    bit-equal here; on the real stack the same comparison is what separates
    batch-shape drift from anything else that could move a cosine.
    """
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    matched = [query for query in measured["queries"] if query["matched"]]
    assert matched, "no query was replayed at matched batching"
    for query in matched:
        block = query["matched"]
        # bit-exact in the sidecar's own dtype: rounding the replay back to
        # float16 reproduces the stored array element for element
        assert block["float16_bit_exact"] is True
        assert block["n_float16_mismatch"] == 0
        # ... and the float32 aggregate, which carries no quantization at all,
        # matches the row's published score exactly
        assert block["aggregate"]["max_abs_delta"] == 0.0
        assert block["aggregate"]["n_above_score_tolerance"] == 0
        # what remains in the per-sample comparison is the sidecar's rounding,
        # nothing else
        assert block["n_above_half_ulp"] == 0
        assert block["max_abs_delta"] <= block["sidecar_half_ulp"]


def test_the_matched_replay_uses_the_production_scoring_function(tmp_path):
    """Not a re-implementation: the engine's own ``_score_one_query`` is called."""
    fixture = _fixture(tmp_path)
    calls = []
    real = me._score_one_query

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("batch_rows"))
        return real(*args, **kwargs)

    me._score_one_query = _spy
    try:
        _measure(fixture)
    finally:
        me._score_one_query = real
    assert calls and set(calls) == {8}


def test_the_tie_path_is_the_probes_own_regeneration(tmp_path):
    """Measured through the gate's code, so the bound cannot be off-path."""
    fixture = _fixture(tmp_path)
    calls = []
    real = op.regenerate_tie_embeddings

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("source_chunk"))
        return real(*args, **kwargs)

    op.regenerate_tie_embeddings = _spy
    try:
        measured = _measure(fixture)
    finally:
        op.regenerate_tie_embeddings = real
    tie_pairs = [pair for pair in measured["pairs"] if pair["path"] == "tie"]
    assert len(calls) == len(tie_pairs)
    assert set(calls) == {1}


def test_every_measured_pair_is_labelled_with_the_gpu_that_produced_it(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    seen = {}
    for pair in measured["pairs"]:
        seen.setdefault(pair["room_id"], set()).add(pair["same_gpu"])
        assert pair["device"] == "cpu"
    assert seen["A/A_idx_1"] == {True}                      # its shard IS the device
    assert seen["B/B_idx_2"] == {False}                     # produced elsewhere


def test_the_matched_replay_is_only_run_on_the_producing_gpu(tmp_path):
    """A cross-GPU 'matched' replay would not be a matched replay."""
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    for query in measured["queries"]:
        if query["matched"] is not None:
            assert query["same_gpu"] is True


def test_a_selected_query_missing_from_the_stream_is_refused(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="never appeared in the stream"):
        dm.run_measurement(
            fixture["engine"], list(fixture["items"])[:1], fixture["records"], fixture["plan"],
            fixture["run_dir"], device="cpu", shard_by_room=fixture["shard_by_room"],
            shard_devices=fixture["shard_devices"],
            binding_sha256=fixture["binding_sha256"], binding=fixture["binding"],
            tau=fx.FIXTURE_TAU, num_samples=fx.FIXTURE_SAMPLES, batch_rows=8, source_chunk=2,
            per_room=4)


# --------------------------------------------------------------------------- #
# the substitution matrix
# --------------------------------------------------------------------------- #
def _entry(query_id, room_id, obs, gen, stored):
    return {"query_id": query_id, "room_id": room_id,
            "obs_embedding": np.asarray(obs, dtype=np.float32),
            "embeddings": np.asarray(gen, dtype=np.float32),
            "stored": np.asarray(stored, dtype=np.float64)}


def test_the_substitution_matrix_is_every_ordered_cross_pair():
    entries = [_entry("a", "R", [1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0]),
               _entry("b", "R", [0.0, 1.0], [[0.0, 1.0], [0.0, 1.0]], [1.0, 1.0]),
               _entry("c", "S", [1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0])]
    deltas = dm.substitution_deltas(entries)
    assert len(deltas) == 3 * 2
    assert not any(entry["query_id"] == entry["observation_query_id"] for entry in deltas)
    by_pair = {(entry["query_id"], entry["observation_query_id"]): entry for entry in deltas}
    # a's generations against b's orthogonal observation: cosine 0 against a
    # stored 1.0, so the gate would see a full unit of movement
    assert by_pair[("a", "b")]["max_abs_delta"] == pytest.approx(1.0)
    # ... and a's against c's identical observation moves nothing, which is the
    # honest worst case a real bank can contain
    assert by_pair[("a", "c")]["max_abs_delta"] == pytest.approx(0.0)
    assert by_pair[("a", "b")]["same_room"] is True
    assert by_pair[("a", "c")]["same_room"] is False


def test_the_substitution_matrix_refuses_repeated_query_ids():
    entry = _entry("a", "R", [1.0, 0.0], [[1.0, 0.0]], [1.0])
    with pytest.raises(ValueError, match="ids repeat"):
        dm.substitution_deltas([entry, dict(entry)])


def test_the_substitution_delta_is_the_gates_own_arithmetic(tmp_path):
    """Substituting an observation reproduces what the gate would have computed."""
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    entries = measured["substitution_entries"]
    deltas = dm.substitution_deltas(entries)
    lookup = {(entry["query_id"], entry["observation_query_id"]): entry["max_abs_delta"]
              for entry in deltas}
    victim, donor = entries[0], entries[1]
    # mirrors the implementation's dtypes exactly: the cosines are formed in
    # float32, as the scorer forms them, and only then widened
    cosines = (torch.as_tensor(donor["obs_embedding"]).float().reshape(1, -1)
               @ torch.as_tensor(victim["embeddings"]).float().T).double().numpy()
    expected = float(np.abs(cosines.reshape(-1)
                            - np.asarray(victim["stored"], dtype=np.float64)).max())
    assert lookup[(victim["query_id"], donor["query_id"])] == pytest.approx(expected)


def test_the_substitution_summary_reports_the_minimum_not_the_mean(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    summary = dm.summarize_substitution(dm.substitution_deltas(measured["substitution_entries"]))
    assert summary["overall"]["min"] <= summary["overall"]["median"]
    assert summary["worst_case_pair"]["max_abs_delta"] == summary["overall"]["min"]
    assert set(summary["per_query_min"]) == {entry["query_id"]
                                             for entry in measured["substitution_entries"]}


# --------------------------------------------------------------------------- #
# distributions and the bound
# --------------------------------------------------------------------------- #
def test_round_up_2sig_never_rounds_a_bound_down():
    assert dm.round_up_2sig(0.0029301) == pytest.approx(0.0030)
    assert dm.round_up_2sig(0.0030) == pytest.approx(0.0030)
    assert dm.round_up_2sig(1.234e-4) == pytest.approx(1.3e-4)
    assert dm.round_up_2sig(0.0) == 0.0
    for value in (1e-5, 3.7e-4, 2.93e-3, 0.0448):
        assert dm.round_up_2sig(value) >= value


def test_the_bound_sits_between_the_two_measured_distributions():
    bound = dm.derive_bound(1e-3, 0.05, safety_factor=1.5)
    assert bound["ok"] is True
    assert bound["value"] == pytest.approx(1.5e-3)
    assert bound["separation_ratio"] == pytest.approx(0.05 / 1.5e-3, rel=1e-6)


def test_overlapping_distributions_derive_NO_bound():
    """The r9r instruction: if they do not separate, STOP -- do not pick one."""
    bound = dm.derive_bound(1e-2, 2e-2, safety_factor=1.5)
    assert bound["ok"] is False
    assert "do not separate cleanly" in bound["why"]
    assert bound["separation_ratio"] < bound["min_separation_required"]


def test_a_zero_substitution_movement_derives_no_bound():
    assert dm.derive_bound(1e-3, 0.0)["ok"] is False


def test_the_honest_maximum_is_taken_over_every_measured_slice(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    summary = dm.summarize_records(measured["pairs"])
    # the default quantity is the EXCESS over the sidecar's float16 rounding,
    # because the gate adds that rounding back per query -- deriving a bound
    # from the raw delta would count the quantization twice
    assert dm.honest_max(summary) == pytest.approx(
        max(pair["excess_over_half_ulp"] for pair in measured["pairs"]))
    assert dm.honest_max(summary, key="per_candidate_max_abs_delta") == pytest.approx(
        max(pair["max_abs_delta"] for pair in measured["pairs"]))
    assert dm.honest_max(summary) <= dm.honest_max(summary, key="per_candidate_max_abs_delta")


def test_the_summary_splits_by_path_and_by_gpu(tmp_path):
    fixture = _fixture(tmp_path)
    summary = dm.summarize_records(_measure(fixture)["pairs"])
    assert set(summary) == {"tie", "matched"}
    assert "same_gpu" in summary["tie"] and "cross_gpu" in summary["tie"]
    assert summary["matched"]["all"]["n_candidates"] > 0
    for block in summary["tie"].values():
        assert block["score_tolerance"] == me.SCORE_TOLERANCE
        assert block["n_aggregate_above_score_tolerance"] >= 0


# --------------------------------------------------------------------------- #
# the artifacts
# --------------------------------------------------------------------------- #
def _report(fixture, measured):
    deltas = dm.substitution_deltas(measured["substitution_entries"])
    substitution = dm.summarize_substitution(deltas)
    substitution["pairs"] = deltas
    report = dm.build_report(measured, device="cpu", run_dir=fixture["run_dir"],
                             shard_devices=fixture["shard_devices"],
                             provenance={"binding_sha256": fixture["binding_sha256"]},
                             protocol={"seed": me.SEED}, substitution=substitution)
    report["bound"] = dm.derive_bound(
        dm.honest_max(report["summary"]), substitution["overall"]["min"])
    return report


def test_the_artifacts_publish_the_distributions_and_the_rule(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    report = _report(fixture, measured)
    published = dm.write_report(str(tmp_path / "drift"), report,
                                matched_arrays=measured["matched_arrays"])
    saved = json.load(open(published["json"]))
    assert saved["selection"]["rule"] == dm.SELECTION_RULE_NOTE
    assert saved["gpu_axis_note"] == dm.GPU_AXIS_NOTE
    assert saved["shard_devices"] == fixture["shard_devices"]
    assert len(saved["pairs"]) == len(measured["pairs"])
    markdown = open(published["markdown"]).read()
    for phrase in ("## 1. Regeneration drift", "## 2. Substitution movement", "## 3. The bound",
                   "Matched-batching replay"):
        assert phrase in markdown, phrase
    arrays = np.load(published["npz"])
    assert any(key.startswith("matched|") for key in arrays.files)


def test_the_markdown_tables_are_well_formed(tmp_path):
    fixture = _fixture(tmp_path)
    markdown = dm.render_markdown(_report(fixture, _measure(fixture))).split("\n")
    tables, current = [], []
    for line in markdown:
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    assert tables
    for table in tables:
        widths = {line.count("|") for line in table}
        assert len(widths) == 1, table


def test_the_merged_report_renders_without_a_single_device_field(tmp_path):
    """A merged report has ``devices``, not ``device`` -- the markdown must cope."""
    fixture = _fixture(tmp_path)
    first = _report(fixture, _measure(fixture))
    second = json.loads(json.dumps(first))
    second["device"] = "cuda:9"
    merged = dm.merge_device_reports([first, second])
    assert "device" not in merged and merged["devices"] == ["cpu", "cuda:9"]
    markdown = dm.render_markdown(merged)
    assert "cpu, cuda:9" in markdown
    assert f"{len(merged['pairs'])} (query, candidate) measurements" in markdown
    published = dm.write_report(str(tmp_path / "merged"), merged)
    assert os.path.isfile(published["markdown"])


def test_merging_the_devices_unions_the_distributions(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _measure(fixture)
    first = _report(fixture, measured)
    second = json.loads(json.dumps(first))
    second["device"] = "cuda:9"
    merged = dm.merge_device_reports([first, second])
    assert merged["devices"] == ["cpu", "cuda:9"]
    assert len(merged["pairs"]) == 2 * len(first["pairs"])
    assert merged["substitution"]["n_pairs"] == 2 * first["substitution"]["n_pairs"]
    # the merged bound is derived over the union, so it can only be >= either
    bound = dm.derive_bound(dm.honest_max(merged["summary"]),
                            merged["substitution"]["overall"]["min"])
    assert bound["measured_honest_max"] >= first["bound"]["measured_honest_max"]


# --------------------------------------------------------------------------- #
# CLI refusals -- no GPU, no run
# --------------------------------------------------------------------------- #
def test_the_cli_refuses_without_the_shard_device_map(tmp_path):
    args = dm.parse_args(["--out-dir", str(tmp_path / "out"), "--run-dir", str(tmp_path),
                          "--audit-report", "a", "--context-manifest", "b",
                          "--ckpt-path", "c", "--model-config", "d", "--dataset-config", "e"])
    with pytest.raises(SystemExit, match="shard -> device map"):
        dm.validate_args(args)


def test_the_cli_refuses_to_write_into_the_measured_run(tmp_path):
    args = dm.parse_args(["--out-dir", str(tmp_path), "--run-dir", str(tmp_path),
                          "--audit-report", "a", "--context-manifest", "b", "--ckpt-path", "c",
                          "--model-config", "d", "--dataset-config", "e",
                          "--shard-device", "x=cpu"])
    with pytest.raises(SystemExit, match="may not be the measured run directory"):
        dm.validate_args(args)


def test_merge_mode_needs_no_run_or_checkpoint(tmp_path):
    args = dm.parse_args(["--merge", "one.json", "--out-dir", str(tmp_path)])
    assert dm.validate_args(args) is True


# --------------------------------------------------------------------------- #
# r9u item A: the margin, measured on the path the gate actually runs
# --------------------------------------------------------------------------- #
def test_the_note_says_why_the_old_margin_did_not_transfer():
    for phrase in ("RETIRED single-candidate", "does not bound", "dot product",
                   "blocker 1"):
        assert phrase in dm.MATCHED_SUBSTITUTION_NOTE, phrase


def test_cell_half_ulp_is_elementwise_and_pins_the_scalar_helper():
    """One bound per cell -- the scalar helper is its maximum, nothing else."""
    for values in ([0.5, 0.25, 0.0, -0.5], [0.9995] * 4, [-0.5, 0.5]):
        stored = np.asarray(values, dtype=np.float16)
        cells = dm.cell_half_ulp(stored)
        assert cells.shape == stored.shape
        # the scalar helper is exactly this array's maximum -- the cross-pin
        assert float(cells.max()) == pytest.approx(mr.float16_half_ulp(stored))
        assert float(cells.min()) > 0.0
    # a small-magnitude cell gets a FAR tighter bound than the array's maximum,
    # which is the whole point of gating elementwise: r9s let the 0.0 cell hide
    # under the 0.5 cell's tolerance
    mixed = dm.cell_half_ulp(np.asarray([0.5, 0.25, 0.0, -0.5], dtype=np.float16))
    assert mixed[2] < mixed[0] / 1000
    assert mixed[1] == pytest.approx(mixed[0] / 2)
    # both neighbours, so a negative binade boundary is not halved
    assert mixed[3] == pytest.approx(mixed[0])
    assert dm.cell_half_ulp(np.asarray([-0.5], dtype=np.float16))[0] == pytest.approx(
        dm.cell_half_ulp(np.asarray([0.5], dtype=np.float16))[0])
    with pytest.raises(ValueError, match="at least one sample"):
        dm.cell_half_ulp(np.zeros(0, dtype=np.float16))


def _matched_entries(fixture):
    """One matched replay per probe query, with its embeddings cached."""
    selected = dm.attach_receiver_groups(fixture["plan"], dm.select_queries(fixture["plan"],
                                                                           per_room=1))
    by_id = {entry["query_id"]: entry for entry in selected}
    records = {record["query_id"]: record for record in fixture["records"]}
    out = []
    for obs, md in fixture["items"]:
        query_id = f"{md['idx']}|{md['relpath']}"
        if query_id not in by_id:
            continue
        entry = by_id[query_id]
        guarded = me.GuardedMetadata(md)
        me.verify_context_record(guarded, records[query_id], int(md["idx"]))
        row = op.load_grid_row(fixture["run_dir"], entry["query"],
                               binding_sha256=fixture["binding_sha256"])
        obs_embedding = torch.as_tensor(
            fixture["engine"].embedder(torch.as_tensor(obs)))[0].float()
        context = me.context_conditioning(fixture["engine"].conditioner, guarded, "cpu")
        summary, _pc, deltas, embeddings = dm.measure_matched_query_with_embeddings(
            fixture["engine"], entry["query"], guarded, context, entry["receiver_id"],
            entry["union"], entry["positions_cam"], row, row["_sims"], obs_embedding,
            num_samples=fx.FIXTURE_SAMPLES, batch_rows=8, source_chunk=2, tau=fx.FIXTURE_TAU,
            candidate_rows=())
        out.append({"query_id": query_id, "room_id": entry["room_id"],
                    "obs_embedding": obs_embedding.numpy(),
                    "embeddings": embeddings.numpy(),
                    "stored": np.asarray(row["_sims"], dtype=np.float16),
                    "cell_tolerance": dm.cell_half_ulp(row["_sims"]),
                    "summary": summary, "deltas": deltas})
    return out


def test_the_cached_embeddings_are_the_replays_own(tmp_path):
    """The cache lines up with the sidecar rows, or nothing else means anything."""
    fixture = _fixture(tmp_path)
    for entry in _matched_entries(fixture):
        stored = entry["stored"].astype(np.float64)
        recomputed = np.abs(
            (torch.as_tensor(entry["embeddings"]).float()
             @ torch.as_tensor(entry["obs_embedding"]).float().reshape(-1)).double().numpy()
            - stored)
        assert recomputed.shape == stored.shape
        assert np.abs(recomputed - entry["deltas"]).max() <= 1e-6


def test_a_cache_that_does_not_line_up_is_refused(tmp_path):
    """A stray embed call would shift every row; it must not pass silently."""
    fixture = _fixture(tmp_path)
    selected = dm.attach_receiver_groups(fixture["plan"], dm.select_queries(fixture["plan"],
                                                                           per_room=1))
    entry = selected[0]
    record = next(r for r in fixture["records"] if r["query_id"] == entry["query_id"])
    obs, md = next((obs, md) for obs, md in fixture["items"]
                   if f"{md['idx']}|{md['relpath']}" == entry["query_id"])
    guarded = me.GuardedMetadata(md)
    me.verify_context_record(guarded, record, int(md["idx"]))
    row = op.load_grid_row(fixture["run_dir"], entry["query"],
                           binding_sha256=fixture["binding_sha256"])
    obs_embedding = torch.as_tensor(
        fixture["engine"].embedder(torch.as_tensor(obs)))[0].float()
    context = me.context_conditioning(fixture["engine"].conditioner, guarded, "cpu")

    real = dm.measure_matched_query

    def _with_a_stray_embed(engine, *args, **kwargs):
        engine.embedder(torch.zeros(1, 1, 16))         # one extra call, wrong shape count
        return real(engine, *args, **kwargs)

    dm.measure_matched_query = _with_a_stray_embed
    try:
        with pytest.raises(ValueError, match="would not line up with the sidecar rows"):
            dm.measure_matched_query_with_embeddings(
                fixture["engine"], entry["query"], guarded, context, entry["receiver_id"],
                entry["union"], entry["positions_cam"], row, row["_sims"], obs_embedding,
                num_samples=fx.FIXTURE_SAMPLES, batch_rows=8, source_chunk=2,
                tau=fx.FIXTURE_TAU, candidate_rows=())
    finally:
        dm.measure_matched_query = real


def test_the_matched_substitution_matrix_is_the_gates_own_arithmetic(tmp_path):
    fixture = _fixture(tmp_path)
    entries = _matched_entries(fixture)
    deltas = dm.matched_substitution_deltas(entries)
    assert len(deltas) == len(entries) * (len(entries) - 1)
    assert not any(entry["query_id"] == entry["observation_query_id"] for entry in deltas)
    lookup = {(entry["query_id"], entry["observation_query_id"]): entry for entry in deltas}
    victim, donor = entries[0], entries[1]
    flat = torch.as_tensor(victim["embeddings"]).float()
    cosines = (flat @ torch.as_tensor(donor["obs_embedding"]).float().reshape(-1)).double()
    expected = float(np.abs(cosines.numpy()
                            - victim["stored"].astype(np.float64)).max())
    entry = lookup[(victim["query_id"], donor["query_id"])]
    assert entry["max_abs_delta"] == pytest.approx(expected)
    # ... and the movement, which carries no sidecar quantization at all
    right = (flat @ torch.as_tensor(victim["obs_embedding"]).float().reshape(-1)).double()
    assert entry["max_abs_movement"] == pytest.approx(
        float(np.abs(cosines.numpy() - right.numpy()).max()))
    assert entry["n_cells"] == victim["stored"].size


def test_the_matrix_refuses_repeated_query_ids(tmp_path):
    fixture = _fixture(tmp_path)
    entries = _matched_entries(fixture)
    with pytest.raises(ValueError, match="ids repeat"):
        dm.matched_substitution_deltas([entries[0], dict(entries[0])])


def test_the_matched_gate_refuses_a_margin_it_cannot_clear():
    """Item A's stop condition, and the elementwise detection requirement."""
    wide = {"n_pairs": 240, "n_pairs_undetected": 0, "overall": {"min": 5e-3}}
    assert dm.derive_matched_gate(wide, 2.44e-4)["ok"] is True
    assert dm.derive_matched_gate(wide, 2.44e-4)["separation_ratio"] > 20
    narrow = dm.derive_matched_gate({"n_pairs": 240, "n_pairs_undetected": 0,
                                     "overall": {"min": 5e-4}}, 2.44e-4)
    assert narrow["ok"] is False and "STOP and report" in narrow["why"]
    # a pair no elementwise cell would have caught refuses whatever the ratio
    blind = dm.derive_matched_gate({"n_pairs": 240, "n_pairs_undetected": 1,
                                    "overall": {"min": 5e-3}}, 2.44e-4)
    assert blind["ok"] is False
    assert dm.derive_matched_gate({"n_pairs": 0, "n_pairs_undetected": 0,
                                   "overall": {"min": 0.0}}, 2.44e-4)["ok"] is False


def test_the_matched_measurement_runs_end_to_end(tmp_path):
    fixture = _fixture(tmp_path)
    measured = dm.run_matched_substitution(
        fixture["engine"], list(fixture["items"]), fixture["records"], fixture["plan"],
        fixture["run_dir"], device="cpu", binding_sha256=fixture["binding_sha256"],
        binding=fixture["binding"], num_samples=fx.FIXTURE_SAMPLES, tau=fx.FIXTURE_TAU)
    probes = me.registered_probe_queries(fixture["plan"])
    assert {query["query_id"] for query in measured["queries"]} == set(probes.values())
    for query in measured["queries"]:
        # every replay is at the ROW's batching and bit-exact on this fixture
        assert query["batch_rows"] == fx.FIXTURE_ADVISORY["batch_rows"]
        assert query["source_chunk"] == fx.FIXTURE_ADVISORY["source_chunk"]
        assert query["float16_bit_exact"] is True
        assert query["n_cells_over_own_tolerance"] == 0
        assert query["honest_aggregate_max_abs_delta"] == 0.0

    deltas = dm.matched_substitution_deltas(measured["entries"])
    substitution = dm.summarize_matched_substitution(deltas)
    assert substitution["n_pairs"] == len(deltas)
    assert substitution["overall"]["min"] >= 0.0
    tolerance = max(query["cell_tolerance_max"] for query in measured["queries"])
    gate = dm.derive_matched_gate(substitution, tolerance)
    report = dm.build_matched_report(measured, device="cpu", run_dir=fixture["run_dir"],
                                     provenance={}, protocol={"seed": me.SEED},
                                     substitution=dict(substitution, pairs=deltas), gate=gate,
                                     supersedes="old.json")
    published = dm.write_matched_report(str(tmp_path / "matched"), report)
    saved = json.load(open(published["json"]))
    assert saved["path"] == "matched_batching_whole_query_replay"
    assert saved["supersedes"] == "old.json"
    markdown = open(published["markdown"]).read()
    for phrase in ("## Honest replay, per probe query", "## Substituted observations",
                   "## The gate", "cells over own tolerance"):
        assert phrase in markdown, phrase


def test_the_donor_bank_is_every_query_not_just_the_probe_set(tmp_path):
    """Without it a same-room substitution is unrepresentable (one probe/room)."""
    fixture = _fixture(tmp_path)
    measured = dm.run_matched_substitution(
        fixture["engine"], list(fixture["items"]), fixture["records"], fixture["plan"],
        fixture["run_dir"], device="cpu", binding_sha256=fixture["binding_sha256"],
        binding=fixture["binding"], num_samples=fx.FIXTURE_SAMPLES, tau=fx.FIXTURE_TAU)
    assert measured["n_donors"] == len(fixture["items"])
    assert measured["n_donors"] > len(measured["queries"])
    deltas = dm.matched_substitution_deltas(measured["entries"])
    # every probe query against every OTHER query, its own excluded by id
    assert len(deltas) == len(measured["queries"]) * (measured["n_donors"] - 1)
    assert not any(entry["query_id"] == entry["observation_query_id"] for entry in deltas)
    # the same-room donors the probe set alone could never supply
    same_room = [entry for entry in deltas if entry["same_room"]]
    assert same_room, "no same-room donor reached the matrix"
    assert any(entry["same_receiver"] for entry in deltas)
    summary = dm.summarize_matched_substitution(deltas)
    assert summary["same_receiver"]["n"] >= 1
    assert summary["same_room"]["n"] >= summary["same_receiver"]["n"]


def test_a_probe_query_missing_from_the_stream_is_refused(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="never appeared in the stream"):
        dm.run_matched_substitution(
            fixture["engine"], list(fixture["items"])[:1], fixture["records"], fixture["plan"],
            fixture["run_dir"], device="cpu", binding_sha256=fixture["binding_sha256"],
            binding=fixture["binding"], num_samples=fx.FIXTURE_SAMPLES, tau=fx.FIXTURE_TAU)


def test_the_retired_label_does_not_rewrite_the_artifact_it_labels(tmp_path):
    original = str(tmp_path / "drift_measurement.json")
    me.write_json(original, {"experiment": "r9r", "substitution": {"overall": {"min": 1.0}}})
    before = me.file_sha256(original)
    path = dm.write_retired_path_label(str(tmp_path), [original], "matched.json")
    assert me.file_sha256(original) == before          # byte-identical, still evidence
    payload = json.load(open(path))
    assert payload["path"] == "retired_changed_batching_single_candidate"
    assert payload["superseded_by"] == "matched.json"
    assert "never bounded the matched gate" in payload["what_does_not"]
    assert "bit-exactness control" in payload["what_still_stands"]
