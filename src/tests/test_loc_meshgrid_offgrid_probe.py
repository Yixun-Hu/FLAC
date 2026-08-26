"""exp_22 R1 controls -- the off-grid truth probe and the AGREE calibration (§2).

The probe is the one place in exp_22 that is ALLOWED to read the held-out target,
so most of these tests are containment tests: the truth may place a generation
whose score is reported, and it may not place a candidate, win an argmax or reach
a published metric. The rest are binding tests -- a control generated under a
different checkpoint, scorer, context draw or candidate manifest is not
comparable to the run it is ranked against, and is refused.

The generation stack is the same synthetic one the report fixture drives through
``run_pass``, so the whole control runs on CPU against a real, digest-verified
run directory.
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_offgrid_probe as op
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization.reaggregate import decode_scores

# The fixture builders live beside the report they were written for; loading the
# module by path (the pattern src/tests/test_loc_meshgrid_engine.py already uses
# for the G1 audit tool) reuses ONE definition of the fixture run rather than
# keeping a second, drifting copy here.
_SPEC = importlib.util.spec_from_file_location(
    "_meshgrid_report_fixtures",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "test_loc_meshgrid_report.py"))
fx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fx)


def _probe_fixture(tmp_path):
    fixture = fx.build_fixture_run(tmp_path)
    fixture["engine"] = fx.SyntheticEngine()
    fixture["out_dir"] = str(tmp_path / "probe")
    return fixture


def _run(fixture, **kwargs):
    return op.run_probe(fixture["engine"], list(fixture["items"]), fixture["records"],
                        fixture["plan"], fixture["run_dir"], fixture["out_dir"],
                        metadata_root=fixture["metadata_root"],
                        binding_sha256=fixture["binding_sha256"],
                        tau=fx.FIXTURE_TAU, num_samples=fx.FIXTURE_SAMPLES,
                        prefixes=fx.FIXTURE_PREFIXES, **kwargs)


# --------------------------------------------------------------------------- #
# containment: what the control is allowed to do with the truth
# --------------------------------------------------------------------------- #
def test_the_control_label_states_the_containment_it_promises():
    for phrase in ("READS THE HELD-OUT TARGET", "NEVER inserted into any candidate set",
                   "never competes in any argmax", "never becomes a prediction",
                   "never enters any published"):
        assert phrase in op.CONTROL_LABEL


def test_the_calibration_label_says_which_real_bank_it_uses():
    assert "context RIRs" in op.CALIBRATION_LABEL
    assert "neither is a localization metric" in op.CALIBRATION_LABEL


def test_the_probe_binds_everything_the_run_binds_except_the_dump_authority():
    assert set(op.PROBE_BINDING_FIELDS) == set(me.RUN_BINDING_FIELDS) - {"dump_cases_sha256"}
    for field in ("ckpt_sha256", "agree_ckpt_sha256", "d1_manifest_sha256",
                  "g1_report_sha256", "room_manifest_sha256", "model_config_sha256",
                  "dataset_config_sha256", "branch", "tau", "seed", "num_samples",
                  "noise_policy", "steps", "cfg_scale", "cond_method", "scorer_readout",
                  "cond_autocast"):
        assert field in op.PROBE_BINDING_FIELDS


# --------------------------------------------------------------------------- #
# the binding gate
# --------------------------------------------------------------------------- #
def test_the_probe_accepts_the_runs_own_binding(tmp_path):
    fixture = _probe_fixture(tmp_path)
    gate = op.assert_probe_binding(fixture["run_dir"], fixture["binding"])
    assert gate["binding_sha256"] == fixture["binding_sha256"]
    assert "dump_cases_sha256" not in gate["fields_checked"]


@pytest.mark.parametrize("field, value", [
    ("ckpt_sha256", "0" * 64),
    ("agree_ckpt_sha256", "0" * 64),
    ("d1_manifest_sha256", "0" * 64),
    ("g1_report_sha256", "0" * 64),
    ("room_manifest_sha256", {"A/A_idx_1": "0" * 64}),
    ("tau", 0.2),
    ("seed", 7),
    ("cond_autocast", "off"),
])
def test_the_probe_refuses_a_binding_that_moved(tmp_path, field, value):
    fixture = _probe_fixture(tmp_path)
    binding = dict(fixture["binding"])
    binding[field] = value
    with pytest.raises(ValueError, match=field):
        op.assert_probe_binding(fixture["run_dir"], binding)


def test_a_different_dump_case_list_does_not_refuse_the_probe(tmp_path):
    fixture = _probe_fixture(tmp_path)
    binding = dict(fixture["binding"], dump_cases_sha256="a" * 64)
    assert op.assert_probe_binding(fixture["run_dir"], binding)["binding_sha256"]


def test_the_probe_refuses_an_edited_published_binding(tmp_path):
    fixture = _probe_fixture(tmp_path)
    path = os.path.join(fixture["run_dir"], me.BINDING_FILENAME)
    payload = json.load(open(path))
    payload["seed"] = 7
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="does not match its own content"):
        op.assert_probe_binding(fixture["run_dir"], fixture["binding"])


def test_a_probe_binding_missing_a_registered_field_is_refused(tmp_path):
    fixture = _probe_fixture(tmp_path)
    binding = {key: value for key, value in fixture["binding"].items()
               if key != "scorer_readout"}
    with pytest.raises(ValueError, match="missing the registered field"):
        op.assert_probe_binding(fixture["run_dir"], binding)


# --------------------------------------------------------------------------- #
# the noise: the truth shares the grid candidates' draws
# --------------------------------------------------------------------------- #
def test_only_common_random_numbers_can_key_an_off_grid_draw():
    assert op.assert_offgrid_noise_policy(me.REGISTERED_NOISE_POLICY) is True
    with pytest.raises(ValueError, match="has no value for a point that is not a candidate"):
        op.assert_offgrid_noise_policy("per_candidate")
    with pytest.raises(ValueError, match="needs the registered noise policy"):
        op.truth_noise(42, "q|a.wav", 4, (2, 4), policy="per_candidate")


def test_the_truth_generation_is_drawn_from_the_grid_candidates_latents():
    truth = op.truth_noise(42, "q|a.wav", 8, (2, 4))
    for candidate in (0, 17, 5294):
        grid = me.noise_block(42, "q|a.wav", [candidate], 8, (2, 4))
        assert torch.equal(truth, grid)
    # ... and a different query, or a different seed, is a different draw
    assert not torch.equal(truth, op.truth_noise(42, "q|b.wav", 8, (2, 4)))
    assert not torch.equal(truth, op.truth_noise(43, "q|a.wav", 8, (2, 4)))
    assert tuple(truth.shape) == (8, 2, 4)


# --------------------------------------------------------------------------- #
# the probe set
# --------------------------------------------------------------------------- #
def test_the_probe_set_is_one_lexicographically_first_query_per_room(tmp_path):
    fixture = _probe_fixture(tmp_path)
    probes = me.registered_probe_queries(fixture["plan"])
    assert op.assert_registered_probe_set(probes, fixture["plan"]) == \
        [probes[room] for room in sorted(fixture["plan"].rooms)]
    # A/A_idx_1's three queries are S001, S003, S005 -> the S001 relpath wins
    assert probes["A/A_idx_1"].endswith("S001_R002_hybrid_IR.wav")


def test_a_probe_set_that_misses_a_room_is_refused(tmp_path):
    fixture = _probe_fixture(tmp_path)
    probes = me.registered_probe_queries(fixture["plan"])
    probes.pop("B/B_idx_2")
    with pytest.raises(ValueError, match="does not cover the audited rooms"):
        op.assert_registered_probe_set(probes, fixture["plan"])


def test_a_probe_set_that_names_one_query_twice_is_refused(tmp_path):
    fixture = _probe_fixture(tmp_path)
    probes = me.registered_probe_queries(fixture["plan"])
    probes["B/B_idx_2"] = probes["A/A_idx_1"]
    with pytest.raises(ValueError, match="same query for two rooms"):
        op.assert_registered_probe_set(probes, fixture["plan"])


# --------------------------------------------------------------------------- #
# the rank against the grid
# --------------------------------------------------------------------------- #
def _grid_row(sims, tau=0.1, prefixes=(1, 2)):
    coordinates = [[float(i), 0.0, 0.0] for i in range(len(sims))]
    scored = me.score_query(torch.as_tensor(sims, dtype=torch.float32),
                            list(range(len(sims))), np.asarray(coordinates), tau=tau,
                            prefixes=prefixes)
    return {"by_k": {str(k): block for k, block in scored["by_k"].items()}}


def test_the_rank_counts_the_candidates_that_beat_the_truth():
    row = _grid_row([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    grid = decode_scores(row["by_k"]["2"]["scores_hex"]).tolist()
    ranked = op.rank_against_grid(row, {2: {"lme": grid[1] + 1e-6, "mean": 0.0}})[2]
    assert ranked["rank"] == 2                      # only candidate 2 is better
    assert ranked["n_grid_better"] == 1
    assert ranked["n_grid_tied"] == 0
    assert ranked["n_candidates"] == 3
    assert ranked["best_grid_score"] == pytest.approx(grid[2])
    assert ranked["truth_minus_best_grid"] < 0.0


def test_a_truth_that_beats_every_candidate_ranks_first_and_ties_are_reported():
    row = _grid_row([[0.1, 0.1], [0.5, 0.5]])
    grid = decode_scores(row["by_k"]["2"]["scores_hex"]).tolist()
    best = op.rank_against_grid(row, {2: {"lme": max(grid) + 1.0, "mean": 0.0}})[2]
    assert best["rank"] == 1 and best["n_grid_better"] == 0
    assert best["percentile"] == pytest.approx(1.0)
    tied = op.rank_against_grid(row, {2: {"lme": grid[1], "mean": 0.0}})[2]
    assert tied["rank"] == 1 and tied["n_grid_tied"] == 1
    assert tied["truth_minus_best_grid"] == pytest.approx(0.0)


def test_the_rank_can_be_taken_against_the_s_mean_diagnostic_too():
    row = _grid_row([[0.1, 0.9], [0.5, 0.5]])
    means = decode_scores(row["by_k"]["2"]["mean_scores_hex"]).tolist()
    ranked = op.rank_against_grid(row, {2: {"lme": 0.0, "mean": means[0] + 1e-6}},
                                  aggregator="mean")[2]
    assert ranked["n_grid_better"] == 0
    assert ranked["grid_prediction_index"] == row["by_k"]["2"]["mean_prediction_index"]


# --------------------------------------------------------------------------- #
# the calibration
# --------------------------------------------------------------------------- #
def test_the_calibration_reports_both_distributions():
    engine = fx.SyntheticEngine()
    real = torch.arange(4 * 16, dtype=torch.float32).reshape(4, 1, 16) * 0.01
    generated = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 1, 8) * 0.02
    obs = engine.embedder(torch.full((1, 1, 16), 0.5))[0]
    record = op.calibration_record(engine.embedder, obs, real, generated)
    assert record["real_summary"]["n"] == 4
    assert record["generated_summary"]["n"] == 8
    assert len(record["real"]) == 4 and len(record["generated"]) == 8
    assert all(-1.0001 <= v <= 1.0001 for v in record["real"] + record["generated"])
    assert record["gap_mean_real_minus_generated"] == pytest.approx(
        record["real_summary"]["mean"] - record["generated_summary"]["mean"])
    assert op.CALIBRATION_LABEL == record["label"]


def test_the_calibration_refuses_a_real_bank_of_the_wrong_shape():
    engine = fx.SyntheticEngine()
    obs = engine.embedder(torch.full((1, 1, 16), 0.5))[0]
    with pytest.raises(ValueError, match=r"must be \[N, 1, T\]"):
        op.calibration_record(engine.embedder, obs, torch.zeros(4, 3, 16),
                              torch.zeros(8, 1, 8))


# --------------------------------------------------------------------------- #
# the whole control, end to end on the fixture run
# --------------------------------------------------------------------------- #
def test_the_probe_runs_both_controls_on_one_query_per_room(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert [record["room_id"] for record in records] == ["A/A_idx_1", "B/B_idx_2"]
    for record in records:
        assert record["control_label"] == op.CONTROL_LABEL
        assert sorted(int(k) for k in record["rank_lme"]) == list(fx.FIXTURE_PREFIXES)
        assert sorted(int(k) for k in record["rank_mean"]) == list(fx.FIXTURE_PREFIXES)
        for k in fx.FIXTURE_PREFIXES:
            block = record["rank_lme"][str(k)]
            assert 1 <= block["rank"] <= block["n_candidates"] + 1
            assert block["n_candidates"] == record["n_candidates"]
        assert len(record["truth_sims"]) == fx.FIXTURE_SAMPLES
        assert record["e_oracle"] > 0.0
        assert record["truth_is_a_candidate"] is False
        assert len(record["calibration"]["real"]) == 8
        assert len(record["calibration"]["generated"]) == fx.FIXTURE_SAMPLES


def test_the_probe_stages_its_waveforms_and_publishes_them_with_the_manifest(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    # nothing is a finalized artifact yet: the manifest does not exist
    for record in records:
        assert record["waveform_published"] is False
        assert not os.path.isfile(os.path.join(fixture["out_dir"],
                                               record["waveform_path"]))
        staged = os.path.join(fixture["out_dir"], record["waveform_staged_path"])
        assert os.path.isfile(staged)
        assert op.WAVEFORM_STAGING_DIRNAME in record["waveform_staged_path"]

    op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                          fixture["binding_sha256"], provenance={},
                          tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    for record in records:
        path = os.path.join(fixture["out_dir"], record["waveform_path"])
        assert record["waveform_published"] is True
        assert os.path.isfile(path)
        assert me.file_sha256(path) == record["waveform_sha256"]
        with np.load(path) as data:
            assert data["waveforms"].shape[0] == fx.FIXTURE_SAMPLES
            assert data["context_audio"].shape[0] == 8
            assert data["truth_xyz"].tolist() == record["truth_xyz"]
            assert data["receiver_xyz"].tolist() == record["receiver_xyz"]
            # a dump read on its own still says what it is (r9 finding 9)
            assert str(data["control_label"]) == op.CONTROL_LABEL
            assert str(data["subset"]) == mr.SUBSET_LABEL
            assert str(data["agree_leakage_caveat"]) == me.AGREE_LEAKAGE_CAVEAT
            assert str(data["query_id"]) == record["query_id"]
        assert "announcement 08" in record["waveform_note"]
    assert not os.path.isdir(os.path.join(fixture["out_dir"],
                                          op.WAVEFORM_STAGING_DIRNAME))


def test_a_partial_control_leaves_quarantined_dumps_never_unmanifested_finals(tmp_path):
    fixture = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="the stream ended before"):
        op.run_probe(fixture["engine"], list(fixture["items"])[:1], fixture["records"],
                     fixture["plan"], fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)
    staging = os.path.join(fixture["out_dir"], op.WAVEFORM_STAGING_DIRNAME)
    published = os.path.join(fixture["out_dir"], op.WAVEFORM_DIRNAME)
    assert sorted(os.listdir(staging)) == ["offgrid_A_A_idx_1_q00000.npz"]
    # ... and nothing was finalized beside them
    assert [name for name in os.listdir(published) if name.endswith(".npz")] == []
    assert not os.path.isfile(os.path.join(fixture["out_dir"], op.PROBE_REPORT_JSON))


def test_the_probe_generation_is_deterministic(tmp_path):
    fixture = _probe_fixture(tmp_path)
    first = _run(fixture)
    second = _run(fixture)
    assert [r["truth_sims"] for r in first] == [r["truth_sims"] for r in second]
    assert [r["waveform_sha256"] for r in first] == [r["waveform_sha256"] for r in second]


def test_the_probe_refuses_a_grid_row_from_another_binding(tmp_path):
    fixture = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot be used as the probe's grid reference"):
        op.run_probe(fixture["engine"], list(fixture["items"]), fixture["records"],
                     fixture["plan"], fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"], binding_sha256="0" * 64,
                     tau=fx.FIXTURE_TAU, num_samples=fx.FIXTURE_SAMPLES,
                     prefixes=fx.FIXTURE_PREFIXES)


def test_the_probe_refuses_a_truncated_stream(tmp_path):
    fixture = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="the stream ended before"):
        op.run_probe(fixture["engine"], list(fixture["items"])[:1], fixture["records"],
                     fixture["plan"], fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)


def test_the_probe_refuses_a_truth_whose_oracle_is_not_the_manifests(tmp_path):
    fixture = _probe_fixture(tmp_path)
    scene, scene_id = "A/A_idx_1".split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = [1.11, 1.11, 0.61]
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="not the one the audit measured"):
        _run(fixture)


def test_the_probe_refuses_a_non_registered_noise_policy(tmp_path):
    fixture = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="needs the registered noise policy"):
        _run(fixture, noise_policy="per_candidate")


# --------------------------------------------------------------------------- #
# the published artifacts
# --------------------------------------------------------------------------- #
def test_the_probe_report_is_stamped_with_every_caveat(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                                      fixture["binding_sha256"],
                                      provenance={"run_dir": fixture["run_dir"]},
                                      tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    payload = json.load(open(published["json"]))
    assert payload["control_label"] == op.CONTROL_LABEL
    assert payload["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    assert payload["subset"] == mr.SUBSET_LABEL
    assert payload["scorer_readout_deviation"] == me.SCORER_READOUT_DEVIATION
    assert payload["binding_sha256"] == fixture["binding_sha256"]
    assert "dump_cases_sha256" not in payload["binding"]
    assert payload["n_queries"] == 2

    summary = payload["summary"]
    assert sorted(int(k) for k in summary["by_k"]) == list(fx.FIXTURE_PREFIXES)
    assert summary["calibration"]["real"]["n"] == 16          # 2 queries x 8 contexts
    assert summary["calibration"]["generated"]["n"] == 2 * fx.FIXTURE_SAMPLES

    markdown = open(published["markdown"]).read()
    assert op.CONTROL_LABEL in markdown
    assert me.AGREE_LEAKAGE_CAVEAT in markdown
    assert mr.SUBSET_LABEL in markdown


def test_the_probe_summary_refuses_an_empty_record_set():
    with pytest.raises(ValueError, match="at least one record"):
        op.summarize_probe([])


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_defaults_are_the_registered_protocol():
    args = op.parse_args(["--ckpt-path", "c.ckpt", "--run-dir", "run", "--out-dir", "out",
                          "--non-canonical"])
    assert args.seed == me.SEED and args.tau == me.TAU
    assert args.num_samples == me.NUM_SAMPLES
    assert args.k_prefixes == list(me.K_PREFIXES)
    assert args.noise_policy == me.REGISTERED_NOISE_POLICY
    assert args.steps == me.STEPS and args.cfg_scale == me.CFG_SCALE
    assert args.cond_method == "vanilla"
    assert op.validate_args(args) is True


@pytest.mark.parametrize("argv", [
    ["--noise-policy", "per_candidate"],
    ["--cond-method", "fa_invariant"],
    ["--num-samples", "4"],
    ["--tau", "0"],
    ["--k-prefixes", "1", "1", "8"],
])
def test_the_cli_refuses_an_unregistered_protocol(argv):
    args = op.parse_args(["--ckpt-path", "c.ckpt", "--run-dir", "run", "--out-dir", "out"]
                         + argv)
    with pytest.raises(SystemExit):
        op.validate_args(args)


def test_a_control_may_not_write_into_the_run_it_reports_against(tmp_path):
    args = op.parse_args(["--ckpt-path", "c.ckpt", "--run-dir", str(tmp_path),
                          "--out-dir", str(tmp_path), "--non-canonical"])
    with pytest.raises(SystemExit, match="may not be the scored run directory"):
        op.validate_args(args)


def test_the_probe_reuses_the_drivers_binding_and_item_unpacker():
    import localize_meshgrid

    # ONE binding builder and ONE stream unpacker across the driver and this
    # control: a second copy could drift from the registered field list
    assert callable(localize_meshgrid.build_run_binding)
    assert callable(localize_meshgrid._iter_items)
    args = op.parse_args(["--ckpt-path", "c.ckpt", "--run-dir", "r", "--out-dir", "o"])
    for field in ("context_manifest", "k_prefixes", "num_samples", "tau", "seed",
                  "noise_policy", "steps", "cfg_scale", "cond_method", "cond_autocast",
                  "dataset_config", "dump_cases_sha256"):
        assert hasattr(args, field), f"build_run_binding reads args.{field}"


# --------------------------------------------------------------------------- #
# r9c: the blockers the Codex r9 review found in this control
# --------------------------------------------------------------------------- #
def _census_args(fixture):
    """The two D1 inputs the run census is taken against."""
    return fixture["records"], fixture["context_manifest"]


def _published_binding(run_dir):
    """The binding as PUBLISHED -- advisory and declared_rooms included."""
    return mr.load_published_binding(run_dir)[0]


def test_the_probe_requires_the_complete_merged_run_it_ranks_against(tmp_path):
    """B4: every shard shares the strict binding digest, so it proved nothing."""
    fixture = _probe_fixture(tmp_path)
    verdict = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    assert verdict["census"]["n_queries"] == fixture["totals"]["queries"]
    assert verdict["identity_join"]["n_queries"] == fixture["totals"]["queries"]
    assert verdict["merge"]["declared_rooms"] == sorted(fx.FIXTURE_QUERIES)

    # one shard of that same run carries the SAME binding and is refused
    shard = fixture["shards"][0]
    shard_binding = _published_binding(shard)
    with pytest.raises(ValueError, match="publishes no merge_report.json"):
        op.assert_probe_run_census(shard, shard_binding, fixture["binding_sha256"],
                                   fixture["plan"], *_census_args(fixture),
                                   totals=fixture["totals"])
    with pytest.raises(ValueError, match="have no published row"):
        op.assert_probe_run_census(shard, shard_binding, fixture["binding_sha256"],
                                   fixture["plan"], *_census_args(fixture),
                                   totals=fixture["totals"], single_shard=True)


def test_the_probe_binding_alone_cannot_tell_a_shard_from_the_merge(tmp_path):
    fixture = _probe_fixture(tmp_path)
    # the r9 gate: it passes on a single shard, which is the whole finding
    assert op.assert_probe_binding(fixture["shards"][0],
                                   fixture["binding"])["binding_sha256"] == \
        fixture["binding_sha256"]


def test_the_probe_refuses_a_swapped_audit_or_manifest(tmp_path):
    fixture = _probe_fixture(tmp_path)
    other = fx._fixture_audit(tmp_path / "other", stamp="a second valid audit")
    other_plan = me.load_audit_plan(other)
    with pytest.raises(ValueError, match="g1_report_sha256"):
        op.assert_probe_run_census(fixture["run_dir"],
                                   _published_binding(fixture["run_dir"]),
                                   fixture["binding_sha256"], other_plan,
                                   *_census_args(fixture), totals=fixture["totals"])


def test_a_foreign_row_at_the_expected_path_cannot_supply_the_grid_scores(tmp_path):
    """B4: the row was digest-checked but never joined to THIS query."""
    import shutil

    fixture = _probe_fixture(tmp_path)
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    probe_query = next(q for q in room.queries if q.position == 0)
    other_query = next(q for q in room.queries if q.position == 2)
    target = me.query_artifact_paths(fixture["run_dir"], probe_query.room_id,
                                     probe_query.position)["row"]
    source = me.query_artifact_paths(fixture["run_dir"], other_query.room_id,
                                     other_query.position)["row"]
    shutil.copyfile(source, target)

    # the row is intact and its digests all verify -- it is simply a different query
    assert me.verify_query_artifact(target,
                                    binding_sha256=fixture["binding_sha256"])["ok"]
    with pytest.raises(ValueError, match="does not match the candidate manifest"):
        op.load_grid_row(fixture["run_dir"], probe_query,
                         binding_sha256=fixture["binding_sha256"])


def test_the_grid_row_protocol_is_checked_against_the_binding(tmp_path):
    fixture = _probe_fixture(tmp_path)
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    query = next(q for q in room.queries if q.position == 0)
    assert op.load_grid_row(fixture["run_dir"], query,
                            binding_sha256=fixture["binding_sha256"],
                            binding=fixture["binding"])["query_id"] == query.query_id
    with pytest.raises(ValueError, match="tau"):
        op.load_grid_row(fixture["run_dir"], query,
                         binding_sha256=fixture["binding_sha256"],
                         binding=dict(fixture["binding"], tau=0.25))


def test_the_binding_gate_runs_before_anything_reaches_a_device(tmp_path, monkeypatch):
    """B5: AGREE used to be loaded onto --device to build the binding."""
    import src.localization.agree_embed as agree_embed

    fixture = _probe_fixture(tmp_path)
    order = []

    def _gate(args, model_config, agree_path):
        order.append("gate")
        raise ValueError("the gate refused this run")

    def _load(*_args, **_kwargs):
        order.append("agree")
        raise AssertionError("the scorer reached a device before the gate ran")

    monkeypatch.setattr(op, "gate_run", _gate)
    monkeypatch.setattr(agree_embed, "load_agree_audio", _load)
    with pytest.raises(ValueError, match="the gate refused this run"):
        op.main(["--ckpt-path", fixture["context_manifest"],
                 "--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                 "--agree-ckpt", fixture["context_manifest"],
                 "--audit-report", fixture["audit_report"],
                 "--context-manifest", fixture["context_manifest"],
                 "--metadata-root", fixture["metadata_root"],
                 "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"]])
    assert order == ["gate"]


def test_gate_run_touches_no_device_and_returns_the_whole_ladder(tmp_path, monkeypatch):
    fixture = _probe_fixture(tmp_path)
    args = op.parse_args(["--ckpt-path", fixture["context_manifest"],
                          "--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                          "--audit-report", fixture["audit_report"],
                          "--context-manifest", fixture["context_manifest"],
                          "--metadata-root", fixture["metadata_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
                          "--device", "cuda:0"])
    monkeypatch.setattr(op.torch, "load", lambda *a, **k: {"model_config": {}})
    monkeypatch.setattr(mq, "load_manifest",
                        lambda path, **k: json.load(open(path)))
    monkeypatch.setattr(op.me, "file_sha256", me.file_sha256)

    captured = {}

    def _binding(args_, plan, **kwargs):
        captured["kwargs"] = kwargs
        return dict(fixture["binding"])

    import localize_meshgrid as driver
    monkeypatch.setattr(driver, "build_run_binding", _binding)
    monkeypatch.setattr(op, "_load_and_validate_checkpoint", lambda *a, **k: {})
    plan, manifest, binding, gate, _ckpt = op.gate_run(args, {},
                                                       fixture["context_manifest"],
                                                       totals=fixture["totals"],
                                                       require_manifest_census=False)
    assert sorted(plan.rooms) == sorted(fx.FIXTURE_QUERIES)
    assert gate["census"]["n_queries"] == fixture["totals"]["queries"]
    assert gate["identity_join"]["n_queries"] == fixture["totals"]["queries"]
    assert gate["registered_protocol"]["is_registered"] is True
    # the AGREE identity came from a FILE digest, never from a loaded model
    assert captured["kwargs"]["agree_sha256"] == me.file_sha256(fixture["context_manifest"])


def test_the_probe_checks_the_truth_as_a_vector_against_the_loader(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert all(record["truth_vector_drift_m"] == pytest.approx(0.0, abs=1e-6)
               for record in records)

    mirrored = fx._mirrored_truth("A/A_idx_1")
    scene, scene_id = "A/A_idx_1".split("/")
    path = os.path.join(fixture["metadata_root"], scene, scene_id, "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = mirrored
    with open(path, "w") as handle:
        json.dump(payload, handle)
    # the scalar oracle is blind to it; the vector check is not
    with pytest.raises(ValueError, match="is not the one the query was held out from"):
        _run(fixture)


def test_a_rank_one_tie_is_not_counted_as_beating_every_candidate(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    largest = str(max(fx.FIXTURE_PREFIXES))
    # force one query into a tie with the best grid candidate and one strictly above
    records[0]["rank_lme"][largest].update(rank=1, n_grid_better=0, n_grid_tied=1)
    records[1]["rank_lme"][largest].update(rank=1, n_grid_better=0, n_grid_tied=0)
    block = op.summarize_probe(records, prefixes=fx.FIXTURE_PREFIXES)["by_k"][largest]
    assert block["n_rank_one"] == 2
    assert block["n_truth_beats_every_candidate"] == 1
    assert block["n_truth_ties_the_best"] == 1
    assert "strictly better" in block["rank_one_note"]


def test_the_probe_report_carries_the_run_gate_and_every_disclosure(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    gate = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={"run_dir": fixture["run_dir"]}, tau=fx.FIXTURE_TAU,
        prefixes=fx.FIXTURE_PREFIXES,
        gate={key: gate[key] for key in ("census", "identity_join", "merge",
                                         "single_shard", "single_shard_note",
                                         "registered_protocol")})
    payload = json.load(open(published["json"]))
    assert payload["latency_scope_note"] == mr.LATENCY_SCOPE_NOTE
    assert payload["truth_binding_note"] == mr.TRUTH_BINDING_NOTE
    assert payload["controls_elsewhere"] == mr.CONTROLS_ELSEWHERE
    assert payload["run_gate"]["census"]["n_queries"] == fixture["totals"]["queries"]
    assert payload["single_shard"] is False
    markdown = open(published["markdown"]).read()
    assert "strictly beats every candidate" in markdown
    assert "ties the best" in markdown
    assert mr.TRUTH_BINDING_NOTE in markdown


# --------------------------------------------------------------------------- #
# r9d: the residuals the consolidated Codex r9c re-review left open
# --------------------------------------------------------------------------- #
def test_a_failed_gate_never_imports_eval_flac(tmp_path):
    """B5: eval_FLAC's function defaults call torch.cuda.is_available() at import.

    Run in a subprocess, because by the time this file executes the module may
    already be resident from another test -- and the claim is about what a
    refused run touches, which only a fresh interpreter can settle.
    """
    import subprocess
    import sys
    import textwrap

    fixture = _probe_fixture(tmp_path)
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {os.getcwd()!r})
        from src.localization import meshgrid_offgrid_probe as op
        assert "eval_FLAC" not in sys.modules, "importing the control pulled eval_FLAC"
        args = op.parse_args([
            "--ckpt-path", {fixture["context_manifest"]!r},
            "--run-dir", {fixture["shards"][0]!r},
            "--out-dir", {fixture["out_dir"]!r},
            "--audit-report", {fixture["audit_report"]!r},
            "--context-manifest", {fixture["context_manifest"]!r},
            "--metadata-root", {fixture["metadata_root"]!r},
            "--expect-metadata-bank-sha256", {fixture["metadata_bank_sha256"]!r}])
        try:
            op.gate_run(args, {{}}, {fixture["context_manifest"]!r},
                        totals={fixture["totals"]!r}, require_manifest_census=False)
        except ValueError:
            pass                      # WHICH gate refuses is not the claim here
        else:
            raise AssertionError("the gate should have refused this run")
        assert "eval_FLAC" not in sys.modules, "a REFUSED run still imported eval_FLAC"
        assert "eval_localization" not in sys.modules
        print("OK")
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "OK" in done.stdout


def test_the_checkpoint_is_read_only_after_every_gate_has_passed(tmp_path, monkeypatch):
    """The same claim, cheaply: a refused gate never reaches the checkpoint."""
    import localize_meshgrid as driver

    fixture = _probe_fixture(tmp_path)
    touched = []
    monkeypatch.setattr(op, "_load_and_validate_checkpoint",
                        lambda *a, **k: touched.append("ckpt") or {})
    # a binding that MATCHES, so the merge-report gate is the one that fires
    monkeypatch.setattr(driver, "build_run_binding",
                        lambda *a, **k: dict(fixture["binding"]))
    args = op.parse_args(["--ckpt-path", fixture["context_manifest"],
                          "--run-dir", fixture["shards"][0],
                          "--out-dir", fixture["out_dir"],
                          "--audit-report", fixture["audit_report"],
                          "--context-manifest", fixture["context_manifest"],
                          "--metadata-root", fixture["metadata_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"]])
    with pytest.raises(ValueError, match="merge_report.json"):
        op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)
    assert touched == []

    # and when every gate passes, the checkpoint IS read -- the test would be
    # vacuous if the call had simply been removed
    touched.clear()
    args = op.parse_args(["--ckpt-path", fixture["context_manifest"],
                          "--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                          "--audit-report", fixture["audit_report"],
                          "--context-manifest", fixture["context_manifest"],
                          "--metadata-root", fixture["metadata_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"]])
    op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                require_manifest_census=False)
    assert touched == ["ckpt"]


def test_a_canonical_control_requires_the_pre_registered_bank_digest():
    base = ["--ckpt-path", "c.ckpt", "--run-dir", "r", "--out-dir", "o"]
    with pytest.raises(SystemExit, match="requires the PRE-REGISTERED"):
        op.validate_args(op.parse_args(base))
    assert op.validate_args(op.parse_args(base + ["--non-canonical"])) is True
    assert op.validate_args(op.parse_args(
        base + ["--expect-metadata-bank-sha256", "a" * 64])) is True


def test_the_probe_digest_mode_needs_nothing_but_the_tree():
    args = op.parse_args(["--print-metadata-bank-digest"])
    assert args.ckpt_path is None and args.run_dir is None
    assert op.validate_args(args) is True


def test_the_probe_canonical_status_names_each_relaxation(tmp_path):
    fixture = _probe_fixture(tmp_path)
    gate = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    gate["metadata_bank_expected"] = fixture["metadata_bank_sha256"]
    assert op.probe_canonical_status(gate)["canonical"] is True

    gate["metadata_bank_expected"] = None
    status = op.probe_canonical_status(gate)
    assert [reason["gate"] for reason in status["reasons"]] == ["metadata_bank"]
    assert "NON-CANONICAL" in status["note"]

    gate["single_shard"] = True
    assert sorted(reason["gate"] for reason in op.probe_canonical_status(gate)["reasons"]) == \
        ["merge_report", "metadata_bank"]


def test_a_dump_carries_its_own_sensitivity_status_and_disclosures(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=False)
    op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                          fixture["binding_sha256"], provenance={},
                          tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    for record in records:
        assert record["sensitivity_status"] == op.CANONICAL_STATUS_CANONICAL
        with np.load(os.path.join(fixture["out_dir"], record["waveform_path"])) as data:
            assert str(data["sensitivity_status"]) == op.CANONICAL_STATUS_CANONICAL
            assert str(data["latency_scope_note"]) == mr.LATENCY_SCOPE_NOTE
            assert str(data["truth_binding_note"]) == mr.TRUTH_BINDING_NOTE
            assert json.loads(str(data["controls_elsewhere"])) == mr.CONTROLS_ELSEWHERE


def test_a_non_canonical_run_stamps_every_dump_as_one(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=True)
    for record in records:
        assert record["sensitivity_status"] == op.CANONICAL_STATUS_NON_CANONICAL
        staged = os.path.join(fixture["out_dir"], record["waveform_staged_path"])
        with np.load(staged) as data:
            assert "NON-CANONICAL" in str(data["sensitivity_status"])


def test_the_manifest_is_written_before_any_final_leaves_quarantine(tmp_path, monkeypatch):
    """M9: a crash during the move must not leave unmanifested finals."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    seen = {}

    real_publish = op.publish_probe_waveforms

    def _publish(out_dir, recs):
        seen["manifest_existed"] = os.path.isfile(
            os.path.join(str(out_dir), op.PROBE_REPORT_JSON))
        seen["markdown_existed"] = os.path.isfile(
            os.path.join(str(out_dir), op.PROBE_REPORT_MARKDOWN))
        return real_publish(out_dir, recs)

    monkeypatch.setattr(op, "publish_probe_waveforms", _publish)
    op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                          fixture["binding_sha256"], provenance={},
                          tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    assert seen["manifest_existed"] is True
    assert seen["markdown_existed"] is True


def test_a_failed_move_rolls_every_dump_back_into_quarantine(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert len(records) >= 2
    # the second dump vanishes between staging and publication
    os.remove(os.path.join(fixture["out_dir"], records[1]["waveform_staged_path"]))
    with pytest.raises(ValueError, match="the staged dump"):
        op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                              fixture["binding_sha256"], provenance={},
                              tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    # the first one was moved and is rolled back: no partial published set
    published = os.path.join(fixture["out_dir"], op.WAVEFORM_DIRNAME)
    assert [name for name in os.listdir(published) if name.endswith(".npz")] == []
    assert os.path.isfile(os.path.join(fixture["out_dir"],
                                       records[0]["waveform_staged_path"]))


def test_a_published_set_that_does_not_match_the_manifest_is_refused(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                          fixture["binding_sha256"], provenance={},
                          tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    assert op.verify_published_probe(fixture["out_dir"], records)["verified"] is True
    os.remove(os.path.join(fixture["out_dir"], records[0]["waveform_path"]))
    with pytest.raises(ValueError, match="does not match the manifest"):
        op.verify_published_probe(fixture["out_dir"], records)


def test_the_off_grid_markdown_renders_the_latency_scope_and_controls(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate={"single_shard": False, "metadata_bank_expected": "a" * 64,
              "registered_protocol": {"is_registered": True, "deviations": {}}})
    markdown = open(published["markdown"]).read()
    assert mr.LATENCY_SCOPE_NOTE in markdown
    assert "§2 controls that are NOT in this report" in markdown
    for name in mr.CONTROLS_ELSEWHERE:
        assert name in markdown
    assert "NON-CANONICAL" not in markdown


def test_a_non_canonical_control_says_so_in_its_markdown(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate={"single_shard": True, "metadata_bank_expected": None,
              "registered_protocol": {"is_registered": True, "deviations": {}}})
    payload = json.load(open(published["json"]))
    assert payload["canonical_status"]["canonical"] is False
    markdown = open(published["markdown"]).read()
    assert "NON-CANONICAL" in markdown
    assert "`metadata_bank`" in markdown


# --------------------------------------------------------------------------- #
# r9g: the residuals the Codex r9f verify pass left open in this control
# --------------------------------------------------------------------------- #
def _gate_args(fixture, run_dir=None, **extra):
    argv = ["--ckpt-path", fixture["context_manifest"],
            "--run-dir", run_dir or fixture["run_dir"],
            "--out-dir", fixture["out_dir"],
            "--audit-report", fixture["audit_report"],
            "--context-manifest", fixture["context_manifest"],
            "--metadata-root", fixture["metadata_root"]]
    for flag, value in extra.items():
        option = "--" + flag.replace("_", "-")
        argv += [option] if value is True else [option, str(value)]
    return op.parse_args(argv)


def _matching_binding(monkeypatch, fixture):
    """Make build_run_binding return the fixture's, so later gates are reached."""
    import localize_meshgrid as driver

    monkeypatch.setattr(driver, "build_run_binding", lambda *a, **k: dict(fixture["binding"]))
    monkeypatch.setattr(op, "_load_and_validate_checkpoint", lambda *a, **k: {})


def test_the_probe_computes_the_bank_and_compares_it_to_the_pin(tmp_path, monkeypatch):
    """B3 residual: r9d stored the expected string and never read the tree."""
    fixture = _probe_fixture(tmp_path)
    _matching_binding(monkeypatch, fixture)

    args = _gate_args(fixture, expect_metadata_bank_sha256=fixture["metadata_bank_sha256"])
    _plan, _manifest, _binding, gate, _ckpt = op.gate_run(
        args, {}, fixture["context_manifest"], totals=fixture["totals"],
        require_manifest_census=False)
    assert gate["metadata_bank_sha256"] == fixture["metadata_bank_sha256"]
    assert gate["metadata_bank"]["pinned"] is True
    assert gate["non_canonical"] is False

    # a WRONG pin is refused -- which r9d could not do, having never computed it
    wrong = _gate_args(fixture, expect_metadata_bank_sha256="a" * 64)
    with pytest.raises(ValueError, match="not the registered ones"):
        op.gate_run(wrong, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)


def test_a_wrong_bank_pin_refuses_before_the_checkpoint_is_touched(tmp_path, monkeypatch):
    """B3 + B5 together: the bank gate must precede the eval_FLAC import."""
    import localize_meshgrid as driver

    fixture = _probe_fixture(tmp_path)
    touched = []
    monkeypatch.setattr(driver, "build_run_binding", lambda *a, **k: dict(fixture["binding"]))
    monkeypatch.setattr(op, "_load_and_validate_checkpoint",
                        lambda *a, **k: touched.append("ckpt") or {})
    args = _gate_args(fixture, expect_metadata_bank_sha256="a" * 64)
    with pytest.raises(ValueError, match="not the registered ones"):
        op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)
    assert touched == []


def test_an_edited_truth_tree_is_caught_by_the_probes_own_bank_gate(tmp_path, monkeypatch):
    fixture = _probe_fixture(tmp_path)
    _matching_binding(monkeypatch, fixture)
    args = _gate_args(fixture, expect_metadata_bank_sha256=fixture["metadata_bank_sha256"])
    op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                require_manifest_census=False)

    mirrored = fx._mirrored_truth("A/A_idx_1")
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = mirrored
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="not the registered ones"):
        op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)


def test_the_probe_run_gate_derives_the_receipt_from_the_rows(tmp_path):
    """B4 residual: r9d passed derived=None, so nothing was cross-checked."""
    fixture = _probe_fixture(tmp_path)
    verdict = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    assert verdict["derived"]["source_rows"] == fixture["totals"]["source_rows"]
    assert verdict["batching"]["batching"] == fx.FIXTURE_ADVISORY
    assert verdict["merge"]["receipt_cross_checked_against_rows"] is True


def test_the_probe_refuses_a_receipt_the_rows_do_not_support(tmp_path):
    fixture = _probe_fixture(tmp_path)
    path = os.path.join(fixture["run_dir"], "merge_report.json")
    pristine = json.load(open(path))
    totals = dict(pristine["totals"], source_rows=int(pristine["totals"]["source_rows"]) + 1)
    me.write_json(path, dict(pristine, totals=totals))
    with pytest.raises(ValueError, match="source_rows"):
        op.assert_probe_run_census(
            fixture["run_dir"], _published_binding(fixture["run_dir"]),
            fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
            totals=dict(fixture["totals"], source_rows=totals["source_rows"]))


def test_the_probe_refuses_mixed_or_stripped_batching(tmp_path):
    fixture = _probe_fixture(tmp_path)
    path = me.query_artifact_paths(fixture["run_dir"], "A/A_idx_1", 0)["row"]
    row = json.load(open(path))
    row["batching"] = {"batch_rows": 512, "source_chunk": 2}
    row["row_sha256"] = me.row_digest(row)
    me.write_json(path, row)
    with pytest.raises(ValueError, match="different batchings"):
        op.assert_probe_run_census(
            fixture["run_dir"], _published_binding(fixture["run_dir"]),
            fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
            totals=fixture["totals"])

    row["batching"] = {}
    row["row_sha256"] = me.row_digest(row)
    me.write_json(path, row)
    with pytest.raises(ValueError, match="no complete batching stamp"):
        op.assert_probe_run_census(
            fixture["run_dir"], _published_binding(fixture["run_dir"]),
            fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
            totals=fixture["totals"])


def test_the_publication_gate_carries_every_verdict_the_status_reads(tmp_path):
    """Item 3: r9d hand-listed a subset and dropped metadata_bank_expected."""
    fixture = _probe_fixture(tmp_path)
    gate = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    gate.update({"metadata_bank_expected": fixture["metadata_bank_sha256"],
                 "metadata_bank_sha256": fixture["metadata_bank_sha256"],
                 "metadata_bank": {"pinned": True}, "non_canonical": False})
    sliced = op.publication_gate(gate)
    for field in ("single_shard", "registered_protocol", "metadata_bank_expected"):
        assert field in sliced, f"{field} is read by probe_canonical_status"
    assert op.probe_canonical_status(sliced) == op.probe_canonical_status(gate)
    assert op.probe_canonical_status(sliced)["canonical"] is True


def test_json_markdown_and_npz_agree_that_a_run_is_canonical(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=False)
    gate = {"single_shard": False, "metadata_bank_expected": fixture["metadata_bank_sha256"],
            "registered_protocol": {"is_registered": True, "deviations": {}}}
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate=op.publication_gate(gate))
    payload = json.load(open(published["json"]))
    markdown = open(published["markdown"]).read()
    assert payload["canonical_status"]["canonical"] is True
    assert "NON-CANONICAL" not in markdown
    for record in payload["records"]:
        assert record["sensitivity_status"] == op.CANONICAL_STATUS_CANONICAL
        with np.load(os.path.join(fixture["out_dir"], record["waveform_path"])) as data:
            assert str(data["sensitivity_status"]) == op.CANONICAL_STATUS_CANONICAL


def test_json_markdown_and_npz_agree_that_a_run_is_not_canonical(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=True)
    gate = {"single_shard": False, "metadata_bank_expected": None,
            "registered_protocol": {"is_registered": True, "deviations": {}}}
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate=op.publication_gate(gate))
    payload = json.load(open(published["json"]))
    markdown = open(published["markdown"]).read()
    assert payload["canonical_status"]["canonical"] is False
    assert "NON-CANONICAL" in markdown
    for record in payload["records"]:
        assert record["sensitivity_status"] == op.CANONICAL_STATUS_NON_CANONICAL
        with np.load(os.path.join(fixture["out_dir"], record["waveform_path"])) as data:
            assert "NON-CANONICAL" in str(data["sensitivity_status"])


def test_a_rename_failure_mid_loop_leaves_quarantine_complete(tmp_path, monkeypatch):
    """M9 residual: os.replace sat OUTSIDE the rollback handler."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert len(records) >= 2

    real_replace = os.replace
    calls = {"n": 0}

    def _failing(src, dst):
        # let the manifest's own tmp->final renames through; fail the Nth NPZ move
        if str(dst).endswith(".npz"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("injected rename failure on the 2nd dump")
        return real_replace(src, dst)

    monkeypatch.setattr(op.os, "replace", _failing)
    with pytest.raises(OSError, match="injected rename failure"):
        op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                              fixture["binding_sha256"], provenance={},
                              tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    monkeypatch.undo()

    # quarantine holds every dump again, and no final was left behind
    staging = os.path.join(fixture["out_dir"], op.WAVEFORM_STAGING_DIRNAME)
    published_dir = os.path.join(fixture["out_dir"], op.WAVEFORM_DIRNAME)
    assert sorted(os.listdir(staging)) == sorted(
        os.path.basename(record["waveform_staged_path"]) for record in records)
    assert [name for name in os.listdir(published_dir) if name.endswith(".npz")] == []
    assert all(record["waveform_published"] is False for record in records)
    # the safety-net manifest is on disk and says publication did NOT complete
    payload = json.load(open(os.path.join(fixture["out_dir"], op.PROBE_REPORT_JSON)))
    assert payload["publication"]["completed"] is False


def test_a_successful_run_records_publication_as_complete(tmp_path):
    """The r9f nit: a successful JSON persisted waveform_published=false."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                                      fixture["binding_sha256"], provenance={},
                                      tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    payload = json.load(open(published["json"]))
    assert payload["publication"]["completed"] is True
    assert payload["publication"]["n_published"] == len(records)
    assert payload["publication"]["verified"] is True
    for record in payload["records"]:
        assert record["waveform_published"] is True
    assert published["sha256"]["json"] == me.file_sha256(published["json"])
    assert "manifest is written and fsynced BEFORE" in payload["publication"]["note"]
