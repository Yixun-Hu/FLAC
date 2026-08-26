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


def test_the_probe_saves_the_generated_waveforms_with_their_digest(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    for record in records:
        path = os.path.join(fixture["out_dir"], record["waveform_path"])
        assert os.path.isfile(path)
        assert me.file_sha256(path) == record["waveform_sha256"]
        with np.load(path) as data:
            assert data["waveforms"].shape[0] == fx.FIXTURE_SAMPLES
            assert data["context_audio"].shape[0] == 8
            assert data["truth_xyz"].tolist() == record["truth_xyz"]
            assert data["receiver_xyz"].tolist() == record["receiver_xyz"]
        assert "announcement 08" in record["waveform_note"]


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
    with pytest.raises(ValueError, match="not looking at the same query"):
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
    args = op.parse_args(["--ckpt-path", "c.ckpt", "--run-dir", "run", "--out-dir", "out"])
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
                          "--out-dir", str(tmp_path)])
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
