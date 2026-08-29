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
import re

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

    # materialize the observed-RIR files the pre-registered bank digests as REAL
    # wavs whose decode is bit-equal to the tensor the fixture stream hands over,
    # so the byte -> tensor single path is exercised rather than stubbed
    import torchaudio

    root = str(tmp_path / "dataset")
    items = []
    for obs, md in fixture["items"]:
        path = os.path.join(root, md["relpath"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, obs.reshape(1, -1), FIXTURE_SAMPLE_RATE,
                        encoding="PCM_F", bits_per_sample=32)
        items.append((obs, dict(md, path=path)))
    fixture["items"] = items
    fixture["dataset_root"] = root
    fixture["sample_size"] = int(items[0][0].shape[-1])
    fixture["observation_decoder"] = lambda path, expected: op.read_verified_observation(
        path, expected, sample_rate=FIXTURE_SAMPLE_RATE,
        sample_size=fixture["sample_size"], force_channels="mono")
    fixture["metadata_bank"] = mr.compute_metadata_bank_digest(
        fixture["context_manifest"], fixture["metadata_root"],
        require_manifest_census=False)
    # the PRE-REGISTRATION step, run exactly as the CLI runs it: no run dir, no
    # checkpoint, no scorer
    bank = op.compute_observation_bank_digest(fixture["audit_report"],
                                              fixture["context_manifest"],
                                              dataset_root=root,
                                              require_manifest_census=False)
    fixture["observation_bank"] = bank
    fixture["observation_bank_sha256"] = bank["observation_bank_sha256"]
    return fixture


FIXTURE_SAMPLE_RATE = 22050


def _run(fixture, stream=None, **kwargs):
    """Walk the fixture as a CANONICAL control unless the caller says otherwise."""
    kwargs.setdefault("observation_bank", fixture["observation_bank"])
    kwargs.setdefault("observation_decoder", fixture["observation_decoder"])
    kwargs.setdefault("metadata_bank", fixture["metadata_bank"])
    return op.run_probe(fixture["engine"],
                        list(fixture["items"] if stream is None else stream),
                        fixture["records"],
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
    # the verified-pair gate now catches the edit BEFORE the oracle does, which
    # is the r9l item-1 fix: a pair file edited after the freeze cannot supply a
    # truth at all
    with pytest.raises(ValueError, match="the truth being read is not the registered one"):
        _run(fixture)
    # ... and with no bank supplied the scalar oracle check is still the backstop
    with pytest.raises(ValueError, match="not the one the audit measured"):
        _run(fixture, metadata_bank=None)


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
                 "--dataset-root", fixture["dataset_root"],
                 "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
                 "--expect-observation-bank-sha256",
                 fixture["observation_bank_sha256"]])
    assert order == ["gate"]


def test_gate_run_touches_no_device_and_returns_the_whole_ladder(tmp_path, monkeypatch):
    fixture = _probe_fixture(tmp_path)
    args = op.parse_args(["--ckpt-path", fixture["context_manifest"],
                          "--run-dir", fixture["run_dir"], "--out-dir", fixture["out_dir"],
                          "--audit-report", fixture["audit_report"],
                          "--context-manifest", fixture["context_manifest"],
                          "--metadata-root", fixture["metadata_root"],
                          "--dataset-root", fixture["dataset_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
                          "--expect-observation-bank-sha256",
                          fixture["observation_bank_sha256"],
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
    # the verified-pair gate refuses it outright now; the vector check remains
    # the backstop when no bank is supplied, and the scalar oracle is blind to it
    with pytest.raises(ValueError, match="the truth being read is not the registered one"):
        _run(fixture)
    with pytest.raises(ValueError, match="is not the one the query was held out from"):
        _run(fixture, metadata_bank=None)


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
            "--dataset-root", {fixture["dataset_root"]!r},
            "--expect-metadata-bank-sha256", {fixture["metadata_bank_sha256"]!r},
            "--expect-observation-bank-sha256", {fixture["observation_bank_sha256"]!r}])
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
                          "--dataset-root", fixture["dataset_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
                          "--expect-observation-bank-sha256",
                          fixture["observation_bank_sha256"]])
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
                          "--dataset-root", fixture["dataset_root"],
                          "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
                          "--expect-observation-bank-sha256",
                          fixture["observation_bank_sha256"]])
    op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                require_manifest_census=False)
    assert touched == ["ckpt"]


def test_a_canonical_control_requires_the_pre_registered_bank_digest():
    base = ["--ckpt-path", "c.ckpt", "--run-dir", "r", "--out-dir", "o"]
    with pytest.raises(SystemExit, match="requires the PRE-REGISTERED"):
        op.validate_args(op.parse_args(base))
    assert op.validate_args(op.parse_args(base + ["--non-canonical"])) is True
    assert op.validate_args(op.parse_args(
        base + ["--expect-metadata-bank-sha256", "a" * 64,
                "--expect-observation-bank-sha256", "b" * 64])) is True


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
    gate["observation_bank_expected"] = fixture["observation_bank_sha256"]
    assert op.probe_canonical_status(gate)["canonical"] is True

    gate["metadata_bank_expected"] = None
    status = op.probe_canonical_status(gate)
    assert [reason["gate"] for reason in status["reasons"]] == ["metadata_bank"]
    assert "NON-CANONICAL" in status["note"]

    gate["observation_bank_expected"] = None
    assert sorted(reason["gate"] for reason in op.probe_canonical_status(gate)["reasons"]) == \
        ["metadata_bank", "observation_bank"]

    gate["single_shard"] = True
    assert sorted(reason["gate"] for reason in op.probe_canonical_status(gate)["reasons"]) == \
        ["merge_report", "metadata_bank", "observation_bank"]


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
              "observation_bank_expected": "b" * 64,
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
              "observation_bank_expected": None,
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
            "--metadata-root", fixture["metadata_root"],
            "--dataset-root", fixture["dataset_root"],
            "--expect-metadata-bank-sha256", fixture["metadata_bank_sha256"],
            "--expect-observation-bank-sha256", fixture["observation_bank_sha256"]]
    # extras are appended, and argparse lets a later flag override an earlier one
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
                 "metadata_bank": {"pinned": True},
                 "observation_bank_expected": fixture["observation_bank_sha256"],
                 "observation_bank_sha256": fixture["observation_bank_sha256"],
                 "observation_bank": {"pinned": True}, "non_canonical": False})
    sliced = op.publication_gate(gate)
    for field in ("single_shard", "registered_protocol", "metadata_bank_expected"):
        assert field in sliced, f"{field} is read by probe_canonical_status"
    assert op.probe_canonical_status(sliced) == op.probe_canonical_status(gate)
    assert op.probe_canonical_status(sliced)["canonical"] is True


def test_json_markdown_and_npz_agree_that_a_run_is_canonical(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=False)
    gate = {"single_shard": False, "metadata_bank_expected": fixture["metadata_bank_sha256"],
            "observation_bank_expected": fixture["observation_bank_sha256"],
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
            "observation_bank_expected": None,
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


# --------------------------------------------------------------------------- #
# r9j: the last items from the Codex r9i verify pass
# --------------------------------------------------------------------------- #
def test_the_observation_is_tied_to_the_frozen_rows_it_is_ranked_against(tmp_path):
    """Item 2: no artifact digests the observation, so the tie is functional."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    for record in records:
        verdict = record["observation_continuity"]
        assert verdict["ok"] is True
        assert verdict["max_abs_delta"] <= verdict["tolerance"]
        # the tolerance is built from REGISTERED constants, not chosen here
        assert verdict["tolerance"] > me.SCORE_TOLERANCE
        # the check lands on the row's HEADLINE prediction, where the result is
        assert verdict["candidate_index"] == record["rank_lme"][
            str(max(fx.FIXTURE_PREFIXES))]["grid_prediction_index"]
        assert 0 <= verdict["candidate_row"] < record["n_candidates"]
        assert verdict["num_samples"] == fx.FIXTURE_SAMPLES
        assert len(verdict["stored"]) == len(verdict["rederived"]) == fx.FIXTURE_SAMPLES
        assert "THE TIE, over the TENSOR PATH" in verdict["note"]


def test_a_substituted_observation_is_refused_against_the_frozen_rows(tmp_path):
    """The exploit: score a different observation against the same rows."""
    fixture = _probe_fixture(tmp_path)
    items = [(obs, md) for obs, md in fixture["items"]]
    # the first probe query's observation becomes a different waveform
    obs, md = items[0]
    # a RAMP, not an offset: the synthetic embedder L2-normalizes, so a constant
    # observation scaled by a constant embeds identically -- the substitution has
    # to change the observation's SHAPE for the fixture to express it at all
    ramp = torch.arange(obs.shape[-1], dtype=obs.dtype).reshape(1, 1, -1) * 0.05
    substituted = obs + ramp
    assert not torch.allclose(fixture["engine"]._embed(substituted),
                              fixture["engine"]._embed(obs)), "the substitution must be visible"
    items[0] = (substituted, md)
    with pytest.raises(ValueError, match="not the observation those rows were scored against"):
        op.run_probe(fixture["engine"], items, fixture["records"], fixture["plan"],
                     fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)


def test_the_observation_note_states_both_the_pin_and_the_tie():
    note = op.OBSERVATION_BINDING_NOTE
    assert "bound TWICE" in note
    assert "THE PIN, over SOURCE BYTES" in note
    assert "THE TIE, over the TENSOR PATH" in note
    assert "PRE-REGISTERED before any result exists" in note
    # it still says what no run artifact does
    assert "the observation is in none of them" in note
    assert "CONTEXT RIRs" in note and "verify_context_record" in note
    # and why neither check subsumes the other
    assert "passes the pin and fails the tie" in note


def test_the_observation_digests_are_recorded_from_the_verified_read(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    for record in records:
        observation = record["observation"]
        assert len(observation["tensor_sha256"]) == 64
        # the digest came from the ONE verified read, not from a second open
        assert observation["pinned"] is True
        assert observation["source_sha256"] == record["observation_source"]["sha256"]
        assert observation["shape"][0] == 1

    # unpinned (non-canonical) runs still record what they read, and say so
    unpinned = _run(fixture, observation_bank=None, observation_decoder=None)
    for record in unpinned:
        assert record["observation"]["pinned"] is False


def test_disabling_the_continuity_check_makes_the_control_non_canonical(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, verify_observation=False)
    assert records[0]["observation_continuity"] is None
    summary = records[0]["observation_continuity_summary"]
    assert summary["ok"] is False
    status = op.probe_canonical_status(
        {"metadata_bank_expected": "a" * 64, "observation_bank_expected": "b" * 64,
         "observation_continuity": summary})
    assert status["canonical"] is False
    assert "observation_continuity" in [reason["gate"] for reason in status["reasons"]]


def test_a_declared_non_canonical_run_marks_json_markdown_and_npz(tmp_path):
    """Item 4: a valid pin PLUS --non-canonical used to yield canonical JSON."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture, non_canonical=True)
    gate = {"single_shard": False,
            "metadata_bank_expected": fixture["metadata_bank_sha256"],
            "observation_bank_expected": fixture["observation_bank_sha256"],
            "registered_protocol": {"is_registered": True, "deviations": {}},
            "non_canonical": True, "non_canonical_declared": True}
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate=op.publication_gate(gate))
    payload = json.load(open(published["json"]))
    assert payload["canonical_status"]["canonical"] is False
    assert "declared_non_canonical" in [reason["gate"]
                                        for reason in payload["canonical_status"]["reasons"]]
    assert "NON-CANONICAL" in open(published["markdown"]).read()
    for record in payload["records"]:
        assert record["sensitivity_status"] == op.CANONICAL_STATUS_NON_CANONICAL
        with np.load(os.path.join(fixture["out_dir"], record["waveform_path"])) as data:
            assert "NON-CANONICAL" in str(data["sensitivity_status"])


def test_the_status_can_never_be_more_canonical_than_the_gate(tmp_path):
    """Fail-closed: an unenumerated reason is still a reason."""
    status = op.probe_canonical_status(
        {"metadata_bank_expected": "a" * 64, "observation_bank_expected": "b" * 64,
         "single_shard": False,
         "registered_protocol": {"is_registered": True, "deviations": {}},
         "non_canonical": True})
    assert status["canonical"] is False
    assert [reason["gate"] for reason in status["reasons"]] == ["non_canonical_flag"]
    # and the flag survives the publication slice
    assert "non_canonical" in op.PUBLICATION_GATE_FIELDS
    assert "non_canonical_declared" in op.PUBLICATION_GATE_FIELDS
    sliced = op.publication_gate({"non_canonical": True, "non_canonical_declared": True,
                                  "metadata_bank_expected": "a" * 64,
                                  "observation_bank_expected": "b" * 64})
    assert op.probe_canonical_status(sliced)["canonical"] is False


def test_publication_journals_every_intended_rename_before_the_first(tmp_path):
    """Item 3: an in-process handler cannot survive SIGKILL."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    seen = {}
    real_replace = os.replace

    def _spy(src, dst):
        if str(dst).endswith(".npz") and "journal" not in seen:
            path = os.path.join(fixture["out_dir"], op.PUBLICATION_JOURNAL)
            seen["journal"] = json.load(open(path)) if os.path.isfile(path) else None
        return real_replace(src, dst)

    op.os.replace = _spy
    try:
        op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                              fixture["binding_sha256"], provenance={},
                              tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    finally:
        op.os.replace = real_replace

    # the journal existed, listed EVERY final, and said publication was unfinished
    assert seen["journal"] is not None
    assert seen["journal"]["completed"] is False
    assert sorted(move["final"] for move in seen["journal"]["moves"]) == \
        sorted(record["waveform_path"] for record in records)
    # ... and it is marked complete once every rename landed and verified
    done = json.load(open(os.path.join(fixture["out_dir"], op.PUBLICATION_JOURNAL)))
    assert done["completed"] is True and done["n_published"] == len(records)


def test_a_crash_after_the_nth_rename_is_recovered_into_quarantine(tmp_path):
    """A hard crash reaches no handler; the journal is what survives it."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert len(records) >= 2

    # simulate the crash: journal, move the first final, then stop dead --
    # no rollback, no completion, exactly what SIGKILL leaves behind
    op.write_publication_journal(fixture["out_dir"], records)
    first = records[0]
    source = os.path.join(fixture["out_dir"], first["waveform_staged_path"])
    target = os.path.join(fixture["out_dir"], first["waveform_path"])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.replace(source, target)
    assert os.path.isfile(target)                 # a partial final set exists

    # the next startup path finds the incomplete journal and quarantines it
    recovery = op.recover_publication(fixture["out_dir"])
    assert recovery["recovered"] is True
    assert recovery["n_quarantined"] == 1
    assert not os.path.isfile(target)
    assert os.path.isfile(source)
    published = os.path.join(fixture["out_dir"], op.WAVEFORM_DIRNAME)
    assert [name for name in os.listdir(published) if name.endswith(".npz")] == []
    assert not os.path.isfile(os.path.join(fixture["out_dir"], op.PUBLICATION_JOURNAL))


def test_write_probe_report_recovers_a_crashed_predecessor_first(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    op.write_publication_journal(fixture["out_dir"], records)
    first = records[0]
    os.makedirs(os.path.join(fixture["out_dir"], op.WAVEFORM_DIRNAME), exist_ok=True)
    os.replace(os.path.join(fixture["out_dir"], first["waveform_staged_path"]),
               os.path.join(fixture["out_dir"], first["waveform_path"]))

    published = op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                                      fixture["binding_sha256"], provenance={},
                                      tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    payload = json.load(open(published["json"]))
    assert payload["publication"]["recovery"]["recovered"] is True
    assert payload["publication"]["recovery"]["n_quarantined"] == 1
    assert payload["publication"]["completed"] is True
    assert op.verify_published_probe(fixture["out_dir"], records)["verified"] is True


def test_a_complete_journal_is_not_rolled_back(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    op.write_probe_report(fixture["out_dir"], records, fixture["binding"],
                          fixture["binding_sha256"], provenance={},
                          tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES)
    recovery = op.recover_publication(fixture["out_dir"])
    assert recovery["recovered"] is False
    assert recovery["reason"] == "the journal is complete"
    for record in records:
        assert os.path.isfile(os.path.join(fixture["out_dir"], record["waveform_path"]))
    assert op.recover_publication(str(tmp_path / "nowhere"))["reason"] == "no journal"


# --------------------------------------------------------------------------- #
# r9j2: the observation is PINNED by a pre-registered digest, not only tied
# --------------------------------------------------------------------------- #
def test_the_observation_digest_mode_needs_no_run_no_ckpt_and_no_gpu(tmp_path):
    fixture = _probe_fixture(tmp_path)
    args = op.parse_args(["--print-observation-digest"])
    assert args.run_dir is None and args.out_dir is None and args.ckpt_path is None
    assert op.validate_args(args) is True

    verdict = op.compute_observation_bank_digest(fixture["audit_report"],
                                                 fixture["context_manifest"],
                                                 dataset_root=fixture["dataset_root"],
                                                 require_manifest_census=False)
    # one entry per REGISTERED probe query -- the audit's own rule, not a list
    probes = me.registered_probe_queries(fixture["plan"])
    assert sorted(verdict["queries"]) == sorted(probes.values())
    assert verdict["n_queries"] == len(probes)
    for query_id, entry in verdict["queries"].items():
        path = os.path.join(fixture["dataset_root"], entry["relpath"])
        assert entry["sha256"] == me.file_sha256(path)
        assert entry["n_bytes"] == os.path.getsize(path)
    assert len(verdict["observation_bank_sha256"]) == 64
    assert "--print-observation-digest" in verdict["how_to_register"]


def test_the_observation_bank_digest_is_deterministic_and_content_addressed(tmp_path):
    fixture = _probe_fixture(tmp_path)
    again = op.compute_observation_bank_digest(fixture["audit_report"],
                                               fixture["context_manifest"],
                                               dataset_root=fixture["dataset_root"],
                                               require_manifest_census=False)
    assert again["observation_bank_sha256"] == fixture["observation_bank_sha256"]

    # one edited observation byte changes it
    entry = next(iter(fixture["observation_bank"]["queries"].values()))
    path = os.path.join(fixture["dataset_root"], entry["relpath"])
    with open(path, "ab") as handle:
        handle.write(b"\x00")
    moved = op.compute_observation_bank_digest(fixture["audit_report"],
                                               fixture["context_manifest"],
                                               dataset_root=fixture["dataset_root"],
                                               require_manifest_census=False)
    assert moved["observation_bank_sha256"] != fixture["observation_bank_sha256"]


def test_a_missing_observation_refuses_rather_than_banking_fifteen(tmp_path):
    fixture = _probe_fixture(tmp_path)
    entry = next(iter(fixture["observation_bank"]["queries"].values()))
    os.remove(os.path.join(fixture["dataset_root"], entry["relpath"]))
    with pytest.raises(ValueError, match="could not be read"):
        op.compute_observation_bank_digest(fixture["audit_report"],
                                           fixture["context_manifest"],
                                           dataset_root=fixture["dataset_root"],
                                           require_manifest_census=False)


def test_the_digest_reads_each_observation_exactly_once(tmp_path):
    """The r9j item-1 pattern: what is said about a file is said about one read."""
    fixture = _probe_fixture(tmp_path)
    wanted = {os.path.realpath(os.path.join(fixture["dataset_root"], entry["relpath"]))
              for entry in fixture["observation_bank"]["queries"].values()}
    opened = []
    real_open = open
    import builtins

    def _spy(path, *args, **kwargs):
        if os.path.realpath(str(path)) in wanted:
            opened.append(os.path.realpath(str(path)))
        return real_open(path, *args, **kwargs)

    builtins.open = _spy
    try:
        op.compute_observation_bank_digest(fixture["audit_report"],
                                           fixture["context_manifest"],
                                           dataset_root=fixture["dataset_root"],
                                           require_manifest_census=False)
    finally:
        builtins.open = real_open
    assert sorted(opened) == sorted(wanted)          # exactly one read each


def test_a_canonical_probe_requires_the_pre_registered_observation_pin():
    base = ["--ckpt-path", "c.ckpt", "--run-dir", "r", "--out-dir", "o",
            "--expect-metadata-bank-sha256", "a" * 64]
    with pytest.raises(SystemExit, match="PRE-REGISTERED observed-RIR bank"):
        op.validate_args(op.parse_args(base))
    assert op.validate_args(op.parse_args(base + ["--non-canonical"])) is True
    assert op.validate_args(op.parse_args(
        base + ["--expect-observation-bank-sha256", "b" * 64])) is True
    # the two pre-registration modes are run one at a time
    with pytest.raises(SystemExit, match="one at a time"):
        op.validate_args(op.parse_args(["--print-metadata-bank-digest",
                                        "--print-observation-digest"]))


def test_a_wrong_observation_pin_refuses_before_the_checkpoint(tmp_path, monkeypatch):
    """The pin is compared in the same pre-import window as the metadata bank."""
    import localize_meshgrid as driver

    fixture = _probe_fixture(tmp_path)
    touched = []
    monkeypatch.setattr(driver, "build_run_binding", lambda *a, **k: dict(fixture["binding"]))
    monkeypatch.setattr(op, "_load_and_validate_checkpoint",
                        lambda *a, **k: touched.append("ckpt") or {})
    args = _gate_args(fixture)
    args.expect_observation_bank_sha256 = "0" * 64
    with pytest.raises(ValueError, match="not the registered ones"):
        op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)
    assert touched == []


def test_the_gate_computes_and_pins_the_observation_bank(tmp_path, monkeypatch):
    fixture = _probe_fixture(tmp_path)
    _matching_binding(monkeypatch, fixture)
    _plan, _manifest, _binding, gate, _ckpt = op.gate_run(
        _gate_args(fixture), {}, fixture["context_manifest"], totals=fixture["totals"],
        require_manifest_census=False)
    assert gate["observation_bank_sha256"] == fixture["observation_bank_sha256"]
    assert gate["observation_bank"]["pinned"] is True
    assert sorted(gate["observation_bank"]["queries"]) == \
        sorted(me.registered_probe_queries(fixture["plan"]).values())
    assert gate["non_canonical"] is False
    assert op.probe_canonical_status(op.publication_gate(gate))["canonical"] is True


def test_an_unpinned_observation_bank_makes_the_control_non_canonical(tmp_path, monkeypatch):
    fixture = _probe_fixture(tmp_path)
    _matching_binding(monkeypatch, fixture)
    args = _gate_args(fixture)
    args.expect_observation_bank_sha256 = None
    with pytest.raises(ValueError, match="requires the PRE-REGISTERED observed-RIR bank"):
        op.gate_run(args, {}, fixture["context_manifest"], totals=fixture["totals"],
                    require_manifest_census=False)
    args.non_canonical = True
    _plan, _manifest, _binding, gate, _ckpt = op.gate_run(
        args, {}, fixture["context_manifest"], totals=fixture["totals"],
        require_manifest_census=False)
    assert gate["observation_bank"]["pinned"] is False
    assert gate["non_canonical"] is True
    status = op.probe_canonical_status(op.publication_gate(gate))
    assert "observation_bank" in [reason["gate"] for reason in status["reasons"]]


def test_the_loader_must_read_the_very_file_the_bank_digested(tmp_path):
    """A divergent root cannot satisfy the bank while the loader reads elsewhere."""
    fixture = _probe_fixture(tmp_path)
    # a second, pristine tree that satisfies the frozen digest byte for byte
    other = str(tmp_path / "elsewhere")
    items = []
    for obs, md in fixture["items"]:
        path = os.path.join(other, md["relpath"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:                 # DIFFERENT bytes
            handle.write(f"substituted:{md['relpath']}".encode())
        items.append((obs, dict(md, path=path)))

    # the decode reads THOSE bytes, so the pin fails on the bytes that would have
    # been scored -- there is no second read to restore the registered file into
    with pytest.raises(ValueError, match="the bytes being decoded are not the registered ones"):
        op.run_probe(fixture["engine"], items, fixture["records"], fixture["plan"],
                     fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES,
                     observation_bank=fixture["observation_bank"],
                     observation_decoder=fixture["observation_decoder"],
                     metadata_bank=fixture["metadata_bank"])


def test_a_probe_run_records_which_source_bytes_it_verified(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    for record in records:
        source = record["observation_source"]
        assert source["ok"] is True
        assert source["sha256"] == \
            fixture["observation_bank"]["queries"][record["query_id"]]["sha256"]
        assert record["observation_source_pinned"] is True


def test_the_pin_and_the_tie_are_independent_gates(tmp_path):
    """Byte-identical files loaded through a changed tensor path fail the tie."""
    fixture = _probe_fixture(tmp_path)
    items = list(fixture["items"])
    obs, md = items[0]
    # the FILE is untouched -- the bank still matches -- but the tensor the
    # loader hands over is different, which only the functional tie can see
    ramp = torch.arange(obs.shape[-1], dtype=obs.dtype).reshape(1, 1, -1) * 0.05
    items[0] = (obs + ramp, md)
    # the FILE still satisfies the pin, so the single-path decode catches the
    # divergence between the registered bytes and the tensor handed over
    with pytest.raises(ValueError, match="not the tensor the loader handed over"):
        _run(fixture, stream=items)
    # and with no bank at all, the functional tie is what sees it
    with pytest.raises(ValueError, match="not the observation those rows were scored against"):
        op.run_probe(fixture["engine"], items, fixture["records"], fixture["plan"],
                     fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)


# --------------------------------------------------------------------------- #
# r9m: verified buffers all the way through the probe
# --------------------------------------------------------------------------- #
def test_the_runtime_truth_comes_from_a_verified_pair_buffer(tmp_path):
    """Item 1: r9j2 gated the bank, then built a fresh unchecked resolver.

    This mirrors the verified-pair pattern the retrieval control adopted in r9k
    (its own `assert_verified_pair` seam); the two are deliberately separate
    implementations because a round may not edit another round's file, and the
    cross-pin is this comment plus the shared TruthResolver they both call.
    """
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    assert len(records) >= 1

    # a pair file edited AFTER the freeze cannot supply a truth any more
    mirrored = fx._mirrored_truth("A/A_idx_1")
    path = os.path.join(fixture["metadata_root"], "A", "A_idx_1", "S001_R002.json")
    payload = json.load(open(path))
    payload["src_loc"] = mirrored
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with pytest.raises(ValueError, match="the truth being read is not the registered one"):
        _run(fixture)


def test_a_pair_file_the_bank_never_covered_supplies_no_truth(tmp_path):
    fixture = _probe_fixture(tmp_path)
    empty = dict(fixture["metadata_bank"], queries={})
    with pytest.raises(ValueError, match="does not cover this query"):
        _run(fixture, metadata_bank=empty)


def test_the_probe_reads_each_observation_exactly_once_at_runtime(tmp_path):
    """Item 2: there is no second open to restore the registered file into."""
    fixture = _probe_fixture(tmp_path)
    probes = set(me.registered_probe_queries(fixture["plan"]).values())
    watched = [md["path"] for _obs, md in fixture["items"]
               if f"{md['idx']}|{md['relpath']}" in probes]
    assert watched

    opened = []
    real_open = open
    import builtins

    def _spy(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and \
                os.path.realpath(str(path)) in {os.path.realpath(p) for p in watched}:
            opened.append(os.path.realpath(str(path)))
        return real_open(path, *args, **kwargs)

    builtins.open = _spy
    try:
        _run(fixture)
    finally:
        builtins.open = real_open
    for path in watched:
        assert opened.count(os.path.realpath(path)) == 1, opened


def test_the_scored_tie_and_calibration_share_one_observation_tensor(tmp_path):
    """The tie, the truth scores and the calibration read ONE embedding."""
    fixture = _probe_fixture(tmp_path)
    seen = []
    real = op.calibration_record

    def _spy(embedder, obs_embedding, context_audio, generated):
        seen.append(obs_embedding)
        return real(embedder, obs_embedding, context_audio, generated)

    op.calibration_record = _spy
    real_tie = op.assert_observation_continuity
    ties = []

    def _tie_spy(engine, query, md, context, row, sims, obs_embedding, **kwargs):
        ties.append(obs_embedding)
        return real_tie(engine, query, md, context, row, sims, obs_embedding, **kwargs)

    op.assert_observation_continuity = _tie_spy
    try:
        records = _run(fixture)
    finally:
        op.calibration_record = real
        op.assert_observation_continuity = real_tie

    assert len(seen) == len(ties) == len(records)
    for calibration_embedding, tie_embedding in zip(seen, ties):
        # the SAME object, not merely equal tensors
        assert calibration_embedding is tie_embedding


def test_the_decoded_observation_is_bit_equal_to_the_loaders_tensor(tmp_path):
    fixture = _probe_fixture(tmp_path)
    for obs, md in fixture["items"]:
        verified = fixture["observation_decoder"](md["path"], None)
        assert torch.equal(verified["tensor"].reshape(-1), obs.reshape(-1))
        assert verified["sha256"] == me.file_sha256(md["path"])
    # and a divergent decode is a refusal, not a silent second tensor
    obs, md = fixture["items"][0]
    with pytest.raises(ValueError, match="not the tensor the loader handed over"):
        op.assert_decoded_observation_matches("q", obs + 1.0, obs)
    with pytest.raises(ValueError, match="the control's decode and the released loader's"):
        op.assert_decoded_observation_matches("q", obs.reshape(-1)[:4], obs)


def test_the_decoder_reproduces_the_released_eval_transform(tmp_path):
    """The decode is the loader's: load -> pad/crop (no randomize) -> mono."""
    import torchaudio
    from src.data.utils import Mono, PadCrop_Normalized_T

    path = str(tmp_path / "probe_obs.wav")
    signal = torch.linspace(-0.5, 0.5, 40).reshape(1, -1)
    torchaudio.save(path, signal, FIXTURE_SAMPLE_RATE, encoding="PCM_F", bits_per_sample=32)

    verified = op.read_verified_observation(path, None, sample_rate=FIXTURE_SAMPLE_RATE,
                                            sample_size=32, force_channels="mono")
    audio, _sr = torchaudio.load(path, format="wav")
    reference = torch.nn.Sequential(Mono())(
        PadCrop_Normalized_T(32, FIXTURE_SAMPLE_RATE, randomize=False)(audio)[0])
    assert torch.equal(verified["tensor"], reference)
    assert verified["sha256"] == me.file_sha256(path)

    # a wrong pin refuses on the bytes that would have been decoded
    with pytest.raises(ValueError, match="the bytes being decoded are not the registered"):
        op.read_verified_observation(path, "0" * 64, sample_rate=FIXTURE_SAMPLE_RATE,
                                     sample_size=32)


def test_a_bank_without_a_decoder_is_refused_rather_than_trusted(tmp_path):
    fixture = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="no decoder"):
        _run(fixture, observation_decoder=None)


def test_the_probe_reads_each_grid_row_and_sidecar_exactly_once(tmp_path):
    """Item 3 in the probe: load_grid_row keeps the sidecar it verified."""
    fixture = _probe_fixture(tmp_path)
    probes = me.registered_probe_queries(fixture["plan"])
    watched = []
    for room_id, query_id in probes.items():
        room = me.load_room_plan(fixture["plan"], room_id)
        query = next(q for q in room.queries if q.query_id == query_id)
        entry = me.query_artifact_paths(fixture["run_dir"], query.room_id, query.position)
        watched += [entry["row"], entry["sims"]]

    opened = []
    real_open = open
    import builtins

    def _spy(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and \
                os.path.realpath(str(path)) in {os.path.realpath(p) for p in watched}:
            opened.append(os.path.realpath(str(path)))
        return real_open(path, *args, **kwargs)

    builtins.open = _spy
    try:
        _run(fixture)
    finally:
        builtins.open = real_open
    for path in watched:
        assert opened.count(os.path.realpath(path)) == 1, (path, opened)


def test_the_observation_decoder_is_built_from_the_pinned_configs(tmp_path):
    model_config = {"sample_rate": FIXTURE_SAMPLE_RATE, "sample_size": 32, "audio_channels": 1}
    decoder = op.build_observation_decoder(
        model_config, os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                   "acousticroom_unseeneval.json"))
    import torchaudio

    path = str(tmp_path / "obs.wav")
    torchaudio.save(path, torch.zeros(1, 32), FIXTURE_SAMPLE_RATE,
                    encoding="PCM_F", bits_per_sample=32)
    verified = decoder(path, me.file_sha256(path))
    assert tuple(verified["tensor"].shape) == (1, 32)
    with pytest.raises(ValueError, match="not the registered ones"):
        decoder(path, "0" * 64)


# --------------------------------------------------------------------------- #
# r9m2: the census's verdict does not expire when the census returns
# --------------------------------------------------------------------------- #
def _coherent_swap(run_dir, room_id, position, mutate_scores=True):
    """Replace a row AND its sidecar coherently -- both self-digests recomputed.

    This is the r9n exploit exactly: the replacement verifies against itself, so
    a phase that re-reads and re-verifies accepts it. Only a binding to what an
    EARLIER phase saw can tell the difference.
    """
    entry = me.query_artifact_paths(run_dir, room_id, position)
    row = json.load(open(entry["row"]))
    sims = np.load(entry["sims"])

    forged = np.asarray(sims, dtype=np.float16).copy()
    if mutate_scores and forged.shape[0] > 1:
        # move a NON-headline candidate, so the continuity check's slice is
        # untouched and only the ranking sees the difference
        headline = int(row["by_k"][str(max(int(k) for k in row["by_k"]))]["prediction_row"])
        row_index = 0 if headline != 0 else forged.shape[0] - 1
        forged[row_index, :] = np.float16(0.99)
    np.save(entry["sims"], forged)

    # rebuild the row's own claims around the forged sidecar
    largest = str(max(int(k) for k in row["by_k"]))
    coordinates = np.asarray([row["by_k"][largest]["prediction_xyz"]] * forged.shape[0],
                             dtype=np.float64)
    rebuilt = me.score_query(torch.as_tensor(forged.astype(np.float32)),
                             [int(i) for i in row["candidate_indices"]], coordinates,
                             tau=float(row["tau"]),
                             prefixes=tuple(int(k) for k in row["k_prefixes"]))
    row["by_k"] = {str(k): block for k, block in rebuilt["by_k"].items()}
    row["sims_sha256"] = me.file_sha256(entry["sims"])
    row["row_sha256"] = me.row_digest(row)
    me.write_json(entry["row"], row)
    return entry


def test_a_coherent_row_and_sidecar_swap_verifies_against_itself(tmp_path):
    """The exploit is real: the replacement passes a fresh verification."""
    fixture = _probe_fixture(tmp_path)
    entry = _coherent_swap(fixture["run_dir"], "A/A_idx_1", 0)
    verdict = mr.read_verified_query_artifact(entry["row"],
                                              binding_sha256=fixture["binding_sha256"])
    assert verdict["ok"] is True, verdict["reason"]


def test_a_swap_between_the_census_and_the_walk_is_refused(tmp_path):
    """... and binding to the census snapshot is what catches it."""
    fixture = _probe_fixture(tmp_path)
    gate = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    snapshot = gate["artifact_snapshot"]
    assert sorted(snapshot) == sorted(r["query_id"] for r in fixture["records"])

    _coherent_swap(fixture["run_dir"], "A/A_idx_1", 0)
    with pytest.raises(ValueError, match="it was replaced between the two phases"):
        _run(fixture, artifact_snapshot=snapshot)


def test_the_walk_reads_are_bound_to_the_census_snapshot(tmp_path):
    """A spy proves every grid artifact the walk reads is snapshot-bound."""
    fixture = _probe_fixture(tmp_path)
    gate = op.assert_probe_run_census(
        fixture["run_dir"], _published_binding(fixture["run_dir"]),
        fixture["binding_sha256"], fixture["plan"], *_census_args(fixture),
        totals=fixture["totals"])
    snapshot = gate["artifact_snapshot"]

    checked = []
    real = mr.assert_matches_snapshot

    def _spy(query_id, verdict, snap):
        checked.append((query_id, verdict["row_bytes_sha256"], verdict["sims_bytes_sha256"]))
        return real(query_id, verdict, snap)

    mr.assert_matches_snapshot = _spy
    try:
        records = _run(fixture, artifact_snapshot=snapshot)
    finally:
        mr.assert_matches_snapshot = real

    assert len(checked) == len(records)
    for query_id, row_digest, sims_digest in checked:
        assert row_digest == snapshot[query_id]["row_bytes_sha256"]
        assert sims_digest == snapshot[query_id]["sims_bytes_sha256"]


def test_a_query_the_census_never_saw_supplies_no_grid_row(tmp_path):
    fixture = _probe_fixture(tmp_path)
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    query = next(q for q in room.queries if q.position == 0)
    with pytest.raises(ValueError, match="does not cover this query"):
        op.load_grid_row(fixture["run_dir"], query,
                         binding_sha256=fixture["binding_sha256"], snapshot={})


def test_either_half_of_a_swap_is_caught_on_its_own(tmp_path):
    fixture = _probe_fixture(tmp_path)
    rows, _sims, snapshot = mr.verify_rows_with_sidecars(fixture["run_dir"],
                                                         fixture["binding_sha256"])
    room = me.load_room_plan(fixture["plan"], "A/A_idx_1")
    query = next(q for q in room.queries if q.position == 0)
    entry = me.query_artifact_paths(fixture["run_dir"], query.room_id, query.position)
    assert op.load_grid_row(fixture["run_dir"], query,
                            binding_sha256=fixture["binding_sha256"],
                            snapshot=snapshot)["query_id"] == query.query_id

    # the ROW's bytes move (a whitespace-only rewrite keeps every claim, and
    # every self-digest, identical -- only the file's bytes differ)
    row = json.load(open(entry["row"]))
    with open(entry["row"], "w") as handle:
        handle.write(json.dumps(row, sort_keys=True, indent=2) + "\n")
    assert me.verify_query_artifact(entry["row"],
                                    binding_sha256=fixture["binding_sha256"])["ok"] is True
    with pytest.raises(ValueError, match="the row read now hashes to"):
        op.load_grid_row(fixture["run_dir"], query,
                         binding_sha256=fixture["binding_sha256"], snapshot=snapshot)


def test_the_snapshot_note_says_why_self_digests_are_not_enough():
    note = mr.ARTIFACT_SNAPSHOT_NOTE
    assert "digests the row's CLAIMS" in note
    assert "recomputes both" in note
    assert "discards them therefore proves nothing" in note
    assert "every later read is held to it" in note


# --------------------------------------------------------------------------- #
# r9p: the tie's yardstick is the changed-batching envelope, not the fixed one
# --------------------------------------------------------------------------- #
#: The REAL field failure, from the merged P1 run: MeetingRoom_idx_20 S001_R001
#: (query 1389), headline candidate 57 at prediction_row 41. The row is stamped
#: batch_rows=256 / source_chunk=16; the tie regenerates that one candidate at 8
#: generated rows with the source branch at chunk 1, and measured 2.93e-3 against
#: the old 1.24e-3 tolerance while the byte pin and decode-equality both passed.
FIELD_FAILURE_STORED = (0.74609, 0.72266, 0.73242, 0.66602, 0.7334, 0.73682, 0.71436, 0.74268)
FIELD_FAILURE_DELTA = 0.00293


def test_the_real_field_failure_passes_under_the_recalibrated_tolerance():
    stored = np.asarray(FIELD_FAILURE_STORED, dtype=np.float16)
    tolerance = op.observation_continuity_tolerance(stored)
    assert FIELD_FAILURE_DELTA <= tolerance, (FIELD_FAILURE_DELTA, tolerance)
    # ... and it genuinely would have been refused before, which is the bug
    old = float(me.SCORE_TOLERANCE) + mr.float16_half_ulp(stored)
    assert FIELD_FAILURE_DELTA > old
    assert old == pytest.approx(0.00124, abs=1e-5)          # the reported figure
    assert tolerance == pytest.approx(0.00414, abs=1e-5)
    # honest about the margin: the field delta sits at 71% of the new bound, so
    # the recalibration is not generous -- it is the engine's own envelope, and a
    # noticeably larger delta would still refuse
    assert FIELD_FAILURE_DELTA / tolerance == pytest.approx(0.707, abs=0.01)


def test_the_tie_tolerance_is_cross_pinned_to_the_engines_measurement():
    """The literal here must stay the number the engine documents."""
    assert op.CHANGED_BATCHING_TIE_TOLERANCE == op.ENGINE_CHANGED_BATCHING_DRIFT == 3.9e-3
    # the engine states it in prose, so the pin is on that text
    assert "3.9e-3" in me.BATCHING_CAVEAT
    assert "changed batch shape perturbs an output" in me.BATCHING_CAVEAT
    # and it is deliberately NOT the fixed-batching aggregate bound
    assert op.CHANGED_BATCHING_TIE_TOLERANCE > me.SCORE_TOLERANCE
    assert "SCORE_TOLERANCE" in op.TIE_TOLERANCE_NOTE
    assert "PER-SAMPLE" in op.TIE_TOLERANCE_NOTE


def test_the_tolerance_is_the_engine_bound_plus_the_sidecar_half_ulp():
    for values in ((0.74609,) * 8, (-0.5,) * 8, (0.001,) * 8):
        stored = np.asarray(values, dtype=np.float16)
        assert op.observation_continuity_tolerance(stored) == pytest.approx(
            op.CHANGED_BATCHING_TIE_TOLERANCE + mr.float16_half_ulp(stored))


def test_a_substituted_observation_still_refuses_by_a_wide_margin(tmp_path):
    """Detection power survives the recalibration.

    The substitution is a RAMP rather than the fixture's other receiver: the
    fixture's four observations are constant-valued waveforms differing only in
    amplitude, and the synthetic embedder L2-normalizes, so a different
    receiver's RIR embeds IDENTICALLY here and no gate of any tolerance could
    see it. Asserted below rather than assumed, so the day the fixture grows a
    shaped waveform this test starts using it.
    """
    fixture = _probe_fixture(tmp_path)
    items = [(obs, md) for obs, md in fixture["items"]]
    obs, md = items[0]
    donor = next(other for other, meta in items if meta["relpath"] != md["relpath"])
    assert not torch.equal(donor.reshape(-1), obs.reshape(-1))
    assert torch.allclose(fixture["engine"]._embed(donor), fixture["engine"]._embed(obs)), (
        "fixture assumption: constant observations are invisible to this embedder")

    ramp = torch.arange(obs.shape[-1], dtype=obs.dtype).reshape(1, 1, -1) * 0.05
    substituted = obs + ramp
    assert not torch.allclose(fixture["engine"]._embed(substituted),
                              fixture["engine"]._embed(obs)), "the substitution must be visible"
    items[0] = (substituted, md)

    with pytest.raises(ValueError,
                       match="not the observation those rows were scored against") as raised:
        op.run_probe(fixture["engine"], items, fixture["records"], fixture["plan"],
                     fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)

    message = str(raised.value)
    delta = float(re.search(r"max \|delta\| ([0-9.eE+-]+)", message).group(1))
    tolerance = float(re.search(r"tolerance of ([0-9.eE+-]+)", message).group(1))
    assert tolerance == pytest.approx(op.CHANGED_BATCHING_TIE_TOLERANCE, abs=5e-4)
    # WIDE: the substitution has to be an order above the widened bound, not a
    # near miss, or the recalibration would have bought the refusal's silence.
    # Measured on this fixture: delta 0.0478 against tolerance 0.00414 (11.5x),
    # and 16x the real field tie of 2.93e-3. On the real run the separation is
    # wider still -- query 1389's cosines span -0.2937..0.7461, i.e. 25-250x the
    # tie -- but the synthetic embedder's weaker contrast is the floor asserted.
    assert delta > 10 * tolerance, (delta, tolerance)
    assert delta > 10 * FIELD_FAILURE_DELTA, (delta, FIELD_FAILURE_DELTA)


def test_the_honest_tie_and_the_substituted_tie_are_orders_apart(tmp_path):
    """The separation the recalibration relies on, measured on one fixture."""
    fixture = _probe_fixture(tmp_path)
    honest = _run(fixture)[0]["observation_continuity"]
    assert honest["within_tolerance"] is True and honest["refused"] is False

    items = [(obs, md) for obs, md in fixture["items"]]
    obs, md = items[0]
    ramp = torch.arange(obs.shape[-1], dtype=obs.dtype).reshape(1, 1, -1) * 0.05
    items[0] = (obs + ramp, md)
    with pytest.raises(ValueError) as raised:
        op.run_probe(fixture["engine"], items, fixture["records"], fixture["plan"],
                     fixture["run_dir"], fixture["out_dir"],
                     metadata_root=fixture["metadata_root"],
                     binding_sha256=fixture["binding_sha256"], tau=fx.FIXTURE_TAU,
                     num_samples=fx.FIXTURE_SAMPLES, prefixes=fx.FIXTURE_PREFIXES)
    cheating = float(re.search(r"max \|delta\| ([0-9.eE+-]+)", str(raised.value)).group(1))
    # measured: honest 2.14e-4, substituted 0.0478 -- 223x apart, and the widened
    # tolerance sits between them with an order of headroom on either side
    assert cheating > 100 * honest["max_abs_delta"]
    assert cheating > 10 * honest["tolerance"]
    assert honest["max_abs_delta"] < honest["tolerance"] < cheating
    # the published span is the readers' version of this same comparison
    assert honest["query_cosine_span"] > 0.0
    assert honest["separation_vs_span"] > 1.0


def test_every_artifact_publishes_the_measured_tie_delta(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate={"single_shard": False,
              "metadata_bank_expected": fixture["metadata_bank_sha256"],
              "observation_bank_expected": fixture["observation_bank_sha256"],
              "registered_protocol": {"is_registered": True, "deviations": {}}})

    payload = json.load(open(published["json"]))
    assert payload["tie_tolerance_note"] == op.TIE_TOLERANCE_NOTE
    for record in payload["records"]:
        tie = record["observation_continuity"]
        for field in ("max_abs_delta", "tolerance", "headroom", "within_tolerance",
                      "refused", "query_cosine_span", "separation_vs_span",
                      "changed_batching_component"):
            assert field in tie, field
        assert tie["within_tolerance"] is True and tie["refused"] is False
        assert tie["changed_batching_component"] == op.CHANGED_BATCHING_TIE_TOLERANCE

    markdown = open(published["markdown"]).read()
    assert "Observation-continuity tie — measured delta vs tolerance" in markdown
    assert "query cosine span" in markdown
    for record in payload["records"]:
        assert mr.format_number(record["observation_continuity"]["max_abs_delta"], 6) in markdown

    for record in payload["records"]:
        with np.load(os.path.join(fixture["out_dir"], record["waveform_path"])) as data:
            assert float(data["tie_max_abs_delta"]) == pytest.approx(
                record["observation_continuity"]["max_abs_delta"])
            assert float(data["tie_tolerance"]) == pytest.approx(
                record["observation_continuity"]["tolerance"])
            assert bool(data["tie_within_tolerance"]) is True
            assert bool(data["tie_refused"]) is False
            assert str(data["tie_tolerance_note"]) == op.TIE_TOLERANCE_NOTE


def test_the_tie_calls_the_source_branch_on_a_single_position(tmp_path):
    """The diagnosis behind the recalibration, pinned rather than asserted in prose.

    The run's source branch is called on the receiver's whole candidate union in
    chunks (``source_chunk`` positions per forward); the tie calls it on ONE
    position, so the two are a batch-1-versus-many pair no matter what chunk is
    passed. That is the difference the widened tolerance pays for.
    """
    fixture = _probe_fixture(tmp_path)
    real = me.source_conditioning
    shapes = []

    def _spy(conditioner, md, positions_cam, device, **kwargs):
        shapes.append((np.asarray(positions_cam).shape, kwargs.get("chunk")))
        return real(conditioner, md, positions_cam, device, **kwargs)

    me.source_conditioning = _spy
    try:
        _run(fixture)
    finally:
        me.source_conditioning = real

    assert shapes, "the tie never reached the source branch"
    assert all(shape[0] == 1 for shape, _chunk in shapes), shapes
    # ... while the rows being checked were produced in wider batches, which is
    # the whole asymmetry (the real run: source_chunk 16, batch_rows 256)
    row = op.load_grid_row(fixture["run_dir"],
                           next(q for q in me.load_room_plan(
                               fixture["plan"], fixture["records"][0]["room_id"]).queries
                               if q.query_id == fixture["records"][0]["query_id"]),
                           binding_sha256=fixture["binding_sha256"])
    # the fixture is small (batch_rows == num_samples, so one candidate per
    # forward), but its source chunk is already wider than the tie's single call
    assert int(row["batching"]["source_chunk"]) > 1
    assert int(row["batching"]["source_chunk"]) > max(
        shape[0] for shape, _chunk in shapes)


def test_the_tie_table_is_a_well_formed_markdown_table(tmp_path):
    """A raw ``|`` in a header (e.g. "max |delta|") silently splits the cell."""
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    published = op.write_probe_report(
        fixture["out_dir"], records, fixture["binding"], fixture["binding_sha256"],
        provenance={}, tau=fx.FIXTURE_TAU, prefixes=fx.FIXTURE_PREFIXES,
        gate={"single_shard": False,
              "metadata_bank_expected": fixture["metadata_bank_sha256"],
              "observation_bank_expected": fixture["observation_bank_sha256"],
              "registered_protocol": {"is_registered": True, "deviations": {}}})
    markdown = open(published["markdown"]).read().split("\n")
    start = markdown.index("## Observation-continuity tie — measured delta vs tolerance")
    table = [line for line in markdown[start:start + 6 + len(records)]
             if line.startswith("|")]
    assert len(table) == 2 + len(records)                 # header, rule, one per query
    widths = {line.count("|") for line in table}
    assert len(widths) == 1, table
    assert widths == {9}                                  # 8 columns


def test_the_run_summary_reports_the_spread_not_just_a_flag(tmp_path):
    fixture = _probe_fixture(tmp_path)
    records = _run(fixture)
    summary = records[0]["observation_continuity_summary"]
    assert summary["ok"] is True
    assert summary["checked"] == len(records)
    assert summary["min_headroom"] > 0.0
    assert summary["changed_batching_component"] == op.CHANGED_BATCHING_TIE_TOLERANCE
    assert sorted(summary["per_query_delta"]) == sorted(r["query_id"] for r in records)
    assert summary["min_separation_vs_span"] > 0.0
