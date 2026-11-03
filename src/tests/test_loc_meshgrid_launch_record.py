"""exp_22 R2-A -- launch records as per-invocation SEGMENTS (plan r2 §3 R2-A, Codex B4).

The off-grid probe already owns a reviewed launch-record mechanism -- exact argv,
full git SHA, tracked-clean tree, hostname and the CUDA runtime's own UUID for the
designated card, all compared against the executing machine at admission. The
full pass reuses it unchanged; what is new is the SEGMENT model around it:

* a record is PER INVOCATION, so a shard that is resumed has one segment per
  invocation and every row names the segment that produced it;
* the run binding stays common across shards (the merge's equal-digest rule), so
  provenance lives beside it, never inside it;
* a merge requires every row to resolve to a present, hash-verified,
  environment-verified record, and carries all of them across with their row
  ranges; the report re-derives that census and a gap makes it non-canonical.

Everything here runs on CPU: the environment and the CUDA runtime are injected
through the probe module's own seams, exactly as its tests do.
"""
import hashlib
import json
import os

import pytest

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_offgrid_probe as op
from src.localization import meshgrid_segments as ms

import test_loc_meshgrid_offgrid_probe as OPT    # pytest prepends src/tests


ARGV = ["--ckpt-path", "weights/exp20/YAW_40k.ckpt", "--cond-method", "vanilla",
        "--frame-avg-angles", "0", "--rotate-deg", "0", "--cond-autocast", "default",
        "--batch-rows", "256", "--source-chunk", "16", "--rooms", "Cafe/Cafe_idx_1",
        "--device", "cuda:0", "--out-dir", "out"]


def emit(tmp_path, argv=ARGV, name="launch.json", device="cuda:0", environment=None,
         **record_overrides):
    """A record exactly as ``--emit-launch-record`` writes one, on an injected machine."""
    logical = int(device.split(":")[-1])
    record = op.build_launch_record(
        list(argv) + ["--emit-launch-record", name], device=device,
        environment=OPT._environment() if environment is None else environment,
        runtime=OPT._runtime(OPT._FIXTURE_GPUS[logical]["uuid"], logical))
    record.update(record_overrides)
    path = str(tmp_path / name)
    me.write_json(path, record)
    return path


def admit(path, argv=ARGV, device="cuda:0", environment=None):
    """The probe module's verified admission, against the same injected machine."""
    return OPT._verify(path, list(argv) + ["--launch-record", path], device=device,
                       environment=environment)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_position_ranges_are_runs_of_consecutive_positions():
    assert ms.position_ranges([]) == []
    assert ms.position_ranges([4]) == [[4, 4]]
    assert ms.position_ranges([9, 0, 1, 2, 5, 6]) == [[0, 2], [5, 6], [9, 9]]
    with pytest.raises(ValueError, match="twice"):
        ms.position_ranges([1, 1])


def test_the_probe_mechanism_is_reused_not_copied():
    assert ms.ENVIRONMENT_AXES is op.ENVIRONMENT_AXES
    assert ms.LAUNCH_RECORD_FIELDS is op.LAUNCH_RECORD_FIELDS
    assert ms.read_verified_launch_record is op.read_verified_launch_record
    assert ms.build_launch_record is op.build_launch_record


# --------------------------------------------------------------------------- #
# opening a segment
# --------------------------------------------------------------------------- #
def test_opening_a_segment_copies_the_verified_record_and_appends_it(tmp_path):
    path = emit(tmp_path)
    admission = admit(path)
    out = str(tmp_path / "shard")
    entry = ms.open_segment(out, path, admission, resume=False)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert entry["launch_record_sha256"] == digest == admission["sha256"]
    assert entry["segment_index"] == 0 and entry["resume"] is False
    copied = os.path.join(out, entry["record_file"])
    assert entry["record_file"] == os.path.join(ms.RECORDS_DIRNAME, f"{digest}.json")
    assert open(copied, "rb").read() == open(path, "rb").read()   # the BYTES, not a re-dump
    ledger = ms.read_ledger(out)
    assert [s["launch_record_sha256"] for s in ledger["segments"]] == [digest]
    assert ledger["segments"][0]["admission"]["environment_verified"] is True
    assert ledger["segments"][0]["admission"]["execution_gpu_uuid"] == \
        OPT._FIXTURE_GPUS[0]["uuid"]
    assert ms.ledger_shas(out) == [digest]


def test_a_record_that_changed_since_admission_is_refused(tmp_path):
    path = emit(tmp_path)
    admission = admit(path)
    record = json.load(open(path))
    record["hostname"] = "elsewhere"
    me.write_json(path, record)
    with pytest.raises(ValueError, match="changed since it was admitted"):
        ms.open_segment(str(tmp_path / "shard"), path, admission, resume=False)
    assert ms.read_ledger(str(tmp_path / "shard")) is None


def test_an_unverified_admission_cannot_open_a_segment(tmp_path):
    path = emit(tmp_path)
    admission = dict(admit(path), environment_verified=False)
    with pytest.raises(ValueError, match="environment"):
        ms.open_segment(str(tmp_path / "shard"), path, admission, resume=False)


def test_a_record_vouches_for_exactly_one_invocation(tmp_path):
    path = emit(tmp_path)
    out = str(tmp_path / "shard")
    ms.open_segment(out, path, admit(path), resume=False)
    with pytest.raises(ValueError, match="one invocation"):
        ms.open_segment(out, path, admit(path), resume=True)


def test_a_resume_opens_a_new_segment_under_a_new_record(tmp_path):
    out = str(tmp_path / "shard")
    first = emit(tmp_path, name="first.json")
    ms.open_segment(out, first, admit(first), resume=False)
    resumed_argv = ARGV + ["--resume"]
    second = emit(tmp_path, argv=resumed_argv, name="second.json")
    entry = ms.open_segment(out, second, admit(second, argv=resumed_argv), resume=True)
    assert entry["segment_index"] == 1 and entry["resume"] is True
    assert len(ms.ledger_shas(out)) == 2
    # the original record does not describe the resumed command (argv differs)
    with pytest.raises(ValueError, match="different command"):
        admit(first, argv=resumed_argv)


def test_closing_a_segment_records_what_it_produced(tmp_path):
    path = emit(tmp_path)
    out = str(tmp_path / "shard")
    entry = ms.open_segment(out, path, admit(path), resume=False)
    ms.close_segment(out, entry["launch_record_sha256"], {"n_scored": 3, "n_skipped": 1},
                     at_utc="2026-11-06T00:00:00+00:00")
    closed = ms.read_ledger(out)["segments"][0]
    assert closed["closed_utc"] == "2026-11-06T00:00:00+00:00"
    assert (closed["n_scored"], closed["n_skipped"]) == (3, 1)
    with pytest.raises(ValueError, match="no open segment"):
        ms.close_segment(out, "0" * 64, {"n_scored": 0})


# --------------------------------------------------------------------------- #
# verifying a segment after the fact
# --------------------------------------------------------------------------- #
def _opened(tmp_path):
    path = emit(tmp_path)
    out = str(tmp_path / "shard")
    entry = ms.open_segment(out, path, admit(path), resume=False)
    return out, entry


def test_a_healthy_segment_verifies(tmp_path):
    out, entry = _opened(tmp_path)
    verdict = ms.verify_segment(out, entry)
    assert verdict["ok"] is True and verdict["reasons"] == []
    assert verdict["facts"]["git_sha"] == "a" * 40
    assert verdict["facts"]["execution_gpu_uuid"] == OPT._FIXTURE_GPUS[0]["uuid"]


@pytest.mark.parametrize("damage, match", [
    ("missing", "missing"),
    ("tampered", "does not hash"),
    ("relocated", "not the record path"),
    ("not_verified", "environment_verified"),
    ("axis_failed", "gpu_uuid"),
    ("axes_short", "axes"),
    ("sha_mismatch", "admission names"),
    ("uuid_mismatch", "execution_gpu_uuid"),
])
def test_every_way_a_segment_can_fail_is_named(tmp_path, damage, match):
    out, entry = _opened(tmp_path)
    copied = os.path.join(out, entry["record_file"])
    entry = json.loads(json.dumps(entry))
    if damage == "missing":
        os.remove(copied)
    elif damage == "tampered":
        record = json.load(open(copied))
        record["hostname"] = "elsewhere"
        me.write_json(copied, record)
    elif damage == "relocated":
        entry["record_file"] = "elsewhere.json"
    elif damage == "not_verified":
        entry["admission"]["environment_verified"] = False
    elif damage == "axis_failed":
        entry["admission"]["environment_comparison"][-1]["verdict"] = "fail"
    elif damage == "axes_short":
        entry["admission"]["environment_comparison"] = \
            entry["admission"]["environment_comparison"][:2]
    elif damage == "sha_mismatch":
        entry["admission"]["sha256"] = "0" * 64
    elif damage == "uuid_mismatch":
        entry["admission"]["execution_gpu_uuid"] = OPT._FIXTURE_GPUS[1]["uuid"]
    verdict = ms.verify_segment(out, entry)
    assert verdict["ok"] is False
    assert any(match in reason for reason in verdict["reasons"]), verdict["reasons"]


def test_a_record_written_from_a_dirty_tree_never_verifies(tmp_path):
    path = emit(tmp_path)
    out = str(tmp_path / "shard")
    entry = ms.open_segment(out, path, admit(path), resume=False)
    # forge a dirty record under the SAME name with a matching admission digest
    copied = os.path.join(out, entry["record_file"])
    record = json.load(open(copied))
    record["git_status_dirty"] = True
    me.write_json(copied, record)
    entry = json.loads(json.dumps(entry))
    entry["launch_record_sha256"] = entry["admission"]["sha256"] = me.file_sha256(copied)
    entry["record_file"] = ms.record_relpath(entry["launch_record_sha256"])
    os.replace(copied, os.path.join(out, entry["record_file"]))
    verdict = ms.verify_segment(out, entry)
    assert verdict["ok"] is False
    assert any("tracked-dirty" in reason for reason in verdict["reasons"])


# --------------------------------------------------------------------------- #
# the census: every row resolves, or the reasons say which do not
# --------------------------------------------------------------------------- #
def _rows(sha, positions):
    return [{"query_id": f"{p}|q", "position": p, "launch_record_sha256": sha}
            for p in positions]


def test_a_complete_census_publishes_every_record_with_its_row_ranges(tmp_path):
    out = str(tmp_path / "shard")
    first = emit(tmp_path, name="first.json")
    one = ms.open_segment(out, first, admit(first), resume=False)
    resumed = ARGV + ["--resume"]
    second = emit(tmp_path, argv=resumed, name="second.json")
    two = ms.open_segment(out, second, admit(second, argv=resumed), resume=True)
    rows = (_rows(one["launch_record_sha256"], [0, 1, 2, 7])
            + _rows(two["launch_record_sha256"], [3, 4, 5, 6]))
    census = ms.provenance_census(out, rows)
    assert census["complete"] is True and census["reasons"] == []
    assert census["generation"] == "launch_records"
    by_sha = {record["launch_record_sha256"]: record for record in census["records"]}
    assert by_sha[one["launch_record_sha256"]]["position_ranges"] == [[0, 2], [7, 7]]
    assert by_sha[two["launch_record_sha256"]]["position_ranges"] == [[3, 6]]
    assert by_sha[two["launch_record_sha256"]]["n_rows"] == 4
    assert by_sha[two["launch_record_sha256"]]["resume"] is True
    assert census["git_shas"] == ["a" * 40]


def test_a_row_without_a_record_or_with_an_unknown_one_breaks_the_census(tmp_path):
    out, entry = _opened(tmp_path)
    sha = entry["launch_record_sha256"]
    unstamped = _rows(sha, [0, 1]) + [{"query_id": "2|q", "position": 2}]
    census = ms.provenance_census(out, unstamped)
    assert census["complete"] is False and census["n_unstamped"] == 1
    assert any("no launch record" in reason for reason in census["reasons"])
    unknown = _rows(sha, [0, 1]) + _rows("f" * 64, [2])
    census = ms.provenance_census(out, unknown)
    assert census["complete"] is False and census["n_unknown"] == 1
    assert any("unknown" in reason for reason in census["reasons"])


def test_a_run_with_no_ledger_has_no_launch_provenance(tmp_path):
    census = ms.provenance_census(str(tmp_path / "legacy"), [{"query_id": "0|q",
                                                              "position": 0}])
    assert census["generation"] == "none" and census["complete"] is False
    assert census["records"] == []


def test_a_segment_that_fails_verification_breaks_the_census(tmp_path):
    out, entry = _opened(tmp_path)
    os.remove(os.path.join(out, entry["record_file"]))
    census = ms.provenance_census(out, _rows(entry["launch_record_sha256"], [0]))
    assert census["complete"] is False
    assert any("missing" in reason for reason in census["reasons"])


def test_a_ledgers_row_range_claim_must_match_the_rows(tmp_path):
    out, entry = _opened(tmp_path)
    ledger = ms.read_ledger(out)
    ledger["segments"][0]["n_rows"] = 5
    ledger["segments"][0]["position_ranges"] = [[0, 4]]
    me.write_json(os.path.join(out, ms.SEGMENTS_FILENAME), ledger)
    census = ms.provenance_census(out, _rows(entry["launch_record_sha256"], [0, 1]))
    assert census["complete"] is False
    assert any("claims" in reason for reason in census["reasons"])


# --------------------------------------------------------------------------- #
# carrying segments across a merge
# --------------------------------------------------------------------------- #
def test_a_merge_carries_every_record_and_its_row_ranges_across(tmp_path):
    shards = {}
    for name, positions in (("cafe", [0, 1]), ("rest", [2, 3, 4])):
        argv = [token if token != "out" else name for token in ARGV]
        path = emit(tmp_path, argv=argv, name=f"{name}.json")
        out = str(tmp_path / name)
        entry = ms.open_segment(out, path, admit(path, argv=argv), resume=False)
        shards[out] = _rows(entry["launch_record_sha256"], positions)
    merged = str(tmp_path / "merged")
    census = ms.merge_ledgers(shards, merged)
    assert census["complete"] is True and len(census["records"]) == 2
    ledger = ms.read_ledger(merged)
    assert sorted(segment["shard"] for segment in ledger["segments"]) == sorted(shards)
    for segment in ledger["segments"]:
        assert os.path.isfile(os.path.join(merged, segment["record_file"]))
    rows = [row for rows in shards.values() for row in rows]
    again = ms.provenance_census(merged, rows)
    assert again["complete"] is True
    assert sorted(r["position_ranges"] for r in again["records"]) == [[[0, 1]], [[2, 4]]]


def test_a_merge_refuses_a_shard_whose_census_is_incomplete(tmp_path):
    path = emit(tmp_path)
    out = str(tmp_path / "shard")
    entry = ms.open_segment(out, path, admit(path), resume=False)
    rows = _rows(entry["launch_record_sha256"], [0]) + [{"query_id": "1|q", "position": 1}]
    with pytest.raises(ValueError, match="launch provenance"):
        ms.merge_ledgers({out: rows}, str(tmp_path / "merged"))


# --------------------------------------------------------------------------- #
# the engine: rows name their segment, resume accounting, the merge gate
# --------------------------------------------------------------------------- #
import test_loc_meshgrid_engine as ENG                        # noqa: E402


def _segment(tmp_path, out, name, argv=ARGV, resume=False):
    path = emit(tmp_path, argv=argv, name=name)
    return ms.open_segment(out, path, admit(path, argv=argv), resume=resume)


def _score(out, items, records, plan, binding, rooms=None, sha=None, done=()):
    summary = me.run_pass(ENG.SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8, source_chunk=3, rooms=rooms,
                          binding_sha256=me.binding_sha256(binding), done=done,
                          launch_record_sha256=sha)
    me.write_json(os.path.join(out, "run_summary.json"), summary)
    return summary


def test_every_row_names_the_record_of_the_invocation_that_produced_it(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    out = str(tmp_path / "run")
    me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3})
    entry = _segment(tmp_path, out, "one.json")
    summary = _score(out, items, records, plan, binding, sha=entry["launch_record_sha256"])
    assert summary["launch_record_sha256"] == entry["launch_record_sha256"]
    rows = me.read_rows(out)
    assert {row["launch_record_sha256"] for row in rows} == {entry["launch_record_sha256"]}
    # the stamp is a CLAIM of the row, so the row digest authenticates it
    path = me.query_artifact_paths(out, rows[0]["room_id"], rows[0]["position"])["row"]
    me.write_json(path, dict(rows[0], launch_record_sha256="f" * 64))
    assert me.verify_query_artifact(path)["ok"] is False


def test_a_pass_without_a_record_writes_exactly_the_rows_it_always_did(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    out = str(tmp_path / "run")
    _score(out, items, records, plan, ENG._binding())
    assert all("launch_record_sha256" not in row for row in me.read_rows(out))


def test_a_resume_is_a_new_segment_and_rows_are_attributed_to_their_own(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    out = str(tmp_path / "run")
    me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3})
    first = _segment(tmp_path, out, "one.json")
    _score(out, items, records, plan, binding, sha=first["launch_record_sha256"])
    # the first invocation "crashed" before publishing positions 1 and 3
    for room, position in (("A/A_idx_1", 1), ("B/B_idx_2", 3)):
        os.remove(me.query_artifact_paths(out, room, position)["row"])
    resumed = ARGV + ["--resume"]
    second = _segment(tmp_path, out, "two.json", argv=resumed, resume=True)
    done, rejected = me.completed_queries(out, binding_sha256=me.binding_sha256(binding),
                                          launch_records=set(ms.ledger_shas(out)))
    assert len(done) == 2 and rejected == []
    _score(out, items, records, plan, binding, sha=second["launch_record_sha256"], done=done)
    census = ms.provenance_census(out, me.read_rows(out))
    assert census["complete"] is True
    ranges = {r["launch_record_sha256"]: r["position_ranges"] for r in census["records"]}
    assert ranges == {first["launch_record_sha256"]: [[0, 0], [2, 2]],
                      second["launch_record_sha256"]: [[1, 1], [3, 3]]}


def test_a_resume_regenerates_a_row_whose_record_this_run_does_not_know(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    out = str(tmp_path / "run")
    me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3})
    entry = _segment(tmp_path, out, "one.json")
    _score(out, items, records, plan, binding, sha=entry["launch_record_sha256"])
    # a row smuggled in under a record this run never admitted
    _score(str(tmp_path / "elsewhere"), items, records, plan, binding, sha="e" * 64)
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)
    foreign = me.query_artifact_paths(str(tmp_path / "elsewhere"), "A/A_idx_1", 0)
    for key in ("row", "sims"):
        with open(foreign[key], "rb") as source, open(path[key], "wb") as target:
            target.write(source.read())
    done, rejected = me.completed_queries(out, binding_sha256=me.binding_sha256(binding),
                                          launch_records=set(ms.ledger_shas(out)))
    assert len(done) == 3
    assert [verdict["query_id"] for verdict in rejected] == [records[0]["query_id"]]
    assert "launch record" in rejected[0]["reason"]


def _r2_shards(tmp_path, stamp=True):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    dirs = {}
    for name, rooms in (("a", ["A/A_idx_1"]), ("b", ["B/B_idx_2"])):
        out = str(tmp_path / name)
        me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3},
                         declared_rooms=rooms)
        argv = [token if token != "out" else name for token in ARGV]
        sha = _segment(tmp_path, out, f"{name}.json", argv=argv)["launch_record_sha256"]
        _score(out, items, records, plan, binding, rooms=rooms, sha=sha if stamp else None)
        dirs[name] = out
    pairs, unions = ENG._fixture_totals()
    totals = {"rooms": 2, "queries": len(records), "candidate_query_pairs": pairs,
              "source_rows": unions, "generated_waveforms": pairs * 4}
    return plan, records, dirs, totals, binding


def test_an_r2_merge_carries_every_record_and_publishes_the_census(tmp_path):
    plan, records, dirs, totals, binding = _r2_shards(tmp_path)
    merged = str(tmp_path / "merged")
    report = me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    provenance = report["launch_provenance"]
    assert provenance["complete"] is True and len(provenance["records"]) == 2
    assert sorted(r["position_ranges"] for r in provenance["records"]) == [
        [[0, 2]], [[3, 3]]]
    assert ms.provenance_census(merged, me.read_rows(merged))["complete"] is True
    published = json.load(open(os.path.join(merged, me.BINDING_FILENAME)))
    assert published["binding_sha256"] == me.binding_sha256(binding)
    assert published["frame_avg_angles"] == [0.0] and "fa_partition" in published


def test_an_r2_merge_refuses_a_row_without_a_record(tmp_path):
    plan, records, dirs, totals, _binding = _r2_shards(tmp_path, stamp=False)
    with pytest.raises(ValueError, match="launch provenance"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)
    assert not os.path.exists(str(tmp_path / "m"))


def test_an_r2_merge_refuses_a_row_naming_an_unknown_record(tmp_path):
    plan, records, dirs, totals, _binding = _r2_shards(tmp_path)
    os.remove(os.path.join(dirs["b"], ms.SEGMENTS_FILENAME))
    with pytest.raises(ValueError, match="launch provenance"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)


def test_a_legacy_merge_states_that_it_predates_launch_records(tmp_path):
    plan, records, dirs, totals = ENG._shards(tmp_path)
    report = me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "merged"), plan, records,
                             totals=totals)
    assert report["launch_provenance"]["generation"] == "legacy_r1_binding"
    assert report["launch_provenance"]["complete"] is False
    assert not os.path.isfile(os.path.join(str(tmp_path / "merged"), ms.SEGMENTS_FILENAME))


# --------------------------------------------------------------------------- #
# the driver: emit, admit, fail-fast preflight, segments around the pass
# --------------------------------------------------------------------------- #
import localize_meshgrid as driver                             # noqa: E402
from src.localization import meshgrid_report as mr           # noqa: E402

MODEL_CONFIG = os.path.join("src", "configs", "model_configs", "FLAC", "AR", "FLAC_AR.json")
FA_TRAINING = {"cond_method": "fa_invariant", "frame_avg_angles": [0.0, 90.0, 180.0, 270.0]}
YAW_TRAINING = {"yaw_aug": {"enabled": True, "img_w": 512, "seed": 42}}


def _args(tmp_path, *extra, protocol=None, record=True, validate=True):
    out_dir, report_path, _base = ENG._fixture_audit(tmp_path)
    manifest = tmp_path / "d1.json"
    manifest.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    argv = (["--ckpt-path", "x.ckpt", "--audit-report", report_path,
             "--context-manifest", str(manifest), "--out-dir", str(tmp_path / "run"),
             "--device", "cuda:0"]
            + list(protocol if protocol is not None else ENG._VANILLA_PROTOCOL) + list(extra)
            + (["--launch-record", str(tmp_path / "launch.json")] if record else []))
    args = driver.parse_args(argv)
    if validate:            # a record-less FULL pass is refused by design; its argv is
        driver.validate_args(args)       # what an emission describes
    return args, argv, me.load_audit_plan(report_path)


def _preflight(args, plan, arm, training):
    model_config = json.load(open(MODEL_CONFIG))
    return driver.preflight(args, plan, model_config, {"model_config": {"training": training}},
                            ckpt_sha256=mr.REGISTERED_CKPT_SHA256[arm],
                            agree_sha256=mr.REGISTERED_ARTIFACT_SHA256["agree_ckpt_sha256"],
                            model_config_sha256=mr.REGISTERED_ARTIFACT_SHA256[
                                "model_config_sha256"])


def test_validate_args_no_longer_preempts_the_fa_refusal(tmp_path):
    args, _argv, _plan = _args(tmp_path, protocol=ENG._FA_PROTOCOL)
    assert args.cond_method == "fa_invariant"          # reached here: no early refusal


def test_fa_flags_reach_the_engine_refusal_only_after_every_binding_check(tmp_path,
                                                                           monkeypatch):
    """The ordering the plan registers: binding checks FIRST, engine refusal SECOND."""
    calls = []
    for name, module in (("assert_registered_protocol", mr),
                         ("checkpoint_protocol_reasons", mr), ("assert_cacheable", me)):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, (lambda real, name: lambda *a, **k: (
            calls.append(name), real(*a, **k))[1])(real, name))
    real_validate = driver.validate_checkpoint
    monkeypatch.setattr(driver, "validate_checkpoint", lambda *a, **k: (
        calls.append("validate_checkpoint"), real_validate(*a, **k))[1])

    args, _argv, plan = _args(tmp_path, protocol=ENG._FA_PROTOCOL)
    # (a) a BF checkpoint trained fa, the registered orbit and partition: every
    # binding check passes, and only then does the engine refuse fa
    with pytest.raises(ValueError, match="fa_invariant"):
        _preflight(args, plan, "BF", FA_TRAINING)
    assert calls == ["assert_registered_protocol", "checkpoint_protocol_reasons",
                     "validate_checkpoint", "assert_cacheable"]
    # (b) the same flags against the P1 checkpoint: the ARM check names the
    # mismatch and the engine is never asked
    calls.clear()
    with pytest.raises(SystemExit, match="arm/protocol"):
        _preflight(args, plan, "P1", {})
    assert "assert_cacheable" not in calls
    # (c) a BF digest whose file was trained vanilla: the embedded protocol refuses
    calls.clear()
    with pytest.raises(SystemExit, match="embedded training protocol"):
        _preflight(args, plan, "BF", {})
    assert "assert_cacheable" not in calls


def test_a_canonical_vanilla_pass_is_held_to_its_arm_before_any_device_work(tmp_path):
    args, _argv, plan = _args(tmp_path)
    binding, verdict = _preflight(args, plan, "YAW", YAW_TRAINING)
    assert verdict["registered"]["arm"] == "YAW" and verdict["registered"]["is_registered"]
    assert binding["frame_avg_angles"] == [0.0] and binding["fa_partition"] is None
    # a vanilla canonical run under the BF checkpoint is the catastrophic mismatch
    with pytest.raises(SystemExit, match="arm/protocol"):
        _preflight(args, plan, "BF", FA_TRAINING)
    # --expect-arm is held too
    args, _argv, plan = _args(tmp_path, "--expect-arm", "P1")
    with pytest.raises(SystemExit, match="--expect-arm"):
        _preflight(args, plan, "YAW", YAW_TRAINING)


def test_a_diagnostic_may_deviate_but_never_against_its_own_file(tmp_path):
    args, _argv, plan = _args(tmp_path, "--probe", "1", record=False)
    # a vanilla probe of the BF digest passes the registry as a stamped deviation
    # but the file itself was trained fa, so the embedded protocol still refuses
    with pytest.raises(SystemExit, match="embedded training protocol"):
        _preflight(args, plan, "BF", FA_TRAINING)
    _binding, verdict = _preflight(args, plan, "P1", {})
    assert verdict["registered"]["arm"] == "P1"


@pytest.mark.parametrize("broken, axis", [
    ({"git_sha": "b" * 40, "git_sha_capture": OPT._ok("b" * 40)}, "git_sha"),
    ({"git_status_dirty": True, "git_status_capture": OPT._ok(" M src/x.py")},
     "git_tracked_clean"),
    ({"hostname": "elsewhere", "hostname_capture": OPT._ok("elsewhere")}, "hostname"),
])
def test_the_driver_refuses_a_record_that_fails_any_environment_axis(tmp_path, broken, axis):
    args, argv, _plan = _args(tmp_path)
    emit(tmp_path, argv=[t for t in argv if t not in ("--launch-record",
                                                      str(tmp_path / "launch.json"))])
    with pytest.raises(SystemExit, match=axis):
        driver.admit_launch_record(args, argv, environment=OPT._environment(**broken),
                                   runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0))
    with pytest.raises(SystemExit, match="gpu_uuid"):
        driver.admit_launch_record(args, argv, environment=OPT._environment(),
                                   runtime=OPT._runtime(OPT._FIXTURE_GPUS[1]["uuid"], 0))
    admitted = driver.admit_launch_record(
        args, argv, environment=OPT._environment(),
        runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0))
    assert admitted["environment_verified"] is True


def test_the_driver_emits_the_record_for_the_command_it_will_run(tmp_path):
    args, argv, _plan = _args(tmp_path, record=False, validate=False)
    path = str(tmp_path / "emitted.json")
    emit_argv = argv + ["--emit-launch-record", path]
    emit_args = driver.parse_args(emit_argv)
    driver.validate_args(emit_args)
    assert driver.emit_launch_record(emit_args, emit_argv, environment=OPT._environment(),
                                     runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0)) == 0
    record = json.load(open(path))
    assert record["argv"] == argv                       # the RUN's argv, flag pair stripped
    assert record["execution_gpu_uuid"] == OPT._FIXTURE_GPUS[0]["uuid"]
    run_argv = argv + ["--launch-record", path]
    run_args = driver.parse_args(run_argv)
    assert driver.admit_launch_record(
        run_args, run_argv, environment=OPT._environment(),
        runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0))["sha256"] == \
        me.file_sha256(path)
    # emission that cannot state a fact is refused, never written
    with pytest.raises(SystemExit, match="cannot be written"):
        driver.emit_launch_record(emit_args, emit_argv, environment=OPT._environment(),
                                  runtime=OPT._runtime_failed())


def _claim(tmp_path, argv_extra=(), name="launch.json"):
    args, argv, plan = _args(tmp_path, *argv_extra, record=False, validate=False)
    path = emit(tmp_path, argv=argv, name=name)
    run_argv = argv + ["--launch-record", path]
    run_args = driver.parse_args(run_argv)
    driver.validate_args(run_args)
    launch = driver.admit_launch_record(run_args, run_argv, environment=OPT._environment(),
                                        runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0))
    binding = driver.build_run_binding(run_args, plan, ckpt_sha256="1" * 64,
                                       agree_sha256="2" * 64, model_config_sha256="3" * 64)
    return run_args, binding, launch


def test_claiming_the_out_dir_opens_one_segment_per_invocation(tmp_path):
    advisory = {"source_chunk": 16, "batch_rows": 256}
    args, binding, launch = _claim(tmp_path)
    done, segment = driver.claim_out_dir(args, binding, advisory, launch)
    assert done == set() and segment["segment_index"] == 0 and segment["resume"] is False
    assert ms.ledger_shas(args.out_dir) == [launch["sha256"]]
    # the same record cannot vouch for a second invocation
    with pytest.raises(ValueError, match="one invocation"):
        driver.claim_out_dir(args, binding, advisory, launch)
    # a resume is a new invocation under a new record: a new segment
    resumed, binding2, launch2 = _claim(tmp_path, ("--resume",), name="resume.json")
    done, segment = driver.claim_out_dir(resumed, binding2, advisory, launch2)
    assert segment["segment_index"] == 1 and segment["resume"] is True
    assert ms.ledger_shas(args.out_dir) == [launch["sha256"], launch2["sha256"]]


def test_a_diagnostic_never_claims_the_out_dir_or_opens_a_segment(tmp_path):
    args, _argv, _plan = _args(tmp_path, "--probe", "1", record=False)
    assert driver.writes_query_artifacts(args) is False
    assert not os.path.exists(os.path.join(args.out_dir, ms.SEGMENTS_FILENAME))


def test_emission_may_write_into_the_shard_directory_before_it_exists(tmp_path):
    """Plan r2 §3 R2-C emits <out-dir>/launch_record.json BEFORE the shard runs."""
    args, argv, _plan = _args(tmp_path, record=False, validate=False)
    path = os.path.join(args.out_dir, "launch_record.json")
    assert not os.path.exists(args.out_dir)
    emit_argv = argv + ["--emit-launch-record", path]
    emit_args = driver.parse_args(emit_argv)
    assert driver.emit_launch_record(emit_args, emit_argv, environment=OPT._environment(),
                                     runtime=OPT._runtime(OPT._FIXTURE_GPUS[0]["uuid"], 0)) == 0
    assert json.load(open(path))["argv"] == argv


# --------------------------------------------------------------------------- #
# hardening (self-review before hand-back)
# --------------------------------------------------------------------------- #
def test_a_negative_zero_binds_exactly_as_zero():
    """Python says -0.0 == 0.0; the canonical digest does not, so shards would not merge."""
    base = ["--ckpt-path", "x.ckpt", "--cond-method", "vanilla", "--cond-autocast", "default",
            "--batch-rows", "256", "--source-chunk", "16"]
    plain = driver.parse_args(base + ["--frame-avg-angles", "0", "--rotate-deg", "0"])
    negative = driver.parse_args(base + ["--frame-avg-angles", "-0", "--rotate-deg", "-0"])
    assert driver.protocol_binding(plain) == driver.protocol_binding(negative)
    from src.localization.crossarm import canonical_sha256

    assert canonical_sha256(driver.protocol_binding(negative)) == \
        canonical_sha256(driver.protocol_binding(plain))


def test_a_resume_reads_the_segment_stamp_from_the_verified_buffer(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    out = str(tmp_path / "run")
    _score(out, items, records, plan, binding, sha="a" * 64)
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]
    assert me.verify_query_artifact(path)["launch_record_sha256"] == "a" * 64


def test_an_unreadable_ledger_is_a_reason_never_a_legacy_run(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / ms.SEGMENTS_FILENAME).write_text("{not json")
    census = ms.provenance_census(str(out), [{"query_id": "0|q", "position": 0}])
    assert census["complete"] is False and census["generation"] == "unreadable"
    assert any("not readable" in reason for reason in census["reasons"])


def test_a_record_copied_into_the_merge_is_re_verified_on_the_copied_bytes(tmp_path):
    source = tmp_path / "record.json"
    source.write_bytes(b'{"a": 1}')
    digest = hashlib.sha256(b'{"a": 1}').hexdigest()
    target = tmp_path / "out" / "copy.json"
    ms.copy_verified(str(source), str(target), digest)
    assert target.read_bytes() == b'{"a": 1}'
    with pytest.raises(ValueError, match="changed"):
        ms.copy_verified(str(source), str(tmp_path / "out" / "other.json"), "0" * 64)
    assert not (tmp_path / "out" / "other.json").exists()


# --------------------------------------------------------------------------- #
# exp_22 rA-F1 (Codex rA BLOCKING; plan r2 amendment A2): ONE reviewed pin per run
# --------------------------------------------------------------------------- #
SHA_B = "b" * 40


def _at_commit(git_sha):
    """The executing machine at another commit (the record AND the admission see it)."""
    return OPT._environment(git_sha=git_sha, git_sha_capture=OPT._ok(git_sha))


def _hand_appended_segment(tmp_path, out, name, argv, environment, resume=True):
    """A segment appended WITHOUT open_segment: a run assembled by hand (or by old code)."""
    path = emit(tmp_path, argv=argv, name=name, environment=environment)
    admission = admit(path, argv=argv, environment=environment)
    digest = admission["sha256"]
    ms.copy_verified(path, os.path.join(out, ms.record_relpath(digest)), digest)
    ledger = ms.read_ledger(out)
    ledger["segments"].append({
        "segment_index": len(ledger["segments"]), "launch_record_sha256": digest,
        "record_file": ms.record_relpath(digest), "resume": bool(resume),
        "opened_utc": "2026-11-02T00:00:00+00:00",
        "admission": {field: admission.get(field) for field in ms.ADMISSION_FIELDS}})
    me.write_json(os.path.join(out, ms.SEGMENTS_FILENAME), ledger)
    return digest


def test_a_resume_at_another_commit_is_refused_before_it_opens_a_segment(tmp_path):
    out = str(tmp_path / "shard")
    first = emit(tmp_path, name="first.json")
    ms.open_segment(out, first, admit(first), resume=False)
    before = open(os.path.join(out, ms.SEGMENTS_FILENAME), "rb").read()
    resumed = ARGV + ["--resume"]
    second = emit(tmp_path, argv=resumed, name="second.json", environment=_at_commit(SHA_B))
    admission = admit(second, argv=resumed, environment=_at_commit(SHA_B))
    assert admission["environment_verified"] is True        # a healthy record, other commit
    with pytest.raises(ValueError, match="one reviewed pin"):
        ms.open_segment(out, second, admission, resume=True)
    # refused before anything moved: the ledger and the records are as they were
    assert open(os.path.join(out, ms.SEGMENTS_FILENAME), "rb").read() == before
    assert not os.path.exists(os.path.join(out, ms.record_relpath(admission["sha256"])))
    # the same resume at the run's own commit opens normally
    again = emit(tmp_path, argv=resumed, name="third.json")
    assert ms.open_segment(out, again, admit(again, argv=resumed), resume=True)[
        "segment_index"] == 1


def test_a_census_over_segments_at_two_commits_is_incomplete(tmp_path):
    out, entry = _opened(tmp_path)
    resumed = ARGV + ["--resume"]
    other = _hand_appended_segment(tmp_path, out, "other.json", resumed, _at_commit(SHA_B))
    rows = _rows(entry["launch_record_sha256"], [0, 1]) + _rows(other, [2])
    census = ms.provenance_census(out, rows)
    assert census["git_shas"] == ["a" * 40, SHA_B]
    assert all(record["verified"] for record in census["records"])   # each record is fine
    assert census["complete"] is False
    assert any("2 git SHAs" in reason and "one reviewed pin" in reason
               for reason in census["reasons"])
    # a segment that produced no row still belongs to the run: still refused
    assert ms.provenance_census(out, rows[:2])["complete"] is False


def test_a_single_commit_run_keeps_a_complete_census(tmp_path):
    out, entry = _opened(tmp_path)
    resumed = ARGV + ["--resume"]
    other = _hand_appended_segment(tmp_path, out, "same.json", resumed, None)
    census = ms.provenance_census(out, _rows(entry["launch_record_sha256"], [0])
                                  + _rows(other, [1]))
    assert census["git_shas"] == ["a" * 40] and census["complete"] is True, census["reasons"]


def _two_commit_shards(tmp_path):
    shards = {}
    for name, positions, environment in (("cafe", [0, 1], None),
                                         ("rest", [2, 3, 4], _at_commit(SHA_B))):
        argv = [token if token != "out" else name for token in ARGV]
        path = emit(tmp_path, argv=argv, name=f"{name}.json", environment=environment)
        out = str(tmp_path / name)
        entry = ms.open_segment(out, path, admit(path, argv=argv, environment=environment),
                                resume=False)
        shards[out] = _rows(entry["launch_record_sha256"], positions)
    return shards


def test_a_merge_of_shards_at_two_commits_is_refused_before_anything_is_written(tmp_path):
    shards = _two_commit_shards(tmp_path)
    # each shard on its own is a complete single-commit census ...
    assert all(ms.provenance_census(out, rows)["complete"] for out, rows in shards.items())
    merged = str(tmp_path / "merged")
    # ... but the run they would form is not: the dry run refuses, and so does the merge
    with pytest.raises(ValueError, match="one reviewed pin"):
        ms.merge_ledgers(shards, merged, dry_run=True)
    with pytest.raises(ValueError, match="one reviewed pin"):
        ms.merge_ledgers(shards, merged)
    assert not os.path.exists(merged)


def test_the_engine_merge_refuses_shards_at_two_commits_and_writes_nothing(tmp_path):
    plan, records, items = ENG._aligned(tmp_path)
    binding = ENG._r2_binding()
    dirs = {}
    for name, rooms, environment in (("a", ["A/A_idx_1"], None),
                                     ("b", ["B/B_idx_2"], _at_commit(SHA_B))):
        out = str(tmp_path / name)
        me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3},
                         declared_rooms=rooms)
        argv = [token if token != "out" else name for token in ARGV]
        path = emit(tmp_path, argv=argv, name=f"{name}.json", environment=environment)
        sha = ms.open_segment(out, path, admit(path, argv=argv, environment=environment),
                              resume=False)["launch_record_sha256"]
        _score(out, items, records, plan, binding, rooms=rooms, sha=sha)
        dirs[name] = out
    pairs, unions = ENG._fixture_totals()
    totals = {"rooms": 2, "queries": len(records), "candidate_query_pairs": pairs,
              "source_rows": unions, "generated_waveforms": pairs * 4}
    merged = str(tmp_path / "merged")
    with pytest.raises(ValueError, match="one reviewed pin"):
        me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    assert not os.path.exists(merged)
