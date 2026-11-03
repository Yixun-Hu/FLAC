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
