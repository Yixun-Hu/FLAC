"""exp_22 R2-A -- launch provenance for the full pass, one SEGMENT per invocation.

Plan r2 §3 R2-A and the Codex plan review (B4): the off-grid probe's reviewed
launch-record mechanism -- exact argv, full git SHA, tracked-clean tree, hostname
and the CUDA runtime's own UUID for the designated card, compared against the
executing machine at admission -- is IMPORTED here and reused unchanged. What
this module adds is the segment model the full pass needs around it:

* a record describes exactly ONE invocation, so a shard that is resumed holds one
  segment per invocation, and every row it publishes names the segment (the
  record's sha256) that produced it;
* provenance lives BESIDE the run binding, never inside it: the binding is
  common to every shard of a run (the merge's equal-binding-digest rule), while
  argv, card and record differ per shard and per resume;
* the record's bytes are copied into the run directory at admission, so the run
  carries the very file that vouched for it and anyone can re-hash it later;
* a census -- every row resolves to a present, hash-verified,
  environment-verified record -- gates the merge and is re-derived by the report.
"""
import hashlib
import json
import os

from datetime import datetime, timezone

from src.localization import meshgrid_engine as me
from src.localization.meshgrid_offgrid_probe import (  # noqa: F401 -- reused, not copied
    ENVIRONMENT_AXES, LAUNCH_RECORD_FIELDS, LAUNCH_RECORD_NOTE, build_launch_record,
    normalize_gpu_uuid, read_verified_launch_record, write_launch_record)

SEGMENTS_FILENAME = "launch_segments.json"
RECORDS_DIRNAME = "launch_records"

SEGMENT_NOTE = (
    "LAUNCH SEGMENTS (plan r2 §3 R2-A, Codex B4). One launch record per INVOCATION: a shard "
    "run once has one segment, a resumed shard one more per resume, and every row carries the "
    "sha256 of the record whose invocation produced it. The record bytes are copied into the "
    "run directory at admission; the admission verdict (four environment axes, all passed) is "
    "kept beside them. A merge requires every row to resolve to a present, hash-verified, "
    "environment-verified record and carries every record across with its row ranges; the "
    "report re-derives that census, and any gap makes the report NON-CANONICAL. The run "
    "binding is untouched: it stays common to all shards")

#: what a segment keeps of the admission verdict -- enough to re-check it later.
ADMISSION_FIELDS = ("sha256", "argv", "git_sha", "git_status_dirty", "hostname",
                    "executing_device", "execution_gpu_uuid", "execution_gpu_name",
                    "environment_comparison", "environment_axes_passed",
                    "environment_axes_expected", "environment_verified", "recorded_utc")


def record_relpath(sha256):
    """Where a record with this digest lives inside a run directory."""
    return os.path.join(RECORDS_DIRNAME, f"{sha256}.json")


def position_ranges(positions):
    """``[[first, last], ...]`` over runs of consecutive stream positions."""
    ordered = sorted(int(p) for p in positions)
    if len(set(ordered)) != len(ordered):
        raise ValueError("a stream position appears twice; one row per position")
    ranges = []
    for position in ordered:
        if ranges and position == ranges[-1][1] + 1:
            ranges[-1][1] = position
        else:
            ranges.append([position, position])
    return ranges


def read_ledger(run_dir):
    """The run's segment ledger, or ``None`` when it has never had one."""
    path = os.path.join(str(run_dir), SEGMENTS_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def ledger_shas(run_dir):
    return [str(s["launch_record_sha256"]) for s in (read_ledger(run_dir) or {}).get(
        "segments", [])]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_segment(out_dir, record_path, admission, *, resume, at_utc=None):
    """Copy the admitted record into the run and append its segment -- or refuse."""
    with open(str(record_path), "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != str(admission.get("sha256")):
        raise ValueError(f"the launch record at {record_path!r} changed since it was admitted "
                         f"(admitted {str(admission.get('sha256'))[:12]}..., now "
                         f"{digest[:12]}...); it no longer vouches for this invocation")
    if admission.get("environment_verified") is not True:
        raise ValueError("the launch record's environment was not verified on every axis; a "
                         f"segment opens only on a fully verified admission. {LAUNCH_RECORD_NOTE}")
    ledger = read_ledger(out_dir) or {"segments": [], "note": SEGMENT_NOTE}
    if digest in {str(s["launch_record_sha256"]) for s in ledger["segments"]}:
        raise ValueError(f"launch record {digest[:12]}... already opened a segment of this run; a "
                         "record vouches for one invocation only -- emit a new one for this "
                         "invocation (a --resume is a new segment)")
    target = os.path.join(str(out_dir), record_relpath(digest))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target + ".tmp", "wb") as handle:
        handle.write(raw)
    os.replace(target + ".tmp", target)
    entry = {"segment_index": len(ledger["segments"]), "launch_record_sha256": digest,
             "record_file": record_relpath(digest), "resume": bool(resume),
             "opened_utc": at_utc or _now(),
             "admission": {field: admission.get(field) for field in ADMISSION_FIELDS}}
    ledger["segments"].append(entry)
    me.write_json(os.path.join(str(out_dir), SEGMENTS_FILENAME), ledger)
    return entry


def close_segment(out_dir, sha256, summary, at_utc=None):
    """Stamp what an invocation produced into its segment (operational disclosure, H4)."""
    ledger = read_ledger(out_dir) or {"segments": []}
    matches = [s for s in ledger["segments"] if s["launch_record_sha256"] == str(sha256)]
    if len(matches) != 1:
        raise ValueError(f"there is no open segment {str(sha256)[:12]}... in {out_dir}")
    matches[0].update({"closed_utc": at_utc or _now(),
                       "n_scored": int(summary.get("n_scored", 0)),
                       "n_skipped": int(summary.get("n_skipped", 0))})
    me.write_json(os.path.join(str(out_dir), SEGMENTS_FILENAME), ledger)
    return matches[0]


def _record_reasons(record, digest):
    reasons = [f"the record is missing {field!r}" for field in LAUNCH_RECORD_FIELDS
               if not record.get(field)]
    if len(str(record.get("git_sha") or "")) != 40:
        reasons.append(f"the record's git_sha {record.get('git_sha')!r} is not a full commit id")
    if not all(str(gpu.get("uuid") or "").startswith("GPU-") for gpu in record.get("gpus") or []):
        reasons.append("the record's GPU entries are not nvidia-smi UUIDs")
    if record.get("git_status_dirty") is not False:
        reasons.append(f"the record was written from a tracked-dirty (or unstated) tree "
                       f"(git_status_dirty={record.get('git_status_dirty')!r})")
    if not record.get("execution_gpu_uuid"):
        reasons.append("the record designates no execution GPU UUID")
    return reasons


def _admission_reasons(admission, record, digest):
    reasons = []
    if str(admission.get("sha256")) != digest:
        reasons.append(f"the admission names {str(admission.get('sha256'))[:12]}..., not this "
                       f"record {digest[:12]}...")
    if admission.get("environment_verified") is not True:
        reasons.append("the admission's environment_verified is not true")
    axes = admission.get("environment_comparison") or []
    if sorted(axis.get("axis") for axis in axes) != sorted(ENVIRONMENT_AXES) or not (
            admission.get("environment_axes_passed")
            == admission.get("environment_axes_expected") == len(ENVIRONMENT_AXES)):
        reasons.append(f"the admission does not carry the {len(ENVIRONMENT_AXES)} environment "
                       f"axes {list(ENVIRONMENT_AXES)} all passed")
    reasons += [f"environment axis {axis.get('axis')} did not pass: {axis.get('why')}"
                for axis in axes if axis.get("verdict") != "pass"]
    for field in ("git_sha", "hostname"):
        if str(admission.get(field)) != str(record.get(field)):
            reasons.append(f"the admission's {field} {admission.get(field)!r} is not the "
                           f"record's {record.get(field)!r}")
    if normalize_gpu_uuid(admission.get("execution_gpu_uuid") or "") != normalize_gpu_uuid(
            record.get("execution_gpu_uuid") or ""):
        reasons.append("the admission's execution_gpu_uuid is not the card the record "
                       "designates")
    return reasons


def verify_segment(run_dir, entry):
    """Re-check one segment from the bytes it carries: hash, shape, admission."""
    digest = str(entry.get("launch_record_sha256"))
    verdict = {"ok": False, "sha256": digest, "reasons": [], "facts": {}}
    if entry.get("record_file") != record_relpath(digest):
        verdict["reasons"].append(f"{entry.get('record_file')!r} is not the record path "
                                  f"{record_relpath(digest)!r} its digest names")
        return verdict
    path = os.path.join(str(run_dir), record_relpath(digest))
    if not os.path.isfile(path):
        verdict["reasons"].append(f"the record file {record_relpath(digest)!r} is missing")
        return verdict
    with open(path, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != digest:
        verdict["reasons"].append(f"the record file does not hash to {digest[:12]}...; it was "
                                  "edited after admission")
        return verdict
    try:
        record = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:            # noqa: BLE001 -- a reason
        verdict["reasons"].append(f"the record is not readable JSON: {error}")
        return verdict
    verdict["reasons"] = (_record_reasons(record, digest)
                          + _admission_reasons(entry.get("admission") or {}, record, digest))
    verdict["facts"] = {field: record.get(field) for field in
                        ("git_sha", "hostname", "execution_gpu_uuid", "execution_gpu_name",
                         "execution_device", "recorded_utc", "argv")}
    verdict["ok"] = not verdict["reasons"]
    return verdict
