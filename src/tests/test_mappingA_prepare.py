"""Mapping-A audio union + amplitude policy (exp_21, contract B, cycles 3-4).

exp_19 published only the 21 selected tx-groups per room; Mapping A needs the exact
union of its target and context captures (~10,368 WAVs). Codex M1: that union has
never been amplitude-audited, and the registered x3 scalar was derived from the
Mapping-H trained supports -- a different population. So the union is audited
BEFORE anything is written, and ANY violation stops the run with a measured report
for Yixun. Never drop items, never auto-adjust the scalar.

Synthetic fixtures only; the real corpus is read-only.
"""
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import prepare_mappingA as prep_a  # noqa: E402

from test_raf_prepare_data import _default_groups, _rir, write_room  # noqa: E402

assert os.path.dirname(os.path.abspath(prep_a.__file__)) == _RAF_DIR


# --------------------------------------------------------------------------- #
# union enumeration
# --------------------------------------------------------------------------- #
def _item(room, placement, slot, target_capture, context_captures):
    return {
        "room": room, "placement_id": placement, "mic_slot": slot,
        "target_capture_id": target_capture,
        "context": [{"capture_id": c} for c in context_captures],
    }


def test_the_union_is_exactly_the_target_and_context_captures():
    items = [
        _item("EmptyRoom", "p000", 0, "000001", ["000010", "000011"]),
        _item("EmptyRoom", "p000", 1, "000002", ["000010", "000012"]),
    ]
    union = prep_a.enumerate_audio_union(items)
    assert union == {"EmptyRoom": ["000001", "000002", "000010", "000011", "000012"]}


def test_the_union_deduplicates_shared_context_captures():
    """A context capture reused by many items is written once, not once per item."""
    items = [_item("EmptyRoom", "p000", slot, f"{slot:06d}", ["000100"] * 1)
             for slot in range(5)]
    union = prep_a.enumerate_audio_union(items)
    assert union["EmptyRoom"].count("000100") == 1
    assert len(union["EmptyRoom"]) == 6


def test_the_union_separates_rooms():
    items = [_item("EmptyRoom", "p000", 0, "000001", ["000010"]),
             _item("FurnishedRoom", "p000", 0, "000001", ["000010"])]
    union = prep_a.enumerate_audio_union(items)
    assert set(union) == {"EmptyRoom", "FurnishedRoom"}
    assert union["EmptyRoom"] == union["FurnishedRoom"] == ["000001", "000010"]


def test_the_union_counts_are_reported_for_the_compute_estimate():
    items = [_item("EmptyRoom", "p000", s, f"{s:06d}", [f"{100 + s}".zfill(6)])
             for s in range(4)]
    report = prep_a.union_report(prep_a.enumerate_audio_union(items), items)
    assert report["n_items"] == 4
    assert report["n_captures"] == 8
    assert report["by_room"]["EmptyRoom"]["n_captures"] == 8
    assert report["by_room"]["EmptyRoom"]["n_items"] == 4
    json.dumps(report)


def test_an_item_without_context_is_refused():
    with pytest.raises(ValueError):
        prep_a.enumerate_audio_union([_item("EmptyRoom", "p000", 0, "000001", [])])


def test_a_target_that_is_also_its_own_context_is_refused():
    """Self-context would leak the answer into the conditioning."""
    with pytest.raises(ValueError) as exc:
        prep_a.enumerate_audio_union(
            [_item("EmptyRoom", "p000", 0, "000001", ["000001", "000010"])])
    assert "000001" in str(exc.value)


# --------------------------------------------------------------------------- #
# amplitude audit over the exact union (M1)
# --------------------------------------------------------------------------- #
@pytest.fixture
def union_room(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.2)
    return os.path.join(str(tmp_path), "archived", "EmptyRoom")


def _overwrite(room_dir, capture_id, signal):
    sf.write(os.path.join(room_dir, "data", capture_id, "rir.wav"), signal, 48000,
             subtype="FLOAT")


def test_a_clean_union_passes_and_reports_measurements(union_room):
    ids = [f"{i:06d}" for i in range(6)]
    report = prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    assert report["passed"] is True
    assert report["n_captures"] == 6
    assert report["scalar"] == 3.0
    assert 0.1 < report["max_raw_peak"] < 0.25
    assert report["max_scaled_peak"] == pytest.approx(report["max_raw_peak"] * 3.0)
    assert report["n_clipping"] == 0 and report["n_below_threshold"] == 0
    assert report["violations"] == []
    json.dumps(report)


def test_a_clipping_capture_aborts_with_a_measured_report(union_room):
    """A 0.7 peak survives resampling at ~0.5, so x3 lands past 1.0: the loader
    would clamp it and distort the very waveform the metric scores.

    (The fixture peak is set at 48 kHz; the anti-alias filter costs ~30%, which is
    why the trigger level is not simply 0.999/3.)
    """
    _overwrite(union_room, "000003", _rir(3, peak=0.7))
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    report = exc.value.report
    assert report["passed"] is False
    assert report["n_clipping"] == 1
    offender = next(v for v in report["violations"] if v["capture_id"] == "000003")
    assert offender["kind"] == "clipping"
    assert offender["scaled_peak"] > report["clip_ceiling"]
    assert offender["scaled_peak"] == pytest.approx(offender["raw_peak"] * 3.0)
    assert "stop" in str(exc.value).lower()
    json.dumps(report)


def test_a_still_silent_capture_aborts_with_a_measured_report(union_room):
    """Post-scale below the loader's -60 dB gate means silent substitution at eval
    time -- an item that is not the item the manifest claims."""
    _overwrite(union_room, "000004", _rir(4, peak=1e-5))
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    report = exc.value.report
    assert report["n_below_threshold"] == 1
    offender = next(v for v in report["violations"] if v["capture_id"] == "000004")
    assert offender["kind"] == "below_threshold"
    assert offender["scaled_dbfs"] < -60.0


def test_a_nonfinite_or_empty_capture_aborts(union_room):
    signal = _rir(5, peak=0.2)
    signal[10] = np.nan
    _overwrite(union_room, "000005", signal)
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, [f"{i:06d}" for i in range(6)],
                                     scalar=3.0)
    assert any(v["kind"] == "non_finite" for v in exc.value.report["violations"])


def test_the_audit_never_drops_items_or_adjusts_the_scalar(union_room):
    """M1's registered behaviour: the report lists what a different scalar WOULD
    require, but the run stops for a human decision either way."""
    _overwrite(union_room, "000003", _rir(3, peak=0.7))
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    report = exc.value.report
    assert report["scalar"] == 3.0                    # unchanged
    assert report["n_captures"] == len(ids)           # nothing dropped
    assert "max_admissible_scalar" in report          # measured option, not applied
    assert report["max_admissible_scalar"] < 3.0
    assert report["decision_required"] is True


def test_the_audit_reports_every_violation_not_just_the_first(union_room):
    _overwrite(union_room, "000002", _rir(2, peak=0.7))
    _overwrite(union_room, "000003", _rir(3, peak=0.9))
    _overwrite(union_room, "000004", _rir(4, peak=1e-5))
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, [f"{i:06d}" for i in range(6)],
                                     scalar=3.0)
    report = exc.value.report
    assert len(report["violations"]) == 3
    assert {v["capture_id"] for v in report["violations"]} == {"000002", "000003",
                                                               "000004"}
    # ordered worst-first so the report reads as evidence
    assert report["violations"][0]["capture_id"] == "000003"


def test_the_audit_measures_the_resampled_signal(union_room):
    """The peak that matters is the one that gets written, i.e. after resampling."""
    ids = [f"{i:06d}" for i in range(3)]
    report = prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    assert report["sample_rate"] == 22050
    assert report["source_sample_rate"] == 48000
    # a 48 kHz peak of 0.2 loses a little in the anti-alias filter
    assert report["max_raw_peak"] < 0.2


def test_the_audit_is_read_only(union_room):
    before = {p: os.path.getmtime(os.path.join(union_room, "data", p, "rir.wav"))
              for p in (f"{i:06d}" for i in range(4))}
    prep_a.audit_amplitude_union(union_room, list(before), scalar=3.0)
    for capture_id, mtime in before.items():
        assert os.path.getmtime(
            os.path.join(union_room, "data", capture_id, "rir.wav")) == mtime


# --------------------------------------------------------------------------- #
# cycle 4: writing the union, with Mapping-H byte-identity provenance
# --------------------------------------------------------------------------- #
def test_the_union_is_written_scaled_and_read_back(union_room, tmp_path):
    ids = [f"{i:06d}" for i in range(4)]
    out = tmp_path / "runtime" / "mappingA" / "EmptyRoom"
    report = prep_a.write_union(union_room, str(out), ids, scalar=3.0)
    assert report["n_files"] == 4
    assert report["scalar"] == 3.0
    for capture_id in ids:
        path = out / "mono_rirs_22050Hz" / f"{capture_id}.wav"
        assert path.exists()
        info = sf.info(str(path))
        assert info.samplerate == 22050 and info.channels == 1
        assert info.subtype == "FLOAT"
        entry = report["files"][capture_id]
        assert entry["roundtrip_max_abs_error"] == 0.0
        assert len(entry["sha256"]) == 64
        assert entry["peak"] <= prep_a.CLIP_CEILING


def test_the_written_peak_is_the_scaled_peak(union_room, tmp_path):
    ids = ["000000"]
    plain = prep_a.write_union(union_room, str(tmp_path / "a" / "EmptyRoom"), ids,
                               scalar=1.0)
    scaled = prep_a.write_union(union_room, str(tmp_path / "b" / "EmptyRoom"), ids,
                                scalar=3.0)
    assert scaled["files"]["000000"]["peak"] == pytest.approx(
        3.0 * plain["files"]["000000"]["peak"], rel=1e-6)


def test_a_capture_shared_with_mapping_h_records_verified_provenance(union_room,
                                                                     tmp_path):
    """Mapping A publishes to disjoint roots, so 'shared' is a PROVENANCE claim:
    the bytes are identical to what the Mapping-H publication holds, verified by
    hash, with the exp_19 generation recorded."""
    ids = ["000000", "000001"]
    mapping_h = tmp_path / "runtime" / "RAF" / "EmptyRoom"
    prep_a.write_union(union_room, str(mapping_h), ids, scalar=3.0)

    out = tmp_path / "runtime" / "mappingA" / "EmptyRoom"
    report = prep_a.write_union(union_room, str(out), ids, scalar=3.0,
                                mappingH_room_dir=str(mapping_h),
                                mappingH_generation="46a43f4ce82b")
    for capture_id in ids:
        shared = report["files"][capture_id]["shared_with_mappingH"]
        assert shared["verified_identical"] is True
        assert shared["generation"] == "46a43f4ce82b"
        assert shared["sha256"] == report["files"][capture_id]["sha256"]
    assert report["n_shared_with_mappingH"] == 2


def test_a_capture_that_differs_from_mapping_h_aborts(union_room, tmp_path):
    """Two publications disagreeing about the same capture is a real defect -- a
    different scalar, a different source file, or a stale generation."""
    ids = ["000000"]
    mapping_h = tmp_path / "runtime" / "RAF" / "EmptyRoom"
    prep_a.write_union(union_room, str(mapping_h), ids, scalar=2.0)   # WRONG scalar
    with pytest.raises(ValueError) as exc:
        prep_a.write_union(union_room, str(tmp_path / "out" / "EmptyRoom"), ids,
                           scalar=3.0, mappingH_room_dir=str(mapping_h),
                           mappingH_generation="46a43f4ce82b")
    assert "000000" in str(exc.value)
    assert "byte-identical" in str(exc.value)


def test_a_capture_absent_from_mapping_h_is_recorded_as_new(union_room, tmp_path):
    """Most of the union is new: exp_19 published only its 21 selected groups."""
    mapping_h = tmp_path / "runtime" / "RAF" / "EmptyRoom"
    prep_a.write_union(union_room, str(mapping_h), ["000000"], scalar=3.0)
    report = prep_a.write_union(union_room, str(tmp_path / "out" / "EmptyRoom"),
                                ["000000", "000001"], scalar=3.0,
                                mappingH_room_dir=str(mapping_h),
                                mappingH_generation="46a43f4ce82b")
    assert report["files"]["000000"]["shared_with_mappingH"]["verified_identical"] is True
    assert report["files"]["000001"]["shared_with_mappingH"] is None
    assert report["n_shared_with_mappingH"] == 1
    assert report["n_new"] == 1


def test_writing_refuses_a_clipping_scalar(union_room, tmp_path):
    _overwrite(union_room, "000000", _rir(0, peak=0.9))
    with pytest.raises(ValueError) as exc:
        prep_a.write_union(union_room, str(tmp_path / "out" / "EmptyRoom"),
                           ["000000"], scalar=3.0)
    assert "clip" in str(exc.value).lower()


def test_the_write_report_is_json_safe(union_room, tmp_path):
    report = prep_a.write_union(union_room, str(tmp_path / "out" / "EmptyRoom"),
                                [f"{i:06d}" for i in range(3)], scalar=3.0)
    json.dumps(report, allow_nan=False)
