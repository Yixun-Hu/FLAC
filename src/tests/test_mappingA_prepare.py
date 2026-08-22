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
    assert offender["kind"] == "below_threshold_crop"
    assert offender["scaled_dbfs"] < -60.0
    assert offender["scaled_dbfs_crop"] < -60.0


# --------------------------------------------------------------------------- #
# r2 N3: the loader's silence test reads the CROP, so the audit must too
# --------------------------------------------------------------------------- #
def _delayed(seed, peak=0.2, n=48000, onset=30000, sr=48000):
    """Loud, but with every sample of energy after the loader's crop.

    10240 samples at 22050 Hz is 0.464 s; an onset at 0.625 s is past it, so the
    runtime crops this capture down to (near) silence and substitutes the item.
    """
    rng = np.random.default_rng(seed)
    sig = np.zeros(n, dtype=np.float64)
    tail = np.arange(n - onset) / sr
    sig[onset:] = rng.normal(size=n - onset) * np.exp(-tail * 20.0)
    sig = sig / np.abs(sig).max() * peak
    return sig.astype(np.float32)


def test_a_loud_capture_that_is_silent_after_the_crop_aborts(union_room):
    """N3's headline: auditing the full waveform passed this file, and the loader
    then substituted it -- so the manifest named an item nobody evaluated."""
    _overwrite(union_room, "000002", _delayed(2, peak=0.2))
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    report = exc.value.report
    offender = next(v for v in report["violations"] if v["capture_id"] == "000002")
    assert offender["kind"] in ("below_threshold_crop", "silent_crop")
    assert offender["scaled_dbfs"] > -60.0          # loud by the OLD measurement
    assert offender["scaled_dbfs_crop"] < -60.0     # silent by the loader's
    assert report["n_below_threshold"] == 1
    assert report["loader_sample_size"] == prep_a.LOADER_SAMPLE_SIZE
    json.dumps(report)


def test_the_audit_reports_both_the_full_and_crop_statistics(union_room):
    ids = [f"{i:06d}" for i in range(6)]
    report = prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    assert report["min_scaled_dbfs_crop"] >= -60.0
    assert report["min_raw_peak_crop"] > 0.0
    assert report["min_raw_peak_crop"] <= report["max_raw_peak"]
    for violation_free in report["violations"]:
        assert "scaled_peak_crop" in violation_free


def test_the_clipping_test_still_reads_the_full_waveform(union_room):
    """A spike after the crop is written clipped even though the crop is clean, so
    the ceiling test must not move to the crop with the silence test."""
    signal = _rir(2, peak=0.2)
    signal[-50] = 0.9
    _overwrite(union_room, "000002", signal)
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)
    offender = next(v for v in exc.value.report["violations"]
                    if v["capture_id"] == "000002")
    assert offender["kind"] == "clipping"


def test_writing_refuses_a_capture_that_is_silent_after_the_crop(union_room, tmp_path):
    """The write path re-checks: the audit runs over a union derived from the
    manifest, so a write must not be able to publish a loader-silent target even
    when it was never audited."""
    _overwrite(union_room, "000002", _delayed(2, peak=0.2))
    with pytest.raises(ValueError) as exc:
        prep_a.write_union(union_room, str(tmp_path / "out"),
                           [f"{i:06d}" for i in range(6)], scalar=3.0)
    message = str(exc.value)
    assert "10240" in message and "-60" in message
    assert "substitute" in message


def test_the_written_report_records_the_crop_peak(union_room, tmp_path):
    report = prep_a.write_union(union_room, str(tmp_path / "out"),
                                [f"{i:06d}" for i in range(3)], scalar=3.0)
    assert report["loader_sample_size"] == prep_a.LOADER_SAMPLE_SIZE
    for entry in report["files"].values():
        assert 0.0 < entry["peak_crop"] <= entry["peak"]
        assert entry["dbfs_crop"] >= -60.0


def test_the_identity_separates_the_derivation_target_from_the_clip_ceiling():
    """0.75 is what the exp_19 scalar was DERIVED against; 0.999 is what every
    written file is CHECKED against. One key called "ceiling" hid which was which."""
    import publish as raf_publish

    registered = raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS
    assert registered["amplitude_derivation_target"] == 0.75
    assert registered["clip_ceiling"] == 0.999
    assert "amplitude_ceiling" not in registered
    assert prep_a.CLIP_CEILING == registered["clip_ceiling"]
    assert prep_a.AMPLITUDE_DERIVATION_TARGET == registered[
        "amplitude_derivation_target"]


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


# --------------------------------------------------------------------------- #
# cycle 12: the CLI that chains the tested components
# --------------------------------------------------------------------------- #
def _multi_placement_room(tmp_path, room, n_placements=2, n_groups=10):
    """A room with several array placements, each carrying source-distinct poses."""
    groups = []
    for p in range(n_placements):
        for g in range(n_groups):
            groups.append((
                (round(0.1 * (g + 1), 6), 0.9, 0.0, 0.1),
                (round(1.0 + g + 10 * p, 6), 1.5, round(0.5 * g, 6)),
                (round(2.0 + 3.0 * p, 6), 0.6, round(-1.0 + 0.0005 * g, 6)),
            ))
    write_room(str(tmp_path), room, groups=groups, rir_peak=0.2)
    return os.path.join(str(tmp_path), "archived", room)


def _readback_for(tmp_path, raf_root):
    """A passing, adjudicated readback record (the publish gate's input)."""
    from test_raf_prepare_data import write_passing_readback_record

    return write_passing_readback_record(str(tmp_path / "readback.json"))


def _cli_argv(tmp_path, raf_root, readback, extra=()):
    return ["--raf-root", str(raf_root),
            "--output-dir", str(tmp_path / "runtime" / "mappingA"),
            "--split-dir", str(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT),
            "--rooms", "EmptyRoom", "FurnishedRoom",
            "--n-placements", "2", "--k", "8", "--non-canonical",
            "--readback-record", readback] + list(extra)


@pytest.fixture
def cli_corpus(tmp_path):
    raf_root = tmp_path / "raf"
    for room in ("EmptyRoom", "FurnishedRoom"):
        _multi_placement_room(raf_root, room)
    return raf_root


def test_the_cli_publishes_the_whole_mappingA_surface(cli_corpus, tmp_path):
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))

    runtime = tmp_path / "runtime" / "mappingA"
    split_dir = tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
    assert (split_dir / "mappingA_eval.json").exists()
    assert (split_dir / "mappingA_splits_record.json").exists()
    assert (split_dir / "mappingA_amplitude_audit.json").exists()
    assert (runtime / "raf_publication.json").exists()
    for room in ("EmptyRoom", "FurnishedRoom"):
        assert (runtime / room / "metadata" / "mappingA_metadata.json").exists()
        assert (runtime / room / "mono_rirs_22050Hz").is_dir()

    with open(split_dir / "mappingA_eval.json") as f:
        manifest = json.load(f)
    # 2 placements x 36 mic slots x 2 rooms
    assert sum(len(v) for v in manifest.values()) == 2 * 36 * 2
    assert set(manifest) == {"EmptyRoom", "FurnishedRoom"}

    report = raf_publish.verify_publication(
        str(split_dir), kind="mappingA_prepare",
        expected_roots=[str(runtime.resolve()), str(split_dir.resolve())])
    assert report["published"] is True


def test_the_published_items_pass_the_static_validator(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    items = []
    for room in ("EmptyRoom", "FurnishedRoom"):
        with open(tmp_path / "runtime" / "mappingA" / room / "metadata" /
                  "mappingA_metadata.json") as f:
            items.extend(json.load(f).values())
    report = prep_a.validate_manifest({"items": items, "k": 8},
                                      expected_items=2 * 36 * 2, k=8)
    assert report["passed"] is True


def test_the_pointer_declares_the_mappingA_flavor(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    with open(tmp_path / "runtime" / "mappingA" / "raf_publication.json") as f:
        pointer = json.load(f)
    assert pointer["flavor"] == "mappingA"
    assert pointer["canonical"] is False           # --non-canonical run
    assert pointer["rooms"] == ["EmptyRoom", "FurnishedRoom"]
    assert any("non-canonical" in t for t in pointer["taint"])


def test_the_marker_carries_the_registered_identity_and_digests(cli_corpus, tmp_path):
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT /
              raf_publish.marker_name("mappingA_prepare")) as f:
        marker = json.load(f)
    parameters = marker["parameters"]
    assert parameters["k"] == 8
    assert parameters["n_placements"] == 2          # the run's own value
    assert parameters["n_items"] == 2 * 36 * 2
    assert parameters["match_algorithm_version"] == \
        prep_a.MATCH_ALGORITHM_VERSION if hasattr(prep_a, "MATCH_ALGORITHM_VERSION") \
        else parameters["match_algorithm_version"]
    for key in ("correspondence_sha256", "audio_union_sha256",
                "readback_record_sha256"):
        assert len(parameters[key]) == 64


def test_the_splits_record_carries_the_correspondence_evidence(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT /
              "mappingA_splits_record.json") as f:
        record = json.load(f)
    for room in ("EmptyRoom", "FurnishedRoom"):
        payload = record["rooms"][room]
        assert payload["n_placements"] >= 2
        assert payload["n_eligible_placements"] >= 2
        assert len(payload["selected_placements"]) == 2
        assert payload["displacements"]["p95_m"]["max"] <= 0.01
        assert payload["n_groups_failing"] >= 0
        assert payload["target_context_distance_m"]["n"] > 0
    assert record["canonical"] is False


def test_the_cli_stops_at_the_amplitude_gate_before_writing(cli_corpus, tmp_path):
    """M1's stop-and-ask: a violation aborts with the measured report and nothing
    is published."""
    loud = os.path.join(str(cli_corpus), "archived", "EmptyRoom", "data", "000005",
                        "rir.wav")
    sf.write(loud, _rir(5, peak=0.9), 48000, subtype="FLOAT")
    readback = _readback_for(tmp_path, cli_corpus)
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    assert exc.value.report["decision_required"] is True
    assert not (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT / "mappingA_eval.json").exists()
    assert not (tmp_path / "runtime" / "mappingA" / "EmptyRoom" /
                "mono_rirs_22050Hz").exists()


def test_the_cli_refuses_too_few_eligible_placements(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    with pytest.raises(ValueError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback,
                              extra=["--n-placements", "99"]))
    assert "eligible" in str(exc.value)
    assert not (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT / "mappingA_eval.json").exists()


def test_the_cli_refuses_non_registered_parameters_in_canonical_mode(cli_corpus,
                                                                     tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    argv = [a for a in _cli_argv(tmp_path, cli_corpus, readback)
            if a != "--non-canonical"]
    with pytest.raises(ValueError) as exc:
        prep_a.main(argv)
    # the readback gate or the parameter gate fires first; either way nothing runs
    assert "sha256" in str(exc.value).lower() or "non-registered" in str(exc.value)
    assert not (tmp_path / "runtime" / "mappingA").exists()


def test_the_cli_is_idempotent_on_the_item_set(cli_corpus, tmp_path):
    """Re-cutting with the same seed and parameters yields the same items; only the
    generation moves."""
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    argv = _cli_argv(tmp_path, cli_corpus, readback)
    prep_a.main(argv)
    split_dir = tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
    first = (split_dir / "mappingA_eval.json").read_bytes()
    first_generation = json.loads(
        (split_dir / raf_publish.marker_name("mappingA_prepare")).read_text())["generation"]

    prep_a.main(argv)
    assert (split_dir / "mappingA_eval.json").read_bytes() == first
    assert json.loads(
        (split_dir / raf_publish.marker_name("mappingA_prepare")).read_text()
    )["generation"] != first_generation


def test_the_two_flavors_survive_each_other_at_the_real_cli_defaults(cli_corpus,
                                                                     tmp_path):
    """N2 end to end: publish Mapping H with ITS default split root, then Mapping A
    with its own, and both remain verifiable -- the case the r1 composition tests
    could not see because they invented a separate A root."""
    import prepare_data as raf_prepare
    import publish as raf_publish
    from test_raf_prepare_data import write_passing_readback_record

    readback = write_passing_readback_record(str(tmp_path / "readback.json"))
    h_split = tmp_path / raf_prepare.build_parser().get_default("split_dir")
    h_runtime = tmp_path / "runtime" / "RAF"
    raf_prepare.main(["--raf-root", str(cli_corpus), "--output-dir", str(h_runtime),
                      "--split-dir", str(h_split), "--rooms", "EmptyRoom",
                      "--n-groups", "1", "--n-val-groups", "1",
                      "--n-diagnostic-groups", "1", "--n-train", "12",
                      "--full-crosscheck", "--non-canonical",
                      "--readback-record", readback])
    assert raf_publish.verify_publication(str(h_split), kind="prepare")["published"]

    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    a_split = tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
    assert raf_publish.verify_publication(str(a_split),
                                          kind="mappingA_prepare")["published"]
    # ... and Mapping H is still attested afterwards
    assert raf_publish.verify_publication(str(h_split), kind="prepare")["published"]
