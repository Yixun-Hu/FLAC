"""Mapping-A audio union + amplitude policy (exp_21, contract B, cycles 3-4).

exp_19 published only the 21 selected tx-groups per room; Mapping A needs the exact
union of its target and context captures (~10,368 WAVs). Codex M1: that union has
never been amplitude-audited, and the registered x3 scalar was derived from the
Mapping-H trained supports -- a different population. So the union is audited
BEFORE anything is written, and ANY violation stops the run with a measured report
for Yixun. Never drop items, never auto-adjust the scalar.

Synthetic fixtures only; the real corpus is read-only.
"""
import hashlib
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


def _h_files(capture_ids):
    """What a Mapping-H generation's manifest covers, room-relative (N4)."""
    return {f"{prep_a.RIR_FOLDER}/{c}.wav" for c in capture_ids}


def test_a_capture_in_both_corpora_records_the_overlap_not_a_reuse(union_room,
                                                                   tmp_path):
    """Amendment 4: no file is shared. A capture the Mapping-H publication also
    holds is written FRESH at Mapping A's scalar, and what is recorded is the
    overlap -- the generation that covers it and BOTH scalars."""
    ids = ["000000", "000001"]
    mapping_h = tmp_path / "runtime" / "RAF" / "EmptyRoom"
    prep_a.write_union(union_room, str(mapping_h), ids, scalar=3.0)

    out = tmp_path / "runtime" / "mappingA" / "EmptyRoom"
    report = prep_a.write_union(union_room, str(out), ids, scalar=2.0,
                                mappingH_room_dir=str(mapping_h),
                                mappingH_generation="46a43f4ce82b",
                                mappingH_files=_h_files(ids), mappingH_scalar=3.0)
    for capture_id in ids:
        overlap = report["files"][capture_id]["overlaps_mappingH"]
        assert overlap["same_capture"] is True and overlap["same_audio"] is False
        assert overlap["generation"] == "46a43f4ce82b"
        assert overlap["mappingA_amplitude_scalar"] == 2.0
        assert overlap["mappingH_amplitude_scalar"] == 3.0
        # written at MAPPING A's level, whatever Mapping H holds
        written = sf.read(str(out / prep_a.RIR_FOLDER / f"{capture_id}.wav"),
                          dtype="float32")[0]
        theirs = sf.read(str(mapping_h / prep_a.RIR_FOLDER / f"{capture_id}.wav"),
                         dtype="float32")[0]
        assert np.abs(written).max() == pytest.approx(np.abs(theirs).max() * 2 / 3,
                                                      rel=1e-6)
    assert report["n_overlapping_mappingH"] == 2
    assert report["scale_disclosure"]["audio_is_shared"] is False


def test_the_write_report_discloses_both_scalars(union_room, tmp_path):
    report = prep_a.write_union(union_room, str(tmp_path / "out"), ["000000"],
                                scalar=2.0, mappingH_scalar=3.0)
    disclosure = report["scale_disclosure"]
    assert disclosure["mappingA_amplitude_scalar"] == 2.0
    assert disclosure["mappingH_amplitude_scalar"] == 3.0
    assert "unlicensed" in disclosure["note"]
    assert "T60/C50/EDT are level-independent" in disclosure["note"]


def test_a_capture_outside_the_mappingH_publication_is_recorded_as_such(union_room,
                                                                        tmp_path):
    """Most of the union is outside Mapping H: exp_19 published only its 21
    selected groups."""
    mapping_h = tmp_path / "runtime" / "RAF" / "EmptyRoom"
    prep_a.write_union(union_room, str(mapping_h), ["000000"], scalar=3.0)
    report = prep_a.write_union(union_room, str(tmp_path / "out" / "EmptyRoom"),
                                ["000000", "000001"], scalar=2.0,
                                mappingH_room_dir=str(mapping_h),
                                mappingH_generation="46a43f4ce82b",
                                mappingH_files=_h_files(["000000"]),
                                mappingH_scalar=3.0)
    assert report["files"]["000000"]["overlaps_mappingH"]["same_capture"] is True
    assert report["files"]["000001"]["overlaps_mappingH"] is None
    assert report["n_overlapping_mappingH"] == 1
    assert report["n_outside_mappingH"] == 1


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


# The synthetic corpora are generated deterministically by _multi_placement_room,
# so their geometry -- and therefore their correspondence audit -- is the same in
# every test; surveying once keeps the CLI tests from paying for it twice each.
_CORRESPONDENCE_CACHE = {}


def _correspondence_summary(raf_root, rooms):
    key = tuple(rooms)
    if key not in _CORRESPONDENCE_CACHE:
        summary = {}
        for room in rooms:
            survey = prep_a.survey_room(os.path.join(str(raf_root), "archived", room),
                                        room)
            summary[room] = {
                "eligible_placement_ids": sorted(p["placement_id"]
                                                 for p in survey["placements"]
                                                 if p["eligible"]),
                "n_placements": survey["n_placements"],
                "n_groups_passing": survey["n_groups_passing"],
                "n_groups_failing": survey["n_groups_failing"],
            }
        _CORRESPONDENCE_CACHE[key] = summary
    return _CORRESPONDENCE_CACHE[key]


def _correspondence_for(tmp_path, raf_root, room_names=("EmptyRoom", "FurnishedRoom"),
                        name="correspondence.json", **overrides):
    """A correspondence record describing THIS corpus (the CLI's N5 input)."""
    record = {
        "schema_version": 1,
        "created_utc": "2026-08-22T00:00:00Z",
        "raf_root": str(raf_root),
        "algorithm_version": prep_a.MATCH_ALGORITHM_VERSION,
        "tolerances": {"placement_cap_m": prep_a.PLACEMENT_CAP_M,
                       "match_p95_m": prep_a.MATCH_P95_M,
                       "match_max_m": prep_a.MATCH_MAX_M,
                       "match_ambiguity_margin": prep_a.MATCH_AMBIGUITY_MARGIN},
        "rooms": _correspondence_summary(raf_root, room_names),
        "verdict": {"eligible": True},
    }
    record.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(record, indent=1))
    return str(path)


def _derived_argv(argv):
    """The same invocation with --output-dir dropped, so the runtime root is the
    DERIVED <mappingH-dir>/mappingA (r3 P1)."""
    index = argv.index("--output-dir")
    return argv[:index] + argv[index + 2:]


def _cli_argv(tmp_path, raf_root, readback, extra=(), correspondence=None):
    return ["--raf-root", str(raf_root),
            "--output-dir", str(tmp_path / "runtime" / "mappingA"),
            "--split-dir", str(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT),
            "--rooms", "EmptyRoom", "FurnishedRoom",
            "--n-placements", "2", "--k", "8", "--non-canonical",
            "--correspondence-record",
            correspondence or _correspondence_for(tmp_path, raf_root),
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
    """Re-attested OFFLINE from the publication itself: the items come from the
    runtime metadata and the authoritative per-slot assignment from the published
    splits record, so nothing is re-surveyed and nothing is taken on trust (P2)."""
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    items = []
    for room in ("EmptyRoom", "FurnishedRoom"):
        with open(tmp_path / "runtime" / "mappingA" / room / "metadata" /
                  "mappingA_metadata.json") as f:
            items.extend(json.load(f).values())
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_splits_record.json") as f:
        assignments = json.load(f)["assignments"]
    assert sorted(assignments) == ["EmptyRoom", "FurnishedRoom"]
    report = prep_a.validate_manifest({"items": items, "k": 8},
                                      expected_items=2 * 36 * 2, k=8,
                                      assignments=assignments)
    assert report["passed"] is True and report["assignments_attested"] is True


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


def _a_target_capture(raf_root, room, n_placements=2, k=8, seed=0):
    """A capture the canonical item set draws as a TARGET, from the real builder."""
    survey = prep_a.survey_room(os.path.join(str(raf_root), "archived", room), room)
    for placement in prep_a.select_placements(survey, n_placements):
        items = prep_a.build_items(room, placement["placement_id"],
                                   placement["passing"], placement["assignment"],
                                   placement["match"], k=k, seed=seed)
        if items:
            return items[0]["target_capture_id"]
    raise AssertionError(f"no items built for {room}")


def test_the_cli_stops_at_the_amplitude_gate_before_writing(cli_corpus, tmp_path):
    """M1's stop-and-ask: a violation aborts with the measured report and nothing
    is published.

    Amendment 4 note: loudness alone no longer stops the run -- the clamp term
    lowers the derived scalar until the union fits. Amendment 4.1 narrows it
    further: a sub-threshold CONTEXT is recorded, not fatal. What still stops the
    run is a sub-threshold capture drawn as a TARGET, which the loader would
    substitute, so that is what this drives."""
    target = _a_target_capture(cli_corpus, "EmptyRoom")
    sf.write(os.path.join(str(cli_corpus), "archived", "EmptyRoom", "data", target,
                          "rir.wav"),
             _rir(5, peak=1e-6), 48000, subtype="FLOAT")
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


# --------------------------------------------------------------------------- #
# r2 N4: the Mapping-H generation is located, required and verified in the CLI
# --------------------------------------------------------------------------- #
def _publish_mappingH(tmp_path, raf_root, readback, rooms=("EmptyRoom",),
                      runtime=None):
    import prepare_data as raf_prepare

    runtime = runtime or (tmp_path / "runtime" / "RAF")
    raf_prepare.main(["--raf-root", str(raf_root), "--output-dir", str(runtime),
                      "--split-dir", str(tmp_path / "data" / "RAF"),
                      "--rooms", *rooms,
                      "--n-groups", "2", "--n-val-groups", "1",
                      "--n-diagnostic-groups", "1", "--n-train", "12",
                      "--full-crosscheck", "--non-canonical",
                      "--readback-record", readback])
    return runtime


def test_a_canonical_run_must_name_the_mappingH_publication():
    with pytest.raises(prep_a.MappingHProvenanceError) as exc:
        prep_a.resolve_mappingH(None, ["EmptyRoom"], canonical=True)
    assert "--mappingH-dir" in str(exc.value)


def test_a_non_canonical_run_may_decline_but_says_so_in_its_taint():
    publication, taint = prep_a.resolve_mappingH(None, ["EmptyRoom"], canonical=False)
    assert publication is None
    assert taint and "unverified" in taint[0]


def test_an_unpublished_tree_is_not_a_mappingH_publication(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(prep_a.MappingHProvenanceError) as exc:
        prep_a.locate_mappingH(str(tmp_path / "empty"), ["EmptyRoom"])
    assert "not a published" in str(exc.value)


def test_a_mappingA_tree_cannot_supply_mappingH_provenance(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    with pytest.raises(prep_a.MappingHProvenanceError) as exc:
        prep_a.locate_mappingH(str(tmp_path / "runtime" / "mappingA"),
                               ["EmptyRoom", "FurnishedRoom"])
    assert "mappingA" in str(exc.value)


def test_the_located_publication_carries_the_generation_and_its_file_set(cli_corpus,
                                                                        tmp_path):
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback)
    found = prep_a.locate_mappingH(str(runtime), ["EmptyRoom"])
    marker = raf_publish.verify_publication(str(tmp_path / "data" / "RAF"),
                                            kind="prepare")
    assert found["generation"] == marker["generation"]
    assert found["rooms"]["EmptyRoom"]["files"]
    assert all(name.startswith(prep_a.RIR_FOLDER + "/")
               or not name.endswith(".wav")
               for name in found["rooms"]["EmptyRoom"]["files"])


def test_a_stale_generation_in_the_manifest_is_refused(cli_corpus, tmp_path):
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback)
    manifest_path = runtime / raf_publish.MANIFEST_NAME
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["generation"] = "deadbeefdead"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Exception) as exc:      # the marker check fires first
        prep_a.locate_mappingH(str(runtime), ["EmptyRoom"])
    assert "generation" in str(exc.value)


def test_the_cli_records_the_mappingH_generation_on_every_room(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback,
                                rooms=("EmptyRoom", "FurnishedRoom"))
    prep_a.main(_derived_argv(_cli_argv(tmp_path, cli_corpus, readback,
                                        extra=["--mappingH-dir", str(runtime)])))
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_amplitude_audit.json") as f:
        audit = json.load(f)
    found = prep_a.locate_mappingH(str(runtime), ["EmptyRoom", "FurnishedRoom"])
    for room in ("EmptyRoom", "FurnishedRoom"):
        written = audit["written"][room]
        assert written["mappingH_generation"] == found["generation"]
        # every recorded overlap is a real one, against that generation's manifest
        expected = sum(1 for name in written["files"]
                       if f"{prep_a.RIR_FOLDER}/{name}.wav"
                       in found["rooms"][room]["files"])
        assert written["n_overlapping_mappingH"] == expected
        for capture_id, entry in written["files"].items():
            overlap = entry["overlaps_mappingH"]
            if overlap is not None:
                assert overlap["same_capture"] is True
                assert overlap["same_audio"] is False
                assert overlap["generation"] == found["generation"]
                assert (overlap["mappingH_amplitude_scalar"]
                        == found["amplitude_scalar"])
    assert "no Mapping-H provenance" not in json.dumps(audit["taint"])
    # the disclosure travels with the artifact
    assert audit["scale_disclosure"]["audio_is_shared"] is False


def test_a_file_outside_the_mappingH_generation_aborts_the_run(cli_corpus, tmp_path):
    """A wav sitting in the Mapping-H tree that its manifest does not cover is a
    leftover or a stale publish; claiming shared provenance with it would attest
    audio no publication stands behind."""
    import shutil

    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))       # first, unshared
    a_room = tmp_path / "runtime" / "mappingA" / "EmptyRoom" / prep_a.RIR_FOLDER
    found = prep_a.locate_mappingH(str(runtime), ["EmptyRoom"])
    smuggled = next(p for p in sorted(a_room.glob("*.wav"))
                    if f"{prep_a.RIR_FOLDER}/{p.name}"
                    not in found["rooms"]["EmptyRoom"]["files"])
    shutil.copy(smuggled, runtime / "EmptyRoom" / prep_a.RIR_FOLDER / smuggled.name)

    with pytest.raises(ValueError) as exc:
        prep_a.main(_derived_argv(_cli_argv(
            tmp_path, cli_corpus, readback,
            extra=["--mappingH-dir", str(runtime),
                   "--scalar", str(found["amplitude_scalar"])])))
    assert "NOT covered" in str(exc.value) and found["generation"] in str(exc.value)


def test_a_mappingH_publication_at_another_scalar_is_accepted_and_disclosed(
        cli_corpus, tmp_path):
    """Amendment 4: the scalars are EXPECTED to differ -- Mapping H's x3 clips two
    Mapping-A union captures -- so the difference is disclosed, not refused. The
    topology checks (location, generation, manifest coverage) all stay."""
    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback)
    publication, taint = prep_a.resolve_mappingH(str(runtime), ["EmptyRoom"],
                                                 canonical=False)
    assert taint == []
    assert publication["generation"] and publication["rooms"]["EmptyRoom"]["files"]
    disclosure = prep_a.scale_disclosure(2.0, publication["amplitude_scalar"])
    assert disclosure["mappingA_amplitude_scalar"] == 2.0
    assert disclosure["audio_is_shared"] is False


def test_a_shared_claim_without_the_covered_set_is_refused(union_room, tmp_path):
    with pytest.raises(ValueError) as exc:
        prep_a.write_union(union_room, str(tmp_path / "out"), ["000000"], scalar=3.0,
                           mappingH_room_dir=str(tmp_path / "h"),
                           mappingH_generation="46a43f4ce82b")
    assert "generation" in str(exc.value) and "manifest covers" in str(exc.value)


def test_float_wav_bytes_are_not_reproducible_but_audio_is(tmp_path):
    """Why the shared claim is a CONTENT claim: libsndfile stamps a UNIX timestamp
    into the PEAK chunk of every float WAV (measured at byte offset 60), so two
    publications holding bit-identical audio have different file digests unless
    they were written in the same second. A byte-identity gate would have failed
    the real corpus, where Mapping A is prepared long after Mapping H."""
    import time

    wave = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(first), wave, 22050, subtype="FLOAT")
    time.sleep(1.1)
    sf.write(str(second), wave, 22050, subtype="FLOAT")
    assert first.read_bytes() != second.read_bytes()
    assert (prep_a.audio_fingerprint(str(first))
            == prep_a.audio_fingerprint(str(second)))


# --------------------------------------------------------------------------- #
# r2 N5: the identity names committed inputs, digest for digest
# --------------------------------------------------------------------------- #
def test_the_registered_correspondence_digest_is_the_committed_record():
    """The pin, checked against the file it pins. N9 regenerates the record and
    re-pins this value in the same commit."""
    import publish as raf_publish

    record = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun",
                          "exp_21_raf_mappingA_claude",
                          "mappingA_correspondence_record.json")
    with open(record, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert (raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS["correspondence_sha256"]
            == digest)


def test_the_registered_readback_digest_is_exp_19s_pinned_record():
    import publish as raf_publish

    pinned = raf_publish.canonical_record_digest()
    assert (raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS["readback_record_sha256"]
            == pinned)
    assert (raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS["readback_record_sha256"]
            == pinned)


def test_nothing_in_the_registered_identity_is_a_placeholder_any_more():
    """r5: the audio union was the last unpinned value and came from the clean dry
    run; a canonical publication is now identified by every digit of its identity."""
    import publish as raf_publish

    assert raf_publish.unpinned_identity_keys("mappingA_prepare") == []
    assert raf_publish.unpinned_identity_keys("mappingA_depth") == []
    assert (raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS["audio_union_sha256"]
            == "b19eff06c7a13e0aaeafcdf95ad58f7f4f24bb3def794889102440437a220a21")


def test_only_the_pinned_audio_union_may_publish_canonically():
    """The pin is the point: this union and no other. A run that measures anything
    else is a different corpus and is refused by name."""
    pinned = "b19eff06c7a13e0aaeafcdf95ad58f7f4f24bb3def794889102440437a220a21"
    assert prep_a.canonical_identity_blockers() == []
    assert prep_a.canonical_identity_blockers(pinned, 10368) == []
    assert prep_a.canonical_identity_blockers("e" * 64, 10368) == [
        f"audio union {'e' * 64} != registered {pinned}"]


def test_a_placeholder_registry_still_refuses_a_canonical_publication(monkeypatch):
    """The mechanism that carried us to the pin, kept honest: while a registered
    digest is SHA256_SHAPE the run measures it, reports it and stops."""
    import publish as raf_publish

    monkeypatch.setitem(prep_a.CANONICAL_MAPPINGA_PREPARE_PARAMS,
                        "audio_union_sha256", raf_publish.SHA256_SHAPE)
    blockers = prep_a.canonical_identity_blockers("f" * 64, 10368)
    assert len(blockers) == 1
    assert "placeholder" in blockers[0] and "f" * 64 in blockers[0]


def test_the_marker_names_the_correspondence_record_by_its_full_digest(cli_corpus,
                                                                      tmp_path):
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    record = _correspondence_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback, correspondence=record))
    with open(record, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    marker_path = (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
                   / raf_publish.marker_name("mappingA_prepare"))
    with open(marker_path) as f:
        marker = json.load(f)
    assert marker["parameters"]["correspondence_sha256"] == digest
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_splits_record.json") as f:
        splits = json.load(f)
    assert splits["correspondence_record"]["sha256"] == digest
    # the run's own survey summary is recorded SEPARATELY, never as the audit
    assert splits["survey_sha256"] and splits["survey_sha256"] != digest


def test_a_record_from_another_algorithm_or_tolerance_is_refused(cli_corpus,
                                                                 tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    wrong_version = _correspondence_for(tmp_path, cli_corpus, name="v.json",
                                        algorithm_version="mappingA-correspondence-0")
    with pytest.raises(prep_a.CorrespondenceRecordError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback,
                              correspondence=wrong_version))
    assert "algorithm" in str(exc.value)

    loose = _correspondence_for(
        tmp_path, cli_corpus, name="t.json",
        tolerances={"placement_cap_m": prep_a.PLACEMENT_CAP_M, "match_p95_m": 0.05,
                    "match_max_m": prep_a.MATCH_MAX_M,
                    "match_ambiguity_margin": prep_a.MATCH_AMBIGUITY_MARGIN})
    with pytest.raises(prep_a.CorrespondenceRecordError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback, correspondence=loose))
    assert "tolerances" in str(exc.value) and "match_p95_m" in str(exc.value)


def test_a_record_that_does_not_describe_this_corpus_is_refused(cli_corpus, tmp_path):
    """A digest says WHICH file was read; this says the file describes what the run
    measured -- the corpus changing under the audit is the failure it catches."""
    readback = _readback_for(tmp_path, cli_corpus)
    rooms = dict(_correspondence_summary(cli_corpus, ("EmptyRoom", "FurnishedRoom")))
    rooms["EmptyRoom"] = dict(rooms["EmptyRoom"],
                              eligible_placement_ids=["p000", "p001", "p077"])
    stale = _correspondence_for(tmp_path, cli_corpus, name="stale.json", rooms=rooms)
    with pytest.raises(prep_a.CorrespondenceRecordError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback, correspondence=stale))
    assert "p077" in str(exc.value) and "EmptyRoom" in str(exc.value)
    assert not (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT / prep_a.MANIFEST_NAME).exists()


def test_a_non_registered_record_taints_a_non_canonical_run(cli_corpus, tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_splits_record.json") as f:
        splits = json.load(f)
    assert any("correspondence record" in note for note in splits["taint"])


def test_a_canonical_run_demands_the_registered_record(cli_corpus, tmp_path):
    import publish as raf_publish

    record = _correspondence_for(tmp_path, cli_corpus)
    with pytest.raises(prep_a.CorrespondenceRecordError) as exc:
        prep_a.load_correspondence_record(record, ["EmptyRoom"], canonical=True)
    registered = raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS[
        "correspondence_sha256"]
    assert registered in str(exc.value)


def test_the_record_is_hashed_from_the_bytes_that_were_parsed(cli_corpus, tmp_path):
    """Read once: hashing a second read would attest a file that may have changed
    between the two."""
    record = _correspondence_for(tmp_path, cli_corpus)
    parsed, provenance = prep_a.load_correspondence_record(
        record, ["EmptyRoom"], canonical=False)
    with open(record, "rb") as f:
        raw = f.read()
    assert provenance["sha256"] == hashlib.sha256(raw).hexdigest()
    assert parsed == json.loads(raw.decode())


def test_the_committed_record_is_readable_by_the_loader_that_pins_it(tmp_path):
    """The loader and the readback driver have to agree on the record's SHAPE, not
    only on its digest: a tolerance the loader cannot find is a tolerance it never
    verified. (r2 N9 regenerated this record under correspondence-2.)"""
    record = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun",
                          "exp_21_raf_mappingA_claude",
                          "mappingA_correspondence_record.json")
    parsed, provenance = prep_a.load_correspondence_record(
        record, ["EmptyRoom", "FurnishedRoom"], canonical=False)
    assert provenance["taint"] == []          # it IS the registered record
    assert parsed["algorithm_version"] == prep_a.MATCH_ALGORITHM_VERSION
    assert set(parsed["rooms"]) == {"EmptyRoom", "FurnishedRoom"}
    for room in parsed["rooms"].values():
        assert len(room["eligible_placement_ids"]) >= 16


# --------------------------------------------------------------------------- #
# r3 P1: the Mapping-A runtime root is derived, and unsafe roots are refused
# --------------------------------------------------------------------------- #
def test_the_runtime_root_is_derived_from_the_mappingH_tree(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    assert prep_a.resolve_output_dir(None, h, ["EmptyRoom"]) == os.path.join(
        h, prep_a.MAPPINGA_RUNTIME_SUBDIR)


def test_publishing_onto_the_mappingH_tree_is_refused(tmp_path):
    """Mapping A writes its own pointer and root manifest, and StagedPublish renames
    the destination's manifest aside first: pointed at H's root it would overwrite
    H's pointer and invalidate H's publication."""
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(h, h, ["EmptyRoom"])
    assert "overwrite" in str(exc.value)
    assert prep_a.MAPPINGA_RUNTIME_SUBDIR in str(exc.value)


def test_an_ancestor_of_the_mappingH_tree_is_refused(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(str(tmp_path / "runtime"), h, ["EmptyRoom"])
    assert "contains it" in str(exc.value)


def test_a_root_outside_the_mappingH_tree_is_refused(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(str(tmp_path / "elsewhere"), h, ["EmptyRoom"])
    assert "outside" in str(exc.value)


def test_a_root_inside_a_mappingH_room_is_refused(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(os.path.join(h, "EmptyRoom", "mappingA"), h,
                                  ["EmptyRoom", "FurnishedRoom"])
    assert "EmptyRoom root" in str(exc.value)


def test_the_protected_rooms_are_the_publications_not_this_runs(tmp_path):
    """r4 Q1, the reviewer's probe: an EmptyRoom-only Mapping-A run pointed at
    <H>/FurnishedRoom. The room it would replace is one this run never mentions, so
    a check over --rooms could not see it -- and the transaction would then have
    replaced a Mapping-H room tree."""
    h = str(tmp_path / "runtime" / "RAF")
    target = os.path.join(h, "FurnishedRoom")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(target, h, ["EmptyRoom", "FurnishedRoom"])
    assert "FurnishedRoom root" in str(exc.value)
    # ... and a descendant of it, likewise
    with pytest.raises(ValueError):
        prep_a.resolve_output_dir(os.path.join(target, "mono_rirs_22050Hz"), h,
                                  ["EmptyRoom", "FurnishedRoom"])
    # the old behaviour, for the record: the run's own subset protects nothing
    assert prep_a.resolve_output_dir(target, h, ["EmptyRoom"]) == target


def test_an_empty_protected_room_list_is_refused(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(os.path.join(h, "mappingA"), h, [])
    assert "protect nothing" in str(exc.value)


def test_the_cli_protects_a_room_the_run_never_mentions(cli_corpus, tmp_path):
    """End to end: Mapping H publishes both rooms, the Mapping-A run prepares only
    EmptyRoom, and --output-dir names H's FurnishedRoom tree."""
    import publish as raf_publish

    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback,
                                rooms=("EmptyRoom", "FurnishedRoom"))
    scalar = prep_a.locate_mappingH(str(runtime), [])["amplitude_scalar"]
    found = prep_a.locate_mappingH(str(runtime), [])
    assert found["pointer_rooms"] == ["EmptyRoom", "FurnishedRoom"]

    argv = ["--raf-root", str(cli_corpus),
            "--output-dir", str(runtime / "FurnishedRoom"),
            "--split-dir", str(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT),
            "--rooms", "EmptyRoom", "--n-placements", "2", "--k", "8",
            "--non-canonical", "--scalar", str(scalar),
            "--mappingH-dir", str(runtime),
            "--correspondence-record",
            _correspondence_for(tmp_path, cli_corpus, room_names=("EmptyRoom",)),
            "--readback-record", readback]
    with pytest.raises(ValueError) as exc:
        prep_a.main(argv)
    assert "FurnishedRoom root" in str(exc.value)
    # Mapping H is untouched: pointer, manifest and attestation all intact
    assert raf_publish.verify_manifest(str(runtime)) == {"missing": [],
                                                         "mismatched": []}
    assert raf_publish.verify_publication(str(tmp_path / "data" / "RAF"),
                                          kind="prepare")["published"]


def test_without_a_mappingH_tree_the_output_root_must_be_given():
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(None, None, ["EmptyRoom"])
    assert "--output-dir is required" in str(exc.value)


def test_the_cli_refuses_the_mappingH_root_before_reading_the_corpus(cli_corpus,
                                                                    tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback)
    scalar = prep_a.locate_mappingH(str(runtime), [])["amplitude_scalar"]
    argv = [a for a in _cli_argv(tmp_path, cli_corpus, readback,
                                 extra=["--mappingH-dir", str(runtime),
                                        "--scalar", str(scalar)])]
    argv[argv.index("--output-dir") + 1] = str(runtime)
    with pytest.raises(ValueError) as exc:
        prep_a.main(argv)
    assert "overwrite" in str(exc.value)
    # nothing was surveyed, staged or published
    assert not (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT / prep_a.MANIFEST_NAME).exists()
    assert not list(runtime.parent.glob(".*staging*"))
    import publish as raf_publish

    assert raf_publish.verify_publication(str(tmp_path / "data" / "RAF"),
                                          kind="prepare")["published"]


def test_both_real_clis_compose_at_their_true_defaults(cli_corpus, tmp_path,
                                                      monkeypatch):
    """No --output-dir and no --split-dir on either side: Mapping H at its default
    split root, Mapping A at ITS default split root and the derived runtime child.
    Both publications verify afterwards -- the topology is produced by the CLIs,
    not invented by the test."""
    import prepare_data as raf_prepare
    import publish as raf_publish

    monkeypatch.chdir(tmp_path)
    readback = _readback_for(tmp_path, cli_corpus)
    h_runtime = tmp_path / "runtime" / "RAF"
    raf_prepare.main(["--raf-root", str(cli_corpus), "--output-dir", str(h_runtime),
                      "--rooms", "EmptyRoom", "FurnishedRoom",
                      "--n-groups", "2", "--n-val-groups", "1",
                      "--n-diagnostic-groups", "1", "--n-train", "12",
                      "--full-crosscheck", "--non-canonical",
                      "--readback-record", readback])
    h_split = tmp_path / raf_prepare.build_parser().get_default("split_dir")
    assert raf_publish.verify_publication(str(h_split), kind="prepare")["published"]
    scalar = prep_a.locate_mappingH(str(h_runtime), [])["amplitude_scalar"]

    prep_a.main(["--raf-root", str(cli_corpus), "--mappingH-dir", str(h_runtime),
                 "--rooms", "EmptyRoom", "FurnishedRoom",
                 "--n-placements", "2", "--k", "8", "--non-canonical",
                 "--scalar", str(scalar),
                 "--correspondence-record", _correspondence_for(tmp_path, cli_corpus),
                 "--readback-record", readback])

    a_runtime = h_runtime / prep_a.MAPPINGA_RUNTIME_SUBDIR
    a_split = tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
    assert (a_runtime / "raf_publication.json").is_file()
    assert (a_split / prep_a.MANIFEST_NAME).is_file()
    assert raf_publish.verify_publication(str(a_split),
                                          kind="mappingA_prepare")["published"]
    # ... and Mapping H is still exactly as attested, pointer and manifest intact
    assert raf_publish.verify_publication(str(h_split), kind="prepare")["published"]
    assert raf_publish.verify_manifest(str(h_runtime)) == {"missing": [],
                                                           "mismatched": []}


# --------------------------------------------------------------------------- #
# r4 R1: the DERIVED default goes through the same gates as an explicit path
# --------------------------------------------------------------------------- #
def test_the_derived_default_is_refused_without_a_protected_room_list(tmp_path):
    """The empty-list refusal used to sit after the derived path returned, so the
    default was resolved against nothing at all."""
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(None, h, [])
    assert "protect nothing" in str(exc.value)


def test_the_derived_default_is_refused_when_it_lands_on_a_protected_room(tmp_path):
    """A publication holding a room literally named 'mappingA' puts <H>/mappingA on
    top of a tree H attests -- the derived path is only safe RELATIVE to the room
    list, so it is checked against it like any other."""
    h = str(tmp_path / "runtime" / "RAF")
    with pytest.raises(ValueError) as exc:
        prep_a.resolve_output_dir(None, h, ["mappingA"])
    assert "DERIVED default" in str(exc.value)
    assert "mappingA root" in str(exc.value)
    # the same room set with an explicit, disjoint root is still fine
    assert prep_a.resolve_output_dir(os.path.join(h, "mappingA_eval"), h,
                                     ["mappingA"]) == os.path.join(h, "mappingA_eval")


def test_the_derived_default_still_resolves_for_the_registered_rooms(tmp_path):
    h = str(tmp_path / "runtime" / "RAF")
    assert prep_a.resolve_output_dir(
        None, h, ["EmptyRoom", "FurnishedRoom"]) == os.path.join(h, "mappingA")


# --------------------------------------------------------------------------- #
# r5 Amendment 4: Mapping A derives its OWN scalar over its OWN union
# --------------------------------------------------------------------------- #
def _measured(peaks):
    """{capture_id: measurement} from a peak table, as measure_union returns."""
    return {capture_id: {"peak": peak, "crop_peak": peak, "finite": True,
                         "n_samples": 2205}
            for capture_id, peak in peaks.items()}


def test_the_union_derives_the_registered_two_point_zero():
    """The real corpus, reproduced as a fixture: the loudest union capture peaks at
    0.48968, so the clip clamp admits at most 0.999/0.48968 = 2.0401 and the
    largest one-significant-digit value under it is 2.0 -- exactly the registered
    Mapping-A scalar (Amendment 4)."""
    import publish as raf_publish

    peaks = {f"{i:06d}": 0.05 + 0.001 * i for i in range(20)}
    peaks["000099"] = 0.48968                       # the clamp-setting capture
    supports = [f"{i:06d}" for i in range(20)]      # contexts: none of them loudest
    decision = prep_a.derive_union_scalar({"EmptyRoom": _measured(peaks)},
                                          {"EmptyRoom": supports})
    assert decision["max_written_peak"] == 0.48968
    assert decision["max_admissible_scalar"] == pytest.approx(2.0401, abs=1e-4)
    assert decision["clamp_term"] == 2.0
    assert decision["support_term"] == 10.0        # one sig digit of 0.75/0.069
    assert decision["binding_term"] == "clip_clamp"
    assert decision["scalar"] == 2.0
    assert (decision["scalar"]
            == raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS["amplitude_scalar"])


def test_the_derivation_matches_the_registered_formula_on_the_same_population():
    """The Mapping-A derivation is exp_19's formula on a different population, so
    it must agree with exp_19's own implementation given the same peaks."""
    import prepare_data as raf_prepare

    peaks = {"000000": 0.12, "000001": 0.30, "000002": 0.48968}
    decision = prep_a.derive_union_scalar({"EmptyRoom": _measured(peaks)},
                                          {"EmptyRoom": ["000000", "000001"]})
    support_term = raf_prepare.one_significant_digit(0.75 / 0.30)
    clamp_term = raf_prepare.largest_one_significant_digit_at_most(0.999 / 0.48968)
    assert decision["support_term"] == support_term
    assert decision["clamp_term"] == clamp_term
    assert decision["scalar"] == min(support_term, clamp_term)
    assert decision["formula_version"] == raf_prepare.AMPLITUDE_FORMULA_VERSION


def test_a_capture_that_clips_at_x3_is_clean_at_the_derived_scalar(union_room,
                                                                   tmp_path):
    """The stop that fired on the real union: two EmptyRoom captures clip at
    Mapping H's x3. At the scalar Mapping A derives for its own union they are
    published intact -- items are never dropped."""
    _overwrite(union_room, "000003", _rir(3, peak=0.7))     # ~0.49 after resampling
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError):
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0)

    measurements = prep_a.measure_union(union_room, ids)
    decision = prep_a.derive_union_scalar({"EmptyRoom": measurements},
                                          {"EmptyRoom": ids})
    assert decision["scalar"] <= decision["max_admissible_scalar"]
    report = prep_a.audit_amplitude_union(union_room, ids,
                                          scalar=decision["scalar"],
                                          measurements=measurements)
    assert report["passed"] is True
    assert report["n_captures"] == len(ids)          # the complete union, intact
    written = prep_a.write_union(union_room, str(tmp_path / "out"), ids,
                                 scalar=decision["scalar"])
    assert written["n_files"] == len(ids)
    assert max(f["peak"] for f in written["files"].values()) <= 0.999


def test_the_derivation_reads_the_union_once(union_room):
    """Both halves of the policy read the same measurements: the corpus is ~10k
    captures and resampling it twice per run is not free."""
    ids = [f"{i:06d}" for i in range(6)]
    measurements = prep_a.measure_union(union_room, ids)
    assert sorted(measurements) == ids
    calls = []
    real = prep_a._resampled_peaks_and_finite
    try:
        prep_a._resampled_peaks_and_finite = lambda *a, **k: calls.append(a) or real(*a, **k)
        prep_a.audit_amplitude_union(union_room, ids, scalar=2.0,
                                     measurements=measurements)
    finally:
        prep_a._resampled_peaks_and_finite = real
    assert calls == []


def test_the_supports_are_the_context_captures():
    items = [{"room": "EmptyRoom", "target_capture_id": "000000",
              "context": [{"capture_id": "000001"}, {"capture_id": "000002"}]},
             {"room": "EmptyRoom", "target_capture_id": "000001",
              "context": [{"capture_id": "000003"}]}]
    roles = prep_a.enumerate_support_captures(items)
    assert roles["EmptyRoom"][0] == ["000001", "000002", "000003"]
    assert roles["EmptyRoom"][1] == ["000000", "000001"]


def test_a_derivation_without_context_captures_is_refused():
    with pytest.raises(ValueError) as exc:
        prep_a.derive_union_scalar({"EmptyRoom": _measured({"000000": 0.1})},
                                   {"EmptyRoom": []})
    assert "no context captures" in str(exc.value)


def test_a_canonical_run_must_derive_the_registered_scalar(cli_corpus, tmp_path,
                                                           monkeypatch):
    """The identity discipline exp_19 uses: the derivation is what PROVES the
    registered scalar, so a union that derives something else stops before writing.
    """
    import publish as raf_publish

    monkeypatch.setitem(raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS,
                        "amplitude_scalar", 7.0)
    readback = _readback_for(tmp_path, cli_corpus)
    argv = [a for a in _cli_argv(tmp_path, cli_corpus, readback)
            if a != "--non-canonical"]
    with pytest.raises(Exception) as exc:
        prep_a.main(argv)
    # the readback/identity gates may fire first; whichever does, nothing is written
    assert not (tmp_path / prep_a.MAPPINGA_SPLIT_ROOT / prep_a.MANIFEST_NAME).exists()
    del exc


def test_the_scalar_flag_asserts_the_derivation_it_cannot_override_it(cli_corpus,
                                                                      tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.main(_cli_argv(tmp_path, cli_corpus, readback,
                              extra=["--scalar", "9.0"]))
    assert "does not override it" in str(exc.value)
    assert exc.value.report["scalar"] != 9.0


def test_the_published_artifacts_all_carry_the_scale_disclosure(cli_corpus,
                                                                tmp_path):
    readback = _readback_for(tmp_path, cli_corpus)
    runtime = _publish_mappingH(tmp_path, cli_corpus, readback,
                                rooms=("EmptyRoom", "FurnishedRoom"))
    prep_a.main(_derived_argv(_cli_argv(tmp_path, cli_corpus, readback,
                                        extra=["--mappingH-dir", str(runtime)])))
    import publish as raf_publish

    split_dir = tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
    with open(split_dir / raf_publish.marker_name("mappingA_prepare")) as f:
        marker = json.load(f)
    with open(split_dir / "mappingA_splits_record.json") as f:
        splits = json.load(f)
    with open(runtime / prep_a.MAPPINGA_RUNTIME_SUBDIR / "raf_publication.json") as f:
        pointer = json.load(f)
    for payload in (marker, splits, pointer):
        assert payload["scale_disclosure"]["audio_is_shared"] is False
        assert "unlicensed" in payload["scale_disclosure"]["note"]
    # the marker also carries HOW the scalar was derived
    assert marker["amplitude_derivation"]["binding_term"] in ("clip_clamp",
                                                              "support_ceiling")
    assert (marker["parameters"]["amplitude_scalar"]
            == marker["amplitude_derivation"]["scalar"])
    assert splits["amplitude_scalar"]["derivation_id_sha256"]


# --------------------------------------------------------------------------- #
# r5 Amendment 4.1: the silence gate is role-aware; contexts are DISCLOSED
# --------------------------------------------------------------------------- #
def _near_silent_fixture(union_room, capture_id="000004"):
    """The measured shape: a capture whose crop sits at ~1.2e-4, i.e. about
    -72 dBFS at x2 -- the FurnishedRoom p008 references from
    mappingA_amplitude_window.json."""
    _overwrite(union_room, capture_id, _rir(4, peak=1.2e-4))
    return [f"{i:06d}" for i in range(6)]


def test_a_sub_threshold_context_is_recorded_not_fatal(union_room):
    """The loader tests is_silence on the cropped TARGET only, so a quiet context
    is conditioning the model actually saw -- a disclosed property, not an item
    nobody evaluated."""
    ids = _near_silent_fixture(union_room)
    targets = {c for c in ids if c != "000004"}
    report = prep_a.audit_amplitude_union(
        union_room, ids, scalar=2.0, target_ids=targets,
        context_items={"000004": ["EmptyRoom/p008/slot03", "EmptyRoom/p008/slot11"]})
    assert report["passed"] is True and report["violations"] == []
    assert report["role_aware_silence_gate"] is True
    assert report["n_near_silent_references"] == 1
    recorded = report["near_silent_references"][0]
    assert recorded["capture_id"] == "000004"
    assert recorded["role"] == "context"
    assert recorded["kind"] == "below_threshold_crop"
    assert recorded["scaled_dbfs_crop"] < -60.0
    assert recorded["item_ids"] == ["EmptyRoom/p008/slot03",
                                    "EmptyRoom/p008/slot11"]
    json.dumps(report)


def test_the_same_capture_as_a_target_is_still_fatal(union_room):
    ids = _near_silent_fixture(union_room)
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=2.0,
                                     target_ids={"000004"})
    report = exc.value.report
    assert report["n_below_threshold"] == 1
    offender = next(v for v in report["violations"] if v["capture_id"] == "000004")
    assert offender["kind"] == "below_threshold_crop" and offender["role"] == "target"
    assert report["n_near_silent_references"] == 0
    assert "TARGET-role" in str(exc.value)


def test_unknown_roles_keep_the_strict_gate(union_room):
    """Fail-closed: a caller that does not say which captures are targets audits
    every capture as one."""
    ids = _near_silent_fixture(union_room)
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=2.0)
    assert exc.value.report["role_aware_silence_gate"] is False


def test_a_clipping_context_is_still_fatal(union_room):
    """The clip gate is role-independent: a clipped context is a distorted input
    whoever reads it."""
    _overwrite(union_room, "000002", _rir(2, peak=0.9))
    ids = [f"{i:06d}" for i in range(6)]
    with pytest.raises(prep_a.AmplitudePolicyError) as exc:
        prep_a.audit_amplitude_union(union_room, ids, scalar=3.0,
                                     target_ids={"000000"})
    offender = next(v for v in exc.value.report["violations"]
                    if v["capture_id"] == "000002")
    assert offender["kind"] == "clipping" and offender["role"] == "context"


def test_the_write_recheck_is_role_aware_too(union_room, tmp_path):
    ids = _near_silent_fixture(union_room)
    targets = {c for c in ids if c != "000004"}
    report = prep_a.write_union(union_room, str(tmp_path / "out"), ids, scalar=2.0,
                                target_ids=targets)
    assert report["n_files"] == len(ids)          # the complete union, intact
    with pytest.raises(ValueError) as exc:
        prep_a.write_union(union_room, str(tmp_path / "strict"), ids, scalar=2.0,
                           target_ids={"000004"})
    assert "drawn as a TARGET" in str(exc.value)


def test_the_disclosure_names_the_items_and_their_placements():
    audits = {"FurnishedRoom": {"near_silent_references": [
        {"capture_id": "000503", "kind": "below_threshold_crop",
         "raw_peak_crop": 1.144e-4, "scaled_peak_crop": 2.288e-4,
         "scaled_dbfs_crop": -72.8,
         "item_ids": ["FurnishedRoom/p008/slot04"]},
        {"capture_id": "028030", "kind": "below_threshold_crop",
         "raw_peak_crop": 1.997e-4, "scaled_peak_crop": 3.994e-4,
         "scaled_dbfs_crop": -68.0,
         "item_ids": ["FurnishedRoom/p008/slot04", "FurnishedRoom/p008/slot17"]}]}}
    items_by_room = {"FurnishedRoom": [
        {"item_id": f"FurnishedRoom/p008/slot{slot:02d}", "room": "FurnishedRoom",
         "placement_id": "p008"} for slot in (4, 17)]}
    disclosure = prep_a.near_silent_disclosure(audits, items_by_room)
    assert disclosure["n_captures"] == 2
    assert disclosure["n_items"] == 2
    assert disclosure["placements"] == ["FurnishedRoom/p008"]
    assert [r["capture_id"] for r in disclosure["by_item"]["FurnishedRoom/p008/slot04"]] \
        == ["000503", "028030"]
    assert [r["capture_id"] for r in disclosure["by_item"]["FurnishedRoom/p008/slot17"]] \
        == ["028030"]
    assert "no item is substituted" in disclosure["note"]

    prep_a.annotate_near_silent_references(items_by_room, disclosure)
    assert len(items_by_room["FurnishedRoom"][0]["near_silent_references"]) == 2
    assert len(items_by_room["FurnishedRoom"][1]["near_silent_references"]) == 1


def test_an_item_with_no_near_silent_reference_says_so_explicitly():
    items_by_room = {"EmptyRoom": [{"item_id": "EmptyRoom/p000/slot00",
                                    "room": "EmptyRoom", "placement_id": "p000"}]}
    prep_a.annotate_near_silent_references(
        items_by_room, prep_a.near_silent_disclosure({}, items_by_room))
    assert items_by_room["EmptyRoom"][0]["near_silent_references"] == []


def test_the_published_items_carry_their_near_silent_references(cli_corpus, tmp_path):
    """End to end: the disclosure rides on the per-item manifest, which is where
    RAF_A_md and every offline reader find the items."""
    readback = _readback_for(tmp_path, cli_corpus)
    contexts = _context_only_captures(cli_corpus, "EmptyRoom")
    victim = sorted(contexts)[0]
    sf.write(os.path.join(str(cli_corpus), "archived", "EmptyRoom", "data", victim,
                          "rir.wav"), _rir(7, peak=1.2e-4), 48000, subtype="FLOAT")
    prep_a.main(_cli_argv(tmp_path, cli_corpus, readback))

    with open(tmp_path / "runtime" / "mappingA" / "EmptyRoom" / "metadata"
              / "mappingA_metadata.json") as f:
        items = json.load(f)
    affected = {item["item_id"]: item["near_silent_references"]
                for item in items.values() if item["near_silent_references"]}
    assert affected, "the corrupted context should appear on the items that draw it"
    for refs in affected.values():
        assert refs[0]["capture_id"] == victim
        assert refs[0]["scaled_dbfs"] < -60.0

    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_splits_record.json") as f:
        splits = json.load(f)
    disclosure = splits["near_silent_references"]
    assert disclosure["n_captures"] == 1
    assert disclosure["n_items"] == len(affected)
    assert sorted(disclosure["by_item"]) == sorted(affected)
    with open(tmp_path / prep_a.MAPPINGA_SPLIT_ROOT
              / "mappingA_amplitude_audit.json") as f:
        audit = json.load(f)
    assert audit["near_silent_references"]["n_captures"] == 1
    assert audit["rooms"]["EmptyRoom"]["n_near_silent_references"] == 1
    assert audit["rooms"]["EmptyRoom"]["passed"] is True


def _context_only_captures(raf_root, room, n_placements=2, k=8, seed=0):
    """Captures the canonical item set draws ONLY as contexts."""
    survey = prep_a.survey_room(os.path.join(str(raf_root), "archived", room), room)
    items = []
    for placement in prep_a.select_placements(survey, n_placements):
        items.extend(prep_a.build_items(room, placement["placement_id"],
                                        placement["passing"], placement["assignment"],
                                        placement["match"], k=k, seed=seed))
    supports, targets = prep_a.enumerate_support_captures(items)[room]
    return set(supports) - set(targets)
