"""Tests for ``data/RAF/readback_audit.py`` and the publish gates (exp_19 r2).

Codex R4 (+ R13's decision inputs), contracts Amendment 2: the code labelled the
quaternion order and the gauge "unverified/candidate" while nothing gated on
resolving them, and headline T60 was enabled unconditionally. This adds the
reproducible audit artifact and makes canonical preparation/rendering refuse to
run without a passing one.

Oracles here are hand-derived: onsets are placed at literal sample indices and
the delay fit is over exactly ``t = d / 343 + delay``.
"""
import json
import math
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import prepare_data as raf_prepare  # noqa: E402
import readback_audit as raf_readback  # noqa: E402
import render_depth as raf_render  # noqa: E402

from test_raf_prepare_data import (  # noqa: E402
    N_MICS, _default_groups, write_room,  # noqa: F401
)

assert os.path.dirname(os.path.abspath(raf_readback.__file__)) == _RAF_DIR

SPEED_OF_SOUND = 343.0


# --------------------------------------------------------------------------- #
# onset detection
# --------------------------------------------------------------------------- #
def _impulse(n=4800, at=100, amp=0.01):
    x = np.zeros(n, dtype=np.float32)
    x[at] = amp
    x[at + 5:at + 50] = amp * 0.1
    return x


def test_detect_onset_finds_the_impulse_sample():
    assert raf_readback.detect_onset(_impulse(at=137), threshold_db=-20.0) == 137


def test_detect_onset_ignores_low_level_noise_before_the_impulse():
    x = _impulse(at=500)
    rng = np.random.default_rng(0)
    x[:500] = rng.normal(size=500).astype(np.float32) * 1e-5   # -60 dB re peak
    assert raf_readback.detect_onset(x, threshold_db=-20.0) == 500


def test_detect_onset_is_fail_closed_on_silence():
    with pytest.raises(ValueError):
        raf_readback.detect_onset(np.zeros(100, dtype=np.float32))


# --------------------------------------------------------------------------- #
# constant-delay fit
# --------------------------------------------------------------------------- #
def _exact_onsets(distances, delay=0.004):
    return [d / SPEED_OF_SOUND + delay for d in distances]


def test_delay_fit_recovers_the_speed_of_sound_and_the_constant_delay():
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    fit = raf_readback.fit_constant_delay(d, _exact_onsets(d, delay=0.004))
    assert fit["slope_s_per_m"] == pytest.approx(1.0 / SPEED_OF_SOUND, rel=1e-9)
    assert fit["intercept_s"] == pytest.approx(0.004, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["slope_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert fit["passed"] is True


def test_delay_fit_fails_when_the_slope_is_off_by_more_than_20_percent():
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    onsets = [x * 1.25 for x in _exact_onsets(d, delay=0.0)]   # slope +25%
    fit = raf_readback.fit_constant_delay(d, onsets)
    assert fit["slope_ratio"] == pytest.approx(1.25, rel=1e-6)
    assert fit["passed"] is False
    assert "slope" in " ".join(fit["reasons"])


def test_delay_fit_fails_when_the_relationship_is_noise():
    rng = np.random.default_rng(1)
    d = list(np.linspace(1.0, 5.0, 40))
    onsets = list(rng.normal(size=40) * 0.01 + 0.01)
    fit = raf_readback.fit_constant_delay(d, onsets)
    assert fit["r2"] < 0.8
    assert fit["passed"] is False
    assert "r2" in " ".join(fit["reasons"])


def test_delay_fit_needs_enough_points_and_spread():
    with pytest.raises(ValueError):
        raf_readback.fit_constant_delay([1.0], [0.003])
    with pytest.raises(ValueError):
        raf_readback.fit_constant_delay([2.0] * 5, [0.006] * 5)   # no spread


def test_delay_fit_values_are_json_safe():
    d = [1.0, 2.0, 3.0]
    json.dumps(raf_readback.fit_constant_delay(d, _exact_onsets(d)), allow_nan=False)


# --------------------------------------------------------------------------- #
# crop-vs-full T30 validity
# --------------------------------------------------------------------------- #
def _decaying(n=33075, sr=22050, tau=0.08, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    return (rng.normal(size=n) * np.exp(-t / tau)).astype(np.float32) * 0.01


def test_t30_validity_counts_crop_and_full_separately():
    waves = [_decaying(seed=i) for i in range(3)] + [np.zeros(33075, dtype=np.float32)]
    report = raf_readback.t30_validity(waves, sr=22050, crop=10240)
    assert report["n"] == 4
    assert report["decay_db"] == 30
    assert report["crop_samples"] == 10240
    assert report["valid_full"] >= 3
    assert report["invalid_full"] == 4 - report["valid_full"]
    assert 0.0 <= report["valid_rate_crop"] <= 1.0
    assert report["crop_invalidates"] == \
        len([1 for a, b in zip(report["per_item_full"], report["per_item_crop"])
             if a and not b])
    json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------- #
# quaternion order diagnostics
# --------------------------------------------------------------------------- #
def test_quaternion_forward_diagnostics_distinguish_the_two_readings():
    """q = (0.70710678, 0, 0, 0.70710678), hand-derived under both readings.

    Read as wxyz: w = cos(45 deg), and the imaginary part is the THIRD raw axis,
    RAF Z (left). A +90 deg rotation about it takes RAF forward (1,0,0) -> (0,1,0),
    i.e. the source ends up pointing at the ceiling (RAF Y is up).
    Read as xyzw: the imaginary part is the FIRST raw axis, RAF X, and a rotation
    about the forward axis leaves forward (1,0,0) fixed.

    That is exactly the diagnostic value: one reading tilts every source 90 deg
    out of the room, the other does not.
    """
    q = [0.70710678, 0.0, 0.0, 0.70710678]
    diag = raf_readback.quaternion_forward_diagnostics(q)
    np.testing.assert_allclose(diag["wxyz"]["forward_raf"], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(diag["xyzw"]["forward_raf"], [1.0, 0.0, 0.0], atol=1e-6)
    # RAF (0,1,0) -> pipeline (X, Z, Y) = (0,0,1): straight up, the pipeline's
    # vertical axis being the third component
    np.testing.assert_allclose(diag["wxyz"]["forward_pipeline"], [0.0, 0.0, 1.0], atol=1e-6)
    assert diag["wxyz"]["elevation_deg"] == pytest.approx(90.0, abs=1e-4)
    assert diag["xyzw"]["elevation_deg"] == pytest.approx(0.0, abs=1e-4)
    assert diag["identity_quat_reading"] in ("wxyz", "xyzw", "ambiguous")


def test_quaternion_diagnostics_are_json_safe():
    json.dumps(raf_readback.quaternion_forward_diagnostics([1.0, 0.0, 0.0, 0.0]),
               allow_nan=False)


# --------------------------------------------------------------------------- #
# the record + the gate
# --------------------------------------------------------------------------- #
@pytest.fixture
def audited_room(tmp_path):
    """A synthetic room whose onsets really are distance/343 + a constant delay."""
    room = "EmptyRoom"
    write_room(str(tmp_path), room, groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", room)
    index = raf_prepare.load_room_index(room_dir)
    sr = 48000
    for record in index:
        d = float(np.linalg.norm(record["rx_xyz"] - record["tx_xyz"]))
        at = int(round((d / SPEED_OF_SOUND + 0.004) * sr))
        sf.write(os.path.join(room_dir, "data", record["capture_id"], "rir.wav"),
                 _impulse(n=72000, at=at), sr, subtype="FLOAT")
    return str(tmp_path), room_dir


def test_readback_record_structure_and_verdict(audited_room, tmp_path):
    raf_root, _ = audited_room
    out = tmp_path / "raf_readback_record.json"
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom",
                       "--out", str(out), "--n-onset-samples", "36"])
    with open(out) as f:
        record = json.load(f)
    assert record["schema_version"] == raf_readback.RECORD_SCHEMA_VERSION
    room = record["rooms"]["EmptyRoom"]
    assert room["onset"]["passed"] is True
    assert room["onset"]["slope_ratio"] == pytest.approx(1.0, abs=0.2)
    assert room["t30_validity"]["n"] > 0
    assert room["amplitude"]["peak_stats"]["count"] > 0
    assert "wxyz" in room["quaternion"] and "xyzw" in room["quaternion"]
    assert record["decisions"]["t60_headline"]["resolution"] in ("headline", "demoted")
    assert record["decisions"]["amplitude_scalar"]["derived_from"] == "train supports only"
    assert record["adjudication"] == {"gauge_pinned": None, "quat_order_pinned": None}
    assert record["verdict"]["passed"] is True
    with open(out) as f:                      # strict JSON: no NaN/Infinity tokens
        json.loads(f.read(), parse_constant=_reject_constant)


def _reject_constant(name):
    raise AssertionError(f"non-standard JSON constant in the record: {name}")


def test_readback_record_can_pin_the_gauge_and_quaternion_order(audited_room, tmp_path):
    raf_root, _ = audited_room
    out = tmp_path / "rec.json"
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom", "--out", str(out),
                       "--n-onset-samples", "36", "--pin-gauge", "RAF_TO_PIPELINE",
                       "--pin-quat", "wxyz"])
    with open(out) as f:
        record = json.load(f)
    assert record["adjudication"] == {"gauge_pinned": "RAF_TO_PIPELINE",
                                      "quat_order_pinned": "wxyz"}


def _pinned_record(tmp_path, audited_room, name="rec.json"):
    raf_root, _ = audited_room
    out = tmp_path / name
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom", "--out", str(out),
                       "--n-onset-samples", "36", "--pin-gauge", "RAF_TO_PIPELINE",
                       "--pin-quat", "wxyz"])
    return str(out)


def test_gate_accepts_a_passing_pinned_record(audited_room, tmp_path):
    path = _pinned_record(tmp_path, audited_room)
    record = raf_readback.load_passing_record(path, canonical=False)
    assert record["verdict"]["passed"] is True


def test_gate_rejects_a_missing_record(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        raf_readback.load_passing_record(str(tmp_path / "nope.json"), canonical=False)


def test_gate_rejects_an_unpinned_record(audited_room, tmp_path):
    raf_root, _ = audited_room
    out = tmp_path / "unpinned.json"
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom", "--out", str(out),
                       "--n-onset-samples", "36"])
    with pytest.raises(ValueError) as exc:
        raf_readback.load_passing_record(str(out), canonical=False)
    assert "gauge" in str(exc.value) or "quat" in str(exc.value)


def test_gate_rejects_a_failed_record(audited_room, tmp_path):
    path = _pinned_record(tmp_path, audited_room)
    with open(path) as f:
        record = json.load(f)
    record["verdict"] = {"passed": False, "reasons": ["onset fit failed"]}
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(ValueError):
        raf_readback.load_passing_record(path, canonical=False)


def test_gate_rejects_a_foreign_schema(audited_room, tmp_path):
    path = _pinned_record(tmp_path, audited_room)
    with open(path) as f:
        record = json.load(f)
    record["schema_version"] = raf_readback.RECORD_SCHEMA_VERSION + 1
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(ValueError):
        raf_readback.load_passing_record(path, canonical=False)


# --------------------------------------------------------------------------- #
# publish gates
# --------------------------------------------------------------------------- #
def _prepare_argv(tmp_path, raf_root, readback=None, non_canonical=True):
    argv = ["--raf-root", str(raf_root), "--output-dir", str(tmp_path / "runtime" / "RAF"),
            "--split-dir", str(tmp_path / "splits"), "--rooms", "EmptyRoom",
            "--n-groups", "1", "--n-val-groups", "1", "--n-diagnostic-groups", "1",
            "--n-train", "12", "--full-crosscheck"]
    if non_canonical:
        argv = argv + ["--non-canonical"]
    if readback is not None:
        argv += ["--readback-record", readback]
    return argv


def test_prepare_refuses_to_publish_without_a_readback_record(tmp_path):
    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    with pytest.raises(SystemExit):
        raf_prepare.main(_prepare_argv(tmp_path, raf_root))


def test_prepare_refuses_a_failed_readback_record(tmp_path, audited_room):
    path = _pinned_record(tmp_path, audited_room)
    with open(path) as f:
        record = json.load(f)
    record["verdict"]["passed"] = False
    with open(path, "w") as f:
        json.dump(record, f)
    raf_root = tmp_path / "raf2"
    write_room(str(raf_root), "EmptyRoom")
    with pytest.raises(ValueError):
        raf_prepare.main(_prepare_argv(tmp_path, raf_root, readback=path))


def test_prepare_records_the_readback_provenance(tmp_path, audited_room):
    path = _pinned_record(tmp_path, audited_room)
    raf_root = tmp_path / "raf3"
    write_room(str(raf_root), "EmptyRoom")
    raf_prepare.main(_prepare_argv(tmp_path, raf_root, readback=path))
    with open(tmp_path / "splits" / "raf_splits_record.json") as f:
        record = json.load(f)
    provenance = record["readback_record"]
    assert provenance["path"] == os.path.abspath(path)
    assert len(provenance["sha256"]) == 64
    assert provenance["gauge_pinned"] == "RAF_TO_PIPELINE"
    assert provenance["quat_order_pinned"] == "wxyz"


def test_render_depth_refuses_canonical_mode_without_a_readback_record(tmp_path):
    from test_raf_render_depth import _write_fixture

    raf_root, out, _ = _write_fixture(tmp_path)
    with pytest.raises(SystemExit):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom"])


def test_render_depth_runs_with_a_passing_record(tmp_path, audited_room):
    from test_raf_render_depth import _write_fixture

    path = _pinned_record(tmp_path, audited_room)
    raf_root, out, groups = _write_fixture(tmp_path)
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", path,
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["readback_record"]["gauge_pinned"] == "RAF_TO_PIPELINE"


# --------------------------------------------------------------------------- #
# R13: amplitude audit
# --------------------------------------------------------------------------- #
def test_amplitude_audit_reads_back_every_written_wav(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ["000000", "000001"])
    for entry in audit["files"].values():
        assert entry["roundtrip_max_abs_error"] == 0.0    # float32 WAV is exact
        assert entry["roundtrip_samples"] == entry["n_samples"]
    assert audit["roundtrip_max_abs_error"] == 0.0


def test_amplitude_audit_is_strict_json_safe_for_a_silent_file(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.0)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ["000000"])
    entry = audit["files"]["000000"]
    assert entry["peak"] == 0.0
    assert entry["dbfs"] == raf_prepare.DBFS_FLOOR
    assert math.isfinite(entry["dbfs"])
    json.dumps(audit, allow_nan=False)


def test_amplitude_audit_separates_roles(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    roles = {f"{i:06d}": ("support" if i < 4 else "test") for i in range(8)}
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"),
        [f"{i:06d}" for i in range(8)], roles=roles)
    assert set(audit["by_role"]) == {"support", "test"}
    assert audit["by_role"]["support"]["count"] == 4
    assert audit["by_role"]["test"]["count"] == 4
    for cid, entry in audit["files"].items():
        assert entry["role"] == roles[cid]


def test_amplitude_audit_scale_decision_uses_train_supports_only(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    roles = {f"{i:06d}": ("train" if i < 4 else "test") for i in range(8)}
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"),
        [f"{i:06d}" for i in range(8)], roles=roles)
    decision = audit["scale_decision"]
    assert decision["derived_from"] == "trained supports only (split_role == 'train')"
    assert decision["applied_scalar"] is None
    assert decision["train_support_peak_median"] == pytest.approx(
        float(np.median([audit["files"][f"{i:06d}"]["peak"] for i in range(4)])))
    assert set(audit["comparison"]) == {"HAA", "AR", "note"}


def test_amplitude_audit_applies_a_scalar_uniformly_when_one_is_given(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    roles = {"000000": "train", "000001": "train"}
    plain = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "a" / "EmptyRoom"), ["000000", "000001"], roles=roles)
    scaled = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "b" / "EmptyRoom"), ["000000", "000001"], roles=roles,
        scale=2.0,
        scale_provenance=raf_prepare.derivation_id_hash(["000000", "000001"]))
    assert scaled["scale_decision"]["applied_scalar"] == 2.0
    for cid in ("000000", "000001"):
        assert scaled["files"][cid]["peak"] == pytest.approx(
            2.0 * plain["files"][cid]["peak"], rel=1e-6)


# --------------------------------------------------------------------------- #
# r3 S1: canonical publication authenticates the pinned record
# --------------------------------------------------------------------------- #
_PINNED_RECORD = os.path.join(_REPO_ROOT, "data", "RAF", "raf_readback_record.json")
_PINNED_SHA256 = "e879768f8b4a152fb79db670e31a211165bcbaff6746bed64ac1f8a6aec0f01e"


def test_the_committed_record_is_the_pinned_one():
    import hashlib

    with open(_PINNED_RECORD, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == _PINNED_SHA256
    assert raf_readback.CANONICAL_RECORD_SHA256 == _PINNED_SHA256
    assert raf_readback.CANONICAL_GAUGE == "RAF_TO_PIPELINE:(X,Z,Y)"
    assert raf_readback.CANONICAL_QUAT_ORDER == "xyzw"


def test_canonical_gate_accepts_the_pinned_record():
    record = raf_readback.load_passing_record(_PINNED_RECORD, canonical=True)
    assert record["adjudication"]["gauge_pinned"] == raf_readback.CANONICAL_GAUGE
    assert record["adjudication"]["quat_order_pinned"] == raf_readback.CANONICAL_QUAT_ORDER
    assert set(record["rooms"]) == set(raf_readback.CANONICAL_ROOMS)


def test_canonical_gate_rejects_a_synthetic_record(tmp_path):
    """The exact bypass the review demonstrated: rooms={}, no measurements."""
    from test_raf_prepare_data import write_passing_readback_record

    path = write_passing_readback_record(str(tmp_path / "synthetic.json"))
    with pytest.raises(ValueError) as exc:
        raf_readback.load_passing_record(path, canonical=True)
    assert "sha256" in str(exc.value).lower()
    # ... while the explicitly non-canonical path still accepts it, tainted
    record = raf_readback.load_passing_record(path, canonical=False)
    assert record["verdict"]["passed"] is True


def _tampered(tmp_path, mutate, name="tampered.json"):
    with open(_PINNED_RECORD) as f:
        record = json.load(f)
    mutate(record)
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(record, f, indent=4)
    return str(path)


def _drop_room(record):
    record["rooms"].pop("FurnishedRoom")


def _superseded_quat(record):
    record["adjudication"]["quat_order_pinned"] = "wxyz"


def _unpin_gauge(record):
    record["adjudication"]["gauge_pinned"] = "RAF_TO_PIPELINE"


def _drop_onset(record):
    record["rooms"]["EmptyRoom"].pop("onset")


def _fail_onset(record):
    record["rooms"]["EmptyRoom"]["onset"]["passed"] = False


@pytest.mark.parametrize("mutate,needle", [
    (_drop_room, "room"),
    (_superseded_quat, "quat"),
    (_unpin_gauge, "gauge"),
    (_drop_onset, "onset"),
    (_fail_onset, "onset"),
])
def test_canonical_gate_rejects_a_tampered_record(tmp_path, mutate, needle):
    """Every one of these still has verdict.passed=true and two non-empty pins."""
    path = _tampered(tmp_path, mutate)
    with pytest.raises(ValueError) as exc:
        raf_readback.load_passing_record(path, canonical=True)
    message = str(exc.value).lower()
    assert "sha256" in message or needle in message


def test_canonical_gate_checks_the_content_not_only_the_hash(tmp_path):
    """A record that hashes differently is rejected on the hash; one that somehow
    matched must still satisfy every content rule, so the rules are exercised with
    the hash check disabled."""
    path = _tampered(tmp_path, _superseded_quat)
    with pytest.raises(ValueError) as exc:
        raf_readback.assert_canonical_content(json.load(open(path)), path)
    assert "quat" in str(exc.value).lower()


def test_canonical_gate_requires_a_matching_raf_root(tmp_path):
    with pytest.raises(ValueError) as exc:
        raf_readback.load_passing_record(_PINNED_RECORD, canonical=True,
                                         expected_raf_root="/somewhere/else")
    assert "raf_root" in str(exc.value)
    raf_readback.load_passing_record(
        _PINNED_RECORD, canonical=True,
        expected_raf_root="/media/diskstation/yixunhu/raf_dataset")


def test_provenance_records_canonicality_and_taint(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record

    record = raf_readback.load_passing_record(_PINNED_RECORD, canonical=True)
    provenance = raf_readback.record_provenance(_PINNED_RECORD, record, canonical=True)
    assert provenance["canonical"] is True
    assert provenance["taint"] == []
    assert provenance["sha256"] == _PINNED_SHA256

    path = write_passing_readback_record(str(tmp_path / "synthetic.json"))
    synthetic = raf_readback.load_passing_record(path, canonical=False)
    tainted = raf_readback.record_provenance(path, synthetic, canonical=False)
    assert tainted["canonical"] is False
    assert any("non-canonical" in t for t in tainted["taint"])


def test_prepare_refuses_a_synthetic_record_without_the_non_canonical_flag(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record, write_room

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    argv = _prepare_argv(tmp_path, raf_root, non_canonical=False,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    with pytest.raises(ValueError) as exc:
        raf_prepare.main(argv)          # no --non-canonical
    assert "sha256" in str(exc.value).lower() or "non-canonical" in str(exc.value)


def test_prepare_taints_every_artifact_in_non_canonical_mode(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record, write_room

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    raf_prepare.main(argv + ["--non-canonical"])
    split_dir = tmp_path / "splits"
    for name in ("raf_splits_record.json", "raf_amplitude_audit.json"):
        with open(split_dir / name) as f:
            payload = json.load(f)
        assert payload["canonical"] is False
        assert any("non-canonical" in t for t in payload["taint"])
        assert payload["readback_record"]["canonical"] is False


def test_render_depth_taints_the_qa_record_in_non_canonical_mode(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record
    from test_raf_render_depth import _write_fixture

    raf_root, out, _ = _write_fixture(tmp_path)
    path = write_passing_readback_record(str(tmp_path / "rb.json"))
    with pytest.raises(ValueError):
        raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                         "--rooms", "EmptyRoom", "--readback-record", path])
    raf_render.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                     "--rooms", "EmptyRoom", "--readback-record", path,
                     "--non-canonical"])
    with open(out / "EmptyRoom" / "depth_images" / "raf_depth_qa.json") as f:
        qa = json.load(f)
    assert qa["canonical"] is False
    assert any("non-canonical" in t for t in qa["taint"])


# --------------------------------------------------------------------------- #
# r3 S6: an amplitude scalar derives from TRAINED supports only
# --------------------------------------------------------------------------- #
def _roles_fixture(tmp_path):
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    roles = {"000000": "train", "000001": "train", "000002": "support",
             "000003": "test", "000004": "val", "000005": "diagnostic"}
    return room_dir, roles


def test_scale_decision_excludes_validation_supports(tmp_path):
    """S6: role 'support' is a VALIDATION support in canonical metadata, so it may
    not enter a statistic labelled 'train supports only'."""
    room_dir, roles = _roles_fixture(tmp_path)
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), sorted(roles), roles=roles)
    decision = audit["scale_decision"]
    assert decision["derived_from"] == "trained supports only (split_role == 'train')"
    assert decision["n_train_supports"] == 2
    assert decision["derivation_ids"] == ["000000", "000001"]
    trained_peaks = [audit["files"][c]["peak"] for c in ("000000", "000001")]
    assert decision["train_support_peak_median"] == pytest.approx(
        float(np.median(trained_peaks)))
    assert decision["applied_scalar"] is None


def test_scale_derivation_provenance_is_a_hash_of_the_id_set(tmp_path):
    import hashlib

    room_dir, roles = _roles_fixture(tmp_path)
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), sorted(roles), roles=roles)
    expected = hashlib.sha256("000000;000001".encode("utf-8")).hexdigest()
    assert audit["scale_decision"]["derivation_id_sha256"] == expected
    assert raf_prepare.derivation_id_hash(["000001", "000000"]) == expected


def test_applying_a_scalar_requires_matching_derivation_provenance(tmp_path):
    room_dir, roles = _roles_fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        raf_prepare.resample_and_write(
            room_dir, str(tmp_path / "a" / "EmptyRoom"), sorted(roles), roles=roles,
            scale=2.0)
    assert "provenance" in str(exc.value).lower()

    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(
            room_dir, str(tmp_path / "b" / "EmptyRoom"), sorted(roles), roles=roles,
            scale=2.0, scale_provenance="0" * 64)

    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "c" / "EmptyRoom"), sorted(roles), roles=roles,
        scale=2.0, scale_provenance=raf_prepare.derivation_id_hash(["000000", "000001"]))
    assert audit["scale_decision"]["applied_scalar"] == 2.0
    assert audit["scale_decision"]["provenance_verified"] is True


def test_scalar_cannot_be_applied_without_any_trained_support(tmp_path):
    room_dir, roles = _roles_fixture(tmp_path)
    only_val = {cid: "val" for cid in roles}
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(
            room_dir, str(tmp_path / "d" / "EmptyRoom"), sorted(only_val),
            roles=only_val, scale=2.0,
            scale_provenance=raf_prepare.derivation_id_hash([]))


# --------------------------------------------------------------------------- #
# r4 T2: canonical mode enforces the registered parameter set
# --------------------------------------------------------------------------- #
def test_canonical_parameters_are_the_registered_ones():
    assert raf_prepare.CANONICAL_PARAMS == {
        "rooms": ("EmptyRoom", "FurnishedRoom"), "n_groups": 16, "n_val_groups": 4,
        "n_train": 12, "n_diagnostic_groups": 1, "full_crosscheck": True,
        "allow_nonuniform": True,
    }


def _canonical_args(**overrides):
    import argparse

    values = {"rooms": ["EmptyRoom", "FurnishedRoom"], "n_groups": 16,
              "n_val_groups": 4, "n_train": 12, "n_diagnostic_groups": 1,
              "full_crosscheck": True, "allow_nonuniform": True}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_canonical_parameter_gate_accepts_the_registered_set():
    assert raf_prepare.assert_canonical_parameters(_canonical_args()) == []


@pytest.mark.parametrize("overrides,needle", [
    ({"rooms": ["EmptyRoom"]}, "rooms"),
    ({"rooms": ["EmptyRoom", "FurnishedRoom", "Extra"]}, "rooms"),
    ({"n_groups": 8}, "n_groups"),
    ({"n_val_groups": 2}, "n_val_groups"),
    ({"n_train": 6}, "n_train"),
    ({"n_diagnostic_groups": 0}, "n_diagnostic_groups"),
    ({"full_crosscheck": False}, "full_crosscheck"),
])
def test_canonical_parameter_gate_rejects_deviations(overrides, needle):
    with pytest.raises(ValueError) as exc:
        raf_prepare.assert_canonical_parameters(_canonical_args(**overrides))
    assert needle in str(exc.value)
    assert "--non-canonical" in str(exc.value)


def test_non_canonical_mode_records_parameter_deviations(tmp_path):
    """The synthetic CLI runs deviate from the canonical set by design; that is
    recorded rather than silently allowed."""
    from test_raf_prepare_data import write_passing_readback_record, write_room

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    raf_prepare.main(argv)
    with open(tmp_path / "splits" / "raf_splits_record.json") as f:
        record = json.load(f)
    assert any("parameter" in t for t in record["taint"])
    assert record["params"]["canonical_parameters"] is False


def test_the_parameter_set_joins_the_marker_identity(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record, write_room
    import publish as raf_publish

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom")
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    raf_prepare.main(argv)
    with open(tmp_path / "splits" / raf_publish.marker_name("prepare")) as f:
        marker = json.load(f)
    assert marker["canonical"] is False
    assert marker["parameters"]["n_groups"] == 1
    assert marker["parameters"]["rooms"] == ["EmptyRoom"]
    assert marker["readback_record"]["sha256"]


# --------------------------------------------------------------------------- #
# r4 T1: read-once digest, full sub-verdict validation, corpus binding
# --------------------------------------------------------------------------- #
def test_record_is_read_once_and_the_parsed_bytes_are_what_is_hashed():
    payload, digest, record = raf_readback.read_record_once(_PINNED_RECORD)
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == digest == _PINNED_SHA256
    assert record == json.loads(payload)


def test_a_file_swapped_after_parsing_cannot_supply_the_pinned_bytes(tmp_path):
    """T1's TOCTOU: parse, then reopen to hash, and a swap in between let forged
    content ride on the pinned digest. One descriptor, one read, one digest."""
    path = tmp_path / "record.json"
    forged = _tampered(tmp_path, _superseded_quat, name="forged.json")
    with open(forged) as f:
        path.write_text(f.read())
    payload, digest, record = raf_readback.read_record_once(str(path))
    with open(_PINNED_RECORD, "rb") as f:
        path.write_bytes(f.read())          # swap AFTER the read
    assert record["adjudication"]["quat_order_pinned"] == "wxyz"
    assert digest != _PINNED_SHA256         # the digest describes what was parsed


def test_provenance_carries_the_authenticated_digest(tmp_path):
    record = raf_readback.load_passing_record(_PINNED_RECORD, canonical=True)
    provenance = raf_readback.record_provenance(_PINNED_RECORD, record, canonical=True)
    assert provenance["sha256"] == record["__authenticated_sha256__"] == _PINNED_SHA256


def _fail_t30(record):
    record["rooms"]["EmptyRoom"]["t30_validity"]["valid_full"] = 0


def _empty_amplitude(record):
    record["rooms"]["EmptyRoom"]["amplitude"]["peak_stats"]["count"] = 0


def _crosscheck_mismatch(record):
    record["rooms"]["EmptyRoom"]["crosscheck"]["mismatches"] = 3


def _no_quaternion_readings(record):
    record["rooms"]["EmptyRoom"]["quaternion"]["identity_readings"] = {}


@pytest.mark.parametrize("mutate,needle", [
    (_fail_t30, "t30"),
    (_empty_amplitude, "amplitude"),
    (_crosscheck_mismatch, "cross-check"),
    (_no_quaternion_readings, "quaternion"),
])
def test_every_sub_verdict_is_validated(tmp_path, mutate, needle):
    """T1: the content check only tested block presence, onset.passed and a
    non-empty T60 resolution."""
    path = _tampered(tmp_path, mutate)
    with pytest.raises(ValueError) as exc:
        raf_readback.assert_canonical_content(json.load(open(path)), path)
    assert needle in str(exc.value).lower()


def test_corpus_binding_uses_the_room_index_digests(tmp_path):
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    digests = raf_readback.room_index_digests(
        os.path.join(str(tmp_path), "archived", "EmptyRoom"))
    assert set(digests) == {"n_captures", "all_tx_pos_sha256", "all_rx_pos_sha256",
                            "rx_trailing_sentinel_dropped"}
    assert digests["n_captures"] == 36
    assert len(digests["all_tx_pos_sha256"]) == 64

    record = {"rooms": {"EmptyRoom": {"room_index": digests}}}
    assert raf_readback.verify_corpus_binding(record, str(tmp_path), ["EmptyRoom"]) == []

    # a different corpus: same shape, different bytes
    other = tmp_path / "other"
    write_room(str(other), "EmptyRoom", groups=_default_groups(2))
    problems = raf_readback.verify_corpus_binding(record, str(other), ["EmptyRoom"])
    assert problems and "EmptyRoom" in problems[0]


def test_corpus_binding_falls_back_to_capture_counts_when_digests_are_absent(tmp_path):
    """The pinned record predates the digest block; the counts it DOES carry still
    catch the operator error of pointing at the wrong corpus."""
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    record = {"rooms": {"EmptyRoom": {"n_captures": 36}}}
    problems = raf_readback.verify_corpus_binding(record, str(tmp_path), ["EmptyRoom"])
    assert problems == []
    wrong = {"rooms": {"EmptyRoom": {"n_captures": 47484}}}
    problems = raf_readback.verify_corpus_binding(wrong, str(tmp_path), ["EmptyRoom"])
    assert problems and "capture count" in problems[0]


def test_the_audit_records_room_index_digests_going_forward(audited_room, tmp_path):
    raf_root, _ = audited_room
    out = tmp_path / "rec.json"
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom", "--out", str(out),
                       "--n-onset-samples", "36"])
    with open(out) as f:
        record = json.load(f)
    digests = record["rooms"]["EmptyRoom"]["room_index"]
    assert digests["n_captures"] == 36
    assert len(digests["all_rx_pos_sha256"]) == 64


def test_the_pinned_record_predates_the_digest_block_and_that_is_recorded():
    """Registered residual: re-pinning the canonical record to carry file digests
    needs a real-corpus audit rerun and a new pinned hash (Planner's call)."""
    record = raf_readback.load_passing_record(_PINNED_RECORD, canonical=True)
    assert "room_index" not in record["rooms"]["EmptyRoom"]
    provenance = raf_readback.record_provenance(_PINNED_RECORD, record, canonical=True)
    assert any("capture counts" in note for note in provenance["corpus_binding"])
