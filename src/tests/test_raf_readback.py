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
    record = raf_readback.load_passing_record(path)
    assert record["verdict"]["passed"] is True


def test_gate_rejects_a_missing_record(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        raf_readback.load_passing_record(str(tmp_path / "nope.json"))


def test_gate_rejects_an_unpinned_record(audited_room, tmp_path):
    raf_root, _ = audited_room
    out = tmp_path / "unpinned.json"
    raf_readback.main(["--raf-root", raf_root, "--rooms", "EmptyRoom", "--out", str(out),
                       "--n-onset-samples", "36"])
    with pytest.raises(ValueError) as exc:
        raf_readback.load_passing_record(str(out))
    assert "gauge" in str(exc.value) or "quat" in str(exc.value)


def test_gate_rejects_a_failed_record(audited_room, tmp_path):
    path = _pinned_record(tmp_path, audited_room)
    with open(path) as f:
        record = json.load(f)
    record["verdict"] = {"passed": False, "reasons": ["onset fit failed"]}
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(ValueError):
        raf_readback.load_passing_record(path)


def test_gate_rejects_a_foreign_schema(audited_room, tmp_path):
    path = _pinned_record(tmp_path, audited_room)
    with open(path) as f:
        record = json.load(f)
    record["schema_version"] = raf_readback.RECORD_SCHEMA_VERSION + 1
    with open(path, "w") as f:
        json.dump(record, f)
    with pytest.raises(ValueError):
        raf_readback.load_passing_record(path)


# --------------------------------------------------------------------------- #
# publish gates
# --------------------------------------------------------------------------- #
def _prepare_argv(tmp_path, raf_root, readback=None):
    argv = ["--raf-root", str(raf_root), "--output-dir", str(tmp_path / "runtime" / "RAF"),
            "--split-dir", str(tmp_path / "splits"), "--rooms", "EmptyRoom",
            "--n-groups", "1", "--n-val-groups", "1", "--n-diagnostic-groups", "1",
            "--n-train", "12", "--full-crosscheck"]
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
                     "--rooms", "EmptyRoom", "--readback-record", path])
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
    assert decision["derived_from"] == "train supports only"
    assert decision["applied_scalar"] is None
    assert decision["train_support_peak_median"] == pytest.approx(
        float(np.median([audit["files"][f"{i:06d}"]["peak"] for i in range(4)])))
    assert set(audit["comparison"]) == {"HAA", "AR", "note"}


def test_amplitude_audit_applies_a_scalar_uniformly_when_one_is_given(tmp_path):
    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    plain = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "a" / "EmptyRoom"), ["000000", "000001"])
    scaled = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "b" / "EmptyRoom"), ["000000", "000001"], scale=2.0)
    assert scaled["scale_decision"]["applied_scalar"] == 2.0
    for cid in ("000000", "000001"):
        assert scaled["files"][cid]["peak"] == pytest.approx(
            2.0 * plain["files"][cid]["peak"], rel=1e-6)
