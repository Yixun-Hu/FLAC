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
_PINNED_SHA256 = "9288181be62bf8b4669880522fadaab18527facb2749837f768572069f4876c3"


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
        "n_train": 12, "n_diagnostic_groups": 1, "seed": 0, "full_crosscheck": True,
        "allow_nonuniform": True, "amplitude_ceiling": 0.75, "amplitude_scalar": 3.0,
        "amplitude_formula_version": "9.2", "amplitude_derivation_ids": 408,
        "amplitude_derivation_sha256":
            "8a740feef8f430dbc2e65d8f3d5eefa3d6b191c00c615ff758163c7428eef00d",
    }


def _canonical_args(**overrides):
    import argparse

    values = {"rooms": ["EmptyRoom", "FurnishedRoom"], "n_groups": 16,
              "n_val_groups": 4, "n_train": 12, "n_diagnostic_groups": 1,
              "seed": 0, "full_crosscheck": True, "allow_nonuniform": True}
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
    ({"seed": 999}, "seed"),
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
    assert marker["parameters"]["amplitude_formula_version"] == "9.2"
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
    """Records written before the digest block existed still bind by capture count,
    which catches the operator error of pointing at the wrong corpus."""
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


def test_the_pinned_record_binds_the_corpus_by_pose_digests():
    """Re-pinned on the real corpus with the r4 audit code: the canonical record now
    carries per-room pose-file digests, so the gate binds on them rather than on
    capture counts alone."""
    record = raf_readback.load_passing_record(_PINNED_RECORD, canonical=True)
    for room in ("EmptyRoom", "FurnishedRoom"):
        digests = record["rooms"][room]["room_index"]
        assert len(digests["all_tx_pos_sha256"]) == 64
        assert len(digests["all_rx_pos_sha256"]) == 64
        assert digests["rx_trailing_sentinel_dropped"] is True
    assert record["rooms"]["EmptyRoom"]["room_index"]["n_captures"] == 47484
    assert record["rooms"]["FurnishedRoom"]["room_index"]["n_captures"] == 39132
    provenance = raf_readback.record_provenance(_PINNED_RECORD, record, canonical=True)
    assert provenance["corpus_binding"] == [
        "EmptyRoom: pose-file digests", "FurnishedRoom: pose-file digests"]


@pytest.mark.skipif(
    not os.path.isdir("/media/diskstation/yixunhu/raf_dataset/archived"),
    reason="the RAF corpus is not mounted in this checkout")
def test_the_pinned_record_matches_the_corpus_it_audited():
    """Integration: the recorded digests are the digests of the corpus on disk."""
    record = raf_readback.load_passing_record(
        _PINNED_RECORD, canonical=True,
        expected_raf_root="/media/diskstation/yixunhu/raf_dataset")
    assert record["params"]["raf_root"] == "/media/diskstation/yixunhu/raf_dataset"


# --------------------------------------------------------------------------- #
# r7 Amendment 9: the registered amplitude scalar, DERIVED not hardcoded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (3.0013, 3.0), (2.4, 2.0), (0.75 / 0.24989, 3.0), (12.7, 10.0), (0.043, 0.04),
])
def test_one_significant_digit_rounding(value, expected):
    assert raf_prepare.one_significant_digit(value) == pytest.approx(expected)


def test_scalar_is_derived_from_the_train_support_peaks(tmp_path):
    """scalar = 1 sig digit of (ceiling / max trained-support peak). The registered
    corpus gives 0.75/0.24989 -> 3.0, but nothing here hardcodes 3."""
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.25)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_amplitude_scalar(room_dir, trained)
    assert decision["ceiling"] == 0.75
    # the peak is measured on the RESAMPLED signal -- what actually gets written --
    # and the scalar follows the formula from THAT, never from a constant
    assert 0.15 < decision["max_train_support_peak"] < 0.25
    assert decision["scalar"] == pytest.approx(
        raf_prepare.one_significant_digit(0.75 / decision["max_train_support_peak"]))
    assert decision["derivation_ids"] == trained
    assert decision["derivation_id_sha256"] == raf_prepare.derivation_id_hash(trained)
    assert "one significant digit" in decision["formula"]


def test_scalar_derivation_ignores_non_trained_roles(tmp_path):
    """A loud TEST file must not shrink the scalar: the derivation set is
    split_role == 'train' only."""
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.25)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    import soundfile as sf_mod
    from test_raf_prepare_data import _rir

    loud = _rir(99) / 0.01 * 0.9
    sf_mod.write(os.path.join(room_dir, "data", "000020", "rir.wav"), loud, 48000,
                 subtype="FLOAT")
    quiet_only = raf_prepare.derive_amplitude_scalar(room_dir, [f"{i:06d}" for i in range(4)])
    with_loud = raf_prepare.derive_amplitude_scalar(
        room_dir, [f"{i:06d}" for i in range(4)] + ["000020"])
    assert with_loud["max_train_support_peak"] > quiet_only["max_train_support_peak"]
    assert with_loud["scalar"] < quiet_only["scalar"]


def test_scalar_derivation_requires_trained_supports(tmp_path):
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1))
    with pytest.raises(ValueError):
        raf_prepare.derive_amplitude_scalar(
            os.path.join(str(tmp_path), "archived", "EmptyRoom"), [])


def test_applied_scalar_lifts_quiet_files_and_never_clips(tmp_path):
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.25)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    ids = [f"{i:06d}" for i in range(8)]
    roles = {cid: ("train" if i < 4 else "test") for i, cid in enumerate(ids)}
    decision = raf_prepare.derive_amplitude_scalar(room_dir, ids[:4])
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ids, roles=roles,
        scale=decision["scalar"], scale_provenance=decision["derivation_id_sha256"])
    assert audit["scale_decision"]["applied_scalar"] == pytest.approx(decision["scalar"])
    for entry in audit["files"].values():
        assert entry["peak"] <= 0.999
        assert entry["roundtrip_max_abs_error"] == 0.0
    assert audit["n_silent"] == 0


def test_a_scalar_that_would_clip_is_refused(tmp_path):
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=0.25)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    ids = [f"{i:06d}" for i in range(4)]
    roles = {cid: "train" for cid in ids}
    with pytest.raises(ValueError) as exc:
        raf_prepare.resample_and_write(
            room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ids, roles=roles,
            scale=9.0, scale_provenance=raf_prepare.derivation_id_hash(ids))
    assert "clip" in str(exc.value).lower()


def test_a_scalar_that_leaves_silent_files_is_refused(tmp_path):
    """Post-scale assertion: after applying a scalar, NO file may still sit below
    the loader's -60 dBFS substitution threshold."""
    from test_raf_prepare_data import _default_groups, write_room

    write_room(str(tmp_path), "EmptyRoom", groups=_default_groups(1), rir_peak=1e-5)
    room_dir = os.path.join(str(tmp_path), "archived", "EmptyRoom")
    ids = [f"{i:06d}" for i in range(4)]
    roles = {cid: "train" for cid in ids}
    with pytest.raises(ValueError) as exc:
        raf_prepare.resample_and_write(
            room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ids, roles=roles,
            scale=2.0, scale_provenance=raf_prepare.derivation_id_hash(ids))
    assert "-60" in str(exc.value) or "silent" in str(exc.value)


def test_cli_derives_records_and_binds_the_scalar(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record, write_room
    import publish as raf_publish

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", rir_peak=0.25)
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    raf_prepare.main(argv)
    split_dir = tmp_path / "splits"
    with open(split_dir / "raf_amplitude_audit.json") as f:
        audit = json.load(f)
    decision = audit["rooms"]["EmptyRoom"]["scale_decision"]
    derived = decision["applied_scalar"]
    assert derived == pytest.approx(raf_prepare.one_significant_digit(
        0.75 / decision["max_train_support_peak"]))
    assert len(decision["derivation_id_sha256"]) == 64
    assert decision["derivation_id_sha256"] == raf_prepare.derivation_id_hash(
        decision["derivation_ids"])
    with open(split_dir / "raf_splits_record.json") as f:
        record = json.load(f)
    assert record["amplitude_scalar"]["scalar"] == pytest.approx(derived)
    assert record["amplitude_scalar"]["formula_version"] == "9.2"
    with open(split_dir / raf_publish.marker_name("prepare")) as f:
        marker = json.load(f)
    assert marker["parameters"]["amplitude_scalar"] == pytest.approx(derived)
    assert marker["parameters"]["amplitude_ceiling"] == 0.75
    assert marker["parameters"]["amplitude_formula_version"] == "9.2"
    assert len(marker["parameters"]["amplitude_derivation_sha256"]) == 64
    assert marker["parameters"]["amplitude_derivation_ids"] == decision["n_train_supports"]


def test_republication_keeps_split_content_byte_identical(tmp_path):
    """Amendment 9 requires a NEW generation with IDENTICAL split content: the
    seed and the FPS selection are untouched, only amplitudes change."""
    from test_raf_prepare_data import write_passing_readback_record, write_room
    import publish as raf_publish

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", rir_peak=0.25)
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    split_dir = tmp_path / "splits"
    raf_prepare.main(argv)
    first = {name: (split_dir / name).read_bytes()
             for name in ("train_base.json", "val_base.json", "test_base.json",
                          "diagnostic_base.json")}
    with open(split_dir / raf_publish.marker_name("prepare")) as f:
        first_generation = json.load(f)["generation"]

    raf_prepare.main(argv)                      # republish
    for name, payload in first.items():
        assert (split_dir / name).read_bytes() == payload, name
    with open(split_dir / raf_publish.marker_name("prepare")) as f:
        assert json.load(f)["generation"] != first_generation


# --------------------------------------------------------------------------- #
# r7 Amendment 9.1: the clip-clamp term
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("limit,expected", [
    (3.9977, 3.0), (5.0, 5.0), (0.999 / 0.24989, 3.0), (9.99, 9.0), (0.42, 0.4),
    (1.0, 1.0), (23.0, 20.0),
])
def test_largest_one_significant_digit_at_most(limit, expected):
    assert raf_prepare.largest_one_significant_digit_at_most(limit) == pytest.approx(
        expected)
    assert raf_prepare.largest_one_significant_digit_at_most(limit) <= limit


def _room_with_peaks(tmp_path, support_peak, loud_peak=None, name="EmptyRoom"):
    """A room whose trained supports peak at ``support_peak`` and whose test file
    optionally peaks louder."""
    import soundfile as sf_mod
    from test_raf_prepare_data import _default_groups, _rir, write_room

    write_room(str(tmp_path), name, groups=_default_groups(1), rir_peak=support_peak)
    room_dir = os.path.join(str(tmp_path), "archived", name)
    if loud_peak is not None:
        loud = _rir(77) / 0.01 * loud_peak
        sf_mod.write(os.path.join(room_dir, "data", "000020", "rir.wav"), loud, 48000,
                     subtype="FLOAT")
    return room_dir


def test_the_clamp_binds_when_a_written_file_would_clip(tmp_path):
    """The real-corpus shape: quiet supports set a large support term, and a loud
    TEST file (not in the derivation set) would clip under it."""
    room_dir = _room_with_peaks(tmp_path, support_peak=0.05, loud_peak=0.9)
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_amplitude_scalar(
        room_dir, trained, written_ids=trained + ["000020"])
    assert decision["clamp_engaged"] is True
    assert decision["binding_term"] == "clip_clamp"
    assert decision["scalar"] == decision["clamp_term"] < decision["support_term"]
    # the bound actually holds: nothing written can exceed the clip ceiling
    assert decision["max_written_peak"] * decision["scalar"] <= raf_prepare.CLIP_CEILING


def test_the_support_term_binds_when_nothing_would_clip(tmp_path):
    room_dir = _room_with_peaks(tmp_path, support_peak=0.25)
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_amplitude_scalar(
        room_dir, trained, written_ids=[f"{i:06d}" for i in range(8)])
    assert decision["clamp_engaged"] is False
    assert decision["binding_term"] == "support_ceiling"
    assert decision["scalar"] == decision["support_term"] <= decision["clamp_term"]


def test_the_clamp_only_ever_lowers_the_scalar(tmp_path):
    """Global information is a SAFETY bound, never a target statistic."""
    room_dir = _room_with_peaks(tmp_path, support_peak=0.05, loud_peak=0.9)
    trained = [f"{i:06d}" for i in range(4)]
    unclamped = raf_prepare.derive_amplitude_scalar(room_dir, trained)
    clamped = raf_prepare.derive_amplitude_scalar(
        room_dir, trained, written_ids=trained + ["000020"])
    assert clamped["scalar"] <= unclamped["scalar"]
    assert clamped["support_term"] == unclamped["support_term"]


def test_the_clamped_scalar_survives_the_post_scale_assertions(tmp_path):
    """End to end: the r7 abort that this amendment answers must not recur."""
    room_dir = _room_with_peaks(tmp_path, support_peak=0.05, loud_peak=0.9)
    ids = [f"{i:06d}" for i in range(4)] + ["000020"]
    roles = {cid: ("train" if i < 4 else "test") for i, cid in enumerate(ids)}
    decision = raf_prepare.derive_amplitude_scalar(
        room_dir, ids[:4], written_ids=ids)
    audit = raf_prepare.resample_and_write(
        room_dir, str(tmp_path / "runtime" / "EmptyRoom"), ids, roles=roles,
        scale=decision["scalar"], scale_provenance=decision["derivation_id_sha256"])
    assert max(e["peak"] for e in audit["files"].values()) <= raf_prepare.CLIP_CEILING
    assert audit["n_silent"] == 0


def test_the_cli_records_the_binding_term(tmp_path):
    from test_raf_prepare_data import write_passing_readback_record, write_room

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", rir_peak=0.25)
    argv = _prepare_argv(tmp_path, raf_root,
                         readback=write_passing_readback_record(str(tmp_path / "rb.json")))
    raf_prepare.main(argv)
    with open(tmp_path / "splits" / "raf_amplitude_audit.json") as f:
        decision = json.load(f)["rooms"]["EmptyRoom"]["scale_decision"]
    assert decision["binding_term"] in ("support_ceiling", "clip_clamp")
    assert decision["clamp_engaged"] in (True, False)
    assert decision["max_written_peak"] >= decision["max_train_support_peak"]
    assert decision["max_written_peak"] * decision["applied_scalar"] <= \
        raf_prepare.CLIP_CEILING


# --------------------------------------------------------------------------- #
# r7 Amendment 9.2 F2/F3: ONE global scalar, provenance in the identity
# --------------------------------------------------------------------------- #
def test_the_scalar_is_derived_once_over_both_rooms(tmp_path):
    """F2: per-room derivation would scale the two rooms differently, making
    cross-room metrics incomparable. Here EmptyRoom's supports are louder, so the
    GLOBAL support term must follow them, not FurnishedRoom's."""
    loud = _room_with_peaks(tmp_path / "a", support_peak=0.25, name="EmptyRoom")
    quiet = _room_with_peaks(tmp_path / "b", support_peak=0.05, name="FurnishedRoom")
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_global_amplitude_scalar({
        "EmptyRoom": (loud, trained, trained),
        "FurnishedRoom": (quiet, trained, trained),
    })
    per_room_loud = raf_prepare.derive_amplitude_scalar(loud, trained, trained)
    per_room_quiet = raf_prepare.derive_amplitude_scalar(quiet, trained, trained)
    assert per_room_loud["scalar"] != per_room_quiet["scalar"]   # they WOULD differ
    assert decision["scalar"] == per_room_loud["scalar"]         # global follows the max
    assert decision["rooms"] == ["EmptyRoom", "FurnishedRoom"]
    assert decision["max_train_support_peak"] == pytest.approx(
        max(per_room_loud["max_train_support_peak"],
            per_room_quiet["max_train_support_peak"]))


def test_the_global_derivation_set_is_room_qualified(tmp_path):
    loud = _room_with_peaks(tmp_path / "a", support_peak=0.25, name="EmptyRoom")
    quiet = _room_with_peaks(tmp_path / "b", support_peak=0.05, name="FurnishedRoom")
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_global_amplitude_scalar({
        "EmptyRoom": (loud, trained, trained),
        "FurnishedRoom": (quiet, trained, trained),
    })
    assert decision["derivation_ids"][0] == "EmptyRoom/000000"
    assert "FurnishedRoom/000000" in decision["derivation_ids"]
    assert decision["n_train_supports"] == 8
    assert decision["derivation_id_sha256"] == raf_prepare.derivation_id_hash(
        decision["derivation_ids"])
    # bare ids would collide across rooms; qualified ones do not
    assert decision["derivation_id_sha256"] != raf_prepare.derivation_id_hash(trained)


def test_the_global_clamp_sees_a_loud_file_in_the_other_room(tmp_path):
    quiet = _room_with_peaks(tmp_path / "a", support_peak=0.05, name="EmptyRoom")
    loud = _room_with_peaks(tmp_path / "b", support_peak=0.05, loud_peak=0.9,
                            name="FurnishedRoom")
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_global_amplitude_scalar({
        "EmptyRoom": (quiet, trained, trained),
        "FurnishedRoom": (loud, trained, trained + ["000020"]),
    })
    assert decision["clamp_engaged"] is True
    assert decision["binding_term"] == "clip_clamp"
    assert decision["max_written_peak"] * decision["scalar"] <= raf_prepare.CLIP_CEILING


def test_a_room_without_trained_supports_fails_closed(tmp_path):
    room = _room_with_peaks(tmp_path, support_peak=0.25)
    with pytest.raises(ValueError) as exc:
        raf_prepare.derive_global_amplitude_scalar({
            "EmptyRoom": (room, [f"{i:06d}" for i in range(4)], []),
            "FurnishedRoom": (room, [], []),
        })
    assert "FurnishedRoom" in str(exc.value)


def test_the_written_scalar_must_match_the_global_derivation_set(tmp_path):
    """The r6 provenance rule still holds with a GLOBAL set: the writer verifies
    the hash against the ids it was told the scalar came from."""
    room = _room_with_peaks(tmp_path, support_peak=0.25)
    ids = [f"{i:06d}" for i in range(4)]
    roles = {cid: "train" for cid in ids}
    qualified = [f"EmptyRoom/{cid}" for cid in ids]
    with pytest.raises(ValueError):
        raf_prepare.resample_and_write(
            room, str(tmp_path / "x" / "EmptyRoom"), ids, roles=roles, scale=2.0,
            scale_provenance=raf_prepare.derivation_id_hash(qualified))
    audit = raf_prepare.resample_and_write(
        room, str(tmp_path / "y" / "EmptyRoom"), ids, roles=roles, scale=2.0,
        scale_provenance=raf_prepare.derivation_id_hash(qualified),
        scale_provenance_ids=qualified)
    assert audit["scale_decision"]["derivation_ids"] == qualified


def test_canonical_mode_refuses_a_scalar_that_is_not_the_registered_one(tmp_path):
    """Amendment 9.1: a corpus that yields a different scalar stops the run."""
    room = _room_with_peaks(tmp_path, support_peak=0.25)
    trained = [f"{i:06d}" for i in range(4)]
    decision = raf_prepare.derive_global_amplitude_scalar(
        {"EmptyRoom": (room, trained, trained)})
    assert raf_prepare.assert_registered_scalar(decision, registered=decision["scalar"])
    with pytest.raises(ValueError) as exc:
        raf_prepare.assert_registered_scalar(decision, registered=3.0)
    assert "Refusing to write any WAV" in str(exc.value)


def test_the_registered_scalar_gate_precedes_every_write(tmp_path):
    """The gate is reached with NOTHING written: an early canonical refusal leaves
    no runtime tree at all."""
    import inspect

    source = inspect.getsource(raf_prepare.main)
    gate = source.index("assert_registered_scalar")
    first_write = source.index("resample_and_write(")
    assert gate < first_write

    from test_raf_prepare_data import write_room

    raf_root = tmp_path / "raf"
    write_room(str(raf_root), "EmptyRoom", rir_peak=0.25)
    out = tmp_path / "runtime" / "RAF"
    with pytest.raises(ValueError):
        raf_prepare.main(["--raf-root", str(raf_root), "--output-dir", str(out),
                          "--split-dir", str(tmp_path / "splits"), "--rooms",
                          "EmptyRoom", "--n-groups", "1", "--n-val-groups", "1",
                          "--n-diagnostic-groups", "1", "--n-train", "12",
                          "--full-crosscheck", "--readback-record",
                          os.path.join(_REPO_ROOT, "data", "RAF",
                                       "raf_readback_record.json")])
    assert not (out / "EmptyRoom" / "mono_rirs_22050Hz").exists()
