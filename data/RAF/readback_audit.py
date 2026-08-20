"""RAF readback audit: the artifact canonical preparation is gated on.

exp_19 r2, Codex R4 / contracts Amendment 2. Before any canonical RAF artifact is
published, the facts the plan promised to measure at the readback rung have to be
measured, written down, and adjudicated:

* onset-vs-distance regression per room (constant acquisition delay + 1/343 s/m),
* crop-vs-full T30 validity, which decides whether T60 stays a headline metric,
* amplitude distribution (the fixed-scalar decision's input; the scalar itself is
  derived in ``prepare_data`` from TRAIN SUPPORTS ONLY),
* both quaternion column-order readings' implied-forward vectors, and the gauge,
  so the Planner can PIN them from evidence instead of a docstring caveat.

``prepare_data.py`` and ``render_depth.py`` refuse to publish canonical artifacts
unless ``--readback-record`` points at a record that passed AND carries the
Planner's pinning.

Usage:
    python data/RAF/readback_audit.py --raf-root /path/to/raf_dataset \\
        --rooms EmptyRoom FurnishedRoom --out data/RAF/raf_readback_record.json \\
        [--pin-gauge "RAF_TO_PIPELINE:(X,Z,Y)" --pin-quat xyzw]
"""
import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # sibling scripts, not an installed package
    sys.path.insert(0, _HERE)
from raf_common import (  # noqa: E402
    RAF_TO_PIPELINE,
    dbfs as _dbfs,
    distance_stats as _distance_stats,
)

# prepare_data imports THIS module for its publish gate, so its own heavy readers
# are imported lazily inside audit_room to keep the module graph acyclic.

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = 1

# The ONE record canonical RAF artifacts may be published under (Amendment 5, S1).
# Binding publication to this digest is what stops a hand-written JSON carrying
# `verdict.passed=true` from authorising a canonical publish: the review
# demonstrated exactly that bypass with `rooms={}` and the superseded wxyz pin.
CANONICAL_RECORD_SHA256 = "9288181be62bf8b4669880522fadaab18527facb2749837f768572069f4876c3"
CANONICAL_GAUGE = "RAF_TO_PIPELINE:(X,Z,Y)"
CANONICAL_QUAT_ORDER = "xyzw"
CANONICAL_ROOMS = ("EmptyRoom", "FurnishedRoom")
# Every measurement block the adjudication rests on. Their presence is what makes
# the pins evidence rather than assertion.
REQUIRED_ROOM_BLOCKS = ("onset", "t30_validity", "amplitude", "quaternion", "crosscheck")
SPEED_OF_SOUND = 343.0
ONSET_THRESHOLD_DB = -20.0
SLOPE_TOLERANCE = 0.20          # +-20% of 1/343 s/m
MIN_R2 = 0.8
T30_DECAY_DB = 30               # mirrors RT60Error's HAA/RAF policy
LOADER_CROP = 10240
# Preregistered demotion rule (plan Rev 2 section 2): if cropping to the metric
# window invalidates more than this fraction of the RIRs that are valid over the
# full recording, T60 stops being a headline number.
T60_CROP_INVALIDATION_LIMIT = 0.05


def detect_onset(wave, threshold_db=ONSET_THRESHOLD_DB):
    """First sample whose |amplitude| exceeds ``threshold_db`` below the peak.

    Fail-closed on an all-zero signal: "the onset is sample 0" would be a fact
    about silence, and it would drag the regression it feeds.
    """
    arr = np.asarray(wave, dtype=np.float64)
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak <= 0.0:
        raise ValueError("cannot detect an onset in an all-zero signal")
    threshold = peak * (10.0 ** (threshold_db / 20.0))
    hits = np.nonzero(np.abs(arr) >= threshold)[0]
    if hits.size == 0:
        raise ValueError("no sample reaches the onset threshold")
    return int(hits[0])


def fit_constant_delay(distances, onsets_s, tolerance=SLOPE_TOLERANCE, min_r2=MIN_R2):
    """Least-squares ``onset = slope * distance + delay`` with the registered gate.

    PASS iff R^2 >= 0.8 and the slope is within +-20% of 1/343 s/m. The intercept
    is the implied constant acquisition delay -- the only delay handling the plan
    preregistered (a constant, never a per-item correction).
    """
    d = np.asarray(distances, dtype=np.float64)
    t = np.asarray(onsets_s, dtype=np.float64)
    if d.shape != t.shape or d.ndim != 1:
        raise ValueError(f"distances/onsets must be matching 1-D arrays, got {d.shape}/{t.shape}")
    if d.size < 3:
        raise ValueError(f"need at least 3 points for a delay fit, got {d.size}")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(t)):
        raise ValueError("delay fit inputs must be finite")
    spread = float(d.max() - d.min())
    if spread <= 1e-6:
        raise ValueError(
            f"distances span only {spread:.3g} m: the slope would be unidentifiable")

    slope, intercept = np.polyfit(d, t, 1)
    predicted = slope * d + intercept
    ss_res = float(np.sum((t - predicted) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    expected = 1.0 / SPEED_OF_SOUND
    ratio = float(slope) / expected

    reasons = []
    if r2 < min_r2:
        reasons.append(f"r2 {r2:.4f} < {min_r2}")
    if abs(ratio - 1.0) > tolerance:
        reasons.append(f"slope ratio {ratio:.4f} outside 1 +- {tolerance}")
    return {
        "n": int(d.size),
        "slope_s_per_m": float(slope),
        "intercept_s": float(intercept),
        "implied_constant_delay_s": float(intercept),
        "expected_slope_s_per_m": expected,
        "slope_ratio": ratio,
        "r2": float(r2),
        "residual_std_s": float(np.std(t - predicted)),
        "distance_span_m": spread,
        "threshold_db": float(ONSET_THRESHOLD_DB),
        "tolerance": float(tolerance),
        "min_r2": float(min_r2),
        "passed": not reasons,
        "reasons": reasons,
    }


def t30_validity(waves, sr=22050, crop=LOADER_CROP, decay_db=T30_DECAY_DB):
    """Per-item T30 measurability over the full recording vs the metric crop.

    Mirrors the metric stack's policy (pyroomacoustics ``measure_rt60`` at 30 dB,
    which is what ``RT60Error`` uses for HAA/RAF) so the rate reported here is the
    rate the evaluation will actually see.
    """
    import pyroomacoustics

    def _valid(x):
        if x.size == 0 or not np.isfinite(x).all() or float(np.abs(x).max()) <= 0.0:
            return False
        try:
            value = pyroomacoustics.experimental.measure_rt60(
                np.asarray(x, dtype=np.float64), fs=sr, decay_db=decay_db)
        except (ValueError, IndexError):
            return False
        return bool(np.isfinite(value) and value > 0)

    full = [_valid(np.asarray(w)) for w in waves]
    cropped = [_valid(np.asarray(w)[:crop]) for w in waves]
    n = len(waves)
    return {
        "n": n,
        "sr": int(sr),
        "crop_samples": int(crop),
        "decay_db": int(decay_db),
        "valid_full": int(sum(full)),
        "invalid_full": int(n - sum(full)),
        "valid_crop": int(sum(cropped)),
        "invalid_crop": int(n - sum(cropped)),
        "valid_rate_full": float(sum(full) / n) if n else 0.0,
        "valid_rate_crop": float(sum(cropped) / n) if n else 0.0,
        "crop_invalidates": int(sum(1 for a, b in zip(full, cropped) if a and not b)),
        "crop_invalidation_rate": (
            float(sum(1 for a, b in zip(full, cropped) if a and not b) / sum(full))
            if sum(full) else 0.0),
        "per_item_full": full,
        "per_item_crop": cropped,
    }


def _rotation_matrix(w, x, y, z):
    """Rotation matrix of the unit quaternion (w, x, y, z)."""
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n <= 0:
        raise ValueError("cannot build a rotation from a zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_forward_diagnostics(quat, forward_raf=(1.0, 0.0, 0.0)):
    """Where the source points under BOTH readings of the raw 4-tuple.

    The order is PINNED as xyzw (real part last), stated verbatim by the RAF release
    documentation and consistent with these diagnostics -- horizontal speakers under
    xyzw, 90-degree tilted ones under wxyz. Both readings are still rendered on
    every audit so the pin keeps showing its evidence rather than becoming folklore;
    v1 never rotates with the quaternion in any case (grouping only).
    """
    q = [float(v) for v in np.asarray(quat, dtype=np.float64).ravel()]
    if len(q) != 4:
        raise ValueError(f"quaternion must have 4 components, got {len(q)}")
    forward = np.asarray(forward_raf, dtype=np.float64)
    readings = {
        "wxyz": _rotation_matrix(q[0], q[1], q[2], q[3]),
        "xyzw": _rotation_matrix(q[3], q[0], q[1], q[2]),
    }
    out = {}
    for name, R in readings.items():
        f_raf = R @ forward
        out[name] = {
            "forward_raf": [float(v) for v in f_raf],
            "forward_pipeline": [float(v) for v in (RAF_TO_PIPELINE @ f_raf)],
            "elevation_deg": float(np.degrees(np.arcsin(
                np.clip(f_raf[1] / max(np.linalg.norm(f_raf), 1e-12), -1.0, 1.0)))),
        }
    # A useful tell: an identity rotation reads as (1,0,0,0) under wxyz and as
    # (0,0,0,1) under xyzw, so a corpus of near-(1,0,0,0) tuples is wxyz-shaped.
    if abs(q[0]) > 0.99 and max(abs(v) for v in q[1:]) < 0.01:
        identity = "wxyz"
    elif abs(q[3]) > 0.99 and max(abs(v) for v in q[:3]) < 0.01:
        identity = "xyzw"
    else:
        identity = "ambiguous"
    out["identity_quat_reading"] = identity
    return out


def audit_room(room_dir, room, n_onset_samples=200, seed=0, crosscheck_sample=200,
               full_crosscheck=False):
    """Measure one room: onsets, T30 validity, amplitude, quaternion readings."""
    from prepare_data import crosscheck_captures, group_captures, load_room_index

    index = load_room_index(room_dir)
    crosscheck = crosscheck_captures(room_dir, index, n_sample=crosscheck_sample,
                                     seed=seed, full=full_crosscheck)
    groups, group_report = group_captures(index, allow_nonuniform=True)

    rng = np.random.default_rng(seed)
    n = min(int(n_onset_samples), len(index))
    chosen = sorted(int(i) for i in rng.choice(len(index), size=n, replace=False))

    distances, onsets_s, peaks, waves, failures = [], [], [], [], []
    for i in chosen:
        record = index[i]
        path = os.path.join(room_dir, "data", record["capture_id"], "rir.wav")
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = wave[:, 0]
        peaks.append(float(np.abs(wave).max()))
        waves.append(wave)
        try:
            onset = detect_onset(wave)
        except ValueError as e:
            failures.append({"capture_id": record["capture_id"], "reason": str(e)})
            continue
        distances.append(float(np.linalg.norm(record["rx_xyz"] - record["tx_xyz"])))
        onsets_s.append(onset / float(sr))

    onset_fit = fit_constant_delay(distances, onsets_s)
    validity = t30_validity([w for w in waves], sr=sr, crop=LOADER_CROP)
    validity.pop("per_item_full", None)
    validity.pop("per_item_crop", None)

    quaternion = {"wxyz": [], "xyzw": [], "identity_readings": {}}
    for g in groups[:min(len(groups), 50)]:
        diag = quaternion_forward_diagnostics(g["quat_canon"])
        quaternion["wxyz"].append(diag["wxyz"]["forward_raf"])
        quaternion["xyzw"].append(diag["xyzw"]["forward_raf"])
        key = diag["identity_quat_reading"]
        quaternion["identity_readings"][key] = quaternion["identity_readings"].get(key, 0) + 1

    return {
        "n_captures": len(index),
        "room_index": {
            "n_captures": len(index),
            "all_tx_pos_sha256": sha256_of(os.path.join(room_dir, "metadata", "all_tx_pos.txt")),
            "all_rx_pos_sha256": sha256_of(os.path.join(room_dir, "metadata", "all_rx_pos.txt")),
            "rx_trailing_sentinel_dropped": bool(index.rx_trailing_sentinel_dropped),
        },
        "n_groups": group_report["n_groups"],
        "size_histogram": group_report["size_histogram"],
        "nonuniform": group_report["nonuniform"],
        "crosscheck": crosscheck,
        "rx_trailing_sentinel_dropped": index.rx_trailing_sentinel_dropped,
        "onset": onset_fit,
        "onset_failures": failures,
        "t30_validity": validity,
        "amplitude": {
            "peak_stats": _distance_stats(peaks),
            "dbfs_stats": _distance_stats([_dbfs(p) for p in peaks]),
            "n_below_60dbfs": int(sum(1 for p in peaks if _dbfs(p) < -60.0)),
            "source_sr": int(sr),
        },
        "quaternion": quaternion,
    }


def build_record(rooms, params, pin_gauge=None, pin_quat=None):
    """Assemble the record and compute the mechanical verdict."""
    reasons = []
    for room, payload in rooms.items():
        if not payload["onset"]["passed"]:
            reasons.extend(f"{room}: onset fit {r}" for r in payload["onset"]["reasons"])

    worst_invalidation = max(
        (p["t30_validity"]["crop_invalidation_rate"] for p in rooms.values()),
        default=0.0)
    resolution = ("headline" if worst_invalidation <= T60_CROP_INVALIDATION_LIMIT
                  else "demoted")
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "created_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "params": params,
        "rooms": rooms,
        "decisions": {
            "t60_headline": {
                "rule": ("T60 stays a headline metric iff cropping to the metric "
                         f"window invalidates <= {T60_CROP_INVALIDATION_LIMIT:.0%} of "
                         "the RIRs that are valid over the full recording"),
                "worst_crop_invalidation_rate": float(worst_invalidation),
                "limit": T60_CROP_INVALIDATION_LIMIT,
                "resolution": resolution,
            },
            "amplitude_scalar": {
                "rule": ("none unless the audit shows RAF off-scale versus HAA/AR; "
                         "if applied, ONE scalar applied identically to targets and "
                         "context"),
                "derived_from": "train supports only",
                "applied_scalar": None,
            },
            "delay_handling": {
                "rule": "constant acquisition delay only, never a per-item correction",
                "per_room_delay_s": {r: p["onset"]["implied_constant_delay_s"]
                                     for r, p in rooms.items()},
            },
        },
        # The Planner pins these FROM this artifact; the publish gate refuses to
        # run while either is null, which is what stops canonical outputs being
        # produced under a gauge that has not been adjudicated from this audit.
        "adjudication": {"gauge_pinned": pin_gauge, "quat_order_pinned": pin_quat},
        "verdict": {"passed": not reasons, "reasons": reasons},
    }


def sha256_of(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_record_once(path):
    """Read the record ONCE and return ``(payload, digest, parsed)`` (T1).

    One open, one read: the digest describes exactly the bytes that were parsed.
    Parsing and then reopening to hash left a window in which a swapped file could
    supply forged content to the parser and the pinned bytes to the hasher.
    """
    import hashlib

    with open(path, "rb") as f:
        payload = f.read()
    digest = hashlib.sha256(payload).hexdigest()
    record = json.loads(payload.decode("utf-8"))
    return payload, digest, record


def room_index_digests(room_dir):
    """Corpus fingerprint the audit can record and the gates can re-verify (T1).

    The pose files ARE the index the whole preparation rests on, and they are small
    -- so their digests bind a publication to the corpus it was audited against
    without hashing 43 GB of audio (out of scope per the registered threat model).
    """
    from prepare_data import _capture_dirs, load_room_index

    meta = os.path.join(room_dir, "metadata")
    index = load_room_index(room_dir)
    return {
        "n_captures": len(index),
        "all_tx_pos_sha256": sha256_of(os.path.join(meta, "all_tx_pos.txt")),
        "all_rx_pos_sha256": sha256_of(os.path.join(meta, "all_rx_pos.txt")),
        "rx_trailing_sentinel_dropped": bool(index.rx_trailing_sentinel_dropped),
    }


def verify_corpus_binding(record, raf_root, rooms):
    """Re-verify the record's corpus fingerprint against the corpus being read.

    Returns a list of problems (empty when consistent). Records that carry
    ``room_index`` are checked on the file digests; the currently pinned record
    predates that block, so it falls back to the capture counts it does carry --
    enough to catch the operator error of pointing a canonical run at a different
    corpus, which is the registered threat model.
    """
    problems = []
    for room in rooms:
        payload = (record.get("rooms") or {}).get(room)
        if payload is None:
            problems.append(f"{room}: not audited by this record")
            continue
        room_dir = os.path.join(raf_root, "archived", room)
        recorded = payload.get("room_index")
        try:
            if recorded:
                actual = room_index_digests(room_dir)
                for key in sorted(recorded):
                    if actual.get(key) != recorded[key]:
                        problems.append(
                            f"{room}: {key} is {actual.get(key)!r} on disk but the "
                            f"record audited {recorded[key]!r}")
            else:
                from prepare_data import _capture_dirs

                n_captures = len(_capture_dirs(room_dir))
                if n_captures != payload.get("n_captures"):
                    problems.append(
                        f"{room}: capture count {n_captures} on disk but the record "
                        f"audited {payload.get('n_captures')}")
        except (OSError, ValueError) as e:
            problems.append(f"{room}: corpus unreadable for binding ({e})")
    return problems


def assert_canonical_content(record, path):
    """Every content rule canonical publication depends on (S1).

    Checked separately from the digest so the rules are testable, and so a failure
    says WHICH fact is missing rather than only that a hash differed.
    """
    adjudication = record.get("adjudication") or {}
    gauge = adjudication.get("gauge_pinned")
    quat = adjudication.get("quat_order_pinned")
    if gauge != CANONICAL_GAUGE:
        raise ValueError(
            f"{path}: gauge_pinned is {gauge!r}, canonical publication requires "
            f"{CANONICAL_GAUGE!r} (Amendment 4 pinned it from the readback evidence)")
    if quat != CANONICAL_QUAT_ORDER:
        raise ValueError(
            f"{path}: quat_order_pinned is {quat!r}, canonical publication requires "
            f"{CANONICAL_QUAT_ORDER!r} (the RAF release states real-part-last)")

    rooms = record.get("rooms") or {}
    if set(rooms) != set(CANONICAL_ROOMS):
        raise ValueError(
            f"{path}: audited rooms are {sorted(rooms)}, canonical publication "
            f"requires exactly {sorted(CANONICAL_ROOMS)}")
    for room, payload in sorted(rooms.items()):
        missing = [block for block in REQUIRED_ROOM_BLOCKS if block not in payload]
        if missing:
            raise ValueError(
                f"{path}: room {room} is missing measurement blocks {missing}; the "
                "pins would be assertions rather than evidence")
        if payload["onset"].get("passed") is not True:
            raise ValueError(
                f"{path}: room {room}'s onset-vs-distance fit did not pass "
                f"({payload['onset'].get('reasons')})")
    if not (record.get("decisions", {}).get("t60_headline", {}).get("resolution")):
        raise ValueError(f"{path}: the T60 headline decision is unresolved")

    # T1: every sub-verdict, not only block presence and onset.passed.
    for room, payload in sorted(rooms.items()):
        t30 = payload["t30_validity"]
        if not t30.get("n") or not t30.get("valid_full"):
            raise ValueError(
                f"{path}: room {room}'s t30 validity block measured nothing "
                f"(n={t30.get('n')}, valid_full={t30.get('valid_full')})")
        amplitude = payload["amplitude"]
        if not (amplitude.get("peak_stats") or {}).get("count"):
            raise ValueError(
                f"{path}: room {room}'s amplitude block measured no files")
        crosscheck = payload["crosscheck"]
        if not crosscheck.get("checked") or crosscheck.get("mismatches"):
            raise ValueError(
                f"{path}: room {room}'s per-capture cross-check checked "
                f"{crosscheck.get('checked')} captures with "
                f"{crosscheck.get('mismatches')} mismatches")
        quaternion = payload["quaternion"]
        if not quaternion.get("identity_readings"):
            raise ValueError(
                f"{path}: room {room} carries no quaternion order diagnostics, so "
                "the xyzw pin rests on nothing")
        if payload.get("onset_failures"):
            raise ValueError(
                f"{path}: room {room} has {len(payload['onset_failures'])} onset "
                "detection failures")
    return record


def load_passing_record(path, canonical=True, expected_raf_root=None):
    """Publish gate: return the record, or raise saying exactly what is missing.

    ``canonical=True`` (the default, i.e. production) authenticates the record
    against the pinned digest AND re-checks every content rule. ``canonical=False``
    is the explicit synthetic/test mode: the structural checks still apply, but
    the outputs it authorises are TAINTED in every artifact.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"readback record not found: {path}. Canonical RAF artifacts may only be "
            "published from a passing readback audit (data/RAF/readback_audit.py).")
    _payload, digest, record = read_record_once(path)
    # The digest of the bytes that were actually parsed travels with the record, so
    # provenance can never describe a different file than the gate validated (T1).
    record["__authenticated_sha256__"] = digest
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"readback record {path} has schema_version "
            f"{record.get('schema_version')!r}, expected {RECORD_SCHEMA_VERSION}")
    verdict = record.get("verdict") or {}
    if verdict.get("passed") is not True:
        raise ValueError(
            f"readback record {path} did not pass: {verdict.get('reasons')}")
    adjudication = record.get("adjudication") or {}
    missing = [k for k in ("gauge_pinned", "quat_order_pinned")
               if not adjudication.get(k)]
    if missing:
        raise ValueError(
            f"readback record {path} is unadjudicated: {missing} still null. The "
            "gauge and the quaternion column order must be PINNED from the audit "
            "before canonical artifacts are published.")

    if canonical:
        if digest != CANONICAL_RECORD_SHA256:
            raise ValueError(
                f"readback record {path} has sha256 {digest}, which is not the "
                f"pinned canonical record ({CANONICAL_RECORD_SHA256}). Canonical "
                "artifacts are published only under the committed record; pass "
                "--non-canonical for synthetic runs (their outputs are tainted).")
        assert_canonical_content(record, path)
        if expected_raf_root is not None:
            audited_root = record.get("params", {}).get("raf_root")
            if os.path.abspath(str(audited_root)) != os.path.abspath(str(expected_raf_root)):
                raise ValueError(
                    f"readback record {path} audited raf_root {audited_root!r}, but "
                    f"this run reads {expected_raf_root!r}: the audit does not "
                    "describe this corpus")
            # A pathname authenticates nothing behind a moved mount, so the corpus
            # itself is fingerprinted too (T1).
            problems = verify_corpus_binding(record, expected_raf_root,
                                             list(record.get("rooms") or {}))
            if problems:
                raise ValueError(
                    f"readback record {path} does not describe the corpus at "
                    f"{expected_raf_root}: " + "; ".join(problems))
    return record


def record_provenance(path, record, canonical=True):
    """Compact provenance block for the artifacts published under this record."""
    digest = record.get("__authenticated_sha256__") or sha256_of(path)
    binding = []
    for room, payload in sorted((record.get("rooms") or {}).items()):
        binding.append(
            f"{room}: pose-file digests"
            if payload.get("room_index") else
            f"{room}: capture counts only (record predates the room_index block)")
    taint = [] if canonical else [
        "non-canonical publication: the readback record was not authenticated "
        f"against the pinned {CANONICAL_RECORD_SHA256[:12]} record"]
    return {
        "path": os.path.abspath(path),
        "sha256": digest,
        "canonical": bool(canonical),
        "taint": taint,
        "corpus_binding": binding,
        "schema_version": record["schema_version"],
        "created_utc": record.get("created_utc"),
        "gauge_pinned": record["adjudication"]["gauge_pinned"],
        "quat_order_pinned": record["adjudication"]["quat_order_pinned"],
        "t60_headline": record["decisions"]["t60_headline"]["resolution"],
    }


def build_parser():
    parser = argparse.ArgumentParser(description="RAF readback audit (exp_19 R4)")
    parser.add_argument('--raf-root', required=True)
    parser.add_argument('--rooms', nargs='+', default=['EmptyRoom', 'FurnishedRoom'])
    parser.add_argument('--out', default='data/RAF/raf_readback_record.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-onset-samples', type=int, default=200)
    parser.add_argument('--crosscheck-sample', type=int, default=200)
    parser.add_argument('--full-crosscheck', action='store_true')
    parser.add_argument('--pin-gauge', default=None,
                        help="name of the gauge the Planner pins from this audit")
    parser.add_argument('--pin-quat', default=None, choices=['wxyz', 'xyzw'],
                        help="quaternion column order the Planner pins from this audit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rooms = {}
    for room in args.rooms:
        room_dir = os.path.join(args.raf_root, "archived", room)
        logger.info("auditing %s", room_dir)
        rooms[room] = audit_room(room_dir, room, n_onset_samples=args.n_onset_samples,
                                 seed=args.seed,
                                 crosscheck_sample=args.crosscheck_sample,
                                 full_crosscheck=args.full_crosscheck)
        fit = rooms[room]["onset"]
        logger.info("%s: onset slope %.6g s/m (%.1f%% of 1/343), r2 %.4f, delay %.6g s -> %s",
                    room, fit["slope_s_per_m"], 100 * fit["slope_ratio"], fit["r2"],
                    fit["intercept_s"], "PASS" if fit["passed"] else "FAIL")

    params = {"raf_root": args.raf_root, "rooms": list(args.rooms), "seed": args.seed,
              "n_onset_samples": args.n_onset_samples,
              "crosscheck": "full" if args.full_crosscheck else f"sample:{args.crosscheck_sample}"}
    record = build_record(rooms, params, pin_gauge=args.pin_gauge, pin_quat=args.pin_quat)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=4, allow_nan=False)
    logger.info("readback record -> %s (verdict: %s)", args.out,
                "PASS" if record["verdict"]["passed"] else "FAIL")
    return 0 if record["verdict"]["passed"] else 1


if __name__ == '__main__':
    raise SystemExit(main())
