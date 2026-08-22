"""Mapping-A preparation: audio union, amplitude policy, items, publication.

exp_21, contract section B. Mapping A needs a different audio population from
Mapping H: exp_19 published only the 21 selected tx-groups per room, while an
item here conditions a target on OTHER SOURCES heard by the same microphone, so
the union of target + context captures spans far more of the corpus (~10,368
files).

Two things follow, and both are registered (Codex M1):

* that union has to be amplitude-audited BEFORE anything is written, because the
  x3 scalar was derived from Mapping H's trained supports -- a different
  population, which says nothing about these files;
* any violation STOPS the run with a measured report. Never drop an item (the
  manifest would then describe a set that was never evaluated) and never
  auto-adjust the scalar (a silently different normalisation is not comparable to
  the Mapping-H publication it will be read beside).
"""
import os
import sys

import librosa
import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # sibling scripts, not an installed package
    sys.path.insert(0, _HERE)
from prepare_data import (  # noqa: E402
    CLIP_CEILING,
    SILENCE_THRESHOLD_DB,
    SOURCE_SR,
    TARGET_SR,
)
from raf_common import dbfs  # noqa: E402


class AmplitudePolicyError(RuntimeError):
    """The audio union violates the registered amplitude policy.

    Carries the measured ``report`` so the run stops with evidence rather than a
    message: the decision that follows (a different scalar, a different ceiling, or
    accepting the loss) is Yixun's, and it needs the numbers.
    """

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def enumerate_audio_union(items):
    """The EXACT set of captures Mapping A needs, per room.

    Deduplicated: one context capture serves many items, and writing it per item
    would multiply the corpus by K. Returns ``{room: [capture_id, ...]}`` sorted, so
    the enumeration is reproducible and hashable.
    """
    union = {}
    for item in items:
        room = item["room"]
        target = str(item["target_capture_id"])
        context = [str(c["capture_id"]) for c in item.get("context", [])]
        if not context:
            raise ValueError(
                f"{room} {item.get('placement_id')} slot {item.get('mic_slot')}: "
                "item has no context captures")
        if target in context:
            raise ValueError(
                f"{room} {item.get('placement_id')} slot {item.get('mic_slot')}: "
                f"target capture {target} appears in its own context, which would "
                "leak the answer into the conditioning")
        bucket = union.setdefault(room, set())
        bucket.add(target)
        bucket.update(context)
    return {room: sorted(ids) for room, ids in sorted(union.items())}


def union_report(union, items):
    """Counts behind the compute estimate (plan section 7 forbids unmeasured quotes)."""
    per_room = {}
    for room, ids in union.items():
        per_room[room] = {
            "n_captures": len(ids),
            "n_items": sum(1 for i in items if i["room"] == room),
        }
    return {
        "n_items": len(items),
        "n_captures": sum(len(ids) for ids in union.values()),
        "rooms": sorted(union),
        "by_room": per_room,
    }


def _resampled_peak_and_finite(path, orig_sr, target_sr):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != orig_sr:
        raise ValueError(f"{path}: expected {orig_sr} Hz, got {sr} Hz")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {audio.shape[1]} channels")
    wave = audio[:, 0]
    finite = bool(np.isfinite(wave).all())
    if not finite or wave.size == 0:
        return 0.0, finite, wave.size
    out = np.asarray(librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr),
                     dtype=np.float32)
    if not np.isfinite(out).all():
        return 0.0, False, int(out.size)
    return float(np.abs(out).max()), True, int(out.size)


def largest_admissible_scalar(max_peak, clip_ceiling=CLIP_CEILING):
    """The largest scalar that keeps this union clip-free -- MEASURED, not applied."""
    if max_peak <= 0:
        return float("inf")
    return float(clip_ceiling / max_peak)


def audit_amplitude_union(room_dir, capture_ids, scalar, orig_sr=SOURCE_SR,
                          target_sr=TARGET_SR, clip_ceiling=CLIP_CEILING,
                          silence_db=SILENCE_THRESHOLD_DB):
    """Amplitude audit over the exact Mapping-A union, BEFORE anything is written.

    Three registered conditions, measured on the RESAMPLED signal (what would
    actually be published):

    * ``peak * scalar <= clip_ceiling`` -- the loader clamps to [-1, 1], so a
      clipped file is a distorted target scored against an undistorted reference;
    * ``dBFS(peak * scalar) >= silence_db`` -- below the loader's gate the item is
      silently substituted, so the manifest would describe an item nobody evaluated;
    * finite and non-empty.

    Returns the measured report when clean. Otherwise raises
    ``AmplitudePolicyError`` carrying the same report: every violation, worst
    first, plus the largest scalar this union would admit -- offered as evidence for
    the decision, never applied.
    """
    peaks, violations = [], []
    for capture_id in capture_ids:
        path = os.path.join(room_dir, "data", capture_id, "rir.wav")
        peak, finite, n_samples = _resampled_peak_and_finite(path, orig_sr, target_sr)
        scaled = peak * scalar
        record = {
            "capture_id": capture_id,
            "raw_peak": peak,
            "scaled_peak": scaled,
            "scaled_dbfs": dbfs(scaled),
            "n_samples": n_samples,
        }
        if not finite or n_samples == 0:
            violations.append(dict(record, kind="non_finite"))
            continue
        peaks.append(peak)
        if peak <= 0.0:
            violations.append(dict(record, kind="silent_source"))
        elif scaled > clip_ceiling:
            violations.append(dict(record, kind="clipping"))
        elif dbfs(scaled) < silence_db:
            violations.append(dict(record, kind="below_threshold"))

    max_peak = max(peaks) if peaks else 0.0
    violations.sort(key=lambda v: (-v["scaled_peak"], v["capture_id"]))
    report = {
        "scalar": float(scalar),
        "clip_ceiling": float(clip_ceiling),
        "silence_threshold_db": float(silence_db),
        "source_sample_rate": int(orig_sr),
        "sample_rate": int(target_sr),
        "n_captures": len(capture_ids),
        "max_raw_peak": max_peak,
        "max_scaled_peak": max_peak * scalar,
        "min_raw_peak": min(peaks) if peaks else 0.0,
        "min_scaled_dbfs": dbfs(min(peaks) * scalar) if peaks else dbfs(0.0),
        "n_clipping": sum(1 for v in violations if v["kind"] == "clipping"),
        "n_below_threshold": sum(1 for v in violations
                                 if v["kind"] == "below_threshold"),
        "n_non_finite": sum(1 for v in violations if v["kind"] == "non_finite"),
        # Measured OPTION, never applied: the run stops for a human decision.
        "max_admissible_scalar": largest_admissible_scalar(max_peak, clip_ceiling),
        "violations": violations,
        "passed": not violations,
        "decision_required": bool(violations),
    }
    if violations:
        raise AmplitudePolicyError(
            f"{room_dir}: the Mapping-A audio union violates the registered "
            f"amplitude policy at scalar x{scalar} -- {report['n_clipping']} "
            f"clipping, {report['n_below_threshold']} below {silence_db} dBFS, "
            f"{report['n_non_finite']} non-finite of {len(capture_ids)} captures. "
            "STOPPING for a registered amplitude-policy decision: items are never "
            "dropped and the scalar is never auto-adjusted (the largest scalar this "
            f"union admits is {report['max_admissible_scalar']:.4f}, recorded as "
            "evidence, not applied).",
            report)
    return report
