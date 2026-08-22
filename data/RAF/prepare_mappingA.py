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
from publish import sha256_file  # noqa: E402
from mappingA_common import (  # noqa: E402
    MATCH_AMBIGUITY_MARGIN,
    MATCH_MAX_M,
    MATCH_P95_M,
    select_target,
    source_xyz_key,
    stable_item_context,
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


# --------------------------------------------------------------------------- #
# Writing the union
# --------------------------------------------------------------------------- #
RIR_FOLDER = "mono_rirs_22050Hz"


def write_union(room_dir, out_room_dir, capture_ids, scalar, orig_sr=SOURCE_SR,
                target_sr=TARGET_SR, clip_ceiling=CLIP_CEILING,
                mappingH_room_dir=None, mappingH_generation=None):
    """Resample, scale and publish the Mapping-A union, with H provenance.

    Mapping A publishes into DISJOINT roots, so "shared with Mapping H" is a
    provenance claim rather than a storage trick: for every capture that the
    Mapping-H publication also holds, the bytes are compared and recorded as
    verified-identical together with the exp_19 generation. A disagreement aborts --
    two publications differing on the same capture means a different scalar, a
    different source, or a stale generation, and reading Mapping-A results beside
    Mapping-H ones would then be comparing different audio.

    Every written file is read back and compared, as in exp_19.
    """
    dest = os.path.join(out_room_dir, RIR_FOLDER)
    os.makedirs(dest, exist_ok=True)
    files, n_shared, roundtrip_max = {}, 0, 0.0

    for capture_id in capture_ids:
        src = os.path.join(room_dir, "data", capture_id, "rir.wav")
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        if sr != orig_sr:
            raise ValueError(f"{src}: expected {orig_sr} Hz, got {sr} Hz")
        if audio.shape[1] != 1:
            raise ValueError(f"{src}: expected mono, got {audio.shape[1]} channels")
        wave = audio[:, 0]
        if not np.isfinite(wave).all():
            raise ValueError(f"{src}: source holds non-finite samples")

        out = np.asarray(librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr),
                         dtype=np.float32)
        out = np.asarray(out * float(scalar), dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError(f"{src}: resampling produced non-finite samples")
        peak = float(np.abs(out).max())
        if peak > clip_ceiling:
            raise ValueError(
                f"{src}: scaled signal clips (peak {peak:.6f} > {clip_ceiling}) at "
                f"x{scalar}. The union audit exists to catch this before any write; "
                "publishing a clipped target would distort the waveform the metric "
                "scores.")

        out_path = os.path.join(dest, f"{capture_id}.wav")
        sf.write(out_path, out, target_sr, subtype="FLOAT")
        back, back_sr = sf.read(out_path, dtype="float32", always_2d=True)
        if back_sr != target_sr or back.shape[1] != 1 or back.shape[0] != out.shape[0]:
            raise ValueError(
                f"{out_path}: read back as {back_sr} Hz / {back.shape} , expected "
                f"{target_sr} Hz mono / {out.shape}")
        roundtrip = float(np.abs(back[:, 0] - out).max())
        roundtrip_max = max(roundtrip_max, roundtrip)
        digest = sha256_file(out_path)

        shared = None
        if mappingH_room_dir:
            h_path = os.path.join(mappingH_room_dir, RIR_FOLDER, f"{capture_id}.wav")
            if os.path.isfile(h_path):
                h_digest = sha256_file(h_path)
                if h_digest != digest:
                    raise ValueError(
                        f"capture {capture_id} is NOT byte-identical to the "
                        f"Mapping-H publication ({h_digest[:12]} vs {digest[:12]}): "
                        "the two publications disagree about the same audio, so a "
                        "different scalar, source file or generation is in play")
                shared = {"path": os.path.abspath(h_path), "sha256": h_digest,
                          "generation": mappingH_generation,
                          "verified_identical": True}
                n_shared += 1

        files[capture_id] = {
            "sha256": digest,
            "peak": peak,
            "dbfs": dbfs(peak),
            "n_samples": int(out.shape[0]),
            "roundtrip_max_abs_error": roundtrip,
            "shared_with_mappingH": shared,
        }

    return {
        "n_files": len(files),
        "n_shared_with_mappingH": n_shared,
        "n_new": len(files) - n_shared,
        "scalar": float(scalar),
        "source_sample_rate": int(orig_sr),
        "sample_rate": int(target_sr),
        "subtype": "FLOAT",
        "mappingH_generation": mappingH_generation,
        "roundtrip_max_abs_error": roundtrip_max,
        "files": files,
    }


# --------------------------------------------------------------------------- #
# Items and the static manifest validator (plan section 3, M5)
# --------------------------------------------------------------------------- #
CANONICAL_N_PLACEMENTS = 16
CANONICAL_K = 8
CANONICAL_ARRAY_SIZE = 36
CANONICAL_N_ITEMS = CANONICAL_N_PLACEMENTS * CANONICAL_ARRAY_SIZE * 2  # both rooms


class ManifestError(ValueError):
    """The Mapping-A manifest violates a registered invariant.

    Carries the full ``report`` -- every violation, not the first -- because the
    manifest is what every arm and seed conditions on, and a partial diagnosis
    would send the run round again.
    """

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def build_items(room, placement_id, poses, assignment, match, k=CANONICAL_K, seed=0):
    """One item per microphone slot for a placement.

    The target POSE is chosen once per placement (hash-uniform, M8) and then held
    out at every mic slot, so the 36 items of a placement differ only in which
    microphone is listening -- which is exactly the quantity Mapping A varies. The
    context is drawn per item, so two slots do not share a conditioning set by
    construction.
    """
    target = select_target(room, placement_id, poses, seed=seed)
    target_key = str(target["group_key"])
    items = []
    for slot in range(CANONICAL_ARRAY_SIZE):
        drawn = stable_item_context(room, placement_id, slot, target, poses, k=k,
                                    seed=seed)
        target_row = assignment[target_key][slot]
        rx_target = np.asarray(target["rx_xyz_p"], dtype=np.float64)[target_row]
        context = []
        for pose in drawn["context"]:
            key = str(pose["group_key"])
            row = assignment[key][slot]
            rx = np.asarray(pose["rx_xyz_p"], dtype=np.float64)[row]
            context.append({
                "capture_id": pose["capture_ids"][row],
                "group_key": key,
                "xyz_key": source_xyz_key(pose["tx_xyz"]),
                "tx_p": [float(v) for v in pose["tx_xyz_p"]],
                "rx_p": [float(v) for v in rx],
                # every context's own receiver is recorded, and its displacement
                # from the target's receiver bounds the same-listener claim (M3)
                "rx_displacement_m": float(np.linalg.norm(rx - rx_target)),
            })
        items.append({
            "item_id": f"{room}/{placement_id}/slot{slot:02d}",
            "room": room,
            "placement_id": placement_id,
            "mic_slot": slot,
            "target_capture_id": target["capture_ids"][target_row],
            "target_group_key": target_key,
            "target_xyz_key": source_xyz_key(target["tx_xyz"]),
            "tx_p": [float(v) for v in target["tx_xyz_p"]],
            "rx_target_p": [float(v) for v in rx_target],
            "rx_target_height_raf_m": float(rx_target[2]),
            "depth_file": f"{room}_{placement_id}_slot{slot:02d}_depth_image.npy",
            "context": context,
            "match": dict(match[target_key]),
            "context_digest": drawn["digest"],
            "context_pool_size": drawn["pool_size"],
        })
    return items


def validate_manifest(manifest, expected_items=CANONICAL_N_ITEMS, k=CANONICAL_K,
                      max_displacement_m=MATCH_MAX_M):
    """Static validator over the whole manifest (M5), run before publication.

    Every condition here is a way the row could stop measuring what it claims:
    a repeated target (the same problem scored twice), a context holding the
    answer, a context source standing where the target stands (the "unseen source
    position" claim), or a context recorded at a different microphone (the
    "same listener" claim). All violations are collected before raising.
    """
    items = manifest["items"]
    violations = []

    def add(kind, item, detail):
        violations.append({"kind": kind, "item_id": item.get("item_id"),
                           "detail": detail})

    seen_items, seen_targets = {}, {}
    for item in items:
        item_id = item["item_id"]
        if item_id in seen_items:
            add("duplicate_item", item, f"item id {item_id} appears twice")
        seen_items[item_id] = True

        target = item["target_capture_id"]
        if target in seen_targets:
            add("duplicate_target", item,
                f"target capture {target} is also item {seen_targets[target]}")
        seen_targets[target] = item_id

        context = item["context"]
        capture_ids = [c["capture_id"] for c in context]
        if len(context) != k:
            add("context_size", item, f"{len(context)} context captures, expected {k}")
        if len(set(capture_ids)) != len(capture_ids):
            add("context_not_distinct", item,
                f"context captures are not distinct: {sorted(capture_ids)}")
        if target in capture_ids:
            add("target_in_context", item,
                f"target capture {target} appears in its own context")
        for entry in context:
            if entry["xyz_key"] == item["target_xyz_key"]:
                add("context_source_position", item,
                    f"context {entry['capture_id']} stands at the target source "
                    f"position {entry['xyz_key']}")
            if entry["rx_displacement_m"] > max_displacement_m:
                add("mic_displacement", item,
                    f"context {entry['capture_id']} was recorded "
                    f"{entry['rx_displacement_m']:.4f} m from the target microphone "
                    f"(> {max_displacement_m} m)")

        match = item.get("match") or {}
        if match.get("p95_m", 0.0) > MATCH_P95_M:
            add("failed_match", item,
                f"correspondence p95 {match['p95_m']:.4f} m > {MATCH_P95_M} m")
        if match.get("max_m", 0.0) > MATCH_MAX_M:
            add("failed_match", item,
                f"correspondence max {match['max_m']:.4f} m > {MATCH_MAX_M} m")
        if match.get("min_ambiguity_margin", float("inf")) < MATCH_AMBIGUITY_MARGIN:
            add("ambiguous_match", item,
                f"correspondence margin {match['min_ambiguity_margin']:.2f} < "
                f"{MATCH_AMBIGUITY_MARGIN}")

    report = {
        "n_items": len(items),
        "expected_items": int(expected_items),
        "k": int(k),
        "n_unique_targets": len(seen_targets),
        "n_unique_item_ids": len(seen_items),
        "max_displacement_m": float(max_displacement_m),
        "violations": violations,
        "passed": not violations and len(items) == expected_items,
    }
    if len(items) != expected_items:
        report["violations"] = violations + [{
            "kind": "item_count", "item_id": None,
            "detail": f"{len(items)} items, expected {expected_items}"}]
        raise ManifestError(
            f"Mapping-A manifest holds {len(items)} items, expected {expected_items}",
            report)
    if violations:
        kinds = sorted({v["kind"] for v in violations})
        raise ManifestError(
            f"Mapping-A manifest violates {len(violations)} registered invariants "
            f"({kinds}); first: {violations[0]['detail']}", report)
    return report
