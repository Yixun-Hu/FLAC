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

import argparse
import hashlib
import json
import logging
import math

import librosa
import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # sibling scripts, not an installed package
    sys.path.insert(0, _HERE)
from prepare_data import (  # noqa: E402
    AMPLITUDE_CEILING as AMPLITUDE_DERIVATION_TARGET,
    AMPLITUDE_FORMULA_VERSION,
    RAF_UP_AXIS,
    derivation_id_hash,
    largest_one_significant_digit_at_most,
    one_significant_digit,
    CLIP_CEILING,
    LOADER_SAMPLE_SIZE,
    SILENCE_THRESHOLD_DB,
    SOURCE_SR,
    TARGET_SR,
)
from publish import (  # noqa: E402
    CANONICAL_MAPPINGA_PREPARE_PARAMS,
    SHA256_SHAPE,
    unpinned_identity_keys,
    MANIFEST_NAME as PUBLISH_MANIFEST_NAME,
    PublishTransaction,
    sha256_file,
    verify_publication,
)
from prepare_data import _write_json, load_room_index, group_captures  # noqa: E402
from raf_common import farthest_point_selection  # noqa: E402
from readback_audit import load_passing_record, record_provenance  # noqa: E402
from mappingA_common import (  # noqa: E402
    CANONICAL_ARRAY_SIZE as ARRAY_SIZE,
    MATCH_ALGORITHM_VERSION,
    MATCH_AMBIGUITY_MARGIN,
    PLACEMENT_CAP_M,
    cluster_placements,
    match_mics,
    MATCH_MAX_M,
    MATCH_P95_M,
    select_target,
    source_xyz_key,
    stable_item_context,
)
from raf_common import dbfs  # noqa: E402


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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


def _resampled_peaks_and_finite(path, orig_sr, target_sr,
                                sample_size=LOADER_SAMPLE_SIZE):
    """Peak of the whole resampled signal AND of the crop the loader will keep.

    N3: the loader crops to ``sample_size`` BEFORE its silence test, so the crop
    peak -- not the full-waveform peak -- decides whether an item is substituted.
    A late-arriving RIR whose direct sound lands after the crop reads loud here and
    silent there, and the manifest would describe an item nobody evaluated.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != orig_sr:
        raise ValueError(f"{path}: expected {orig_sr} Hz, got {sr} Hz")
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {audio.shape[1]} channels")
    wave = audio[:, 0]
    finite = bool(np.isfinite(wave).all())
    if not finite or wave.size == 0:
        return 0.0, 0.0, finite, wave.size
    out = np.asarray(librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr),
                     dtype=np.float32)
    if not np.isfinite(out).all():
        return 0.0, 0.0, False, int(out.size)
    crop = out[:sample_size]
    crop_peak = float(np.abs(crop).max()) if crop.size else 0.0
    return float(np.abs(out).max()), crop_peak, True, int(out.size)


def largest_admissible_scalar(max_peak, clip_ceiling=CLIP_CEILING):
    """The largest scalar that keeps this union clip-free -- MEASURED, not applied."""
    if max_peak <= 0:
        return float("inf")
    return float(clip_ceiling / max_peak)


def measure_union(room_dir, capture_ids, orig_sr=SOURCE_SR, target_sr=TARGET_SR,
                  sample_size=LOADER_SAMPLE_SIZE):
    """Resample every union capture ONCE and record what the policy needs.

    Amendment 4 derives the scalar from the union's own peaks and then audits the
    union at that scalar; both read these measurements, so the corpus is resampled
    once rather than twice (~10k files per run).
    """
    measurements = {}
    for capture_id in capture_ids:
        path = os.path.join(room_dir, "data", capture_id, "rir.wav")
        if measurements is not None and capture_id in measurements:
            measured = measurements[capture_id]
            peak, crop_peak = measured["peak"], measured["crop_peak"]
            finite, n_samples = measured["finite"], measured["n_samples"]
        else:
            peak, crop_peak, finite, n_samples = _resampled_peaks_and_finite(
                path, orig_sr, target_sr, sample_size)
        measurements[capture_id] = {"peak": peak, "crop_peak": crop_peak,
                                    "finite": bool(finite),
                                    "n_samples": int(n_samples)}
    return measurements


def context_item_index(items):
    """{room: {context capture id: [item ids that draw it]}} (Amendment 4.1).

    A near-silent REFERENCE is a fact about the items that condition on it, so the
    disclosure names them rather than the capture alone.
    """
    index = {}
    for item in items:
        room = index.setdefault(item["room"], {})
        for entry in item["context"]:
            room.setdefault(entry["capture_id"], []).append(item["item_id"])
    return {room: {capture: sorted(set(ids)) for capture, ids in sorted(captures.items())}
            for room, captures in sorted(index.items())}


def enumerate_support_captures(items):
    """{room: (support ids, target ids)} -- the union split by ROLE.

    Mapping A has no train/test split, so the derivation's "support" set is the
    conditioning population: every capture that is drawn as a CONTEXT. That is the
    Mapping-A analogue of exp_19's trained supports -- the references the model is
    given -- while the targets are the waveforms being scored. Both sets are
    recorded, and the clamp term below covers ALL written files regardless of role.
    """
    supports, targets = {}, {}
    for item in items:
        room = item["room"]
        targets.setdefault(room, set()).add(item["target_capture_id"])
        for entry in item["context"]:
            supports.setdefault(room, set()).add(entry["capture_id"])
    rooms = sorted(set(supports) | set(targets))
    return {room: (sorted(supports.get(room, ())), sorted(targets.get(room, ())))
            for room in rooms}


def derive_union_scalar(measurements_by_room, supports_by_room,
                        ceiling=AMPLITUDE_DERIVATION_TARGET,
                        clip_ceiling=CLIP_CEILING):
    """ONE scalar for the Mapping-A corpus, derived over ITS union (Amendment 4).

    The registered formula, unchanged from exp_19 and evaluated on Mapping A's own
    population::

        scalar = min( one-significant-digit(ceiling / max SUPPORT peak),
                      largest one-significant-digit value keeping
                      max WRITTEN peak x scalar <= clip_ceiling )

    Mapping H's x3 was derived over a different population (its 21 selected groups)
    and says nothing about this union: two EmptyRoom union captures clip at x3,
    which is what the pre-authorised amplitude stop reported. The support term is
    the target statistic; the clamp is a fail-closed safety bound that can only
    LOWER the scalar, never raise it.

    Peaks come from ``measure_union``, so nothing is read twice.
    """
    if not measurements_by_room:
        raise ValueError("cannot derive a Mapping-A scalar with no rooms")

    qualified, support_peak, written_peak, n_written = [], 0.0, 0.0, 0
    per_room = {}
    for room in sorted(measurements_by_room):
        measurements = measurements_by_room[room]
        support_ids = sorted(supports_by_room.get(room) or ())
        if not support_ids:
            raise ValueError(f"{room}: no context captures to derive from")
        missing = [c for c in support_ids if c not in measurements]
        if missing:
            raise ValueError(
                f"{room}: {len(missing)} support captures were never measured "
                f"(e.g. {missing[:3]}); the derivation set must be inside the union")
        room_support = max(measurements[c]["peak"] for c in support_ids)
        room_written = max(m["peak"] for m in measurements.values())
        per_room[room] = {"max_support_peak": room_support,
                          "max_written_peak": room_written,
                          "n_supports": len(support_ids),
                          "n_written": len(measurements)}
        qualified.extend(f"{room}/{c}" for c in support_ids)
        support_peak = max(support_peak, room_support)
        written_peak = max(written_peak, room_written)
        n_written += len(measurements)

    if support_peak <= 0.0:
        raise ValueError("every context capture is silent; no scalar can be derived")
    if written_peak <= 0.0:
        raise ValueError("every union capture is silent")

    support_term = one_significant_digit(ceiling / support_peak)
    clamp_term = largest_one_significant_digit_at_most(clip_ceiling / written_peak)
    scalar = min(support_term, clamp_term)
    qualified = sorted(qualified)
    return {
        "formula": ("min( one significant digit of (ceiling / max CONTEXT peak), "
                    "largest one-significant-digit value with max UNION peak x "
                    "scalar <= clip_ceiling ), measured on the resampled signal "
                    "over the whole Mapping-A union"),
        "formula_version": AMPLITUDE_FORMULA_VERSION,
        "population": "mappingA_union",
        "ceiling": float(ceiling),
        "clip_ceiling": float(clip_ceiling),
        "max_support_peak": support_peak,
        "max_written_peak": written_peak,
        "support_term": support_term,
        "clamp_term": clamp_term,
        "clamp_engaged": bool(clamp_term < support_term),
        "binding_term": "clip_clamp" if clamp_term < support_term else "support_ceiling",
        "scalar": scalar,
        "max_admissible_scalar": largest_admissible_scalar(written_peak, clip_ceiling),
        "rooms": sorted(measurements_by_room),
        "per_room": per_room,
        "derivation_ids": qualified,
        "derivation_id_sha256": derivation_id_hash(qualified),
        "n_supports": len(qualified),
        "n_written": n_written,
    }


def audit_amplitude_union(room_dir, capture_ids, scalar, orig_sr=SOURCE_SR,
                          target_sr=TARGET_SR, clip_ceiling=CLIP_CEILING,
                          silence_db=SILENCE_THRESHOLD_DB,
                          sample_size=LOADER_SAMPLE_SIZE, measurements=None,
                          target_ids=None, context_items=None):
    """Amplitude audit over the exact Mapping-A union, BEFORE anything is written.

    Three registered conditions, measured on the RESAMPLED signal (what would
    actually be published):

    * ``full peak * scalar <= clip_ceiling`` -- the loader clamps to [-1, 1], so a
      clipped file is a distorted target scored against an undistorted reference.
      Measured on the FULL signal: a sample clipped anywhere is written clipped;
    * ``dBFS(crop peak * scalar) >= silence_db`` -- measured on the first
      ``sample_size`` samples, because that is what the loader tests. Below its gate
      the item is silently substituted, so the manifest would describe an item
      nobody evaluated (N3: auditing the full peak here passed late-arriving RIRs
      that the runtime would then drop);
    * finite and non-empty.

    Amendment 4.1 -- the silence gate is ROLE-AWARE. ``SampleDataset.__getitem__``
    tests ``is_silence`` on the cropped TARGET and substitutes the item; RAF_A_md
    loads context audio with no amplitude test at all, so a quiet reference is
    conditioning the model saw, not an item nobody evaluated. Measured basis
    (mappingA_amplitude_window.json): no scalar satisfies both ends over the whole
    union -- lifting every sub-threshold capture needs x8.74 while the clip cap is
    x2.0401 -- and all 21 sub-threshold captures are context-only. So the gate is
    fatal for captures that appear in any item's TARGET role and RECORDS the rest,
    naming the items that draw them. The clip gate is unchanged over ALL files: a
    clipped context is a distorted input regardless of who reads it.

    ``target_ids=None`` means "roles unknown", and then EVERY capture is treated as
    a target -- the strict pre-Amendment behaviour, so an uninformed caller cannot
    silently get the weaker gate.

    Returns the measured report when clean. Otherwise raises
    ``AmplitudePolicyError`` carrying the same report: every violation, worst
    first, plus the largest scalar this union would admit -- offered as evidence for
    the decision, never applied.
    """
    peaks, crop_peaks, violations = [], [], []
    near_silent = []
    context_items = context_items or {}
    for capture_id in capture_ids:
        # unknown roles => strict: everything is audited as a target
        is_target = target_ids is None or capture_id in target_ids
        path = os.path.join(room_dir, "data", capture_id, "rir.wav")
        if measurements is not None and capture_id in measurements:
            measured = measurements[capture_id]
            peak, crop_peak = measured["peak"], measured["crop_peak"]
            finite, n_samples = measured["finite"], measured["n_samples"]
        else:
            peak, crop_peak, finite, n_samples = _resampled_peaks_and_finite(
                path, orig_sr, target_sr, sample_size)
        scaled, scaled_crop = peak * scalar, crop_peak * scalar
        record = {
            "capture_id": capture_id,
            "raw_peak": peak,
            "raw_peak_crop": crop_peak,
            "scaled_peak": scaled,
            "scaled_peak_crop": scaled_crop,
            "scaled_dbfs": dbfs(scaled),
            "scaled_dbfs_crop": dbfs(scaled_crop),
            "n_samples": n_samples,
        }
        record["role"] = "target" if is_target else "context"
        if not finite or n_samples == 0:
            # unreadable is not a level fact: fatal whatever the role
            violations.append(dict(record, kind="non_finite"))
            continue
        peaks.append(peak)
        crop_peaks.append(crop_peak)
        silence_kind = None
        if peak <= 0.0:
            silence_kind = "silent_source"
        elif scaled > clip_ceiling:
            # the clip gate is role-independent: a clipped context is a distorted
            # input regardless of who reads it
            violations.append(dict(record, kind="clipping"))
        elif crop_peak <= 0.0:
            silence_kind = "silent_crop"
        elif dbfs(scaled_crop) < silence_db:
            # the LOADER's test: the first sample_size samples, after scaling
            silence_kind = "below_threshold_crop"

        if silence_kind is not None:
            if is_target:
                violations.append(dict(record, kind=silence_kind))
            else:
                near_silent.append(dict(
                    record, kind=silence_kind,
                    item_ids=list(context_items.get(capture_id, ()))))

    max_peak = max(peaks) if peaks else 0.0
    violations.sort(key=lambda v: (-v["scaled_peak"], v["capture_id"]))
    report = {
        "scalar": float(scalar),
        "clip_ceiling": float(clip_ceiling),
        "silence_threshold_db": float(silence_db),
        "source_sample_rate": int(orig_sr),
        "sample_rate": int(target_sr),
        "loader_sample_size": int(sample_size),
        "n_captures": len(capture_ids),
        "max_raw_peak": max_peak,
        "max_scaled_peak": max_peak * scalar,
        "min_raw_peak": min(peaks) if peaks else 0.0,
        "min_scaled_dbfs": dbfs(min(peaks) * scalar) if peaks else dbfs(0.0),
        # the crop statistics the loader's silence gate actually reads
        "min_raw_peak_crop": min(crop_peaks) if crop_peaks else 0.0,
        "min_scaled_dbfs_crop": (dbfs(min(crop_peaks) * scalar) if crop_peaks
                                 else dbfs(0.0)),
        "n_clipping": sum(1 for v in violations if v["kind"] == "clipping"),
        "n_below_threshold": sum(1 for v in violations
                                 if v["kind"] in ("below_threshold_crop",
                                                  "silent_crop",
                                                  "silent_source")),
        # Amendment 4.1: context-only sub-threshold captures, RECORDED with the
        # items that draw them. Not violations -- the loader never substitutes on
        # them -- but a disclosed property of the published corpus.
        "role_aware_silence_gate": target_ids is not None,
        "n_target_role_captures": (len(capture_ids) if target_ids is None
                                   else sum(1 for c in capture_ids if c in target_ids)),
        "n_near_silent_references": len(near_silent),
        "near_silent_references": sorted(near_silent,
                                         key=lambda r: (r["scaled_dbfs_crop"],
                                                        r["capture_id"])),
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
            f"clipping, {report['n_below_threshold']} TARGET-role captures below "
            f"{silence_db} dBFS over the loader's {sample_size}-sample crop, "
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


class CorrespondenceRecordError(RuntimeError):
    """The committed correspondence audit that authorises a Mapping-A selection."""


def load_correspondence_record(path, rooms, canonical, raf_root=None):
    """Read the committed correspondence record ONCE and digest what was read (N5).

    The record is the evidence that these placements can be matched at all, so the
    publication's ``correspondence_sha256`` must be the digest of the FULL committed
    file -- not, as before, a hash of a summary the run computed about itself, which
    attested nothing a reader could check against the repository. Reading and
    hashing the same bytes closes the window where the file changes between the two.

    The record must also describe THIS corpus and THIS algorithm: version,
    tolerances, rooms, and -- canonically -- the RAF root it audited.
    """
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise CorrespondenceRecordError(f"{path}: not valid JSON ({e})")

    registered = CANONICAL_MAPPINGA_PREPARE_PARAMS["correspondence_sha256"]
    taint = []
    if canonical and digest != registered:
        raise CorrespondenceRecordError(
            f"{path} hashes to {digest}, not the registered correspondence record "
            f"{registered}. A canonical publication is authorised by the committed "
            "audit, not by whatever record a run was pointed at.")
    if digest != registered:
        taint.append(f"correspondence record {digest} is not the registered "
                     f"{registered}")

    if record.get("algorithm_version") != MATCH_ALGORITHM_VERSION:
        raise CorrespondenceRecordError(
            f"{path} audits algorithm {record.get('algorithm_version')!r}, this run "
            f"uses {MATCH_ALGORITHM_VERSION!r}")
    tolerances = record.get("tolerances") or {}
    # the readback driver's own key names -- checked against the committed record
    # in the tests, because a tolerance this code cannot find is a tolerance it
    # cannot verify
    expected_tolerances = {"placement_cap_m": PLACEMENT_CAP_M,
                           "match_p95_m": MATCH_P95_M,
                           "match_max_m": MATCH_MAX_M,
                           "match_ambiguity_margin": MATCH_AMBIGUITY_MARGIN}
    wrong = {k: (tolerances.get(k), v) for k, v in expected_tolerances.items()
             if tolerances.get(k) != v}
    if wrong:
        raise CorrespondenceRecordError(
            f"{path} was audited under different tolerances: "
            + "; ".join(f"{k}={got!r} != {want!r}" for k, (got, want) in
                        sorted(wrong.items())))
    missing = [room for room in rooms if room not in (record.get("rooms") or {})]
    if missing:
        raise CorrespondenceRecordError(f"{path} audits no {missing}")
    if canonical and raf_root and (os.path.realpath(record.get("raf_root") or "")
                                   != os.path.realpath(raf_root)):
        raise CorrespondenceRecordError(
            f"{path} audited {record.get('raf_root')!r}, this run reads {raf_root!r}")

    return record, {"path": os.path.abspath(path), "sha256": digest,
                    "canonical": canonical and not taint, "taint": taint,
                    "algorithm_version": record.get("algorithm_version"),
                    "created_utc": record.get("created_utc")}


def cross_check_correspondence(record, room, survey):
    """The audited room and the surveyed room must be the same room (N5).

    A digest proves which file was read; this proves the file describes what the
    run just measured. A corpus that changed under the audit shows up here as a
    different eligible-placement set or a different pass/fail count.
    """
    audited = (record.get("rooms") or {})[room]
    surveyed_eligible = sorted(p["placement_id"] for p in survey["placements"]
                               if p["eligible"])
    problems = []
    if sorted(audited.get("eligible_placement_ids") or []) != surveyed_eligible:
        problems.append(
            f"eligible placements {sorted(audited.get('eligible_placement_ids') or [])}"
            f" != surveyed {surveyed_eligible}")
    for key in ("n_groups_passing", "n_groups_failing", "n_placements"):
        if key in audited and audited[key] != survey[key]:
            problems.append(f"{key} {audited[key]} != surveyed {survey[key]}")
    if problems:
        raise CorrespondenceRecordError(
            f"{room}: the correspondence record does not describe the corpus this "
            "run surveyed -- " + "; ".join(problems))
    return {"eligible_placement_ids": surveyed_eligible,
            "n_groups_passing": survey["n_groups_passing"],
            "n_groups_failing": survey["n_groups_failing"]}


def audio_fingerprint(path):
    """Content identity for a published WAV: rate, length and exact samples.

    NOT the file digest. libsndfile stamps a UNIX timestamp into the ``PEAK`` chunk
    of every float WAV, so two publications holding bit-identical audio have
    different file bytes unless they were written in the same second -- measured,
    at offset 60. The claim worth making about shared audio is that the samples are
    the same, and that is what this hashes.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"{path}: expected mono, got {audio.shape[1]} channels")
    wave = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    digest = hashlib.sha256(wave.tobytes()).hexdigest()
    return {"audio_sha256": digest, "sample_rate": int(sr), "n_samples": int(wave.size)}


class MappingHProvenanceError(RuntimeError):
    """The Mapping-H publication whose topology and overlap Mapping A records."""


def scale_disclosure(mappingA_scalar, mappingH_scalar=None):
    """The registered cross-mapping scale disclosure (Amendment 4).

    Mapping A publishes its complete union at its own derived scalar; Mapping H is
    at x3.0. Rather than reconcile the two -- which would mean either clipping two
    EmptyRoom captures or dropping them, and items are never dropped -- the
    difference is DISCLOSED wherever the corpus or its results are described.
    """
    return {
        "mappingA_amplitude_scalar": float(mappingA_scalar),
        "mappingH_amplitude_scalar": (None if mappingH_scalar is None
                                      else float(mappingH_scalar)),
        "audio_is_shared": False,
        "note": (
            f"Mapping-A audio is written at x{float(mappingA_scalar)} over its "
            "COMPLETE union; the Mapping-H publication it overlaps is at "
            f"x{mappingH_scalar if mappingH_scalar is not None else 3.0}. The two "
            "corpora hold the same captures at different levels, so no file is "
            "shared and cross-mapping ABSOLUTE level-dependent comparisons "
            "(multi-resolution L1, Env) are unlicensed. Within-Mapping-A contrasts "
            "are unaffected, and T60/C50/EDT are level-independent."),
    }


def locate_mappingH(runtime_dir, rooms, require_canonical=False):
    """Find, require and VERIFY the Mapping-H publication (N4).

    "Shared with Mapping H" is a provenance claim, and a claim needs a generation
    to be a claim about anything: the file on disk could be a leftover from an
    interrupted publish, a stale generation, or a tree nobody attested. This
    resolves the publication the claim will name -- pointer, prepare marker,
    generation, and the manifest file set that generation actually covers -- and
    refuses everything else. The r1 CLI never called this at all: ``write_union``
    took the parameters and no caller passed them, so every production run silently
    recorded ``shared_with_mappingH: null`` for files it did share.
    """
    if not runtime_dir:
        raise MappingHProvenanceError("no Mapping-H runtime directory was given")
    pointer_path = os.path.join(runtime_dir, PUBLICATION_POINTER)
    if not os.path.isfile(pointer_path):
        raise MappingHProvenanceError(
            f"{pointer_path} does not exist: {runtime_dir} is not a published "
            "Mapping-H runtime tree")
    with open(pointer_path) as f:
        pointer = json.load(f)
    flavor = pointer.get("flavor", "mappingH")
    if flavor != "mappingH":
        raise MappingHProvenanceError(
            f"{pointer_path} declares flavor {flavor!r}: Mapping-A audio cannot take "
            "its provenance from a Mapping-A publication")
    split_dir = pointer.get("split_dir")
    if not split_dir:
        raise MappingHProvenanceError(f"{pointer_path} names no split_dir")

    report = verify_publication(
        split_dir, kind="prepare",
        expected_roots=[os.path.abspath(runtime_dir), os.path.abspath(split_dir)])
    if not report["published"]:
        raise MappingHProvenanceError(
            f"the Mapping-H publication at {split_dir} is not valid: "
            f"{report['reason']}")
    generation = report["generation"]
    if not generation:
        raise MappingHProvenanceError(
            f"the Mapping-H prepare marker at {split_dir} carries no generation, so "
            "a shared-audio claim could not name the publication it came from")
    # The amplitude identity lives in the MARKER (exp_19 writes the derived scalar
    # there, not in the pointer), which is also what verify_publication authenticated.
    marker = report.get("marker") or {}
    if require_canonical and (marker.get("canonical") is not True
                              or marker.get("taint")):
        raise MappingHProvenanceError(
            f"the Mapping-H publication at {split_dir} is not canonical "
            f"(canonical={marker.get('canonical')!r}, taint={marker.get('taint')!r}): "
            "a canonical Mapping-A publication cannot claim shared audio with a "
            "tainted one")

    manifest_path = os.path.join(runtime_dir, PUBLISH_MANIFEST_NAME)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("generation") != generation:
        raise MappingHProvenanceError(
            f"{manifest_path} attests generation {manifest.get('generation')}, but "
            f"the marker attests {generation}")

    parameters = marker.get("parameters") or {}
    if "amplitude_scalar" not in parameters:
        raise MappingHProvenanceError(
            f"the Mapping-H marker at {split_dir} records no amplitude_scalar, so it "
            "cannot be known whether its audio was normalised like Mapping A's")

    covered = {}
    for name in manifest["files"]:
        room, _, relative = name.partition("/")
        if relative:
            covered.setdefault(room, set()).add(relative)
    return {
        "runtime_dir": os.path.abspath(runtime_dir),
        "split_dir": os.path.abspath(split_dir),
        # Q1: every room THIS publication holds, whatever subset the caller asked
        # about -- these are the trees a Mapping-A output root must stay out of.
        "pointer_rooms": [str(room) for room in (pointer.get("rooms") or [])],
        "generation": generation,
        "amplitude_scalar": float(parameters["amplitude_scalar"]),
        "canonical": bool(marker.get("canonical")),
        "taint": list(marker.get("taint") or []),
        "rooms": {room: {"dir": os.path.join(os.path.abspath(runtime_dir), room),
                         "files": covered.get(room, set())}
                  for room in rooms},
        "n_files": manifest["n_files"],
    }


MAPPINGA_RUNTIME_SUBDIR = "mappingA"


def _is_within(child, parent):
    """True when ``child`` is ``parent`` or sits underneath it (no symlink games)."""
    child = os.path.realpath(child)
    parent = os.path.realpath(parent)
    return child == parent or child.startswith(parent + os.sep)


def resolve_output_dir(output_dir, mappingH_dir, protected_rooms):
    """Where Mapping A publishes its runtime tree, and where it may NOT (P1).

    Mapping A writes its own ``raf_publication.json`` and its own root manifest,
    and StagedPublish renames the destination root's existing manifest aside before
    swapping. Pointed at Mapping H's runtime root, it therefore overwrites H's
    pointer and invalidates H's prepare attestation -- and the listener render then
    collides with H's per-room depth roots. That is a wrong-flag operator error, so
    it is refused HERE, before a single file is read or written, rather than
    discovered afterwards in a broken publication.

    With a Mapping-H tree in hand the default is ``<H>/mappingA``: a proper child,
    disjoint from every root H attests, and self-evidently the same corpus. An
    explicit ``--output-dir`` must satisfy the same relation.

    Q1: ``protected_rooms`` is EVERY room the Mapping-H publication holds, taken
    from its pointer -- not the rooms this run happens to prepare. An
    EmptyRoom-only Mapping-A run pointed at ``<H>/FurnishedRoom`` was accepted
    before, and the transaction would then have replaced a Mapping-H room tree the
    run never mentions.

    R1: the DERIVED path is checked exactly like an explicit one. It used to
    return before the gates on the grounds that it is safe by construction -- but
    it is safe only relative to a room list, and a publication holding a room
    literally named ``mappingA`` puts the derived root on top of a protected tree.
    "Safe by construction" that is never checked is an assumption, and this
    function exists to refuse assumptions.
    """
    if not mappingH_dir:
        if not output_dir:
            raise ValueError(
                "--output-dir is required when no --mappingH-dir is given: without "
                "the Mapping-H tree there is nothing to derive the Mapping-A runtime "
                "root from.")
        return os.path.abspath(output_dir)

    h_root = os.path.abspath(mappingH_dir)
    # BEFORE any root is resolved: with no protected rooms there is nothing to
    # check a root against, derived or not.
    if not protected_rooms:
        raise ValueError(
            "refusing to resolve a Mapping-A runtime root without the Mapping-H "
            "publication's room list: the rooms it protects come from ITS pointer, "
            "and an empty list would protect nothing.")

    derived = not output_dir
    out = (os.path.join(h_root, MAPPINGA_RUNTIME_SUBDIR) if derived
           else os.path.abspath(output_dir))
    if _is_within(h_root, out):
        # covers equality and any ancestor of the Mapping-H tree
        raise ValueError(
            f"refusing to publish Mapping A at {out}: it is the Mapping-H runtime "
            f"tree {h_root} or contains it. Mapping A writes its own "
            f"{PUBLICATION_POINTER} and root manifest there, which would overwrite "
            "Mapping H's pointer and invalidate its publication. Omit --output-dir "
            f"to use {os.path.join(h_root, MAPPINGA_RUNTIME_SUBDIR)}.")
    if not _is_within(out, h_root):
        raise ValueError(
            f"refusing to publish Mapping A at {out}: it is outside the Mapping-H "
            f"runtime tree {h_root}. The two corpora share audio by provenance, so "
            "the Mapping-A tree is a child of the publication it cites.")
    for room in protected_rooms:
        room_root = os.path.join(h_root, room)
        if _is_within(out, room_root):
            raise ValueError(
                f"refusing to publish Mapping A at "
                f"{out}{' (the DERIVED default)' if derived else ''}: it is inside "
                f"Mapping H's {room} root, whose audio and depth_images directories "
                "H attests. The protected rooms are the publication's own "
                f"({sorted(protected_rooms)}), not this run's subset."
                + (f" Pass --output-dir explicitly: this publication holds a room "
                   f"named {MAPPINGA_RUNTIME_SUBDIR!r}, so the derived default "
                   "lands on top of it." if derived else ""))
    return out


def resolve_mappingH(mappingH_dir, rooms, canonical):
    """The CLI's Mapping-H requirement, as one testable decision (N4).

    Canonical runs MUST name the publication they share audio with; a
    non-canonical run may decline, and then says so in its taint rather than
    quietly publishing every shared capture as new.
    """
    if mappingH_dir:
        publication = locate_mappingH(mappingH_dir, rooms,
                                      require_canonical=canonical)
        # Amendment 4: agreement is no longer required. Mapping A derives its own
        # scalar over its own union (Mapping H's x3 clips two of these captures),
        # so a difference is expected and is DISCLOSED rather than refused.
        return publication, []
    if canonical:
        raise MappingHProvenanceError(
            "a canonical Mapping-A publication must name the Mapping-H publication "
            "it shares audio with: pass --mappingH-dir <exp_19 runtime tree>. "
            "Without it every shared capture is published as new and unverified, and "
            "the two corpora could silently diverge on the same file.")
    return None, ["no Mapping-H provenance: --mappingH-dir was not given, so shared "
                  "audio is recorded as new and unverified"]


def write_union(room_dir, out_room_dir, capture_ids, scalar, orig_sr=SOURCE_SR,
                target_sr=TARGET_SR, clip_ceiling=CLIP_CEILING,
                mappingH_room_dir=None, mappingH_generation=None,
                mappingH_files=None, mappingH_scalar=None,
                sample_size=LOADER_SAMPLE_SIZE, silence_db=SILENCE_THRESHOLD_DB,
                target_ids=None):
    """Resample, scale and publish the Mapping-A union at ITS OWN scalar.

    Amendment 4: nothing is reused from Mapping H. Every union member is written
    fresh at the Mapping-A scalar, because the two publications are at different
    levels by decision -- Mapping H's x3 clips two EmptyRoom union captures, and
    items are never dropped. What is recorded instead is the OVERLAP: which
    captures the Mapping-H publication also holds, under which generation, and
    both scalars, so a reader can see exactly what the two corpora share (the
    captures) and what they do not (the levels).

    Every written file is read back and compared, as in exp_19, and both amplitude
    conditions are RE-CHECKED here on the bytes about to be published (N3): the
    audit runs over a union computed from the manifest, so a write that reached this
    point with a different file, scalar or sample rate must still not be able to
    publish a clipped or loader-silent target.

    Amendment 4.1: the silence recheck is role-aware for the same reason the audit
    is -- only a TARGET can be substituted by the loader. ``target_ids=None`` keeps
    the strict behaviour, so an uninformed caller cannot get the weaker check.
    """
    if mappingH_room_dir and (mappingH_generation is None or mappingH_files is None):
        raise ValueError(
            "recording the Mapping-H overlap needs the generation and the file set "
            "that generation's manifest covers; counting whatever happens to sit in "
            "the tree would attest leftovers or a stale publish (N4)")

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

        crop_peak = float(np.abs(out[:sample_size]).max()) if out.size else 0.0
        is_target = target_ids is None or capture_id in target_ids
        if is_target and dbfs(crop_peak) < silence_db:
            raise ValueError(
                f"{src}: the first {sample_size} samples peak at {crop_peak:.6g} "
                f"({dbfs(crop_peak):.1f} dBFS) after x{scalar}, below the loader's "
                f"{silence_db} dBFS gate. It is drawn as a TARGET, so the runtime "
                "would substitute the item and the manifest would name an item that "
                "was never evaluated.")

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
        # content identity, independent of the PEAK-chunk timestamp in the header
        audio_digest = audio_fingerprint(out_path)["audio_sha256"]

        # Amendment 4: OVERLAP, not reuse. The capture exists in both corpora at
        # different levels, so its bytes are neither copied nor compared -- only
        # the fact of the overlap is recorded, against the generation that covers it.
        shared = None
        if mappingH_room_dir:
            h_relative = f"{RIR_FOLDER}/{capture_id}.wav"
            h_path = os.path.join(mappingH_room_dir, RIR_FOLDER, f"{capture_id}.wav")
            if os.path.isfile(h_path):
                if h_relative not in mappingH_files:
                    raise ValueError(
                        f"{h_path} exists but is NOT covered by Mapping-H generation "
                        f"{mappingH_generation}'s manifest: it is a leftover or a "
                        "stale file, and recording an overlap with it would attest "
                        "audio no publication stands behind")
                shared = {"path": os.path.abspath(h_path),
                          "generation": mappingH_generation,
                          "mappingH_amplitude_scalar": (
                              None if mappingH_scalar is None else float(mappingH_scalar)),
                          "mappingA_amplitude_scalar": float(scalar),
                          "same_capture": True,
                          "same_audio": False,
                          "reason": "written at different amplitude scalars "
                                    "(Amendment 4); no file is shared"}
                n_shared += 1

        files[capture_id] = {
            "sha256": digest,
            "audio_sha256": audio_digest,
            "peak": peak,
            "peak_crop": crop_peak,
            "dbfs": dbfs(peak),
            "dbfs_crop": dbfs(crop_peak),
            "n_samples": int(out.shape[0]),
            "roundtrip_max_abs_error": roundtrip,
            "overlaps_mappingH": shared,
        }

    return {
        "n_files": len(files),
        "n_overlapping_mappingH": n_shared,
        "n_outside_mappingH": len(files) - n_shared,
        "scale_disclosure": scale_disclosure(scalar, mappingH_scalar),
        "scalar": float(scalar),
        "loader_sample_size": int(sample_size),
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
# N6: the REGISTERED count, cross-checked against how it is composed. A canonical
# run validates against this, never against the number of items it happened to
# build -- that comparison passes by construction.
CANONICAL_N_ITEMS = CANONICAL_MAPPINGA_PREPARE_PARAMS["n_items"]
assert CANONICAL_N_ITEMS == CANONICAL_N_PLACEMENTS * CANONICAL_ARRAY_SIZE * 2


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
        # N8: the tracked height is a RAW RAF quantity (metres above the y=0 ground
        # plane), so it is read from the RAW row, not from a pipeline coordinate
        # that happens to hold it under the current gauge. Reading rx_target[2] was
        # correct only because RAF_TO_PIPELINE maps RAF y to pipeline z; a gauge
        # change would have silently turned it into a different axis.
        rx_target_raw = np.asarray(target["rx_xyz"], dtype=np.float64)[target_row]
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
                # N6: which physical row this group's slot resolved to, and the
                # correspondence evidence for the group it came from -- the
                # manifest must carry what its "same microphone" claim rests on
                "rx_row": int(row),
                "match": dict(match[key]),
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
            "rx_target_row": int(target_row),
            "rx_target_height_raf_m": float(rx_target_raw[RAF_UP_AXIS]),
            "depth_file": f"{room}_{placement_id}_slot{slot:02d}_depth_image.npy",
            "context": context,
            "match": dict(match[target_key]),
            "context_digest": drawn["digest"],
            "context_pool_size": drawn["pool_size"],
        })
    return items


def _is_xyz(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in value))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value):
    """P2: a distance is finite. NaN in particular compares False against every
    threshold, so an unmeasured match would pass every gate it was tested against."""
    return _is_number(value) and math.isfinite(float(value))


def _is_margin(value):
    """The ambiguity margin is a RATIO, and +inf is its meaningful extreme: an
    exact hit whose next-nearest mic is strictly further away (every placement
    medoid against itself). NaN is still refused -- that is an absent measurement,
    not a decisive one."""
    return _is_number(value) and not math.isnan(float(value)) and float(value) >= 0.0


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _is_index(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# N6: the FULL schema, checked without defaults. ``match.get("p95_m", 0.0)`` used
# to pass an item that carried no correspondence evidence at all -- the absent
# field read as a perfect score.
ITEM_SCHEMA = {
    "item_id": lambda v: isinstance(v, str) and v,
    "room": lambda v: isinstance(v, str) and v,
    "placement_id": lambda v: isinstance(v, str) and v,
    "mic_slot": _is_index,
    "target_capture_id": lambda v: isinstance(v, str) and v,
    "target_group_key": lambda v: isinstance(v, str) and v,
    "target_xyz_key": lambda v: isinstance(v, str) and v,
    "tx_p": _is_xyz,
    "rx_target_p": _is_xyz,
    "rx_target_row": _is_index,
    "rx_target_height_raf_m": _is_number,
    "depth_file": lambda v: isinstance(v, str) and v.endswith(".npy"),
    "context": lambda v: isinstance(v, list),
    "match": lambda v: isinstance(v, dict),
}
CONTEXT_SCHEMA = {
    "capture_id": lambda v: isinstance(v, str) and v,
    "group_key": lambda v: isinstance(v, str) and v,
    "xyz_key": lambda v: isinstance(v, str) and v,
    "tx_p": _is_xyz,
    "rx_p": _is_xyz,
    "rx_row": _is_index,
    "rx_displacement_m": _is_finite_number,
    "match": lambda v: isinstance(v, dict),
}
# P2: the FULL correspondence evidence, all of it finite. p95/max/margin summarise
# 36 slots into three numbers; the digest names the assignment those numbers came
# from and the residual says whether one rigid motion explains it.
MATCH_SCHEMA = {"p95_m": _is_finite_number, "max_m": _is_finite_number,
                "min_ambiguity_margin": _is_margin,
                "evidence_sha256": _is_sha256,
                "rigid_residual_rms_m": _is_finite_number}


def _attest_row(add, item, room_rows, group_key, slot, recorded_row, who):
    """The recorded row must BE the row the correspondence assigned that slot.

    ``rx_row`` inside 0..35 says nothing: the reviewer's probe put slot 0 on row 23
    and every gate passed. "The same microphone" is the assignment's claim, so it
    is checked against the assignment.
    """
    if not room_rows or group_key not in room_rows:
        add("unattested_assignment", item,
            f"{who}: no authoritative assignment for group {group_key} in room "
            f"{item['room']}")
        return
    rows = room_rows[group_key]
    if slot >= len(rows):
        add("unattested_assignment", item,
            f"{who}: the assignment for group {group_key} covers {len(rows)} slots, "
            f"not slot {slot}")
        return
    if int(rows[slot]) != int(recorded_row):
        add("wrong_row", item,
            f"{who} records row {recorded_row} for mic slot {slot}, but the "
            f"correspondence assigned row {rows[slot]} -- the item is conditioned on "
            "a different microphone than it claims")


def _schema_problems(payload, schema, where):
    problems = []
    for key, ok in sorted(schema.items()):
        if key not in payload:
            problems.append(f"{where} is missing {key}")
        elif not ok(payload[key]):
            problems.append(f"{where}.{key}={payload[key]!r} has the wrong shape")
    return problems


def _match_problems(match, where):
    problems = _schema_problems(match, MATCH_SCHEMA, where)
    if problems:
        return problems
    if match["p95_m"] > MATCH_P95_M:
        problems.append(f"{where} p95 {match['p95_m']:.4f} m > {MATCH_P95_M} m")
    if match["max_m"] > MATCH_MAX_M:
        problems.append(f"{where} max {match['max_m']:.4f} m > {MATCH_MAX_M} m")
    if match["min_ambiguity_margin"] < MATCH_AMBIGUITY_MARGIN:
        problems.append(f"{where} margin {match['min_ambiguity_margin']:.2f} < "
                        f"{MATCH_AMBIGUITY_MARGIN}")
    return problems


def validate_manifest(manifest, expected_items=CANONICAL_N_ITEMS, k=CANONICAL_K,
                      max_displacement_m=MATCH_MAX_M, array_size=CANONICAL_ARRAY_SIZE,
                      assignments=None, allow_unattested=False):
    """Static validator over the whole manifest (M5), run before publication.

    Every condition here is a way the row could stop measuring what it claims:
    a repeated target (the same problem scored twice), a context holding the
    answer, a context source standing where the target stands (the "unseen source
    position" claim), or a context recorded at a different microphone (the
    "same listener" claim). All violations are collected before raising.
    """
    # P2: the AUTHORITATIVE per-slot correspondence, {room: {group_key: [row per
    # slot]}}. Without it the validator can only check that a row is inside the
    # array -- which is why an item for slot 0 naming row 23 passed. A caller with
    # no assignment must say so, rather than getting the weaker check silently.
    if assignments is None and not allow_unattested:
        raise ValueError(
            "validate_manifest needs the authoritative assignments to attest that "
            "each recorded row IS the row the Hungarian match gave that group's mic "
            "slot; pass assignments={room: {group_key: [row per slot]}}, or "
            "allow_unattested=True to check everything else and say so.")

    items = manifest["items"]
    violations = []

    def add(kind, item, detail):
        violations.append({"kind": kind, "item_id": item.get("item_id"),
                           "detail": detail})

    seen_items, seen_targets = {}, {}
    for position, item in enumerate(items):
        schema = _schema_problems(item, ITEM_SCHEMA, f"item[{position}]")
        if schema:
            for problem in schema:
                add("schema", item, problem)
            continue
        item_id = item["item_id"]
        if item_id in seen_items:
            add("duplicate_item", item, f"item id {item_id} appears twice")
        seen_items[item_id] = True

        target = item["target_capture_id"]
        if target in seen_targets:
            add("duplicate_target", item,
                f"target capture {target} is also item {seen_targets[target]}")
        seen_targets[target] = item_id

        # N6: the id, the slot and the depth file must name the same slot -- the
        # metadata hook resolves the panorama by that name.
        slot = item["mic_slot"]
        if slot >= array_size:
            add("mic_slot", item, f"slot {slot} is outside the {array_size}-mic array")
        if not item_id.endswith(f"slot{slot:02d}"):
            add("mic_slot", item, f"item id {item_id} does not name slot {slot}")
        if f"slot{slot:02d}_depth_image.npy" not in item["depth_file"]:
            add("mic_slot", item,
                f"depth file {item['depth_file']} does not name slot {slot}")
        if item["rx_target_row"] >= array_size:
            add("mic_slot", item,
                f"target row {item['rx_target_row']} is outside the array")

        room_rows = (assignments or {}).get(item["room"])
        if assignments is not None:
            _attest_row(add, item, room_rows, item["target_group_key"], slot,
                        item["rx_target_row"], "target")

        for problem in _match_problems(item["match"], "target correspondence"):
            kind = "ambiguous_match" if "margin" in problem else "failed_match"
            add("schema" if "missing" in problem or "shape" in problem else kind,
                item, problem)

        context = item["context"]
        rx_target = np.asarray(item["rx_target_p"], dtype=np.float64)
        tx_target = np.asarray(item["tx_p"], dtype=np.float64)
        capture_ids = [c["capture_id"] for c in context]
        if len(context) != k:
            add("context_size", item, f"{len(context)} context captures, expected {k}")
        if len(set(capture_ids)) != len(capture_ids):
            add("context_not_distinct", item,
                f"context captures are not distinct: {sorted(capture_ids)}")
        if target in capture_ids:
            add("target_in_context", item,
                f"target capture {target} appears in its own context")
        group_keys = [c.get("group_key") for c in context]
        if len(set(group_keys)) != len(group_keys):
            add("context_not_distinct", item,
                f"context groups are not distinct: {sorted(str(g) for g in group_keys)}")
        for position, entry in enumerate(context):
            where = f"context[{position}]"
            schema = _schema_problems(entry, CONTEXT_SCHEMA, where)
            if schema:
                for problem in schema:
                    add("schema", item, problem)
                continue
            if entry["group_key"] == item["target_group_key"]:
                add("target_in_context", item,
                    f"context {entry['capture_id']} is the target's own group")
            if entry["xyz_key"] == item["target_xyz_key"]:
                add("context_source_position", item,
                    f"context {entry['capture_id']} stands at the target source "
                    f"position {entry['xyz_key']}")
            # N6: RECOMPUTED from the recorded coordinates, so a doctored or stale
            # key cannot certify a source position or a displacement it is not.
            if np.allclose(np.asarray(entry["tx_p"], dtype=np.float64), tx_target,
                           rtol=0.0, atol=1e-9):
                add("context_source_position", item,
                    f"context {entry['capture_id']} stands at the target source "
                    f"coordinates {list(tx_target)} despite key {entry['xyz_key']}")
            recomputed = float(np.linalg.norm(
                np.asarray(entry["rx_p"], dtype=np.float64) - rx_target))
            if abs(recomputed - float(entry["rx_displacement_m"])) > 1e-9:
                add("displacement_mismatch", item,
                    f"context {entry['capture_id']} records "
                    f"{entry['rx_displacement_m']:.6f} m but its poses give "
                    f"{recomputed:.6f} m")
            if recomputed > max_displacement_m:
                add("mic_displacement", item,
                    f"context {entry['capture_id']} was recorded "
                    f"{recomputed:.4f} m from the target microphone "
                    f"(> {max_displacement_m} m)")
            if entry["rx_row"] >= array_size:
                add("mic_slot", item,
                    f"context {entry['capture_id']} row {entry['rx_row']} is outside "
                    f"the {array_size}-mic array")
            if assignments is not None:
                _attest_row(add, item, room_rows, entry["group_key"], slot,
                            entry["rx_row"], f"context {entry['capture_id']}")
            for problem in _match_problems(entry["match"],
                                           f"context {entry['capture_id']}"):
                kind = "ambiguous_match" if "margin" in problem else "failed_match"
                add("schema" if "missing" in problem or "shape" in problem else kind,
                    item, problem)

    report = {
        "n_items": len(items),
        "expected_items": int(expected_items),
        "assignments_attested": assignments is not None,
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


# --------------------------------------------------------------------------- #
# CLI: chain the tested components into one publication
# --------------------------------------------------------------------------- #
MIN_ELIGIBLE_GROUPS = 9          # per placement (plan section 2)
PUBLICATION_POINTER = "raf_publication.json"
METADATA_NAME = "mappingA_metadata.json"
MANIFEST_NAME = "mappingA_eval.json"
# Registered Mapping-A split root (Amendment 1, N2).
MAPPINGA_SPLIT_ROOT = "data/RAF_mappingA"


def canonical_digest(payload):
    """sha256 over a canonical JSON rendering (stable across machines)."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def survey_room(room_dir, room):
    """Placements, correspondence and eligibility for one room.

    Exactly the readback rung's computation, so the publication is cut from the
    same evidence the rung reported rather than from a second implementation.
    """
    index = load_room_index(room_dir)
    groups, group_report = group_captures(index, allow_nonuniform=True)
    sized = [g for g in groups if g["size"] == ARRAY_SIZE]
    excluded = [{"group_key": g["group_key"], "size": g["size"]}
                for g in groups if g["size"] != ARRAY_SIZE]

    clusters = cluster_placements(sized)
    by_key = {g["group_key"]: g for g in sized}
    placements, n_pass, n_fail, p95s = [], 0, 0, []
    for cluster in clusters:
        passing, assignment, match = [], {}, {}
        for key in cluster["member_keys"]:
            report = match_mics(cluster["template_rx"], by_key[key]["rx_xyz_p"])
            p95s.append(report["p95_m"])
            if report["passed"]:
                n_pass += 1
                passing.append(by_key[key])
                assignment[key] = report["assignment"]
                match[key] = {"p95_m": report["p95_m"], "max_m": report["max_m"],
                              "min_ambiguity_margin": report["min_ambiguity_margin"],
                              # N9: the per-slot assignment this group's items rest
                              # on, named by its own digest rather than by three
                              # summary statistics several assignments could share
                              "evidence_sha256": report["evidence_sha256"],
                              "rigid_residual_rms_m": report["rigid_residual_rms_m"]}
            else:
                n_fail += 1
        distinct = {source_xyz_key(g["tx_xyz"]) for g in passing}
        placements.append({
            "placement_id": cluster["placement_id"],
            "centroid_p": [float(v) for v in cluster["centroid_p"]],
            "n_groups": len(cluster["member_keys"]),
            "n_passing": len(passing),
            "n_passing_source_distinct": len(distinct),
            "eligible": len(distinct) >= MIN_ELIGIBLE_GROUPS,
            "passing": passing,
            "assignment": assignment,
            "match": match,
        })

    return {
        "room": room,
        "n_captures": len(index),
        "n_groups": len(groups),
        "excluded_wrong_size": excluded,
        "n_placements": len(clusters),
        "n_groups_passing": n_pass,
        "n_groups_failing": n_fail,
        "placement_p95_m": p95s,
        "placements": placements,
        "size_histogram": group_report["size_histogram"],
    }


def select_placements(survey, n_placements):
    """Farthest-point selection over ELIGIBLE placement centroids.

    FPS here and only here: placement COVERAGE is a spatial question (the plan
    wants the room sampled, not one corner of it), while the target pose within a
    placement is hash-uniform because that estimand is general unseen-source
    performance, not spatial stress (M8).
    """
    eligible = [p for p in survey["placements"] if p["eligible"]]
    if len(eligible) < n_placements:
        raise ValueError(
            f"{survey['room']}: only {len(eligible)} eligible placements "
            f"(>= {MIN_ELIGIBLE_GROUPS} passing source-distinct groups), need "
            f"{n_placements}. Correspondence excluded {survey['n_groups_failing']} of "
            f"{survey['n_groups']} groups before eligibility.")
    centroids = np.vstack([p["centroid_p"] for p in eligible])
    picks = farthest_point_selection(centroids, n_placements)
    return [eligible[i] for i in picks]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare the RAF Mapping-A (unseen-source) evaluation set")
    parser.add_argument('--raf-root', required=True)
    # P1: DERIVED from the Mapping-H runtime tree when omitted (<H>/mappingA). An
    # explicit value must still be a proper disjoint child of it -- see
    # resolve_output_dir for why equal and ancestor roots are refused.
    parser.add_argument('--output-dir', default=None)
    # N2: DISJOINT from Mapping H's data/RAF. Manifests are per-directory, so a
    # shared split root means both flavors overwrite one raf_publish_manifest.json
    # and each publish invalidates the other flavor's marker-to-manifest digest --
    # exactly the exp_19 r4-T4 failure, re-entering through the default value.
    parser.add_argument('--split-dir', default=MAPPINGA_SPLIT_ROOT)
    parser.add_argument('--rooms', nargs='+', default=['EmptyRoom', 'FurnishedRoom'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-placements', type=int, default=CANONICAL_N_PLACEMENTS)
    parser.add_argument('--k', type=int, default=CANONICAL_K)
    # Amendment 4: the scalar is DERIVED over the Mapping-A union. The flag is an
    # assertion -- a value that disagrees with the derivation stops the run rather
    # than overriding it.
    parser.add_argument('--scalar', type=float, default=None,
                        help="assert the derived amplitude scalar (optional)")
    # N4: the publication a "shared with Mapping H" claim names. Required for a
    # canonical run; without it a non-canonical run is tainted and claims nothing.
    parser.add_argument('--mappingH-dir', default=None,
                        help="published Mapping-H runtime tree (exp_19 --output-dir)")
    parser.add_argument('--readback-record', required=True)
    # N5: the committed correspondence audit. Its FULL digest is the publication's
    # correspondence_sha256, and canonically it must be the registered record.
    parser.add_argument('--correspondence-record', required=True)
    parser.add_argument('--non-canonical', action='store_true',
                        help="synthetic/test mode: the readback record is not "
                             "authenticated and every artifact is tainted")
    return parser


def near_silent_disclosure(audits, items_by_room):
    """The per-item near-silent-reference disclosure (Amendment 4.1).

    A context below the loader's gate is never substituted -- RAF_A_md has no
    amplitude test -- so it is not a violation. It IS a property of the items that
    condition on it: they carry a reference with effectively no energy, and that
    has to be visible in the record, per item, wherever those items are read.
    """
    by_item, captures, placements = {}, [], set()
    item_placement = {item["item_id"]: (item["room"], item["placement_id"])
                      for items in items_by_room.values() for item in items}
    for room in sorted(audits):
        for record in audits[room].get("near_silent_references", ()):
            entry = {"room": room, "capture_id": record["capture_id"],
                     "kind": record["kind"], "crop_peak": record["raw_peak_crop"],
                     "scaled_crop_peak": record["scaled_peak_crop"],
                     "scaled_dbfs": record["scaled_dbfs_crop"],
                     "item_ids": list(record.get("item_ids") or ())}
            captures.append(entry)
            for item_id in entry["item_ids"]:
                by_item.setdefault(item_id, []).append(
                    {"capture_id": entry["capture_id"],
                     "crop_peak": entry["crop_peak"],
                     "scaled_dbfs": entry["scaled_dbfs"]})
                if item_id in item_placement:
                    placements.add("/".join(item_placement[item_id]))
    return {
        "n_captures": len(captures),
        "n_items": len(by_item),
        "placements": sorted(placements),
        "captures": captures,
        "by_item": {item_id: sorted(refs, key=lambda r: r["capture_id"])
                    for item_id, refs in sorted(by_item.items())},
        "note": ("these captures sit below the loader's silence threshold at the "
                 "published scalar in a CONTEXT role only. SampleDataset tests "
                 "is_silence on the cropped TARGET and substitutes there; RAF_A_md "
                 "loads context audio with no amplitude test, so no item is "
                 "substituted and every item is evaluated as the manifest says. "
                 "The affected items condition on a reference with effectively no "
                 "energy, which is a measurement condition, not a defect -- it is "
                 "identical across arms and carries a labelled sensitivity row."),
    }


def annotate_near_silent_references(items_by_room, disclosure):
    """Attach each item's near-silent references to the item itself (in place)."""
    by_item = disclosure.get("by_item") or {}
    for items in items_by_room.values():
        for item in items:
            item["near_silent_references"] = list(by_item.get(item["item_id"], ()))
    return items_by_room


def canonical_identity_blockers(audio_union_digest=None, n_captures=None):
    """What still stands between this run and a canonical publication (N5).

    ``SHA256_SHAPE`` lets a marker name ANY well-formed digest, which is only
    acceptable while a value is genuinely unknowable. The audio-union digest is
    knowable exactly once -- from the canonical generation itself -- so a canonical
    run measures it, reports it for pinning, and refuses; every other registered
    digest is a committed input and must already be pinned.

    ``audio_union_digest=None`` means "not measured yet", so the same check can run
    before the survey and again after it.
    """
    blockers = []
    for key in unpinned_identity_keys("mappingA_prepare"):
        if key == "audio_union_sha256":
            continue
        blockers.append(f"registered {key} is still a placeholder")
    registered = CANONICAL_MAPPINGA_PREPARE_PARAMS["audio_union_sha256"]
    if audio_union_digest is not None:
        if registered is SHA256_SHAPE:
            blockers.append(
                f"registered audio_union_sha256 is still a placeholder -- this run "
                f"measures {audio_union_digest} over {n_captures} captures; pin that "
                "value (from a --non-canonical dry run) and re-run, or a canonical "
                "marker could name any union")
        elif audio_union_digest != registered:
            blockers.append(f"audio union {audio_union_digest} != registered "
                            f"{registered}")
    return blockers


def parameter_identity(args, n_items, digests, scalar):
    return {
        "rooms": list(args.rooms),
        "n_placements": int(args.n_placements),
        "k": int(args.k),
        "n_items": int(n_items),
        "match_algorithm_version": MATCH_ALGORITHM_VERSION,
        "match_p95_m": MATCH_P95_M,
        "match_max_m": MATCH_MAX_M,
        "match_ambiguity_margin": MATCH_AMBIGUITY_MARGIN,
        "placement_cap_m": PLACEMENT_CAP_M,
        "amplitude_scalar": float(scalar),
        # N3: 0.75 is the TARGET the exp_19 scalar was derived against (headroom
        # over the trained supports); 0.999 is the ceiling this run ENFORCES on
        # every written file. Naming both "ceiling" hid which number was checked.
        "amplitude_derivation_target": float(AMPLITUDE_DERIVATION_TARGET),
        "clip_ceiling": float(CLIP_CEILING),
        "correspondence_sha256": digests["correspondence"],
        "audio_union_sha256": digests["audio_union"],
        "readback_record_sha256": digests["readback"],
    }


def canonical_parameter_deviations(args):
    registered = CANONICAL_MAPPINGA_PREPARE_PARAMS
    deviations = []
    if list(args.rooms) != list(registered["rooms"]):
        deviations.append(f"rooms {list(args.rooms)} != {registered['rooms']}")
    if int(args.n_placements) != registered["n_placements"]:
        deviations.append(
            f"n_placements {args.n_placements} != {registered['n_placements']}")
    if int(args.k) != registered["k"]:
        deviations.append(f"k {args.k} != {registered['k']}")
    if args.scalar is not None and float(args.scalar) != registered["amplitude_scalar"]:
        deviations.append(
            f"amplitude_scalar {args.scalar} != {registered['amplitude_scalar']}")
    if int(args.seed) != 0:
        deviations.append(f"seed {args.seed} != 0")
    return deviations


def main(argv=None):
    args = build_parser().parse_args(argv)
    canonical = not args.non_canonical

    # Publish gate first: nothing is read or written under an unadjudicated record.
    readback = load_passing_record(args.readback_record, canonical=canonical,
                                   expected_raf_root=args.raf_root if canonical else None)
    readback_provenance = record_provenance(args.readback_record, readback,
                                            canonical=canonical)
    taint = list(readback_provenance["taint"])
    deviations = canonical_parameter_deviations(args)
    if canonical and deviations:
        raise ValueError(
            "refusing a canonical Mapping-A publication with non-registered "
            "parameters: " + "; ".join(deviations) + ". Pass --non-canonical for an "
            "experiment (its artifacts are tainted).")
    if deviations:
        taint.append("non-registered parameters: " + "; ".join(deviations))

    correspondence, correspondence_provenance = load_correspondence_record(
        args.correspondence_record, args.rooms, canonical, raf_root=args.raf_root)
    taint.extend(correspondence_provenance["taint"])

    if canonical:
        # cheap gate first: refuse before the survey if the identity cannot be met
        blockers = canonical_identity_blockers()
        if blockers:
            raise ValueError("refusing a canonical Mapping-A publication: "
                             + "; ".join(blockers))

    mappingH, h_taint = resolve_mappingH(args.mappingH_dir, args.rooms, canonical)
    taint.extend(h_taint)
    if mappingH:
        logger.info("Mapping-H publication %s (generation %s, %d files)",
                    mappingH["split_dir"], mappingH["generation"],
                    mappingH["n_files"])

    # P1: resolved (and refused) BEFORE the survey, so a wrong-flag run costs
    # nothing and can never touch the Mapping-H publication.
    args.output_dir = resolve_output_dir(
        args.output_dir, args.mappingH_dir,
        mappingH["pointer_rooms"] if mappingH else args.rooms)
    logger.info("Mapping-A runtime tree: %s", args.output_dir)

    # Pass 1: survey, select placements, build items. NOTHING is written yet -- the
    # amplitude audit over the resulting union has to pass first (M1).
    surveys, items_by_room, assignments = {}, {}, {}
    for room in args.rooms:
        room_dir = os.path.join(args.raf_root, "archived", room)
        logger.info("surveying %s", room_dir)
        survey = survey_room(room_dir, room)
        selected = select_placements(survey, args.n_placements)
        logger.info("%s: %d placements, %d eligible, %d selected", room,
                    survey["n_placements"],
                    sum(1 for p in survey["placements"] if p["eligible"]),
                    len(selected))
        items = []
        for placement in selected:
            items.extend(build_items(room, placement["placement_id"],
                                     placement["passing"], placement["assignment"],
                                     placement["match"], k=args.k, seed=args.seed))
            # P2: the authoritative per-slot correspondence for the groups these
            # items were cut from, kept for the validator and published with the
            # splits record so the attestation can be re-run offline.
            assignments.setdefault(room, {}).update(
                {key: [int(row) for row in rows]
                 for key, rows in placement["assignment"].items()})
        cross_check_correspondence(correspondence, room, survey)
        survey["selected_placements"] = [p["placement_id"] for p in selected]
        surveys[room] = survey
        items_by_room[room] = items

    all_items = [item for room in args.rooms for item in items_by_room[room]]
    n_items = len(all_items)
    # N6: canonically the REGISTERED count, otherwise the count the requested
    # parameters imply. Validating against len(all_items) passed by construction.
    expected_items = (CANONICAL_N_ITEMS if canonical else
                      args.n_placements * ARRAY_SIZE * len(args.rooms))
    validate_manifest({"items": all_items, "k": args.k},
                      expected_items=expected_items, k=args.k,
                      assignments=assignments)

    union = enumerate_audio_union(all_items)
    counts = union_report(union, all_items)
    logger.info("audio union: %d captures for %d items", counts["n_captures"],
                counts["n_items"])
    # Amendment 4: ONE resample pass over the union, then the registered formula
    # over THIS union, then the audit at the scalar that formula gave.
    measurements = {room: measure_union(os.path.join(args.raf_root, "archived", room),
                                        union[room])
                    for room in args.rooms}
    roles = enumerate_support_captures(all_items)
    scale_decision = derive_union_scalar(
        measurements, {room: roles[room][0] for room in args.rooms})
    scalar = scale_decision["scalar"]
    logger.info("Mapping-A amplitude scalar x%g = min(support %g, clamp %g) "
                "[bound by %s]; max context peak %.5f, max union peak %.5f over "
                "%d captures", scalar, scale_decision["support_term"],
                scale_decision["clamp_term"], scale_decision["binding_term"],
                scale_decision["max_support_peak"],
                scale_decision["max_written_peak"], scale_decision["n_written"])

    registered_scalar = CANONICAL_MAPPINGA_PREPARE_PARAMS["amplitude_scalar"]
    if canonical and scalar != registered_scalar:
        raise AmplitudePolicyError(
            f"the Mapping-A union derives x{scalar}, not the registered "
            f"x{registered_scalar}: a canonical publication must reproduce the "
            "registered identity, and the derivation is what proves it.",
            scale_decision)
    if args.scalar is not None and float(args.scalar) != scalar:
        raise AmplitudePolicyError(
            f"--scalar {args.scalar} does not match the scalar this union derives "
            f"(x{scalar}). The scalar is derived, never supplied: the flag asserts "
            "the derivation, it does not override it.", scale_decision)
    if scalar != registered_scalar:
        taint.append(f"amplitude scalar x{scalar} != registered x{registered_scalar}")

    # Amendment 4.1: the audit is told which captures are TARGETS (the only role
    # the loader's silence check can fire on) and which items draw each context.
    context_index = context_item_index(all_items)
    audits = {}
    for room in args.rooms:
        room_dir = os.path.join(args.raf_root, "archived", room)
        audits[room] = audit_amplitude_union(
            room_dir, union[room], scalar=scalar, measurements=measurements[room],
            target_ids=set(roles[room][1]),
            context_items=context_index.get(room, {}))
        logger.info("%s: amplitude audit clean (max scaled peak %.4f); "
                    "%d near-silent context references recorded", room,
                    audits[room]["max_scaled_peak"],
                    audits[room]["n_near_silent_references"])

    near_silent = near_silent_disclosure(audits, items_by_room)
    if near_silent["n_captures"]:
        logger.info("near-silent references: %d captures affecting %d of %d items "
                    "in placements %s -- recorded, not fatal (Amendment 4.1)",
                    near_silent["n_captures"], near_silent["n_items"], n_items,
                    near_silent["placements"])

    digests = {
        # the COMMITTED record, digested as read -- not a summary of this run (N5)
        "correspondence": correspondence_provenance["sha256"],
        "audio_union": canonical_digest(union),
        "readback": readback_provenance["sha256"],
        # kept for the record: what this run measured about its own survey
        "survey": canonical_digest({
            room: {"n_placements": s["n_placements"],
                   "selected": s["selected_placements"],
                   "n_groups_passing": s["n_groups_passing"],
                   "n_groups_failing": s["n_groups_failing"]}
            for room, s in surveys.items()}),
    }
    parameters = parameter_identity(args, n_items, digests, scalar)
    disclosure = scale_disclosure(
        scalar, mappingH["amplitude_scalar"] if mappingH else None)
    if canonical:
        blockers = canonical_identity_blockers(digests["audio_union"],
                                               counts["n_captures"])
        if blockers:
            raise ValueError("refusing a canonical Mapping-A publication: "
                             + "; ".join(blockers))

    # Amendment 4.1: the disclosure travels ON the items, so every reader of the
    # per-item manifest sees it. (The loader's eval JSON is a scene->filenames map
    # by contract -- src/data/dataset.py iterates its keys as scenes -- so the
    # per-item disclosure belongs in the per-item manifest, not there.)
    annotate_near_silent_references(items_by_room, near_silent)

    # Pass 2: stage the whole publication, then commit both roots together.
    with PublishTransaction(args.split_dir, kind="mappingA_prepare") as txn:
        staged_runtime = txn.stage(args.output_dir)
        staged_splits = txn.stage(args.split_dir)

        write_reports = {}
        for room in args.rooms:
            h_room = (mappingH["rooms"][room] if mappingH else
                      {"dir": None, "files": None})
            write_reports[room] = write_union(
                os.path.join(args.raf_root, "archived", room),
                os.path.join(staged_runtime.staging_dir, room), union[room],
                scalar=scalar,
                mappingH_room_dir=h_room["dir"],
                mappingH_generation=mappingH["generation"] if mappingH else None,
                mappingH_files=h_room["files"],
                mappingH_scalar=mappingH["amplitude_scalar"] if mappingH else None,
                target_ids=set(roles[room][1]))
            _write_json(staged_runtime.path(room, "metadata", METADATA_NAME),
                        {item["target_capture_id"]: item
                         for item in items_by_room[room]})

        _write_json(staged_runtime.path(PUBLICATION_POINTER), {
            "split_dir": os.path.abspath(args.split_dir),
            "output_dir": os.path.abspath(args.output_dir),
            "rooms": list(args.rooms),
            "flavor": "mappingA",
            "canonical": canonical,
            "taint": taint,
            "parameters": parameters,
            "scale_disclosure": disclosure,
            "readback_record": readback_provenance,
        })

        manifest = {room: sorted(f"{item['target_capture_id']}.wav"
                                 for item in items_by_room[room])
                    for room in args.rooms}
        _write_json(staged_splits.path(MANIFEST_NAME), manifest)
        _write_json(staged_splits.path("mappingA_splits_record.json"),
                    build_splits_record(surveys, items_by_room, counts, parameters,
                                        canonical, taint, readback_provenance,
                                        correspondence_provenance, digests["survey"],
                                        assignments, scale_decision, disclosure,
                                        near_silent))
        _write_json(staged_splits.path("mappingA_amplitude_audit.json"),
                    {"parameters": parameters, "canonical": canonical, "taint": taint,
                     "amplitude_scalar": scale_decision,
                     "scale_disclosure": disclosure,
                     "near_silent_references": near_silent,
                     "rooms": audits, "written": write_reports})

        marker = txn.commit(
            expectations={
                staged_runtime.dest_root: [PUBLICATION_POINTER] + [
                    f"{room}/metadata/{METADATA_NAME}" for room in args.rooms],
                staged_splits.dest_root: [MANIFEST_NAME, "mappingA_splits_record.json",
                                          "mappingA_amplitude_audit.json"]},
            validate_json=True,
            extra={"canonical": canonical, "taint": taint, "parameters": parameters,
                   "canonical_parameters": not deviations,
                   # Amendment 4: the levels differ from Mapping H's by decision,
                   # and the marker is where a consumer learns it
                   "scale_disclosure": disclosure,
                   "amplitude_derivation": scale_decision,
                   "near_silent_references": {
                       "n_captures": near_silent["n_captures"],
                       "n_items": near_silent["n_items"],
                       "placements": near_silent["placements"]},
                   "readback_record": readback_provenance})

    logger.info("published Mapping-A generation %s: %d items, %d captures",
                marker["generation"][:12], n_items, counts["n_captures"])
    return 0


def build_splits_record(surveys, items_by_room, counts, parameters, canonical, taint,
                        readback_provenance, correspondence_provenance=None,
                        survey_sha256=None, assignments=None, scale_decision=None,
                        disclosure=None, near_silent=None):
    """The committed description of how the Mapping-A set was cut."""
    rooms = {}
    for room, survey in surveys.items():
        items = items_by_room[room]
        displacements = [c["rx_displacement_m"] for item in items
                         for c in item["context"]]
        distances = [float(np.linalg.norm(np.asarray(c["tx_p"]) -
                                          np.asarray(item["tx_p"])))
                     for item in items for c in item["context"]]
        rooms[room] = {
            "n_captures": survey["n_captures"],
            "n_groups": survey["n_groups"],
            "n_placements": survey["n_placements"],
            "n_eligible_placements": sum(1 for p in survey["placements"]
                                         if p["eligible"]),
            "selected_placements": survey["selected_placements"],
            "n_groups_passing": survey["n_groups_passing"],
            "n_groups_failing": survey["n_groups_failing"],
            "excluded_wrong_size": survey["excluded_wrong_size"],
            "n_items": len(items),
            "displacements": {
                "p95_m": _distribution(survey["placement_p95_m"]),
                "context_to_target_m": _distribution(displacements),
            },
            "target_context_distance_m": _distribution(distances),
        }
    return {"parameters": parameters, "canonical": canonical, "taint": taint,
            "readback_record": readback_provenance,
            # N5: the audit that authorised the selection, and separately what this
            # run measured about its own survey (identical claims, different sources)
            "correspondence_record": correspondence_provenance,
            "survey_sha256": survey_sha256,
            # P2: the authoritative per-slot correspondence the items rest on, so
            # validate_manifest can attest a published manifest without re-surveying
            "assignments": assignments,
            # Amendment 4: how the scalar was derived, and that the two corpora sit
            # at different levels
            "amplitude_scalar": scale_decision,
            "scale_disclosure": disclosure,
            # Amendment 4.1: which items carry a near-silent reference, and which
            "near_silent_references": near_silent,
            "union": counts, "rooms": rooms}


def _distribution(values):
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "min": float(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()), "mean": float(arr.mean())}


if __name__ == '__main__':
    raise SystemExit(main())
