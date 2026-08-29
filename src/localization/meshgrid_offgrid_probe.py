"""exp_22 R1 controls -- the off-grid truth probe and the AGREE calibration (§2).

Two of the §2 controls cannot be read off the published rows, because both need
generations the registered pass deliberately never made. They live here, in one
tool, because they share every input: the same sixteen queries, the same frozen
stack, the same observation embedding.

**Off-grid truth probe.** For exactly the lexicographically first query of each
of the sixteen included rooms, generate ``K`` RIRs at the CONTINUOUS ground-truth
source position ``x*_s`` -- the position the half-metre lattice can only
approximate -- score them against the observation and report where that score
would have ranked among the query's grid candidates. It answers one question:
how much of the residual error is the grid's quantization rather than the
model's.

This control READS THE GROUND TRUTH, by design and by registration. That is
legitimate here and nowhere else in exp_22: the truth position is used only to
place a generation whose score is REPORTED, never to place a candidate. It never
enters any candidate set, any argmax, any prediction or any published
localization metric -- see :data:`CONTROL_LABEL`, which is stamped into every
record and every artifact this module writes.

**Real-vs-generated AGREE calibration.** On the same sixteen queries, compare
``cos(E(h_obs), E(h_real, other))`` against ``cos(E(h_obs), E(h_generated))``.
The real bank is the query's own frozen D1 context RIRs: real measured RIRs of
the same room, at the same receiver, from other sources -- the only real-RIR bank
the registered pass materializes, and one whose identity is already pinned by the
D1 manifest's per-context sha256. If the generated distribution sits far below
the real one, the scorer is being asked to rank inside a domain gap, and that is
a property of the embedding rather than of the localization.

Nothing runs until the artifacts agree: the published run binding must hash to
its own content and must match, field by field, the binding this probe builds
from its own checkpoint, scorer, D1 manifest, G1 report and sampler settings.
The probe is otherwise a normal announcement-08 citizen: every generated
waveform is saved with its sha256 and a manifest.
"""
import argparse
import hashlib
import io
import json
import os

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization import scoring as sc
from src.localization.reaggregate import decode_scores

#: stamped into every record, every JSON and the markdown. The label is the
#: control's containment: it says what the probe is allowed to do with the truth.
CONTROL_LABEL = (
    "OFF-GRID TRUTH CONTROL -- this probe generates at the CONTINUOUS ground-truth source "
    "position x*_s and therefore READS THE HELD-OUT TARGET, by design and by registration "
    "(inherited plan §2). Its generation is NEVER inserted into any candidate set, never "
    "competes in any argmax, never becomes a prediction and never enters any published "
    "localization metric; it exists only to report how the truth position would have SCORED "
    "against the grid the engine actually searched")

CALIBRATION_LABEL = (
    "REAL-VS-GENERATED AGREE CALIBRATION -- cos(E(h_obs), E(h_real,other)) over the query's "
    "frozen D1 context RIRs (real measured RIRs of the same room, same receiver, other sources; "
    "their bytes are pinned by the D1 manifest's per-context sha256) against cos(E(h_obs), "
    "E(h_generated)) over the off-grid truth generations. Both distributions are reported; "
    "neither is a localization metric, and the comparison diagnoses the embedding's domain gap "
    "only")

#: the binding fields the probe must agree with the published run on.
#:
#: Everything the run binding pins EXCEPT ``dump_cases_sha256``: that field is
#: the localization pass's dump AUTHORITY, and this probe is not a localization
#: pass -- it dumps exactly the sixteen registered off-grid probe queries, which
#: the announcement-08 exemption names directly rather than through a case list.
PROBE_BINDING_FIELDS = tuple(field for field in me.RUN_BINDING_FIELDS
                             if field != "dump_cases_sha256")

#: the sentinel candidate index the off-grid draw is keyed with.
#:
#: Under the registered common-random-numbers policy the key does not depend on
#: the candidate at all, so the truth generation is drawn from EXACTLY the same K
#: latents as every grid candidate of that query -- which is what makes the rank
#: comparison a comparison between POSITIONS. A per-candidate policy has no key
#: for a point that is not a candidate, so it is refused rather than invented.
OFFGRID_CANDIDATE_SENTINEL = -1

WAVEFORM_DIRNAME = "waveforms"
PROBE_REPORT_JSON = "offgrid_probe_report.json"
PROBE_REPORT_MARKDOWN = "offgrid_probe_report.md"

#: What binds the probe's LIVE observation to the frozen grid rows -- and, just
#: as importantly, what does not.
#:
#: Surveyed across the published artifacts: the D1 record pins the eight CONTEXT
#: RIRs (``context_audio_sha256``, verified per query by
#: ``meshgrid_engine.verify_context_record``) and the context poses; the engine
#: row pins its own claims and the similarity sidecar (``sims_sha256``). NOTHING
#: digests the observed RIR -- not the manifest, not the row, not the binding
#: (checked over all 1,566 published rows). No field is invented here to pretend
#: otherwise (Codex r9i review, item 2).
#:
#: What IS available is a functional tie, and it is stronger than a digest of a
#: file nobody registered: one of the row's OWN stored similarities is
#: re-derived from the live observation, using the same keyed noise and the same
#: candidate, and must reproduce the frozen value. s[x, k] = cos(E(h_obs),
#: E(h_hat[x, k])), so a different observation moves it directly and by O(1),
#: while the only admissible difference is the batch-shape noise the engine
#: already registers a bound for.
OBSERVATION_BINDING_NOTE = (
    "the observed RIR is bound TWICE, because the two checks answer different questions and "
    "neither answers the other's. (1) THE PIN, over SOURCE BYTES: the observation-bank digest "
    "covers the on-disk bytes of the sixteen registered probe queries' RIR files, is computed "
    "with no run and no model, and is PRE-REGISTERED before any result exists -- the same "
    "chronology that closes the pair-metadata bank (Planner RULING 2). A canonical control "
    "requires it back as --expect-observation-bank-sha256, recomputes it, and additionally "
    "requires the file the released loader actually opened to hash to that query's pinned "
    "digest, so a divergent dataset root cannot satisfy the bank while the loader reads "
    "elsewhere. (2) THE TIE, over the TENSOR PATH: no registered run artifact digests the "
    "observation -- the D1 record pins the eight CONTEXT RIRs (verified per query by the "
    "engine's verify_context_record) and the row pins its own claims and its similarity "
    "sidecar, but the observation is in none of them (Codex r9i review, item 2) -- so the live "
    "observation is tied to the frozen rows FUNCTIONALLY: the query's candidates are regenerated "
    "from the same keyed noise and the same conditioning, and their cosines against the LIVE "
    "observation must reproduce the similarities the row already published. As of r9s the "
    "replay runs at the row's OWN stamped batching over the whole query, where the computation "
    "is bit-exact (r9r measured 11,577 candidates with zero exceptions), so the admissible "
    "difference is the float16 sidecar's own half-ulp and nothing else, and the float32 "
    "aggregate must match the row's published score at exactly 0. It is NOT the engine's "
    "SCORE_TOLERANCE, which is an aggregate bound for a CHANGED-batching replay "
    "(MATCHED_BATCHING_TIE). Because "
    "s[x, k] = cos(E(h_obs), E(h_hat[x, k])), a substituted observation moves that number "
    "directly. Together: the pin says these are the registered bytes, the tie says the tensor "
    "those bytes decoded to is the one the frozen rows were scored against -- a byte-identical "
    "file loaded through a changed crop or sample rate passes the pin and fails the tie")

#: the §2 sibling this report must not contradict (Codex r9t blocker 5).
RETRIEVAL_HANDOFF_FILENAME = "retrieval_control_handoff.json"

RETRIEVAL_RECONCILIATION_NOTE = (
    "§2 COMPLETENESS. The sparse-bank AGREE retrieval control is a different tool with its own "
    "run, so this report cannot know its status by inspection -- and r9s shipped a bundle whose "
    "off-grid report still asserted a stale pending status while the retrieval report declared a "
    "canonical "
    "run (Codex r9t blocker 5). A canonical off-grid control therefore requires that control's "
    "own retrieval_control_handoff.json, reads its status, canonicality and headline out of it, "
    "and records the handoff's digest, so the two halves of the bundle cannot disagree about "
    "whether §2 is complete")

#: the launch provenance the SOP requires to exist AT LAUNCH, not afterwards.
LAUNCH_RECORD_FIELDS = ("argv", "git_sha", "hostname", "gpus")

LAUNCH_RECORD_NOTE = (
    "LAUNCH PROVENANCE (experiment_SOP.md:37, Codex r9t blocker 4). Logical CUDA indices and "
    "input digests do not say which machine, which commit or which physical card produced a "
    "result. A canonical run requires a launch record written BEFORE it starts -- exact argv, "
    "the repository's git SHA, the hostname and the nvidia-smi UUID of every visible GPU -- and "
    "its sha256 is hashed into this report's provenance. Produce it with the very command you "
    "are about to run: replace --launch-record PATH with --emit-launch-record PATH, which writes "
    "the record and exits, then run the same command with --launch-record PATH. The recorded "
    "argv is compared against the run's own argv, so a record written for a different command "
    "refuses rather than vouching for this one. Admission then COMPARES the record against the "
    "executing environment -- git SHA equal to HEAD, same hostname, a clean TRACKED tree both "
    "when the record was written and while it runs, and the CUDA runtime's own UUID for the "
    "executing device EQUAL to the one the record designated at emission (not merely among the "
    "nvidia-smi entries it enumerated, which are context) -- and names any mismatch (Codex r9v "
    "residual 4b, r9z shred 2, r9z3)")


DIRTY_SEMANTICS_NOTE = (
    "WHAT 'DIRTY' MEANS (Codex r9v residual 4a). git_status_dirty is TRACKED modifications and "
    "staged changes only -- `git status --porcelain --untracked-files=no` -- because that is the "
    "question the record has to answer: is the code that ran the code that is committed. "
    "Untracked paths are listed separately under untracked_paths and are INFORMATIONAL: this "
    "repository's runtime directories (outputs_loc/, AcousticRooms/, weights/, wandb/) are "
    "gitignored-or-untracked by design, and r9u's record read dirty:true from exactly those "
    "while its tracked tree at HEAD 57b6f52 was clean. Recording both fields makes that "
    "auditable instead of ambiguous, and a TRACKED-dirty record refuses canonical admission")


CAPTURE_FAILURE_NOTE = (
    "CAPTURE FAILURES FAIL CLOSED (Codex r9x residual 4). Every environment fact this comparison "
    "needs is captured with an explicit verdict, never with a value that doubles as its own "
    "failure. r9w read the tracked-dirty state from `git status --porcelain --untracked-files=no` "
    "and passed it through bool(): a clean tree returns an empty string and a FAILED capture "
    "returned None, and bool() maps both to False, so a git that could not run read as 'clean'. "
    "The same shape sat in the SHA and hostname comparisons, which were guarded by `if "
    "environment.get(...)` and therefore SKIPPED when the capture failed, while the record still "
    "came back environment_verified. An unavailable fact is now a refusal with a named reason, "
    "and environment_verified is true only when every axis affirmatively passed")

#: the axes a launch record is held to. environment_verified needs all of them.
ENVIRONMENT_AXES = ("git_sha", "git_tracked_clean", "hostname", "gpu_uuid")


def _capture(command, timeout=60):
    """``{"ok", "value", "error"}`` -- a capture that cannot be mistaken for a value.

    The whole point is that ``ok`` is separate from ``value``: an empty stdout is
    a legitimate answer for ``git status`` and a total failure for ``git
    rev-parse``, and only the caller knows which. Returning ``None`` for both, as
    r9w did, is what let a failed capture read as a clean tree.
    """
    import subprocess

    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.SubprocessError) as error:       # noqa: BLE001 -- recorded
        return {"ok": False, "value": None,
                "error": f"{' '.join(str(part) for part in command)!r} could not be run: {error}"}
    if done.returncode != 0:
        return {"ok": False, "value": None,
                "error": f"{' '.join(str(part) for part in command)!r} exited "
                         f"{done.returncode}: {(done.stderr or '').strip()[:200]}"}
    return {"ok": True, "value": done.stdout.strip(), "error": None}


#: how the executing card is identified, and why nothing infers it.
RUNTIME_UUID_NOTE = (
    "THE RUNTIME ANSWERS FOR ITS OWN DEVICE (Codex r9z3). r9z2 resolved the executing card by "
    "mapping the logical ordinal through CUDA_VISIBLE_DEVICES onto nvidia-smi's index order. "
    "That inference is wrong in general: CUDA_DEVICE_ORDER selects between FASTEST_FIRST (the "
    "default) and PCI_BUS_ID, so the CUDA runtime's ordinals need not follow NVML's indices at "
    "all, and the two orderings can disagree without either being wrong. The designation is now "
    "taken from the runtime itself -- torch.cuda.get_device_properties(<ordinal>).uuid, which "
    "answers for the device the run will actually use under whatever visibility and ordering are "
    "in force -- at emission AND at admission, and the two must be equal. The nvidia-smi "
    "enumeration is still recorded, as CONTEXT ONLY: it says what the machine had, not which "
    "card was used. If the runtime cannot answer -- no CUDA, an ordinal it does not have, or a "
    "torch too old to expose the UUID -- emission refuses rather than falling back to a guess")


def normalize_gpu_uuid(value):
    """``GPU-<hex>`` from whatever form a UUID arrives in.

    ``torch.cuda.get_device_properties(...).uuid`` is a ``_CUuuid`` whose ``str``
    is the bare hex form; nvidia-smi prints the same value with a ``GPU-``
    prefix. One spelling, so an equality test means what it looks like.
    """
    text = str(value).strip()
    if not text:
        return ""
    return text if text.startswith("GPU-") else f"GPU-{text}"


def runtime_gpu_uuid(device):
    """The UUID the CUDA RUNTIME reports for the device this run will use.

    Not inferred from an ordinal, an environment variable or an NVML index --
    asked of the runtime that is about to execute, which is the only party that
    knows how ``CUDA_VISIBLE_DEVICES`` and ``CUDA_DEVICE_ORDER`` combine on this
    machine. See :data:`RUNTIME_UUID_NOTE`.
    """
    text = str(device)
    if not text.startswith("cuda"):
        return {"ok": False, "uuid": None, "logical_index": None,
                "error": f"the executing device {device!r} is not a CUDA device, so the runtime "
                         "has no card to answer for"}
    try:
        logical = int(text.split(":")[-1]) if ":" in text else 0
    except ValueError:
        return {"ok": False, "uuid": None, "logical_index": None,
                "error": f"the executing device {device!r} names no ordinal the runtime can "
                         "resolve"}
    try:
        import torch
    except ImportError as error:                                 # noqa: BLE001 -- recorded
        return {"ok": False, "uuid": None, "logical_index": logical,
                "error": f"torch could not be imported, so the runtime cannot be asked: {error}"}
    if not torch.cuda.is_available():
        return {"ok": False, "uuid": None, "logical_index": logical,
                "error": "the CUDA runtime reports no available device, so nothing can answer "
                         "for the card this run would use"}
    count = int(torch.cuda.device_count())
    if logical >= count:
        return {"ok": False, "uuid": None, "logical_index": logical, "device_count": count,
                "error": f"the run uses {device!r} but the CUDA runtime exposes {count} "
                         "device(s), so that ordinal names no card"}
    try:
        properties = torch.cuda.get_device_properties(logical)
    except (RuntimeError, AssertionError, AttributeError) as error:   # noqa: BLE001 -- recorded
        return {"ok": False, "uuid": None, "logical_index": logical, "device_count": count,
                "error": f"the runtime could not describe device {logical}: {error}"}
    raw = getattr(properties, "uuid", None)
    uuid = normalize_gpu_uuid(raw) if raw is not None else ""
    if not uuid or uuid == "GPU-":
        return {"ok": False, "uuid": None, "logical_index": logical, "device_count": count,
                "error": "this torch does not expose a device UUID "
                         "(torch.cuda.get_device_properties(...).uuid), so the card cannot be "
                         "identified without inferring it -- which is what this check exists to "
                         "stop"}
    return {"ok": True, "uuid": uuid, "logical_index": logical, "device_count": count,
            "name": getattr(properties, "name", None), "error": None}


def current_environment():
    """The machine as it is RIGHT NOW, with a verdict on every capture.

    Separate from :func:`build_launch_record` so admission can compare the two:
    a record is a claim about an environment, and a claim nothing is checked
    against vouches for nothing (Codex r9v residual 4b). Each fact carries its
    own ``*_capture`` verdict so an unavailable one cannot be mistaken for a
    benign value (Codex r9x residual 4a; :data:`CAPTURE_FAILURE_NOTE`).
    """
    import platform

    sha = _capture(["git", "rev-parse", "HEAD"])
    # TRACKED only: --untracked-files=no. The untracked list is gathered
    # separately and never decides anything (DIRTY_SEMANTICS_NOTE)
    tracked = _capture(["git", "status", "--porcelain", "--untracked-files=no"])
    everything = _capture(["git", "status", "--porcelain", "--untracked-files=normal"])
    listed = _capture(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])

    gpus, gpu_capture = [], dict(listed)
    if listed["ok"]:
        for line in (listed["value"] or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                gpus.append({"index": int(parts[0]), "uuid": parts[1],
                             "name": parts[2] if len(parts) > 2 else None})
        if not gpus:
            gpu_capture = {"ok": False, "value": listed["value"],
                           "error": "nvidia-smi ran but enumerated no usable index/UUID rows, so "
                                    "no physical card can be identified"}
        elif not all(str(gpu["uuid"]).startswith("GPU-") for gpu in gpus):
            gpu_capture = {"ok": False, "value": listed["value"],
                           "error": f"nvidia-smi returned entries that are not GPU- UUIDs: "
                                    f"{[gpu['uuid'] for gpu in gpus][:3]}"}

    hostname = platform.node()
    host_capture = ({"ok": True, "value": hostname, "error": None} if hostname else
                    {"ok": False, "value": None,
                     "error": "platform.node() returned an empty hostname, so this machine does "
                              "not identify itself"})
    untracked = sorted(line[3:] for line in (everything["value"] or "").splitlines()
                       if line.startswith("?? ")) if everything["ok"] else []
    return {"git_sha": sha["value"] if sha["ok"] else None,
            "git_sha_capture": sha,
            # None, not False: "could not look" is not "nothing changed"
            "git_status_dirty": (bool(tracked["value"]) if tracked["ok"] else None),
            "git_status_capture": tracked,
            "git_tracked_changes": sorted((tracked["value"] or "").splitlines())
                                   if tracked["ok"] else [],
            "untracked_paths": untracked,
            "untracked_capture": everything,
            "hostname": hostname or None,
            "hostname_capture": host_capture,
            "gpus": gpus,
            "gpu_capture": gpu_capture,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            # context only: the runtime is asked directly, so neither of these
            # decides which card a run used (RUNTIME_UUID_NOTE)
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "capture_failure_note": CAPTURE_FAILURE_NOTE,
            "dirty_semantics_note": DIRTY_SEMANTICS_NOTE}


def compare_environment(record, environment, device=None, runtime=None):
    """One verdict per axis: recorded value, live value, pass or why not.

    Persisted whole into the report's provenance, so the check a canonical run
    passed is auditable rather than implied by a single boolean (Codex r9x
    residual 4c). Every axis must be present and pass; a capture that failed is
    a refusal with the capture's own error, never a skip.
    """
    axes = []

    def _axis(name, recorded, live, ok, why=None):
        axes.append({"axis": name, "recorded": recorded, "live": live,
                     "verdict": "pass" if ok else "fail", "why": why})

    sha_capture = environment.get("git_sha_capture") or {}
    if not sha_capture.get("ok") or not environment.get("git_sha"):
        _axis("git_sha", record.get("git_sha"), None, False,
              f"the live commit could not be captured: "
              f"{sha_capture.get('error') or 'no git_sha in the environment'}")
    elif str(record.get("git_sha")) != str(environment["git_sha"]):
        _axis("git_sha", record.get("git_sha"), environment["git_sha"], False,
              f"the record says {str(record.get('git_sha'))[:12]}... and HEAD is "
              f"{str(environment['git_sha'])[:12]}..., so the code that ran is not the code the "
              "record names")
    else:
        _axis("git_sha", record.get("git_sha"), environment["git_sha"], True)

    status_capture = environment.get("git_status_capture") or {}
    live_dirty = environment.get("git_status_dirty")
    recorded_dirty = record.get("git_status_dirty")
    if not isinstance(recorded_dirty, bool):
        # NO bool(None) PATH (Codex r9z shred 1): an absent or null dirty field
        # is not "clean", it is a record that never stated the fact
        _axis("git_tracked_clean", recorded_dirty, live_dirty, False,
              f"the record's tracked-tree state is {recorded_dirty!r}, not a boolean; a record "
              "that does not state whether its tree was clean cannot vouch that its git SHA "
              "describes the code that ran")
    elif not status_capture.get("ok") or live_dirty is None:
        _axis("git_tracked_clean", record.get("git_status_dirty"), None, False,
              f"the live tracked-tree state could not be captured: "
              f"{status_capture.get('error') or 'no git_status_dirty in the environment'}")
    elif recorded_dirty:
        _axis("git_tracked_clean", True, live_dirty, False,
              "the RECORD was written from a tracked-dirty tree "
              f"({record.get('git_tracked_changes') or 'changes not enumerated'}), so its git "
              "SHA does not describe the code it ran")
    elif bool(live_dirty):
        _axis("git_tracked_clean", record.get("git_status_dirty"), True, False,
              "this tree has TRACKED modifications right now "
              f"({environment.get('git_tracked_changes') or 'changes not enumerated'}), so the "
              "running code is not the committed code the record names")
    else:
        _axis("git_tracked_clean", False, False, True)

    host_capture = environment.get("hostname_capture") or {}
    if not host_capture.get("ok") or not environment.get("hostname"):
        _axis("hostname", record.get("hostname"), None, False,
              f"the live hostname could not be captured: "
              f"{host_capture.get('error') or 'no hostname in the environment'}")
    elif str(record.get("hostname")) != str(environment["hostname"]):
        _axis("hostname", record.get("hostname"), environment["hostname"], False,
              f"the record says {record.get('hostname')!r} and this machine is "
              f"{environment['hostname']!r}")
    else:
        _axis("hostname", record.get("hostname"), environment["hostname"], True)

    # THE DESIGNATED CARD, asked of the RUNTIME (Codex r9z shred 2, r9z3). The
    # nvidia-smi set is context: membership in it passes a run that changed
    # visibility onto a different card the set happens to contain, and mapping
    # an ordinal onto NVML indices is wrong whenever CUDA_DEVICE_ORDER disagrees
    # with NVML's ordering. The runtime answers for the card it will actually
    # use, at emission and here, and the two must be EQUAL (RUNTIME_UUID_NOTE).
    designated = record.get("execution_gpu_uuid")
    recorded_uuids = sorted({str(gpu.get("uuid") or "") for gpu in (record.get("gpus") or [])})
    if not designated:
        _axis("gpu_uuid", None, None, False,
              "the record designates no execution GPU UUID, so there is no card for this run to "
              "be held to; it was written before the designation existed, or by a path that "
              "could not resolve one")
    elif device is None:
        _axis("gpu_uuid", designated, None, False,
              "no executing device was supplied, so the physical card this run uses was never "
              "identified")
    else:
        live = runtime_gpu_uuid(device) if runtime is None else dict(runtime)
        if not live.get("ok"):
            _axis("gpu_uuid", designated, None, False,
                  f"the CUDA runtime could not identify the card this run uses: "
                  f"{live.get('error') or 'no verdict was recorded'}")
        elif normalize_gpu_uuid(live["uuid"]) != normalize_gpu_uuid(designated):
            _axis("gpu_uuid", designated, live["uuid"], False,
                  f"the runtime reports {live['uuid']} for {device!r} but the record designates "
                  f"{designated} (emitted for {record.get('execution_device')!r}); the card in "
                  f"use is not the card this record was written for"
                  + (", though it is among the ones nvidia-smi enumerated -- which is exactly "
                     "why membership is not the test"
                     if normalize_gpu_uuid(live["uuid"])
                     in {normalize_gpu_uuid(uuid) for uuid in recorded_uuids} else ""))
        else:
            _axis("gpu_uuid", designated, live["uuid"], True)
            axes[-1]["logical_index"] = live.get("logical_index")
            axes[-1]["runtime_device_count"] = live.get("device_count")
            axes[-1]["source"] = "cuda_runtime"
            # context only: what the machine had, not which card was used
            axes[-1]["nvidia_smi_uuid_set"] = recorded_uuids
            axes[-1]["visible_devices"] = environment.get("cuda_visible_devices")
            axes[-1]["device_order"] = environment.get("cuda_device_order")


    verified = bool(axes and len(axes) == len(ENVIRONMENT_AXES)
                    and all(axis["verdict"] == "pass" for axis in axes))
    return {"axes": axes, "verified": verified,
            "n_axes": len(axes), "n_passed": sum(1 for a in axes if a["verdict"] == "pass"),
            "failures": [axis for axis in axes if axis["verdict"] != "pass"],
            "capture_failure_note": CAPTURE_FAILURE_NOTE}


def build_launch_record(argv, *, device=None, emit_flag="--emit-launch-record",
                        record_flag="--launch-record", environment=None, runtime=None):
    """Everything SOP:37 wants, gathered from the machine that is about to run.

    ``argv`` is stored with the emit/record flag pair stripped, so the record
    written by ``--emit-launch-record PATH`` describes exactly the run that
    ``--launch-record PATH`` then performs.

    EMISSION FAILS CLOSED TOO (Codex r9z shred 1). r9y hardened admission but
    still let emission write whatever it managed to capture, so a machine whose
    git could not run produced a record with ``git_status_dirty: None`` -- a
    field admission then had to treat as a value. A record that cannot state
    every fact it exists to state is not written at all.

    THE DESIGNATED CARD (Codex r9z shred 2). The executing device's PHYSICAL
    UUID is resolved here, through the emission-time ``CUDA_VISIBLE_DEVICES``,
    and stored as ``execution_gpu_uuid``. Admission then requires the run-time
    resolution to EQUAL it: membership in the recorded set is not enough,
    because a visibility change between emit and run can land on a different
    card that the set happens to contain.
    """
    environment = current_environment() if environment is None else dict(environment)
    refusals = []
    for field, capture, why, kind in (
            ("git_sha", environment.get("git_sha_capture") or {},
             "the source commit", "text"),
            ("git_status_dirty", environment.get("git_status_capture") or {},
             "the tracked-tree state", "bool"),
            ("hostname", environment.get("hostname_capture") or {}, "the hostname", "text"),
            ("gpus", environment.get("gpu_capture") or {}, "the GPU enumeration", "list")):
        value = environment.get(field)
        if not capture.get("ok"):
            refusals.append(f"{why} could not be captured: "
                            f"{capture.get('error') or 'no verdict was recorded'}")
        elif value is None:
            refusals.append(f"{why} came back empty despite a successful capture")
        # NON-EMPTY, not merely non-None (Codex r9z3 shred 1). An empty string is
        # what a capture that "succeeded" and produced nothing looks like, and it
        # is exactly as unusable as a missing one
        elif kind == "text" and not str(value).strip():
            refusals.append(f"{why} came back as an empty string despite a successful capture")
        elif kind == "list" and not list(value):
            refusals.append(f"{why} came back as an empty list despite a successful capture")
        # the tracked-dirty field must be a BOOL: that is the one field whose
        # false value is legitimate, which is exactly why it must not be able to
        # arrive absent or as something else (CAPTURE_FAILURE_NOTE)
        elif kind == "bool" and not isinstance(value, bool):
            refusals.append("the tracked-tree state is not a boolean, so the record would carry "
                            "a dirty field that means nothing")
    resolved = None
    if not refusals:
        # THE RUNTIME, not an inference (RUNTIME_UUID_NOTE)
        resolved = runtime_gpu_uuid(device) if runtime is None else dict(runtime)
        if not resolved.get("ok"):
            refusals.append(f"the CUDA runtime could not identify the card this run would use: "
                            f"{resolved.get('error') or 'no verdict was recorded'}")
    if refusals:
        raise ValueError(
            "this launch record cannot be written -- " + "; ".join(refusals)
            + f". A record that cannot state a fact must not record a value for it. "
              f"{LAUNCH_RECORD_NOTE} {CAPTURE_FAILURE_NOTE} {RUNTIME_UUID_NOTE}")

    record = {"argv": strip_launch_flags(argv, flags=(emit_flag, record_flag)),
              "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              # the card this record DESIGNATES, taken from the CUDA runtime at
              # emission. The nvidia-smi set below is context, never the test
              "execution_device": str(device),
              "execution_gpu_uuid": normalize_gpu_uuid(resolved["uuid"]),
              "execution_logical_index": resolved.get("logical_index"),
              "execution_gpu_name": resolved.get("name"),
              "execution_runtime_device_count": resolved.get("device_count"),
              "execution_uuid_source": "cuda_runtime",
              "runtime_uuid_note": RUNTIME_UUID_NOTE,
              "note": LAUNCH_RECORD_NOTE}
    record.update({key: environment.get(key) for key in
                   ("git_sha", "git_status_dirty", "git_tracked_changes", "untracked_paths",
                    "hostname", "gpus", "cuda_visible_devices", "cuda_device_order",
                    "dirty_semantics_note")})
    return record


def strip_launch_flags(argv, flags=("--emit-launch-record", "--launch-record")):
    """``argv`` without the launch-record flag and its value."""
    out, skip = [], False
    for token in [str(item) for item in (argv or [])]:
        if skip:
            skip = False
            continue
        if token in flags:
            skip = True
            continue
        if any(token.startswith(f"{flag}=") for flag in flags):
            continue
        out.append(token)
    return out


def write_launch_record(path, argv, device=None, runtime=None):
    record = build_launch_record(argv, device=device, runtime=runtime)
    me.write_json(str(path), record)
    return record


def read_verified_launch_record(path, argv, device=None, environment=None,
                                runtime=None):
    """ONE read: hash the bytes, parse the same buffer, then hold it to the run.

    Two rounds of checks, both fail-closed. The record's own shape first -- it
    must name this command, a full commit id, a host and real nvidia-smi UUIDs.
    Then, and this is what r9u was missing (Codex r9v residual 4b), the record is
    COMPARED against the environment executing right now: the same commit at
    HEAD, the same hostname, the executing device's physical UUID among the
    recorded ones, and a clean TRACKED tree on both sides. A claim nobody checks
    against the machine it claims about vouches for nothing.

    ``environment`` is injectable so the comparison itself can be tested on
    every axis; production passes ``None`` and reads the live machine.
    """
    with open(str(path), "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:            # noqa: BLE001 -- a refusal
        raise ValueError(f"the launch record at {path!r} is not readable JSON: {error}. "
                         f"{LAUNCH_RECORD_NOTE}") from error
    missing = [field for field in LAUNCH_RECORD_FIELDS if not record.get(field)]
    if missing:
        raise ValueError(f"the launch record at {path!r} is missing {missing}; SOP:37 wants the "
                         f"exact command, the source SHA, the host and the physical GPU "
                         f"identity. {LAUNCH_RECORD_NOTE}")
    if len(str(record["git_sha"])) != 40:
        raise ValueError(f"the launch record's git_sha {record['git_sha']!r} is not a full "
                         f"40-character commit id; a short or absent SHA does not identify the "
                         f"code that ran. {LAUNCH_RECORD_NOTE}")
    recorded = [str(token) for token in record["argv"]]
    running = strip_launch_flags(argv)
    if recorded != running:
        raise ValueError(
            f"the launch record at {path!r} was written for a different command: it records "
            f"{recorded[:6]}... and this run is {running[:6]}.... A record that does not describe "
            f"this invocation vouches for nothing. {LAUNCH_RECORD_NOTE}")
    uuids = [str(gpu.get("uuid") or "") for gpu in record["gpus"]]
    if not all(uuid.startswith("GPU-") for uuid in uuids):
        raise ValueError(f"the launch record's GPU entries {uuids[:3]} are not nvidia-smi UUIDs; "
                         f"a logical index does not identify a physical card. {LAUNCH_RECORD_NOTE}")
    # NOTE: r9w compared the logical ordinal against the record's PHYSICAL
    # indices here. That check is gone rather than fixed: under
    # CUDA_VISIBLE_DEVICES the two numbering schemes are different things, and
    # the gpu_uuid axis below answers the real question -- which physical card
    # is this run executing on, and does the record name it (Codex r9x 4b).

    # AGAINST THE EXECUTING ENVIRONMENT (Codex r9v residual 4b, r9x residual 4).
    # Up to here the record has only been checked for internal shape. Every axis
    # is now compared and its verdict kept: a failed capture refuses by name
    # rather than being skipped, and environment_verified is true only when all
    # four axes affirmatively passed.
    environment = current_environment() if environment is None else dict(environment)
    comparison = compare_environment(record, environment, device=device, runtime=runtime)
    if comparison["failures"]:
        raise ValueError(
            f"the launch record at {path!r} does not describe the environment this run is "
            f"executing in -- "
            + "; ".join(f"{axis['axis']}: {axis['why']}" for axis in comparison["failures"])
            + f". {LAUNCH_RECORD_NOTE} {DIRTY_SEMANTICS_NOTE} {CAPTURE_FAILURE_NOTE}")

    return {"path": str(path), "sha256": digest, "n_bytes": len(raw),
            "argv": recorded, "git_sha": str(record["git_sha"]),
            # a bool by now: the git_tracked_clean axis refuses anything else,
            # so there is no bool(None) left anywhere (Codex r9z shred 1)
            "git_status_dirty": record["git_status_dirty"],
            "git_tracked_changes": list(record.get("git_tracked_changes") or []),
            # informational, never a refusal: this repo's runtime dirs are
            # untracked by design (DIRTY_SEMANTICS_NOTE)
            "untracked_paths": list(record.get("untracked_paths") or []),
            "n_untracked_paths": len(record.get("untracked_paths") or []),
            "hostname": str(record["hostname"]),
            "gpus": record["gpus"],
            "executing_device": (None if device is None else str(device)),
            "execution_gpu_uuid": record.get("execution_gpu_uuid"),
            "execution_gpu_name": record.get("execution_gpu_name"),
            "execution_uuid_source": record.get("execution_uuid_source"),
            "recorded_execution_device": record.get("execution_device"),
            # the whole comparison, axis by axis, so a reviewer reads what
            # passed instead of trusting a boolean (Codex r9x residual 4c)
            "environment_comparison": comparison["axes"],
            "environment_axes_passed": comparison["n_passed"],
            "environment_axes_expected": len(ENVIRONMENT_AXES),
            "environment_verified": bool(comparison["verified"]),
            "live_environment": {"git_sha": environment.get("git_sha"),
                                 "git_status_dirty": environment.get("git_status_dirty"),
                                 "hostname": environment.get("hostname"),
                                 "cuda_visible_devices":
                                     environment.get("cuda_visible_devices"),
                                 "gpus": environment.get("gpus")},
            "cuda_visible_devices": record.get("cuda_visible_devices"),
            "recorded_utc": record.get("recorded_utc"),
            "capture_failure_note": CAPTURE_FAILURE_NOTE,
            "dirty_semantics_note": DIRTY_SEMANTICS_NOTE,
            "note": LAUNCH_RECORD_NOTE}


def read_retrieval_handoff(path):
    """The sibling control's own handoff -- one read, hashed and parsed together."""
    with open(str(path), "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:            # noqa: BLE001 -- a refusal
        raise ValueError(f"the retrieval handoff at {path!r} is not readable JSON: {error}. "
                         f"{RETRIEVAL_RECONCILIATION_NOTE}") from error
    for field in ("control_key", "status", "canonical"):
        if payload.get(field) is None:
            raise ValueError(f"the retrieval handoff at {path!r} does not state {field!r}; this "
                             f"report cannot reconcile §2 completeness against it. "
                             f"{RETRIEVAL_RECONCILIATION_NOTE}")
    return {"path": str(path), "sha256": digest,
            "control_key": str(payload["control_key"]),
            "status": str(payload["status"]),
            "canonical": bool(payload["canonical"]),
            "headline": payload.get("headline") or {},
            "report_json": payload.get("report_json"),
            "report_sha256": payload.get("report_sha256"),
            "created_utc": payload.get("created_utc"),
            "note": RETRIEVAL_RECONCILIATION_NOTE}


def _headline_phrase(headline, key, digits=3, unit=""):
    entry = (headline or {}).get(key) or {}
    if entry.get("point") is None:
        return None
    return (f"{key} {float(entry['point']):.{digits}f}{unit} "
            f"[{float(entry.get('ci_lo', float('nan'))):.{digits}f}, "
            f"{float(entry.get('ci_hi', float('nan'))):.{digits}f}]")


def reconcile_controls_elsewhere(handoff=None, base=None):
    """``CONTROLS_ELSEWHERE`` with the retrieval entry told the truth.

    With a handoff the entry names that control's status, canonicality, headline
    and report digest; without one it says plainly that this report was not given
    the sibling's status, instead of asserting a stale "run pending" (Codex r9t
    blocker 5).
    """
    controls = dict(base if base is not None else mr.CONTROLS_ELSEWHERE)
    key = "agree_oracle_retrieval_over_the_metadata_bank"
    original = controls.get(key, "")
    prefix = original.split(" -- ", 1)[0] if " -- " in original else key
    if handoff is None:
        controls[key] = (f"{prefix} -- STATUS NOT RECONCILED: this run was given no "
                         f"{RETRIEVAL_HANDOFF_FILENAME}, so it does not state whether §2's "
                         f"retrieval control has run. {RETRIEVAL_RECONCILIATION_NOTE}")
        return controls
    if str(handoff.get("control_key")) != key:
        raise ValueError(f"the handoff describes {handoff.get('control_key')!r}, not {key!r}; "
                         f"it is not this §2 control's handoff. "
                         f"{RETRIEVAL_RECONCILIATION_NOTE}")
    phrases = [phrase for phrase in
               (_headline_phrase(handoff["headline"], "median_e_loc", unit=" m"),
                _headline_phrase(handoff["headline"], "median_e_excess", unit=" m"),
                _headline_phrase(handoff["headline"], "success_raw@1.0"))
               if phrase]
    controls[key] = (
        f"{prefix} -- {handoff['status']}"
        f"{' (CANONICAL)' if handoff['canonical'] else ' (NOT canonical)'}"
        f", reported in {handoff.get('report_json')} "
        f"[{str(handoff.get('report_sha256'))[:12]}...]"
        + (f": {'; '.join(phrases)}" if phrases else "")
        + ". Its candidate set is the sparse metadata bank, never the dense grid, so its oracle "
          "floor is its own and its numbers are never comparable to this report's")
    return controls


#: how the observation bank is pre-registered, stated the way the metadata bank's is.
OBSERVATION_BANK_PREREGISTRATION_NOTE = (
    "compute the digest with `python -m src.localization.meshgrid_offgrid_probe "
    "--print-observation-digest --audit-report <G1> --context-manifest <D1> --dataset-root "
    "<root>`, commit the value, and pass it back as --expect-observation-bank-sha256 on every "
    "canonical run. It needs no run directory, no checkpoint and no GPU, so it can be -- and "
    "must be -- frozen before any localization quality has been read; that ordering is the whole "
    "argument, exactly as it is for the pair-metadata bank")

#: the intent record that survives a hard crash.
PUBLICATION_JOURNAL = "offgrid_publication_journal.json"

JOURNAL_NOTE = (
    "an in-process rollback cannot survive SIGKILL, a power loss, or an interrupt landing in the "
    "gap between a rename and the bookkeeping that records it (Codex r9i review, item 3). So "
    "every intended rename is journalled and fsynced BEFORE the first one runs, and the journal "
    "is marked complete only after all of them have landed and been re-verified. A journal found "
    "incomplete at startup means a previous attempt died mid-move: every final it names is "
    "moved back to quarantine before anything else happens, so the directory returns to the one "
    "state a partial publication may leave behind")

#: the publish contract, stated inside the artifact that depends on it.
PUBLICATION_ORDER_NOTE = (
    "the manifest is written and fsynced BEFORE any dump leaves quarantine, so a crash can "
    "never leave a finalized file no manifest names; the dumps are then moved with every rename "
    "inside a rollback handler, so a failure mid-move returns the whole set to quarantine; and "
    "the manifest is rewritten once publication is complete and every file has been re-verified "
    "against the digest it was staged with. publication.completed = false therefore means the "
    "dumps are in quarantine (or the process died between the move and the rewrite, which "
    "verify_published_probe resolves), and true means the published set is complete")


# --------------------------------------------------------------------------- #
# the binding gate
# --------------------------------------------------------------------------- #
def assert_probe_binding(run_dir, binding, fields=PROBE_BINDING_FIELDS):
    """The probe continues the SAME experiment as the run it reports against.

    The published binding is recomputed from its own content first (so a hand-
    edited file cannot vouch for itself), then compared field by field against
    the binding this probe built from the checkpoint, the scorer, the D1 manifest,
    the G1 report and the sampler settings it was actually given.
    """
    published, published_sha = mr.load_published_binding(run_dir)
    differing = {}
    for field in fields:
        if field not in binding:
            raise ValueError(f"the probe binding is missing the registered field {field!r}; "
                             "every quantity that decides a score must be pinned before a "
                             "control is generated")
        if published.get(field) != binding.get(field):
            differing[field] = {"published": published.get(field), "probe": binding.get(field)}
    if differing:
        raise ValueError(
            f"the probe does not run the protocol the published run was scored under: "
            f"{sorted(differing)} differ (published binding {published_sha[:12]}...). "
            f"First mismatch: {sorted(differing)[0]} = "
            f"{differing[sorted(differing)[0]]!r}. A control generated under a different "
            "checkpoint, scorer, context draw, candidate manifest or sampler setting cannot be "
            "compared against the run's scores")
    return {"binding_sha256": published_sha, "fields_checked": list(fields),
            "published": published,
            "published_checked": {field: published[field] for field in fields}}


def assert_probe_run_census(run_dir, binding, binding_sha256, plan, records,
                            context_manifest, totals=None, single_shard=False,
                            expect_ckpt_sha256=None, allow_protocol_deviation=False):
    """The run this control ranks against IS the complete, merged, registered pass.

    Every shard of a run shares the strict binding digest, so the binding alone
    cannot tell a finished 5,337-query merge from one shard of it -- and a rank
    "of 5,337 queries" taken against a partial directory would be a different
    claim than the one the report makes (Codex r9 review, finding 4). The full
    artifact ladder the R1 report applies is applied here too, reusing it rather
    than re-deriving a weaker copy: the supplied D1/G1/room manifests must be the
    bound ones, the binding must be the registered protocol, the directory must
    carry its merge report, every row and sidecar must re-verify, the census must
    hold and the D1/G1/row identities must be one set.
    """
    artifacts = mr.assert_artifact_hashes(binding, plan, context_manifest)
    registered = mr.assert_registered_protocol(binding, expect_ckpt_sha256=expect_ckpt_sha256,
                                               allow_deviation=allow_protocol_deviation)
    # the census does not discard what it verified: it keeps the sha256 of the
    # BYTES each row and sidecar had, and every later read is held to it, so a
    # coherent replacement between the census and the walk cannot pass by
    # recomputing its own self-digests (Codex r9n review)
    rows, _sims, snapshot = mr.verify_rows_with_sidecars(run_dir, binding_sha256)
    # the receipt is checked against the ROWS here too. r9d handed
    # assert_merge_report derived=None from this path, so the control accepted a
    # receipt no row supported and never looked at the batching stamps at all
    # (Codex r9f review, B4) -- the one ladder, applied the one way.
    derived = mr.derive_run_facts(rows)
    batching = mr.assert_uniform_batching(rows, binding.get("advisory"))
    merge = (None if single_shard
             else mr.assert_merge_report(run_dir, binding, binding_sha256, plan, totals=totals,
                                         derived=derived))
    census = mr.assert_census(rows, records, totals=totals)
    mr.assert_row_protocol(rows, binding)
    identity_join = mr.assert_identity_join(plan, records, rows)
    return {"artifacts": artifacts, "registered_protocol": registered, "merge": merge,
            "derived": derived, "batching": batching, "artifact_snapshot": snapshot,
            "artifact_snapshot_note": mr.ARTIFACT_SNAPSHOT_NOTE,
            "census": census, "identity_join": identity_join,
            "single_shard": bool(single_shard),
            "single_shard_note": mr.SINGLE_SHARD_NOTE if single_shard else None}


def assert_registered_probe_set(probes, plan):
    """The probe set IS ``one lexicographically first query per included room``.

    ``registered_probe_queries`` derives it from the manifests; this asserts the
    derived set covers every audited room exactly once, so a probe cannot quietly
    run on fifteen rooms.
    """
    rooms = sorted(plan.rooms)
    if sorted(probes) != rooms:
        missing = sorted(set(rooms) - set(probes))
        extra = sorted(set(probes) - set(rooms))
        raise ValueError(f"the off-grid probe set does not cover the audited rooms: missing "
                         f"{missing}, unexpected {extra}; §2 registers exactly one probe query "
                         "per included room")
    identities = [probes[room] for room in rooms]
    if len(set(identities)) != len(identities):
        raise ValueError("the off-grid probe set names the same query for two rooms")
    return identities


# --------------------------------------------------------------------------- #
# the generation at the continuous truth
# --------------------------------------------------------------------------- #
def assert_offgrid_noise_policy(policy):
    """Only common random numbers can key a draw at a non-candidate point."""
    if str(policy) != me.REGISTERED_NOISE_POLICY:
        raise ValueError(
            f"the off-grid truth probe needs the registered noise policy "
            f"{me.REGISTERED_NOISE_POLICY!r}, not {policy!r}: under common random numbers the "
            "truth generation is drawn from exactly the K latents every grid candidate of that "
            "query was drawn from, which is what makes its score comparable to theirs. A "
            "per-candidate key has no value for a point that is not a candidate")
    return True


def truth_noise(seed, query_id, num_samples, latent_shape, policy=me.NOISE_KEY_POLICY,
                device="cpu"):
    """The ``[K, C, T]`` latent noise the truth position is generated from."""
    assert_offgrid_noise_policy(policy)
    block = me.noise_block(seed, query_id, [OFFGRID_CANDIDATE_SENTINEL], int(num_samples),
                           latent_shape, policy=policy, device=device)
    return block


def generate_at_truth(engine, md, receiver_xyz, truth_xyz, *, query_id, seed=me.SEED,
                      num_samples=me.NUM_SAMPLES, noise_policy=me.NOISE_KEY_POLICY,
                      source_chunk=1, context=None):
    """Generate ``K`` RIRs at the continuous truth -> ``[K, 1, T]`` waveforms.

    The conditioning is assembled through the engine's own two branches, so the
    truth generation differs from a candidate generation in exactly one input --
    the source pose -- and in nothing else. The truth is taken from the caller
    (the pair metadata), never from ``md``, so the loader item may stay guarded.
    """
    assert_offgrid_noise_policy(noise_policy)
    receiver = np.asarray(receiver_xyz, dtype=np.float64).reshape(3)
    truth = np.asarray(truth_xyz, dtype=np.float64).reshape(3)
    if not (np.isfinite(receiver).all() and np.isfinite(truth).all()):
        raise ValueError(f"{query_id}: the receiver and the continuous truth must be finite")
    position_cam = (truth - receiver).reshape(1, 3)

    context = (me.context_conditioning(engine.conditioner, md, engine.device)
               if context is None else context)
    source = me.source_conditioning(engine.conditioner, {"depth": md["depth"]}, position_cam,
                                    engine.device, chunk=int(source_chunk))
    noise = truth_noise(seed, query_id, num_samples, engine.latent_shape,
                        policy=noise_policy, device=engine.device)
    rows = torch.zeros(int(num_samples), dtype=torch.long)
    merged = me.expand_conditioning(context, source, rows, engine.device)
    latents = engine.sampler(noise, engine.cond_inputs_fn(merged))
    return engine.decoder(latents).clamp(-1.0, 1.0)


#: where the released split's relpaths are rooted.
DEFAULT_DATASET_ROOT = "AcousticRooms"


def resolve_observation_path(dataset_root, record):
    """``<dataset_root>/<relpath>`` -- the file the released loader opens."""
    relpath = record.get("relpath") or record.get("path")
    if not relpath:
        raise ValueError(f"{record.get('query_id')!r}: the context manifest record carries no "
                         "relpath, so its observed RIR cannot be located")
    return os.path.join(str(dataset_root), str(relpath))


def digest_file_once(path):
    """``(sha256, n_bytes)`` from a SINGLE read -- the r9j item-1 pattern.

    Nothing is parsed out of a wav, but the discipline is the same one the pair
    metadata needed: whatever is later said about this file is said about the
    bytes that were actually read, not about a second read of the same name.
    """
    with open(str(path), "rb") as handle:
        raw = handle.read()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def compute_observation_bank_digest(audit_report, context_manifest,
                                    dataset_root=DEFAULT_DATASET_ROOT,
                                    require_manifest_census=True, branch=None):
    """The PRE-REGISTRATION entry point for the observed RIRs.

    Deterministic and run-free: the sixteen registered probe queries come from
    the G1 audit's own rule (``registered_probe_queries``), their observed RIRs
    are located through the D1 manifest's relpaths, and each is digested from a
    single read. No checkpoint, no scorer, no GPU and no ``eval_FLAC`` import is
    involved, so the value can be frozen before any localization quality exists
    -- which is the entire argument for it (Planner RULING 2).
    """
    plan = me.load_audit_plan(audit_report, branch=branch)
    manifest = mq.load_manifest(context_manifest, require_census=require_manifest_census)
    records = {str(record["query_id"]): record for record in manifest["records"]}

    probes = me.registered_probe_queries(plan)
    assert_registered_probe_set(probes, plan)

    queries, missing = {}, []
    for room_id in sorted(probes):
        query_id = probes[room_id]
        record = records.get(query_id)
        if record is None:
            missing.append(query_id)
            continue
        path = resolve_observation_path(dataset_root, record)
        if not os.path.isfile(path):
            missing.append(f"{query_id} -> {path}")
            continue
        digest, n_bytes = digest_file_once(path)
        queries[query_id] = {"room_id": room_id,
                             "relpath": str(record.get("relpath")),
                             "path": path,
                             "sha256": digest,
                             "n_bytes": int(n_bytes)}
    if missing:
        raise ValueError(
            f"{len(missing)} registered probe observation(s) could not be read (first "
            f"{missing[:3]}); the bank covers all {len(probes)} or it is not the registered bank")

    from src.localization.crossarm import canonical_sha256

    bank = canonical_sha256({query_id: [entry["relpath"], entry["sha256"]]
                             for query_id, entry in sorted(queries.items())})
    return {"observation_bank_sha256": bank,
            "n_queries": len(queries),
            "queries": queries,
            "dataset_root": str(dataset_root),
            "audit_report": str(audit_report),
            "audit_report_sha256": plan.report_sha256,
            "context_manifest": str(context_manifest),
            "context_manifest_sha256": me.file_sha256(context_manifest),
            "how_to_register": OBSERVATION_BANK_PREREGISTRATION_NOTE,
            "note": OBSERVATION_BINDING_NOTE}


def assert_observation_bank(found, expected=None, allow_unpinned=False):
    """The observations came out of the PRE-REGISTERED bank.

    The same shape, and the same refusal, as the pair-metadata bank: recording a
    digest and feeding it back later proves stability, not origin, so
    trust-on-first-use is not a canonical mode.
    """
    if expected and str(expected) != str(found):
        raise ValueError(
            f"the observed-RIR bank this control reads hashes to {str(found)[:16]}... but the "
            f"registered bank is {str(expected)[:16]}...; the observations behind every rank and "
            "every calibration cosine are not the registered ones")
    if not expected and not allow_unpinned:
        raise ValueError(
            "a canonical off-grid control requires the PRE-REGISTERED observed-RIR bank digest, "
            f"and none was supplied. The bank this run reads hashes to {str(found)}. "
            f"{OBSERVATION_BANK_PREREGISTRATION_NOTE}. Pass --non-canonical to run a diagnostic "
            "instead")
    return {"observation_bank_sha256": str(found), "pinned": bool(expected),
            "preregistration_note": OBSERVATION_BANK_PREREGISTRATION_NOTE,
            "note": OBSERVATION_BINDING_NOTE}


def decode_observation(raw, *, sample_rate, sample_size, force_channels="mono",
                       fmt="wav"):
    """Decode observation BYTES the way the released eval loader decodes the file.

    ``SampleDataset.load_file`` -> ``PadCrop_Normalized_T(randomize=False)`` ->
    the ``force_channels`` encoding, with ``augs`` off, which is what the eval
    dataset config pins. Reproduced here over a buffer rather than a path so the
    tensor that gets scored comes out of the bytes that were hashed; the result
    is asserted bit-equal to the loader's own tensor, so a divergence is a
    refusal rather than a silent second decode (Codex r9l review, item 2).
    """
    import torchaudio
    from src.data.utils import Mono, PadCrop_Normalized_T, PseudoStereo, Stereo

    audio, in_sr = torchaudio.load(io.BytesIO(raw), format=fmt)
    if int(in_sr) != int(sample_rate):
        from torchaudio import transforms as T

        audio = T.Resample(in_sr, int(sample_rate), lowpass_filter_width=128)(audio)
    chunk = PadCrop_Normalized_T(int(sample_size), int(sample_rate), randomize=False)(audio)[0]
    encoding = torch.nn.Sequential(
        Stereo() if force_channels == "stereo" else torch.nn.Identity(),
        PseudoStereo(sample_rate=int(sample_rate)) if force_channels == "pseudostereo"
        else torch.nn.Identity(),
        Mono() if force_channels == "mono" else torch.nn.Identity())
    return encoding(chunk)


def read_verified_observation(path, expected, *, sample_rate, sample_size,
                              force_channels="mono"):
    """ONE read: hash the bytes against the pin, decode the SAME buffer.

    r9j2 hashed the file AFTER the loader had already decoded it, so restoring
    the registered file between the two made the pin pass over bytes nothing was
    scored from (Codex r9l review, item 2). Here the digest and the tensor come
    out of one buffer, and there is no second open to restore anything into.
    """
    with open(str(path), "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != str(expected):
        raise ValueError(
            f"the observation at {path!r} hashes to {digest[:16]}... but the pre-registered "
            f"bank records {str(expected)[:16]}...; the bytes being decoded are not the "
            f"registered ones. {OBSERVATION_BINDING_NOTE}")
    tensor = decode_observation(raw, sample_rate=sample_rate, sample_size=sample_size,
                                force_channels=force_channels)
    return {"sha256": digest, "n_bytes": len(raw), "tensor": tensor, "path": str(path)}


def assert_decoded_observation_matches(query_id, decoded, loader_tensor):
    """The tensor decoded from the VERIFIED bytes is the loader's own tensor.

    If these agree, the two are the same object in every sense that matters and
    the verified one can be used everywhere below -- which is what makes the
    byte-to-tensor path single. If they ever disagree, this control's decode and
    the released loader's have diverged, and that is a refusal rather than a
    silent choice between two tensors.
    """
    decoded = torch.as_tensor(decoded).detach().cpu().float()
    loader = torch.as_tensor(loader_tensor).detach().cpu().float()
    if decoded.numel() != loader.numel():
        raise ValueError(
            f"{query_id}: the observation decoded from the verified bytes has "
            f"{decoded.numel()} samples but the loader handed over {loader.numel()}; the "
            "control's decode and the released loader's have diverged")
    if not torch.equal(decoded.reshape(-1), loader.reshape(-1)):
        drift = float((decoded.reshape(-1) - loader.reshape(-1)).abs().max())
        raise ValueError(
            f"{query_id}: the observation decoded from the verified bytes is not the tensor the "
            f"loader handed over (max |diff| {drift:.3g}); either the bytes on disk are not what "
            f"the loader read, or the two decodes disagree. {OBSERVATION_BINDING_NOTE}")
    return loader.reshape(loader_tensor.shape) if hasattr(loader_tensor, "shape") else loader


def assert_observation_source(query_id, loader_path, expected):
    """The file the LOADER opened is the file the bank digested.

    The bank is computed under ``--dataset-root``; the loader resolves its own
    path from the dataset config. A divergent root would let a pristine tree
    satisfy the frozen digest while the observations came from somewhere else --
    the r9i review found exactly that shape of hole in the retrieval control, and
    it is closed here by digesting what the loader actually read rather than
    trusting the two roots to agree.
    """
    if not loader_path:
        raise ValueError(f"{query_id}: the loader did not say which file it read, so its "
                         "observation cannot be joined to the pre-registered bank")
    if not os.path.isfile(str(loader_path)):
        raise ValueError(f"{query_id}: the loader names {loader_path!r}, which is not a file")
    found, n_bytes = digest_file_once(loader_path)
    if found != str(expected):
        raise ValueError(
            f"{query_id}: the observation the loader read from {loader_path!r} hashes to "
            f"{found[:16]}... but the pre-registered bank covers {str(expected)[:16]}...; the "
            f"bank and the loader are not reading the same bytes. {OBSERVATION_BINDING_NOTE}")
    return {"ok": True, "sha256": found, "n_bytes": int(n_bytes),
            "loader_path": str(loader_path)}


def observation_digests(obs_wav, source_path=None, source_sha256=None):
    """Record what the observation IS -- WITHOUT reopening the file.

    ``source_sha256`` is the digest the verified single read already produced.
    Recomputing it here would be a second open of the very file whose
    single-read property is the point (Codex r9l review, item 2), so the file is
    hashed here only when nothing verified it -- the unpinned, non-canonical
    path, where there is no single-read guarantee to preserve.
    """
    tensor = torch.as_tensor(obs_wav).detach().cpu().float().contiguous()
    out = {"tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
           "shape": [int(v) for v in tensor.shape],
           "source_path": None if source_path is None else str(source_path),
           "source_sha256": None if source_sha256 is None else str(source_sha256),
           "pinned": bool(source_sha256),
           "note": OBSERVATION_BINDING_NOTE}
    if source_sha256 is None and source_path and os.path.isfile(str(source_path)):
        out["source_sha256"] = me.file_sha256(str(source_path))
    return out


#: THE GATE'S EVIDENCE -- the matched path, and ONLY the matched path.
#:
#: Split from the retired path's numbers deliberately (Codex r9v residual 1).
#: r9u kept both in one dict and the report then sliced a "tie_evidence" block
#: that joined the retired 6.67e-3 / 8,064 to the matched 85.4x separation, and
#: the markdown attributed the new margin to the old pair count. Two dicts make
#: that mistake unavailable rather than merely discouraged: nothing the gate
#: publishes may read from :data:`RETIRED_PATH_EVIDENCE`.
#:
#: Measured 2026-08-29 by :mod:`src.localization.meshgrid_drift_measurement`
#: ``--matched-substitution`` against the merged P1 run on cuda:0 (663 s), log
#: ``loc_meshgrid_2026-08-29_01:51:30_r9u_matched_substitution.log``: each of the
#: sixteen registered probe queries replayed ONCE at its row's own stamped
#: batching, its generated embeddings cached, and every other in-scope query's
#: observation scored against them.
MATCHED_PATH_EVIDENCE = {
    "path": "matched_batching_whole_query_replay",
    "date": "2026-08-29",
    "artifact": "outputs_loc/exp22/r9r_drift_measurement/matched_substitution/"
                "matched_substitution_measurement.json",
    # the tracked mirror, so the evidence join survives a fresh checkout. Same
    # distributions; only the 85,376 raw pair records stay in outputs_loc
    "artifact_mirror": "worklog/worklog_yixun/exp_22_loc_meshgrid_claude/"
                       "r9r_drift_measurement/matched_substitution_measurement.json",
    "log": "worklog/worklog_yixun/exp_22_loc_meshgrid_claude/"
           "loc_meshgrid_2026-08-29_01:51:30_r9u_matched_substitution.log",
    # (1) the honest replay: what the gate expects to see every time
    "n_replayed_queries": 16,
    "n_replay_candidates": 11577,
    "max_abs_delta": 2.440810203552246e-4,
    "max_abs_aggregate_delta": 0.0,
    "float16_bit_exact": True,
    "n_cells_over_own_tolerance": 0,
    "wall_seconds": 663,
    # (2) the adversary, measured on THIS path
    "n_substitution_pairs": 85376,
    "n_donor_observations": 5337,
    "substitution_min": 2.0847943e-2,
    "substitution_median": 0.630453,
    "substitution_max": 1.465234,
    "substitution_same_room_min": 2.0847943e-2,
    "n_substitution_same_room_pairs": 5321,
    "substitution_same_receiver_min": 0.181648567,
    "n_substitution_same_receiver_pairs": 143,
    "substitution_cross_room_min": 0.09369415,
    "n_substitution_undetected": 0,
    # (3) and therefore
    "tolerance": 2.44140625e-4,
    "separation": 85.393173,
    "min_separation_required": 5.0,
}

#: THE RETIRED PATH's numbers -- superseded, never the gate's evidence.
#:
#: r9r measured the single-candidate, changed-batching regeneration: its drift
#: distribution is still the diagnostic's reference (that IS the path the
#: diagnostic runs), and its substitution distribution is kept only so the
#: record of what r9s wrongly quoted survives. Codex r9t blocker 1 is exactly
#: this: agreement between two paths on the RIGHT observation bounds nothing
#: about the WRONG one, so 6.67e-3 over 8,064 pairs never described the matched
#: gate. Labelled at every use.
#:
#: Artifact ``outputs_loc/exp22/r9r_drift_measurement/merged/drift_measurement.json``
#: (retired_path_label.json beside it), logs
#: ``loc_meshgrid_2026-08-28_23:49:00_r9r_drift_cuda{0,1}.log``. Sample: seed
#: 20260828, 4 queries per room over 16 rooms x 2 candidates, both devices.
RETIRED_PATH_EVIDENCE = {
    "path": "retired_changed_batching_single_candidate",
    "superseded_by": MATCHED_PATH_EVIDENCE["artifact"],
    "label": "SUPERSEDED for detection power (Codex r9t blocker 1); still the "
             "non-gating diagnostic's own reference distribution",
    "artifact": "outputs_loc/exp22/r9r_drift_measurement/merged/drift_measurement.json",
    "date": "2026-08-29",
    "selection_seed": 20260828,
    # the diagnostic's reference distribution -- this part is NOT superseded,
    # because the diagnostic still runs exactly this path
    "n_measurements": 256,
    "max_abs_delta": 3.6296844482421875e-3,
    "excess_over_half_ulp_max": 3.5076141357421875e-3,
    "q99_abs_delta": 2.93e-3,
    "median_abs_delta": 5.13e-4,
    "max_abs_aggregate_delta": 1.1619329452514648e-3,
    "n_aggregate_above_score_tolerance": 4,
    # the substitution distribution that never described the matched gate
    "n_substitution_pairs": 8064,
    "substitution_min": 6.668746471405029e-3,
    "substitution_median": 0.569884,
    "substitution_max": 1.409295,
    # r9r's verdict on the changed-batching bound: it could not be established,
    # which is what sent the gate to matched batching (r9s, RULING 3)
    "candidate_bound_at_safety_1_5": 5.3e-3,
    "separation_of_candidate_bound": 1.3,
    "bound_established": False,
}


def tie_evidence():
    """What the GATE rests on -- matched path only, artifact path included.

    A function rather than a slice of a bigger dict, so no future caller can
    reach past it into the retired numbers (Codex r9v residual 1).
    """
    return dict(MATCHED_PATH_EVIDENCE)


def retired_path_evidence():
    """The superseded numbers, under a label that says so."""
    return dict(RETIRED_PATH_EVIDENCE)


#: THE GATE (r9s, Planner RULING 3). Adopted on the r9r measurement's evidence.
#:
#: The tie no longer regenerates one candidate at a batch shape the run never
#: used. It replays the WHOLE QUERY at the row's own stamped batching --
#: ``ReceiverCache`` over the receiver's candidate union at the row's
#: ``source_chunk``, then the query's candidates through the engine's own
#: ``_score_one_query`` at the row's ``batch_rows`` -- and requires the result to
#: reproduce the frozen artifacts exactly, on two counts that are asserted
#: separately because they fail differently:
#:
#: * every per-sample cosine, against the float16 sidecar, within the sidecar's
#:   OWN half-ulp and nothing more. There is no drift term: at matched batching
#:   there is no drift to allow;
#: * the float32 log-mean-exp aggregate, against the row's published
#:   ``scores_hex``, at exactly 0.0. That comparison carries no quantization at
#:   all, so "exactly" is a statement the arithmetic can actually make.
#:
#: The evidence is r9r's, measured on this very run and published under
#: ``outputs_loc/exp22/r9r_drift_measurement/`` (mirrored into the experiment
#: folder). Replaying 16 whole queries at matched batching -- 11,577 candidates,
#: 92,616 generated waveforms, on both devices -- came back BIT-EXACT: the
#: float16 round-trip returned the stored sidecar element for element and the
#: float32 aggregate matched to exactly 0.0, with zero exceptions. The same
#: queries through the old single-candidate path moved per-sample cosines by up
#: to 3.63e-3 and aggregates by up to 1.16e-3 (past ``SCORE_TOLERANCE`` on 4 of
#: 256), which is why that path is now a diagnostic and not the gate.
#:
#: DETECTION POWER, measured ON THIS PATH (r9u, Codex r9t blocker 1). r9s quoted
#: a margin that r9r had measured against generations from the RETIRED
#: single-candidate path; agreement between two paths on the RIGHT observation
#: bounds nothing about the WRONG one, so the number was re-measured where the
#: gate lives. Each of the sixteen probe queries was replayed once at its row's
#: own batching, its generated embeddings cached, and every other in-scope
#: query's observation scored against them: 85,376 ordered pairs against 5,337
#: donor observations -- including 5,321 same-room and 143 SAME-RECEIVER ones,
#: which the sixteen-query set alone could never have supplied. The closest
#: adversary moves the gate's comparison by 0.020848 (same room; the closest
#: same-receiver one by 0.1816), against a most-permissive cell tolerance of
#: 2.44e-4: a **85.4x separation**, and every one of the 85,376 pairs is caught
#: ELEMENTWISE, the sparsest by 206 cells. The retired path's 6.67e-3 is kept
#: only as that diagnostic's own reference.
#: COST: the replay generates every candidate of each probe query, ~10 minutes
#: for the sixteen (the run's own throughput; Cafe's 5,295 candidates dominate).
#: It is stamped into the report so nobody has to rediscover it.
MATCHED_BATCHING_TIE = (
    "MATCHED-BATCHING TIE (r9s, Planner RULING 3). The live observation is tied to the frozen "
    "rows by replaying the WHOLE query at the row's own stamped batching -- the receiver's "
    "candidate union through the source branch at the row's source_chunk, then the query's "
    "candidates through the engine's own _score_one_query at the row's batch_rows -- and "
    "requiring the replay to reproduce the published artifacts exactly: every per-sample cosine "
    "within the float16 sidecar's own half-ulp (no drift term, because at matched batching there "
    "is no drift), and the float32 log-mean-exp aggregate equal to the row's scores_hex at "
    "exactly 0.0. Evidence: the r9r measurement replayed 16 whole queries this way on both "
    "devices -- 11,577 candidates, 92,616 waveforms -- and was bit-exact on every one, while the "
    "superseded single-candidate path at a batch shape the run never used moved cosines by up to "
    "3.63e-3 and aggregates past SCORE_TOLERANCE on 4 of 256 measurements. Detection power is "
    "measured ON THIS PATH (r9u): the sixteen replays' generated embeddings were cached and "
    "scored against every other in-scope query's observation -- 85,376 ordered pairs over 5,337 "
    "donors, 5,321 of them same-room and 143 same-receiver -- and the closest adversary moves "
    "the gate's comparison by 0.020848 against a 2.44e-4 tolerance, a 85.4x separation with "
    "every pair caught elementwise. r9s quoted 6.67e-3/27x from the RETIRED path's generations, "
    "which bounded nothing about this gate (Codex r9t blocker 1). Distributions: "
    "outputs_loc/exp22/r9r_drift_measurement/matched_substitution/ and the retired path's under "
    "../merged/ (mirrored to "
    "worklog/worklog_yixun/exp_22_loc_meshgrid_claude/r9r_drift_measurement/)")

#: what the gate costs, stamped so the operator is not surprised by it.
MATCHED_BATCHING_TIE_COST_NOTE = (
    "COST OF THE TIE: the matched-batching replay generates every candidate of each probe query "
    "at the row's own batch_rows, so it is not free -- about 10 minutes for the sixteen registered "
    "probe queries at the P1 run's own throughput, dominated by Cafe_idx_1's 5,295 candidates and "
    "Auditorium_idx_1's 3,722. That is the price of a gate whose expectation is bit-exactness "
    "rather than a tolerance; the superseded single-candidate check cost 8 generations per query "
    "and bought a bound that could not be established (r9r)")

#: the old path, kept because it is nearly free and now well characterized.
CHANGED_BATCHING_DIAGNOSTIC_NOTE = (
    "NON-GATING DIAGNOSTIC. The single-candidate regeneration at a CHANGED batch shape (one "
    "source position, num_samples generated rows) is still run and still published, because it "
    "costs 8 generations and its distribution is now characterized: over r9r's 256 measurements "
    "across all 16 rooms it ran to a median 5.13e-4, q99 2.93e-3 and max 3.63e-3 per-sample "
    "|delta|, with aggregate shifts up to 1.16e-3. It decides NOTHING -- a value inside that "
    "reference distribution is expected, and a value outside it is a signal to investigate the "
    "backbone's batch behaviour, not a reason to refuse an observation. The gate is the "
    "matched-batching replay above")

#: what the query's own cosine span is, and what it is NOT.
DYNAMIC_RANGE_NOTE = (
    "query_cosine_span is the spread of THIS query's own stored cosines and query_cosine_span_"
    "over_delta divides it by the measured drift. Both are DYNAMIC RANGE, not detection "
    "evidence: they say nothing about how far a substituted observation moves this number. r9p "
    "published the ratio as 'separation_vs_span' and read it as substitution evidence, which "
    "Codex r9q rejected (item 3). The measured detection margin is measured_substitution_min -- "
    "the smallest movement over the MATCHED path's 85,376 ordered substituted-observation pairs "
    "(r9u), which is the path this gate runs on. It is carried beside these two so the "
    "difference cannot be misread again, and it is NOT the retired path's 8,064-pair figure, "
    "which described a regeneration this gate no longer performs (Codex r9v residual 1)")

TIE_TOLERANCE_NOTE = MATCHED_BATCHING_TIE


def observation_continuity_tolerance(stored):
    """The ONLY admissible difference: the float16 sidecar's own half-ulp.

    No drift term. The replay runs at the row's own batching, where the r9r
    measurement found the computation bit-exact over 11,577 candidates, so the
    only thing left between the replay and the stored array is the rounding the
    sidecar applied when it was written. See :data:`MATCHED_BATCHING_TIE`.
    """
    return float(mr.float16_half_ulp(np.asarray(stored, dtype=np.float16)))


def tie_candidate_row(row, aggregator=mr.HEADLINE_AGGREGATOR):
    """Which candidate the tie lands on: the row's own headline prediction.

    Factored out so the r9r drift measurement selects the same candidate the
    gate does when it measures the gate's own case, rather than a second copy
    of the rule that could drift from it.
    """
    largest = str(max(int(k) for k in row["by_k"]))
    block = row["by_k"][largest]
    candidate_row = int(block["prediction_row"] if aggregator == "lme"
                        else block["mean_prediction_row"])
    return candidate_row, int(row["candidate_indices"][candidate_row]), int(largest)


def regenerate_tie_embeddings(engine, query, md, context, candidate_row, candidate_index, *,
                              seed=me.SEED, num_samples=me.NUM_SAMPLES,
                              noise_policy=me.NOISE_KEY_POLICY, source_chunk=1):
    """The tie's regeneration, as ``[1, K, D]`` embeddings -- ONE implementation.

    The gate calls this, and so does the r9r drift measurement
    (:mod:`src.localization.meshgrid_drift_measurement`), so the distribution
    the bound is derived from is measured through the code path the bound is
    then applied to. A second copy would let the two diverge silently, which is
    precisely the failure the measurement exists to close.

    Note the batch shapes, because they are the measured quantity: ONE position
    through the source branch (batch 1 whatever ``source_chunk`` says, since
    ``source_conditioning`` chunks the position list) and ``num_samples``
    generated rows through the DiT, the VAE and AGREE -- against production's
    16-position source calls and 256-row forwards.
    """
    num_samples = int(num_samples)
    coordinates = np.asarray(query.coordinates, dtype=np.float64)
    position_cam = (coordinates[candidate_row]
                    - np.asarray(query.receiver_xyz, dtype=np.float64)).reshape(1, 3)

    source = me.source_conditioning(engine.conditioner, {"depth": md["depth"]}, position_cam,
                                    engine.device, chunk=int(source_chunk))
    noise = me.noise_block(seed, query.query_id, [int(candidate_index)], num_samples,
                           engine.latent_shape, policy=noise_policy, device=engine.device)
    merged = me.expand_conditioning(context, source,
                                    torch.zeros(num_samples, dtype=torch.long), engine.device)
    wavs = engine.decoder(engine.sampler(noise, engine.cond_inputs_fn(merged))).clamp(-1.0, 1.0)
    return torch.as_tensor(engine.embedder(wavs)).float().reshape(1, num_samples, -1).cpu()


def row_batching(row):
    """The batch shapes THIS row was produced at -- fail-closed, never defaulted.

    The tie's whole premise is that it runs at the row's own batching, so a row
    that does not say what that was cannot be tied to. Guessing the registered
    defaults would silently reintroduce exactly the changed-batching comparison
    r9r retired.
    """
    batching = row.get("batching") or {}
    missing = [key for key in ("batch_rows", "source_chunk") if batching.get(key) is None]
    if missing:
        raise ValueError(
            f"the row for {row.get('query_id')!r} does not record {missing}; the matched-batching "
            f"tie replays the query at the shapes the row was PRODUCED at, and a row that does "
            f"not state them cannot be replayed at them. {MATCHED_BATCHING_TIE}")
    return int(batching["batch_rows"]), int(batching["source_chunk"])


def changed_batching_diagnostic(engine, query, md, context, row, sims, obs_embedding,
                                candidate_row, candidate_index, *, seed=me.SEED,
                                num_samples=me.NUM_SAMPLES, noise_policy=me.NOISE_KEY_POLICY,
                                source_chunk=1):
    """The retired gate, kept as a NON-GATING published diagnostic.

    Eight generations at a batch shape the run never used. It decides nothing --
    see :data:`CHANGED_BATCHING_DIAGNOSTIC_NOTE` -- but it is nearly free and
    r9r characterized its distribution over 256 measurements, so publishing it
    beside that reference is worth more than dropping it.
    """
    from src.localization.scoring import cosine_sims

    num_samples = int(num_samples)
    stored = np.asarray(sims, dtype=np.float16)[candidate_row, :num_samples]
    embeddings = regenerate_tie_embeddings(engine, query, md, context, candidate_row,
                                           candidate_index, seed=seed, num_samples=num_samples,
                                           noise_policy=noise_policy, source_chunk=source_chunk)
    rederived = cosine_sims(torch.as_tensor(obs_embedding).float().reshape(-1),
                            embeddings)[0].double().numpy()
    delta = float(np.abs(rederived - stored.astype(np.float64)).max())
    # the reference distribution is the RETIRED path's, which is correct here and
    # only here: this diagnostic IS that path. It is labelled so, and it is never
    # the gate's evidence (Codex r9v residual 1)
    retired = RETIRED_PATH_EVIDENCE
    reference = {"path": str(retired["path"]),
                 "label": str(retired["label"]),
                 "n_measurements": int(retired["n_measurements"]),
                 "median": float(retired["median_abs_delta"]),
                 "q99": float(retired["q99_abs_delta"]),
                 "max": float(retired["max_abs_delta"]),
                 "artifact": str(retired["artifact"])}
    return {"gating": False,
            "max_abs_delta": delta,
            "inside_reference_distribution": bool(delta <= reference["max"]),
            "reference_distribution": reference,
            "candidate_row": int(candidate_row), "candidate_index": int(candidate_index),
            "num_samples": num_samples,
            "stored": [float(v) for v in stored],
            "rederived": [float(v) for v in rederived],
            "note": CHANGED_BATCHING_DIAGNOSTIC_NOTE}


def assert_observation_continuity(engine, query, md, context, row, sims, obs_embedding, *,
                                  receiver_id, union, positions_cam,
                                  seed=me.SEED, num_samples=me.NUM_SAMPLES,
                                  noise_policy=me.NOISE_KEY_POLICY, source_chunk=1,
                                  tau=me.TAU, aggregator=mr.HEADLINE_AGGREGATOR,
                                  diagnostic=True):
    """The live observation IS the one the frozen rows were scored against.

    The WHOLE query is replayed at the row's own stamped batching, through the
    engine's own production functions, and must reproduce the frozen artifacts
    exactly: every per-sample cosine inside the float16 sidecar's own half-ulp,
    and the float32 aggregate equal to the row's published score at 0.0. See
    :data:`MATCHED_BATCHING_TIE` for the evidence this expectation rests on,
    :data:`OBSERVATION_BINDING_NOTE` for why the observation is bound this way
    at all, and :data:`MATCHED_BATCHING_TIE_COST_NOTE` for what it costs.

    ``source_chunk`` here is the DIAGNOSTIC's chunk, not the gate's: the gate
    takes both batch shapes from the row.
    """
    # imported inside the call because the measurement module imports this one
    # at module level; the replay machinery lives there, is TDD-covered there,
    # and is not copied here -- a second implementation of "the production
    # path" would stop being the production path (r9r)
    from src.localization import meshgrid_drift_measurement as dm

    candidate_row, candidate_index, largest = tie_candidate_row(row, aggregator=aggregator)
    num_samples = int(num_samples)
    batch_rows, row_source_chunk = row_batching(row)

    summary, per_candidate, deltas = dm.measure_matched_query(
        engine, query, md, context, receiver_id, union, positions_cam, row, sims, obs_embedding,
        seed=seed, num_samples=num_samples, noise_policy=noise_policy, batch_rows=batch_rows,
        source_chunk=row_source_chunk, tau=tau, candidate_rows=(candidate_row,))

    stored = np.asarray(sims, dtype=np.float16)[candidate_row, :num_samples]
    headline = per_candidate[0]
    deltas = np.asarray(deltas, dtype=np.float64)
    # PER ELEMENT (Codex r9t blocker 2). r9s compared the query-wide maximum
    # delta against a query-wide maximum half-ulp, so a low-magnitude cell --
    # whose own float16 gap can be a thousand times smaller -- could cross its
    # cell and still pass under some other cell's larger bound. Every cosine is
    # now held to the gap of the cell it is compared against.
    cell_tolerance = dm.cell_half_ulp(sims)
    if cell_tolerance.shape != deltas.shape:
        raise ValueError(f"{query.query_id}: the replay is {deltas.shape} and the sidecar's "
                         f"per-cell bounds are {cell_tolerance.shape}; they are not the same "
                         "array and the gate would compare the wrong cells")
    violations = deltas > cell_tolerance
    n_violations = int(violations.sum())
    delta = float(deltas.max())
    worst = np.unravel_index(int(np.argmax(deltas)), deltas.shape)
    # the bound reported beside the worst delta is that CELL's bound, not the
    # array's maximum, so "max_abs_delta <= tolerance" reads about one cell
    tolerance = float(cell_tolerance[worst])
    headroom = float((cell_tolerance - deltas).min())
    if float(cell_tolerance.max()) != float(summary["sidecar_half_ulp"]):
        raise ValueError(f"{query.query_id}: the tie's per-cell bounds top out at "
                         f"{float(cell_tolerance.max())!r} but the replay's sidecar half-ulp is "
                         f"{summary['sidecar_half_ulp']!r}; they are the same quantity and a "
                         "divergence means one of them is not this query's sidecar")
    aggregate_delta = float(summary["aggregate"]["max_abs_delta"])
    # three criteria, decided separately because they fail differently: the
    # per-cell one sees float16 rounding, the float16 round-trip sees a value
    # that no longer rounds to the stored cell at all -- which r9s recorded but
    # did not gate on (blocker 2) -- and the aggregate comparison is float32
    # against float32, with no rounding to hide in
    within = bool(n_violations == 0)
    bit_exact = bool(int(summary["n_float16_mismatch"]) == 0)
    exact = bool(aggregate_delta == 0.0)

    # the query's own cosine span is DYNAMIC RANGE, not detection evidence: it
    # says how far this query's cosines spread, which is not how far a
    # substituted observation moves them. r9p divided one by the other and
    # called the ratio "separation" (Codex r9q, item 3). It is still published,
    # because a reader wants the scale, but it is labelled for what it is and
    # the real detection margin comes from the r9r measurement
    whole = np.asarray(sims, dtype=np.float64)
    span = float(whole.max() - whole.min())
    verdict = {"ok": bool(within and bit_exact and exact),
               "within_tolerance": within,
               "float16_round_trip_exact": bit_exact,
               "aggregate_exact": exact,
               "refused": bool(not (within and bit_exact and exact)),
               "gate": "matched_batching_replay",
               "per_element_gate": True,
               "max_abs_delta": delta,
               "tolerance": tolerance,
               "tolerance_max": float(cell_tolerance.max()),
               "tolerance_min": float(cell_tolerance.min()),
               "headroom": headroom,
               "n_elements": int(deltas.size),
               "n_violations": n_violations,
               "worst_element": {"candidate_row": int(worst[0]), "sample": int(worst[1]),
                                 "delta": delta, "tolerance": tolerance,
                                 "stored": float(np.asarray(sims, dtype=np.float16)[worst])},
               "aggregate_max_abs_delta": aggregate_delta,
               "float16_bit_exact": bool(summary["float16_bit_exact"]),
               "n_float16_mismatch": int(summary["n_float16_mismatch"]),
               "n_candidates_replayed": int(summary["n_candidates"]),
               "n_union": int(summary["n_union"]),
               "batch_rows": batch_rows, "source_chunk": row_source_chunk,
               "per_sample_quantiles": dict(summary["quantiles"]),
               "n_above_half_ulp": int(summary["n_above_half_ulp"]),
               "aggregate_quantiles": dict(summary["aggregate"]["quantiles"]),
               "query_cosine_span": span,
               "query_cosine_span_over_delta": (float(span / delta) if delta > 0
                                                else float("inf")),
               "dynamic_range_note": DYNAMIC_RANGE_NOTE,
               "measured_substitution_min": float(MATCHED_PATH_EVIDENCE["substitution_min"]),
               "measured_separation": (
                   float(MATCHED_PATH_EVIDENCE["substitution_min"]
                         / float(cell_tolerance.max())) if cell_tolerance.max() > 0
                   else float("inf")),
               "k": int(largest),
               "candidate_index": candidate_index, "candidate_row": candidate_row,
               "num_samples": num_samples,
               "stored": [float(v) for v in stored],
               "rederived": list(headline["rederived"]),
               "cost_note": MATCHED_BATCHING_TIE_COST_NOTE,
               "tolerance_note": TIE_TOLERANCE_NOTE,
               "note": OBSERVATION_BINDING_NOTE}
    if diagnostic:
        verdict["changed_batching_diagnostic"] = changed_batching_diagnostic(
            engine, query, md, context, row, sims, obs_embedding, candidate_row, candidate_index,
            seed=seed, num_samples=num_samples, noise_policy=noise_policy,
            source_chunk=source_chunk)
    if not verdict["ok"]:
        failed = ", ".join(
            reason for reason, broken in (
                (f"{n_violations} of {deltas.size} cosines are outside their OWN cell's float16 "
                 "half-ulp", not within),
                (f"{summary['n_float16_mismatch']} cosines no longer round to the float16 cell "
                 "the sidecar stores", not bit_exact),
                ("the float32 aggregate score differs from the row's published value",
                 not exact)) if broken)
        raise ValueError(
            f"{query.query_id}: replayed at the row's OWN batching (batch_rows={batch_rows}, "
            f"source_chunk={row_source_chunk}) over all {summary['n_candidates']} candidates, "
            f"this observation does not reproduce what the frozen row published -- {failed}. "
            f"Worst cell: |delta| {delta:.3g} against its own bound {tolerance:.3g} (row "
            f"{int(worst[0])}, sample {int(worst[1])}); max aggregate |delta| "
            f"{aggregate_delta:.3g} against an exact 0. At matched batching the computation is "
            f"bit-exact (r9r: 11,577 candidates, zero exceptions), so this is not numerical "
            f"noise. The closest measured substituted observation moves the tie by "
            f"{MATCHED_PATH_EVIDENCE['substitution_min']:.3g} on this same path "
            f"({MATCHED_PATH_EVIDENCE['n_substitution_pairs']:,} ordered pairs). "
            f"s[x, k] = cos(E(h_obs), E(h_hat)), so the observation being scored here "
            f"is not the observation those rows were scored against. {OBSERVATION_BINDING_NOTE}")
    return verdict


def truth_scores(embedder, obs_embedding, waveforms, tau=me.TAU, prefixes=me.K_PREFIXES):
    """``(sims [1, K], {K: {lme, mean}})`` for the truth generations."""
    embeddings = embedder(waveforms)
    embeddings = torch.as_tensor(embeddings).float().reshape(1, int(waveforms.shape[0]), -1)
    sims = sc.cosine_sims(torch.as_tensor(obs_embedding).float().reshape(-1), embeddings)
    blocks = me.nested_scores(sims, tau=tau, prefixes=prefixes)
    return sims, {int(k): {"lme": float(block["scores"][0]),
                           "mean": float(block["mean_scores"][0])}
                  for k, block in blocks.items()}


def rank_against_grid(row, scores_by_k, aggregator=mr.HEADLINE_AGGREGATOR):
    """Where the truth's score sits among the query's GRID candidate scores.

    The grid scores are the row's own float32 ``scores_hex`` -- the published
    numbers, not a recomputation -- so the rank is against exactly what the
    engine ranked. ``rank = 1`` means the truth would have beaten every candidate.
    """
    key = "scores_hex" if aggregator == "lme" else "mean_scores_hex"
    out = {}
    for k, block in sorted(row["by_k"].items(), key=lambda item: int(item[0])):
        k = int(k)
        if k not in scores_by_k:
            continue
        grid = decode_scores(block[key]).double()
        score = float(scores_by_k[k][aggregator])
        n_better = int((grid > score).sum())
        n_tied = int((grid == score).sum())
        best = float(grid.max())
        out[k] = {
            "truth_score": score,
            "rank": n_better + 1,
            "n_candidates": int(grid.numel()),
            "n_grid_better": n_better,
            "n_grid_tied": n_tied,
            "percentile": float((grid < score).double().mean()),
            "best_grid_score": best,
            "truth_minus_best_grid": score - best,
            "grid_prediction_index": int(block["prediction_index"] if aggregator == "lme"
                                         else block["mean_prediction_index"]),
        }
    return out


def calibration_record(embedder, obs_embedding, context_audio, generated_waveforms):
    """The two cosine distributions of the §2 real-vs-generated calibration."""
    obs = torch.as_tensor(obs_embedding).float().reshape(-1)
    real_wavs = torch.as_tensor(context_audio).float()
    if real_wavs.ndim == 2:                                   # [N, T] -> [N, 1, T]
        real_wavs = real_wavs.unsqueeze(1)
    if real_wavs.ndim != 3 or real_wavs.shape[1] != 1:
        raise ValueError(f"the real context bank must be [N, 1, T], got "
                         f"{tuple(real_wavs.shape)}")
    real_emb = torch.as_tensor(embedder(real_wavs)).float()
    real = sc.cosine_sims(obs, real_emb.reshape(1, real_emb.shape[0], -1))[0]

    gen_emb = torch.as_tensor(embedder(generated_waveforms)).float()
    generated = sc.cosine_sims(obs, gen_emb.reshape(1, gen_emb.shape[0], -1))[0]
    real_summary, generated_summary = _distribution(real), _distribution(generated)
    return {"label": CALIBRATION_LABEL,
            "real": [float(v) for v in real],
            "generated": [float(v) for v in generated],
            "real_summary": real_summary,
            "generated_summary": generated_summary,
            # taken from the float64 summaries, so the gap is exactly the
            # difference of the two means the report publishes
            "gap_mean_real_minus_generated": float(real_summary["mean"]
                                                   - generated_summary["mean"])}


def _distribution(values):
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        raise ValueError("a calibration distribution must be non-empty")
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "min": float(array.min()), "median": float(np.median(array)),
            "max": float(array.max())}


# --------------------------------------------------------------------------- #
# artifacts (announcement 08)
# --------------------------------------------------------------------------- #
#: where a query's dump waits until the whole control has succeeded.
WAVEFORM_STAGING_DIRNAME = os.path.join(WAVEFORM_DIRNAME, ".partial")

WAVEFORM_NOTE = ("off-grid truth generations [K, 1, T], the observation and the real context "
                 "bank they are calibrated against (announcement 08 exp_22 exemption: the "
                 "sixteen registered probe queries)")

#: what a dump says about its own standing when the caller did not say.
CANONICAL_STATUS_UNKNOWN = ("status not declared by the caller; consult the probe report's "
                            "canonical_status")
CANONICAL_STATUS_CANONICAL = "CANONICAL: every registered gate of the run and this control passed"
CANONICAL_STATUS_NON_CANONICAL = (
    "NON-CANONICAL: a gate was relaxed or unmet (see canonical_status in the probe report); "
    "these generations are a diagnostic and may not be quoted as the registered result")


def write_probe_waveforms(out_dir, room_id, position, waveforms, observation, context_audio,
                          truth_xyz, receiver_xyz, query_id=None, status_label=None,
                          continuity=None):
    """One probe query's dump, STAGED with its digest and its own labels.

    Staged, not published: a dump finalized the moment its query finished would
    survive a later failure as an unmanifested file holding generations at the
    ground truth (Codex r9 review, finding 9). :func:`publish_probe_waveforms`
    moves the whole set into place only after all sixteen have succeeded and the
    manifest exists; anything left behind stays under ``waveforms/.partial/``,
    quarantined and obviously incomplete.

    The labels travel INSIDE the npz as well, because a waveform file read on its
    own -- which is exactly how a dump gets used -- would otherwise carry
    generations at the held-out truth with nothing saying so.
    """
    staging = os.path.join(str(out_dir), WAVEFORM_STAGING_DIRNAME)
    os.makedirs(staging, exist_ok=True)
    name = f"offgrid_{me.room_stem(room_id)}_q{int(position):05d}.npz"
    path = os.path.join(staging, name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle,
                 waveforms=np.asarray(torch.as_tensor(waveforms).detach().cpu().numpy(),
                                      dtype=np.float32),
                 observation=np.asarray(torch.as_tensor(observation).detach().cpu()
                                        .reshape(-1).numpy(), dtype=np.float32),
                 context_audio=np.asarray(torch.as_tensor(context_audio).detach().cpu()
                                          .numpy(), dtype=np.float32),
                 truth_xyz=np.asarray(truth_xyz, dtype=np.float64).reshape(3),
                 receiver_xyz=np.asarray(receiver_xyz, dtype=np.float64).reshape(3),
                 query_id=np.array(str(query_id or "")),
                 room_id=np.array(str(room_id)),
                 control_label=np.array(CONTROL_LABEL),
                 calibration_label=np.array(CALIBRATION_LABEL),
                 subset=np.array(mr.SUBSET_LABEL),
                 agree_leakage_caveat=np.array(me.AGREE_LEAKAGE_CAVEAT),
                 scorer_readout_deviation=np.array(me.SCORER_READOUT_DEVIATION),
                 # a dump gets read on its own, so it carries the same
                 # disclosures the JSON does (Codex r9c review, disclosure minor)
                 latency_scope_note=np.array(mr.LATENCY_SCOPE_NOTE),
                 truth_binding_note=np.array(mr.TRUTH_BINDING_NOTE),
                 controls_elsewhere=np.array(json.dumps(mr.CONTROLS_ELSEWHERE,
                                                        sort_keys=True)),
                 sensitivity_status=np.array(str(status_label or CANONICAL_STATUS_UNKNOWN)),
                 # the tie's measured numbers, so a dump read on its own carries
                 # the gate's verdict AND its separation (Codex r9p)
                 tie_max_abs_delta=np.array(float((continuity or {}).get("max_abs_delta",
                                                                         float("nan")))),
                 tie_tolerance=np.array(float((continuity or {}).get("tolerance",
                                                                     float("nan")))),
                 tie_within_tolerance=np.array(bool((continuity or {}).get("within_tolerance",
                                                                           False))),
                 tie_refused=np.array(bool((continuity or {}).get("refused", False))),
                 tie_headroom=np.array(float((continuity or {}).get("headroom",
                                                                     float("nan")))),
                 # dynamic range, labelled as such -- NOT detection evidence
                 tie_query_cosine_span=np.array(float((continuity or {}).get(
                     "query_cosine_span", float("nan")))),
                 tie_query_cosine_span_over_delta=np.array(float((continuity or {}).get(
                     "query_cosine_span_over_delta", float("nan")))),
                 tie_dynamic_range_note=np.array(DYNAMIC_RANGE_NOTE),
                 # the MEASURED detection margin and where the bound stands
                 tie_measured_substitution_min=np.array(
                     float(MATCHED_PATH_EVIDENCE["substitution_min"])),
                 # the count the margin belongs to, so a dump read alone cannot
                 # attribute it to the retired path's 8,064 (Codex r9v residual 1)
                 tie_measured_substitution_pairs=np.array(
                     int(MATCHED_PATH_EVIDENCE["n_substitution_pairs"])),
                 tie_measured_substitution_path=np.array(str(MATCHED_PATH_EVIDENCE["path"])),
                 tie_evidence_artifact=np.array(str(MATCHED_PATH_EVIDENCE["artifact"])),
                 tie_retired_path_label=np.array(str(RETIRED_PATH_EVIDENCE["label"])),
                 tie_measured_separation=np.array(float((continuity or {}).get(
                     "measured_separation", float("nan")))),
                 tie_gate=np.array("matched_batching_replay"),
                 tie_aggregate_max_abs_delta=np.array(float((continuity or {}).get(
                     "aggregate_max_abs_delta", float("nan")))),
                 tie_float16_bit_exact=np.array(bool((continuity or {}).get("float16_bit_exact",
                                                                            False))),
                 tie_cost_note=np.array(MATCHED_BATCHING_TIE_COST_NOTE),
                 tie_tolerance_note=np.array(TIE_TOLERANCE_NOTE),
                 waveform_note=np.array(WAVEFORM_NOTE))
    os.replace(tmp, path)
    return {"waveform_path": os.path.join(WAVEFORM_DIRNAME, name),
            "sensitivity_status": str(status_label or CANONICAL_STATUS_UNKNOWN),
            "waveform_staged_path": os.path.relpath(path, str(out_dir)),
            "waveform_sha256": me.file_sha256(path),
            "waveform_published": False,
            "waveform_note": WAVEFORM_NOTE}


def _fsync_dir(path):
    """Flush a directory entry, so a rename survives a crash. Best effort."""
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(handle)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


def _rollback_published(moved):
    """Put every already-moved dump back in quarantine, and make it durable.

    A half-moved set is the one state the publish contract forbids: either the
    quarantine holds everything, or the published set is complete (Codex r9c
    review, M9). Rolling a rename back is another rename, so the recovery is as
    reliable as the move was -- and each reversal is followed by a directory
    fsync, so a crash during the recovery cannot leave the reversal itself
    half-durable (Codex r9f review, M9).
    """
    for target, source in reversed(moved):
        try:
            os.makedirs(os.path.dirname(source), exist_ok=True)
            os.replace(target, source)
            _fsync_dir(os.path.dirname(source))
            _fsync_dir(os.path.dirname(target))
        except OSError:                       # noqa: PERF203 -- best effort by design
            pass
    return len(moved)


def publish_probe_waveforms(out_dir, records):
    """Move every staged dump into place -- only once the control is complete.

    The digest was taken at staging time and the bytes do not change, so the
    published file's sha256 is the one the manifest already records. Any failure
    rolls every completed move back into quarantine, so the directory is never
    left holding a partial published set.
    """
    # the crash-safe half: an fsynced statement of every rename about to happen,
    # written before the first one (Codex r9i review, item 3)
    write_publication_journal(out_dir, records)
    published, moved = [], []
    # EVERY step of the loop is inside the handler, the rename included. r9d left
    # os.replace outside it, so a rename that failed part-way through -- a full
    # disk, a permission change, a cross-device target -- kept the already-moved
    # subset published (Codex r9f review, M9). The journal covers what no handler
    # can: the process not reaching the handler at all.
    try:
        for record in records:
            source = os.path.join(str(out_dir), record["waveform_staged_path"])
            target = os.path.join(str(out_dir), record["waveform_path"])
            if not os.path.isfile(source):
                raise ValueError(f"{record['query_id']}: the staged dump {source!r} is gone; "
                                 "the control may not publish a manifest naming a file it "
                                 "cannot move")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(source, target)
            moved.append((target, source))
            _fsync_dir(os.path.dirname(target))
            if me.file_sha256(target) != record["waveform_sha256"]:
                raise ValueError(f"{record['query_id']}: the published dump does not match the "
                                 "digest recorded at staging time")
            record["waveform_published"] = True
            published.append(record["waveform_path"])
        complete_publication_journal(out_dir, len(published))
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-loop must not be
        # the one way to leave a partial published set behind
        for record in records:
            record["waveform_published"] = False
        _rollback_published(moved)
        raise
    staging = os.path.join(str(out_dir), WAVEFORM_STAGING_DIRNAME)
    if os.path.isdir(staging) and not os.listdir(staging):
        os.rmdir(staging)
    return published


def journal_path(out_dir):
    return os.path.join(str(out_dir), PUBLICATION_JOURNAL)


def write_publication_journal(out_dir, records):
    """State every intended rename, durably, BEFORE the first one happens."""
    from datetime import datetime, timezone

    payload = {
        "experiment": "exp_22 loc_meshgrid off-grid publication journal",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": False,
        "note": JOURNAL_NOTE,
        "moves": [{"query_id": record["query_id"],
                   "final": record["waveform_path"],
                   "staged": record["waveform_staged_path"],
                   "sha256": record["waveform_sha256"]}
                  for record in records],
    }
    return _write_json_durably(journal_path(out_dir), payload)


def complete_publication_journal(out_dir, n_published):
    """Mark the journal complete -- only after every move landed and verified."""
    path = journal_path(out_dir)
    with open(path) as handle:
        payload = json.load(handle)
    payload.update({"completed": True, "n_published": int(n_published)})
    return _write_json_durably(path, payload)


def recover_publication(out_dir):
    """Quarantine the finals of an interrupted publication, before anything else.

    The one state a partial publication may leave behind is "everything is in
    quarantine". A journal that exists and is not complete says a previous
    attempt died between the first rename and the last verification, so every
    final it names is moved back to its staged path -- and only then may a caller
    proceed. A complete journal, or none at all, is a no-op.
    """
    path = journal_path(out_dir)
    if not os.path.isfile(path):
        return {"recovered": False, "reason": "no journal", "n_quarantined": 0}
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("completed"):
        return {"recovered": False, "reason": "the journal is complete", "n_quarantined": 0}

    quarantined, missing = [], []
    for move in payload.get("moves") or []:
        final = os.path.join(str(out_dir), move["final"])
        staged = os.path.join(str(out_dir), move["staged"])
        if not os.path.isfile(final):
            if not os.path.isfile(staged):
                missing.append(move["final"])
            continue
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        os.replace(final, staged)
        _fsync_dir(os.path.dirname(staged))
        _fsync_dir(os.path.dirname(final))
        quarantined.append(move["final"])
    os.remove(path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)))
    return {"recovered": True, "reason": "an interrupted publication was rolled back",
            "n_quarantined": len(quarantined), "quarantined": quarantined,
            "missing": missing, "note": JOURNAL_NOTE}


#: what the publication path needs out of the verified gate. r9d hand-listed a
#: subset here and dropped ``metadata_bank_expected``, so the JSON and the
#: Markdown always said non-canonical while the NPZ labels said canonical
#: (Codex r9f review). One definition now, and a test pins that it carries
#: everything ``probe_canonical_status`` reads.
PUBLICATION_GATE_FIELDS = ("census", "identity_join", "merge", "derived", "batching",
                           "artifact_snapshot_note",
                           "single_shard", "single_shard_note", "registered_protocol",
                           "metadata_bank", "metadata_bank_sha256", "metadata_bank_expected",
                           "observation_continuity", "observation_bank",
                           "observation_bank_sha256", "observation_bank_expected",
                           "launch_record", "retrieval_handoff",
                           "non_canonical", "non_canonical_declared")


def publication_gate(gate):
    """The verified gate, sliced for publication without losing a verdict."""
    return {field: gate[field] for field in PUBLICATION_GATE_FIELDS if field in gate}


def build_observation_decoder(model_config, dataset_config_path):
    """A ``(path, expected_sha256) -> verified observation`` bound to the configs.

    The sample rate, the sample size and the channel policy come from the very
    model and dataset configs the released loader is built from -- both of which
    the run binding already pins -- so the decode this control performs is the
    decode the run performed, and :func:`assert_decoded_observation_matches`
    refuses if that ever stops being true.
    """
    with open(str(dataset_config_path)) as handle:
        dataset_config = json.load(handle)
    force_channels = ("mono" if int(model_config.get("audio_channels", 1)) == 1
                      else dataset_config.get("force_channels", "stereo"))

    def _decode(path, expected):
        return read_verified_observation(path, expected,
                                         sample_rate=int(model_config["sample_rate"]),
                                         sample_size=int(model_config["sample_size"]),
                                         force_channels=force_channels)
    return _decode


def probe_canonical_status(gate):
    """Whether this control may be quoted as the registered off-grid result.

    Mirrors ``meshgrid_report.canonical_status``: one authority, read by the
    JSON, the Markdown and the embedded NPZ label alike.
    """
    gate = gate or {}
    reasons = []
    if gate.get("single_shard"):
        reasons.append({"gate": "merge_report",
                        "why": "the run this control ranks against publishes no census-gated "
                               "merge receipt",
                        "note": mr.SINGLE_SHARD_NOTE})
    registered = gate.get("registered_protocol") or {}
    if registered and not registered.get("is_registered", True):
        reasons.append({"gate": "registered_protocol",
                        "why": f"the run binding deviates on "
                               f"{sorted(registered.get('deviations') or {})}",
                        "note": mr.CKPT_SHA256_NOTE})
    if not gate.get("metadata_bank_expected"):
        reasons.append({"gate": "metadata_bank",
                        "why": "no pre-registered pair-metadata bank digest was supplied",
                        "note": mr.METADATA_BANK_PREREGISTRATION_NOTE})
    if not gate.get("observation_bank_expected"):
        reasons.append({"gate": "observation_bank",
                        "why": "no pre-registered observed-RIR bank digest was supplied",
                        "note": OBSERVATION_BANK_PREREGISTRATION_NOTE})
    # FAIL-CLOSED PRESENCE (Codex r9t blocker 3). r9s refused only when the
    # record was present AND said ok is False, so an absent, empty or partial
    # continuity result canonicalized -- the one shape a broken run is most
    # likely to produce. A canonical control now requires a COMPLETE record: one
    # tie verdict per probe query, all of them ok.
    continuity = gate.get("observation_continuity")
    if not isinstance(continuity, dict) or not continuity:
        reasons.append({"gate": "observation_continuity",
                        "why": "no observation-continuity record was published, so nothing ties "
                               "the loaded observation to the frozen rows this control ranks "
                               "against",
                        "note": OBSERVATION_BINDING_NOTE})
    elif not continuity.get("ok", False):
        reasons.append({"gate": "observation_continuity",
                        "why": continuity.get(
                            "why", "the live observation could not be tied to the frozen rows"),
                        "note": OBSERVATION_BINDING_NOTE})
    else:
        checked = int(continuity.get("checked") or 0)
        expected = int(continuity.get("n_expected") or 0)
        if not expected or checked != expected:
            reasons.append({"gate": "observation_continuity",
                            "why": f"the continuity record covers {checked} of "
                                   f"{expected or 'an unstated number of'} probe queries; a "
                                   "partial tie leaves the uncovered queries' observations "
                                   "unbound",
                            "note": OBSERVATION_BINDING_NOTE})
        elif len(continuity.get("per_query_delta") or {}) != expected:
            reasons.append({"gate": "observation_continuity",
                            "why": f"the continuity record claims {expected} queries but "
                                   f"publishes {len(continuity.get('per_query_delta') or {})} "
                                   "per-query deltas; the record is not self-consistent",
                            "note": OBSERVATION_BINDING_NOTE})
    # r9r: the tie PASSING is not enough while what it passes against is a
    # provisional number the measurement declined to confirm. A control whose
    # gate has no established bound may not be quoted as the registered result,
    # so the status says so rather than leaving the reader to know it
    # The operator's own declaration. r9d propagated it and r9g put it in the
    # publication slice, but nothing READ it, so a valid pin plus --non-canonical
    # produced canonical JSON/Markdown beside non-canonical NPZs (Codex r9i
    # review, item 4). Read explicitly, and then joined fail-closed below.
    # r9t blockers 4 and 5: provenance and §2 reconciliation are admission
    # criteria, not decorations, so their absence is named here rather than
    # discovered by a reviewer reading the bundle
    launch = gate.get("launch_record")
    if not launch:
        reasons.append({"gate": "launch_record",
                        "why": "no launch record was supplied, so the exact command, the source "
                               "commit, the host and the physical GPU that produced this control "
                               "are unrecorded",
                        "note": LAUNCH_RECORD_NOTE})
    elif not launch.get("environment_verified"):
        passed = launch.get("environment_axes_passed")
        reasons.append({"gate": "launch_record_environment",
                        "why": "the launch record was not verified against the executing "
                               f"environment on every axis ({passed} of "
                               f"{launch.get('environment_axes_expected', len(ENVIRONMENT_AXES))}"
                               " passed); a record that was not fully compared identifies "
                               "nothing",
                        "note": CAPTURE_FAILURE_NOTE})
    if not gate.get("retrieval_handoff"):
        reasons.append({"gate": "retrieval_handoff",
                        "why": "the sparse-bank retrieval control's handoff was not supplied, so "
                               "this report cannot state whether §2 is complete without risking "
                               "contradicting it",
                        "note": RETRIEVAL_RECONCILIATION_NOTE})
    if gate.get("non_canonical_declared"):
        reasons.append({"gate": "declared_non_canonical",
                        "why": "the operator ran this control with --non-canonical",
                        "note": mr.NON_CANONICAL_NOTE})
    # FAIL-CLOSED: whatever the derived flag says, the status may never come out
    # more canonical than it. A reason we failed to enumerate is still a reason.
    if gate.get("non_canonical") and not reasons:
        reasons.append({"gate": "non_canonical_flag",
                        "why": "the verified gate carries non_canonical = True without naming a "
                               "reason this function knows how to enumerate; the status refuses "
                               "to be more canonical than the gate it was handed",
                        "note": mr.NON_CANONICAL_NOTE})
    return {"canonical": not reasons, "reasons": reasons,
            "note": None if not reasons else mr.NON_CANONICAL_NOTE}


def write_probe_report(out_dir, records, binding, binding_sha256, provenance,
                       tau=me.TAU, prefixes=me.K_PREFIXES, gate=None,
                       controls_elsewhere=None):
    """Publish the probe's JSON + markdown, both stamped with every caveat.

    The order is the contract (Codex r9c review, M9): the summary is computed,
    then the JSON and the Markdown are written and flushed to disk, and only
    then do the staged dumps leave quarantine -- with any failure during the move
    rolling every completed one back. A crash therefore leaves either a
    quarantine with no manifest, or a manifest whose every named file is present
    and digest-verified; never a scatter of unmanifested finals.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    records = list(records)
    # a previous attempt may have died mid-move; put its finals back in
    # quarantine before this one writes a thing (Codex r9i review, item 3)
    recovery = recover_publication(out_dir)
    # summarize FIRST: a refusal in here must not leave published dumps behind
    summary = summarize_probe(records, prefixes=prefixes)
    report = {
        "experiment": "exp_22 loc_meshgrid R1 off-grid truth probe + AGREE calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control_label": CONTROL_LABEL,
        "calibration_label": CALIBRATION_LABEL,
        "subset": mr.SUBSET_LABEL,
        "agree_leakage_caveat": me.AGREE_LEAKAGE_CAVEAT,
        "scorer_readout_deviation": me.SCORER_READOUT_DEVIATION,
        "batching_caveat": me.BATCHING_CAVEAT,
        "binding_sha256": binding_sha256,
        "binding": {field: binding[field] for field in PROBE_BINDING_FIELDS
                    if field in binding},
        "provenance": dict(provenance or {}),
        # the run-level gates this control was admitted under; None only when the
        # caller ran the pass directly, which the CLI never does
        "run_gate": gate,
        "single_shard": bool((gate or {}).get("single_shard")),
        "single_shard_note": (gate or {}).get("single_shard_note"),
        "canonical_status": probe_canonical_status(gate),
        # every artifact of this round carries the same disclosures, so an
        # off-grid output read on its own cannot lose them (r9 finding 10)
        "latency_scope_note": mr.LATENCY_SCOPE_NOTE,
        "truth_binding_note": mr.TRUTH_BINDING_NOTE,
        "tie_tolerance_note": TIE_TOLERANCE_NOTE,
        # r9s: the gate's price and the retired check's new status, stamped at
        # the top of the report so neither has to be rediscovered from a record
        "tie_cost_note": MATCHED_BATCHING_TIE_COST_NOTE,
        "changed_batching_diagnostic_note": CHANGED_BATCHING_DIAGNOSTIC_NOTE,
        # THE GATE'S EVIDENCE: matched path only, with its own artifact path and
        # its own pair count. r9u sliced a block that joined the retired
        # 6.67e-3 / 8,064 to the matched separation (Codex r9v residual 1)
        "tie_evidence": tie_evidence(),
        # ... and the superseded numbers, under a label, never mixed in
        "retired_path_evidence": retired_path_evidence(),
        # reconciled against the sibling control's own handoff when one was
        # supplied, so the bundle cannot contradict itself about §2 (r9t B5)
        "controls_elsewhere": (reconcile_controls_elsewhere(None)
                               if controls_elsewhere is None else dict(controls_elsewhere)),
        "retrieval_reconciliation_note": RETRIEVAL_RECONCILIATION_NOTE,
        "launch_record_note": LAUNCH_RECORD_NOTE,
        "protocol": {"tau": float(tau), "k_prefixes": [int(k) for k in prefixes],
                     "noise_policy": me.REGISTERED_NOISE_POLICY,
                     "noise_note": "the truth generation is keyed by the query, not by a "
                                   "candidate, so it is drawn from exactly the K latents every "
                                   "grid candidate of that query was drawn from"},
        "n_queries": len(records),
        "records": list(records),
        "summary": summary,
        "publication": {"completed": False, "note": PUBLICATION_ORDER_NOTE,
                        "journal_note": JOURNAL_NOTE, "recovery": recovery},
    }
    path = os.path.join(str(out_dir), PROBE_REPORT_JSON)
    markdown = os.path.join(str(out_dir), PROBE_REPORT_MARKDOWN)

    # pass 1 -- the SAFETY NET. Every file is named with the digest it will have,
    # and publication is honestly recorded as not yet done, so a crash during the
    # move leaves a manifest that says so rather than one that lies.
    _write_json_durably(path, mr.jsonable(report))
    _write_text_durably(markdown, render_markdown(report))

    # the manifest is on disk and fsynced; only now do the finals leave quarantine
    publish_probe_waveforms(out_dir, records)
    verified = verify_published_probe(out_dir, records)

    # pass 2 -- the COMPLETION RECORD. r9d serialized the records before the
    # publication flag was set, so a successful run persisted
    # waveform_published=false (Codex r9f review, nit). The same payload is
    # rewritten once publication is complete and verified.
    report["records"] = list(records)
    report["publication"] = {"completed": True, "n_published": verified["n_published"],
                             "verified": True, "note": PUBLICATION_ORDER_NOTE,
                             "journal_note": JOURNAL_NOTE, "recovery": recovery}
    _write_json_durably(path, mr.jsonable(report))
    _write_text_durably(markdown, render_markdown(report))
    return {"json": path, "markdown": markdown, "publication": report["publication"],
            "sha256": {"json": me.file_sha256(path), "markdown": me.file_sha256(markdown)}}


def _fsync_path(path):
    """Flush a written file, and the directory entry that names it, to disk."""
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _write_json_durably(path, payload):
    """``meshgrid_engine.write_json`` plus the fsync the publish order needs."""
    me.write_json(path, payload)
    return _fsync_path(path)


def _write_text_durably(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return _fsync_path(path)


def verify_published_probe(out_dir, records, recover=False):
    """Every file the manifest names is present, and is the file it names.

    ``recover=True`` makes this a startup path: an incomplete journal is rolled
    back into quarantine first, so a caller that verifies after a crash is told
    the truth about a quarantined set rather than about a half-published one.
    """
    if recover:
        recover_publication(out_dir)
    missing, wrong = [], []
    for record in records:
        path = os.path.join(str(out_dir), record["waveform_path"])
        if not os.path.isfile(path):
            missing.append(record["waveform_path"])
        elif me.file_sha256(path) != record["waveform_sha256"]:
            wrong.append(record["waveform_path"])
    if missing or wrong:
        raise ValueError(
            f"the published dump set does not match the manifest just written: {len(missing)} "
            f"file(s) missing (first {missing[:3]}) and {len(wrong)} with a different digest "
            f"(first {wrong[:3]}); the control publishes a complete set or none")
    return {"n_published": len(records), "verified": True}


def summarize_probe(records, prefixes=me.K_PREFIXES):
    """Rank and calibration distributions over the sixteen probe queries."""
    records = list(records)
    if not records:
        raise ValueError("the probe summary needs at least one record")
    by_k = {}
    for k in (int(p) for p in prefixes):
        ranks = np.asarray([record["rank_lme"][str(k)]["rank"] for record in records],
                           dtype=np.float64)
        deltas = np.asarray([record["rank_lme"][str(k)]["truth_minus_best_grid"]
                             for record in records], dtype=np.float64)
        percentiles = np.asarray([record["rank_lme"][str(k)]["percentile"]
                                  for record in records], dtype=np.float64)
        # rank 1 covers both "beat everything" and "tied the best", and those are
        # different claims: a tie means the truth position scored no better than
        # some grid candidate (Codex r9 review, finding 10)
        ties = np.asarray([record["rank_lme"][str(k)]["n_grid_tied"] for record in records],
                          dtype=np.int64)
        strictly_better = int(((ranks == 1.0) & (ties == 0)).sum())
        by_k[str(k)] = {
            "n_queries": len(records),
            "rank": {"mean": float(ranks.mean()), "median": float(np.median(ranks)),
                     "min": float(ranks.min()), "max": float(ranks.max())},
            "n_truth_beats_every_candidate": strictly_better,
            "n_truth_ties_the_best": int(((ranks == 1.0) & (ties > 0)).sum()),
            "n_rank_one": int((ranks == 1.0).sum()),
            "rank_one_note": "rank 1 means no grid candidate scored HIGHER; "
                             "n_truth_beats_every_candidate counts only the strictly better "
                             "cases, and n_truth_ties_the_best the rest",
            "truth_minus_best_grid": {"mean": float(deltas.mean()),
                                      "median": float(np.median(deltas)),
                                      "min": float(deltas.min()), "max": float(deltas.max())},
            "percentile": {"mean": float(percentiles.mean()),
                           "median": float(np.median(percentiles))}}
    real = np.concatenate([np.asarray(record["calibration"]["real"], dtype=np.float64)
                           for record in records])
    generated = np.concatenate([np.asarray(record["calibration"]["generated"],
                                           dtype=np.float64) for record in records])
    return {"by_k": by_k,
            "calibration": {"real": _distribution(real), "generated": _distribution(generated),
                            "gap_mean_real_minus_generated": float(real.mean()
                                                                   - generated.mean()),
                            "label": CALIBRATION_LABEL}}


def render_markdown(report):
    """A short human-readable summary; the JSON carries everything."""
    lines = ["# exp_22 R1 — off-grid truth probe + real-vs-generated AGREE calibration", ""]
    lines.append(f"Generated {report['created_utc']}.")
    lines.append("")
    lines.append(f"> **{report['control_label']}**")
    lines.append("")
    lines.append(f"- **Scope:** {report['subset']}")
    lines.append(f"- **Run binding:** `{report['binding_sha256']}`")
    lines.append(f"- **AGREE leakage caveat:** {report['agree_leakage_caveat']}")
    lines.append(f"- **Scorer readout deviation:** {report['scorer_readout_deviation']}")
    lines.append(f"- **Truth binding:** {report['truth_binding_note']}")
    lines.append(f"- **Latency scope:** {report['latency_scope_note']}")
    lines.append("")
    status = report.get("canonical_status") or {}
    if status.get("reasons"):
        lines.append(f"> **{status['note']}**")
        lines.append(">")
        for reason in status["reasons"]:
            lines.append(f"> - `{reason['gate']}` — {reason['why']}")
        lines.append("")
    if report.get("single_shard"):
        lines.append(f"> **{report['single_shard_note']}**")
        lines.append("")
    lines.append("## Off-grid truth rank against the grid (log-mean-exp)")
    lines.append("")
    lines.append("| K | median rank | min | max | truth strictly beats every candidate | "
                 "ties the best | median (truth − best grid) |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, block in sorted(report["summary"]["by_k"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {k} | {mr.format_number(block['rank']['median'], 1)} | "
                     f"{mr.format_number(block['rank']['min'], 0)} | "
                     f"{mr.format_number(block['rank']['max'], 0)} | "
                     f"{block['n_truth_beats_every_candidate']}/{block['n_queries']} | "
                     f"{block['n_truth_ties_the_best']}/{block['n_queries']} | "
                     f"{mr.format_number(block['truth_minus_best_grid']['median'], 5)} |")
    lines.append("")
    lines.append(f"_{report['summary']['by_k'][str(max(int(k) for k in report['summary']['by_k']))]['rank_one_note']}_")
    lines.append("")
    lines.append("## Real vs generated AGREE cosine")
    lines.append("")
    lines.append(f"> {report['summary']['calibration']['label']}")
    lines.append("")
    calibration = report["summary"]["calibration"]
    lines.append("| bank | n | mean | sd | min | median | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("real", "generated"):
        block = calibration[name]
        lines.append(f"| {name} | {block['n']} | {mr.format_number(block['mean'], 4)} | "
                     f"{mr.format_number(block['sd'], 4)} | {mr.format_number(block['min'], 4)} | "
                     f"{mr.format_number(block['median'], 4)} | {mr.format_number(block['max'], 4)} |")
    lines.append("")
    lines.append(f"Mean gap (real − generated): "
                 f"{mr.format_number(calibration['gap_mean_real_minus_generated'], 4)}")
    lines.append("")
    largest_k = max(int(k) for k in report["protocol"]["k_prefixes"])
    lines.append("## Per-query")
    lines.append("")
    lines.append(f"| room | query | e_oracle (m) | rank @K={largest_k} | "
                 f"truth − best grid @K={largest_k} | mean real cos | mean generated cos |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in report["records"]:
        largest = str(max(int(k) for k in record["rank_lme"]))
        block = record["rank_lme"][largest]
        lines.append(
            f"| {record['room_id']} | `{record['query_id'].split('|')[0]}` | "
            f"{mr.format_number(record['e_oracle'], 3)} | {block['rank']} | "
            f"{mr.format_number(block['truth_minus_best_grid'], 5)} | "
            f"{mr.format_number(record['calibration']['real_summary']['mean'], 4)} | "
            f"{mr.format_number(record['calibration']['generated_summary']['mean'], 4)} |")
    lines.append("")
    lines.append("## Observation-continuity tie — matched-batching replay")
    lines.append("")
    lines.append(f"> {report.get('tie_tolerance_note') or TIE_TOLERANCE_NOTE}")
    lines.append("")
    lines.append(f"> _Cost:_ {MATCHED_BATCHING_TIE_COST_NOTE}")
    lines.append("")
    lines.append(f"> _Dynamic range, not detection:_ {DYNAMIC_RANGE_NOTE}")
    lines.append("")
    # the evidence block, spelled out with the counts each number belongs to --
    # r9u's markdown attributed the matched margin to the retired path's pair
    # count (Codex r9v residual 1)
    evidence = report.get("tie_evidence") or tie_evidence()
    retired = report.get("retired_path_evidence") or retired_path_evidence()
    lines.append("### Evidence for this gate — MATCHED path only")
    lines.append("")
    lines.append(f"- measured on `{evidence['path']}`, artifact "
                 f"`{evidence['artifact']}`")
    lines.append(f"- honest replay: {evidence['n_replayed_queries']} queries / "
                 f"{evidence['n_replay_candidates']:,} candidates, max abs delta "
                 f"{mr.format_number(evidence['max_abs_delta'], 8)}, aggregate "
                 f"{mr.format_number(evidence['max_abs_aggregate_delta'], 8)}, float16 "
                 f"round-trip exact {mr.format_number(evidence['float16_bit_exact'])}, "
                 f"{evidence['n_cells_over_own_tolerance']} cells over their own bound")
    lines.append(f"- substituted observations: **{evidence['n_substitution_pairs']:,} ordered "
                 f"pairs** over {evidence['n_donor_observations']:,} donors — min "
                 f"**{mr.format_number(evidence['substitution_min'], 6)}** "
                 f"(same-room {evidence['n_substitution_same_room_pairs']:,} pairs, min "
                 f"{mr.format_number(evidence['substitution_same_room_min'], 6)}; "
                 f"same-receiver {evidence['n_substitution_same_receiver_pairs']} pairs, min "
                 f"{mr.format_number(evidence['substitution_same_receiver_min'], 6)}), "
                 f"{evidence['n_substitution_undetected']} undetected")
    lines.append(f"- separation: **{mr.format_number(evidence['separation'], 1)}x** against a "
                 f"{mr.format_number(evidence['tolerance'], 8)} tolerance (required >= "
                 f"{mr.format_number(evidence['min_separation_required'], 1)}x)")
    lines.append("")
    lines.append(f"> _Retired path ({retired['path']}), {retired['label']}:_ its substitution "
                 f"minimum {mr.format_number(retired['substitution_min'], 6)} over "
                 f"{retired['n_substitution_pairs']:,} pairs is NOT this gate's margin; its "
                 f"drift distribution (median "
                 f"{mr.format_number(retired['median_abs_delta'], 6)}, q99 "
                 f"{mr.format_number(retired['q99_abs_delta'], 6)}, max "
                 f"{mr.format_number(retired['max_abs_delta'], 6)} over "
                 f"{retired['n_measurements']} measurements) is the non-gating diagnostic's "
                 f"reference. Artifact `{retired['artifact']}`")
    lines.append("")
    # "max abs delta" rather than "max |delta|": raw pipes would split the cell
    lines.append("| room | query | candidates replayed | batching | max abs delta | "
                 "tolerance (half-ulp) | headroom | aggregate abs delta | bit-exact | "
                 "dyn. range: query cosine span | dyn. range: span / delta | "
                 "measured substitution min | measured separation | diagnostic (non-gating) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for record in report["records"]:
        tie = record.get("observation_continuity")
        if not tie:
            lines.append(f"| {record['room_id']} | `{record['query_id'].split('|')[0]}` | "
                         + " — |" * 12)
            continue
        diagnostic = tie.get("changed_batching_diagnostic") or {}
        lines.append(
            f"| {record['room_id']} | `{record['query_id'].split('|')[0]}` | "
            f"{mr.format_number(tie['n_candidates_replayed'])} | "
            f"{tie['batch_rows']}/{tie['source_chunk']} | "
            f"{mr.format_number(tie['max_abs_delta'], 6)} | "
            f"{mr.format_number(tie['tolerance'], 6)} | "
            f"{mr.format_number(tie['headroom'], 6)} | "
            f"{mr.format_number(tie['aggregate_max_abs_delta'], 8)} | "
            f"{mr.format_number(tie['float16_bit_exact'])} | "
            f"{mr.format_number(tie['query_cosine_span'], 4)} | "
            f"{mr.format_number(tie['query_cosine_span_over_delta'], 1)}x | "
            f"{mr.format_number(tie['measured_substitution_min'], 6)} | "
            f"{mr.format_number(tie['measured_separation'], 2)}x | "
            f"{mr.format_number(diagnostic.get('max_abs_delta'), 6)} |")
    lines.append("")
    lines.append(f"> _Non-gating diagnostic:_ {CHANGED_BATCHING_DIAGNOSTIC_NOTE}")
    lines.append("")
    lines.append("## §2 controls that are NOT in this report")
    lines.append("")
    for name, where in sorted(report["controls_elsewhere"].items()):
        lines.append(f"- **{name}** — {where}")
    lines.append("")
    lines.append(f"_Latency scope:_ {report['latency_scope_note']}")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# the pass
# --------------------------------------------------------------------------- #
def load_grid_row(run_dir, query, binding_sha256=None, binding=None, snapshot=None):
    """The published row the truth score is ranked against, fully joined first.

    A generic digest check proves a row is intact; it does not prove it is THIS
    query's row. Rows are addressed by ``(room, position)``, so a same-binding row
    left at the expected path by another query would have silently supplied its
    grid scores to the rank (Codex r9 review, finding 4). The engine's own
    identity join answers that question -- it compares the row's query id,
    receiver, branch and full candidate index list against the G1 plan -- and the
    row's protocol is checked against the binding on top of it.
    """
    paths = me.query_artifact_paths(str(run_dir), query.room_id, int(query.position))
    # ONE read of the row and ONE of its sidecar, both parsed out of the buffers
    # that were verified -- the probe used to verify, then reopen, then reopen
    # the sidecar again for the continuity check (Codex r9l review, item 3)
    verdict = mr.read_verified_query_artifact(paths["row"], binding_sha256=binding_sha256)
    if not verdict["ok"]:
        raise ValueError(f"{query.room_id} q{int(query.position):05d}: the published row cannot "
                         f"be used as the probe's grid reference: {verdict['reason']}")
    # ... and it is the artifact the CENSUS verified, byte for byte. Without
    # this the census's work expired the moment it returned (Codex r9n review).
    if snapshot is not None:
        mr.assert_matches_snapshot(query.query_id, verdict, snapshot)
    row = verdict["row"]
    mr.assert_row_matches_plan(row, query)
    if binding is not None:
        mr.assert_row_protocol([row], binding)
    row["_sims"] = verdict["sims"]
    return row


def run_probe(engine, stream, records, plan, run_dir, out_dir, *, metadata_root,
              binding_sha256=None, binding=None, seed=me.SEED, tau=me.TAU,
              num_samples=me.NUM_SAMPLES, prefixes=me.K_PREFIXES,
              noise_policy=me.NOISE_KEY_POLICY, source_chunk=1, non_canonical=None,
              verify_observation=True, observation_bank=None, observation_decoder=None,
              metadata_bank=None, artifact_snapshot=None, on_record=None):
    """Walk the registered stream and run both controls on the sixteen queries.

    The stream is the released loader in D1 order and is walked ONCE, exactly as
    the scored pass walks it, because every query's context draw depends on the
    complete pass; only the sixteen registered positions are generated.
    """
    assert_offgrid_noise_policy(noise_policy)
    probes = me.registered_probe_queries(plan)
    assert_registered_probe_set(probes, plan)
    wanted = {probes[room]: room for room in probes}

    by_id = {record["query_id"]: record for record in records}
    by_position = {int(record["position"]): record for record in records}
    missing = sorted(query_id for query_id in wanted if query_id not in by_id)
    if missing:
        raise ValueError(f"the probe queries {missing[:3]} are not in the context manifest; the "
                         "probe set and the registered subset disagree")

    # the probe SET comes from the engine's registered rule and the query PLANS
    # are then looked up by identity. That reads every room manifest a second
    # time (~330 MB in production) rather than re-implementing the selection rule
    # here, where a second copy could drift from the registered one.
    query_plans, query_groups = {}, {}
    for room_id in sorted(plan.rooms):
        room_plan = me.load_room_plan(plan, room_id)
        query_id = probes[room_id]
        query_plans[query_id] = next(query for query in room_plan.queries
                                     if query.query_id == query_id)
        # r9s: the matched-batching tie replays the query through the receiver's
        # candidate union, exactly as the scored pass did, so the union and its
        # camera-frame positions are gathered here -- while the room manifest is
        # already open -- rather than re-read per query
        group = next(candidate for candidate in me.receiver_groups(room_plan)
                     if any(entry.query_id == query_id for entry in candidate.queries))
        base = np.asarray(room_plan.base, dtype=np.float64)
        query_groups[query_id] = {
            "receiver_id": str(group.receiver_id),
            "union": [int(index) for index in group.union],
            "positions_cam": (base[np.asarray(group.union, dtype=np.int64)]
                              - np.asarray(group.receiver_xyz, dtype=np.float64))}

    status_label = (CANONICAL_STATUS_UNKNOWN if non_canonical is None else
                    (CANONICAL_STATUS_NON_CANONICAL if non_canonical
                     else CANONICAL_STATUS_CANONICAL))
    # the runtime truth path consumes VERIFIED buffers: every pair file it reads
    # must be one the frozen bank covered and must still hash to what the bank
    # recorded, and the coordinates come out of that same buffer. r9j2 gated the
    # bank and then built a fresh, unchecked resolver, so a pair JSON edited
    # after the gate could still supply a mirrored src_loc (Codex r9l review,
    # item 1). This mirrors the verified-pair pattern the retrieval control
    # adopted in r9k; the two are cross-pinned by a test rather than shared,
    # because the rounds may not edit each other's files.
    resolver = mr.TruthResolver(
        metadata_root,
        expected=(None if metadata_bank is None else
                  {query_id: entry["sha256"]
                   for query_id, entry in (metadata_bank.get("queries") or {}).items()}))
    out, seen = [], set()
    for position, (obs_wav, raw_md) in enumerate(stream):
        record = by_position.get(position)
        if record is None or record["query_id"] not in wanted:
            continue
        md = me.GuardedMetadata(raw_md)
        me.verify_context_record(md, record, position)
        query = query_plans[record["query_id"]]
        if obs_wav is None:
            raise ValueError(f"stream position {position}: the loader returned no observed "
                             "waveform; there is nothing to calibrate against")

        metadata_receiver, truth = resolver.resolve(record)
        mr.assert_receiver_matches(query.query_id, metadata_receiver, query.receiver_xyz)
        coordinates = np.asarray(query.coordinates, dtype=np.float64)
        distances = np.linalg.norm(coordinates - truth.reshape(1, 3), axis=1)
        e_oracle = float(distances.min())
        mr.assert_grid_oracle(query.query_id, coordinates, query.oracle, truth)
        # the injective check: this control HAS the stream, so it can compare the
        # truth as a VECTOR against the loader's own target instead of relying on
        # the scalar oracle, which two truths mirrored inside one lattice cell
        # would share (Codex r9 review, finding 3)
        truth_vector_drift = mr.assert_truth_vector(query.query_id, truth, query.receiver_xyz,
                                                    raw_md["source"])

        row = load_grid_row(run_dir, query, binding_sha256=binding_sha256, binding=binding,
                            snapshot=artifact_snapshot)

        # THE PIN, and the byte -> tensor single path. The observation file is
        # read ONCE, hashed against the frozen bank, and the tensor everything
        # below is scored from is decoded out of that same buffer. r9j2 hashed
        # the file after the loader had already decoded it, so restoring the
        # registered file in between made the pin pass over bytes nothing was
        # scored from (Codex r9l review, item 2).
        source, observation = None, obs_wav
        if observation_bank:
            entry = (observation_bank.get("queries") or {}).get(query.query_id)
            if entry is None:
                raise ValueError(f"{query.query_id} is a registered probe query but the "
                                 "pre-registered observation bank does not cover it")
            if observation_decoder is None:
                raise ValueError(
                    f"{query.query_id}: a pre-registered observation bank was supplied but no "
                    "decoder, so the verified bytes could not be turned into the tensor that is "
                    "scored; the pin would then cover a file nothing was decoded from")
            verified = observation_decoder(raw_md.get("path"), entry["sha256"])
            observation = assert_decoded_observation_matches(query.query_id, verified["tensor"],
                                                             obs_wav)
            source = {"ok": True, "sha256": verified["sha256"],
                      "n_bytes": int(verified["n_bytes"]),
                      "loader_path": str(verified["path"]),
                      "decoded_from_verified_bytes": True,
                      "note": OBSERVATION_BINDING_NOTE}

        # ONE embedding, from the ONE verified tensor: the tie, the truth scores
        # and the calibration all read this object, so they cannot disagree about
        # what the observation was (Codex r9l review, item 2)
        obs_embedding = torch.as_tensor(
            engine.embedder(torch.as_tensor(observation).to(engine.device))
        )[0].float().cpu()

        # the query's context branch, computed once and reused by both the truth
        # generation and the observation-continuity check
        query_context = me.context_conditioning(engine.conditioner, md, engine.device)

        # ... and the TIE: the tensor those bytes decoded to is the one these
        # frozen rows were scored against (Codex r9i review, item 2). The sidecar
        # is the one the row verification already parsed -- no reopen.
        continuity = None
        if verify_observation:
            group = query_groups[record["query_id"]]
            continuity = assert_observation_continuity(
                engine, query, md, query_context, row, row["_sims"], obs_embedding,
                receiver_id=group["receiver_id"], union=group["union"],
                positions_cam=group["positions_cam"],
                seed=seed, num_samples=num_samples, noise_policy=noise_policy,
                source_chunk=source_chunk, tau=tau)

        waveforms = generate_at_truth(engine, md, query.receiver_xyz, truth,
                                      query_id=query.query_id, seed=seed,
                                      num_samples=num_samples, noise_policy=noise_policy,
                                      source_chunk=source_chunk, context=query_context)
        sims, scores = truth_scores(engine.embedder, obs_embedding, waveforms, tau=tau,
                                    prefixes=prefixes)
        calibration = calibration_record(engine.embedder, obs_embedding,
                                         md["context_audio"], waveforms)
        dump = write_probe_waveforms(out_dir, query.room_id, query.position, waveforms,
                                     observation, md["context_audio"], truth,
                                     query.receiver_xyz,
                                     query_id=query.query_id, status_label=status_label,
                                     continuity=continuity)

        record_out = {
            "control_label": CONTROL_LABEL,
            "query_id": query.query_id, "room_id": query.room_id,
            "position": int(query.position), "receiver_id": query.receiver_id,
            "receiver_xyz": [float(v) for v in query.receiver_xyz],
            "truth_xyz": [float(v) for v in truth],
            "truth_vector_drift_m": float(truth_vector_drift),
            "observation_continuity": continuity,
            "observation_source": source,
            "observation": observation_digests(
                observation, raw_md.get("path"),
                source_sha256=None if source is None else source["sha256"]),
            "n_candidates": int(query.n_candidates), "num_samples": int(num_samples),
            "e_oracle": e_oracle,
            "truth_is_a_candidate": bool(distances.min() == 0.0),
            "truth_sims": [float(v) for v in sims.reshape(-1)],
            "truth_scores": {str(k): value for k, value in scores.items()},
            "rank_lme": {str(k): value for k, value in
                         rank_against_grid(row, scores, aggregator="lme").items()},
            "rank_mean": {str(k): value for k, value in
                          rank_against_grid(row, scores, aggregator="mean").items()},
            "calibration": calibration,
            "grid_row_sha256": row.get("row_sha256"),
            "grid_sims_sha256": row.get("sims_sha256"),
        }
        record_out.update(dump)
        out.append(record_out)
        seen.add(query.query_id)
        if on_record is not None:
            on_record(record_out)

    for record in out:
        record["observation_source_pinned"] = bool(observation_bank)
    if verify_observation:
        deltas = [record["observation_continuity"]["max_abs_delta"] for record in out]
        tolerances = [record["observation_continuity"]["tolerance"] for record in out]
        spans = [record["observation_continuity"]["query_cosine_span_over_delta"]
                 for record in out]
        continuity_summary = {"ok": True, "checked": len(out),
                              # what "complete" means, so the status can check it
                              "n_expected": len(out),
                              "queries": [record["query_id"] for record in out],
                              "max_abs_delta": (max(deltas) if deltas else 0.0),
                              "min_headroom": (min(t - d for d, t in zip(deltas, tolerances))
                                               if deltas else 0.0),
                              "tolerance": (max(tolerances) if tolerances else 0.0),
                              "gate": "matched_batching_replay",
                              "aggregate_max_abs_delta":
                                  (max(record["observation_continuity"]
                                       ["aggregate_max_abs_delta"] for record in out)
                                   if out else 0.0),
                              "all_float16_bit_exact":
                                  all(record["observation_continuity"]["float16_bit_exact"]
                                      for record in out),
                              "n_candidates_replayed":
                                  sum(record["observation_continuity"]["n_candidates_replayed"]
                                      for record in out),
                              "cost_note": MATCHED_BATCHING_TIE_COST_NOTE,
                              # dynamic range, named as such (Codex r9q item 3)
                              "min_query_cosine_span_over_delta": (min(spans) if spans
                                                                   else float("inf")),
                              "dynamic_range_note": DYNAMIC_RANGE_NOTE,
                              # the MEASURED margin, and the standing verdict
                              "measured_substitution_min":
                                  float(MATCHED_PATH_EVIDENCE["substitution_min"]),
                              "measured_separation":
                                  (float(MATCHED_PATH_EVIDENCE["substitution_min"]
                                         / max(tolerances)) if tolerances else float("inf")),
                              "changed_batching_diagnostic_note":
                                  CHANGED_BATCHING_DIAGNOSTIC_NOTE,
                              "per_query_delta": {record["query_id"]:
                                                  record["observation_continuity"][
                                                      "max_abs_delta"] for record in out},
                              "tolerance_note": TIE_TOLERANCE_NOTE,
                              "note": OBSERVATION_BINDING_NOTE}
    else:
        continuity_summary = {"ok": False, "checked": 0, "n_expected": len(out),
                              "queries": [],
                              "why": "the observation-continuity check was disabled, so nothing "
                                     "ties the loaded observation to the frozen rows",
                              "note": OBSERVATION_BINDING_NOTE}
    for record in out:
        record["observation_continuity_summary"] = continuity_summary

    absent = sorted(set(wanted) - seen)
    if absent:
        # the staged dumps stay in waveforms/.partial/: quarantined and obviously
        # incomplete, never a finalized file no manifest names (r9 finding 9)
        raise ValueError(f"the stream ended before {len(absent)} probe queries were reached "
                         f"(first {absent[:3]}); a partial control may not be published. "
                         f"{len(out)} staged dump(s) remain under "
                         f"{WAVEFORM_STAGING_DIRNAME}/ and are not manifested")
    out.sort(key=lambda record: int(record["position"]))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt-path", default=None,
                        help="the frozen FLAC checkpoint the run was scored under; required "
                             "except in --print-metadata-bank-digest mode")
    parser.add_argument("--run-dir", default=None,
                        help="the MERGED I1 run directory the probe ranks against")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-config",
                        default=os.path.join("src", "configs", "model_configs", "FLAC", "AR",
                                             "FLAC_AR.json"))
    parser.add_argument("--dataset-config",
                        default=os.path.join("src", "configs", "dataset_configs", "AR", "eval",
                                             "acousticroom_unseeneval.json"))
    parser.add_argument("--context-manifest",
                        default=os.path.join("outputs_loc", "exp22",
                                             "d1_context_manifest.json"))
    parser.add_argument("--audit-report",
                        default=os.path.join("outputs_loc", "exp22", "g1_audit",
                                             "geometry_audit_report.json"))
    parser.add_argument("--metadata-root",
                        default=os.path.join("AcousticRooms", "metadata"))
    parser.add_argument("--agree-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--cond-method", default="vanilla", choices=["vanilla", "fa_invariant"])
    parser.add_argument("--cond-autocast", default="default",
                        choices=["default", "bf16", "off"])
    parser.add_argument("--seed", type=int, default=me.SEED)
    parser.add_argument("--tau", type=float, default=me.TAU)
    parser.add_argument("--num-samples", type=int, default=me.NUM_SAMPLES)
    parser.add_argument("--k-prefixes", type=int, nargs="+", default=list(me.K_PREFIXES))
    parser.add_argument("--noise-policy", default=me.NOISE_KEY_POLICY,
                        choices=list(me.NOISE_KEY_POLICIES))
    parser.add_argument("--steps", type=int, default=me.STEPS)
    parser.add_argument("--cfg-scale", type=float, default=me.CFG_SCALE)
    parser.add_argument("--source-chunk", type=int, default=1)
    parser.add_argument("--single-shard", action="store_true",
                        help="rank against a directory that carries no merge_report.json. "
                             "Relaxes only the merge-only gates; the artifact-hash joins, the "
                             "row census, the identity join and every digest still apply, and "
                             "the control is stamped as non-canonical")
    parser.add_argument("--expect-ckpt-sha256", default=None,
                        help="enforce the run binding's ckpt_sha256 against this value")
    parser.add_argument("--allow-protocol-deviation", action="store_true",
                        help="run even though the run binding is not the registered protocol; "
                             "the artifacts are then stamped as a sensitivity check")
    parser.add_argument("--expect-metadata-bank-sha256", default=None,
                        help="the PRE-REGISTERED pair-metadata bank digest the continuous "
                             "truths must come out of. Required for a canonical control; "
                             "obtain it with --print-metadata-bank-digest and commit it before "
                             "any result exists")
    parser.add_argument("--non-canonical", action="store_true",
                        help="run without a pre-registered metadata-bank digest. "
                             "Trust-on-first-use is not a canonical mode, so the report, the "
                             "markdown and every NPZ are stamped NON-CANONICAL")
    parser.add_argument("--print-metadata-bank-digest", action="store_true",
                        help="PRE-REGISTRATION MODE: compute the pair-metadata bank digest from "
                             "--context-manifest and --metadata-root, print it and exit")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="where the split's relpaths are rooted; the observed RIRs are read "
                             "from <root>/<relpath>")
    parser.add_argument("--expect-observation-bank-sha256", default=None,
                        help="the PRE-REGISTERED observed-RIR bank digest the sixteen probe "
                             "observations must come out of. Required for a canonical control; "
                             "obtain it with --print-observation-digest and commit it before "
                             "any result exists")
    parser.add_argument("--print-observation-digest", action="store_true",
                        help="PRE-REGISTRATION MODE: compute the observed-RIR bank digest over "
                             "the sixteen registered probe queries from --audit-report, "
                             "--context-manifest and --dataset-root, print the per-query digests "
                             "and the combined value, and exit. Needs no run directory, no "
                             "checkpoint and no GPU")
    parser.add_argument("--launch-record", default=None,
                        help="PATH to the launch record this run is described by -- argv, git "
                             "SHA, hostname and nvidia-smi GPU UUIDs, written BEFORE the run. "
                             "Required for a canonical control (experiment_SOP.md:37); its "
                             "sha256 is hashed into the report's provenance")
    parser.add_argument("--emit-launch-record", default=None,
                        help="PROVENANCE MODE: write the launch record for THIS command to PATH "
                             "and exit. Run the identical command again with --launch-record "
                             "PATH to perform the run the record describes")
    parser.add_argument("--retrieval-handoff", default=None,
                        help=f"PATH to the sparse-bank retrieval control's "
                             f"{RETRIEVAL_HANDOFF_FILENAME}. Required for a canonical control: "
                             "this report states §2 completeness and may not contradict the "
                             "sibling control about it")
    # the probe registers no dump case list: announcement 08 names its sixteen
    # queries directly. Present so build_run_binding can be reused unchanged.
    parser.add_argument("--dump-cases-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    """Startup refusals -- before a checkpoint is read or a GPU is touched."""
    if args.print_metadata_bank_digest and args.print_observation_digest:
        _refuse("--print-metadata-bank-digest and --print-observation-digest are separate "
                "pre-registration modes; run one at a time so each printed value is "
                "unambiguous")
    if args.print_metadata_bank_digest or args.print_observation_digest:
        return True
    if args.emit_launch_record:
        return True
    for name in ("ckpt_path", "run_dir", "out_dir"):
        if not getattr(args, name):
            _refuse(f"--{name.replace('_', '-')} is required to run the control")
    if not args.expect_metadata_bank_sha256 and not args.non_canonical:
        _refuse("a canonical off-grid control requires the PRE-REGISTERED pair-metadata bank "
                f"digest. {mr.METADATA_BANK_PREREGISTRATION_NOTE}. Pass --non-canonical to run "
                "a diagnostic instead")
    if not args.expect_observation_bank_sha256 and not args.non_canonical:
        _refuse("a canonical off-grid control requires the PRE-REGISTERED observed-RIR bank "
                f"digest. {OBSERVATION_BANK_PREREGISTRATION_NOTE}. Pass --non-canonical to run "
                "a diagnostic instead")
    if not args.launch_record and not args.non_canonical:
        _refuse(f"a canonical off-grid control requires --launch-record. {LAUNCH_RECORD_NOTE}. "
                "Pass --non-canonical to run a diagnostic instead")
    if not args.retrieval_handoff and not args.non_canonical:
        _refuse(f"a canonical off-grid control requires --retrieval-handoff, the sparse-bank "
                f"control's {RETRIEVAL_HANDOFF_FILENAME}. "
                f"{RETRIEVAL_RECONCILIATION_NOTE}. Pass --non-canonical to run a diagnostic "
                "instead")
    if args.noise_policy != me.REGISTERED_NOISE_POLICY:
        _refuse(f"--noise-policy {args.noise_policy!r} cannot key an off-grid draw; "
                f"{me.REGISTERED_NOISE_POLICY!r} is the registered policy and the only one "
                "under which the truth generation shares the grid candidates' latents")
    if args.cond_method != "vanilla":
        _refuse(f"cond_method={args.cond_method!r}: the registered exp_22 arm is vanilla")
    prefixes = [int(k) for k in args.k_prefixes]
    if sorted(set(prefixes)) != sorted(prefixes) or min(prefixes) < 1:
        _refuse(f"--k-prefixes must be distinct positive integers, got {prefixes}")
    if max(prefixes) != int(args.num_samples):
        _refuse(f"--num-samples must equal the largest prefix ({max(prefixes)}), not "
                f"{args.num_samples}: the prefixes are nested reads of ONE sequence")
    if float(args.tau) <= 0.0:
        _refuse(f"--tau must be > 0, got {args.tau}")
    if os.path.abspath(str(args.out_dir)) == os.path.abspath(str(args.run_dir)):
        _refuse("--out-dir may not be the scored run directory: a control never writes into "
                "the artifact set it reports against")
    return True


def _load_and_validate_checkpoint(args, model_config):
    """Read the checkpoint to CPU and validate it -- AFTER the artifact gates.

    Isolated in its own function because ``validate_checkpoint`` is the first
    thing in this tool that reaches ``eval_FLAC``, whose module-level function
    defaults call ``torch.cuda.is_available()``. No allocation happens, but the
    r9c review is right that a refused run should not have gone near the device
    layer at all, so nothing calls this until every gate has passed.
    """
    from localize_meshgrid import validate_checkpoint

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    validate_checkpoint(args, model_config, ckpt)
    return ckpt


def gate_run(args, model_config, agree_path, totals=None,
             require_manifest_census=True):
    """Every artifact gate, on CPU, BEFORE anything reaches a device.

    The r9 probe built its binding out of a LOADED AGREE model, which put the
    scorer on ``--device`` before the gate that decides whether this control may
    run at all (Codex r9 review, finding 5). Nothing here needs a device: the
    AGREE identity is a file digest, the checkpoint is read to CPU, and the whole
    artifact ladder -- audit chain, context manifest, binding, hash joins,
    registered protocol, merge report, row census, identity join -- is applied.
    Only after this returns does the caller load a model.

    ``totals`` and ``require_manifest_census`` exist for fixtures, exactly as
    ``evaluate_run``'s do, and are never passed by ``main``: a real run is always
    held to the registered census.
    """
    # localize_meshgrid itself imports only torch + src.localization, so the
    # binding builder is safe to reach for here; validate_checkpoint is NOT --
    # its body imports eval_localization -> eval_FLAC, whose function defaults
    # evaluate torch.cuda.is_available() at import time (Codex r9c review, B5).
    # It is therefore called only after every artifact gate has passed.
    from localize_meshgrid import build_run_binding

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest,
                                require_census=require_manifest_census)
    binding = build_run_binding(args, plan, ckpt_sha256=me.file_sha256(args.ckpt_path),
                                agree_sha256=me.file_sha256(agree_path),
                                model_config_sha256=me.file_sha256(args.model_config))
    gate = assert_probe_binding(args.run_dir, binding)
    gate.update(assert_probe_run_census(
        args.run_dir, gate["published"], gate["binding_sha256"], plan,
        manifest["records"], args.context_manifest,
        totals=totals, single_shard=args.single_shard,
        expect_ckpt_sha256=args.expect_ckpt_sha256,
        allow_protocol_deviation=args.allow_protocol_deviation))
    # r9d only STORED the expected string here, so the control never read the
    # tree it takes its truths from and any nonempty value passed (Codex r9f
    # review, B3). The bank is computed over the same records the run is bound
    # to and compared against the pre-registered digest -- and it happens here,
    # before _load_and_validate_checkpoint, so a bank mismatch refuses without
    # eval_FLAC ever being imported (B5 ordering preserved).
    bank = mr.compute_metadata_bank_digest(args.context_manifest, args.metadata_root,
                                           records=manifest["records"])
    gate["metadata_bank"] = mr.assert_metadata_bank(
        bank["metadata_bank_sha256"], expected=args.expect_metadata_bank_sha256,
        allow_unpinned=args.non_canonical)
    gate["metadata_bank"]["queries"] = bank["queries"]
    gate["metadata_bank_sha256"] = bank["metadata_bank_sha256"]
    gate["metadata_bank_expected"] = args.expect_metadata_bank_sha256
    # the observed-RIR bank, on the same terms and in the same window as the
    # pair-metadata bank: computed here, compared here, and both of them before
    # _load_and_validate_checkpoint ever reaches eval_FLAC (r9j2)
    observations = compute_observation_bank_digest(
        args.audit_report, args.context_manifest, dataset_root=args.dataset_root,
        require_manifest_census=require_manifest_census, branch=args.branch)
    gate["observation_bank"] = assert_observation_bank(
        observations["observation_bank_sha256"],
        expected=args.expect_observation_bank_sha256, allow_unpinned=args.non_canonical)
    gate["observation_bank"]["queries"] = observations["queries"]
    gate["observation_bank_sha256"] = observations["observation_bank_sha256"]
    gate["observation_bank_expected"] = args.expect_observation_bank_sha256

    gate["non_canonical_declared"] = bool(args.non_canonical)
    gate["non_canonical"] = bool(args.non_canonical
                                 or not gate["metadata_bank"]["pinned"]
                                 or not gate["observation_bank"]["pinned"]
                                 or args.single_shard
                                 or not gate["registered_protocol"]["is_registered"])
    ckpt = _load_and_validate_checkpoint(args, model_config)
    return plan, manifest, binding, gate, ckpt


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    validate_args(args)
    if args.emit_launch_record:
        # PROVENANCE MODE: no gates, no checkpoint, no GPU work -- it exists so
        # the record can be written from the very command about to be run
        record = write_launch_record(args.emit_launch_record, argv, device=args.device)
        print(json.dumps(mr.jsonable(record), indent=2, sort_keys=True))
        print(f"\nlaunch record -> {args.emit_launch_record}")
        print(f"\nnow run the identical command with --launch-record "
              f"{args.emit_launch_record}")
        return 0
    if args.print_metadata_bank_digest:
        verdict = mr.compute_metadata_bank_digest(args.context_manifest, args.metadata_root)
        print(json.dumps(mr.jsonable(verdict), indent=2, sort_keys=True))
        print(f"\nmetadata_bank_sha256 = {verdict['metadata_bank_sha256']}")
        print(f"\n{mr.METADATA_BANK_PREREGISTRATION_NOTE}")
        return 0
    if args.print_observation_digest:
        # no run directory, no checkpoint, no scorer, no eval_FLAC: this must be
        # runnable before the merge exists, which is the point of it
        verdict = compute_observation_bank_digest(args.audit_report, args.context_manifest,
                                                  dataset_root=args.dataset_root,
                                                  branch=args.branch)
        print(json.dumps(mr.jsonable(verdict), indent=2, sort_keys=True))
        print(f"\nper-query observed-RIR digests ({verdict['n_queries']} registered probe "
              "queries):")
        for query_id, entry in sorted(verdict["queries"].items()):
            print(f"  {entry['sha256']}  {entry['n_bytes']:>9,} B  {query_id}")
        print(f"\nobservation_bank_sha256 = {verdict['observation_bank_sha256']}")
        print(f"\n{OBSERVATION_BANK_PREREGISTRATION_NOTE}")
        return 0
    # the provenance surfaces, read (and refused) before any device work
    launch = (read_verified_launch_record(args.launch_record, argv, device=args.device)
              if args.launch_record else None)
    handoff = (read_retrieval_handoff(args.retrieval_handoff)
               if args.retrieval_handoff else None)

    print(f"{CONTROL_LABEL}\n")
    print(f"AGREE LEAKAGE CAVEAT: {me.AGREE_LEAKAGE_CAVEAT}")
    if launch:
        print(f"launch record {launch['sha256'][:12]}... git {launch['git_sha'][:12]}"
              f"{' (DIRTY TREE)' if launch['git_status_dirty'] else ''} on "
              f"{launch['hostname']}, GPUs "
              + ", ".join(f"{gpu['index']}:{str(gpu['uuid'])[:16]}..."
                          for gpu in launch["gpus"]))
    if handoff:
        print(f"§2 retrieval control: {handoff['status']} "
              f"({'canonical' if handoff['canonical'] else 'not canonical'}), handoff "
              f"{handoff['sha256'][:12]}...")
    # r9s: the tie now replays whole queries, so the operator is told the price
    # before the wait rather than after it
    print(f"\nTIE: {MATCHED_BATCHING_TIE_COST_NOTE}\n")
    if args.single_shard:
        print(f"\n{mr.SINGLE_SHARD_NOTE}\n")

    # the driver's own item unpacker -- one implementation, not a second copy
    from localize_meshgrid import _iter_items as iter_stream_items

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    # resolve the configured scorer only when the operator did not name one
    agree_path = args.agree_ckpt or \
        mq.with_resolved_agree(model_config)["training"]["metrics"]["AGREE_ckpt"]

    # EVERY gate first, on CPU. Nothing below this line may run if one refuses.
    plan, manifest, binding, gate, ckpt = gate_run(args, model_config, agree_path)
    # the provenance surfaces are admission criteria, so they join the gate the
    # status reads rather than sitting only in the provenance block
    gate["launch_record"] = launch
    gate["retrieval_handoff"] = handoff
    gate["non_canonical"] = bool(gate.get("non_canonical") or not launch or not handoff)
    records = manifest["records"]
    print(f"binding gate passed against {args.run_dir}: {gate['binding_sha256'][:12]}... "
          f"({len(gate['fields_checked'])} fields); run census "
          f"{gate['census']['n_queries']:,} queries / {gate['census']['n_rooms']} rooms, "
          f"identity join over {gate['identity_join']['n_queries']:,} identities")

    from src.localization.agree_embed import load_agree_audio

    agree = load_agree_audio(agree_path, args.device)
    engine, context = me.build_mesh_engine(
        args.ckpt_path, model_config, agree, device=args.device,
        cond_method=args.cond_method, cond_autocast=args.cond_autocast,
        steps=args.steps, cfg_scale=args.cfg_scale, ckpt=ckpt)
    print(f"weights: {context['weights_source']}, latent {context['latent_shape']}")

    loader, facts = mq.build_release_stack(args.dataset_config, args.model_config)
    me.assert_release_rng_state(manifest)
    print(f"release call graph reproduced: {facts['call_graph']}")

    def _announce(record):
        largest = str(max(int(k) for k in record["rank_lme"]))
        block = record["rank_lme"][largest]
        print(f"  {record['room_id']}: the truth position ranks {block['rank']} of "
              f"{block['n_candidates']} grid candidates at K={largest} "
              f"(truth - best grid = {block['truth_minus_best_grid']:+.5f})", flush=True)

    probe_records = run_probe(
        engine, iter_stream_items(loader), records, plan, args.run_dir, args.out_dir,
        metadata_root=args.metadata_root, binding_sha256=gate["binding_sha256"],
        binding=gate["published"], seed=args.seed, tau=args.tau,
        num_samples=args.num_samples,
        prefixes=tuple(int(k) for k in args.k_prefixes), noise_policy=args.noise_policy,
        source_chunk=args.source_chunk, non_canonical=gate["non_canonical"],
        observation_bank=gate["observation_bank"],
        observation_decoder=build_observation_decoder(model_config, args.dataset_config),
        metadata_bank=gate["metadata_bank"],
        artifact_snapshot=gate["artifact_snapshot"], on_record=_announce)
    gate["observation_continuity"] = (probe_records[0]["observation_continuity_summary"]
                                      if probe_records else
                                      {"ok": False, "checked": 0,
                                       "why": "no probe query was reached",
                                       "note": OBSERVATION_BINDING_NOTE})
    published = write_probe_report(args.out_dir, probe_records, binding,
                                   gate["binding_sha256"],
                                   provenance={"run_dir": str(args.run_dir),
                                               "audit_report": str(args.audit_report),
                                               "audit_report_sha256": plan.report_sha256,
                                               "context_manifest": str(args.context_manifest),
                                               "agree_ckpt": agree_path,
                                               "agree_ckpt_sha256": agree.ckpt_sha256,
                                               "device": str(args.device),
                                               "launch_record": launch,
                                               "retrieval_handoff": handoff},
                                   tau=args.tau,
                                   prefixes=tuple(int(k) for k in args.k_prefixes),
                                   gate=publication_gate(gate),
                                   controls_elsewhere=reconcile_controls_elsewhere(handoff))
    status = json.load(open(published["json"]))["canonical_status"]
    print(f"\n{len(probe_records)} probe queries -> {published['json']}")
    print(f"  markdown -> {published['markdown']}")
    if status["canonical"]:
        print("  CANONICAL: every registered gate of the run and this control passed")
    else:
        print(f"  {status['note']}")
        for reason in status["reasons"]:
            print(f"    - {reason['gate']}: {reason['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
