#!/usr/bin/env python
"""exp_18 (loc_invert): localize a hidden source by inverting a frozen FLAC.

For each held-out query RIR the driver enumerates the room's metadata-declared
candidate sources, regenerates one RIR per candidate under common random numbers,
scores each against the observed RIR in AGREE's audio embedding space, and
predicts the argmax candidate. It is an evaluation driver only: no training, no
model edits, and every protocol quantity it applies is registered in
``worklog/worklog_yixun/exp_18_loc_invert_claude/plan_loc_invert.md`` (Rev 3 §2).

Reuse boundary (plan §4.5): identity/fingerprint/provenance/integrity helpers are
imported from ``eval_FLAC`` and the candidate, scoring and scorer-readout logic
from ``src.localization`` -- this file adds the protocol wiring around them and
duplicates none of it. The model build, EMA remap and sampling follow
``eval_FLAC.evaluate_model``'s lines of record; ``parity_check_one_query`` is the
standing proof of that (C8).
"""
import argparse
import contextlib
import copy
import hashlib
import itertools
import json
import math
import subprocess
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio

from eval_FLAC import (CONTEXT_ID_PRECISION, canonical_stream_hash, check_load_integrity,
                       orbit_provenance,
                       resolve_are_from_checkpoint, resolve_cond_autocast,
                       resolve_weights_source, sample_context_ids, sample_target_id, source_sha)
from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.inference.sampling import sample_discrete_euler
from src.models.factory import create_model_from_config
from src.training.diffusion import invariant_conditioning
from src.training.factory import create_training_wrapper_from_config
from src.data.dataset import create_dataloader_from_config, get_audio_filenames
from src.localization.agree_embed import MAX_LEN, embed_rirs, load_agree_audio, sha256_file
from src.localization.reaggregate import (decode_scores, decode_sims, encode_sims,
                                          reaggregate)
from src.localization.candidates import (CandidateSet, assert_gt_matches_loader,
                                         build_candidate_set, candidate_metadata,
                                         crosscheck_sources_vs_files,
                                         enumerate_metadata_sources, find_pair_metadata,
                                         parse_ir_filename, project_to_camera)
from src.localization.scoring import (DEFAULT_RADII, aggregate, clustered_bootstrap_ci,
                                      context_conditioned_baseline, cosine_sims,
                                      localization_error, nearest_context_baseline, noise_key,
                                      paired_room_clustered_test, power_statistic, predict_index,
                                      summarize, uniform_baseline)


def _identities(filenames, root_paths):
    """``'<position>|<relpath>'`` per file, mirroring SampleDataset's own relpath
    derivation (every root that is a substring is applied in order, last wins)."""
    identities = []
    for idx, filename in enumerate(filenames):
        relpath = None
        for root_path in root_paths:
            if root_path in filename:
                relpath = os.path.relpath(filename, root_path)
        identities.append(f"{idx}|{relpath if relpath is not None else filename}")
    return identities


def expected_split_identities_from_config(dataset_config):
    """The registered enumeration, derived from the SPLIT JSON and folder layout.

    Independent of the dataset object being audited (r3 review finding 1): an
    expectation read off ``loader.dataset`` proves only that the object agrees
    with itself. This rebuilds the file list the way ``SampleDataset`` builds it
    -- ``get_audio_filenames`` on the same config values, concatenated in config
    order -- so the audit compares the run against the SPLIT.
    """
    roots, filenames = [], []
    for audio_dir in dataset_config.get("datasets", None) or []:
        path = audio_dir["path"]
        roots.append(path)
        filenames.extend(get_audio_filenames(
            paths=path, keywords=None, json_file_path=audio_dir.get("json_file_path"),
            folder_name=audio_dir.get("folder_name")))
    return _identities(filenames, roots)


def assert_scored_stream(scored, expected):
    """End-of-run gate: the scored stream IS the registered split; returns its hash."""
    scored, expected = list(scored), list(expected)
    if len(scored) != len(expected):
        raise SystemExit(
            f"identity gate ABORT: scored {len(scored)} queries, the split declares "
            f"{len(expected)}; a truncated or over-long run must not be summarized")
    scored_rooms = {room_id_from_relpath(identity.split("|", 1)[1]) for identity in scored}
    expected_rooms = {room_id_from_relpath(identity.split("|", 1)[1]) for identity in expected}
    if scored_rooms != expected_rooms:
        raise SystemExit(
            f"identity gate ABORT: scored rooms {sorted(scored_rooms)} != split rooms "
            f"{sorted(expected_rooms)}")
    return split_hash(scored)


def expected_split_identities(dataset):
    """The identities a split declares, in dataset order, without loading audio.

    Mirrors ``SampleDataset.__getitem__``'s own ``relpath`` derivation (every root
    that is a substring of the filename is applied in order, last one winning) so
    the expectation is built from the file list alone -- an audit that re-derived
    its expectation from the loaded stream would prove nothing.
    """
    return _identities(dataset.filenames, dataset.root_paths)


def split_hash(identities):
    """sha256 over the ordered identity list, LF-joined and UTF-8 encoded."""
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def _iter_metadata(source):
    """Yield metadata dicts from a dataloader (``(reals, [md, ...])``) or an iterable."""
    for item in source:
        if isinstance(item, dict):
            yield item
        else:
            _reals, metadata = item
            for md in metadata:
                yield md


def audit_split_identities(source, expected):
    """Fail-closed identity audit (plan §2.1); returns the split hash.

    ``SampleDataset`` silently substitutes a random other item when a file fails
    to load or is silent, and the substitute carries its own idx/relpath. The walk
    therefore compares ``sample_target_id`` position by position against the
    expectation and aborts with ``SystemExit`` at the FIRST mismatch -- including a
    stream that is longer or shorter than the split.
    """
    expected = list(expected)
    observed = []
    for position, md in enumerate(_iter_metadata(source)):
        if position >= len(expected):
            raise SystemExit(
                f"identity audit ABORT: stream is longer than the split "
                f"({position + 1} > {len(expected)} items)")
        identity = sample_target_id(md)
        if identity != expected[position]:
            raise SystemExit(
                f"identity audit ABORT at position {position}: expected "
                f"{expected[position]!r}, got {identity!r} (silent substitution?)")
        observed.append(identity)
    if len(observed) != len(expected):
        raise SystemExit(
            f"identity audit ABORT: stream is shorter than the split "
            f"({len(observed)} < {len(expected)} items)")
    return split_hash(observed)


def build_noise_bank(seed, query_id, num_samples, latent_shape, device="cpu"):
    """The query's K latent noise draws, keyed by ``(seed, query_id, k)`` (plan §2.3).

    One dedicated ``torch.Generator`` per draw -- never the global stream, which
    the conditioner/dataloader also advance -- so the bank is identical on every
    machine, survives a resume, and is SHARED across the query's candidates
    (common random numbers, C10). Draws are made on CPU and then moved, exactly as
    ``evaluate_model`` does, so the bank does not depend on the compute device.
    """
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError(f"num_samples (K) must be >= 1, got {num_samples}")
    shape = tuple(int(s) for s in latent_shape)
    if len(shape) != 2 or any(s < 1 for s in shape):
        raise ValueError(f"latent_shape must be [channels, samples] with positive dims, got {shape}")

    draws = []
    for k in range(num_samples):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_key(seed, query_id, k))
        draws.append(torch.randn(shape, generator=generator))
    return torch.stack(draws).to(device)


@dataclass
class Engine:
    """The generation + scoring stack, as callables.

    Keeping the stack behind one seam is what makes the query layout testable
    without a GPU: a run builds it from the real module (``build_engine``), the
    tests build it from recording fakes. ``sampler`` receives ``(noise,
    cond_inputs)`` and returns latents; ``decoder`` maps latents to waveforms;
    ``embedder`` maps waveforms to L2-normalized embeddings.
    """
    device: str
    io_channels: int
    latent_samples: int
    conditioner: object
    cond_inputs_fn: object
    sampler: object
    decoder: object
    embedder: object


def _expand_cond_inputs(cond_inputs, rows_of_candidate):
    """Select one conditioning row per generated row (candidate-major)."""
    expanded = {}
    for key, value in cond_inputs.items():
        if value is None:
            expanded[key] = None
        elif torch.is_tensor(value):
            expanded[key] = value.index_select(0, rows_of_candidate)
        else:
            raise ValueError(f"conditioning input {key!r} is neither a tensor nor None")
    return expanded


def candidate_camera_positions(cand_set):
    """The candidate set's world positions in the receiver's camera frame ``[M, 3]``."""
    return np.stack([project_to_camera(cand_set.rec_loc, xyz) for xyz in cand_set.xyz_world])


def _resolve_cuda_index(device):
    """The CUDA index a device string names, or ``None`` off CUDA."""
    name = str(device)
    if not name.startswith("cuda") or not torch.cuda.is_available():
        return None
    return int(name.split(":", 1)[1]) if ":" in name else torch.cuda.current_device()


def _sync(device):
    """Drain the REQUESTED device (r4 review H1).

    ``torch.cuda.synchronize()`` without an index waits on the current device,
    which for an ``R0`` pinned to GPU 1 could return while its work is still
    queued -- reporting a fraction of the real time.
    """
    index = _resolve_cuda_index(device)
    if index is not None:
        torch.cuda.synchronize(index)


@contextlib.contextmanager
def _timed(timings, name, device):
    """Time one component with a LEADING and a TRAILING synchronization.

    Leading, so a previous query's outstanding work is not billed to this
    interval; trailing, so the interval actually contains the work it names --
    scoring used to stop its clock before the ``.cpu()`` that does the waiting.
    """
    _sync(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started)


def reset_peak_memory(device):
    """Start the peak-memory measurement for this run (R0's fit probe)."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device if ":" in str(device) else None)


def read_peak_memory(device):
    """Peak allocated bytes since the reset, or ``None`` off CUDA."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device if ":" in str(device) else None))
    return None


PROBE_COMPONENTS = ("conditioning", "sampling", "decode", "embed", "scoring",
                    "context")
#: separately measured whole-query wall time (not a component sum).
PROBE_WALL = "total_wall"


def probe_summary(rows, peak_memory_bytes):
    """Aggregate the per-query component timings (plan Rev 3.1 §3).

    R0's registered fit/timing probe is just a smoke run's summary, so the
    instrumentation is always on rather than living in a separate script.
    """
    timed = [row["timings_s"] for row in rows if row.get("timings_s")]
    if not timed:
        return None
    components = {}
    for name in PROBE_COMPONENTS:
        values = np.asarray([float(t.get(name, 0.0)) for t in timed], dtype=np.float64)
        components[name] = {"mean": float(values.mean()),
                            "p50": float(np.percentile(values, 50, method="linear")),
                            "p95": float(np.percentile(values, 95, method="linear"))}
    def _block(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mean": float(values.mean()),
                "p50": float(np.percentile(values, 50, method="linear")),
                "p95": float(np.percentile(values, 95, method="linear"))}

    totals = [sum(float(t.get(name, 0.0)) for name in PROBE_COMPONENTS) for t in timed]
    walls = [float(t[PROBE_WALL]) for t in timed if t.get(PROBE_WALL) is not None]
    return {"n_queries": len(timed), "peak_memory_bytes": peak_memory_bytes,
            "components": components, "total_s": _block(totals),
            "total_wall_s": _block(walls) if walls else None}


def run_query(engine, base_md, cand_set, noise, obs_wav, batch_size=64, control="none",
              return_wavs=False):
    """Score every candidate of one query: sims ``[M, K]``.

    Layout is candidate-major -- row ``m * K + k`` is candidate ``m`` with the
    query's noise draw ``k`` -- so all candidates share the same K draws (common
    random numbers, C10) and per-candidate results cannot depend on the batching.
    The conditioner runs ONCE over the M candidate metadata dicts (plan §2.3), the
    GT geometry invariant is checked against the loader's own metadata before any
    generation (O6), and ``control='constant_source'`` freezes every candidate's
    CONDITIONING at the candidate centroid (§2.8.1) so a working pipeline must
    collapse to the context-conditioned baseline.

    The two position arrays are kept apart on purpose (r3 review finding 3): the
    candidate geometry is what rows, context membership and the baselines are
    computed from and is never modified, while only the conditioning positions are
    substituted by the control. Overwriting the geometry would make every
    candidate look absent from the context and invalidate the very comparison the
    control exists to make.
    """
    if control not in ("none", "constant_source"):
        raise ValueError(f"unknown control {control!r} (expected 'none' or 'constant_source')")
    assert_gt_matches_loader(cand_set, base_md)

    candidate_positions = candidate_camera_positions(cand_set)
    conditioning_positions = candidate_positions
    if control == "constant_source":
        conditioning_positions = np.repeat(
            candidate_positions.mean(axis=0, keepdims=True), candidate_positions.shape[0], axis=0)

    num_candidates = candidate_positions.shape[0]
    num_samples = int(noise.shape[0])
    metadata = [candidate_metadata(base_md, conditioning_positions[m])
                for m in range(num_candidates)]

    timings = {name: 0.0 for name in PROBE_COMPONENTS if name != "context"}
    with _timed(timings, "conditioning", engine.device):
        conditioning = engine.conditioner(metadata, engine.device)
        cond_inputs = engine.cond_inputs_fn(conditioning)

    total_rows = num_candidates * num_samples
    rows = torch.arange(total_rows)
    candidate_of_row = torch.div(rows, num_samples, rounding_mode="floor")
    sample_of_row = rows % num_samples

    wavs = []
    for start in range(0, total_rows, max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), total_rows)
        chunk_noise = noise.index_select(0, sample_of_row[start:stop]).to(engine.device)
        chunk_cond = _expand_cond_inputs(cond_inputs, candidate_of_row[start:stop].to(engine.device))
        with _timed(timings, "sampling", engine.device):
            latents = engine.sampler(chunk_noise, chunk_cond)
        with _timed(timings, "decode", engine.device):
            wavs.append(engine.decoder(latents).clamp(-1.0, 1.0))
    wavs = torch.cat(wavs, dim=0)

    with _timed(timings, "embed", engine.device):
        embeddings = engine.embedder(wavs)
        obs_embedding = engine.embedder(obs_wav.to(engine.device))[0]
    with _timed(timings, "scoring", engine.device):
        # the .cpu() transfer is the actual wait: it belongs INSIDE this interval
        sims = cosine_sims(obs_embedding,
                           embeddings.reshape(num_candidates, num_samples, -1)).float().cpu()

    out = {"sims": sims, "cand_cam_xyz": candidate_positions,
           "conditioning_xyz_cam": conditioning_positions, "control": control,
           "num_candidates": num_candidates, "num_samples": num_samples,
           "timings_s": timings}
    if return_wavs:
        out["wavs"] = wavs
    return out


def room_id_from_relpath(relpath):
    """``'<scene>/<scene_id>'`` -- the 17-room key of plan §2.6."""
    parts = str(relpath).replace(os.sep, "/").strip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"relpath must contain <scene>/<scene_id>/<file>: {relpath!r}")
    return f"{parts[-3]}/{parts[-2]}"


def render_position_id(xyz):
    """One candidate position in ``sample_context_ids``' rendering.

    Same rule, deliberately: float32 (what the loader's ``context_poses`` carry),
    ``CONTEXT_ID_PRECISION`` decimals, ``-0.0`` normalized. The candidate's
    camera-frame position is bit-identical to the loader's projection of the same
    source, so equal positions render to equal strings.
    """
    values = []
    for value in np.asarray(xyz, dtype=np.float32).reshape(-1):
        value = float(value)
        if value == 0.0:
            value = 0.0
        values.append(f"{value:.{CONTEXT_ID_PRECISION}f}")
    return ",".join(values)


def context_membership_mask(cand_cam_xyz, context_ids, gt_index=None):
    """Which candidates the conditioning already reveals (plan §2.2, C1).

    Fail-closed (r3 review finding 7). A set-membership test would silently treat
    an unmatched context source as "not a candidate" and thereby ENLARGE the
    eligible set, which inflates the headroom of the information-matched
    baseline -- the registered comparison target. So: candidate fingerprints must
    be unique, every ordered context id must resolve to exactly one candidate, and
    the GT may never be a context member (the context is drawn from OTHER sources
    by construction). Repeated context ids are allowed: ``AR_md`` falls back to
    ``np.random.choice(replace=True)`` in rooms with too few sources.
    """
    positions = np.asarray(cand_cam_xyz)
    index_of = {}
    for index, xyz in enumerate(positions):
        fingerprint = render_position_id(xyz)
        if fingerprint in index_of:
            raise ValueError(
                f"candidates {index_of[fingerprint]} and {index} render to the same position "
                f"fingerprint {fingerprint!r}; candidate identity is ambiguous")
        index_of[fingerprint] = index

    mask = [False] * len(positions)
    for position, context_id in enumerate(context_ids):
        if context_id not in index_of:
            raise ValueError(
                f"context source {position} ({context_id!r}) matches no candidate of this query; "
                "the conditioning and the candidate set disagree about the geometry")
        mask[index_of[context_id]] = True

    if gt_index is not None and mask[int(gt_index)]:
        raise ValueError(
            f"GT candidate {int(gt_index)} is a context member; context is drawn from OTHER "
            "sources by construction, so this query's conditioning already reveals the answer")
    return mask


def gt_reciprocal_rank(scores, gt_index, available=None):
    """Reciprocal rank of the GT candidate, ties broken by lowest index.

    Ranked over the AVAILABLE candidates only: in ``gt_rir`` mode a candidate
    without a measured file carries a placeholder score, and letting a placeholder
    out-rank the GT would understate the oracle (r3 review finding 8).
    """
    scores = torch.as_tensor(scores).reshape(-1)
    gt_index = int(gt_index)
    if available is not None:
        keep = [i for i, flag in enumerate(available) if flag]
        if gt_index not in keep:
            raise ValueError(f"GT candidate {gt_index} is not available; its rank is undefined")
        gt_index, scores = keep.index(gt_index), scores[torch.tensor(keep, dtype=torch.long)]
    gt_score = scores[gt_index]
    better = int((scores > gt_score).sum())
    tied_before = int((scores[:gt_index] == gt_score).sum())
    return 1.0 / (better + tied_before + 1)


def build_row(query_id, room_id, relpath, receiver_node, cand_set, cam_xyz, sims,
              context_mask, noise_keys, tau, agg, control, score_source, smoke,
              available=None, identity_index=None, substituted=False,
              context_xyz_cam=None, context_sims_hex=None, timings=None):
    """One JSONL query record: raw evidence first, derived quantities alongside.

    ``sims_hex``/``scores_hex`` are the exact float32 values (O18), so the
    aggregation, prediction and error can all be re-derived offline. The
    prediction is taken over the AVAILABLE candidates only -- a measured RIR that
    does not exist shrinks the oracle's eligibility, never the candidate set --
    and the eligible-set size counts the non-context candidates, the
    information-matched comparison target (C1).
    """
    sims = sims.detach().cpu().float()
    num_candidates = int(sims.shape[0])
    available = [True] * num_candidates if available is None else [bool(a) for a in available]
    if len(available) != num_candidates:
        raise ValueError(f"available has {len(available)} entries for {num_candidates} candidates")
    usable = [i for i, flag in enumerate(available) if flag]
    if not usable:
        raise ValueError("no candidate is available; nothing could be predicted")

    scores = aggregate(sims, agg, tau if agg == "lme" else None)
    usable_index = torch.tensor(usable, dtype=torch.long)
    pred_index = usable[predict_index(scores.index_select(0, usable_index))]
    gt_index = cand_set.gt_index
    error = localization_error(cand_set.xyz_world[pred_index], cand_set.gt_xyz)
    context_mask = [bool(m) for m in context_mask]

    row = {
        "query_id": query_id,
        "room_id": room_id,
        "relpath": relpath,
        "receiver_node": int(receiver_node),
        "gt_node": int(cand_set.gt_node),
        "gt_index": int(gt_index),
        "gt_xyz_world": [float(v) for v in cand_set.gt_xyz],
        "gt_xyz_cam": [float(v) for v in np.asarray(cam_xyz)[gt_index]],
        "candidate_nodes": [int(n) for n in cand_set.nodes],
        "candidate_xyz_world": [[float(v) for v in row] for row in cand_set.xyz_world],
        "candidate_xyz_cam": [[float(v) for v in row] for row in np.asarray(cam_xyz)],
        "context_member": context_mask,
        "candidate_available": available,
        "n_candidates": num_candidates,
        "n_samples": int(sims.shape[1]),
        "n_eligible": int(sum(1 for m in context_mask if not m)),
        "n_available": len(usable),
        "gt_only": int(sum(1 for m in context_mask if not m)) == 1,
        "sims_hex": encode_sims(sims),
        "scores_hex": encode_sims(scores.unsqueeze(0))[0],
        "noise_keys": [int(k) for k in noise_keys],
        "pred_index": int(pred_index),
        "pred_node": int(cand_set.nodes[pred_index]),
        "pred_xyz_world": [float(v) for v in cand_set.xyz_world[pred_index]],
        "e_loc": float(error),
        "top1": 1.0 if pred_index == gt_index else 0.0,
        "rr": gt_reciprocal_rank(scores, gt_index, available=available),
        "power_statistic": (float(power_statistic(sims))
                            if sims.shape[0] > 1 and sims.shape[1] > 1 else None),
        "tau": float(tau) if tau is not None else None,
        "agg": agg,
        "control": control,
        "score_source": score_source,
        "identity_index": None if identity_index is None else int(identity_index),
        "substituted": bool(substituted),
        "smoke": bool(smoke),
        "timings_s": (None if timings is None
                      else {**{name: float(timings.get(name, 0.0)) for name in PROBE_COMPONENTS},
                            PROBE_WALL: float(timings.get(PROBE_WALL, 0.0))}),
    }
    if context_xyz_cam is not None and context_sims_hex is not None:
        # optional evidence for the non-generative control (O10)
        row["context_xyz_cam"] = [[float(v) for v in xyz] for xyz in context_xyz_cam]
        row["context_sims_hex"] = list(context_sims_hex)
    return row


def write_row(handle, row):
    """Append one JSONL row and flush it: a killed run keeps every finished query."""
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def read_rows(path):
    """Read a JSONL row file back."""
    rows = []
    with open(str(path), "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _query_record(row, extra=None):
    record = {"query_id": row["query_id"], "room_id": row["room_id"], "e_loc": row["e_loc"],
              "top1": row["top1"], "rr": row["rr"]}
    if extra:
        record.update(extra)
    return record


def _baseline_records(rows, kind, radii):
    """Per-query baseline expectations as summarize-shaped records.

    The whole per-candidate distance list is carried, so :func:`summarize` applies
    the same 1/M within-query weighting it applies to a FLAC row (C1/C3). ``rr`` is
    deliberately absent: a random guesser has no ranking, and inventing one would
    be a protocol invention rather than a measurement.
    """
    records = []
    for row in rows:
        candidates = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
        gt = np.asarray(row["gt_xyz_world"], dtype=np.float64)
        if kind == "uniform":
            out = uniform_baseline(candidates, gt, radii=radii)
        else:
            out = context_conditioned_baseline(candidates, gt, row["context_member"], radii=radii)
        records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                        "distances": out["distances"], "top1": out["top1"]})
    return records


def _nearest_context_records(rows, masked):
    """Records for the non-generative control (O10), or ``None`` if the rows do
    not carry the observed-vs-context similarities it needs."""
    records = []
    for row in rows:
        if "context_xyz_cam" not in row or "context_sims_hex" not in row:
            return None
        cand_cam = np.asarray(row["candidate_xyz_cam"], dtype=np.float64)
        ctx_cam = np.asarray(row["context_xyz_cam"], dtype=np.float64)
        sims = decode_scores(row["context_sims_hex"])
        eligible = [not m for m in row["context_member"]] if masked else None
        pred = nearest_context_baseline(cand_cam, ctx_cam, sims, eligible_mask=eligible)
        candidates = np.asarray(row["candidate_xyz_world"], dtype=np.float64)
        records.append({
            "query_id": row["query_id"], "room_id": row["room_id"],
            "e_loc": localization_error(candidates[pred], np.asarray(row["gt_xyz_world"])),
            "top1": 1.0 if pred == row["gt_index"] else 0.0})
    return records


def summarize_run(rows, radii=DEFAULT_RADII, bootstrap_n=10000, bootstrap_seed=0):
    """Aggregate JSONL rows into the run summary (plan §2.6).

    Everything statistical goes through ``scoring.summarize`` -- the FLAC rows and
    both random baselines under identical weighting and boundary conventions -- so
    a comparison can never be an artifact of two aggregation paths. The
    zero-headroom (GT-only eligible set) queries are reported separately AND
    excluded from a second information-matched block, as registered.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("summarize_run needs at least one row")

    eligible_sizes = [int(row["n_eligible"]) for row in rows]
    histogram = {}
    for size in eligible_sizes:
        histogram[str(size)] = histogram.get(str(size), 0) + 1
    gt_only_rows = [row for row in rows if row["gt_only"]]
    kept = [row for row in rows if not row["gt_only"]]
    context_predictions = [1.0 if row["context_member"][row["pred_index"]] else 0.0 for row in rows]

    raw_control = _nearest_context_records(rows, masked=False)
    masked_control = _nearest_context_records(rows, masked=True)
    kept_masked_control = _nearest_context_records(kept, masked=True) if kept else None

    # Statistics (plan §2.6/C3/O12). The paired tests compare FLAC to the
    # information-matched baseline and to the non-generative control on the SAME
    # queries; the CI is the 17-room clustered interval on the pooled median.
    flac_points = [{"query_id": row["query_id"], "room_id": row["room_id"],
                    "e_loc": row["e_loc"]} for row in rows]
    context_points = [{"query_id": rec["query_id"], "room_id": rec["room_id"],
                       "distances": rec["distances"]}
                      for rec in _baseline_records(rows, "context", radii)]
    statistics = {
        "clustered_ci": clustered_bootstrap_ci(flac_points, n=bootstrap_n, seed=bootstrap_seed),
        "paired_vs_context_conditioned": paired_room_clustered_test(
            flac_points, context_points, n=bootstrap_n, seed=bootstrap_seed),
        "paired_vs_nearest_context_masked": (
            paired_room_clustered_test(
                flac_points,
                [{"query_id": rec["query_id"], "room_id": rec["room_id"], "e_loc": rec["e_loc"]}
                 for rec in masked_control],
                n=bootstrap_n, seed=bootstrap_seed) if masked_control else None),
    }

    power_values = [row["power_statistic"] for row in rows
                    if row.get("power_statistic") is not None]
    power_block = None
    if power_values:
        power_block = {"n_queries": len(power_values),
                       "mean": float(np.mean(power_values)),
                       "median": float(np.median(power_values)),
                       "min": float(np.min(power_values)),
                       "max": float(np.max(power_values))}

    return {
        "flac": summarize([_query_record(row) for row in rows], radii=radii),
        "flac_excl_gt_only": (summarize([_query_record(row) for row in kept], radii=radii)
                              if kept else None),
        "statistics": statistics,
        "power_statistic": power_block,
        "baselines": {
            "uniform": summarize(_baseline_records(rows, "uniform", radii), radii=radii),
            "context_conditioned": summarize(
                _baseline_records(rows, "context", radii), radii=radii),
            "context_conditioned_excl_gt_only": (
                summarize(_baseline_records(kept, "context", radii), radii=radii) if kept else None),
        },
        "controls": {
            "nearest_context_raw": summarize(raw_control, radii=radii) if raw_control else None,
            "nearest_context_masked": (
                summarize(masked_control, radii=radii) if masked_control else None),
            "nearest_context_masked_excl_gt_only": (
                summarize(kept_masked_control, radii=radii) if kept_masked_control else None),
        },
        "eligible_set_sizes": {
            "histogram": histogram,
            "min": min(eligible_sizes),
            "max": max(eligible_sizes),
            "mean": float(np.mean(eligible_sizes)),
        },
        "gt_only": {
            "n_queries": len(gt_only_rows),
            "rooms": sorted({row["room_id"] for row in gt_only_rows}),
        },
        "context_member_prediction_rate": float(np.mean(context_predictions)),
        "n_queries": len(rows),
        "n_rooms": len({row["room_id"] for row in rows}),
    }


def _file_sha256(path):
    """sha256 of a config FILE's contents (a path alone proves nothing, O17)."""
    try:
        return sha256_file(path)
    except (OSError, TypeError):
        return "n/a"


def _flash_attn_available():
    import importlib.util
    return importlib.util.find_spec("flash_attn") is not None


def _package_version(name):
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "n/a"


def context_stream_digest(rows):
    """Digest of the ordered per-query context draw actually used (O8).

    The context sources are drawn per item inside the dataloader workers, so the
    only proof of what a run conditioned on is the ordered fingerprint stream the
    rows carry. Serialized through eval_FLAC's own canonical stream hash.
    """
    stream = [tuple(render_position_id(xyz) for xyz in row["context_xyz_cam"])
              for row in rows if "context_xyz_cam" in row]
    if not stream:
        return "n/a"
    return canonical_stream_hash(stream)


def is_registered_run(args, dataset_config):
    """A registered unseen generative run -- the only kind that produces a headline."""
    return (bool((dataset_config or {}).get("unseeneval", False))
            and args.score_source == "flac" and not args.smoke)


def assert_registration_sha(args, dataset_config):
    """O17: a registered unseen run must name BOTH its committed protocol manifest
    and the commit that carries it."""
    if is_registered_run(args, dataset_config):
        if not args.registration_sha:
            _refuse("--registration-sha is required for a registered unseen run (O17): the "
                    "parameter file must be committed BEFORE the run")
        if not args.registration_manifest:
            _refuse("--registration-manifest is required for a registered unseen run (O17): "
                    "the locked protocol must be machine-checkable, not a bare SHA")
    return args


def device_provenance(device):
    """Identify the device the run ACTUALLY used (full-review F6).

    ``device_name`` used to query CUDA index 0 regardless of the requested device,
    which is simply wrong for ``--device cuda:1`` and useless for cross-machine
    reproduction. The requested string, the resolved index, the name, the compute
    capability and the UUID (when the build exposes it) are all recorded.
    """
    requested = str(device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        index = int(requested.split(":", 1)[1]) if ":" in requested else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return {"device_requested": requested, "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "device_capability": list(torch.cuda.get_device_capability(index)),
                "device_uuid": str(getattr(properties, "uuid", "n/a"))}
    return {"device_requested": requested, "device_index": None, "device_name": "cpu",
            "device_capability": None, "device_uuid": "n/a"}


def build_provenance(args, ckpt_sha256, agree_sha256, split_hash, weights_source, n_queries,
                     dataset_config=None, context_digest=None, candidate_manifest_sha256=None):
    """Everything needed to reproduce or falsify the run, in one record.

    Announcement 05: a row is only interpretable next to the protocol it was
    produced under, so every pinned quantity is written down -- including the
    dataloader parallelism, which the per-item context draw depends on (O8), the
    resolved weights source (O9) and the scorer readout (§2.4).
    """
    orbit_execution, frame_avg_cap = orbit_provenance(args.cond_method)
    angles = "n/a" if args.cond_method != "fa_invariant" else (
        None if args.frame_avg_angles is None else [float(a) for a in args.frame_avg_angles])
    return {
        "experiment": "exp_18_loc_invert",
        "source_sha": source_sha(),
        "model_config": args.model_config,
        "dataset_config": args.dataset_config,
        "ckpt_path": args.ckpt_path,
        "ckpt_sha256": ckpt_sha256,
        "agree_ckpt": args.agree_ckpt,
        "agree_sha256": agree_sha256,
        "split_hash": split_hash,
        "n_queries": int(n_queries),
        "weights_source": weights_source,
        "readout": "mean",
        "num_samples": effective_num_samples(args),
        "tau": float(args.tau) if args.tau is not None else None,
        "agg": args.agg,
        "steps": int(args.steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
        "cond_method": args.cond_method,
        "frame_avg_angles": angles,
        "rotate_deg": float(args.rotate_deg),
        "cond_autocast": args.cond_autocast,
        "orbit_execution": orbit_execution,
        "frame_avg_fwd_cap": frame_avg_cap,
        "score_source": args.score_source,
        "control": args.control,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "smoke": bool(args.smoke),
        "max_queries": None if args.max_queries is None else int(args.max_queries),
        "eval_name": args.eval_name,
        "registration_sha": args.registration_sha or "n/a",
        "model_config_sha256": _file_sha256(args.model_config),
        "dataset_config_sha256": _file_sha256(args.dataset_config),
        "context_k": (((dataset_config or {}).get("modalities") or {})
                      .get("acoustic_context", {}) or {}).get("max_context"),
        "loader_shuffle": False,
        "loader_drop_last": bool((dataset_config or {}).get("drop_last", True)),
        "context_stream_digest": context_digest or "n/a",
        "candidate_manifest_sha256": candidate_manifest_sha256 or "n/a",
        **device_provenance(getattr(args, "device", "cpu")),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "torchaudio_version": _package_version("torchaudio"),
        "transformers_version": _package_version("transformers"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "flash_attn_available": _flash_attn_available(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def effective_num_samples(args):
    """K actually used: the oracle scores exactly one measured RIR per candidate."""
    return 1 if args.score_source == "gt_rir" else int(args.num_samples)


def artifact_stem(args):
    """A file stem that names the CELL, not just the run (full-review F5).

    Diagnostic cells differing only in control mode, autocast, aggregation, tau or
    scorer used to collide and overwrite one another. Every cell-defining field is
    in the name, so two different protocols can never share a path.
    """
    scorer = os.path.splitext(os.path.basename(str(args.agree_ckpt)))[0] or "none"
    marker = ("smoke" if args.smoke
              else ("registered" if getattr(args, "registration_manifest", None) else "dev"))
    parts = [str(args.eval_name), args.score_source, f"ctl-{args.control}", args.cond_method,
             f"ac-{args.cond_autocast}", args.agg]
    if args.agg == "lme" and args.tau is not None:
        parts.append(f"tau{float(args.tau):g}")
    parts += [f"K{effective_num_samples(args)}", f"seed{int(args.seed)}", f"scorer-{scorer}",
              marker]
    return "_".join(parts)


def artifact_paths(args, overwrite=None):
    """``{rows, summary, manifest}`` paths for this cell; refuses to clobber.

    ``os.replace`` would silently overwrite a finished artifact, and a leftover
    ``.partial`` means an earlier run of this exact cell died -- both are refused
    unless ``--overwrite`` is given explicitly.
    """
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = artifact_stem(args)
    paths = {"rows": os.path.join(out_dir, f"{stem}_rows.jsonl"),
             "summary": os.path.join(out_dir, f"{stem}_summary.json"),
             "manifest": os.path.join(out_dir, f"{stem}_manifest.json")}
    if overwrite is None:
        overwrite = bool(getattr(args, "overwrite", False))
    if not overwrite:
        for kind, path in paths.items():
            for candidate in (path, path + ".partial"):
                if os.path.exists(candidate):
                    _refuse(f"{kind} target already exists: {candidate}. This cell has already "
                            "been run (or died mid-run); pass --overwrite to replace it")
    return paths


def jsonable(obj):
    """Normalize a summary for JSON: numeric dict keys become strings explicitly.

    ``success`` is keyed by radius (a float), and json would coerce those keys
    silently. Doing it here makes the on-disk schema a stated contract -- readers
    look up ``success["1.0"]`` -- instead of an encoder side effect.
    """
    if isinstance(obj, dict):
        return {(str(k) if isinstance(k, (int, float)) else k): jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_summary(path, summary, provenance, overwrite=True):
    """Write the summary record (provenance first, then the aggregates).

    ``overwrite`` defaults to True because the run mode already refused a
    colliding target in :func:`artifact_paths`; the atomic write is what matters
    here.
    """
    return write_json_atomic(path, {"provenance": jsonable(provenance),
                                    "summary": jsonable(summary)}, overwrite=overwrite)


AR_SAMPLE_RATE = 22050


def measured_rir_paths(room_wav_dir, cand_set, receiver_node):
    """Per candidate, the measured ``S<cand>_R<rec>`` file, or ``None`` if absent.

    Matched on parsed numeric identity over the directory listing, never on a
    reconstructed name: the wav namespace pads node ids inconsistently.
    """
    room_wav_dir = str(room_wav_dir)
    by_source = {}
    if os.path.isdir(room_wav_dir):
        for fname in sorted(os.listdir(room_wav_dir)):
            try:
                src, rec = parse_ir_filename(fname)
            except ValueError:
                continue
            if rec != int(receiver_node):
                continue
            if src in by_source:
                raise ValueError(
                    f"two files claim S{src}_R{receiver_node} in {room_wav_dir}: "
                    f"{os.path.basename(by_source[src])} and {fname}; which RIR the oracle "
                    "scored would be unresolvable")
            by_source[src] = os.path.join(room_wav_dir, fname)
    return [by_source.get(int(node)) for node in cand_set.nodes]


def load_measured_rirs(room_wav_dir, cand_set, receiver_node):
    """``(wavs [A, 1, MAX_LEN], available [M], paths [M])`` for the oracle mode.

    Only the files that exist are loaded: a missing pair shrinks what the oracle
    can report, never the candidate set (plan §2.2). Each file is clamped and
    cropped/padded to ``MAX_LEN`` -- the metric route's own window -- before the
    shared preprocessing runs.
    """
    paths = measured_rir_paths(room_wav_dir, cand_set, receiver_node)
    available = [path is not None for path in paths]
    wavs = []
    for path in paths:
        if path is None:
            continue
        wav, rate = torchaudio.load(path)
        if rate != AR_SAMPLE_RATE:
            raise ValueError(f"{path}: IR sampling rate must be {AR_SAMPLE_RATE}, got {rate}")
        wav = wav[:1, :MAX_LEN].clamp(-1.0, 1.0)
        if wav.shape[-1] < MAX_LEN:
            wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[-1]))
        wavs.append(wav.unsqueeze(0))
    if not wavs:
        raise ValueError(f"no measured RIR for receiver {receiver_node} in {room_wav_dir}")
    return torch.cat(wavs, dim=0), available, paths


def run_query_gt_rir(engine, cand_set, room_wav_dir, receiver_node, obs_wav):
    """Measured-RIR oracle (plan §2.6): score the real RIRs, generate nothing.

    Needs no FLAC checkpoint, so it runs the moment the dataset lands. Candidates
    whose measured file is missing get a placeholder similarity and are marked
    unavailable, so the row still carries the full candidate set while the
    prediction is taken over the available ones only.
    """
    timings = {name: 0.0 for name in PROBE_COMPONENTS if name != "context"}
    with _timed(timings, "decode", engine.device):         # file load stands in for decode
        wavs, available, _paths = load_measured_rirs(room_wav_dir, cand_set, receiver_node)
    with _timed(timings, "embed", engine.device):
        embeddings = engine.embedder(wavs.to(engine.device))
        obs_embedding = engine.embedder(obs_wav.to(engine.device))[0]
    with _timed(timings, "scoring", engine.device):
        present = cosine_sims(obs_embedding, embeddings.unsqueeze(1)).float().cpu()

    sims = torch.zeros(len(cand_set.nodes), 1, dtype=torch.float32)
    cursor = 0
    for index, flag in enumerate(available):
        if flag:
            sims[index, 0] = present[cursor, 0]
            cursor += 1
    gt_index = cand_set.gt_index
    if not available[gt_index]:
        raise ValueError(
            f"the identity (GT) candidate S{cand_set.gt_node}_R{receiver_node} has no measured "
            f"RIR in {room_wav_dir}; the oracle would be scoring a query whose own answer is "
            "not in the scored set")
    return {"sims": sims, "available": available, "cand_cam_xyz": candidate_camera_positions(cand_set),
            "identity_index": gt_index,
            "num_candidates": len(cand_set.nodes), "num_samples": 1, "timings_s": timings}


def _finite_angle(value):
    """argparse type: a frame-average angle must be a finite number."""
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"frame-average angle must be finite, got {value!r}")
    return number


def parse_args(argv=None):
    """CLI for the exp_18 driver; the registered defaults are the plan's §2.3 pins."""
    parser = argparse.ArgumentParser(
        description="exp_18 loc_invert: source localization by analysis-by-synthesis inversion")
    parser.add_argument("--mode", choices=["run", "readback", "scorer-noise", "reaggregate"],
                        default="run", help="run: score queries; readback: the R-1 "
                             "dataset gate; scorer-noise: the §2.8.3 measurement; "
                             "reaggregate: R1's offline sweep")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--agree-ckpt", default=None)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="K samples per candidate (required for --score-source flac)")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--agg", choices=["lme", "mean", "max"], default="lme")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-method", choices=["vanilla", "fa_invariant"], default="vanilla")
    parser.add_argument("--frame-avg-angles", type=_finite_angle, nargs="+", default=None)
    parser.add_argument("--rotate-deg", type=float, default=0.0)
    parser.add_argument("--cond-autocast", choices=["default", "bf16", "off"], default="default")
    parser.add_argument("--score-source", choices=["flac", "gt_rir"], default="flac")
    parser.add_argument("--control", choices=["none", "constant_source"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--out-dir", default="worklog/worklog_yixun/exp_18_loc_invert_claude")
    parser.add_argument("--eval-name", default="exp18_loc_invert")
    parser.add_argument("--rows", nargs="+", default=None,
                        help="rows JSONL file(s) for --mode reaggregate")
    parser.add_argument("--noise-draws", type=int, default=100,
                        help="sampled-readout draws for --mode scorer-noise (§2.8.3)")
    parser.add_argument("--noise-wavs", nargs="+", default=None,
                        help="explicit RIR files for --mode scorer-noise")
    parser.add_argument("--noise-wav-count", type=int, default=4,
                        help="how many split RIRs to measure when --noise-wavs is absent")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing artifact for this exact cell")
    parser.add_argument("--registration-manifest", default=None,
                        help="committed JSON manifest locking the registered protocol (O17)")
    parser.add_argument("--registration-sha", default=None,
                        help="commit SHA of the pre-registered params file (O17); "
                             "required for a registered unseen generative run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--parity-check", action="store_true",
                        help="run the one-query eval_FLAC parity harness and exit (C8)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"exp_18 loc_invert REFUSED: {message}")


def _finite_flag(value, name):
    number = float(value)
    if not math.isfinite(number):
        _refuse(f"{name} must be a finite number, got {value}")
    return number


def validate_args(args):
    """Every fail-closed rule that can be checked before touching a file or a GPU."""
    if args.mode != "run":
        # Auxiliary modes score nothing generative; only the run mode needs the
        # generative protocol flags to be complete.
        if args.mode == "reaggregate" and not args.rows:
            _refuse("--rows is required for --mode reaggregate")
        if args.mode == "scorer-noise":
            if not args.agree_ckpt:
                _refuse("--agree-ckpt is required for --mode scorer-noise")
            if int(args.noise_draws) < 2:
                _refuse(f"--noise-draws must be >= 2, got {args.noise_draws}")
            if int(args.noise_wav_count) < 1:
                _refuse(f"--noise-wav-count must be >= 1, got {args.noise_wav_count}")
        return args
    if not args.agree_ckpt:
        _refuse("--agree-ckpt is required to score RIRs")
    _finite_flag(args.rotate_deg, "--rotate-deg")
    _finite_flag(args.cfg_scale, "--cfg-scale")
    if args.tau is not None:
        _finite_flag(args.tau, "--tau")
    if int(args.steps) < 1:
        _refuse(f"--steps must be >= 1, got {args.steps}")
    if int(args.num_workers) < 1:
        # create_dataloader_from_config hardcodes persistent_workers=True, which
        # torch refuses at num_workers=0; refuse here rather than crash in the loader.
        _refuse(f"--num-workers must be >= 1 (the release dataloader sets "
                f"persistent_workers=True), got {args.num_workers}")
    if args.max_queries is not None and int(args.max_queries) < 1:
        _refuse(f"--max-queries must be >= 1, got {args.max_queries}")
    if float(args.rotate_deg) != 0.0:
        _refuse(f"--rotate-deg {args.rotate_deg} is not implemented in this driver; the "
                "registered protocol is rotate_deg=0 (announcement 05)")
    if args.max_queries is not None and not args.smoke:
        _refuse("--max-queries truncates the split and is only allowed with --smoke, so a "
                "truncated run can never be mistaken for a headline artifact")
    if args.agg == "lme" and (args.tau is None or float(args.tau) <= 0.0):
        _refuse(f"--tau must be > 0 for --agg lme, got {args.tau}")
    if args.num_samples is not None and int(args.num_samples) < 1:
        _refuse(f"--num-samples (K) must be >= 1, got {args.num_samples}")
    if args.score_source == "flac" and args.num_samples is None:
        _refuse("--num-samples (K) is required for --score-source flac")
    if int(args.batch_size) < 1:
        _refuse(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.agg == "lme" and args.tau is not None and not math.isfinite(float(args.tau)):
        _refuse(f"--tau must be finite for --agg lme, got {args.tau}")
    if args.score_source == "gt_rir":
        for flag, value in (("--ckpt-path", args.ckpt_path),
                            ("--parity-check", args.parity_check),
                            ("--control constant_source", args.control == "constant_source")):
            if value:
                _refuse(f"{flag} is meaningless under --score-source gt_rir (no generation "
                        "happens); refusing rather than recording a protocol that did not run")
    if args.score_source == "flac" and not args.ckpt_path:
        _refuse("--ckpt-path is required to score generated RIRs; only --score-source gt_rir "
                "(the measured-RIR oracle) runs without a checkpoint")
    return args


def assert_rectified_flow(model_config):
    """The registered generator is rectified flow; anything else is another model."""
    objective = ((model_config or {}).get("model") or {}).get("diffusion", {}) \
        .get("diffusion_objective")
    if objective != "rectified_flow":
        _refuse(f"model config declares diffusion_objective={objective!r}; this driver "
                "evaluates the registered rectified-flow generator only")


def assert_no_are(embedded_model_config, file_model_config):
    """Refuse ARE checkpoints: the sampler emits a residual there, so its samples
    are not the quantity this protocol scores (C5/C8)."""
    try:
        are_lambda, source, _anchor = resolve_are_from_checkpoint(
            embedded_model_config, file_model_config, None)
    except ValueError as err:
        # ARE is in play but unresolvable (e.g. no embedded config to bind to).
        # Either way this driver will not run it; surface it as a startup refusal.
        _refuse(f"ARE checkpoint check failed: {err}")
    if are_lambda is not None:
        _refuse(f"the checkpoint declares ARE (lambda={are_lambda}, source={source}); exp_18 "
                "scores vanilla FLAC samples, not anchor residuals")


def prepare_state_dict(ckpt, training_config):
    """``(state_dict, weights_source)`` -- evaluate_model's EMA lines of record.

    Mirrors ``eval_FLAC.evaluate_model`` (eval_FLAC.py:1146-1167) exactly: strip
    the ``diffusion.`` prefix, resolve which weights will ACTUALLY be used (O9),
    fold ``diffusion_ema.ema_model.*`` in as ``model.*`` when the config asks for
    EMA and the checkpoint carries it, and flip ``use_ema`` off so the wrapper
    does not rebuild a shadow at eval time. Divergence here is a silent
    wrong-weights bug, which is why it is a separately tested function.
    """
    state_dict = ckpt["state_dict"]
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "")] = state_dict.pop(key)

    weights_source = resolve_weights_source(training_config, state_dict.keys())
    if (training_config or {}).get("use_ema", False) and any(
            k.startswith("diffusion_ema.ema_model.") for k in state_dict.keys()):
        print("Using EMA model")
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(key)
        training_config["use_ema"] = False
    return state_dict, weights_source


def build_engine(args, agree=None, device=None, model_config=None, ckpt=None):
    """Build the frozen generator and wrap it as an :class:`Engine`.

    Follows ``eval_FLAC.evaluate_model``'s lines of record (eval_FLAC.py:1134-1198)
    step for step -- matmul precision, config load, ARE refusal, state-dict remap,
    load-integrity check, wrapper construction, eval/no-grad, device move, latent
    length from the pretransform's downsampling ratio -- because any divergence
    would mean exp_18 scores a different generative process than the release
    evaluation does. :func:`parity_check_one_query` is the standing proof (C8).
    """
    device = device or getattr(args, "device", "cpu")
    torch.set_float32_matmul_precision("medium")

    if model_config is None:
        with open(args.model_config) as handle:
            model_config = json.load(handle)
    file_model_config = copy.deepcopy(model_config)
    assert_rectified_flow(model_config)

    training_config = model_config.get("training", None)
    if ckpt is None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
    # Earlier than evaluate_model does it: an ARE artifact must never reach a model build.
    assert_no_are(ckpt.get("model_config"), file_model_config)
    state_dict, weights_source = prepare_state_dict(ckpt, training_config)

    model_obj = create_model_from_config(model_config)
    missing, unexpected = model_obj.load_state_dict(state_dict, strict=False)
    check_load_integrity(missing, unexpected, False)

    model_config["training"] = training_config
    module = create_training_wrapper_from_config(model_config, model_obj)
    module.eval().requires_grad_(False)
    module.to(device)
    with torch.amp.autocast(device):
        model = module.diffusion.model

    if module.diffusion.pretransform is not None:
        latent_samples = model_config["sample_size"] // module.diffusion.pretransform.downsampling_ratio
    else:
        latent_samples = model_config["sample_size"]

    ac_enabled, ac_dtype = resolve_cond_autocast(args.cond_autocast)
    frame_avg_angles = tuple(
        float(a) for a in (args.frame_avg_angles if args.frame_avg_angles else DEFAULT_FRAME_ANGLES))

    def cond_autocast_ctx():
        if not ac_enabled:
            return contextlib.nullcontext()
        if ac_dtype is None:
            return torch.amp.autocast(device)
        return torch.amp.autocast(device, dtype=ac_dtype)

    def conditioner(metadata, _device):
        with cond_autocast_ctx():
            if args.cond_method == "fa_invariant":
                return invariant_conditioning(module.diffusion.conditioner, metadata,
                                              module.device, frame_avg_angles)
            return module.diffusion.conditioner(metadata, module.device)

    def sampler(noise, cond_inputs):
        with torch.no_grad():
            return sample_discrete_euler(model, noise, args.steps, **cond_inputs,
                                         cfg_scale=args.cfg_scale,
                                         dist_shift=module.diffusion.dist_shift,
                                         batch_cfg=True, disable_tqdm=True)

    def decoder(latents):
        with torch.no_grad():
            if module.diffusion.pretransform is not None:
                return module.diffusion.pretransform.decode(latents)
            return latents

    def embedder(wavs):
        if agree is None:
            raise ValueError("no AGREE scorer was loaded; embedding is unavailable")
        return embed_rirs(agree.model, wavs, device, readout="mean")

    engine = Engine(device=device, io_channels=module.diffusion.io_channels,
                    latent_samples=latent_samples, conditioner=conditioner,
                    cond_inputs_fn=module.diffusion.get_conditioning_inputs,
                    sampler=sampler, decoder=decoder, embedder=embedder)
    context = {"module": module, "model": model, "model_config": model_config,
               "weights_source": weights_source, "device": device,
               "latent_shape": (module.diffusion.io_channels, latent_samples)}
    return engine, context


def parity_check_one_query(args, engine, context, metadata, noise):
    """C8: the driver's generation path vs a straight-line ``evaluate_model`` replay.

    Same checkpoint, same metadata, same noise through (i) the engine's closures
    and (ii) the reference call sequence written out here from
    ``eval_FLAC.evaluate_model`` (conditioning under the resolved autocast ->
    ``get_conditioning_inputs`` -> ``sample_discrete_euler`` with cfg_scale,
    dist_shift and batch_cfg -> pretransform decode -> clamp). Identical waveforms
    is the contract; anything else is a divergence, not a variant.
    """
    module, model = context["module"], context["model"]
    noise = noise.to(engine.device)

    driver = engine.decoder(engine.sampler(
        noise, engine.cond_inputs_fn(engine.conditioner(metadata, engine.device)))).clamp(-1.0, 1.0)

    ac_enabled, ac_dtype = resolve_cond_autocast(args.cond_autocast)
    if not ac_enabled:
        reference_ctx = contextlib.nullcontext()
    elif ac_dtype is None:
        reference_ctx = torch.amp.autocast(context["device"])
    else:
        reference_ctx = torch.amp.autocast(context["device"], dtype=ac_dtype)

    with torch.no_grad():
        with reference_ctx:
            conditioning = module.diffusion.conditioner(metadata, module.device)
        cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)
        reference = sample_discrete_euler(model, noise, args.steps, **cond_inputs,
                                          cfg_scale=args.cfg_scale,
                                          dist_shift=module.diffusion.dist_shift,
                                          batch_cfg=True, disable_tqdm=True)
        if module.diffusion.pretransform is not None:
            reference = module.diffusion.pretransform.decode(reference)
        reference = reference.clamp(-1.0, 1.0)

    return {"match": bool(torch.equal(driver, reference)),
            "max_abs_diff": float((driver - reference).abs().max()),
            "shape": list(driver.shape)}


def dataset_folder_from_md(md):
    """The dataset root the loader read this item from.

    Same derivation as ``AR_md.get_custom_metadata`` (AR_md.py:11-14): peel the
    relative path off the absolute one, so the metadata authority is found the
    way the release loader finds it rather than by a driver-side convention.
    """
    full_path, rel_path = md["path"], md["relpath"]
    common_suffix = os.path.commonpath([full_path[::-1], rel_path[::-1]])[::-1]
    return full_path[: -len(common_suffix)]


def query_candidate_set(md):
    """The metadata-declared candidate set for one query (C7)."""
    return build_candidate_set(md["path"], os.path.join(dataset_folder_from_md(md), "metadata"))


def context_evidence(engine, md, obs_wav):
    """``{context_xyz_cam, context_sims_hex}`` for the O10 control, or ``None``.

    The control needs the similarity between the observed RIR and each CONTEXT
    RIR; both the context waveforms and their camera-frame positions are already
    in the loader's metadata, so no extra file is read and no generation happens.
    """
    poses, audio = md.get("context_poses"), md.get("context_audio")
    if poses is None or audio is None:
        return None
    embeddings = engine.embedder(audio.to(engine.device))
    obs_embedding = engine.embedder(obs_wav.to(engine.device))[0]
    sims = cosine_sims(obs_embedding, embeddings.unsqueeze(1)).reshape(1, -1)
    return {"context_xyz_cam": [[float(v) for v in row] for row in poses],
            "context_sims_hex": encode_sims(sims)[0]}


def _iter_items(loader):
    """Yield ``(reals_i, md_i)`` one query at a time from a batched loader.

    The loader keeps the PINNED ``batch_size``/``num_workers`` -- the per-item
    context draw depends on both (O8), so changing them would change the
    conditioning -- while the driver walks the batch item by item, because the
    M x K candidate expansion is what fills a GPU batch inside :func:`run_query`.
    """
    for reals, metadata in loader:
        for index, md in enumerate(metadata):
            yield (None if reals is None else reals[index:index + 1]), md


def process_query(args, engine, context, md, obs_wav):
    """One query end to end: candidate set -> generation/oracle -> scored row.

    The whole body sits inside a separately synchronized wall-clock timer, and
    the context-control embedding is timed as its own component, so R0's probe
    accounts for every part of a scored query (r4 review H1).
    """
    timings = {}
    with _timed(timings, PROBE_WALL, engine.device):
        query_id = sample_target_id(md)
        room_id = room_id_from_relpath(md["relpath"])
        gt_node, receiver_node = parse_ir_filename(md["path"])
        manifest = context.get("manifest")
        if manifest is None:
            _refuse("no frozen candidate manifest in the run context; per-query disk enumeration "
                    "was removed so that M cannot change mid-run (plan Rev 3.1 §2)")
        cand_set = candidate_set_from_manifest(manifest, room_id, gt_node, receiver_node)
        # BEFORE anything can touch the poses: the context fingerprint identifies the draw.
        context_ids = sample_context_ids(md)

        context_k = context.get("context_k")
        if context_k is not None:
            assert_query_context(md, context_k)
        # Membership is resolved BEFORE any generation so a conditioning/candidate
        # geometry disagreement aborts instead of quietly enlarging the eligible set.
        candidate_cams = candidate_camera_positions(cand_set)
        context_mask = context_membership_mask(candidate_cams, context_ids,
                                               gt_index=cand_set.gt_index)

        if args.score_source == "gt_rir":
            outcome = run_query_gt_rir(engine, cand_set, os.path.dirname(md["path"]),
                                       receiver_node, obs_wav)
            noise_keys, available = [], outcome["available"]
            identity_index = outcome["identity_index"]
        else:
            noise = build_noise_bank(args.seed, query_id, args.num_samples,
                                     context["latent_shape"])
            noise_keys = [noise_key(args.seed, query_id, k)
                          for k in range(int(args.num_samples))]
            outcome = run_query(engine, md, cand_set, noise, obs_wav,
                                batch_size=args.batch_size, control=args.control)
            available, identity_index = None, None

        timings.update(outcome.get("timings_s") or {})
        with _timed(timings, "context", engine.device):
            evidence = context_evidence(engine, md, obs_wav) or {}

    return build_row(
        query_id=query_id, room_id=room_id, relpath=md["relpath"], receiver_node=receiver_node,
        cand_set=cand_set, cam_xyz=outcome["cand_cam_xyz"], sims=outcome["sims"],
        context_mask=context_mask,
        noise_keys=noise_keys, tau=args.tau, agg=args.agg, control=args.control,
        score_source=args.score_source, smoke=bool(args.smoke), available=available,
        identity_index=identity_index, substituted=False,
        context_xyz_cam=evidence.get("context_xyz_cam"),
        context_sims_hex=evidence.get("context_sims_hex"),
        timings=timings)


def run_evaluation(args, loader, engine, context, ckpt_sha256, agree_sha256, expected=None,
                   dataset_config=None):
    """Score every query under a fail-closed identity contract (plan §2.1, C2).

    The audit is IN the scoring loop, not a separate earlier pass (r3 review
    finding 1): ``SampleDataset`` substitutes a random other item on a load
    failure, so a pre-pass that saw the right item proves nothing about the pass
    that produced the numbers. Every position's identity is checked against the
    registered enumeration BEFORE that query is generated, the run is gated on the
    scored count and room set at the end, the recorded split hash is computed over
    the SCORED stream, and the artifacts are written to ``.partial`` paths and
    renamed only once the gate passes -- so a final-named file always denotes a
    fully verified run. A separate pre-flight audit remains available as
    :func:`audit_split_identities`, but it is no longer load-bearing.
    """
    if expected is None:
        expected = expected_split_identities_from_config(load_dataset_config(args))
    expected = list(expected)
    if args.smoke and args.max_queries is not None:
        expected = expected[: int(args.max_queries)]
    if not expected:
        _refuse("the registered split enumerates no queries")

    paths = artifact_paths(args)
    rows_path, summary_path = paths["rows"], paths["summary"]
    partial_rows = rows_path + ".partial"
    rows, scored, seen_rooms = [], [], set()
    reset_peak_memory(context.get("device", "cpu"))

    with open(partial_rows, "w") as handle:
        for position, (obs_wav, md) in enumerate(itertools.islice(_iter_items(loader),
                                                                  len(expected))):
            identity = sample_target_id(md)
            if identity != expected[position]:
                raise SystemExit(
                    f"identity gate ABORT at position {position}: expected "
                    f"{expected[position]!r}, got {identity!r} (silent substitution?); "
                    "no query is scored under an unverified identity")
            row = process_query(args, engine, context, md, obs_wav)
            write_row(handle, row)
            rows.append(row)
            scored.append(identity)
            if row["room_id"] not in seen_rooms:
                seen_rooms.add(row["room_id"])
                print(f"[{len(rows)}/{len(expected)}] room {row['room_id']}")

    if context.get("context_k") is not None:
        assert_context_evidence_complete(rows, context["context_k"])
    split = assert_scored_stream(scored, expected)
    print(f"identity gate passed: {len(scored)} queries, split_hash={split[:12]}...")

    summary = summarize_run(rows)
    summary["probe"] = probe_summary(rows, read_peak_memory(context.get("device", "cpu")))
    manifest = context.get("manifest")
    provenance = build_provenance(args, ckpt_sha256, agree_sha256, split,
                                  context["weights_source"], len(rows),
                                  dataset_config=dataset_config,
                                  context_digest=context_stream_digest(rows),
                                  candidate_manifest_sha256=(manifest_sha256(manifest)
                                                             if manifest else None))
    write_summary(summary_path, summary, provenance)
    if manifest is not None:
        write_json_atomic(paths["manifest"], manifest, overwrite=True)
    os.replace(partial_rows, rows_path)          # publish only verified artifacts
    return {"rows_path": rows_path, "summary_path": summary_path,
            "manifest_path": paths["manifest"], "rows": rows,
            "summary": summary, "provenance": provenance}


def scoring_only_engine(agree, device):
    """An :class:`Engine` that can score but not generate (measured-RIR oracle).

    The oracle needs no FLAC checkpoint, so it must be able to run the moment the
    dataset lands (plan §2.6, run R-1); the generation callables refuse rather
    than exist as stubs that could quietly return something.
    """
    def _unavailable(*_args, **_kwargs):
        raise ValueError("generation is unavailable under --score-source gt_rir")

    return Engine(device=device, io_channels=0, latent_samples=0, conditioner=_unavailable,
                  cond_inputs_fn=_unavailable, sampler=_unavailable, decoder=_unavailable,
                  embedder=lambda wavs: embed_rirs(agree.model, wavs, device, readout="mean"))


def build_dataloader(args, model_config, dataset_config=None):
    """The eval loader at the PINNED parallelism (O8), unshuffled."""
    if dataset_config is None:
        dataset_config = load_dataset_config(args)
    return create_dataloader_from_config(
        dataset_config, batch_size=args.batch_size, num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1), shuffle=False)


def main(argv=None):
    """CLI entry: validate, build, audit, evaluate, summarize."""
    args = validate_args(parse_args(argv))
    # O16 first (this reads the dataset config only for smoke/parity runs), then
    # the CPU-only artifact refusals: objective and ARE are checked before the
    # scorer or the generator is constructed, let alone moved to a device (F9).
    if args.mode == "readback":
        return run_readback(args)
    if args.mode == "scorer-noise":
        return run_scorer_noise(args)
    if args.mode == "reaggregate":
        return run_reaggregate(args)

    validate_dataset_split(args)
    model_config, ckpt = load_and_validate_artifacts(args)
    dataset_config = load_dataset_config(args)
    assert_registration_sha(args, dataset_config)        # O17, before any model load

    # Freeze the candidate manifest and resolve every locked quantity while this is
    # still pure disk work, so a registration mismatch costs no model load (F4/F7).
    candidate_manifest = None if args.parity_check else manifest_for_dataset_config(dataset_config)
    manifest_hash = manifest_sha256(candidate_manifest) if candidate_manifest else "n/a"
    ckpt_sha256 = sha256_file(args.ckpt_path) if args.score_source == "flac" else "n/a"
    agree_sha256 = sha256_file(args.agree_ckpt)
    verify_registration(args, dataset_config, {
        "model_config_sha256": _file_sha256(args.model_config),
        "dataset_config_sha256": _file_sha256(args.dataset_config),
        "ckpt_sha256": ckpt_sha256, "agree_sha256": agree_sha256,
        "num_samples": effective_num_samples(args), "tau": args.tau, "agg": args.agg,
        "cond_method": args.cond_method, "cond_autocast": args.cond_autocast,
        "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed,
        "readout": "mean", "candidate_manifest_sha256": manifest_hash})

    torch.set_float32_matmul_precision("medium")
    agree = load_agree_audio(args.agree_ckpt, args.device)
    if args.score_source == "gt_rir":
        engine = scoring_only_engine(agree, args.device)
        context = {"weights_source": "n/a", "latent_shape": None, "device": args.device}
    else:
        engine, context = build_engine(args, agree=agree, device=args.device,
                                       model_config=model_config, ckpt=ckpt)

    # Seeded exactly where evaluate_model seeds it -- after the model build and
    # before the loader, because the per-item context draw happens in the workers.
    pl.seed_everything(args.seed, workers=True)
    loader = build_dataloader(args, model_config, dataset_config)

    if args.parity_check:
        obs_wav, md = next(_iter_items(loader))
        noise = build_noise_bank(args.seed, sample_target_id(md), 1, context["latent_shape"])
        result = parity_check_one_query(args, engine, context, [md], noise)
        print(f"parity_check_one_query: {result}")
        if not result["match"]:
            _refuse(f"generation parity with eval_FLAC FAILED: {result}")
        return result

    expected = expected_split_identities_from_config(dataset_config)
    print(f"registered split: {len(expected)} queries")
    context["manifest"] = candidate_manifest
    context["context_k"] = resolve_context_k(dataset_config)
    print(f"candidate manifest frozen: {len(candidate_manifest['rooms'])} rooms, "
          f"sha256={manifest_hash[:12]}...")
    result = run_evaluation(args, loader, engine, context, ckpt_sha256, agree_sha256,
                            expected=expected, dataset_config=dataset_config)
    summary = result["summary"]
    print(f"rows:    {result['rows_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"primary ({summary['flac']['primary_name']}): {summary['flac']['primary']:.4f} m over "
          f"{summary['n_queries']} queries / {summary['n_rooms']} rooms")
    print(f"context-conditioned baseline: "
          f"{summary['baselines']['context_conditioned']['primary']:.4f} m")
    return result



def load_and_validate_artifacts(args):
    """CPU-only artifact validation, BEFORE anything is built or moved to a device.

    Reads the model config and (for generative runs) the checkpoint's metadata and
    refuses a foreign objective or an ARE artifact there and then -- previously
    those refusals lived in ``build_engine``, i.e. after the AGREE scorer had
    already been constructed on the target device (r3 review finding 9).
    """
    with open(args.model_config) as handle:
        model_config = json.load(handle)
    assert_rectified_flow(model_config)
    ckpt = None
    if args.score_source == "flac":
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        assert_no_are(ckpt.get("model_config"), copy.deepcopy(model_config))
    return model_config, ckpt


def load_dataset_config(args):
    """Parse the dataset config (content, not just the path)."""
    with open(args.dataset_config) as handle:
        return json.load(handle)


def validate_dataset_split(args, dataset_config=None):
    """O16: debug-shaped runs may only read a SEEN split.

    ``--smoke`` and ``--parity-check`` are iterated during development; pointing
    them at the held-out split would leak it before the registered run, which is
    exactly what pre-registration exists to prevent. Registered unseen runs --
    including the checkpoint-free R-1 oracle -- are untouched.
    """
    if not (args.smoke or args.parity_check or args.mode == "scorer-noise"):
        return dataset_config
    config = dataset_config if dataset_config is not None else load_dataset_config(args)
    if bool(config.get("unseeneval", False)) or not bool(config.get("seeneval", False)):
        _refuse(
            f"--smoke/--parity-check require a SEEN-split dataset config; {args.dataset_config!r} "
            f"declares seeneval={config.get('seeneval')!r}, unseeneval={config.get('unseeneval')!r} "
            "(O16: the held-out split is not a debugging surface)")
    return config


DEFAULT_IR_FOLDER = "single_channel_ir_1"


def build_room_manifest(dataset_root, split_config, folder_name=DEFAULT_IR_FOLDER):
    """Freeze the candidate authority for the whole run (plan Rev 3.1 §2).

    Enumerating candidates from disk per query let M -- and therefore the
    conditioning batch composition, which the registered bf16 autocast is
    sensitive to -- change between seeds if the dataset moved underneath the run.
    This walks every room of the split ONCE, records the metadata-declared nodes
    with their (consistency-checked) coordinates, the receiver positions the split
    actually uses, and which nodes have any wav at all, and is then consumed from
    memory. It is JSON-serializable and hashed into provenance.
    """
    dataset_root = str(dataset_root)
    rooms = {}
    for scene in sorted(split_config):
        for scene_id in sorted(split_config[scene]):
            room_id = f"{scene}/{scene_id}"
            meta_dir = os.path.join(dataset_root, "metadata", scene, scene_id)
            wav_dir = os.path.join(dataset_root, folder_name, scene, scene_id)
            if not os.path.isdir(meta_dir):
                raise SystemExit(
                    f"candidate manifest ABORT: {room_id} is in the split but has no metadata "
                    f"directory at {meta_dir}; the candidate authority is incomplete")
            try:
                sources = enumerate_metadata_sources(meta_dir)
            except ValueError as err:
                raise SystemExit(f"candidate manifest ABORT for {room_id}: {err}")

            wav_nodes, receivers = set(), {}
            if os.path.isdir(wav_dir):
                for fname in os.listdir(wav_dir):
                    try:
                        src, _rec = parse_ir_filename(fname)
                    except ValueError:
                        continue
                    wav_nodes.add(src)
            for fname in sorted(split_config[scene][scene_id]):
                try:
                    _src, rec = parse_ir_filename(fname)
                except ValueError:
                    raise SystemExit(f"candidate manifest ABORT: {room_id} lists a non-IR "
                                     f"file {fname!r}")
                if str(rec) in receivers:
                    continue
                pair_path = None
                for node in sorted(sources):
                    pair_path = find_pair_metadata(meta_dir, node, rec)
                    if pair_path is not None:
                        break
                if pair_path is None:
                    raise SystemExit(
                        f"candidate manifest ABORT: {room_id} receiver {rec} has no pair "
                        f"metadata; its query frame cannot be established")
                with open(pair_path) as handle:
                    rec_loc = json.load(handle)["rec_loc"]
                receivers[str(rec)] = [float(v) for v in rec_loc]

            nodes = sorted(sources)
            rooms[room_id] = {
                "scene": scene,
                "scene_id": scene_id,
                "nodes": nodes,
                "xyz_world": [[float(v) for v in sources[node]] for node in nodes],
                "wav_nodes": sorted(wav_nodes),
                "receivers": receivers,
                "n_metadata_sources": len(nodes),
                "n_wav_sources": len(wav_nodes),
            }
    return {"dataset_root": dataset_root, "folder_name": folder_name, "rooms": rooms}


def manifest_sha256(manifest):
    """sha256 over the canonical JSON of the frozen manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_set_from_manifest(manifest, room_id, gt_node, rec_node):
    """The query's candidate set, from the frozen manifest -- no disk access."""
    room = manifest["rooms"].get(room_id)
    if room is None:
        raise ValueError(f"room {room_id!r} is not in the frozen candidate manifest")
    rec_loc = room["receivers"].get(str(int(rec_node)))
    if rec_loc is None:
        raise ValueError(f"receiver {rec_node} of {room_id} is not in the frozen manifest")
    nodes = [int(n) for n in room["nodes"]]
    if int(gt_node) not in nodes:
        raise ValueError(f"GT source {gt_node} is not a candidate of {room_id}")
    xyz = np.asarray(room["xyz_world"], dtype=np.float64)
    return CandidateSet(nodes=nodes, xyz_world=xyz, rec_loc=np.asarray(rec_loc, dtype=np.float64),
                        gt_node=int(gt_node), gt_xyz=xyz[nodes.index(int(gt_node))])


def manifest_for_dataset_config(dataset_config):
    """Freeze the candidate manifest for every room the dataset config's split names."""
    entries = dataset_config.get("datasets", None) or []
    if len(entries) != 1:
        _refuse(f"the candidate manifest expects exactly one dataset entry, got {len(entries)}")
    entry = entries[0]
    with open(entry["json_file_path"]) as handle:
        split = json.load(handle)
    return build_room_manifest(entry["path"], split,
                               folder_name=entry.get("folder_name", DEFAULT_IR_FOLDER))


def resolve_context_k(dataset_config):
    """The registered context size K_ctx, or a refusal.

    The nearest-context control is a registered success criterion (plan §2.6), so
    a configuration that cannot produce it is refused up front rather than
    published with the control missing (full-review F3).
    """
    acoustic = ((dataset_config or {}).get("modalities") or {}).get("acoustic_context") or {}
    if not acoustic.get("load", False):
        _refuse("the dataset config does not load acoustic_context; the registered "
                "nearest-context control (O10) could not be computed")
    context_k = acoustic.get("max_context")
    if not isinstance(context_k, int) or context_k < 1:
        _refuse(f"the dataset config declares max_context={context_k!r}; a positive integer "
                "context size is required")
    return int(context_k)


def assert_query_context(md, context_k):
    """Every query must carry exactly the configured context, in loader shape."""
    poses, audio = md.get("context_poses"), md.get("context_audio")
    if poses is None or audio is None:
        raise ValueError("query metadata carries no context_poses/context_audio; the registered "
                         "nearest-context control cannot be computed for it")
    if not isinstance(poses, torch.Tensor) or poses.dim() != 2 or poses.shape[1] != 3:
        raise ValueError(f"context_poses must be a [K, 3] tensor, got "
                         f"{tuple(poses.shape) if isinstance(poses, torch.Tensor) else type(poses)}")
    if poses.dtype != torch.float32:
        raise ValueError(f"context_poses must be float32 (the fingerprint rendering is not "
                         f"dtype-stable), got {poses.dtype}")
    if not isinstance(audio, torch.Tensor) or audio.dim() not in (2, 3):
        raise ValueError("context_audio must be a [K, T] or [K, 1, T] tensor")
    if poses.shape[0] != context_k or audio.shape[0] != context_k:
        raise ValueError(
            f"query carries {poses.shape[0]} context poses / {audio.shape[0]} context RIRs, "
            f"but the dataset config registers K_ctx={context_k}")
    if not bool(torch.isfinite(poses).all()):
        raise ValueError("context_poses must be finite")
    return context_k


def assert_context_evidence_complete(rows, context_k):
    """Publication gate: every row carries the full-length control evidence."""
    for row in rows:
        evidence = row.get("context_xyz_cam")
        if evidence is None:
            raise SystemExit(
                f"context gate ABORT: query {row.get('query_id')!r} has no context evidence; "
                "the registered nearest-context control would silently vanish from the summary")
        if len(evidence) != int(context_k):
            raise SystemExit(
                f"context gate ABORT: query {row.get('query_id')!r} carries {len(evidence)} "
                f"context sources, the dataset config registers K_ctx={context_k}")
    return True


#: fields a registered run's committed manifest must lock (plan Rev 3.1 §4).
REGISTRATION_LOCKED_FIELDS = (
    "model_config_sha256", "dataset_config_sha256", "ckpt_sha256", "agree_sha256",
    "num_samples", "tau", "agg", "cond_method", "cond_autocast", "steps", "cfg_scale",
    "readout", "candidate_manifest_sha256",
)


def _repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def verify_registration_commit(manifest_path, registration_sha, repo_root=None):
    """Return the manifest, having proved it IS the committed one.

    A registration SHA that is merely a non-empty string proves nothing. This
    resolves it as a real commit and byte-compares the committed blob against the
    local file, so a manifest edited after registration cannot be used.
    """
    repo_root = repo_root or _repo_root()
    manifest_path = os.path.abspath(str(manifest_path))
    if not os.path.isfile(manifest_path):
        _refuse(f"--registration-manifest not found: {manifest_path}")
    relpath = os.path.relpath(manifest_path, repo_root)

    resolved = subprocess.run(["git", "cat-file", "-e", f"{registration_sha}^{{commit}}"],
                              cwd=repo_root, capture_output=True)
    if resolved.returncode != 0:
        _refuse(f"--registration-sha {registration_sha!r} does not resolve to a commit in "
                f"{repo_root}")
    show = subprocess.run(["git", "show", f"{registration_sha}:{relpath}"],
                          cwd=repo_root, capture_output=True)
    if show.returncode != 0:
        _refuse(f"commit {registration_sha} does not contain {relpath!r}; the registration "
                "manifest was not committed before the run (O17)")
    with open(manifest_path, "rb") as handle:
        local = handle.read()
    if show.stdout != local:
        _refuse(f"{relpath!r} differs from the version committed at {registration_sha}; the "
                "registered protocol was edited after registration")
    return json.loads(local.decode("utf-8"))


def check_registration_fields(manifest, resolved, registered):
    """Every locked field must match the resolved run state; refuse otherwise."""
    for field in REGISTRATION_LOCKED_FIELDS:
        if field not in manifest:
            _refuse(f"registration manifest does not lock {field!r}")
        locked, actual = manifest[field], resolved.get(field)
        if field == "candidate_manifest_sha256" and locked == "tbd":
            if registered:
                _refuse("registration manifest still has candidate_manifest_sha256='tbd'; a "
                        "registered run must lock the frozen candidate manifest")
            continue
        if isinstance(locked, (int, float)) and isinstance(actual, (int, float)):
            match = float(locked) == float(actual)
        else:
            match = locked == actual
        if not match:
            _refuse(f"registered {field} is {locked!r} but this run resolves {actual!r}")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        _refuse("registration manifest does not lock a non-empty 'seeds' list")
    if int(resolved["seed"]) not in [int(s) for s in seeds]:
        _refuse(f"--seed {resolved['seed']} is not one of the registered seeds {list(seeds)}")
    return True


def verify_registration(args, dataset_config, resolved, repo_root=None):
    """Machine-checked registration gate (full-review F4), before any model load."""
    registered = is_registered_run(args, dataset_config)
    if not args.registration_manifest:
        if registered:
            _refuse("--registration-manifest is required for a registered unseen run")
        return False
    manifest = verify_registration_commit(args.registration_manifest, args.registration_sha,
                                          repo_root=repo_root)
    check_registration_fields(manifest, resolved, registered)
    return True


#: the registered unseen-split shape (plan Rev 3.1 §1, established at rung 4).
REGISTERED_UNSEEN_ROOMS = 17
REGISTERED_UNSEEN_SOURCES = 10
#: the ONE anomaly the protocol expects: metadata source 10 has no wavs.
REGISTERED_METADATA_ONLY = {
    "LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30": [10],
}
#: the depth panorama shape the loader's projection assumes.
DEPTH_MAP_SHAPE = (256, 512)


def run_readback(args):
    """R-1's dataset gate (plan §6 rung 4, Rev 3.1 §3): does the data hold up?

    Against the registered UNSEEN split this is an executable invariant, not a
    report (r4 review H2): exactly ``REGISTERED_UNSEEN_ROOMS`` rooms with exactly
    ``REGISTERED_UNSEEN_SOURCES`` metadata sources each, and the single expected
    metadata-only anomaly (LivingRoomsWithHallway_idx_30's source 10) -- anything
    else, INCLUDING that anomaly disappearing, is a FAILURE. Every depth map the
    split's receivers reference is LOADED and checked for shape, dtype and
    finiteness; one wav per (room, source) is decoded and checked for rate,
    channels, length and finiteness. Nonzero exit on any failure.
    """
    dataset_config = load_dataset_config(args)
    entry = (dataset_config.get("datasets") or [None])[0]
    if entry is None:
        _refuse("the dataset config declares no datasets")
    with open(entry["json_file_path"]) as handle:
        split = json.load(handle)
    manifest = build_room_manifest(entry["path"], split,
                                   folder_name=entry.get("folder_name", DEFAULT_IR_FOLDER))
    root, folder = manifest["dataset_root"], manifest["folder_name"]
    registered = bool(dataset_config.get("unseeneval", False))

    rooms, failures, warnings = {}, [], []
    for room_id, room in sorted(manifest["rooms"].items()):
        scene, scene_id = room["scene"], room["scene_id"]
        wav_dir = os.path.join(root, folder, scene, scene_id)
        depth_dir = os.path.join(root, "depth_map", scene, scene_id)
        cross = crosscheck_sources_vs_files(room["nodes"], wav_dir)

        split_files = list(split[scene][scene_id])
        split_sources, receivers, missing_files, one_per_source = set(), set(), [], {}
        for fname in split_files:
            src, rec = parse_ir_filename(fname)
            split_sources.add(src)
            receivers.add(rec)
            path = os.path.join(wav_dir, fname)
            if not os.path.isfile(path):
                missing_files.append(fname)
            else:
                one_per_source.setdefault(src, path)

        depth_bad = []
        for rec in sorted(receivers):
            depth_path = os.path.join(depth_dir, f"{rec}.npy")
            if not os.path.isfile(depth_path):
                depth_bad.append(f"R{rec}: missing")
                continue
            try:
                depth = np.load(depth_path)
            except Exception as err:                       # noqa: BLE001 - reported, not raised
                depth_bad.append(f"R{rec}: unreadable ({type(err).__name__})")
                continue
            if tuple(depth.shape) != DEPTH_MAP_SHAPE:
                depth_bad.append(f"R{rec}: shape {tuple(depth.shape)} != {DEPTH_MAP_SHAPE}")
            elif not np.issubdtype(depth.dtype, np.floating):
                depth_bad.append(f"R{rec}: dtype {depth.dtype} is not floating")
            elif not bool(np.isfinite(depth).all()):
                depth_bad.append(f"R{rec}: contains non-finite values")

        wav_bad, sample_rates = [], set()
        for src, path in sorted(one_per_source.items()):
            try:
                wav, rate = torchaudio.load(path)
            except Exception as err:                       # noqa: BLE001 - reported, not raised
                wav_bad.append(f"S{src}: unreadable ({type(err).__name__})")
                continue
            sample_rates.add(int(rate))
            if int(rate) != AR_SAMPLE_RATE:
                wav_bad.append(f"S{src}: sample rate {rate} != {AR_SAMPLE_RATE}")
            elif wav.shape[0] != 1:
                wav_bad.append(f"S{src}: {wav.shape[0]} channels, expected mono")
            elif wav.shape[-1] < 1:
                wav_bad.append(f"S{src}: empty waveform")
            elif not bool(torch.isfinite(wav).all()):
                wav_bad.append(f"S{src}: contains non-finite samples")

        rooms[room_id] = {
            "metadata_nodes": room["nodes"], "wav_nodes": room["wav_nodes"],
            "metadata_only_nodes": cross["missing_files"], "wav_only_nodes": cross["extra_files"],
            "split_files": len(split_files), "split_sources": sorted(split_sources),
            "split_receivers": len(receivers), "missing_split_files": missing_files,
            "depth_checked": len(receivers), "depth_bad": depth_bad,
            "wav_checked": len(one_per_source), "wav_bad": wav_bad,
            "sample_rates": sorted(sample_rates),
        }
        if cross["extra_files"]:
            failures.append(f"{room_id}: wav sources without metadata: {cross['extra_files']}")
        expected_only = REGISTERED_METADATA_ONLY.get(room_id, [])
        if cross["missing_files"]:
            if registered and cross["missing_files"] != expected_only:
                failures.append(
                    f"{room_id}: unregistered metadata-only sources {cross['missing_files']} "
                    f"(registered: {expected_only or 'none'})")
            else:
                warnings.append(
                    f"{room_id}: metadata-only sources (no wavs): {cross['missing_files']}")
        elif registered and expected_only:
            failures.append(
                f"{room_id}: the registered metadata-only source {expected_only} is GONE; the "
                "candidate count changed under the protocol (plan Rev 3.1 §1)")
        if registered and len(room["nodes"]) != REGISTERED_UNSEEN_SOURCES:
            failures.append(f"{room_id}: {len(room['nodes'])} metadata sources, the registered "
                            f"unseen split has {REGISTERED_UNSEEN_SOURCES}")
        if not split_sources <= set(room["nodes"]):
            failures.append(f"{room_id}: split names sources absent from metadata: "
                            f"{sorted(split_sources - set(room['nodes']))}")
        if missing_files:
            failures.append(f"{room_id}: {len(missing_files)} split files are absent on disk "
                            f"(first: {missing_files[:3]})")
        if depth_bad:
            failures.append(f"{room_id}: depth maps invalid: {depth_bad[:3]}")
        if wav_bad:
            failures.append(f"{room_id}: wav readback invalid: {wav_bad[:3]}")
        if not one_per_source:
            failures.append(f"{room_id}: no split file present to read back")

    if registered and len(rooms) != REGISTERED_UNSEEN_ROOMS:
        failures.append(f"{len(rooms)} rooms in the split, the registered unseen split has "
                        f"{REGISTERED_UNSEEN_ROOMS}")

    report = {
        "mode": "readback", "dataset_config": args.dataset_config,
        "dataset_root": root, "n_rooms": len(rooms),
        "registered_check": {
            "enforced": registered, "n_rooms": len(rooms),
            "expected_rooms": REGISTERED_UNSEEN_ROOMS if registered else None,
            "expected_sources": REGISTERED_UNSEEN_SOURCES if registered else None,
            "expected_metadata_only": REGISTERED_METADATA_ONLY if registered else {}},
        "manifest_sha256": manifest_sha256(manifest),
        "source_sha": source_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rooms": rooms, "warnings": warnings, "failures": failures, "ok": not failures,
    }
    report_path = write_report(args, report, "readback")
    report["report_path"] = report_path

    print(f"readback: {len(rooms)} rooms, {len(warnings)} warnings, {len(failures)} failures "
          f"-> {report_path}")
    for line in warnings:
        print(f"  WARNING {line}")
    for line in failures:
        print(f"  FAILURE {line}")
    if failures:
        raise SystemExit(f"readback gate FAILED with {len(failures)} failures; see {report_path}")
    return report


def write_json_atomic(path, payload, overwrite=False):
    """The ONE report writer: tmp file + ``os.replace``, refusing to clobber.

    Every mode goes through this (r4 review M3). A finished report is evidence:
    a later failed run must not be able to erase it, and a leftover ``.partial``
    means an earlier attempt died, which is also a refusal.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not overwrite:
        for candidate in (path, path + ".partial"):
            if os.path.exists(candidate):
                _refuse(f"report target already exists: {candidate}; pass --overwrite to replace")
    tmp = path + ".partial"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def _short(value, size=10):
    return str(value)[:size]


def _rows_digest(paths):
    """One digest over the CONTENT of the input row files."""
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths or []):
        digest.update(sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def aux_stem(args, kind):
    """Content-addressed stem for an auxiliary mode's report.

    Two runs of the same mode over different inputs must not collide, so the
    stem carries short hashes of the dataset config, the scorer checkpoint and
    the input rows (whichever the mode consumes) plus its protocol knobs.
    """
    parts = [str(args.eval_name), kind, f"ds-{_short(_file_sha256(args.dataset_config))}"]
    if kind == "scorer-noise":
        scorer = os.path.splitext(os.path.basename(str(args.agree_ckpt)))[0] or "none"
        parts += [f"scorer-{scorer}", f"sha-{_short(_file_sha256(args.agree_ckpt))}",
                  f"n{int(args.noise_draws)}", f"seed{int(args.seed)}"]
    elif kind == "reaggregate":
        parts.append(f"rows-{_short(_rows_digest(getattr(args, 'rows', None)))}")
    return "_".join(parts)


def write_report(args, report, kind):
    """Write one auxiliary-mode JSON report, atomically and without clobbering."""
    path = os.path.join(str(args.out_dir), f"{aux_stem(args, kind)}.json")
    return write_json_atomic(path, report, overwrite=bool(getattr(args, "overwrite", False)))


def _cos_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "p5": float(np.percentile(values, 5, method="linear")),
            "p50": float(np.percentile(values, 50, method="linear")),
            "max": float(values.max())}


def measure_scorer_noise(mean_embeddings, sampled_draws):
    """Quantify what the registered mean readout removes (plan §2.8.3).

    ``mean_embeddings`` is ``[B, D]`` from the deterministic readout,
    ``sampled_draws`` is ``[N, B, D]`` from the stock stochastic one. Reports, per
    wav and pooled, the pairwise cosine distribution ACROSS draws and the cosine
    of each draw against the mean readout -- i.e. the scorer noise the registered
    readout eliminates by construction.
    """
    if sampled_draws.ndim != 3 or sampled_draws.shape[0] < 2:
        raise ValueError("sampled_draws must be [N, B, D] with N >= 2 draws")
    mean_embeddings = mean_embeddings.detach().cpu().double()
    sampled_draws = sampled_draws.detach().cpu().double()

    per_wav, all_pairwise, all_vs_mean = [], [], []
    for index in range(mean_embeddings.shape[0]):
        draws = sampled_draws[:, index, :]
        gram = draws @ draws.T
        upper = gram[torch.triu(torch.ones_like(gram), diagonal=1) > 0]
        vs_mean = draws @ mean_embeddings[index]
        per_wav.append({"pairwise": _cos_stats(upper.numpy()),
                        "vs_mean": _cos_stats(vs_mean.numpy())})
        all_pairwise.append(upper.numpy())
        all_vs_mean.append(vs_mean.numpy())
    return {"n_draws": int(sampled_draws.shape[0]), "n_wavs": int(mean_embeddings.shape[0]),
            "per_wav": per_wav,
            "aggregate": {"pairwise": _cos_stats(np.concatenate(all_pairwise)),
                          "vs_mean": _cos_stats(np.concatenate(all_vs_mean))}}


def _load_noise_wavs(paths):
    wavs = []
    for path in paths:
        wav, rate = torchaudio.load(str(path))
        if rate != AR_SAMPLE_RATE:
            raise ValueError(f"{path}: sample rate must be {AR_SAMPLE_RATE}, got {rate}")
        wav = wav[:1, :MAX_LEN].clamp(-1.0, 1.0)
        if wav.shape[-1] < MAX_LEN:
            wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[-1]))
        wavs.append(wav.unsqueeze(0))
    return torch.cat(wavs, dim=0)


def resolve_noise_wavs(args):
    """Explicit ``--noise-wavs``, else the first files of the (seen) split."""
    if args.noise_wavs:
        return [str(p) for p in args.noise_wavs]
    dataset_config = load_dataset_config(args)
    entry = (dataset_config.get("datasets") or [None])[0]
    if entry is None:
        _refuse("the dataset config declares no datasets")
    with open(entry["json_file_path"]) as handle:
        split = json.load(handle)
    folder = entry.get("folder_name", DEFAULT_IR_FOLDER)
    paths = []
    for scene in sorted(split):
        for scene_id in sorted(split[scene]):
            for fname in sorted(split[scene][scene_id]):
                candidate = os.path.join(entry["path"], folder, scene, scene_id, fname)
                if os.path.isfile(candidate):
                    paths.append(candidate)
                if len(paths) >= int(args.noise_wav_count):
                    return paths
    if not paths:
        _refuse("no wav from the split is present to measure scorer noise on")
    return paths


def run_scorer_noise(args):
    """--mode scorer-noise: the §2.8.3 diagnostic. Loads AGREE only."""
    validate_dataset_split(args)
    paths = resolve_noise_wavs(args)
    agree = load_agree_audio(args.agree_ckpt, args.device)
    wavs = _load_noise_wavs(paths)

    mean_embeddings = embed_rirs(agree.model, wavs, args.device, readout="mean")
    draws = torch.stack([embed_rirs(agree.model, wavs, args.device, readout="sample")
                         for _ in range(int(args.noise_draws))])
    report = measure_scorer_noise(mean_embeddings, draws)
    report.update({
        "mode": "scorer-noise", "agree_ckpt": args.agree_ckpt,
        "agree_sha256": agree.ckpt_sha256, "wavs": paths, "device": args.device,
        "source_sha": source_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    report_path = write_report(args, report, "scorer-noise")
    report["report_path"] = report_path
    pairwise = report["aggregate"]["pairwise"]
    print(f"scorer noise over {report['n_draws']} sampled draws x {report['n_wavs']} RIRs: "
          f"pairwise cos mean={pairwise['mean']:.6f} p5={pairwise['p5']:.6f} "
          f"min={pairwise['min']:.6f} -> {report_path}")
    return report


def run_reaggregate(args):
    """--mode reaggregate: R1's offline tau/aggregation/K' sweep + selection."""
    report = reaggregate(args.rows)
    report.update({"mode": "reaggregate", "source_sha": source_sha(),
                   "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    report_path = write_report(args, report, "reaggregate")
    report["report_path"] = report_path
    chosen = report["selected"]
    print(f"reaggregate: {report['n_rows']} rows, {len(report['sweep'])} configurations -> "
          f"{report_path}")
    print(f"registered selection: {chosen['method']} tau={chosen['tau']} K'={chosen['k_prime']} "
          f"(pooled mean e_loc {chosen['pooled_mean_e_loc']:.4f} m)")
    return report

if __name__ == "__main__":
    main()
