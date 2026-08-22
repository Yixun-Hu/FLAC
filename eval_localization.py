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
import re
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
from src.data.dataset import (create_dataloader_from_config, get_audio_filenames,
                              is_silence)
from src.localization.agree_embed import MAX_LEN, embed_rirs, load_agree_audio, sha256_file
from src.localization.reaggregate import (decode_scores, decode_sims, encode_sims,
                                          reaggregate)
from src.localization.candidates import (CandidateSet, assert_gt_matches_loader,
                                         build_candidate_set, candidate_metadata,
                                         crosscheck_sources_vs_files,
                                         enumerate_metadata_sources, find_pair_metadata,
                                         merge_position_duplicates, parse_ir_filename,
                                         project_to_camera)
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
#: Separately measured whole-query wall time (not a component sum). It covers
#: candidate resolution, membership, generation/oracle and the context control --
#: i.e. everything process_query does -- but NOT the row construction or the JSONL
#: write, which happen in run_evaluation after the timer closes.
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
              context_xyz_cam=None, context_sims_hex=None, timings=None, merge_map=None,
              oracle_source_nodes=None):
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
        "merge_map": dict(merge_map or {}),
        "oracle_source_nodes": (None if oracle_source_nodes is None
                                else [None if n is None else int(n)
                                      for n in oracle_source_nodes]),
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
                     dataset_config=None, context_digest=None, candidate_manifest_sha256=None,
                     split_file_sha256=None, merge_groups=None, metric_registerable=None):
    """Everything needed to reproduce or falsify the run, in one record.

    Announcement 05: a row is only interpretable next to the protocol it was
    produced under, so every pinned quantity is written down -- including the
    dataloader parallelism, which the per-item context draw depends on (O8), the
    resolved weights source (O9) and the scorer readout (§2.4).
    """
    # the DECLARED plan, not the module default: exp_20 conditions one angle per
    # forward, so provenance must say so rather than inherit "batched"/64
    fa_state = fa_protocol_state(args)
    orbit_execution, frame_avg_cap = orbit_provenance(args.cond_method)
    if fa_state is not None:
        orbit_execution = fa_state["orbit_execution"]
        frame_avg_cap = "candidate_micro_batch"
    # a frame-average row must state the orbit it actually ran, and the default
    # orbit is still an orbit -- recording None there left the row silent (r1)
    angles = "n/a" if args.cond_method != "fa_invariant" else (
        fa_state["frame_avg_angles"] if fa_state is not None else
        (None if args.frame_avg_angles is None else [float(a) for a in args.frame_avg_angles]))
    record = {
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
        "registration_sha_resolved": getattr(args, "registration_sha_resolved", None) or "n/a",
        "registration_manifest": args.registration_manifest or "n/a",
        "model_config_sha256": _file_sha256(args.model_config),
        "dataset_config_sha256": _file_sha256(args.dataset_config),
        "context_k": (((dataset_config or {}).get("modalities") or {})
                      .get("acoustic_context", {}) or {}).get("max_context"),
        "loader_shuffle": False,
        "loader_drop_last": bool((dataset_config or {}).get("drop_last", True)),
        "context_stream_digest": context_digest or "n/a",
        "candidate_manifest_sha256": candidate_manifest_sha256 or "n/a",
        "candidate_merge_groups": merge_groups,

        "split_file_sha256": split_file_sha256 or "n/a",
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
    # R4 fields appear ONLY when metrics actually ran: a --metrics-off run keeps
    # the r7-era provenance schema byte-for-byte (r4m review finding 7, and the
    # registered R2/R2b runs are exactly that shape).
    if metric_registerable is not None or getattr(args, "metric_registration", None):
        record["metric_registerable"] = metric_registerable
        record["metric_registration"] = getattr(args, "metric_registration", None) or "n/a"
        record["metric_registration_sha_resolved"] = getattr(
            args, "metric_registration_sha_resolved", None) or "n/a"
    # r2 F6: EVERY row states where its conditioning method was bound -- a
    # checkpoint-bound vanilla arm and an unbound stripped release are different
    # facts, and only recording it for FA rows hid both.
    record["cond_method_binding"] = getattr(args, "cond_method_binding", None)
    if fa_state is not None:
        record["frame_avg_chunk_plan"] = fa_state["frame_avg_chunk_plan"]
        record["fa_protocol"] = {field: fa_state[field] for field in sorted(fa_state)}
        # a registration commit need only be an ANCESTOR, so the executable FA
        # source can move after the manifest says per_angle: hash it at run time
        from src.localization.crossarm import fa_source_shas

        record["fa_source_shas"] = fa_source_shas()
    return record


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
    if getattr(args, "verify_against", None):
        # a verification replay publishes its OWN artifacts; the original run's are
        # evidence and are never rewritten
        parts.append("replay")
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


def _measured_rir_lookup(room_wav_dir, cand_set, receiver_node, merge_map=None):
    """``[(path|None, node|None)]`` per candidate, canonical file first.

    Under Rev 3.2 a candidate can be a merged GROUP of nodes at one position. The
    canonical node's measured file is preferred; if it is absent, any member's
    file measures the same position, and WHICH node supplied it is recorded.
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

    groups = {int(canonical): [int(m) for m in members]
              for canonical, members in (merge_map or {}).items()}
    lookup = []
    for node in cand_set.nodes:
        for candidate_node in [int(node)] + [m for m in groups.get(int(node), [])
                                             if m != int(node)]:
            if candidate_node in by_source:
                lookup.append((by_source[candidate_node], candidate_node))
                break
        else:
            lookup.append((None, None))
    return lookup


def measured_rir_paths(room_wav_dir, cand_set, receiver_node, merge_map=None):
    """Per candidate, the measured file to score (canonical first), or ``None``."""
    return [path for path, _node in _measured_rir_lookup(room_wav_dir, cand_set, receiver_node,
                                                         merge_map=merge_map)]


def load_measured_rirs(room_wav_dir, cand_set, receiver_node, merge_map=None):
    """``(wavs [A, 1, MAX_LEN], available [M], source_nodes [M])`` for the oracle.

    Only the files that exist are loaded: a missing pair shrinks what the oracle
    can report, never the candidate set (plan §2.2). Each file is clamped and
    cropped/padded to ``MAX_LEN`` -- the metric route's own window -- before the
    shared preprocessing runs. ``source_nodes`` names the node whose file was
    actually read, which differs from the candidate under a Rev 3.2 merge.
    """
    lookup = _measured_rir_lookup(room_wav_dir, cand_set, receiver_node, merge_map=merge_map)
    available = [path is not None for path, _node in lookup]
    source_nodes = [node for _path, node in lookup]
    wavs = []
    for path, _node in lookup:
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
    return torch.cat(wavs, dim=0), available, source_nodes


def run_query_gt_rir(engine, cand_set, room_wav_dir, receiver_node, obs_wav, merge_map=None):
    """Measured-RIR oracle (plan §2.6): score the real RIRs, generate nothing.

    Needs no FLAC checkpoint, so it runs the moment the dataset lands. Candidates
    whose measured file is missing get a placeholder similarity and are marked
    unavailable, so the row still carries the full candidate set while the
    prediction is taken over the available ones only.
    """
    timings = {name: 0.0 for name in PROBE_COMPONENTS if name != "context"}
    with _timed(timings, "decode", engine.device):         # file load stands in for decode
        wavs, available, source_nodes = load_measured_rirs(room_wav_dir, cand_set, receiver_node,
                                                           merge_map=merge_map)
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
            "identity_index": gt_index, "oracle_source_nodes": source_nodes,
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
    parser.add_argument("--mode", choices=["run", "readback", "scorer-noise", "reaggregate",
                                           "metrics-calibrate", "metrics-retrieval",
                                           "metrics-report"],
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
    parser.add_argument("--metrics-rows", default=None, metavar="METRICS.jsonl",
                        help="seen metrics-JSONL consumed by --mode metrics-calibrate")
    parser.add_argument("--rows", nargs="+", default=None,
                        help="rows JSONL file(s) for --mode reaggregate")
    parser.add_argument("--noise-draws", type=int, default=100,
                        help="sampled-readout draws for --mode scorer-noise (§2.8.3)")
    parser.add_argument("--noise-wavs", nargs="+", default=None,
                        help="explicit RIR files for --mode scorer-noise")
    parser.add_argument("--noise-wav-count", type=int, default=4,
                        help="how many split RIRs to measure when --noise-wavs is absent")
    parser.add_argument("--readback-decode-all", action="store_true",
                        help="--mode readback: decode EVERY wav of the split, not one per "
                             "(room, source); finds silent/short/corrupt individual files")
    parser.add_argument("--dump-waveforms", default=None, metavar="DIR",
                        help="save the exactly-as-scored predicted RIRs per query "
                             "(announcement 08); generative runs only")
    parser.add_argument("--metrics", action="store_true",
                        help="compute the R4 non-AGREE metric families on the replay pass "
                             "(requires --dump-waveforms: one snapshot feeds both)")
    parser.add_argument("--metric-delta-max", type=int, default=None,
                        help="registered M1/M5 alignment bound; must be on rir_metrics' grid")
    parser.add_argument("--metric-t30-backend", choices=["pyroomacoustics", "torch"],
                        default="pyroomacoustics")
    parser.add_argument("--metric-sensitivities", action="store_true",
                        help="compute the declared seen sensitivity battery on every "
                             "SENSITIVITY_STRIDE-th query (calibration passes only)")
    parser.add_argument("--metric-secondaries", action="store_true",
                        help="also compute the declared secondaries: M2 complex-STFT, "
                             "M3 band/hilbert envelopes and M5 GCC-PHAT")
    parser.add_argument("--report-input", nargs="+", default=None,
                        metavar="SEED:METRICS_JSONL:ROWS_JSONL",
                        help="--mode metrics-report: one published unseen pass per seed, as "
                             "seed:metrics-jsonl:replay-rows-jsonl (repeatable)")
    parser.add_argument("--fa-parity-check", action="store_true",
                        help="run the exp_20 frame-average parity gate on ONE real query "
                             "(driver conditioning vs an eval_FLAC-faithful replay) and write "
                             "its evidence record; requires --cond-method fa_invariant")
    parser.add_argument("--oracle-inputs", nargs="+", default=None,
                        metavar="SEED:ORACLE_ROWS_JSONL",
                        help="--mode metrics-report: the published --mode metrics-retrieval "
                             "rows of each seed, for the measured-candidate oracle ceiling")
    parser.add_argument("--report-registration", default=None, metavar="PATH",
                        help="--mode metrics-report: the FROZEN metric registration manifest "
                             "(required); the report must cover exactly its seeds and families")
    parser.add_argument("--report-expect-queries", type=int, default=None,
                        help="--mode metrics-report: queries each seed must cover (default: "
                             "the registered split size resolved from --dataset-config)")
    parser.add_argument("--report-seen", default=None, metavar="METRICS_JSONL:ROWS_JSONL",
                        help="--mode metrics-report: the SEEN calibration pass, for the "
                             "seen-vs-unseen control and the sensitivity battery")
    parser.add_argument("--report-families", nargs="+", default=None,
                        help="--mode metrics-report: restrict the reported families "
                             "(default: the five primaries plus the declared secondaries)")
    parser.add_argument("--report-bootstrap", type=int, default=None,
                        help="--mode metrics-report: room-clustered bootstrap resamples "
                             "(default: the registered 10000)")
    parser.add_argument("--report-hash-inputs", action="store_true",
                        help="--mode metrics-report: record a sha256 of every input stream")
    parser.add_argument("--verify-context-digest", default=None, metavar="SHA256",
                        help="--mode metrics-retrieval: refuse unless the pass's own context "
                             "draw matches the paired replay's context-stream digest")
    parser.add_argument("--calibration-identities", default=None, metavar="PATH",
                        help="committed identity stream the calibration rows must match "
                             "exactly (a rows JSONL or a JSON list of query ids)")
    parser.add_argument("--register-seeds", default=None,
                        help="seeds to lock into the draft metric manifest, e.g. '42 43 44'")
    parser.add_argument("--register-candidate-manifest", default=None,
                        help="candidate-manifest sha256 to lock into the draft")
    parser.add_argument("--register-identity-digest", default=None,
                        help="R2 identity-stream digest to lock into the draft")
    parser.add_argument("--register-r2-manifest", nargs="+", default=None,
                        help="R2/R2b registration manifests to digest into the draft")
    parser.add_argument("--metric-registration", default=None, metavar="MANIFEST.json",
                        help="committed metric-registration manifest (required for --metrics "
                             "on an unseeneval config)")
    parser.add_argument("--verify-against", default=None, metavar="ROWS.jsonl",
                        help="regenerate and verify every per-sample similarity against a "
                             "published rows file (announcement 08 back-fill)")
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
        if args.mode == "metrics-retrieval":
            for flag, value in (("--ckpt-path", args.ckpt_path),
                                ("--dump-waveforms", args.dump_waveforms),
                                ("--verify-against", args.verify_against)):
                if value:
                    _refuse(f"{flag} is meaningless under --mode metrics-retrieval: it "
                            "generates nothing")
            if args.metric_delta_max is not None:
                from src.localization.rir_metrics import M1_DELTA_GRID
                if int(args.metric_delta_max) not in M1_DELTA_GRID:
                    _refuse(f"--metric-delta-max {args.metric_delta_max} is not on the "
                            f"registered grid {M1_DELTA_GRID}")
        if args.mode == "metrics-report":
            if not args.report_input:
                _refuse("--report-input is required for --mode metrics-report: one "
                        "seed:metrics-jsonl:rows-jsonl triple per published seed")
            for spec in args.report_input:
                if len(str(spec).split(":")) != 3:
                    _refuse(f"--report-input {spec!r} is not seed:metrics-jsonl:rows-jsonl")
            for spec in (args.oracle_inputs or []):
                if len(str(spec).split(":")) != 2:
                    _refuse(f"--oracle-inputs {spec!r} is not seed:oracle-rows-jsonl")
            if not args.report_registration:
                _refuse("--report-registration is required for --mode metrics-report: the "
                        "report is bound to the frozen metric manifest, not to whichever "
                        "files an operator happened to pass")
            if not args.registration_sha:
                _refuse("--registration-sha is required with --report-registration")
            if args.report_seen and len(str(args.report_seen).split(":")) != 2:
                _refuse(f"--report-seen {args.report_seen!r} is not metrics-jsonl:rows-jsonl")
        if args.mode == "metrics-calibrate" and not args.metrics_rows:
            _refuse("--metrics-rows is required for --mode metrics-calibrate")
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
        for flag, value in (("--dump-waveforms", args.dump_waveforms),
                            ("--ckpt-path", args.ckpt_path),
                            ("--parity-check", args.parity_check),
                            ("--control constant_source", args.control == "constant_source")):
            if value:
                _refuse(f"{flag} is meaningless under --score-source gt_rir (no generation "
                        "happens); refusing rather than recording a protocol that did not run")
    if args.metrics:
        if not args.dump_waveforms:
            _refuse("--metrics requires --dump-waveforms: the metric families and the npz dump "
                    "must consume ONE snapshot of the scored waveforms, never a re-decode")
        if args.metric_delta_max is not None:
            from src.localization.rir_metrics import M1_DELTA_GRID
            if int(args.metric_delta_max) not in M1_DELTA_GRID:
                _refuse(f"--metric-delta-max {args.metric_delta_max} is not on the registered "
                        f"grid {M1_DELTA_GRID}")
    if args.verify_against and not args.dump_waveforms:
        _refuse("--verify-against is the announcement-08 back-fill: it must also "
                "--dump-waveforms, otherwise the pass verifies but saves nothing")
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

    fa_execution = {}

    def conditioner(metadata, _device):
        with cond_autocast_ctx():
            return conditioning_call(args.cond_method, module.diffusion.conditioner,
                                     metadata, module.device, frame_avg_angles,
                                     record=fa_execution)

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
               "fa_execution": fa_execution,
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


DUMP_README = """# exp_18 predicted-RIR waveform dump

One `.npz` per scored query, named `<position>_<sanitized query id>.npz`, where
`position` is the query's index in the scored stream. Each file holds two float32
arrays:

- `pred` -- `[M, K, 10240]`: the generated RIRs EXACTLY as scored (rectified-flow
  sample -> pretransform decode -> clamp to [-1, 1]). The M axis is the candidate
  axis and its order is the candidate order recorded in `*_waveforms.json`
  (`candidate_nodes` / `candidate_xyz_world`, canonical node order); the K axis is the sample axis in
  generation order (noise draw k).
- `obs` -- `[10240]`: the observed RIR h_obs in the same scored window.

Window convention: 22050 Hz, the first 10240 samples (the loader's pad-crop),
clamped to [-1, 1]. The AGREE scorer additionally consumes only the FIRST 8000
samples of each waveform, so a waveform-level analysis that compares `pred`
against `obs` should state which window it uses -- the full 10240 dumped here, or
the 8000 the similarity numbers were computed on.

`*_waveforms.json` carries, per query, the file path + sha256 and the geometry
(room, GT node/xyz, candidate nodes/xyz in M-axis order, predicted index), so
this directory is self-contained for external analysis. The rows JSONL remains
the authority for the similarities themselves.

```python
import numpy as np
with np.load("0000000_0_single_channel_ir_1_Cafe_Cafe_idx_1_S003_R011_hybrid_IR_wav.npz") as z:
    pred, obs = z["pred"], z["obs"]      # [M, K, 10240], [10240]
print(pred.shape, obs.shape, pred.dtype)
```
"""


def sha256_bytes(payload):
    """sha256 over raw bytes (the dumped .npz file)."""
    return hashlib.sha256(payload).hexdigest()


def waveform_filename(position, query_id):
    """``<position>_<sanitized query id>.npz`` -- stream order first, so a
    directory listing is the scored order."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(query_id)).strip("_")
    return f"{int(position):07d}_{safe}.npz"


def prepare_dump_dir(args):
    """Create (or validate) the waveform dump directory.

    A dump that mixes two runs' waveforms would be unusable, so a non-empty
    directory is refused unless --overwrite. The README is written at creation so
    the directory explains itself to whoever analyses it later.
    """
    dump_dir = str(args.dump_waveforms)
    if os.path.isdir(dump_dir):
        existing = [name for name in os.listdir(dump_dir) if not name.startswith(".")]
        if existing and not bool(getattr(args, "overwrite", False)):
            _refuse(f"--dump-waveforms directory {dump_dir!r} is not empty "
                    f"({len(existing)} entries); pass --overwrite to reuse it")
    else:
        os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, "README.md"), "w") as handle:
        handle.write(DUMP_README)
    return dump_dir


def dump_query_waveforms_legacy(dump_dir, position, query_id, wavs, num_candidates,
                                num_samples, obs_wav):
    """The r7-era dump route, preserved LITERALLY for runs with --metrics off.

    The registered R2/R2b runs take this path: no snapshot object, no digest pass
    over the array, exactly the r7 expression for both arrays (r4m review finding
    7). A byte-level golden test compares it against the r7 module itself.
    """
    pred = wavs.detach().cpu().float().reshape(num_candidates, num_samples, -1).numpy()
    obs = obs_wav.detach().cpu().float().reshape(-1).numpy()
    name = waveform_filename(position, query_id)
    path = os.path.join(str(dump_dir), name)
    tmp = path + ".partial"
    with open(tmp, "wb") as handle:
        np.savez(handle, pred=pred, obs=obs)
    os.replace(tmp, path)
    with open(path, "rb") as handle:
        return name, sha256_bytes(handle.read())


def waveform_snapshot(wavs, num_candidates, num_samples, obs_wav):
    """The ONE immutable float32 snapshot of a query's scored waveforms.

    R4-COMPOSITION GUARD (r7 review): the npz dump and the R4 metric families must
    consume the same numbers -- no second decode, no in-place op -- so both are
    served from this single detached CPU copy, and its digest is re-checked after
    every consumer has run.
    """
    pred = wavs.detach().to(torch.float32).cpu().reshape(
        num_candidates, num_samples, -1).contiguous()
    obs = obs_wav.detach().to(torch.float32).cpu().reshape(-1).contiguous()
    return {"pred": pred, "obs": obs, "sha256": snapshot_digest(pred, obs)}


def snapshot_digest(pred, obs):
    """Digest of the snapshot's bytes, used to prove no consumer mutated it."""
    digest = hashlib.sha256()
    digest.update(pred.numpy().tobytes())
    digest.update(obs.numpy().tobytes())
    return digest.hexdigest()


def dump_query_waveforms(dump_dir, position, query_id, snapshot):
    """Write one query's ``pred``/``obs`` arrays; return ``(relpath, sha256)``."""
    pred = snapshot["pred"].numpy()
    obs = snapshot["obs"].numpy()
    name = waveform_filename(position, query_id)
    path = os.path.join(str(dump_dir), name)
    tmp = path + ".partial"
    with open(tmp, "wb") as handle:
        np.savez(handle, pred=pred, obs=obs)
    os.replace(tmp, path)
    with open(path, "rb") as handle:
        return name, sha256_bytes(handle.read())


def process_query(args, engine, context, md, obs_wav, dump=None, sink=None):
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
        room_entry = manifest["rooms"][room_id]
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
                                       receiver_node, obs_wav,
                                       merge_map=room_entry.get("merge_map"))
            noise_keys, available = [], outcome["available"]
            identity_index = outcome["identity_index"]
        else:
            noise = build_noise_bank(args.seed, query_id, args.num_samples,
                                     context["latent_shape"])
            noise_keys = [noise_key(args.seed, query_id, k)
                          for k in range(int(args.num_samples))]
            outcome = run_query(engine, md, cand_set, noise, obs_wav,
                                batch_size=args.batch_size, control=args.control,
                                return_wavs=dump is not None)
            available, identity_index = None, None

        timings.update(outcome.get("timings_s") or {})
        with _timed(timings, "context", engine.device):
            evidence = context_evidence(engine, md, obs_wav) or {}

    row = build_row(
        query_id=query_id, room_id=room_id, relpath=md["relpath"], receiver_node=receiver_node,
        cand_set=cand_set, cam_xyz=outcome["cand_cam_xyz"], sims=outcome["sims"],
        context_mask=context_mask,
        noise_keys=noise_keys, tau=args.tau, agg=args.agg, control=args.control,
        score_source=args.score_source, smoke=bool(args.smoke), available=available,
        identity_index=identity_index, substituted=False,
        context_xyz_cam=evidence.get("context_xyz_cam"),
        context_sims_hex=evidence.get("context_sims_hex"),
        timings=timings, merge_map=room_entry.get("merge_map"),
        oracle_source_nodes=outcome.get("oracle_source_nodes"))
    attach_fa_execution(row, (context or {}).get("fa_execution"))
    if dump is not None and outcome.get("wavs") is not None:
        if sink is None:
            # --metrics off: the r7 route, unchanged (finding 7 / R2b firewall)
            row["waveform_path"], row["waveform_sha256"] = dump_query_waveforms_legacy(
                dump["dir"], dump["position"], row["query_id"], outcome["wavs"],
                outcome["num_candidates"], outcome["num_samples"], obs_wav)
        else:
            snapshot = waveform_snapshot(outcome["wavs"], outcome["num_candidates"],
                                         outcome["num_samples"], obs_wav)
            row["waveform_path"], row["waveform_sha256"] = dump_query_waveforms(
                dump["dir"], dump["position"], row["query_id"], snapshot)
            sink["snapshot"] = snapshot
    return row


def run_evaluation(args, loader, engine, context, ckpt_sha256, agree_sha256, expected=None,
                   dataset_config=None, paths=None):
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

    paths = paths or artifact_paths(args)      # main claims them before any model load
    rows_path, summary_path = paths["rows"], paths["summary"]
    partial_rows = rows_path + ".partial"
    dump_dir = prepare_dump_dir(args) if getattr(args, "dump_waveforms", None) else None
    published, published_sha = ({}, None)
    if getattr(args, "verify_against", None):
        # r7 review LOW: prove we are replaying the SAME protocol before generating
        preflight = preflight_verify_against(args, expected_queries=len(expected))
        published, published_sha = load_published_rows(args.verify_against)
        print(f"replay preflight passed: {preflight['n_rows']} published rows, "
              f"sha256={published_sha[:12]}..., provenance checked="
              f"{preflight['provenance_checked']}")

    metrics_config = None
    if getattr(args, "metrics", False):
        # a registered run's config comes from the verified manifest, never from
        # the CLI (r4m review finding 1)
        metrics_config = (getattr(args, "registered_metric_config", None)
                          or metric_config_from_args(args))
    metrics_rows, metrics_path, metrics_handle = [], None, None
    if metrics_config is not None:
        metrics_path = os.path.join(os.path.dirname(rows_path),
                                    f"{artifact_stem(args)}_metrics.jsonl")
        metrics_handle = open(metrics_path + ".partial", "w")
        print(f"R4 metrics enabled: {metrics_config.payload()}")
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
            sink = {} if metrics_config is not None else None
            row = process_query(args, engine, context, md, obs_wav,
                                dump=None if dump_dir is None
                                else {"dir": dump_dir, "position": position}, sink=sink)
            if metrics_config is not None:
                metrics_rows.append(score_query_metrics(
                    args, row, position, sink.get("snapshot"), md, metrics_config,
                    metrics_handle))
            if published_sha is not None:
                verify_row_against_published(row, published)
            write_row(handle, row)
            rows.append(row)
            scored.append(identity)
            if row["room_id"] not in seen_rooms:
                seen_rooms.add(row["room_id"])
                print(f"[{len(rows)}/{len(expected)}] room {row['room_id']}")

    if metrics_handle is not None:
        metrics_handle.close()      # flushed -- but NOT named until every gate passes
    if context.get("context_k") is not None:
        assert_context_evidence_complete(rows, context["context_k"])
    fa_locked = context.get("fa_locked_plan")
    if fa_locked:
        print(f"fa partition gate passed: {assert_fa_execution_matches(rows, fa_locked)} "
              "queries executed the registered plan")
    split = assert_scored_stream(scored, expected)
    print(f"identity gate passed: {len(scored)} queries, split_hash={split[:12]}...")

    summary = summarize_run(rows)
    summary["probe"] = probe_summary(rows, read_peak_memory(context.get("device", "cpu")))
    if metrics_config is not None:
        summary["metrics"] = {"n_queries": len(metrics_rows), "path": metrics_path,
                              "families": sorted({family for row in metrics_rows
                                                  for family in row["families"]
                                                  if "_delta" not in family}),
                              "config": metrics_config.payload()}
    if published_sha is not None:
        summary["verify_against"] = {"rows_path": str(args.verify_against),
                                     "rows_sha256": published_sha,
                                     "n_verified": len(rows), "all_match": True}
    manifest = context.get("manifest")
    provenance = build_provenance(args, ckpt_sha256, agree_sha256, split,
                                  context["weights_source"], len(rows),
                                  dataset_config=dataset_config,
                                  context_digest=context_stream_digest(rows),
                                  candidate_manifest_sha256=(manifest_sha256(manifest)
                                                             if manifest else None),
                                  split_file_sha256=(_file_sha256(split_path_of(dataset_config))
                                                     if dataset_config else None),
                                  merge_groups=(merge_group_count(manifest)
                                                if manifest else None),
                                  metric_registerable=(metric_registerable_payload()
                                                       if metrics_config is not None else None))
    write_summary(summary_path, summary, provenance)
    waveform_manifest_path = None
    if dump_dir is not None:
        waveform_manifest_path = write_json_atomic(
            os.path.join(os.path.dirname(rows_path), f"{artifact_stem(args)}_waveforms.json"),
            build_waveform_manifest(args, rows, dump_dir, os.path.basename(rows_path)),
            overwrite=True)
    if manifest is not None:
        write_json_atomic(paths["manifest"], manifest, overwrite=True)
    # Publish as ONE set, and only here: the identity loop, the context end-gate,
    # the scored-stream gate and summary construction have all passed by now, so
    # no final-named artifact can describe a run that failed its own gates
    # (r4m3 finding 6 residual -- the rename used to sit above the end gates).
    if metrics_handle is not None:
        os.replace(metrics_path + ".partial", metrics_path)
    os.replace(partial_rows, rows_path)          # publish only verified artifacts
    return {"rows_path": rows_path, "summary_path": summary_path,
            "manifest_path": paths["manifest"], "rows": rows, "metrics_path": metrics_path,
            "metrics_rows": metrics_rows,
            "waveform_manifest_path": waveform_manifest_path,
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
    if args.mode == "metrics-calibrate":
        return run_metrics_calibrate(args)
    if args.mode == "metrics-report":
        return run_metrics_report(args)
    if getattr(args, "fa_parity_check", False):
        return run_fa_parity_check(args)
    if args.mode == "metrics-retrieval":
        dataset_config = load_dataset_config(args)
        assert_registered_split(args, dataset_config)
        assert_metric_registration(args, dataset_config)
        model_config = read_model_config(args)
        manifest = manifest_for_dataset_config(dataset_config)
        if args.metric_registration:
            args.registered_metric_config = verify_metric_registration(
                args, dataset_config, candidate_manifest_sha256=manifest_sha256(manifest),
                identity_digest=split_hash(
                    expected_split_identities_from_config(dataset_config)))
        pl.seed_everything(args.seed, workers=True)
        loader = build_dataloader(args, model_config, dataset_config)
        context = {"manifest": manifest, "context_k": resolve_context_k(dataset_config),
                   "device": args.device, "weights_source": "n/a"}
        return run_metrics_retrieval(args, loader, None, context,
                                     dataset_config=dataset_config)

    # ---- everything cheap and refusable first (r4 review M5) -----------------
    # configs read + hashed, registration verified, context resolved, output
    # targets claimed -- all before a single byte of a checkpoint is deserialized.
    dataset_config = load_dataset_config(args)
    validate_dataset_split(args, dataset_config)
    model_config = read_model_config(args)
    context_k = resolve_context_k(dataset_config)
    split_check = assert_registered_split(args, dataset_config)
    assert_registration_sha(args, dataset_config)
    assert_metric_registration(args, dataset_config)

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
        "readout": "mean", "candidate_manifest_sha256": manifest_hash,
        "split_file_sha256": split_check["file_sha256"],
        "batch_size": args.batch_size, "num_workers": args.num_workers})
    registered_metric_config = None
    if getattr(args, "metric_registration", None):
        registered_metric_config = verify_metric_registration(
            args, dataset_config,
            candidate_manifest_sha256=manifest_hash if manifest_hash != "n/a" else None,
            identity_digest=split_hash(expected_split_identities_from_config(dataset_config)))
    args.registered_metric_config = registered_metric_config
    paths = None if args.parity_check else artifact_paths(args)

    # ---- only now: deserialize, construct, load data ------------------------
    ckpt = load_checkpoint_and_validate(args, model_config)
    torch.set_float32_matmul_precision("medium")
    agree = load_agree_audio(args.agree_ckpt, args.device)
    if args.score_source == "gt_rir":
        engine = scoring_only_engine(agree, args.device)
        context = {"weights_source": "n/a", "latent_shape": None, "device": args.device}
    else:
        engine, context = build_engine(args, agree=agree, device=args.device,
                                       model_config=model_config, ckpt=ckpt)

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
    context["context_k"] = context_k
    # exp_20 F4: every FA query is held to the registered partition at the end gate
    context["fa_locked_plan"] = getattr(args, "fa_locked_plan", None) or (
        {field: value for field, value in (fa_protocol_state(args) or {}).items()
         if field in FA_EXECUTION_FIELDS} or None)
    print(f"candidate manifest frozen: {len(candidate_manifest['rooms'])} rooms, "
          f"sha256={manifest_hash[:12]}...")
    result = run_evaluation(args, loader, engine, context, ckpt_sha256, agree_sha256,
                            expected=expected, dataset_config=dataset_config, paths=paths)
    summary = result["summary"]
    print(f"rows:    {result['rows_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"primary ({summary['flac']['primary_name']}): {summary['flac']['primary']:.4f} m over "
          f"{summary['n_queries']} queries / {summary['n_rooms']} rooms")
    print(f"context-conditioned baseline: "
          f"{summary['baselines']['context_conditioned']['primary']:.4f} m")
    return result



def read_model_config(args):
    """Read the model config and refuse a foreign objective -- no deserialization."""
    with open(args.model_config) as handle:
        model_config = json.load(handle)
    assert_rectified_flow(model_config)
    return model_config


def conditioning_call(cond_method, conditioner, metadata, device, frame_avg_angles,
                      record=None):
    """One conditioning forward, with the frame-average CHUNK PLAN made explicit.

    ``invariant_conditioning`` partitions the orbit into forwards of at most
    ``max_fwd_samples`` rows; left unset it inherits the module constant (64),
    which for a 10-candidate query puts all four angles in ONE forward. exp_20
    declares the plan instead (announcement 06 §3): the cap is this call's
    candidate micro-batch, so the orbit runs one forward per angle.
    """
    if cond_method == "fa_invariant":
        from src.localization.crossarm import fa_conditioning

        return fa_conditioning(conditioner, metadata, device, frame_avg_angles, record=record)
    if record is not None:
        record.clear()
        record.update({"cond_method": "vanilla", "n_orbit_forwards": 0})
    return conditioner(metadata, device)


#: the executed-partition fields a row publishes and the end gate compares.
FA_EXECUTION_FIELDS = ("cap_policy", "frame_avg_fwd_cap", "candidate_micro_batch",
                       "orbit_size", "angles_per_chunk", "n_orbit_forwards",
                       "shared_angle_count")


def attach_fa_execution(row, observed):
    """Publish the partition a query actually executed (FA rows only).

    Vanilla rows are untouched, so exp_18's row schema is unchanged; an FA row
    carries cheap ints the end gate then checks against the manifest.
    """
    if not observed or observed.get("cond_method") != "fa_invariant":
        return row
    row["fa_execution"] = {key: observed[key] for key in
                           ("partition",) + FA_EXECUTION_FIELDS if key in observed}
    return row


def assert_fa_execution_matches(rows, locked):
    """End gate: EVERY query's executed partition equals the registered plan.

    The manifest locks numbers; this proves the run produced them query by query
    rather than asserting it once at startup (r2 F4).
    """
    checked = 0
    for row in rows:
        observed = row.get("fa_execution")
        if not observed:
            raise SystemExit(f"fa gate ABORT: query {row.get('query_id')!r} carries no "
                             "fa_execution; a frame-average run must publish the partition "
                             "it executed for every query")
        for field in FA_EXECUTION_FIELDS:
            if field not in locked:
                continue
            want, got = locked[field], observed.get(field)
            if isinstance(want, str) or isinstance(got, str):
                same = str(want) == str(got)
            else:
                same = want is not None and got is not None and int(want) == int(got)
            if not same:
                raise SystemExit(
                    f"fa gate ABORT: query {row.get('query_id')!r} executed {field}={got!r} "
                    f"but the registration locks {want!r}; the orbit was partitioned "
                    "differently from the registered plan")
        checked += 1
    return checked


def fa_protocol_state(args):
    """The frame-average protocol this run resolves, or ``None`` for vanilla."""
    from src.localization.crossarm import FA_ANGLES, fa_run_state

    if getattr(args, "cond_method", "vanilla") != "fa_invariant":
        return None
    angles = args.frame_avg_angles if getattr(args, "frame_avg_angles", None) else FA_ANGLES
    return fa_run_state(args.cond_method, frame_avg_angles=angles,
                        rotate_deg=getattr(args, "rotate_deg", 0.0) or 0.0,
                        cond_autocast=getattr(args, "cond_autocast", "default"))


def assert_fa_registration(manifest, args):
    """A frame-average run must have its WHOLE conditioning protocol registered.

    Inert for vanilla runs, so every manifest exp_18 committed stays valid.
    """
    from src.localization.crossarm import fa_reasons

    state = fa_protocol_state(args)
    if state is None:
        return None
    for reason in fa_reasons(manifest, state):
        _refuse(reason)
    # a registration commit need only be an ANCESTOR, so the executable FA source
    # can move afterwards: the manifest pins the blobs and the run compares them
    # the numbers the gate will hold every query to: the manifest's when a
    # manifest was verified, else the run's own declared plan
    args.fa_locked_plan = {field: manifest[field] for field in FA_EXECUTION_FIELDS
                           if field in manifest} or {
        field: state[field] for field in FA_EXECUTION_FIELDS if field in state}
    pinned = manifest.get("fa_source_shas")
    if pinned:
        from src.localization.crossarm import fa_source_shas

        current = fa_source_shas()
        for relpath in sorted(pinned):
            if pinned[relpath] != current.get(relpath):
                _refuse(f"registered fa source {relpath} is {str(pinned[relpath])[:12]}... "
                        f"but this run executes {str(current.get(relpath))[:12]}...; the "
                        "frame-average code changed after registration")
    return state


def load_checkpoint_and_validate(args, model_config):
    """Deserialize the checkpoint and refuse an ARE artifact (CPU, no model build).

    Also binds ``--cond-method`` to the checkpoint itself where the file can
    answer: exp_20's arms embed their training config, so conditioning a
    frame-averaged model as vanilla (or the reverse) is refused here, before any
    GPU work. The released EMA checkpoint embeds no config at all -- for that
    file the method is not detectable and the binding rests on the registration
    manifest, which the run records rather than hides.
    """
    if args.score_source != "flac":
        return None
    from src.localization.crossarm import cond_method_binding

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    assert_no_are(ckpt.get("model_config"), copy.deepcopy(model_config))
    binding = cond_method_binding(
        ckpt, getattr(args, "cond_method", "vanilla"),
        manifest=getattr(args, "verified_registration_manifest", None),
        manifest_verified=bool(getattr(args, "verified_registration_manifest", None)),
        registered=bool(getattr(args, "registration_manifest", None)))
    args.cond_method_binding = binding
    for reason in binding["reasons"]:
        _refuse(reason)
    return ckpt


def load_and_validate_artifacts(args):
    """CPU-only artifact validation, BEFORE anything is built or moved to a device.

    Reads the model config and (for generative runs) the checkpoint's metadata and
    refuses a foreign objective or an ARE artifact there and then -- previously
    those refusals lived in ``build_engine``, i.e. after the AGREE scorer had
    already been constructed on the target device (r3 review finding 9).
    """
    model_config = read_model_config(args)
    return model_config, load_checkpoint_and_validate(args, model_config)


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
                sources = enumerate_metadata_sources(meta_dir, allow_duplicate_positions=True)
            except ValueError as err:
                raise SystemExit(f"candidate manifest ABORT for {room_id}: {err}")
            # Rev 3.2: two labels at one position are ONE candidate.
            merged, merge_groups = merge_position_duplicates(sources)

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
                for node in sorted(sources):               # any member proves the frame
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

            nodes = sorted(merged)
            rooms[room_id] = {
                "scene": scene,
                "scene_id": scene_id,
                "nodes": nodes,
                # Only NON-TRIVIAL groups, so a clean room's map is simply {}. A
                # clean room's CANDIDATES and SCORES are unchanged by the merge;
                # rows/provenance gained merge_map, oracle_source_nodes and
                # candidate_merge_groups fields, and the manifest schema (hence its
                # sha256) changed -- "computation-identical", not "byte-identical".
                "merge_map": {str(canonical): members
                              for canonical, members in sorted(merge_groups.items())
                              if len(members) > 1},
                "member_nodes": sorted(sources),
                "xyz_world": [[float(v) for v in merged[node]] for node in nodes],
                "wav_nodes": sorted(wav_nodes),
                "receivers": receivers,
                "n_metadata_sources": len(nodes),
                "n_member_sources": len(sources),
                "n_wav_sources": len(wav_nodes),
            }
    return {"dataset_root": dataset_root, "folder_name": folder_name, "rooms": rooms}


def manifest_sha256(manifest):
    """sha256 over the canonical JSON of the frozen manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_node(room, node):
    """The candidate a source node belongs to (Rev 3.2): itself, or its group's
    canonical node when it was merged into one."""
    node = int(node)
    for canonical, members in (room.get("merge_map") or {}).items():
        if node in [int(m) for m in members]:
            return int(canonical)
    return node


def merge_group_count(manifest):
    """How many non-trivial merge groups the frozen manifest holds."""
    return sum(len(room.get("merge_map") or {}) for room in manifest["rooms"].values())


def candidate_set_from_manifest(manifest, room_id, gt_node, rec_node):
    """The query's candidate set, from the frozen manifest -- no disk access."""
    room = manifest["rooms"].get(room_id)
    if room is None:
        raise ValueError(f"room {room_id!r} is not in the frozen candidate manifest")
    rec_loc = room["receivers"].get(str(int(rec_node)))
    if rec_loc is None:
        raise ValueError(f"receiver {rec_node} of {room_id} is not in the frozen manifest")
    nodes = [int(n) for n in room["nodes"]]
    gt_node = canonical_node(room, gt_node)
    if gt_node not in nodes:
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
    "readout", "candidate_manifest_sha256", "split_file_sha256",
)

#: The loader parallelism is part of the pinned evaluation protocol (O8): it
#: fixes iteration and worker determinism across arms. It is NOT the
#: frame-average micro-batch -- that is the query's candidate count (M = 10),
#: since the conditioner is called once per query with the whole candidate set
#: (r2 re-review nit). exp_18's registrations are frozen and do not lock these,
#: so they are checked WHEN LOCKED and REQUIRED of any manifest naming an arm.
REGISTRATION_MATCHED_IF_PRESENT = ("batch_size", "num_workers")


#: an immutable git object id: 40-hex (sha1) or 64-hex (sha256 repositories).
_FULL_OBJECT_ID = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


def _repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def verify_registration_commit(manifest_path, registration_sha, repo_root=None):
    """Prove the manifest IS the registered one; return it and the resolved id.

    Four separate requirements (r4 review M4): the value must be an immutable
    full object id (``HEAD``, a branch, a tag or an abbreviation is refused by
    FORMAT, since those can move or be ambiguous); it must resolve to a commit;
    that commit must contain byte-identical manifest content; and it must be an
    ANCESTOR of the executing HEAD, which is what makes "registered before the
    run" checkable rather than asserted. The manifest must also live inside the
    repository worktree, or "committed" would be meaningless.
    """
    repo_root = os.path.realpath(str(repo_root or _repo_root()))
    sha = str(registration_sha or "")
    if not _FULL_OBJECT_ID.match(sha):
        _refuse(f"--registration-sha must be a full 40- or 64-hex object id, got {sha!r}; "
                "HEAD, branch/tag names and abbreviations can move and are refused")

    manifest_path = os.path.realpath(str(manifest_path))
    if not os.path.isfile(manifest_path):
        _refuse(f"--registration-manifest not found: {manifest_path}")
    if os.path.commonpath([manifest_path, repo_root]) != repo_root:
        _refuse(f"--registration-manifest {manifest_path!r} is outside the repository "
                f"{repo_root!r}; a file that is not in the repo cannot have been committed")
    relpath = os.path.relpath(manifest_path, repo_root)

    if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                      cwd=repo_root, capture_output=True).returncode != 0:
        _refuse(f"--registration-sha {sha!r} does not resolve to a commit in {repo_root}")
    show = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=repo_root, capture_output=True)
    if show.returncode != 0:
        _refuse(f"commit {sha} does not contain {relpath!r}; the registration manifest was not "
                "committed before the run (O17)")
    with open(manifest_path, "rb") as handle:
        local = handle.read()
    if show.stdout != local:
        _refuse(f"{relpath!r} differs from the version committed at {sha}; the registered "
                "protocol was edited after registration")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                      cwd=repo_root, capture_output=True).returncode != 0:
        _refuse(f"registration commit {sha} is not an ancestor of the executing HEAD; the "
                "protocol must be registered on the history this run is executing")
    resolved = subprocess.run(["git", "rev-parse", f"{sha}^{{commit}}"], cwd=repo_root,
                              capture_output=True, text=True)
    return {"manifest": json.loads(local.decode("utf-8")),
            "resolved_sha": resolved.stdout.strip() or sha}


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

    arm_bound = manifest.get("arm") is not None
    for field in REGISTRATION_MATCHED_IF_PRESENT:
        if field not in manifest:
            if arm_bound:
                _refuse(f"registration manifest names arm {manifest['arm']!r} but does not "
                        f"lock {field!r}; the loader parallelism decides the candidate "
                        "micro-batch and is part of an exp_20 protocol")
            continue
        locked, actual = manifest[field], resolved.get(field)
        if locked != actual:
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
    verified = verify_registration_commit(args.registration_manifest, args.registration_sha,
                                          repo_root=repo_root)
    check_registration_fields(verified["manifest"], resolved, registered)
    assert_fa_registration(verified["manifest"], args)       # inert unless fa_invariant
    # the resolved immutable id is what provenance records, not the string typed
    args.registration_sha_resolved = verified["resolved_sha"]
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
#: the CANONICAL unseen split, pinned by content (r5 review, H2 residual).
#: The gate used to read its authority from the very file it was validating, so a
#: truncated or same-shaped substituted split passed and the identity audit then
#: agreed with the altered authority.
UNSEEN_SPLIT_FILE_SHA256 = "9a9d817abc3e19f41351e07325ffa929c1f2846d0c77e80d538d9e4a21342ba8"
UNSEEN_SPLIT_N_FILES = 6337
UNSEEN_SPLIT_N_ROOMS = 17
UNSEEN_ROOM_NODE_MAP_SHA256 = "38c07598fc070cff50684907ce6bcd5d5bee29a249693a441d41b78eafef33a0"
#: shortest waveform every scoring path can consume unchanged. The target is
#: pad_crop'd to sample_size = 10240, context RIRs to 9600 and the oracle to
#: 8000, so a wav >= 10240 is score-identical however long it is, while a shorter
#: one silently changes oracle, context and query scores.
MIN_WAV_SAMPLES = 10240


def room_node_map_digest(split):
    """sha256 over ``{room_id: sorted source nodes}`` derived from the split's own
    file names -- the identity of the split's room/source structure."""
    node_map = {}
    for scene in split:
        for scene_id, files in split[scene].items():
            node_map[f"{scene}/{scene_id}"] = sorted({parse_ir_filename(f)[0] for f in files})
    payload = json.dumps(node_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_registered_split(split_path, enforced=True):
    """Check a split file against the pinned canonical unseen split."""
    with open(str(split_path), "rb") as handle:
        raw = handle.read()
    split = json.loads(raw.decode("utf-8"))
    checks = {
        "enforced": bool(enforced),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "n_files": sum(len(files) for rooms in split.values() for files in rooms.values()),
        "n_rooms": sum(len(rooms) for rooms in split.values()),
        "room_node_map_sha256": room_node_map_digest(split),
        "expected": {"file_sha256": UNSEEN_SPLIT_FILE_SHA256, "n_files": UNSEEN_SPLIT_N_FILES,
                     "n_rooms": UNSEEN_SPLIT_N_ROOMS,
                     "room_node_map_sha256": UNSEEN_ROOM_NODE_MAP_SHA256},
        "failures": [],
    }
    if not enforced:
        return checks
    if checks["file_sha256"] != UNSEEN_SPLIT_FILE_SHA256:
        checks["failures"].append(
            f"split byte digest {checks['file_sha256'][:16]}... != the registered "
            f"{UNSEEN_SPLIT_FILE_SHA256[:16]}...")
    if checks["n_files"] != UNSEEN_SPLIT_N_FILES:
        checks["failures"].append(
            f"split enumerates {checks['n_files']} identities, the registered unseen split has "
            f"{UNSEEN_SPLIT_N_FILES}")
    if checks["n_rooms"] != UNSEEN_SPLIT_N_ROOMS:
        checks["failures"].append(
            f"split enumerates {checks['n_rooms']} rooms, the registered unseen split has "
            f"{UNSEEN_SPLIT_N_ROOMS}")
    if checks["room_node_map_sha256"] != UNSEEN_ROOM_NODE_MAP_SHA256:
        checks["failures"].append(
            f"split room/node map digest {checks['room_node_map_sha256'][:16]}... != the "
            f"registered {UNSEEN_ROOM_NODE_MAP_SHA256[:16]}...")
    return checks


def split_path_of(dataset_config):
    """The split JSON the dataset config points at."""
    entry = (dataset_config.get("datasets") or [None])[0]
    if entry is None:
        _refuse("the dataset config declares no datasets")
    return entry["json_file_path"]


def assert_registered_split(args, dataset_config):
    """Run-mode startup gate: an unseen run must read the CANONICAL split."""
    # skip_split_digest exists only for synthetic test fixtures with the registered
    # SHAPE; no CLI flag sets it, so an operator cannot turn the gate off.
    enforced = bool(dataset_config.get("unseeneval", False)) and not getattr(
        args, "skip_split_digest", False)
    checks = verify_registered_split(split_path_of(dataset_config), enforced=enforced)
    if checks["failures"]:
        _refuse("the dataset config does not point at the registered unseen split: "
                + "; ".join(checks["failures"]))
    return checks


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
    decode_all = bool(getattr(args, "readback_decode_all", False))
    registered = bool(dataset_config.get("unseeneval", False))
    split_check = verify_registered_split(
        entry["json_file_path"],
        enforced=registered and not getattr(args, "skip_split_digest", False))

    rooms, failures, warnings = {}, [], list(split_check["failures"] and [])
    failures.extend(split_check["failures"])
    for room_id, room in sorted(manifest["rooms"].items()):
        scene, scene_id = room["scene"], room["scene_id"]
        wav_dir = os.path.join(root, folder, scene, scene_id)
        depth_dir = os.path.join(root, "depth_map", scene, scene_id)
        cross = crosscheck_sources_vs_files(room.get("member_nodes", room["nodes"]), wav_dir)

        split_files = list(split[scene][scene_id])
        split_sources, receivers, missing_files, one_per_source = set(), set(), [], {}
        every_file = {}
        for fname in split_files:
            src, rec = parse_ir_filename(fname)
            split_sources.add(src)
            receivers.add(rec)
            path = os.path.join(wav_dir, fname)
            if not os.path.isfile(path):
                missing_files.append(fname)
            else:
                one_per_source.setdefault(src, path)
                every_file[fname] = path
        # r7 item 4: one wav per (room, source) missed a SILENT file the dataset
        # then substituted mid-run; --readback-decode-all decodes the whole split.
        to_decode = ({fname: path for fname, path in every_file.items()} if decode_all
                     else {os.path.basename(p): p for p in one_per_source.values()})

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

        wav_bad, sample_rates, lengths = [], set(), []
        for label, path in sorted(to_decode.items()):
            try:
                wav, rate = torchaudio.load(path)
            except Exception as err:                       # noqa: BLE001 - reported, not raised
                wav_bad.append(f"{label}: unreadable ({type(err).__name__})")
                continue
            sample_rates.add(int(rate))
            lengths.append(int(wav.shape[-1]))
            src = label
            if int(rate) != AR_SAMPLE_RATE:
                wav_bad.append(f"{src}: sample rate {rate} != {AR_SAMPLE_RATE}")
            elif wav.shape[0] != 1:
                wav_bad.append(f"{src}: {wav.shape[0]} channels, expected mono")
            elif decode_all and is_silence(wav):
                # the dataset substitutes a random other item for a silent file,
                # which corrupts the identity stream mid-run
                wav_bad.append(f"{src}: silent (below the loader's -60 dB threshold)")
            elif wav.shape[-1] < MIN_WAV_SAMPLES:
                # Score-relevant: the target is pad_crop'd to 10240, context RIRs to
                # 9600 and the oracle window to 8000, so a shorter wav changes what
                # every scoring path sees. Anything >= 10240 is score-identical.
                wav_bad.append(f"{src}: {wav.shape[-1]} samples is shorter than the scored "
                               f"prefix ({MIN_WAV_SAMPLES})")
            elif not bool(torch.isfinite(wav).all()):
                wav_bad.append(f"{src}: contains non-finite samples")

        rooms[room_id] = {
            "metadata_nodes": room["nodes"], "wav_nodes": room["wav_nodes"],
            "metadata_only_nodes": cross["missing_files"], "wav_only_nodes": cross["extra_files"],
            "split_files": len(split_files), "split_sources": sorted(split_sources),
            "split_receivers": len(receivers), "missing_split_files": missing_files,
            "depth_checked": len(receivers), "depth_bad": depth_bad,
            "wav_checked": len(to_decode), "wav_bad": wav_bad,
            "sample_rates": sorted(sample_rates),
            "wav_lengths": ({"min": min(lengths), "max": max(lengths),
                             "mean": float(np.mean(lengths))} if lengths else None),
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

    all_lengths = [length for room in rooms.values() if room["wav_lengths"]
                   for length in (room["wav_lengths"]["min"], room["wav_lengths"]["max"])]
    report = {
        "mode": "readback", "dataset_config": args.dataset_config,
        "dataset_root": root, "n_rooms": len(rooms),
        "split_check": split_check,
        "decode_all": decode_all,
        "decoded_files": sum(room["wav_checked"] for room in rooms.values()),
        "min_wav_samples": MIN_WAV_SAMPLES,
        "wav_length_rationale": (
            "every scoring path consumes a prefix (target pad_crop 10240, context 9600, oracle "
            "8000), so >= 10240 samples is score-identical and < 10240 is a failure"),
        "wav_lengths": ({"min": min(all_lengths), "max": max(all_lengths),
                         "mean": float(np.mean(all_lengths))} if all_lengths else None),
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

    if split_check["enforced"]:
        status = "PASS" if not split_check["failures"] else "FAIL"
        print(f"split digest [{status}]: file {split_check['file_sha256'][:12]}..., "
              f"{split_check['n_files']} identities, {split_check['n_rooms']} rooms, "
              f"room/node map {split_check['room_node_map_sha256'][:12]}...")
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

    NOT an interprocess lock: the existence check and the rename are separate
    steps, so two runs started simultaneously against the same target can both
    pass the check. It defends against the sequential mistake (rerunning a cell,
    or a failed run erasing a passing one), not against concurrent writers.
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


def aux_stem(args, kind, wavs=None):
    """Content-addressed stem for an auxiliary mode's report.

    Two runs of the same mode over different inputs must not collide, so the
    stem carries short hashes of the dataset config, the scorer checkpoint and
    the input rows (whichever the mode consumes) plus its protocol knobs.
    """
    parts = [str(args.eval_name), kind, f"ds-{_short(_file_sha256(args.dataset_config))}"]
    if kind == "scorer-noise":
        scorer = os.path.splitext(os.path.basename(str(args.agree_ckpt)))[0] or "none"
        parts += [f"scorer-{scorer}", f"sha-{_short(_file_sha256(args.agree_ckpt))}",
                  f"n{int(args.noise_draws)}", f"w{int(args.noise_wav_count)}",
                  f"seed{int(args.seed)}"]
        if wavs:
            # the SELECTED wav set, so two measurements over different RIRs cannot
            # share a report path even at identical knobs
            digest = hashlib.sha256(
                "\n".join(sorted(os.path.realpath(str(p)) for p in wavs)).encode("utf-8"))
            parts.append(f"wavs-{_short(digest.hexdigest())}")
    elif kind == "reaggregate":
        parts.append(f"rows-{_short(_rows_digest(getattr(args, 'rows', None)))}")
    return "_".join(parts)


def write_report(args, report, kind, wavs=None):
    """Write one auxiliary-mode JSON report, atomically and without clobbering."""
    path = os.path.join(str(args.out_dir), f"{aux_stem(args, kind, wavs=wavs)}.json")
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


def _split_wav_paths(args):
    """Every wav the configured split enumerates, as real paths."""
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
                paths.append(os.path.join(entry["path"], folder, scene, scene_id, fname))
    return paths


def resolve_noise_wavs(args):
    """Explicit ``--noise-wavs``, else the first files of the split.

    Explicit paths must belong to the CONFIGURED split's own enumeration (r4
    review M6): otherwise a seen dataset config plus an arbitrary path would have
    measured unseen RIRs while the report claimed a seen run.
    """
    enumerated = _split_wav_paths(args)
    if args.noise_wavs:
        allowed = {os.path.realpath(p) for p in enumerated}
        chosen = []
        for path in args.noise_wavs:
            if os.path.realpath(str(path)) not in allowed:
                _refuse(f"--noise-wavs {path!r} is not part of the configured split "
                        f"({args.dataset_config}); scorer noise is measured on that split only")
            chosen.append(str(path))
        return chosen
    present = [path for path in enumerated if os.path.isfile(path)]
    if not present:
        _refuse("no wav from the split is present to measure scorer noise on")
    return present[: int(args.noise_wav_count)]


def run_scorer_noise(args):
    """--mode scorer-noise: the §2.8.3 diagnostic. Loads AGREE only."""
    validate_dataset_split(args)
    paths = resolve_noise_wavs(args)
    agree = load_agree_audio(args.agree_ckpt, args.device)
    wavs = _load_noise_wavs(paths)

    mean_embeddings = embed_rirs(agree.model, wavs, args.device, readout="mean")
    # The stock readout samples from the GLOBAL stream, so seed it immediately
    # before drawing and record the seed: a measurement nobody can reproduce is
    # not a measurement (r4 review M6).
    torch.manual_seed(int(args.seed))
    draws = torch.stack([embed_rirs(agree.model, wavs, args.device, readout="sample")
                         for _ in range(int(args.noise_draws))])
    report = measure_scorer_noise(mean_embeddings, draws)
    report.update({
        "mode": "scorer-noise", "agree_ckpt": args.agree_ckpt,
        "agree_sha256": agree.ckpt_sha256, "wavs": paths, "device": args.device,
        "seed": int(args.seed),
        "source_sha": source_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    report_path = write_report(args, report, "scorer-noise", wavs=paths)
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


def load_published_rows(path):
    """``(by_query_id, sha256)`` for a completed run's rows file."""
    rows = read_rows(path)
    by_query = {}
    for row in rows:
        by_query[row["query_id"]] = row
    return by_query, sha256_file(path)


def verify_row_against_published(row, published):
    """Fail closed unless every logged per-sample similarity is reproduced EXACTLY.

    The noise bank is keyed by (seed, query_id, k), so a replay of a completed run
    must re-derive bit-identical waveforms and therefore bit-identical
    similarities; the comparison is on the exact float32 hex, not a tolerance.
    """
    reference = published.get(row["query_id"])
    if reference is None:
        raise SystemExit(
            f"verification ABORT: query {row['query_id']!r} is not in the published rows file; "
            "the replay is not scoring the same split")
    got, want = row["sims_hex"], reference["sims_hex"]
    if len(got) != len(want) or any(len(a) != len(b) for a, b in zip(got, want)):
        raise SystemExit(
            f"verification ABORT at {row['query_id']!r}: sims shape "
            f"{len(got)}x{len(got[0]) if got else 0} != published "
            f"{len(want)}x{len(want[0]) if want else 0}")
    for m, (row_got, row_want) in enumerate(zip(got, want)):
        for k, (a, b) in enumerate(zip(row_got, row_want)):
            if a != b:
                raise SystemExit(
                    f"verification ABORT at {row['query_id']!r}: similarity differs at "
                    f"m={m}, k={k} (replay {float.fromhex(a)!r} vs published "
                    f"{float.fromhex(b)!r})")
    return True


def build_waveform_manifest(args, rows, dump_dir, rows_stem):
    """Index of the waveform dump, carrying the geometry alongside the checksums.

    External waveform analyses (announcement 08's purpose) should need only this
    directory: per query the file, its sha256, the room, the GT node/position and
    the candidate nodes/positions in the SAME order as the dumped ``pred`` M axis.
    The rows JSONL stays the authority for the similarities.
    """
    waveforms = {}
    for row in rows:
        if not row.get("waveform_path"):
            continue
        waveforms[row["query_id"]] = {
            "path": row["waveform_path"], "sha256": row["waveform_sha256"],
            "room_id": row["room_id"], "gt_node": row["gt_node"],
            "gt_xyz_world": row["gt_xyz_world"],
            "candidate_nodes": row["candidate_nodes"],
            "candidate_xyz_world": row["candidate_xyz_world"],
            "pred_index": row["pred_index"], "n_candidates": row["n_candidates"],
            "n_samples": row["n_samples"],
        }
    return {
        "stem": artifact_stem(args), "rows_stem": rows_stem, "dump_dir": str(dump_dir),
        "n_queries": len(waveforms), "arrays": {"pred": "[M, K, samples] float32",
                                                "obs": "[samples] float32"},
        "registration_sha": getattr(args, "registration_sha_resolved", None)
        or args.registration_sha or "n/a",
        "source_sha": source_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "waveforms": waveforms,
    }


# --------------------------------------------------------------------------- #
# R4 metric families on the replay pass (plan_loc_invert_R4 §1)
# --------------------------------------------------------------------------- #
def rir_metrics_compute(pred, obs, ctx, config):
    """Indirection so tests can observe exactly what the metrics saw."""
    from src.localization.rir_metrics import compute_metrics
    return compute_metrics(pred, obs, ctx, config)


def rir_metrics_window(x):
    """The metric families' common analysis window (imported, never redefined)."""
    from src.localization.rir_metrics import common_window
    return common_window(x)


def metric_config_from_args(args):
    """The MetricConfig this run applies. Nothing is chosen here.

    ``delta_max`` comes from the CLI/registration manifest; on a SEEN calibration
    run (no registered value) the whole pre-listed grid is emitted so calibration
    can select from it -- selection never happens inside a scoring pass.
    """
    from src.localization.rir_metrics import M1_DELTA_GRID, MetricConfig

    registered = args.metric_delta_max is not None
    delta_max = int(args.metric_delta_max) if registered else M1_DELTA_GRID[0]
    grid = () if registered else tuple(M1_DELTA_GRID)
    return MetricConfig(delta_max=delta_max, delta_grid=grid,
                        t30_backend=args.metric_t30_backend,
                        m4_mu=getattr(args, "metric_m4_mu", None),
                        m4_sigma=getattr(args, "metric_m4_sigma", None),
                        secondaries=bool(args.metric_secondaries))


def _encode_vector(values):
    """Exact float32 hex for a 1-D distance vector (the sims codec, reused)."""
    return encode_sims(torch.as_tensor(values).reshape(1, -1))[0]


def build_metrics_row(row, position, metrics, config, num_context, sensitivities=None):
    """One metrics-JSONL record: every family's raw distances plus its readout."""
    from src.localization.rir_metrics import (aggregate_over_k, K_AGGREGATION_PRIMARY,
                                              K_AGGREGATION_SECONDARIES,
                                              predict_from_distances)

    families = {}
    for name, distances in metrics["candidates"].items():
        aggregations = {}
        for how in (K_AGGREGATION_PRIMARY,) + tuple(K_AGGREGATION_SECONDARIES):
            aggregations[how] = _encode_vector(aggregate_over_k(distances, how))
        primary = aggregate_over_k(distances, K_AGGREGATION_PRIMARY)
        pred_index = predict_from_distances(primary)
        families[name] = {
            "candidates_hex": encode_sims(distances),
            "context_hex": _encode_vector(metrics["context"][name]),
            "aggregations": aggregations,
            "pred_index": int(pred_index),
            "pred_node": int(row["candidate_nodes"][pred_index]),
            "correct": bool(pred_index == row["gt_index"]),
        }

    diagnostics = metrics["diagnostics"]
    m4_block = {}
    if "m4_features" in diagnostics:
        m4_block = {
            "features": diagnostics["m4_features"].tolist(),
            "obs_features": diagnostics["m4_obs_features"].tolist(),
            "context_features": diagnostics["m4_context_features"].tolist(),
            "mask": [bool(v) for v in diagnostics["m4_mask"]],
            "dropped": diagnostics["m4_dropped"],
        }
    return {
        "query_id": row["query_id"], "room_id": row["room_id"], "position": int(position),
        "n_candidates": int(row["n_candidates"]), "n_samples": int(row["n_samples"]),
        "n_context": int(num_context),
        "candidate_nodes": row["candidate_nodes"], "gt_index": int(row["gt_index"]),
        "gt_node": int(row["gt_node"]), "context_member": row["context_member"],
        "candidate_xyz_world": row["candidate_xyz_world"],
        "gt_xyz_world": row["gt_xyz_world"],
        "agree_pred_index": int(row["pred_index"]), "agree_e_loc": float(row["e_loc"]),
        "waveform_path": row.get("waveform_path"),
        "families": families,
        "m4": m4_block,
        "m5_lags": (diagnostics["m5_lags"].tolist() if "m5_lags" in diagnostics else None),
        "m5_gcc_lags": (diagnostics["m5_gcc_lags"].tolist()
                        if "m5_gcc_lags" in diagnostics else None),
        "m5_gcc_context_lags": (diagnostics["m5_gcc_context_lags"].tolist()
                                if "m5_gcc_context_lags" in diagnostics else None),
        "sensitivities": sensitivities,
        "m5_context_lags": (diagnostics["m5_context_lags"].tolist()
                            if "m5_context_lags" in diagnostics else None),
        "metric_config": metrics["config"],
        "waveform_source": "replay_snapshot",
        "tail_provenance": (
            "samples 8000-9600 are deterministic-replay data: they were regenerated by the "
            "same keyed noise bank, NOT independently verified against the original run "
            "(the published sims only constrain the first 8000 samples)"),
    }


def preflight_verify_against(args, expected_queries):
    """Refuse a replay that is not replaying the SAME protocol (r7 review LOW).

    Runs BEFORE any generation: the reference file must be complete and unique,
    its per-row protocol fields must match this CLI, and -- when the sibling
    summary is present -- so must the run-level provenance (seed, conditioning,
    steps, cfg-scale, checkpoint).
    """
    path = str(args.verify_against)
    rows = read_rows(path)
    if not rows:
        _refuse(f"--verify-against {path!r} contains no rows")
    identities = [row["query_id"] for row in rows]
    if len(set(identities)) != len(identities):
        _refuse(f"--verify-against {path!r} has duplicate query ids; it is not a run's rows file")
    if expected_queries is not None and len(rows) != int(expected_queries):
        _refuse(f"--verify-against {path!r} has {len(rows)} rows but this run scores "
                f"{expected_queries}; the replay would not cover the published run")

    expected_row = {"tau": (float(args.tau) if args.tau is not None else None),
                    "agg": args.agg, "n_samples": effective_num_samples(args),
                    "control": args.control, "score_source": args.score_source}
    for field, wanted in expected_row.items():
        values = {row.get(field) for row in rows}
        if len(values) != 1:
            _refuse(f"--verify-against {path!r} mixes {field!r} values {sorted(values)}")
        found = values.pop()
        if field == "tau" and found is not None and wanted is not None:
            match = float(found) == float(wanted)
        else:
            match = found == wanted
        if not match:
            _refuse(f"replay protocol mismatch: published {field}={found!r}, this run "
                    f"{field}={wanted!r}")

    summary_path = path.replace("_rows.jsonl", "_summary.json")
    provenance = None
    if summary_path != path and os.path.isfile(summary_path):
        with open(summary_path) as handle:
            provenance = json.load(handle).get("provenance")
    if provenance:
        for field, wanted in (("seed", int(args.seed)), ("cond_method", args.cond_method),
                              ("cond_autocast", args.cond_autocast), ("steps", int(args.steps)),
                              ("cfg_scale", float(args.cfg_scale)),
                              ("rotate_deg", float(args.rotate_deg))):
            found = provenance.get(field)
            if found is not None and found != wanted:
                _refuse(f"replay provenance mismatch on {field}: published {found!r}, this run "
                        f"{wanted!r}")
    return {"rows_path": path, "n_rows": len(rows), "rows_sha256": sha256_file(path),
            "provenance_checked": provenance is not None}


def metric_registerable_payload():
    """The frozen REGISTERABLE set of rir_metrics, for provenance/registration."""
    from src.localization.rir_metrics import registerable_payload
    return registerable_payload()


def score_query_metrics(args, row, position, snapshot, md, config, handle):
    """Score one query's five metric families from the SNAPSHOT and stream the row.

    The snapshot is the immutable copy the npz dump was written from, so the two
    artifacts cannot describe different waveforms; its digest is re-checked after
    the metrics have run, and a change aborts the run rather than publishing a
    dump and a metrics file that disagree (R4-COMPOSITION GUARD).
    """
    if snapshot is None:
        _refuse(f"query {row['query_id']!r} produced no waveform snapshot; --metrics cannot "
                "run without --dump-waveforms")
    pred = rir_metrics_window(snapshot["pred"])
    obs = rir_metrics_window(snapshot["obs"].reshape(1, -1))[0]
    context_audio = md.get("context_audio")
    if context_audio is None:
        _refuse(f"query {row['query_id']!r} carries no context_audio; the metric-matched "
                "retrieval control (plan R4 §2) could not be computed")
    ctx = rir_metrics_window(torch.as_tensor(context_audio).reshape(
        torch.as_tensor(context_audio).shape[0], -1))

    metrics = rir_metrics_compute(pred, obs, ctx, config)
    # The seen sensitivity battery on the declared deterministic subset: every
    # SENSITIVITY_STRIDE-th query of the stream (plan R4 §3, registered rule).
    sensitivities = None
    if getattr(args, "metric_sensitivities", False):
        from src.localization.rir_metrics import SENSITIVITY_STRIDE, sensitivity_variants
        if int(position) % int(SENSITIVITY_STRIDE) == 0:
            variants = sensitivity_variants(pred, obs, config)
            sensitivities = {name: {family: encode_sims(values)
                                    for family, values in block.items()}
                             for name, block in variants.items()}
    if snapshot_digest(snapshot["pred"], snapshot["obs"]) != snapshot["sha256"]:
        _refuse(f"query {row['query_id']!r}: the waveform snapshot CHANGED while it was being "
                "consumed; the dump and the metrics would describe different waveforms")

    metrics_row = build_metrics_row(row, position, metrics, config, ctx.shape[0],
                                    sensitivities=sensitivities)
    write_row(handle, metrics_row)
    return metrics_row


def _metric_mode(args):
    """Does this invocation compute metrics at all (any mode)?"""
    return bool(getattr(args, "metrics", False)) or getattr(args, "mode", "run") in (
        "metrics-retrieval",)


def assert_metric_registration(args, dataset_config):
    """Any UNSEEN metric work requires a committed metric manifest.

    Every mode, not just --metrics: metrics-retrieval scores the same families on
    the same held-out data (r4m review finding 1). Seen calibration runs
    deliberately have no manifest -- that is where the constants are chosen -- but
    they still record the full REGISTERABLE payload in provenance.
    """
    if not _metric_mode(args):
        return False
    if bool((dataset_config or {}).get("unseeneval", False)):
        if not args.metric_registration:
            _refuse("--metric-registration is required to run --metrics on the unseen split: "
                    "no metric constant may be chosen, or changed, on held-out data")
        if not args.registration_sha:
            _refuse("--registration-sha is required with --metric-registration")
        return True
    return False


def metric_config_from_manifest(manifest):
    """Build the MetricConfig FROM the verified manifest -- no CLI reconstruction.

    The frozen z-normalization has no CLI representation at all, so a registered
    run that rebuilt its config from flags could never apply it (r4m review
    finding 1). The manifest is the authority; the CLI only points at it.
    """
    from src.localization.rir_metrics import MetricConfig

    config = manifest.get("metric_config")
    if not isinstance(config, dict):
        _refuse("the metric manifest does not carry a 'metric_config' block")
    required = ("delta_max", "t30_backend", "m4_mu", "m4_sigma", "window_samples",
                "param_window_samples", "lam", "sample_rate", "families", "secondaries",
                "delta_grid")
    for key in required:
        if key not in config:
            _refuse(f"the metric manifest does not lock metric_config.{key}")
    if config["delta_grid"]:
        _refuse("a registered metric manifest must fix ONE delta_max, not a calibration grid")
    for key in ("m4_mu", "m4_sigma"):
        if config[key] is None:
            _refuse(f"the metric manifest leaves metric_config.{key} unset; a registered run "
                    "must apply the frozen z-normalization")
    return MetricConfig(delta_max=int(config["delta_max"]),
                        window_samples=int(config["window_samples"]),
                        param_window_samples=int(config["param_window_samples"]),
                        lam=float(config["lam"]), t30_backend=config["t30_backend"],
                        sample_rate=int(config["sample_rate"]),
                        m4_mu=config["m4_mu"], m4_sigma=config["m4_sigma"],
                        families=tuple(config["families"]),
                        secondaries=bool(config["secondaries"]), delta_grid=())


#: the source of the registered metric DEFINITIONS. The `registerable` block
#: locks every constant, but not a formula, so the module itself is pinned: the
#: blob at the manifest's own source_sha, the blob at the registration commit and
#: the file this process imports must all be one file (r4m4 finding 1). Only the
#: definitions are pinned -- the driver around them keeps being fixed between
#: registration and the unseen passes, which is why it is not on this list.
METRIC_SOURCE_FILES = ("src/localization/rir_metrics.py",)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _blob_at(repo_root, sha, relpath, label):
    """The bytes of ``relpath`` at commit ``sha``, or a refusal."""
    show = subprocess.run(["git", "show", f"{sha}:{relpath}"], cwd=repo_root,
                          capture_output=True)
    if show.returncode != 0:
        _refuse(f"commit {sha} does not contain {relpath!r} ({label}); a digest can only be "
                "verified against a path the commit actually carries")
    return show.stdout


def verify_registered_sources(manifest, resolved_sha, repo_root):
    """Bind the manifest to the metric CODE it was calibrated with.

    ``source_sha`` is what the calibration pass recorded as its HEAD. It must be
    an immutable full object id, resolve to a commit, and carry byte-identical
    metric sources to both the registration commit and the working tree that is
    about to execute -- otherwise the frozen constants describe formulas that no
    longer exist.
    """
    sha = manifest.get("source_sha")
    if not isinstance(sha, str) or not _FULL_OBJECT_ID.match(sha):
        _refuse(f"the metric manifest records source_sha={sha!r}; a registered manifest must "
                "record the full 40- or 64-hex commit its calibration pass ran on")
    if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo_root,
                      capture_output=True).returncode != 0:
        _refuse(f"the metric manifest's source_sha {sha} does not resolve to a commit in "
                f"{repo_root}")
    digests = {}
    for relpath in METRIC_SOURCE_FILES:
        at_source = _blob_at(repo_root, sha, relpath, "metric source at source_sha")
        at_registration = _blob_at(repo_root, resolved_sha, relpath,
                                   "metric source at the registration commit")
        if at_source != at_registration:
            _refuse(f"{relpath} differs between the manifest's source_sha {sha[:12]}... and the "
                    f"registration commit {resolved_sha[:12]}...; the registered constants were "
                    "calibrated with different code than they were registered with")
        local_path = os.path.join(repo_root, relpath)
        if not os.path.isfile(local_path):
            _refuse(f"{relpath} is missing from {repo_root}; the registered metric source is "
                    "not the source this run would execute")
        with open(local_path, "rb") as handle:
            local = handle.read()
        if local != at_registration:
            _refuse(f"{relpath} in the worktree differs from the version committed at the "
                    f"registration commit {resolved_sha[:12]}...; the metric definitions were "
                    "edited after registration")
        digests[relpath] = hashlib.sha256(local).hexdigest()
    return {"source_sha": sha, "sources": digests}


def verify_registered_r2_manifests(manifest, resolved_sha, repo_root, registered):
    """Byte-check the R2/R2b candidate manifests the registration pins.

    The digests are only evidence if they are checked against the manifests as
    COMMITTED at the registration sha, so each key must be a repository-relative
    path that commit contains. A registered run may not leave the block empty.
    """
    digests = manifest.get("r2_manifest_digests")
    if digests in (None, {}) or not isinstance(digests, dict):
        if registered:
            _refuse("the metric manifest locks no r2_manifest_digests; a registered run must "
                    "pin the R2/R2b candidate manifests it is scored against")
        return {}
    verified = {}
    for relpath, digest in sorted(digests.items()):
        if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
            _refuse(f"r2_manifest_digests[{relpath!r}] is {digest!r}, not a sha256")
        actual = hashlib.sha256(
            _blob_at(repo_root, resolved_sha, str(relpath), "R2 manifest")).hexdigest()
        if actual != digest:
            _refuse(f"r2_manifest_digests[{relpath!r}] is {digest[:16]}... but the copy "
                    f"committed at {resolved_sha[:12]}... hashes to {actual[:16]}...")
        verified[str(relpath)] = actual
    return verified


def verify_metric_registration(args, dataset_config, repo_root=None,
                               candidate_manifest_sha256=None, identity_digest=None):
    """Machine-check the metric manifest and RETURN the config it registers.

    Commit rules are the R2 gate's own machinery (full-hex id, in-worktree,
    byte-identical, ancestor of HEAD). What is locked: rir_metrics' whole
    REGISTERABLE set, the entire MetricConfig, the registered seeds, the candidate
    manifest and the R2 identity stream this run must be scoring.
    """
    registered = assert_metric_registration(args, dataset_config)
    if not args.metric_registration:
        return None
    verified = verify_registration_commit(args.metric_registration, args.registration_sha,
                                          repo_root=repo_root)
    manifest = verified["manifest"]

    locked = manifest.get("registerable")
    if not isinstance(locked, dict):
        _refuse("the metric manifest does not lock a 'registerable' block")
    current = metric_registerable_payload()
    for key, value in sorted(current.items()):
        if key not in locked:
            _refuse(f"the metric manifest does not lock {key!r}")
        if locked[key] != value:
            _refuse(f"registered {key} is {locked[key]!r} but rir_metrics now defines {value!r}")
    for key in sorted(set(locked) - set(current)):
        _refuse(f"the metric manifest locks {key!r}, which rir_metrics no longer defines")

    config = metric_config_from_manifest(manifest)
    # the manifest is only evidence about the code and the candidate sets it was
    # produced from if those are checked too (r4m4 finding 1)
    repo = os.path.realpath(str(repo_root or _repo_root()))
    sources = verify_registered_sources(manifest, verified["resolved_sha"], repo)
    r2_manifests = verify_registered_r2_manifests(manifest, verified["resolved_sha"], repo,
                                                  registered)

    seeds = manifest.get("seeds")
    if registered:
        if not isinstance(seeds, (list, tuple)) or not seeds:
            _refuse("the metric manifest does not lock a non-empty 'seeds' list")
        if int(args.seed) not in [int(s) for s in seeds]:
            _refuse(f"--seed {args.seed} is not one of the registered metric seeds {list(seeds)}")
        for field, actual, label in (("candidate_manifest_sha256", candidate_manifest_sha256,
                                      "candidate manifest"),
                                     ("r2_identity_digest", identity_digest, "identity stream")):
            locked_value = manifest.get(field)
            if locked_value in (None, "tbd"):
                _refuse(f"the metric manifest leaves {field} unset; a registered run must lock "
                        f"the {label}")
            if actual is not None and locked_value != actual:
                _refuse(f"registered {label} is {locked_value!r} but this run resolves "
                        f"{actual!r}")
    args.metric_registration_sha_resolved = verified["resolved_sha"]
    args.metric_registration_bindings = {"source": sources, "r2_manifests": r2_manifests}
    return config


def _load_identity_stream(path):
    """A committed identity stream: a rows JSONL or a JSON list of query ids."""
    path = str(path)
    if path.endswith(".jsonl"):
        return [row["query_id"] for row in read_rows(path)]
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("identities")
    if not isinstance(payload, list):
        _refuse(f"--calibration-identities {path!r} is neither a rows JSONL nor a JSON list")
    return [str(item) for item in payload]


def _top1_for_delta(rows, family_prefix, delta, base_delta):
    """Dev top-1 of one family at one grid value, over the seen metrics rows."""
    from src.localization.rir_metrics import (K_AGGREGATION_PRIMARY, aggregate_over_k,
                                              predict_from_distances)
    name = family_prefix if int(delta) == int(base_delta) else f"{family_prefix}_delta{int(delta)}"
    hits, total = 0, 0
    for row in rows:
        block = row["families"].get(name)
        if block is None:
            return None
        distances = decode_sims(block["candidates_hex"])
        scores = aggregate_over_k(distances, K_AGGREGATION_PRIMARY)
        hits += int(predict_from_distances(scores) == int(row["gt_index"]))
        total += 1
    return {"delta_max": int(delta), "top1": hits / total if total else float("nan"),
            "n_queries": total}


def summarize_sensitivity(rows):
    """What the sensitivity battery in THESE rows actually covers.

    The report used to declare a battery it never inspected (r4m4 finding 4). It
    now describes only what the supplied rows carry, under the registered variant
    names -- and refuses a battery that is present but does not cover every
    declared variant with every declared family, because a partial battery
    reported as a battery is a false claim about what was tested.
    """
    from src.localization.rir_metrics import (SENSITIVITY_STRIDE, SENSITIVITY_VARIANTS)

    declared_variants = list(SENSITIVITY_VARIANTS)
    carriers = [row for row in rows if row.get("sensitivities")]
    summary = {"declared_variants": declared_variants,
               "declared_families": None, "stride": int(SENSITIVITY_STRIDE),
               "n_rows": len(rows), "n_rows_with_battery": len(carriers),
               "positions": [row.get("position") for row in carriers],
               "per_variant": {},
               "status": "not computed: the supplied metrics rows carry no battery"}
    if not carriers:
        return summary

    families = sorted({family for row in carriers
                       for block in row["sensitivities"].values() for family in block})
    summary["declared_families"] = families
    for row in carriers:
        battery = row["sensitivities"]
        missing = [name for name in declared_variants if name not in battery]
        if missing:
            _refuse(f"query {row.get('query_id')!r} carries a sensitivity battery without "
                    f"{missing}; the registered battery is {declared_variants}")
        for name in declared_variants:
            absent = [family for family in families if family not in battery[name]]
            if absent:
                _refuse(f"query {row.get('query_id')!r}: sensitivity variant {name!r} covers "
                        f"{sorted(battery[name])} but the battery reports {families}; a "
                        "partial battery may not be reported as one")
    summary["per_variant"] = {
        name: {"n_queries": len(carriers), "families": families} for name in declared_variants}
    summary["status"] = "computed"
    return summary


def run_metrics_calibrate(args):
    """--mode metrics-calibrate: choose delta_max and freeze mu/sigma. Seen only.

    Consumes a SEEN metrics-JSONL (the replay of R1's prefix) and never touches
    data: the registered delta_max is the pre-listed grid value with the best dev
    top-1 (ties to the smallest, as registered), and M4's z-normalization
    statistics are the seen mean/std of each feature. Emits the draft metric
    manifest, the per-feature discrimination diagnostics and the sensitivity
    battery. Deterministic; no unseen access.
    """
    import numpy as np
    from src.localization.rir_metrics import (M1_DELTA_GRID, M4_FEATURES,
                                              registerable_payload)

    dataset_config = load_dataset_config(args)
    if bool(dataset_config.get("unseeneval", False)):
        _refuse("--mode metrics-calibrate runs on the seen split only: every R4 constant is "
                "chosen from seen data, and the freeze must predate any unseen pass")

    rows = read_rows(args.metrics_rows)
    if not rows:
        _refuse(f"--metrics-rows {args.metrics_rows!r} contains no rows")

    # Authenticate the calibration stream itself -- FAIL-CLOSED (r4m4 finding 1):
    # every registered constant below is chosen from these rows, so a pass that
    # cannot name the queries it calibrated on may not produce a draft at all.
    identities = [row["query_id"] for row in rows]
    if len(set(identities)) != len(identities):
        _refuse(f"--metrics-rows {args.metrics_rows!r} contains duplicate query ids")
    if not getattr(args, "calibration_identities", None):
        _refuse("--calibration-identities is required by --mode metrics-calibrate: the "
                "registered delta_max and the frozen m4 mu/sigma are chosen from these rows, "
                "and an unauthenticated stream cannot say which queries they came from")
    identity_check = {"n_identities": len(identities), "authenticated": False,
                      "source": str(args.calibration_identities),
                      "source_sha256": _file_sha256(args.calibration_identities)}
    expected_stream = _load_identity_stream(args.calibration_identities)
    if len(expected_stream) != len(identities):
        _refuse(f"the calibration rows cover {len(identities)} queries but the committed "
                f"stream declares {len(expected_stream)}")
    for position, (found, wanted) in enumerate(zip(identities, expected_stream)):
        if found != wanted:
            _refuse(f"calibration identity mismatch at position {position}: expected "
                    f"{wanted!r}, got {found!r}")
    identity_check["authenticated"] = True

    grid = []
    base_delta = int((rows[0].get("metric_config") or {}).get("delta_max", M1_DELTA_GRID[0]))
    for delta in M1_DELTA_GRID:
        entry = _top1_for_delta(rows, "m1", delta, base_delta)
        if entry is None:
            _refuse(f"the metrics rows do not carry M1 at delta_max={delta}; a calibration pass "
                    "must emit the whole pre-listed grid")
        grid.append(entry)
    best = max(entry["top1"] for entry in grid)
    selected = min(entry["delta_max"] for entry in grid if entry["top1"] == best)

    features, dropped_total = [], 0
    for row in rows:
        block = row.get("m4") or {}
        if block.get("features"):
            features.append(np.asarray(block["features"], dtype=np.float64).reshape(
                -1, len(M4_FEATURES)))
        dropped_total += int((block.get("dropped") or {}).get("n_dropped", 0))
    if features:
        stacked = np.concatenate(features)
        finite = np.where(np.isfinite(stacked), stacked, np.nan)
        mu = np.nanmean(finite, axis=0)
        sigma = np.nanstd(finite, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
    else:
        stacked = np.zeros((0, len(M4_FEATURES)))
        mu, sigma = np.zeros(len(M4_FEATURES)), np.ones(len(M4_FEATURES))

    # Per-feature discrimination (plan R4 §1): does this feature separate
    # CANDIDATES more than it separates the K samples of one candidate, and how
    # well does it localize on its own? Reporting only -- it selects nothing.
    per_feature = []
    for index, name in enumerate(M4_FEATURES):
        between, within, hits, total = [], [], 0, 0
        for row in rows:
            block = row.get("m4") or {}
            if not block.get("features") or not block.get("obs_features"):
                continue
            feats = np.asarray(block["features"], dtype=np.float64)[..., index]   # [M, K]
            if not np.isfinite(feats).all():
                continue
            per_candidate = feats.mean(axis=-1)
            between.append(float(np.var(per_candidate)))
            within.append(float(np.mean(np.var(feats, axis=-1))))
            obs_value = float(np.asarray(block["obs_features"], dtype=np.float64)[index])
            if np.isfinite(obs_value):
                distances = np.abs(per_candidate - obs_value)
                hits += int(int(np.argmin(distances)) == int(row["gt_index"]))
                total += 1
        per_feature.append({
            "feature": name,
            "between_var": float(np.mean(between)) if between else float("nan"),
            "within_var": float(np.mean(within)) if within else float("nan"),
            "power": (float(np.mean(between) / np.mean(within))
                      if between and np.mean(within) > 0 else float("nan")),
            "top1": (hits / total) if total else float("nan"),
            "n_queries": total,
        })

    seeds = [int(s) for s in str(getattr(args, "register_seeds", None) or "42 43 44").split()]
    # Repository-RELATIVE keys: the unseen pass verifies each digest against the
    # copy committed at the registration sha, and a bare basename names nothing a
    # commit can be asked for (r4m4 finding 1).
    r2_digests = {}
    repo = os.path.realpath(_repo_root())
    for path in (getattr(args, "register_r2_manifest", None) or []):
        real = os.path.realpath(str(path))
        if not os.path.isfile(real):
            _refuse(f"--register-r2-manifest {path!r} does not exist")
        if os.path.commonpath([real, repo]) != repo:
            _refuse(f"--register-r2-manifest {path!r} is outside the repository {repo!r}; the "
                    "registration can only pin a manifest that gets committed with it")
        r2_digests[os.path.relpath(real, repo)] = _file_sha256(real)
    draft = {
        "registerable": registerable_payload(),
        "metric_config": {"delta_max": int(selected), "t30_backend": args.metric_t30_backend,
                          "m4_mu": [float(v) for v in mu],
                          "m4_sigma": [float(v) for v in sigma],
                          "window_samples": rm_window_samples(),
                          "param_window_samples": rm_param_window_samples(),
                          "lam": float(registerable_payload()["m2_lambda"]),
                          "sample_rate": int(registerable_payload()["sample_rate"]),
                          "families": ["m1", "m2", "m3", "m4", "m5"],
                          "secondaries": True, "delta_grid": []},
        # the unseen passes must be pinned to these too (r4m review finding 1)
        "seeds": seeds,
        "candidate_manifest_sha256": getattr(args, "register_candidate_manifest", None) or "tbd",
        "r2_identity_digest": getattr(args, "register_identity_digest", None) or "tbd",
        "r2_manifest_digests": r2_digests,
        "calibration_identities": getattr(args, "calibration_identities", None) or "n/a",
        "metrics_rows": str(args.metrics_rows),
        "metrics_rows_sha256": sha256_file(args.metrics_rows),
        "source_sha": source_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = {
        "mode": "metrics-calibrate", "n_rows": len(rows),
        "metrics_rows": str(args.metrics_rows),
        "delta_max": {"grid": grid, "selected": int(selected), "objective": "dev_top1",
                      "tie_break": "smallest"},
        "m4": {"mu": [float(v) for v in mu], "sigma": [float(v) for v in sigma],
               "n_observations": int(stacked.shape[0]),
               "dropped_features_total": int(dropped_total)},
        "identity_check": identity_check,
        "diagnostics": {"per_feature": per_feature},
        "sensitivity": summarize_sensitivity(rows),
        "draft_manifest": draft,
        "source_sha": source_sha(),
        "created_utc": draft["created_utc"],
    }
    report_path = write_report(args, report, "metrics-calibrate")
    draft_path = os.path.join(str(args.out_dir), f"{args.eval_name}_metric_registration.json")
    write_json_atomic(draft_path, draft, overwrite=bool(getattr(args, "overwrite", False)))
    report["report_path"] = report_path
    report["draft_manifest_path"] = draft_path

    print(f"metrics calibration over {len(rows)} seen queries -> {report_path}")
    for entry in grid:
        marker = " <- registered" if entry["delta_max"] == selected else ""
        print(f"  delta_max={entry['delta_max']:>4}: dev top-1 {entry['top1']:.4f}{marker}")
    print(f"draft metric manifest -> {draft_path}")
    return report


def run_fa_parity_check(args):
    """--fa-parity-check: the exp_20 FA gate, with the run's own arguments.

    The record is written before the verdict is acted on, so a failure leaves
    the evidence behind rather than only a non-zero exit.
    """
    from src.localization import crossarm

    if getattr(args, "cond_method", "vanilla") != "fa_invariant":
        _refuse("--fa-parity-check needs --cond-method fa_invariant: there is no orbit to "
                "compare on a vanilla run")
    record = crossarm.run_fa_parity(args.ckpt_path, args.model_config,
                                    dataset_config=args.dataset_config, device=args.device,
                                    args=args)
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    record_path = os.path.join(out_dir, f"{args.eval_name}_fa_parity.json")
    write_json_atomic(record_path, jsonable(record), overwrite=True)
    print(f"fa-parity record -> {record_path}")
    if not record.get("passed"):
        failed = {mode: result.get("reasons") for mode, result in record["results"].items()
                  if not result.get("match")}
        _refuse(f"fa-parity gate FAILED: {failed}; the record is at {record_path}")
    print("fa-parity gate PASSED "
          + ", ".join(f"{mode}: max_abs_diff={result.get('max_abs_diff')}"
                      for mode, result in record["results"].items()))
    return {"record_path": record_path, "record": record}


def run_metrics_report(args):
    """--mode metrics-report: the reviewed offline aggregation of R4's streams.

    Pure re-reading of published artifacts -- no dataset, no checkpoint, no GPU
    and no re-scoring: every distance was computed by the registered passes and
    is decoded bit-exactly. The module does the work; this is the CLI around it.
    """
    from src.localization import metrics_report as mreport

    families = tuple(args.report_families) if args.report_families else mreport.REPORT_FAMILIES
    n_boot = int(args.report_bootstrap or mreport.BOOTSTRAP_N)

    # ---- the frozen experiment, not the operator's file list (r4m6 F3) ------
    verified = verify_registration_commit(args.report_registration, args.registration_sha)
    manifest = verified["manifest"]
    registered_seeds = sorted(int(s) for s in (manifest.get("seeds") or []))
    registered_families = tuple((manifest.get("metric_config") or {}).get("families") or ())
    if not registered_seeds or not registered_families:
        _refuse(f"{args.report_registration!r} registers no seeds/families to bind the report to")
    expect_queries = args.report_expect_queries
    if expect_queries is None:
        expect_queries = len(expected_split_identities_from_config(load_dataset_config(args)))
    oracle_by_seed = {}
    for spec in (args.oracle_inputs or []):
        seed, oracle_path = str(spec).split(":")
        if not os.path.isfile(oracle_path):
            _refuse(f"--oracle-inputs names a file that does not exist: {oracle_path}")
        oracle_by_seed[int(seed)] = oracle_path

    seed_reports = []
    for spec in args.report_input:
        seed, metrics_path, rows_path = str(spec).split(":")
        for path in (metrics_path, rows_path):
            if not os.path.isfile(path):
                _refuse(f"--report-input names a file that does not exist: {path}")
        print(f"[metrics-report] scanning seed {seed}: {os.path.basename(metrics_path)}")
        try:
            scan = mreport.scan_seed(metrics_path, rows_path, families=families,
                                     seed=int(seed),
                                     oracle_path=oracle_by_seed.get(int(seed)),
                                     expect_queries=int(expect_queries))
        except ValueError as error:                # library refusals are run refusals
            _refuse(f"seed {seed}: {error}")
        missing = [f for f in registered_families if f not in scan["families_present"]]
        if missing:
            _refuse(f"seed {seed} is missing the registered families {missing}; the report "
                    "covers every registered primary or refuses")
        if oracle_by_seed and int(seed) not in oracle_by_seed:
            _refuse(f"--oracle-inputs covers {sorted(oracle_by_seed)} but not seed {seed}")
        seed_reports.append(mreport.build_seed_report(scan, seed=int(seed), n_boot=n_boot))
        print(f"[metrics-report] seed {seed}: {scan['n_queries']} queries, "
              f"{scan['n_rooms']} rooms, families {scan['families_present']}")

    seeds = sorted(r["seed"] for r in seed_reports)
    if seeds != registered_seeds:
        _refuse(f"the report covers seeds {seeds} but {args.report_registration!r} registers "
                f"{registered_seeds}; the R4 answer is the registered seeds or nothing")
    if sorted(oracle_by_seed) not in ([], registered_seeds):
        _refuse(f"--oracle-inputs covers seeds {sorted(oracle_by_seed)}, not the registered "
                f"{registered_seeds}")
    expected_tests = 2 * len(registered_families)
    for report_block in seed_reports:
        found = len(report_block["primary_comparisons"])
        if found != expected_tests:
            _refuse(f"seed {report_block['seed']} produced {found} primary comparisons; the "
                    f"registered plan has {expected_tests} (5 families x 2 references)")

    seen_scan = None
    if args.report_seen:
        seen_metrics, seen_rows = str(args.report_seen).split(":")
        print("[metrics-report] scanning the seen calibration pass")
        seen_scan = mreport.scan_seed(seen_metrics, seen_rows, families=families)

    report = mreport.build_report(seed_reports, families=families, seen_scan=seen_scan,
                                  n_boot=n_boot,
                                  hash_inputs=bool(args.report_hash_inputs),
                                  extra_provenance={"registration": {
                                      "manifest": str(args.report_registration),
                                      "sha": verified["resolved_sha"],
                                      "seeds": registered_seeds,
                                      "families": list(registered_families),
                                      "queries_per_seed": int(expect_queries)}})
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"{args.eval_name}_metrics_report.json")
    markdown_path = os.path.join(out_dir, f"{args.eval_name}_metrics_report.md")
    write_json_atomic(report_path, jsonable(report), overwrite=True)
    with open(markdown_path + ".partial", "w") as handle:
        handle.write(mreport.render_markdown(report))
    os.replace(markdown_path + ".partial", markdown_path)
    print(f"metrics-report -> {report_path}")
    print(f"markdown block -> {markdown_path}")
    return {"report_path": report_path, "markdown_path": markdown_path, "report": report}


def run_metrics_retrieval(args, loader, engine, context, expected=None, dataset_config=None):
    """--mode metrics-retrieval: the §3 controls that need no generation.

    Per query, under EVERY metric family: the context RIRs scored against the
    observation (the metric-matched retrieval control of §2, raw and
    eligible-masked), the measured-candidate oracle ceiling where the files exist,
    and the context / non-context split. Nothing is generated and no checkpoint is
    loaded, so this runs whenever the dataset is present.
    """
    from src.localization.rir_metrics import metric_matched_retrieval

    dataset_config = dataset_config or load_dataset_config(args)
    # On unseen data this control is only a control if it is bound to the replay
    # it will be compared against (r4m4 finding 5): a differently-drawn context
    # set makes the comparison meaningless, so the pairing digest is mandatory
    # and is required BEFORE anything is computed or written.
    unseen = bool(dataset_config.get("unseeneval", False))
    paired_digest = getattr(args, "verify_context_digest", None)
    if unseen and not paired_digest:
        _refuse("--verify-context-digest is required for an unseen --mode metrics-retrieval "
                "pass: the control must be pinned to the context stream of the replay it is "
                "compared against, or an unbound artifact would be published as one")
    if expected is None:
        expected = expected_split_identities_from_config(dataset_config)
    expected = list(expected)
    if args.smoke and args.max_queries is not None:
        expected = expected[: int(args.max_queries)]

    config = getattr(args, "registered_metric_config", None) or metric_config_from_args(args)
    stem = f"{args.eval_name}_metrics_retrieval_seed{int(args.seed)}_ds-{_short(_file_sha256(args.dataset_config))}"
    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, f"{stem}_rows.jsonl")
    summary_path = os.path.join(out_dir, f"{stem}_summary.json")
    overwrite = bool(getattr(args, "overwrite", False))
    if not overwrite:
        for candidate in (rows_path, rows_path + ".partial", summary_path):
            if os.path.exists(candidate):
                _refuse(f"metrics-retrieval target already exists: {candidate}; pass --overwrite")

    manifest = context.get("manifest")
    families = tuple(config.families)
    rows, meta, scored = [], [], []
    with open(rows_path + ".partial", "w") as handle:
        for position, (obs_wav, md) in enumerate(itertools.islice(_iter_items(loader),
                                                                  len(expected))):
            identity = sample_target_id(md)
            if identity != expected[position]:
                raise SystemExit(f"identity gate ABORT at position {position}: expected "
                                 f"{expected[position]!r}, got {identity!r}")
            room_id = room_id_from_relpath(md["relpath"])
            gt_node, receiver_node = parse_ir_filename(md["path"])
            cand_set = candidate_set_from_manifest(manifest, room_id, gt_node, receiver_node)
            room_entry = manifest["rooms"][room_id]
            candidate_cams = candidate_camera_positions(cand_set)
            context_mask = context_membership_mask(candidate_cams, sample_context_ids(md),
                                                   gt_index=cand_set.gt_index)

            obs = rir_metrics_window(torch.as_tensor(obs_wav).reshape(1, -1))[0]
            ctx_audio = torch.as_tensor(md["context_audio"])
            ctx = rir_metrics_window(ctx_audio.reshape(ctx_audio.shape[0], -1))
            ctx_poses = torch.as_tensor(md["context_poses"]).float().numpy()

            # the measured candidate RIRs, where they exist: the oracle ceiling
            oracle_wavs, available, oracle_source_nodes = load_measured_rirs(
                os.path.dirname(md["path"]), cand_set, receiver_node,
                merge_map=room_entry.get("merge_map"))
            oracle = rir_metrics_window(oracle_wavs.reshape(oracle_wavs.shape[0], -1))
            metrics = rir_metrics_compute(oracle.unsqueeze(1), obs, ctx, config)

            eligible = [not member for member in context_mask]
            row_families = {}
            for family in families:
                ctx_distances = metrics["context"][family]
                raw = metric_matched_retrieval(candidate_cams, ctx_poses, ctx_distances)
                masked = metric_matched_retrieval(candidate_cams, ctx_poses, ctx_distances,
                                                  eligible_mask=eligible)
                oracle_distances = metrics["candidates"][family].reshape(-1)
                oracle_pred = oracle_prediction(oracle_distances, available)
                row_families[family] = {
                    "context_hex": _encode_vector(ctx_distances),
                    "oracle_hex": _encode_vector(oracle_distances),
                    "retrieval_pred_index": int(raw),
                    "retrieval_masked_pred_index": int(masked),
                    "retrieval_correct": bool(raw == cand_set.gt_index),
                    "retrieval_masked_correct": bool(masked == cand_set.gt_index),
                    "oracle_pred_index": int(oracle_pred),
                    "oracle_correct": bool(oracle_pred == cand_set.gt_index),
                }
            row = {"query_id": identity, "room_id": room_id, "position": position,
                   "n_candidates": len(cand_set.nodes), "n_context": int(ctx.shape[0]),
                   "candidate_nodes": [int(n) for n in cand_set.nodes],
                   "gt_index": int(cand_set.gt_index), "gt_node": int(cand_set.gt_node),
                   "context_member": context_mask,
                   "candidate_available": [bool(a) for a in available],
                   "oracle_source_nodes": [None if n is None else int(n)
                                           for n in oracle_source_nodes],
                   "context_fingerprints": sample_context_ids(md),
                   "families": row_families, "metric_config": config.payload()}
            write_row(handle, row)
            rows.append(row)
            meta.append({"context_member": context_mask})
            scored.append(identity)

    # Nothing is published until EVERY gate passes (r4m review finding 6).
    split_hash_value = assert_scored_stream(scored, expected)
    # the ordered context draw of THIS pass, in eval_FLAC's canonical serialization
    context_digest = canonical_stream_hash([tuple(row["context_fingerprints"])
                                            for row in rows]) if rows else "n/a"
    if paired_digest:
        if context_digest != paired_digest:
            raise SystemExit(
                f"context stream digest mismatch: this pass drew {context_digest[:16]}... but "
                f"the paired replay recorded {str(paired_digest)[:16]}...; the "
                "control would not be matched to the run it is compared against")
    # every artifact says whether it is bound, so an unbound seen pass can never
    # be read later as if it had been paired (r4m4 finding 5)
    context_binding = {
        "required": bool(unseen), "verified": bool(paired_digest),
        "expected": str(paired_digest) if paired_digest else None,
        "digest": context_digest,
        "status": ("bound to the paired replay" if paired_digest
                   else "unbound: seen split, no paired replay digest was supplied")}
    summary_families = {}
    for family in families:
        correct = [row["families"][family]["retrieval_correct"] for row in rows]
        masked = [row["families"][family]["retrieval_masked_correct"] for row in rows]
        oracle = [row["families"][family]["oracle_correct"] for row in rows]
        context_hits = [row["context_member"][row["families"][family]["retrieval_pred_index"]]
                        for row in rows]
        summary_families[family] = {
            "retrieval_top1": float(np.mean(correct)),
            "retrieval_masked_top1": float(np.mean(masked)),
            "oracle_top1": float(np.mean(oracle)),
            "context_member_rate": float(np.mean(context_hits)),
            "split": retrieval_split(rows, family),
        }
    summary = {"mode": "metrics-retrieval", "n_queries": len(rows),
               "n_rooms": len({row["room_id"] for row in rows}),
               "families": summary_families, "split_hash": split_hash_value,
               "context_stream_digest": context_digest,
               "context_binding": context_binding,
               "context_coverage": context_coverage(rows),
               "seed": int(args.seed), "metric_config": config.payload()}
    provenance = {"mode": "metrics-retrieval", "source_sha": source_sha(),
                  "dataset_config": args.dataset_config,
                  "dataset_config_sha256": _file_sha256(args.dataset_config),
                  "split_hash": split_hash_value,
                  "candidate_manifest_sha256": manifest_sha256(manifest) if manifest else "n/a",
                  "metric_registerable": metric_registerable_payload(),
                  "metric_registration": getattr(args, "metric_registration", None) or "n/a",
                  "context_binding": context_binding,
                  "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json_atomic(summary_path, {"provenance": provenance, "summary": jsonable(summary)},
                      overwrite=True)
    os.replace(rows_path + ".partial", rows_path)      # publish the set, gates first
    print(f"metrics-retrieval: {len(rows)} queries -> {rows_path}")
    return {"rows_path": rows_path, "summary_path": summary_path, "rows": rows,
            "rows_meta": meta, "summary": summary, "provenance": provenance}


def rm_window_samples():
    from src.localization.rir_metrics import WINDOW_SAMPLES
    return int(WINDOW_SAMPLES)


def rm_param_window_samples():
    from src.localization.rir_metrics import PARAM_WINDOW_SAMPLES
    return int(PARAM_WINDOW_SAMPLES)


def oracle_prediction(compact_distances, available):
    """Map an argmin over the COMPACT available-only array back to a candidate.

    The oracle loads only the measured files that exist, so its distance vector is
    shorter than the candidate set and indexed differently; indexing it with an
    original candidate index mis-maps whenever an unavailable candidate is not the
    trailing one (r4m review finding 5).
    """
    usable = [index for index, flag in enumerate(available) if flag]
    compact_distances = torch.as_tensor(compact_distances).reshape(-1)
    if compact_distances.numel() != len(usable):
        raise ValueError(f"oracle distances cover {compact_distances.numel()} entries but "
                         f"{len(usable)} candidates are available")
    if not usable:
        raise ValueError("no candidate has a measured RIR; the oracle is undefined")
    return int(usable[int(torch.argmin(compact_distances))])


def retrieval_split(rows, family):
    """Split by whether the PREDICTED candidate is a context member.

    Splitting on ``context_member[gt_index]`` produced an always-empty "context"
    bucket, because a GT that appears in its own context aborts upstream. What the
    control actually asks is whether the retrieval landed on a context source.
    """
    buckets = {"context": [], "non_context": []}
    for row in rows:
        block = row["families"][family]
        predicted = int(block["retrieval_pred_index"])
        key = "context" if row["context_member"][predicted] else "non_context"
        buckets[key].append(bool(block["retrieval_correct"]))
    return {name: {"n": len(values),
                   "top1": (float(np.mean(values)) if values else None)}
            for name, values in buckets.items()}


def context_coverage(rows):
    """How many candidates each query's context actually covers."""
    counts = [int(sum(1 for member in row["context_member"] if member)) for row in rows]
    return {"per_query": counts, "mean": float(np.mean(counts)) if counts else None,
            "min": int(min(counts)) if counts else None,
            "max": int(max(counts)) if counts else None}

if __name__ == "__main__":
    main()
