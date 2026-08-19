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
from src.localization.candidates import (assert_gt_matches_loader, build_candidate_set,
                                         candidate_metadata, parse_ir_filename,
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
        latents = engine.sampler(chunk_noise, chunk_cond)
        wavs.append(engine.decoder(latents).clamp(-1.0, 1.0))
    wavs = torch.cat(wavs, dim=0)

    embeddings = engine.embedder(wavs)
    obs_embedding = engine.embedder(obs_wav.to(engine.device))[0]
    sims = cosine_sims(obs_embedding, embeddings.reshape(num_candidates, num_samples, -1))

    out = {"sims": sims.float().cpu(), "cand_cam_xyz": candidate_positions,
           "conditioning_xyz_cam": conditioning_positions, "control": control,
           "num_candidates": num_candidates, "num_samples": num_samples}
    if return_wavs:
        out["wavs"] = wavs
    return out


def encode_sims(sims):
    """``[M, K]`` similarities as exact hex floats (``float.hex``).

    The registered aggregation must be reproducible offline from the logged rows
    (O18), so the serialization is lossless rather than pretty: widening a float32
    to float64 is exact, and ``float.fromhex`` inverts it bit for bit.
    """
    return [[float(v).hex() for v in row] for row in sims.detach().cpu().float()]


def decode_sims(payload):
    """Inverse of :func:`encode_sims` -> float32 ``[M, K]``."""
    return torch.tensor([[float.fromhex(v) for v in row] for row in payload], dtype=torch.float32)


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


def decode_scores(payload):
    """Inverse of the row's ``scores_hex`` -> float32 ``[M]``."""
    return decode_sims([payload])[0]


def build_row(query_id, room_id, relpath, receiver_node, cand_set, cam_xyz, sims,
              context_mask, noise_keys, tau, agg, control, score_source, smoke,
              available=None, identity_index=None, substituted=False,
              context_xyz_cam=None, context_sims_hex=None):
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


def assert_registration_sha(args, dataset_config):
    """O17: a registered unseen generative run must name its pre-registration commit."""
    registered = (bool((dataset_config or {}).get("unseeneval", False))
                  and args.score_source == "flac" and not args.smoke)
    if registered and not args.registration_sha:
        _refuse(
            "--registration-sha is required for a registered unseen run (O17): the parameter "
            "file must be committed BEFORE the run and its SHA recorded in the manifest")
    return args


def build_provenance(args, ckpt_sha256, agree_sha256, split_hash, weights_source, n_queries,
                     dataset_config=None, context_digest=None):
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
        "device_name": (torch.cuda.get_device_name(0)
                        if str(getattr(args, "device", "cpu")).startswith("cuda")
                        and torch.cuda.is_available() else "cpu"),
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


def output_paths(out_dir, eval_name, num_samples, seed, smoke, score_source="flac"):
    """``(rows_path, summary_path)``; the seed and K are in the name because the
    protocol runs three seeds, and a smoke run is stamped so it can never be
    mistaken for a headline artifact."""
    os.makedirs(str(out_dir), exist_ok=True)
    tag = "gt_rir_K1" if score_source == "gt_rir" else f"K{int(num_samples)}"
    stem = f"{eval_name}_{tag}_seed{int(seed)}" + ("_smoke" if smoke else "")
    return (os.path.join(str(out_dir), f"{stem}_rows.jsonl"),
            os.path.join(str(out_dir), f"{stem}_summary.json"))


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


def write_summary(path, summary, provenance):
    """Write the summary record (provenance first, then the aggregates)."""
    with open(str(path), "w") as handle:
        json.dump({"provenance": jsonable(provenance), "summary": jsonable(summary)}, handle,
                  sort_keys=True, indent=2)
        handle.write("\n")
    return str(path)


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
    wavs, available, _paths = load_measured_rirs(room_wav_dir, cand_set, receiver_node)
    embeddings = engine.embedder(wavs.to(engine.device))
    obs_embedding = engine.embedder(obs_wav.to(engine.device))[0]
    present = cosine_sims(obs_embedding, embeddings.unsqueeze(1))

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
            "num_candidates": len(cand_set.nodes), "num_samples": 1}


def parse_args(argv=None):
    """CLI for the exp_18 driver; the registered defaults are the plan's §2.3 pins."""
    parser = argparse.ArgumentParser(
        description="exp_18 loc_invert: source localization by analysis-by-synthesis inversion")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--agree-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="K samples per candidate (required for --score-source flac)")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--agg", choices=["lme", "mean", "max"], default="lme")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-method", choices=["vanilla", "fa_invariant"], default="vanilla")
    parser.add_argument("--frame-avg-angles", type=float, nargs="+", default=None)
    parser.add_argument("--rotate-deg", type=float, default=0.0)
    parser.add_argument("--cond-autocast", choices=["default", "bf16", "off"], default="default")
    parser.add_argument("--score-source", choices=["flac", "gt_rir"], default="flac")
    parser.add_argument("--control", choices=["none", "constant_source"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--out-dir", default="worklog/worklog_yixun/exp_18_loc_invert_claude")
    parser.add_argument("--eval-name", default="exp18_loc_invert")
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
    """One query end to end: candidate set -> generation/oracle -> scored row."""
    query_id = sample_target_id(md)
    cand_set = query_candidate_set(md)
    room_id = room_id_from_relpath(md["relpath"])
    _src_node, receiver_node = parse_ir_filename(md["path"])
    # BEFORE anything can touch the poses: the context fingerprint identifies the draw.
    context_ids = sample_context_ids(md)

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
        noise = build_noise_bank(args.seed, query_id, args.num_samples, context["latent_shape"])
        noise_keys = [noise_key(args.seed, query_id, k) for k in range(int(args.num_samples))]
        outcome = run_query(engine, md, cand_set, noise, obs_wav, batch_size=args.batch_size,
                            control=args.control)
        available, identity_index = None, None

    evidence = context_evidence(engine, md, obs_wav) or {}
    return build_row(
        query_id=query_id, room_id=room_id, relpath=md["relpath"], receiver_node=receiver_node,
        cand_set=cand_set, cam_xyz=outcome["cand_cam_xyz"], sims=outcome["sims"],
        context_mask=context_mask,
        noise_keys=noise_keys, tau=args.tau, agg=args.agg, control=args.control,
        score_source=args.score_source, smoke=bool(args.smoke), available=available,
        identity_index=identity_index, substituted=False,
        context_xyz_cam=evidence.get("context_xyz_cam"),
        context_sims_hex=evidence.get("context_sims_hex"))


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

    rows_path, summary_path = output_paths(args.out_dir, args.eval_name,
                                           effective_num_samples(args), args.seed, args.smoke,
                                           score_source=args.score_source)
    partial_rows, partial_summary = rows_path + ".partial", summary_path + ".partial"
    rows, scored, seen_rooms = [], [], set()

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

    split = assert_scored_stream(scored, expected)
    print(f"identity gate passed: {len(scored)} queries, split_hash={split[:12]}...")

    summary = summarize_run(rows)
    provenance = build_provenance(args, ckpt_sha256, agree_sha256, split,
                                  context["weights_source"], len(rows),
                                  dataset_config=dataset_config,
                                  context_digest=context_stream_digest(rows))
    write_summary(partial_summary, summary, provenance)
    os.replace(partial_rows, rows_path)          # publish only verified artifacts
    os.replace(partial_summary, summary_path)
    return {"rows_path": rows_path, "summary_path": summary_path, "rows": rows,
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
    validate_dataset_split(args)
    model_config, ckpt = load_and_validate_artifacts(args)
    torch.set_float32_matmul_precision("medium")

    agree = load_agree_audio(args.agree_ckpt, args.device)
    if args.score_source == "gt_rir":
        engine = scoring_only_engine(agree, args.device)
        context = {"weights_source": "n/a", "latent_shape": None, "device": args.device}
        ckpt_sha256 = "n/a"
    else:
        engine, context = build_engine(args, agree=agree, device=args.device,
                                       model_config=model_config, ckpt=ckpt)
        ckpt_sha256 = sha256_file(args.ckpt_path)

    # Seeded exactly where evaluate_model seeds it -- after the model build and
    # before the loader, because the per-item context draw happens in the workers.
    pl.seed_everything(args.seed, workers=True)
    dataset_config = load_dataset_config(args)
    assert_registration_sha(args, dataset_config)        # O17, before any query runs
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
    result = run_evaluation(args, loader, engine, context, ckpt_sha256, agree.ckpt_sha256,
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
    if not (args.smoke or args.parity_check):
        return dataset_config
    config = dataset_config if dataset_config is not None else load_dataset_config(args)
    if bool(config.get("unseeneval", False)) or not bool(config.get("seeneval", False)):
        _refuse(
            f"--smoke/--parity-check require a SEEN-split dataset config; {args.dataset_config!r} "
            f"declares seeneval={config.get('seeneval')!r}, unseeneval={config.get('unseeneval')!r} "
            "(O16: the held-out split is not a debugging surface)")
    return config

if __name__ == "__main__":
    main()
