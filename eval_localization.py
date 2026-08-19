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
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from eval_FLAC import CONTEXT_ID_PRECISION, sample_target_id
from src.localization.candidates import (assert_gt_matches_loader, candidate_metadata,
                                         project_to_camera)
from src.localization.scoring import (aggregate, cosine_sims, localization_error,
                                      noise_key, predict_index)


def expected_split_identities(dataset):
    """The identities a split declares, in dataset order, without loading audio.

    Mirrors ``SampleDataset.__getitem__``'s own ``relpath`` derivation (every root
    that is a substring of the filename is applied in order, last one winning) so
    the expectation is built from the file list alone -- an audit that re-derived
    its expectation from the loaded stream would prove nothing.
    """
    identities = []
    for idx, filename in enumerate(dataset.filenames):
        relpath = None
        for root_path in dataset.root_paths:
            if root_path in filename:
                relpath = os.path.relpath(filename, root_path)
        identities.append(f"{idx}|{relpath if relpath is not None else filename}")
    return identities


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
    generation (O6), and ``control='constant_source'`` freezes every candidate at
    the candidate centroid (§2.8.1) so a working pipeline must collapse to the
    context-conditioned baseline.
    """
    if control not in ("none", "constant_source"):
        raise ValueError(f"unknown control {control!r} (expected 'none' or 'constant_source')")
    assert_gt_matches_loader(cand_set, base_md)

    positions = candidate_camera_positions(cand_set)
    if control == "constant_source":
        positions = np.repeat(positions.mean(axis=0, keepdims=True), positions.shape[0], axis=0)

    num_candidates = positions.shape[0]
    num_samples = int(noise.shape[0])
    metadata = [candidate_metadata(base_md, positions[m]) for m in range(num_candidates)]

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

    out = {"sims": sims.float().cpu(), "cand_cam_xyz": positions, "control": control,
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


def context_membership_mask(cand_cam_xyz, context_ids):
    """Which candidates the conditioning already reveals (plan §2.2, C1)."""
    wanted = set(context_ids)
    return [render_position_id(xyz) in wanted for xyz in np.asarray(cand_cam_xyz)]


def gt_reciprocal_rank(scores, gt_index):
    """Reciprocal rank of the GT candidate, ties broken by lowest index."""
    scores = torch.as_tensor(scores).reshape(-1)
    gt_index = int(gt_index)
    gt_score = scores[gt_index]
    better = int((scores > gt_score).sum())
    tied_before = int((scores[:gt_index] == gt_score).sum())
    return 1.0 / (better + tied_before + 1)


def decode_scores(payload):
    """Inverse of the row's ``scores_hex`` -> float32 ``[M]``."""
    return decode_sims([payload])[0]


def build_row(query_id, room_id, relpath, receiver_node, cand_set, cam_xyz, sims,
              context_mask, noise_keys, tau, agg, control, score_source, smoke,
              available=None, identity_index=None, substituted=False):
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

    return {
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
        "rr": gt_reciprocal_rank(scores, gt_index),
        "tau": float(tau) if tau is not None else None,
        "agg": agg,
        "control": control,
        "score_source": score_source,
        "identity_index": None if identity_index is None else int(identity_index),
        "substituted": bool(substituted),
        "smoke": bool(smoke),
    }


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
