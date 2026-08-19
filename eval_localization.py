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
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
import torchaudio

from eval_FLAC import (CONTEXT_ID_PRECISION, orbit_provenance, resolve_are_from_checkpoint,
                       sample_target_id, source_sha)
from src.localization.agree_embed import MAX_LEN
from src.localization.candidates import (assert_gt_matches_loader, candidate_metadata,
                                         parse_ir_filename, project_to_camera)
from src.localization.scoring import (DEFAULT_RADII, aggregate, context_conditioned_baseline,
                                      cosine_sims, localization_error,
                                      nearest_context_baseline, noise_key, predict_index,
                                      summarize, uniform_baseline)


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


def summarize_run(rows, radii=DEFAULT_RADII):
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

    return {
        "flac": summarize([_query_record(row) for row in rows], radii=radii),
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


def build_provenance(args, ckpt_sha256, agree_sha256, split_hash, weights_source, n_queries):
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
        "num_samples": int(args.num_samples),
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
        "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def output_paths(out_dir, eval_name, num_samples, seed, smoke):
    """``(rows_path, summary_path)``; the seed and K are in the name because the
    protocol runs three seeds, and a smoke run is stamped so it can never be
    mistaken for a headline artifact."""
    os.makedirs(str(out_dir), exist_ok=True)
    stem = f"{eval_name}_K{int(num_samples)}_seed{int(seed)}" + ("_smoke" if smoke else "")
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
            if rec == int(receiver_node):
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
    return {"sims": sims, "available": available, "cand_cam_xyz": candidate_camera_positions(cand_set),
            "identity_index": gt_index if available[gt_index] else None,
            "num_candidates": len(cand_set.nodes), "num_samples": 1}


def parse_args(argv=None):
    """CLI for the exp_18 driver; the registered defaults are the plan's §2.3 pins."""
    parser = argparse.ArgumentParser(
        description="exp_18 loc_invert: source localization by analysis-by-synthesis inversion")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--agree-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, required=True, help="K samples per candidate")
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--parity-check", action="store_true",
                        help="run the one-query eval_FLAC parity harness and exit (C8)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"exp_18 loc_invert REFUSED: {message}")


def validate_args(args):
    """Every fail-closed rule that can be checked before touching a file or a GPU."""
    if float(args.rotate_deg) != 0.0:
        _refuse(f"--rotate-deg {args.rotate_deg} is not implemented in this driver; the "
                "registered protocol is rotate_deg=0 (announcement 05)")
    if args.max_queries is not None and not args.smoke:
        _refuse("--max-queries truncates the split and is only allowed with --smoke, so a "
                "truncated run can never be mistaken for a headline artifact")
    if args.agg == "lme" and (args.tau is None or float(args.tau) <= 0.0):
        _refuse(f"--tau must be > 0 for --agg lme, got {args.tau}")
    if int(args.num_samples) < 1:
        _refuse(f"--num-samples (K) must be >= 1, got {args.num_samples}")
    if int(args.batch_size) < 1:
        _refuse(f"--batch-size must be >= 1, got {args.batch_size}")
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
