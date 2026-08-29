"""exp_22 r9r -- MEASURE the observation-continuity tie's honest drift.

Round r9p *derived* the tie's tolerance: it took the engine's aggregate
changed-batching bound, multiplied it by ``sqrt(K)`` on an independence
argument, and swapped in a conditioner-token magnitude (``3.9e-3``) that was
never expressed in cosine units. Codex r9q rejected all three steps, and it was
right:

* ``SCORE_TOLERANCE`` is itself a CHANGED-batching aggregate bound, not a
  fixed-batching one, so it was never the "wrong yardstick because it assumes
  fixed batching" the r9p docstring claimed;
* the ``sqrt(K)`` step needs independent per-sample errors, and the three
  near-envelope queries of probe v2 are sign-COHERENT (8/8 negative, 7/8
  positive, 8/8 negative) with aggregate shifts of -0.53e-3, +1.16e-3 and
  -1.14e-3 -- two of them already past ``SCORE_TOLERANCE``. Deterministic
  correlated drift does not average down;
* ``3.9e-3`` is a difference between CONDITIONER TOKENS, measured before the
  DiT, the VAE and AGREE. Nothing maps it into a cosine.

So this module does not argue. It measures, on the real merged run, the two
distributions the gate actually sits between:

1. **Regeneration drift** -- for a stratified, seeded sample of
   ``(query, candidate)`` pairs across all sixteen rooms, the per-sample cosines
   the tie's own code path re-derives against the float16 sidecar the run
   published, plus the re-derived log-mean-exp aggregate against the row's
   float32 ``scores_hex``. Measured along two axes: the GPU (rows were produced
   on both -- Cafe on one device, the other fifteen rooms on the other) and the
   batch shape (the tie's single-position, K-row path against a replay at the
   run's OWN production batching, which is the bit-exactness control).
2. **Substitution movement** -- the same re-derived generations scored against
   every OTHER measured query's observation. That is the real detection margin:
   what the gate must still catch. It replaces r9p's "separation", which divided
   a query's cosine SPAN by its drift and measured dynamic range, not detection.

The bound is then the top of distribution 1 times a stated safety factor, and it
is only admissible if it sits far below the bottom of distribution 2. If the two
distributions do not separate, this module says so and derives nothing.

Read-only on the run directory. It generates ephemerally, publishes to its own
output directory, and never writes into the artifact set it measures.
"""
import argparse
import json
import os
import time

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me
from src.localization import meshgrid_offgrid_probe as op
from src.localization import meshgrid_queries as mq
from src.localization import meshgrid_report as mr
from src.localization import scoring as sc
from src.localization.reaggregate import decode_scores

#: The measurement's own seed. Fixed here, in the code, so the sample is a
#: property of the round rather than of the invocation.
DRIFT_SELECTION_SEED = 20260828

#: queries drawn per room (16 rooms x 4 = 64 pairs at the headline candidate).
QUERIES_PER_ROOM = 4

#: how far above the measured honest maximum the bound is placed.
SAFETY_FACTOR = 1.5

SELECTION_RULE_NOTE = (
    "DETERMINISTIC SELECTION RULE. Rooms are taken in sorted order. Within room i (0-based over "
    "the sorted room list) the first selected query is the room's REGISTERED PROBE QUERY -- the "
    "one the gate itself checks, so the measured population contains every case the gate will "
    "ever see -- and the remaining QUERIES_PER_ROOM-1 are drawn without replacement by "
    "numpy.random.default_rng([DRIFT_SELECTION_SEED, i]) from the room's other queries sorted by "
    "position. Within each selected query the first candidate is the row's OWN headline "
    "prediction row (tie_candidate_row, the gate's rule, not a copy of it) and the second is "
    "drawn by numpy.random.default_rng([DRIFT_SELECTION_SEED, position]) from the row's other "
    "candidate rows. Nothing about the sample depends on a delta, so it cannot be steered "
    "toward or away from the answer")

BATCH_SHAPE_NOTE = (
    "THE TWO PATHS. 'tie' is the gate's path: the source branch is called on ONE candidate "
    "position (batch 1, whatever source_chunk says, because source_conditioning chunks the "
    "position list) and num_samples rows go through the DiT, the VAE and AGREE. 'matched' is the "
    "run's own path: the receiver's whole candidate union through the source branch at the run's "
    "source_chunk, then the query's candidates in batch_rows-sized forwards -- ReceiverCache + "
    "meshgrid_engine._score_one_query, the same functions the scored pass called. A 'matched' "
    "delta is what remains when batch shape is held equal, so it separates batch-shape drift "
    "from everything else that could move a cosine")

GPU_AXIS_NOTE = (
    "THE GPU AXIS. The P1 pass ran as two shards on two devices, so a row's drift depends on "
    "whether it is re-derived on the device that produced it. The room -> shard map is read from "
    "the run's own merge_report.json; the shard -> device map is external knowledge (the "
    "operator's launch record) and is passed in explicitly and stamped into this artifact rather "
    "than inferred. 'same_gpu' means the measuring device is the one that produced the row")

SUBSTITUTION_NOTE = (
    "THE DETECTION MARGIN. For each measured query the tie's re-derived generations are scored "
    "against every OTHER measured query's observation and compared to the SAME frozen sidecar "
    "slice, which is exactly the arithmetic the gate would perform if it were handed the wrong "
    "observation. The minimum over all ordered cross pairs is the worst case the gate must still "
    "catch. This replaces r9p's 'separation_vs_span', which divided a query's own cosine SPAN by "
    "its drift: dynamic range, not substitution evidence (Codex r9q, item 3)")

MEASUREMENT_JSON = "drift_measurement.json"
MEASUREMENT_MARKDOWN = "drift_measurement.md"
MEASUREMENT_NPZ = "drift_deltas.npz"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# which rows were produced where
# --------------------------------------------------------------------------- #
def room_shard_map(merge_report):
    """``{room_id: shard_dir}`` from the run's OWN merge report."""
    out = {}
    for shard in merge_report.get("shards") or []:
        for room_id in shard.get("rooms") or []:
            if room_id in out:
                raise ValueError(f"room {room_id!r} is claimed by two shards; the merge report "
                                 "cannot say where its rows were produced")
            out[str(room_id)] = str(shard.get("dir"))
    if not out:
        raise ValueError("the merge report declares no shard rooms; without it there is no "
                         "room -> device map and the GPU axis cannot be measured")
    return out


def parse_shard_devices(pairs):
    """``["cafe=cuda:0", ...] -> {"cafe": "cuda:0"}`` -- the operator's launch record."""
    out = {}
    for pair in pairs or []:
        if "=" not in str(pair):
            raise ValueError(f"--shard-device expects <shard-substring>=<device>, got {pair!r}")
        key, device = str(pair).split("=", 1)
        key, device = key.strip(), device.strip()
        if not key or not device:
            raise ValueError(f"--shard-device expects <shard-substring>=<device>, got {pair!r}")
        out[key] = device
    return out


def shard_device(shard_dir, devices):
    """The device a shard ran on, matched by substring on the shard directory."""
    hits = sorted(key for key in devices if key in str(shard_dir))
    if len(hits) != 1:
        raise ValueError(
            f"the shard {shard_dir!r} matches {hits or 'no'} --shard-device key(s); the GPU axis "
            "is only measurable when every shard maps to exactly one device")
    return devices[hits[0]]


# --------------------------------------------------------------------------- #
# the sample
# --------------------------------------------------------------------------- #
def select_queries(plan, *, seed=DRIFT_SELECTION_SEED, per_room=QUERIES_PER_ROOM):
    """The stratified, seeded query sample -- see :data:`SELECTION_RULE_NOTE`."""
    probes = me.registered_probe_queries(plan)
    chosen = []
    for ordinal, room_id in enumerate(sorted(plan.rooms)):
        room_plan = me.load_room_plan(plan, room_id)
        queries = sorted(room_plan.queries, key=lambda q: int(q.position))
        by_id = {q.query_id: q for q in queries}
        registered = probes.get(room_id)
        picked = []
        if registered is not None and registered in by_id:
            picked.append(by_id[registered])
        pool = [q for q in queries if q.query_id not in {p.query_id for p in picked}]
        want = max(0, int(per_room) - len(picked))
        if want and pool:
            rng = np.random.default_rng([int(seed), int(ordinal)])
            take = rng.choice(len(pool), size=min(want, len(pool)), replace=False)
            picked.extend(pool[int(i)] for i in sorted(int(v) for v in take))
        for query in picked:
            chosen.append({"room_id": room_id, "query_id": query.query_id,
                           "position": int(query.position), "query": query,
                           "is_registered_probe_query": query.query_id == registered})
    return chosen


def select_candidate_rows(row, *, seed=DRIFT_SELECTION_SEED, extra=1):
    """``[headline_row, other_row...]`` -- the gate's candidate first."""
    headline, _index, _k = op.tie_candidate_row(row)
    n_candidates = int(row["n_candidates"])
    rows = [int(headline)]
    if extra > 0 and n_candidates > 1:
        rng = np.random.default_rng([int(seed), int(row["position"])])
        pool = [i for i in range(n_candidates) if i != int(headline)]
        take = rng.choice(len(pool), size=min(int(extra), len(pool)), replace=False)
        rows.extend(pool[int(i)] for i in sorted(int(v) for v in take))
    return rows


# --------------------------------------------------------------------------- #
# one measurement
# --------------------------------------------------------------------------- #
def compare_slice(stored, rederived, *, stored_aggregate=None, tau=me.TAU):
    """The delta bundle for ONE candidate's K samples.

    Sign coherence is reported because it is the property that killed the
    ``sqrt(K)`` argument: errors that all point the same way do not average down
    into the aggregate, they accumulate into it.
    """
    stored = np.asarray(stored, dtype=np.float64).reshape(-1)
    rederived = np.asarray(rederived, dtype=np.float64).reshape(-1)
    if stored.shape != rederived.shape:
        raise ValueError(f"stored {stored.shape} and rederived {rederived.shape} disagree on K")
    signed = rederived - stored
    positive = int((signed > 0).sum())
    negative = int((signed < 0).sum())
    # the sidecar is float16, so part of every delta is the sidecar's own
    # rounding rather than drift. The gate adds this per query already, so the
    # quantity a bound is derived from is the EXCESS over it -- otherwise the
    # rounding would be counted twice
    half_ulp = float(mr.float16_half_ulp(np.asarray(stored, dtype=np.float16)))
    max_abs = float(np.abs(signed).max())
    out = {"k": int(stored.shape[0]),
           "stored": [float(v) for v in stored],
           "rederived": [float(v) for v in rederived],
           "signed_deltas": [float(v) for v in signed],
           "max_abs_delta": max_abs,
           "sidecar_half_ulp": half_ulp,
           "excess_over_half_ulp": float(max(0.0, max_abs - half_ulp)),
           "mean_signed_delta": float(signed.mean()),
           "n_positive": positive, "n_negative": negative,
           "sign_coherent": bool(positive == 0 or negative == 0),
           "sign_coherence": float(max(positive, negative) / max(1, stored.shape[0]))}
    aggregate = float(sc.aggregate(torch.as_tensor(rederived).float().reshape(1, -1),
                                   method="lme", tau=float(tau))[0])
    out["rederived_aggregate"] = aggregate
    if stored_aggregate is not None:
        out["stored_aggregate"] = float(stored_aggregate)
        out["aggregate_delta"] = float(aggregate - float(stored_aggregate))
        out["abs_aggregate_delta"] = abs(out["aggregate_delta"])
    return out


def row_aggregate(row, candidate_row, k=None):
    """The row's OWN float32 published score for one candidate at prefix ``k``.

    float32, not the float16 sidecar: the aggregate comparison therefore carries
    no sidecar quantization at all, which is what makes it directly comparable
    to ``SCORE_TOLERANCE``.
    """
    key = str(k if k is not None else max(int(v) for v in row["by_k"]))
    scores = decode_scores(row["by_k"][key]["scores_hex"])
    return float(np.asarray(scores, dtype=np.float64)[int(candidate_row)])


def measure_tie_pair(engine, query, md, context, row, sims, obs_embedding, candidate_row, *,
                     seed=me.SEED, num_samples=me.NUM_SAMPLES,
                     noise_policy=me.NOISE_KEY_POLICY, source_chunk=1, tau=me.TAU):
    """One ``(query, candidate)`` measured through the GATE's own path.

    Returns the delta bundle plus the re-derived embeddings, which the
    substitution matrix consumes so that measurement 2 is scored off exactly the
    generations measurement 1 was scored off.
    """
    candidate_row = int(candidate_row)
    candidate_index = int(row["candidate_indices"][candidate_row])
    num_samples = int(num_samples)
    stored = np.asarray(sims, dtype=np.float16)[candidate_row, :num_samples]
    embeddings = op.regenerate_tie_embeddings(
        engine, query, md, context, candidate_row, candidate_index, seed=seed,
        num_samples=num_samples, noise_policy=noise_policy, source_chunk=source_chunk)
    rederived = sc.cosine_sims(torch.as_tensor(obs_embedding).float().reshape(-1),
                               embeddings)[0].double().numpy()
    record = compare_slice(stored, rederived, tau=tau,
                           stored_aggregate=row_aggregate(row, candidate_row))
    record.update({"path": "tie", "candidate_row": candidate_row,
                   "candidate_index": candidate_index})
    return record, embeddings.reshape(num_samples, -1)


def measure_matched_query(engine, query, md, context, receiver_id, union, positions_cam,
                          row, sims, obs_embedding, *, seed=me.SEED,
                          num_samples=me.NUM_SAMPLES, noise_policy=me.NOISE_KEY_POLICY,
                          batch_rows=256, source_chunk=16, tau=me.TAU, candidate_rows=()):
    """The whole query replayed at the RUN'S OWN batching -- the control.

    Everything the tie changes is put back: the receiver's full candidate union
    goes through the source branch at the run's ``source_chunk``, and the
    query's candidates are generated in ``batch_rows``-sized forwards, through
    ``ReceiverCache`` and ``meshgrid_engine._score_one_query`` -- the functions
    the scored pass itself called. A near-zero result here is what makes the
    batch shape, rather than the checkpoint, the loader or the observation, the
    thing that moved the tie's cosines.
    """
    # the private helper is imported ON PURPOSE: a local re-implementation of
    # the production scoring loop would no longer be the production path, and
    # being the production path is the entire content of this control
    cache = me.ReceiverCache.build(engine.conditioner, receiver_id, {"depth": md["depth"]},
                                   [int(i) for i in union],
                                   np.asarray(positions_cam, dtype=np.float64), engine.device,
                                   chunk=int(source_chunk))
    replayed = me._score_one_query(                                   # noqa: SLF001 -- see above
        engine, query, context, cache, obs_embedding, seed=seed, num_samples=int(num_samples),
        noise_policy=noise_policy, batch_rows=int(batch_rows), timer=me._Timer(engine.device))
    replayed = np.asarray(replayed.double().numpy(), dtype=np.float64)

    stored_all = np.asarray(sims, dtype=np.float16).astype(np.float64)
    if stored_all.shape != replayed.shape:
        raise ValueError(f"the replay is {replayed.shape} but the sidecar is {stored_all.shape}; "
                         "the replay is not this query's")
    # the sidecar is float16, so the honest floor of this comparison is the
    # sidecar's own rounding, reported beside the deltas rather than subtracted
    deltas = np.abs(replayed - stored_all)
    half_ulp = float(mr.float16_half_ulp(np.asarray(sims, dtype=np.float16)))
    # THE bit-exactness statement: if the replay is the production computation,
    # rounding it back to the sidecar's dtype reproduces the sidecar exactly.
    # Comparing float64 against a float16 store can never read zero, so this is
    # the only form in which "bit-exact" is a meaningful claim here
    n_mismatch = int((replayed.astype(np.float16) != np.asarray(sims, dtype=np.float16)).sum())
    signed = replayed - stored_all
    coherent = int(((signed > 0).all(axis=1) | (signed < 0).all(axis=1)).sum())
    summary = {"path": "matched", "n_candidates": int(stored_all.shape[0]),
               "k": int(stored_all.shape[1]),
               "max_abs_delta": float(deltas.max()),
               "quantiles": quantiles(deltas.reshape(-1)),
               "sidecar_half_ulp": half_ulp,
               "n_float16_mismatch": n_mismatch,
               "float16_bit_exact": bool(n_mismatch == 0),
               "n_above_half_ulp": int((deltas > half_ulp).sum()),
               "share_above_half_ulp": float((deltas > half_ulp).mean()),
               "n_sign_coherent_candidates": coherent,
               "share_sign_coherent": float(coherent / max(1, stored_all.shape[0])),
               "n_union": int(len(union)), "batch_rows": int(batch_rows),
               "source_chunk": int(source_chunk)}
    stored_scores = np.asarray(decode_scores(
        row["by_k"][str(max(int(v) for v in row["by_k"]))]["scores_hex"]), dtype=np.float64)
    rederived_scores = sc.aggregate(torch.as_tensor(replayed).float(), method="lme",
                                    tau=float(tau)).double().numpy()
    aggregate_deltas = np.abs(rederived_scores - stored_scores)
    summary["aggregate"] = {"max_abs_delta": float(aggregate_deltas.max()),
                            "quantiles": quantiles(aggregate_deltas),
                            "n_above_score_tolerance":
                                int((aggregate_deltas > me.SCORE_TOLERANCE).sum()),
                            "score_tolerance": float(me.SCORE_TOLERANCE)}
    per_candidate = []
    for candidate_row in candidate_rows:
        candidate_row = int(candidate_row)
        entry = compare_slice(stored_all[candidate_row], replayed[candidate_row], tau=tau,
                              stored_aggregate=float(stored_scores[candidate_row]))
        entry.update({"path": "matched", "candidate_row": candidate_row,
                      "candidate_index": int(row["candidate_indices"][candidate_row])})
        per_candidate.append(entry)
    return summary, per_candidate, deltas


# --------------------------------------------------------------------------- #
# the substitution matrix
# --------------------------------------------------------------------------- #
def substitution_deltas(entries):
    """``max_k |cos(E(obs_j), E(h_hat_i,k)) - stored_i,k|`` for every ordered i != j.

    ``entries`` are ``{query_id, obs_embedding [D], embeddings [K, D], stored [K]}``
    -- one per measured query, from the tie path.
    """
    ids = [str(entry["query_id"]) for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("the substitution matrix needs one entry per query; ids repeat")
    obs = torch.stack([torch.as_tensor(entry["obs_embedding"]).float().reshape(-1)
                       for entry in entries])
    out = []
    for i, entry in enumerate(entries):
        embeddings = torch.as_tensor(entry["embeddings"]).float()
        stored = np.asarray(entry["stored"], dtype=np.float64).reshape(-1)
        # [n_obs, K] cosines of THIS query's generations against every observation
        cosines = (obs @ embeddings.T).double().numpy()
        for j in range(len(entries)):
            if i == j:
                continue
            delta = float(np.abs(cosines[j] - stored).max())
            out.append({"query_id": ids[i], "observation_query_id": ids[j],
                        "same_room": bool(entry.get("room_id") == entries[j].get("room_id")),
                        "max_abs_delta": delta})
    return out


# --------------------------------------------------------------------------- #
# distributions and the bound
# --------------------------------------------------------------------------- #
QUANTILES = (0.5, 0.9, 0.99, 1.0)


def quantiles(values, points=QUANTILES):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {}
    return {f"q{int(round(p * 100)):02d}": float(np.quantile(values, p)) for p in points}


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "median": float(np.median(values)),
            "quantiles": quantiles(values)}


def summarize_records(records):
    """Per-sample and per-candidate distributions, split by path and GPU axis."""
    out = {}
    for path in sorted({str(record["path"]) for record in records}):
        subset = [record for record in records if str(record["path"]) == path]
        block = {"all": _block(subset)}
        for axis in ("same_gpu", "cross_gpu"):
            want = axis == "same_gpu"
            rows = [record for record in subset if bool(record.get("same_gpu")) == want]
            if rows:
                block[axis] = _block(rows)
        out[path] = block
    return out


def _block(records):
    per_sample = np.concatenate([np.abs(np.asarray(record["signed_deltas"], dtype=np.float64))
                                 for record in records]) if records else np.zeros(0)
    per_candidate = [float(record["max_abs_delta"]) for record in records]
    excess = [float(record["excess_over_half_ulp"]) for record in records
              if "excess_over_half_ulp" in record]
    aggregate = [float(record["abs_aggregate_delta"]) for record in records
                 if "abs_aggregate_delta" in record]
    coherent = [bool(record["sign_coherent"]) for record in records]
    return {"n_candidates": len(records),
            "per_sample_abs_delta": summarize(per_sample),
            "per_candidate_max_abs_delta": summarize(per_candidate),
            "excess_over_half_ulp": summarize(excess),
            "abs_aggregate_delta": summarize(aggregate),
            "n_aggregate_above_score_tolerance":
                int(sum(1 for value in aggregate if value > me.SCORE_TOLERANCE)),
            "score_tolerance": float(me.SCORE_TOLERANCE),
            "share_sign_coherent": (float(sum(coherent) / len(coherent)) if coherent else None)}


def round_up_2sig(value):
    """Round UP to two significant figures -- a bound is never rounded down."""
    value = float(value)
    if value <= 0.0:
        return 0.0
    exponent = int(np.floor(np.log10(value))) - 1
    scale = 10.0 ** exponent
    return float(np.ceil(value / scale) * scale)


def derive_bound(honest_max, substitution_min, *, safety_factor=SAFETY_FACTOR,
                 min_separation=5.0):
    """The bound, and whether the two measured distributions admit one at all.

    ``ok`` is False -- and no bound is offered -- when the honest maximum times
    the safety factor does not sit at least ``min_separation`` times below the
    smallest substituted movement. There is no fallback: a gate wedged between
    overlapping distributions is not a gate.
    """
    honest_max = float(honest_max)
    substitution_min = float(substitution_min)
    raw = honest_max * float(safety_factor)
    value = round_up_2sig(raw)
    separation = (substitution_min / value) if value > 0 else float("inf")
    ok = bool(value > 0.0 and substitution_min > 0.0 and separation >= float(min_separation))
    return {"ok": ok,
            "measured_honest_max": honest_max,
            "safety_factor": float(safety_factor),
            "raw": raw, "value": value,
            "substitution_min": substitution_min,
            "separation_ratio": float(separation),
            "min_separation_required": float(min_separation),
            "why": ("the bound sits between the measured distributions with the required "
                    "separation" if ok else
                    "the honest drift and the substituted movement do not separate cleanly; no "
                    "bound is derived (r9r: STOP and report rather than pick one)")}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def attach_receiver_groups(plan, entries):
    """Give each selected query its receiver group's union and camera positions.

    Needed only by the matched-batching replay, and gathered here so the room
    manifests are read once for the whole sample rather than once per query.
    """
    by_room = {}
    for entry in entries:
        by_room.setdefault(entry["room_id"], []).append(entry)
    for room_id, room_entries in by_room.items():
        room_plan = me.load_room_plan(plan, room_id)
        base = np.asarray(room_plan.base, dtype=np.float64)
        groups = list(me.receiver_groups(room_plan))
        for entry in room_entries:
            group = next(g for g in groups
                         if any(q.query_id == entry["query_id"] for q in g.queries))
            entry["receiver_id"] = str(group.receiver_id)
            entry["union"] = [int(i) for i in group.union]
            entry["positions_cam"] = (base[np.asarray(group.union, dtype=np.int64)]
                                      - np.asarray(group.receiver_xyz, dtype=np.float64))
    return entries


def run_measurement(engine, stream, records, plan, run_dir, *, device, shard_by_room,
                    shard_devices, binding_sha256=None, binding=None, seed=me.SEED,
                    num_samples=me.NUM_SAMPLES, noise_policy=me.NOISE_KEY_POLICY, tau=me.TAU,
                    tie_source_chunk=1, batch_rows=256, source_chunk=16,
                    per_room=QUERIES_PER_ROOM, extra_candidates=1, selection_seed=None,
                    matched_per_room=1, on_query=None):
    """Walk the released stream once and measure the sample it carries.

    The walk is the run's own walk: every query's context draw depends on the
    complete pass, so the stream is consumed from position 0 exactly as the
    scored pass consumed it, and only the selected positions are generated.
    """
    selection_seed = DRIFT_SELECTION_SEED if selection_seed is None else int(selection_seed)
    selected = attach_receiver_groups(
        plan, select_queries(plan, seed=selection_seed, per_room=per_room))
    wanted = {entry["query_id"]: entry for entry in selected}
    matched_wanted = set()
    per_room_seen = {}
    for entry in selected:
        seen = per_room_seen.get(entry["room_id"], 0)
        if seen < int(matched_per_room):
            matched_wanted.add(entry["query_id"])
            per_room_seen[entry["room_id"]] = seen + 1

    by_position = {int(record["position"]): record for record in records}
    pair_records, query_records, substitution_entries, matched_arrays = [], [], [], {}
    for position, (obs_wav, raw_md) in enumerate(stream):
        record = by_position.get(position)
        if record is None or record["query_id"] not in wanted:
            continue
        entry = wanted[record["query_id"]]
        md = me.GuardedMetadata(raw_md)
        me.verify_context_record(md, record, position)
        if obs_wav is None:
            raise ValueError(f"stream position {position}: the loader returned no observation")
        query = entry["query"]
        row = op.load_grid_row(run_dir, query, binding_sha256=binding_sha256, binding=binding)
        sims = row["_sims"]

        shard = shard_by_room[entry["room_id"]]
        produced_on = shard_device(shard, shard_devices)
        same_gpu = str(produced_on) == str(device)

        obs_embedding = torch.as_tensor(
            engine.embedder(torch.as_tensor(obs_wav).to(engine.device)))[0].float().cpu()
        context = me.context_conditioning(engine.conditioner, md, engine.device)
        candidate_rows = select_candidate_rows(row, seed=selection_seed, extra=extra_candidates)

        started = time.perf_counter()
        common = {"room_id": entry["room_id"], "query_id": entry["query_id"],
                  "position": position, "device": str(device),
                  "produced_on": str(produced_on), "same_gpu": bool(same_gpu),
                  "shard": str(shard), "receiver_id": entry["receiver_id"],
                  "is_registered_probe_query": bool(entry["is_registered_probe_query"]),
                  "n_candidates": int(row["n_candidates"]),
                  "row_batching": dict(row.get("batching") or {})}
        for rank, candidate_row in enumerate(candidate_rows):
            measured, embeddings = measure_tie_pair(
                engine, query, md, context, row, sims, obs_embedding, candidate_row,
                seed=seed, num_samples=num_samples, noise_policy=noise_policy,
                source_chunk=tie_source_chunk, tau=tau)
            measured.update(common)
            measured["is_headline_candidate"] = bool(rank == 0)
            pair_records.append(measured)
            if rank == 0:
                substitution_entries.append(
                    {"query_id": entry["query_id"], "room_id": entry["room_id"],
                     "obs_embedding": obs_embedding.numpy(),
                     "embeddings": embeddings.numpy(),
                     "stored": list(measured["stored"])})

        matched_summary = None
        if entry["query_id"] in matched_wanted and same_gpu:
            matched_summary, matched_pairs, deltas = measure_matched_query(
                engine, query, md, context, entry["receiver_id"], entry["union"],
                entry["positions_cam"], row, sims, obs_embedding,
                seed=seed, num_samples=num_samples, noise_policy=noise_policy,
                batch_rows=batch_rows, source_chunk=source_chunk, tau=tau,
                candidate_rows=candidate_rows)
            matched_summary.update(common)
            for measured in matched_pairs:
                measured.update(common)
                measured["is_headline_candidate"] = bool(
                    int(measured["candidate_row"]) == int(candidate_rows[0]))
                pair_records.append(measured)
            matched_arrays[f"matched|{entry['query_id']}"] = deltas.astype(np.float32)

        query_records.append(dict(common, seconds=float(time.perf_counter() - started),
                                  candidate_rows=[int(v) for v in candidate_rows],
                                  matched=matched_summary))
        if on_query is not None:
            on_query(query_records[-1])
        if len(query_records) == len(wanted):
            break

    missing = sorted(set(wanted) - {record["query_id"] for record in query_records})
    if missing:
        raise ValueError(f"{len(missing)} selected queries never appeared in the stream "
                         f"(first {missing[:3]}); the sample is not the one the rule selected")
    return {"pairs": pair_records, "queries": query_records,
            "substitution_entries": substitution_entries,
            "matched_arrays": matched_arrays,
            "selection": {"seed": selection_seed, "per_room": int(per_room),
                          "extra_candidates": int(extra_candidates),
                          "matched_per_room": int(matched_per_room),
                          "n_queries": len(query_records), "n_pairs": len(pair_records),
                          "rule": SELECTION_RULE_NOTE}}


# --------------------------------------------------------------------------- #
# publication
# --------------------------------------------------------------------------- #
def build_report(measured, *, device, run_dir, shard_devices, provenance, protocol,
                 substitution=None, bound=None):
    """The measurement artifact -- distributions, not verdicts."""
    pairs = measured["pairs"]
    report = {"experiment": "exp_22 r9r observation-continuity drift measurement",
              "created_utc": utc_now(),
              "device": str(device), "run_dir": str(run_dir),
              "shard_devices": dict(shard_devices),
              "selection": measured["selection"],
              "protocol": dict(protocol),
              "provenance": dict(provenance),
              "selection_rule_note": SELECTION_RULE_NOTE,
              "batch_shape_note": BATCH_SHAPE_NOTE,
              "gpu_axis_note": GPU_AXIS_NOTE,
              "substitution_note": SUBSTITUTION_NOTE,
              "summary": summarize_records(pairs),
              "queries": measured["queries"],
              "pairs": pairs}
    if substitution is not None:
        report["substitution"] = substitution
    if bound is not None:
        report["bound"] = bound
    return report


def summarize_substitution(deltas):
    """min/median/max over the ordered cross pairs, plus the per-query worst case."""
    values = [float(entry["max_abs_delta"]) for entry in deltas]
    per_query = {}
    for entry in deltas:
        key = str(entry["query_id"])
        value = float(entry["max_abs_delta"])
        per_query[key] = min(per_query.get(key, float("inf")), value)
    same_room = [float(entry["max_abs_delta"]) for entry in deltas if entry.get("same_room")]
    return {"n_pairs": len(values), "note": SUBSTITUTION_NOTE,
            "overall": summarize(values),
            "same_room": summarize(same_room),
            "cross_room": summarize([float(entry["max_abs_delta"]) for entry in deltas
                                     if not entry.get("same_room")]),
            "per_query_min": {key: float(value) for key, value in sorted(per_query.items())},
            "worst_case_pair": (min(deltas, key=lambda entry: float(entry["max_abs_delta"]))
                                if deltas else None)}


def honest_max(summary, key="excess_over_half_ulp"):
    """The top of the measured honest distribution, over BOTH paths and BOTH GPUs.

    The gate does not get to know which device produced a row or how the run was
    batched, so the bound is taken over everything measured rather than over the
    friendliest slice. The default quantity is the EXCESS over the sidecar's own
    float16 rounding, because the gate adds that rounding back per query.
    """
    values = [block["all"].get(key, {}).get("max", 0.0)
              for block in summary.values() if block.get("all", {}).get("n_candidates")]
    return float(max(values)) if values else 0.0


def render_markdown(report):
    devices = report.get("devices") or [report.get("device")]
    lines = [f"# {report['experiment']}", "",
             f"- created: `{report['created_utc']}`",
             f"- run: `{report['run_dir']}`",
             f"- measured on: `{', '.join(str(device) for device in devices)}`, shard devices: "
             f"`{json.dumps(report['shard_devices'], sort_keys=True)}`",
             f"- sample: {report['selection']['n_queries']} queries per device / "
             f"{len(report['pairs'])} (query, candidate) measurements in total, seed "
             f"{report['selection']['seed']}", ""]
    lines += ["## Selection rule", "", f"> {report['selection_rule_note']}", "",
              "## The two paths", "", f"> {report['batch_shape_note']}", "",
              "## The GPU axis", "", f"> {report['gpu_axis_note']}", ""]

    lines += ["## 1. Regeneration drift", "",
              "| path | slice | candidates | per-sample max | per-sample q99 | per-sample median "
              "| per-candidate max | aggregate max | aggregate > SCORE_TOLERANCE | sign-coherent |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(report["summary"]):
        for slice_name, block in sorted(report["summary"][path].items()):
            per_sample = block["per_sample_abs_delta"]
            per_candidate = block["per_candidate_max_abs_delta"]
            aggregate = block["abs_aggregate_delta"]
            lines.append(
                f"| {path} | {slice_name} | {block['n_candidates']} | "
                f"{mr.format_number(per_sample.get('max'), 6)} | "
                f"{mr.format_number((per_sample.get('quantiles') or {}).get('q99'), 6)} | "
                f"{mr.format_number(per_sample.get('median'), 6)} | "
                f"{mr.format_number(per_candidate.get('max'), 6)} | "
                f"{mr.format_number(aggregate.get('max'), 6)} | "
                f"{block['n_aggregate_above_score_tolerance']} of {aggregate.get('n', 0)} | "
                f"{mr.format_number(block['share_sign_coherent'], 3)} |")
    lines.append("")

    matched = [query for query in report["queries"] if query.get("matched")]
    if matched:
        lines += ["### Matched-batching replay (the bit-exactness control)", "",
                  "| room | query | candidates | union | max abs delta | share above sidecar "
                  "half-ulp | aggregate max | aggregate > SCORE_TOLERANCE |",
                  "|---|---|---|---|---|---|---|---|"]
        for query in matched:
            block = query["matched"]
            lines.append(
                f"| {query['room_id']} | `{str(query['query_id']).split('|')[0]}` | "
                f"{block['n_candidates']} | {block['n_union']} | "
                f"{mr.format_number(block['max_abs_delta'], 6)} | "
                f"{mr.format_number(block['share_above_half_ulp'], 4)} | "
                f"{mr.format_number(block['aggregate']['max_abs_delta'], 6)} | "
                f"{block['aggregate']['n_above_score_tolerance']} of {block['n_candidates']} |")
        lines.append("")

    substitution = report.get("substitution")
    if substitution:
        overall = substitution["overall"]
        lines += ["## 2. Substitution movement", "", f"> {substitution['note']}", "",
                  f"- ordered cross pairs: **{substitution['n_pairs']}**",
                  f"- minimum: **{mr.format_number(overall.get('min'), 6)}**, median "
                  f"{mr.format_number(overall.get('median'), 6)}, max "
                  f"{mr.format_number(overall.get('max'), 6)}",
                  f"- same-room pairs: n={substitution['same_room'].get('n', 0)}, min "
                  f"{mr.format_number(substitution['same_room'].get('min'), 6)}",
                  f"- cross-room pairs: n={substitution['cross_room'].get('n', 0)}, min "
                  f"{mr.format_number(substitution['cross_room'].get('min'), 6)}", ""]

    bound = report.get("bound")
    if bound:
        lines += ["## 3. The bound", "",
                  f"- measured honest maximum: **{mr.format_number(bound['measured_honest_max'], 6)}**",
                  f"- safety factor: {mr.format_number(bound['safety_factor'], 2)} -> raw "
                  f"{mr.format_number(bound['raw'], 6)} -> rounded up "
                  f"**{mr.format_number(bound['value'], 6)}**",
                  f"- smallest substituted movement: "
                  f"**{mr.format_number(bound['substitution_min'], 6)}**",
                  f"- separation: **{mr.format_number(bound['separation_ratio'], 1)}x** "
                  f"(required >= {mr.format_number(bound['min_separation_required'], 1)}x)",
                  f"- admissible: **{'YES' if bound['ok'] else 'NO'}** -- {bound['why']}", ""]
    return "\n".join(lines) + "\n"


def write_report(out_dir, report, matched_arrays=None, pair_deltas=None):
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = os.path.join(str(out_dir), MEASUREMENT_JSON)
    markdown_path = os.path.join(str(out_dir), MEASUREMENT_MARKDOWN)
    me.write_json(json_path, mr.jsonable(report))
    with open(markdown_path, "w") as handle:
        handle.write(render_markdown(report))
    npz_path = os.path.join(str(out_dir), MEASUREMENT_NPZ)
    payload = {key: np.asarray(value) for key, value in (matched_arrays or {}).items()}
    if pair_deltas:
        for key, value in pair_deltas.items():
            payload[key] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(npz_path, **payload)
    return {"json": json_path, "markdown": markdown_path, "npz": npz_path}


def merge_device_reports(reports):
    """Combine the per-device measurements into ONE distribution and bound.

    Each device measured the SAME pairs, so the union is what the gate faces:
    a control may be run on either device against rows produced on either.
    """
    if not reports:
        raise ValueError("nothing to merge")
    pairs, queries, substitution = [], [], []
    for report in reports:
        pairs.extend(report["pairs"])
        queries.extend(report["queries"])
        substitution.extend((report.get("substitution") or {}).get("pairs") or [])
    merged = {"experiment": reports[0]["experiment"] + " (merged over devices)",
              "created_utc": utc_now(),
              "run_dir": reports[0]["run_dir"],
              "devices": sorted({str(report["device"]) for report in reports}),
              "shard_devices": reports[0]["shard_devices"],
              "selection": reports[0]["selection"],
              "protocol": reports[0]["protocol"],
              "provenance": {str(report["device"]): report["provenance"] for report in reports},
              "selection_rule_note": SELECTION_RULE_NOTE,
              "batch_shape_note": BATCH_SHAPE_NOTE,
              "gpu_axis_note": GPU_AXIS_NOTE,
              "substitution_note": SUBSTITUTION_NOTE,
              "summary": summarize_records(pairs),
              "queries": queries, "pairs": pairs,
              "sources": [{"device": report["device"], "json": report.get("_path")}
                          for report in reports]}
    if substitution:
        merged["substitution"] = summarize_substitution(substitution)
        merged["substitution"]["pairs"] = substitution
    return merged


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="exp_22 r9r: measure the observation-continuity tie's honest drift and the "
                    "substituted-observation movement it has to separate from")
    parser.add_argument("--merge", nargs="+", default=None,
                        help="MERGE MODE: combine per-device measurement JSONs into one "
                             "distribution and derive the bound. No GPU, no run directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--audit-report", default=None)
    parser.add_argument("--context-manifest", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--agree-ckpt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--shard-device", action="append", default=None,
                        help="<shard-dir-substring>=<device>, once per shard; the operator's "
                             "launch record, stamped into the artifact")
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
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--source-chunk", type=int, default=16)
    parser.add_argument("--tie-source-chunk", type=int, default=1)
    parser.add_argument("--per-room", type=int, default=QUERIES_PER_ROOM)
    parser.add_argument("--extra-candidates", type=int, default=1)
    parser.add_argument("--matched-per-room", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=DRIFT_SELECTION_SEED)
    parser.add_argument("--safety-factor", type=float, default=SAFETY_FACTOR)
    parser.add_argument("--dump-cases-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _refuse(message):
    raise SystemExit(f"REFUSED: {message}")


def validate_args(args):
    if args.merge:
        return True
    for name in ("run_dir", "out_dir", "audit_report", "context_manifest", "ckpt_path",
                 "model_config", "dataset_config"):
        if not getattr(args, name):
            _refuse(f"--{name.replace('_', '-')} is required to measure")
    if not args.shard_device:
        _refuse("--shard-device <shard>=<device> is required: without the shard -> device map "
                "the same-GPU / cross-GPU axis cannot be labelled, and mislabelling it would "
                "make the measurement worse than none")
    if os.path.abspath(str(args.out_dir)) == os.path.abspath(str(args.run_dir)):
        _refuse("--out-dir may not be the measured run directory")
    return True


def _run_merge(args):
    reports = []
    for path in args.merge:
        with open(str(path)) as handle:
            report = json.load(handle)
        report["_path"] = str(path)
        reports.append(report)
    merged = merge_device_reports(reports)
    substitution = merged.get("substitution") or {}
    bound = derive_bound(honest_max(merged["summary"]),
                         (substitution.get("overall") or {}).get("min", 0.0),
                         safety_factor=float(args.safety_factor))
    merged["bound"] = bound
    published = write_report(args.out_dir, merged)
    print(render_markdown(merged))
    print(f"merged measurement -> {published['json']}")
    if not bound["ok"]:
        print("\nNO BOUND DERIVED: " + bound["why"])
    return 0


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.merge:
        return _run_merge(args)

    from localize_meshgrid import _iter_items as iter_stream_items
    from localize_meshgrid import build_run_binding

    with open(args.model_config) as handle:
        model_config = json.load(handle)
    agree_path = args.agree_ckpt or \
        mq.with_resolved_agree(model_config)["training"]["metrics"]["AGREE_ckpt"]

    plan = me.load_audit_plan(args.audit_report, branch=args.branch)
    manifest = mq.load_manifest(args.context_manifest)
    binding = build_run_binding(args, plan, ckpt_sha256=me.file_sha256(args.ckpt_path),
                                agree_sha256=me.file_sha256(agree_path),
                                model_config_sha256=me.file_sha256(args.model_config))
    # the measurement runs the SAME protocol the rows were scored under, or the
    # deltas it reports are protocol differences wearing a batching label
    gate = op.assert_probe_binding(args.run_dir, binding)
    with open(os.path.join(str(args.run_dir), "merge_report.json")) as handle:
        merge_report = json.load(handle)
    shard_by_room = room_shard_map(merge_report)
    shard_devices = parse_shard_devices(args.shard_device)
    print(f"binding gate passed against {args.run_dir}: {gate['binding_sha256'][:12]}... "
          f"({len(gate['fields_checked'])} fields)")
    print(f"room -> shard from the run's merge report; shard -> device {shard_devices}")

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
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

    def _announce(query):
        matched = query.get("matched")
        print(f"  {query['room_id']} q{query['position']:05d} "
              f"({'same' if query['same_gpu'] else 'CROSS'}-gpu, {query['seconds']:.1f}s)"
              + (f" matched max |delta| {matched['max_abs_delta']:.3g} over "
                 f"{matched['n_candidates']} candidates" if matched else ""), flush=True)

    measured = run_measurement(
        engine, iter_stream_items(loader), manifest["records"], plan, args.run_dir,
        device=args.device, shard_by_room=shard_by_room, shard_devices=shard_devices,
        binding_sha256=gate["binding_sha256"], binding=gate["published"],
        seed=args.seed, num_samples=args.num_samples, noise_policy=args.noise_policy,
        tau=args.tau, tie_source_chunk=args.tie_source_chunk, batch_rows=args.batch_rows,
        source_chunk=args.source_chunk, per_room=args.per_room,
        extra_candidates=args.extra_candidates, matched_per_room=args.matched_per_room,
        selection_seed=args.selection_seed, on_query=_announce)

    deltas = substitution_deltas(measured["substitution_entries"])
    substitution = summarize_substitution(deltas)
    substitution["pairs"] = deltas
    report = build_report(
        measured, device=args.device, run_dir=args.run_dir, shard_devices=shard_devices,
        provenance={"audit_report": str(args.audit_report),
                    "audit_report_sha256": plan.report_sha256,
                    "context_manifest": str(args.context_manifest),
                    "ckpt_path": str(args.ckpt_path),
                    "ckpt_sha256": binding["ckpt_sha256"],
                    "agree_ckpt": agree_path, "agree_ckpt_sha256": agree.ckpt_sha256,
                    "binding_sha256": gate["binding_sha256"]},
        protocol={"seed": int(args.seed), "tau": float(args.tau),
                  "num_samples": int(args.num_samples),
                  "noise_policy": str(args.noise_policy), "steps": int(args.steps),
                  "cfg_scale": float(args.cfg_scale), "batch_rows": int(args.batch_rows),
                  "source_chunk": int(args.source_chunk),
                  "tie_source_chunk": int(args.tie_source_chunk)},
        substitution=substitution)
    published = write_report(args.out_dir, report, matched_arrays=measured["matched_arrays"])
    print(render_markdown(report))
    print(f"measurement -> {published['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
