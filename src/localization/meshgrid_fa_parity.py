"""exp_22 R2-B -- the frame-average parity gates of the BF mesh pass (plan r2 §3 R2-B / R2-C step 1).

A wrong cache silently changes every BF number, so the five-way split
(``meshgrid_engine.FA_BRANCH_IDS``) is held to the released conditioning by
three gates before the BF pass (Codex plan review H3), each a record-free
diagnostic that writes one evidence JSON for the Planner to commit:

* **G1** ``fa_split_parity`` -- the split's DECOMPOSITION against the released
  ``invariant_conditioning`` at IDENTICAL batching (each split batch run at the
  released call's batch shape), autocast off: bit-exact on the five conditioner
  outputs, their masks and the inputs the DiT consumes.
* **G2** ``fa_cache_parity`` -- cached against uncached memoization at matched
  batching, with candidate-perturbation counter-tests: bit-exact.
* **G3** ``fa_production_parity`` -- production batching (256/16, the registered
  autocast) against a faithful replay of the released path per scored batch:
  per-key max |diff| within the pre-registered 1e-3, plus score and argmax
  stability, full and tail micro-batches covered.

Evidence carries its inputs, per-key digests and deltas, the verdict, the
EXECUTED partition, the git SHA (and whether the tree was tracked-clean), the
host and -- on a GPU -- the CUDA runtime's own card UUID.
"""
import hashlib
import json
import os

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me

#: the gates, their artifact stems and their PRE-REGISTERED tolerances (plan r2
#: §3 R2-B). A tolerance is not an argument: a bar the caller picks can be argued down.
GATE_NAMES = {"G1": "fa_split_parity", "G2": "fa_cache_parity", "G3": "fa_production_parity",
              "G0": "vanilla_replay_regression"}
GATE_TOLERANCES = {"G1": 0.0, "G2": 0.0, "G3": 1e-3, "G0": 0.0}

#: the vanilla no-regression gate's frozen inputs: the observation-bank digest of
#: the 16 registered probe queries (loc_meshgrid_preregistered_digests.json, frozen
#: 2026-08-26 before any quality was read) and the committed canonical P1 off-grid
#: probe report, whose per-query grid row/sidecar digests the replay is held to
FROZEN_OBSERVATION_BANK_SHA256 = (
    "ee2ba80ad4f27b61f4d48e24b4c31cd0d7af5891259aa4f92385cb4400841c6b")
COMMITTED_PROBE_REPORT = os.path.join("worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude",
                                      "offgrid_probe_report.json")
#: binding fields that decide no number: the dump authority governs dumps only
REPLAY_BINDING_EXEMPT = ("dump_cases_sha256",)
GATE_NOTES = {
    "G1": "GATE 1 (plan r2 §3 R2-B, Codex H3): the five-way split's decomposition against the "
          "released invariant_conditioning at IDENTICAL batching -- every split batch is run at "
          "the released call's batch shape (the query row repeated once per candidate, the "
          "candidates themselves), with --cond-autocast off -- bit-exact on all five conditioner "
          "outputs, their masks and the inputs the DiT consumes. It proves WHICH inputs each "
          "tensor is computed from; gate 2 proves the caches, gate 3 the production batching",
    "G2": "GATE 2 (plan r2 §3 R2-B, Codex H3): cached against uncached at MATCHED batching (one "
          "candidate per call on both sides): the production path -- the per-query context "
          "branch, the receiver cache built through ReceiverCache, each scored batch's own "
          "context_poses -- against the released fa_conditioning of each candidate alone, "
          "bit-exact; the four counter-tests (moved candidates, a per-query context_poses "
          "cache, unaveraged ViT ids, swapped candidate rows) must each be DETECTED, or the "
          "comparison could not have failed. The production-batching difference is reported "
          "as informational (gate 3 owns it)",
    "G3": "GATE 3 (plan r2 §3 R2-B, Codex H3): the PRODUCTION path at production batching (the "
          "per-query context at batch 1, the receiver cache over the receiver's whole union at "
          "--source-chunk, context_poses per scored batch of --batch-rows / K candidates, the "
          "registered --cond-autocast) against a faithful replay -- exp_20's released "
          "fa_conditioning over each scored batch's candidates -- with per-key max |diff| <= "
          "1e-3 on the five tensors, their masks and the DiT inputs, the executed partition "
          "equal to the declared one, and score/argmax stability: each probe query generated "
          "under both conditionings with the same keyed noise, |dS| <= SCORE_TOLERANCE and no "
          "argmax moved at any K. Full AND tail scored and source batches must be covered. The "
          "evidence carries no prediction, score or margin -- only deltas, agreement booleans and "
          "at-risk flags -- so no BF localization quality is read before the pass",
    "G0": "VANILLA NO-REGRESSION (plan r2 §3 R2-B): after the R2-B change, the vanilla path "
          "replays each registered probe query of the published P1 run at the row's OWN stamped "
          "batching (the matched-batching tie of the r9s/r9u probe, meshgrid_drift_measurement."
          "measure_matched_query) and must reproduce its float16 sidecar exactly and its float32 "
          "aggregate to exactly 0.0; the published rows must still carry the grid digests the "
          "committed P1 probe report froze, the observation bank must hash to the frozen "
          "ee2ba80a..., and the replay must run the published protocol. Architecture-bound: run "
          "it on the A6000 class the P1 pass ran on",
}

#: the counter-tests of gate 2 and the ids each one must move
COUNTER_KEYS = {"moved_candidates": ("source", "source_vit", "context_poses"),
                "per_query_context_poses": ("context_poses",),
                "unaveraged_vit": ("source_vit", "context_poses_vit"),
                "swapped_candidates": ("source", "source_vit", "context_poses")}


def require_split(engine):
    """The gates compare the ENGAGED split; an engine without one has nothing to gate."""
    if not isinstance(getattr(engine, "split", None), me.FaSplit):
        raise ValueError("the frame-average parity gates need an engine built with the five-way "
                         "split engaged (--cond-method fa_invariant --frame-avg-angles 0 90 180 "
                         "270); this engine has no split")
    return engine.split


def without_target(md):
    """The query's metadata with the held-out target DROPPED, never read."""
    if isinstance(md, me.GuardedMetadata):
        return md.without_target()
    return {key: value for key, value in md.items() if key not in me.GuardedMetadata.BLOCKED}


def _digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype != torch.bfloat16:
        return me.tensor_digest(tensor)
    header = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
    return hashlib.sha256(header.encode("utf-8")
                          + tensor.view(torch.int16).numpy().tobytes()).hexdigest()


def tensor_pair(left, right, tolerance):
    """Exact facts about two tensors: shapes, dtypes, digests, finiteness, max |diff|."""
    a = torch.as_tensor(left).detach().cpu()
    b = torch.as_tensor(right).detach().cpu()
    facts = {"shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)],
             "digest": [_digest(a), _digest(b)],
             "finite": bool(torch.isfinite(a.double()).all() and torch.isfinite(b.double()).all()),
             "reasons": []}
    if a.shape != b.shape:
        facts.update(max_abs_diff=float("inf"), bitwise=False)
        facts["reasons"].append(f"shape {list(a.shape)} vs {list(b.shape)}")
    else:
        facts["max_abs_diff"] = (float((a.double() - b.double()).abs().max())
                                 if a.numel() else 0.0)
        facts["bitwise"] = bool(a.dtype == b.dtype and torch.equal(a, b))
    if a.dtype != b.dtype:
        facts["reasons"].append(f"dtype {a.dtype} vs {b.dtype}: equal values in different "
                                "storage are not parity")
    if not facts["finite"]:
        facts["reasons"].append("a non-finite value is present; a verdict over it is meaningless")
    if facts["max_abs_diff"] > float(tolerance):
        facts["reasons"].append(f"max |diff| {facts['max_abs_diff']:.3g} > {tolerance}")
    return facts


def _mask_pair(left, right, tolerance):
    if left is not None and right is not None:
        return tensor_pair(left, right, tolerance)
    same = left is None and right is None
    return {"none": [left is None, right is None], "bitwise": same,
            "max_abs_diff": 0.0 if same else float("inf"), "finite": True,
            "reasons": [] if same else ["a mask is present on one side only"]}


def compare_conditioning(left, right, tolerance):
    """``{id: {tensor, mask}}`` facts for two conditioner-output blocks, and the reasons."""
    out, reasons = {}, []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            reasons.append(f"{key}: present on one side only")
            continue
        out[key] = {"tensor": tensor_pair(left[key][0], right[key][0], tolerance),
                    "mask": _mask_pair(left[key][1], right[key][1], tolerance)}
        for kind in ("tensor", "mask"):
            reasons += [f"{key}.{kind}: {why}" for why in out[key][kind]["reasons"]]
    return out, reasons


def compare_inputs(left, right, tolerance):
    """The same facts for the inputs the DiT consumes (``cond_inputs_fn``'s dicts)."""
    out, reasons = {}, []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            out[name] = {"bitwise": False, "max_abs_diff": float("inf"), "finite": True,
                         "reasons": ["present on one side only"]}
        else:
            out[name] = tensor_pair(a, b, tolerance)
        reasons += [f"dit_inputs.{name}: {why}" for why in out[name]["reasons"]]
    return out, reasons


def fold_per_key(per_key, conditioning, inputs):
    """Accumulate one chunk's facts into the gate's per-key summary (max over chunks)."""
    rows = [(key, entry["tensor"], entry["mask"]) for key, entry in conditioning.items()]
    rows += [(f"dit_inputs.{name}", facts, None) for name, facts in inputs.items()]
    for name, facts, mask in rows:
        entry = per_key.setdefault(name, {"max_abs_diff": 0.0, "mask_max_abs_diff": 0.0,
                                          "bitwise": True, "finite": True, "dtypes": [],
                                          "shapes": []})
        entry["max_abs_diff"] = max(entry["max_abs_diff"], facts["max_abs_diff"])
        entry["bitwise"] = bool(entry["bitwise"] and facts["bitwise"]
                                and (mask is None or mask["bitwise"]))
        entry["finite"] = bool(entry["finite"] and facts["finite"])
        if mask is not None:
            entry["mask_max_abs_diff"] = max(entry["mask_max_abs_diff"], mask["max_abs_diff"])
        for field, value in (("dtypes", "/".join(facts.get("dtype") or [])),
                             ("shapes", "/".join(str(shape) for shape in facts.get("shape") or []))):
            if value and value not in entry[field]:
                entry[field].append(value)
    return per_key


def coverage(sizes, micro):
    """How many full and tail micro-batches a gate exercised; both are required."""
    full = sum(1 for size in sizes if size == micro)
    tail = sum(1 for size in sizes if size < micro)
    return {"full": full, "tail": tail, "complete": bool(full and tail)}


def gate_report(gate, reasons, **fields):
    """The verdict block every gate shares."""
    return {"gate": gate, "name": GATE_NAMES[gate], "verdict": "FAIL" if reasons else "PASS",
            "match": not reasons, "reasons": list(reasons),
            "tolerance": GATE_TOLERANCES[gate], "note": GATE_NOTES[gate], **fields}


def query_inputs(query, n, **extra):
    return {"query_id": query.query_id, "room_id": query.room_id,
            "receiver_id": query.receiver_id, "position": int(query.position),
            "n_candidates": int(n),
            "candidate_indices": [int(i) for i in query.candidate_indices[:n]], **extra}


def split_parity(engine, query, md, *, micro_batch, n_candidates=None):
    """G1 -- the split's decomposition against the released path at IDENTICAL batching.

    Each chunk of ``micro_batch`` candidates is conditioned twice: by exp_20's
    ``fa_conditioning`` (``invariant_conditioning`` at cap = batch, the released
    path) and by the split's three batches, each run at THAT call's batch shape.
    """
    from src.localization.candidates import candidate_metadata
    from src.localization.crossarm import fa_conditioning

    split = require_split(engine)
    micro = max(1, int(micro_batch))
    wanted = micro + max(1, micro // 2) if n_candidates is None else int(n_candidates)
    n = max(1, min(query.n_candidates, wanted))
    positions = me.candidate_inputs(engine, query, md)["positions"][:n]
    free = without_target(md)
    query_row = {key: md[key] for key in me.FA_QUERY_KEYS}
    recorder, chunks, reasons, per_key = me.FaPartitionRecorder(), [], [], {}
    for start in range(0, n, micro):
        chunk = positions[start:start + micro]
        record = {}
        released = fa_conditioning(engine.conditioner,
                                   [candidate_metadata(free, position) for position in chunk],
                                   engine.device, split.angles, record=record)
        batches = {"context_vit": [query_row] * len(chunk),
                   "context_scored_batch": [candidate_metadata(
                       {"context_poses": md["context_poses"]}, position) for position in chunk],
                   "source_cache": [candidate_metadata({"depth": md["depth"]}, position)
                                    for position in chunk]}
        parts = {}
        for branch, rows in batches.items():
            parts.update(me.fa_branch(engine.conditioner, branch, rows, engine.device,
                                      split.angles, recorder, configured=micro))
        conditioning, why = compare_conditioning(parts, released, GATE_TOLERANCES["G1"])
        inputs, why_inputs = compare_inputs(engine.cond_inputs_fn(parts),
                                            engine.cond_inputs_fn(released),
                                            GATE_TOLERANCES["G1"])
        reasons += [f"chunk@{start}: {item}" for item in why + why_inputs]
        fold_per_key(per_key, conditioning, inputs)
        chunks.append({"start": start, "n": len(chunk), "full": len(chunk) == micro,
                       "released_partition": record, "conditioning": conditioning,
                       "dit_inputs": inputs})
    covered = coverage([chunk["n"] for chunk in chunks], micro)
    if not covered["complete"]:
        reasons.append(f"coverage: {covered['full']} full and {covered['tail']} tail "
                       "micro-batch(es); a gate covers both (raise --parity-candidates)")
    return gate_report("G1", reasons, chunks=chunks, per_key=per_key, coverage=covered,
                       inputs=query_inputs(query, n, micro_batch=micro,
                                           angles=list(split.angles)),
                       executed_partition=recorder.report())


def finalize_evidence(report, device, environment=None, runtime=None, inputs=None):
    """The gate's evidence: its report plus the run's provenance (record-free diagnostic)."""
    from src.localization import meshgrid_offgrid_probe as mop

    environment = mop.current_environment() if environment is None else environment
    on_gpu = str(device).startswith("cuda")
    if on_gpu and runtime is None:
        runtime = mop.runtime_gpu_uuid(device)
    dirty = environment.get("git_status_dirty")
    evidence = dict(report)
    evidence["inputs"] = dict(report.get("inputs") or {}, **(inputs or {}))
    evidence["provenance"] = {
        "git_sha": environment.get("git_sha"),
        "tracked_clean": None if dirty is None else not bool(dirty),
        "hostname": environment.get("hostname"), "device": str(device),
        "gpu_uuid": (runtime or {}).get("uuid") if on_gpu else None,
        "gpu_runtime": runtime if on_gpu else None,
        "torch": torch.__version__, "record_free": True}
    evidence["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return evidence


def write_evidence(out_dir, evidence):
    """Publish one gate's evidence atomically as ``<out_dir>/<gate name>.json``."""
    os.makedirs(str(out_dir), exist_ok=True)
    path = os.path.join(str(out_dir), f"{evidence['name']}.json")
    return me.write_json(path, evidence)


def gate_engine(engine, angles=None, record=True):
    """A shallow view of ``engine`` with its OWN split, so a gate never writes into a pass."""
    import copy

    view = copy.copy(engine)
    view.split = me.FaSplit(angles=tuple(engine.split.angles if angles is None else angles),
                            recorder=me.FaPartitionRecorder() if record else None)
    return view


def concat_blocks(blocks):
    """Per-batch ``{id: [tensor, mask]}`` blocks concatenated along the candidate axis."""
    out = {}
    for key in blocks[0]:
        masks = [block[key][1] for block in blocks]
        out[key] = [torch.cat([block[key][0] for block in blocks], dim=0),
                    None if masks[0] is None else torch.cat(masks, dim=0)]
    return out


def production_conditioning(view, query, md, indices, positions, *, source_chunk, micro):
    """The PRODUCTION code path at the given batching, one row per candidate: the per-query
    context, the receiver cache (``ReceiverCache.build``) and ``scored_conditioning``."""
    context = me.query_context(view, md)
    cache = me.ReceiverCache.build(view.conditioner, query.receiver_id, {"depth": md["depth"]},
                                   indices, positions, view.device, chunk=source_chunk,
                                   fa=view.split)
    candidate = {"context_poses": md["context_poses"], "positions": positions}
    rows_all = cache.rows_for(indices)
    return concat_blocks([me.scored_conditioning(view, context, cache, rows_all[start:start + micro],
                                                 candidate, start=start, stop=start + micro,
                                                 num_samples=1, configured=micro)
                          for start in range(0, len(indices), micro)])


def released_conditioning(engine, md, positions, angles, per_candidate=False, records=None):
    """exp_20's ``fa_conditioning`` (the released path) over the candidates: all in one
    call, or each candidate alone (``per_candidate``); ``records`` collects each call's
    executed orbit partition."""
    from src.localization.candidates import candidate_metadata
    from src.localization.crossarm import fa_conditioning

    free = without_target(md)
    groups = [[p] for p in positions] if per_candidate else [list(positions)]
    records = [] if records is None else records
    blocks = []
    for group in groups:
        records.append({})
        blocks.append(fa_conditioning(engine.conditioner,
                                      [candidate_metadata(free, p) for p in group],
                                      engine.device, angles, record=records[-1]))
    return concat_blocks(blocks)


def cache_parity(engine, query, md, *, n_candidates=None, source_chunk=me.SOURCE_CHUNK,
                 micro_batch=1):
    """G2 -- the caches are an exact memoization at matched batching, and the check bites."""
    split = require_split(engine)
    n = max(2, min(query.n_candidates,
                   2 * int(source_chunk) if n_candidates is None else int(n_candidates)))
    indices = [int(i) for i in query.candidate_indices[:n]]
    positions = me.candidate_inputs(engine, query, md)["positions"][:n]
    uncached = released_conditioning(engine, md, positions, split.angles, per_candidate=True)
    memo_view = gate_engine(engine)
    cached = production_conditioning(memo_view, query, md, indices, positions, source_chunk=1,
                                     micro=1)
    conditioning, why = compare_conditioning(cached, uncached, GATE_TOLERANCES["G2"])
    inputs, why_inputs = compare_inputs(memo_view.cond_inputs_fn(cached),
                                        engine.cond_inputs_fn(uncached), GATE_TOLERANCES["G2"])
    reasons = [f"memoization: {item}" for item in why + why_inputs]

    blind = gate_engine(engine, record=False)
    mutants = {
        "moved_candidates": production_conditioning(blind, query, md, indices, positions + 1.0,
                                                    source_chunk=1, micro=1),
        "per_query_context_poses": dict(cached, context_poses=[
            cached["context_poses"][0][:1].expand_as(cached["context_poses"][0]),
            cached["context_poses"][1]]),
        "unaveraged_vit": production_conditioning(gate_engine(engine, angles=(0.0,), record=False),
                                                  query, md, indices, positions, source_chunk=1,
                                                  micro=1),
        "swapped_candidates": {key: [value[0].flip(0), value[1]]
                               for key, value in cached.items()}}
    counters = {}
    for name, mutant in mutants.items():
        moved = {key: tensor_pair(mutant[key][0], uncached[key][0], 0.0)["max_abs_diff"]
                 for key in COUNTER_KEYS[name]}
        counters[name] = {"keys": moved, "detected": all(value > 0.0 for value in moved.values())}
        if not counters[name]["detected"]:
            reasons.append(f"counter-test {name} did not bite on {sorted(COUNTER_KEYS[name])}: "
                           "a comparison that cannot fail proves nothing")

    batched_view = gate_engine(engine)
    production = production_conditioning(batched_view, query, md, indices, positions,
                                         source_chunk=source_chunk, micro=int(micro_batch))
    batched, _ = compare_conditioning(production, released_conditioning(
        engine, md, positions, split.angles), float("inf"))
    per_key = fold_per_key({}, conditioning, inputs)
    return gate_report(
        "G2", reasons, per_key=per_key, counter_tests=counters,
        memoization={"match": not why and not why_inputs, "conditioning": conditioning,
                     "dit_inputs": inputs},
        batched={"informational": True, "keys": batched,
                 "max_abs_diff": max(entry["tensor"]["max_abs_diff"]
                                     for entry in batched.values()),
                 "executed_partition": batched_view.split.recorder.report(),
                 "note": "production chunks against ONE released call over all candidates: a "
                         "nonzero value is the backbone's own batch-shape nondeterminism; gate 3 "
                         "owns the production comparison"},
        inputs=query_inputs(query, n, source_chunk=int(source_chunk),
                            micro_batch=int(micro_batch), angles=list(split.angles)),
        executed_partition=memo_view.split.recorder.report())


def probe_case(room_plan, query, md, obs_wav):
    """One gate-3 case: the query, its metadata and observation, and its receiver's WHOLE
    candidate union in camera frame -- the source cache the production pass builds."""
    group = next(group for group in me.receiver_groups(room_plan)
                 if any(entry.query_id == query.query_id for entry in group.queries))
    union = [int(index) for index in group.union]
    positions = (np.asarray(room_plan.base, dtype=np.float64)[np.asarray(union, dtype=np.int64)]
                 - np.asarray(group.receiver_xyz, dtype=np.float64))
    return {"query": query, "md": md, "obs_wav": obs_wav, "union": union,
            "union_positions": positions}


def _score_deltas(production, replayed, indices, tau, prefixes):
    """Per-K score and argmax stability -- deltas and booleans, never a score itself."""
    out, why = {}, []
    left = me.nested_scores(production, tau=tau, prefixes=prefixes)
    right = me.nested_scores(replayed, tau=tau, prefixes=prefixes)
    for k in left:
        a, b = left[k]["scores"], right[k]["scores"]
        delta = float((a.double() - b.double()).abs().max())
        equal = me.argmax_by_global_index(a, indices) == me.argmax_by_global_index(b, indices)
        out[str(k)] = {"max_abs_delta": delta, "argmax_equal": bool(equal),
                       "at_risk": not me.is_argmax_stable(me.top1_margin(a))}
        if delta > me.SCORE_TOLERANCE:
            why.append(f"K={k}: score |diff| {delta:.3g} > SCORE_TOLERANCE {me.SCORE_TOLERANCE}")
        if not equal:
            why.append(f"K={k}: the argmax moved between production and replay")
    return out, why


def _chain(chains, conditioning, inputs):
    """Extend each key's ordered per-side sha256 chain with this chunk's tensor digests."""
    rows = [(key, entry["tensor"]) for key, entry in conditioning.items()]
    rows += [(f"dit_inputs.{name}", facts) for name, facts in inputs.items()]
    for name, facts in rows:
        sides = chains.setdefault(name, [hashlib.sha256(), hashlib.sha256()])
        for side, digest in zip(sides, facts.get("digest") or ["", ""]):
            side.update(str(digest).encode("utf-8"))


def _production_query(view, engine, case, *, micro, batch_rows, source_chunk, num_samples,
                      seed, noise_policy, tau, prefixes, per_key, sizes, chains):
    from src.localization.scoring import cosine_sims

    query, md = case["query"], case["md"]
    obs = engine.embedder(torch.as_tensor(case["obs_wav"]).to(engine.device))[0].float().cpu()
    context = me.query_context(view, md)
    cache = me.ReceiverCache.build(view.conditioner, query.receiver_id, {"depth": md["depth"]},
                                   case["union"], case["union_positions"], view.device,
                                   chunk=source_chunk, fa=view.split)
    candidate = me.candidate_inputs(view, query, md)
    production = me._score_one_query(                                 # noqa: SLF001 -- THE path
        view, query, context, cache, obs, seed=seed, num_samples=num_samples,
        noise_policy=noise_policy, batch_rows=batch_rows, timer=me._Timer(view.device),
        candidate=candidate)
    indices = [int(i) for i in query.candidate_indices]
    rows_all, parts, why, chunks = cache.rows_for(indices), [], [], []
    for start in range(0, len(indices), micro):
        slice_indices = indices[start:start + micro]
        produced = me.scored_conditioning(view, context, cache, rows_all[start:start + micro],
                                          candidate, start=start, stop=start + micro,
                                          num_samples=1, configured=micro, record=False)
        replayed_partition = []
        replay = released_conditioning(engine, md, candidate["positions"][start:start + micro],
                                       view.split.angles, records=replayed_partition)
        conditioning, w_cond = compare_conditioning(produced, replay, GATE_TOLERANCES["G3"])
        inputs, w_inputs = compare_inputs(engine.cond_inputs_fn(produced),
                                          engine.cond_inputs_fn(replay), GATE_TOLERANCES["G3"])
        fold_per_key(per_key, conditioning, inputs)
        _chain(chains, conditioning, inputs)
        why += [f"chunk@{start}: {item}" for item in w_cond + w_inputs]
        sizes["scored"].append(len(slice_indices))
        chunks.append({"start": start, "n": len(slice_indices),
                       "max_abs_diff": max(e["tensor"]["max_abs_diff"]
                                           for e in conditioning.values()),
                       "bitwise": all(e["tensor"]["bitwise"] and e["mask"]["bitwise"]
                                      for e in conditioning.values()),
                       "replay_partition": replayed_partition[0].get("partition")})
        # the replay's own generation: the production chunking and the same keyed noise
        noise = me.noise_block(seed, query.query_id, slice_indices, num_samples,
                               engine.latent_shape, policy=noise_policy, device=engine.device)
        rows = torch.arange(len(slice_indices)).repeat_interleave(int(num_samples))
        merged = {key: me._select(entry, rows.to(entry[0].device))   # noqa: SLF001
                  for key, entry in replay.items()}
        wavs = engine.decoder(engine.sampler(noise, engine.cond_inputs_fn(merged))).clamp(-1.0, 1.0)
        parts.append(cosine_sims(obs, engine.embedder(wavs).reshape(
            len(slice_indices), int(num_samples), -1)).float().cpu())
    union = len(case["union"])
    sizes["source"] += [min(int(source_chunk), union - start)
                        for start in range(0, union, int(source_chunk))]
    scores, why_scores = _score_deltas(production, torch.cat(parts, dim=0), indices, tau,
                                       prefixes)
    return ({"inputs": query_inputs(query, len(indices), n_union=union), "chunks": chunks,
             "score_stability": scores}, why + why_scores)


def production_parity(engine, cases, *, batch_rows, source_chunk, num_samples=me.NUM_SAMPLES,
                      seed=me.SEED, noise_policy=me.NOISE_KEY_POLICY, tau=me.TAU,
                      prefixes=me.K_PREFIXES, declared_partition=None):
    """G3 -- production batching against a faithful replay, with score/argmax stability."""
    split = require_split(engine)
    view = gate_engine(engine)
    micro = max(1, int(batch_rows) // int(num_samples))
    entries, per_key, reasons, sizes = [], {}, [], {"scored": [], "source": []}
    chains = {}
    for case in cases:
        entry, why = _production_query(view, engine, case, micro=micro, batch_rows=batch_rows,
                                       source_chunk=source_chunk, num_samples=num_samples,
                                       seed=seed, noise_policy=noise_policy, tau=tau,
                                       prefixes=prefixes, per_key=per_key, sizes=sizes,
                                       chains=chains)
        entries.append(entry)
        reasons += [f"{case['query'].query_id}: {item}" for item in why]
    covered = {"scored": coverage(sizes["scored"], micro),
               "source": coverage(sizes["source"], int(source_chunk))}
    reasons += [f"coverage: {name} batches {facts['full']} full / {facts['tail']} tail; a gate "
                "covers both" for name, facts in covered.items() if not facts["complete"]]
    executed = view.split.recorder.report()
    if declared_partition is not None:
        reasons += [f"executed partition differs from the declared one at {path}"
                    for path in me.fa_partition_differences(declared_partition,
                                                            executed["executed"])]
        reasons += [f"executed partition violation: {item}" for item in executed["violations"]]
    by_k = {}
    for entry in entries:
        for k, facts in entry["score_stability"].items():
            block = by_k.setdefault(k, {"max_abs_delta": 0.0, "n_queries": 0,
                                        "n_argmax_equal": 0, "n_at_risk": 0})
            block["max_abs_delta"] = max(block["max_abs_delta"], facts["max_abs_delta"])
            block["n_queries"] += 1
            block["n_argmax_equal"] += int(facts["argmax_equal"])
            block["n_at_risk"] += int(facts["at_risk"])
    return gate_report(
        "G3", reasons, queries=entries, per_key=per_key, coverage=covered, score_stability=by_k,
        digest_chains={name: [side.hexdigest() for side in sides]
                       for name, sides in chains.items()},
        score_tolerance=me.SCORE_TOLERANCE, declared_partition=declared_partition,
        inputs={"queries": [entry["inputs"]["query_id"] for entry in entries],
                "batch_rows": int(batch_rows), "source_chunk": int(source_chunk),
                "num_samples": int(num_samples), "micro_batch": micro, "seed": int(seed),
                "noise_policy": str(noise_policy), "tau": float(tau),
                "prefixes": [int(k) for k in prefixes], "angles": list(split.angles)},
        executed_partition=executed)


def frozen_grid_digests(probe_report_path=COMMITTED_PROBE_REPORT):
    """``{query_id: {row_sha256, sims_sha256}}`` the committed P1 probe report froze."""
    with open(str(probe_report_path)) as handle:
        report = json.load(handle)
    return {str(record["query_id"]): {"row_sha256": record["grid_row_sha256"],
                                      "sims_sha256": record["grid_sims_sha256"]}
            for record in report["records"]}


def vanilla_replay_regression(engine, cases, run_dir, *, frozen_digests, observation_bank_sha256,
                              expected_observation_bank_sha256=FROZEN_OBSERVATION_BANK_SHA256,
                              run_binding=None, seed=me.SEED, num_samples=me.NUM_SAMPLES,
                              noise_policy=me.NOISE_KEY_POLICY, tau=me.TAU):
    """G0 -- the vanilla path still reproduces the published P1 rows of the probe queries."""
    from src.localization import meshgrid_drift_measurement as dm
    from src.localization import meshgrid_offgrid_probe as mop
    from src.localization import meshgrid_report as mr

    if getattr(engine, "split", None) is not None or str(engine.cond_method) != "vanilla":
        raise ValueError("the no-regression replay gates the vanilla path; this engine runs the fa "
                         "split, which gates 1-3 cover")
    published, published_sha = mr.load_published_binding(run_dir)
    reasons = []
    if str(observation_bank_sha256) != str(expected_observation_bank_sha256):
        reasons.append(f"the observation bank hashes to {str(observation_bank_sha256)[:12]}..., "
                       f"not the frozen {str(expected_observation_bank_sha256)[:12]}...")
    if run_binding is not None:
        moved = [field for field in me.RUN_BINDING_FIELDS if field not in REPLAY_BINDING_EXEMPT
                 and run_binding.get(field) != published.get(field)]
        reasons += [f"the replay does not run the published protocol: {field} differs"
                    for field in moved]
    entries = []
    for case in cases:
        query, md = case["query"], case["md"]
        row = mop.load_grid_row(run_dir, query, binding_sha256=published_sha, binding=published)
        stored = {"row_sha256": row.get("row_sha256"), "sims_sha256": row.get("sims_sha256")}
        batch_rows, source_chunk = mop.row_batching(row)
        obs = engine.embedder(torch.as_tensor(case["obs_wav"]).to(engine.device))[0].float().cpu()
        matched, _per_candidate, _deltas = dm.measure_matched_query(
            engine, query, md, me.query_context(engine, md), query.receiver_id, case["union"],
            case["union_positions"], row, row["_sims"], obs, seed=seed, num_samples=num_samples,
            noise_policy=noise_policy, batch_rows=batch_rows, source_chunk=source_chunk, tau=tau)
        entry = {"query_id": query.query_id, "room_id": query.room_id,
                 "frozen_digests_equal": frozen_digests.get(query.query_id) == stored,
                 "published_digests": stored, "matched": matched}
        if not entry["frozen_digests_equal"]:
            reasons.append(f"{query.query_id}: the published row no longer carries the frozen "
                           "grid digests of the committed P1 probe report")
        if not matched["float16_bit_exact"]:
            reasons.append(f"{query.query_id}: {matched['n_float16_mismatch']} sidecar cells are "
                           "not reproduced float16-exactly")
        if matched["aggregate"]["max_abs_delta"] != 0.0:
            reasons.append(f"{query.query_id}: the float32 aggregate moved by "
                           f"{matched['aggregate']['max_abs_delta']:.3g}")
        entries.append(entry)
    if not entries:
        reasons.append("no registered probe query was replayed")
    return gate_report("G0", reasons, queries=entries,
                       inputs={"run_dir": str(run_dir), "published_binding_sha256": published_sha,
                               "observation_bank_sha256": str(observation_bank_sha256),
                               "expected_observation_bank_sha256":
                                   str(expected_observation_bank_sha256),
                               "queries": [entry["query_id"] for entry in entries]},
                       executed_partition=None)
