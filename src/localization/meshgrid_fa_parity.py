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

import torch

from src.localization import meshgrid_engine as me

#: the gates, their artifact stems and their PRE-REGISTERED tolerances (plan r2
#: §3 R2-B). A tolerance is not an argument: a bar the caller picks can be argued down.
GATE_NAMES = {"G1": "fa_split_parity", "G2": "fa_cache_parity", "G3": "fa_production_parity"}
GATE_TOLERANCES = {"G1": 0.0, "G2": 0.0, "G3": 1e-3}
GATE_NOTES = {
    "G1": "GATE 1 (plan r2 §3 R2-B, Codex H3): the five-way split's decomposition against the "
          "released invariant_conditioning at IDENTICAL batching -- every split batch is run at "
          "the released call's batch shape (the query row repeated once per candidate, the "
          "candidates themselves), with --cond-autocast off -- bit-exact on all five conditioner "
          "outputs, their masks and the inputs the DiT consumes. It proves WHICH inputs each "
          "tensor is computed from; gate 2 proves the caches, gate 3 the production batching",
}


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
                                          "bitwise": True, "finite": True, "dtypes": []})
        entry["max_abs_diff"] = max(entry["max_abs_diff"], facts["max_abs_diff"])
        entry["bitwise"] = bool(entry["bitwise"] and facts["bitwise"]
                                and (mask is None or mask["bitwise"]))
        entry["finite"] = bool(entry["finite"] and facts["finite"])
        if mask is not None:
            entry["mask_max_abs_diff"] = max(entry["mask_max_abs_diff"], mask["max_abs_diff"])
        dtypes = "/".join(facts.get("dtype") or [])
        if dtypes and dtypes not in entry["dtypes"]:
            entry["dtypes"].append(dtypes)
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
