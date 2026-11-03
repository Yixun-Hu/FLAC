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
* **G3** ``fa_production_parity`` -- production batching (256/16) against a
  faithful replay of the released path per scored batch, with autocast OFF (fp32;
  plan r2 amendment A4 i): per-key max |diff| within the pre-registered 1e-3,
  plus score and argmax stability, full and tail micro-batches covered.
* **G3D** ``fa_production_drift`` -- the same comparison under the resolved
  production autocast (``default``: float16 on CUDA), PUBLISHED AS A DISCLOSED
  DRIFT MEASUREMENT with no pass/fail verdict (A4 ii).

Evidence carries its inputs, per-key digests and deltas, the verdict, the
EXECUTED partition, the git SHA (and whether the tree was tracked-clean), the
host and -- on a GPU -- the CUDA runtime's own card UUID.
"""
import hashlib
import json
import os
import re

from datetime import datetime, timezone

import numpy as np
import torch

from src.localization import meshgrid_engine as me

#: the gates, their artifact stems and their PRE-REGISTERED tolerances (plan r2
#: §3 R2-B). A tolerance is not an argument: a bar the caller picks can be argued down.
GATE_NAMES = {"G1": "fa_split_parity", "G2": "fa_cache_parity", "G3": "fa_production_parity",
              "G3D": "fa_production_drift", "G0": "vanilla_replay_regression",
              "GV": "vanilla_cache_parity"}
#: ``None``: a measurement, not a gate -- no tolerance exists to pass or fail (A4 ii)
GATE_TOLERANCES = {"G1": 0.0, "G2": 0.0, "G3": 1e-3, "G3D": None, "G0": 0.0, "GV": 0.0}
#: each gate's registered arm(s) and the ONLY binding deviations its registration admits
#: (Codex rB finding 3; plan r2 amendment A4 v): G1 and the G3 logic gate run autocast off,
#: nothing else may move. GV is the vanilla cache parity run AS a gate (--expect-arm YAW)
GATE_IDENTITY = {"G0": (("P1",), {}), "G1": (("BF",), {"cond_autocast": "off"}),
                 "G2": (("BF",), {}), "G3": (("BF",), {"cond_autocast": "off"}),
                 "G3D": (("BF",), {}), "GV": (("P1", "YAW"), {})}
#: the autocast each gate is registered at (A4 i/ii): G1 and the G3 logic gate run with
#: autocast OFF; the drift measurement runs the production flag, resolved by the device
GATE_AUTOCAST = {"G1": ("off", False), "G3": ("off", False), "G3D": ("default", True)}
#: the engine's own vanilla batching caveat, the drift measurement's published context
VANILLA_BATCHING_CAVEAT_MAX_ABS_DIFF = 3.9e-3

#: the vanilla no-regression gate's frozen inputs: the observation-bank digest of
#: the 16 registered probe queries (loc_meshgrid_preregistered_digests.json, frozen
#: 2026-08-26 before any quality was read) and the committed canonical P1 off-grid
#: probe report, whose per-query grid row/sidecar digests the replay is held to
FROZEN_OBSERVATION_BANK_SHA256 = (
    "ee2ba80ad4f27b61f4d48e24b4c31cd0d7af5891259aa4f92385cb4400841c6b")
COMMITTED_PROBE_REPORT = os.path.join("worklog", "worklog_yixun", "exp_22_loc_meshgrid_claude",
                                      "offgrid_probe_report.json")
#: the committed report's BYTES, pinned (Codex rB finding 2; plan r2 amendment A4 v): the
#: frozen grid anchor is not caller-replaceable -- a relocated copy is admitted only if it
#: hashes to this, and anything else is refused before a replay starts
COMMITTED_PROBE_REPORT_SHA256 = (
    "8c5fd5e8b35e380fe6bc66a328eb8353ce5ef75b56fc02a587ddb5c85957ffd3")
#: ... and its CONTENT, registered beside its bytes (Codex rB-F2 finding 2): the canonical
#: digest (``crossarm.canonical_sha256``, as ``load_probe_anchor`` computes it) of its ORDERED
#: probe query ids and of its frozen grid-digest map ``{query_id: {row_sha256, sims_sha256}}``,
#: computed from the committed file -- an evidence's anchor block is authenticated by these,
#: not by the sha string it claims (src/tests re-derives both from the file's bytes)
COMMITTED_PROBE_QUERY_IDS_SHA256 = (
    "4f5743a8c7dcead5486281560a60e709ff93e44e315acf16722962790ab82382")
COMMITTED_PROBE_FROZEN_DIGESTS_SHA256 = (
    "9ecef47d94165e375c87c98597815a8b0c1ef3b8e83fe75112519ab20f31011c")
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
    "G3": "GATE 3 (plan r2 §3 R2-B, Codex H3; amendment A4 i): the PRODUCTION path at "
          "production batching (the per-query context at batch 1, the receiver cache over the "
          "receiver's whole union at --source-chunk, context_poses per scored batch of "
          "--batch-rows / K candidates) with --cond-autocast off (fp32) against a faithful "
          "replay -- exp_20's released "
          "fa_conditioning over each scored batch's candidates -- with per-key max |diff| <= "
          "1e-3 on the five tensors, their masks and the DiT inputs, the executed partition "
          "equal to the declared one, and score/argmax stability: each probe query generated "
          "under both conditionings with the same keyed noise, |dS| <= SCORE_TOLERANCE and no "
          "argmax moved at any K. Full AND tail scored and source batches must be covered. The "
          "evidence carries no prediction, score or margin -- only deltas, agreement booleans and "
          "at-risk flags -- so no BF localization quality is read before the pass. The probe set "
          "is anchored by the pinned committed probe report, and every clause re-verifies from "
          "this JSON alone (meshgrid_fa_parity.reverify_gate_evidence; Codex rB-F1 finding 2)",
    "G3D": "DRIFT MEASUREMENT (plan r2 amendment A4 ii): gate 3's comparison run under the "
           "resolved PRODUCTION autocast (--cond-autocast default: float16 on CUDA) and published "
           "with NO pass/fail verdict -- per-key deltas, score deltas, argmax agreement -- beside "
           "the engine's own vanilla batching caveat (3.9e-3 between a batch-1 and an "
           "8-candidate context call under fp16) as context. No fp16 tolerance is chosen before "
           "or after the BF values are seen: tolerance, score_tolerance and match are null (A6 "
           "iii). 'complete' says only that the comparison's structure held (every probe query "
           "of the pinned anchor, full and tail batches, the declared partition); it re-verifies "
           "from this JSON alone (meshgrid_fa_parity.reverify_gate_evidence)",
    "GV": "VANILLA CACHE PARITY AS A GATE (plan r2 §3 R2-C step 1; Codex rB finding 3): the "
          "§1.5 cached-vs-uncached memoization proof and its counter-test on the first registered "
          "query, run under the arm --expect-arm names (YAW for R2) and held to that arm's "
          "registered identity and to captured provenance",
    "G0": "VANILLA NO-REGRESSION (plan r2 §3 R2-B): after the R2-B change, the vanilla path "
          "replays each registered probe query of the published P1 run at the row's OWN stamped "
          "batching (the matched-batching tie of the r9s/r9u probe, meshgrid_drift_measurement."
          "measure_matched_query) and must reproduce its float16 sidecar exactly and its float32 "
          "aggregate to exactly 0.0; the published rows must still carry the grid digests the "
          "committed P1 probe report froze, the observation bank must hash to the frozen "
          "ee2ba80a..., and the replay must run the published protocol. Architecture-bound: run "
          "it on the A6000 class the P1 pass ran on",
}

#: how a reader re-verifies a gate's evidence (Codex rB-F1 finding 2)
REVERIFY_NOTE = (
    "Every PASS clause of this evidence re-verifies from this JSON alone: "
    "src.localization.meshgrid_fa_parity.reverify_gate_evidence(<this JSON>) returns [] for a "
    "clean PASS (for the G3D measurement: a complete one) and names every clause that does not "
    "hold otherwise; tolerances, the pinned anchor and the registered batching come from the "
    "code, never from the evidence")

#: the counter-tests of gate 2 and the ids each one must move
COUNTER_KEYS = {"moved_candidates": ("source", "source_vit", "context_poses"),
                "per_query_context_poses": ("context_poses",),
                "unaveraged_vit": ("source_vit", "context_poses_vit"),
                "swapped_candidates": ("source", "source_vit", "context_poses")}


def require_split(engine):
    """The gates compare the ENGAGED split; an engine without one -- or with a malformed
    one (Codex rB NIT 8) -- has nothing to gate."""
    if not isinstance(getattr(engine, "split", None), me.FaSplit):
        raise ValueError("the frame-average parity gates need an engine built with the five-way "
                         "split engaged (--cond-method fa_invariant --frame-avg-angles 0 90 180 "
                         "270); this engine has no split")
    me.assert_cacheable(getattr(engine, "cond_method", None), engine.split)
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


def _fmt(value):
    return f"{value:.3g}" if isinstance(value, (int, float)) else repr(value)


def pair_reasons(facts, tolerance):
    """The tensor clause over a pair's SERIALIZED facts -- one predicate, applied by the gates
    and again by :func:`reverify_gate_evidence`: one shape, one dtype, finite values and (for
    a gate) max |diff| within the registered tolerance (a NaN is never within it)."""
    shape, dtype = facts.get("shape") or [None, None], facts.get("dtype") or [None, None]
    reasons = [f"shape {shape[0]} vs {shape[1]}"] if shape[0] != shape[1] else []
    if dtype[0] != dtype[1]:
        reasons.append(f"dtype {dtype[0]} vs {dtype[1]}: equal values in different storage are "
                       "not parity")
    if facts.get("finite") is not True:
        reasons.append("a non-finite value is present; a verdict over it is meaningless")
    found = facts.get("max_abs_diff")
    if tolerance is not None and not (isinstance(found, (int, float))
                                      and found <= float(tolerance)):
        reasons.append(f"max |diff| {_fmt(found)} > {tolerance}")
    return reasons


def mask_reasons(facts, tolerance):
    """The mask clause: present on both sides or on neither, then the tensor clause."""
    if "none" in facts:
        return [] if facts.get("none") == [True, True] else ["a mask is present on one side only"]
    return pair_reasons(facts, tolerance)


def tensor_pair(left, right, tolerance):
    """Exact facts about two tensors: shapes, dtypes, digests, finiteness, max |diff|."""
    a = torch.as_tensor(left).detach().cpu()
    b = torch.as_tensor(right).detach().cpu()
    facts = {"shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)],
             "digest": [_digest(a), _digest(b)],
             "finite": bool(torch.isfinite(a.double()).all() and torch.isfinite(b.double()).all())}
    if a.shape != b.shape:
        facts.update(max_abs_diff=float("inf"), bitwise=False)
    else:
        facts["max_abs_diff"] = (float((a.double() - b.double()).abs().max())
                                 if a.numel() else 0.0)
        facts["bitwise"] = bool(a.dtype == b.dtype and torch.equal(a, b))
    facts["reasons"] = pair_reasons(facts, tolerance)
    return facts


def _mask_pair(left, right, tolerance):
    if left is not None and right is not None:
        return tensor_pair(left, right, tolerance)
    same = left is None and right is None
    sides = [None if mask is None else torch.as_tensor(mask).detach().cpu()
             for mask in (left, right)]
    facts = {"none": [left is None, right is None], "bitwise": same,
             "max_abs_diff": 0.0 if same else float("inf"), "finite": True,
             # presence is a fact of the comparison: it enters the digest chain too
             "digest": ["none" if side is None else _digest(side) for side in sides],
             "dtype": ["none" if side is None else str(side.dtype) for side in sides],
             "shape": [None if side is None else list(side.shape) for side in sides]}
    return dict(facts, reasons=mask_reasons(facts, tolerance))


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
            for field, value in (("mask_dtypes", "/".join(mask.get("dtype") or [])),
                                 ("mask_shapes", "/".join(str(s) for s in mask.get("shape")
                                                          or []))):
                if value not in entry.setdefault(field, []):
                    entry[field].append(value)
        for field, value in (("dtypes", "/".join(facts.get("dtype") or [])),
                             ("shapes", "/".join(str(shape) for shape in facts.get("shape") or []))):
            if value and value not in entry[field]:
                entry[field].append(value)
    return per_key


def replay_partition_reasons(record, n, orbit_size):
    """The released replay's OWN orbit partition is exp_20's per-angle plan at cap = its
    batch, or the comparison was not against the released path (Codex rB finding 7)."""
    wanted = {"partition": [int(n)] * (int(orbit_size) - 1), "frame_avg_fwd_cap": int(n),
              "candidate_micro_batch": int(n), "angles_per_chunk": 1,
              "n_orbit_forwards": int(orbit_size) - 1}
    found = {key: (record or {}).get(key) for key in wanted}
    return [] if found == wanted else [f"replay orbit partition {found} is not exp_20's "
                                       f"per-angle plan {wanted}"]


def coverage(sizes, micro):
    """How many full and tail micro-batches a gate exercised; both are required."""
    full = sum(1 for size in sizes if size == micro)
    tail = sum(1 for size in sizes if size < micro)
    return {"full": full, "tail": tail, "complete": bool(full and tail)}


def probe_query_set(registered, consumed):
    """The registered probe-query set against the cases a gate consumed (Codex rB finding 1;
    plan r2 amendment A4 v): every one, once, nothing else -- ordered keys and digests."""
    from src.localization.crossarm import canonical_sha256

    registered = [str(query) for query in (registered or [])]
    consumed = [str(query) for query in consumed]
    block = {"registered": registered, "registered_sha256": canonical_sha256(registered),
             "consumed": consumed, "consumed_sha256": canonical_sha256(consumed),
             "n_registered": len(registered), "n_consumed": len(consumed),
             "missing": [query for query in registered if query not in set(consumed)],
             "unexpected": [query for query in consumed if query not in set(registered)],
             "repeated": sorted({query for query in consumed if consumed.count(query) > 1})}
    reasons = [] if registered else ["no registered probe-query set was declared; a gate over "
                                     "an unstated set proves nothing"]
    reasons += [f"{len(block[name])} {what} (first {block[name][:3]}); a gate consumes every "
                f"registered probe query exactly once" for name, what in (
                    ("missing", "registered probe queries were not consumed"),
                    ("unexpected", "consumed queries are not registered probe queries"),
                    ("repeated", "registered probe queries were consumed twice")) if block[name]]
    block["complete"] = not reasons
    return block, reasons


def anchor_reasons(block, key_set_equal):
    """The anchor clause over the SERIALIZED anchor block (Codex rB-F1 finding 2): the pinned
    committed probe report, its ordered ids matching their digest, and the gate's registered
    and consumed queries exactly its key set."""
    from src.localization.crossarm import canonical_sha256

    if not isinstance(block, dict) or not block.get("sha256"):
        return ["anchor: the evidence names no pinned probe-report anchor, so its registered "
                "probe set is not shown to be the committed report's"]
    ids, reasons = list(block.get("query_ids") or []), []
    if not (block.get("sha256") == block.get("pinned_sha256") == COMMITTED_PROBE_REPORT_SHA256):
        reasons.append(f"anchor: the probe report {str(block.get('sha256'))[:12]}... (stated pin "
                       f"{str(block.get('pinned_sha256'))[:12]}...) is not the pinned committed "
                       f"probe report {COMMITTED_PROBE_REPORT_SHA256[:12]}...")
    if (block.get("query_ids_sha256") != canonical_sha256(ids) or len(set(ids)) != len(ids)
            or block.get("n_queries") != len(ids)):
        reasons.append("anchor: its ordered query ids do not match their digest or count")
    if block.get("query_ids_sha256") != COMMITTED_PROBE_QUERY_IDS_SHA256:
        reasons.append(f"anchor: its ordered query ids are not the registered content of the "
                       f"pinned probe report ({str(block.get('query_ids_sha256'))[:12]}... vs "
                       f"{COMMITTED_PROBE_QUERY_IDS_SHA256[:12]}...; Codex rB-F2 finding 2)")
    if block.get("frozen_digests_sha256") != COMMITTED_PROBE_FROZEN_DIGESTS_SHA256:
        reasons.append("anchor: its frozen grid-digest map is not the registered content of the "
                       "pinned probe report")
    if key_set_equal is not True:
        reasons.append("anchor: the registered and consumed probe queries are not exactly the "
                       "anchor's key set")
    return reasons


def anchor_evidence(anchor, registered, consumed):
    """``(block, key_set_equal, reasons)``: the anchor as G0 publishes it, for G3 / G3D."""
    block = {key: (anchor or {}).get(key) for key in ANCHOR_EVIDENCE}
    equal = key_set_equal(block, registered, consumed)
    return block, equal, anchor_reasons(block if anchor else None, equal)


def _content_reasons(ids, frozen):
    """An anchor's ordered ids (``None``: not compared) and frozen map against the registered
    content of the pinned report (Codex rB-F2 finding 2)."""
    from src.localization.crossarm import canonical_sha256

    reasons = [] if ids is None or canonical_sha256([str(query) for query in ids]) == (
        COMMITTED_PROBE_QUERY_IDS_SHA256) else [
        "its ordered query ids are not the registered content of the pinned probe report"]
    return reasons + ([] if canonical_sha256(frozen) == COMMITTED_PROBE_FROZEN_DIGESTS_SHA256
                      else ["its frozen grid-digest map is not the registered content of the "
                            "pinned probe report"])


def key_set_equal(block, registered, consumed):
    """The anchor's query ids are exactly the registered AND the consumed ones."""
    ids = sorted(str(query) for query in (block or {}).get("query_ids") or [])
    return bool(ids) and ids == sorted(registered or []) == sorted(consumed or [])


def gate_report(gate, reasons, **fields):
    """The verdict block every gate shares; a measurement (no tolerance) carries none."""
    if GATE_TOLERANCES[gate] is None:
        verdict, match, fields = "measurement", None, dict(fields, complete=not reasons)
    else:
        verdict, match = ("FAIL" if reasons else "PASS"), not reasons
    return {"gate": gate, "name": GATE_NAMES[gate], "verdict": verdict, "match": match,
            "reasons": list(reasons), "tolerance": GATE_TOLERANCES[gate],
            "note": GATE_NOTES[gate], **fields}


def autocast_reasons(gate, inputs):
    """A gate registered at an autocast (A4 i/ii) names the RESOLVED state it ran, or fails."""
    if gate not in GATE_AUTOCAST:
        return []
    requested, enabled = GATE_AUTOCAST[gate]
    found = (inputs.get("cond_autocast_requested"), inputs.get("cond_autocast_enabled"))
    if found != (requested, enabled) or not inputs.get("cond_autocast_dtype"):
        return [f"autocast: {GATE_NAMES[gate]} is registered at --cond-autocast {requested} "
                f"(enabled={enabled}), but the evidence records requested/enabled/dtype "
                f"{found + (inputs.get('cond_autocast_dtype'),)}"]
    return []


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
        why += replay_partition_reasons(record, len(chunk), len(split.angles))
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


GATE_REPORT_KEYS = ("gate", "name", "verdict", "match", "reasons", "tolerance", "note",
                    "complete")


def gate_identity(gate, binding, expect_arm=None):
    """The gate's arm identity through the FULL registered-binding check (Codex rB finding 3):
    the checkpoint must be one of the gate's arms and the binding must deviate from that
    arm's registered protocol on exactly the gate's admitted fields -- ``(block, reasons)``."""
    from src.localization import meshgrid_report as mr

    arms, admitted = GATE_IDENTITY[gate]
    block = {"gate": gate, "registered_arms": list(arms), "expect_arm": expect_arm,
             "admitted_deviations": dict(admitted), "arm": None, "deviations": None,
             "ckpt_sha256": (binding or {}).get("ckpt_sha256")}
    try:
        verdict = mr.assert_registered_protocol(dict(binding or {}), allow_deviation=True,
                                                expect_arm=expect_arm)
    except (ValueError, TypeError, KeyError) as error:
        reasons = [f"identity: {error}"]
    else:
        block.update(arm=verdict["arm"], deviations={field: entry["run"] for field, entry
                                                     in verdict["deviations"].items()})
        reasons = [] if verdict["arm"] in arms else [
            f"identity: {GATE_NAMES[gate]} runs the registered {'/'.join(arms)} arm; this "
            f"checkpoint is {verdict['arm'] or 'no registered arm'}"]
        if block["deviations"] != admitted:
            reasons.append(f"identity: the binding deviates from arm {verdict['arm']}'s "
                           f"registered protocol on {sorted(block['deviations'])}; "
                           f"{GATE_NAMES[gate]} admits exactly {admitted or 'none'}")
    block.update(registered_for_gate=not reasons, reasons=reasons)
    return block, reasons


def identity_reasons(inputs):
    identity = inputs.get("identity") or {}
    if identity.get("registered_for_gate") is True:
        return []
    return list(identity.get("reasons") or ["identity: the evidence names no registered arm "
                                            "identity for this gate (Codex rB finding 3)"])


def provenance_reasons(provenance):
    """A PASS rests on CAPTURED provenance (Codex rB finding 3): a full git SHA, a tracked-clean
    tree, the host and -- on a GPU -- the CUDA runtime's own card UUID."""
    reasons = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("git_sha") or "")):
        reasons.append(f"provenance: the git SHA was not captured as a full commit id "
                       f"({provenance.get('git_sha')!r})")
    if provenance.get("tracked_clean") is not True:
        reasons.append(f"provenance: the tracked tree was dirty or its state was not captured "
                       f"(tracked_clean={provenance.get('tracked_clean')!r})")
    if not provenance.get("hostname"):
        reasons.append("provenance: the hostname was not captured")
    if str(provenance.get("device")).startswith("cuda") and not str(
            provenance.get("gpu_uuid") or "").startswith("GPU-"):
        reasons.append(f"provenance: the CUDA runtime did not report the card's UUID "
                       f"({(provenance.get('gpu_runtime') or {}).get('error')})")
    return reasons


def finalize_evidence(report, device, environment=None, runtime=None, inputs=None):
    """The gate's evidence: its report, the run's provenance, and the verdict over ALL clauses.

    A numerical PASS stands only if the gate ran its registered autocast, under its
    registered arm identity, with its provenance captured; each clause is published.
    """
    from src.localization import meshgrid_offgrid_probe as mop

    environment = mop.current_environment() if environment is None else environment
    on_gpu = str(device).startswith("cuda")
    if on_gpu and runtime is None:
        runtime = mop.runtime_gpu_uuid(device)
    dirty = environment.get("git_status_dirty")
    merged = dict(report.get("inputs") or {}, **(inputs or {}))
    provenance = {
        "git_sha": environment.get("git_sha"),
        "tracked_clean": None if dirty is None else not bool(dirty),
        "hostname": environment.get("hostname"), "device": str(device),
        "gpu_uuid": (runtime or {}).get("uuid") if on_gpu else None,
        "gpu_runtime": runtime if on_gpu else None,
        "torch": torch.__version__, "record_free": True}
    clauses = {"autocast": autocast_reasons(report["gate"], merged),
               "identity": identity_reasons(merged), "provenance": provenance_reasons(provenance)}
    provenance.update(complete=not clauses["provenance"], reasons=clauses["provenance"])
    fields = {key: value for key, value in report.items() if key not in GATE_REPORT_KEYS}
    evidence = gate_report(report["gate"], list(report["reasons"]) + [
        reason for name in ("autocast", "identity", "provenance") for reason in clauses[name]],
        **dict(fields, inputs=merged))
    evidence["provenance"] = provenance
    evidence["verdict_clauses"] = {"comparison": not report["reasons"],
                                   **{name: not why for name, why in clauses.items()}}
    evidence["reverify_note"] = REVERIFY_NOTE
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
    wanted = 2 * int(source_chunk) if n_candidates is None else int(n_candidates)
    n = min(query.n_candidates, max(2, wanted))
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


def _score_deltas(production, replayed, indices, tau, prefixes, enforce=True):
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
        if enforce:                    # the drift measurement publishes, it does not judge
            why += score_reasons(k, out[str(k)])
    return out, why


def score_reasons(k, facts):
    """The score clause of one K over its serialized facts: |dS| within SCORE_TOLERANCE (a
    NaN never is) and the argmax unmoved."""
    delta, why = facts.get("max_abs_delta"), []
    if not (isinstance(delta, (int, float)) and delta <= me.SCORE_TOLERANCE):
        why.append(f"K={k}: score |diff| {_fmt(delta)} > SCORE_TOLERANCE {me.SCORE_TOLERANCE}")
    if facts.get("argmax_equal") is not True:
        why.append(f"K={k}: the argmax moved between production and replay")
    return why


def score_summary(entries):
    """Per-K stability over the queries: the worst |dS| and the agreement counts."""
    by_k = {}
    for entry in entries:
        for k, facts in entry["score_stability"].items():
            block = by_k.setdefault(k, {"max_abs_delta": 0.0, "n_queries": 0,
                                        "n_argmax_equal": 0, "n_at_risk": 0})
            block["max_abs_delta"] = max(block["max_abs_delta"], facts["max_abs_delta"])
            block["n_queries"] += 1
            block["n_argmax_equal"] += int(facts["argmax_equal"])
            block["n_at_risk"] += int(facts["at_risk"])
    return by_k


def chunk_summary(conditioning):
    """A chunk's headline over its five tensors: the worst |diff| and bitwise identity."""
    return {"max_abs_diff": max(e["tensor"]["max_abs_diff"] for e in conditioning.values()),
            "bitwise": all(e["tensor"]["bitwise"] and e["mask"]["bitwise"]
                           for e in conditioning.values())}


def source_sizes(union, chunk):
    """The source-cache batch sizes a receiver union of ``union`` makes at ``chunk``."""
    return [min(int(chunk), int(union) - start) for start in range(0, int(union), int(chunk))]


def _chain(chains, conditioning, inputs):
    """Extend each key's ordered per-side sha256 chain with this chunk's tensor and mask
    digests (an absent mask chains as ``none``)."""
    rows = [(key, entry["tensor"]) for key, entry in conditioning.items()]
    rows += [(f"{key}.mask", entry["mask"]) for key, entry in conditioning.items()]
    rows += [(f"dit_inputs.{name}", facts) for name, facts in inputs.items()]
    for name, facts in rows:
        sides = chains.setdefault(name, [hashlib.sha256(), hashlib.sha256()])
        for side, digest in zip(sides, facts.get("digest") or ["", ""]):
            side.update(str(digest).encode("utf-8"))


def _production_query(view, engine, case, *, micro, batch_rows, source_chunk, num_samples,
                      seed, noise_policy, tau, prefixes, per_key, sizes, chains, tolerance):
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
        conditioning, w_cond = compare_conditioning(produced, replay, tolerance)
        inputs, w_inputs = compare_inputs(engine.cond_inputs_fn(produced),
                                          engine.cond_inputs_fn(replay), tolerance)
        w_cond += replay_partition_reasons(replayed_partition[0], len(slice_indices),
                                           len(view.split.angles))
        fold_per_key(per_key, conditioning, inputs)
        _chain(chains, conditioning, inputs)
        why += [f"chunk@{start}: {item}" for item in w_cond + w_inputs]
        sizes["scored"].append(len(slice_indices))
        chunks.append({"start": start, "n": len(slice_indices), **chunk_summary(conditioning),
                       # what the verdict was judged on, whole (Codex rB-F1 finding 2)
                       "replay_partition": dict(replayed_partition[0]),
                       "conditioning": conditioning, "dit_inputs": inputs})
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
    sizes["source"] += source_sizes(union, source_chunk)
    scores, why_scores = _score_deltas(production, torch.cat(parts, dim=0), indices, tau,
                                       prefixes, enforce=tolerance is not None)
    return ({"inputs": query_inputs(query, len(indices), n_union=union), "chunks": chunks,
             "score_stability": scores}, why + why_scores)


def production_parity(engine, cases, *, batch_rows, source_chunk, num_samples=me.NUM_SAMPLES,
                      seed=me.SEED, noise_policy=me.NOISE_KEY_POLICY, tau=me.TAU,
                      prefixes=me.K_PREFIXES, declared_partition=None, expected_queries=None,
                      measurement=False, anchor=None):
    """G3 -- production batching against a faithful replay, with score/argmax stability; with
    ``measurement`` the same comparison as G3D, the drift measurement (no verdict, A4 ii)."""
    gate = "G3D" if measurement else "G3"
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
                                       chains=chains, tolerance=GATE_TOLERANCES[gate])
        entries.append(entry)
        reasons += [f"{case['query'].query_id}: {item}" for item in why]
    consumed = [entry["inputs"]["query_id"] for entry in entries]
    queries, why_queries = probe_query_set(expected_queries, consumed)
    anchored, key_set_equal, why_anchor = anchor_evidence(anchor, queries["registered"], consumed)
    reasons += why_queries + why_anchor
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
    by_k = score_summary(entries)
    context = {} if not measurement else {"context": {
        "vanilla_batching_caveat_max_abs_diff": VANILLA_BATCHING_CAVEAT_MAX_ABS_DIFF,
        "batching_caveat": me.BATCHING_CAVEAT}}
    return gate_report(
        gate, reasons, queries=entries, probe_queries=queries, per_key=per_key, coverage=covered,
        score_stability=by_k, **context,
        digest_chains={name: [side.hexdigest() for side in sides]
                       for name, sides in chains.items()},
        score_tolerance=None if measurement else me.SCORE_TOLERANCE,
        declared_partition=declared_partition,
        inputs={"queries": [entry["inputs"]["query_id"] for entry in entries],
                "batch_rows": int(batch_rows), "source_chunk": int(source_chunk),
                "num_samples": int(num_samples), "micro_batch": micro, "seed": int(seed),
                "noise_policy": str(noise_policy), "tau": float(tau),
                "prefixes": [int(k) for k in prefixes], "angles": list(split.angles),
                "probe_report": anchored, "anchor_key_set_equal": key_set_equal},
        # the raw call counts: the executed block is re-derivable from the evidence alone
        executed_partition=dict(executed, counts=view.split.recorder.to_counts()))


def load_probe_anchor(path=COMMITTED_PROBE_REPORT):
    """The committed P1 probe report as the gates' frozen anchor, its bytes pinned by digest.

    Read ONCE, so the digest is of the bytes parsed; any other bytes are refused,
    wherever they live. Returns the path, the digest, the ordered query ids (the
    registered 16) with their digest, and ``{query_id: {row_sha256, sims_sha256}}``.
    """
    from src.localization.crossarm import canonical_sha256

    with open(str(path), "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != COMMITTED_PROBE_REPORT_SHA256:
        raise ValueError(f"the probe report {str(path)!r} hashes to {digest[:12]}..., not the "
                         f"committed canonical P1 probe report {COMMITTED_PROBE_REPORT_SHA256[:12]}"
                         "...; the frozen anchor is pinned by digest and not caller-replaceable "
                         "(Codex rB finding 2)")
    records = json.loads(raw.decode("utf-8"))["records"]
    ids = [str(record["query_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"the probe report {str(path)!r} names a query twice")
    digests = {str(record["query_id"]): {"row_sha256": record["grid_row_sha256"],
                                         "sims_sha256": record["grid_sims_sha256"]}
               for record in records}
    content = (canonical_sha256(ids), canonical_sha256(digests))
    if content != (COMMITTED_PROBE_QUERY_IDS_SHA256, COMMITTED_PROBE_FROZEN_DIGESTS_SHA256):
        raise ValueError(f"the probe report {str(path)!r} is not the registered content of the "
                         f"pinned report: its ordered ids / frozen map hash to "
                         f"{content[0][:12]}... / {content[1][:12]}... (Codex rB-F2 finding 2)")
    return {"path": str(path), "sha256": digest, "pinned_sha256": COMMITTED_PROBE_REPORT_SHA256,
            "query_ids": ids, "query_ids_sha256": content[0], "n_queries": len(ids),
            "frozen_digests_sha256": content[1], "digests": digests}


def frozen_grid_digests(probe_report_path=COMMITTED_PROBE_REPORT):
    """``{query_id: {row_sha256, sims_sha256}}`` the committed (pinned) P1 probe report froze."""
    return load_probe_anchor(probe_report_path)["digests"]


#: what of the anchor a gate's evidence publishes (the digests themselves stay in the report)
ANCHOR_EVIDENCE = ("path", "sha256", "pinned_sha256", "query_ids", "query_ids_sha256",
                   "n_queries", "frozen_digests_sha256")


def vanilla_replay_regression(engine, cases, run_dir, *, anchor, observation_bank_sha256,
                              expected_observation_bank_sha256=FROZEN_OBSERVATION_BANK_SHA256,
                              run_binding=None, seed=me.SEED, num_samples=me.NUM_SAMPLES,
                              noise_policy=me.NOISE_KEY_POLICY, tau=me.TAU, expected_queries=None):
    """G0 -- the vanilla path still reproduces the published P1 rows of the probe queries."""
    from src.localization import meshgrid_drift_measurement as dm
    from src.localization import meshgrid_offgrid_probe as mop
    from src.localization import meshgrid_report as mr

    if getattr(engine, "split", None) is not None or str(engine.cond_method) != "vanilla":
        raise ValueError("the no-regression replay gates the vanilla path; this engine runs the fa "
                         "split, which gates 1-3 cover")
    published, published_sha = mr.load_published_binding(run_dir)
    reasons = []
    if not (anchor.get("sha256") == anchor.get("pinned_sha256") == COMMITTED_PROBE_REPORT_SHA256):
        reasons.append(f"the frozen anchor {str(anchor.get('sha256'))[:12]}... is not the pinned "
                       f"committed probe report {COMMITTED_PROBE_REPORT_SHA256[:12]}...")
    frozen_digests = anchor.get("digests") or {}
    reasons += [f"the frozen anchor: {why}" for why in _content_reasons(
        anchor.get("query_ids") or [], frozen_digests)]
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
                 # both sides of the comparison, so it re-verifies from the JSON alone
                 "frozen_digests": frozen_digests.get(query.query_id),
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
    queries, why_queries = probe_query_set(expected_queries,
                                           [entry["query_id"] for entry in entries])
    reasons += why_queries
    key_set_equal = (sorted(entry["query_id"] for entry in entries)
                     == sorted(anchor.get("query_ids") or []) == sorted(frozen_digests))
    if not key_set_equal:
        reasons.append("the replayed queries are not exactly the anchor's key set "
                       f"({len(entries)} replayed, {len(anchor.get('query_ids') or [])} frozen)")
    return gate_report("G0", reasons, queries=entries, probe_queries=queries,
                       inputs={"run_dir": str(run_dir), "published_binding_sha256": published_sha,
                               "published_binding": {field: published[field] for field
                                                     in me.binding_fields(published)},
                               "probe_report": {key: anchor.get(key) for key in ANCHOR_EVIDENCE},
                               "anchor_key_set_equal": key_set_equal,
                               "observation_bank_sha256": str(observation_bank_sha256),
                               "expected_observation_bank_sha256":
                                   str(expected_observation_bank_sha256),
                               "queries": [entry["query_id"] for entry in entries]},
                       executed_partition=None)


# --------------------------------------------------------------------------- #
# exp_22 rB-F2 (Codex rB-F1 finding 2): every PASS clause re-verified from the JSON alone
# --------------------------------------------------------------------------- #
#: the five conditioner ids every fa comparison block covers
FA_IDS = tuple(key for ids in me.FA_BRANCH_IDS.values() for key in ids)
#: the identity block's fields, re-derived from the evidence's own binding
IDENTITY_FIELDS = ("gate", "registered_arms", "expect_arm", "admitted_deviations", "arm",
                   "deviations", "ckpt_sha256", "registered_for_gate")
MALFORMED = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def conditioning_reasons(conditioning, tolerance, ids=FA_IDS):
    """The per-key clauses of one serialized comparison block: every id, tensor and mask."""
    reasons = [f"{key}: absent from the comparison" for key in ids if key not in conditioning]
    for key, entry in conditioning.items():
        reasons += [f"{key}.tensor: {why}" for why in pair_reasons(entry["tensor"], tolerance)]
        reasons += [f"{key}.mask: {why}" for why in mask_reasons(entry["mask"], tolerance)]
    return reasons


def inputs_reasons(dit_inputs, tolerance):
    """The DiT-input clauses of one serialized block (a one-sided input carries no shape)."""
    if not dit_inputs:
        return ["dit_inputs: the inputs the DiT consumes were not compared"]
    return [f"dit_inputs.{name}: {why}" for name, facts in dit_inputs.items()
            for why in (["present on one side only"] if "shape" not in facts
                        else pair_reasons(facts, tolerance))]


def executed_reasons(executed, declared):
    """The executed partition re-derived from its raw call counts, held to the declaration."""
    again = me.FaPartitionRecorder.from_counts(executed.get("counts") or []).report()
    reasons = [] if all(executed.get(key) == again[key] for key in (
        "executed", "violations", "branches")) else [
        "executed partition: the published block is not the one its call counts yield"]
    reasons += [f"executed partition differs from the declared one at {path}"
                for path in me.fa_partition_differences(declared, again["executed"])]
    return reasons + [f"executed partition violation: {item}" for item in again["violations"]]


def _reverify_production(evidence, inputs):
    """G3 / G3D over the serialized evidence: the registered production conditions, the
    anchored probe set, each query's chunk tiling, replay record, per-key facts and score
    stability, the summaries re-derived from the chunks, coverage, the executed partition."""
    from src.localization import meshgrid_report as mr

    tolerance, binding = GATE_TOLERANCES[evidence["gate"]], inputs.get("binding") or {}
    micro, chunk_size = int(inputs["micro_batch"]), int(inputs["source_chunk"])
    bound = {"batch_rows": mr.REGISTERED_BATCH_ROWS, "source_chunk": mr.REGISTERED_SOURCE_CHUNK,
             "num_samples": binding.get("num_samples"), "prefixes": binding.get("k_prefixes"),
             "tau": binding.get("tau"), "seed": binding.get("seed"),
             "noise_policy": binding.get("noise_policy"),
             "angles": binding.get("frame_avg_angles"),
             "micro_batch": mr.REGISTERED_BATCH_ROWS // int(binding.get("num_samples") or 1)}
    reasons = [f"inputs: {field} {inputs.get(field)!r} is not the registered / bound {value!r}"
               for field, value in bound.items() if inputs.get(field) != value]
    if evidence.get("declared_partition") != binding.get("fa_partition"):
        reasons.append("partition: the declared partition is not the run binding's fa_partition")
    queries = evidence.get("queries") or []
    consumed = [str(entry["inputs"]["query_id"]) for entry in queries]
    block, why = probe_query_set((evidence.get("probe_queries") or {}).get("registered"), consumed)
    reasons += why
    if evidence.get("probe_queries") != block or inputs.get("queries") != consumed:
        reasons.append("probe_queries: the published set is not the one its registered and "
                       "consumed ids yield")
    equal = key_set_equal(inputs.get("probe_report"), block["registered"], consumed)
    reasons += anchor_reasons(inputs.get("probe_report"), equal)
    if inputs.get("anchor_key_set_equal") is not equal:
        reasons.append("anchor: the published anchor_key_set_equal is not the recomputed one")
    scored, source, per_key, chains = [], [], {}, {}
    prefixes = sorted(str(int(k)) for k in inputs.get("prefixes") or [])
    for entry in queries:
        query = entry["inputs"]
        n, qid = int(query["n_candidates"]), query["query_id"]
        tiling = [[start, min(micro, n - start)] for start in range(0, n, micro)]
        if ([[chunk.get("start"), chunk.get("n")] for chunk in entry["chunks"]] != tiling
                or len(query.get("candidate_indices") or []) != n):
            reasons.append(f"{qid}: its chunks do not tile its {n} candidates at micro {micro}")
        for chunk in entry["chunks"]:
            cond, dit = chunk["conditioning"], chunk["dit_inputs"]
            why = conditioning_reasons(cond, tolerance) + inputs_reasons(dit, tolerance)
            why += replay_partition_reasons(chunk.get("replay_partition"), chunk["n"],
                                            len(inputs.get("angles") or []))
            if {key: chunk.get(key) for key in ("max_abs_diff", "bitwise")} != chunk_summary(cond):
                why.append("its max |diff| / bitwise headline is not its facts'")
            reasons += [f"{qid}: chunk@{chunk.get('start')}: {item}" for item in why]
            fold_per_key(per_key, cond, dit)
            _chain(chains, cond, dit)
            scored.append(chunk["n"])
        source += source_sizes(query["n_union"], chunk_size)
        if sorted(entry["score_stability"]) != prefixes:
            reasons.append(f"{qid}: its score stability is not published for exactly K {prefixes}")
        if tolerance is not None:
            reasons += [f"{qid}: {item}" for k, facts in entry["score_stability"].items()
                        for item in score_reasons(k, facts)]
    if per_key != evidence.get("per_key"):
        reasons.append("per_key: the published per-key summary is not the fold of its chunks")
    if {name: [side.hexdigest() for side in sides] for name, sides in chains.items()} != \
            evidence.get("digest_chains"):
        reasons.append("digest_chains: the published chains are not its chunks' digests")
    covered = {"scored": coverage(scored, micro), "source": coverage(source, chunk_size)}
    reasons += [f"coverage: {name} batches {facts['full']} full / {facts['tail']} tail; a gate "
                "covers both" for name, facts in covered.items() if not facts["complete"]]
    if covered != evidence.get("coverage"):
        reasons.append("coverage: the published coverage is not the one its chunks yield")
    if score_summary(queries) != evidence.get("score_stability"):
        reasons.append("score_stability: the published summary is not its queries'")
    return reasons + executed_reasons(evidence.get("executed_partition") or {},
                                      evidence.get("declared_partition"))


def _reverify_identity(gate, inputs):
    """The arm identity re-derived from the evidence's own run binding (Codex rB finding 3)."""
    binding = inputs.get("binding")
    if not isinstance(binding, dict):
        return ["identity: the evidence carries no run binding to re-derive its identity from"]
    stored = inputs.get("identity") or {}
    block, reasons = gate_identity(gate, binding, stored.get("expect_arm"))
    moved = [field for field in IDENTITY_FIELDS if stored.get(field) != block[field]]
    return reasons + ([f"identity: the published {moved} differ from the binding's"]
                      if moved else [])


def _reverify_provenance(provenance):
    reasons = provenance_reasons(provenance)
    if provenance.get("complete") is not (not reasons) or provenance.get("reasons") != reasons:
        reasons.append("provenance: the published complete / reasons are not its fields'")
    return reasons


def _verdict_reasons(gate, evidence, clauses):
    """The published verdict, tolerances and clause flags against the re-verified clauses."""
    failed = any(clauses.values())
    reasons = [] if evidence.get("name") == GATE_NAMES[gate] else [
        f"verdict: the evidence is named {evidence.get('name')!r}, not {GATE_NAMES[gate]!r}"]
    if evidence.get("tolerance") != GATE_TOLERANCES[gate]:
        reasons.append(f"tolerance: the evidence publishes {evidence.get('tolerance')!r}, not the "
                       f"registered {GATE_TOLERANCES[gate]!r}")
    wanted = {"G3": me.SCORE_TOLERANCE, "G3D": None}
    if gate in wanted and evidence.get("score_tolerance") != wanted[gate]:
        reasons.append(f"score_tolerance: the evidence publishes "
                       f"{evidence.get('score_tolerance')!r}, not {wanted[gate]!r}")
    if GATE_TOLERANCES[gate] is None:
        if (evidence.get("verdict"), evidence.get("match")) != ("measurement", None):
            reasons.append("verdict: a measurement carries verdict 'measurement' and match null")
        if evidence.get("complete") is not (not failed) or (evidence.get("reasons") and not failed):
            reasons.append(f"complete: published {evidence.get('complete')!r}, re-verified "
                           f"{not failed}")
    elif (evidence.get("verdict"), evidence.get("match"), evidence.get("reasons")) != (
            "PASS", True, []):
        reasons.append(f"verdict: the evidence claims {evidence.get('verdict')!r} (match "
                       f"{evidence.get('match')!r}, reasons {len(evidence.get('reasons') or [])}"
                       "), not a clean PASS")
    published = evidence.get("verdict_clauses") or {}
    return reasons + [f"verdict_clauses: {name} is published {published.get(name)!r} but "
                      f"re-verifies {not why}" for name, why in clauses.items()
                      if published.get(name) is not (not why)]


def _orbit_reasons(inputs):
    angles = (inputs.get("binding") or {}).get("frame_avg_angles")
    return [] if inputs.get("angles") == angles else [
        f"inputs: the compared orbit {inputs.get('angles')!r} is not the binding's {angles!r}"]


def _reverify_split(evidence, inputs):
    """G1: the chunks tile the compared candidates, each bit-exact on the five ids, their
    masks and the DiT inputs at the released per-angle replay record; full + tail coverage."""
    micro, n, tolerance = int(inputs["micro_batch"]), int(inputs["n_candidates"]), 0.0
    chunks, per_key, reasons = evidence.get("chunks") or [], {}, _orbit_reasons(inputs)
    tiling = [[start, min(micro, n - start), n - start >= micro] for start in range(0, n, micro)]
    if ([[chunk.get("start"), chunk.get("n"), chunk.get("full")] for chunk in chunks] != tiling
            or len(inputs.get("candidate_indices") or []) != n):
        reasons.append(f"its chunks do not tile its {n} candidates at micro {micro}")
    for chunk in chunks:
        why = conditioning_reasons(chunk["conditioning"], tolerance)
        why += inputs_reasons(chunk["dit_inputs"], tolerance) + replay_partition_reasons(
            chunk.get("released_partition"), chunk["n"], len(inputs.get("angles") or []))
        reasons += [f"chunk@{chunk.get('start')}: {item}" for item in why]
        fold_per_key(per_key, chunk["conditioning"], chunk["dit_inputs"])
    covered = coverage([chunk["n"] for chunk in chunks], micro)
    if not covered["complete"] or covered != evidence.get("coverage"):
        reasons.append(f"coverage: {covered['full']} full / {covered['tail']} tail re-derived "
                       f"(published {evidence.get('coverage')}); a gate covers both")
    if per_key != evidence.get("per_key"):
        reasons.append("per_key: the published per-key summary is not the fold of its chunks")
    return reasons


def _reverify_cache(evidence, inputs):
    """G2: the memoization is bit-exact on the five ids, masks and DiT inputs, and each of
    the four counter-tests bites on exactly its keys."""
    memo, tolerance = evidence.get("memoization") or {}, GATE_TOLERANCES["G2"]
    why = conditioning_reasons(memo["conditioning"], tolerance)
    why += inputs_reasons(memo["dit_inputs"], tolerance)
    reasons = _orbit_reasons(inputs) + [f"memoization: {item}" for item in why]
    if memo.get("match") is not (not why):
        reasons.append(f"memoization: published match {memo.get('match')!r}, re-verified {not why}")
    counters = evidence.get("counter_tests") or {}
    if set(counters) != set(COUNTER_KEYS):
        reasons.append(f"counter-test: published {sorted(counters)}, registered "
                       f"{sorted(COUNTER_KEYS)}")
    for name, keys in COUNTER_KEYS.items():
        moved = (counters.get(name) or {}).get("keys") or {}
        bites = set(moved) == set(keys) and all(value > 0.0 for value in moved.values())
        if not bites or (counters.get(name) or {}).get("detected") is not bites:
            reasons.append(f"counter-test {name} did not bite on {sorted(keys)} (published "
                           f"detected {(counters.get(name) or {}).get('detected')!r})")
    if fold_per_key({}, memo["conditioning"], memo["dit_inputs"]) != evidence.get("per_key"):
        reasons.append("per_key: the published per-key summary is not the memoization's fold")
    return reasons


def _reverify_vanilla_cache(evidence, inputs):
    """GV: the §1.5 memoization matched on every vanilla id and its counter-test bit."""
    report = evidence.get("vanilla_cache_parity") or {}
    memo, counter = report.get("memoization") or {}, report.get("counter_test") or {}
    keys, ids = memo.get("keys") or {}, me.CONTEXT_COND_IDS + me.SOURCE_COND_IDS
    memo_ok = sorted(keys) == sorted(ids) and all(entry.get("equal") is True
                                                  for entry in keys.values())
    found = counter.get("max_abs_diff")
    bites = isinstance(found, (int, float)) and found > 0.0
    reasons = [] if memo_ok else [f"memoization: the vanilla memoization is not exact on "
                                  f"every id of {list(ids)}"]
    reasons += [] if bites else [f"counter-test: the vanilla counter-test did not bite ({found!r})"]
    if (memo.get("match"), counter.get("detected"), report.get("match")) != (
            memo_ok, bites, memo_ok and bites):
        reasons.append("memoization: the published match / detected flags are not re-verified")
    if (inputs.get("query_id"), inputs.get("n_candidates")) != (report.get("query_id"),
                                                                report.get("n_candidates")):
        reasons.append("inputs: the gate's query is not the one its parity report covers")
    return reasons


def _reverify_replay(evidence, inputs):
    """G0: the probe set and the pinned anchor, the frozen observation bank, the published
    protocol replayed, and per query the frozen grid digests, the float16-exact sidecar and
    the 0.0 aggregate."""
    queries = evidence.get("queries") or []
    consumed = [str(entry["query_id"]) for entry in queries]
    block, reasons = probe_query_set((evidence.get("probe_queries") or {}).get("registered"),
                                     consumed)
    if evidence.get("probe_queries") != block or inputs.get("queries") != consumed:
        reasons.append("probe_queries: the published set is not the one its registered and "
                       "consumed ids yield")
    anchor = inputs.get("probe_report") or {}
    equal = sorted(consumed) == sorted(anchor.get("query_ids") or []) and all(
        isinstance(entry.get("frozen_digests"), dict) for entry in queries)
    reasons += anchor_reasons(anchor, equal)
    if inputs.get("anchor_key_set_equal") is not equal:
        reasons.append("anchor: the published anchor_key_set_equal is not the recomputed one")
    # the map the frozen clause compared, re-assembled from the queries: replacing the frozen
    # and published maps together no longer re-verifies (Codex rB-F2 finding 2)
    reasons += [f"G0's queries: {why}" for why in _content_reasons(
        None, {str(entry["query_id"]): entry.get("frozen_digests") for entry in queries})]
    banks = (inputs.get("observation_bank_sha256"), inputs.get("expected_observation_bank_sha256"))
    if banks != (FROZEN_OBSERVATION_BANK_SHA256, FROZEN_OBSERVATION_BANK_SHA256):
        reasons.append(f"the observation bank {str(banks[0])[:12]}... is not the frozen "
                       f"{FROZEN_OBSERVATION_BANK_SHA256[:12]}...")
    published, replayed = inputs.get("published_binding") or {}, inputs.get("binding") or {}
    if me.binding_sha256(published) != inputs.get("published_binding_sha256"):
        reasons.append("the published binding does not hash to its published_binding_sha256")
    reasons += [f"the replay does not run the published protocol: {field} differs"
                for field in me.RUN_BINDING_FIELDS if field not in REPLAY_BINDING_EXEMPT
                and replayed.get(field) != published.get(field)]
    for entry in queries:
        frozen = isinstance(entry.get("frozen_digests"), dict) and entry.get(
            "frozen_digests") == entry.get("published_digests")
        if not frozen or entry.get("frozen_digests_equal") is not frozen:
            reasons.append(f"{entry['query_id']}: the published row's grid digests are not the "
                           "frozen ones of the committed P1 probe report")
        matched = entry["matched"]
        if matched.get("float16_bit_exact") is not True or matched.get("n_float16_mismatch") != 0:
            reasons.append(f"{entry['query_id']}: the sidecar is not reproduced float16-exactly")
        if matched["aggregate"]["max_abs_delta"] != 0.0:
            reasons.append(f"{entry['query_id']}: the float32 aggregate moved by "
                           f"{_fmt(matched['aggregate']['max_abs_delta'])}")
    return reasons


#: the gates :func:`reverify_gate_evidence` re-verifies, by their comparison clauses
REVERIFIERS = {"G1": _reverify_split, "G2": _reverify_cache, "G3": _reverify_production,
               "G3D": _reverify_production, "G0": _reverify_replay,
               "GV": _reverify_vanilla_cache}


def reverify_gate_evidence(evidence):
    """Every PASS clause of a gate's evidence recomputed FROM THE SERIALIZED JSON ONLY (Codex
    rB-F1 finding 2): ``[]`` iff it re-verifies as the clean PASS it claims -- for the G3D
    measurement, as COMPLETE -- else every named reason. No file, model or tensor is read:
    the clause predicates the gates themselves apply run over the published facts, every
    summary is re-derived from the parts it summarizes, the identity from the published
    binding, and the registered constants (tolerances, the pinned anchor, the production
    batching) come from this code, never from the evidence."""
    gate = evidence.get("gate") if isinstance(evidence, dict) else None
    if gate not in REVERIFIERS:
        return [f"gate: {gate!r} is not one of the re-verifiable gates {sorted(REVERIFIERS)}"]
    inputs, clauses = evidence.get("inputs") or {}, {}
    for name, check in (("comparison", lambda: REVERIFIERS[gate](evidence, inputs)),
                        ("autocast", lambda: autocast_reasons(gate, inputs)),
                        ("identity", lambda: _reverify_identity(gate, inputs)),
                        ("provenance",
                         lambda: _reverify_provenance(evidence.get("provenance") or {}))):
        try:
            clauses[name] = list(check())
        except MALFORMED as error:
            clauses[name] = [f"{name}: the evidence is malformed ({type(error).__name__}: "
                             f"{error})"]
    return [reason for why in clauses.values() for reason in why] + _verdict_reasons(
        gate, evidence, clauses)
