"""exp_37 latency summary library (plan_loc_efficiency.md §1 formulas, §10 B1 / B2 time axis / CPU co-tenancy).

The bundle's formulas, unchanged (``tools/summarize_core_forward_latency.py``): per query
``L_q = core_forward_total_q + selector_q`` (the AGREE selector's median ``scoring_total_seconds``); s/query = mean /
median / P90 (``np.quantile`` linear) over the queries; ms/candidate dataset-wide = ``sum L / sum |C|``, query median
/ P90 of ``L_q / |C_q|``. Every run is validated against its arm's protocol (conditioning, the pinned local checkpoint,
K_ctx, K_gen 1, the bundle's timing flags, SDPA) and its host records (a timing with another compute process on its
card, or on an unidentified card, is never valid). Never ``elapsed_seconds``.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.localization import exp37_identity as ident
from src.localization.pilot import canonical_sha256

ARM_PROTOCOL = ident.ARMS
SELECTOR_RECORD_TYPES = ("exp37_agree_selector_latency", "localization_selector_latency_kctx8_kgen1")
# a selection served by a selector vector measured on a registered superset (records contained identically)
REGISTERED_SUPERSETS = {
    "6eeeec401c3f63e47ab446b4b43efe2f1db260bad9decf74de423d0872b659a2":          # the 64-query seed-42 pilot
        ("9bf75810a7fe6972049650748e015f9856b586f0c66d0554ef1b67c6bf8e041b",),   # the 128-query latency selection
}
CPU_COTENANT_PCPU = 10.0
SPLIT_TOLERANCE = 0.02          # plan §10: outside 2 % the split is withheld, never the headline
STAGES = ("context_conditioning", "candidate_conditioning", "dit", "vae_decode")
C4 = [0.0, 90.0, 180.0, 270.0]


def _hashed(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return payload


def summarize(values: list[float]) -> dict:
    """The bundle's ``summarize`` (tools/summarize_core_forward_latency.py), verbatim."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("latency values must be a nonempty finite vector")
    return {"query_count": len(array), "mean_seconds": float(array.mean()), "median_seconds": float(np.median(array)),
            "p90_seconds": float(np.quantile(array, 0.9)), "minimum_seconds": float(array.min()),
            "maximum_seconds": float(array.max()), "summed_seconds": float(array.sum())}


def latency_statistics(core: dict, selector: dict, counts: dict) -> dict:
    """s/query and ms/candidate statistics in the bundle's operation order (sorted query index)."""
    if set(core) != set(counts) or not set(counts) <= set(selector):
        raise RuntimeError("latency coverage mismatch between the run, the selector vector and the selection")
    ordered = sorted(counts)
    values = {i: float(core[i] + selector[i]) for i in ordered}
    total = sum(counts[i] for i in ordered)
    per_candidate = summarize([values[i] / counts[i] for i in ordered])
    per_candidate["amortized_seconds"] = float(sum(values.values()) / total)
    per_candidate["candidate_evaluations"] = total
    return {"overall": summarize([values[i] for i in ordered]), "per_candidate": per_candidate,
            "latency_seconds": {str(i): values[i] for i in ordered}}


def ordered_subsequence(small: list, big: list) -> bool:
    """True when every record of ``small`` occurs, identical and in order, among ``big``'s records."""
    position = 0
    for record in small:
        while position < len(big) and big[position] != record:
            position += 1
        if position == len(big):
            return False
        position += 1
    return True


def load_selector(path: Path, selection: dict) -> dict:
    """Per selection query: the selector term and its peak. The vector must be the registered AGREE on the registered
    context manifest, measured on this selection or on a REGISTERED superset; each row must name the same query,
    room, candidate count (and candidate hash when recorded)."""
    payload = _hashed(path)
    if payload.get("record_type") not in SELECTOR_RECORD_TYPES or int(payload.get("repeat_count", -1)) != 3 \
            or int(payload.get("candidate_batch_size", -1)) != 64:
        raise RuntimeError(f"{path}: not a 3-repeat, batch-64 AGREE selector record")
    if payload.get("agree_checkpoint_file_sha256") != ident.AGREE_SHA256:
        raise RuntimeError(f"{path}: not the registered AGREE checkpoint")
    if payload.get("context_manifest_sha256") != ident.CONTEXT_MANIFEST_SHA256:
        raise RuntimeError(f"{path}: not the registered context manifest")
    if payload.get("selection_sha256") != selection["sha256"] and payload.get("selection_sha256") not in \
            REGISTERED_SUPERSETS.get(selection["sha256"], ()):
        raise RuntimeError(f"{path}: measured on another selection that is not a registered superset of this one")
    rows = {int(row["query_index"]): row for row in payload["queries"]}
    out = {}
    for record in selection["records"]:
        index = int(record["index"])
        row = rows.get(index)
        if row is None:
            raise RuntimeError(f"{path}: selector coverage misses query {index}")
        keys = ["query_id", "room", "candidate_count"] + (
            ["candidate_indices_sha256"] if "candidate_indices_sha256" in record and row.get("candidate_indices_sha256") else [])
        if any(row[key] != record[key] for key in keys):
            raise RuntimeError(f"{path}: selector identity mismatch at query {index}")
        out[index] = {"scoring_total_seconds": float(row["agree"]["median_seconds"]["scoring_total_seconds"]),
                      "peak_memory_allocated_bytes": row.get("peak_memory_allocated_bytes"),
                      "peak_memory_reserved_bytes": row.get("peak_memory_reserved_bytes")}
    return out


def _norm_uuid(value) -> str:
    text = str(value or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def _snapshot_verdict(snapshot: dict, name: str, gpu: list, cpu: list) -> tuple[bool, bool]:
    """Collect one snapshot's GPU / CPU co-tenants; return (card identified, own pid seen on the card)."""
    pid, card = int(snapshot["pid"]), _norm_uuid((snapshot.get("cuda_device") or {}).get("uuid"))
    apps = snapshot.get("gpu_compute_apps", {})
    own_seen = False
    for line in apps.get("lines", []):
        fields = [f.strip() for f in line.split(",")]
        if _norm_uuid(fields[0]) == card and int(fields[1]) != pid:
            gpu.append(f"{name}: {line}")
        own_seen |= _norm_uuid(fields[0]) == card and int(fields[1]) == pid
    for line in snapshot.get("top_cpu_processes", {}).get("lines", []):
        fields = line.split()
        if int(fields[0]) != pid and float(fields[3]) >= CPU_COTENANT_PCPU and line not in cpu:
            cpu.append(line)
    return bool(card) and "lines" in apps, own_seen


def host_cotenancy(run_dir: Path) -> dict:
    """GPU co-tenants (another pid on the run's card -> invalid; no record, an unidentified card, or the run never
    seen on its own card -- a uuid-matching failure that could hide co-tenants -> invalid) and CPU co-tenants
    (other processes at >= 10 %CPU in the ``ps`` top lists -- ps pcpu is a lifetime average, so this is a coarse
    record for the Planner's judgement, not a verdict)."""
    records = sorted((Path(run_dir) / "host_records").glob("*.json"))
    gpu, cpu, unknown, own_seen = [], [], not records, False
    for path in records:
        identified, seen = _snapshot_verdict(json.loads(path.read_text())["snapshot"], path.name, gpu, cpu)
        unknown |= not identified
        own_seen |= seen
    return {"timing_valid": not gpu and not unknown and own_seen, "gpu_cotenants": gpu, "cpu_cotenants": cpu,
            "host_record_count": len(records), "unidentified_card": unknown, "own_process_seen_on_card": own_seen}


def selector_cotenancy(payload: dict) -> dict:
    """The selector vector's own end-of-run snapshot, judged like a run (no snapshot -> not timing-valid)."""
    gpu, cpu = [], []
    snapshot = payload.get("host")
    identified, own_seen = _snapshot_verdict(snapshot, "selector", gpu, cpu) if snapshot else (False, False)
    return {"timing_valid": identified and own_seen and not gpu, "gpu_cotenants": gpu, "cpu_cotenants": cpu,
            "unidentified_card": not identified, "own_process_seen_on_card": own_seen}


def load_arm_run(run_dir: Path, *, arm: str, selection: dict, audit: dict, seed: int = 42, context_count: int = 8,
                 stage_split: bool = False, subset_manifest: dict | None = None, frame_angles=None) -> dict:
    """``{query index: {core, context, candidate, peak_memory_bytes, candidate_count}}`` of one validated run;
    a headline run must be uninstrumented and a split run must be the split pass (plus its stage fields); the
    orbit (``frame_angles``, default C4 for fa) is checked; a rotated run is never a timing source; a nested-1.0 m
    timing pass is bound to its ``subset_manifest`` (counts and each query's subset sha256)."""
    run_dir = Path(run_dir)
    manifest = _hashed(run_dir / "run_manifest.json")
    identity = manifest["identity"]
    if subset_manifest is not None and subset_manifest.get("parent_pilot_sha256") != selection["sha256"]:
        raise RuntimeError("the subset manifest is not a subset of this selection")
    want = ident.registered_identity(arm, selection=selection, audit=audit, seed=seed, timing=True,
                                     context_count=context_count, frame_angles=frame_angles, stage_split=stage_split,
                                     candidate_subset="nested_1m" if subset_manifest is not None else None)
    bad = ident.identity_reasons(identity, want, audit)
    if bad:
        raise RuntimeError(f"{run_dir}: {bad} (the {arm} K_ctx={context_count} seed-{seed} registered protocol)")
    protocol = {"name": f"fem_core_aligned_kctx{context_count}_kgen1", "context_count": int(context_count)}
    subset_rows = {int(r["index"]): r for r in (subset_manifest or {}).get("records", [])}
    rows = {}
    for record in selection["records"]:
        index = int(record["index"])
        query = _hashed(run_dir / "queries" / f"query_{index:05d}.json")
        expected = subset_rows[index]["candidate_count"] if subset_manifest is not None else record["candidate_count"]
        if query.get("run_manifest_sha256") != manifest["sha256"] or int(query["candidate_count"]) != int(
                expected) or query.get("query_id") != record["query_id"]:
            raise RuntimeError(f"{run_dir}: query {index} does not belong to this run / selection")
        if subset_manifest is not None and (query.get("candidate_subset") or {}).get(
                "indices_sha256") != subset_rows[index]["candidate_indices_sha256"]:
            raise RuntimeError(f"{run_dir}: query {index} ran another candidate subset than the subset manifest's")
        given = query.get("latency_protocol") or {}
        if any(given.get(k) != v for k, v in protocol.items()) or bool(given.get("stage_split")) != stage_split:
            raise RuntimeError(f"{run_dir}: query {index} latency_protocol is not {protocol} (stage_split={stage_split})")
        seconds = query["latency_seconds"]
        rows[index] = {"core": float(seconds["core_forward_total"]), "context": float(seconds["context_conditioning"]),
                       "candidate": float(seconds["candidate_conditioning_and_generation"]),
                       "peak_memory_bytes": int(query["peak_memory_bytes"]),
                       "candidate_count": int(query["candidate_count"])}
        if stage_split:
            rows[index].update(stage_seconds={k: float(query["stage_seconds"][k]) for k in STAGES},
                               stage_sum=float(query["stage_sum_seconds"]),
                               stage_peak_memory=query["stage_peak_memory"])
    return rows


def split_comparison(headline: dict, split: dict, tolerance: float = SPLIT_TOLERANCE) -> dict:
    """The split pass's core totals vs the uninstrumented headline's: publishable only if every query AND the
    aggregate agree within ``tolerance``; the aggregate-only rule and the split's internal stage-sum check are
    reported beside it."""
    if set(headline) != set(split):
        raise RuntimeError("split / headline query coverage mismatch")
    ordered = sorted(headline)
    rel = np.array([(split[i]["core"] - headline[i]["core"]) / headline[i]["core"] for i in ordered])
    total = sum(headline[i]["core"] for i in ordered)
    aggregate = (sum(split[i]["core"] for i in ordered) - total) / total
    internal = np.array([(split[i]["stage_sum"] - split[i]["core"]) / split[i]["core"] for i in ordered])
    strict = bool(abs(aggregate) <= tolerance and (np.abs(rel) <= tolerance).all())
    return {"tolerance": tolerance, "aggregate_relative_difference": float(aggregate),
            "per_query_relative_difference": {str(i): float(v) for i, v in zip(ordered, rel)},
            "per_query_max_abs": float(np.abs(rel).max()), "per_query_median_abs": float(np.median(np.abs(rel))),
            "per_query_within_fraction": float((np.abs(rel) <= tolerance).mean()),
            "internal_stage_sum_vs_core_max_abs": float(np.abs(internal).max()),
            "aggregate_rule_pass": bool(abs(aggregate) <= tolerance), "strict_rule_pass": strict,
            "split_publishable": strict}


def stage_breakdown(split: dict, selector: dict, counts: dict) -> dict:
    """Per stage (+ the AGREE selector): summed s, s/query, dataset-wide ms/candidate, fraction of the total; the
    memory segments' absolute peaks (max over queries -- peaks are never added)."""
    ordered, total = sorted(counts), sum(counts.values())
    sums = {name: sum(split[i]["stage_seconds"][name] for i in ordered) for name in STAGES}
    sums["agree_selector"] = sum(selector[i] for i in ordered)
    grand = sum(sums.values())
    seconds = {name: {"summed_seconds": float(value), "mean_seconds_per_query": float(value / len(ordered)),
                      "ms_per_candidate_dataset_wide": float(1000.0 * value / total), "fraction": float(value / grand)}
               for name, value in sums.items()}
    peaks = {}
    for i in ordered:
        for name, peak in split[i]["stage_peak_memory"].items():
            if peak is None or peak.get("allocated_bytes") is None:
                continue
            old = peaks.get(name, {"allocated_bytes": 0, "reserved_bytes": 0})
            peaks[name] = {k: max(old[k], int(peak[k])) for k in ("allocated_bytes", "reserved_bytes")}
    return {"seconds": seconds, "peak_memory": peaks}


def load_c2f_run(run_dir: Path, *, arm: str, selection: dict, audit: dict, top_k: int = 5) -> dict:
    """``{query index: {c2f_total, core, union_count, generated_count}}`` of one validated c2f sequential pass."""
    run_dir = Path(run_dir)
    manifest = _hashed(run_dir / "run_manifest.json")
    identity = manifest["identity"]
    want = ident.registered_identity(arm, selection=selection, audit=audit, seed=42, timing=True,
                                     candidate_subset="c2f", c2f_top_k=top_k)
    bad = ident.identity_reasons(identity, want, audit)
    if bad:
        raise RuntimeError(f"{run_dir}: {bad} (the {arm} c2f k = {top_k} registered protocol)")
    rows = {}
    for record in selection["records"]:
        index = int(record["index"])
        query = _hashed(run_dir / "queries" / f"query_{index:05d}.json")
        protocol = query.get("latency_protocol") or {}
        if query.get("run_manifest_sha256") != manifest["sha256"] or query.get("query_id") != record["query_id"] or \
                protocol.get("name") != "exp37_c2f_sequential_kctx8_kgen1" or protocol.get("top_k") != int(top_k) or \
                len(query["c2f"]["union_indices"]) != int(query["candidate_count"]):
            raise RuntimeError(f"{run_dir}: query {index} is not a c2f (k = {top_k}) result of this run / selection")
        rows[index] = {"c2f_total": float(query["latency_seconds"]["c2f_total"]),
                       "core": float(query["latency_seconds"]["core_forward_total"]),
                       "union_count": int(query["candidate_count"]), "generated_count": int(query["c2f"]["generated_count"])}
    return rows
