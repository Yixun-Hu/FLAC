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

from src.localization.pilot import canonical_sha256

ARM_PROTOCOL = {
    "P1": {"cond": "vanilla", "ckpt": "c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c"},
    "YAW": {"cond": "vanilla", "ckpt": "ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7"},
    "BF": {"cond": "fa_invariant", "ckpt": "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328"},
}
SELECTOR_RECORD_TYPES = ("exp37_agree_selector_latency", "localization_selector_latency_kctx8_kgen1")
TIMING_IDENTITY = {"score_sample_counts": [1], "measure_core_forward": True, "synchronize_timing": True,
                   "warmup_query_count": 1, "candidate_batch_size": 64, "tau": 0.1, "attention_backend": "sdpa"}
CPU_COTENANT_PCPU = 10.0
SPLIT_TOLERANCE = 0.02          # plan §10: outside 2 % the split is withheld, never the headline
STAGES = ("context_conditioning", "candidate_conditioning", "dit", "vae_decode")


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


def load_selector(path: Path, selection: dict) -> dict:
    """Per selection query: the selector term and its peak; the vector may cover MORE queries (e.g. the 128 vector
    serving the 64-query pilot), never fewer, and each row must name the same query and candidate count."""
    payload = _hashed(path)
    if payload.get("record_type") not in SELECTOR_RECORD_TYPES or int(payload.get("repeat_count", -1)) != 3 \
            or int(payload.get("candidate_batch_size", -1)) != 64:
        raise RuntimeError(f"{path}: not a 3-repeat, batch-64 AGREE selector record")
    rows = {int(row["query_index"]): row for row in payload["queries"]}
    out = {}
    for record in selection["records"]:
        index = int(record["index"])
        row = rows.get(index)
        if row is None:
            raise RuntimeError(f"{path}: selector coverage misses query {index}")
        if any(row[key] != record[key] for key in ("query_id", "room", "candidate_count")):
            raise RuntimeError(f"{path}: selector identity mismatch at query {index}")
        out[index] = {"scoring_total_seconds": float(row["agree"]["median_seconds"]["scoring_total_seconds"]),
                      "peak_memory_allocated_bytes": row.get("peak_memory_allocated_bytes"),
                      "peak_memory_reserved_bytes": row.get("peak_memory_reserved_bytes")}
    return out


def _norm_uuid(value) -> str:
    text = str(value or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def host_cotenancy(run_dir: Path) -> dict:
    """GPU co-tenants (another pid on the run's card -> invalid; no record or an unidentified card -> invalid) and
    CPU co-tenants (other processes at >= 10 %CPU in the top lists -- recorded for the Planner's judgement)."""
    records = sorted((Path(run_dir) / "host_records").glob("*.json"))
    gpu, cpu, unknown = [], [], not records
    for path in records:
        snapshot = json.loads(path.read_text())["snapshot"]
        pid, card = int(snapshot["pid"]), _norm_uuid((snapshot.get("cuda_device") or {}).get("uuid"))
        unknown |= not card or "lines" not in snapshot["gpu_compute_apps"]
        for line in snapshot["gpu_compute_apps"].get("lines", []):
            fields = [f.strip() for f in line.split(",")]
            if _norm_uuid(fields[0]) == card and int(fields[1]) != pid:
                gpu.append(f"{path.name}: {line}")
        for line in snapshot.get("top_cpu_processes", {}).get("lines", []):
            fields = line.split()
            if int(fields[0]) != pid and float(fields[3]) >= CPU_COTENANT_PCPU and line not in cpu:
                cpu.append(line)
    return {"timing_valid": not gpu and not unknown, "gpu_cotenants": gpu, "cpu_cotenants": cpu,
            "host_record_count": len(records), "unidentified_card": unknown}


def load_arm_run(run_dir: Path, *, arm: str, selection: dict, context_count: int, stage_split: bool = False) -> dict:
    """``{query index: {core, context, candidate, peak_memory_bytes, candidate_count}}`` of one validated run;
    a headline run must be uninstrumented and a split run must be the split pass (plus its stage fields)."""
    run_dir = Path(run_dir)
    manifest = _hashed(run_dir / "run_manifest.json")
    identity = manifest["identity"]
    want = {**TIMING_IDENTITY, "conditioning_method": ARM_PROTOCOL[arm]["cond"],
            "checkpoint_sha256": ARM_PROTOCOL[arm]["ckpt"], "n_context": int(context_count),
            "pilot_manifest_sha256": selection["sha256"],
            "query_indices": [int(r["index"]) for r in selection["records"]]}
    bad = [key for key, value in want.items() if identity.get(key) != value]
    if bool(identity.get("stage_split")) != stage_split:
        bad.append("stage_split")
    if bad:
        raise RuntimeError(f"{run_dir}: identity fields {bad} differ from the {arm} K_ctx={context_count} protocol")
    protocol = {"name": f"fem_core_aligned_kctx{context_count}_kgen1", "context_count": int(context_count)}
    rows = {}
    for record in selection["records"]:
        index = int(record["index"])
        query = _hashed(run_dir / "queries" / f"query_{index:05d}.json")
        if query.get("run_manifest_sha256") != manifest["sha256"] or int(query["candidate_count"]) != int(
                record["candidate_count"]) or query.get("query_id") != record["query_id"]:
            raise RuntimeError(f"{run_dir}: query {index} does not belong to this run / selection")
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
