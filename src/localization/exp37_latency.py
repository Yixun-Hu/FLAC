"""exp_37 latency summary library (plan_loc_efficiency.md §1 formulas, §10 B1 / B2 time axis / CPU co-tenancy).

The bundle's formulas, unchanged (``tools/summarize_core_forward_latency.py``): per query
``L_q = core_forward_total_q + selector_q`` (the AGREE selector's median ``scoring_total_seconds``); s/query = mean /
median / P90 (``np.quantile`` linear) over the queries; ms/candidate dataset-wide = ``sum L / sum |C|``, query median
/ P90 of ``L_q / |C_q|``. Every run is validated against its arm's protocol (conditioning, the pinned local checkpoint,
K_ctx, K_gen 1, the bundle's timing flags, SDPA) and its run-bound co-tenancy attestation (the monitor over the
whole timed interval plus bound host records; ``host_cotenancy`` / ``selector_cotenancy``). Never ``elapsed_seconds``.

Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from src.localization import exp37_identity as ident
from src.localization import exp37_queries as queries
from src.localization.pilot import canonical_sha256

ARM_PROTOCOL = ident.ARMS
SELECTOR_RECORD_TYPES = ("exp37_agree_selector_latency", "localization_selector_latency_kctx8_kgen1")
# a selection served by a selector vector measured on a registered superset (records contained identically)
REGISTERED_SUPERSETS = {
    "6eeeec401c3f63e47ab446b4b43efe2f1db260bad9decf74de423d0872b659a2": {        # the 64-query seed-42 pilot
        "sha256": "9bf75810a7fe6972049650748e015f9856b586f0c66d0554ef1b67c6bf8e041b",   # the 128-query selection
        "path": "NeurIPS_2026_Workshop_Latency_Reproducibility/data/selection/frozen_16room_128.json"},
}
SELECTOR_FIELDS = ("observed_encode_seconds", "generated_encode_seconds", "similarity_seconds", "argmax_seconds",
                   "scoring_total_seconds")
SELECTOR_TIMED_FUNCTION = "tools/benchmark_localization_selector_latency.py::benchmark_agree_query"
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


def _claimed_selection(payload: dict, selection: dict, path) -> dict:
    """The selection a vector claims: this one, or its REGISTERED superset (loaded from its registered path and
    hash-checked) of which this selection must be an ordered subsequence."""
    claimed = payload.get("selection_sha256")
    if claimed == selection["sha256"]:
        return selection
    entry = REGISTERED_SUPERSETS.get(selection["sha256"])
    if not entry or claimed != entry["sha256"]:
        raise RuntimeError(f"{path}: measured on another selection that is not a registered superset of this one")
    location = Path(entry["path"])
    superset = _hashed(location if location.is_absolute() else ident.REPO_ROOT / location)
    if superset["sha256"] != entry["sha256"] or not ordered_subsequence(selection["records"], superset["records"]):
        raise RuntimeError(f"{path}: the registered superset on disk is not the claimed one or does not contain "
                           "this selection identically and in order")
    return superset


def _selector_row_reasons(row: dict, *, peaks: bool) -> list[str]:
    agree = row.get("agree") or {}
    repeats, medians = agree.get("repeat_seconds"), agree.get("median_seconds")
    if not isinstance(repeats, dict) or not isinstance(medians, dict) or set(repeats) != set(SELECTOR_FIELDS) or \
            set(medians) != set(SELECTOR_FIELDS):
        return ["the repeat / median fields are not the bundle's five"]
    if not all(isinstance(repeats[f], list) and len(repeats[f]) == 3 and all(queries.finite_nonnegative(v)
                                                                            for v in repeats[f]) for f in repeats):
        return ["a field does not hold three finite nonnegative repeats"]
    bad = []
    if any(medians[f] != float(statistics.median(float(v) for v in repeats[f])) for f in SELECTOR_FIELDS):
        bad.append("a recorded median is not the median of its three repeats")
    if any(repeats["scoring_total_seconds"][r] != repeats["observed_encode_seconds"][r]
           + repeats["generated_encode_seconds"][r] + repeats["similarity_seconds"][r] + repeats["argmax_seconds"][r]
           for r in range(3)):
        bad.append("a repeat's scoring total is not the sum of its four parts")
    if peaks and not all(isinstance(row.get(k), int) and not isinstance(row.get(k), bool) and row[k] >= 0
                         for k in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes")):
        bad.append("the peak memory fields are not nonnegative integers")
    return bad


def load_selector(path: Path, selection: dict, *, bundle: bool = False) -> dict:
    """Per selection query: the selector term and its peaks, from a vector validated in full (Codex code review 2,
    HIGH 3): the registered exp_37 vector (``bundle``: the latency bundle's own, for its oracle test only) on the
    registered AGREE / context manifest, measured on this selection or its registered superset, whose rows UNIQUELY
    and EXACTLY cover the claimed selection with the same query / room / count / candidate hash and complete,
    recomputable repeat evidence."""
    payload = _hashed(path)
    if payload.get("record_type") != ("localization_selector_latency_kctx8_kgen1" if bundle
                                      else "exp37_agree_selector_latency"):
        raise RuntimeError(f"{path}: not the expected selector record type")
    if not bundle and payload.get("timed_function") != SELECTOR_TIMED_FUNCTION:
        raise RuntimeError(f"{path}: the timed function is not the bundle's benchmark_agree_query")
    if payload.get("device") != "cuda:0" or payload.get("repeat_count") != 3 or payload.get("candidate_batch_size") != 64:
        raise RuntimeError(f"{path}: not a 3-repeat, batch-64 AGREE selector record on device cuda:0")
    if payload.get("agree_checkpoint_file_sha256") != ident.AGREE_SHA256:
        raise RuntimeError(f"{path}: not the registered AGREE checkpoint")
    if payload.get("context_manifest_sha256") != ident.CONTEXT_MANIFEST_SHA256:
        raise RuntimeError(f"{path}: not the registered context manifest")
    claimed = _claimed_selection(payload, selection, path)
    rows = payload.get("queries") or []
    indices = [int(row["query_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"{path}: duplicate query rows")
    if payload.get("query_count") != len(rows):
        raise RuntimeError(f"{path}: query_count does not match its rows")
    if sorted(indices) != sorted(int(r["index"]) for r in claimed["records"]):
        raise RuntimeError(f"{path}: the rows' coverage is not exactly the claimed selection")
    by_index = {int(row["query_index"]): row for row in rows}
    for record in claimed["records"]:
        row = by_index[int(record["index"])]
        if any(row.get(k) != record.get(k) for k in ("query_id", "room", "candidate_count", "candidate_indices_sha256")) \
                or not record.get("candidate_indices_sha256"):
            raise RuntimeError(f"{path}: selector identity mismatch at query {record['index']}")
        bad = _selector_row_reasons(row, peaks=not bundle)
        if bad:
            raise RuntimeError(f"{path}: query {record['index']}: {bad}")
    return {int(r["index"]): {"scoring_total_seconds": float(by_index[int(r["index"])]["agree"]["median_seconds"][
        "scoring_total_seconds"]), "peak_memory_allocated_bytes": by_index[int(r["index"])].get("peak_memory_allocated_bytes"),
        "peak_memory_reserved_bytes": by_index[int(r["index"])].get("peak_memory_reserved_bytes")}
        for r in selection["records"]}


def snapshot_reasons(snapshot, producer: dict, label: str) -> list[str]:
    """A host snapshot taken by the producer itself on its card, with both host commands successful."""
    from src.localization import exp37_cotenancy as cot

    if not isinstance(snapshot, dict):
        return [f"{label}: no host snapshot"]
    bad = []
    if (snapshot.get("pid"), snapshot.get("hostname")) != (producer["pid"], producer["hostname"]) or cot._uuid(
            (snapshot.get("cuda_device") or {}).get("uuid")) != cot._uuid(producer["card_uuid"]):
        bad.append(f"{label}: the snapshot's pid / host / card are not the producer's")
    for command in ("gpu_compute_apps", "top_cpu_processes"):
        done = snapshot.get(command) or {}
        if done.get("error") or done.get("returncode") != 0 or not isinstance(done.get("lines"), list):
            bad.append(f"{label}: the host command {command} failed or is incomplete")
    return bad


def host_cotenancy(run_dir: Path) -> dict:
    """A run's timing attestation (Codex code reviews 1 / 2, BLOCKING 3 / HIGH 1): the producer record (pid + /proc
    start time, host, card; bound to the run manifest before the measurement), the monitor's verdict over the whole
    timed interval for exactly that producer, and exactly one hashed start / loaded / end host record, each bound to the
    run, labelled as its file, taken by the producer on its card with both host commands successful."""
    from src.localization import exp37_cotenancy as cot

    run_dir = Path(run_dir)
    try:
        manifest = _hashed(run_dir / "run_manifest.json")
        producer = _hashed(run_dir / "producer.json")
    except (OSError, ValueError, RuntimeError) as error:
        return {"timing_valid": False, "reasons": [f"no readable run manifest / producer record: {error}"]}
    reasons, records = [], {}
    if producer.get("record_type") != "exp37_producer" or producer.get("run_manifest_sha256") != manifest["sha256"]:
        reasons.append("the producer record is not this run's")
    for path in sorted((run_dir / "host_records").glob("*.json")):
        try:
            record = _hashed(path)
        except (OSError, ValueError, RuntimeError):
            reasons.append(f"host record {path.name} is unhashed or corrupt")
            continue
        if record.get("run_manifest_sha256") != manifest["sha256"] or record.get("label") != path.name.split("_")[0]:
            reasons.append(f"host record {path.name} is bound to another run or mislabelled")
        reasons += snapshot_reasons(record.get("snapshot"), producer, f"host record {path.name}")
        records[path.name] = record["sha256"]
    labels = sorted(name.split("_")[0] for name in records)
    if labels != ["end", "loaded", "start"]:
        reasons.append(f"the host records {labels} are not exactly start / loaded / end")
    verdict = cot.monitor_verdict(run_dir / "cotenancy", binding_sha256=manifest["sha256"],
                                  producer={k: producer.get(k) for k in cot.PRODUCER_KEYS})
    return {"timing_valid": bool(verdict["timing_valid"] and not reasons), "reasons": verdict["reasons"] + reasons,
            "monitor": verdict, "host_records": records, "producer_sha256": producer["sha256"]}


SELECTOR_BOUND = ("selection_sha256", "context_manifest_sha256", "agree_checkpoint_file_sha256", "candidate_batch_size",
                  "repeat_count", "device")


def selector_cotenancy(path: Path) -> dict:
    """A selector vector's attestation: its binding (hashed before the measurement: this file, the payload's selection /
    AGREE / context / batch / repeats / device, and the producer pid + /proc start time, host, card), the monitor
    verdict in ``<file>.cotenancy`` for exactly that producer, and the payload's host snapshot by the same producer."""
    from src.localization import exp37_cotenancy as cot

    path = Path(path)
    try:
        payload = _hashed(path)
    except (OSError, ValueError, RuntimeError) as error:
        return {"timing_valid": False, "reasons": [f"unreadable selector vector: {error}"]}
    binding = payload.get("binding")
    if not isinstance(binding, dict) or binding.get("sha256") != canonical_sha256(
            {k: v for k, v in binding.items() if k != "sha256"}):
        return {"timing_valid": False, "reasons": ["the selector vector has no hashed binding (no monitored run)"]}
    differ = [k for k in SELECTOR_BOUND if binding.get(k) != payload.get(k)]
    if Path(str(binding.get("output", ""))).name != path.name or binding.get("record_type") != "exp37_selector_binding":
        differ.append("output")
    if differ:
        return {"timing_valid": False, "reasons": [f"the selector binding does not describe this vector: {differ}"]}
    producer = {k: binding.get(k) for k in cot.PRODUCER_KEYS}
    verdict = cot.monitor_verdict(Path(f"{path}.cotenancy"), binding_sha256=binding["sha256"], producer=producer)
    reasons = verdict["reasons"] + (snapshot_reasons(payload.get("host"), producer, "the selector's host snapshot")
                                    if not producer_missing(producer) else [])
    return {"timing_valid": bool(verdict["timing_valid"] and not reasons), "reasons": reasons, "monitor": verdict,
            "binding_sha256": binding["sha256"]}


def producer_missing(producer: dict) -> bool:
    return any(producer.get(k) in (None, "") for k in ("pid", "pid_start_ticks", "hostname", "card_uuid"))


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
        if subset_manifest is None:
            queries.check_query(query, record, run_sha256=manifest["sha256"], label=f"{run_dir}: ")
        else:
            sub = subset_rows.get(index) or {}
            if sub.get("query_id") != record["query_id"] or (query.get("candidate_subset") or {}).get(
                    "indices_sha256") != sub.get("candidate_indices_sha256"):
                raise RuntimeError(f"{run_dir}: query {index} ran another candidate subset than the subset manifest's")
            queries.check_query(query, record, run_sha256=manifest["sha256"], label=f"{run_dir}: ",
                                active_sha256=sub["candidate_indices_sha256"], count=int(sub["candidate_count"]))
        given = query.get("latency_protocol") or {}
        if any(given.get(k) != v for k, v in protocol.items()) or bool(given.get("stage_split")) != stage_split:
            raise RuntimeError(f"{run_dir}: query {index} latency_protocol is not {protocol} (stage_split={stage_split})")
        seconds = query["latency_seconds"]
        if not all(queries.finite_nonnegative(seconds.get(k)) for k in (
                "core_forward_total", "context_conditioning", "candidate_conditioning_and_generation")) or not \
                isinstance(query.get("peak_memory_bytes"), int) or query["peak_memory_bytes"] < 0:
            raise RuntimeError(f"{run_dir}: query {index} latency / memory fields are not finite nonnegative numbers")
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


def _arrays(run_dir: Path, query: dict):
    path = Path(run_dir) / query["arrays_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != query.get("arrays_sha256"):
        raise RuntimeError(f"{path} does not match its query's arrays_sha256")
    with np.load(path) as arrays:
        return arrays["candidates"], arrays["similarities"]


C2F_PARTS = ("observed_encoding", "context_conditioning", "coarse_generation", "coarse_scoring", "selection",
             "refined_generation", "refined_scoring", "union_argmax")
C2F_SELECTOR_PARTS = ("observed_encoding", "coarse_scoring", "selection", "refined_scoring", "union_argmax")
C2F_STATISTIC = "per-component median over the repeats on the exact generated tensors (the bundle selector's median-of-three)"


def c2f_protocol(top_k: int) -> dict:
    """The exact latency_protocol block a c2f timing pass records (exp37_c2f.execute_query_c2f)."""
    return {"name": "exp37_c2f_sequential_kctx8_kgen1", "context_count": 8, "top_k": int(top_k),
            "chebyshev_radius_m": 0.5, "observed_encoding_included": True, "candidate_scoring_included": True,
            "input_loading_included": False, "candidate_filtering_included": False, "selector_repeat_count": 3,
            "selector_statistic": C2F_STATISTIC}


def _c2f_timing_reasons(query: dict, top_k: int) -> list[str]:
    """Review 2 HIGH 5: the exact repeat-evidence and latency schema of a c2f timing record."""
    bad = []
    if query.get("latency_protocol") != c2f_protocol(top_k):
        bad.append("the latency protocol block is not the registered c2f one")
    repeats, seconds = query.get("selector_repeat_seconds"), query.get("latency_seconds")
    if not isinstance(repeats, dict) or set(repeats) != set(C2F_SELECTOR_PARTS) or not all(
            isinstance(repeats[p], list) and len(repeats[p]) == 3 and all(queries.finite_nonnegative(v) for v in repeats[p])
            for p in repeats):
        return bad + ["the selector repeat evidence is not the five parts x three finite nonnegative samples"]
    if not isinstance(seconds, dict) or set(seconds) != set(C2F_PARTS) | {"c2f_total", "core_forward_total"} or \
            not all(queries.finite_nonnegative(v) for v in seconds.values()):
        return bad + ["the latency components are not the registered finite nonnegative set"]
    if any(seconds[p] != statistics.median(repeats[p]) for p in C2F_SELECTOR_PARTS):
        bad.append("a selector part is not the median of its three repeats")
    if seconds["c2f_total"] != float(sum(seconds[p] for p in C2F_PARTS)):
        bad.append("c2f_total is not the sum of the components")
    if seconds["core_forward_total"] != float(seconds["context_conditioning"] + seconds["coarse_generation"]
                                              + seconds["refined_generation"]):
        bad.append("core_forward_total is not context + coarse + refined generation")
    return bad


def _c2f_reasons(query: dict, full, cands, sims, top_k: int) -> list[str]:
    """Every registered property of one c2f record, recomputed from its own arrays and the parent's full grid."""
    from src.localization import exp37_grid as grid
    c2f, bad = query["c2f"], []
    union = np.asarray(c2f["union_indices"])
    if union.ndim != 1 or not len(union) or np.any(np.diff(union) <= 0) or union[0] < 0 or union[-1] >= len(full):
        return ["the union indices are not sorted, unique and inside the parent grid"]
    if hashlib.sha256(union.astype("<u4").tobytes()).hexdigest() != c2f["union_indices_sha256"]:
        bad.append("the recorded union hash is not the union's")
    if not np.array_equal(cands, full[union]) or cands.dtype != full.dtype:
        bad.append("the stored candidates are not the parent's full[union] rows")
    scores = np.full(len(full), np.nan)
    scores[union] = grid.query_scores(sims, 0.1)
    coarse = grid.nested_subset(full)
    top = grid.top_k(scores, coarse, top_k) if not np.isnan(scores[coarse]).any() else np.array([], dtype=int)
    if top.tolist() != c2f["top_indices"]:
        bad.append("the top-k indices do not follow from the stored coarse scores")
    near = np.zeros(len(full), dtype=bool)
    for index in top:
        near |= np.abs(full.astype(np.float64) - full[index].astype(np.float64)).max(axis=1) <= grid.CHEBYSHEV_RADIUS_M \
            + grid.TOLERANCE_M
    refined = np.flatnonzero(near)
    new = np.setdiff1d(refined, coarse)
    if not np.array_equal(np.union1d(coarse, refined), union):
        bad.append("the union is not the coarse set plus the Chebyshev-0.5 m refinement")
    counts = (c2f["full_candidate_count"], c2f["coarse_count"], c2f["refined_count"], c2f["new_count"],
              c2f["generated_count"], len(union))
    if counts != (len(full), len(coarse), len(refined), len(new), len(coarse) + len(new), int(query["candidate_count"])):
        bad.append("the full / coarse / refined / generated / union counts disagree")
    if int(union[int(np.argmax(scores[union]))]) != c2f["winner_full_index"]:
        bad.append("the winner is not the union argmax")
    if grid.recomputed_k1_metrics(cands, sims, query["source_global"], 0.1) != query["metrics"]["1"]:
        bad.append("the recorded K=1 block is not reproduced")
    if (c2f.get("top_k"), c2f.get("chebyshev_radius_m")) != (top_k, 0.5):
        bad.append("the radius / top_k of the c2f record are not the registered ones")
    return bad


def load_c2f_run(run_dir: Path, *, arm: str, selection: dict, audit: dict, parent_run_dir: Path, top_k: int = 5,
                 seed: int = 42, timing: bool = True) -> dict:
    """``{query index: {c2f_total, core, union_count, generated_count, readout}}`` of one c2f run -- the seed-42
    sequential TIMING pass (``timing``) or an accuracy-only c2f run of seed 43 / 44 -- every record re-verified
    against its PARENT (the same arm's 0.5 m run of the same seed, itself identity-checked): the selection rule (full
    parent records, subset c2f records with the recomputed union hash), the union / top-k / refinement / counts /
    winner / K=1 block, and for a timing pass the exact repeat and latency schema. ``readout`` is the query's
    accuracy row (fresh oracle over its union; the c2f point's accuracy of record)."""
    from src.localization import exp37_grid as grid
    run_dir, parent_run_dir = Path(run_dir), Path(parent_run_dir)
    manifest, parent = _hashed(run_dir / "run_manifest.json"), _hashed(parent_run_dir / "run_manifest.json")
    mode = None if timing else "inline"
    for label, payload, extra in (("c2f", manifest, {"candidate_subset": "c2f", "c2f_top_k": top_k}),
                                  ("parent", parent, {})):
        want = ident.registered_identity(arm, selection=selection, audit=audit, seed=seed, timing=timing,
                                         waveform_mode=mode, **extra)
        bad = ident.identity_reasons(payload["identity"], want, audit)
        if bad:
            raise RuntimeError(f"{run_dir if label == 'c2f' else parent_run_dir}: {bad} (the {arm} seed-{seed} {label} "
                               f"{'timing' if timing else 'accuracy-only'} registered protocol)")
    rows = {}
    for record in selection["records"]:
        index = int(record["index"])
        query = _hashed(run_dir / "queries" / f"query_{index:05d}.json")
        parent_query = _hashed(parent_run_dir / "queries" / f"query_{index:05d}.json")
        queries.check_query(parent_query, record, run_sha256=parent["sha256"], label=f"{parent_run_dir}: ")
        union = np.asarray(query.get("c2f", {}).get("union_indices", []))
        queries.check_query(query, record, run_sha256=manifest["sha256"], label=f"{run_dir}: ", count=len(union),
                            active_sha256=hashlib.sha256(union.astype("<u4").tobytes()).hexdigest())
        full, _parent_sims = _arrays(parent_run_dir, parent_query)
        cands, sims = _arrays(run_dir, query)
        bad = _c2f_reasons(query, full, cands, sims, int(top_k))
        if timing:
            bad += _c2f_timing_reasons(query, int(top_k))
        elif any(k in query for k in ("latency_seconds", "selector_repeat_seconds", "latency_protocol")):
            bad.append("an accuracy-only c2f record carries timing fields")
        if bad:
            raise RuntimeError(f"{run_dir}: query {index}: {bad}")
        scores = grid.query_scores(sims, 0.1)
        readout = {**grid.subset_readout(cands, scores, np.asarray(query["source_global"]), np.arange(len(cands))),
                   "room": query["room"], "query_index": index}
        seconds = query.get("latency_seconds") or {}
        rows[index] = {"c2f_total": float(seconds["c2f_total"]) if timing else None,
                       "core": float(seconds["core_forward_total"]) if timing else None,
                       "union_count": int(query["candidate_count"]), "generated_count": int(query["c2f"]["generated_count"]),
                       "readout": readout}
    return rows