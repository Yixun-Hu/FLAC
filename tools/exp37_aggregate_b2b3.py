#!/usr/bin/env python3
"""exp_37 B2 / B3 aggregation (plan_loc_efficiency.md §4, §5, §10): one spec JSON in -> a hashed summary JSON, a
markdown table and the two-panel B2 Pareto figure (PNG + PDF).

B2 -- per arm (FLAC P1, OrbitRIR B-F) x grid: accuracy = the pooled 64-query metrics of each seed's run, mean and
sample SD over seeds 42-44 (the 0.5 m runs give the 0.5 m and nested-1.0 m readouts -- offline-exact with full-grid
common random numbers; the 0.75 m runs their own full readout; the c2f point from sequential c2f runs for all three
seeds: the seed-42 timing pass via its replay and the seed-43 / 44 accuracy runs b2a_c2f_*, the parent-derived
offline c2f readout reported only as a parity cross-check and the c2f_k5_offline sensitivity row); time = s/query,
the query-micro mean of L_q = core_forward_total_q + the shape-matched AGREE selector term, measured ONLY at seed 42
(D-a: seeds 43 / 44 are accuracy-only, no timing variance claim; nested 1.0 m from its dedicated subset pass + the
nested selector vector; c2f = its measured sequential total); never ``elapsed_seconds``. The Pareto frontier is computed over these internal points only; FewShotRIR and FEM are
external reference markers. c2f k in {1, 3, 10} is a post-primary sensitivity (accuracy only).
B3 -- OrbitRIR orbit {0} / {0,180} / C4 x theta {0, 90} x seeds: accuracy per (orbit, theta); orbit latency from
B1-style passes on the 128-query selection. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization import exp37_aggregate as agg  # noqa: E402
from src.localization import exp37_identity as ident  # noqa: E402
from src.localization import exp37_grid as grid  # noqa: E402
from src.localization import exp37_latency as lat  # noqa: E402
from src.localization import exp37_sources as sources  # noqa: E402
from src.localization.pilot import canonical_sha256, load_pilot_manifest  # noqa: E402

ARMS = ("P1", "BF")
LABELS = {"P1": "FLAC (P1@40k)", "BF": "OrbitRIR (B-F@40k)"}
GRIDS = ("0.50", "0.75", "1.00_nested", "c2f_k5")
GRID_LABELS = {"0.50": "0.5 m", "0.75": "0.75 m", "1.00_nested": "1.0 m (nested)", "c2f_k5": "c2f (1.0 -> 0.5 m, k = 5)"}
SENSITIVITY = ("c2f_k1", "c2f_k3", "c2f_k10", "c2f_k5_offline")    # offline readouts of the 0.5 m runs (accuracy only)
METRICS = ("median_localization_error_m", "success_0_5m", "success_1_0m", "oracle_normalized_success_0_5m",
           "mean_oracle_error_m", "mean_candidate_count")
ORBITS = {"0": [0.0], "0_180": [0.0, 180.0], "c4": [0.0, 90.0, 180.0, 270.0]}
EXP23_SUMMARY_SHA256 = "674998f9e3e76fad2641e6b164a31dca1b73d0cfb3a0a5120b3f0e180c514ad2"   # its canonical sha256
LATENCY_CSV_SHA256 = "3cac4a9c92f0943a57c0a228f94ca4d8d9357ea244214e202238cdb01b0191b8"     # = the bundle's SHA256SUMS

# the registered coverage (Codex code reviews 1 / 2, BLOCKING 4, HIGH 6): every cell exactly once, n = 3 seeds
SPEC_SCHEMA = 3
SEEDS = (42, 43, 44)
B2_GRIDS = ("0.50", "0.75")
TIMING_KINDS = ("nested_1m", "c2f")
C2F_ACCURACY_SEEDS = (43, 44)
B3_ORBITS = ("0", "0_180", "c4")
B3_THETAS = (0, 90)
SYNTHESIS_K = (1, 8)
INPUT_KEYS = ("pilot_0p50", "pilot_0p75", "audit_0p50", "audit_0p75", "nested_manifest", "selector_0p50",
              "selector_0p75", "selector_nested")
TOP_KEYS = ("schema_version", *INPUT_KEYS, "b2_runs", "b2_timing", "b2_c2f_runs", "b3_runs", "b3_latency",
            "b3_synthesis", "external", "require_timing_replays")


class SpecError(ValueError):
    """The spec is not the registered schema / coverage (nothing is loaded or written)."""


def _path(value) -> bool:
    return isinstance(value, str) and bool(value)


def _coverage(label, keys, want) -> list[str]:
    missing = sorted(set(want) - set(keys), key=str)
    extra = sorted({k for k in keys if k not in want or keys.count(k) > 1}, key=str)
    return [] if not missing and not extra else [f"{label} must be exactly {len(want)} registered cells "
                                                 f"(missing {missing}, extra or duplicated {extra})"]


def _entries(spec, label, keys) -> tuple[list[dict], list[str]]:
    items = spec.get(label)
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return [], [f"{label} must be a list of objects"]
    bad = [f"{label} entry {i} keys are not exactly {sorted(keys)}" for i in items if set(i) != set(keys)]
    bad += [f"{label} entry {i}: dir must be a nonempty path" for i in items if not _path(i.get("dir"))]
    return items, bad


def _synthesis_reasons(section) -> list[str]:
    from src.localization import exp37_synthesis as syn
    if not isinstance(section, dict):
        return ["b3_synthesis is missing: fill it from the exp_36 leaf, or mark it pending"]
    if section.get("status") == "pending":
        want = {"status": "pending", "leaf_commit": None, "runbook": None, "metrics_keys": [], "cells": []}
        ok = set(section) == set(want) | {"note"} and all(section[k] == v for k, v in want.items()) and _path(
            section.get("note"))
        return [] if ok else ["b3_synthesis is pending but not exactly the pending placeholder"]
    if section.get("status") != "complete":
        return ["b3_synthesis.status must be pending or complete"]
    if set(section) != {"status", "leaf_commit", "runbook", "metrics_keys", "cells"}:
        return ["b3_synthesis (complete) keys are not exactly status / leaf_commit / runbook / metrics_keys / cells"]
    bad, cells = [], section["cells"] if isinstance(section["cells"], list) else []
    commit = section["leaf_commit"]
    if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):  # 40-hex
        bad.append("b3_synthesis.leaf_commit must be the 40-hex exp_36 leaf commit")
    if not _path(section["runbook"]):
        bad.append("b3_synthesis needs its runbook")
    if section["metrics_keys"] != list(syn.METRICS_KEYS):            # review 3 H-02: exactly, in order
        bad.append(f"b3_synthesis.metrics_keys must be exactly {list(syn.METRICS_KEYS)}, not {section['metrics_keys']!r}")
    if not all(isinstance(c, dict) and set(c) == set(syn.CELL_KEYS) for c in cells):
        return bad + [f"b3_synthesis cells must be objects with exactly the keys {list(syn.CELL_KEYS)}"]
    bad += _coverage("b3_synthesis.cells", [(c["orbit"], c["theta"], c["k_context"]) for c in cells],
                     [(o, t, k) for o in B3_ORBITS for t in B3_THETAS for k in SYNTHESIS_K])
    for c in cells:
        if (c["orbit"], c["theta"], c["k_context"]) not in [(o, t, k) for o in B3_ORBITS for t in B3_THETAS
                                                            for k in SYNTHESIS_K]:
            continue
        name, _block, source = syn.cell_name(c["orbit"], c["theta"], c["k_context"])
        hexes = all(syn.is_hex64(c[k]) for k in syn.SHA_KEY.values())         # lowercase 64-hex (H-02)
        if (c["seed"], c["source"], c["cell"]) != (42, source, name) or not hexes or not all(
                _path(c[k]) for k in syn.SHA_KEY):
            bad.append(f"b3_synthesis cell {name}: seed 42, source {source}, its paths and 64-hex sha256 are required")
    return bad


def check_spec(spec: dict) -> None:
    """The registered schema-3 spec, exactly, before anything is loaded; SpecError lists every deviation."""
    if not isinstance(spec, dict):
        raise SpecError("the spec is not a JSON object")
    bad = [] if set(spec) == set(TOP_KEYS) else [
        f"the spec keys are not exactly the schema-3 keys (missing {sorted(set(TOP_KEYS) - set(spec))}, extra "
        f"{sorted(set(spec) - set(TOP_KEYS))})"]
    if spec.get("schema_version") != SPEC_SCHEMA:
        bad.append(f"schema_version must be {SPEC_SCHEMA}")
    bad += [f"{key} must be a nonempty path" for key in INPUT_KEYS if not _path(spec.get(key))]
    if spec.get("require_timing_replays") is not True:
        bad.append("require_timing_replays must be true (D-c: every timing pass has its artifact replay)")
    runs, more = _entries(spec, "b2_runs", ("arm", "grid", "seed", "timing", "dir", "replay"))
    bad += more + _coverage("b2_runs", [(i.get("arm"), i.get("grid"), i.get("seed")) for i in runs],
                            [(a, g, s) for a in ARMS for g in B2_GRIDS for s in SEEDS])
    for i in runs:
        name = f"b2_runs {i.get('arm')} {i.get('grid')} seed {i.get('seed')}"
        if type(i.get("seed")) is not int:
            bad.append(f"{name}: the seed must be an integer")
        elif i.get("seed") == 42 and not (i.get("timing") is True and _path(i.get("replay"))):
            bad.append(f"{name}: the seed 42 run is the timing pass (timing true) with its replay")
        elif i.get("seed") != 42 and not (i.get("timing") is False and i.get("replay") is None):
            bad.append(f"{name}: an accuracy-only run (D-a: timing false, replay null)")
    timing, more = _entries(spec, "b2_timing", ("arm", "kind", "dir", "replay"))
    bad += more + _coverage("b2_timing", [(i.get("arm"), i.get("kind")) for i in timing],
                            [(a, k) for a in ARMS for k in TIMING_KINDS])
    c2f_runs, more = _entries(spec, "b2_c2f_runs", ("arm", "seed", "dir"))
    bad += more + _coverage("b2_c2f_runs", [(i.get("arm"), i.get("seed")) for i in c2f_runs],
                            [(a, s) for a in ARMS for s in C2F_ACCURACY_SEEDS])
    b3, more = _entries(spec, "b3_runs", ("orbit", "theta", "seed", "dir"))
    bad += more + _coverage("b3_runs", [(i.get("orbit"), i.get("theta"), i.get("seed")) for i in b3],
                            [(o, t, s) for o in B3_ORBITS for t in B3_THETAS for s in SEEDS if (o, t) != ("c4", 0)])
    bad += [f"b3_runs {i.get('orbit')} theta {i.get('theta')!r}: theta and seed must be integers" for i in b3
            if type(i.get("theta")) is not int or type(i.get("seed")) is not int]
    latency = spec.get("b3_latency")
    if not isinstance(latency, dict) or set(latency) != {"selection", "selector", "runs"} or not (
            _path(latency.get("selection")) and _path(latency.get("selector"))):
        bad.append("b3_latency must be exactly {selection, selector, runs} with nonempty paths")
        latency = {"runs": []}
    lruns, more = _entries(latency, "runs", ("orbit", "dir", "replay"))
    bad += [f"b3_latency {m}" for m in more] + _coverage("b3_latency.runs", [r.get("orbit") for r in lruns],
                                                         list(B3_ORBITS))
    bad += [f"{i.get('dir')}: a timing pass needs its artifact replay (D-c)" for i in timing + lruns
            if not _path(i.get("replay"))]
    bad += _synthesis_reasons(spec.get("b3_synthesis"))
    external = spec.get("external")
    if not isinstance(external, dict) or set(external) != {"accuracy", "latency_csv"} or not all(
            _path(v) for v in external.values()):
        bad.append("external must be exactly {accuracy, latency_csv} with nonempty paths")
    if bad:
        raise SpecError("; ".join(bad))


def _hashed(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return payload


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input(path, inventory: dict, key: str) -> dict:
    payload = _hashed(path)
    inventory["inputs"][key] = {"path": str(path), "sha256": payload["sha256"], "file_sha256": _file_sha256(path)}
    return payload


def _selector(path, selection, inventory: dict, key: str):
    rows, host = lat.load_selector(path, selection), lat.selector_cotenancy(path)
    monitor = host.get("monitor") or {}
    inventory["selectors"][key] = {"path": str(path), "sha256": _hashed(path)["sha256"], "file_sha256": _file_sha256(path),
                                   "binding_sha256": host.get("binding_sha256"), "timing_valid": host["timing_valid"],
                                   "monitor_head_sha256": monitor.get("monitor_head_sha256"),
                                   "markers_head_sha256": monitor.get("markers_head_sha256")}
    return {i: r["scoring_total_seconds"] for i, r in rows.items()}, host


def _attested(run_dir, entry: dict) -> bool:
    """The run's co-tenancy attestation into its inventory entry; True when the timing is attested."""
    host = lat.host_cotenancy(run_dir)
    monitor = host.get("monitor") or {}
    entry["attestation"] = {"timing_valid": host["timing_valid"], "reasons": host["reasons"],
                            "monitor_head_sha256": monitor.get("monitor_head_sha256"),
                            "markers_head_sha256": monitor.get("markers_head_sha256"),
                            "foreign_cpu_cores_mean": monitor.get("foreign_cpu_cores_mean"),
                            "host_records": host.get("host_records", {})}
    return bool(host["timing_valid"])


def _timing_only(run_dir, replay, *, arm, selection, audit, **modes) -> dict:
    """A timing-only pass's inventory entry (and its replay, validated when given)."""
    if replay:
        return sources.accuracy_source(run_dir, arm=arm, selection=selection, audit=audit, seed=42, timing=True,
                                       replay_dir=replay, **modes)
    manifest = _hashed(Path(run_dir) / "run_manifest.json")
    return {"run_dir": str(run_dir), "run_manifest_sha256": manifest["sha256"], "timing": True,
            "queries": sources.query_inventory(run_dir, manifest, [int(r["index"]) for r in selection["records"]])}


def _stats(by_seed: dict, label: str) -> dict:
    if sorted(by_seed) != list(SEEDS):
        raise RuntimeError(f"{label}: an accuracy statistic needs exactly the seeds {list(SEEDS)} (has {sorted(by_seed)})")
    return {**{m: agg.seed_stats(by_seed[s][m] for s in SEEDS) for m in METRICS}, "seeds": list(SEEDS)}


def build_b2(spec: dict, inventory: dict) -> tuple[dict, dict]:
    pilot = _input(spec["pilot_0p50"], inventory, "pilot_0p50")
    pilot75 = _input(spec["pilot_0p75"], inventory, "pilot_0p75")
    audits = {"0.50": _input(spec["audit_0p50"], inventory, "audit_0p50"),
              "0.75": _input(spec["audit_0p75"], inventory, "audit_0p75")}
    nested = _input(spec["nested_manifest"], inventory, "nested_manifest")
    indices = [int(r["index"]) for r in pilot["records"]]
    if [int(r["index"]) for r in pilot75["records"]] != indices or nested["parent_pilot_sha256"] != pilot["sha256"]:
        raise RuntimeError("the 0.75 m pilot / nested manifest do not cover the 0.5 m pilot's queries")
    terms = {"0.50": _selector(spec["selector_0p50"], pilot, inventory, "selector_0p50"),
             "0.75": _selector(spec["selector_0p75"], pilot75, inventory, "selector_0p75"),
             "1.00_nested": _selector(spec["selector_nested"], nested, inventory, "selector_nested")}
    summaries = {(a, g): {} for a in ARMS for g in GRIDS + SENSITIVITY}
    seconds, valid = {(a, g): {} for a in ARMS for g in GRIDS}, {(a, g): {} for a in ARMS for g in GRIDS}
    for item in spec["b2_runs"]:
        arm, g, seed, timing, run_dir = item["arm"], item["grid"], int(item["seed"]), bool(item["timing"]), item["dir"]
        selection, audit = (pilot, audits["0.50"]) if g == "0.50" else (pilot75, audits["0.75"])
        entry = sources.accuracy_source(run_dir, arm=arm, selection=selection, audit=audit, seed=seed, timing=timing,
                                        replay_dir=item.get("replay"))
        rows = grid.readout_run(run_dir, pilot) if g == "0.50" else {
            "0.75": grid.readout_run(run_dir, pilot75, full_only=True)["full"]}
        for name, summary in grid.summarize_readout(rows).items():   # c2f_k5 of record = the sequential runs below
            summaries[(arm, "c2f_k5_offline" if name == "c2f_k5" else name)][seed] = summary
        if timing:
            core = {i: r["core"] for i, r in lat.load_arm_run(run_dir, arm=arm, selection=selection, audit=audit,
                                                              seed=seed).items()}
            seconds[(arm, g)][seed] = agg.mean_latency(core, terms[g][0], indices)
            valid[(arm, g)][seed] = _attested(run_dir, entry) and bool(terms[g][1]["timing_valid"])
        inventory["runs"][run_dir] = entry
    parents = {(i["arm"], int(i["seed"])): i["dir"] for i in spec["b2_runs"] if i["grid"] == "0.50"}
    c2f_rows = {a: {} for a in ARMS}
    for item in spec["b2_timing"]:
        arm, kind, run_dir = item["arm"], item["kind"], item["dir"]
        modes = {"candidate_subset": kind, **({"c2f_top_k": 5} if kind == "c2f" else {})}
        entry = _timing_only(run_dir, item.get("replay"), arm=arm, selection=pilot, audit=audits["0.50"], **modes)
        if kind == "nested_1m":
            core = {i: r["core"] for i, r in lat.load_arm_run(run_dir, arm=arm, selection=pilot, audit=audits["0.50"],
                                                              subset_manifest=nested).items()}
            key, value = (arm, "1.00_nested"), agg.mean_latency(core, terms["1.00_nested"][0], indices)
            selector_ok = bool(terms["1.00_nested"][1]["timing_valid"])
        else:
            rows = lat.load_c2f_run(run_dir, arm=arm, selection=pilot, audit=audits["0.50"], top_k=5,
                                    parent_run_dir=parents[(arm, 42)], seed=42, timing=True)
            key, value, selector_ok = (arm, "c2f_k5"), sum(rows[i]["c2f_total"] for i in indices) / len(indices), True
            c2f_rows[arm][42] = (run_dir, rows)
        seconds[key][42], valid[key][42] = value, _attested(run_dir, entry) and selector_ok
        inventory["runs"][run_dir] = entry
    for item in spec["b2_c2f_runs"]:                    # review 2 BLOCKING 3: the c2f accuracy is sequential, n = 3
        arm, seed, run_dir = item["arm"], int(item["seed"]), item["dir"]
        inventory["runs"][run_dir] = sources.accuracy_source(run_dir, arm=arm, selection=pilot, audit=audits["0.50"],
                                                             seed=seed, timing=False, candidate_subset="c2f", c2f_top_k=5)
        c2f_rows[arm][seed] = (run_dir, lat.load_c2f_run(run_dir, arm=arm, selection=pilot, audit=audits["0.50"],
                                                         top_k=5, parent_run_dir=parents[(arm, seed)], seed=seed,
                                                         timing=False))
    parity = {a: {} for a in ARMS}
    for arm in ARMS:
        for seed, (run_dir, rows) in sorted(c2f_rows[arm].items()):
            summaries[(arm, "c2f_k5")][seed] = grid.summarize_readout(
                {"c2f_k5": [rows[i]["readout"] for i in indices]})["c2f_k5"]
            parity[arm][str(seed)] = c2f_parity(run_dir, parents[(arm, seed)], pilot)
    points = []
    for arm in ARMS:
        for g in GRIDS:
            times = seconds[(arm, g)]
            if set(times) != {42}:
                raise RuntimeError(f"B2 {arm} {g}: the time axis is the seed-42 pass only (D-a)")
            points.append({"arm": arm, "label": LABELS[arm], "grid": g, "grid_label": GRID_LABELS[g],
                           "accuracy": _stats(summaries[(arm, g)], f"B2 {arm} {g}"), "seconds_per_query": times[42],
                           "timing_valid": bool(valid[(arm, g)][42])})                    # the seed-42 headline
    sensitivity = [{"arm": a, "grid": g, "accuracy": _stats(summaries[(a, g)], f"B2 {a} {g}"),
                    "post_primary_sensitivity": True} for a in ARMS for g in SENSITIVITY]
    return {"query_count": len(indices), "points": points, "sensitivity": sensitivity, "c2f_parity": parity}, summaries


def c2f_parity(c2f_dir, parent_dir, selection: dict) -> dict:
    """The REPORTED cross-check of a sequential c2f run against its parent 0.5 m run's offline c2f (never a gate):
    per query, whether the parent-derived top-k / union and winner equal the run's and whether the run's union
    similarities are bitwise the parent's; the largest absolute similarity difference over equal unions."""
    union_equal = winner_equal = bitwise = 0
    delta = 0.0
    for record in selection["records"]:
        name = f"query_{int(record['index']):05d}.json"
        query, parent = _hashed(Path(c2f_dir) / "queries" / name), _hashed(Path(parent_dir) / "queries" / name)
        cands, sims = lat._arrays(c2f_dir, query)
        full, parent_sims = lat._arrays(parent_dir, parent)
        derived = grid.c2f_indices(full, grid.query_scores(parent_sims, 0.1), k=5)["union"]
        union = np.asarray(query["c2f"]["union_indices"])
        winner = int(derived[int(np.argmax(grid.query_scores(parent_sims[derived], 0.1)))])
        union_equal += int(np.array_equal(derived, union))
        winner_equal += int(winner == int(query["c2f"]["winner_full_index"]))
        if np.array_equal(derived, union):
            bitwise += int(np.array_equal(parent_sims[union], sims))
            delta = max(delta, float(np.max(np.abs(parent_sims[union].astype(np.float64) - sims.astype(np.float64)))))
    return {"queries": len(selection["records"]), "union_equal": union_equal, "winner_equal": winner_equal,
            "bitwise_equal": bitwise, "max_abs_similarity_delta": delta}


def frontier(points: list) -> dict:
    """The internal Pareto frontier -- only ever over attested timings (build_summary computes it only then)."""
    if not all(p["timing_valid"] for p in points):
        raise RuntimeError("a Pareto frontier is never computed from an unattested timing")
    return {metric: agg.pareto_frontier([(p["seconds_per_query"], p["accuracy"][metric]["mean"]) for p in points],
                                        minimize_y=minimize)
            for metric, minimize in (("median_localization_error_m", True), ("success_1_0m", False))}


def _synthesis(section: dict, inventory: dict) -> dict:
    """The B3 synthesis section: pending (diagnostic only) or validated cell by cell from its own artifacts."""
    if section["status"] != "complete":
        return {"status": "pending", "note": section.get("note"), "cells": []}
    from src.localization import exp37_synthesis as syn
    out = syn.validate_section(section)
    for cell in section["cells"]:
        for key, sha_key in syn.SHA_KEY.items():
            inventory["synthesis"][cell[key]] = cell[sha_key]
    return out


def build_b3(spec: dict, b2_summaries: dict, inventory: dict) -> dict:
    pilot, audit = _hashed(spec["pilot_0p50"]), _hashed(spec["audit_0p50"])
    cells = {("c4", 0): {s: b2_summaries[("BF", "0.50")][s] for s in SEEDS}}       # C4 theta 0 = B2's B-F 0.50
    for item in spec["b3_runs"]:
        orbit, theta, seed, run_dir = item["orbit"], int(item["theta"]), int(item["seed"]), item["dir"]
        inventory["runs"][run_dir] = sources.accuracy_source(run_dir, arm="BF", selection=pilot, audit=audit, seed=seed,
                                                             timing=False, frame_angles=ORBITS[orbit],
                                                             rotate_deg=float(theta))
        cells.setdefault((orbit, theta), {})[seed] = grid.summarize_readout(
            {"full": grid.readout_run(run_dir, pilot, full_only=True)["full"]})["full"]
    accuracy = [{"orbit": o, "theta": t, "angles": ORBITS[o], "accuracy": _stats(cells[(o, t)], f"B3 {o} theta {t}")}
                for o in B3_ORBITS for t in B3_THETAS]
    selection = _input(spec["b3_latency"]["selection"], inventory, "latency_selection")
    terms, selector_host = _selector(spec["b3_latency"]["selector"], selection, inventory, "latency_selector")
    counts = {int(r["index"]): int(r["candidate_count"]) for r in selection["records"]}
    latency = []
    for item in spec["b3_latency"]["runs"]:
        orbit, run_dir = item["orbit"], item["dir"]
        entry = _timing_only(run_dir, item.get("replay"), arm="BF", selection=selection, audit=audit,
                             frame_angles=ORBITS[orbit])
        rows = lat.load_arm_run(run_dir, arm="BF", selection=selection, audit=audit, frame_angles=ORBITS[orbit])
        stats = lat.latency_statistics({i: r["core"] for i, r in rows.items()}, terms, counts)
        latency.append({"orbit": orbit, "angles": ORBITS[orbit], "overall": stats["overall"],
                        "per_candidate": stats["per_candidate"],
                        "timing_valid": _attested(run_dir, entry) and bool(selector_host["timing_valid"])})
        inventory["runs"][run_dir] = entry
    return {"accuracy": accuracy, "latency": latency, "synthesis": _synthesis(spec["b3_synthesis"], inventory)}


def load_external(external: dict, pilot: dict, inventory: dict) -> dict:
    """The FewShotRIR / FEM reference markers from the PINNED exp_23 summary and bundle latency CSV."""
    summary = _hashed(external["accuracy"])
    if summary["sha256"] != EXP23_SUMMARY_SHA256:
        raise RuntimeError(f"{external['accuracy']}: not the pinned exp_23 summary ({EXP23_SUMMARY_SHA256[:12]}...)")
    csv_sha256 = _file_sha256(external["latency_csv"])
    if csv_sha256 != LATENCY_CSV_SHA256:
        raise RuntimeError(f"{external['latency_csv']}: not the pinned bundle latency CSV ({LATENCY_CSV_SHA256[:12]}...)")
    with open(external["latency_csv"]) as handle:
        rows = list(csv.DictReader(handle))
    inventory.update(exp23_summary={"path": str(external["accuracy"]), "sha256": summary["sha256"],
                                    "file_sha256": _file_sha256(external["accuracy"])},
                     latency_csv={"path": str(external["latency_csv"]), "file_sha256": csv_sha256})
    return agg.external_markers(summary, rows, pilot)


def build_summary(spec: dict) -> dict:
    check_spec(spec)
    inventory = {"inputs": {}, "selectors": {}, "runs": {}, "external": {}, "synthesis": {}}
    b2, summaries = build_b2(spec, inventory)
    b3 = build_b3(spec, summaries, inventory)
    external = load_external(spec["external"], _hashed(spec["pilot_0p50"]), inventory["external"])
    reasons = [f"B2 {p['arm']} {p['grid']}: the seed-42 timing is not attested" for p in b2["points"]
               if not p["timing_valid"]]
    reasons += [f"B3 orbit {r['orbit']}: the latency timing is not attested" for r in b3["latency"] if not r["timing_valid"]]
    if b3["synthesis"]["status"] != "complete":
        reasons.append("B3 synthesis is pending: the exp_36 leaf cells are not filled in the spec")
    b2["frontier"] = None if reasons else frontier(b2["points"])
    body = {"schema_version": SPEC_SCHEMA, "record_type": "exp37_b2_b3_DIAGNOSTIC" if reasons else "exp37_b2_b3_summary",
            "publishable": not reasons, "reasons": reasons, "spec_sha256": canonical_sha256(spec), "spec": spec,
            "b2": b2, "b3": b3, "external": external, "inventory": inventory}
    return {**body, "sha256": canonical_sha256(body)}


def _cell(stats: dict, metric: str, percent: bool = False) -> str:
    s = stats[metric]
    scale, unit = (100.0, "%") if percent else (1.0, "")
    sd = "" if s["sd"] is None else f" ± {scale * s['sd']:.{1 if percent else 3}f}"
    return f"{scale * s['mean']:.{1 if percent else 3}f}{sd}{unit} (n={s['n']})"


def render_markdown(summary: dict) -> str:
    b2 = summary["b2"]
    lines = [] if summary["publishable"] else [
        "# DIAGNOSTIC -- NOT PUBLISHABLE (no frontier and no figure are produced)", "",
        *[f"- {reason}" for reason in summary["reasons"]], ""]
    lines += [f"# exp_37 B2 candidate density / coarse-to-fine and B3 orbit size ({b2['query_count']} queries)", "",
             "B2 accuracy: mean ± sample SD over seeds 42-44 of the pooled per-run metrics; s/query = the seed-42 "
             "pass only (D-a; no timing-variance claim), query-micro mean of L_q = core_forward_total + shape-matched "
             "AGREE selector (c2f: its sequential total). 1.0 m = the nested integer subset of the 0.5 m grid with "
             "full-grid common random numbers. c2f accuracy = the sequential c2f runs (seed 42 the timed pass, 43 / 44 "
             "accuracy runs); the offline c2f readout of the 0.5 m runs is a reported cross-check.", "",
             "| Method | Grid | s/query | candidates / query | median error [m] | SR@0.5 m | SR@1 m | res.-aware SR@0.5 | timing |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for p in b2["points"]:
        a, t = p["accuracy"], p["seconds_per_query"]
        lines.append(f"| {p['label']} | {p['grid_label']} | {'-' if t is None else f'{t:.3f}'} | "
                     f"{a['mean_candidate_count']['mean']:.1f} | {_cell(a, 'median_localization_error_m')} | "
                     f"{_cell(a, 'success_0_5m', True)} | {_cell(a, 'success_1_0m', True)} | "
                     f"{_cell(a, 'oracle_normalized_success_0_5m', True)} | "
                     f"{'valid' if p['timing_valid'] else 'NOT PUBLISHABLE / missing'} |")
    for name, m in summary.get("external", {}).items():
        acc = m["accuracy"]
        lines.append(f"| {m['label']} | reference | {m['seconds_per_query']:.3f} | - | {acc['median_localization_error_m']:.3f} | "
                     f"{100 * acc['success_0_5m']:.1f}% | {100 * acc['success_1_0m']:.1f}% | "
                     f"{100 * acc['oracle_normalized_success_0_5m']:.1f}% | external |")
    for metric, members in (b2["frontier"] or {}).items():
        lines.append(f"\nPareto frontier ({metric}, internal points only): " +
                     ", ".join(f"{b2['points'][i]['label']} {b2['points'][i]['grid_label']}" for i in members))
    for arm, by_seed in b2.get("c2f_parity", {}).items():
        lines.append(f"\nc2f cross-check {LABELS[arm]} (sequential vs the parent's offline c2f, per seed): " + "; ".join(
            f"s{seed} union {v['union_equal']}/{v['queries']}, winner {v['winner_equal']}/{v['queries']}, bitwise "
            f"{v['bitwise_equal']}/{v['queries']}, max |dsim| {v['max_abs_similarity_delta']:.2e}"
            for seed, v in sorted(by_seed.items())))
    lines += ["", "c2f top-k sensitivity and the offline k = 5 readout (post-primary; accuracy only):", "",
              "| Method | Grid | median error [m] | SR@1 m |", "|---|---|---:|---:|"]
    lines += [f"| {LABELS[s['arm']]} | {s['grid']} | {_cell(s['accuracy'], 'median_localization_error_m')} | "
              f"{_cell(s['accuracy'], 'success_1_0m', True)} |" for s in b2["sensitivity"]]
    lines += ["", "## B3 OrbitRIR inference orbit (localization; synthesis cells are evaluated separately)", "",
              "| Orbit | theta | median error [m] | SR@0.5 m | SR@1 m | res.-aware SR@0.5 |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['angles']} | {r['theta']} | {_cell(r['accuracy'], 'median_localization_error_m')} | "
              f"{_cell(r['accuracy'], 'success_0_5m', True)} | {_cell(r['accuracy'], 'success_1_0m', True)} | "
              f"{_cell(r['accuracy'], 'oracle_normalized_success_0_5m', True)} |" for r in summary["b3"]["accuracy"]]
    if summary["b3"]["latency"]:
        lines += ["", "| Orbit | mean s/query | median | P90 | ms/candidate | timing |", "|---|---:|---:|---:|---:|---|"]
        lines += [f"| {r['angles']} | {r['overall']['mean_seconds']:.3f} | {r['overall']['median_seconds']:.3f} | "
                  f"{r['overall']['p90_seconds']:.3f} | {1000 * r['per_candidate']['amortized_seconds']:.3f} | "
                  f"{'valid' if r['timing_valid'] else 'NOT PUBLISHABLE'} |" for r in summary["b3"]["latency"]]
    synthesis = summary["b3"]["synthesis"]
    lines += ["", "## B3 synthesis cells (eval_FLAC, seed 42; the exp_36 leaf)", ""]
    if synthesis["status"] != "complete":
        lines.append(f"PENDING: {synthesis.get('note')}")
    else:
        keys = synthesis["metrics_keys"]
        lines += [f"Leaf commit `{synthesis['leaf_commit']}`; runbook: {synthesis['runbook']}. Values = the per-scene "
                  "mean of each metric (the paper's convention).", "",
                  "| Orbit | theta | K | source | " + " | ".join(keys) + " |", "|---|---:|---:|---|" + "---:|" * len(keys)]
        lines += [f"| {ORBITS[c['orbit']]} | {c['theta']} | {c['k_context']} | {c['source']} | "
                  + " | ".join(f"{c['per_scene_mean'][k]:.3f}" for k in keys) + " |" for c in synthesis["cells"]]
    inventory = summary["inventory"]
    lines += ["", f"Sources: {len(inventory['runs'])} runs, {len(inventory['selectors'])} selector vectors, "
              f"{len(inventory['inputs'])} input manifests, the pinned exp_23 summary and bundle latency CSV, "
              f"{len(inventory['synthesis'])} synthesis files -- every sha256 in the JSON inventory; spec "
              f"`{summary['spec_sha256']}`."]
    return "\n".join(lines) + "\n"


# The dataviz reference palette (validated slots 1-2 of its categorical order; light surface), ink tokens.
SERIES = {"P1": "#2a78d6", "BF": "#eb6834"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
MARKERS = {"0.50": "o", "0.75": "s", "1.00_nested": "^", "c2f_k5": "D"}
REFERENCE_MARKERS = {"few_shot_rir": ("P", "Few-ShotRIR (ref.)"), "fem_omp": ("X", "FEM-OMP (ref.)")}


def render_figure(summary: dict, stem: Path) -> None:
    if not summary["publishable"] or summary["b2"]["frontier"] is None:
        raise RuntimeError("the Pareto figure is drawn only from a publishable summary")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points, external = summary["b2"]["points"], summary.get("external", {})
    xs = [p["seconds_per_query"] for p in points if p["seconds_per_query"]] + [m["seconds_per_query"] for m in external.values()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    fig.patch.set_facecolor(INK["surface"])
    panels = (("median_localization_error_m", "Median localization error [m]  (lower is better)", True),
              ("success_1_0m", "Success rate @ 1 m  (higher is better)", False))
    for ax, (metric, title, minimize) in zip(axes, panels):
        ax.set_facecolor(INK["surface"])
        ax.set_xscale("log")
        ax.grid(True, which="major", color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK["axis"])
        ax.tick_params(colors=INK["secondary"], labelsize=7)
        frontier = sorted((points[i]["seconds_per_query"], points[i]["accuracy"][metric]["mean"])
                          for i in summary["b2"]["frontier"][metric])
        if frontier:
            ax.plot(*zip(*frontier), color=INK["secondary"], linewidth=1.0, drawstyle="steps-post", zorder=1)
        for arm in ARMS:
            mine = sorted((p for p in points if p["arm"] == arm and p["seconds_per_query"]),
                          key=lambda p: p["seconds_per_query"])
            if not mine:
                continue
            ax.plot([p["seconds_per_query"] for p in mine], [p["accuracy"][metric]["mean"] for p in mine],
                    color=SERIES[arm], linewidth=1.5, alpha=0.55, zorder=2)
            for p in mine:
                stats = p["accuracy"][metric]
                ax.errorbar(p["seconds_per_query"], stats["mean"], yerr=stats["sd"] or 0.0, fmt=MARKERS[p["grid"]],
                            markersize=6.5, color=SERIES[arm], markeredgecolor=INK["surface"], markeredgewidth=1.2,
                            ecolor=SERIES[arm], elinewidth=1.0, capsize=0, zorder=3)
        for name, marker in external.items():
            shape, short = REFERENCE_MARKERS[name]
            x, y = marker["seconds_per_query"], marker["accuracy"][metric]
            ax.scatter([x], [y], marker=shape, s=48, facecolor="none", edgecolor=INK["muted"], linewidth=1.2, zorder=3)
            right_edge = x >= max(xs)                        # keep the label inside the axes
            ax.annotate(short, (x, y), xytext=(-5 if right_edge else 5, 5), textcoords="offset points", fontsize=6.5,
                        color=INK["secondary"], ha="right" if right_edge else "left", va="bottom")
        ax.set_title(title, fontsize=8.5, color=INK["primary"], loc="left")
        ax.set_xlabel("s / query  (log scale; $L_q$ = core forward + AGREE selector)", fontsize=7.5,
                      color=INK["secondary"])
        if xs:
            ax.set_xlim(min(xs) / 1.8, max(xs) * 2.2)
        ys = [p["accuracy"][metric]["mean"] + (p["accuracy"][metric]["sd"] or 0.0) for p in points]
        ys += [m["accuracy"][metric] for m in external.values()]
        ax.set_ylim(0.0, 1.0 if not minimize else 1.18 * max(ys + [0.1]))
    handles = [Line2D([], [], color=SERIES[a], linewidth=2.5, label=LABELS[a]) for a in ARMS]
    handles += [Line2D([], [], color=INK["muted"], marker=MARKERS[g], linestyle="none", label=GRID_LABELS[g]) for g in GRIDS]
    handles += [Line2D([], [], color=INK["muted"], marker=REFERENCE_MARKERS[n][0], markerfacecolor="none", linestyle="none",
                       label=external[n]["label"]) for n in external]
    handles.append(Line2D([], [], color=INK["secondary"], linewidth=1.0, label="Pareto frontier (FLAC / OrbitRIR points only)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=6.5, labelcolor=INK["primary"])
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor=INK["surface"])
    fig.savefig(stem.with_suffix(".pdf"), facecolor=INK["surface"])
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--figure-stem", type=Path)
    args = parser.parse_args()
    published = [args.output_json, args.output_md] + (
        [args.figure_stem.with_suffix(".png"), args.figure_stem.with_suffix(".pdf")] if args.figure_stem else [])
    diagnostic = [args.output_json.with_suffix(".DIAGNOSTIC.json"), args.output_md.with_suffix(".DIAGNOSTIC.md")]
    if any(path.exists() for path in published + diagnostic):
        print("REFUSED: an output exists (summaries, diagnostics and figures are written once)", file=sys.stderr)
        return 2
    try:
        summary = build_summary(json.loads(args.spec.read_text()))
    except SpecError as error:
        print(f"REFUSED (spec): {error}", file=sys.stderr)
        return 2
    from src.localization import exp37_publish
    record, others = (published[0], published[1:]) if summary["publishable"] else (diagnostic[0], diagnostic[1:])

    def render(staging):                                  # review 2 HIGH 7: one transaction, the JSON last
        (staging / others[0].name).write_text(render_markdown(summary))
        if summary["publishable"] and args.figure_stem is not None:
            render_figure(summary, staging / args.figure_stem.name)
        body = {k: v for k, v in summary.items() if k != "sha256"}
        body["published_files"] = exp37_publish.staged_sha256(staging, others)
        (staging / record.name).write_text(json.dumps({**body, "sha256": canonical_sha256(body)}, indent=2,
                                                      sort_keys=True) + "\n")
    exp37_publish.publish(others + [record], render)
    print(render_markdown(summary))
    return 0 if summary["publishable"] else 3


if __name__ == "__main__":
    sys.exit(main())
