#!/usr/bin/env python3
"""exp_37 B2 / B3 aggregation (plan_loc_efficiency.md §4, §5, §10): one spec JSON in -> a hashed summary JSON, a
markdown table and the two-panel B2 Pareto figure (PNG + PDF).

B2 -- per arm (FLAC P1, OrbitRIR B-F) x grid: accuracy = the pooled 64-query metrics of each seed's run, mean and
sample SD over seeds 42-44 (0.5 m runs give the 0.5 m, nested-1.0 m and c2f readouts -- offline-exact with full-grid
common random numbers; 0.75 m runs their own full readout); time = s/query, the query-micro mean of
L_q = core_forward_total_q + the shape-matched AGREE selector term (seed 42 = the headline, 43 / 44 = repeats;
nested 1.0 m from its dedicated subset pass + the nested selector vector; c2f = its measured sequential total);
never ``elapsed_seconds``. The Pareto frontier is computed over these internal points only; FewShotRIR and FEM are
external reference markers. c2f k in {1, 3, 10} is a post-primary sensitivity (accuracy only).
B3 -- OrbitRIR orbit {0} / {0,180} / C4 x theta {0, 90} x seeds: accuracy per (orbit, theta); orbit latency from
B1-style passes on the 128-query selection. Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-c.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization import exp37_aggregate as agg  # noqa: E402
from src.localization import exp37_grid as grid  # noqa: E402
from src.localization import exp37_latency as lat  # noqa: E402
from src.localization.pilot import canonical_sha256, load_pilot_manifest  # noqa: E402

ARMS = ("P1", "BF")
LABELS = {"P1": "FLAC (P1@40k)", "BF": "OrbitRIR (B-F@40k)"}
GRIDS = ("0.50", "0.75", "1.00_nested", "c2f_k5")
GRID_LABELS = {"0.50": "0.5 m", "0.75": "0.75 m", "1.00_nested": "1.0 m (nested)", "c2f_k5": "c2f (1.0 -> 0.5 m, k = 5)"}
SENSITIVITY = ("c2f_k1", "c2f_k3", "c2f_k10")
METRICS = ("median_localization_error_m", "success_0_5m", "success_1_0m", "oracle_normalized_success_0_5m",
           "mean_oracle_error_m", "mean_candidate_count")
ORBITS = {"0": [0.0], "0_180": [0.0, 180.0], "c4": [0.0, 90.0, 180.0, 270.0]}


def _hashed(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != canonical_sha256({k: v for k, v in payload.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return payload


def _selector(path, selection):
    rows = lat.load_selector(path, selection)
    return {i: r["scoring_total_seconds"] for i, r in rows.items()}, lat.selector_cotenancy(json.loads(Path(path).read_text()))


def _seed_of(run_dir) -> int:
    return int(_hashed(Path(run_dir) / "run_manifest.json")["identity"]["sample_seed"])


def _stats(by_seed: dict) -> dict:
    seeds = sorted(by_seed)
    return {**{m: agg.seed_stats(by_seed[s][m] for s in seeds) for m in METRICS}, "seeds": seeds}


def build_b2(spec: dict) -> dict:
    pilot, pilot75 = load_pilot_manifest(spec["pilot_0p50"]), load_pilot_manifest(spec["pilot_0p75"])
    nested = _hashed(spec["nested_manifest"])
    indices = [int(r["index"]) for r in pilot["records"]]
    if [int(r["index"]) for r in pilot75["records"]] != indices or nested["parent_pilot_sha256"] != pilot["sha256"]:
        raise RuntimeError("the 0.75 m pilot / nested manifest do not cover the 0.5 m pilot's queries")
    terms = {"0.50": _selector(spec["selector_0p50"], pilot), "0.75": _selector(spec["selector_0p75"], pilot75),
             "1.00_nested": _selector(spec["selector_nested"], nested)}
    summaries = {(a, g): {} for a in ARMS for g in GRIDS + SENSITIVITY}
    seconds, valid = {(a, g): {} for a in ARMS for g in GRIDS}, {(a, g): [] for a in ARMS for g in GRIDS}
    for item in spec["b2_runs"]:
        arm, g, seed, run_dir = item["arm"], item["grid"], int(item["seed"]), item["dir"]
        selection = {"0.50": pilot, "0.75": pilot75}[g]
        core = {i: r["core"] for i, r in lat.load_arm_run(run_dir, arm=arm, selection=selection, context_count=8).items()}
        if _seed_of(run_dir) != seed or seed in summaries[(arm, g)]:
            raise RuntimeError(f"{run_dir}: not the seed-{seed} {arm} {g} run, or a duplicate")
        rows = grid.readout_run(run_dir, pilot) if g == "0.50" else {"0.75": grid.readout_run(
            run_dir, pilot75, full_only=True)["full"]}
        for name, summary in grid.summarize_readout(rows).items():
            summaries[(arm, name)][seed] = summary
        seconds[(arm, g)][seed] = agg.mean_latency(core, terms[g][0], indices)
        valid[(arm, g)].append(lat.host_cotenancy(run_dir)["timing_valid"] and terms[g][1]["timing_valid"])
    for item in spec.get("b2_timing", []):
        arm, kind, run_dir = item["arm"], item["kind"], item["dir"]
        if kind == "nested_1m":
            core = {i: r["core"] for i, r in lat.load_arm_run(run_dir, arm=arm, selection=pilot, context_count=8,
                                                              subset_manifest=nested).items()}
            key, value, ok = (arm, "1.00_nested"), agg.mean_latency(core, terms["1.00_nested"][0], indices), \
                terms["1.00_nested"][1]["timing_valid"]
        elif kind == "c2f":
            rows = lat.load_c2f_run(run_dir, arm=arm, selection=pilot, top_k=5)
            key, value, ok = (arm, "c2f_k5"), sum(rows[i]["c2f_total"] for i in indices) / len(indices), True
        else:
            raise ValueError(f"unknown timing pass kind {kind!r}")
        if _seed_of(run_dir) != 42 or 42 in seconds[key]:
            raise RuntimeError(f"{run_dir}: the dedicated {kind} timing pass must be the single seed-42 pass")
        seconds[key][42] = value
        valid[key].append(ok and lat.host_cotenancy(run_dir)["timing_valid"])
    points = []
    for arm in ARMS:
        for g in GRIDS:
            if not summaries[(arm, g)]:
                continue
            times = seconds[(arm, g)]
            points.append({"arm": arm, "label": LABELS[arm], "grid": g, "grid_label": GRID_LABELS[g],
                           "accuracy": _stats(summaries[(arm, g)]), "seconds_per_query": times.get(42),
                           "seconds_per_query_repeats": {str(s): v for s, v in sorted(times.items()) if s != 42},
                           "timing_valid": bool(valid[(arm, g)]) and all(valid[(arm, g)]) and 42 in times})
    timed = [i for i, p in enumerate(points) if p["seconds_per_query"] is not None]
    frontier = {metric: [timed[j] for j in agg.pareto_frontier(
        [(points[i]["seconds_per_query"], points[i]["accuracy"][metric]["mean"]) for i in timed], minimize_y=minimize)]
        for metric, minimize in (("median_localization_error_m", True), ("success_1_0m", False))}
    sensitivity = [{"arm": a, "grid": g, "accuracy": _stats(summaries[(a, g)]), "post_primary_sensitivity": True}
                   for a in ARMS for g in SENSITIVITY if summaries[(a, g)]]
    return {"query_count": len(indices), "points": points, "frontier": frontier, "sensitivity": sensitivity}


def _b3_reasons(identity: dict, orbit: str, theta: float, seed: int, pilot: dict) -> list[str]:
    want = {"conditioning_method": "fa_invariant", "checkpoint_sha256": lat.ARM_PROTOCOL["BF"]["ckpt"],
            "frame_average_angles": ORBITS[orbit], "pilot_manifest_sha256": pilot["sha256"], "n_context": 8,
            "query_indices": [int(r["index"]) for r in pilot["records"]], "sample_seed": seed, "tau": 0.1,
            "score_sample_counts": [1], "candidate_batch_size": 64, "attention_backend": "sdpa"}
    bad = [key for key, value in want.items() if identity.get(key) != value]
    return bad + (["rotate_deg"] if float(identity.get("rotate_deg", 0.0)) != float(theta) else [])


def build_b3(spec: dict) -> dict:
    pilot = load_pilot_manifest(spec["pilot_0p50"])
    by_cell = {}
    for item in spec.get("b3_runs", []):
        orbit, theta, seed, run_dir = item["orbit"], float(item["theta"]), int(item["seed"]), item["dir"]
        bad = _b3_reasons(_hashed(Path(run_dir) / "run_manifest.json")["identity"], orbit, theta, seed, pilot)
        cell = by_cell.setdefault((orbit, theta), {})
        if bad or seed in cell:
            raise RuntimeError(f"{run_dir}: identity fields {bad} differ from B-F orbit {orbit} theta {theta} seed {seed}"
                               " (or a duplicate)")
        cell[seed] = grid.summarize_readout({"full": grid.readout_run(run_dir, pilot, full_only=True)["full"]})["full"]
    accuracy = [{"orbit": o, "theta": int(t), "angles": ORBITS[o], "accuracy": _stats(cell)}
                for (o, t), cell in sorted(by_cell.items())]
    latency = []
    if spec.get("b3_latency"):
        selection = load_pilot_manifest(spec["b3_latency"]["selection"])
        terms, selector_host = _selector(spec["b3_latency"]["selector"], selection)
        counts = {int(r["index"]): int(r["candidate_count"]) for r in selection["records"]}
        for item in spec["b3_latency"]["runs"]:
            rows = lat.load_arm_run(item["dir"], arm="BF", selection=selection, context_count=8,
                                    frame_angles=ORBITS[item["orbit"]])
            stats = lat.latency_statistics({i: r["core"] for i, r in rows.items()}, terms, counts)
            latency.append({"orbit": item["orbit"], "angles": ORBITS[item["orbit"]], "overall": stats["overall"],
                            "per_candidate": stats["per_candidate"],
                            "timing_valid": lat.host_cotenancy(item["dir"])["timing_valid"] and selector_host["timing_valid"]})
    return {"accuracy": accuracy, "latency": latency}


def build_summary(spec: dict) -> dict:
    body = {"schema_version": 1, "record_type": "exp37_b2_b3_summary", "spec": spec, "b2": build_b2(spec),
            "b3": build_b3(spec)}
    if spec.get("external"):
        pilot = load_pilot_manifest(spec["pilot_0p50"])
        with open(spec["external"]["latency_csv"]) as handle:
            body["external"] = agg.external_markers(json.loads(Path(spec["external"]["accuracy"]).read_text()),
                                                    list(csv.DictReader(handle)), pilot)
    return {**body, "sha256": canonical_sha256(body)}
