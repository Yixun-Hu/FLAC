#!/usr/bin/env python3
"""exp_37 B1 latency summary: arms x K_ctx on one selection (plan_loc_efficiency.md §2 B1, §10).

Rows: the bundle's s/query mean / median / P90 and ms/candidate dataset-wide / query median / P90 of
L_q = core_forward_total_q + the AGREE selector term (``src/localization/exp37_latency.py``); memory as DIFFERENT
quantities -- the whole-query allocated peak (bundle-compatible), the selector's own peak, their MAXIMUM as the
combined inference peak (never a sum), and the loaded-weight footprint; the host-record co-tenancy verdict (a row
whose card held another compute process is not publishable; the CLI then exits 3). A stage split (the separate
instrumented pass) is published only when its core totals agree with the headline within 2 % per query AND in
aggregate; otherwise only its comparison is recorded -- the headline is never withheld.
Written by the exp_37 Coder (Claude Opus 5.5, max effort) per the Coder brief r1, round B-b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(REPO_ROOT))

from src.localization import exp37_latency as lat  # noqa: E402
from src.localization.pilot import canonical_sha256  # noqa: E402

LABELS = {"P1": "Vanilla FLAC (P1@40k)", "YAW": "Yaw-Augmented FLAC (YAW@40k)", "BF": "OrbitRIR (B-F@40k, C4 FA)"}
MIB = 1024 * 1024


def _manifest_sha(run_dir: Path) -> str:
    return json.loads((Path(run_dir) / "run_manifest.json").read_text())["sha256"]


def weights_footprint(run_dir: Path) -> dict | None:
    loaded = sorted((Path(run_dir) / "host_records").glob("loaded_*.json"))
    if not loaded:
        return None
    return {**(json.loads(loaded[0].read_text()).get("extra") or {}), "loaded_record_count": len(loaded)}


def build_summary(selection: dict, selector_path: Path, runs: list, splits: list = ()) -> dict:
    keys = [(arm, int(k)) for arm, k, _d in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate arm / K_ctx headline run")
    split_dirs = {(arm, int(k)): Path(d) for arm, k, d in splits}
    if not set(split_dirs) <= set(keys) or len(split_dirs) != len(splits):
        raise ValueError("a stage split has no headline run of the same arm / K_ctx (or is duplicated)")
    selector = lat.load_selector(selector_path, selection)
    terms = {i: row["scoring_total_seconds"] for i, row in selector.items()}
    selector_peaks = [row["peak_memory_allocated_bytes"] for row in selector.values()]
    selector_peak = max(selector_peaks) if all(p is not None for p in selector_peaks) else None
    counts = {int(r["index"]): int(r["candidate_count"]) for r in selection["records"]}
    rows = []
    for arm, k, run_dir in runs:
        head = lat.load_arm_run(run_dir, arm=arm, selection=selection, context_count=int(k))
        stats = lat.latency_statistics({i: r["core"] for i, r in head.items()}, terms, counts)
        host = lat.host_cotenancy(run_dir)
        peaks = [r["peak_memory_bytes"] for r in head.values()]
        row = {"arm": arm, "label": LABELS[arm], "context_count": int(k), "run_dir": str(run_dir),
               "run_manifest_sha256": _manifest_sha(run_dir), "timing_valid": host["timing_valid"], "host": host,
               "overall": stats["overall"], "per_candidate": stats["per_candidate"],
               "latency_seconds": stats["latency_seconds"],
               "memory": {"query_peak_allocated_max_bytes": max(peaks),
                          "query_peak_allocated_median_bytes": float(np.median(peaks)),
                          "selector_peak_allocated_max_bytes": selector_peak,
                          "combined_inference_peak_allocated_bytes": max(max(peaks), selector_peak or 0),
                          "weights_footprint": weights_footprint(run_dir)}}
        if (arm, int(k)) in split_dirs:
            split_dir = split_dirs[(arm, int(k))]
            split = lat.load_arm_run(split_dir, arm=arm, selection=selection, context_count=int(k), stage_split=True)
            comparison = lat.split_comparison(head, split)
            split_host = lat.host_cotenancy(split_dir)
            entry = {"run_dir": str(split_dir), "run_manifest_sha256": _manifest_sha(split_dir), "host": split_host,
                     "comparison": comparison,
                     "published": bool(comparison["split_publishable"] and split_host["timing_valid"])}
            if entry["published"]:
                entry["breakdown"] = lat.stage_breakdown(split, terms, counts)
            row["stage_split"] = entry
        rows.append(row)
    selector_payload = json.loads(Path(selector_path).read_text())
    selector_host = lat.selector_cotenancy(selector_payload)
    body = {"schema_version": 1, "record_type": "exp37_latency_summary", "selection_sha256": selection["sha256"],
            "selector_file": str(selector_path), "selector_sha256": selector_payload.get("sha256"),
            "selector_host": selector_host, "selector_timing_valid": selector_host["timing_valid"],
            "query_count": len(counts), "candidate_evaluations": sum(counts.values()),
            "formulas": "L_q = core_forward_total_q + selector median scoring_total_q; s/query over queries; "
                        "ms/candidate = 1000 sum L / sum |C| and query median / P90 of 1000 L_q / |C_q| (bundle)",
            "rows": rows}
    return {**body, "sha256": canonical_sha256(body)}


def render_markdown(summary: dict) -> str:
    lines = [f"# exp_37 latency summary ({summary['query_count']} queries, {summary['candidate_evaluations']:,} "
             "candidates; K_gen 1, batch 64, SDPA)", "",
             "AGREE selector vector: " + ("valid (exclusive card)" if summary["selector_timing_valid"] else
                                          "NOT PUBLISHABLE (co-tenant, unknown card or no host snapshot)"), "",
             "| Arm | K_ctx | Mean [s/query] | Median | P90 | ms/cand (dataset) | Query median | Query P90 | "
             "Query peak [MiB] | Combined peak [MiB] | Timing |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        o, c, m = row["overall"], row["per_candidate"], row["memory"]
        lines.append(f"| {row['label']} | {row['context_count']} | {o['mean_seconds']:.3f} | {o['median_seconds']:.3f} | "
                     f"{o['p90_seconds']:.3f} | {1000 * c['amortized_seconds']:.3f} | {1000 * c['median_seconds']:.3f} | "
                     f"{1000 * c['p90_seconds']:.3f} | {m['query_peak_allocated_max_bytes'] / MIB:.0f} | "
                     f"{m['combined_inference_peak_allocated_bytes'] / MIB:.0f} | "
                     f"{'valid' if row['timing_valid'] else 'NOT PUBLISHABLE (co-tenant / unknown card)'} |")
    lines += ["", "Stage split (dataset-wide ms/candidate; published only within 2 % of the headline):", "",
              "| Arm | K_ctx | context | cand. conditioning | DiT | VAE decode | AGREE | aggregate / max per-query diff |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        split = row.get("stage_split")
        if split is None:
            continue
        comp = split["comparison"]
        diff = f"{100 * comp['aggregate_relative_difference']:+.2f} % / {100 * comp['per_query_max_abs']:.2f} %"
        if not split["published"]:
            lines.append(f"| {row['label']} | {row['context_count']} | WITHHELD | | | | | {diff} |")
            continue
        sec = split["breakdown"]["seconds"]
        cells = " | ".join(f"{sec[name]['ms_per_candidate_dataset_wide']:.3f}" for name in
                           ("context_conditioning", "candidate_conditioning", "dit", "vae_decode", "agree_selector"))
        lines.append(f"| {row['label']} | {row['context_count']} | {cells} | {diff} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--split", nargs=3, action="append", default=[], metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection.get("sha256") != canonical_sha256({k: v for k, v in selection.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {args.selection}")
    for arm, _k, _d in args.run + args.split:
        if arm not in LABELS:
            raise ValueError(f"unknown arm {arm}")
    summary = build_summary(selection, args.selector, [(a, int(k), Path(d)) for a, k, d in args.run],
                            [(a, int(k), Path(d)) for a, k, d in args.split])
    for path, text in ((args.output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n"),
                       (args.output_md, render_markdown(summary))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, path)
    print(render_markdown(summary))
    valid = summary["selector_timing_valid"] and all(row["timing_valid"] for row in summary["rows"])
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
