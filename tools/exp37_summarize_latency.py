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
from src.localization import exp37_sources as sources  # noqa: E402
from src.localization.pilot import canonical_sha256  # noqa: E402

LABELS = {"P1": "Vanilla FLAC (P1@40k)", "YAW": "Yaw-Augmented FLAC (YAW@40k)", "BF": "OrbitRIR (B-F@40k, C4 FA)"}
MIB = 1024 * 1024
REGISTERED_ROWS = sorted((arm, k) for arm in ("P1", "YAW", "BF") for k in (8, 1))


class CoverageError(ValueError):
    """Not the registered B1 table (nothing is written)."""


def check_coverage(runs, splits, replays=(), split_replays=(), *, require_replays=False) -> None:
    keys = sorted((arm, int(k)) for arm, k, _d in runs)
    if keys != REGISTERED_ROWS:
        raise CoverageError(f"the B1 table is exactly {REGISTERED_ROWS} (got {keys})")
    split_keys = sorted((arm, int(k)) for arm, k, _d in splits)
    k8, k1 = [(a, 8) for a in ("BF", "P1", "YAW")], [(a, 1) for a in ("BF", "P1", "YAW")]
    if split_keys not in (sorted(k8), sorted(k8 + k1)):
        raise CoverageError("the stage splits are the three K_ctx 8 passes, plus all three K_ctx 1 passes or none")
    for label, given, passes in (("--replay", replays, keys), ("--split-replay", split_replays, split_keys)):
        have = sorted((arm, int(k)) for arm, k, _d in given)
        if len(set(have)) != len(have) or not set(have) <= set(passes) or (require_replays and have != passes):
            raise CoverageError(f"{label}: one replay per pass at most{' and exactly one each' if require_replays else ''}")


def _entry(run_dir, replay, *, arm, selection, audit, k, stage_split=False) -> dict:
    """The inventory entry of one timed pass: manifest + per-query artifacts (or its validated replay)."""
    if replay is not None:
        entry = sources.accuracy_source(run_dir, arm=arm, selection=selection, audit=audit, seed=42, timing=True,
                                        replay_dir=replay, context_count=k, stage_split=stage_split)
    else:
        manifest = json.loads((Path(run_dir) / "run_manifest.json").read_text())
        entry = {"run_dir": str(run_dir), "run_manifest_sha256": manifest["sha256"], "timing": True,
                 "queries": sources.query_inventory(run_dir, manifest, [int(r["index"]) for r in selection["records"]])}
    host = lat.host_cotenancy(run_dir)
    monitor = host.get("monitor") or {}
    entry["attestation"] = {"timing_valid": host["timing_valid"], "reasons": host["reasons"],
                            "monitor_head_sha256": monitor.get("monitor_head_sha256"),
                            "markers_head_sha256": monitor.get("markers_head_sha256"),
                            "host_records": host.get("host_records", {})}
    return entry


def _manifest_sha(run_dir: Path) -> str:
    return json.loads((Path(run_dir) / "run_manifest.json").read_text())["sha256"]


def weights_footprint(run_dir: Path) -> dict | None:
    loaded = sorted((Path(run_dir) / "host_records").glob("loaded_*.json"))
    if not loaded:
        return None
    return {**(json.loads(loaded[0].read_text()).get("extra") or {}), "loaded_record_count": len(loaded)}


def load_audit(path: Path) -> dict:
    audit = json.loads(Path(path).read_text())
    if audit.get("sha256") != canonical_sha256({k: v for k, v in audit.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {path}")
    return audit


def build_summary(selection: dict, selector_path: Path, runs: list, splits: list = (), *, audit: dict,
                  replays: dict | None = None, split_replays: dict | None = None) -> dict:
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
    replays, split_replays, inventory_runs = dict(replays or {}), dict(split_replays or {}), {}
    rows = []
    for arm, k, run_dir in runs:
        head = lat.load_arm_run(run_dir, arm=arm, selection=selection, audit=audit, seed=42, context_count=int(k))
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
        inventory_runs[str(run_dir)] = _entry(run_dir, replays.get((arm, int(k))), arm=arm, selection=selection,
                                              audit=audit, k=int(k))
        if (arm, int(k)) in split_dirs:
            split_dir = split_dirs[(arm, int(k))]
            inventory_runs[str(split_dir)] = _entry(split_dir, split_replays.get((arm, int(k))), arm=arm,
                                                    selection=selection, audit=audit, k=int(k), stage_split=True)
            split = lat.load_arm_run(split_dir, arm=arm, selection=selection, audit=audit, seed=42,
                                     context_count=int(k), stage_split=True)
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
    selector_host = lat.selector_cotenancy(selector_path)
    monitor = selector_host.get("monitor") or {}
    reasons = [f"{row['arm']} K_ctx {row['context_count']}: the timing is not attested" for row in rows
               if not row["timing_valid"]] + ([] if selector_host["timing_valid"] else ["the selector vector is not attested"])
    inventory = {"selection": {"sha256": selection["sha256"]}, "geometry_audit": {"sha256": audit["sha256"]},
                 "selector": {"path": str(selector_path), "sha256": selector_payload.get("sha256"),
                              "binding_sha256": selector_host.get("binding_sha256"),
                              "monitor_head_sha256": monitor.get("monitor_head_sha256"),
                              "markers_head_sha256": monitor.get("markers_head_sha256")},
                 "runs": inventory_runs}
    body = {"schema_version": 2, "record_type": "exp37_latency_summary" if not reasons else "exp37_latency_DIAGNOSTIC",
            "publishable": not reasons, "reasons": reasons, "inventory": inventory,
            "selection_sha256": selection["sha256"],
            "selector_file": str(selector_path), "selector_sha256": selector_payload.get("sha256"),
            "geometry_audit_sha256": audit["sha256"], "selector_host": selector_host, "selector_timing_valid": selector_host["timing_valid"],
            "query_count": len(counts), "candidate_evaluations": sum(counts.values()),
            "formulas": "L_q = core_forward_total_q + selector median scoring_total_q; s/query over queries; "
                        "ms/candidate = 1000 sum L / sum |C| and query median / P90 of 1000 L_q / |C_q| (bundle)",
            "rows": rows}
    return {**body, "sha256": canonical_sha256(body)}


def render_markdown(summary: dict) -> str:
    lines = [] if summary.get("publishable", True) else [
        "# DIAGNOSTIC -- NOT PUBLISHABLE", "", *[f"- {reason}" for reason in summary["reasons"]], ""]
    lines += [f"# exp_37 latency summary ({summary['query_count']} queries, {summary['candidate_evaluations']:,} "
             "candidates; K_gen 1, batch 64, SDPA)", "",
             "AGREE selector vector: " + ("valid (exclusive card)" if summary["selector_timing_valid"] else
                                          "NOT PUBLISHABLE (see selector_host.reasons)"), "",
             "| Arm | K_ctx | Mean [s/query] | Median | P90 | ms/cand (dataset) | Query median | Query P90 | "
             "Query peak [MiB] | Combined peak [MiB] | Timing |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        o, c, m = row["overall"], row["per_candidate"], row["memory"]
        lines.append(f"| {row['label']} | {row['context_count']} | {o['mean_seconds']:.3f} | {o['median_seconds']:.3f} | "
                     f"{o['p90_seconds']:.3f} | {1000 * c['amortized_seconds']:.3f} | {1000 * c['median_seconds']:.3f} | "
                     f"{1000 * c['p90_seconds']:.3f} | {m['query_peak_allocated_max_bytes'] / MIB:.0f} | "
                     f"{m['combined_inference_peak_allocated_bytes'] / MIB:.0f} | "
                     f"{'valid' if row['timing_valid'] else 'NOT PUBLISHABLE (see host.reasons)'} |")
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
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--split", nargs=3, action="append", default=[], metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--replay", nargs=3, action="append", default=[], metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--split-replay", nargs=3, action="append", default=[], metavar=("ARM", "K_CTX", "DIR"))
    parser.add_argument("--require-replays", action="store_true", help="every pass needs its artifact replay")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_json, args.output_md, args.output_json.with_suffix(".DIAGNOSTIC.json"),
               args.output_md.with_suffix(".DIAGNOSTIC.md")]
    if any(path.exists() for path in outputs):
        print("REFUSED: an output exists (summaries and diagnostics are written once)", file=sys.stderr)
        return 2
    selection = json.loads(args.selection.read_text())
    if selection.get("sha256") != canonical_sha256({k: v for k, v in selection.items() if k != "sha256"}):
        raise RuntimeError(f"stale or corrupt sha256 in {args.selection}")
    for arm, _k, _d in args.run + args.split + args.replay + args.split_replay:
        if arm not in LABELS:
            raise ValueError(f"unknown arm {arm}")
    try:
        check_coverage(args.run, args.split, args.replay, args.split_replay, require_replays=args.require_replays)
    except CoverageError as error:
        print(f"REFUSED (coverage): {error}", file=sys.stderr)
        return 2
    summary = build_summary(selection, args.selector, [(a, int(k), Path(d)) for a, k, d in args.run],
                            [(a, int(k), Path(d)) for a, k, d in args.split], audit=load_audit(args.geometry_audit),
                            replays={(a, int(k)): Path(d) for a, k, d in args.replay},
                            split_replays={(a, int(k)): Path(d) for a, k, d in args.split_replay})
    targets = outputs[:2] if summary["publishable"] else outputs[2:]
    for path, text in zip(targets, (json.dumps(summary, indent=2, sort_keys=True) + "\n", render_markdown(summary))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, path)
    print(render_markdown(summary))
    return 0 if summary["publishable"] else 3


if __name__ == "__main__":
    sys.exit(main())
