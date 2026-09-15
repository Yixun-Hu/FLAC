"""Validation and aggregation for completed exp_09 pilot arms."""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.localization.pilot import canonical_sha256
from src.localization.runner import completed_query_result, verify_hashed_payload


def load_completed_arm(output_dir: Path | str, pilot_manifest: dict) -> tuple[dict, list[dict]]:
    output_dir = Path(output_dir)
    run_path = output_dir / "run_manifest.json"
    if not run_path.is_file():
        raise RuntimeError(f"missing run manifest: {run_path}")
    run = json.loads(run_path.read_text())
    verify_hashed_payload(run, "run manifest")
    identity = run["identity"]
    if identity["pilot_manifest_sha256"] != pilot_manifest["sha256"]:
        raise RuntimeError("run does not use the requested pilot manifest")
    expected_indices = [int(item["index"]) for item in pilot_manifest["records"]]
    if identity["query_indices"] != expected_indices:
        raise RuntimeError("run does not contain the complete 64-query pilot")
    results = []
    for record in pilot_manifest["records"]:
        result = completed_query_result(
            output_dir,
            query_index=int(record["index"]),
            query_id=record["query_id"],
            candidate_count=int(record["candidate_count"]),
            run_sha256=run["sha256"],
        )
        if result is None:
            raise RuntimeError(f"incomplete query: {record['query_id']}")
        results.append(result)
    return run, results


def _summarize_metric_rows(rows: list[dict]) -> dict:
    errors = np.asarray([item["localization_error_m"] for item in rows], dtype=np.float64)
    oracle = np.asarray([item["oracle_error_m"] for item in rows], dtype=np.float64)
    excess = np.asarray([item["excess_error_m"] for item in rows], dtype=np.float64)
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        grouped[item["room"]].append(float(item["localization_error_m"]))
    room_means = [statistics.mean(values) for _room, values in sorted(grouped.items())]
    return {
        "query_count": len(rows),
        "room_count": len(grouped),
        "mean_localization_error_m": float(errors.mean()),
        "median_localization_error_m": float(np.median(errors)),
        "room_macro_mean_localization_error_m": float(statistics.mean(room_means)),
        "mean_oracle_error_m": float(oracle.mean()),
        "median_oracle_error_m": float(np.median(oracle)),
        "mean_excess_error_m": float(excess.mean()),
        "median_excess_error_m": float(np.median(excess)),
        "success_0_5m": float(np.mean([item["success_0_5m"] for item in rows])),
        "success_1_0m": float(np.mean([item["success_1_0m"] for item in rows])),
        "oracle_normalized_success_0_5m": float(
            np.mean([item["oracle_normalized_success_0_5m"] for item in rows])
        ),
        "oracle_normalized_success_1_0m": float(
            np.mean([item["oracle_normalized_success_1_0m"] for item in rows])
        ),
    }


def summarize_arm(results: list[dict]) -> dict:
    output = {}
    for count in (1, 4, 8):
        rows = []
        for result in results:
            row = dict(result["metrics"][str(count)])
            row["room"] = result["room"]
            rows.append(row)
        output[str(count)] = _summarize_metric_rows(rows)
    return output


def summarize_random_baseline(results: list[dict]) -> dict:
    rows = []
    for result in results:
        row = dict(result["random_candidate_metrics"])
        row["room"] = result["room"]
        rows.append(row)
    return _summarize_metric_rows(rows)


def aggregate_pilot(pilot_manifest: dict, arm_dirs: dict[str, Path | str]) -> dict:
    if set(arm_dirs) != {"vanilla", "fa_bf"}:
        raise ValueError("exactly the vanilla and fa_bf arms are required")
    runs = {}
    results = {}
    for arm, path in arm_dirs.items():
        runs[arm], results[arm] = load_completed_arm(path, pilot_manifest)
    vanilla_identity = runs["vanilla"]["identity"]
    fa_identity = runs["fa_bf"]["identity"]
    shared_keys = (
        "agree_checkpoint_sha256",
        "context_manifest_sha256",
        "geometry_audit_sha256",
        "pilot_manifest_sha256",
        "query_indices",
        "n_context",
        "score_sample_counts",
        "tau",
        "sample_seed",
        "candidate_batch_size",
        "sampler",
    )
    if any(vanilla_identity[key] != fa_identity[key] for key in shared_keys):
        raise RuntimeError("model arms do not share the frozen evaluation protocol")
    for vanilla, fa in zip(results["vanilla"], results["fa_bf"]):
        if (
            vanilla["query_id"] != fa["query_id"]
            or vanilla["candidate_indices_sha256"] != fa["candidate_indices_sha256"]
            or vanilla["random_candidate_metrics"] != fa["random_candidate_metrics"]
        ):
            raise RuntimeError("model arms do not share query candidates/baseline")
    payload = {
        "schema_version": 1,
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_count": pilot_manifest["query_count"],
        "room_count": pilot_manifest["room_count"],
        "n_context": vanilla_identity["n_context"],
        "score_sample_counts": vanilla_identity["score_sample_counts"],
        "tau": vanilla_identity["tau"],
        "arms": {
            "vanilla": {
                "run_manifest_sha256": runs["vanilla"]["sha256"],
                "summary": summarize_arm(results["vanilla"]),
            },
            "fa_bf": {
                "run_manifest_sha256": runs["fa_bf"]["sha256"],
                "summary": summarize_arm(results["fa_bf"]),
            },
        },
        "random_candidate_baseline": summarize_random_baseline(results["vanilla"]),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def render_markdown(aggregate: dict) -> str:
    lines = [
        "# Exp_09 64-query localization pilot results",
        "",
        f"Pilot SHA-256: `{aggregate['pilot_manifest_sha256']}`. "
        f"Scope: {aggregate['query_count']} queries / {aggregate['room_count']} rooms; "
        f"N_ctx={aggregate['n_context']}; K_gen={aggregate['score_sample_counts']}.",
        "",
        "| Arm | K_gen | Mean error (m) | Median error (m) | Success@0.5 | Success@1.0 | Oracle-normalized@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("vanilla", "fa_bf"):
        for count in ("1", "4", "8"):
            item = aggregate["arms"][arm]["summary"][count]
            lines.append(
                f"| {arm} | {count} | {item['mean_localization_error_m']:.3f} | "
                f"{item['median_localization_error_m']:.3f} | {item['success_0_5m']:.3f} | "
                f"{item['success_1_0m']:.3f} | {item['oracle_normalized_success_0_5m']:.3f} |"
            )
    random = aggregate["random_candidate_baseline"]
    lines += [
        f"| random candidate | — | {random['mean_localization_error_m']:.3f} | "
        f"{random['median_localization_error_m']:.3f} | {random['success_0_5m']:.3f} | "
        f"{random['success_1_0m']:.3f} | {random['oracle_normalized_success_0_5m']:.3f} |",
        "",
        "This is a room-stratified diagnostic pilot (four targets per room), not the complete 5,337-query unseen-room evaluation.",
        "",
    ]
    return "\n".join(lines)


def save_aggregate(aggregate: dict, json_path: Path, markdown_path: Path) -> None:
    verify = {key: value for key, value in aggregate.items() if key != "sha256"}
    if aggregate.get("sha256") != canonical_sha256(verify):
        raise ValueError("aggregate hash is stale")
    for path, text in (
        (json_path, json.dumps(aggregate, indent=2, sort_keys=True) + "\n"),
        (markdown_path, render_markdown(aggregate)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, path)
