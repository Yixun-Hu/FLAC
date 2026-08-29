#!/usr/bin/env python3
"""Run the matched-band FEM--AGREE selector on frozen Depth-AABB queries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.fem_agree import (
    exact_bandlimit_waveforms,
    peak_normalize_waveforms,
    score_fem_waveforms_with_agree,
)
from src.baselines.fem_pipeline import (
    load_tetrahedral_mesh_npz,
    prepare_fem_query_interpolation,
    prepare_fem_room_operators,
    run_fem_sabine_forward,
)
from src.baselines.fem_sabine import bandlimited_response_to_waveform
from src.baselines.fem_solver import SparseDirectSolveSession, SparseDirectSolverOptions
from src.localization.ar_queries import load_context_manifest
from src.localization.baseline_runner import score_fem_room_helps_candidates
from src.localization.engine import (
    filter_frozen_query_candidates,
    load_agree_retrieval,
    load_frozen_query,
    reconstruct_room_base_candidates,
)
from src.localization.pilot import canonical_sha256
from src.localization.runner import file_sha256, verify_hashed_payload
from src.localization.scoring import localization_metrics, stable_argmax


SAMPLE_COUNTS = (1, 4, 8)
FREQUENCY_BAND_HZ = (80.0, 300.0)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def hashed(payload: dict) -> dict:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def load_hashed_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text())
    verify_hashed_payload(payload, label)
    return payload


def compact_forward_audit(audit: dict) -> dict:
    profile = audit["solver_profile"]
    timing_keys = (
        "symbolic_analysis_seconds",
        "factorization_seconds",
        "solve_seconds",
        "system_construction_seconds",
        "residual_and_sampling_seconds",
    )
    return {
        key: value for key, value in audit.items() if key != "solver_profile"
    } | {
        "solver_profile": {
            "frequency_count": len(profile),
            "backend": profile[0]["backend"],
            "threads": profile[0]["threads"],
            **{
                f"total_{key}": float(sum(float(row[key]) for row in profile))
                for key in timing_keys
            },
        }
    }


def resolve_records(selection: dict, context: dict, query_index: int | None) -> list[tuple[dict, dict]]:
    context_by_index = {int(row["index"]): row for row in context["records"]}
    resolved = []
    for selected in selection["records"]:
        index = int(selected["index"])
        if query_index is not None and index != query_index:
            continue
        try:
            record = context_by_index[index]
        except KeyError as error:
            raise RuntimeError(f"query {index} is absent from the context manifest") from error
        for key in ("query_id", "scene", "room"):
            if selected[key] != record[key]:
                raise RuntimeError(f"query {index} has mismatched {key}")
        resolved.append((selected, record))
    if query_index is not None and len(resolved) != 1:
        raise RuntimeError(f"query {query_index} is not unique in the selection")
    return resolved


def validate_source_result(
    source_result_dir: Path,
    selected: dict,
    candidates: np.ndarray,
) -> tuple[dict, Path]:
    index = int(selected["index"])
    path = source_result_dir / f"query_{index:05d}_depth_aabb_result.json"
    source = load_hashed_json(path, f"source FEM query {index}")
    if source.get("method") != "fem_sabine_depth_aabb":
        raise RuntimeError(f"query {index} source method mismatch")
    if source.get("query_id") != selected["query_id"]:
        raise RuntimeError(f"query {index} source identity mismatch")
    if int(source.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError(f"query {index} source candidate count mismatch")
    if source.get("candidate_indices_sha256") not in (
        None,
        selected["candidate_indices_sha256"],
    ):
        raise RuntimeError(f"query {index} source candidate hash mismatch")
    arrays_path = source_result_dir / source["arrays_file"]
    if file_sha256(arrays_path) != source["arrays_sha256"]:
        raise RuntimeError(f"query {index} source arrays hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        stored_candidates = archive["candidates"]
    if not np.array_equal(stored_candidates, candidates.astype(np.float32)):
        raise RuntimeError(f"query {index} candidate coordinates differ from source FEM")
    mesh_path = source_result_dir / source["mesh_file"]
    if file_sha256(mesh_path) != source["mesh_sha256"]:
        raise RuntimeError(f"query {index} source mesh hash mismatch")
    return source, mesh_path


def response_cache_is_complete(path: Path, audit_path: Path, expected: dict) -> bool:
    if not path.is_file() or not audit_path.is_file():
        return False
    audit = load_hashed_json(audit_path, "FEM response cache")
    return (
        audit.get("method") == "fem_sabine_depth_aabb_response_cache"
        and all(audit.get(key) == value for key, value in expected.items())
        and audit.get("response_file_sha256") == file_sha256(path)
    )


def solve_one(
    *,
    selected: dict,
    record: dict,
    geometry_audit: dict,
    room_base_cache: dict[str, np.ndarray],
    dataset_root: Path,
    source_result_dir: Path,
    output_dir: Path,
    selection_sha256: str,
    context_sha256: str,
    solver_options: SparseDirectSolverOptions,
) -> dict:
    index = int(selected["index"])
    response_dir = output_dir / "responses"
    response_path = response_dir / f"query_{index:05d}_response.npz"
    audit_path = response_dir / f"query_{index:05d}.json"
    expected = {
        "query_index": index,
        "query_id": selected["query_id"],
        "selection_sha256": selection_sha256,
        "context_manifest_sha256": context_sha256,
        "candidate_indices_sha256": selected["candidate_indices_sha256"],
    }
    if response_cache_is_complete(response_path, audit_path, expected):
        return {"query_index": index, "status": "resume", "seconds": 0.0}

    started = time.perf_counter()
    if record["room"] not in room_base_cache:
        room_base_cache[record["room"]] = reconstruct_room_base_candidates(
            record["room"], geometry_audit
        )
    candidates = filter_frozen_query_candidates(
        record, geometry_audit, room_base_cache[record["room"]]
    )
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError(f"query {index} selection candidate count mismatch")
    source, mesh_path = validate_source_result(source_result_dir, selected, candidates)
    mesh, mesh_metadata = load_tetrahedral_mesh_npz(mesh_path)
    observed, metadata = load_frozen_query(record, dataset_root)
    operators = prepare_fem_room_operators(mesh)
    receiver_load, candidate_interpolation = prepare_fem_query_interpolation(
        operators,
        np.asarray(record["receiver_global"], dtype=np.float64),
        candidates,
    )
    solve_started = time.perf_counter()
    with SparseDirectSolveSession(solver_options) as session:
        forward = run_fem_sabine_forward(
            mesh,
            receiver_point=np.asarray(record["receiver_global"], dtype=np.float64),
            candidate_points=candidates,
            context_waveforms=metadata["context_audio"][:8],
            construct_waveforms=False,
            room_operators=operators,
            receiver_load=receiver_load,
            candidate_interpolation=candidate_interpolation,
            solver_options=solver_options,
            solver_session=session,
        )
    solve_seconds = time.perf_counter() - solve_started

    _scores, recovery = score_fem_room_helps_candidates(
        forward.response,
        forward.bin_indices,
        observed,
        sample_count=observed.shape[-1],
    )
    omp_prediction = int(recovery.support[0])
    if omp_prediction != int(source["metrics"]["prediction_index"]):
        raise RuntimeError(f"query {index} rerun does not reproduce source OMP winner")

    atomic_npz(
        response_path,
        response=forward.response.numpy().astype(np.complex64),
        candidates=candidates.astype(np.float64),
        bin_indices=forward.bin_indices.astype(np.int64),
        frequencies_hz=forward.frequencies_hz.astype(np.float64),
    )
    payload = hashed(
        {
            "schema_version": 1,
            "method": "fem_sabine_depth_aabb_response_cache",
            **expected,
            "room": record["room"],
            "candidate_count": len(candidates),
            "source_fem_result_sha256": source["sha256"],
            "source_mesh_sha256": source["mesh_sha256"],
            "mesh_metadata": mesh_metadata,
            "frequency_band_hz": list(FREQUENCY_BAND_HZ),
            "context_count": 8,
            "fem_audit": compact_forward_audit(forward.audit),
            "omp_prediction_index": omp_prediction,
            "omp_relative_residual_norm": recovery.relative_residual_norm,
            "response_file": response_path.name,
            "response_file_sha256": file_sha256(response_path),
            "runtime_seconds": {
                "solve": solve_seconds,
                "total": time.perf_counter() - started,
            },
        }
    )
    atomic_json(audit_path, payload)
    return {
        "query_index": index,
        "status": "completed",
        "seconds": payload["runtime_seconds"]["total"],
    }


def score_result_is_complete(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    payload = load_hashed_json(path, "FEM--AGREE result")
    return (
        payload.get("method") == "fem_sabine_depth_aabb_agree"
        and all(payload.get(key) == value for key, value in expected.items())
    )


@torch.inference_mode()
def score_one(
    *,
    selected: dict,
    record: dict,
    dataset_root: Path,
    source_result_dir: Path,
    output_dir: Path,
    retrieval,
    agree_sha256: str,
    selection_sha256: str,
    context_sha256: str,
    device: str,
    candidate_batch_size: int,
    score_seed: int,
    target_peak: float,
    tau: float,
) -> dict:
    index = int(selected["index"])
    response_path = output_dir / "responses" / f"query_{index:05d}_response.npz"
    response_audit_path = output_dir / "responses" / f"query_{index:05d}.json"
    response_audit = load_hashed_json(response_audit_path, f"FEM response query {index}")
    if response_audit["response_file_sha256"] != file_sha256(response_path):
        raise RuntimeError(f"query {index} response cache hash mismatch")
    expected = {
        "query_index": index,
        "query_id": selected["query_id"],
        "selection_sha256": selection_sha256,
        "context_manifest_sha256": context_sha256,
        "candidate_indices_sha256": selected["candidate_indices_sha256"],
        "agree_checkpoint_sha256": agree_sha256,
        "response_cache_sha256": response_audit["sha256"],
    }
    result_dir = output_dir / "results"
    result_path = result_dir / f"query_{index:05d}.json"
    if score_result_is_complete(result_path, expected):
        return {"query_index": index, "status": "resume", "seconds": 0.0}

    started = time.perf_counter()
    with np.load(response_path, allow_pickle=False) as archive:
        response = torch.from_numpy(archive["response"])
        candidates = archive["candidates"]
        bin_indices = archive["bin_indices"]
        frequencies = archive["frequencies_hz"]
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError(f"query {index} cached candidate count mismatch")
    observed, _metadata = load_frozen_query(record, dataset_root)
    candidate_waveforms = bandlimited_response_to_waveform(
        response, bin_indices, sample_count=observed.shape[-1]
    )
    observed_waveform = exact_bandlimit_waveforms(
        observed.unsqueeze(0), bin_indices
    )
    normalized_candidates, candidate_peaks = peak_normalize_waveforms(
        candidate_waveforms, target_peak=target_peak
    )
    normalized_observed, observed_peak = peak_normalize_waveforms(
        observed_waveform, target_peak=target_peak
    )
    agree_started = time.perf_counter()
    scores_by_k, similarities, rng_seeds = score_fem_waveforms_with_agree(
        retrieval,
        normalized_candidates,
        normalized_observed,
        query_index=index,
        score_seed=score_seed,
        device=device,
        candidate_batch_size=candidate_batch_size,
        sample_counts=SAMPLE_COUNTS,
        tau=tau,
    )
    agree_seconds = time.perf_counter() - agree_started
    truth = np.asarray(record["source_global"], dtype=np.float64)
    metrics_by_k = {}
    for count, scores in scores_by_k.items():
        prediction_index = stable_argmax(scores)
        metrics = localization_metrics(candidates, truth, prediction_index)
        metrics.update(
            {
                "prediction_global": candidates[prediction_index].astype(float).tolist(),
                "winning_score": float(scores[prediction_index]),
                "mean_candidate_score": float(scores.mean()),
            }
        )
        metrics_by_k[str(count)] = metrics

    source = load_hashed_json(
        source_result_dir / f"query_{index:05d}_depth_aabb_result.json",
        f"source FEM query {index}",
    )
    arrays_path = result_dir / f"query_{index:05d}_scores.npz"
    atomic_npz(
        arrays_path,
        candidates=candidates.astype(np.float32),
        similarities=similarities.numpy().astype(np.float32),
        scores=np.stack([scores_by_k[count].numpy() for count in SAMPLE_COUNTS], axis=1).astype(
            np.float32
        ),
        sample_counts=np.asarray(SAMPLE_COUNTS, dtype=np.int64),
        frequencies_hz=frequencies.astype(np.float64),
    )
    payload = hashed(
        {
            "schema_version": 1,
            "method": "fem_sabine_depth_aabb_agree",
            "interpretation": (
                "Depth-AABB FEM response scored by frozen AGREE after identical exact "
                "80--300 Hz bandlimiting and per-waveform peak normalization"
            ),
            **expected,
            "scene": record["scene"],
            "room": record["room"],
            "receiver_id": selected["receiver_id"],
            "candidate_count": len(candidates),
            "context_count": 8,
            "frequency_band_hz": list(FREQUENCY_BAND_HZ),
            "frequency_count": len(frequencies),
            "waveform_preprocessing": {
                "observed_and_candidate_bandlimit_identical": True,
                "sample_alignment_preserved": True,
                "normalization": "independent_per_waveform_absolute_peak",
                "target_peak": target_peak,
                "observed_pre_normalization_peak": float(observed_peak[0]),
                "candidate_pre_normalization_peak_min": float(candidate_peaks.min()),
                "candidate_pre_normalization_peak_median": float(candidate_peaks.median()),
                "candidate_pre_normalization_peak_max": float(candidate_peaks.max()),
            },
            "agree_protocol": {
                "sample_counts": list(SAMPLE_COUNTS),
                "log_mean_exp_tau": tau,
                "score_seed": score_seed,
                "rng_seeds": rng_seeds,
                "candidate_batch_size": candidate_batch_size,
                "observation_encode_count": 1,
            },
            "metrics_by_k_agree": metrics_by_k,
            "source_fem_omp_metrics": source["metrics"],
            "source_fem_result_sha256": source["sha256"],
            "arrays_file": arrays_path.name,
            "arrays_sha256": file_sha256(arrays_path),
            "runtime_seconds": {
                "agree_scoring": agree_seconds,
                "total": time.perf_counter() - started,
            },
        }
    )
    atomic_json(result_path, payload)
    return {
        "query_index": index,
        "status": "completed",
        "seconds": payload["runtime_seconds"]["total"],
    }


def aggregate_metric_rows(rows: list[tuple[str, dict]]) -> dict:
    errors = np.asarray([float(metric["localization_error_m"]) for _room, metric in rows])
    rooms: dict[str, list[dict]] = defaultdict(list)
    for room, metric in rows:
        rooms[room].append(metric)
    return {
        "query_count": len(rows),
        "room_count": len(rooms),
        "mean_localization_error_m": float(errors.mean()),
        "median_localization_error_m": float(np.median(errors)),
        "success_0_5m": float(np.mean([metric["success_0_5m"] for _room, metric in rows])),
        "success_1_0m": float(np.mean([metric["success_1_0m"] for _room, metric in rows])),
        "oracle_normalized_success_0_5m": float(
            np.mean([metric["oracle_normalized_success_0_5m"] for _room, metric in rows])
        ),
        "room_macro_mean_localization_error_m": float(
            np.mean(
                [
                    np.mean([metric["localization_error_m"] for metric in metrics])
                    for metrics in rooms.values()
                ]
            )
        ),
        "room_macro_success_0_5m": float(
            np.mean([np.mean([metric["success_0_5m"] for metric in metrics]) for metrics in rooms.values()])
        ),
        "room_macro_success_1_0m": float(
            np.mean([np.mean([metric["success_1_0m"] for metric in metrics]) for metrics in rooms.values()])
        ),
        "room_macro_oracle_normalized_success_0_5m": float(
            np.mean(
                [
                    np.mean([metric["oracle_normalized_success_0_5m"] for metric in metrics])
                    for metrics in rooms.values()
                ]
            )
        ),
    }


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# FEM--AGREE on the Depth-AABB matched subset",
        "",
        (
            f"Scope: {summary['query_count']} strict-coverage queries / "
            f"{summary['room_count']} rooms; K_ctx=8; exact 80--300 Hz matched-band "
            "waveforms; independent per-waveform peak normalization to 0.95."
        ),
        "",
        "| Selector | AGREE samples | Mean error [m] | Median error [m] | SR@0.5m | SR@1.0m | Resolution-aware SR@0.5m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for count in SAMPLE_COUNTS:
        metric = summary["fem_agree_by_k"][str(count)]
        lines.append(
            "| FEM--AGREE | "
            f"{count} | {metric['mean_localization_error_m']:.3f} | "
            f"{metric['median_localization_error_m']:.3f} | "
            f"{100 * metric['success_0_5m']:.1f}% | "
            f"{100 * metric['success_1_0m']:.1f}% | "
            f"{100 * metric['oracle_normalized_success_0_5m']:.1f}% |"
        )
    omp = summary["fem_omp_reference"]
    lines.append(
        "| FEM--OMP reference | -- | "
        f"{omp['mean_localization_error_m']:.3f} | "
        f"{omp['median_localization_error_m']:.3f} | "
        f"{100 * omp['success_0_5m']:.1f}% | "
        f"{100 * omp['success_1_0m']:.1f}% | "
        f"{100 * omp['oracle_normalized_success_0_5m']:.1f}% |"
    )
    lines.extend(
        [
            "",
            "AGREE samples are stochastic audio-encoder samples aggregated with the same nested log-mean-exp rule (tau=0.1); they are not FEM solves or K_gen samples.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(output_dir: Path, records: list[tuple[dict, dict]], selection_sha256: str) -> dict:
    results = []
    for selected, record in records:
        index = int(selected["index"])
        path = output_dir / "results" / f"query_{index:05d}.json"
        if not path.is_file():
            raise RuntimeError(f"missing FEM--AGREE result for query {index}")
        payload = load_hashed_json(path, f"FEM--AGREE query {index}")
        results.append((record["room"], payload))
    agree_by_k = {
        str(count): aggregate_metric_rows(
            [(room, payload["metrics_by_k_agree"][str(count)]) for room, payload in results]
        )
        for count in SAMPLE_COUNTS
    }
    omp = aggregate_metric_rows(
        [(room, payload["source_fem_omp_metrics"]) for room, payload in results]
    )
    summary = hashed(
        {
            "schema_version": 1,
            "method": "fem_sabine_depth_aabb_agree",
            "selection_sha256": selection_sha256,
            "query_count": len(results),
            "room_count": len({room for room, _payload in results}),
            "fem_agree_by_k": agree_by_k,
            "fem_omp_reference": omp,
        }
    )
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(render_summary_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("solve", "score", "all"), default="all")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-result-dir", type=Path, required=True)
    parser.add_argument("--agree-ckpt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-batch-size", type=int, default=32)
    parser.add_argument("--score-seed", type=int, default=42)
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--query-index", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--solver-backend", choices=("superlu", "mkl_pardiso"), default="mkl_pardiso"
    )
    parser.add_argument("--solver-threads", type=int, default=24)
    parser.add_argument("--mkl-runtime", type=Path)
    args = parser.parse_args()
    if args.candidate_batch_size <= 0 or args.solver_threads <= 0:
        raise ValueError("batch size and solver threads must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard count)")
    if args.query_index is not None and args.shard_count != 1:
        raise ValueError("query-index and sharding are mutually exclusive")
    if args.mkl_runtime is not None:
        if not args.mkl_runtime.is_file():
            raise FileNotFoundError(args.mkl_runtime)
        os.environ["MKL_RT"] = str(args.mkl_runtime.resolve())
    if args.stage in ("score", "all") and args.agree_ckpt is None:
        raise ValueError("--agree-ckpt is required for scoring")

    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("output must remain inside the repository") from error
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = load_hashed_json(args.selection, "Depth-AABB selection")
    context = load_context_manifest(args.context_manifest)
    geometry_audit = load_hashed_json(args.geometry_audit, "geometry audit")
    records = resolve_records(selection, context, args.query_index)
    if args.shard_count != 1:
        records = [
            item
            for ordinal, item in enumerate(records)
            if ordinal % args.shard_count == args.shard_index
        ]
    if not records:
        raise RuntimeError("selection resolved no queries")

    if args.stage in ("solve", "all"):
        solver_options = SparseDirectSolverOptions(
            backend=args.solver_backend, threads=args.solver_threads
        )
        room_base_cache: dict[str, np.ndarray] = {}
        for ordinal, (selected, record) in enumerate(records, start=1):
            outcome = solve_one(
                selected=selected,
                record=record,
                geometry_audit=geometry_audit,
                room_base_cache=room_base_cache,
                dataset_root=args.dataset_root.resolve(),
                source_result_dir=args.source_result_dir.resolve(),
                output_dir=output_dir,
                selection_sha256=selection["sha256"],
                context_sha256=context["sha256"],
                solver_options=solver_options,
            )
            print(f"[solve {ordinal}/{len(records)}] {json.dumps(outcome, sort_keys=True)}", flush=True)

    if args.stage in ("score", "all"):
        agree_path = args.agree_ckpt.resolve()
        agree_sha256 = file_sha256(agree_path)
        torch.manual_seed(args.score_seed)
        retrieval = load_agree_retrieval(agree_path, args.device)
        for ordinal, (selected, record) in enumerate(records, start=1):
            outcome = score_one(
                selected=selected,
                record=record,
                dataset_root=args.dataset_root.resolve(),
                source_result_dir=args.source_result_dir.resolve(),
                output_dir=output_dir,
                retrieval=retrieval,
                agree_sha256=agree_sha256,
                selection_sha256=selection["sha256"],
                context_sha256=context["sha256"],
                device=args.device,
                candidate_batch_size=args.candidate_batch_size,
                score_seed=args.score_seed,
                target_peak=args.target_peak,
                tau=args.tau,
            )
            print(f"[score {ordinal}/{len(records)}] {json.dumps(outcome, sort_keys=True)}", flush=True)

    if args.query_index is None and args.shard_count == 1 and args.stage in ("score", "all"):
        print(json.dumps(summarize(output_dir, records, selection["sha256"]), indent=2))


if __name__ == "__main__":
    main()
