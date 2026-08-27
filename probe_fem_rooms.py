#!/usr/bin/env python3
"""Run a resumable three-frequency FEM smoke test across audited rooms."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from probe_fem_small_room import _solve_dictionary, _three_probe_bins
from src.baselines.fem_pipeline import FEM_MAXIMUM_EDGE_M
from src.baselines.fem_sabine import estimate_context_t60, sabine_boundary
from src.baselines.room_helps_sparse import (
    extract_rir_frequency_response,
    room_helps_pulse_omp,
)
from src.localization.ar_queries import load_context_manifest
from src.localization.baseline_experiment import (
    load_room_tetrahedral_mesh,
    load_tetrahedral_mesh_manifest,
)
from src.localization.engine import load_frozen_query, reconstruct_query_candidates
from src.localization.pilot import canonical_sha256, load_pilot_manifest, resolve_pilot_records
from src.localization.runner import file_sha256
from src.baselines.fem_solver import (
    SparseDirectSolveSession,
    SparseDirectSolverOptions,
    assemble_p1_matrices,
    barycentric_interpolation_matrix,
    build_barycentric_point_locator,
    resolve_sparse_direct_backend,
)


SCHEMA_VERSION = 1
DEFAULT_CONTEXT_COUNTS = (1, 8)


def _load_geometry_audit(path: Path) -> dict:
    payload = json.loads(path.read_text())
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_sha256(content):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    return payload


def _hashed_payload(payload: dict) -> dict:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    content["sha256"] = canonical_sha256(content)
    return content


def _atomic_json(path: Path, payload: dict) -> None:
    content = _hashed_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_completed(path: Path, *, run_sha256: str, query_id: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    expected = payload.pop("sha256", None)
    if expected != canonical_sha256(payload):
        raise RuntimeError(f"stale FEM smoke-test hash: {path}")
    payload["sha256"] = expected
    if payload.get("run_manifest_sha256") != run_sha256:
        raise RuntimeError(f"FEM smoke-test run identity mismatch: {path}")
    if payload.get("query_id") != query_id:
        raise RuntimeError(f"FEM smoke-test query identity mismatch: {path}")
    return payload


def _run_query(
    *,
    selected: dict,
    record: dict,
    geometry_audit: dict,
    tetra_manifest: dict,
    dataset_root: Path,
    context_counts: tuple[int, ...],
    maximum_edge_m: float,
    maximum_residual: float,
    run_sha256: str,
    solver_options: SparseDirectSolverOptions,
) -> dict:
    started = time.perf_counter()
    room = selected["room"]
    observed, metadata = load_frozen_query(record, dataset_root)
    candidates = reconstruct_query_candidates(record, geometry_audit)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("candidate count differs from frozen pilot")
    truth = np.asarray(record["source_global"], dtype=np.float64)
    oracle_index = int(np.argmin(np.linalg.norm(candidates - truth, axis=1)))
    bin_indices, frequencies = _three_probe_bins(22050, observed.shape[-1])
    observed_bins = extract_rir_frequency_response(
        observed, bin_indices, sample_count=observed.shape[-1]
    )

    mesh, mesh_metadata = load_room_tetrahedral_mesh(
        room, tetra_manifest, geometry_audit
    )
    matrices = assemble_p1_matrices(mesh)
    point_locator = build_barycentric_point_locator(mesh)
    receiver_interpolation = barycentric_interpolation_matrix(
        mesh,
        np.asarray(record["receiver_global"], dtype=np.float64).reshape(1, 3),
        locator=point_locator,
    ).toarray()[0]
    candidate_interpolation = barycentric_interpolation_matrix(
        mesh, candidates, locator=point_locator
    )
    if matrices.maximum_element_edge_m > maximum_edge_m + 1e-12:
        raise ValueError(
            f"mesh hmax {matrices.maximum_element_edge_m:.9g} exceeds "
            f"the {maximum_edge_m:.9g} m smoke-test gate"
        )

    contexts = {}
    all_gates = {
        "mesh_resolution": True,
        "mesh_quality": bool(matrices.minimum_element_mean_ratio > 0.0),
    }
    with SparseDirectSolveSession(solver_options) as solver_session:
        for context_count in context_counts:
            stage_started = time.perf_counter()
            t60 = estimate_context_t60(metadata["context_audio"][:context_count])
            boundary = sabine_boundary(
                matrices.volume_m3, matrices.surface_area_m2, t60.t60_seconds
            )
            response, residuals, reciprocity, solver_profile = _solve_dictionary(
                mesh,
                matrices,
                np.asarray(record["receiver_global"], dtype=np.float64),
                candidates,
                frequencies,
                boundary.normalized_impedance,
                diagnostic_frequency_index=len(frequencies) // 2,
                receiver_interpolation=receiver_interpolation,
                candidate_interpolation=candidate_interpolation,
                point_locator=point_locator,
                solver_options=solver_options,
                solver_session=solver_session,
            )
            if not np.isfinite(response).all() or not np.isfinite(residuals).all():
                raise RuntimeError("FEM smoke-test response contains non-finite values")

            synthetic_gain = 0.7 - 0.2j
            synthetic = room_helps_pulse_omp(
                response,
                synthetic_gain * response[oracle_index],
                source_count=1,
            )
            real = room_helps_pulse_omp(response, observed_bins, source_count=1)
            winner = int(real.support[0])
            coefficient = real.coefficients[0]
            gates = {
                "solver_residual": bool(float(residuals.max()) <= maximum_residual),
                "reciprocity": bool(reciprocity["relative_difference"] <= 1e-10),
                "passive_dft_sign": bool(reciprocity["passive_dft_sign"]),
                "synthetic_omp_support": bool(synthetic.support == (oracle_index,)),
                "synthetic_omp_residual": bool(
                    synthetic.relative_residual_norm <= 1e-10
                ),
            }
            all_gates.update(
                {f"K{context_count}_{name}": value for name, value in gates.items()}
            )
            contexts[str(context_count)] = {
                "t60_seconds": float(t60.t60_seconds),
                "t60_valid_contexts": int(t60.valid_count),
                "t60_invalid_contexts": int(t60.invalid_count),
                "sabine_absorption": float(boundary.absorption),
                "normalized_impedance": float(boundary.normalized_impedance),
                "maximum_relative_solver_residual": float(residuals.max()),
                "solver_profile": solver_profile,
                "reciprocity": reciprocity,
                "synthetic_omp": {
                    "support": list(synthetic.support),
                    "relative_residual_norm": float(
                        synthetic.relative_residual_norm
                    ),
                },
                "real_three_frequency_omp": {
                    "support": list(real.support),
                    "coefficient_real": float(coefficient.real),
                    "coefficient_imag": float(coefficient.imag),
                    "relative_residual_norm": float(real.relative_residual_norm),
                    "localization_error_m": float(
                        np.linalg.norm(candidates[winner] - truth)
                    ),
                    "oracle_error_m": float(
                        np.linalg.norm(candidates[oracle_index] - truth)
                    ),
                },
                "stage_seconds": float(time.perf_counter() - stage_started),
                "gates": gates,
            }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_manifest_sha256": run_sha256,
        "query_index": int(selected["index"]),
        "query_id": selected["query_id"],
        "scene": selected["scene"],
        "room": room,
        "candidate_count": len(candidates),
        "oracle_candidate_index": oracle_index,
        "mesh": {
            "metadata": mesh_metadata,
            "node_count": len(mesh.nodes),
            "element_count": len(mesh.elements),
            "maximum_element_edge_m": float(matrices.maximum_element_edge_m),
            "minimum_element_mean_ratio": float(matrices.minimum_element_mean_ratio),
            "volume_m3": float(matrices.volume_m3),
            "surface_area_m2": float(matrices.surface_area_m2),
        },
        "probe_bin_indices": bin_indices.tolist(),
        "probe_frequencies_hz": frequencies.tolist(),
        "context_counts": list(context_counts),
        "contexts": contexts,
        "gates": all_gates,
        "passed": bool(all(all_gates.values())),
        "solver": {
            "backend": solver_session.options.backend,
            "superlu_ordering": solver_session.options.superlu_ordering,
            "threads": solver_session.options.threads,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "maximum_resident_memory_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    del matrices, mesh
    gc.collect()
    return payload


def _available_memory_gib() -> float:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0])
    return fields["MemAvailable"] / 1024**2


def _validate_parallel_resources(
    *, room_workers: int, solver_threads: int, minimum_memory_gib_per_worker: float
) -> None:
    if room_workers <= 0:
        raise ValueError("room workers must be positive")
    if solver_threads <= 0:
        raise ValueError("solver threads must be positive")
    if minimum_memory_gib_per_worker <= 0:
        raise ValueError("minimum memory per worker must be positive")
    logical_cpus = os.cpu_count() or 1
    if room_workers * solver_threads > logical_cpus:
        raise ValueError(
            "room_workers * solver_threads exceeds available logical CPUs"
        )
    required_memory = room_workers * minimum_memory_gib_per_worker
    available_memory = _available_memory_gib()
    if required_memory > available_memory:
        raise RuntimeError(
            f"parallel FEM launch requires {required_memory:.1f} GiB but only "
            f"{available_memory:.1f} GiB is available"
        )


def _run_query_worker(ordinal: int, total: int, kwargs: dict) -> dict:
    print(
        f"[{ordinal}/{total}] test {kwargs['selected']['room']} "
        f"pid={os.getpid()}",
        flush=True,
    )
    return _run_query(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tetra-mesh-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-counts", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--maximum-edge-m", type=float, default=FEM_MAXIMUM_EDGE_M)
    parser.add_argument("--maximum-residual", type=float, default=1e-8)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--solver-backend",
        choices=("auto", "superlu", "mkl_pardiso"),
        default="auto",
    )
    parser.add_argument(
        "--superlu-ordering",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default="MMD_AT_PLUS_A",
    )
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--room-workers", type=int, default=1)
    parser.add_argument("--minimum-memory-gib-per-worker", type=float, default=48.0)
    parser.add_argument(
        "--mkl-runtime",
        type=Path,
        help="optional explicit libmkl_rt path, exported as MKL_RT before backend load",
    )
    args = parser.parse_args()

    if args.mkl_runtime is not None:
        runtime = args.mkl_runtime.resolve()
        if not runtime.is_file():
            raise FileNotFoundError(runtime)
        os.environ["MKL_RT"] = str(runtime)
    resolved_backend = resolve_sparse_direct_backend(args.solver_backend)
    solver_options = SparseDirectSolverOptions(
        backend=resolved_backend,
        superlu_ordering=args.superlu_ordering,
        threads=args.solver_threads,
    )
    _validate_parallel_resources(
        room_workers=args.room_workers,
        solver_threads=args.solver_threads,
        minimum_memory_gib_per_worker=args.minimum_memory_gib_per_worker,
    )

    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("FEM smoke-test output must stay inside the worktree") from error
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = tuple(int(value) for value in args.context_counts)
    if counts != tuple(sorted(set(counts))) or any(value <= 0 for value in counts):
        raise ValueError("context counts must be unique, increasing, and positive")
    context_manifest = load_context_manifest(args.context_manifest)
    geometry_audit = _load_geometry_audit(args.geometry_audit)
    pilot_manifest = load_pilot_manifest(args.pilot_manifest)
    joined = resolve_pilot_records(pilot_manifest, context_manifest, geometry_audit)
    if args.query_limit is not None:
        if args.query_limit <= 0 or args.query_limit > len(joined):
            raise ValueError("query limit is outside the pilot range")
        joined = joined[: args.query_limit]
    tetra_manifest = load_tetrahedral_mesh_manifest(args.tetra_mesh_manifest)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": "fem_three_frequency_room_smoke",
        "tetra_manifest_sha256": file_sha256(args.tetra_mesh_manifest),
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_indices": [int(selected["index"]) for selected, _record, _ in joined],
        "context_counts": list(counts),
        "maximum_edge_m": float(args.maximum_edge_m),
        "maximum_residual": float(args.maximum_residual),
        "frequency_selection": "first_middle_last_bins_in_80_300_hz",
        "solver_backend": resolved_backend,
        "superlu_ordering": args.superlu_ordering,
        "solver_threads": int(args.solver_threads),
        "room_workers": int(args.room_workers),
        "minimum_memory_gib_per_worker": float(
            args.minimum_memory_gib_per_worker
        ),
    }
    run_path = output_dir / "run_manifest.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text())
        expected = existing.pop("sha256", None)
        if expected != canonical_sha256(existing) or existing != identity:
            raise RuntimeError("FEM smoke-test run manifest mismatch")
        existing["sha256"] = expected
        run_manifest = existing
    else:
        run_manifest = _hashed_payload(identity)
        _atomic_json(run_path, run_manifest)
    run_sha256 = run_manifest["sha256"]

    completed = 0
    failures = 0
    pending = []
    for ordinal, (selected, record, _geometry) in enumerate(joined, start=1):
        result_path = output_dir / "queries" / f"query_{int(selected['index']):05d}.json"
        previous = _load_completed(
            result_path,
            run_sha256=run_sha256,
            query_id=selected["query_id"],
        )
        if previous is not None:
            completed += int(previous.get("passed", False))
            failures += int(not previous.get("passed", False))
            print(f"[{ordinal}/{len(joined)}] resume {selected['room']}", flush=True)
            continue
        pending.append(
            (
                ordinal,
                selected,
                result_path,
                {
                    "selected": selected,
                    "record": record,
                    "geometry_audit": geometry_audit,
                    "tetra_manifest": tetra_manifest,
                    "dataset_root": args.dataset_root,
                    "context_counts": counts,
                    "maximum_edge_m": float(args.maximum_edge_m),
                    "maximum_residual": float(args.maximum_residual),
                    "run_sha256": run_sha256,
                    "solver_options": solver_options,
                },
            )
        )

    def record_result(ordinal, selected, result_path, result=None, error=None):
        nonlocal completed, failures
        if error is not None:
            result = {
                "schema_version": SCHEMA_VERSION,
                "run_manifest_sha256": run_sha256,
                "query_index": int(selected["index"]),
                "query_id": selected["query_id"],
                "scene": selected["scene"],
                "room": selected["room"],
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
            _atomic_json(result_path, result)
            failures += 1
            print(
                f"[{ordinal}/{len(joined)}] fail {selected['room']}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            return
        _atomic_json(result_path, result)
        completed += int(result["passed"])
        failures += int(not result["passed"])
        print(
            f"[{ordinal}/{len(joined)}] {'pass' if result['passed'] else 'fail'} "
            f"{selected['room']} seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )

    if args.room_workers == 1:
        for ordinal, selected, result_path, kwargs in pending:
            print(f"[{ordinal}/{len(joined)}] test {selected['room']}", flush=True)
            try:
                result = _run_query(**kwargs)
            except Exception as error:
                record_result(ordinal, selected, result_path, error=error)
                if not args.continue_on_error:
                    raise
            else:
                record_result(ordinal, selected, result_path, result=result)
    else:
        process_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.room_workers,
            mp_context=process_context,
        ) as executor:
            futures = {
                executor.submit(
                    _run_query_worker, ordinal, len(joined), kwargs
                ): (ordinal, selected, result_path)
                for ordinal, selected, result_path, kwargs in pending
            }
            for future in concurrent.futures.as_completed(futures):
                ordinal, selected, result_path = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    record_result(ordinal, selected, result_path, error=error)
                    if not args.continue_on_error:
                        for other in futures:
                            other.cancel()
                        raise
                else:
                    record_result(ordinal, selected, result_path, result=result)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_manifest_sha256": run_sha256,
        "requested_rooms": len(joined),
        "passed_rooms": completed,
        "failed_rooms": failures,
        "output_dir": str(output_dir),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
