"""Resumable frozen-model localization execution for exp_09."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS,
    FA_CONTEXT_CONDITIONING_IDS,
    FA_DYNAMIC_CONDITIONING_IDS,
    SCORE_SAMPLE_COUNTS,
    SOURCE_CONDITIONING_IDS,
    cache_conditioning_branch,
    cache_invariant_conditioning_branch,
    candidate_metadata,
    candidate_seed,
    encode_audio_features,
    filter_frozen_query_candidates,
    generate_and_score_batch,
    generate_rir_batch,
    load_agree_retrieval,
    load_flac_module,
    load_frozen_query,
    reconstruct_room_base_candidates,
)
from src.localization.pilot import canonical_sha256, resolve_pilot_records
from src.localization.scoring import (
    deterministic_random_candidate,
    localization_metrics,
    log_mean_exp_scores,
    stable_argmax,
)


RUN_SCHEMA_VERSION = 1
QUERY_SCHEMA_VERSION = 1


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _hashed_payload(payload: dict) -> dict:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def verify_hashed_payload(payload: dict, label: str) -> None:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != canonical_sha256(content):
        raise RuntimeError(f"{label} SHA-256 mismatch")


def initialize_run(output_dir: Path, identity: dict) -> dict:
    """Create or strictly reopen a run manifest."""

    path = output_dir / "run_manifest.json"
    expected = _hashed_payload(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "identity": identity,
        }
    )
    if path.exists():
        existing = json.loads(path.read_text())
        verify_hashed_payload(existing, "run manifest")
        if existing != expected:
            raise RuntimeError("existing run manifest does not match requested execution")
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("nonempty output directory has no matching run manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, expected)
    return expected


def _cache_branch(module, metadata, ids, device: str, cond_method: str):
    if cond_method == "fa_invariant":
        return cache_invariant_conditioning_branch(
            module.diffusion.conditioner,
            metadata,
            ids,
            device,
            DEFAULT_FRAME_ANGLES,
        )
    return cache_conditioning_branch(
        module.diffusion.conditioner,
        metadata,
        ids,
        device,
    )


def _repeat_branch(branch: dict, repeats: int) -> dict:
    output = {}
    for key, (tokens, mask) in branch.items():
        repeated_tokens = torch.repeat_interleave(tokens, repeats, dim=0)
        repeated_mask = (
            torch.repeat_interleave(mask, repeats, dim=0) if mask is not None else None
        )
        output[key] = [repeated_tokens, repeated_mask]
    return output


def _query_paths(output_dir: Path, query_index: int) -> tuple[Path, Path]:
    stem = f"query_{query_index:05d}"
    return output_dir / "queries" / f"{stem}.json", output_dir / "queries" / f"{stem}.npz"


def completed_query_result(
    output_dir: Path,
    *,
    query_index: int,
    query_id: str,
    candidate_count: int,
    run_sha256: str,
    score_sample_counts: tuple[int, ...] = SCORE_SAMPLE_COUNTS,
) -> dict | None:
    json_path, npz_path = _query_paths(output_dir, query_index)
    if not json_path.exists():
        return None
    result = json.loads(json_path.read_text())
    verify_hashed_payload(result, f"query {query_index} result")
    if (
        result.get("query_index") != query_index
        or result.get("query_id") != query_id
        or result.get("candidate_count") != candidate_count
        or result.get("run_manifest_sha256") != run_sha256
    ):
        raise RuntimeError(f"query {query_index} resume identity mismatch")
    if not npz_path.is_file() or file_sha256(npz_path) != result.get("arrays_sha256"):
        raise RuntimeError(f"query {query_index} array artifact is missing or corrupt")
    with np.load(npz_path) as arrays:
        candidates = arrays["candidates"]
        similarities = arrays["similarities"]
        if candidates.shape != (candidate_count, 3):
            raise RuntimeError(f"query {query_index} candidate array shape mismatch")
        if similarities.shape != (candidate_count, max(score_sample_counts)):
            raise RuntimeError(f"query {query_index} score array shape mismatch")
        if not np.isfinite(candidates).all() or not np.isfinite(similarities).all():
            raise RuntimeError(f"query {query_index} artifact contains non-finite values")
    return result


def save_query_result(
    output_dir: Path,
    *,
    result: dict,
    candidates: np.ndarray,
    similarities: np.ndarray,
) -> dict:
    json_path, npz_path = _query_paths(output_dir, int(result["query_index"]))
    _atomic_npz(
        npz_path,
        candidates=np.asarray(candidates, dtype=np.float32),
        similarities=np.asarray(similarities, dtype=np.float32),
    )
    payload = dict(result)
    payload["arrays_file"] = str(npz_path.relative_to(output_dir))
    payload["arrays_sha256"] = file_sha256(npz_path)
    payload = _hashed_payload(payload)
    _atomic_json(json_path, payload)
    return payload


def _build_query_result(
    *,
    selected: dict,
    record: dict,
    candidates: np.ndarray,
    similarities: torch.Tensor,
    run_sha256: str,
    elapsed_seconds: float,
    peak_memory_bytes: int,
    tau: float,
    random_seed: int,
    score_sample_counts: tuple[int, ...] = SCORE_SAMPLE_COUNTS,
) -> dict:
    score_vectors = log_mean_exp_scores(similarities, score_sample_counts, tau=tau)
    metrics = {}
    truth = np.asarray(record["source_global"], dtype=np.float64)
    for count, scores in score_vectors.items():
        winner = stable_argmax(scores)
        values = localization_metrics(candidates, truth, winner)
        values["prediction_global"] = candidates[winner].astype(float).tolist()
        values["winning_score"] = float(scores[winner])
        values["mean_candidate_score"] = float(scores.mean())
        metrics[str(count)] = values
    random_index = deterministic_random_candidate(
        int(selected["index"]), len(candidates), seed=random_seed
    )
    random_metrics = localization_metrics(candidates, truth, random_index)
    random_metrics["prediction_global"] = candidates[random_index].astype(float).tolist()
    return {
        "schema_version": QUERY_SCHEMA_VERSION,
        "run_manifest_sha256": run_sha256,
        "query_index": int(selected["index"]),
        "query_id": selected["query_id"],
        "scene": selected["scene"],
        "room": selected["room"],
        "receiver_id": selected["receiver_id"],
        "candidate_count": len(candidates),
        "candidate_indices_sha256": selected["candidate_indices_sha256"],
        "source_global": list(map(float, record["source_global"])),
        "receiver_global": list(map(float, record["receiver_global"])),
        "n_context": len(record["contexts"]),
        "score_sample_counts": list(score_sample_counts),
        "tau": float(tau),
        "metrics": metrics,
        "random_candidate_metrics": random_metrics,
        "elapsed_seconds": float(elapsed_seconds),
        "peak_memory_bytes": int(peak_memory_bytes),
    }


@torch.inference_mode()
def execute_query(
    *,
    module,
    retrieval,
    selected: dict,
    record: dict,
    audit: dict,
    room_base: np.ndarray,
    dataset_root: Path,
    device: str,
    cond_method: str,
    candidate_batch_size: int,
    sample_seed: int,
    tau: float,
    run_sha256: str,
    score_sample_counts: tuple[int, ...] = SCORE_SAMPLE_COUNTS,
    synchronize_timing: bool = False,
    measure_core_forward: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run and score all candidates for one frozen target query."""

    device_type = torch.device(device).type
    if synchronize_timing and device_type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    candidates = filter_frozen_query_candidates(record, audit, room_base)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != 8:
        raise RuntimeError("formal pilot requires N_ctx=8")

    observed_gpu = observed.unsqueeze(0).to(device=device, dtype=torch.float32)
    observation_features = encode_audio_features(retrieval, observed_gpu)

    if measure_core_forward and device_type == "cuda":
        torch.cuda.synchronize(device)
    context_started = time.perf_counter()
    with torch.amp.autocast(device_type):
        context = _cache_branch(
            module,
            [metadata],
            FA_CONTEXT_CONDITIONING_IDS
            if cond_method == "fa_invariant"
            else CONTEXT_CONDITIONING_IDS,
            device,
            cond_method,
        )
    if measure_core_forward and device_type == "cuda":
        torch.cuda.synchronize(device)
    context_seconds = (
        time.perf_counter() - context_started if measure_core_forward else 0.0
    )

    maximum_samples = max(score_sample_counts)
    score_chunks = []
    candidate_generation_seconds = 0.0
    for batch_start in range(0, len(candidates), candidate_batch_size):
        batch_stop = min(batch_start + candidate_batch_size, len(candidates))
        if measure_core_forward and device_type == "cuda":
            torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        candidate_items = candidate_metadata(
            metadata,
            candidates[batch_start:batch_stop],
            record["receiver_global"],
        )
        with torch.amp.autocast(device_type):
            source = _cache_branch(
                module,
                candidate_items,
                SOURCE_CONDITIONING_IDS,
                device,
                cond_method,
            )
            dynamic = (
                _cache_branch(
                    module,
                    candidate_items,
                    FA_DYNAMIC_CONDITIONING_IDS,
                    device,
                    cond_method,
                )
                if cond_method == "fa_invariant"
                else None
            )
        repeated_source = _repeat_branch(source, maximum_samples)
        repeated_dynamic = _repeat_branch(dynamic, maximum_samples) if dynamic else None
        seeds = [
            candidate_seed(sample_seed, int(record["index"]), candidate_index, sample_index)
            for candidate_index in range(batch_start, batch_stop)
            for sample_index in range(maximum_samples)
        ]
        if measure_core_forward:
            generated = generate_rir_batch(
                module,
                repeated_source,
                context,
                seeds,
                dynamic_branch=repeated_dynamic,
            )
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            candidate_generation_seconds += time.perf_counter() - generation_started
            generated_features = encode_audio_features(retrieval, generated)
            scores = generated_features @ observation_features.T
        else:
            scores = generate_and_score_batch(
                module,
                retrieval,
                repeated_source,
                context,
                observation_features,
                seeds,
                dynamic_branch=repeated_dynamic,
            )
        score_chunks.append(scores.reshape(batch_stop - batch_start, maximum_samples).cpu())

    similarities = torch.cat(score_chunks, dim=0).float()
    if similarities.shape != (len(candidates), maximum_samples):
        raise RuntimeError("generated score matrix has an unexpected shape")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device_type == "cuda" else 0
    )
    if synchronize_timing and device_type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    result = _build_query_result(
        selected=selected,
        record=record,
        candidates=candidates,
        similarities=similarities,
        run_sha256=run_sha256,
        elapsed_seconds=elapsed_seconds,
        peak_memory_bytes=peak_memory,
        tau=tau,
        random_seed=sample_seed,
        score_sample_counts=score_sample_counts,
    )
    if measure_core_forward:
        result["latency_protocol"] = {
            "name": "fem_core_aligned_kctx8_kgen1",
            "context_count": 8,
            "generated_rirs_per_candidate": maximum_samples,
            "input_loading_included": False,
            "candidate_filtering_included": False,
            "candidate_scoring_included": False,
            "candidate_selection_and_metrics_included": False,
            "result_serialization_included": False,
        }
        result["latency_seconds"] = {
            "context_conditioning": context_seconds,
            "candidate_conditioning_and_generation": candidate_generation_seconds,
            "core_forward_total": context_seconds + candidate_generation_seconds,
        }
    return result, candidates, similarities.numpy()


def run_localization(
    *,
    model_config_path: Path,
    checkpoint_path: Path,
    agree_checkpoint_path: Path,
    context_manifest: dict,
    geometry_audit: dict,
    pilot_manifest: dict,
    dataset_root: Path,
    output_dir: Path,
    device: str,
    cond_method: str,
    candidate_batch_size: int = 64,
    sample_seed: int = 42,
    tau: float = 0.1,
    query_limit: int | None = None,
    score_sample_counts: tuple[int, ...] = SCORE_SAMPLE_COUNTS,
    synchronize_timing: bool = False,
    warmup_query_count: int = 0,
    measure_core_forward: bool = False,
) -> dict:
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    if cond_method not in ("vanilla", "fa_invariant"):
        raise ValueError("unsupported conditioning method")
    score_sample_counts = tuple(int(count) for count in score_sample_counts)
    if (
        not score_sample_counts
        or score_sample_counts != tuple(sorted(set(score_sample_counts)))
        or any(count <= 0 for count in score_sample_counts)
    ):
        raise ValueError("score_sample_counts must be unique, increasing, and positive")
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must be nonnegative")
    if measure_core_forward and score_sample_counts != (1,):
        raise ValueError("core-forward latency requires exactly one RIR per candidate")
    audit_content = {key: value for key, value in geometry_audit.items() if key != "sha256"}
    if geometry_audit.get("sha256") != canonical_sha256(audit_content):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("formal localization requires an available CUDA device")
    joined = resolve_pilot_records(pilot_manifest, context_manifest, geometry_audit)
    if query_limit is not None:
        if query_limit <= 0 or query_limit > len(joined):
            raise ValueError("query_limit is outside the pilot range")
        joined = joined[:query_limit]

    identity = {
        "model_config_sha256": file_sha256(model_config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "agree_checkpoint_sha256": file_sha256(agree_checkpoint_path),
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_indices": [int(item[0]["index"]) for item in joined],
        "conditioning_method": cond_method,
        "frame_average_angles": (
            list(DEFAULT_FRAME_ANGLES) if cond_method == "fa_invariant" else None
        ),
        "n_context": 8,
        "score_sample_counts": list(score_sample_counts),
        "tau": float(tau),
        "sample_seed": int(sample_seed),
        "candidate_batch_size": int(candidate_batch_size),
        "synchronize_timing": bool(synchronize_timing),
        "warmup_query_count": int(warmup_query_count),
        "measure_core_forward": bool(measure_core_forward),
        "sampler": {"type": "rectified_flow_discrete_euler", "steps": 1, "cfg_scale": 1.0},
    }
    run_manifest = initialize_run(output_dir, identity)
    run_sha256 = run_manifest["sha256"]

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    module, _config = load_flac_module(model_config_path, checkpoint_path, device)
    retrieval = load_agree_retrieval(agree_checkpoint_path, device)

    room_bases: dict[str, np.ndarray] = {}
    if warmup_query_count > len(joined):
        raise ValueError("warmup_query_count exceeds the selected query count")
    for selected, record, _geometry in joined[:warmup_query_count]:
        room = selected["room"]
        if room not in room_bases:
            room_bases[room] = reconstruct_room_base_candidates(room, geometry_audit)
        execute_query(
            module=module,
            retrieval=retrieval,
            selected=selected,
            record=record,
            audit=geometry_audit,
            room_base=room_bases[room],
            dataset_root=dataset_root,
            device=device,
            cond_method=cond_method,
            candidate_batch_size=candidate_batch_size,
            sample_seed=sample_seed,
            tau=tau,
            run_sha256=run_sha256,
            score_sample_counts=score_sample_counts,
            synchronize_timing=synchronize_timing,
            measure_core_forward=measure_core_forward,
        )
    if warmup_query_count:
        print(f"completed {warmup_query_count} in-process warm-up queries", flush=True)

    completed = 0
    for ordinal, (selected, record, _geometry) in enumerate(joined, start=1):
        previous = completed_query_result(
            output_dir,
            query_index=int(selected["index"]),
            query_id=selected["query_id"],
            candidate_count=int(selected["candidate_count"]),
            run_sha256=run_sha256,
            score_sample_counts=score_sample_counts,
        )
        if previous is not None:
            completed += 1
            print(f"[{ordinal}/{len(joined)}] resume {selected['query_id']}", flush=True)
            continue
        room = selected["room"]
        if room not in room_bases:
            room_bases[room] = reconstruct_room_base_candidates(room, geometry_audit)
        torch.cuda.reset_peak_memory_stats(device)
        result, candidates, similarities = execute_query(
            module=module,
            retrieval=retrieval,
            selected=selected,
            record=record,
            audit=geometry_audit,
            room_base=room_bases[room],
            dataset_root=dataset_root,
            device=device,
            cond_method=cond_method,
            candidate_batch_size=candidate_batch_size,
            sample_seed=sample_seed,
            tau=tau,
            run_sha256=run_sha256,
            score_sample_counts=score_sample_counts,
            synchronize_timing=synchronize_timing,
            measure_core_forward=measure_core_forward,
        )
        save_query_result(
            output_dir,
            result=result,
            candidates=candidates,
            similarities=similarities,
        )
        completed += 1
        print(
            f"[{ordinal}/{len(joined)}] complete {selected['query_id']} "
            f"candidates={len(candidates)} seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )
    return {
        "run_manifest_sha256": run_sha256,
        "completed_queries": completed,
        "requested_queries": len(joined),
        "output_dir": str(output_dir),
    }
