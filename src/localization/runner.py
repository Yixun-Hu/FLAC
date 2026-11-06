"""Resumable frozen-model localization execution for exp_09."""

from __future__ import annotations

import functools
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
from src.localization.mesh_remap import resolve_mesh_root
from src.localization.pilot import canonical_sha256, resolve_pilot_records
from src.localization.run_provenance import (
    attention_identity,
    no_flash_launch_handoff,
    weights_footprint,
    write_host_record,
)
from src.localization.scoring import (
    deterministic_random_candidate,
    localization_metrics,
    log_mean_exp_scores,
    stable_argmax,
)


RUN_SCHEMA_VERSION = 1
QUERY_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_CONTEXT_COUNTS = (8, 1)
K1_CONTEXT_SELECTION = {
    "model_contexts": "contexts[0] of the frozen context manifest (its exp_09 seed-42 loader order)",
    "candidate_filter_contexts": "all 8 manifest contexts (candidate_filter_context_sources_global)",
}


def context_count_view(record: dict, context_count: int) -> dict:
    """exp_37 K_ctx: 8 = the frozen record itself; 1 = ``contexts[0]`` fed to the model while the candidate
    filter keeps all 8 manifest contexts, so the candidate set and its audit hash do not change."""
    if context_count not in SUPPORTED_CONTEXT_COUNTS:
        raise ValueError(f"unsupported context count {context_count}; supported: {SUPPORTED_CONTEXT_COUNTS}")
    if len(record["contexts"]) != 8 or len(record["context_sources_global"]) != 8:
        raise ValueError("the frozen manifest records must carry 8 contexts")
    if context_count == 8:
        return record
    if "candidate_filter_context_sources_global" in record:
        raise ValueError("the record already carries a candidate-filter override")
    view = dict(record)
    view["contexts"] = list(record["contexts"][:context_count])
    view["context_sources_global"] = [list(p) for p in record["context_sources_global"][:context_count]]
    view["candidate_filter_context_sources_global"] = [list(p) for p in record["context_sources_global"]]
    return view


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


def _cache_branch(
    module, metadata, ids, device: str, cond_method: str, *, frame_angles=DEFAULT_FRAME_ANGLES
):
    if cond_method == "fa_invariant":
        return cache_invariant_conditioning_branch(
            module.diffusion.conditioner,
            metadata,
            ids,
            device,
            frame_angles,
        )
    return cache_conditioning_branch(
        module.diffusion.conditioner,
        metadata,
        ids,
        device,
    )


FRAME_ANGLE_SETS = ((0.0,), (0.0, 180.0), tuple(DEFAULT_FRAME_ANGLES))


def validate_frame_angles(cond_method: str, frame_angles) -> None:
    """exp_37 B3: an orbit applies to fa_invariant only and starts at the 0-degree identity."""
    angles = tuple(float(angle) for angle in frame_angles)
    if cond_method != "fa_invariant" and angles != tuple(DEFAULT_FRAME_ANGLES):
        raise ValueError("frame-average angles apply to the fa_invariant conditioning only")
    if not angles or angles[0] != 0.0:
        raise ValueError("frame-average angles must start with the 0-degree identity")


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
    context_count: int = 8,
    frame_angles: tuple = DEFAULT_FRAME_ANGLES,
    candidate_subset=None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run and score all candidates for one frozen target query.

    exp_37 plumbing (registered in test_exp37_headline_timer_guard.py): ``frame_angles`` (fa_invariant orbit, default
    the bundle's C4) and ``candidate_subset`` (a callable returning FULL 0.5 m indices; the seeds keep them).
    """

    validate_frame_angles(cond_method, frame_angles)
    device_type = torch.device(device).type
    if synchronize_timing and device_type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    observed, metadata = load_frozen_query(record, dataset_root)
    candidates = filter_frozen_query_candidates(record, audit, room_base)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("pilot and reconstructed candidate counts differ")
    if len(record["contexts"]) != context_count:
        raise RuntimeError(f"formal pilot requires N_ctx={context_count}")
    candidate_indices = range(len(candidates))
    if candidate_subset is not None:
        full_count, chosen = len(candidates), np.asarray(candidate_subset(candidates))
        if (chosen.ndim != 1 or not len(chosen) or not np.issubdtype(chosen.dtype, np.integer)
                or np.any(np.diff(chosen) <= 0) or chosen[0] < 0 or chosen[-1] >= full_count):
            raise ValueError("candidate_subset must return sorted, unique, in-range full 0.5 m indices")
        candidates, candidate_indices = candidates[chosen], [int(index) for index in chosen]

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
            frame_angles=frame_angles,
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
                frame_angles=frame_angles,
            )
            dynamic = (
                _cache_branch(
                    module,
                    candidate_items,
                    FA_DYNAMIC_CONDITIONING_IDS,
                    device,
                    cond_method,
                    frame_angles=frame_angles,
                )
                if cond_method == "fa_invariant"
                else None
            )
        repeated_source = _repeat_branch(source, maximum_samples)
        repeated_dynamic = _repeat_branch(dynamic, maximum_samples) if dynamic else None
        seeds = [
            candidate_seed(sample_seed, int(record["index"]), candidate_index, sample_index)
            for candidate_index in candidate_indices[batch_start:batch_stop]
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
    if candidate_subset is not None:
        result["candidate_subset"] = {
            "name": candidate_subset.__name__,
            "full_candidate_count": full_count,
            "indices_sha256": hashlib.sha256(chosen.astype("<u4").tobytes()).hexdigest(),
        }
    if measure_core_forward:
        result["latency_protocol"] = {
            "name": f"fem_core_aligned_kctx{context_count}_kgen1",
            "context_count": context_count,
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


def select_queries(
    joined: list, *, query_limit: int | None = None, query_index: int | None = None
) -> list:
    """The pilot prefix (``query_limit``), exactly one pilot query (``query_index``, exp_37 G0), or all."""
    if query_limit is not None and query_index is not None:
        raise ValueError("query_limit and query_index are mutually exclusive")
    if query_index is not None:
        chosen = [item for item in joined if int(item[0]["index"]) == int(query_index)]
        if len(chosen) != 1:
            raise ValueError(f"query index {query_index} is not in the pilot manifest exactly once")
        return chosen
    if query_limit is not None:
        if query_limit <= 0 or query_limit > len(joined):
            raise ValueError("query_limit is outside the pilot range")
        return joined[:query_limit]
    return joined


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
    query_index: int | None = None,
    mesh_root: Path | None = None,
    require_attention_backend: str | None = None,
    context_count: int = 8,
    stage_split: bool = False,
    frame_angles: tuple | None = None,
    candidate_subset: str | None = None,
    c2f_top_k: int = 5,
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
    if context_count not in SUPPORTED_CONTEXT_COUNTS:
        raise ValueError(f"unsupported context count {context_count}; supported: {SUPPORTED_CONTEXT_COUNTS}")
    if measure_core_forward and score_sample_counts != (1,):
        raise ValueError("core-forward latency requires exactly one RIR per candidate")
    if stage_split and not (measure_core_forward and synchronize_timing):
        raise ValueError("the stage split pass needs --measure-core-forward and --synchronize-latency")
    angles = DEFAULT_FRAME_ANGLES if frame_angles is None else tuple(float(a) for a in frame_angles)
    if frame_angles is not None and (cond_method != "fa_invariant" or angles not in FRAME_ANGLE_SETS):
        raise ValueError("--frame-avg-angles must be a registered fa_invariant orbit: {0}, {0, 180} or C4")
    if candidate_subset not in (None, "nested_1m", "c2f"):
        raise ValueError(f"unknown candidate subset {candidate_subset!r}")
    if candidate_subset and not measure_core_forward:
        raise ValueError("the nested / c2f passes are timing passes: --measure-core-forward is required")
    if candidate_subset and stage_split:
        raise ValueError("a candidate subset pass is never a stage split pass")
    if candidate_subset and (Path(output_dir) / "run_manifest.json").exists():
        raise RuntimeError("a nested / c2f pass is never resumed: use a fresh output directory")
    audit_content = {key: value for key, value in geometry_audit.items() if key != "sha256"}
    if geometry_audit.get("sha256") != canonical_sha256(audit_content):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    joined = resolve_pilot_records(pilot_manifest, context_manifest, geometry_audit)
    joined = select_queries(joined, query_limit=query_limit, query_index=query_index)
    # exp_37 provenance, verified before anything is written: the attention backend actually selected (and,
    # when SDPA is required, the no-flash launcher's in-process hand-off), the --mesh-root map (audit untouched).
    attention = attention_identity(REPO_ROOT, require_attention_backend)
    handoff = no_flash_launch_handoff(REPO_ROOT) if require_attention_backend else None
    mesh_remap = resolve_mesh_root(geometry_audit, mesh_root) if mesh_root is not None else None
    mesh_paths = {room: row["mesh_path"] for room, row in (mesh_remap or {"rooms": {}})["rooms"].items()}
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("formal localization requires an available CUDA device")

    identity = {
        "model_config_sha256": file_sha256(model_config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "agree_checkpoint_sha256": file_sha256(agree_checkpoint_path),
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_indices": [int(item[0]["index"]) for item in joined],
        "conditioning_method": cond_method,
        "frame_average_angles": list(angles) if cond_method == "fa_invariant" else None,
        "n_context": int(context_count),
        "score_sample_counts": list(score_sample_counts),
        "tau": float(tau),
        "sample_seed": int(sample_seed),
        "candidate_batch_size": int(candidate_batch_size),
        "synchronize_timing": bool(synchronize_timing),
        "warmup_query_count": int(warmup_query_count),
        "measure_core_forward": bool(measure_core_forward),
        "sampler": {"type": "rectified_flow_discrete_euler", "steps": 1, "cfg_scale": 1.0},
        **attention,
    }
    if mesh_remap is not None:
        identity["mesh_root_override"] = mesh_remap
    if context_count != 8:
        identity["context_selection"] = K1_CONTEXT_SELECTION
    if stage_split:
        identity["stage_split"] = True
    if candidate_subset:
        identity["candidate_subset"] = candidate_subset
    if candidate_subset == "c2f":
        identity.update(c2f_top_k=int(c2f_top_k), c2f_chebyshev_radius_m=0.5)
    run_manifest = initialize_run(output_dir, identity)
    run_sha256 = run_manifest["sha256"]
    write_host_record(
        output_dir, "start", device=device, handoff=handoff, run_manifest_sha256=run_sha256
    )

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    module, _config = load_flac_module(model_config_path, checkpoint_path, device)
    flac_footprint = weights_footprint(device)
    retrieval = load_agree_retrieval(agree_checkpoint_path, device)
    write_host_record(
        output_dir, "loaded", device=device, handoff=handoff, run_manifest_sha256=run_sha256,
        extra={"memory_after_flac_load": flac_footprint, "memory_after_agree_load": weights_footprint(device)},
    )

    subset_fn = None
    if stage_split:
        from src.localization import stage_split as split_module

        executor = split_module.execute_query_stage_split
    elif candidate_subset == "c2f":
        from src.localization import exp37_c2f

        executor = functools.partial(exp37_c2f.execute_query_c2f, top_k_count=int(c2f_top_k))
    else:
        executor = execute_query
        if candidate_subset == "nested_1m":
            from src.localization.exp37_grid import nested_subset as subset_fn
    room_bases: dict[str, np.ndarray] = {}
    if warmup_query_count > len(joined):
        raise ValueError("warmup_query_count exceeds the selected query count")
    for selected, record, _geometry in joined[:warmup_query_count]:
        record = context_count_view(record, context_count)
        room = selected["room"]
        if room not in room_bases:
            room_bases[room] = reconstruct_room_base_candidates(
                room, geometry_audit, mesh_path=mesh_paths.get(room)
            )
        executor(
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
            context_count=context_count,
            frame_angles=angles,
            candidate_subset=subset_fn,
        )
    if warmup_query_count:
        print(f"completed {warmup_query_count} in-process warm-up queries", flush=True)

    completed = 0
    for ordinal, (selected, record, _geometry) in enumerate(joined, start=1):
        record = context_count_view(record, context_count)
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
            room_bases[room] = reconstruct_room_base_candidates(
                room, geometry_audit, mesh_path=mesh_paths.get(room)
            )
        torch.cuda.reset_peak_memory_stats(device)
        result, candidates, similarities = executor(
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
            context_count=context_count,
            frame_angles=angles,
            candidate_subset=subset_fn,
        )
        if context_count != 8:
            result["context_selection"] = {"indices": list(range(context_count)), "paths": list(record["contexts"])}
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
    write_host_record(
        output_dir, "end", device=device, handoff=handoff, run_manifest_sha256=run_sha256
    )
    return {
        "run_manifest_sha256": run_sha256,
        "completed_queries": completed,
        "requested_queries": len(joined),
        "output_dir": str(output_dir),
    }
