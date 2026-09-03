"""Resumable localization execution for material-blind Room Helps baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.fem_pipeline import (
    load_tetrahedral_mesh_npz,
    prepare_fem_query_interpolation,
    prepare_fem_room_operators,
    run_fem_sabine_forward,
)
from src.baselines.fem_solver import (
    SparseDirectSolveSession,
    SparseDirectSolverOptions,
    resolve_sparse_direct_backend,
)
from src.localization.baseline_runner import (
    score_fem_room_helps_candidates,
    score_few_shot_candidates,
)
from src.localization.engine import (
    encode_audio_features,
    filter_frozen_query_candidates,
    load_agree_retrieval,
    load_frozen_query,
    reconstruct_room_base_candidates,
)
from src.localization.fewshot_rir import (
    NEAR_CONTEXT_PROTOCOL,
    load_fewshot_rir_query,
    score_fewshot_rir_candidates,
)
from src.localization.pilot import canonical_sha256, resolve_pilot_records
from src.localization.runner import (
    _atomic_json,
    _atomic_npz,
    _hashed_payload,
    _query_paths,
    file_sha256,
    initialize_run,
    verify_hashed_payload,
)
from src.localization.scoring import (
    deterministic_random_candidate,
    localization_metrics,
    stable_argmax,
)
from src.models import create_model_from_config


BASELINE_CONTEXT_COUNTS = (1, 8)
BASELINE_QUERY_SCHEMA_VERSION = 2
TETRA_MESH_MANIFEST_SCHEMA_VERSION = 1


def filter_execution_rooms(
    joined: list[tuple[dict, dict, dict]],
    skip_rooms: tuple[str, ...] = (),
) -> list[tuple[dict, dict, dict]]:
    """Apply a resume-safe execution-only room filter.

    The full pilot remains in the run identity.  This filter only controls which
    query artifacts are produced by the current invocation, so a later launch
    can resume the same output directory with a different execution subset.
    """

    skipped = {str(room) for room in skip_rooms}
    if not skipped:
        return list(joined)
    available = {str(selected["room"]) for selected, _record, _geometry in joined}
    unknown = skipped - available
    if unknown:
        raise ValueError(f"skip_rooms contains unknown pilot rooms: {sorted(unknown)}")
    scheduled = [
        item for item in joined if str(item[0]["room"]) not in skipped
    ]
    if not scheduled:
        raise ValueError("skip_rooms excludes every pilot query")
    return scheduled


def load_few_shot_waveform_checkpoint(
    model_config_path: Path | str,
    checkpoint_path: Path | str,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Strictly load either a Lightning-wrapped or direct model state dictionary."""

    config = json.loads(Path(model_config_path).read_text())
    if config.get("model_type") != "few_shot_rir_waveform":
        raise ValueError("checkpoint config is not Few-ShotRIR-Waveform")
    model = create_model_from_config(config)
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(bundle, dict) and bundle.get("model_config") not in (None, config):
        raise ValueError("checkpoint embedded model config differs from the requested config")
    state = bundle.get("state_dict", bundle) if isinstance(bundle, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty state dictionary")
    keys = tuple(state)
    if all(key.startswith("model.") for key in keys):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, config


def load_fewshot_rir_checkpoint(
    model_config_path: Path | str,
    checkpoint_path: Path | str,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Strictly load a FewshotRiR Lightning or direct state dictionary."""

    config = json.loads(Path(model_config_path).read_text())
    if config.get("model_type") != "FewshotRiR":
        raise ValueError("checkpoint config is not FewshotRiR")
    model = create_model_from_config(config)
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(bundle, dict) and bundle.get("model_config") not in (None, config):
        raise ValueError("checkpoint embedded model config differs from the requested config")
    state = bundle.get("state_dict", bundle) if isinstance(bundle, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty state dictionary")
    keys = tuple(state)
    if all(key.startswith("model.") for key in keys):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, config


def load_tetrahedral_mesh_manifest(path: Path | str) -> dict:
    """Load a room-to-NPZ map and resolve relative entries against the manifest."""

    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != TETRA_MESH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported tetrahedral mesh manifest schema")
    rooms = manifest.get("rooms")
    if not isinstance(rooms, dict) or not rooms:
        raise ValueError("tetrahedral mesh manifest must contain rooms")
    resolved_rooms = {}
    for room, entry in rooms.items():
        if not isinstance(entry, dict) or set(entry) != {"path", "npz_sha256"}:
            raise ValueError("each tetrahedral room entry requires path and npz_sha256")
        mesh_path = Path(entry["path"])
        mesh_hash = entry["npz_sha256"]
        if not isinstance(mesh_hash, str) or len(mesh_hash) != 64 or any(
            character not in "0123456789abcdef" for character in mesh_hash
        ):
            raise ValueError("tetrahedral NPZ SHA-256 must be lowercase hexadecimal")
        resolved_rooms[str(room)] = {
            "path": str(
                (mesh_path if mesh_path.is_absolute() else path.parent / mesh_path).resolve()
            ),
            "npz_sha256": mesh_hash,
        }
    resolved = dict(manifest)
    resolved["rooms"] = resolved_rooms
    return resolved


def load_room_tetrahedral_mesh(
    room: str,
    tetra_manifest: dict,
    geometry_audit: dict,
):
    """Load one air mesh only when its provenance matches the official OBJ hash."""

    if tetra_manifest.get("schema_version") != TETRA_MESH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported tetrahedral mesh manifest schema")
    try:
        entry = tetra_manifest["rooms"][room]
        expected_surface_hash = geometry_audit["rooms"][room]["mesh_sha256"]
    except KeyError as error:
        raise KeyError(f"room {room!r} lacks an audited tetrahedral mesh") from error
    if not isinstance(entry, dict) or set(entry) != {"path", "npz_sha256"}:
        raise ValueError("invalid tetrahedral room manifest entry")
    path = entry["path"]
    if file_sha256(path) != entry["npz_sha256"]:
        raise RuntimeError(f"room {room!r} tetrahedral NPZ hash mismatch")
    mesh, metadata = load_tetrahedral_mesh_npz(path)
    if metadata["source_mesh_sha256"] != expected_surface_hash:
        raise RuntimeError(f"room {room!r} tetrahedral/source surface mesh hash mismatch")
    return mesh, metadata


def build_baseline_query_result(
    *,
    query_index: int,
    query_id: str,
    scene: str,
    room: str,
    receiver_id: str | int,
    candidates: np.ndarray,
    source_global: np.ndarray,
    receiver_global: np.ndarray,
    candidate_scores: torch.Tensor,
    context_counts: tuple[int, ...] = BASELINE_CONTEXT_COUNTS,
    random_seed: int,
    elapsed_seconds: float,
    diagnostics: dict,
) -> dict:
    """Build deterministic K-context metrics with the common stable argmax rule."""

    candidates = np.asarray(candidates, dtype=np.float64)
    values = torch.as_tensor(candidate_scores, dtype=torch.float32)
    counts = tuple(int(count) for count in context_counts)
    if values.shape != (len(candidates), len(counts)):
        raise ValueError("candidate_scores must have one column per context count")
    if counts != tuple(sorted(set(counts))) or any(count <= 0 for count in counts):
        raise ValueError("context counts must be unique, increasing, and positive")
    if not torch.isfinite(values).all():
        raise ValueError("baseline candidate scores must be finite")
    truth = np.asarray(source_global, dtype=np.float64)
    metrics = {}
    for column, count in enumerate(counts):
        scores = values[:, column]
        winner = stable_argmax(scores)
        result = localization_metrics(candidates, truth, winner)
        result["prediction_global"] = candidates[winner].astype(float).tolist()
        result["winning_score"] = float(scores[winner])
        result["mean_candidate_score"] = float(scores.mean())
        metrics[str(count)] = result
    random_index = deterministic_random_candidate(query_index, len(candidates), seed=random_seed)
    random_metrics = localization_metrics(candidates, truth, random_index)
    random_metrics["prediction_global"] = candidates[random_index].astype(float).tolist()
    return {
        "schema_version": BASELINE_QUERY_SCHEMA_VERSION,
        "query_index": int(query_index),
        "query_id": str(query_id),
        "scene": str(scene),
        "room": str(room),
        "receiver_id": receiver_id,
        "candidate_count": len(candidates),
        "source_global": truth.astype(float).tolist(),
        "receiver_global": np.asarray(receiver_global, dtype=float).tolist(),
        "context_counts": list(counts),
        "metrics": metrics,
        "random_candidate_metrics": random_metrics,
        "elapsed_seconds": float(elapsed_seconds),
        "diagnostics": diagnostics,
    }


def _completed_baseline_query(
    output_dir: Path,
    *,
    query_index: int,
    query_id: str,
    candidate_count: int,
    run_sha256: str,
    context_counts: tuple[int, ...] = BASELINE_CONTEXT_COUNTS,
) -> dict | None:
    json_path, npz_path = _query_paths(output_dir, query_index)
    if not json_path.exists():
        return None
    result = json.loads(json_path.read_text())
    verify_hashed_payload(result, f"baseline query {query_index} result")
    if (
        result.get("query_id") != query_id
        or result.get("candidate_count") != candidate_count
        or result.get("run_manifest_sha256") != run_sha256
    ):
        raise RuntimeError(f"baseline query {query_index} resume identity mismatch")
    if not npz_path.is_file() or file_sha256(npz_path) != result.get("arrays_sha256"):
        raise RuntimeError(f"baseline query {query_index} array artifact is missing or corrupt")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if arrays["candidates"].shape != (candidate_count, 3) or arrays[
            "candidate_scores"
        ].shape != (candidate_count, len(context_counts)):
            raise RuntimeError(f"baseline query {query_index} array shape mismatch")
        if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
            raise RuntimeError(f"baseline query {query_index} artifact contains non-finite values")
    return result


def _save_baseline_query(
    output_dir: Path,
    *,
    result: dict,
    candidates: np.ndarray,
    candidate_scores: np.ndarray,
) -> dict:
    json_path, npz_path = _query_paths(output_dir, int(result["query_index"]))
    _atomic_npz(
        npz_path,
        candidates=np.asarray(candidates, dtype=np.float32),
        candidate_scores=np.asarray(candidate_scores, dtype=np.float32),
    )
    payload = dict(result)
    payload["arrays_file"] = str(npz_path.relative_to(output_dir))
    payload["arrays_sha256"] = file_sha256(npz_path)
    payload = _hashed_payload(payload)
    _atomic_json(json_path, payload)
    return payload


@torch.inference_mode()
def execute_baseline_query(
    *,
    method: str,
    predictor,
    retrieval,
    selected: dict,
    record: dict,
    geometry_audit: dict,
    room_base: np.ndarray,
    tetrahedral_mesh,
    fem_room_operators=None,
    fem_solver_options: SparseDirectSolverOptions | None = None,
    fem_solver_session: SparseDirectSolveSession | None = None,
    dataset_root: Path,
    device: str | torch.device,
    candidate_batch_size: int,
    random_seed: int,
    run_sha256: str,
    context_counts: tuple[int, ...] = BASELINE_CONTEXT_COUNTS,
    synchronize_timing: bool = False,
    measure_core_forward: bool = False,
    fewshot_rir_options: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Execute one query for FewshotRiR, legacy waveform, or FEM-Sabine synthesis."""

    device_type = torch.device(device).type
    if synchronize_timing and device_type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if method == "FewshotRiR":
        fewshot_rir_options = dict(fewshot_rir_options or {})
        observed, metadata = load_fewshot_rir_query(
            record,
            dataset_root,
            sample_rate=int(fewshot_rir_options.get("sample_rate", 22050)),
            sample_size=int(fewshot_rir_options.get("sample_size", 10240)),
            n_fft=int(fewshot_rir_options.get("n_fft", 511)),
            hop_length=int(fewshot_rir_options.get("hop_length", 40)),
            win_length=int(fewshot_rir_options.get("win_length", 248)),
            depth_size=tuple(fewshot_rir_options.get("depth_size", (128, 256))),
            depth_max_m=float(fewshot_rir_options.get("depth_max_m", 67.16327)),
        )
    else:
        observed, metadata = load_frozen_query(record, dataset_root)
    candidates = filter_frozen_query_candidates(record, geometry_audit, room_base)
    if len(candidates) != int(selected["candidate_count"]):
        raise RuntimeError("baseline candidate count differs from the frozen pilot")
    observation_features = None
    if method in ("FewshotRiR", "few_shot_rir_waveform"):
        if retrieval is None:
            raise ValueError("Few-ShotRIR localization requires the frozen AGREE scorer")
        observation_features = encode_audio_features(
            retrieval, observed.unsqueeze(0).to(device=device, dtype=torch.float32)
        )
    score_columns = []
    core_timing: dict[str, float] | None = {} if measure_core_forward else None
    diagnostics: dict[str, dict] = {}
    fem_receiver_load = None
    fem_candidate_interpolation = None
    if method == "fem_sabine":
        if tetrahedral_mesh is None:
            raise ValueError("FEM-Sabine requires a tetrahedral mesh")
        fem_room_operators = fem_room_operators or prepare_fem_room_operators(
            tetrahedral_mesh
        )
        fem_receiver_load, fem_candidate_interpolation = prepare_fem_query_interpolation(
            fem_room_operators,
            np.asarray(record["receiver_global"]),
            candidates,
        )
    for context_count in context_counts:
        if len(record["contexts"]) < context_count:
            raise RuntimeError("frozen query contains too few ordered contexts")
        if method == "FewshotRiR":
            scores = score_fewshot_rir_candidates(
                predictor,
                retrieval,
                metadata,
                candidates,
                receiver_global=np.asarray(record["receiver_global"]),
                observation_features=observation_features,
                context_count=context_count,
                candidate_batch_size=candidate_batch_size,
                device=device,
                griffin_lim_iterations=int(
                    fewshot_rir_options.get("griffin_lim_iterations", 32)
                ),
                griffin_lim_momentum=float(
                    fewshot_rir_options.get("griffin_lim_momentum", 0.99)
                ),
                core_timing=core_timing,
                synchronize_timing=synchronize_timing,
            )
            diagnostics[str(context_count)] = {
                "prediction": "monaural_magnitude_spectrogram",
                "waveform_reconstruction": "deterministic_griffin_lim",
                "griffin_lim_rand_init": False,
                "griffin_lim_iterations": int(
                    fewshot_rir_options.get("griffin_lim_iterations", 32)
                ),
            }
        elif method == "few_shot_rir_waveform":
            scores = score_few_shot_candidates(
                predictor,
                retrieval,
                metadata,
                candidates,
                receiver_global=np.asarray(record["receiver_global"]),
                observation_features=observation_features,
                context_count=context_count,
                candidate_batch_size=candidate_batch_size,
                device=device,
                core_timing=core_timing,
                synchronize_timing=synchronize_timing,
            )
            diagnostics[str(context_count)] = {"deterministic_waveform_generation": True}
        elif method == "fem_sabine":
            forward = run_fem_sabine_forward(
                tetrahedral_mesh,
                receiver_point=np.asarray(record["receiver_global"]),
                candidate_points=candidates,
                context_waveforms=metadata["context_audio"][:context_count],
                construct_waveforms=False,
                room_operators=fem_room_operators,
                receiver_load=fem_receiver_load,
                candidate_interpolation=fem_candidate_interpolation,
                solver_options=fem_solver_options,
                solver_session=fem_solver_session,
            )
            scores, recovery = score_fem_room_helps_candidates(
                forward.response,
                forward.bin_indices,
                observed,
                sample_count=observed.shape[-1],
            )
            coefficient = recovery.coefficients[0]
            diagnostics[str(context_count)] = {
                "selection_rule": "room_helps_pulse_stacked_omp",
                "source_count": 1,
                "fem": forward.audit,
                "sparse_recovery": {
                    "support": list(recovery.support),
                    "coefficient_real": float(coefficient.real),
                    "coefficient_imag": float(coefficient.imag),
                    "relative_residual_norm": recovery.relative_residual_norm,
                },
            }
        else:
            raise ValueError("unsupported baseline method")
        score_columns.append(scores.float().cpu())
    candidate_scores = torch.stack(score_columns, dim=1)
    if synchronize_timing and device_type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    result = build_baseline_query_result(
        query_index=int(selected["index"]),
        query_id=selected["query_id"],
        scene=selected["scene"],
        room=selected["room"],
        receiver_id=selected["receiver_id"],
        candidates=candidates,
        source_global=np.asarray(record["source_global"]),
        receiver_global=np.asarray(record["receiver_global"]),
        candidate_scores=candidate_scores,
        random_seed=random_seed,
        elapsed_seconds=elapsed_seconds,
        diagnostics=diagnostics,
        context_counts=context_counts,
    )
    result["run_manifest_sha256"] = run_sha256
    result["method"] = method
    result["candidate_score_name"] = (
        "agree_cosine"
        if method in ("FewshotRiR", "few_shot_rir_waveform")
        else "room_helps_projection_fraction"
    )
    if measure_core_forward:
        timing_key = (
            "candidate_conditioning_generation_and_griffin_lim"
            if method == "FewshotRiR"
            else "candidate_conditioning_and_generation"
        )
        core_total = float(core_timing[timing_key])
        result["latency_protocol"] = {
            "name": (
                "fewshot_rir_kctx8_generation_plus_deterministic_griffin_lim"
                if method == "FewshotRiR"
                else "fem_core_aligned_kctx8_kgen1"
            ),
            "context_count": int(context_counts[0]),
            "generated_rirs_per_candidate": 1,
            "input_loading_included": False,
            "candidate_filtering_included": False,
            "candidate_scoring_included": False,
            "candidate_selection_and_metrics_included": False,
            "result_serialization_included": False,
        }
        result["latency_seconds"] = {
            timing_key: core_total,
            "core_forward_total": core_total,
        }
    result["candidate_indices_sha256"] = selected["candidate_indices_sha256"]
    return result, candidates, candidate_scores.numpy()


def run_baseline_localization(
    *,
    method: str,
    agree_checkpoint_path: Path | None,
    context_manifest: dict,
    geometry_audit: dict,
    pilot_manifest: dict,
    dataset_root: Path,
    output_dir: Path,
    device: str,
    candidate_batch_size: int = 64,
    random_seed: int = 42,
    query_limit: int | None = None,
    model_config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    tetra_manifest_path: Path | None = None,
    fem_solver_backend: str = "auto",
    fem_superlu_ordering: str = "MMD_AT_PLUS_A",
    fem_solver_threads: int = 1,
    skip_rooms: tuple[str, ...] = (),
    context_counts: tuple[int, ...] = BASELINE_CONTEXT_COUNTS,
    synchronize_timing: bool = False,
    warmup_query_count: int = 0,
    measure_core_forward: bool = False,
) -> dict:
    """Run a resume-safe paired K=1/8 baseline on frozen exp_09 identities."""

    if method not in ("FewshotRiR", "few_shot_rir_waveform", "fem_sabine"):
        raise ValueError("unsupported baseline method")
    if method == "FewshotRiR":
        if context_manifest.get("protocol", {}).get("name") != NEAR_CONTEXT_PROTOCOL:
            raise ValueError(
                "FewshotRiR requires a frozen near-coincident context manifest and rebuilt geometry/pilot artifacts"
            )
        if any(
            record.get("context_protocol") != NEAR_CONTEXT_PROTOCOL
            for record in context_manifest.get("records", ())
        ):
            raise ValueError("FewshotRiR context manifest contains a non-near-context record")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    context_counts = tuple(int(count) for count in context_counts)
    if (
        not context_counts
        or context_counts != tuple(sorted(set(context_counts)))
        or any(count <= 0 for count in context_counts)
    ):
        raise ValueError("context_counts must be unique, increasing, and positive")
    if any(count > 8 for count in context_counts):
        raise ValueError("formal localization provides at most eight contexts")
    if warmup_query_count < 0:
        raise ValueError("warmup_query_count must be nonnegative")
    if method == "fem_sabine" and warmup_query_count:
        raise ValueError("in-process warm-up is only supported for Few-ShotRIR")
    if measure_core_forward and (
        method not in ("FewshotRiR", "few_shot_rir_waveform") or context_counts != (8,)
    ):
        raise ValueError("core-forward latency requires Few-ShotRIR at K_ctx=8")
    audit_content = {key: value for key, value in geometry_audit.items() if key != "sha256"}
    if geometry_audit.get("sha256") != canonical_sha256(audit_content):
        raise RuntimeError("geometry audit SHA-256 mismatch")
    joined = resolve_pilot_records(pilot_manifest, context_manifest, geometry_audit)
    if query_limit is not None:
        if query_limit <= 0 or query_limit > len(joined):
            raise ValueError("query_limit is outside the pilot range")
        joined = joined[:query_limit]
    scheduled = filter_execution_rooms(joined, skip_rooms)

    predictor = None
    tetra_manifest = None
    method_identity = {}
    if method in ("FewshotRiR", "few_shot_rir_waveform"):
        if (
            model_config_path is None
            or checkpoint_path is None
            or agree_checkpoint_path is None
        ):
            raise ValueError(
                f"{method} requires config, checkpoint, and AGREE paths"
            )
        requested_config = json.loads(Path(model_config_path).read_text())
        if requested_config.get("model_type") != method:
            raise ValueError(f"model config type must match localization method {method}")
        adaptation = dict(requested_config.get("adaptation", {}))
        method_identity = {
            "model_config_sha256": file_sha256(model_config_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "agree_checkpoint_sha256": file_sha256(agree_checkpoint_path),
            "selection_rule": "frozen_agree_cosine",
        }
        if method == "FewshotRiR":
            method_identity.update(
                {
                    "waveform_reconstruction": "deterministic_griffin_lim",
                    "griffin_lim_iterations": int(
                        adaptation.get("griffin_lim_iterations", 32)
                    ),
                    "griffin_lim_momentum": float(
                        adaptation.get("griffin_lim_momentum", 0.99)
                    ),
                }
            )
    else:
        if tetra_manifest_path is None:
            raise ValueError("FEM-Sabine requires a tetra-mesh manifest")
        resolved_backend = resolve_sparse_direct_backend(fem_solver_backend)
        method_identity = {
            "tetra_manifest_sha256": file_sha256(tetra_manifest_path),
            "frequency_band_hz": [80.0, 300.0],
            "selection_rule": "room_helps_pulse_stacked_omp",
            "source_count": 1,
            "solver_backend": resolved_backend,
            "superlu_ordering": fem_superlu_ordering,
            "solver_threads": int(fem_solver_threads),
        }

    identity = {
        "method": method,
        "context_manifest_sha256": context_manifest["sha256"],
        "geometry_audit_sha256": geometry_audit["sha256"],
        "pilot_manifest_sha256": pilot_manifest["sha256"],
        "query_indices": [int(item[0]["index"]) for item in joined],
        "context_counts": list(context_counts),
        "candidate_batch_size": int(candidate_batch_size),
        "random_seed": int(random_seed),
        "synchronize_timing": bool(synchronize_timing),
        "warmup_query_count": int(warmup_query_count),
        "measure_core_forward": bool(measure_core_forward),
        **method_identity,
    }
    run_manifest = initialize_run(output_dir, identity)
    run_sha256 = run_manifest["sha256"]
    fewshot_rir_options = None
    if method in ("FewshotRiR", "few_shot_rir_waveform"):
        if method == "FewshotRiR":
            predictor, config = load_fewshot_rir_checkpoint(
                model_config_path, checkpoint_path, device
            )
            model_options = config.get("model", {})
            adaptation = config.get("adaptation", {})
            fewshot_rir_options = {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"],
                "n_fft": model_options.get("n_fft", 511),
                "hop_length": model_options.get("hop_length", 40),
                "win_length": model_options.get("win_length", 248),
                "depth_size": adaptation.get("depth_size", (128, 256)),
                "depth_max_m": adaptation.get("depth_max_m", 67.16327),
                "griffin_lim_iterations": adaptation.get("griffin_lim_iterations", 32),
                "griffin_lim_momentum": adaptation.get("griffin_lim_momentum", 0.99),
            }
        else:
            predictor, config = load_few_shot_waveform_checkpoint(
                model_config_path, checkpoint_path, device
            )
        if config.get("sample_size") != 10240 or config.get("sample_rate") != 22050:
            raise ValueError("formal Few-ShotRIR baseline requires 10240 samples at 22050 Hz")
    else:
        tetra_manifest = load_tetrahedral_mesh_manifest(tetra_manifest_path)
    retrieval = (
        load_agree_retrieval(agree_checkpoint_path, device)
        if method in ("FewshotRiR", "few_shot_rir_waveform")
        else None
    )

    room_bases: dict[str, np.ndarray] = {}
    active_fem_room = None
    active_fem_mesh = None
    active_fem_operators = None
    active_fem_session = None
    fem_solver_options = (
        SparseDirectSolverOptions(
            backend=method_identity["solver_backend"],
            superlu_ordering=fem_superlu_ordering,
            threads=fem_solver_threads,
        )
        if method == "fem_sabine"
        else None
    )
    if warmup_query_count > len(scheduled):
        raise ValueError("warmup_query_count exceeds the scheduled query count")
    for selected, record, _geometry in scheduled[:warmup_query_count]:
        room = selected["room"]
        if room not in room_bases:
            room_bases[room] = reconstruct_room_base_candidates(room, geometry_audit)
        execute_baseline_query(
            method=method,
            predictor=predictor,
            retrieval=retrieval,
            selected=selected,
            record=record,
            geometry_audit=geometry_audit,
            room_base=room_bases[room],
            tetrahedral_mesh=None,
            dataset_root=dataset_root,
            device=device,
            candidate_batch_size=candidate_batch_size,
            random_seed=random_seed,
            run_sha256=run_sha256,
            context_counts=context_counts,
            synchronize_timing=synchronize_timing,
            measure_core_forward=measure_core_forward,
            fewshot_rir_options=fewshot_rir_options,
        )
    if warmup_query_count:
        print(f"completed {warmup_query_count} in-process warm-up queries", flush=True)

    completed = 0
    for ordinal, (selected, record, _geometry) in enumerate(scheduled, start=1):
        previous = _completed_baseline_query(
            output_dir,
            query_index=int(selected["index"]),
            query_id=selected["query_id"],
            candidate_count=int(selected["candidate_count"]),
            run_sha256=run_sha256,
            context_counts=context_counts,
        )
        if previous is not None:
            completed += 1
            print(
                f"[{ordinal}/{len(scheduled)}] resume {selected['query_id']}",
                flush=True,
            )
            continue
        room = selected["room"]
        if room not in room_bases:
            room_bases[room] = reconstruct_room_base_candidates(room, geometry_audit)
        if method == "fem_sabine" and room != active_fem_room:
            if active_fem_session is not None:
                active_fem_session.close()
            active_fem_mesh, _metadata = load_room_tetrahedral_mesh(
                room, tetra_manifest, geometry_audit
            )
            active_fem_operators = prepare_fem_room_operators(active_fem_mesh)
            active_fem_session = SparseDirectSolveSession(fem_solver_options)
            active_fem_room = room
        result, candidates, candidate_scores = execute_baseline_query(
            method=method,
            predictor=predictor,
            retrieval=retrieval,
            selected=selected,
            record=record,
            geometry_audit=geometry_audit,
            room_base=room_bases[room],
            tetrahedral_mesh=active_fem_mesh,
            fem_room_operators=active_fem_operators,
            fem_solver_options=fem_solver_options,
            fem_solver_session=active_fem_session,
            dataset_root=dataset_root,
            device=device,
            candidate_batch_size=candidate_batch_size,
            random_seed=random_seed,
            run_sha256=run_sha256,
            context_counts=context_counts,
            synchronize_timing=synchronize_timing,
            measure_core_forward=measure_core_forward,
            fewshot_rir_options=fewshot_rir_options,
        )
        _save_baseline_query(
            output_dir,
            result=result,
            candidates=candidates,
            candidate_scores=candidate_scores,
        )
        completed += 1
        print(
            f"[{ordinal}/{len(scheduled)}] complete {selected['query_id']} "
            f"candidates={len(candidates)} seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )
    if active_fem_session is not None:
        active_fem_session.close()
    return {
        "run_manifest_sha256": run_sha256,
        "method": method,
        "completed_queries": completed,
        "requested_queries": len(joined),
        "scheduled_queries": len(scheduled),
        "skipped_queries": len(joined) - len(scheduled),
        "skip_rooms": sorted({str(room) for room in skip_rooms}),
        "output_dir": str(output_dir),
    }
