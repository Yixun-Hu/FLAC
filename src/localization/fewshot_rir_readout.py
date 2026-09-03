"""Continuous-coordinate FewshotRiR localization with a frozen RIR readout."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.localization.fewshot_rir import build_fewshot_rir_candidate_batch


READOUT_QUERY_SCHEMA_VERSION = 1
READOUT_SUMMARY_SCHEMA_VERSION = 1
READOUT_CONTEXT_COUNT = 8
READOUT_GENERATION_COUNT = 1
PAPER_REFERENCE_CONTEXT_COUNT = 20
READOUT_PROTOCOL_NAME = "fewshotrir_ar_k8_gt_query_direct_resnet18_v2"


def readout_protocol() -> dict[str, Any]:
    return {
        "name": READOUT_PROTOCOL_NAME,
        "context_count": READOUT_CONTEXT_COUNT,
        "generation_count": READOUT_GENERATION_COUNT,
        "paper_reference_context_count": PAPER_REFERENCE_CONTEXT_COUNT,
        "directly_comparable_to_paper_n20_sle": False,
        "non_comparability_reason": (
            "this AR protocol and its frozen context manifest use K=8; "
            "the paper SLE table uses N=20"
        ),
    }


def _synchronize(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _spectrogram_diagnostics(values: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(values)
    if not bool(finite.all()):
        raise RuntimeError("localization readout received a non-finite spectrogram")
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.float().std(correction=0)),
    }


def point_localization_metrics(
    prediction_global: np.ndarray,
    target_global: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction_global, dtype=np.float64)
    target = np.asarray(target_global, dtype=np.float64)
    if prediction.shape != (3,) or target.shape != (3,):
        raise ValueError("point localization coordinates must have shape [3]")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("point localization coordinates must be finite")
    signed = prediction - target
    absolute = np.abs(signed)
    euclidean = float(np.linalg.norm(signed))
    return {
        "signed_error_xyz_m": signed.tolist(),
        "absolute_error_xyz_m": absolute.tolist(),
        "l1_distance_m": float(absolute.sum()),
        "coordinate_mae_m": float(absolute.mean()),
        "euclidean_error_m": euclidean,
        "success_0_5m": bool(euclidean <= 0.5),
        "success_1_0m": bool(euclidean <= 1.0),
    }


@torch.inference_mode()
def infer_fewshot_rir_readout_query(
    generator: torch.nn.Module,
    localizer: torch.nn.Module,
    gt_transform: torch.nn.Module,
    *,
    observed_waveform: torch.Tensor,
    context_metadata: dict[str, torch.Tensor | np.ndarray],
    source_global: np.ndarray,
    receiver_global: np.ndarray,
    device: str | torch.device,
    synchronize_timing: bool = False,
) -> dict[str, Any]:
    """Generate exactly one query RIR and regress exactly one continuous point."""

    device = torch.device(device)
    source = np.asarray(source_global, dtype=np.float32)
    receiver = np.asarray(receiver_global, dtype=np.float32)
    if source.shape != (3,) or receiver.shape != (3,):
        raise ValueError("source_global and receiver_global must have shape [3]")
    if observed_waveform.ndim != 2 or observed_waveform.shape[0] != 1:
        raise ValueError("observed_waveform must have shape [1, T]")
    batch = build_fewshot_rir_candidate_batch(
        context_metadata,
        source[None],
        receiver,
        context_count=READOUT_CONTEXT_COUNT,
    )
    if batch["context_depth"].shape[:2] != (1, READOUT_CONTEXT_COUNT):
        raise RuntimeError("readout protocol did not construct eight contexts")
    if batch["query_poses"].shape[:2] != (1, READOUT_GENERATION_COUNT):
        raise RuntimeError("readout protocol did not construct one generation query")
    batch = {key: value.to(device) for key, value in batch.items()}

    generator.eval()
    localizer.eval()
    _synchronize(device, synchronize_timing)
    generation_started = time.perf_counter()
    raw_output = generator(**batch)
    _synchronize(device, synchronize_timing)
    generation_seconds = time.perf_counter() - generation_started
    if raw_output.ndim != 5 or raw_output.shape[:2] != (
        1,
        READOUT_GENERATION_COUNT,
    ) or raw_output.shape[-1] != 1:
        raise RuntimeError(
            "FewshotRiR must return one channel-last log-magnitude spectrogram"
        )
    generated_log_spectrogram = raw_output[:, 0].permute(0, 3, 1, 2).float()

    _synchronize(device, synchronize_timing)
    generated_readout_started = time.perf_counter()
    generated_relative = localizer(generated_log_spectrogram)
    _synchronize(device, synchronize_timing)
    generated_readout_seconds = time.perf_counter() - generated_readout_started

    gt_log_spectrogram = gt_transform(observed_waveform.to(device))
    if gt_log_spectrogram.shape != generated_log_spectrogram.shape:
        raise RuntimeError(
            "generator and GT-localizer log-spectrogram layouts do not match"
        )
    _synchronize(device, synchronize_timing)
    gt_readout_started = time.perf_counter()
    gt_relative = localizer(gt_log_spectrogram)
    _synchronize(device, synchronize_timing)
    gt_readout_seconds = time.perf_counter() - gt_readout_started

    generated_relative_np = generated_relative[0].float().cpu().numpy()
    gt_relative_np = gt_relative[0].float().cpu().numpy()
    return {
        "target_relative": source - receiver,
        "generated_prediction_relative": generated_relative_np,
        "generated_prediction_global": receiver + generated_relative_np,
        "gt_prediction_relative": gt_relative_np,
        "gt_prediction_global": receiver + gt_relative_np,
        "timing_seconds": {
            "fewshotrir_generation": generation_seconds,
            "generated_rir_localizer": generated_readout_seconds,
            "gt_rir_localizer": gt_readout_seconds,
            "generated_pipeline": generation_seconds + generated_readout_seconds,
        },
        "spectrogram_diagnostics": {
            "generated_log_magnitude": _spectrogram_diagnostics(
                generated_log_spectrogram
            ),
            "ground_truth_log_magnitude": _spectrogram_diagnostics(
                gt_log_spectrogram
            ),
        },
    }


def build_readout_query_result(
    *,
    query_index: int,
    query_id: str,
    scene: str,
    room: str,
    receiver_id: str | int,
    source_global: np.ndarray,
    receiver_global: np.ndarray,
    inference: dict[str, Any],
    run_manifest_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    source = np.asarray(source_global, dtype=np.float64)
    receiver = np.asarray(receiver_global, dtype=np.float64)
    generated_global = np.asarray(
        inference["generated_prediction_global"], dtype=np.float64
    )
    gt_global = np.asarray(inference["gt_prediction_global"], dtype=np.float64)
    return {
        "schema_version": READOUT_QUERY_SCHEMA_VERSION,
        "run_manifest_sha256": str(run_manifest_sha256),
        "protocol": {
            **readout_protocol(),
            "query_source_input": "ground_truth_continuous_coordinate",
            "output": "one_continuous_xyz_point",
            "candidate_search": False,
            "agree_scoring": False,
            "griffin_lim": False,
        },
        "query_index": int(query_index),
        "query_id": str(query_id),
        "scene": str(scene),
        "room": str(room),
        "receiver_id": str(receiver_id),
        "target": {
            "source_global": source.tolist(),
            "receiver_global": receiver.tolist(),
            "source_relative_to_receiver": (source - receiver).tolist(),
        },
        "generated_rir_readout": {
            "prediction_relative_to_receiver": np.asarray(
                inference["generated_prediction_relative"], dtype=np.float64
            ).tolist(),
            "prediction_global": generated_global.tolist(),
            "metrics": point_localization_metrics(generated_global, source),
        },
        "ground_truth_rir_readout": {
            "prediction_relative_to_receiver": np.asarray(
                inference["gt_prediction_relative"], dtype=np.float64
            ).tolist(),
            "prediction_global": gt_global.tolist(),
            "metrics": point_localization_metrics(gt_global, source),
        },
        "timing_seconds": {
            **inference["timing_seconds"],
            "end_to_end": float(elapsed_seconds),
        },
        "spectrogram_diagnostics": inference["spectrogram_diagnostics"],
    }


def summarize_readout_results(
    results: list[dict[str, Any]],
    *,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty readout run")
    branches = {}
    for branch in ("generated_rir_readout", "ground_truth_rir_readout"):
        metrics = [result[branch]["metrics"] for result in results]
        euclidean = np.asarray(
            [item["euclidean_error_m"] for item in metrics], dtype=np.float64
        )
        l1 = np.asarray([item["l1_distance_m"] for item in metrics], dtype=np.float64)
        branches[branch] = {
            "query_count": len(metrics),
            "mean_l1_distance_m": float(l1.mean()),
            "coordinate_mae_m": float(l1.mean() / 3.0),
            "mean_euclidean_error_m": float(euclidean.mean()),
            "median_euclidean_error_m": float(np.median(euclidean)),
            "success_rate_0_5m": float(np.mean(euclidean <= 0.5)),
            "success_rate_1_0m": float(np.mean(euclidean <= 1.0)),
        }
    generated_pipeline = np.asarray(
        [result["timing_seconds"]["generated_pipeline"] for result in results],
        dtype=np.float64,
    )
    end_to_end = np.asarray(
        [result["timing_seconds"]["end_to_end"] for result in results],
        dtype=np.float64,
    )
    return {
        "schema_version": READOUT_SUMMARY_SCHEMA_VERSION,
        "run_manifest_sha256": str(run_manifest_sha256),
        "protocol": readout_protocol(),
        "query_count": len(results),
        "metrics": branches,
        "latency_seconds": {
            "mean_generated_pipeline": float(generated_pipeline.mean()),
            "median_generated_pipeline": float(np.median(generated_pipeline)),
            "mean_end_to_end": float(end_to_end.mean()),
        },
    }
