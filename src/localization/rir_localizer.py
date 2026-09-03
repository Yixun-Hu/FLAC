"""Ground-truth-RIR sound-source coordinate regression for AcousticRooms.

This module implements the downstream readout described by Few-ShotRIR:
log-magnitude RIR -> ResNet-18 -> linear relative-coordinate head.  It is
deliberately independent from the acoustic renderer, AGREE, and the discrete
candidate-grid localization protocol.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.baselines.fewshot_rir import _resnet18
from src.data.fewshot_rir import load_ar_inventory, load_ar_positions, load_rir_waveform


RIR_LOCALIZER_CHECKPOINT_SCHEMA_VERSION = 2
RIR_LOCALIZER_MODEL_TYPE = "rir_coordinate_localizer"


@dataclass(frozen=True)
class RIRLocalizationRecord:
    query_id: str
    scene: str
    room: str
    filename: str


class RIRLogMagnitude(nn.Module):
    """Exact odd-FFT log-magnitude transform used by the AR FewshotRiR model."""

    def __init__(
        self,
        *,
        sample_size: int = 10240,
        n_fft: int = 511,
        hop_length: int = 40,
        win_length: int = 248,
        log_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        values = (sample_size, n_fft, hop_length, win_length)
        if any(int(value) <= 0 for value in values):
            raise ValueError("STFT dimensions must be positive")
        if win_length > n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if not math.isfinite(log_epsilon) or log_epsilon <= 0:
            raise ValueError("log_epsilon must be finite and positive")
        self.sample_size = int(sample_size)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.log_epsilon = float(log_epsilon)
        self.output_frequency_bins = self.n_fft // 2 + 1
        # Reflect padding adds exactly n_fft samples, matching the generator's
        # odd-FFT librosa compatibility path.
        self.output_frames = self.sample_size // self.hop_length + 1
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0]
        if waveforms.ndim != 2 or waveforms.shape[-1] != self.sample_size:
            raise ValueError(
                f"waveforms must have shape [B, {self.sample_size}] or [B, 1, {self.sample_size}]"
            )
        if not torch.isfinite(waveforms).all():
            raise ValueError("waveforms must be finite")
        values = waveforms.float()
        left = self.n_fft // 2
        right = self.n_fft - left
        padded = F.pad(values.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=values.device, dtype=values.dtype),
            center=False,
            return_complex=True,
        )
        result = torch.log(spectrum.abs() + self.log_epsilon).unsqueeze(1)
        expected = (self.output_frequency_bins, self.output_frames)
        if result.shape[-2:] != expected:
            raise RuntimeError(
                f"log-magnitude transform produced {tuple(result.shape[-2:])}, expected {expected}"
            )
        return result


class _TinyLocalizerEncoder(nn.Module):
    """Small contract-compatible encoder used only by unit tests."""

    def __init__(self, output_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, output_features),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class RIRCoordinateLocalizer(nn.Module):
    """Single-channel ResNet-18 readout of receiver-relative xyz coordinates."""

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        feature_dimensions: int = 512,
        output_dimensions: int = 3,
    ) -> None:
        super().__init__()
        if int(output_dimensions) != 3:
            raise ValueError("AcousticRooms localization must predict xyz coordinates")
        if architecture == "resnet18":
            if int(feature_dimensions) != 512:
                raise ValueError("ResNet-18 must retain its 512-dimensional feature")
            self.encoder = _resnet18(1)
        elif architecture == "tiny_test":
            self.encoder = _TinyLocalizerEncoder(int(feature_dimensions))
        else:
            raise ValueError(f"unsupported localization architecture: {architecture}")
        self.head = nn.Linear(int(feature_dimensions), int(output_dimensions))
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.architecture = str(architecture)
        self.feature_dimensions = int(feature_dimensions)
        self.output_dimensions = int(output_dimensions)

    def forward(self, log_spectrograms: torch.Tensor) -> torch.Tensor:
        if log_spectrograms.ndim != 4 or log_spectrograms.shape[1] != 1:
            raise ValueError("localizer input must have shape [B, 1, F, W]")
        if not torch.isfinite(log_spectrograms).all():
            raise ValueError("localizer input must be finite")
        return self.head(self.encoder(log_spectrograms.float()))


class AcousticRoomsRIRLocalizationDataset(Dataset):
    """Individual GT AR RIRs paired with receiver-relative source xyz."""

    def __init__(
        self,
        *,
        dataset_root: Path | str,
        split_path: Path | str,
        sample_rate: int = 22050,
        sample_size: int = 10240,
        audio_folder: str = "single_channel_ir_1",
        metadata_folder: str = "metadata",
        included_rooms: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split_path = Path(split_path)
        self.sample_rate = int(sample_rate)
        self.sample_size = int(sample_size)
        self.audio_folder = str(audio_folder)
        self.metadata_folder = str(metadata_folder)
        inventory = load_ar_inventory(self.split_path)
        selected_rooms = None
        if included_rooms is not None:
            selected_rooms = {
                (str(scene), str(room)) for scene, room in included_rooms
            }
            unknown = selected_rooms.difference(inventory)
            if unknown:
                raise ValueError(
                    f"requested rooms are absent from the localization split: {sorted(unknown)}"
                )
            if not selected_rooms:
                raise ValueError("included_rooms cannot be empty")
        records = []
        room_indices: dict[tuple[str, str], list[int]] = {}
        for (scene, room), filenames in sorted(inventory.items()):
            key = (scene, room)
            if selected_rooms is not None and key not in selected_rooms:
                continue
            room_indices[key] = []
            for filename in filenames:
                index = len(records)
                records.append(
                    RIRLocalizationRecord(
                        query_id=str(Path(self.audio_folder) / scene / room / filename),
                        scene=scene,
                        room=room,
                        filename=filename,
                    )
                )
                room_indices[key].append(index)
        if not records:
            raise ValueError("localization split contains no RIRs")
        self.records = tuple(records)
        self.room_indices = {
            key: tuple(indices) for key, indices in sorted(room_indices.items())
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        waveform = load_rir_waveform(
            self.dataset_root / record.query_id,
            sample_rate=self.sample_rate,
            sample_size=self.sample_size,
        )
        source, receiver = load_ar_positions(
            self.dataset_root,
            record.scene,
            record.room,
            record.filename,
            metadata_folder=self.metadata_folder,
        )
        relative = torch.from_numpy((source - receiver).astype(np.float32))
        return {
            "waveform": waveform,
            "relative_source": relative,
            "source_global": torch.from_numpy(source.astype(np.float32)),
            "receiver_global": torch.from_numpy(receiver.astype(np.float32)),
            "query_id": record.query_id,
            "scene": record.scene,
            "room": record.room,
        }

def split_localizer_rooms(
    split_path: Path | str,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Create a deterministic room-disjoint train/validation partition."""

    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    rooms = tuple(sorted(load_ar_inventory(split_path)))
    if len(rooms) < 2:
        raise ValueError("a room-disjoint split requires at least two rooms")
    validation_count = min(
        len(rooms) - 1,
        max(1, int(round(len(rooms) * fraction))),
    )
    permutation = np.random.default_rng(int(seed)).permutation(len(rooms))
    validation_positions = set(int(value) for value in permutation[:validation_count])
    training = tuple(room for index, room in enumerate(rooms) if index not in validation_positions)
    validation = tuple(room for index, room in enumerate(rooms) if index in validation_positions)
    return training, validation


def localizer_model_options(model_config: dict[str, Any]) -> dict[str, Any]:
    if model_config.get("model_type") != RIR_LOCALIZER_MODEL_TYPE:
        raise ValueError("model config is not an RIR coordinate localizer")
    if int(model_config.get("audio_channels", 0)) != 1:
        raise ValueError("the AR RIR localizer requires monaural spectrograms")
    options = dict(model_config.get("model", {}))
    options.setdefault("architecture", "resnet18")
    options.setdefault("feature_dimensions", 512)
    options.setdefault("output_dimensions", 3)
    return options


def localizer_transform_from_config(model_config: dict[str, Any]) -> RIRLogMagnitude:
    if model_config.get("model_type") != RIR_LOCALIZER_MODEL_TYPE:
        raise ValueError("model config is not an RIR coordinate localizer")
    preprocessing = dict(model_config.get("preprocessing", {}))
    if preprocessing.get("input_normalization") != "none":
        raise ValueError("the paper-aligned localizer does not apply input z-score")
    return RIRLogMagnitude(
        sample_size=int(model_config["sample_size"]),
        n_fft=int(preprocessing.get("n_fft", 511)),
        hop_length=int(preprocessing.get("hop_length", 40)),
        win_length=int(preprocessing.get("win_length", 248)),
        log_epsilon=float(preprocessing.get("log_epsilon", 1e-8)),
    )


def build_rir_localizer(
    model_config: dict[str, Any],
) -> RIRCoordinateLocalizer:
    return RIRCoordinateLocalizer(
        **localizer_model_options(model_config),
    )


def load_rir_localizer_checkpoint(
    checkpoint_path: Path | str,
    device: str | torch.device,
) -> tuple[RIRCoordinateLocalizer, RIRLogMagnitude, dict[str, Any]]:
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("RIR localizer checkpoint must be an object")
    if bundle.get("schema_version") != RIR_LOCALIZER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported RIR localizer checkpoint schema")
    if bundle.get("model_type") != RIR_LOCALIZER_MODEL_TYPE:
        raise ValueError("checkpoint is not an RIR coordinate localizer")
    model_config = bundle.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("RIR localizer checkpoint lacks model config")
    model = build_rir_localizer(model_config)
    state = bundle.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("RIR localizer checkpoint lacks state_dict")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    transform = localizer_transform_from_config(model_config).eval().to(device)
    return model, transform, bundle


def coordinate_error_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must share shape [B, 3]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target coordinates must be finite")
    difference = (prediction.float() - target.float()).abs()
    euclidean = torch.linalg.vector_norm(prediction.float() - target.float(), dim=1)
    return {
        "count": int(prediction.shape[0]),
        "absolute_coordinate_error_sum_m": float(difference.sum().item()),
        "euclidean_error_sum_m": float(euclidean.sum().item()),
        "euclidean_errors_m": [float(value) for value in euclidean.tolist()],
        "success_0_5m_count": int((euclidean <= 0.5).sum().item()),
        "success_1_0m_count": int((euclidean <= 1.0).sum().item()),
    }


def finalize_coordinate_metrics(sums: dict[str, Any]) -> dict[str, float | int]:
    count = int(sums.get("count", 0))
    if count <= 0:
        raise ValueError("coordinate metrics require at least one query")
    absolute_sum = float(sums["absolute_coordinate_error_sum_m"])
    euclidean_sum = float(sums["euclidean_error_sum_m"])
    euclidean_errors = np.asarray(sums["euclidean_errors_m"], dtype=np.float64)
    if euclidean_errors.shape != (count,) or not np.isfinite(euclidean_errors).all():
        raise ValueError("coordinate metric accumulator is malformed")
    return {
        "query_count": count,
        "mean_l1_distance_m": absolute_sum / count,
        "coordinate_mae_m": absolute_sum / (count * 3),
        "mean_euclidean_error_m": euclidean_sum / count,
        "median_euclidean_error_m": float(np.median(euclidean_errors)),
        "success_rate_0_5m": int(sums["success_0_5m_count"]) / count,
        "success_rate_1_0m": int(sums["success_1_0m_count"]) / count,
    }


def localizer_checkpoint_payload(
    *,
    model: RIRCoordinateLocalizer,
    model_config: dict[str, Any],
    step: int,
    best_validation_l1_m: float,
    run_manifest_sha256: str,
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RIR_LOCALIZER_CHECKPOINT_SCHEMA_VERSION,
        "model_type": RIR_LOCALIZER_MODEL_TYPE,
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "step": int(step),
        "best_validation_l1_m": float(best_validation_l1_m),
        "run_manifest_sha256": str(run_manifest_sha256),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    return payload
