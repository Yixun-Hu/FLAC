"""Room-level AcousticRooms episodes for the official-architecture FewshotRiR."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


_RIR_PATTERN = re.compile(r"^S0*(\d+)_R0*(\d+)_hybrid_IR\.wav$")


def parse_rir_filename(filename: str) -> tuple[int, int]:
    match = _RIR_PATTERN.match(Path(filename).name)
    if match is None:
        raise ValueError(f"invalid AcousticRooms RIR filename: {filename}")
    return int(match.group(1)), int(match.group(2))


def load_ar_inventory(path: str | os.PathLike[str]) -> dict[tuple[str, str], list[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"AcousticRooms inventory must be an object: {path}")
    rooms: dict[tuple[str, str], list[str]] = {}
    for scene, scene_rooms in payload.items():
        if not isinstance(scene_rooms, dict):
            raise ValueError(f"scene {scene!r} must map room names to RIR lists")
        for room, names in scene_rooms.items():
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError(f"room {scene}/{room} must contain RIR filenames")
            rooms[(str(scene), str(room))] = list(names)
    return rooms


def load_rir_waveform(
    path: str | os.PathLike[str],
    *,
    sample_rate: int,
    sample_size: int,
) -> torch.Tensor:
    """Load one monaural RIR and deterministically pad/crop it to the model length."""

    try:
        import torchaudio
    except ImportError as error:  # pragma: no cover - environment setup
        raise ImportError("FewshotRiR AcousticRooms loading requires torchaudio") from error
    waveform, rate = torchaudio.load(str(path))
    if int(rate) != int(sample_rate):
        raise ValueError(f"{path} has sample rate {rate}, expected {sample_rate}")
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"{path} must be monaural, got shape {tuple(waveform.shape)}")
    waveform = waveform[0].float()
    if waveform.numel() < sample_size:
        waveform = F.pad(waveform, (0, sample_size - waveform.numel()))
    else:
        waveform = waveform[:sample_size]
    return waveform.clamp(-1.0, 1.0)


def rir_magnitude_spectrogram(
    waveforms: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Return linear magnitudes with the official channel-last STFT layout."""

    if waveforms.ndim < 1:
        raise ValueError("waveforms must have at least one dimension")
    sample_size = waveforms.shape[-1]
    flattened = waveforms.reshape(-1, sample_size).float()
    window = torch.hann_window(win_length, device=flattened.device, dtype=flattened.dtype)
    left = n_fft // 2
    right = n_fft - left
    padded = F.pad(flattened.unsqueeze(1), (left, right), mode="reflect").squeeze(1)
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return spectrum.abs().reshape(*waveforms.shape[:-1], *spectrum.shape[-2:], 1)


def load_ar_positions(
    dataset_root: str | os.PathLike[str],
    scene: str,
    room: str,
    filename: str,
    *,
    metadata_folder: str = "metadata",
) -> tuple[np.ndarray, np.ndarray]:
    source, receiver = parse_rir_filename(filename)
    metadata_path = (
        Path(dataset_root)
        / metadata_folder
        / scene
        / room
        / f"S00{source}_R00{receiver}.json"
    )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    source_position = np.asarray(metadata["src_loc"], dtype=np.float32)
    receiver_position = np.asarray(metadata["rec_loc"], dtype=np.float32)
    if source_position.shape != (3,) or receiver_position.shape != (3,):
        raise ValueError(f"invalid source/receiver location in {metadata_path}")
    if not np.isfinite(source_position).all() or not np.isfinite(receiver_position).all():
        raise ValueError(f"non-finite source/receiver location in {metadata_path}")
    return source_position, receiver_position


def select_near_coincident_contexts(
    dataset_root: str | os.PathLike[str],
    scene: str,
    room: str,
    names: list[str] | tuple[str, ...],
    *,
    metadata_folder: str = "metadata",
) -> tuple[str, ...]:
    """Approximate upstream colocated echoes with the nearest AR RIR per source.

    AcousticRooms has no emitted-and-recorded-at-the-same-pose measurement.  A
    numeric ``Sxxx_Rxxx`` index match is not a spatial match, so select the
    available receiver closest to each distinct source using metadata geometry.
    """

    source_positions: dict[int, np.ndarray] = {}
    receiver_positions: dict[int, np.ndarray] = {}
    available_by_source: dict[int, list[tuple[int, str]]] = {}
    for name in sorted(set(names)):
        source_index, receiver_index = parse_rir_filename(name)
        available_by_source.setdefault(source_index, []).append((receiver_index, name))
        if source_index in source_positions and receiver_index in receiver_positions:
            continue
        source, receiver = load_ar_positions(
            dataset_root,
            scene,
            room,
            name,
            metadata_folder=metadata_folder,
        )
        if source_index in source_positions and not np.allclose(
            source_positions[source_index], source, atol=1e-6, rtol=0
        ):
            raise ValueError(f"inconsistent source position for {scene}/{room}/S{source_index}")
        if receiver_index in receiver_positions and not np.allclose(
            receiver_positions[receiver_index], receiver, atol=1e-6, rtol=0
        ):
            raise ValueError(
                f"inconsistent receiver position for {scene}/{room}/R{receiver_index}"
            )
        source_positions[source_index] = source
        receiver_positions[receiver_index] = receiver

    selected = []
    for source_index in sorted(available_by_source):
        source = source_positions[source_index]
        choices = available_by_source[source_index]
        _, filename = min(
            choices,
            key=lambda item: (
                float(np.linalg.norm(source - receiver_positions[item[0]])),
                item[0],
                item[1],
            ),
        )
        selected.append(filename)
    if not selected:
        raise ValueError(f"room {scene}/{room} contains no context RIR candidates")
    return tuple(selected)


def load_ar_depth(
    dataset_root: str | os.PathLike[str],
    scene: str,
    room: str,
    receiver: int,
    *,
    depth_folder: str = "depth_map",
    depth_size: tuple[int, int] = (128, 256),
    depth_max_m: float = 67.16327,
) -> torch.Tensor:
    if not math.isfinite(depth_max_m) or depth_max_m <= 0:
        raise ValueError("depth_max_m must be finite and positive")
    depth_path = Path(dataset_root) / depth_folder / scene / room / f"{receiver}.npy"
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth map must be 2-D: {depth_path}")
    depth = np.nan_to_num(depth, nan=0.0, posinf=depth_max_m, neginf=0.0)
    values = torch.from_numpy(depth).clamp(0.0, depth_max_m) / float(depth_max_m)
    values = F.interpolate(
        values[None, None],
        size=tuple(int(value) for value in depth_size),
        mode="bilinear",
        align_corners=False,
    )
    return values[0]


@dataclass(frozen=True)
class RoomEpisode:
    scene: str
    room: str
    queries: tuple[str, ...]
    near_contexts: tuple[str, ...]


class AcousticRoomsFewshotRiRDataset(Dataset):
    """Sample K nearest-endpoint contexts and Q target RIRs per room episode."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        sample_rate: int,
        sample_size: int,
    ) -> None:
        super().__init__()
        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("FewshotRiR dataset config requires a dataset object")
        self.dataset_root = Path(dataset["path"])
        self.audio_folder = str(dataset.get("audio_folder", "single_channel_ir_1"))
        self.metadata_folder = str(dataset.get("metadata_folder", "metadata"))
        self.depth_folder = str(dataset.get("depth_folder", "depth_map"))
        query_rooms = load_ar_inventory(dataset["query_json_file_path"])
        context_rooms = load_ar_inventory(dataset["context_json_file_path"])

        episodes: list[RoomEpisode] = []
        for key, query_names in sorted(query_rooms.items()):
            if key not in context_rooms:
                raise ValueError(f"context inventory does not contain room {key[0]}/{key[1]}")
            near_contexts = select_near_coincident_contexts(
                self.dataset_root,
                key[0],
                key[1],
                context_rooms[key],
                metadata_folder=self.metadata_folder,
            )
            context_set = set(near_contexts)
            queries = tuple(
                name for name in query_names if name not in context_set
            )
            if not queries:
                raise ValueError(
                    f"room {key[0]}/{key[1]} contains no query RIRs outside its contexts"
                )
            episodes.append(RoomEpisode(key[0], key[1], queries, near_contexts))
        if not episodes:
            raise ValueError("FewshotRiR dataset contains no rooms")
        self.rooms = tuple(episodes)
        self.episodes_per_room = int(config.get("episodes_per_room", 1))
        self.max_context = int(config.get("max_context", 8))
        self.max_queries = int(config.get("max_queries", 60))
        self.depth_size = tuple(int(value) for value in config.get("depth_size", (128, 256)))
        self.depth_max_m = float(config.get("depth_max_m", 67.16327))
        self.seed = int(config.get("seed", 0))
        self.sample_rate = int(sample_rate)
        self.sample_size = int(sample_size)
        self.n_fft = int(config.get("n_fft", 511))
        self.hop_length = int(config.get("hop_length", 40))
        self.win_length = int(config.get("win_length", 248))
        if self.episodes_per_room <= 0 or self.max_context <= 0 or self.max_queries <= 0:
            raise ValueError("episode and context/query counts must be positive")
        if len(self.depth_size) != 2 or any(value <= 0 for value in self.depth_size):
            raise ValueError("depth_size must contain two positive dimensions")

    def __len__(self) -> int:
        return len(self.rooms) * self.episodes_per_room

    @staticmethod
    def _sample_names(
        rng: np.random.Generator,
        names: tuple[str, ...],
        count: int,
    ) -> list[str]:
        if count <= len(names):
            indices = rng.choice(len(names), size=count, replace=False)
        else:
            # Match the release: use every available pose once in randomized
            # order, then fill only the shortage with replacement.
            first = rng.choice(len(names), size=len(names), replace=False)
            repeated = rng.choice(len(names), size=count - len(names), replace=True)
            indices = np.concatenate((first, repeated))
        return [names[int(index)] for index in indices]

    def _audio_path(self, room: RoomEpisode, filename: str) -> Path:
        return self.dataset_root / self.audio_folder / room.scene / room.room / filename

    def __getitem__(self, index: int) -> dict[str, Any]:
        room = self.rooms[index % len(self.rooms)]
        cycle = index // len(self.rooms)
        room_index = index % len(self.rooms)
        rng = np.random.default_rng(self.seed + cycle * len(self.rooms) + room_index)
        context_names = self._sample_names(rng, room.near_contexts, self.max_context)
        query_names = self._sample_names(rng, room.queries, self.max_queries)

        context_waveforms = torch.stack(
            [
                load_rir_waveform(
                    self._audio_path(room, name),
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                )
                for name in context_names
            ]
        )
        query_waveforms = torch.stack(
            [
                load_rir_waveform(
                    self._audio_path(room, name),
                    sample_rate=self.sample_rate,
                    sample_size=self.sample_size,
                )
                for name in query_names
            ]
        )
        context_locations = [
            load_ar_positions(
                self.dataset_root,
                room.scene,
                room.room,
                name,
                metadata_folder=self.metadata_folder,
            )
            for name in context_names
        ]
        query_locations = [
            load_ar_positions(
                self.dataset_root,
                room.scene,
                room.room,
                name,
                metadata_folder=self.metadata_folder,
            )
            for name in query_names
        ]
        anchor = context_locations[0][1]
        context_poses = torch.from_numpy(
            np.stack(
                [np.concatenate((receiver - anchor, source - anchor)) for source, receiver in context_locations]
            ).astype(np.float32)
        )
        query_poses = torch.from_numpy(
            np.stack(
                [np.concatenate((receiver - anchor, source - anchor)) for source, receiver in query_locations]
            ).astype(np.float32)
        )
        context_depth = torch.stack(
            [
                load_ar_depth(
                    self.dataset_root,
                    room.scene,
                    room.room,
                    parse_rir_filename(name)[1],
                    depth_folder=self.depth_folder,
                    depth_size=self.depth_size,
                    depth_max_m=self.depth_max_m,
                )
                for name in context_names
            ]
        )
        stft_options = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
        }
        return {
            "context_depth": context_depth,
            "context_magnitude": rir_magnitude_spectrogram(context_waveforms, **stft_options),
            "context_poses": context_poses,
            "target_magnitude": rir_magnitude_spectrogram(query_waveforms, **stft_options),
            "query_poses": query_poses,
            "context_mask": torch.ones(self.max_context, dtype=torch.bool),
            "query_mask": torch.ones(self.max_queries, dtype=torch.bool),
            "scene": room.scene,
            "room": room.room,
        }


def create_fewshot_rir_dataloader_from_config(
    dataset_config: dict[str, Any],
    *,
    batch_size: int,
    sample_size: int,
    sample_rate: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = AcousticRoomsFewshotRiRDataset(
        dataset_config,
        sample_rate=sample_rate,
        sample_size=sample_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=bool(dataset_config.get("drop_last", True)),
    )
