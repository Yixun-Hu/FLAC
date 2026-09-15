"""AcousticRooms query order and frozen-context manifest contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ContextProtocol:
    """Released exp_01 loader settings that determine the global RNG stream."""

    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = False
    max_context: int = 8


@dataclass(frozen=True)
class QueryRecord:
    index: int
    scene: str
    room: str
    filename: str
    relpath: str
    rir_path: str
    eligible_context_relpaths: tuple[str, ...]

    @property
    def query_id(self) -> str:
        return self.relpath

    @property
    def eligible_context_count(self) -> int:
        return len(self.eligible_context_relpaths)


@lru_cache(maxsize=None)
def _room_inventory(room_dir: str) -> tuple[frozenset[int], frozenset[str]]:
    names = tuple(os.listdir(room_dir))
    source_ids = frozenset(int(name.split("_")[0][1:]) for name in names)
    return source_ids, frozenset(names)


def _eligible_contexts(rir_path: Path, dataset_root: Path) -> tuple[str, ...]:
    filename = rir_path.name
    source_id = int(filename.split("_")[0][1:])
    receiver_token = filename.split("_")[1]
    source_ids, filenames = _room_inventory(str(rir_path.parent))

    # Intentionally reproduce AR_md.py:94-102, including set iteration and the
    # S010 -> S0010 filename quirk. Do not sort this released eligible pool.
    remaining = list(set(source_ids).difference({source_id}))
    selected = []
    for node in remaining:
        candidate = f"S00{node}_{receiver_token}_hybrid_IR.wav"
        if candidate in filenames:
            selected.append(str((rir_path.parent / candidate).relative_to(dataset_root)))
    return tuple(selected)


def parse_split_queries(split_path: Path | str, dataset_root: Path | str) -> tuple[QueryRecord, ...]:
    """Parse the JSON in the exact insertion/list order used by json_scandir."""

    split_path = Path(split_path)
    dataset_root = Path(dataset_root)
    split = json.loads(split_path.read_text())
    records: list[QueryRecord] = []
    for scene, rooms in split.items():
        if not isinstance(rooms, dict):
            raise ValueError("AcousticRooms split must map scenes to room dictionaries")
        for room, filenames in rooms.items():
            for filename in filenames:
                relpath = Path("single_channel_ir_1") / scene / room / filename
                rir_path = dataset_root / relpath
                if not rir_path.is_file():
                    raise FileNotFoundError(rir_path)
                records.append(
                    QueryRecord(
                        index=len(records),
                        scene=scene,
                        room=room,
                        filename=filename,
                        relpath=str(relpath),
                        rir_path=str(rir_path),
                        eligible_context_relpaths=_eligible_contexts(rir_path, dataset_root),
                    )
                )
    return tuple(records)


def context_availability_histogram(records: Iterable[QueryRecord]) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for record in records:
        count = record.eligible_context_count
        histogram[count] = histogram.get(count, 0) + 1
    return dict(sorted(histogram.items()))


def _canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def attach_context_selections(
    queries: Sequence[QueryRecord],
    selections: Sequence[Sequence[str]],
    protocol: ContextProtocol,
) -> dict:
    """Freeze already-materialized original-loader selections into a manifest."""

    if not queries or [q.index for q in queries] != list(range(len(queries))):
        raise ValueError("queries must retain complete full split order before filtering")
    if len(selections) != len(queries):
        raise ValueError("one context selection is required for every full-split query")

    records = []
    for query, chosen in zip(queries, selections):
        chosen = [str(path) for path in chosen]
        if len(chosen) != protocol.max_context:
            raise ValueError(f"{query.query_id} does not have width {protocol.max_context}")
        eligible = set(query.eligible_context_relpaths)
        if any(path not in eligible for path in chosen):
            raise ValueError(f"{query.query_id} contains an ineligible or target context")
        records.append(
            {
                "index": query.index,
                "query_id": query.query_id,
                "scene": query.scene,
                "room": query.room,
                "filename": query.filename,
                "eligible_context_count": query.eligible_context_count,
                "contexts": chosen,
            }
        )

    payload = {
        "schema_version": 1,
        "protocol": asdict(protocol),
        "full_query_count": len(records),
        "records": records,
    }
    payload["sha256"] = _canonical_sha(payload)
    return payload


def save_context_manifest(manifest: dict, path: Path | str) -> None:
    path = Path(path)
    expected = manifest.get("sha256")
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    if expected != _canonical_sha(content):
        raise ValueError("context manifest hash is stale or invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_context_manifest(path: Path | str) -> dict:
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256", None)
    actual = _canonical_sha(manifest)
    if expected != actual:
        raise ValueError("context manifest SHA-256 mismatch")
    manifest["sha256"] = expected
    return manifest


def filter_materialized_scope(manifest: dict, excluded_room: str, expected_excluded: int) -> dict:
    if len(manifest.get("records", ())) != manifest.get("full_query_count"):
        raise ValueError("only a complete materialized full-split manifest may be filtered")
    kept = [record for record in manifest["records"] if record["room"] != excluded_room]
    excluded = len(manifest["records"]) - len(kept)
    if excluded != expected_excluded:
        raise ValueError(f"expected {expected_excluded} excluded queries, got {excluded}")
    payload = {
        "schema_version": manifest["schema_version"],
        "protocol": copy.deepcopy(manifest["protocol"]),
        "source_manifest_sha256": manifest["sha256"],
        "excluded_room": excluded_room,
        "excluded_count": excluded,
        "records": copy.deepcopy(kept),
    }
    payload["sha256"] = _canonical_sha(payload)
    return payload


def clone_with_candidate(metadata: dict, candidate_global, receiver_global) -> dict:
    """Clone one frozen context and replace only the receiver-relative target."""

    candidate = np.asarray(candidate_global, dtype=np.float32)
    receiver = np.asarray(receiver_global, dtype=np.float32)
    if candidate.shape != (3,) or receiver.shape != (3,):
        raise ValueError("candidate and receiver coordinates must have shape (3,)")
    relative = torch.from_numpy(candidate - receiver)
    cloned = copy.deepcopy(metadata)
    cloned["source"] = relative
    cloned["source_vit"] = relative.unsqueeze(0)
    return cloned
