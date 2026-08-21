"""Materialize the released exp_01 global/per-worker context RNG stream."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

from src.configs.dataset_configs.custom_metadata import AR_md
from src.data.dataset import create_dataloader_from_config
from src.localization.ar_queries import (
    ContextProtocol,
    _canonical_sha,
    attach_context_selections,
    parse_split_queries,
)


def _validate_release_config(config: dict, protocol: ContextProtocol) -> None:
    datasets = config.get("datasets", [])
    acoustic = config.get("modalities", {}).get("acoustic_context", {})
    required = {
        "is_eval": True,
        "unseeneval": True,
        "random_crop": False,
        "augs": False,
    }
    if len(datasets) != 1 or any(config.get(key) != value for key, value in required.items()):
        raise ValueError("dataset config is not the released unseen-eval protocol")
    if acoustic.get("max_context") != protocol.max_context or not acoustic.get("load"):
        raise ValueError("released K=8 acoustic context is required")
    if protocol != ContextProtocol():
        raise ValueError("exp_09 context materialization requires exact exp_01 defaults")


def _global_locations(relpath: str, metadata_root: Path):
    source, receiver = AR_md.get_receiver_source_location(relpath, str(metadata_root))
    return np.asarray(source, dtype=np.float64), np.asarray(receiver, dtype=np.float64)


def materialize_original_contexts(
    dataset_config_path: Path | str,
    dataset_root: Path | str,
    protocol: ContextProtocol = ContextProtocol(),
) -> dict:
    """Run the original full loader once and return a frozen context manifest."""

    dataset_config_path = Path(dataset_config_path)
    dataset_root = Path(dataset_root).resolve()
    config = json.loads(dataset_config_path.read_text())
    _validate_release_config(config, protocol)
    split_path = Path(config["datasets"][0]["json_file_path"])
    queries = parse_split_queries(split_path, dataset_root)
    if len(queries) != 6337:
        raise ValueError(f"expected complete 6,337-query split, got {len(queries)}")

    runtime_config = copy.deepcopy(config)
    runtime_config["datasets"][0]["path"] = str(dataset_root)
    runtime_config["modalities"]["acoustic_context"]["record_paths"] = True

    pl.seed_everything(protocol.seed, workers=True)
    dataloader = create_dataloader_from_config(
        runtime_config,
        batch_size=protocol.batch_size,
        num_workers=protocol.num_workers,
        sample_rate=22050,
        sample_size=10240,
        audio_channels=1,
        shuffle=protocol.shuffle,
    )

    selections: list[list[str] | None] = [None] * len(queries)
    relative_poses: list[np.ndarray | None] = [None] * len(queries)
    for _audio, metadata_batch in dataloader:
        for metadata in metadata_batch:
            index = int(metadata["idx"])
            if index < 0 or index >= len(queries) or metadata["relpath"] != queries[index].relpath:
                raise RuntimeError("loader order changed or SampleDataset recursively replaced a query")
            if selections[index] is not None:
                raise RuntimeError(f"duplicate loader index {index}")
            selections[index] = [str(Path(path)) for path in metadata["context_paths"]]
            relative_poses[index] = metadata["context_poses"].detach().cpu().numpy()

    if any(selection is None for selection in selections):
        raise RuntimeError("loader did not materialize every full-split query")
    manifest = attach_context_selections(queries, selections, protocol)  # type: ignore[arg-type]

    metadata_root = dataset_root / "metadata"
    for query, record, rel_poses in zip(queries, manifest["records"], relative_poses):
        source, receiver = _global_locations(query.relpath, metadata_root)
        context_sources = []
        for context_relpath in record["contexts"]:
            context_source, context_receiver = _global_locations(context_relpath, metadata_root)
            if not np.allclose(context_receiver, receiver, atol=1e-6, rtol=0):
                raise RuntimeError(f"receiver mismatch in {query.query_id}")
            context_sources.append(context_source)
        expected_relative = np.stack(context_sources) - receiver
        if not np.allclose(expected_relative, rel_poses, atol=1e-5, rtol=0):
            raise RuntimeError(f"context pose/path mismatch in {query.query_id}")
        record["source_global"] = source.tolist()
        record["receiver_global"] = receiver.tolist()
        record["context_sources_global"] = np.stack(context_sources).tolist()

    manifest.pop("sha256")
    manifest["split_sha256"] = _canonical_sha(json.loads(split_path.read_text()))
    manifest["dataset_config"] = str(dataset_config_path)
    manifest["sha256"] = _canonical_sha(manifest)
    return manifest
