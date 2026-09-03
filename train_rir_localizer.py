#!/usr/bin/env python3
"""Train the frozen GT-RIR -> ResNet-18 -> relative-xyz readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from src.localization.pilot import canonical_sha256
from src.localization.rir_localizer import (
    RIR_LOCALIZER_CHECKPOINT_SCHEMA_VERSION,
    AcousticRoomsRIRLocalizationDataset,
    build_rir_localizer,
    coordinate_error_sums,
    finalize_coordinate_metrics,
    localizer_checkpoint_payload,
    localizer_transform_from_config,
    split_localizer_rooms,
)
from src.localization.runner import _atomic_json, file_sha256, initialize_run


DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "src/configs/model_configs/baselines/RIRLocalizer_AR.json"
)
DEFAULT_TRAIN_SPLIT = REPO_ROOT / "data/AR/train.json"


class RoomBalancedBatchSampler(Sampler[list[int]]):
    """Stateless per-step sampling, so an interrupted run resumes exactly."""

    def __init__(
        self,
        dataset: AcousticRoomsRIRLocalizationDataset,
        *,
        batch_size: int,
        seed: int,
        start_step: int,
        stop_step: int,
    ) -> None:
        self.rooms = tuple(dataset.room_indices)
        self.room_indices = dataset.room_indices
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        if self.batch_size <= 0 or not 0 <= self.start_step <= self.stop_step:
            raise ValueError("invalid room-balanced batch sampler dimensions")

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            room_positions = rng.integers(len(self.rooms), size=self.batch_size)
            batch = []
            for position in room_positions:
                choices = self.room_indices[self.rooms[int(position)]]
                batch.append(int(choices[int(rng.integers(len(choices)))]))
            yield batch

    def __len__(self) -> int:
        return self.stop_step - self.start_step


def _loader_options(num_workers: int, use_cuda: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(use_cuda),
    }
    if num_workers > 0:
        options["persistent_workers"] = True
    return options


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def meter_l1_distance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-example SLE: |dx| + |dy| + |dz|, measured in meters."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must share shape [B, 3]")
    return (prediction.float() - target.float()).abs().sum(dim=1).mean()


def _accumulate_metrics(total: dict[str, Any], batch: dict[str, Any]) -> None:
    for key in (
        "count",
        "absolute_coordinate_error_sum_m",
        "euclidean_error_sum_m",
        "success_0_5m_count",
        "success_1_0m_count",
    ):
        total[key] = total.get(key, 0) + batch[key]
    total.setdefault("euclidean_errors_m", []).extend(batch["euclidean_errors_m"])


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    transform: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, float | int]:
    model.eval()
    totals: dict[str, Any] = {}
    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        target = batch["relative_source"].to(device, non_blocking=True)
        log_spectrograms = transform(waveforms)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            prediction = model(log_spectrograms)
        metrics = coordinate_error_sums(prediction, target)
        _accumulate_metrics(totals, metrics)
    return finalize_coordinate_metrics(totals)


def _lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    minimum_ratio: float,
) -> float:
    if step < warmup_steps:
        return max(minimum_ratio, (step + 1) / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _checkpoint(
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    best_validation_l1_m: float,
    run_sha256: str,
    bad_validation_count: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = localizer_checkpoint_payload(
        model=model,
        model_config=model_config,
        step=step,
        best_validation_l1_m=best_validation_l1_m,
        run_manifest_sha256=run_sha256,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
    )
    payload["trainer_state"] = {
        "bad_validation_count": int(bad_validation_count),
        "history": history,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--train-split", type=Path, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError("training output must stay inside this worktree") from error
    dataset_root = args.dataset_root.resolve()
    train_split = args.train_split.resolve()
    model_config_path = args.model_config.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not train_split.is_file():
        raise FileNotFoundError(train_split)
    model_config = json.loads(model_config_path.read_text())
    training = dict(model_config.get("training", {}))
    if training.get("loss") != "meter_l1_distance":
        raise ValueError("the localization recipe requires meter_l1_distance")
    if training.get("coordinate_target") != "source_xyz_minus_receiver_xyz_meters":
        raise ValueError("the localization recipe requires meter-space relative xyz")
    if training.get("sampler") != "room_balanced":
        raise ValueError("the localization recipe requires room_balanced sampling")
    if training.get("augmentation") != "none":
        raise ValueError("the frozen localization recipe does not use augmentation")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed = int(args.seed)
    num_workers = int(
        training.get("num_workers", 0) if args.num_workers is None else args.num_workers
    )
    max_steps = int(
        training["max_steps"] if args.max_steps is None else args.max_steps
    )
    batch_size = int(training["batch_size"])
    validation_every = int(training["validation_every_steps"])
    warmup_steps = int(training["warmup_steps"])
    patience = int(training["early_stopping_patience"])
    if min(num_workers, max_steps, batch_size, validation_every, patience) < 0:
        raise ValueError("training dimensions cannot be negative")
    if min(max_steps, batch_size, validation_every, patience) == 0:
        raise ValueError("training dimensions must be positive")
    learning_rate = float(training["learning_rate"])
    minimum_learning_rate = float(training["minimum_learning_rate"])
    if not 0.0 < minimum_learning_rate <= learning_rate:
        raise ValueError("learning rates must satisfy 0 < minimum <= initial")
    precision = str(training.get("precision", "32"))
    if precision not in ("32", "bf16-mixed"):
        raise ValueError("precision must be '32' or 'bf16-mixed'")
    if (
        device.type == "cuda"
        and precision == "bf16-mixed"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the requested CUDA device does not support BF16")

    training_rooms, validation_rooms = split_localizer_rooms(
        train_split,
        validation_fraction=float(training["validation_room_fraction"]),
        seed=seed,
    )
    partition = {
        "method": "numpy.default_rng.PCG64_room_disjoint",
        "seed": seed,
        "source_split": str(train_split),
        "training_rooms": [list(room) for room in training_rooms],
        "validation_rooms": [list(room) for room in validation_rooms],
    }
    partition["sha256"] = canonical_sha256(partition)
    identity = {
        "task": "gt_rir_relative_coordinate_localizer",
        "dataset_root": str(dataset_root),
        "train_split": str(train_split),
        "train_split_sha256": file_sha256(train_split),
        "model_config": str(model_config_path),
        "model_config_sha256": file_sha256(model_config_path),
        "room_partition_sha256": partition["sha256"],
        "seed": seed,
        "num_workers": num_workers,
        "max_steps": max_steps,
    }
    run_manifest = initialize_run(output_dir, identity)
    _atomic_json(output_dir / "room_partition.json", partition)
    run_sha256 = str(run_manifest["sha256"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_options = {
        "dataset_root": dataset_root,
        "split_path": train_split,
        "sample_rate": int(model_config["sample_rate"]),
        "sample_size": int(model_config["sample_size"]),
    }
    train_dataset = AcousticRoomsRIRLocalizationDataset(
        **dataset_options, included_rooms=training_rooms
    )
    validation_dataset = AcousticRoomsRIRLocalizationDataset(
        **dataset_options, included_rooms=validation_rooms
    )
    transform = localizer_transform_from_config(model_config).eval().to(device)
    last_checkpoint_path = output_dir / "last.pt"
    resume_bundle = None
    if last_checkpoint_path.is_file():
        resume_bundle = torch.load(
            last_checkpoint_path, map_location="cpu", weights_only=False
        )
        if resume_bundle.get("schema_version") != RIR_LOCALIZER_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("last checkpoint has an unsupported schema")
        if resume_bundle.get("run_manifest_sha256") != run_sha256:
            raise RuntimeError("last checkpoint belongs to a different run")
        if resume_bundle.get("model_config") != model_config:
            raise RuntimeError("last checkpoint model config mismatch")
    model = build_rir_localizer(model_config).to(device)
    betas = tuple(float(value) for value in training["optimizer_betas"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=float(training["optimizer_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(
            step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            minimum_ratio=minimum_learning_rate / learning_rate,
        ),
    )
    step = 0
    best_validation_l1_m = math.inf
    bad_validation_count = 0
    history: list[dict[str, Any]] = []
    if resume_bundle is not None:
        model.load_state_dict(resume_bundle["state_dict"], strict=True)
        optimizer.load_state_dict(resume_bundle["optimizer_state_dict"])
        scheduler.load_state_dict(resume_bundle["scheduler_state_dict"])
        step = int(resume_bundle["step"])
        best_validation_l1_m = float(resume_bundle["best_validation_l1_m"])
        trainer_state = dict(resume_bundle.get("trainer_state", {}))
        bad_validation_count = int(trainer_state.get("bad_validation_count", 0))
        history = list(trainer_state.get("history", []))

    sampler = RoomBalancedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        seed=seed,
        start_step=step,
        stop_step=max_steps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        **_loader_options(num_workers, device.type == "cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_options(num_workers, device.type == "cuda"),
    )
    train_iterator = iter(train_loader)
    use_bfloat16 = device.type == "cuda" and precision == "bf16-mixed"
    started = time.perf_counter()

    while step < max_steps and bad_validation_count < patience:
        batch = next(train_iterator)
        model.train()
        waveforms = batch["waveform"].to(device, non_blocking=True)
        target = batch["relative_source"].to(device, non_blocking=True)
        with torch.no_grad():
            log_spectrograms = transform(waveforms)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16,
        ):
            prediction = model(log_spectrograms)
            loss = meter_l1_distance_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        step += 1

        if step % validation_every != 0 and step != max_steps:
            continue
        validation_metrics = validate(
            model,
            transform,
            validation_loader,
            device=device,
            use_bfloat16=use_bfloat16,
        )
        entry = {
            "step": step,
            "train_meter_l1_distance_m": float(loss.detach()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
            "validation": validation_metrics,
        }
        history.append(entry)
        current = float(validation_metrics["mean_l1_distance_m"])
        improved = current < best_validation_l1_m
        if improved:
            best_validation_l1_m = current
            bad_validation_count = 0
        else:
            bad_validation_count += 1
        payload = _checkpoint(
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            best_validation_l1_m=best_validation_l1_m,
            run_sha256=run_sha256,
            bad_validation_count=bad_validation_count,
            history=history,
        )
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
        _atomic_torch_save(last_checkpoint_path, payload)
        _atomic_json(output_dir / "history.json", {"validations": history})
        print(json.dumps(entry, sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "run_manifest_sha256": run_sha256,
        "completed_steps": step,
        "requested_max_steps": max_steps,
        "stopped_early": bad_validation_count >= patience,
        "best_validation_l1_m": best_validation_l1_m,
        "training_rir_count": len(train_dataset),
        "validation_rir_count": len(validation_dataset),
        "training_room_count": len(training_rooms),
        "validation_room_count": len(validation_rooms),
        "best_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
