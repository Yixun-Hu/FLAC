"""Lightning training wrapper for the direct-waveform Few-ShotRIR baseline."""

from __future__ import annotations

import random

import pytorch_lightning as pl
import torch

from src.baselines.few_shot_rir_waveform import FewShotRIRWaveformLoss


class FewShotRIRWaveformTrainingWrapper(pl.LightningModule):
    def __init__(self, model, training_config: dict) -> None:
        super().__init__()
        self.model = model
        counts = tuple(int(value) for value in training_config.get("context_counts", range(1, 9)))
        if not counts or any(value <= 0 or value > 8 for value in counts):
            raise ValueError("context_counts must be a nonempty subset of [1, 8]")
        if 1 not in counts or 8 not in counts:
            raise ValueError("context_counts must include the primary K=1 and K=8 settings")
        self.context_counts = counts
        self.learning_rate = float(training_config.get("learning_rate", 1e-4))
        self.weight_decay = float(training_config.get("weight_decay", 1e-3))
        self.losses = FewShotRIRWaveformLoss(**training_config.get("loss", {}))

    def draw_context_count(self) -> int:
        return int(random.choice(self.context_counts))

    def prepare_batch(
        self,
        batch,
        *,
        context_count: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        target, metadata = batch
        if not torch.is_tensor(target) or target.ndim != 3 or target.shape[1] != 1:
            raise ValueError("target batch must have shape [B, 1, T]")
        if not isinstance(metadata, (tuple, list)) or len(metadata) != target.shape[0]:
            raise ValueError("metadata must contain one dictionary per target waveform")
        required = ("depth", "context_audio", "context_poses", "source", "padding_mask")
        if any(any(key not in item for key in required) for item in metadata):
            raise ValueError(f"every metadata item must contain {required}")
        geometry = torch.stack([torch.as_tensor(item["depth"]).float() for item in metadata])
        context_audio = torch.stack(
            [torch.as_tensor(item["context_audio"]).float() for item in metadata]
        )
        context_coordinates = torch.stack(
            [torch.as_tensor(item["context_poses"]).float() for item in metadata]
        )
        context_count = int(context_count)
        if context_count <= 0 or context_count > context_audio.shape[1]:
            raise ValueError("context_count is outside the available ordered contexts")
        source = torch.stack([torch.as_tensor(item["source"]).float() for item in metadata])
        padding_mask = torch.stack(
            [torch.as_tensor(item["padding_mask"]).bool() for item in metadata]
        )
        inputs = {
            "geometry": geometry,
            "context_audio": context_audio,
            "context_coordinates": context_coordinates,
            "query_source": source,
            "query_receiver": torch.zeros_like(source),
            "context_mask": torch.ones(
                target.shape[0], context_audio.shape[1], dtype=torch.bool, device=target.device
            )
            & (
                torch.arange(context_audio.shape[1], device=target.device).unsqueeze(0)
                < context_count
            ),
        }
        return inputs, target.float(), padding_mask

    def _shared_step(self, batch, context_count: int) -> dict[str, torch.Tensor]:
        inputs, target, padding_mask = self.prepare_batch(
            batch, context_count=context_count
        )
        device = target.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prediction = self.model(**inputs)
        return self.losses(prediction, target, padding_mask=padding_mask.to(device))

    def training_step(self, batch, batch_idx):
        values = self._shared_step(batch, self.draw_context_count())
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {f"train/{key}": value for key, value in values.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        return values["loss"]

    def validation_step(self, batch, batch_idx):
        by_count = {
            context_count: self._shared_step(batch, context_count)
            for context_count in (1, 8)
        }
        reconstruction_loss = torch.stack(
            [values["loss"] for values in by_count.values()]
        ).mean()
        if getattr(self, "_trainer", None) is not None:
            for context_count, values in by_count.items():
                self.log_dict(
                    {
                        f"val/k{context_count}/{key}": value
                        for key, value in values.items()
                    },
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
            self.log(
                "val/reconstruction_loss",
                reconstruction_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return reconstruction_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
