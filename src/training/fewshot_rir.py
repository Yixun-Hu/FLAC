"""Lightning training loop for the AcousticRooms FewshotRiR adaptation."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from src.baselines.fewshot_rir import FewshotRiRLoss


class FewshotRiRTrainingWrapper(pl.LightningModule):
    def __init__(self, model, training_config: dict) -> None:
        super().__init__()
        self.model = model
        if "context_counts" in training_config:
            raise ValueError(
                "variable context_counts is not part of upstream Few-ShotRIR training"
            )
        self.context_count = int(training_config.get("context_count", 8))
        if self.context_count != 8:
            raise ValueError("the upstream-aligned AcousticRooms training setting is fixed K=8")
        self.learning_rate = float(training_config.get("learning_rate", 1e-4))
        self.optimizer_epsilon = float(training_config.get("optimizer_epsilon", 1e-5))
        self.optimizer_betas = tuple(
            float(value) for value in training_config.get("optimizer_betas", (0.9, 0.999))
        )
        if len(self.optimizer_betas) != 2:
            raise ValueError("optimizer_betas must contain two values")
        self.losses = FewshotRiRLoss(**training_config.get("loss", {}))

    @staticmethod
    def prepare_batch(
        batch: dict[str, torch.Tensor],
        *,
        context_count: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        required = (
            "context_depth",
            "context_magnitude",
            "context_poses",
            "target_magnitude",
            "query_poses",
            "context_mask",
            "query_mask",
        )
        if not isinstance(batch, dict) or any(key not in batch for key in required):
            raise ValueError(f"FewshotRiR batches must contain {required}")
        available = int(batch["context_depth"].shape[1])
        context_count = int(context_count)
        if context_count <= 0 or context_count > available:
            raise ValueError("context_count is outside the available ordered contexts")
        inputs = {
            "context_depth": batch["context_depth"][:, :context_count].float(),
            "context_spectrograms": batch["context_magnitude"][:, :context_count].float(),
            "context_poses": batch["context_poses"][:, :context_count].float(),
            "query_poses": batch["query_poses"].float(),
            "context_mask": batch["context_mask"][:, :context_count].bool(),
            "query_mask": batch["query_mask"].bool(),
        }
        return inputs, batch["target_magnitude"].float(), batch["query_mask"].bool()

    def _shared_step(self, batch, context_count: int) -> dict[str, torch.Tensor]:
        inputs, target, query_mask = self.prepare_batch(batch, context_count=context_count)
        raw_prediction = self.model(**inputs)
        return self.losses(raw_prediction, target, query_mask)

    def training_step(self, batch, batch_idx):
        values = self._shared_step(batch, self.context_count)
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {f"train/{key}": value for key, value in values.items()},
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
        return values["loss"]

    def validation_step(self, batch, batch_idx):
        values = self._shared_step(batch, self.context_count)
        reconstruction_loss = values["spectral_l1"]
        if getattr(self, "_trainer", None) is not None:
            self.log_dict(
                {f"val/k{self.context_count}/{key}": value for key, value in values.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
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
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.optimizer_betas,
            eps=self.optimizer_epsilon,
        )
