"""PyTorch Lightning wrapper around the ST-GCN classifier."""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn as nn

from .graph import partition_adjacency
from .stgcn import STGCN


class STGCNLightning(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        num_keypoints: int = 17,
        hidden_channels: tuple[int, ...] = (64, 64, 128, 128, 256, 256),
        lr: float = 0.05,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        epochs: int = 80,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        adjacency = torch.from_numpy(partition_adjacency(num_keypoints)).float()
        self.model = STGCN(
            in_channels=in_channels,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            adjacency=adjacency,
            hidden_channels=hidden_channels,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        return self._step(batch, "val")

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )

        epochs = self.hparams.epochs
        warmup = self.hparams.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [scheduler]
