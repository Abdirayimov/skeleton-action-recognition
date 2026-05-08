"""Training entry point: `python -m skeleton_ar_train.train --config train.yaml`."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .data import make_loaders
from .lightning_module import STGCNLightning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader = make_loaders(
        train_npz=cfg["data"]["train_npz"],
        val_npz=cfg["data"]["val_npz"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = STGCNLightning(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        num_keypoints=cfg["model"]["num_keypoints"],
        hidden_channels=tuple(cfg["model"]["hidden_channels"]),
        lr=cfg["optim"]["lr"],
        momentum=cfg["optim"]["momentum"],
        weight_decay=cfg["optim"]["weight_decay"],
        epochs=cfg["optim"]["epochs"],
        warmup_epochs=cfg["optim"]["warmup_epochs"],
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, monitor="val/acc", mode="max",
            filename="stgcn-{epoch:02d}-{val/acc:.3f}",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=cfg["paths"]["log_dir"], name="stgcn")

    trainer = L.Trainer(
        max_epochs=cfg["optim"]["epochs"],
        precision=cfg["trainer"]["precision"],
        gradient_clip_val=cfg["trainer"]["gradient_clip_val"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
