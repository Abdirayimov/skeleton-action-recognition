"""NTU-RGBD-60 dataset loader for the 10-class subset used by skeleton_ar.

The expected on-disk format is a single .npz archive produced by
`prepare_ntu60.py`:

    data:   float32, shape (N, 3, T, V, M)   - skeleton sequences
    labels: int64,   shape (N,)              - action class ids in [0..9]

`T` is the canonical clip length (zero-padded to 30 frames), `V` is 17
(COCO topology after the NTU->COCO mapping), `M` is the maximum number
of persons per clip (we keep 1 for this subset).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NTU60Subset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, npz_path: str | Path, augment: bool = False) -> None:
        super().__init__()
        archive = np.load(npz_path)
        self.x = archive["data"].astype(np.float32)
        self.y = archive["labels"].astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[idx]
        if self.augment:
            sample = self._augment(sample)
        return torch.from_numpy(sample), torch.tensor(self.y[idx])

    @staticmethod
    def _augment(sample: np.ndarray) -> np.ndarray:
        # Light augmentations:
        #   * random Gaussian noise on (x, y) coords
        #   * random scale in [0.9, 1.1]
        scale = np.random.uniform(0.9, 1.1)
        sample[:2] = sample[:2] * scale
        sample[:2] = sample[:2] + np.random.normal(0, 0.005, sample[:2].shape).astype(np.float32)
        return sample


def make_loaders(
    train_npz: str | Path,
    val_npz: str | Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train = NTU60Subset(train_npz, augment=True)
    val = NTU60Subset(val_npz, augment=False)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
