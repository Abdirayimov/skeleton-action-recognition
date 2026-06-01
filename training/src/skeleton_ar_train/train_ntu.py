"""Train the repo's ST-GCN on the public NTU-RGB+D 10-class subset.

This consumes the CTR-GCN-style preprocessed arrays (``x_train.npy`` /
``y_train.npy`` of shape ``(N, T=300, 150)`` where 150 = 2 persons x 25
joints x 3 coords, and one-hot labels), filters them to the ten
single-person classes the repo targets, resamples each clip to a fixed
length, and trains the ST-GCN defined in ``stgcn.py`` on the 25-joint
NTU topology.

Plain PyTorch (no Lightning dependency); the model and graph are the
repo's own. After training it exports the best checkpoint to ONNX and
dumps a handful of held-out test clips for the C++ visual demo.

Usage:
    PYTHONPATH=training/src python -m skeleton_ar_train.train_ntu \
        --data-dir /path/to/CTR-GCN/data/ntu120 \
        --out-dir training/outputs --epochs 25 --frames 64
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .graph import partition_adjacency
from .stgcn import STGCN

# NTU action id (1-based) -> contiguous class index, matching
# configs/labels_ntu60_subset.txt ordering.
NTU_ACTION_TO_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 11: 5, 12: 6, 28: 7, 23: 8, 8: 9}
CLASS_NAMES = [
    "drink_water", "eat_meal", "brushing_teeth", "brushing_hair", "clapping",
    "reading", "writing", "phone_call", "hand_waving", "sitting_down",
]
CLIP_MAGIC = 0x534B4C31  # 'SKL1'


def _to_ctrgcn_layout(x: np.ndarray) -> np.ndarray:
    """(N, T, 150) -> (N, C=3, T, V=25, M=2)."""
    n, t, _ = x.shape
    x = x.reshape(n, t, 2, 25, 3)
    return np.transpose(x, (0, 4, 1, 3, 2))  # N, C, T, V, M


def _valid_length(clip: np.ndarray) -> int:
    """Last frame index (along T) that holds any non-zero joint."""
    # clip: (C, T, V, M)
    energy = np.abs(clip).sum(axis=(0, 2, 3))  # (T,)
    nz = np.nonzero(energy > 1e-6)[0]
    return int(nz[-1]) + 1 if nz.size else clip.shape[1]


def _resample(clip: np.ndarray, frames: int) -> np.ndarray:
    """Temporally resample (C, T, V, M) to (C, frames, V, M)."""
    length = _valid_length(clip)
    idx = np.linspace(0, max(length - 1, 1), frames).round().astype(np.int64)
    idx = np.clip(idx, 0, clip.shape[1] - 1)
    return clip[:, idx, :, :]


def _load_split(data_dir: Path, split: str, frames: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(data_dir / f"x_{split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{split}.npy")
    labels_full = y.argmax(axis=1) if y.ndim > 1 else y  # 0-based action idx
    keep_mask = np.zeros(labels_full.shape[0], dtype=bool)
    remap = np.full(120, -1, dtype=np.int64)
    for action_id, cls in NTU_ACTION_TO_CLASS.items():
        remap[action_id - 1] = cls
    for i, a in enumerate(labels_full):
        if remap[a] >= 0:
            keep_mask[i] = True
    keep = np.nonzero(keep_mask)[0]

    # The ten target actions are single-person; keep person 0 only
    # (M=1), which matches the repo's STGCNClassifier and the live
    # single-person-per-track RTMPose path.
    xs = np.empty((keep.size, 3, frames, 25, 1), dtype=np.float32)
    ys = np.empty((keep.size,), dtype=np.int64)
    for out_i, src_i in enumerate(keep):
        clip = _to_ctrgcn_layout(x[src_i : src_i + 1].astype(np.float32))[0]
        xs[out_i] = _resample(clip, frames)[:, :, :, 0:1]
        ys[out_i] = remap[labels_full[src_i]]
    return xs, ys


def export_onnx(model: nn.Module, path: Path, frames: int) -> None:
    model.eval()
    dummy = torch.randn(1, 3, frames, 25, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["skeleton"], output_names=["logits"],
        dynamic_axes={"skeleton": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX -> {path}")


def export_demo_clips(x: np.ndarray, y: np.ndarray, path: Path, per_class: int = 1) -> None:
    """Dump a few clips per class to a flat binary the C++ demo reads."""
    chosen: list[int] = []
    for cls in range(len(CLASS_NAMES)):
        idx = np.nonzero(y == cls)[0]
        chosen.extend(idx[:per_class].tolist())
    clips = x[chosen]            # (K, 3, T, 25, 2)
    labels = y[chosen].astype(np.int32)
    k, c, t, v, m = clips.shape

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<6i", CLIP_MAGIC, k, c, t, v, m))
        f.write(np.ascontiguousarray(clips, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(labels, dtype=np.int32).tobytes())
    print(f"exported {k} demo clips -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  loading NTU 10-class subset from {args.data_dir} ...")

    x_tr, y_tr = _load_split(args.data_dir, "train", args.frames)
    x_te, y_te = _load_split(args.data_dir, "test", args.frames)
    print(f"train={x_tr.shape}  test={x_te.shape}  classes={len(CLASS_NAMES)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    adjacency = torch.from_numpy(partition_adjacency("ntu-rgb+d")).float()
    model = STGCN(in_channels=3, num_classes=len(CLASS_NAMES), num_keypoints=25,
                  adjacency=adjacency).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "stgcn_ntu10_best.pt"

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += int((pred == yb).sum())
                total += int(yb.numel())
        acc = correct / max(total, 1)
        print(f"epoch {epoch + 1:02d}/{args.epochs}  val_acc={acc:.4f}  "
              f"({time.time() - t0:.1f}s)")
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "val_acc": acc,
                        "frames": args.frames}, best_path)

    print(f"best val_acc={best_acc:.4f}  saved {best_path}")

    # Reload best and export artefacts.
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    export_onnx(model.cpu(), args.out_dir / "stgcn_ntu10.onnx", args.frames)
    export_demo_clips(x_te, y_te, args.out_dir / "demo_clips.bin", per_class=1)
    print(f"\nFINAL best_val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
