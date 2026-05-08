"""Prepare the 10-class subset of NTU-RGBD-60 used by this repository.

Steps:
  1. Read raw `.skeleton` files (NTU's text format).
  2. Keep only the 10 single-person, full-body actions:

         drink_water, eat_meal, brushing_teeth, brushing_hair,
         clapping, reading, writing, phone_call, hand_waving,
         sitting_down

  3. Map NTU's 25-joint body to COCO's 17 keypoints (we drop the
     extra hand / foot joints; for joints that have no direct COCO
     equivalent we duplicate the closest available).
  4. Pad / truncate every clip to `window_frames` frames.
  5. Apply the cross-subject (xsub) split published with NTU-RGBD.
  6. Save a single `.npz` archive per split.

This script is not the full official preprocessing pipeline; it is a
minimal recipe that lets a reader reproduce a working ST-GCN baseline
end-to-end. For published-paper accuracy use the upstream dataloaders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NTU_TO_COCO = {
    # COCO index -> NTU joint id (1-indexed in NTU; offset 1 here).
    0: 4,    # nose       <- head
    1: 4,    # left_eye   <- head (no eye in NTU; duplicate)
    2: 4,    # right_eye  <- head
    3: 4,    # left_ear   <- head
    4: 4,    # right_ear  <- head
    5: 5,    # left_shoulder
    6: 9,    # right_shoulder
    7: 6,    # left_elbow
    8: 10,   # right_elbow
    9: 7,    # left_wrist
    10: 11,  # right_wrist
    11: 13,  # left_hip
    12: 17,  # right_hip
    13: 14,  # left_knee
    14: 18,  # right_knee
    15: 15,  # left_ankle
    16: 19,  # right_ankle
}

NTU60_LABEL_TO_CLASS = {
    1: 0,   # drink water
    2: 1,   # eat meal
    3: 2,   # brushing teeth
    4: 3,   # brushing hair
    10: 4,  # clapping
    11: 5,  # reading
    12: 6,  # writing
    28: 7,  # phone call
    23: 8,  # hand waving
    8: 9,   # sitting down
}

XSUB_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38,
}


def parse_skeleton_file(path: Path) -> np.ndarray | None:
    """Return a (T, 25, 3) tensor for the first body in `path`, or None."""
    with path.open() as f:
        n_frames = int(f.readline().strip())
        out = np.zeros((n_frames, 25, 3), dtype=np.float32)
        for t in range(n_frames):
            n_bodies = int(f.readline().strip())
            if n_bodies == 0:
                continue
            # Body header line (we ignore the body id).
            f.readline()
            n_joints = int(f.readline().strip())
            if n_joints != 25:
                return None
            for j in range(25):
                parts = f.readline().split()
                out[t, j, 0] = float(parts[0])
                out[t, j, 1] = float(parts[1])
                out[t, j, 2] = float(parts[2])
            for _ in range(n_bodies - 1):
                # Skip extra bodies for this 10-class single-person subset.
                f.readline()
                k = int(f.readline().strip())
                for _ in range(k):
                    f.readline()
    return out


def ntu_to_coco(seq: np.ndarray) -> np.ndarray:
    """Project (T, 25, 3) NTU joints to (T, 17, 3) COCO joints."""
    out = np.zeros((seq.shape[0], 17, 3), dtype=np.float32)
    for coco_idx, ntu_id in NTU_TO_COCO.items():
        out[:, coco_idx] = seq[:, ntu_id - 1]
    return out


def fix_temporal_length(seq: np.ndarray, target: int) -> np.ndarray:
    """Pad with zeros or center-crop to `target` frames."""
    t = seq.shape[0]
    if t == target:
        return seq
    if t > target:
        start = (t - target) // 2
        return seq[start : start + target]
    pad = np.zeros((target - t, *seq.shape[1:]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def to_chwm(seq: np.ndarray) -> np.ndarray:
    """(T, V, 3) -> (3, T, V, 1)."""
    return np.transpose(seq, (2, 0, 1))[..., None]


def filename_metadata(name: str) -> tuple[int, int]:
    # Naming convention: SsssCcccPpppRrrrAaaa.skeleton
    subject = int(name[9:12])
    action = int(name[17:20])
    return subject, action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=30)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = [], []
    val_x, val_y = [], []

    for path in sorted(args.raw_dir.glob("*.skeleton")):
        subject, action = filename_metadata(path.name)
        if action not in NTU60_LABEL_TO_CLASS:
            continue
        seq = parse_skeleton_file(path)
        if seq is None:
            continue
        coco = ntu_to_coco(seq)
        coco = fix_temporal_length(coco, args.window_frames)
        sample = to_chwm(coco)
        label = NTU60_LABEL_TO_CLASS[action]
        if subject in XSUB_TRAIN_SUBJECTS:
            train_x.append(sample)
            train_y.append(label)
        else:
            val_x.append(sample)
            val_y.append(label)

    np.savez(
        args.out_dir / "ntu60_xsub_train.npz",
        data=np.stack(train_x), labels=np.array(train_y),
    )
    np.savez(
        args.out_dir / "ntu60_xsub_val.npz",
        data=np.stack(val_x), labels=np.array(val_y),
    )
    print(f"wrote {len(train_x)} train, {len(val_x)} val samples")


if __name__ == "__main__":
    main()
