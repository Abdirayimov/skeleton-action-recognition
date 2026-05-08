# skeleton_ar_train

PyTorch Lightning training pipeline for the ST-GCN engine consumed by
the C++ runtime.

## Quick start

```bash
cd training
pip install -e .[dev]

# 1. Prepare the 10-class NTU-60 subset (see prepare_ntu60.py for the
#    exact label mapping). NTU-RGBD raw .skeleton files must already be
#    on disk; they are not redistributed here.
python -m skeleton_ar_train.prepare_ntu60 \
    --raw-dir /path/to/ntu_rgbd_raw \
    --out-dir data/processed

# 2. Train (single GPU, ~4 hours on RTX 3090).
python -m skeleton_ar_train.train --config configs/train.yaml

# 3. Export the best checkpoint to ONNX for the C++ runtime.
python -m skeleton_ar_train.export_onnx \
    --ckpt outputs/checkpoints/stgcn-best.ckpt \
    --output ../models/onnx/stgcn_ntu60.onnx
```

## What's here

- `graph.py` - COCO-17 partitioned adjacency matrices following Yan et al.
- `stgcn.py` - Pure-PyTorch ST-GCN implementation (no `mmcv` dependency).
- `lightning_module.py` - LR warmup + cosine schedule, val accuracy logged.
- `data.py` - .npz dataset loader with light augmentations.
- `prepare_ntu60.py` - Raw NTU `.skeleton` -> 10-class .npz pipeline.
- `train.py` - YAML-driven entry point.
- `export_onnx.py` - Checkpoint -> ONNX with dynamic batch axis.

## Notes

- The 10-class subset is intentionally small so that a working model
  converges in a few hours on a single GPU. For published-paper
  accuracy on the full 60-class problem, use the original
  [ST-GCN](https://github.com/yysijie/st-gcn) or
  [pyskl](https://github.com/kennymckormick/pyskl) repos.
- The COCO-17 mapping in `prepare_ntu60.py` is intentionally simple
  (duplicating head joints rather than estimating them); accuracy is
  good enough to demonstrate the C++ pipeline works end-to-end but
  could be improved with a proper face-keypoint detector.
- Inference at runtime takes COCO-17 keypoints from RTMPose, so this
  preprocessing must produce the same topology.
