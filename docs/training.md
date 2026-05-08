# Training the action model

The C++ runtime expects an ST-GCN ONNX export. The bundled
`training/` package builds one from NTU-RGBD-60.

## End-to-end recipe

```bash
cd training
pip install -e .[dev]

# 1. Prepare the 10-class subset (see prepare_ntu60.py for the label
#    table). Raw NTU-60 .skeleton files are not redistributed here.
python -m skeleton_ar_train.prepare_ntu60 \
    --raw-dir /path/to/ntu_rgbd_raw \
    --out-dir data/processed

# 2. Train. ~4 hours on a single RTX 3090.
python -m skeleton_ar_train.train --config configs/train.yaml

# 3. Export the best checkpoint.
python -m skeleton_ar_train.export_onnx \
    --ckpt outputs/checkpoints/stgcn-best.ckpt \
    --output ../models/onnx/stgcn_ntu60.onnx
```

Then build the engine and run the C++ binary as usual.

## Why the 10-class subset?

NTU-RGBD-60 has 60 action classes; many of them are pairwise
interactions (M = 2) or hand-only motions that the COCO-17 topology
cannot represent well (e.g., touching nose with finger). The 10-class
subset chosen in `prepare_ntu60.py` is single-person, full-body, and
distinctive enough to demonstrate the pipeline working end-to-end on a
single GPU in a few hours. To train the full vocabulary, use
[ST-GCN](https://github.com/yysijie/st-gcn) or
[pyskl](https://github.com/kennymckormick/pyskl) - both provide
better preprocessing than the minimal recipe here.

## Adapting to your own labels

1. Replace the data preparation script with one that emits a `.npz`
   archive of the same shape: `data` shaped `(N, 3, T, V, 1)`,
   `labels` shaped `(N,)`.
2. Update `configs/labels_ntu60_subset.txt` with your label table.
3. Set `action.num_classes` in `configs/system_config.yaml` to match.
4. Retrain and re-export.

The C++ side does not need any code changes; it reads the class count
from the engine's output binding shape.
