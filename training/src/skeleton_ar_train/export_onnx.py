"""Export a trained ST-GCN checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .lightning_module import STGCNLightning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/onnx/stgcn_ntu60.onnx"))
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--num-keypoints", type=int, default=17)
    args = parser.parse_args()

    model = STGCNLightning.load_from_checkpoint(str(args.ckpt), strict=False)
    model.eval()

    dummy = torch.randn(1, 3, args.window_frames, args.num_keypoints, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
