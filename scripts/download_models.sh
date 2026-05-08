#!/usr/bin/env bash
# Convenience wrapper around the public ONNX exports we need.
#
#   yolov8s.onnx     - Ultralytics YOLOv8s, person class only
#   rtmpose_m.onnx   - mmpose RTMPose-m, COCO 17-keypoint topology
#   stgcn_ntu60.onnx - ST-GCN trained by training/ scripts on NTU-60 subset
#
# The first two are downloaded from public release pages; the third is
# produced locally by training/scripts/export_onnx.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/models/onnx"
mkdir -p "${DEST}"

YOLO_URL="${YOLO_URL:-https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt}"

# YOLOv8 ships as a .pt; users typically convert via:
#   yolo export model=yolov8s.pt format=onnx imgsz=640
#
# We do not bundle the conversion here because it requires PyTorch +
# ultralytics; install those in a venv and run the one-liner above to
# produce yolov8s.onnx, then drop it into models/onnx/.

cat <<EOF
This script intentionally only prints instructions for the three model
checkpoints that skeleton_ar needs. The exact URLs change with each
upstream release, and licensing means we do not redistribute weights.

  1. YOLOv8s
     pip install ultralytics
     yolo export model=yolov8s.pt format=onnx imgsz=640
     mv yolov8s.onnx ${DEST}/

  2. RTMPose-m (COCO 17 keypoints)
     See https://github.com/open-mmlab/mmpose for the latest release.
     Place the export at ${DEST}/rtmpose_m.onnx.

  3. ST-GCN (NTU-60 10-class subset)
     cd training && python -m skeleton_ar_train.export_onnx
     The script writes ${DEST}/stgcn_ntu60.onnx.

After all three ONNX files are present, run scripts/build_engines.sh to
compile FP16 TensorRT engines.
EOF
