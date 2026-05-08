#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${ROOT}/models/onnx"
ENG_DIR="${ROOT}/models/engines"
mkdir -p "${ENG_DIR}"

if [[ -f "${ONNX_DIR}/yolov8s.onnx" ]]; then
    echo "building YOLOv8s engine..."
    trtexec \
        --onnx="${ONNX_DIR}/yolov8s.onnx" \
        --saveEngine="${ENG_DIR}/yolov8s_person_fp16.engine" \
        --fp16 \
        --workspace=4096
fi

if [[ -f "${ONNX_DIR}/rtmpose_m.onnx" ]]; then
    echo "building RTMPose-m engine..."
    trtexec \
        --onnx="${ONNX_DIR}/rtmpose_m.onnx" \
        --saveEngine="${ENG_DIR}/rtmpose_m_fp16.engine" \
        --fp16 \
        --minShapes=input:1x3x256x192 \
        --optShapes=input:8x3x256x192 \
        --maxShapes=input:16x3x256x192 \
        --workspace=4096
fi

if [[ -f "${ONNX_DIR}/stgcn_ntu60.onnx" ]]; then
    echo "building ST-GCN engine..."
    trtexec \
        --onnx="${ONNX_DIR}/stgcn_ntu60.onnx" \
        --saveEngine="${ENG_DIR}/stgcn_ntu60_fp16.engine" \
        --fp16 \
        --minShapes=input:1x3x30x17x1 \
        --optShapes=input:4x3x30x17x1 \
        --maxShapes=input:16x3x30x17x1 \
        --workspace=4096
fi

echo "engines:"
ls -lh "${ENG_DIR}/"
