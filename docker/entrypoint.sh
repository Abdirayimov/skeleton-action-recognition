#!/usr/bin/env bash
set -euo pipefail

ENGINES_DIR="${ENGINES_DIR:-/app/models/engines}"
ONNX_DIR="${ONNX_DIR:-/app/models/onnx}"

build_engine() {
    local name="$1"
    local extra="${2:-}"
    if [[ ! -f "${ENGINES_DIR}/${name}_fp16.engine" && -f "${ONNX_DIR}/${name}.onnx" ]]; then
        echo "[entrypoint] building ${name} engine..."
        mkdir -p "${ENGINES_DIR}"
        # shellcheck disable=SC2086
        trtexec \
            --onnx="${ONNX_DIR}/${name}.onnx" \
            --saveEngine="${ENGINES_DIR}/${name}_fp16.engine" \
            --fp16 \
            --workspace=4096 \
            ${extra}
    fi
}

build_engine yolov8s_person
build_engine rtmpose_m "--minShapes=input:1x3x256x192 --optShapes=input:8x3x256x192 --maxShapes=input:16x3x256x192"
build_engine stgcn_ntu60 "--minShapes=input:1x3x30x17x1 --optShapes=input:4x3x30x17x1 --maxShapes=input:16x3x30x17x1"

exec skeleton_ar_video "$@"
