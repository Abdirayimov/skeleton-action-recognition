#!/usr/bin/env bash
# Convenience wrapper for the offline video CLI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:?usage: infer_video.sh INPUT_VIDEO OUTPUT_VIDEO}"
OUTPUT="${2:?usage: infer_video.sh INPUT_VIDEO OUTPUT_VIDEO}"

"${ROOT}/build/skeleton_ar_video" \
    --config "${ROOT}/configs/system_config.yaml" \
    --labels "${ROOT}/configs/labels_ntu60_subset.txt" \
    --input "${INPUT}" \
    --output "${OUTPUT}"
