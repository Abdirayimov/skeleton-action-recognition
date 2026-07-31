#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace skeleton_ar::config {

struct PipelineConfig {
    std::uint32_t muxer_width = 1280;
    std::uint32_t muxer_height = 720;
    std::uint32_t batch_size = 1;
    std::uint32_t batched_push_timeout_us = 40000;
    bool emit_overlay = true;
};

struct DetectionConfig {
    std::string engine_path;
    std::uint32_t input_width = 640;
    std::uint32_t input_height = 640;
    float confidence_threshold = 0.4f;
    float nms_iou_threshold = 0.5f;
    std::int32_t person_class_id = 0;
};

struct PoseConfig {
    std::string engine_path;
    std::uint32_t input_width = 192;
    std::uint32_t input_height = 256;
    std::uint32_t num_keypoints = 17;
    std::uint32_t batch_size = 16;
};

struct ActionConfig {
    std::string engine_path;
    std::uint32_t num_classes = 10;
    std::uint32_t window_frames = 30;
    std::uint32_t step_frames = 10;
    std::uint32_t num_keypoints = 17;
};

struct TrackingConfig {
    std::uint32_t buffer_size = 30;
    std::uint32_t max_missed_frames = 15;
    float min_keypoint_confidence = 0.3f;
};

struct LoggingConfig {
    std::string level = "info";
    bool json = true;
};

struct SystemConfig {
    PipelineConfig pipeline;
    DetectionConfig detection;
    PoseConfig pose;
    ActionConfig action;
    TrackingConfig tracking;
    LoggingConfig logging;

    /// Class label table; index N corresponds to class id N. Filled in by
    /// `load` from `labels_ntu60_subset.txt` when present alongside the
    /// YAML config (or via the optional `labels_path` field).
    std::vector<std::string> labels;

    /// Load configuration from a YAML file. Throws std::runtime_error on
    /// parse failure or missing required fields.
    static SystemConfig load(const std::string& yaml_path, const std::string& labels_path = "");
};

}  // namespace skeleton_ar::config
