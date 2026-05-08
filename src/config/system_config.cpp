#include "skeleton_ar/config/system_config.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace skeleton_ar::config {

namespace {

template <typename T>
T require(const YAML::Node& node, const std::string& key) {
    if (!node[key]) {
        throw std::runtime_error("missing required config key: " + key);
    }
    return node[key].as<T>();
}

template <typename T>
T optional(const YAML::Node& node, const std::string& key, T fallback) {
    return node[key] ? node[key].as<T>() : fallback;
}

std::vector<std::string> read_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    if (!f.is_open()) return labels;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        labels.push_back(line);
    }
    return labels;
}

}  // namespace

SystemConfig SystemConfig::load(const std::string& yaml_path, const std::string& labels_path) {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    SystemConfig out;

    if (const auto p = root["pipeline"]; p) {
        out.pipeline.muxer_width = optional<std::uint32_t>(p, "muxer_width", 1280);
        out.pipeline.muxer_height = optional<std::uint32_t>(p, "muxer_height", 720);
        out.pipeline.batch_size = optional<std::uint32_t>(p, "batch_size", 1);
        out.pipeline.batched_push_timeout_us =
            optional<std::uint32_t>(p, "batched_push_timeout_us", 40000);
        out.pipeline.emit_overlay = optional<bool>(p, "emit_overlay", true);
    }

    if (const auto d = root["detection"]; d) {
        out.detection.engine_path = require<std::string>(d, "engine_path");
        out.detection.input_width = optional<std::uint32_t>(d, "input_width", 640);
        out.detection.input_height = optional<std::uint32_t>(d, "input_height", 640);
        out.detection.confidence_threshold = optional<float>(d, "confidence_threshold", 0.4f);
        out.detection.nms_iou_threshold = optional<float>(d, "nms_iou_threshold", 0.5f);
        out.detection.person_class_id = optional<std::int32_t>(d, "person_class_id", 0);
    } else {
        throw std::runtime_error("missing 'detection' section in config");
    }

    if (const auto p = root["pose"]; p) {
        out.pose.engine_path = require<std::string>(p, "engine_path");
        out.pose.input_width = optional<std::uint32_t>(p, "input_width", 192);
        out.pose.input_height = optional<std::uint32_t>(p, "input_height", 256);
        out.pose.num_keypoints = optional<std::uint32_t>(p, "num_keypoints", 17);
        out.pose.batch_size = optional<std::uint32_t>(p, "batch_size", 16);
    } else {
        throw std::runtime_error("missing 'pose' section in config");
    }

    if (const auto a = root["action"]; a) {
        out.action.engine_path = require<std::string>(a, "engine_path");
        out.action.num_classes = optional<std::uint32_t>(a, "num_classes", 10);
        out.action.window_frames = optional<std::uint32_t>(a, "window_frames", 30);
        out.action.step_frames = optional<std::uint32_t>(a, "step_frames", 10);
        out.action.num_keypoints = optional<std::uint32_t>(a, "num_keypoints", 17);
    } else {
        throw std::runtime_error("missing 'action' section in config");
    }

    if (const auto t = root["tracking"]; t) {
        out.tracking.buffer_size = optional<std::uint32_t>(t, "buffer_size", 30);
        out.tracking.max_missed_frames = optional<std::uint32_t>(t, "max_missed_frames", 15);
        out.tracking.min_keypoint_confidence =
            optional<float>(t, "min_keypoint_confidence", 0.3f);
    }

    if (const auto l = root["logging"]; l) {
        out.logging.level = optional<std::string>(l, "level", "info");
        out.logging.json = optional<bool>(l, "json", true);
    }

    if (!labels_path.empty()) {
        out.labels = read_labels(labels_path);
    }
    if (out.labels.empty()) {
        // Fall back to the file shipped alongside the default config.
        out.labels = read_labels("configs/labels_ntu60_subset.txt");
    }

    return out;
}

}  // namespace skeleton_ar::config
