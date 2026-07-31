#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"

namespace skeleton_ar::trt {

class TrtEngine;

struct PersonDetection {
    cv::Rect2f bbox;
    float score = 0.0f;
};

/// YOLOv8 person detector wrapper.
///
/// The bundled engine is filtered to COCO class id 0 ("person") only.
/// Output decoding follows the standard YOLOv8 layout where the model
/// emits a single `(B, num_classes + 4, 8400)` tensor; bbox values are
/// already in pixel space relative to the (letterboxed) input.
///
/// Reference:
///   Jocher et al., "YOLOv8 by Ultralytics", 2023.
class YOLOv8Detector {
public:
    explicit YOLOv8Detector(const config::DetectionConfig& cfg);
    ~YOLOv8Detector();

    YOLOv8Detector(const YOLOv8Detector&) = delete;
    YOLOv8Detector& operator=(const YOLOv8Detector&) = delete;

    std::vector<PersonDetection> detect(const cv::Mat& image);

    const config::DetectionConfig& config() const noexcept { return cfg_; }

private:
    config::DetectionConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> input_scratch_;
};

}  // namespace skeleton_ar::trt
