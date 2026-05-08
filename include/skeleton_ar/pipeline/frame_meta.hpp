#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

#include "skeleton_ar/trt/rtmpose_estimator.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"

namespace skeleton_ar::pipeline {

struct DetectedTrack {
    std::uint64_t track_id = 0;
    cv::Rect2f bbox;
    float detection_score = 0.0f;
    trt::KeypointSet keypoints;
    std::optional<trt::ActionPrediction> action;
};

struct FrameResult {
    std::uint64_t frame_number = 0;
    std::int64_t pts_ns = 0;
    std::vector<DetectedTrack> tracks;
};

}  // namespace skeleton_ar::pipeline
