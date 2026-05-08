#pragma once

#include <functional>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/pipeline/frame_meta.hpp"
#include "skeleton_ar/tracking/track_registry.hpp"
#include "skeleton_ar/trt/rtmpose_estimator.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"

namespace skeleton_ar::pipeline {

using ResultCallback = std::function<void(const FrameResult&)>;

/// Connects pose estimation, skeleton buffering, and action
/// classification. Plugged into the DeepStream pipeline as a src-pad
/// probe that fires once per frame after nvtracker.
class ProbeChain {
public:
    ProbeChain(trt::RTMPoseEstimator& pose,
               trt::STGCNClassifier& action,
               tracking::TrackRegistry& registry,
               const config::ActionConfig& action_cfg);

    /// Process a single frame given the raw image plus the tracker's
    /// (track_id, bbox) pairs from this frame.
    FrameResult process(std::uint64_t frame_number, std::int64_t pts_ns,
                        const cv::Mat& image,
                        const std::vector<std::pair<std::uint64_t, cv::Rect2f>>& tracks);

    void set_result_callback(ResultCallback cb) { callback_ = std::move(cb); }

private:
    trt::RTMPoseEstimator& pose_;
    trt::STGCNClassifier& action_;
    tracking::TrackRegistry& registry_;
    config::ActionConfig action_cfg_;
    ResultCallback callback_;
};

}  // namespace skeleton_ar::pipeline
