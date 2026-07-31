#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"

namespace skeleton_ar::trt {

class TrtEngine;

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
};

/// One person's pose: 17 COCO keypoints in original-image coordinates.
using KeypointSet = std::vector<Keypoint>;

/// RTMPose-m wrapper.
///
/// Top-down: the caller provides bounding boxes (e.g., from YOLOv8) and
/// this class crops, normalises, and runs pose estimation in a single
/// batched TRT call. Output is mapped back to the original image frame.
///
/// Reference:
///   Jiang et al., "RTMPose: Real-Time Multi-Person Pose Estimation
///   based on MMPose", arXiv:2303.07399, 2023.
class RTMPoseEstimator {
public:
    explicit RTMPoseEstimator(const config::PoseConfig& cfg);
    ~RTMPoseEstimator();

    RTMPoseEstimator(const RTMPoseEstimator&) = delete;
    RTMPoseEstimator& operator=(const RTMPoseEstimator&) = delete;

    /// Estimate a pose for each bbox. The returned vector has the same
    /// size as `bboxes`; an entry may be empty if cropping fails.
    std::vector<KeypointSet> estimate(const cv::Mat& image, const std::vector<cv::Rect2f>& bboxes);

    const config::PoseConfig& config() const noexcept { return cfg_; }

private:
    config::PoseConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> input_scratch_;
    std::vector<float> output_scratch_;

    /// Per-pose preprocessing scratch (cropped + resized BGR).
    cv::Mat crop_scratch_;

    void run_chunk_(const cv::Mat& image, const std::vector<cv::Rect2f>& bboxes, std::size_t offset,
                    std::size_t count, std::vector<KeypointSet>& out);
};

}  // namespace skeleton_ar::trt
