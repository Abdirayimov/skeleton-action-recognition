#pragma once

#include <Eigen/Core>

#include <memory>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"

namespace skeleton_ar::trt {

class TrtEngine;

struct ActionPrediction {
    std::int32_t class_id = -1;
    float confidence = 0.0f;
};

/// ST-GCN action classifier.
///
/// Input is a `(C=3, T=window_frames, V=num_keypoints, M=1)` skeleton
/// tensor (the 4D form used in the original ST-GCN; M=1 for our single-
/// person windows). Output is a length-`num_classes` softmax over the
/// action vocabulary in `configs/labels_ntu60_subset.txt`.
///
/// Reference:
///   Yan et al., "Spatial Temporal Graph Convolutional Networks for
///   Skeleton-Based Action Recognition", AAAI 2018.
class STGCNClassifier {
public:
    explicit STGCNClassifier(const config::ActionConfig& cfg);
    ~STGCNClassifier();

    STGCNClassifier(const STGCNClassifier&) = delete;
    STGCNClassifier& operator=(const STGCNClassifier&) = delete;

    /// Classify a single (C, T, V, M=1) skeleton tensor. The input is the
    /// row-major flatten of that tensor, length C*T*V (M=1 collapsed).
    ActionPrediction classify(const std::vector<float>& skeleton);

    /// Classify many skeletons at once. The input matrix is shaped
    /// (B, C*T*V) so that each row is one example.
    std::vector<ActionPrediction> classify_batch(const Eigen::MatrixXf& skeletons);

    const config::ActionConfig& config() const noexcept { return cfg_; }

private:
    config::ActionConfig cfg_;
    std::unique_ptr<TrtEngine> engine_;
    std::vector<float> output_scratch_;
};

}  // namespace skeleton_ar::trt
