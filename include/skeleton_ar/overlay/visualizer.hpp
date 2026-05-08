#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "skeleton_ar/pipeline/frame_meta.hpp"

namespace skeleton_ar::overlay {

/// Draw bounding boxes, skeleton edges, and action labels on top of a
/// frame.
///
/// COCO-17 topology: this draws the standard 19-edge skeleton that pairs
/// shoulders, elbows, wrists, hips, knees, and ankles; head edges are
/// drawn from nose to eyes/ears.
class Visualizer {
public:
    explicit Visualizer(const std::vector<std::string>& class_labels);

    /// Render in-place onto `frame`.
    void render(cv::Mat& frame, const pipeline::FrameResult& result) const;

private:
    std::vector<std::string> labels_;
};

}  // namespace skeleton_ar::overlay
