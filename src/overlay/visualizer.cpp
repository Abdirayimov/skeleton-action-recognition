#include "skeleton_ar/overlay/visualizer.hpp"

#include <array>
#include <opencv2/imgproc.hpp>
#include <string>
#include <utility>
#include <vector>

namespace skeleton_ar::overlay {

namespace {

// COCO-17 keypoint indices.
//   0: nose          1: left_eye       2: right_eye
//   3: left_ear      4: right_ear
//   5: left_shoulder 6: right_shoulder
//   7: left_elbow    8: right_elbow
//   9: left_wrist   10: right_wrist
//  11: left_hip     12: right_hip
//  13: left_knee    14: right_knee
//  15: left_ankle   16: right_ankle
constexpr std::array<std::pair<int, int>, 19> kEdges{{
    {0, 1},  {0, 2},   {1, 3},   {2, 4},   {5, 6},   {5, 7},   {7, 9}, {6, 8}, {8, 10}, {5, 11},
    {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {0, 5}, {0, 6}, {3, 4},
}};

cv::Scalar color_for_track(std::uint64_t id) {
    // Cycle through a small fixed palette; consistent per track id.
    static const cv::Scalar palette[] = {
        cv::Scalar(58, 184, 255),   // amber
        cv::Scalar(99, 211, 142),   // green
        cv::Scalar(244, 173, 66),   // orange
        cv::Scalar(120, 105, 245),  // violet
        cv::Scalar(72, 207, 235),   // cyan
        cv::Scalar(255, 95, 128),   // pink
    };
    return palette[id % (sizeof(palette) / sizeof(palette[0]))];
}

}  // namespace

Visualizer::Visualizer(const std::vector<std::string>& class_labels) : labels_(class_labels) {}

void Visualizer::render(cv::Mat& frame, const pipeline::FrameResult& result) const {
    for (const auto& t : result.tracks) {
        const cv::Scalar color = color_for_track(t.track_id);

        cv::rectangle(frame, t.bbox, color, 2);

        // Skeleton edges
        if (t.keypoints.size() >= 17) {
            for (const auto& [a, b] : kEdges) {
                if (t.keypoints[a].score < 0.2f || t.keypoints[b].score < 0.2f)
                    continue;
                cv::line(frame,
                         cv::Point(static_cast<int>(t.keypoints[a].x),
                                   static_cast<int>(t.keypoints[a].y)),
                         cv::Point(static_cast<int>(t.keypoints[b].x),
                                   static_cast<int>(t.keypoints[b].y)),
                         color, 2);
            }
            for (const auto& kp : t.keypoints) {
                if (kp.score < 0.2f)
                    continue;
                cv::circle(frame, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 3,
                           color, -1);
            }
        }

        // Action label
        std::string label = "id " + std::to_string(t.track_id);
        if (t.action) {
            const auto idx = static_cast<std::size_t>(t.action->class_id);
            if (idx < labels_.size()) {
                label += "  " + labels_[idx];
            } else {
                label += "  cls" + std::to_string(t.action->class_id);
            }
            char conf[16];
            std::snprintf(conf, sizeof(conf), " %.2f", static_cast<double>(t.action->confidence));
            label += conf;
        }

        const cv::Point origin(static_cast<int>(t.bbox.x),
                               std::max(0, static_cast<int>(t.bbox.y) - 8));
        cv::putText(frame, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

}  // namespace skeleton_ar::overlay
