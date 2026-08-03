#include "skeleton_ar/tracking/skeleton_buffer.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace skeleton_ar::tracking {

SkeletonBuffer::SkeletonBuffer(const config::TrackingConfig& cfg, std::uint32_t num_keypoints,
                               std::uint32_t window_frames)
    : cfg_(cfg), num_keypoints_(num_keypoints), window_frames_(window_frames) {}

void SkeletonBuffer::push(const trt::KeypointSet& kps) {
    missed_ = 0;

    // Forward-fill low-confidence joints from the previous frame so the
    // action classifier sees a consistent shape rather than (0,0) noise.
    if (kps.size() != num_keypoints_) {
        // Defensive: if upstream gave the wrong shape, drop the frame.
        return;
    }

    trt::KeypointSet filled = kps;
    if (!frames_.empty()) {
        const auto& prev = frames_.back();
        for (std::size_t j = 0; j < filled.size(); ++j) {
            if (filled[j].score < cfg_.min_keypoint_confidence) {
                filled[j] = prev[j];
            }
        }
    } else {
        for (auto& kp : filled) {
            if (kp.score < cfg_.min_keypoint_confidence) {
                kp.x = 0.0f;
                kp.y = 0.0f;
            }
        }
    }

    frames_.push_back(std::move(filled));
    maybe_pop_oldest_();
}

void SkeletonBuffer::mark_missed() {
    ++missed_;
}

void SkeletonBuffer::maybe_pop_oldest_() {
    while (frames_.size() > window_frames_) {
        frames_.pop_front();
    }
}

std::optional<std::vector<float>> SkeletonBuffer::as_tensor() const {
    if (!ready())
        return std::nullopt;

    const auto T = window_frames_;
    const auto V = num_keypoints_;
    std::vector<float> tensor(static_cast<std::size_t>(3) * T * V, 0.0f);

    // Use the most recent T frames if the buffer somehow grew larger.
    const std::size_t start = frames_.size() - T;

    // Step 1: copy raw (x, y, score) into (3, T, V) layout.
    for (std::uint32_t t = 0; t < T; ++t) {
        const auto& kps = frames_[start + t];
        for (std::uint32_t v = 0; v < V; ++v) {
            tensor[(0 * T + t) * V + v] = kps[v].x;
            tensor[(1 * T + t) * V + v] = kps[v].y;
            tensor[(2 * T + t) * V + v] = kps[v].score;
        }
    }

    // Step 2: translate so that the per-clip mean of joints 5/6/11/12
    // (shoulders + hips, the body centroid for COCO-17) sits at the
    // origin.
    constexpr std::array<std::uint32_t, 4> kCentroidJoints{5, 6, 11, 12};
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    std::size_t count = 0;
    for (std::uint32_t t = 0; t < T; ++t) {
        for (auto v : kCentroidJoints) {
            if (v >= V)
                continue;
            mean_x += tensor[(0 * T + t) * V + v];
            mean_y += tensor[(1 * T + t) * V + v];
            ++count;
        }
    }
    if (count > 0) {
        mean_x /= static_cast<float>(count);
        mean_y /= static_cast<float>(count);
        for (std::uint32_t t = 0; t < T; ++t) {
            for (std::uint32_t v = 0; v < V; ++v) {
                tensor[(0 * T + t) * V + v] -= mean_x;
                tensor[(1 * T + t) * V + v] -= mean_y;
            }
        }
    }

    // Step 3: scale so the largest joint distance from origin is ~1.
    float max_norm = 0.0f;
    for (std::uint32_t t = 0; t < T; ++t) {
        for (std::uint32_t v = 0; v < V; ++v) {
            const float x = tensor[(0 * T + t) * V + v];
            const float y = tensor[(1 * T + t) * V + v];
            const float n = std::sqrt(x * x + y * y);
            if (n > max_norm)
                max_norm = n;
        }
    }
    if (max_norm > 1e-6f) {
        const float inv = 1.0f / max_norm;
        for (std::uint32_t t = 0; t < T; ++t) {
            for (std::uint32_t v = 0; v < V; ++v) {
                tensor[(0 * T + t) * V + v] *= inv;
                tensor[(1 * T + t) * V + v] *= inv;
            }
        }
    }

    return tensor;
}

}  // namespace skeleton_ar::tracking
