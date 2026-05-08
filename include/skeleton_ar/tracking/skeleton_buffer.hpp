#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/trt/rtmpose_estimator.hpp"

namespace skeleton_ar::tracking {

/// Per-track sliding window of skeleton observations.
///
/// Stores the last `buffer_size` keypoint sets for one tracked person and
/// produces normalised (3, T, V) tensors ready for the action classifier.
/// Low-confidence joints are forward-filled from the previous frame, or
/// zeroed if no prior observation is available.
class SkeletonBuffer {
public:
    explicit SkeletonBuffer(const config::TrackingConfig& cfg, std::uint32_t num_keypoints,
                            std::uint32_t window_frames);

    /// Push the most recent observation. Caller is expected to have
    /// already mapped keypoints back to original-image coordinates.
    void push(const trt::KeypointSet& kps);

    /// Mark this frame as missed. Skeleton buffer is NOT advanced; this
    /// just bumps the missed-frames counter for eviction logic.
    void mark_missed();

    /// True when at least `window_frames` observations are present.
    bool ready() const noexcept { return frames_.size() >= window_frames_; }

    /// True when the track should be evicted by the owner.
    bool should_evict() const noexcept { return missed_ > cfg_.max_missed_frames; }

    /// Build a normalised `(3, T, V)` row-major float tensor for the
    /// action classifier, returning std::nullopt when not yet ready.
    /// Coordinates are translated so that the spine root is at the
    /// origin, then scaled to roughly [-1, 1] using the largest joint
    /// distance in the buffer.
    std::optional<std::vector<float>> as_tensor() const;

    std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(frames_.size()); }
    std::uint32_t window_frames() const noexcept { return window_frames_; }
    std::uint32_t missed() const noexcept { return missed_; }

private:
    config::TrackingConfig cfg_;
    std::uint32_t num_keypoints_;
    std::uint32_t window_frames_;
    std::deque<trt::KeypointSet> frames_;
    std::uint32_t missed_ = 0;

    void maybe_pop_oldest_();
};

}  // namespace skeleton_ar::tracking
