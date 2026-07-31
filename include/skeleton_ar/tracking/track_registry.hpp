#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/tracking/skeleton_buffer.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"

namespace skeleton_ar::tracking {

struct TrackedPerson {
    std::uint64_t track_id = 0;
    cv::Rect2f bbox;
    SkeletonBuffer buffer;
    std::uint32_t since_last_classified = 0;
    std::optional<trt::ActionPrediction> latest_action;

    TrackedPerson(std::uint64_t id, const config::TrackingConfig& cfg, std::uint32_t num_keypoints,
                  std::uint32_t window_frames)
        : track_id(id), buffer(cfg, num_keypoints, window_frames) {}
};

/// Owns the pool of currently-active tracks and their skeleton buffers.
/// The DeepStream tracker assigns track IDs; we maintain a parallel map
/// from those IDs to skeleton state and the most recent action label.
class TrackRegistry {
public:
    TrackRegistry(const config::TrackingConfig& cfg, std::uint32_t num_keypoints,
                  std::uint32_t window_frames);

    TrackedPerson& upsert(std::uint64_t track_id, const cv::Rect2f& bbox);

    /// Mark every track that did not receive an upsert this frame as
    /// missed; evict any that have crossed `max_missed_frames`.
    void age_unobserved();

    /// Reset the per-frame "observed" flags ahead of the next frame.
    void mark_all_unobserved();

    std::vector<TrackedPerson*> ready_tracks();

    std::vector<TrackedPerson*> all_tracks();

private:
    config::TrackingConfig cfg_;
    std::uint32_t num_keypoints_;
    std::uint32_t window_frames_;
    std::unordered_map<std::uint64_t, TrackedPerson> tracks_;
    std::unordered_map<std::uint64_t, bool> observed_this_frame_;
};

}  // namespace skeleton_ar::tracking
