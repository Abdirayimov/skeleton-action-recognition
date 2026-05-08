#include "skeleton_ar/tracking/track_registry.hpp"

#include <vector>

namespace skeleton_ar::tracking {

TrackRegistry::TrackRegistry(const config::TrackingConfig& cfg, std::uint32_t num_keypoints,
                             std::uint32_t window_frames)
    : cfg_(cfg), num_keypoints_(num_keypoints), window_frames_(window_frames) {}

TrackedPerson& TrackRegistry::upsert(std::uint64_t track_id, const cv::Rect2f& bbox) {
    auto it = tracks_.find(track_id);
    if (it == tracks_.end()) {
        it = tracks_
                 .emplace(std::piecewise_construct, std::forward_as_tuple(track_id),
                          std::forward_as_tuple(track_id, cfg_, num_keypoints_, window_frames_))
                 .first;
    }
    it->second.bbox = bbox;
    observed_this_frame_[track_id] = true;
    return it->second;
}

void TrackRegistry::age_unobserved() {
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        const bool observed = observed_this_frame_[it->first];
        if (!observed) {
            it->second.buffer.mark_missed();
        }
        if (it->second.buffer.should_evict()) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

void TrackRegistry::mark_all_unobserved() {
    for (auto& [k, _] : observed_this_frame_) observed_this_frame_[k] = false;
    for (const auto& [k, _] : tracks_) {
        if (observed_this_frame_.find(k) == observed_this_frame_.end()) {
            observed_this_frame_[k] = false;
        }
    }
}

std::vector<TrackedPerson*> TrackRegistry::ready_tracks() {
    std::vector<TrackedPerson*> out;
    out.reserve(tracks_.size());
    for (auto& [_, t] : tracks_) {
        if (t.buffer.ready()) out.push_back(&t);
    }
    return out;
}

std::vector<TrackedPerson*> TrackRegistry::all_tracks() {
    std::vector<TrackedPerson*> out;
    out.reserve(tracks_.size());
    for (auto& [_, t] : tracks_) out.push_back(&t);
    return out;
}

}  // namespace skeleton_ar::tracking
