#include "skeleton_ar/pipeline/probe_chain.hpp"

#include <Eigen/Core>

#include <utility>

namespace skeleton_ar::pipeline {

ProbeChain::ProbeChain(trt::RTMPoseEstimator& pose,
                       trt::STGCNClassifier& action,
                       tracking::TrackRegistry& registry,
                       const config::ActionConfig& action_cfg)
    : pose_(pose), action_(action), registry_(registry), action_cfg_(action_cfg) {}

FrameResult ProbeChain::process(
    std::uint64_t frame_number, std::int64_t pts_ns, const cv::Mat& image,
    const std::vector<std::pair<std::uint64_t, cv::Rect2f>>& tracks) {
    FrameResult result;
    result.frame_number = frame_number;
    result.pts_ns = pts_ns;

    registry_.mark_all_unobserved();

    // 1. Run pose estimation across all tracker bboxes in one batched call.
    std::vector<cv::Rect2f> bboxes;
    bboxes.reserve(tracks.size());
    for (const auto& [tid, box] : tracks) {
        (void)tid;
        bboxes.push_back(box);
    }
    auto kps_per_track = pose_.estimate(image, bboxes);

    // 2. Push observations into per-track skeleton buffers.
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        auto& tp = registry_.upsert(tracks[i].first, tracks[i].second);
        tp.buffer.push(kps_per_track[i]);
        tp.since_last_classified += 1;
    }

    // 3. Age and evict tracks not seen this frame.
    registry_.age_unobserved();

    // 4. Re-classify any track whose buffer is full and whose step
    //    counter has rolled over.
    std::vector<tracking::TrackedPerson*> due;
    for (auto* tp : registry_.ready_tracks()) {
        if (tp->since_last_classified >= action_cfg_.step_frames) due.push_back(tp);
    }

    if (!due.empty()) {
        const auto T = static_cast<Eigen::Index>(action_cfg_.window_frames);
        const auto V = static_cast<Eigen::Index>(action_cfg_.num_keypoints);
        const Eigen::Index row_size = 3 * T * V;
        Eigen::MatrixXf batch(static_cast<Eigen::Index>(due.size()), row_size);
        std::vector<std::size_t> emitted;
        for (std::size_t i = 0; i < due.size(); ++i) {
            const auto tensor = due[i]->buffer.as_tensor();
            if (!tensor) continue;
            for (Eigen::Index j = 0; j < row_size; ++j) {
                batch(static_cast<Eigen::Index>(emitted.size()), j) = (*tensor)[static_cast<std::size_t>(j)];
            }
            emitted.push_back(i);
        }
        if (!emitted.empty()) {
            Eigen::MatrixXf trimmed = batch.topRows(static_cast<Eigen::Index>(emitted.size()));
            const auto preds = action_.classify_batch(trimmed);
            for (std::size_t k = 0; k < emitted.size(); ++k) {
                auto* tp = due[emitted[k]];
                tp->latest_action = preds[k];
                tp->since_last_classified = 0;
            }
        }
    }

    // 5. Build the per-frame result snapshot.
    for (auto* tp : registry_.all_tracks()) {
        DetectedTrack dt;
        dt.track_id = tp->track_id;
        dt.bbox = tp->bbox;
        if (!tp->buffer.size() == 0u) {
            // Most recently pushed observation, if any.
            // (skeleton_buffer holds frames internally; we have no public
            // accessor for the most recent frame, so re-store it in
            // FrameResult via the latest pose result instead.)
        }
        dt.action = tp->latest_action;
        result.tracks.push_back(std::move(dt));
    }
    // The last-frame keypoints we just pushed are also useful for the
    // overlay; carry them in `result.tracks` in input order.
    for (std::size_t i = 0; i < tracks.size() && i < result.tracks.size(); ++i) {
        result.tracks[i].keypoints = kps_per_track[i];
    }

    if (callback_) callback_(result);
    return result;
}

}  // namespace skeleton_ar::pipeline
