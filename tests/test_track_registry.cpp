#include <gtest/gtest.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>

#include "skeleton_ar/tracking/track_registry.hpp"

namespace {

using skeleton_ar::config::TrackingConfig;
using skeleton_ar::tracking::TrackedPerson;
using skeleton_ar::tracking::TrackRegistry;
using skeleton_ar::trt::Keypoint;
using skeleton_ar::trt::KeypointSet;

constexpr std::uint32_t kJoints = 17;
constexpr std::uint32_t kWindow = 3;

TrackingConfig tracking(std::uint32_t max_missed = 2) {
    TrackingConfig cfg;
    cfg.max_missed_frames = max_missed;
    cfg.min_keypoint_confidence = 0.3f;
    return cfg;
}

KeypointSet pose() {
    KeypointSet kps(kJoints);
    for (auto& kp : kps)
        kp = Keypoint{10.0f, 20.0f, 0.9f};
    return kps;
}

cv::Rect2f box(float x) {
    return cv::Rect2f(x, 100.0f, 40.0f, 100.0f);
}

bool contains_id(const std::vector<TrackedPerson*>& tracks, std::uint64_t id) {
    return std::any_of(tracks.begin(), tracks.end(),
                       [id](const TrackedPerson* t) { return t->track_id == id; });
}

// --------------------------------------------------------------- upsert

TEST(TrackRegistry, StartsEmpty) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    EXPECT_TRUE(r.all_tracks().empty());
    EXPECT_TRUE(r.ready_tracks().empty());
}

TEST(TrackRegistry, CreatesATrackOnFirstSight) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    auto& t = r.upsert(7, box(100.0f));

    EXPECT_EQ(t.track_id, 7u);
    EXPECT_FLOAT_EQ(t.bbox.x, 100.0f);
    EXPECT_EQ(r.all_tracks().size(), 1u);
}

TEST(TrackRegistry, ReturnsTheSameTrackForTheSameId) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    auto& first = r.upsert(7, box(100.0f));
    first.buffer.push(pose());
    auto& second = r.upsert(7, box(150.0f));

    EXPECT_EQ(&first, &second) << "upsert must not replace the skeleton buffer";
    EXPECT_EQ(second.buffer.size(), 1u);
    EXPECT_FLOAT_EQ(second.bbox.x, 150.0f) << "but the box should be refreshed";
    EXPECT_EQ(r.all_tracks().size(), 1u);
}

TEST(TrackRegistry, KeepsDistinctIdsApart) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    r.upsert(1, box(100.0f));
    r.upsert(2, box(600.0f));

    EXPECT_EQ(r.all_tracks().size(), 2u);
    EXPECT_TRUE(contains_id(r.all_tracks(), 1));
    EXPECT_TRUE(contains_id(r.all_tracks(), 2));
}

TEST(TrackRegistry, GivesEachTrackItsOwnBuffer) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    r.upsert(1, box(100.0f)).buffer.push(pose());
    r.upsert(2, box(600.0f));

    for (auto* t : r.all_tracks()) {
        EXPECT_EQ(t->buffer.size(), t->track_id == 1 ? 1u : 0u);
    }
}

// -------------------------------------------------------------- ageing

TEST(TrackRegistry, KeepsATrackThatWasObservedThisFrame) {
    TrackRegistry r{tracking(/*max_missed=*/1), kJoints, kWindow};

    r.mark_all_unobserved();
    r.upsert(7, box(100.0f));
    r.age_unobserved();

    EXPECT_EQ(r.all_tracks().size(), 1u);
}

TEST(TrackRegistry, EvictsATrackOnceItPassesTheMissedFrameBudget) {
    TrackRegistry r{tracking(/*max_missed=*/2), kJoints, kWindow};
    r.upsert(7, box(100.0f));

    for (int frame = 0; frame < 2; ++frame) {
        r.mark_all_unobserved();
        r.age_unobserved();
    }
    EXPECT_EQ(r.all_tracks().size(), 1u) << "still inside the budget";

    r.mark_all_unobserved();
    r.age_unobserved();
    EXPECT_TRUE(r.all_tracks().empty());
}

TEST(TrackRegistry, EvictsOnlyTheTrackThatWentMissing) {
    TrackRegistry r{tracking(/*max_missed=*/1), kJoints, kWindow};
    r.upsert(1, box(100.0f));
    r.upsert(2, box(600.0f));

    for (int frame = 0; frame < 3; ++frame) {
        r.mark_all_unobserved();
        r.upsert(1, box(100.0f));  // only track 1 keeps being seen
        r.age_unobserved();
    }

    ASSERT_EQ(r.all_tracks().size(), 1u);
    EXPECT_EQ(r.all_tracks().front()->track_id, 1u);
}

TEST(TrackRegistry, ReacquiringATrackResetsItsMissedCount) {
    TrackRegistry r{tracking(/*max_missed=*/2), kJoints, kWindow};
    r.upsert(7, box(100.0f));

    r.mark_all_unobserved();
    r.age_unobserved();
    r.mark_all_unobserved();
    r.upsert(7, box(120.0f));  // seen again
    r.upsert(7, box(120.0f)).buffer.push(pose());
    r.age_unobserved();

    for (int frame = 0; frame < 2; ++frame) {
        r.mark_all_unobserved();
        r.age_unobserved();
    }

    EXPECT_EQ(r.all_tracks().size(), 1u) << "the counter should have restarted";
}

TEST(TrackRegistry, AgeUnobservedIsSafeOnAnEmptyRegistry) {
    TrackRegistry r{tracking(), kJoints, kWindow};

    EXPECT_NO_THROW(r.mark_all_unobserved());
    EXPECT_NO_THROW(r.age_unobserved());
}

TEST(TrackRegistry, AnUnseenTrackIsNotEvictedBeforeTheFirstMarkAllUnobserved) {
    // A track created and immediately aged in the same frame is still
    // flagged observed, so it survives.
    TrackRegistry r{tracking(/*max_missed=*/1), kJoints, kWindow};
    r.upsert(7, box(100.0f));

    r.age_unobserved();

    EXPECT_EQ(r.all_tracks().size(), 1u);
}

// ---------------------------------------------------------- readiness

TEST(TrackRegistry, ReportsNoReadyTracksUntilABufferFills) {
    TrackRegistry r{tracking(), kJoints, kWindow};
    auto& t = r.upsert(7, box(100.0f));

    t.buffer.push(pose());
    t.buffer.push(pose());
    EXPECT_TRUE(r.ready_tracks().empty());

    t.buffer.push(pose());
    EXPECT_EQ(r.ready_tracks().size(), 1u);
}

TEST(TrackRegistry, ReportsOnlyTheTracksWithAFullClip) {
    TrackRegistry r{tracking(), kJoints, kWindow};
    auto& ready = r.upsert(1, box(100.0f));
    r.upsert(2, box(600.0f));

    for (std::uint32_t i = 0; i < kWindow; ++i)
        ready.buffer.push(pose());

    const auto tracks = r.ready_tracks();
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks.front()->track_id, 1u);
}

TEST(TrackRegistry, ExposesTheLatestActionSlotPerTrack) {
    TrackRegistry r{tracking(), kJoints, kWindow};
    auto& t = r.upsert(7, box(100.0f));

    EXPECT_FALSE(t.latest_action.has_value());
    EXPECT_EQ(t.since_last_classified, 0u);

    t.latest_action = skeleton_ar::trt::ActionPrediction{3, 0.87f};

    ASSERT_TRUE(r.all_tracks().front()->latest_action.has_value());
    EXPECT_EQ(r.all_tracks().front()->latest_action->class_id, 3);
    EXPECT_FLOAT_EQ(r.all_tracks().front()->latest_action->confidence, 0.87f);
}

}  // namespace
