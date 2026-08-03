#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "skeleton_ar/tracking/skeleton_buffer.hpp"

namespace {

using skeleton_ar::config::TrackingConfig;
using skeleton_ar::tracking::SkeletonBuffer;
using skeleton_ar::trt::Keypoint;
using skeleton_ar::trt::KeypointSet;

constexpr std::uint32_t kJoints = 17;
constexpr std::uint32_t kWindow = 4;

/// The COCO-17 joints the normaliser uses as the body centroid.
constexpr std::uint32_t kCentroid[] = {5, 6, 11, 12};

TrackingConfig tracking(float min_conf = 0.3f, std::uint32_t max_missed = 15) {
    TrackingConfig cfg;
    cfg.min_keypoint_confidence = min_conf;
    cfg.max_missed_frames = max_missed;
    return cfg;
}

/// Every joint at the same point with the same confidence.
KeypointSet uniform_pose(float x, float y, float score = 0.9f, std::uint32_t joints = kJoints) {
    KeypointSet kps(joints);
    for (auto& kp : kps) {
        kp.x = x;
        kp.y = y;
        kp.score = score;
    }
    return kps;
}

/// Index into the (3, T, V) row-major tensor the buffer produces.
float at(const std::vector<float>& tensor, std::uint32_t c, std::uint32_t t, std::uint32_t v,
         std::uint32_t T = kWindow, std::uint32_t V = kJoints) {
    return tensor[(c * T + t) * V + v];
}

// ------------------------------------------------------------- readiness

TEST(SkeletonBuffer, StartsEmptyAndNotReady) {
    const SkeletonBuffer b{tracking(), kJoints, kWindow};

    EXPECT_EQ(b.size(), 0u);
    EXPECT_FALSE(b.ready());
    EXPECT_EQ(b.window_frames(), kWindow);
    EXPECT_FALSE(b.as_tensor().has_value());
}

TEST(SkeletonBuffer, BecomesReadyOnlyAfterAFullWindow) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};

    for (std::uint32_t i = 0; i < kWindow - 1; ++i) {
        b.push(uniform_pose(10.0f, 10.0f));
        EXPECT_FALSE(b.ready()) << "after " << (i + 1) << " frames";
    }

    b.push(uniform_pose(10.0f, 10.0f));
    EXPECT_TRUE(b.ready());
    EXPECT_TRUE(b.as_tensor().has_value());
}

TEST(SkeletonBuffer, NeverGrowsBeyondTheWindow) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};

    for (int i = 0; i < 20; ++i)
        b.push(uniform_pose(10.0f, 10.0f));

    EXPECT_EQ(b.size(), kWindow);
}

TEST(SkeletonBuffer, DropsAFrameWithTheWrongJointCount) {
    // Upstream handing over a differently-shaped pose is a bug, but the
    // buffer must not mix shapes into one clip.
    SkeletonBuffer b{tracking(), kJoints, kWindow};

    b.push(uniform_pose(10.0f, 10.0f, 0.9f, /*joints=*/13));

    EXPECT_EQ(b.size(), 0u);
}

// -------------------------------------------------------------- eviction

TEST(SkeletonBuffer, DoesNotAskToBeEvictedWhileObserved) {
    SkeletonBuffer b{tracking(0.3f, /*max_missed=*/3), kJoints, kWindow};

    b.push(uniform_pose(10.0f, 10.0f));

    EXPECT_EQ(b.missed(), 0u);
    EXPECT_FALSE(b.should_evict());
}

TEST(SkeletonBuffer, CountsMissedFramesWithoutAdvancingTheClip) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    b.push(uniform_pose(10.0f, 10.0f));

    b.mark_missed();
    b.mark_missed();

    EXPECT_EQ(b.missed(), 2u);
    EXPECT_EQ(b.size(), 1u) << "a missed frame must not push a duplicate observation";
}

TEST(SkeletonBuffer, AsksToBeEvictedPastTheMissedFrameBudget) {
    SkeletonBuffer b{tracking(0.3f, /*max_missed=*/3), kJoints, kWindow};
    b.push(uniform_pose(10.0f, 10.0f));

    for (int i = 0; i < 3; ++i)
        b.mark_missed();
    EXPECT_FALSE(b.should_evict()) << "exactly at the budget is still alive";

    b.mark_missed();
    EXPECT_TRUE(b.should_evict());
}

TEST(SkeletonBuffer, AnObservationResetsTheMissedCounter) {
    SkeletonBuffer b{tracking(0.3f, /*max_missed=*/3), kJoints, kWindow};
    b.push(uniform_pose(10.0f, 10.0f));
    b.mark_missed();
    b.mark_missed();

    b.push(uniform_pose(11.0f, 11.0f));

    EXPECT_EQ(b.missed(), 0u);
    EXPECT_FALSE(b.should_evict());
}

// --------------------------------------------------------- forward fill

TEST(SkeletonBuffer, ZeroesLowConfidenceJointsOnTheFirstFrame) {
    // Nothing to fill from yet, so an unreliable joint has to become the
    // origin rather than whatever the detector guessed.
    SkeletonBuffer b{tracking(/*min_conf=*/0.5f), kJoints, kWindow};

    KeypointSet kps = uniform_pose(50.0f, 60.0f, 0.9f);
    kps[3].x = 999.0f;
    kps[3].y = 999.0f;
    kps[3].score = 0.1f;  // below the threshold
    b.push(kps);

    for (std::uint32_t i = 0; i < kWindow; ++i)
        b.push(uniform_pose(50.0f, 60.0f, 0.9f));

    // The zeroed frame has fallen out of the window by now, but the push
    // above must not have thrown or corrupted anything.
    EXPECT_TRUE(b.ready());
}

TEST(SkeletonBuffer, ForwardFillsALowConfidenceJointFromThePreviousFrame) {
    SkeletonBuffer b{tracking(/*min_conf=*/0.5f), kJoints, /*window=*/2};

    b.push(uniform_pose(10.0f, 20.0f, 0.9f));

    // Second frame: joint 0 is unreliable and reports nonsense.
    KeypointSet kps = uniform_pose(10.0f, 20.0f, 0.9f);
    kps[0].x = 999.0f;
    kps[0].y = -999.0f;
    kps[0].score = 0.1f;
    b.push(kps);

    const auto tensor = b.as_tensor();
    ASSERT_TRUE(tensor.has_value());

    // Every joint in both frames is the same point, so after centring the
    // whole tensor collapses to the origin. A leaked 999 would not.
    for (std::uint32_t t = 0; t < 2; ++t) {
        for (std::uint32_t v = 0; v < kJoints; ++v) {
            EXPECT_NEAR(at(*tensor, 0, t, v, 2), 0.0f, 1e-5f) << "x at t=" << t << " v=" << v;
            EXPECT_NEAR(at(*tensor, 1, t, v, 2), 0.0f, 1e-5f) << "y at t=" << t << " v=" << v;
        }
    }
}

TEST(SkeletonBuffer, KeepsAJointExactlyAtTheConfidenceThreshold) {
    // The comparison is strictly less-than, so a joint sitting on the
    // threshold is trusted.
    SkeletonBuffer b{tracking(/*min_conf=*/0.5f), kJoints, /*window=*/1};

    KeypointSet kps = uniform_pose(10.0f, 20.0f, 0.5f);
    b.push(kps);

    EXPECT_EQ(b.size(), 1u);
    EXPECT_TRUE(b.as_tensor().has_value());
}

// ------------------------------------------------------- the tensor

TEST(SkeletonBuffer, ProducesATensorOfTheDocumentedShape) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    for (std::uint32_t i = 0; i < kWindow; ++i)
        b.push(uniform_pose(10.0f, 20.0f));

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    EXPECT_EQ(tensor->size(), static_cast<std::size_t>(3) * kWindow * kJoints);
}

TEST(SkeletonBuffer, CarriesConfidenceThroughOnTheThirdChannel) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    for (std::uint32_t i = 0; i < kWindow; ++i)
        b.push(uniform_pose(10.0f, 20.0f, 0.8f));

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    for (std::uint32_t t = 0; t < kWindow; ++t) {
        EXPECT_FLOAT_EQ(at(*tensor, 2, t, 0), 0.8f) << "t=" << t;
    }
}

TEST(SkeletonBuffer, TranslatesTheBodyCentroidToTheOrigin) {
    // Every joint sits at the same place, so the centroid is that place
    // and the whole clip should end up centred on zero.
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    for (std::uint32_t i = 0; i < kWindow; ++i)
        b.push(uniform_pose(640.0f, 360.0f));

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    for (std::uint32_t v : kCentroid) {
        EXPECT_NEAR(at(*tensor, 0, 0, v), 0.0f, 1e-4f) << "joint " << v;
        EXPECT_NEAR(at(*tensor, 1, 0, v), 0.0f, 1e-4f) << "joint " << v;
    }
}

TEST(SkeletonBuffer, IsInvariantToWhereThePersonStandsInTheFrame) {
    // The same pose at two very different image positions must normalise
    // to the same tensor -- that is the entire point of centring.
    SkeletonBuffer near_left{tracking(), kJoints, kWindow};
    SkeletonBuffer far_right{tracking(), kJoints, kWindow};

    for (std::uint32_t t = 0; t < kWindow; ++t) {
        KeypointSet a(kJoints);
        KeypointSet b(kJoints);
        for (std::uint32_t v = 0; v < kJoints; ++v) {
            const float dx = static_cast<float>(v) * 3.0f;
            const float dy = static_cast<float>(t) * 2.0f;
            a[v] = Keypoint{100.0f + dx, 100.0f + dy, 0.9f};
            b[v] = Keypoint{900.0f + dx, 500.0f + dy, 0.9f};
        }
        near_left.push(a);
        far_right.push(b);
    }

    const auto ta = near_left.as_tensor();
    const auto tb = far_right.as_tensor();

    ASSERT_TRUE(ta.has_value());
    ASSERT_TRUE(tb.has_value());
    for (std::size_t i = 0; i < ta->size(); ++i) {
        EXPECT_NEAR((*ta)[i], (*tb)[i], 1e-4f) << "element " << i;
    }
}

TEST(SkeletonBuffer, ScalesTheLargestJointOffsetToOne) {
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    for (std::uint32_t t = 0; t < kWindow; ++t) {
        KeypointSet kps(kJoints);
        for (std::uint32_t v = 0; v < kJoints; ++v) {
            kps[v] = Keypoint{100.0f + static_cast<float>(v) * 10.0f, 200.0f, 0.9f};
        }
        b.push(kps);
    }

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    float max_norm = 0.0f;
    for (std::uint32_t t = 0; t < kWindow; ++t) {
        for (std::uint32_t v = 0; v < kJoints; ++v) {
            const float x = at(*tensor, 0, t, v);
            const float y = at(*tensor, 1, t, v);
            max_norm = std::max(max_norm, std::sqrt(x * x + y * y));
        }
    }
    EXPECT_NEAR(max_norm, 1.0f, 1e-4f);
}

TEST(SkeletonBuffer, IsInvariantToTheApparentSizeOfThePerson) {
    // The same pose scaled 3x (a person closer to the camera) must
    // normalise to the same tensor.
    SkeletonBuffer small{tracking(), kJoints, kWindow};
    SkeletonBuffer large{tracking(), kJoints, kWindow};

    for (std::uint32_t t = 0; t < kWindow; ++t) {
        KeypointSet a(kJoints);
        KeypointSet b(kJoints);
        for (std::uint32_t v = 0; v < kJoints; ++v) {
            const float dx = static_cast<float>(v) * 4.0f;
            const float dy = static_cast<float>(t) * 6.0f;
            a[v] = Keypoint{dx, dy, 0.9f};
            b[v] = Keypoint{dx * 3.0f, dy * 3.0f, 0.9f};
        }
        small.push(a);
        large.push(b);
    }

    const auto ts = small.as_tensor();
    const auto tl = large.as_tensor();

    ASSERT_TRUE(ts.has_value());
    ASSERT_TRUE(tl.has_value());
    for (std::size_t i = 0; i < ts->size(); ++i) {
        EXPECT_NEAR((*ts)[i], (*tl)[i], 1e-4f) << "element " << i;
    }
}

TEST(SkeletonBuffer, KeepsEveryCoordinateFiniteForADegenerateClip) {
    // Every joint at exactly one point leaves a zero scale; the guard has
    // to stop that becoming a division by zero.
    SkeletonBuffer b{tracking(), kJoints, kWindow};
    for (std::uint32_t i = 0; i < kWindow; ++i)
        b.push(uniform_pose(0.0f, 0.0f));

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    for (const float value : *tensor) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(SkeletonBuffer, UsesTheMostRecentFramesOnceTheWindowSlides) {
    SkeletonBuffer b{tracking(), kJoints, /*window=*/2};

    b.push(uniform_pose(0.0f, 0.0f, 0.1f));  // will slide out
    b.push(uniform_pose(10.0f, 10.0f, 0.5f));
    b.push(uniform_pose(10.0f, 10.0f, 0.7f));

    const auto tensor = b.as_tensor();

    ASSERT_TRUE(tensor.has_value());
    EXPECT_FLOAT_EQ(at(*tensor, 2, 0, 0, 2), 0.5f);
    EXPECT_FLOAT_EQ(at(*tensor, 2, 1, 0, 2), 0.7f);
}

}  // namespace
