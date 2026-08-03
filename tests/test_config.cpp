#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "skeleton_ar/config/system_config.hpp"

namespace {

using skeleton_ar::config::SystemConfig;

std::string fixture(const std::string& name) {
    return std::string(SKAR_TEST_FIXTURES) + "/" + name;
}

std::string repo_config(const std::string& name) {
    return std::string(SKAR_TEST_FIXTURES) + "/../../configs/" + name;
}

/// A config with every field valid, used as the starting point for the
/// programmatic `validate()` tests.
SystemConfig valid_config() {
    SystemConfig cfg;
    cfg.detection.engine_path = "models/engines/detector.engine";
    cfg.pose.engine_path = "models/engines/pose.engine";
    cfg.action.engine_path = "models/engines/action.engine";
    return cfg;
}

// ------------------------------------------------------------- happy path

TEST(SystemConfigLoad, ReadsEveryFieldFromAFullyPopulatedFile) {
    const SystemConfig cfg = SystemConfig::load(fixture("valid_full.yaml"));

    EXPECT_EQ(cfg.pipeline.muxer_width, 1920u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 1080u);
    EXPECT_EQ(cfg.pipeline.batch_size, 4u);
    EXPECT_EQ(cfg.pipeline.batched_push_timeout_us, 33000u);
    EXPECT_FALSE(cfg.pipeline.emit_overlay);

    EXPECT_EQ(cfg.detection.engine_path, "models/engines/yolov8s_person_fp16.engine");
    EXPECT_EQ(cfg.detection.input_width, 960u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.35f);
    EXPECT_FLOAT_EQ(cfg.detection.nms_iou_threshold, 0.6f);
    EXPECT_EQ(cfg.detection.person_class_id, 0);

    EXPECT_EQ(cfg.pose.input_width, 288u);
    EXPECT_EQ(cfg.pose.input_height, 384u);
    EXPECT_EQ(cfg.pose.num_keypoints, 17u);
    EXPECT_EQ(cfg.pose.batch_size, 8u);

    EXPECT_EQ(cfg.action.num_classes, 10u);
    EXPECT_EQ(cfg.action.window_frames, 45u);
    EXPECT_EQ(cfg.action.step_frames, 15u);
    EXPECT_EQ(cfg.action.num_keypoints, 17u);

    EXPECT_EQ(cfg.tracking.buffer_size, 60u);
    EXPECT_EQ(cfg.tracking.max_missed_frames, 20u);
    EXPECT_FLOAT_EQ(cfg.tracking.min_keypoint_confidence, 0.4f);

    EXPECT_EQ(cfg.logging.level, "debug");
    EXPECT_FALSE(cfg.logging.json);
}

TEST(SystemConfigLoad, FillsInDocumentedDefaultsForOmittedFields) {
    const SystemConfig cfg = SystemConfig::load(fixture("minimal.yaml"));

    EXPECT_EQ(cfg.pipeline.muxer_width, 1280u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 720u);
    EXPECT_EQ(cfg.pipeline.batch_size, 1u);
    EXPECT_TRUE(cfg.pipeline.emit_overlay);

    EXPECT_EQ(cfg.detection.input_width, 640u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.4f);

    EXPECT_EQ(cfg.pose.input_width, 192u);
    EXPECT_EQ(cfg.pose.input_height, 256u);
    EXPECT_EQ(cfg.pose.num_keypoints, 17u);

    EXPECT_EQ(cfg.action.num_classes, 10u);
    EXPECT_EQ(cfg.action.window_frames, 30u);
    EXPECT_EQ(cfg.action.step_frames, 10u);

    EXPECT_EQ(cfg.tracking.buffer_size, 30u);
    EXPECT_EQ(cfg.tracking.max_missed_frames, 15u);
    EXPECT_FLOAT_EQ(cfg.tracking.min_keypoint_confidence, 0.3f);

    EXPECT_EQ(cfg.logging.level, "info");
    EXPECT_TRUE(cfg.logging.json);
}

TEST(SystemConfigLoad, AcceptsTheConfigShippedWithTheRepository) {
    EXPECT_NO_THROW({
        const SystemConfig cfg = SystemConfig::load(repo_config("system_config.yaml"),
                                                    repo_config("labels_ntu60_subset.txt"));
        (void)cfg;
    });
}

TEST(SystemConfigLoad, ReadsTheLabelTableAndSkipsComments) {
    // The shipped table has ten labels behind three comment lines, which
    // is what makes it line up with the default num_classes.
    const SystemConfig cfg = SystemConfig::load(repo_config("system_config.yaml"),
                                                repo_config("labels_ntu60_subset.txt"));

    EXPECT_EQ(cfg.labels.size(), cfg.action.num_classes);
    for (const auto& label : cfg.labels) {
        ASSERT_FALSE(label.empty());
        EXPECT_NE(label.front(), '#');
    }
}

// ---------------------------------------------------------------- failures

TEST(SystemConfigLoad, ReportsAMissingFileByName) {
    const std::string path = fixture("this_file_does_not_exist.yaml");

    try {
        SystemConfig::load(path);
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos)
            << "error should name the offending path: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsMalformedYaml) {
    EXPECT_THROW(SystemConfig::load(fixture("malformed.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingActionSection) {
    EXPECT_THROW(SystemConfig::load(fixture("missing_action.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingRequiredKey) {
    try {
        SystemConfig::load(fixture("missing_engine_path.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("engine_path"), std::string::npos) << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAConfidenceThresholdAboveOne) {
    try {
        SystemConfig::load(fixture("invalid_confidence.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("detection.confidence_threshold"), std::string::npos)
            << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAKeypointCountMismatchBetweenPoseAndAction) {
    try {
        SystemConfig::load(fixture("keypoint_mismatch.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("num_keypoints"), std::string::npos) << e.what();
    }
}

TEST(SystemConfigLoad, RejectsABufferTooSmallToHoldAClip) {
    try {
        SystemConfig::load(fixture("buffer_too_small.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("buffer_size"), std::string::npos) << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAStepLargerThanTheWindow) {
    try {
        SystemConfig::load(fixture("step_exceeds_window.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("step_frames"), std::string::npos) << e.what();
    }
}

TEST(SystemConfigLoad, RejectsALabelTableThatDoesNotMatchNumClasses) {
    // The fixture table has three labels; the config declares ten classes.
    try {
        SystemConfig::load(fixture("minimal.yaml"), fixture("labels.txt"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("num_classes"), std::string::npos) << e.what();
    }
}

// ---------------------------------------------------- programmatic checks

TEST(SystemConfigValidate, AcceptsTheStructDefaultsPlusEnginePaths) {
    EXPECT_NO_THROW(valid_config().validate());
}

TEST(SystemConfigValidate, RejectsAnEmptyEnginePath) {
    for (int stage = 0; stage < 3; ++stage) {
        SystemConfig cfg = valid_config();
        if (stage == 0)
            cfg.detection.engine_path.clear();
        if (stage == 1)
            cfg.pose.engine_path.clear();
        if (stage == 2)
            cfg.action.engine_path.clear();
        EXPECT_THROW(cfg.validate(), std::runtime_error) << "stage " << stage;
    }
}

TEST(SystemConfigValidate, RejectsAZeroBatchSize) {
    SystemConfig cfg = valid_config();
    cfg.pipeline.batch_size = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroWindow) {
    SystemConfig cfg = valid_config();
    cfg.action.window_frames = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroKeypointCount) {
    SystemConfig cfg = valid_config();
    cfg.pose.num_keypoints = 0;
    cfg.action.num_keypoints = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsANegativePersonClassId) {
    SystemConfig cfg = valid_config();
    cfg.detection.person_class_id = -1;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroMissedFrameBudget) {
    SystemConfig cfg = valid_config();
    cfg.tracking.max_missed_frames = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAKeypointConfidenceOutsideTheUnitInterval) {
    SystemConfig cfg = valid_config();
    cfg.tracking.min_keypoint_confidence = 1.5f;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, AcceptsAStepEqualToTheWindow) {
    // Non-overlapping clips are a legitimate configuration.
    SystemConfig cfg = valid_config();
    cfg.action.step_frames = cfg.action.window_frames;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(SystemConfigValidate, AcceptsABufferExactlyOneClipLong) {
    SystemConfig cfg = valid_config();
    cfg.tracking.buffer_size = cfg.action.window_frames;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(SystemConfigValidate, AcceptsAnEmptyLabelTable) {
    // Labels are optional; only a populated table has to match.
    SystemConfig cfg = valid_config();
    cfg.labels.clear();
    EXPECT_NO_THROW(cfg.validate());
}

}  // namespace
