// benchmark: per-stage latency for the skeleton_ar pipeline using random
// synthetic inputs. Useful for quick sanity-checks without a real video.

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <random>
#include <string>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/trt/rtmpose_estimator.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"
#include "skeleton_ar/trt/yolov8_detector.hpp"
#include "skeleton_ar/utils/logger.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= v.size())
        return v[lo];
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --config CONFIG_YAML\n";
}

void report(const char* stage, const std::vector<double>& ms) {
    std::cout << stage << "\n"
              << "  p50: " << std::fixed << std::setprecision(3) << percentile(ms, 0.5) << " ms\n"
              << "  p95: " << percentile(ms, 0.95) << " ms\n"
              << "  p99: " << percentile(ms, 0.99) << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc)
            config_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = skeleton_ar::config::SystemConfig::load(config_path);
        skeleton_ar::utils::init_logger("warn", false);

        skeleton_ar::trt::YOLOv8Detector detector(cfg.detection);
        skeleton_ar::trt::RTMPoseEstimator pose(cfg.pose);
        skeleton_ar::trt::STGCNClassifier action(cfg.action);

        // Synthesize a 720p frame and a couple of fake bboxes.
        cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(64, 64, 64));
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> ud(0, 255);
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                auto& px = frame.at<cv::Vec3b>(y, x);
                px[0] = static_cast<unsigned char>(ud(rng));
                px[1] = static_cast<unsigned char>(ud(rng));
                px[2] = static_cast<unsigned char>(ud(rng));
            }
        }
        std::vector<cv::Rect2f> bboxes{
            cv::Rect2f(100, 200, 200, 400),
            cv::Rect2f(700, 150, 220, 480),
        };

        constexpr std::size_t iters = 200;

        // Warm-ups
        for (std::size_t i = 0; i < 5; ++i) {
            (void)detector.detect(frame);
            (void)pose.estimate(frame, bboxes);
        }

        std::vector<double> det_ms;
        det_ms.reserve(iters);
        for (std::size_t i = 0; i < iters; ++i) {
            const auto t0 = Clock::now();
            (void)detector.detect(frame);
            const auto t1 = Clock::now();
            det_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        report("YOLOv8 detect (1 frame)", det_ms);

        std::vector<double> pose_ms;
        pose_ms.reserve(iters);
        for (std::size_t i = 0; i < iters; ++i) {
            const auto t0 = Clock::now();
            (void)pose.estimate(frame, bboxes);
            const auto t1 = Clock::now();
            pose_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        report("RTMPose estimate (2 boxes)", pose_ms);

        std::vector<float> skel(static_cast<std::size_t>(3) * cfg.action.window_frames *
                                cfg.action.num_keypoints);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        for (auto& v : skel)
            v = nd(rng);
        for (std::size_t i = 0; i < 5; ++i)
            (void)action.classify(skel);

        std::vector<double> action_ms;
        action_ms.reserve(iters);
        for (std::size_t i = 0; i < iters; ++i) {
            const auto t0 = Clock::now();
            (void)action.classify(skel);
            const auto t1 = Clock::now();
            action_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        report("ST-GCN classify (1 clip)", action_ms);
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal error: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
