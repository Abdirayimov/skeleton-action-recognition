#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/overlay/visualizer.hpp"
#include "skeleton_ar/pipeline/probe_chain.hpp"
#include "skeleton_ar/tracking/track_registry.hpp"
#include "skeleton_ar/trt/rtmpose_estimator.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"
#include "skeleton_ar/trt/yolov8_detector.hpp"
#include "skeleton_ar/utils/logger.hpp"

namespace {

std::atomic<bool> g_shutdown{false};
void signal_handler(int) { g_shutdown = true; }

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --config CONFIG_YAML --input INPUT_VIDEO --output OUTPUT_VIDEO\n"
              << "       [--labels LABELS_TXT]\n"
              << "\n"
              << "  --config   Path to system_config.yaml.\n"
              << "  --labels   Optional override for action class labels.\n"
              << "  --input    Input video file (mp4, avi, ...). Required.\n"
              << "  --output   Annotated output mp4. Required.\n";
}

}  // namespace

// This binary uses an OpenCV-based fallback driver instead of running
// the full DeepStream pipeline. The DeepStream version
// (DeepStreamPipeline) shows the production layout; this CLI keeps the
// reference repo runnable on machines without a DeepStream install,
// which is the more common case for portfolio review.
int main(int argc, char** argv) {
    std::string config_path;
    std::string labels_path;
    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc) config_path = argv[++i];
        else if (a == "--labels" && i + 1 < argc) labels_path = argv[++i];
        else if ((a == "--input" || a == "-i") && i + 1 < argc) input_path = argv[++i];
        else if ((a == "--output" || a == "-o") && i + 1 < argc) output_path = argv[++i];
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty() || input_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = skeleton_ar::config::SystemConfig::load(config_path, labels_path);
        skeleton_ar::utils::init_logger(cfg.logging.level, cfg.logging.json);
        SPDLOG_INFO("skeleton_ar_video starting");

        skeleton_ar::trt::YOLOv8Detector detector(cfg.detection);
        skeleton_ar::trt::RTMPoseEstimator pose(cfg.pose);
        skeleton_ar::trt::STGCNClassifier action(cfg.action);
        skeleton_ar::tracking::TrackRegistry registry(cfg.tracking, cfg.action.num_keypoints,
                                                      cfg.action.window_frames);
        skeleton_ar::pipeline::ProbeChain probe(pose, action, registry, cfg.action);
        skeleton_ar::overlay::Visualizer viz(cfg.labels);

        cv::VideoCapture cap(input_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("could not open input video: " + input_path);
        }
        const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               (fps > 0.0 ? fps : 25.0), cv::Size(W, H));
        if (!writer.isOpened()) {
            throw std::runtime_error("could not open output writer: " + output_path);
        }

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Naive single-track id assignment: in this OpenCV-only driver,
        // we run a fresh detector per frame and reuse track_id = 0..N-1
        // matching the detection order. The DeepStream variant uses
        // NvDCF for proper temporal tracking.
        std::uint64_t frame_no = 0;
        cv::Mat frame;
        while (cap.read(frame)) {
            if (g_shutdown.load()) break;
            const auto detections = detector.detect(frame);

            std::vector<std::pair<std::uint64_t, cv::Rect2f>> tracks;
            tracks.reserve(detections.size());
            for (std::size_t i = 0; i < detections.size(); ++i) {
                tracks.emplace_back(static_cast<std::uint64_t>(i), detections[i].bbox);
            }

            const auto pts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            const auto result = probe.process(frame_no, pts_ns, frame, tracks);
            viz.render(frame, result);
            writer.write(frame);
            ++frame_no;

            if (frame_no % 30 == 0) {
                SPDLOG_INFO("processed {} frames; tracks={}", frame_no, result.tracks.size());
            }
        }

        SPDLOG_INFO("done: {} frames processed", frame_no);
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
