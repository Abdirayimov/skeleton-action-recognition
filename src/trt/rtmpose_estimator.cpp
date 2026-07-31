#include "skeleton_ar/trt/rtmpose_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "skeleton_ar/trt/trt_engine.hpp"
#include "skeleton_ar/utils/cuda_helpers.hpp"

namespace skeleton_ar::trt {

namespace {

constexpr float kImageMean[3] = {123.675f, 116.28f, 103.53f};  // RGB ImageNet mean
constexpr float kImageStd[3] = {58.395f, 57.12f, 57.375f};     // RGB ImageNet std

void preprocess(const cv::Mat& crop, int out_w, int out_h, float* dst) {
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

    const int channel_stride = out_w * out_h;
    for (int y = 0; y < out_h; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out_w; ++x) {
            const auto& px = row[x];
            const int idx = y * out_w + x;
            // BGR -> RGB then ImageNet normalize.
            const float r = static_cast<float>(px[2]);
            const float g = static_cast<float>(px[1]);
            const float b = static_cast<float>(px[0]);
            dst[0 * channel_stride + idx] = (r - kImageMean[0]) / kImageStd[0];
            dst[1 * channel_stride + idx] = (g - kImageMean[1]) / kImageStd[1];
            dst[2 * channel_stride + idx] = (b - kImageMean[2]) / kImageStd[2];
        }
    }
}

cv::Rect clamp_to(const cv::Rect2f& r, int W, int H) {
    const int x = std::max(0, static_cast<int>(std::floor(r.x)));
    const int y = std::max(0, static_cast<int>(std::floor(r.y)));
    const int x2 = std::min(W, static_cast<int>(std::ceil(r.x + r.width)));
    const int y2 = std::min(H, static_cast<int>(std::ceil(r.y + r.height)));
    if (x2 <= x || y2 <= y)
        return cv::Rect();
    return cv::Rect(x, y, x2 - x, y2 - y);
}

}  // namespace

RTMPoseEstimator::RTMPoseEstimator(const config::PoseConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    const std::size_t per_img = static_cast<std::size_t>(3) * cfg.input_height * cfg.input_width;
    input_scratch_.resize(cfg.batch_size * per_img);
    output_scratch_.resize(static_cast<std::size_t>(cfg.batch_size) * cfg.num_keypoints * 3);
}

RTMPoseEstimator::~RTMPoseEstimator() = default;

std::vector<KeypointSet> RTMPoseEstimator::estimate(const cv::Mat& image,
                                                    const std::vector<cv::Rect2f>& bboxes) {
    std::vector<KeypointSet> out(bboxes.size());
    if (bboxes.empty())
        return out;

    const std::size_t bsz = cfg_.batch_size;
    for (std::size_t i = 0; i < bboxes.size(); i += bsz) {
        const std::size_t count = std::min(bsz, bboxes.size() - i);
        run_chunk_(image, bboxes, i, count, out);
    }
    return out;
}

void RTMPoseEstimator::run_chunk_(const cv::Mat& image, const std::vector<cv::Rect2f>& bboxes,
                                  std::size_t offset, std::size_t count,
                                  std::vector<KeypointSet>& out) {
    const int W = image.cols;
    const int H = image.rows;
    const std::int64_t in_w = static_cast<std::int64_t>(cfg_.input_width);
    const std::int64_t in_h = static_cast<std::int64_t>(cfg_.input_height);
    const std::size_t per_img = static_cast<std::size_t>(3 * in_w * in_h);

    // Track the actual valid crops in this chunk; some bboxes may degenerate
    // after clamping.
    std::vector<cv::Rect> valid_rois(count);
    std::size_t actual = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const auto roi = clamp_to(bboxes[offset + k], W, H);
        if (roi.area() <= 0)
            continue;
        valid_rois[actual] = roi;
        cv::Mat crop = image(roi);
        preprocess(crop, static_cast<int>(in_w), static_cast<int>(in_h),
                   input_scratch_.data() + actual * per_img);
        ++actual;
    }
    if (actual == 0)
        return;

    const std::string input_name = engine_->bindings().front().name;
    engine_->set_input_shape(input_name, {static_cast<std::int64_t>(actual), 3, in_h, in_w});

    utils::CudaStream stream;
    engine_->copy_input(input_name, input_scratch_.data(), actual * per_img * sizeof(float),
                        stream.get());
    engine_->infer(stream.get());

    // RTMPose simcc head produces two outputs (x, y) of shape
    // (B, K, sx) and (B, K, sy). For simplicity we assume a single
    // fused-output ONNX export of shape (B, K, 3) -> (x_norm, y_norm,
    // confidence) in input space.
    std::string out_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    const auto& ob = engine_->binding(out_name);
    if (ob.volume < actual * cfg_.num_keypoints * 3) {
        throw std::runtime_error("unexpected RTMPose output volume");
    }
    engine_->copy_output(out_name, output_scratch_.data(),
                         actual * cfg_.num_keypoints * 3 * sizeof(float), stream.get());
    stream.synchronize();

    // Map back to original image coordinates.
    std::size_t emit = 0;
    for (std::size_t k = 0; k < count; ++k) {
        const auto& roi_src = bboxes[offset + k];
        const auto roi = clamp_to(roi_src, W, H);
        if (roi.area() <= 0) {
            // Leave default (empty) keypoint set in `out`.
            continue;
        }
        const float x_scale = static_cast<float>(roi.width) / static_cast<float>(in_w);
        const float y_scale = static_cast<float>(roi.height) / static_cast<float>(in_h);

        KeypointSet kps(cfg_.num_keypoints);
        const float* src = output_scratch_.data() + emit * cfg_.num_keypoints * 3;
        for (std::uint32_t j = 0; j < cfg_.num_keypoints; ++j) {
            kps[j].x = static_cast<float>(roi.x) + src[j * 3 + 0] * x_scale;
            kps[j].y = static_cast<float>(roi.y) + src[j * 3 + 1] * y_scale;
            kps[j].score = src[j * 3 + 2];
        }
        out[offset + k] = std::move(kps);
        ++emit;
    }
}

}  // namespace skeleton_ar::trt
