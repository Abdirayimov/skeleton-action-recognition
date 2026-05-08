#include "skeleton_ar/trt/stgcn_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "skeleton_ar/trt/trt_engine.hpp"
#include "skeleton_ar/utils/cuda_helpers.hpp"

namespace skeleton_ar::trt {

namespace {

void softmax_inplace(float* x, std::size_t n) {
    if (n == 0) return;
    const float m = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }
    if (sum > 0.0f) {
        for (std::size_t i = 0; i < n; ++i) x[i] /= sum;
    }
}

}  // namespace

STGCNClassifier::STGCNClassifier(const config::ActionConfig& cfg)
    : cfg_(cfg), engine_(std::make_unique<TrtEngine>(cfg.engine_path)) {
    output_scratch_.resize(cfg.num_classes);
}

STGCNClassifier::~STGCNClassifier() = default;

ActionPrediction STGCNClassifier::classify(const std::vector<float>& skeleton) {
    const std::int64_t T = static_cast<std::int64_t>(cfg_.window_frames);
    const std::int64_t V = static_cast<std::int64_t>(cfg_.num_keypoints);
    const std::size_t expected = static_cast<std::size_t>(3 * T * V);
    if (skeleton.size() != expected) {
        throw std::runtime_error("STGCNClassifier::classify: input size mismatch");
    }

    const std::string input_name = engine_->bindings().front().name;
    // (1, 3, T, V, 1)
    engine_->set_input_shape(input_name, {1, 3, T, V, 1});

    utils::CudaStream stream;
    engine_->copy_input(input_name, skeleton.data(), skeleton.size() * sizeof(float),
                        stream.get());
    engine_->infer(stream.get());

    std::string out_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    engine_->copy_output(out_name, output_scratch_.data(),
                         output_scratch_.size() * sizeof(float), stream.get());
    stream.synchronize();

    softmax_inplace(output_scratch_.data(), output_scratch_.size());

    const auto best = std::max_element(output_scratch_.begin(), output_scratch_.end());
    ActionPrediction p;
    p.class_id = static_cast<std::int32_t>(std::distance(output_scratch_.begin(), best));
    p.confidence = *best;
    return p;
}

std::vector<ActionPrediction> STGCNClassifier::classify_batch(const Eigen::MatrixXf& skeletons) {
    const auto B = static_cast<std::int64_t>(skeletons.rows());
    const std::int64_t T = static_cast<std::int64_t>(cfg_.window_frames);
    const std::int64_t V = static_cast<std::int64_t>(cfg_.num_keypoints);

    const std::string input_name = engine_->bindings().front().name;
    engine_->set_input_shape(input_name, {B, 3, T, V, 1});

    utils::CudaStream stream;
    engine_->copy_input(input_name, skeletons.data(),
                        static_cast<std::size_t>(skeletons.size()) * sizeof(float), stream.get());
    engine_->infer(stream.get());

    std::string out_name;
    for (const auto& b : engine_->bindings()) {
        if (!b.is_input) {
            out_name = b.name;
            break;
        }
    }
    std::vector<float> out(static_cast<std::size_t>(B) * cfg_.num_classes);
    engine_->copy_output(out_name, out.data(), out.size() * sizeof(float), stream.get());
    stream.synchronize();

    std::vector<ActionPrediction> result(static_cast<std::size_t>(B));
    for (std::size_t i = 0; i < static_cast<std::size_t>(B); ++i) {
        float* row = out.data() + i * cfg_.num_classes;
        softmax_inplace(row, cfg_.num_classes);
        const auto best = std::max_element(row, row + cfg_.num_classes);
        result[i].class_id = static_cast<std::int32_t>(std::distance(row, best));
        result[i].confidence = *best;
    }
    return result;
}

}  // namespace skeleton_ar::trt
