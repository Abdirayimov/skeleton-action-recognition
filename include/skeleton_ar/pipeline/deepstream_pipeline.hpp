#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "skeleton_ar/config/system_config.hpp"

namespace skeleton_ar::pipeline {

class ProbeChain;

/// Single-source DeepStream pipeline targeting offline video processing:
///
///     filesrc -> decoder -> nvstreammux -> nvinfer (YOLOv8 person)
///              -> nvtracker (NvDCF) -> nvdsosd -> filesink (mp4)
///
/// A src-pad probe attached after nvtracker collects (track_id, bbox)
/// pairs and forwards them to the supplied `ProbeChain`, which runs
/// pose + action classification and writes an overlay before the
/// frame is encoded.
class DeepStreamPipeline {
public:
    DeepStreamPipeline(const config::PipelineConfig& cfg,
                       const std::string& pgie_config_path,
                       const std::string& tracker_config_path,
                       ProbeChain& probe_chain);
    ~DeepStreamPipeline();

    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;

    /// Run the pipeline on `input_uri` and (if `emit_overlay`) encode an
    /// annotated MP4 to `output_path`. Blocks until EOS or `stop()`.
    void run(const std::string& input_uri, const std::string& output_path);

    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    config::PipelineConfig cfg_;
    ProbeChain& probe_chain_;
    std::atomic<bool> running_{false};
};

}  // namespace skeleton_ar::pipeline
