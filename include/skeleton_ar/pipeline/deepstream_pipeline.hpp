#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "skeleton_ar/config/system_config.hpp"

namespace skeleton_ar::pipeline {

class ProbeChain;

/// Single-source DeepStream pipeline, built but not yet driving anything:
///
///     nvurisrcbin -> nvstreammux -> nvinfer (YOLOv8 person)
///                 -> nvtracker (NvDCF) -> nvdsosd -> fakesink
///
/// The design is for a src-pad probe after `nvtracker` to collect
/// (track_id, bbox) pairs, hand them to the supplied `ProbeChain` for
/// pose + action classification, and encode an annotated stream out.
/// Neither half of that exists yet: no probe is attached, so
/// `probe_chain` is held and never called, and the sink is a `fakesink`,
/// so nothing is encoded. Both are roadmap items.
///
/// Nothing in this repo constructs this class - `skeleton_ar_video`
/// drives the same `ProbeChain` over OpenCV instead, and that is the
/// path the demos and the benchmark use.
class DeepStreamPipeline {
public:
    DeepStreamPipeline(const config::PipelineConfig& cfg, const std::string& pgie_config_path,
                       const std::string& tracker_config_path, ProbeChain& probe_chain);
    ~DeepStreamPipeline();

    DeepStreamPipeline(const DeepStreamPipeline&) = delete;
    DeepStreamPipeline& operator=(const DeepStreamPipeline&) = delete;

    /// Run the pipeline on `input_uri`. Blocks until EOS or `stop()`.
    ///
    /// Takes no output path: the sink is a `fakesink` and there is nothing
    /// to write. The parameter used to be here and was discarded with a
    /// cast, which promised a file the caller never got.
    void run(const std::string& input_uri);

    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    config::PipelineConfig cfg_;
    ProbeChain& probe_chain_;
    std::atomic<bool> running_{false};
};

}  // namespace skeleton_ar::pipeline
