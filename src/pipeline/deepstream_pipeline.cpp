#include "skeleton_ar/pipeline/deepstream_pipeline.hpp"

#include <gst/gst.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>

#include "skeleton_ar/pipeline/probe_chain.hpp"

namespace skeleton_ar::pipeline {

struct DeepStreamPipeline::Impl {
    GstElement* pipeline = nullptr;
    GstElement* source = nullptr;
    GstElement* streammux = nullptr;
    GstElement* pgie = nullptr;
    GstElement* tracker = nullptr;
    GstElement* osd = nullptr;
    GstElement* sink = nullptr;
    GMainLoop* loop = nullptr;
};

namespace {

GstElement* make(const char* factory, const char* name) {
    GstElement* e = gst_element_factory_make(factory, name);
    if (!e) {
        throw std::runtime_error(std::string("could not create GStreamer element: ") + factory);
    }
    return e;
}

}  // namespace

DeepStreamPipeline::DeepStreamPipeline(const config::PipelineConfig& cfg,
                                       const std::string& pgie_config_path,
                                       const std::string& tracker_config_path,
                                       ProbeChain& probe_chain)
    : impl_(std::make_unique<Impl>()), cfg_(cfg), probe_chain_(probe_chain) {
    if (!gst_is_initialized())
        gst_init(nullptr, nullptr);

    impl_->pipeline = gst_pipeline_new("skeleton-ar");
    impl_->source = make("nvurisrcbin", "video-source");
    impl_->streammux = make("nvstreammux", "stream-muxer");
    impl_->pgie = make("nvinfer", "primary-detector");
    impl_->tracker = make("nvtracker", "person-tracker");
    impl_->osd = make("nvdsosd", "overlay");
    impl_->sink = make("fakesink", "sink");

    g_object_set(G_OBJECT(impl_->streammux), "batch-size", cfg_.batch_size, "width",
                 cfg_.muxer_width, "height", cfg_.muxer_height, "batched-push-timeout",
                 cfg_.batched_push_timeout_us, "live-source", FALSE, nullptr);

    g_object_set(G_OBJECT(impl_->pgie), "config-file-path", pgie_config_path.c_str(), "batch-size",
                 cfg_.batch_size, nullptr);

    g_object_set(G_OBJECT(impl_->tracker), "tracker-width", 640, "tracker-height", 384,
                 "ll-config-file", tracker_config_path.c_str(), "ll-lib-file",
                 "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so", nullptr);

    g_object_set(G_OBJECT(impl_->osd), "process-mode", 1, nullptr);
    g_object_set(G_OBJECT(impl_->sink), "sync", FALSE, "async", FALSE, "qos", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(impl_->pipeline), impl_->source, impl_->streammux, impl_->pgie,
                     impl_->tracker, impl_->osd, impl_->sink, nullptr);

    GstPad* src_pad = gst_element_get_static_pad(impl_->source, "src");
    GstPad* mux_pad = gst_element_request_pad_simple(impl_->streammux, "sink_0");
    if (!src_pad || !mux_pad || gst_pad_link(src_pad, mux_pad) != GST_PAD_LINK_OK) {
        throw std::runtime_error("failed to link nvurisrcbin -> nvstreammux");
    }
    gst_object_unref(src_pad);
    gst_object_unref(mux_pad);

    if (!gst_element_link_many(impl_->streammux, impl_->pgie, impl_->tracker, impl_->osd,
                               impl_->sink, nullptr)) {
        throw std::runtime_error("failed to link mux -> pgie -> tracker -> osd -> sink");
    }

    // The tracker's src-pad probe would forward (track_id, bbox) tuples
    // to probe_chain_; the wire-up is left to the integration layer to
    // keep this constructor small and testable.
    (void)probe_chain_;
}

DeepStreamPipeline::~DeepStreamPipeline() {
    stop();
    if (impl_->pipeline) {
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
        gst_object_unref(impl_->pipeline);
    }
}

void DeepStreamPipeline::run(const std::string& input_uri, const std::string& output_path) {
    if (running_.exchange(true))
        return;
    g_object_set(G_OBJECT(impl_->source), "uri", input_uri.c_str(), nullptr);

    // Re-targeting the sink to a file is supported by replacing fakesink
    // with an encoder + filesink chain when emit_overlay is true. For the
    // reference implementation we keep fakesink and document this in
    // README.md; readers extending this should swap to nvv4l2h264enc +
    // h264parse + qtmux + filesink.
    (void)output_path;

    impl_->loop = g_main_loop_new(nullptr, FALSE);
    gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
    SPDLOG_INFO("DeepStreamPipeline running on input {}", input_uri);

    g_main_loop_run(impl_->loop);
    gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    running_ = false;
}

void DeepStreamPipeline::stop() {
    if (!running_.exchange(false))
        return;
    if (impl_->loop) {
        g_main_loop_quit(impl_->loop);
        g_main_loop_unref(impl_->loop);
        impl_->loop = nullptr;
    }
    SPDLOG_INFO("DeepStreamPipeline stop requested");
}

}  // namespace skeleton_ar::pipeline
