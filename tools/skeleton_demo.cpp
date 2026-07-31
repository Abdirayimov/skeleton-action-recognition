// skeleton_demo: visualise the trained ST-GCN on held-out NTU clips.
//
// Reads a binary bundle of NTU-25 skeleton clips (produced by
// training/.../train_ntu.py), runs each through the ST-GCN TensorRT
// engine via STGCNClassifier, and rasterises an animated stick figure
// to a sequence of PPM frames. A status bar across the top is green
// when the prediction matches the ground-truth class, red otherwise.
//
// Deliberately free of OpenCV: it draws with a tiny built-in
// rasteriser and writes plain PPM frames, so it needs only TensorRT.
// Encode the frames into a GIF/MP4 with ffmpeg afterwards; the
// per-clip prediction ranges are printed for an optional text overlay.
//
//   skeleton_demo --engine stgcn_ntu10_fp16.engine --clips demo_clips.bin \
//                 --labels configs/labels_ntu60_subset.txt --out-dir frames

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "skeleton_ar/config/system_config.hpp"
#include "skeleton_ar/trt/stgcn_classifier.hpp"
#include "skeleton_ar/utils/logger.hpp"

namespace {

constexpr std::uint32_t kClipMagic = 0x534B4C31;  // 'SKL1'
constexpr int W = 480, H = 540;

// NTU-25 bone list (0-indexed), matching graph.NTU25_EDGES.
constexpr std::array<std::pair<int, int>, 24> kNtuEdges{{
    {0, 1},  {1, 20},  {2, 20},  {3, 2},   {4, 20},  {5, 4},   {6, 5},   {7, 6},
    {8, 20}, {9, 8},   {10, 9},  {11, 10}, {12, 0},  {13, 12}, {14, 13}, {15, 14},
    {16, 0}, {17, 16}, {18, 17}, {19, 18}, {21, 22}, {22, 7},  {23, 24}, {24, 11},
}};

struct Rgb {
    std::uint8_t r, g, b;
};

class Canvas {
public:
    Canvas() : buf_(static_cast<std::size_t>(W) * H * 3) {}
    void fill(Rgb c) {
        for (std::size_t i = 0; i < buf_.size(); i += 3) {
            buf_[i] = c.r;
            buf_[i + 1] = c.g;
            buf_[i + 2] = c.b;
        }
    }
    void px(int x, int y, Rgb c) {
        if (x < 0 || x >= W || y < 0 || y >= H)
            return;
        const std::size_t i = (static_cast<std::size_t>(y) * W + x) * 3;
        buf_[i] = c.r;
        buf_[i + 1] = c.g;
        buf_[i + 2] = c.b;
    }
    void disk(int cx, int cy, int rad, Rgb c) {
        for (int dy = -rad; dy <= rad; ++dy)
            for (int dx = -rad; dx <= rad; ++dx)
                if (dx * dx + dy * dy <= rad * rad)
                    px(cx + dx, cy + dy, c);
    }
    void line(int x0, int y0, int x1, int y1, Rgb c, int thick) {
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true) {
            disk(x0, y0, thick, c);
            if (x0 == x1 && y0 == y1)
                break;
            const int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
    void bar(int y0, int y1, Rgb c) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                px(x, y, c);
    }
    void write_ppm(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << W << " " << H << "\n255\n";
        f.write(reinterpret_cast<const char*>(buf_.data()),
                static_cast<std::streamsize>(buf_.size()));
    }

private:
    std::vector<std::uint8_t> buf_;
};

struct ClipBundle {
    int count = 0, c = 0, t = 0, v = 0, m = 0;
    std::vector<float> data;
    std::vector<std::int32_t> labels;
};

ClipBundle load_clips(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("cannot open clips: " + path);
    std::int32_t hdr[6];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (static_cast<std::uint32_t>(hdr[0]) != kClipMagic)
        throw std::runtime_error("bad magic");
    ClipBundle b;
    b.count = hdr[1];
    b.c = hdr[2];
    b.t = hdr[3];
    b.v = hdr[4];
    b.m = hdr[5];
    const std::size_t n = static_cast<std::size_t>(b.count) * b.c * b.t * b.v * b.m;
    b.data.resize(n);
    f.read(reinterpret_cast<char*>(b.data.data()), static_cast<std::streamsize>(n * sizeof(float)));
    b.labels.resize(static_cast<std::size_t>(b.count));
    f.read(reinterpret_cast<char*>(b.labels.data()),
           static_cast<std::streamsize>(b.count * sizeof(std::int32_t)));
    if (!f)
        throw std::runtime_error("short read on clips file");
    return b;
}

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty() && line[0] != '#')
            out.push_back(line);
    return out;
}

inline float at(const float* clip, int c, int t, int v, int T, int V) {
    return clip[(static_cast<std::size_t>(c) * T + t) * V + v];
}

}  // namespace

int main(int argc, char** argv) {
    std::string engine, clips_path, labels_path, out_dir;
    int hold = 6;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](const std::string& flag) {
            if (i + 1 >= argc)
                throw std::invalid_argument(flag + " expects a value");
            return std::string(argv[++i]);
        };
        if (a == "--engine")
            engine = take(a);
        else if (a == "--clips")
            clips_path = take(a);
        else if (a == "--labels")
            labels_path = take(a);
        else if (a == "--out-dir")
            out_dir = take(a);
        else if (a == "--hold")
            hold = std::stoi(take(a));
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s --engine E --clips C.bin --labels L.txt --out-dir DIR\n",
                        argv[0]);
            return 0;
        }
    }
    if (engine.empty() || clips_path.empty() || labels_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "missing required argument; see --help\n");
        return 1;
    }

    try {
        skeleton_ar::utils::init_logger("info", false);
        const auto labels = load_labels(labels_path);
        const auto bundle = load_clips(clips_path);
        SPDLOG_INFO("loaded {} clips ({}x{}x{}x{})", bundle.count, bundle.c, bundle.t, bundle.v,
                    bundle.m);

        skeleton_ar::config::ActionConfig acfg;
        acfg.engine_path = engine;
        acfg.num_classes = static_cast<std::uint32_t>(labels.size());
        acfg.window_frames = static_cast<std::uint32_t>(bundle.t);
        acfg.num_keypoints = static_cast<std::uint32_t>(bundle.v);
        skeleton_ar::trt::STGCNClassifier classifier(acfg);

        const std::size_t clip_floats =
            static_cast<std::size_t>(bundle.c) * bundle.t * bundle.v * bundle.m;
        const int T = bundle.t, V = bundle.v;

        std::ofstream ranges(out_dir + "/ranges.txt");
        int global = 0, correct_total = 0;

        for (int k = 0; k < bundle.count; ++k) {
            const float* clip = bundle.data.data() + static_cast<std::size_t>(k) * clip_floats;
            std::vector<float> in(clip, clip + clip_floats);
            const auto pred = classifier.classify(in);
            const int gt = bundle.labels[static_cast<std::size_t>(k)];
            const bool ok = pred.class_id == gt;
            correct_total += ok ? 1 : 0;
            const std::string pred_name =
                (pred.class_id >= 0 && pred.class_id < static_cast<int>(labels.size()))
                    ? labels[static_cast<std::size_t>(pred.class_id)]
                    : "unknown";
            SPDLOG_INFO("clip {:2d}: pred={:<16} truth={:<16} {}", k, pred_name,
                        labels[static_cast<std::size_t>(gt)], ok ? "OK" : "X");

            // Bounds for projection (person 0, x/y channels).
            float minx = 1e9f, maxx = -1e9f, miny = 1e9f, maxy = -1e9f;
            for (int t = 0; t < T; ++t) {
                float e = 0;
                for (int v = 0; v < V; ++v)
                    e += std::abs(at(clip, 0, t, v, T, V));
                if (e < 1e-6f)
                    continue;
                for (int v = 0; v < V; ++v) {
                    minx = std::min(minx, at(clip, 0, t, v, T, V));
                    maxx = std::max(maxx, at(clip, 0, t, v, T, V));
                    miny = std::min(miny, at(clip, 1, t, v, T, V));
                    maxy = std::max(maxy, at(clip, 1, t, v, T, V));
                }
            }
            const float scale = std::min(300.0f / std::max(1e-3f, maxx - minx),
                                         400.0f / std::max(1e-3f, maxy - miny));
            const float cx = (minx + maxx) * 0.5f, cy = (miny + maxy) * 0.5f;
            auto proj = [&](float x, float y) {
                return std::pair<int, int>(static_cast<int>(W * 0.5f + (x - cx) * scale),
                                           static_cast<int>(H * 0.5f - (y - cy) * scale + 16));
            };

            const Rgb status = ok ? Rgb{70, 190, 90} : Rgb{60, 60, 220};
            const Rgb bone{70, 150, 210}, joint{250, 200, 60}, bg{26, 24, 22};

            const int clip_start = global;
            for (int t = 0; t < T; ++t) {
                float e = 0;
                for (int v = 0; v < V; ++v)
                    e += std::abs(at(clip, 0, t, v, T, V));
                if (e < 1e-6f)
                    continue;
                Canvas cv;
                cv.fill(bg);
                cv.bar(0, 10, status);
                for (const auto& [a, b] : kNtuEdges) {
                    if (a >= V || b >= V)
                        continue;
                    auto pa = proj(at(clip, 0, t, a, T, V), at(clip, 1, t, a, T, V));
                    auto pb = proj(at(clip, 0, t, b, T, V), at(clip, 1, t, b, T, V));
                    cv.line(pa.first, pa.second, pb.first, pb.second, bone, 2);
                }
                for (int v = 0; v < V; ++v) {
                    auto p = proj(at(clip, 0, t, v, T, V), at(clip, 1, t, v, T, V));
                    cv.disk(p.first, p.second, 3, joint);
                }
                char path[256];
                std::snprintf(path, sizeof(path), "%s/f%05d.ppm", out_dir.c_str(), global);
                cv.write_ppm(path);
                ++global;
            }
            for (int h = 0; h < hold && global > clip_start; ++h) {
                char src[256], dst[256];
                std::snprintf(src, sizeof(src), "%s/f%05d.ppm", out_dir.c_str(), global - 1);
                std::snprintf(dst, sizeof(dst), "%s/f%05d.ppm", out_dir.c_str(), global);
                std::ifstream in_f(src, std::ios::binary);
                std::ofstream out_f(dst, std::ios::binary);
                out_f << in_f.rdbuf();
                ++global;
            }
            // start end pred gt correct  (frame range for an optional overlay)
            ranges << clip_start << " " << (global - 1) << " " << pred_name << " "
                   << labels[static_cast<std::size_t>(gt)] << " " << (ok ? 1 : 0) << "\n";
        }
        SPDLOG_INFO("done: {}/{} correct; {} frames in {}", correct_total, bundle.count, global,
                    out_dir);
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal: {}", e.what());
        return 1;
    }
    return 0;
}
