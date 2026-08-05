# Building

## Native (Ubuntu 22.04)

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    libeigen3-dev libspdlog-dev libyaml-cpp-dev \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

CUDA 12.x, TensorRT 8.6+, and DeepStream 7.x or 8.x must be installed
separately from NVIDIA. Override search paths if needed:

```bash
export TensorRT_ROOT=/usr/local/tensorrt
export DeepStream_ROOT=/opt/nvidia/deepstream/deepstream
```

Configure and build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

You will get:

- `build/skeleton_ar_video` - offline video processor
- `build/skeleton_ar_benchmark` - per-stage latency probe
- `build/skeleton_demo` - ST-GCN over stored clips, renders the README demo

### Options

| Option | Default | What it does |
|---|---|---|
| `SKAR_CPU_ONLY` | `OFF` | Build only `skeleton_ar_core` - config, the skeleton buffer, the track registry and the logger. Skips every CUDA / TensorRT / DeepStream `find_package`, so it configures on a machine with no NVIDIA stack. Forces `BUILD_TOOLS` and `BUILD_DEEPSTREAM` off. |
| `SKAR_BUILD_TESTS` | `OFF` | Build the GoogleTest suite. GoogleTest is fetched at configure time, pinned to `v1.14.0`. |
| `BUILD_TOOLS` | `ON` | Build `skeleton_ar_benchmark` and `skeleton_demo`. |
| `BUILD_DEEPSTREAM` | `ON` | Compile `deepstream_pipeline.cpp` and link GStreamer + the DeepStream SDK. Turns itself off with a warning when the SDK is not found. Note that the pipeline does not run the recognition chain yet - see the README's Limitations. |

`BUILD_TESTS` is accepted as a deprecated alias for `SKAR_BUILD_TESTS`
and warns.

The CPU-only configuration, which is what CI builds:

```bash
cmake -S . -B build-tests -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DSKAR_CPU_ONLY=ON \
      -DSKAR_BUILD_TESTS=ON
cmake --build build-tests -j
ctest --test-dir build-tests --output-on-failure
```

> Verified on: Ubuntu 24.04, GCC 13, CMake 3.28, CPU-only configuration.
> The CUDA / TensorRT / DeepStream configuration is not built by CI and
> needs the SDKs present.

## Docker

```bash
docker compose up --build
```

The image uses the official `nvcr.io/nvidia/deepstream:8.0-gc-triton-devel`
as its toolchain. The runtime stage is the same image; switch to the
`*-base` image for a smaller deployment if you have built the engines
ahead of time.

## TensorRT engines

The repo never commits `.engine` files. Build them once from the public
ONNX checkpoints:

```bash
./scripts/download_models.sh   # prints instructions per model
./scripts/build_engines.sh     # FP16
```

The container's `entrypoint.sh` will also build any missing engines on
first run.

## Common issues

**`could not open input video`** - `cv::VideoCapture` could not decode
the file. On Ubuntu, `apt-get install ffmpeg libavcodec-extra` covers
most codec gaps.

**`ST-GCN classify: input size mismatch`** - the skeleton tensor shape
the C++ runtime built does not match what the engine was exported for.
Check `action.window_frames` and `action.num_keypoints` in
`configs/system_config.yaml` against `training/configs/train.yaml` and
the engine's profile.

**`nvinfer ERROR: ... batch size mismatch`** - rebuild the YOLOv8
engine with `--maxBatch=<n>` matching `pipeline.batch_size`.
