cmake_minimum_required(VERSION 3.22)
project(skeleton_ar VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TOOLS "Build CLI tools (benchmark)" ON)
option(BUILD_DEEPSTREAM "Build the DeepStream pipeline (requires DeepStream SDK + GStreamer)" ON)
option(SKAR_BUILD_TESTS "Build the GoogleTest unit test suite" OFF)
option(SKAR_CPU_ONLY
       "Build only the CPU-only algorithmic core (no CUDA / TensorRT / DeepStream)" OFF)

# Accepted for backwards compatibility with pre-0.2 build scripts.
if(BUILD_TESTS AND NOT SKAR_BUILD_TESTS)
    message(WARNING "BUILD_TESTS is deprecated; use SKAR_BUILD_TESTS")
    set(SKAR_BUILD_TESTS ON)
endif()

if(SKAR_CPU_ONLY)
    # The GPU stages have no meaning without their SDKs; turn them off
    # rather than letting find_package fail late.
    set(BUILD_TOOLS OFF)
    set(BUILD_DEEPSTREAM OFF)
endif()

# ---- Dependencies ----
# The algorithmic core needs only these four; the rest are GPU stages and
# are looked up separately below.
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

if(NOT SKAR_CPU_ONLY)
    find_package(CUDAToolkit 12.0 REQUIRED)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
    find_package(TensorRT REQUIRED)
endif()

if(BUILD_DEEPSTREAM)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gobject-2.0
        glib-2.0
    )
    find_package(DeepStream 8.0)
    if(NOT DeepStream_FOUND)
        message(WARNING "DeepStream SDK not found; disabling DeepStream pipeline")
        set(BUILD_DEEPSTREAM OFF)
    endif()
endif()

# ---- Library: skeleton_ar (static) ----
# ---- Library: skeleton_ar_core (static, CPU only) ----
# Config parsing, the skeleton buffer and the track registry are pure CPU
# work. Keeping them in their own target lets the unit tests -- and CI
# runners without an NVIDIA stack -- build them without CUDA or TensorRT.
add_library(skeleton_ar_core STATIC
    src/config/system_config.cpp
    src/utils/logger.cpp
    src/tracking/skeleton_buffer.cpp
    src/tracking/track_registry.cpp
)
target_include_directories(skeleton_ar_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(skeleton_ar_core
    PUBLIC
        opencv_core
        opencv_imgproc
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
)
set_target_properties(skeleton_ar_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Library: skeleton_ar (static, GPU stages) ----
if(NOT SKAR_CPU_ONLY)
file(GLOB_RECURSE SKEL_AR_SOURCES
    "${CMAKE_SOURCE_DIR}/src/config/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/tracking/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/overlay/*.cpp"
)

# Already compiled into skeleton_ar_core.
foreach(core_src system_config logger skeleton_buffer track_registry)
    list(FILTER SKEL_AR_SOURCES EXCLUDE REGEX "${core_src}\\.cpp$")
endforeach()

if(NOT BUILD_DEEPSTREAM)
    list(FILTER SKEL_AR_SOURCES EXCLUDE REGEX "deepstream_pipeline\\.cpp$")
endif()

add_library(skeleton_ar STATIC ${SKEL_AR_SOURCES})
target_include_directories(skeleton_ar
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(skeleton_ar
    PUBLIC
        skeleton_ar_core
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        TensorRT::nvinfer
        TensorRT::nvonnxparser
)
if(BUILD_DEEPSTREAM)
    target_include_directories(skeleton_ar PUBLIC ${DeepStream_INCLUDE_DIRS})
    target_link_libraries(skeleton_ar PUBLIC
        DeepStream::nvds_meta
        DeepStream::nvbufsurface
        PkgConfig::GST
    )
    target_compile_definitions(skeleton_ar PUBLIC SKEL_AR_HAVE_DEEPSTREAM=1)
endif()
set_target_properties(skeleton_ar PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Executable: skeleton_ar_video (offline video processing) ----
add_executable(skeleton_ar_video src/main.cpp)
target_link_libraries(skeleton_ar_video PRIVATE skeleton_ar)

if(BUILD_TOOLS)
    add_executable(skeleton_ar_benchmark tools/benchmark.cpp)
    target_link_libraries(skeleton_ar_benchmark PRIVATE skeleton_ar)

    # Visualises the trained ST-GCN on exported NTU clips (engine + OpenCV
    # only; no DeepStream needed).
    add_executable(skeleton_demo tools/skeleton_demo.cpp)
    target_link_libraries(skeleton_demo PRIVATE skeleton_ar)
endif()
endif()

if(SKAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TARGET skeleton_ar_video)
    install(TARGETS skeleton_ar_video RUNTIME DESTINATION bin)
endif()
if(BUILD_TOOLS)
    install(TARGETS skeleton_ar_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/skeleton_ar/configs)

message(STATUS "")
message(STATUS "skeleton_ar ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CPU-only:          ${SKAR_CPU_ONLY}")
if(NOT SKAR_CPU_ONLY)
    message(STATUS "  CUDA:              ${CUDAToolkit_VERSION}")
    message(STATUS "  TensorRT:          ${TensorRT_VERSION}")
endif()
message(STATUS "  OpenCV:            ${OpenCV_VERSION}")
message(STATUS "  DeepStream path:   ${BUILD_DEEPSTREAM}")
message(STATUS "  Build tools:       ${BUILD_TOOLS}")
message(STATUS "  Build tests:       ${SKAR_BUILD_TESTS}")
message(STATUS "")
