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
option(BUILD_TESTS "Build unit tests" OFF)

# ---- Dependencies ----
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
    gstreamer-1.0
    gstreamer-app-1.0
    gobject-2.0
    glib-2.0
)

find_package(DeepStream 8.0 REQUIRED)
find_package(TensorRT REQUIRED)

# ---- Library: skeleton_ar (static) ----
file(GLOB_RECURSE SKEL_AR_SOURCES
    "${CMAKE_SOURCE_DIR}/src/config/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/tracking/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/overlay/*.cpp"
)

add_library(skeleton_ar STATIC ${SKEL_AR_SOURCES})
target_include_directories(skeleton_ar
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${DeepStream_INCLUDE_DIRS}
)
target_link_libraries(skeleton_ar
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        DeepStream::nvds_meta
        DeepStream::nvbufsurface
        PkgConfig::GST
)
set_target_properties(skeleton_ar PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Executable: skeleton_ar_video (offline video processing) ----
add_executable(skeleton_ar_video src/main.cpp)
target_link_libraries(skeleton_ar_video PRIVATE skeleton_ar)

if(BUILD_TOOLS)
    add_executable(skeleton_ar_benchmark tools/benchmark.cpp)
    target_link_libraries(skeleton_ar_benchmark PRIVATE skeleton_ar)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS skeleton_ar_video RUNTIME DESTINATION bin)
if(BUILD_TOOLS)
    install(TARGETS skeleton_ar_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/skeleton_ar/configs)

message(STATUS "")
message(STATUS "skeleton_ar ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:         ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV:       ${OpenCV_VERSION}")
message(STATUS "  TensorRT:     ${TensorRT_VERSION}")
message(STATUS "  DeepStream:   ${DeepStream_VERSION}")
message(STATUS "  Build tools:  ${BUILD_TOOLS}")
message(STATUS "  Build tests:  ${BUILD_TESTS}")
message(STATUS "")
