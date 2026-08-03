include(FetchContent)

# Pinned rather than tracking main: an unannounced GoogleTest change should
# never be able to turn this repository's CI red.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# GoogleTest is third-party; this project's warning set is not its contract.
get_directory_property(SKAR_WARNINGS COMPILE_OPTIONS)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "")
FetchContent_MakeAvailable(googletest)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${SKAR_WARNINGS}")

include(GoogleTest)

add_executable(skeleton_ar_tests
    test_skeleton_buffer.cpp
    test_track_registry.cpp
    test_config.cpp
)
target_link_libraries(skeleton_ar_tests PRIVATE skeleton_ar_core GTest::gtest_main)
target_compile_definitions(skeleton_ar_tests
    PRIVATE SKAR_TEST_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

gtest_discover_tests(skeleton_ar_tests)
