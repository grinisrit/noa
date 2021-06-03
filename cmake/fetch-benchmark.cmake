include(FetchContent)

if(NOT TARGET benchmark)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.5.2)

    FetchContent_GetProperties(benchmark)

    if(NOT benchmark_POPULATED)
        FetchContent_Populate(benchmark)

        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "")
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

        add_subdirectory(
            ${benchmark_SOURCE_DIR}
            ${benchmark_BINARY_DIR})
    endif()

    # Compatibility with LibTorch cxx11 ABI
    target_compile_definitions(benchmark PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()