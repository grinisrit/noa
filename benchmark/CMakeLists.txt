####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

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

set(NOA_BENCH_SRC 
    benchmark-dcs.cc)

# Main test executables
add_executable(noa_bench ${NOA_BENCH_SRC})

# Link libraries
target_link_libraries(noa_bench PRIVATE benchmark_main noa)
target_compile_options(noa_bench PRIVATE -Wall -Wextra -Wpedantic -O3)