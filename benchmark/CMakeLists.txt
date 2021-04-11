####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

# Google Benchmark
include(fetch-benchmark)
# Compatibility with LibTorch cxx11 ABI
target_compile_definitions(benchmark PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

# Test data
include(noa-test-data)

set(NOA_BENCH_SRC
        benchmark-dcs.cc)

# Main test executables
add_executable(noa_bench ${NOA_BENCH_SRC})

# Extra include directories
target_include_directories(noa_bench PRIVATE ${NOA_ROOT_DIR}/test)

# Link libraries
target_link_libraries(noa_bench PRIVATE benchmark_main ${PROJECT_NAME})
target_compile_options(noa_bench PRIVATE -Wall -Wextra -Wpedantic -O3)