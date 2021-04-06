####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

# Google Benchmark
include(fetch-benchmark)

# Test data
include(noa-test-data)

set(NOA_BENCH_SRC 
    benchmark-dcs.cc)

# Main test executables
add_executable(noa_bench ${NOA_BENCH_SRC})

# Link libraries
target_link_libraries(noa_bench PRIVATE benchmark_main noa)
target_compile_options(noa_bench PRIVATE -Wall -Wextra -Wpedantic -O3)