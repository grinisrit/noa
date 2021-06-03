####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

# Google Benchmark
include(fetch-benchmark)

# Test data
include(noa-test-data)
# Find OpenMP
include(conda-omp)

set(NOA_BENCH_SRC
        benchmark-dcs.cc)

# Main test executables
if (BUILD_NOA_CUDA)
    add_executable(noa_benchmark ${NOA_BENCH_SRC} benchmark-dcs-cuda.cc ${NOA_ROOT_DIR}/test/kernels.cu)
    set_target_properties(noa_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
    add_executable(noa_benchmark ${NOA_BENCH_SRC})
endif ()
add_dependencies(noa_benchmark test_data)

# Extra include directories
target_include_directories(noa_benchmark PRIVATE ${NOA_ROOT_DIR}/test)

# Link libraries
target_link_libraries(noa_benchmark PRIVATE benchmark_main ${PROJECT_NAME} openmp)
target_compile_options(noa_benchmark
        PRIVATE
        -Wall -Wextra -Wpedantic -O3
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)