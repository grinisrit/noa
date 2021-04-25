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
# Find OpenMP
include(conda-omp)

set(NOA_BENCH_SRC
        benchmark-dcs.cc
        benchmark-dcs-model.cc)

# Main test executables
if (BUILD_CUDA)
    add_executable(noa_bench ${NOA_BENCH_SRC} benchmark-dcs-cuda.cc ${NOA_ROOT_DIR}/test/dcs-kernels.cu)
    set_target_properties(noa_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
    add_executable(noa_bench ${NOA_BENCH_SRC})
endif ()
add_dependencies(noa_bench test_data)

# Extra include directories
target_include_directories(noa_bench PRIVATE ${NOA_ROOT_DIR}/test)

# Link libraries
target_link_libraries(noa_bench PRIVATE benchmark_main ${PROJECT_NAME} openmp)
target_compile_options(noa_bench PRIVATE -Wall -Wextra -Wpedantic -O3)