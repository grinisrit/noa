####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

# Google Benchmark
include(fetch-benchmark)

# Test data
include(noa-test-data)

set(NOA_BENCH_SRC
        measure-dcs-calc.cc)

# Main test executables
if (BUILD_NOA_CUDA)
    set(NOA_BENCH_SRC_CUDA
            "measure-dcs-calc-cuda.cc;${NOA_ROOT_DIR}/test/kernels.cu")
else ()
    set(NOA_BENCH_SRC_CUDA "")
endif ()

add_executable(measure_dcs_calc ${NOA_BENCH_SRC} ${NOA_BENCH_SRC_CUDA})
set_target_properties(measure_dcs_calc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(measure_dcs_calc test_data)

# Extra include directories
target_include_directories(measure_dcs_calc PRIVATE ${NOA_ROOT_DIR}/test)

# Link libraries
target_link_libraries(measure_dcs_calc PRIVATE benchmark_main ${PROJECT_NAME})
target_compile_options(measure_dcs_calc
        PRIVATE -O3
        $<$<COMPILE_LANGUAGE:CXX>: ${W_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${MCXX_CUDA}>)
target_add_openmp( measure_dcs_cal )
