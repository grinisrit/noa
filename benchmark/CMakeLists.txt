####################################################################################################
############################              NOA benchmarking             #############################
####################################################################################################

message(STATUS "Building benchmarks required")

# Google Benchmark
include(fetch-benchmark)

# Test data
include(noa-test-data)

# Find OpenMP
find_package(OpenMP QUIET)

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
target_include_directories(measure_dcs_calc PRIVATE ${NOA_ROOT_DIR}/test ${OpenMP_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(measure_dcs_calc PRIVATE benchmark_main ${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES})
target_compile_options(measure_dcs_calc
        PRIVATE -O3
        $<$<COMPILE_LANGUAGE:CXX>: ${W_FLAGS} ${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${MCXX_CUDA}>)