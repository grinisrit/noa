####################################################################################################
############################              NOA unit testing             #############################
####################################################################################################


# GTest
include(fetch-gtest)

# Test data
include(noa-test-data)

# Find OpenMP
find_package(OpenMP QUIET)

# Find ZLIB
find_package(ZLIB)

# Source files
set(NOA_TEST_SRC
        noa-test-suite.cc
        test-ghmc-sampler.cc
        test-dcs-calc.cc
        test-tnl.cc
        test-trace.cc
        ${NOA_ROOT_DIR}/test/kernels.cc)

if (BUILD_NOA_CUDA)
    set(NOA_TEST_SRC_CUDA
            test-ghmc-sampler-cuda.cc
            test-dcs-calc-cuda.cc
            test-tnl.cu;test-trace.cu
            ${NOA_ROOT_DIR}/test/kernels.cu)
else ()
    set(NOA_TEST_SRC_CUDA "")
endif ()

# Main test executable
add_executable(noa_test_suite ${NOA_TEST_SRC} ${NOA_TEST_SRC_CUDA})
add_dependencies(noa_test_suite test_data)
set_target_properties(noa_test_suite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Extra include directories
target_include_directories(noa_test_suite PRIVATE ${NOA_ROOT_DIR}/test ${OpenMP_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(noa_test_suite PRIVATE gtest ${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES} ZLIB::ZLIB)
target_compile_options(noa_test_suite
        PRIVATE -O3 -DHAVE_ZLIB -DHAVE_OPENMP
        $<$<COMPILE_LANGUAGE:CXX>: ${W_FLAGS} ${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${MCXX_CUDA}>)


# Set up test
add_test(NAME TEST_${PROJECT_NAME}
        COMMAND noa_test_suite
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
