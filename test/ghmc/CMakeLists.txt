####################################################################################################
##################                    GHMC functional tests                    #####################
####################################################################################################

# Gflags
include(fetch-gflags)

# Test data
include(noa-test-data)

# Find OpenMP
find_package(OpenMP QUIET)

# Main executable
add_executable(sampler sampler.cc)

add_dependencies(sampler test_data)

# Extra include directories
target_include_directories(sampler PRIVATE ${NOA_ROOT_DIR}/test ${OpenMP_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(sampler PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(sampler PRIVATE -Wall -Wextra -Wpedantic -O3 ${OpenMP_CXX_FLAGS})
