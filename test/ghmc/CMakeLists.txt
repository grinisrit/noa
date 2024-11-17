####################################################################################################
##################                    GHMC functional tests                    #####################
####################################################################################################

# Gflags
include(fetch-gflags)

# Test data
include(noa-test-data)

# Main executable
add_executable(sampler sampler.cc)

add_dependencies(sampler test_data)

# Extra include directories
target_include_directories(sampler PRIVATE ${NOA_ROOT_DIR}/test)

# Link libraries
target_link_libraries(sampler PRIVATE ${PROJECT_NAME} gflags)
target_compile_options(sampler PRIVATE ${W_FLAGS} -O3)
target_add_openmp( sampler )
