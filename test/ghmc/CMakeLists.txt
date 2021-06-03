####################################################################################################
##################                    GHMC functional tests                    #####################
####################################################################################################

# Gflags
include(fetch-gflags)

# Main executable
add_executable(sampler sampler.cc)

# Extra include directories
target_include_directories(sampler PRIVATE ${NOA_ROOT_DIR}/test/unit)


# Link libraries
target_link_libraries(sampler PRIVATE ${PROJECT_NAME} gflags)
target_compile_options(sampler PRIVATE -Wall -Wextra -Wpedantic -O3)
