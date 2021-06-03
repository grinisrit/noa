####################################################################################################
##################                    GHMC functional tests                    #####################
####################################################################################################

# Gflags
include(fetch-gflags)

# Main executable
add_executable(ghmc_sampler ghmc-sampler.cc)

# Extra include directories
target_include_directories(ghmc_sampler PRIVATE ${NOA_ROOT_DIR}/test/unit)


# Link libraries
target_link_libraries(ghmc_sampler PRIVATE ${PROJECT_NAME} gflags)
target_compile_options(ghmc_sampler PRIVATE -Wall -Wextra -Wpedantic -O3)
