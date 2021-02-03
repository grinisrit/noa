####################################################################################################
####################                      GHMC: unit testing                    ####################
####################################################################################################

message(STATUS "Building tests required")

find_package(GTest REQUIRED)

# Source file
set(GHMC_TEST_SRC 
    sampler.cc
    pms.cc)

# Main test executable
add_executable(test_ghmc ${GHMC_TEST_SRC})

# Link libraries
target_link_libraries(test_ghmc PRIVATE GTest::Main GHMC)
target_compile_options(test_ghmc PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TEST_GHMC COMMAND test_ghmc)