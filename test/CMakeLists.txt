####################################################################################################
#################                     GHMC: testing cmake file                     #################
####################################################################################################

message(STATUS "Building tests required")

find_package(GTest REQUIRED)

# Source file
set(GHMC_TEST_SRC 
    unit.cc
    functional.cc)

# Main test executable
add_executable(tghmc ${GHMC_TEST_SRC})

# Link libraries
target_link_libraries(tghmc PRIVATE GTest::Main torch GHMC)
target_compile_options(tghmc PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TGHMC COMMAND tghmc)