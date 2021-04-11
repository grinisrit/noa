####################################################################################################
############################              NOA unit testing             #############################
####################################################################################################

message(STATUS "Building tests required")

# GTest
include(fetch-gtest)
# Compatibility with LibTorch cxx11 ABI
target_compile_definitions(gtest PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

# Test data
include(noa-test-data)

# Source files
set(NOA_TEST_SRC
        main.cc
        test-ghmc.cc
        test-dcs.cc)


# Main test executables
add_executable(noa_test ${NOA_TEST_SRC})

# Link libraries
target_link_libraries(noa_test PRIVATE gtest ${PROJECT_NAME})
target_compile_options(noa_test PRIVATE -Wall -Wextra -Wpedantic -O3)


# Set up test
add_test(NAME TEST_${PROJECT_NAME}
        COMMAND noa_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})