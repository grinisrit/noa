####################################################################################################
############################              NOA unit testing             #############################
####################################################################################################

message(STATUS "Building tests required")

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.10.x)

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)

    set(INSTALL_GTEST OFF CACHE BOOL "")

    add_subdirectory(
        ${googletest_SOURCE_DIR}
        ${googletest_BINARY_DIR})
endif()


# Source files
set(NOA_TEST_SRC 
    test-ghmc.cc
    test-dcs.cc)


# Main test executables
add_executable(test_noa ${NOA_TEST_SRC})

# Link libraries
target_link_libraries(test_noa PRIVATE gtest_main NOA)
target_compile_options(test_noa PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TEST_NOA COMMAND test_noa)