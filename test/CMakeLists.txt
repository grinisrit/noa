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

    set(BUILD_GMOCK OFF CACHE BOOL "")
    set(INSTALL_GTEST OFF CACHE BOOL "")

    add_subdirectory(
        ${googletest_SOURCE_DIR}
        ${googletest_BINARY_DIR})
endif()


# Source files
set(NOA_TEST_SRC 
    main.cc
    test-ghmc.cc
    test-dcs.cc)


# Main test executables
add_executable(noa_test ${NOA_TEST_SRC})

# Link libraries
target_link_libraries(noa_test PRIVATE gtest noa)
target_compile_options(noa_test PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TEST_NOA COMMAND noa_test)