####################################################################################################
###################              KOTIK: unit testing & benchmarking             ####################
####################################################################################################

message(STATUS "Building tests & benchmarks required")


FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.10.x
  )

FetchContent_GetProperties(googletest)

if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)

    set(INSTALL_GTEST OFF CACHE BOOL "")

    add_subdirectory(
        ${googletest_SOURCE_DIR}
        ${googletest_BINARY_DIR})
endif()


FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.5.2
  )

FetchContent_GetProperties(benchmark)

if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)

    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "")

    add_subdirectory(
        ${benchmark_SOURCE_DIR}
        ${benchmark_BINARY_DIR})
endif()



# Source files
set(KOTIK_TEST_SRC 
    test-ghmc.cc
    test-dcs.cc)

set(KOTIK_BENCH_SRC 
    benchmark-dcs.cc)


# Main test executables
add_executable(test_kotik ${KOTIK_TEST_SRC})
add_executable(benchmark_kotik ${KOTIK_BENCH_SRC})

# Link libraries
target_link_libraries(test_kotik PRIVATE gtest_main KOTIK)
target_compile_options(test_kotik PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(benchmark_kotik PRIVATE benchmark_main KOTIK)
target_compile_options(benchmark_kotik PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TEST_KOTIK COMMAND test_kotik)