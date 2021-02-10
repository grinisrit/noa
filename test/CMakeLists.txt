####################################################################################################
####################              GHMC: unit testing & benchmarking             ####################
####################################################################################################

message(STATUS "Building tests & benchmarks required")

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)


# Source files
set(GHMC_TEST_SRC 
    test-sampler.cc
    test-dcs.cc)

set(GHMC_BENCH_SRC 
    benchmark-dcs.cc)


# Main test executables
add_executable(test_ghmc ${GHMC_TEST_SRC})
add_executable(benchmark_ghmc ${GHMC_BENCH_SRC})

# Link libraries
target_link_libraries(test_ghmc PRIVATE GTest::Main GHMC)
target_compile_options(test_ghmc PRIVATE -Wall -Wextra -Wpedantic -O3)
target_link_libraries(benchmark_ghmc PRIVATE benchmark::benchmark_main GHMC)
target_compile_options(benchmark_ghmc PRIVATE -Wall -Wextra -Wpedantic -O3)

# Set up test
add_test(NAME TEST_GHMC COMMAND test_ghmc)