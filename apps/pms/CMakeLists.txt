####################################################################################################
##################                     GHMC Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# Main executables
add_executable(bmc bmc.cc)

# Link libraries
target_link_libraries(bmc PRIVATE GHMC openmp)
target_compile_options(bmc PRIVATE -Wall -Wextra -Wpedantic -O3)
