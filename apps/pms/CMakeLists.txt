####################################################################################################
##################                     GHMC Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# Main executables
add_executable(load load.cc)

# Link libraries
target_link_libraries(load PRIVATE GHMC openmp)
target_compile_options(load PRIVATE -Wall -Wextra -Wpedantic -O3)
