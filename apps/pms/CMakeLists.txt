####################################################################################################
##################                     NOA Apps: PMS examples                    ###################
####################################################################################################

message(STATUS "Building PMS apps required")

# Gflags
include(fetch-gflags)

# Find OpenMP
include(conda-omp)

# Fetch materials
include(noa-pms-models)

# Main executables
add_executable(pumas pumas.cc)

# Link libraries
target_link_libraries(pumas PRIVATE noa gflags openmp)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3)
