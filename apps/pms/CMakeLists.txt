####################################################################################################
##################                     NOA Apps: PMS examples                    ###################
####################################################################################################

message(STATUS "Building PMS apps required")

# Find OpenMP
include(conda-omp)

# Fetch materials
include(noa-pms-models)

# Main executables
add_executable(pumas pumas.cc)

# Link libraries
target_link_libraries(pumas PRIVATE NOA gflags openmp)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3)
