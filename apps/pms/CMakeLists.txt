####################################################################################################
##################                     GHMC Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# XML parser
find_package(pugixml REQUIRED)

# Command line utilities
find_package(gflags REQUIRED)

# Main executables
add_executable(pumas pumas.cc)

# Link libraries
target_link_libraries(pumas PRIVATE GHMC openmp pugixml gflags)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3)
