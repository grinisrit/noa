####################################################################################################
##################                     GHMC Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# XML parser
find_package(pugixml REQUIRED)

# Main executables
add_executable(bmc bmc.cc)

# Link libraries
target_link_libraries(bmc PRIVATE GHMC openmp pugixml)
target_compile_options(bmc PRIVATE -Wall -Wextra -Wpedantic -O3)
