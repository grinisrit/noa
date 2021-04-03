####################################################################################################
#################                     KOTIK Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# XML parser
find_package(pugixml REQUIRED)


# Main executables
add_executable(pumas pumas.cc)

# Link libraries
target_link_libraries(pumas PRIVATE KOTIK gflags openmp pugixml)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3)
