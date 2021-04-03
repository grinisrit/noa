####################################################################################################
#################                     KOTIK Apps: PMS examples                    ##################
####################################################################################################

message(STATUS "Building PMS apps required")

include(conda-omp)

# Main executables
add_executable(pumas pumas.cc)

include(fetch-pugixml)

target_include_directories(pumas PRIVATE ${PUGIXML_SRC})

# Link libraries
target_link_libraries(pumas PRIVATE KOTIK gflags openmp)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3 -DPUGIXML_HEADER_ONLY)
