####################################################################################################
##################                      QUANT functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Main executable
add_executable(quant heston.cc)

# Link libraries
target_link_libraries(quant PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(quant PRIVATE ${W_FLAGS} -O3 ${OpenMP_CXX_FLAGS})
