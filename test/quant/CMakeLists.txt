####################################################################################################
##################                    QUANT functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Main executable
add_executable(bsm bsm.cc)
add_executable(heston heston.cc)

# Link libraries
target_link_libraries(bsm PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(bsm PRIVATE ${W_FLAGS} -O0 ${OpenMP_CXX_FLAGS})

target_link_libraries(heston PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(heston PRIVATE ${W_FLAGS} -O0 ${OpenMP_CXX_FLAGS})
