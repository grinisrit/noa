####################################################################################################
##################                    QUANT functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Main executable
add_executable(bsm bsm.cc)
add_executable(heston_sim heston_sim.cc)
add_executable(lsm lsm.cc)

# Link libraries
target_link_libraries(bsm PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(bsm PRIVATE ${W_FLAGS} -O0 ${OpenMP_CXX_FLAGS})

target_link_libraries(heston_sim PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(heston_sim PRIVATE ${W_FLAGS} -O0 ${OpenMP_CXX_FLAGS})

target_link_libraries(lsm PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(lsm PRIVATE ${W_FLAGS} -O0 ${OpenMP_CXX_FLAGS})
