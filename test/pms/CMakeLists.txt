####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Fetch materials
include(pumas-materials)

# Main executables
add_executable(muon_model
        muon-model.cc
        ${NOA_ROOT_DIR}/test/kernels.c)
add_dependencies(muon_model pumas_materials)

# Link libraries
target_include_directories(muon_model PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(muon_model PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(muon_model PRIVATE
        PRIVATE -O3
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic ${OpenMP_CXX_FLAGS}>)
