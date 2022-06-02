####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Find ZLIB
find_package(ZLIB)

# Fetch materials
include(pumas-materials)

# Main executables
add_executable(muon_model
        muon-model.cc
        ${NOA_ROOT_DIR}/test/kernels.c
        ${NOA_ROOT_DIR}/test/kernels.cc)
        
add_dependencies(muon_model pumas_materials)
set_target_properties(muon_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_include_directories(muon_model PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(muon_model PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES} ZLIB::ZLIB)
target_compile_options(muon_model PRIVATE
        PRIVATE -O3
        $<$<COMPILE_LANGUAGE:CXX>:${W_FLAGS} ${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${MCXX_CUDA}>)
