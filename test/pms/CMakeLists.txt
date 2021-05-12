####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)
# Compatibility with LibTorch cxx11 ABI
target_compile_definitions(gflags_nothreads_static PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

# Find OpenMP
include(conda-omp)

# Fetch materials
include(noa-pms-models)

# Main executables
add_executable(load_pms_model load-pms-model.cc)
add_dependencies(load_pms_model pms_models)

# Link libraries
target_link_libraries(load_pms_model PRIVATE ${PROJECT_NAME} gflags openmp)
target_compile_options(load_pms_model PRIVATE -Wall -Wextra -Wpedantic -O3)
