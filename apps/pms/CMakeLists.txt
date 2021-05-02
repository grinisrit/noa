####################################################################################################
##################                     NOA Apps: PMS examples                    ###################
####################################################################################################

message(STATUS "Building PMS apps required")

# Gflags
include(fetch-gflags)
# Compatibility with LibTorch cxx11 ABI
target_compile_definitions(gflags_nothreads_static PUBLIC _GLIBCXX_USE_CXX11_ABI=0)

# Find OpenMP
include(conda-omp)

# Fetch materials
include(noa-pms-models)

# Main executables
add_executable(muography muography.cc)
add_dependencies(muography pms_models)

# Link libraries
target_link_libraries(muography PRIVATE ${PROJECT_NAME} gflags openmp)
target_compile_options(muography PRIVATE -Wall -Wextra -Wpedantic -O3)
