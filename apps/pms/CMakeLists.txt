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
add_executable(pumas pumas.cc)
add_dependencies(pumas pms_models)

# Link libraries
target_link_libraries(pumas PRIVATE ${PROJECT_NAME} gflags openmp)
target_compile_options(pumas PRIVATE -Wall -Wextra -Wpedantic -O3)
