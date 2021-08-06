####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Fetch materials
include(noa-pms-models)

# Main executables
add_executable(muon muon.cc)
add_dependencies(muon pms_models)

# Link libraries
target_include_directories(muon PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(muon PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(muon PRIVATE -Wall -Wextra -Wpedantic -O3 ${OpenMP_CXX_FLAGS})
