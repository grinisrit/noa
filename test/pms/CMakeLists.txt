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
add_executable(load_muon_model load-muon-model.cc)
add_dependencies(load_muon_model pms_models)

# Link libraries
target_include_directories(load_muon_model PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(load_muon_model PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES})
target_compile_options(load_muon_model PRIVATE -Wall -Wextra -Wpedantic -O3 ${OpenMP_CXX_FLAGS})
