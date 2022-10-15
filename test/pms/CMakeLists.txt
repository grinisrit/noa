####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find ZLIB
find_package(ZLIB REQUIRED)

# Fetch materials
include(pumas-materials)

# Main executables
add_executable(muon_model
        muon-model.cc)
        
add_dependencies(muon_model pumas_materials)

# Link libraries
target_include_directories(muon_model PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(muon_model PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES} ZLIB::ZLIB)
target_compile_options(muon_model PRIVATE
        PRIVATE -O3 -DHAVE_ZLIB -DHAVE_OPENMP
	$<$<COMPILE_LANGUAGE:CXX>:${W_FLAGS} ${OpenMP_CXX_FLAGS} -fpermissive>)
